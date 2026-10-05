"""
Loitering & Unauthorized Dwell Incident Detection Module
Monitors prolonged stationary or low-movement dwell times within critical
geofenced zones or sensitive corridors, generating early loitering alerts.
"""

import time
from typing import List, Dict, Any, Optional
import numpy as np


class LoiteringDetector:
    """Detects individuals lingering or loitering beyond acceptable duration thresholds."""

    def __init__(self, dwell_thresh_sec: float = 6.0, movement_radius_px: float = 40.0):
        self.dwell_thresh_sec = dwell_thresh_sec
        self.movement_radius_px = movement_radius_px
        # Track ID -> {first_seen: float, anchor_pos: [x, y], zone_name: str}
        self.dwell_registry: Dict[int, Dict[str, Any]] = {}

    def evaluate(
        self,
        active_tracks: List[Dict[str, Any]],
        restricted_zones: Optional[List[Dict[str, Any]]] = None,
        timestamp: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates active person tracks for prolonged loitering behavior.

        Args:
            active_tracks: List of active track dictionaries.
            restricted_zones: List of zone definitions with polygon coordinates.
            timestamp: Current timestamp (seconds).

        Returns:
            List of loitering alert dictionaries.
        """
        now = timestamp if timestamp is not None else time.time()
        alerts = []
        current_track_ids = set()

        person_tracks = [t for t in active_tracks if t["class"] == "person"]

        for track in person_tracks:
            tid = track["id"]
            current_track_ids.add(tid)
            bbox = track["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            pos = np.array([cx, cy])

            # Determine if track is in a restricted zone
            in_zone_name = None
            if restricted_zones:
                for zone in restricted_zones:
                    poly = zone.get("polygon", [])
                    if len(poly) >= 3 and self._point_in_poly(cx, cy, poly):
                        in_zone_name = zone.get("name", "Restricted Area")
                        break

            if tid not in self.dwell_registry:
                self.dwell_registry[tid] = {
                    "first_seen": now,
                    "anchor_pos": pos,
                    "last_pos": pos,
                    "zone_name": in_zone_name,
                    "max_displacement": 0.0,
                }
            else:
                entry = self.dwell_registry[tid]
                entry["last_pos"] = pos
                displacement = float(np.linalg.norm(pos - entry["anchor_pos"]))
                entry["max_displacement"] = max(entry["max_displacement"], displacement)

                # Reset anchor if subject made significant purposeful transit
                if displacement > self.movement_radius_px * 2.5:
                    entry["anchor_pos"] = pos
                    entry["first_seen"] = now
                    entry["max_displacement"] = 0.0
                    entry["zone_name"] = in_zone_name
                    continue

                dwell_duration = now - entry["first_seen"]

                # Loitering triggered if dwelling in zone longer than threshold
                # If inside restricted zone, threshold is reduced by 40%
                eff_threshold = (self.dwell_thresh_sec * 0.6) if in_zone_name else self.dwell_thresh_sec

                if dwell_duration >= eff_threshold and entry["max_displacement"] < self.movement_radius_px * 2.0:
                    confidence = min(0.96, round(0.72 + (dwell_duration / (eff_threshold * 2)) * 0.24, 3))
                    severity = "HIGH" if in_zone_name else "MEDIUM"
                    zone_label = f" in '{in_zone_name}'" if in_zone_name else ""

                    alerts.append({
                        "incident_type": "Suspicious Loitering / Lingering",
                        "severity": severity,
                        "confidence": confidence,
                        "involved_tracks": [tid],
                        "bbox": bbox,
                        "reasons": [
                            f"Subject ID#{tid} lingering{zone_label} for {dwell_duration:.1f}s",
                            f"Stationary radius: {entry['max_displacement']:.1f}px (threshold: {self.movement_radius_px:.1f}px)",
                            f"Dwell threshold exceeded ({eff_threshold:.1f}s configured)",
                        ],
                        "evidence_metrics": {
                            "dwell_duration_sec": round(dwell_duration, 1),
                            "max_displacement_px": round(entry["max_displacement"], 1),
                            "zone": in_zone_name or "General Corridor",
                        },
                    })

        # Purge stale tracks from registry
        stale_ids = [tid for tid in self.dwell_registry if tid not in current_track_ids]
        for tid in stale_ids:
            del self.dwell_registry[tid]

        return alerts

    @staticmethod
    def _point_in_poly(x: float, y: float, poly: List[List[int]]) -> bool:
        """Ray-casting algorithm for point-in-polygon check."""
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

"""
Worker Immobility & "Man-Down" Emergency Detection Module
Monitors fallen or motionless workers for complete lack of movement / unresponsiveness
over an extended temporal window, triggering staged emergency escalation.
"""

import math
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class WorkerImmobilityDetector:
    """
    Detects unresponsiveness and critical worker immobility.
    Analyzes spatial displacement variance, aspect ratio, and prolonged stationary duration.
    """

    def __init__(
        self,
        immobility_time_sec: float = 6.0,
        max_displacement_px: float = 18.0,
        aspect_ratio_threshold: float = 0.95,
    ):
        self.immobility_time_sec = immobility_time_sec
        self.max_displacement_px = max_displacement_px
        self.aspect_ratio_threshold = aspect_ratio_threshold
        # State tracker: track_id -> {start_time, initial_pos, max_dist_seen, last_seen}
        self.immobility_tracker: Dict[int, Dict[str, Any]] = {}

    def _get_centroid(self, bbox: List[int]) -> Tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def evaluate(
        self,
        active_tracks: List[Dict[str, Any]],
        timestamp: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates active person tracks for extended immobility / man-down state.

        Returns:
            List of man-down emergency alerts.
        """
        now = timestamp or time.time()
        alerts = []
        active_ids = set()

        for t in active_tracks:
            if t.get("class") != "person":
                continue

            t_id = t["id"]
            active_ids.add(t_id)
            bbox = t["bbox"]
            w = max(1, bbox[2] - bbox[0])
            h = max(1, bbox[3] - bbox[1])
            aspect_ratio = w / float(h)
            centroid = self._get_centroid(bbox)

            # Check if person is horizontal / prone or sitting collapsed on floor
            is_prone = (aspect_ratio >= self.aspect_ratio_threshold)

            if t_id not in self.immobility_tracker:
                if is_prone:
                    self.immobility_tracker[t_id] = {
                        "first_prone_time": now,
                        "initial_pos": centroid,
                        "last_pos": centroid,
                        "max_displacement": 0.0,
                        "alert_raised": False,
                    }
            else:
                entry = self.immobility_tracker[t_id]
                dx = centroid[0] - entry["initial_pos"][0]
                dy = centroid[1] - entry["initial_pos"][1]
                displacement = math.sqrt(dx * dx + dy * dy)
                entry["last_pos"] = centroid
                entry["max_displacement"] = max(entry["max_displacement"], displacement)

                # If worker stood up and walked away, reset tracker
                if not is_prone and displacement > self.max_displacement_px * 2.0:
                    del self.immobility_tracker[t_id]
                    continue

                # Worker is still in immobility window
                elapsed = now - entry["first_prone_time"]

                if elapsed >= self.immobility_time_sec and entry["max_displacement"] <= self.max_displacement_px:
                    is_critical = elapsed >= (self.immobility_time_sec * 1.8)
                    severity = "CRITICAL" if is_critical else "HIGH"
                    conf = min(0.98, 0.85 + (elapsed / 30.0) * 0.12)

                    alerts.append({
                        "incident_type": "worker_mandown",
                        "track_id": t_id,
                        "bbox": bbox,
                        "confidence": round(conf, 2),
                        "severity": severity,
                        "immobility_duration_sec": round(elapsed, 1),
                        "aspect_ratio": round(aspect_ratio, 2),
                        "reasons": [
                            f"Worker #{t_id} collapsed and unresponsive on floor for {round(elapsed, 1)}s without kinetic recovery.",
                            f"Horizontal body aspect ratio: {round(aspect_ratio, 2)} (Prone threshold: {self.aspect_ratio_threshold}).",
                            f"Total spatial displacement: {round(entry['max_displacement'], 1)} px (Stillness limit: {self.max_displacement_px} px).",
                            "STAGED EMERGENCY DISPATCH: Immediate medical/safety response recommended."
                        ],
                        "evidence_metrics": {
                            "track_id": t_id,
                            "immobility_duration_sec": round(elapsed, 1),
                            "max_displacement_px": round(entry["max_displacement"], 1),
                            "aspect_ratio": round(aspect_ratio, 2),
                            "is_critical_dispatch": is_critical,
                        }
                    })

        # Purge dead tracks from tracker
        dead_keys = [k for k in self.immobility_tracker.keys() if k not in active_ids]
        for k in dead_keys:
            del self.immobility_tracker[k]

        return alerts

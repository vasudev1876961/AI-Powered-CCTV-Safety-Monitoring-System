"""
Restricted Zone Intrusion & Loitering Detection Module
Implements ray-casting point-in-polygon geometry checks to identify unauthorized entry
and extended loitering in secure facilities, construction hazards, or perimeter fences.
"""

from typing import List, Dict, Any, Optional
from SafetySurveillance.tracking.trajectory import TrackHistory


def point_in_polygon(point: tuple, polygon: List[List[int]]) -> bool:
    """Ray-casting algorithm for testing if a 2D point is inside a polygon."""
    x, y = point
    inside = False
    n = len(polygon)
    p1x, p1y = polygon[0]

    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


class IntrusionDetector:
    """Monitors virtual geofence perimeter zones for intrusion and loitering."""

    def __init__(self, restricted_zones: List[Dict[str, Any]] = None, loiter_thresh_sec: float = 6.0):
        self.restricted_zones = restricted_zones or []
        self.loiter_thresh = loiter_thresh_sec

    def evaluate(self, active_tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Checks all active tracks against configured restricted zones."""
        alerts = []
        if not self.restricted_zones:
            return alerts

        for track in active_tracks:
            hist: TrackHistory = track["history"]
            if not hist.centroids:
                continue

            cx, cy = hist.centroids[-1]
            foot_point = (cx, track["bbox"][3])  # Foot contact point on ground plane

            for zone in self.restricted_zones:
                zone_name = zone.get("name", "Restricted Area")
                poly = zone.get("polygon", [])
                if len(poly) < 3:
                    continue

                if point_in_polygon(foot_point, poly):
                    # Person or vehicle is inside zone!
                    dwell_time = hist.get_dwell_time(stationary_threshold_px=40.0)
                    is_loitering = dwell_time >= self.loiter_thresh

                    incident_type = "Perimeter Intrusion" if not is_loitering else "Suspicious Loitering"
                    severity = "CRITICAL" if is_loitering else "HIGH"

                    alerts.append({
                        "incident_type": incident_type,
                        "zone_name": zone_name,
                        "severity": severity,
                        "confidence": 0.94 if is_loitering else 0.89,
                        "track_id": track["id"],
                        "class": track["class"],
                        "bbox": track["bbox"],
                        "reasons": [
                            f"Entity entered {zone_name} boundaries",
                            f"Dwell time inside restricted zone: {dwell_time:.1f}s",
                            f"Breach point: ({int(cx)}, {int(foot_point[1])})",
                        ],
                        "evidence_metrics": {
                            "zone": zone_name,
                            "dwell_time": round(dwell_time, 1),
                            "target_class": track["class"],
                        },
                    })

        return alerts

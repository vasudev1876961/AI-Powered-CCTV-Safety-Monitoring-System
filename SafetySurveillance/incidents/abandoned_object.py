"""
Abandoned Object / Unattended Luggage Incident Detection Module
Monitors bags, backpacks, and luggage left stationary without an owner
in public or sensitive CCTV surveillance zones.
"""

import numpy as np
from typing import List, Dict, Any


class AbandonedObjectDetector:
    """Detects unattended luggage, backpacks, and suspicious stationary packages."""

    def __init__(self, unattended_seconds: float = 5.0, owner_radius_px: float = 120.0):
        self.unattended_seconds = unattended_seconds
        self.owner_radius = owner_radius_px
        self.target_classes = {"backpack", "handbag", "suitcase", "object"}

    def evaluate(self, active_tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scans active objects and correlates with nearest person tracks."""
        alerts = []
        object_tracks = [t for t in active_tracks if t["class"] in self.target_classes]
        person_tracks = [t for t in active_tracks if t["class"] == "person"]

        for obj in object_tracks:
            hist = obj["history"]
            dwell_time = hist.get_dwell_time(stationary_threshold_px=20.0)

            if dwell_time >= self.unattended_seconds:
                # Check distance to nearest person
                obj_center = hist.centroids[-1]
                min_dist_to_person = float("inf")

                for p in person_tracks:
                    p_center = p["history"].centroids[-1]
                    d = np.sqrt((obj_center[0] - p_center[0]) ** 2 + (obj_center[1] - p_center[1]) ** 2)
                    if d < min_dist_to_person:
                        min_dist_to_person = d

                if min_dist_to_person > self.owner_radius:
                    alerts.append({
                        "incident_type": "Abandoned Object / Luggage",
                        "severity": "HIGH",
                        "confidence": 0.88,
                        "track_id": obj["id"],
                        "class": obj["class"],
                        "bbox": obj["bbox"],
                        "reasons": [
                            f"Object ({obj['class']}) stationary for {dwell_time:.1f}s",
                            f"No owner within {self.owner_radius}px perimeter (nearest person: {min_dist_to_person:.1f}px away)",
                        ],
                        "evidence_metrics": {
                            "unattended_duration": round(dwell_time, 1),
                            "nearest_person_dist": round(min_dist_to_person, 1),
                        },
                    })

        return alerts

"""
Violence & Physical Altercation Incident Detection Module
Analyzes multi-person proximity, rapid reciprocal acceleration,
and high-frequency kinetic jitter between interacting subjects.
"""

import numpy as np
from typing import List, Dict, Any, Optional


class ViolenceDetector:
    """Detects physical fights, brawls, and aggressive interactions."""

    def __init__(self, proximity_thresh: float = 85.0, kinetic_thresh: float = 45.0):
        self.proximity_thresh = proximity_thresh
        self.kinetic_thresh = kinetic_thresh

    def evaluate(self, active_tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scans all pairs of persons for fight signatures.

        Returns:
            List of violence incident alerts.
        """
        person_tracks = [t for t in active_tracks if t["class"] == "person"]
        alerts = []

        for i in range(len(person_tracks)):
            for j in range(i + 1, len(person_tracks)):
                p1 = person_tracks[i]
                p2 = person_tracks[j]

                hist1 = p1["history"]
                hist2 = p2["history"]

                if len(hist1.centroids) < 4 or len(hist2.centroids) < 4:
                    continue

                c1 = hist1.centroids[-1]
                c2 = hist2.centroids[-1]
                dist = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

                if dist < self.proximity_thresh:
                    # Calculate kinetic energy and velocity variance
                    v1 = np.array(hist1.velocities[-4:])
                    v2 = np.array(hist2.velocities[-4:])

                    speed1 = np.linalg.norm(v1, axis=1) if len(v1) > 0 else [0]
                    speed2 = np.linalg.norm(v2, axis=1) if len(v2) > 0 else [0]

                    avg_speed = float((np.mean(speed1) + np.mean(speed2)) / 2.0)
                    jitter = float(np.var(speed1) + np.var(speed2))

                    if avg_speed > self.kinetic_thresh or jitter > 120.0:
                        alerts.append({
                            "incident_type": "Violence / Physical Fight",
                            "severity": "CRITICAL",
                            "confidence": min(0.95, round(0.70 + (avg_speed / 200.0) * 0.25, 3)),
                            "involved_tracks": [p1["id"], p2["id"]],
                            "bbox": [
                                min(p1["bbox"][0], p2["bbox"][0]),
                                min(p1["bbox"][1], p2["bbox"][1]),
                                max(p1["bbox"][2], p2["bbox"][2]),
                                max(p1["bbox"][3], p2["bbox"][3]),
                            ],
                            "reasons": [
                                f"High proximity interaction ({dist:.1f}px separation)",
                                f"Rapid kinetic oscillations ({avg_speed:.1f} px/s avg speed)",
                                f"High reciprocal velocity jitter ({jitter:.1f})",
                            ],
                            "evidence_metrics": {
                                "inter_person_dist": round(dist, 1),
                                "kinetic_energy": round(avg_speed, 1),
                                "velocity_jitter": round(jitter, 1),
                            },
                        })

        return alerts

"""
Vehicle-Pedestrian Proximity & Collision Warning Module
Calculates dynamic Time-To-Collision (TTC), relative approach velocity vectors,
and blind-spot proximity hazards between industrial vehicles/forklifts and workers.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class CollisionWarningDetector:
    """
    Monitors relative trajectories and kinematic velocity vectors between
    vehicles (forklifts, AGVs, trucks) and pedestrians to predict collisions.
    """

    def __init__(
        self,
        proximity_threshold_px: float = 120.0,
        critical_ttc_seconds: float = 2.0,
        vehicle_classes: Optional[List[str]] = None,
    ):
        self.proximity_threshold_px = proximity_threshold_px
        self.critical_ttc_seconds = critical_ttc_seconds
        self.vehicle_classes = set(vehicle_classes or ["car", "motorcycle", "truck", "forklift", "bicycle"])
        # History of detected proximity alerts for cooldown & smoothing
        self.collision_state_cache: Dict[str, Dict[str, Any]] = {}

    def _get_centroid(self, bbox: List[int]) -> Tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def _get_velocity(self, track: Dict[str, Any]) -> Tuple[float, float]:
        """Extracts (vx, vy) in px/frame from track history."""
        hist = track.get("history")
        if hist and hasattr(hist, "velocities") and len(hist.velocities) > 0:
            return hist.velocities[-1]
        elif hist and hasattr(hist, "centroids") and len(hist.centroids) >= 2:
            p2 = hist.centroids[-1]
            p1 = hist.centroids[-2]
            return (p2[0] - p1[0], p2[1] - p1[1])
        return (0.0, 0.0)

    def calculate_ttc(
        self,
        c_veh: Tuple[float, float],
        v_veh: Tuple[float, float],
        c_ped: Tuple[float, float],
        v_ped: Tuple[float, float],
        fps: float = 25.0
    ) -> Tuple[float, float, float]:
        """
        Calculates distance, approach speed, and Time-To-Collision (TTC) in seconds.

        Returns:
            Tuple of (distance_px, relative_speed_px_sec, ttc_seconds)
        """
        dx = c_ped[0] - c_veh[0]
        dy = c_ped[1] - c_veh[1]
        distance = math.sqrt(dx * dx + dy * dy)
        if distance < 1e-4:
            return 0.0, 0.0, 0.0

        # Unit vector pointing from vehicle to pedestrian
        ux = dx / distance
        uy = dy / distance

        # Relative velocity in px/frame
        rel_vx = (v_veh[0] - v_ped[0])
        rel_vy = (v_veh[1] - v_ped[1])

        # Closing / approach velocity projected along the line of sight (px/frame)
        approach_speed_frame = (rel_vx * ux + rel_vy * uy)
        approach_speed_sec = approach_speed_frame * fps

        if approach_speed_sec <= 0:
            # Targets are moving apart or stationary relative to each other
            return distance, 0.0, 999.0

        ttc_seconds = distance / max(0.1, approach_speed_sec)
        return distance, approach_speed_sec, round(ttc_seconds, 2)

    def evaluate(
        self,
        active_tracks: List[Dict[str, Any]],
        fps: float = 25.0
    ) -> List[Dict[str, Any]]:
        """
        Evaluates all active tracks for vehicle-pedestrian collision hazards.

        Returns:
            List of collision alerts with trajectory vectors and TTC metadata.
        """
        alerts = []
        vehicles = []
        pedestrians = []

        for t in active_tracks:
            cls_name = str(t.get("class", "")).lower()
            if cls_name in self.vehicle_classes:
                vehicles.append(t)
            elif cls_name == "person":
                pedestrians.append(t)

        if not vehicles or not pedestrians:
            return alerts

        for veh in vehicles:
            v_id = veh["id"]
            v_bbox = veh["bbox"]
            c_veh = self._get_centroid(v_bbox)
            vel_veh = self._get_velocity(veh)

            for ped in pedestrians:
                p_id = ped["id"]
                p_bbox = ped["bbox"]
                c_ped = self._get_centroid(p_bbox)
                vel_ped = self._get_velocity(ped)

                dist_px, rel_speed, ttc = self.calculate_ttc(c_veh, vel_veh, c_ped, vel_ped, fps=fps)

                # Collision hazard conditions
                is_imminent = (ttc <= self.critical_ttc_seconds and dist_px < self.proximity_threshold_px * 2.5)
                is_proximity_breach = (dist_px <= self.proximity_threshold_px)

                if is_imminent or is_proximity_breach:
                    pair_key = f"veh{v_id}_ped{p_id}"
                    severity = "CRITICAL" if (is_imminent and dist_px < self.proximity_threshold_px * 1.5) else "HIGH"
                    conf = 0.94 if is_imminent else 0.85

                    # Enclosing collision bounding box spanning both entities
                    min_x = min(v_bbox[0], p_bbox[0])
                    min_y = min(v_bbox[1], p_bbox[1])
                    max_x = max(v_bbox[2], p_bbox[2])
                    max_y = max(v_bbox[3], p_bbox[3])

                    alerts.append({
                        "incident_type": "collision_hazard",
                        "track_id": f"V{v_id}-P{p_id}",
                        "bbox": [min_x, min_y, max_x, max_y],
                        "confidence": conf,
                        "severity": severity,
                        "ttc_seconds": ttc,
                        "distance_px": round(dist_px, 1),
                        "approach_speed_px_sec": round(rel_speed, 1),
                        "vehicle_id": v_id,
                        "pedestrian_id": p_id,
                        "collision_vector": {
                            "vehicle_center": [int(c_veh[0]), int(c_veh[1])],
                            "pedestrian_center": [int(c_ped[0]), int(c_ped[1])],
                            "vehicle_velocity": [round(vel_veh[0], 2), round(vel_veh[1], 2)],
                            "pedestrian_velocity": [round(vel_ped[0], 2), round(vel_ped[1], 2)],
                        },
                        "reasons": [
                            f"Vehicle #{v_id} and Pedestrian #{p_id} on collision intercept trajectory.",
                            f"Dynamic Time-To-Collision (TTC): {ttc}s (Critical alert threshold: {self.critical_ttc_seconds}s).",
                            f"Separation distance: {round(dist_px, 1)} px | Closing speed: {round(rel_speed, 1)} px/s."
                        ],
                        "evidence_metrics": {
                            "ttc_seconds": ttc,
                            "distance_px": round(dist_px, 1),
                            "approach_speed": round(rel_speed, 1),
                            "vehicle_id": v_id,
                            "pedestrian_id": p_id,
                            "collision_imminent": is_imminent,
                        }
                    })

        return alerts

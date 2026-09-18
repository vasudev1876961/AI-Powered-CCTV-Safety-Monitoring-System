"""
Fall & Collapse Incident Detection Module
Monitors spatial-temporal kinematics: rapid downward vertical velocity,
bounding-box aspect ratio inversion (standing to horizontal), and ground dwell persistence.
"""

from typing import Dict, Any, Optional
from SafetySurveillance.tracking.trajectory import TrackHistory


class FallDetector:
    """Detects slip, trip, fall, and sudden collapse incidents."""

    def __init__(
        self,
        velocity_thresh: float = 80.0,      # Pixels per second downward
        aspect_ratio_thresh: float = 1.05,  # w/h > 1.05 indicates horizontal posture
        dwell_stationary_sec: float = 1.5,  # Remained on floor
    ):
        self.velocity_thresh = velocity_thresh
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.dwell_thresh = dwell_stationary_sec

    def evaluate(self, track: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluates track for fall signature.

        Returns:
            Incident alert dict if fall detected, else None.
        """
        if track["class"] != "person":
            return None

        hist: TrackHistory = track["history"]
        if len(hist.centroids) < 5:
            return None

        # 1. Downward vertical velocity
        downward_vy = hist.get_vertical_velocity(window=6)

        # 2. Current aspect ratio
        current_bbox = track["bbox"]
        w = max(1, current_bbox[2] - current_bbox[0])
        h = max(1, current_bbox[3] - current_bbox[1])
        current_ar = w / float(h)

        # 3. Aspect ratio shift (transition from vertical to horizontal)
        ar_shift = hist.get_aspect_ratio_shift(window=10)

        # 4. Stationary dwell time on ground
        dwell_time = hist.get_dwell_time(stationary_threshold_px=35.0)

        # Combined fall confidence heuristic
        is_horizontal = current_ar > self.aspect_ratio_thresh
        has_fallen = (downward_vy > self.velocity_thresh or ar_shift > 0.4) and is_horizontal
        is_collapsed = is_horizontal and dwell_time >= self.dwell_thresh

        if has_fallen or is_collapsed:
            confidence = 0.85
            if is_collapsed and dwell_time > 3.0:
                confidence = 0.96
            elif has_fallen and downward_vy > 120.0:
                confidence = 0.92

            reasons = []
            if downward_vy > self.velocity_thresh:
                reasons.append(f"Sudden downward velocity: {downward_vy:.1f} px/s")
            if ar_shift > 0.3:
                reasons.append(f"Posture shifted from vertical to horizontal (AR Δ: +{ar_shift:.2f})")
            if dwell_time > 0.5:
                reasons.append(f"Person stationary on ground for {dwell_time:.1f}s")

            return {
                "incident_type": "Fall / Collapse",
                "severity": "CRITICAL" if dwell_time > 2.5 else "HIGH",
                "confidence": round(confidence, 3),
                "track_id": track["id"],
                "bbox": track["bbox"],
                "reasons": reasons,
                "evidence_metrics": {
                    "vertical_velocity": round(downward_vy, 1),
                    "aspect_ratio": round(current_ar, 2),
                    "aspect_ratio_shift": round(ar_shift, 2),
                    "ground_dwell_time": round(dwell_time, 1),
                },
            }

        return None

"""
Personal Protective Equipment (PPE) Compliance Module
Detects industrial safety gear compliance (Hardhat/Safety Helmet & High-Visibility Safety Vest)
in industrial, construction, and hazardous warehouse environments.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class PPEComplianceDetector:
    """
    Evaluates worker bounding boxes for mandatory Personal Protective Equipment (PPE).
    Analyzes chromaticity and spectral properties of upper head (helmet) and torso (vest).
    """

    def __init__(
        self,
        helmet_required: bool = True,
        vest_required: bool = True,
        min_worker_height_px: int = 40,
    ):
        self.helmet_required = helmet_required
        self.vest_required = vest_required
        self.min_worker_height_px = min_worker_height_px
        # Worker track audit cache: track_id -> history of violations
        self.worker_audit_cache: Dict[int, Dict[str, Any]] = {}

    def inspect_worker_crop(self, worker_crop: np.ndarray) -> Dict[str, Any]:
        """
        Inspects worker image crop for presence of safety helmet and high-visibility vest.

        Returns:
            Dictionary with has_helmet, has_vest, helmet_color, vest_color, compliance_score
        """
        if worker_crop is None or worker_crop.size == 0:
            return {
                "has_helmet": False,
                "has_vest": False,
                "helmet_color": "NONE",
                "vest_color": "NONE",
                "compliance_score": 0.0,
            }

        ch, cw = worker_crop.shape[:2]
        if ch < 20 or cw < 10:
            return {
                "has_helmet": True,  # Insufficient resolution to penalize
                "has_vest": True,
                "helmet_color": "UNKNOWN",
                "vest_color": "UNKNOWN",
                "compliance_score": 1.0,
            }

        # 1. Head Region (Top 22% of bounding box)
        head_h = max(4, int(ch * 0.22))
        head_crop = worker_crop[0:head_h, :]

        # 2. Torso Region (25% to 65% of bounding box)
        torso_y1 = int(ch * 0.25)
        torso_y2 = int(ch * 0.65)
        torso_crop = worker_crop[torso_y1:torso_y2, :]

        # Evaluate Helmet
        has_helmet, helmet_color = self._detect_helmet(head_crop)

        # Evaluate Vest
        has_vest, vest_color = self._detect_hi_vis_vest(torso_crop)

        # Compliance Score calculation
        req_count = (1 if self.helmet_required else 0) + (1 if self.vest_required else 0)
        passed_count = (1 if (not self.helmet_required or has_helmet) else 0) + \
                       (1 if (not self.vest_required or has_vest) else 0)
        compliance_score = round(passed_count / max(1, req_count), 2)

        return {
            "has_helmet": has_helmet,
            "has_vest": has_vest,
            "helmet_color": helmet_color,
            "vest_color": vest_color,
            "compliance_score": compliance_score,
        }

    def _detect_helmet(self, head_crop: np.ndarray) -> Tuple[bool, str]:
        """Detects standard industrial helmet colors (Yellow, White, Orange, Blue) in head crop."""
        if head_crop.size == 0:
            return False, "NONE"

        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        total_pixels = head_crop.shape[0] * head_crop.shape[1]

        # Yellow Helmet: H: 20-35, S: 80-255, V: 100-255
        yellow_mask = cv2.inRange(hsv, np.array([18, 70, 90]), np.array([35, 255, 255]))
        yellow_ratio = np.count_nonzero(yellow_mask) / max(1, total_pixels)

        # White Helmet: S < 45, V > 180 (high brightness, low saturation)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
        white_ratio = np.count_nonzero(white_mask) / max(1, total_pixels)

        # Orange Helmet: H: 5-18, S: 100-255, V: 110-255
        orange_mask = cv2.inRange(hsv, np.array([5, 100, 110]), np.array([18, 255, 255]))
        orange_ratio = np.count_nonzero(orange_mask) / max(1, total_pixels)

        # Blue Helmet: H: 95-130, S: 80-255, V: 80-255
        blue_mask = cv2.inRange(hsv, np.array([95, 80, 80]), np.array([130, 255, 255]))
        blue_ratio = np.count_nonzero(blue_mask) / max(1, total_pixels)

        if yellow_ratio > 0.12:
            return True, "YELLOW_HARDHAT"
        if white_ratio > 0.16:
            return True, "WHITE_HARDHAT"
        if orange_ratio > 0.12:
            return True, "ORANGE_HARDHAT"
        if blue_ratio > 0.14:
            return True, "BLUE_HARDHAT"

        return False, "NO_HELMET"

    def _detect_hi_vis_vest(self, torso_crop: np.ndarray) -> Tuple[bool, str]:
        """Detects fluorescent neon yellow/green or safety orange high-vis vest."""
        if torso_crop.size == 0:
            return False, "NONE"

        hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        total_pixels = torso_crop.shape[0] * torso_crop.shape[1]

        # Fluorescent Neon Lime/Yellow Vest: H: 25-45, S: 100-255, V: 110-255
        neon_mask = cv2.inRange(hsv, np.array([25, 90, 110]), np.array([45, 255, 255]))
        neon_ratio = np.count_nonzero(neon_mask) / max(1, total_pixels)

        # Fluorescent Safety Orange Vest: H: 8-22, S: 110-255, V: 120-255
        orange_mask = cv2.inRange(hsv, np.array([8, 110, 120]), np.array([22, 255, 255]))
        orange_ratio = np.count_nonzero(orange_mask) / max(1, total_pixels)

        if neon_ratio > 0.14:
            return True, "NEON_YELLOW_VEST"
        if orange_ratio > 0.14:
            return True, "SAFETY_ORANGE_VEST"

        return False, "NO_VEST"

    def evaluate(
        self,
        active_tracks: List[Dict[str, Any]],
        frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Evaluates all active person tracks in current frame for PPE non-compliance.

        Returns:
            List of PPE violation alerts.
        """
        alerts = []
        if frame is None or frame.size == 0:
            return alerts

        fh, fw = frame.shape[:2]

        for t in active_tracks:
            if t.get("class") != "person":
                continue

            bbox = t["bbox"]
            x1, y1, x2, y2 = bbox
            # Clamp bounds
            x1 = max(0, min(fw - 1, int(x1)))
            y1 = max(0, min(fh - 1, int(y1)))
            x2 = max(x1 + 1, min(fw, int(x2)))
            y2 = max(y1 + 1, min(fh, int(y2)))

            w = x2 - x1
            h = y2 - y1

            if h < self.min_worker_height_px:
                continue

            crop = frame[y1:y2, x1:x2]
            ppe_res = self.inspect_worker_crop(crop)

            track_id = t["id"]
            # Track compliance smoothing in cache
            if track_id not in self.worker_audit_cache:
                self.worker_audit_cache[track_id] = {
                    "violations": 0,
                    "audits": 0,
                    "last_status": ppe_res,
                }

            cache = self.worker_audit_cache[track_id]
            cache["audits"] += 1
            cache["last_status"] = ppe_res

            # Violation trigger condition
            missing_helmet = self.helmet_required and not ppe_res["has_helmet"]
            missing_vest = self.vest_required and not ppe_res["has_vest"]

            if missing_helmet or missing_vest:
                cache["violations"] += 1
            else:
                cache["violations"] = max(0, cache["violations"] - 1)

            # Raise alert once persistence threshold reached (at least 3 frames)
            if cache["violations"] >= 3:
                missing_items = []
                if missing_helmet:
                    missing_items.append("Mandatory Safety Helmet / Hardhat")
                if missing_vest:
                    missing_items.append("High-Visibility Safety Vest")

                severity = "CRITICAL" if (missing_helmet and missing_vest) else "HIGH"
                conf = 0.88 if len(missing_items) == 2 else 0.82

                alerts.append({
                    "incident_type": "ppe_violation",
                    "track_id": track_id,
                    "bbox": bbox,
                    "confidence": conf,
                    "severity": severity,
                    "missing_ppe": missing_items,
                    "compliance_score": ppe_res["compliance_score"],
                    "reasons": [
                        f"Worker #{track_id} in designated safety zone without required PPE: {', '.join(missing_items)}.",
                        f"Spectral inspection: Helmet={ppe_res['helmet_color']}, Vest={ppe_res['vest_color']}.",
                        f"Compliance index: {int(ppe_res['compliance_score'] * 100)}% (Standard threshold: 100%)."
                    ],
                    "evidence_metrics": {
                        "track_id": track_id,
                        "has_helmet": ppe_res["has_helmet"],
                        "has_vest": ppe_res["has_vest"],
                        "helmet_color": ppe_res["helmet_color"],
                        "vest_color": ppe_res["vest_color"],
                        "compliance_score": ppe_res["compliance_score"],
                        "consecutive_violation_frames": cache["violations"],
                    }
                })

        return alerts

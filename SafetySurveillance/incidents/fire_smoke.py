"""
Fire & Smoke Hazard Early Warning Detection Module
Detects early chromatic flame signatures and atmospheric smoke diffusion
in poor CCTV environments using YCrCb/HSV color dynamics and flicker variance.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List


class FireSmokeDetector:
    """Detects optical smoke plumes and active flame signatures in CCTV streams."""

    def __init__(self, min_area_px: int = 400, flicker_history_len: int = 10):
        self.min_area_px = min_area_px
        self.flicker_history: List[float] = []
        self.max_flicker_len = flicker_history_len

    def evaluate(self, frame: np.ndarray, frame_diff: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """
        Analyzes video frame for chromatic flame or diffuse smoke signatures.

        Args:
            frame: Current BGR image frame.
            frame_diff: Optional motion difference frame.

        Returns:
            Alert dictionary if fire/smoke signature is recognized, else None.
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        # Downsample for fast real-time optical analysis
        small = cv2.resize(frame, (320, int(320 * (h / w))))
        sh, sw = small.shape[:2]

        # 1. Flame Detection in YCrCb color space
        # Rules: Y > Cb, Cr > Cb, Cr > 140, Y > 120
        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        y_chan, cr_chan, cb_chan = cv2.split(ycrcb)

        flame_mask = (cr_chan > 140) & (cr_chan > cb_chan) & (y_chan > cb_chan) & (y_chan > 110)
        flame_pixels = int(np.sum(flame_mask))

        # 2. Smoke Detection in HSV color space
        # Smoke characteristic: Low saturation, mid-to-high value, diffuse texture
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv)

        smoke_mask = (s_chan < 55) & (v_chan > 125) & (v_chan < 235)
        smoke_pixels = int(np.sum(smoke_mask))

        # Update flicker history
        current_flame_intensity = flame_pixels / float(sh * sw)
        self.flicker_history.append(current_flame_intensity)
        if len(self.flicker_history) > self.max_flicker_len:
            self.flicker_history.pop(0)

        flicker_var = float(np.var(self.flicker_history)) if len(self.flicker_history) > 3 else 0.0

        scale_x = w / float(sw)
        scale_y = h / float(sh)

        # Check Flame Trigger
        if flame_pixels >= (self.min_area_px * 0.3):
            # Extract bounding contour
            mask_uint8 = (flame_mask.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_c = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(largest_c)
                orig_bbox = [
                    int(x * scale_x),
                    int(y * scale_y),
                    int((x + bw) * scale_x),
                    int((y + bh) * scale_y),
                ]
                conf = min(0.98, round(0.75 + (flame_pixels / 3000.0) * 0.20 + (flicker_var * 2.0), 3))
                return {
                    "incident_type": "Fire & Flame Hazard",
                    "severity": "CRITICAL",
                    "confidence": conf,
                    "bbox": orig_bbox,
                    "reasons": [
                        f"Active optical combustion signature detected ({flame_pixels} px hot-spot)",
                        f"High Cr/Cb chromatic divergence indicative of open flame",
                        f"Temporal luminance flicker variance: {flicker_var:.4f}",
                    ],
                    "evidence_metrics": {
                        "flame_pixel_count": flame_pixels,
                        "flicker_variance": round(flicker_var, 5),
                        "hazard_classification": "OPEN_FLAME",
                    },
                }

        # Check Smoke Trigger
        if smoke_pixels >= (self.min_area_px * 1.8):
            mask_uint8 = (smoke_mask.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_c = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(largest_c)
                orig_bbox = [
                    int(x * scale_x),
                    int(y * scale_y),
                    int((x + bw) * scale_x),
                    int((y + bh) * scale_y),
                ]
                conf = min(0.92, round(0.68 + (smoke_pixels / 8000.0) * 0.22, 3))
                return {
                    "incident_type": "Smoke / Atmospheric Combustion Plume",
                    "severity": "HIGH",
                    "confidence": conf,
                    "bbox": orig_bbox,
                    "reasons": [
                        f"Diffuse low-saturation smoke plume signature detected ({smoke_pixels} px cluster)",
                        "Localized contrast attenuation and luminance obscuration",
                        "Early stage fire emission or hazardous aerosol release",
                    ],
                    "evidence_metrics": {
                        "smoke_pixel_count": smoke_pixels,
                        "hazard_classification": "SMOKE_PLUME",
                    },
                }

        return None

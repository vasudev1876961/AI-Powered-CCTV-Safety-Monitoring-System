"""
CCTV Quality Assessment Module
Quantifies video degradation factors (brightness, blur, noise, contrast)
and classifies the frame state for adaptive enhancement.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple


class CCTVQualityEstimator:
    """
    Evaluates real-time quality metrics on input CCTV video frames.
    Classifies frames into: GOOD, MODERATE, DEGRADED, SEVERELY DEGRADED.
    """

    def __init__(
        self,
        low_light_thresh: float = 50.0,
        severe_low_light: float = 25.0,
        blur_laplacian_thresh: float = 120.0,
        severe_blur_thresh: float = 45.0,
        noise_std_thresh: float = 12.0,
        contrast_rms_thresh: float = 30.0,
    ):
        self.low_light_thresh = low_light_thresh
        self.severe_low_light = severe_low_light
        self.blur_thresh = blur_laplacian_thresh
        self.severe_blur_thresh = severe_blur_thresh
        self.noise_std_thresh = noise_std_thresh
        self.contrast_rms_thresh = contrast_rms_thresh

    def assess_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Calculates all quality metrics for a single frame.

        Args:
            frame: BGR numpy image array.

        Returns:
            Dictionary containing metrics, classification, quality factor, and enhancement triggers.
        """
        if frame is None or frame.size == 0:
            return {
                "quality_state": "SEVERELY DEGRADED",
                "quality_factor": 0.1,
                "brightness": 0.0,
                "blur_score": 0.0,
                "noise_level": 100.0,
                "contrast": 0.0,
                "needs_low_light": True,
                "needs_denoise": True,
                "needs_deblur": True,
            }

        # Convert to grayscale for metric calculations
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Performance Optimization: Downscale proxy for sub-5ms Laplacian and median blur
        # while keeping full numerical fidelity for surveillance classification
        h, w = gray.shape[:2]
        proxy_w = 480
        if w > proxy_w:
            scale = proxy_w / float(w)
            proxy_h = max(1, int(h * scale))
            proxy_gray = cv2.resize(gray, (proxy_w, proxy_h), interpolation=cv2.INTER_AREA)
        else:
            proxy_gray = gray
            scale = 1.0

        # 1. Brightness / Illumination: Mean and 10th percentile
        mean_brightness = float(cv2.mean(gray)[0])
        p10_brightness = float(np.percentile(proxy_gray, 10))

        # 2. Blur Estimation: Variance of the Laplacian using fast single-precision CV_32F
        # Normalize by scale factor squared to maintain calibrated blur thresholds
        laplacian = cv2.Laplacian(proxy_gray, cv2.CV_32F)
        blur_score = float(laplacian.var()) * (1.0 / max(0.01, scale * scale))

        # 3. Noise Estimation: High-pass residual from fast 3x3 median filter on proxy
        median_filtered = cv2.medianBlur(proxy_gray, 3)
        noise_residual = cv2.absdiff(proxy_gray, median_filtered)
        noise_level = float(np.std(noise_residual))

        # 4. Contrast: Root Mean Square (RMS) contrast
        contrast = float(np.std(proxy_gray))

        # Degradation flags
        is_low_light = mean_brightness < self.low_light_thresh
        is_severe_low_light = mean_brightness < self.severe_low_light
        is_blurry = blur_score < self.blur_thresh
        is_severe_blurry = blur_score < self.severe_blur_thresh
        is_noisy = noise_level > self.noise_std_thresh
        is_low_contrast = contrast < self.contrast_rms_thresh

        # Count degradation points
        deg_score = 0
        if is_low_light:
            deg_score += 2 if is_severe_low_light else 1
        if is_blurry:
            deg_score += 2 if is_severe_blurry else 1
        if is_noisy:
            deg_score += 1
        if is_low_contrast:
            deg_score += 1

        # Classify state
        if deg_score == 0:
            quality_state = "GOOD"
            quality_factor = 0.95
        elif deg_score <= 2:
            quality_state = "MODERATE"
            quality_factor = 0.75
        elif deg_score <= 4:
            quality_state = "DEGRADED"
            quality_factor = 0.50
        else:
            quality_state = "SEVERELY DEGRADED"
            quality_factor = 0.25

        return {
            "quality_state": quality_state,
            "quality_factor": quality_factor,
            "brightness": round(mean_brightness, 2),
            "p10_brightness": round(p10_brightness, 2),
            "blur_score": round(blur_score, 2),
            "noise_level": round(noise_level, 2),
            "contrast": round(contrast, 2),
            "needs_low_light": is_low_light,
            "needs_denoise": is_noisy,
            "needs_deblur": is_blurry,
            "needs_contrast": is_low_contrast,
        }

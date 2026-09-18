"""
Denoising Module for Degraded CCTV Footage
Implements edge-preserving Bilateral Filtering, Non-Local Means,
and Median filtering for sensor noise in low-light surveillance.
"""

import cv2
import numpy as np


class FrameDenoiser:
    """Removes high-frequency camera noise while preserving object boundaries."""

    def __init__(self, bilateral_d: int = 7, sigma_color: float = 50.0, sigma_space: float = 50.0):
        self.bilateral_d = bilateral_d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def bilateral_denoise(self, frame: np.ndarray) -> np.ndarray:
        """
        Bilateral filtering preserves crisp edges of moving subjects
        while removing sensor thermal noise.
        """
        return cv2.bilateralFilter(frame, self.bilateral_d, self.sigma_color, self.sigma_space)

    def fast_nlm_denoise(self, frame: np.ndarray, h: float = 10.0) -> np.ndarray:
        """Fast Non-Local Means denoising for severe noise."""
        if len(frame.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(frame, None, h, h, 7, 21)
        return cv2.fastNlMeansDenoising(frame, None, h, 7, 21)

    def median_denoise(self, frame: np.ndarray, ksize: int = 3) -> np.ndarray:
        """Median filter for impulse / salt-and-pepper noise."""
        return cv2.medianBlur(frame, ksize)

    def denoise(self, frame: np.ndarray, noise_level: float = 15.0) -> np.ndarray:
        """Adaptively selects filter strength based on measured noise standard deviation."""
        if noise_level > 25.0:
            return self.fast_nlm_denoise(frame, h=12.0)
        elif noise_level > 12.0:
            return self.bilateral_denoise(frame)
        return frame

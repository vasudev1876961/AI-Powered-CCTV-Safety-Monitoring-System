"""
Low-Light Image Enhancement Module
Implements adaptive CLAHE in LAB space, dynamic Gamma correction,
and Multi-Scale Retinex (MSR) for dark surveillance feeds.
"""

import cv2
import numpy as np


class LowLightEnhancer:
    """Enhances poorly illuminated CCTV frames adaptively."""

    def __init__(self, clip_limit: float = 3.0, grid_size: tuple = (8, 8)):
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhances luminance using Contrast Limited Adaptive Histogram Equalization (CLAHE)
        in the CIE LAB color space to prevent color distortion.
        """
        if len(frame.shape) == 2:
            return self.clahe.apply(frame)

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = self.clahe.apply(l_channel)
        merged_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    def apply_gamma(self, frame: np.ndarray, gamma: float = None) -> np.ndarray:
        """
        Dynamic power-law gamma correction:
        If gamma is None, automatically computes gamma based on frame mean brightness.
        """
        if gamma is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            mean_val = np.mean(gray) / 255.0
            mean_val = np.clip(mean_val, 0.05, 0.95)
            # When mean_val < 0.5 (dark), gamma < 1 (brightens image)
            gamma = float(np.log(0.5) / np.log(mean_val))
            gamma = np.clip(gamma, 0.4, 2.2)

        # When gamma < 1 (dark image), applying power of gamma brightens pixels
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(frame, table)

    def apply_retinex(self, frame: np.ndarray, sigmas: list = [15, 80, 250]) -> np.ndarray:
        """
        Multi-Scale Retinex (MSR) algorithm for extreme dynamic range compression
        in pitch-black or backlit environments.
        """
        img_float = frame.astype(np.float64) + 1.0
        retinex = np.zeros_like(img_float)

        for sigma in sigmas:
            blur = cv2.GaussianBlur(img_float, (0, 0), sigma) + 1.0
            retinex += np.log10(img_float) - np.log10(blur)

        retinex = retinex / len(sigmas)

        # Normalize to [0, 255]
        for i in range(frame.shape[2] if len(frame.shape) == 3 else 1):
            channel = retinex[:, :, i] if len(frame.shape) == 3 else retinex
            c_min = np.percentile(channel, 1)
            c_max = np.percentile(channel, 99)
            channel = np.clip((channel - c_min) / (c_max - c_min + 1e-6) * 255.0, 0, 255)
            if len(frame.shape) == 3:
                retinex[:, :, i] = channel
            else:
                retinex = channel

        return retinex.astype(np.uint8)

    def enhance(self, frame: np.ndarray, mode: str = "adaptive") -> np.ndarray:
        """Pipeline enhancement combining Gamma + CLAHE."""
        enhanced = self.apply_gamma(frame)
        enhanced = self.apply_clahe(enhanced)
        return enhanced

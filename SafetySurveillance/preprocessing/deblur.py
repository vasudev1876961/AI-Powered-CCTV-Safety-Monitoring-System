"""
Deblurring & Edge Sharpening Module
Recovers blurred CCTV video resulting from camera shake,
defocus, or fast subject motion.
"""

import cv2
import numpy as np


class FrameDeblurrer:
    """Sharpens and restores edges in blurred CCTV frames."""

    def __init__(self, unsharp_sigma: float = 1.0, unsharp_strength: float = 1.5):
        self.unsharp_sigma = unsharp_sigma
        self.unsharp_strength = unsharp_strength

    def unsharp_mask(
        self,
        frame: np.ndarray,
        sigma: float = None,
        strength: float = None
    ) -> np.ndarray:
        """
        Unsharp Masking:
        I_sharp = I + strength * (I - GaussianBlur(I))
        """
        s = self.unsharp_sigma if sigma is None else sigma
        k = self.unsharp_strength if strength is None else strength

        blurred = cv2.GaussianBlur(frame, (0, 0), s)
        sharpened = cv2.addWeighted(frame, 1.0 + k, blurred, -k, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def laplacian_sharpen(self, frame: np.ndarray) -> np.ndarray:
        """Kernel-based high-pass boost for fast CPU inference."""
        kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)
        return cv2.filter2D(frame, -1, kernel)

    def deblur(self, frame: np.ndarray, blur_score: float = 80.0) -> np.ndarray:
        """Adaptively sharpens frame if blur index falls below threshold."""
        if blur_score < 50.0:
            # Severe blur: stronger unsharp mask
            return self.unsharp_mask(frame, sigma=1.5, strength=1.8)
        elif blur_score < 120.0:
            # Mild blur
            return self.unsharp_mask(frame, sigma=1.0, strength=1.2)
        return frame

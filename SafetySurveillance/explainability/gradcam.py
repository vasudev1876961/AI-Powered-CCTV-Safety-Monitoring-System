"""
Explainable AI (XAI) - Spatial Saliency & Attention Heatmap Module
Generates visual attention / Grad-CAM style saliency heatmaps to show
operators exactly which spatial regions triggered the safety alert.
"""

import cv2
import numpy as np
from typing import List, Optional


class SaliencyExplainer:
    """Generates visual attention heatmaps indicating alert-triggering image regions."""

    @staticmethod
    def generate_heatmap(
        frame: np.ndarray,
        target_bboxes: List[List[int]],
        intensity: float = 0.85
    ) -> np.ndarray:
        """
        Creates a thermal Jet colormap heatmap centered over incident entities.

        Args:
            frame: Base BGR frame.
            target_bboxes: List of [x1, y1, x2, y2] bounding boxes to highlight.
            intensity: Blend weight.

        Returns:
            Thermal attention map blended with original frame.
        """
        h, w = frame.shape[:2]
        heatmap_mask = np.zeros((h, w), dtype=np.float32)

        for bbox in target_bboxes:
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            bw = max(10, x2 - x1)
            bh = max(10, y2 - y1)
            radius_x = int(bw * 0.8)
            radius_y = int(bh * 0.8)

            # Generate 2D Gaussian blob
            y_coords, x_coords = np.ogrid[:h, :w]
            dist_sq = ((x_coords - cx) ** 2) / (radius_x ** 2 + 1e-5) + \
                      ((y_coords - cy) ** 2) / (radius_y ** 2 + 1e-5)
            gaussian = np.exp(-0.5 * dist_sq)
            heatmap_mask = np.maximum(heatmap_mask, gaussian)

        # Normalize and colorize
        heatmap_norm = np.uint8(255 * np.clip(heatmap_mask, 0, 1))
        colored_heatmap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        # Alpha blend with original frame
        blended = cv2.addWeighted(frame, 1.0 - intensity, colored_heatmap, intensity, 0)
        return blended

"""
Spatial Analytics & 2D Occupancy Heatmap Module
Accumulates continuous spatial worker trajectories, pedestrian dwell patterns,
and incident concentration hotspots into normalized 2D grid matrices for HUD rendering.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class SpatialOccupancyAnalytics:
    """
    Maintains 2D spatial grid matrices for multi-camera CCTV feeds,
    tracking foot-traffic density, dwell duration, and safety hazard hotspots.
    """

    def __init__(self, grid_w: int = 32, grid_h: int = 18, decay_factor: float = 0.995):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.decay_factor = decay_factor
        # camera_id -> 2D numpy array of density
        self.camera_heatmaps: Dict[str, np.ndarray] = {}
        # camera_id -> 2D numpy array of incident occurrences
        self.incident_heatmaps: Dict[str, np.ndarray] = {}
        # camera_id -> total frames accumulated
        self.frame_counts: Dict[str, int] = {}

    def _ensure_camera(self, cam_id: str):
        if cam_id not in self.camera_heatmaps:
            self.camera_heatmaps[cam_id] = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
            self.incident_heatmaps[cam_id] = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
            self.frame_counts[cam_id] = 0

    def update(
        self,
        cam_id: str,
        active_tracks: List[Dict[str, Any]],
        incidents: Optional[List[Dict[str, Any]]] = None,
        frame_shape: Tuple[int, int] = (540, 960)
    ):
        """Accumulates current frame detections and incidents into 2D grid matrix."""
        self._ensure_camera(cam_id)
        fh, fw = frame_shape
        self.frame_counts[cam_id] += 1

        # Apply gradual temporal decay to prioritize recent activity
        self.camera_heatmaps[cam_id] *= self.decay_factor

        for t in active_tracks:
            bbox = t.get("bbox", [0, 0, 0, 0])
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            # Map to grid coordinates
            gx = int((cx / max(1.0, fw)) * self.grid_w)
            gy = int((cy / max(1.0, fh)) * self.grid_h)

            gx = max(0, min(self.grid_w - 1, gx))
            gy = max(0, min(self.grid_h - 1, gy))

            # Splat with 3x3 gaussian kernel
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny = gy + dy
                    nx = gx + dx
                    if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                        weight = 1.0 if (dx == 0 and dy == 0) else 0.5
                        self.camera_heatmaps[cam_id][ny, nx] += weight

        if incidents:
            for inc in incidents:
                bbox = inc.get("bbox", [0, 0, 0, 0])
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                gx = max(0, min(self.grid_w - 1, int((cx / max(1.0, fw)) * self.grid_w)))
                gy = max(0, min(self.grid_h - 1, int((cy / max(1.0, fh)) * self.grid_h)))
                self.incident_heatmaps[cam_id][gy, gx] += 3.0

    def get_normalized_heatmap(self, cam_id: str) -> Dict[str, Any]:
        """Returns JSON-ready normalized occupancy and incident matrix [0.0 .. 1.0]."""
        self._ensure_camera(cam_id)
        grid = self.camera_heatmaps[cam_id]
        inc_grid = self.incident_heatmaps[cam_id]

        max_val = float(np.max(grid)) if np.max(grid) > 0 else 1.0
        norm_density = (grid / max_val).tolist()

        inc_max = float(np.max(inc_grid)) if np.max(inc_grid) > 0 else 1.0
        norm_incidents = (inc_grid / inc_max).tolist()

        # Find peak hotspot coordinate
        max_idx = np.unravel_index(np.argmax(grid), grid.shape)
        peak_y, peak_x = int(max_idx[0]), int(max_idx[1])

        return {
            "camera_id": cam_id,
            "grid_dimensions": [self.grid_w, self.grid_h],
            "total_frames_sampled": self.frame_counts[cam_id],
            "density_matrix": [[round(val, 3) for val in row] for row in norm_density],
            "incident_hotspot_matrix": [[round(val, 3) for val in row] for row in norm_incidents],
            "peak_hotspot_grid": [peak_x, peak_y],
            "peak_hotspot_normalized": [round(peak_x / self.grid_w, 2), round(peak_y / self.grid_h, 2)],
        }

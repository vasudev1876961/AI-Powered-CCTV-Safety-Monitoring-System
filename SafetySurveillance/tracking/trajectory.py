"""
Trajectory & Kinematic Analysis Module
Maintains track history, temporal centroids, velocity vectors,
aspect-ratio dynamics, and spatial dwell time.
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Any


class TrackHistory:
    """Stores and analyzes temporal trajectory data for an individual tracked entity."""

    def __init__(self, track_id: int, max_history: int = 60):
        self.track_id = track_id
        self.max_history = max_history
        self.centroids: List[Tuple[float, float]] = []
        self.bboxes: List[List[int]] = []
        self.timestamps: List[float] = []
        self.velocities: List[Tuple[float, float]] = []  # (vx, vy) in px/sec
        self.aspect_ratios: List[float] = []  # width / height

    def update(self, bbox: List[int], timestamp: float = None):
        """Appends new observation and computes kinematics."""
        t = timestamp if timestamp is not None else time.time()
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        ar = w / float(h)

        if self.timestamps:
            dt = max(1e-3, t - self.timestamps[-1])
            prev_cx, prev_cy = self.centroids[-1]
            vx = (cx - prev_cx) / dt
            vy = (cy - prev_cy) / dt
        else:
            vx, vy = 0.0, 0.0

        self.centroids.append((cx, cy))
        self.bboxes.append(bbox)
        self.timestamps.append(t)
        self.velocities.append((vx, vy))
        self.aspect_ratios.append(ar)

        # Truncate to max history
        if len(self.centroids) > self.max_history:
            self.centroids.pop(0)
            self.bboxes.pop(0)
            self.timestamps.pop(0)
            self.velocities.pop(0)
            self.aspect_ratios.pop(0)

    def get_vertical_velocity(self, window: int = 5) -> float:
        """Returns average vertical velocity over recent window (positive is downward in image coords)."""
        if len(self.velocities) < 2:
            return 0.0
        recent = [v[1] for v in self.velocities[-window:]]
        return float(np.mean(recent))

    def get_aspect_ratio_shift(self, window: int = 8) -> float:
        """
        Returns recent aspect ratio shift:
        Positive values indicate transition from vertical (standing) to horizontal (lying down).
        """
        if len(self.aspect_ratios) < window:
            return 0.0
        initial_ar = np.mean(self.aspect_ratios[-window:-window // 2])
        current_ar = np.mean(self.aspect_ratios[-window // 2:])
        return float(current_ar - initial_ar)

    def get_displacement(self, seconds: float = 3.0) -> float:
        """Total Euclidean displacement over specified past duration."""
        if len(self.timestamps) < 2:
            return 0.0
        now = self.timestamps[-1]
        cutoff = now - seconds
        idx = 0
        for i, t in enumerate(self.timestamps):
            if t >= cutoff:
                idx = i
                break
        start_pt = self.centroids[idx]
        end_pt = self.centroids[-1]
        return float(np.sqrt((end_pt[0] - start_pt[0]) ** 2 + (end_pt[1] - start_pt[1]) ** 2))

    def get_dwell_time(self, stationary_threshold_px: float = 30.0) -> float:
        """Computes how many seconds the track has remained within a stationary radius."""
        if len(self.centroids) < 2:
            return 0.0
        current_pt = self.centroids[-1]
        stationary_time = 0.0
        for i in range(len(self.centroids) - 2, -1, -1):
            pt = self.centroids[i]
            dist = np.sqrt((current_pt[0] - pt[0]) ** 2 + (current_pt[1] - pt[1]) ** 2)
            if dist <= stationary_threshold_px:
                dt = self.timestamps[i + 1] - self.timestamps[i]
                stationary_time += dt
            else:
                break
        return stationary_time

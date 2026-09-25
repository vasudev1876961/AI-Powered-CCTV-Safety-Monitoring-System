"""
Multi-Object Tracker Module
Implements IoU-based persistent track management inspired by ByteTrack,
assigning persistent IDs and maintaining trajectories across video frames.
"""

import time
import numpy as np
from typing import List, Dict, Any
from SafetySurveillance.tracking.trajectory import TrackHistory


def compute_iou(boxA: List[int], boxB: List[int]) -> float:
    """Computes Intersection over Union between two [x1, y1, x2, y2] bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def batch_iou(boxesA: np.ndarray, boxesB: np.ndarray) -> np.ndarray:
    """
    Vectorized calculation of IoU between N boxes in boxesA and M boxes in boxesB.
    Args:
        boxesA: np.ndarray of shape (N, 4) with [x1, y1, x2, y2]
        boxesB: np.ndarray of shape (M, 4) with [x1, y1, x2, y2]
    Returns:
        np.ndarray of shape (N, M) containing IoU values in [0, 1].
    """
    if boxesA.size == 0 or boxesB.size == 0:
        return np.zeros((len(boxesA), len(boxesB)), dtype=np.float32)

    xA = np.maximum(boxesA[:, None, 0], boxesB[None, :, 0])
    yA = np.maximum(boxesA[:, None, 1], boxesB[None, :, 1])
    xB = np.minimum(boxesA[:, None, 2], boxesB[None, :, 2])
    yB = np.minimum(boxesA[:, None, 3], boxesB[None, :, 3])

    inter = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    areaA = np.maximum(1.0, (boxesA[:, 2] - boxesA[:, 0]) * (boxesA[:, 3] - boxesA[:, 1]))
    areaB = np.maximum(1.0, (boxesB[:, 2] - boxesB[:, 0]) * (boxesB[:, 3] - boxesB[:, 1]))
    union = areaA[:, None] + areaB[None, :] - inter

    return (inter / np.maximum(1.0, union)).astype(np.float32)


class MultiObjectTracker:
    """Lightweight and robust multi-object tracker for safety CCTV monitoring."""

    def __init__(self, iou_thresh: float = 0.3, max_lost_frames: int = 30):
        self.iou_thresh = iou_thresh
        self.max_lost_frames = max_lost_frames
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}  # id -> {bbox, class, conf, lost_frames, history}

    def update(self, detections: List[Dict[str, Any]], timestamp: float = None) -> List[Dict[str, Any]]:
        """
        Associates incoming detections with existing tracks using vectorized IoU and velocity prediction.

        Args:
            detections: List of {'bbox': [x1, y1, x2, y2], 'class': str, 'conf': float}
            timestamp: Observation time

        Returns:
            List of active tracks with persistent IDs and trajectory history.
        """
        now = timestamp if timestamp is not None else time.time()
        track_ids = list(self.tracks.keys())
        det_indices = list(range(len(detections)))

        matched_tracks = set()
        matched_dets = set()

        if track_ids and det_indices:
            # Prepare vectorized arrays
            static_boxes = np.array([self.tracks[tid]["bbox"] for tid in track_ids], dtype=np.float32)
            pred_boxes = np.array([
                self.tracks[tid]["history"].get_predicted_bbox(
                    dt=max(0.01, min(0.5, now - self.tracks[tid].get("last_timestamp", now)))
                )
                for tid in track_ids
            ], dtype=np.float32)
            det_boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)

            # High-speed vectorized batch IoU
            iou_static = batch_iou(static_boxes, det_boxes)
            iou_pred = batch_iou(pred_boxes, det_boxes)
            iou_matrix = np.maximum(iou_static, iou_pred)

            # Class consistency prior vectorized mask
            track_classes = np.array([self.tracks[tid]["class"] for tid in track_ids])
            det_classes = np.array([d["class"] for d in detections])
            match_class_mask = (track_classes[:, None] == det_classes[None, :])
            iou_matrix = np.where(match_class_mask, np.minimum(1.0, iou_matrix + 0.05), iou_matrix * 0.5)

            # Greedy bipartite matching
            while True:
                max_iou = np.max(iou_matrix) if iou_matrix.size > 0 else 0
                if max_iou < self.iou_thresh:
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[i]
                d = detections[det_indices[j]]

                # Update matched track
                self.tracks[tid]["bbox"] = d["bbox"]
                self.tracks[tid]["conf"] = d["conf"]
                self.tracks[tid]["class"] = d["class"]
                self.tracks[tid]["lost_frames"] = 0
                self.tracks[tid]["last_timestamp"] = now
                self.tracks[tid]["history"].update(d["bbox"], now)

                matched_tracks.add(tid)
                matched_dets.add(j)

                # Clear matched row and column
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

        # Increment lost frames for unmatched tracks
        lost_to_remove = []
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid]["lost_frames"] += 1
                if self.tracks[tid]["lost_frames"] > self.max_lost_frames:
                    lost_to_remove.append(tid)

        for tid in lost_to_remove:
            del self.tracks[tid]

        # Initialize new tracks for unmatched detections
        for j, d in enumerate(detections):
            if j not in matched_dets:
                new_tid = self.next_id
                self.next_id += 1
                hist = TrackHistory(new_tid)
                hist.update(d["bbox"], now)
                self.tracks[new_tid] = {
                    "id": new_tid,
                    "bbox": d["bbox"],
                    "class": d["class"],
                    "conf": d["conf"],
                    "lost_frames": 0,
                    "history": hist,
                }

        # Return list of active tracks
        active_tracks = []
        for tid, tdata in self.tracks.items():
            if tdata["lost_frames"] == 0:
                active_tracks.append({
                    "id": tid,
                    "bbox": tdata["bbox"],
                    "class": tdata["class"],
                    "conf": tdata["conf"],
                    "history": tdata["history"],
                    "trajectory": list(tdata["history"].centroids),
                })
        return active_tracks

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


class MultiObjectTracker:
    """Lightweight and robust multi-object tracker for safety CCTV monitoring."""

    def __init__(self, iou_thresh: float = 0.3, max_lost_frames: int = 30):
        self.iou_thresh = iou_thresh
        self.max_lost_frames = max_lost_frames
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}  # id -> {bbox, class, conf, lost_frames, history}

    def update(self, detections: List[Dict[str, Any]], timestamp: float = None) -> List[Dict[str, Any]]:
        """
        Associates incoming detections with existing tracks.

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
            # Build IoU cost matrix with velocity-guided motion prediction
            iou_matrix = np.zeros((len(track_ids), len(det_indices)), dtype=np.float32)
            for i, tid in enumerate(track_ids):
                tdata = self.tracks[tid]
                dt = max(0.01, min(0.5, now - tdata.get("last_timestamp", now)))
                pred_bbox = tdata["history"].get_predicted_bbox(dt=dt)

                for j, d_idx in enumerate(det_indices):
                    d_bbox = detections[d_idx]["bbox"]
                    # Calculate IoU with both last known and velocity-predicted positions
                    iou_static = compute_iou(tdata["bbox"], d_bbox)
                    iou_pred = compute_iou(pred_bbox, d_bbox)
                    best_iou = max(iou_static, iou_pred)

                    # Class consistency prior
                    if tdata["class"] == detections[d_idx]["class"]:
                        best_iou = min(1.0, best_iou + 0.05)
                    else:
                        best_iou *= 0.5  # Penalize class mismatch

                    iou_matrix[i, j] = best_iou

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

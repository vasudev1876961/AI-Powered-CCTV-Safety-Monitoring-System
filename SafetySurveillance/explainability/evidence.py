"""
Explainable AI (XAI) - Evidence Aggregator Module
Assembles key-frame temporal sequences, kinematic trajectory logs,
and structured natural-language causal explanations for human operators.
"""

import time
import base64
import cv2
import numpy as np
from typing import List, Dict, Any, Deque
from collections import deque


class EvidenceRecorder:
    """Maintains a rolling buffer of recent frames and generates forensic evidence packs."""

    def __init__(self, buffer_size: int = 90):  # ~3-4 seconds of video
        self.frame_buffer: Deque[Dict[str, Any]] = deque(maxlen=buffer_size)

    def record_frame(self, frame: np.ndarray, frame_id: int, timestamp: float = None):
        """Buffers current frame thumbnail and metadata."""
        now = timestamp or time.time()
        # Downsample thumbnail for lightweight storage
        h, w = frame.shape[:2]
        thumb = cv2.resize(frame, (min(320, w), int(min(320, w) * (h / w))))
        self.frame_buffer.append({
            "frame_id": frame_id,
            "timestamp": now,
            "frame": thumb,
        })

    def assemble_evidence_pack(
        self,
        alert: Dict[str, Any],
        camera_id: str,
        quality_metrics: Dict[str, Any],
        risk_score: float
    ) -> Dict[str, Any]:
        """
        Builds complete forensic evidence package for an incident.

        Returns:
            Evidence package dictionary with key-frames, trajectory, and explanations.
        """
        buffer_len = len(self.frame_buffer)
        key_frames = []

        if buffer_len > 0:
            indices = [
                0,                                  # T-minus
                max(0, buffer_len // 2),            # Approaching
                buffer_len - 1                      # Trigger moment
            ]
            labels = ["Pre-Incident", "Movement Transition", "Incident Trigger"]

            for idx, label in zip(indices, labels):
                item = self.frame_buffer[idx]
                _, enc = cv2.imencode('.jpg', item["frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                b64_str = base64.b64encode(enc).decode('utf-8')
                key_frames.append({
                    "label": label,
                    "frame_id": item["frame_id"],
                    "timestamp": round(item["timestamp"], 2),
                    "image_b64": f"data:image/jpeg;base64,{b64_str}",
                })

        return {
            "evidence_id": f"EV-{int(time.time() * 1000)}",
            "camera_id": camera_id,
            "incident_type": alert["incident_type"],
            "severity": alert["severity"],
            "confidence": alert["confidence"],
            "risk_score": round(risk_score, 3),
            "quality_state": quality_metrics.get("quality_state", "UNKNOWN"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reasons": alert.get("reasons", ["Unusual pattern detected"]),
            "evidence_metrics": alert.get("evidence_metrics", {}),
            "key_frames": key_frames,
        }

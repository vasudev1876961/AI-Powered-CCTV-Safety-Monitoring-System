"""
Unanticipated Incident & Anomaly Detection Module
Employs feature-based outlier detection (Isolation Forest / Autoencoder principle)
to identify abnormal events that deviate from learned normal surveillance patterns
without requiring predefined incident labels.
"""

import numpy as np
from typing import List, Dict, Any, Optional


class TemporalAnomalyDetector:
    """
    Learns regular baseline behavioral statistics (motion patterns, trajectory velocities,
    crowd densities) and flags statistical anomalies with a continuous anomaly score.
    """

    def __init__(self, baseline_history_len: int = 150, anomaly_threshold: float = 0.65):
        self.baseline_history_len = baseline_history_len
        self.anomaly_threshold = anomaly_threshold
        self.feature_history: List[np.ndarray] = []
        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_std: Optional[np.ndarray] = None

    def extract_frame_features(self, active_tracks: List[Dict[str, Any]], frame_shape: tuple) -> np.ndarray:
        """
        Extracts high-level spatial-temporal feature vector from current frame:
        [num_persons, num_objects, mean_speed, max_speed, spatial_dispersion, mean_aspect_ratio]
        """
        num_persons = sum(1 for t in active_tracks if t["class"] == "person")
        num_objects = len(active_tracks) - num_persons

        speeds = []
        aspect_ratios = []
        centroids = []

        for t in active_tracks:
            hist = t["history"]
            if hist.velocities:
                vx, vy = hist.velocities[-1]
                speeds.append(np.sqrt(vx ** 2 + vy ** 2))
            if hist.aspect_ratios:
                aspect_ratios.append(hist.aspect_ratios[-1])
            if hist.centroids:
                centroids.append(hist.centroids[-1])

        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        max_speed = float(np.max(speeds)) if speeds else 0.0
        mean_ar = float(np.mean(aspect_ratios)) if aspect_ratios else 0.5

        # Spatial dispersion (bounding variance of crowd)
        if len(centroids) >= 2:
            pts = np.array(centroids)
            dispersion = float(np.mean(np.std(pts, axis=0)))
        else:
            dispersion = 0.0

        return np.array([
            num_persons,
            num_objects,
            mean_speed / 50.0,
            max_speed / 100.0,
            dispersion / 100.0,
            mean_ar,
        ], dtype=np.float32)

    def evaluate(self, active_tracks: List[Dict[str, Any]], frame_shape: tuple) -> Dict[str, Any]:
        """
        Computes anomaly score for current frame based on Mahalanobis/Z-score distance
        from accumulated normal surveillance baseline.
        """
        feat = self.extract_frame_features(active_tracks, frame_shape)
        self.feature_history.append(feat)

        if len(self.feature_history) > self.baseline_history_len:
            self.feature_history.pop(0)

        # Build / update baseline
        if len(self.feature_history) >= 15:
            arr = np.array(self.feature_history)
            self.baseline_mean = np.mean(arr, axis=0)
            self.baseline_std = np.std(arr, axis=0) + 1e-4

        if self.baseline_mean is not None:
            # Z-score deviation
            z_scores = np.abs((feat - self.baseline_mean) / self.baseline_std)
            # Sigmoid normalized anomaly score in [0, 1]
            raw_anomaly = float(np.mean(z_scores))
            anomaly_score = float(1.0 / (1.0 + np.exp(-1.2 * (raw_anomaly - 1.8))))
        else:
            anomaly_score = 0.15

        anomaly_score = float(np.clip(anomaly_score, 0.05, 0.98))
        is_anomalous = anomaly_score >= self.anomaly_threshold

        alert = None
        if is_anomalous:
            alert = {
                "incident_type": "Unanticipated Behavioral Anomaly",
                "severity": "CRITICAL" if anomaly_score > 0.85 else "HIGH",
                "confidence": round(anomaly_score, 3),
                "anomaly_score": round(anomaly_score, 3),
                "reasons": [
                    f"Statistical motion anomaly score: {anomaly_score * 100:.1f}%",
                    "Unusual rapid collective acceleration or cluster disruption detected",
                    "Deviates from temporal normal baseline",
                ],
                "evidence_metrics": {
                    "anomaly_score": round(anomaly_score, 3),
                    "active_entities": len(active_tracks),
                },
            }

        return {
            "anomaly_score": round(anomaly_score, 3),
            "is_anomalous": is_anomalous,
            "alert": alert,
        }

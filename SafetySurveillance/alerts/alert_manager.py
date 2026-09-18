"""
Alert Management & Multi-Factor Risk Scoring Module
Calculates composite risk scores based on:
Risk Score = P_incident * C_temporal * C_detection * Q_factor
Handles alert throttling, deduplication, and persistence.
"""

import time
from typing import Dict, Any, List, Optional


class SafetyAlertManager:
    """Manages safety notifications and calculates multi-factor risk scores."""

    def __init__(self, cooldown_seconds: float = 4.0):
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_times: Dict[str, float] = {}  # key -> timestamp
        self.alert_history: List[Dict[str, Any]] = []

    def compute_risk_score(
        self,
        incident_conf: float,
        detection_conf: float,
        temporal_conf: float = 0.90,
        quality_factor: float = 0.85
    ) -> float:
        """
        Calculates unified multi-factor risk score:
        Risk = Incident Confidence * Temporal Confidence * Detection Confidence * Quality Factor
        """
        score = float(incident_conf * temporal_conf * detection_conf * quality_factor)
        # Scale slightly to make dynamic range intuitive
        scaled_score = min(0.99, max(0.05, score * 1.35))
        return round(scaled_score, 3)

    def determine_severity(self, risk_score: float) -> str:
        """Categorizes numerical risk score into operational severity levels."""
        if risk_score >= 0.75:
            return "CRITICAL"
        elif risk_score >= 0.55:
            return "HIGH"
        elif risk_score >= 0.35:
            return "MEDIUM"
        return "LOW"

    def process_alert(
        self,
        camera_id: str,
        alert_candidate: Dict[str, Any],
        quality_factor: float,
        detection_conf: float = 0.88,
        temporal_conf: float = 0.92
    ) -> Optional[Dict[str, Any]]:
        """
        Processes candidate alert, applies risk formula, deduplicates, and logs.
        """
        inc_type = alert_candidate["incident_type"]
        track_id = alert_candidate.get("track_id", "multi")
        dedup_key = f"{camera_id}_{inc_type}_{track_id}"
        now = time.time()

        if dedup_key in self.last_alert_times:
            if now - self.last_alert_times[dedup_key] < self.cooldown_seconds:
                return None  # Throttled

        self.last_alert_times[dedup_key] = now

        # Calculate Risk Score
        risk = self.compute_risk_score(
            incident_conf=alert_candidate.get("confidence", 0.85),
            detection_conf=detection_conf,
            temporal_conf=temporal_conf,
            quality_factor=quality_factor
        )
        severity = self.determine_severity(risk)

        final_alert = {
            "id": f"ALT-{int(now * 1000)}",
            "camera_id": camera_id,
            "incident_type": inc_type,
            "severity": severity,
            "risk_score": risk,
            "confidence": alert_candidate.get("confidence", 0.85),
            "timestamp": time.strftime("%H:%M:%S"),
            "full_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "track_id": track_id,
            "bbox": alert_candidate.get("bbox", [0, 0, 0, 0]),
            "reasons": alert_candidate.get("reasons", ["Anomalous event pattern detected"]),
            "evidence_metrics": alert_candidate.get("evidence_metrics", {}),
            "acknowledged": False,
        }

        self.alert_history.insert(0, final_alert)
        if len(self.alert_history) > 100:
            self.alert_history.pop()

        return final_alert

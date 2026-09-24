"""
Unit Tests for Alert Management, Risk Scoring & XAI Evidence Modules
"""

import unittest
import numpy as np

from SafetySurveillance.alerts.alert_manager import SafetyAlertManager
from SafetySurveillance.explainability.evidence import EvidenceRecorder
from SafetySurveillance.explainability.gradcam import SaliencyExplainer


class TestAlertsAndEvidence(unittest.TestCase):
    def setUp(self):
        self.alert_mgr = SafetyAlertManager(cooldown_seconds=1.0)
        self.evidence_rec = EvidenceRecorder(buffer_size=10)

    def test_risk_score_calculation(self):
        # Risk = P(inc) * C(temp) * C(det) * Q(fac)
        # 0.9 * 0.9 * 0.9 * 0.9 = 0.6561 * 1.35 = 0.885
        risk = self.alert_mgr.compute_risk_score(
            incident_conf=0.9,
            detection_conf=0.9,
            temporal_conf=0.9,
            quality_factor=0.9
        )
        self.assertGreater(risk, 0.70)
        self.assertEqual(self.alert_mgr.determine_severity(risk), "CRITICAL")

        # Low risk score test
        low_risk = self.alert_mgr.compute_risk_score(
            incident_conf=0.3,
            detection_conf=0.4,
            temporal_conf=0.4,
            quality_factor=0.4
        )
        self.assertLess(low_risk, 0.35)
        self.assertEqual(self.alert_mgr.determine_severity(low_risk), "LOW")

    def test_alert_processing_and_cooldown(self):
        candidate = {
            "incident_type": "Fall / Collapse",
            "confidence": 0.92,
            "track_id": 1,
            "bbox": [100, 100, 150, 150],
            "reasons": ["Sudden downward acceleration"],
        }

        # First alert must succeed
        alert1 = self.alert_mgr.process_alert("CAM_01", alert_candidate=candidate, quality_factor=0.85)
        self.assertIsNotNone(alert1)
        self.assertEqual(alert1["incident_type"], "Fall / Collapse")
        self.assertIn("id", alert1)

        # Immediate second identical alert must be throttled by cooldown
        alert2 = self.alert_mgr.process_alert("CAM_01", alert_candidate=candidate, quality_factor=0.85)
        self.assertIsNone(alert2)

        # Different camera or track should not be throttled
        cand_other = dict(candidate, track_id=2)
        alert3 = self.alert_mgr.process_alert("CAM_01", alert_candidate=cand_other, quality_factor=0.85)
        self.assertIsNotNone(alert3)

    def test_evidence_recorder_and_packager(self):
        frame = np.full((120, 160, 3), 100, dtype=np.uint8)

        # Buffer frames
        for i in range(5):
            self.evidence_rec.record_frame(frame, frame_id=i, timestamp=float(i))

        alert = {
            "incident_type": "Perimeter Intrusion",
            "severity": "HIGH",
            "confidence": 0.88,
            "reasons": ["Breached fence perimeter"],
            "evidence_metrics": {"dwell_time": 4.2},
        }

        pack = self.evidence_rec.assemble_evidence_pack(
            alert=alert,
            camera_id="CAM_03",
            quality_metrics={"quality_state": "GOOD"},
            risk_score=0.78
        )

        self.assertIn("evidence_id", pack)
        self.assertEqual(pack["camera_id"], "CAM_03")
        self.assertGreater(len(pack["key_frames"]), 0)
        self.assertTrue(pack["key_frames"][0]["image_b64"].startswith("data:image/jpeg;base64,"))

    def test_saliency_explainer(self):
        frame = np.full((100, 100, 3), 50, dtype=np.uint8)
        bboxes = [[20, 20, 60, 60]]
        heatmap = SaliencyExplainer.generate_heatmap(frame, bboxes, intensity=0.8)

        self.assertEqual(heatmap.shape, frame.shape)
        self.assertEqual(heatmap.dtype, np.uint8)
        # Saliency region must differ from original dark frame
        self.assertFalse(np.array_equal(heatmap, frame))


if __name__ == "__main__":
    unittest.main()

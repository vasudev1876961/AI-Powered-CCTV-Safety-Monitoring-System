"""
Integration Tests for FastAPI Surveillance REST Endpoints
"""

import unittest
import base64
import cv2
import numpy as np
from fastapi.testclient import TestClient

from SafetySurveillance.server import app, engine


class TestServerAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_check(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn(data["status"], ["online", "healthy"])
        self.assertIn("active_camera", data)
        self.assertIn("enhancement_mode", data)

    def test_system_info(self):
        resp = self.client.get("/api/system_info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("cameras", data)
        self.assertIn("incident_detectors", data)
        self.assertIn("tracker", data)

    def test_get_and_post_config(self):
        # GET config
        resp_get = self.client.get("/api/config")
        self.assertEqual(resp_get.status_code, 200)

        # POST config
        resp_post = self.client.post("/api/config", json={
            "enhancement_mode": "auto",
            "camera": "CAM_02",
            "fall_velocity_threshold": 85.0
        })
        self.assertEqual(resp_post.status_code, 200)
        data = resp_post.json()
        self.assertEqual(data["status"], "updated")
        self.assertEqual(engine.current_camera, "CAM_02")

    def test_benchmark_results_endpoint(self):
        resp = self.client.get("/benchmark_results.json")
        self.assertEqual(resp.status_code, 200)

    def test_evidence_list_and_detail(self):
        resp = self.client.get("/api/evidence")
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), list)

    def test_process_single_frame(self):
        # Create a small 64x64 synthetic frame and encode as JPEG base64
        frame = np.full((64, 64, 3), 120, dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', frame)
        b64_str = base64.b64encode(buffer).decode('utf-8')

        resp = self.client.post("/api/process_frame", json={"image": f"data:image/jpeg;base64,{b64_str}"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("quality", data)
        self.assertIn("risk_score", data)
        self.assertIn("severity", data)

    def test_snapshot_capture_endpoint(self):
        resp = self.client.get("/api/snapshot/CAM_01")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "image/jpeg")
        self.assertGreater(len(resp.content), 500)

    def test_metrics_summary_endpoint(self):
        resp = self.client.get("/api/metrics/summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("uptime_seconds", data)
        self.assertIn("total_frames_processed", data)
        self.assertIn("severity_distribution", data)

    def test_preset_geofences_endpoints(self):
        # GET presets
        resp = self.client.get("/api/presets/geofences")
        self.assertEqual(resp.status_code, 200)
        presets = resp.json()
        self.assertIsInstance(presets, list)
        self.assertGreaterEqual(len(presets), 1)

        # POST new preset
        new_preset = {
            "name": "Chemical Storage Hazard",
            "polygon": [[50, 50], [250, 50], [250, 250], [50, 250]],
            "description": "Hazardous materials isolation zone"
        }
        resp_post = self.client.post("/api/presets/geofences", json=new_preset)
        self.assertEqual(resp_post.status_code, 200)
        self.assertEqual(resp_post.json()["status"], "created")

    def test_alert_acknowledge_and_search(self):
        # Inject sample alert into alert manager
        alert_candidate = {
            "incident_type": "Fall / Collapse",
            "confidence": 0.92,
            "track_id": 999,
            "bbox": [100, 100, 200, 150],
            "reasons": ["Rapid vertical collapse"]
        }
        created = engine.alert_manager.process_alert("CAM_01", alert_candidate, quality_factor=0.9)
        self.assertIsNotNone(created)
        alert_id = created["id"]

        # Search alert
        resp_search = self.client.get("/api/alerts/search?incident_type=Fall")
        self.assertEqual(resp_search.status_code, 200)
        results = resp_search.json()
        self.assertTrue(any(a["id"] == alert_id for a in results))

        # Acknowledge alert
        resp_ack = self.client.post(f"/api/alerts/{alert_id}/acknowledge")
        self.assertEqual(resp_ack.status_code, 200)
        self.assertEqual(resp_ack.json()["status"], "acknowledged")


if __name__ == "__main__":
    unittest.main()

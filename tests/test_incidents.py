"""
Unit Tests for Incident Recognition & Anomaly Detection Modules
"""

import unittest
import numpy as np

from SafetySurveillance.tracking.trajectory import TrackHistory
from SafetySurveillance.incidents.fall import FallDetector
from SafetySurveillance.incidents.violence import ViolenceDetector
from SafetySurveillance.incidents.intrusion import IntrusionDetector, point_in_polygon
from SafetySurveillance.incidents.abandoned_object import AbandonedObjectDetector
from SafetySurveillance.incidents.anomaly import TemporalAnomalyDetector


class TestIncidents(unittest.TestCase):
    def test_point_in_polygon(self):
        poly = [[100, 100], [300, 100], [300, 300], [100, 300]]
        self.assertTrue(point_in_polygon((200, 200), poly))
        self.assertFalse(point_in_polygon((50, 50), poly))
        self.assertFalse(point_in_polygon((400, 400), poly))

    def test_fall_detector(self):
        fall_det = FallDetector(velocity_thresh=60.0, aspect_ratio_thresh=1.0)
        hist = TrackHistory(track_id=10)
        t = 100.0

        # Simulate person walking upright
        for i in range(5):
            hist.update([100, 100 + i * 2, 140, 200 + i * 2], timestamp=t + i * 0.1)

        track_standing = {
            "id": 10,
            "class": "person",
            "bbox": [100, 110, 140, 210], # w=40, h=100 (ar=0.4)
            "history": hist
        }
        self.assertIsNone(fall_det.evaluate(track_standing))

        # Now simulate sudden downward fall and horizontal collapse
        hist.update([100, 150, 150, 230], timestamp=t + 0.6)
        hist.update([90, 220, 210, 270], timestamp=t + 0.7) # w=120, h=50 (ar=2.4)
        hist.update([90, 222, 210, 272], timestamp=t + 1.2)
        hist.update([90, 222, 210, 272], timestamp=t + 2.5)

        track_fallen = {
            "id": 10,
            "class": "person",
            "bbox": [90, 222, 210, 272],
            "history": hist
        }
        alert = fall_det.evaluate(track_fallen)
        self.assertIsNotNone(alert)
        self.assertIn("Fall", alert["incident_type"])
        self.assertGreater(alert["confidence"], 0.7)
        self.assertIn("reasons", alert)

    def test_violence_detector(self):
        violence_det = ViolenceDetector(proximity_thresh=90.0, kinetic_thresh=40.0)

        # Person 1 and Person 2 far apart
        h1 = TrackHistory(track_id=1)
        h2 = TrackHistory(track_id=2)
        for i in range(5):
            h1.update([50, 50, 90, 140], timestamp=10.0 + i * 0.1)
            h2.update([400, 400, 440, 490], timestamp=10.0 + i * 0.1)

        tracks_far = [
            {"id": 1, "class": "person", "bbox": [50, 50, 90, 140], "history": h1},
            {"id": 2, "class": "person", "bbox": [400, 400, 440, 490], "history": h2}
        ]
        self.assertEqual(len(violence_det.evaluate(tracks_far)), 0)

        # Person 1 and Person 2 close with rapid opposing velocity oscillations (fight)
        h3 = TrackHistory(track_id=3)
        h4 = TrackHistory(track_id=4)
        t = 20.0
        coords3 = [[200, 200, 240, 300], [215, 205, 255, 305], [195, 195, 235, 295], [220, 210, 260, 310], [200, 200, 240, 300]]
        coords4 = [[230, 205, 270, 305], [210, 200, 250, 300], [240, 215, 280, 315], [210, 195, 250, 295], [235, 205, 275, 305]]
        for i in range(5):
            h3.update(coords3[i], timestamp=t + i * 0.05)
            h4.update(coords4[i], timestamp=t + i * 0.05)

        tracks_fight = [
            {"id": 3, "class": "person", "bbox": coords3[-1], "history": h3},
            {"id": 4, "class": "person", "bbox": coords4[-1], "history": h4}
        ]
        alerts = violence_det.evaluate(tracks_fight)
        self.assertGreaterEqual(len(alerts), 1)
        self.assertIn("Violence", alerts[0]["incident_type"])
        self.assertEqual(alerts[0]["severity"], "CRITICAL")

    def test_intrusion_detector(self):
        zones = [{
            "name": "Server Vault",
            "polygon": [[100, 100], [300, 100], [300, 300], [100, 300]]
        }]
        intrusion_det = IntrusionDetector(restricted_zones=zones, loiter_thresh_sec=3.0)

        # Track outside zone
        h_out = TrackHistory(track_id=5)
        h_out.update([10, 10, 40, 80], timestamp=1.0)
        track_out = [{"id": 5, "class": "person", "bbox": [10, 10, 40, 80], "history": h_out}]
        self.assertEqual(len(intrusion_det.evaluate(track_out)), 0)

        # Track inside zone
        h_in = TrackHistory(track_id=6)
        h_in.update([150, 150, 190, 240], timestamp=1.0)
        track_in = [{"id": 6, "class": "person", "bbox": [150, 150, 190, 240], "history": h_in}]
        alerts = intrusion_det.evaluate(track_in)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["incident_type"], "Perimeter Intrusion")

        # Loitering test
        for i in range(10):
            h_in.update([150, 150, 190, 240], timestamp=1.0 + i * 0.5) # 4.5s inside
        alerts_loiter = intrusion_det.evaluate(track_in)
        self.assertEqual(len(alerts_loiter), 1)
        self.assertEqual(alerts_loiter[0]["incident_type"], "Suspicious Loitering")

    def test_abandoned_object_detector(self):
        bag_det = AbandonedObjectDetector(unattended_seconds=2.0, owner_radius_px=100.0)

        # Bag left alone
        h_bag = TrackHistory(track_id=7)
        for i in range(6):
            h_bag.update([200, 200, 230, 230], timestamp=10.0 + i * 0.5)

        # Person is 300px away
        h_person = TrackHistory(track_id=8)
        h_person.update([500, 500, 540, 600], timestamp=13.0)

        active_tracks = [
            {"id": 7, "class": "backpack", "bbox": [200, 200, 230, 230], "history": h_bag},
            {"id": 8, "class": "person", "bbox": [500, 500, 540, 600], "history": h_person},
        ]
        alerts = bag_det.evaluate(active_tracks)
        self.assertEqual(len(alerts), 1)
        self.assertIn("Abandoned Object", alerts[0]["incident_type"])

    def test_temporal_anomaly_detector(self):
        anomaly_det = TemporalAnomalyDetector(baseline_history_len=30, anomaly_threshold=0.60)

        # Train baseline with normal calm motion
        for step in range(20):
            h = TrackHistory(track_id=100)
            h.update([100 + step * 2, 100, 140 + step * 2, 180], timestamp=float(step))
            tracks = [{"id": 100, "class": "person", "bbox": [100 + step * 2, 100, 140 + step * 2, 180], "history": h}]
            res = anomaly_det.evaluate(tracks, (480, 640))
            self.assertIn("anomaly_score", res)

        # Now introduce violent disruption: 8 people moving at extreme velocities
        burst_tracks = []
        for p in range(8):
            hb = TrackHistory(track_id=200 + p)
            hb.update([p * 50, 100, p * 50 + 40, 180], timestamp=25.0)
            hb.update([p * 50 + 200, 350, p * 50 + 240, 430], timestamp=25.05) # huge speed
            burst_tracks.append({"id": 200 + p, "class": "person", "bbox": [p * 50 + 200, 350, p * 50 + 240, 430], "history": hb})

        res_burst = anomaly_det.evaluate(burst_tracks, (480, 640))
        self.assertGreater(res_burst["anomaly_score"], 0.50)


if __name__ == "__main__":
    unittest.main()

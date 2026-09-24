"""
Unit Tests for Tracking & Kinematic Trajectory Modules
"""

import unittest
import time

from SafetySurveillance.tracking.trajectory import TrackHistory
from SafetySurveillance.tracking.tracker import MultiObjectTracker, compute_iou


class TestTrackingKinematics(unittest.TestCase):
    def test_compute_iou(self):
        boxA = [10, 10, 50, 50]
        boxB = [10, 10, 50, 50]
        self.assertAlmostEqual(compute_iou(boxA, boxB), 1.0)

        boxC = [100, 100, 150, 150]
        self.assertAlmostEqual(compute_iou(boxA, boxC), 0.0)

        boxD = [30, 30, 70, 70]
        iou_val = compute_iou(boxA, boxD)
        self.assertGreater(iou_val, 0.0)
        self.assertLess(iou_val, 1.0)

    def test_track_history_kinematics(self):
        hist = TrackHistory(track_id=1, max_history=30)
        t0 = 1000.0

        # Step 1: Standing posture moving down (fall simulation)
        hist.update([100, 50, 140, 150], timestamp=t0)        # w=40, h=100, ar=0.4
        hist.update([100, 80, 140, 180], timestamp=t0 + 0.1)  # moving down
        hist.update([100, 120, 140, 220], timestamp=t0 + 0.2) # moving down
        hist.update([90, 180, 200, 240], timestamp=t0 + 0.3)  # fallen: w=110, h=60, ar=1.83

        self.assertEqual(len(hist.centroids), 4)
        self.assertGreater(hist.get_vertical_velocity(window=3), 100.0)
        self.assertGreater(hist.get_aspect_ratio_shift(window=4), 0.5)

    def test_track_dwell_time(self):
        hist = TrackHistory(track_id=2, max_history=30)
        t = 1000.0

        # Feed stationary points within 10px radius
        for i in range(10):
            hist.update([100, 100, 150, 150], timestamp=t + (i * 0.5))

        dwell = hist.get_dwell_time(stationary_threshold_px=20.0)
        self.assertGreater(dwell, 4.0)

    def test_predicted_bbox(self):
        hist = TrackHistory(track_id=3)
        hist.update([100, 100, 150, 150], timestamp=10.0)
        hist.update([110, 100, 160, 150], timestamp=10.1) # moving right by 100 px/sec

        pred = hist.get_predicted_bbox(dt=0.1)
        # Bbox should be shifted to the right
        self.assertGreater(pred[0], 110)

    def test_multi_object_tracker_lifecycle(self):
        tracker = MultiObjectTracker(iou_thresh=0.25, max_lost_frames=3)

        # Frame 1: One detection
        dets1 = [{"bbox": [50, 50, 90, 130], "class": "person", "conf": 0.9}]
        tracks1 = tracker.update(dets1, timestamp=1.0)
        self.assertEqual(len(tracks1), 1)
        tid = tracks1[0]["id"]

        # Frame 2: Person moves slightly
        dets2 = [{"bbox": [54, 52, 94, 132], "class": "person", "conf": 0.92}]
        tracks2 = tracker.update(dets2, timestamp=1.04)
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0]["id"], tid)  # ID must persist!

        # Frame 3-6: Person disappears (lost frames increment and prune)
        for t_step in range(4):
            tracker.update([], timestamp=2.0 + t_step * 0.1)

        self.assertEqual(len(tracker.tracks), 0)  # Pruned after max_lost_frames


if __name__ == "__main__":
    unittest.main()

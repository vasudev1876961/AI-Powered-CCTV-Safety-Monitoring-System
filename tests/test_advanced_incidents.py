"""
Unit tests for Advanced Incident Detectors:
Loitering, Crowd Density / Stampede Hazard, and Fire & Smoke Early Warning.
"""

import time
import numpy as np
import pytest
from SafetySurveillance.incidents.loitering import LoiteringDetector
from SafetySurveillance.incidents.crowd_density import CrowdDensityDetector
from SafetySurveillance.incidents.fire_smoke import FireSmokeDetector
from SafetySurveillance.tracking.trajectory import TrackHistory


def test_loitering_detector_normal_transit():
    detector = LoiteringDetector(dwell_thresh_sec=4.0, movement_radius_px=30.0)
    now = time.time()

    # Moving person across frames
    track = {
        "id": 1,
        "class": "person",
        "bbox": [100, 100, 150, 220]
    }
    alerts = detector.evaluate([track], timestamp=now)
    assert len(alerts) == 0

    # Person moved 100px away within 5 seconds
    track["bbox"] = [250, 100, 300, 220]
    alerts = detector.evaluate([track], timestamp=now + 5.0)
    # Significant displacement resets dwell -> no alert
    assert len(alerts) == 0


def test_loitering_detector_triggers_alert():
    detector = LoiteringDetector(dwell_thresh_sec=3.0, movement_radius_px=40.0)
    now = time.time()

    track = {
        "id": 2,
        "class": "person",
        "bbox": [100, 100, 150, 220]
    }
    detector.evaluate([track], timestamp=now)

    # Person remains virtually stationary in the same spot for 4.5 seconds
    track["bbox"] = [102, 101, 152, 221]
    alerts = detector.evaluate([track], timestamp=now + 4.5)

    assert len(alerts) == 1
    assert alerts[0]["incident_type"] == "Suspicious Loitering / Lingering"
    assert alerts[0]["evidence_metrics"]["dwell_duration_sec"] >= 4.0


def test_crowd_density_detector_dispersed():
    detector = CrowdDensityDetector(max_density_threshold=4, cluster_dist_px=80.0)

    # 4 people widely separated across camera
    tracks = [
        {"id": 1, "class": "person", "bbox": [50, 50, 90, 150]},
        {"id": 2, "class": "person", "bbox": [250, 50, 290, 150]},
        {"id": 3, "class": "person", "bbox": [500, 50, 540, 150]},
        {"id": 4, "class": "person", "bbox": [700, 50, 740, 150]},
    ]
    alerts = detector.evaluate(tracks)
    assert len(alerts) == 0


def test_crowd_density_detector_overcrowding():
    detector = CrowdDensityDetector(max_density_threshold=4, cluster_dist_px=100.0)

    # 5 people tightly clustered together
    tracks = [
        {"id": 1, "class": "person", "bbox": [200, 200, 240, 300]},
        {"id": 2, "class": "person", "bbox": [220, 210, 260, 310]},
        {"id": 3, "class": "person", "bbox": [240, 195, 280, 295]},
        {"id": 4, "class": "person", "bbox": [215, 230, 255, 330]},
        {"id": 5, "class": "person", "bbox": [235, 220, 275, 320]},
    ]
    alerts = detector.evaluate(tracks)
    assert len(alerts) >= 1
    assert any(a["incident_type"] == "Overcrowding / Density Hazard" for a in alerts)


def test_crowd_density_stampede_panic():
    detector = CrowdDensityDetector(max_density_threshold=10, stampede_velocity_thresh=50.0)

    # 3 clustered people with high collective velocity
    tracks = []
    for i in range(3):
        hist = TrackHistory(track_id=i+1, max_history=20)
        hist.centroids = [(200 + i * 20, 200), (280 + i * 20, 200)]
        hist.timestamps = [10.0, 10.5]
        hist.velocities = [(160.0, 0.0)]
        tracks.append({
            "id": i + 1,
            "class": "person",
            "bbox": [200 + i * 20, 200, 240 + i * 20, 300],
            "history": hist
        })

    alerts = detector.evaluate(tracks)
    assert len(alerts) >= 1
    assert any("Stampede" in a["incident_type"] for a in alerts)


def test_fire_smoke_detector_clean_frame():
    detector = FireSmokeDetector()
    frame = np.full((400, 400, 3), (60, 60, 60), dtype=np.uint8)
    alert = detector.evaluate(frame)
    assert alert is None


def test_fire_smoke_detector_flame_trigger():
    detector = FireSmokeDetector(min_area_px=200)
    # Generate frame with bright orange/red flame patch (high Cr in YCrCb)
    frame = np.full((300, 300, 3), (20, 20, 20), dtype=np.uint8)
    # Bright flame rectangle in BGR (0, 120, 255)
    frame[100:200, 100:200] = (0, 140, 255)

    # Feed multiple frames to establish flicker
    alert = None
    for _ in range(5):
        alert = detector.evaluate(frame)

    assert alert is not None
    assert "Fire" in alert["incident_type"]
    assert alert["severity"] == "CRITICAL"

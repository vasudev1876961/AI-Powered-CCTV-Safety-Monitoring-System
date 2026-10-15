"""
Automated Unit & Integration Test Suite for Advanced Safety Intelligence Modules
Tests PPE Compliance, Vehicle-Pedestrian Collision (TTC), Worker Immobility (Man-Down),
Enterprise Webhook Dispatcher, and Spatial Occupancy Analytics.
"""

import time
import pytest
import numpy as np
from fastapi.testclient import TestClient

from SafetySurveillance.incidents.ppe_compliance import PPEComplianceDetector
from SafetySurveillance.incidents.collision_warning import CollisionWarningDetector
from SafetySurveillance.incidents.worker_immobility import WorkerImmobilityDetector
from SafetySurveillance.alerts.dispatcher import WebhookDispatcher
from SafetySurveillance.evaluation.spatial_analytics import SpatialOccupancyAnalytics
from SafetySurveillance.tracking.trajectory import TrackHistory
from SafetySurveillance.server import app, engine


@pytest.fixture
def test_client():
    return TestClient(app)


# -----------------------------------------------------------------------------
# 1. PPE Compliance Detector Tests
# -----------------------------------------------------------------------------
def test_ppe_detection_compliant_worker():
    detector = PPEComplianceDetector()
    # Create synthetic worker crop: 100x40 px
    # Head with yellow hardhat (H: 25, S: 200, V: 220)
    worker = np.zeros((100, 40, 3), dtype=np.uint8)
    # Head region (0:22)
    worker[0:20, :] = [0, 215, 255]  # BGR yellow
    # Torso region (25:65) with neon lime vest
    worker[25:65, :] = [20, 240, 220]  # BGR neon lime

    result = detector.inspect_worker_crop(worker)
    assert result["has_helmet"] is True
    assert result["has_vest"] is True
    assert result["compliance_score"] == 1.0


def test_ppe_detection_non_compliant_worker():
    detector = PPEComplianceDetector()
    # Dark clothing without helmet or neon vest
    worker = np.full((100, 40, 3), 45, dtype=np.uint8)
    result = detector.inspect_worker_crop(worker)
    assert result["has_helmet"] is False
    assert result["has_vest"] is False
    assert result["compliance_score"] == 0.0

    # Evaluate track over 3 frames to trigger violation alert
    frame = np.full((480, 640, 3), 50, dtype=np.uint8)
    frame[50:150, 50:90] = worker
    track = {"id": 101, "class": "person", "bbox": [50, 50, 90, 150]}

    alerts = []
    for _ in range(3):
        alerts = detector.evaluate([track], frame)

    assert len(alerts) >= 1
    assert alerts[0]["incident_type"] == "ppe_violation"
    assert "Mandatory Safety Helmet / Hardhat" in alerts[0]["missing_ppe"]
    assert alerts[0]["severity"] == "CRITICAL"


# -----------------------------------------------------------------------------
# 2. Collision Warning & TTC Detector Tests
# -----------------------------------------------------------------------------
def test_collision_warning_closing_trajectory():
    detector = CollisionWarningDetector(proximity_threshold_px=150.0, critical_ttc_seconds=2.5)

    # Vehicle at (200, 200), moving right at 20 px/frame towards person at (300, 200)
    dist, rel_speed, ttc = detector.calculate_ttc(
        c_veh=(200.0, 200.0), v_veh=(20.0, 0.0),
        c_ped=(300.0, 200.0), v_ped=(0.0, 0.0),
        fps=25.0
    )
    assert dist == 100.0
    assert rel_speed == 500.0  # 20 px/frame * 25 fps = 500 px/sec
    assert ttc == 0.2  # 100 / 500 = 0.2s

    # Track evaluation test
    veh_track = {
        "id": 1,
        "class": "car",
        "bbox": [160, 170, 240, 230],
        "history": TrackHistory(track_id=1, max_history=10)
    }
    veh_track["history"].centroids = [(180, 200), (200, 200)]
    veh_track["history"].velocities = [(20.0, 0.0)]

    ped_track = {
        "id": 2,
        "class": "person",
        "bbox": [280, 160, 320, 240],
        "history": TrackHistory(track_id=2, max_history=10)
    }
    ped_track["history"].centroids = [(300, 200), (300, 200)]
    ped_track["history"].velocities = [(0.0, 0.0)]

    alerts = detector.evaluate([veh_track, ped_track], fps=25.0)
    assert len(alerts) == 1
    assert alerts[0]["incident_type"] == "collision_hazard"
    assert alerts[0]["ttc_seconds"] <= 2.5
    assert alerts[0]["severity"] == "CRITICAL"


def test_collision_warning_diverging_trajectory():
    detector = CollisionWarningDetector()
    # Moving apart -> TTC should be 999.0
    dist, rel_speed, ttc = detector.calculate_ttc(
        c_veh=(200.0, 200.0), v_veh=(-10.0, 0.0),
        c_ped=(300.0, 200.0), v_ped=(0.0, 0.0),
        fps=25.0
    )
    assert ttc == 999.0


# -----------------------------------------------------------------------------
# 3. Worker Immobility (Man-Down) Tests
# -----------------------------------------------------------------------------
def test_worker_immobility_mandown():
    detector = WorkerImmobilityDetector(immobility_time_sec=4.0, max_displacement_px=15.0)
    t_start = 1000.0

    # Prone track: w=100, h=40 -> aspect ratio = 2.5
    prone_track = {
        "id": 7,
        "class": "person",
        "bbox": [100, 200, 200, 240],
    }

    # First frame at t_start -> initialized in tracker
    alerts_0 = detector.evaluate([prone_track], timestamp=t_start)
    assert len(alerts_0) == 0

    # Intermediate frame at t_start + 2s -> still in window
    alerts_mid = detector.evaluate([prone_track], timestamp=t_start + 2.0)
    assert len(alerts_mid) == 0

    # Elapsed frame at t_start + 5.0s -> triggers man-down alert!
    alerts_final = detector.evaluate([prone_track], timestamp=t_start + 5.0)
    assert len(alerts_final) == 1
    assert alerts_final[0]["incident_type"] == "worker_mandown"
    assert alerts_final[0]["immobility_duration_sec"] >= 4.0


# -----------------------------------------------------------------------------
# 4. Webhook Dispatcher Tests
# -----------------------------------------------------------------------------
def test_webhook_dispatcher_crud_and_formatting():
    dispatcher = WebhookDispatcher()
    # Register webhook
    wh = dispatcher.register_webhook("Test SOC Webhook", "http://localhost:8000/api/mock_webhook", "generic", "HIGH")
    assert wh["name"] == "Test SOC Webhook"

    # Verify listing
    hooks = dispatcher.list_webhooks()
    assert any(h["id"] == wh["id"] for h in hooks)

    # Format Discord payload
    test_alert = {
        "id": "ALT-1234",
        "incident_type": "worker_mandown",
        "severity": "CRITICAL",
        "risk_score": 0.95,
        "camera_id": "CAM_01",
        "reasons": ["Unresponsive worker motionless on floor"],
        "timestamp": "12:00:00"
    }
    discord_payload = dispatcher.format_payload({"type": "discord"}, test_alert)
    assert "embeds" in discord_payload
    assert discord_payload["embeds"][0]["color"] == 16711680  # Red

    # Delete webhook
    deleted = dispatcher.delete_webhook(wh["id"])
    assert deleted is True


# -----------------------------------------------------------------------------
# 5. Spatial Analytics Tests
# -----------------------------------------------------------------------------
def test_spatial_occupancy_analytics():
    analytics = SpatialOccupancyAnalytics(grid_w=32, grid_h=18)
    tracks = [{"bbox": [480, 270, 520, 340]}]
    incidents = [{"bbox": [480, 270, 520, 340]}]

    # Update twice
    analytics.update("CAM_01", tracks, incidents=incidents, frame_shape=(540, 960))
    analytics.update("CAM_01", tracks, incidents=incidents, frame_shape=(540, 960))

    heatmap = analytics.get_normalized_heatmap("CAM_01")
    assert heatmap["camera_id"] == "CAM_01"
    assert heatmap["grid_dimensions"] == [32, 18]
    assert len(heatmap["density_matrix"]) == 18
    assert len(heatmap["density_matrix"][0]) == 32
    # Verify density has accumulated
    assert any(val > 0.0 for row in heatmap["density_matrix"] for val in row)


# -----------------------------------------------------------------------------
# 6. Server REST Endpoints Integration Tests
# -----------------------------------------------------------------------------
def test_server_advanced_endpoints(test_client):
    # 1. Supported incidents includes 11 categories
    r = test_client.get("/api/incidents/types")
    assert r.status_code == 200
    types = [inc["id"] for inc in r.json()["supported_incidents"]]
    assert "ppe" in types
    assert "collision" in types
    assert "mandown" in types
    assert len(types) >= 11

    # 2. PPE compliance endpoint
    r_ppe = test_client.get("/api/ppe/compliance")
    assert r_ppe.status_code == 200
    assert "metrics" in r_ppe.json()

    # 3. Collision warnings endpoint
    r_col = test_client.get("/api/collision/warnings")
    assert r_col.status_code == 200
    assert "active_warnings" in r_col.json()

    # 4. Spatial heatmap endpoint
    r_map = test_client.get("/api/analytics/spatial_heatmap/CAM_01")
    assert r_map.status_code == 200
    assert "density_matrix" in r_map.json()

    # 5. Webhook management & mock receiver
    r_wh = test_client.get("/api/webhooks")
    assert r_wh.status_code == 200

    r_test_wh = test_client.post("/api/webhooks/test", json={"incident_type": "ppe_violation", "severity": "HIGH"})
    assert r_test_wh.status_code == 200
    assert r_test_wh.json()["status"] == "dispatched"

    # 6. Models info telemetry updated
    r_models = test_client.get("/api/models/info")
    assert r_models.status_code == 200
    model_ids = [m["id"] for m in r_models.json()["models"]]
    assert "ppe_engine" in model_ids
    assert "collision_predictor" in model_ids
    assert "webhook_dispatcher" in model_ids
    assert "spatial_analytics" in model_ids

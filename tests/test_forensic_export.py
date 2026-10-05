"""
Unit and integration tests for Cryptographic Forensic Evidence Dossiers,
SHA-256 Chain of Custody Seals, HTML Export, and Advanced REST Endpoints.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from SafetySurveillance.explainability.evidence import EvidenceRecorder, compute_evidence_hash
from SafetySurveillance.server import app, engine


@pytest.fixture
def client():
    return TestClient(app)


def test_evidence_cryptographic_seal():
    recorder = EvidenceRecorder()
    frame = np.full((240, 320, 3), 100, dtype=np.uint8)
    recorder.record_frame(frame, frame_id=1, timestamp=100.0)

    alert = {
        "incident_type": "Restricted Zone Intrusion",
        "severity": "HIGH",
        "confidence": 0.91,
        "reasons": ["Breached security perimeter"],
        "evidence_metrics": {"breach_depth_px": 42.5},
    }

    pack = recorder.assemble_evidence_pack(
        alert=alert,
        camera_id="CAM_03",
        quality_metrics={"quality_state": "GOOD"},
        risk_score=0.78
    )

    assert "tamper_evident_seal" in pack
    assert len(pack["tamper_evident_seal"]) == 64  # SHA-256 hex string
    assert pack["integrity_verified"] is True

    # Test integrity verification
    assert EvidenceRecorder.verify_evidence_integrity(pack) is True

    # Test tamper detection
    tampered_pack = pack.copy()
    tampered_pack["risk_score"] = 0.12  # Alter payload
    assert EvidenceRecorder.verify_evidence_integrity(tampered_pack) is False


def test_forensic_html_report_generation():
    recorder = EvidenceRecorder()
    frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    recorder.record_frame(frame, frame_id=10, timestamp=10.0)

    alert = {
        "incident_type": "Slip, Trip & Fall",
        "severity": "CRITICAL",
        "confidence": 0.95,
        "reasons": ["Sudden vertical velocity collapse", "Post-fall horizontal dwell"],
        "evidence_metrics": {"vertical_velocity": -120.4, "dwell_seconds": 4.2},
    }

    pack = recorder.assemble_evidence_pack(
        alert=alert,
        camera_id="CAM_01",
        quality_metrics={"quality_state": "MODERATE"},
        risk_score=0.88
    )

    html = EvidenceRecorder.generate_forensic_html_report(pack)
    assert "<!DOCTYPE html>" in html
    assert "OFFICIAL FORENSIC EXAMINATION DOSSIER" in html
    assert "SHA-256" in html
    assert pack["tamper_evident_seal"] in html
    assert "Slip, Trip &amp; Fall" in html or "Slip, Trip & Fall" in html


def test_api_matrix_stream(client):
    # Test matrix frame generator directly to verify 2x2 stitched dimensions without infinite stream blocking
    matrix_frame = engine.generate_matrix_frame()
    assert matrix_frame is not None
    assert matrix_frame.shape == (540, 960, 3)


def test_api_reid_endpoints(client):
    res_entities = client.get("/api/reid/entities")
    assert res_entities.status_code == 200
    assert isinstance(res_entities.json(), list)

    res_handovers = client.get("/api/reid/handovers")
    assert res_handovers.status_code == 200
    assert isinstance(res_handovers.json(), list)


def test_api_incident_types(client):
    res = client.get("/api/incidents/types")
    assert res.status_code == 200
    data = res.json()
    assert "supported_incidents" in data
    assert len(data["supported_incidents"]) >= 8


def test_api_evidence_export_and_verify(client):
    # Store synthetic evidence
    alert_id = "test_alert_99"
    engine.evidence_store[alert_id] = {
        "evidence_id": "EV-99999",
        "camera_id": "CAM_01",
        "incident_type": "Physical Violence",
        "severity": "CRITICAL",
        "confidence": 0.94,
        "risk_score": 0.85,
        "quality_state": "GOOD",
        "timestamp": "2026-10-05 23:00:00",
        "reasons": ["High kinetic oscillation"],
        "evidence_metrics": {"kinetic_energy": 88.0},
        "key_frames": [],
    }
    seal = compute_evidence_hash(engine.evidence_store[alert_id])
    engine.evidence_store[alert_id]["tamper_evident_seal"] = seal

    # Test export HTML
    res_export = client.get(f"/api/evidence/{alert_id}/export")
    assert res_export.status_code == 200
    assert "text/html" in res_export.headers.get("content-type", "")
    assert "EV-99999" in res_export.text

    # Test verify
    res_verify = client.get(f"/api/evidence/{alert_id}/verify")
    assert res_verify.status_code == 200
    data = res_verify.json()
    assert data["integrity_verified"] is True
    assert data["status"] == "AUTHENTIC"

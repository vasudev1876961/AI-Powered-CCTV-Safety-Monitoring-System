"""
Unit tests for Cross-Camera Re-Identification (Re-ID) & Handover Tracking.
"""

import time
import numpy as np
import pytest
from SafetySurveillance.tracking.reid import CrossCameraReIDTracker


def test_reid_signature_extraction():
    tracker = CrossCameraReIDTracker()
    # Create synthetic person frame
    frame = np.full((300, 300, 3), (120, 180, 240), dtype=np.uint8)
    bbox = [50, 50, 150, 250]

    sig = tracker.extract_signature(frame, bbox)
    assert sig is not None
    assert isinstance(sig, np.ndarray)
    assert len(sig) == 16 * 8  # 128 bins


def test_reid_signature_comparison():
    tracker = CrossCameraReIDTracker()
    frame1 = np.full((200, 200, 3), (255, 100, 50), dtype=np.uint8)
    frame2 = np.full((200, 200, 3), (255, 100, 50), dtype=np.uint8)
    frame3 = np.full((200, 200, 3), (10, 250, 10), dtype=np.uint8)

    sig1 = tracker.extract_signature(frame1, [20, 20, 180, 180])
    sig2 = tracker.extract_signature(frame2, [20, 20, 180, 180])
    sig3 = tracker.extract_signature(frame3, [20, 20, 180, 180])

    sim_identical = tracker.compare_signatures(sig1, sig2)
    sim_disparate = tracker.compare_signatures(sig1, sig3)

    assert sim_identical > 0.90
    assert sim_identical > sim_disparate


def test_reid_cross_camera_handover():
    tracker = CrossCameraReIDTracker(appearance_thresh=0.50, max_handover_time_sec=30.0)
    now = time.time()

    # Distinctive red jacket person on CAM_01
    frame_cam1 = np.full((400, 400, 3), (30, 30, 220), dtype=np.uint8)
    tracks_cam1 = [{
        "id": 1,
        "class": "person",
        "bbox": [50, 50, 150, 300],
        "conf": 0.92
    }]

    enriched1 = tracker.update_camera_tracks("CAM_01", tracks_cam1, frame_cam1, timestamp=now)
    assert len(enriched1) == 1
    assert enriched1[0]["global_id"] == "G-101"

    # Same person appears on CAM_02 within handover time window
    frame_cam2 = np.full((400, 400, 3), (28, 32, 218), dtype=np.uint8)
    tracks_cam2 = [{
        "id": 5,  # New local ID on CAM_02
        "class": "person",
        "bbox": [60, 60, 160, 310],
        "conf": 0.89
    }]

    enriched2 = tracker.update_camera_tracks("CAM_02", tracks_cam2, frame_cam2, timestamp=now + 5.0)
    assert len(enriched2) == 1
    # Should match global entity G-101
    assert enriched2[0]["global_id"] == "G-101"

    # Verify handover event was logged
    handovers = tracker.get_recent_handovers()
    assert len(handovers) >= 1
    assert handovers[0]["from_camera"] == "CAM_01"
    assert handovers[0]["to_camera"] == "CAM_02"
    assert handovers[0]["global_id"] == "G-101"

    # Verify entity journey
    entities = tracker.get_active_entities()
    assert len(entities) == 1
    assert entities[0]["journey_length"] >= 2

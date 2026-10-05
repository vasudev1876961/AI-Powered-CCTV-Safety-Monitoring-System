"""
Cross-Camera Re-Identification (Re-ID) & Multi-Stream Entity Tracker Module
Maintains global identities across distributed CCTV cameras by extracting
color appearance signatures, spatial-temporal continuity, and handover trajectories.
"""

import time
from typing import List, Dict, Any, Optional
import cv2
import numpy as np


class CrossCameraReIDTracker:
    """Tracks persistent entity identities across multiple independent CCTV camera feeds."""

    def __init__(self, appearance_thresh: float = 0.60, max_handover_time_sec: float = 45.0):
        self.appearance_thresh = appearance_thresh
        self.max_handover_time_sec = max_handover_time_sec
        self.next_global_id = 101

        # Global entities: global_id -> Dict
        self.global_entities: Dict[int, Dict[str, Any]] = {}
        # Handover events log
        self.handover_events: List[Dict[str, Any]] = []
        # Mapping (camera_id, local_track_id) -> global_id
        self.local_to_global: Dict[str, int] = {}

    def extract_signature(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """
        Extracts normalized HSV color histogram feature from person bounding box.

        Returns:
            Normalized 1D feature vector of shape (24,).
        """
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, bbox[0]))
        y1 = max(0, min(h - 1, bbox[1]))
        x2 = max(x1 + 1, min(w, bbox[2]))
        y2 = max(y1 + 1, min(h, bbox[3]))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # 16 bins for Hue, 8 bins for Saturation
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

    def compare_signatures(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Computes histogram correlation similarity in [0, 1]."""
        if sig1 is None or sig2 is None or len(sig1) != len(sig2):
            return 0.0
        # Correlation metric: 1 is identical, -1 is inverse
        val = cv2.compareHist(
            sig1.astype(np.float32),
            sig2.astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        return max(0.0, float(val))

    def update_camera_tracks(
        self,
        camera_id: str,
        tracks: List[Dict[str, Any]],
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Associates local camera tracks with global identities.

        Args:
            camera_id: Camera identifier (e.g. 'CAM_01')
            tracks: List of active local tracks with bbox, id, class
            frame: Video frame for appearance feature extraction
            timestamp: Observation timestamp

        Returns:
            Enriched tracks with assigned 'global_id' and 'reid_confidence'.
        """
        now = timestamp if timestamp is not None else time.time()
        enriched_tracks = []

        for track in tracks:
            if track.get("class") != "person":
                track["global_id"] = None
                enriched_tracks.append(track)
                continue

            local_tid = track["id"]
            local_key = f"{camera_id}_{local_tid}"
            bbox = track["bbox"]
            sig = self.extract_signature(frame, bbox)

            global_id = None
            match_score = 0.0

            # 1. Check if this local track is already assigned
            if local_key in self.local_to_global:
                global_id = self.local_to_global[local_key]
                entity = self.global_entities[global_id]
                entity["last_seen_time"] = now
                entity["current_camera"] = camera_id
                entity["last_bbox"] = bbox
                if sig is not None:
                    # Rolling EMA update of visual signature
                    entity["signature"] = 0.85 * entity["signature"] + 0.15 * sig
                match_score = 1.0

            else:
                # 2. Candidate matching against global entities recently seen in OTHER cameras
                best_gid = None
                best_sim = 0.0

                for gid, entity in self.global_entities.items():
                    time_diff = now - entity["last_seen_time"]
                    # Handover candidate: different camera, seen within handover window
                    if entity["current_camera"] != camera_id and time_diff <= self.max_handover_time_sec:
                        if sig is not None and entity.get("signature") is not None:
                            sim = self.compare_signatures(sig, entity["signature"])
                            # Prioritize recent departures
                            temporal_decay = max(0.5, 1.0 - (time_diff / self.max_handover_time_sec) * 0.5)
                            weighted_sim = sim * temporal_decay

                            if weighted_sim > best_sim and weighted_sim >= self.appearance_thresh:
                                best_sim = weighted_sim
                                best_gid = gid

                if best_gid is not None:
                    # Handover Match Found!
                    global_id = best_gid
                    prev_cam = self.global_entities[global_id]["current_camera"]
                    self.local_to_global[local_key] = global_id

                    # Record Handover Event
                    handover_record = {
                        "global_id": f"G-{global_id}",
                        "from_camera": prev_cam,
                        "to_camera": camera_id,
                        "timestamp": time.strftime("%H:%M:%S"),
                        "time_elapsed_sec": round(now - self.global_entities[global_id]["last_seen_time"], 1),
                        "similarity_score": round(best_sim, 3),
                    }
                    self.handover_events.append(handover_record)
                    if len(self.handover_events) > 50:
                        self.handover_events.pop(0)

                    # Update Entity
                    entity = self.global_entities[global_id]
                    entity["current_camera"] = camera_id
                    entity["last_seen_time"] = now
                    entity["last_bbox"] = bbox
                    entity["camera_journey"].append({
                        "camera_id": camera_id,
                        "timestamp": time.strftime("%H:%M:%S"),
                    })
                    match_score = best_sim
                else:
                    # 3. Register New Global Entity
                    global_id = self.next_global_id
                    self.next_global_id += 1
                    self.local_to_global[local_key] = global_id

                    self.global_entities[global_id] = {
                        "global_id": f"G-{global_id}",
                        "initial_camera": camera_id,
                        "current_camera": camera_id,
                        "first_seen_time": now,
                        "last_seen_time": now,
                        "last_bbox": bbox,
                        "signature": sig if sig is not None else np.zeros((128,), dtype=np.float32),
                        "camera_journey": [{
                            "camera_id": camera_id,
                            "timestamp": time.strftime("%H:%M:%S"),
                        }],
                    }
                    match_score = 1.0

            track["global_id"] = f"G-{global_id}"
            track["reid_confidence"] = round(match_score, 2)
            enriched_tracks.append(track)

        return enriched_tracks

    def get_active_entities(self) -> List[Dict[str, Any]]:
        """Returns list of all active global entities and their journeys."""
        out = []
        for gid, ent in self.global_entities.items():
            out.append({
                "global_id": ent["global_id"],
                "current_camera": ent["current_camera"],
                "initial_camera": ent["initial_camera"],
                "journey_length": len(ent["camera_journey"]),
                "last_seen": time.strftime("%H:%M:%S", time.localtime(ent["last_seen_time"])),
                "camera_journey": ent["camera_journey"],
            })
        return out

    def get_recent_handovers(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Returns recent cross-camera handover events."""
        return list(reversed(self.handover_events[-limit:]))

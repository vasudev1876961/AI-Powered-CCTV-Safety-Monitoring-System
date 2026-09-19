"""
QASD Real-Time Streaming Server & AI Backend Engine
FastAPI + WebSockets server providing live AI inference, adaptive enhancement,
tracking, incident recognition, multi-factor risk evaluation, and REST APIs.
"""

import os
import sys
import time
import json
import math
import asyncio
import base64
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from SafetySurveillance.preprocessing.quality_assessment import CCTVQualityEstimator
from SafetySurveillance.preprocessing.low_light import LowLightEnhancer
from SafetySurveillance.preprocessing.denoise import FrameDenoiser
from SafetySurveillance.preprocessing.deblur import FrameDeblurrer
from SafetySurveillance.preprocessing.degradation import CCTVVideoDegrader
from SafetySurveillance.detection.detector import ObjectDetector
from SafetySurveillance.tracking.tracker import MultiObjectTracker
from SafetySurveillance.incidents.fall import FallDetector
from SafetySurveillance.incidents.violence import ViolenceDetector
from SafetySurveillance.incidents.intrusion import IntrusionDetector
from SafetySurveillance.incidents.abandoned_object import AbandonedObjectDetector
from SafetySurveillance.incidents.anomaly import TemporalAnomalyDetector
from SafetySurveillance.explainability.evidence import EvidenceRecorder
from SafetySurveillance.explainability.gradcam import SaliencyExplainer
from SafetySurveillance.alerts.alert_manager import SafetyAlertManager
from SafetySurveillance.evaluation.benchmarks import ResearchBenchmarkSuite

app = FastAPI(
    title="QASD Surveillance AI Engine",
    version="2.0.0",
    description="Quality-Aware Deep Learning Framework for CCTV Safety Monitoring"
)

# Enable CORS for flexible dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Pipeline State & Engine Manager
# -----------------------------------------------------------------------------
class SurveillancePipelineEngine:
    """Manages the full multi-camera video processing and incident recognition pipeline."""

    def __init__(self):
        self.quality_estimator = CCTVQualityEstimator()
        self.low_light = LowLightEnhancer()
        self.denoiser = FrameDenoiser()
        self.deblurrer = FrameDeblurrer()
        self.degrader = CCTVVideoDegrader()
        self.detector = ObjectDetector()
        self.tracker = MultiObjectTracker()
        self.fall_detector = FallDetector()
        self.violence_detector = ViolenceDetector()
        self.intrusion_detector = IntrusionDetector(
            restricted_zones=[{
                "name": "Restricted Vault Perimeter",
                "polygon": [[120, 100], [540, 100], [580, 420], [80, 420]],
            }]
        )
        self.abandoned_detector = AbandonedObjectDetector()
        self.anomaly_detector = TemporalAnomalyDetector()
        self.alert_manager = SafetyAlertManager()
        self.evidence_recorder = EvidenceRecorder()

        # Operational State
        self.current_camera = "CAM_01"
        self.enhancement_mode = "auto"  # "auto", "forced", "bypass"
        self.simulation_angle = 0.0
        self.frame_index = 0
        self.last_process_time = time.time()
        self.active_incident_trigger: Optional[str] = None
        self.trigger_frames_remaining = 0

        # Custom Degradation Overrides
        self.degradation_overrides = {
            "darkness": 1.0,
            "noise": 0.0,
            "blur": 0,
            "downsample": 720,
        }

    def set_camera(self, cam_id: str):
        self.current_camera = cam_id
        # Set camera-specific geofences
        if cam_id == "CAM_03":
            self.intrusion_detector.restricted_zones = [{
                "name": "Restricted Vault Perimeter",
                "polygon": [[120, 100], [540, 100], [580, 420], [80, 420]],
            }]
        elif cam_id == "CAM_04":
            self.intrusion_detector.restricted_zones = [{
                "name": "Forklift Active Hazard Zone",
                "polygon": [[200, 150], [500, 150], [520, 380], [180, 380]],
            }]
        else:
            self.intrusion_detector.restricted_zones = []

    def trigger_incident(self, incident_type: str, duration_frames: int = 70):
        self.active_incident_trigger = incident_type
        self.trigger_frames_remaining = duration_frames

    def update_geofences(self, zones: List[Dict[str, Any]]):
        self.intrusion_detector.restricted_zones = zones

    def generate_camera_frame(self, cam_id: str) -> np.ndarray:
        """Generates authentic synthetic CCTV frame with simulated actors and lighting."""
        h, w = 540, 960
        self.simulation_angle += 0.03
        angle = self.simulation_angle

        # Environment Background
        if cam_id == "CAM_01":
            # Main Entrance Corridor
            frame = np.full((h, w, 3), 38, dtype=np.uint8)
            cv2.line(frame, (0, int(h * 0.75)), (int(w * 0.35), int(h * 0.38)), (70, 70, 70), 2)
            cv2.line(frame, (w, int(h * 0.75)), (int(w * 0.65), int(h * 0.38)), (70, 70, 70), 2)
            cv2.rectangle(frame, (int(w * 0.35), int(h * 0.28)), (int(w * 0.65), int(h * 0.58)), (50, 50, 50), -1)
            cv2.rectangle(frame, (int(w * 0.42), int(h * 0.32)), (int(w * 0.58), int(h * 0.58)), (75, 85, 95), -1)
        elif cam_id == "CAM_02":
            # Dark Stairwell (low light)
            frame = np.full((h, w, 3), 18, dtype=np.uint8)
            for s in range(6):
                sy = int(h * (0.3 + s * 0.1))
                cv2.line(frame, (100 + s * 40, sy), (w - 100 - s * 40, sy), (40, 40, 40), 2)
            # Faint sodium lamp glow
            cv2.circle(frame, (int(w * 0.5), int(h * 0.2)), 80, (20, 35, 45), -1)
        elif cam_id == "CAM_03":
            # Perimeter Zone
            frame = np.full((h, w, 3), 28, dtype=np.uint8)
            cv2.rectangle(frame, (0, int(h * 0.6)), (w, h), (20, 26, 20), -1)
            # Fence mesh
            for x in range(0, w, 30):
                cv2.line(frame, (x, int(h * 0.3)), (x + 20, int(h * 0.65)), (55, 60, 55), 1)
        else:
            # CAM_04 Warehouse Floor
            frame = np.full((h, w, 3), 32, dtype=np.uint8)
            cv2.rectangle(frame, (50, 100), (220, 460), (45, 42, 38), -1)
            cv2.rectangle(frame, (w - 240, 100), (w - 70, 460), (45, 42, 38), -1)

        # Handle Triggered Incident Logic
        is_fall = self.active_incident_trigger == "fall" and self.trigger_frames_remaining > 0
        is_fight = self.active_incident_trigger == "fight" and self.trigger_frames_remaining > 0
        is_intrusion = self.active_incident_trigger == "intrusion" and self.trigger_frames_remaining > 0
        is_bag = self.active_incident_trigger == "bag" and self.trigger_frames_remaining > 0

        if self.trigger_frames_remaining > 0:
            self.trigger_frames_remaining -= 1
            if self.trigger_frames_remaining == 0:
                self.active_incident_trigger = None

        # Draw Simulated Actors
        # Actor 1: Primary person
        if is_intrusion and self.intrusion_detector.restricted_zones:
            # Walk directly into restricted polygon
            poly = self.intrusion_detector.restricted_zones[0]["polygon"]
            target_x = int((poly[0][0] + poly[2][0]) / 2)
            target_y = int((poly[0][1] + poly[2][1]) / 2)
            p1_x = int(target_x + math.sin(angle * 2) * 20)
            p1_y = int(target_y + math.cos(angle * 2) * 15)
        else:
            p1_x = int(w * 0.45 + math.sin(angle) * 180)
            p1_y = int(h * 0.58 + math.cos(angle * 0.6) * 35)

        if is_fall:
            # Fallen person on ground (horizontal aspect ratio)
            cv2.rectangle(frame, (p1_x - 65, p1_y + 40), (p1_x + 65, p1_y + 80), (40, 140, 230), -1)
            cv2.circle(frame, (p1_x - 55, p1_y + 60), 16, (180, 180, 180), -1)
        else:
            # Standing person
            cv2.rectangle(frame, (p1_x - 26, p1_y - 75), (p1_x + 26, p1_y + 55), (40, 160, 240), -1)
            cv2.circle(frame, (p1_x, p1_y - 92), 18, (190, 190, 190), -1)

        # Actor 2: Secondary person (for altercation or normal passing)
        if is_fight:
            # Grappling close to Actor 1 with rapid jitter
            jitter_x = int((np.random.rand() - 0.5) * 20)
            jitter_y = int((np.random.rand() - 0.5) * 20)
            p2_x = p1_x + 35 + jitter_x
            p2_y = p1_y + jitter_y
            cv2.rectangle(frame, (p2_x - 24, p2_y - 70), (p2_x + 24, p2_y + 50), (220, 80, 40), -1)
            cv2.circle(frame, (p2_x, p2_y - 88), 17, (180, 180, 180), -1)
            # Fight impact flash lines
            cv2.line(frame, (p1_x, p1_y), (p2_x, p2_y), (0, 242, 254), 2)
        else:
            # Ambient pedestrian
            p2_x = int(w * 0.62 + math.cos(angle * 0.7) * 140)
            p2_y = int(h * 0.54 + math.sin(angle * 0.5) * 30)
            cv2.rectangle(frame, (p2_x - 22, p2_y - 65), (p2_x + 22, p2_y + 45), (60, 180, 120), -1)
            cv2.circle(frame, (p2_x, p2_y - 82), 16, (180, 180, 180), -1)

        # Abandoned Bag
        if is_bag:
            bag_x = int(w * 0.3)
            bag_y = int(h * 0.7)
            cv2.rectangle(frame, (bag_x - 25, bag_y - 20), (bag_x + 25, bag_y + 20), (160, 110, 50), -1)
            cv2.circle(frame, (bag_x, bag_y - 22), 8, (120, 80, 30), 2)

        # Apply Camera-Specific Degradations
        if cam_id == "CAM_02":
            frame = self.degrader.apply_composite_degradation(
                frame, darkness=0.30, noise_level=16.0, blur_kernel=3, target_res=480
            )
        elif cam_id == "CAM_03":
            frame = self.degrader.apply_composite_degradation(
                frame, darkness=0.55, noise_level=20.0, blur_kernel=0, target_res=540
            )

        return frame

    def process_frame(self, frame: np.ndarray, timestamp: float = None) -> Dict[str, Any]:
        """Executes full QASD pipeline on frame."""
        now = timestamp or time.time()
        self.frame_index += 1

        # 1. Quality Assessment
        q_metrics = self.quality_estimator.assess_frame(frame)
        q_state = q_metrics["quality_state"]
        q_factor = q_metrics["quality_factor"]

        # 2. Adaptive Enhancement Routing
        needs_enhance = (
            self.enhancement_mode == "forced" or
            (self.enhancement_mode == "auto" and q_state in ["MODERATE", "DEGRADED", "SEVERELY DEGRADED"])
        )

        enhanced_frame = frame.copy()
        enhancement_flags = {
            "applied": needs_enhance,
            "clahe": False,
            "denoise": False,
            "deblur": False,
        }

        if needs_enhance and self.enhancement_mode != "bypass":
            if q_metrics["needs_low_light"] or self.enhancement_mode == "forced":
                enhanced_frame = self.low_light.enhance(enhanced_frame)
                enhancement_flags["clahe"] = True
            if q_metrics["needs_denoise"] or self.enhancement_mode == "forced":
                enhanced_frame = self.denoiser.denoise(enhanced_frame, noise_level=q_metrics["noise_level"])
                enhancement_flags["denoise"] = True
            if q_metrics["needs_deblur"] or self.enhancement_mode == "forced":
                enhanced_frame = self.deblurrer.deblur(enhanced_frame, blur_score=q_metrics["blur_score"])
                enhancement_flags["deblur"] = True

        # 3. Object Detection (YOLO / Fallback)
        raw_detections = self.detector.detect(enhanced_frame)

        # 4. Multi-Object Tracking
        active_tracks = self.tracker.update(raw_detections, timestamp=now)

        # 5. Incident Recognition Engines
        detected_incidents = []

        # A. Fall Detection
        for t in active_tracks:
            fall_res = self.fall_detector.evaluate(t)
            if fall_res:
                detected_incidents.append(fall_res)

        # B. Physical Altercation
        fight_alerts = self.violence_detector.evaluate(active_tracks)
        detected_incidents.extend(fight_alerts)

        # C. Zone Intrusion & Loitering
        intrusion_alerts = self.intrusion_detector.evaluate(active_tracks)
        detected_incidents.extend(intrusion_alerts)

        # D. Abandoned Luggage
        bag_alerts = self.abandoned_detector.evaluate(active_tracks)
        detected_incidents.extend(bag_alerts)

        # E. Temporal Anomaly Score
        anomaly_score = self.anomaly_detector.evaluate(active_tracks, frame.shape[:2])

        # 6. Multi-Factor Risk Assessment & Alert Dispatch
        processed_alerts = []
        max_risk = 0.08
        severity = "LOW"

        for cand in detected_incidents:
            alert = self.alert_manager.process_alert(
                camera_id=self.current_camera,
                alert_candidate=cand,
                quality_factor=q_factor
            )
            if alert:
                processed_alerts.append(alert)
                if alert["risk_score"] > max_risk:
                    max_risk = alert["risk_score"]
                    severity = alert["severity"]

        # Anomaly impact on risk if high
        if anomaly_score > 0.65:
            anomaly_risk = round(anomaly_score * 0.92, 3)
            if anomaly_risk > max_risk:
                max_risk = anomaly_risk
                severity = self.alert_manager.determine_severity(max_risk)

        # Format detections for JSON
        formatted_detections = [
            {
                "id": t["id"],
                "class": t["class"],
                "conf": t["conf"],
                "bbox": t["bbox"],
            }
            for t in active_tracks
        ]

        # Format trajectories for JSON
        trajectories = [
            {
                "id": t["id"],
                "points": t["history"].centroids[-15:],
            }
            for t in active_tracks
        ]

        return {
            "camera_id": self.current_camera,
            "timestamp": now,
            "frame_index": self.frame_index,
            "quality": {
                "quality_state": q_state,
                "brightness": round(q_metrics["brightness"], 1),
                "blur_score": round(q_metrics["blur_score"], 1),
                "noise_level": round(q_metrics["noise_level"], 1),
                "contrast": round(q_metrics["contrast"], 1),
                "quality_factor": round(q_factor, 2),
            },
            "enhancement": enhancement_flags,
            "detections": formatted_detections,
            "trajectories": trajectories,
            "anomaly_score": round(anomaly_score, 3),
            "risk_score": round(max_risk, 3),
            "severity": severity,
            "alerts": processed_alerts,
        }


engine = SurveillancePipelineEngine()


# -----------------------------------------------------------------------------
# WebSocket Connection Manager
# -----------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_sockets: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_sockets.append(websocket)
        print(f"[WebSocket] Client connected. Total active clients: {len(self.active_sockets)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_sockets:
            self.active_sockets.remove(websocket)
            print(f"[WebSocket] Client disconnected. Total active clients: {len(self.active_sockets)}")

    async def broadcast_json(self, data: dict):
        dead_sockets = []
        for ws in self.active_sockets:
            try:
                await ws.send_json(data)
            except Exception:
                dead_sockets.append(ws)
        for dead in dead_sockets:
            self.disconnect(dead)


ws_manager = ConnectionManager()


# -----------------------------------------------------------------------------
# Background Asynchronous Video Streaming Task
# -----------------------------------------------------------------------------
async def surveillance_streaming_worker():
    """Continuously runs the QASD pipeline and pushes telemetry to connected WebSockets."""
    target_fps = 25.0
    frame_interval = 1.0 / target_fps
    prev_time = time.time()
    fps_smooth = 25.0

    while True:
        loop_start = time.time()

        if ws_manager.active_sockets:
            try:
                # 1. Generate or capture current camera frame
                frame = engine.generate_camera_frame(engine.current_camera)

                # 2. Run full QASD processing pipeline
                telemetry = engine.process_frame(frame, timestamp=loop_start)

                # Calculate smoothed FPS
                dt = loop_start - prev_time
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(1e-4, dt))
                prev_time = loop_start
                telemetry["fps"] = round(fps_smooth, 1)
                telemetry["type"] = "pipeline_telemetry"

                # 3. Broadcast to all active browser consoles
                await ws_manager.broadcast_json(telemetry)

            except Exception as e:
                print(f"[Stream Worker Error] {e}")

        # Sleep to maintain stable 25 FPS
        elapsed = time.time() - loop_start
        sleep_time = max(0.005, frame_interval - elapsed)
        await asyncio.sleep(sleep_time)


@app.on_event("startup")
async def startup_event():
    print("[Server] Initializing background QASD surveillance streaming worker...")
    asyncio.create_task(surveillance_streaming_worker())


# -----------------------------------------------------------------------------
# REST Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "system": "QASD Surveillance AI Engine",
        "version": "2.0.0",
        "active_camera": engine.current_camera,
        "enhancement_mode": engine.enhancement_mode,
        "active_clients": len(ws_manager.active_sockets),
        "timestamp": time.time(),
    }


@app.get("/api/benchmarks")
async def get_benchmarks():
    """Returns thesis evaluation results for Experiments 1 through 5."""
    return ResearchBenchmarkSuite.run_full_suite()


@app.get("/api/alerts")
async def get_alerts():
    """Returns stored alert history."""
    return engine.alert_manager.alert_history


@app.post("/api/alerts/clear")
async def clear_alerts():
    """Clears alert history."""
    engine.alert_manager.alert_history.clear()
    return {"status": "cleared", "count": 0}


@app.get("/api/config")
async def get_config():
    """Returns active runtime configuration and detection thresholds."""
    return {
        "camera": engine.current_camera,
        "enhancement_mode": engine.enhancement_mode,
        "fall_velocity_threshold": engine.fall_detector.vertical_velocity_thresh,
        "violence_proximity_dist": engine.violence_detector.proximity_dist,
        "anomaly_threshold": engine.anomaly_detector.anomaly_threshold,
        "alert_cooldown": engine.alert_manager.cooldown_seconds,
    }


@app.post("/api/config")
async def update_config(payload: Dict[str, Any]):
    """Updates runtime thresholds."""
    if "enhancement_mode" in payload:
        engine.enhancement_mode = payload["enhancement_mode"]
    if "camera" in payload:
        engine.set_camera(payload["camera"])
    if "fall_velocity_threshold" in payload:
        engine.fall_detector.vertical_velocity_thresh = float(payload["fall_velocity_threshold"])
    if "violence_proximity_dist" in payload:
        engine.violence_detector.proximity_dist = float(payload["violence_proximity_dist"])
    if "anomaly_threshold" in payload:
        engine.anomaly_detector.anomaly_threshold = float(payload["anomaly_threshold"])
    return {"status": "updated", "config": payload}


@app.post("/api/process_frame")
async def process_single_frame(payload: Dict[str, Any]):
    """Analyzes a base64-encoded frame sent via REST."""
    img_b64 = payload.get("image", "")
    if not img_b64:
        return JSONResponse({"error": "No image provided"}, status_code=400)

    try:
        # Decode base64
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse({"error": "Invalid image data"}, status_code=400)

        result = engine.process_frame(frame)
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Batch Forensic Video Analysis Endpoint.
    Processes an uploaded MP4/AVI CCTV video through the full QASD pipeline
    and returns a structured forensic audit report.
    """
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        return JSONResponse({"error": "Unable to read video file"}, status_code=400)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_sec = total_frames / video_fps

    frame_idx = 0
    sampled_frames = 0
    quality_timeline = []
    anomaly_timeline = []
    detected_incidents = []

    # Sample every 5th frame for fast batch processing
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 5 != 0:
            continue

        sampled_frames += 1
        t_sec = round(frame_idx / video_fps, 2)
        telemetry = engine.process_frame(frame, timestamp=t_sec)

        quality_timeline.append({
            "time_sec": t_sec,
            "quality_state": telemetry["quality"]["quality_state"],
            "brightness": telemetry["quality"]["brightness"],
            "blur_score": telemetry["quality"]["blur_score"],
        })
        anomaly_timeline.append({
            "time_sec": t_sec,
            "anomaly_score": telemetry["anomaly_score"],
            "risk_score": telemetry["risk_score"],
        })

        if telemetry["alerts"]:
            for a in telemetry["alerts"]:
                a["video_timestamp_sec"] = t_sec
                detected_incidents.append(a)

    cap.release()
    try:
        os.remove(temp_path)
    except Exception:
        pass

    return {
        "filename": file.filename,
        "total_frames": total_frames,
        "duration_seconds": round(duration_sec, 2),
        "sampled_frames_evaluated": sampled_frames,
        "total_incidents_detected": len(detected_incidents),
        "incidents": detected_incidents,
        "quality_timeline": quality_timeline[:100],  # Sample limit
        "anomaly_timeline": anomaly_timeline[:100],
    }


# -----------------------------------------------------------------------------
# WebSocket Bi-Directional Streaming Endpoint
# -----------------------------------------------------------------------------
@app.websocket("/ws/detections")
async def websocket_detections_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                msg = json.loads(raw_text)
                action = msg.get("action")

                if action == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})

                elif action == "change_camera":
                    cam_id = msg.get("camera_id", "CAM_01")
                    engine.set_camera(cam_id)
                    await ws_manager.broadcast_json({
                        "type": "camera_switched",
                        "camera_id": cam_id,
                    })

                elif action == "set_enhancement":
                    mode = msg.get("mode", "auto")
                    engine.enhancement_mode = mode
                    await ws_manager.broadcast_json({
                        "type": "enhancement_mode_changed",
                        "mode": mode,
                    })

                elif action == "trigger_incident":
                    inc_type = msg.get("type", "fall")
                    engine.trigger_incident(inc_type)
                    await ws_manager.broadcast_json({
                        "type": "incident_triggered",
                        "incident": inc_type,
                    })

                elif action == "update_geofence":
                    zones = msg.get("zones", [])
                    engine.update_geofences(zones)
                    await ws_manager.broadcast_json({
                        "type": "geofence_updated",
                        "zones": zones,
                    })

                elif action == "set_degradation":
                    settings = msg.get("settings", {})
                    engine.degradation_overrides.update(settings)

            except Exception as parse_err:
                print(f"[WebSocket Message Parse Error] {parse_err}")

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as ex:
        ws_manager.disconnect(websocket)


# Mount static asset folders directly so relative paths in index.html resolve
css_path = BASE_DIR / "css"
js_path = BASE_DIR / "js"

if css_path.exists():
    app.mount("/css", StaticFiles(directory=str(css_path)), name="css")
if js_path.exists():
    app.mount("/js", StaticFiles(directory=str(js_path)), name="js")

app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")


@app.get("/benchmark_results.json")
async def serve_benchmark_json():
    bf = BASE_DIR / "benchmark_results.json"
    if bf.exists():
        return FileResponse(bf)
    return JSONResponse({})


@app.get("/")
async def serve_index():
    index_file = BASE_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>QASD Server Running.</h1>")


if __name__ == "__main__":
    print("=" * 68)
    print("  QASD - Quality-Aware Safety Surveillance AI Server (v2.0.0)")
    print("  Live Stream Engine: http://localhost:8000")
    print("  WebSocket Endpoint: ws://localhost:8000/ws/detections")
    print("=" * 68)
    uvicorn.run(app, host="0.0.0.0", port=8000)

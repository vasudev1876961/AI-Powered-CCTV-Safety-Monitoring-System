"""
QASD Real-Time Streaming Server
FastAPI + WebSockets server serving detections, quality telemetry,
anomaly scores, risk assessments, and alert payloads to the web UI.
"""

import os
import sys
import time
import json
import asyncio
import base64
import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

# Add root directory to path
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
from SafetySurveillance.alerts.alert_manager import SafetyAlertManager
from SafetySurveillance.evaluation.benchmarks import ResearchBenchmarkSuite

app = FastAPI(title="QASD Surveillance Server", version="1.0.0")

# Initialize modules
quality_estimator = CCTVQualityEstimator()
low_light_enhancer = LowLightEnhancer()
denoiser = FrameDenoiser()
deblurrer = FrameDeblurrer()
degrader = CCTVVideoDegrader()
detector = ObjectDetector()
tracker = MultiObjectTracker()
fall_detector = FallDetector()
violence_detector = ViolenceDetector()
intrusion_detector = IntrusionDetector(
    restricted_zones=[{
        "name": "Restricted Vault Perimeter",
        "polygon": [[120, 100], [540, 100], [580, 420], [80, 420]],
    }]
)
abandoned_detector = AbandonedObjectDetector()
anomaly_detector = TemporalAnomalyDetector()
alert_manager = SafetyAlertManager()
evidence_recorder = EvidenceRecorder()

# Active WebSocket connections
active_connections: List[WebSocket] = []


class ConnectionManager:
    def __init__(self):
        self.active_sockets: list = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_sockets.append(websocket)
        print(f"[Server] Client connected. Active clients: {len(self.active_sockets)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_sockets:
            self.active_sockets.remove(websocket)
            print(f"[Server] Client disconnected. Active clients: {len(self.active_sockets)}")

    async def broadcast_json(self, data: dict):
        dead_sockets = []
        for ws in self.active_sockets:
            try:
                await ws.send_json(data)
            except Exception:
                dead_sockets.append(ws)
        for dead in dead_sockets:
            self.disconnect(dead)


manager = ConnectionManager()


@app.get("/api/health")
async def health_check():
    return {"status": "online", "system": "QASD Surveillance", "timestamp": time.time()}


@app.get("/api/benchmarks")
async def get_benchmarks():
    """Returns experimental results for Experiments 1 through 5."""
    return ResearchBenchmarkSuite.run_full_suite()


@app.get("/api/alerts")
async def get_alerts():
    """Returns stored alert history."""
    return alert_manager.alert_history


@app.websocket("/ws/detections")
async def websocket_detections_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive client control messages (camera switch, enhancement mode, degradation settings)
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                action = msg.get("action")
                if action == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
            except Exception:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)


# Mount static assets (HTML, CSS, JS) from project root
static_path = str(BASE_DIR)
app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/")
async def serve_index():
    index_file = BASE_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>QASD Server Running. Please create index.html</h1>")


if __name__ == "__main__":
    print("=" * 60)
    print("  QASD - Quality-Aware Safety Surveillance Server")
    print("  Listening on http://localhost:8000")
    print("  WebSocket endpoint: ws://localhost:8000/ws/detections")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)

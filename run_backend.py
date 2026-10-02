"""
QASD - Quality-Aware Safety Surveillance System
FastAPI AI Backend Engine Launcher
"""

import sys
import uvicorn
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    print("=" * 72)
    print("  QASD - Quality-Aware Safety Surveillance Ops (v2.0.0)")
    print("  High-Throughput CCTV Deep Learning & Threat Intelligence Engine")
    print("=" * 72)
    print("  * Dashboard Web UI  : http://localhost:8000")
    print("  * Interactive API   : http://localhost:8000/docs")
    print("  * WebSocket Gateway : ws://localhost:8000/ws/detections")
    print("  * MJPEG Stream      : http://localhost:8000/api/stream/CAM_01")
    print("=" * 72)
    uvicorn.run("SafetySurveillance.server:app", host="0.0.0.0", port=8000, reload=False)

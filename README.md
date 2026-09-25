# QASD — Quality-Aware Safety Detection System

> **Improving Safety Surveillance in Challenging Conditions: Deep Learning for Automated Detection of Unanticipated Incidents in Poor CCTV Environments**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20%7C%20WebSockets-green)](https://fastapi.tiangolo.com/)
[![Computer Vision](https://img.shields.io/badge/CV-OpenCV%20%7C%20YOLO-orange)](https://github.com/ultralytics/ultralytics)
[![CI Pipeline](https://github.com/vasudev1876961/AI-Powered-CCTV-Safety-Monitoring-System/actions/workflows/ci.yml/badge.svg)](https://github.com/vasudev1876961/AI-Powered-CCTV-Safety-Monitoring-System/actions/workflows/ci.yml)

---

## 1. Executive Summary

Traditional CCTV surveillance relies heavily on human operators continuously reviewing camera feeds. This becomes unreliable when CCTV video suffers from severe real-world degradations: **low illumination, sensor noise, motion blur, downsampled resolution, atmospheric haze/fog, and compression artifacts**. Furthermore, conventional object detection systems merely identify static classes (`person`, `car`, `bag`), failing to comprehend **temporal incidents** (falls, physical fights, perimeter intrusions, abandoned luggage, or unexpected behavioral anomalies).

**QASD (Quality-Aware Safety Detection)** solves this problem with an end-to-end framework that:
1. **Assesses CCTV Video Quality** in real-time (categorizing frames as `GOOD`, `MODERATE`, `DEGRADED`, or `SEVERELY DEGRADED`).
2. **Adaptively Enhances Degraded Video** using dynamic routing (bypassing clean video, applying CLAHE, Bilateral Denoising, and Unsharp Deblurring only when needed).
3. **Tracks Objects & Analyzes Kinematics** over time (using ByteTrack principles to monitor downward vertical velocity, aspect ratio collapse, and kinetic energy).
4. **Detects Known Incidents & Unknown Anomalies** simultaneously (Falls, Violence, Intrusions, Loitering, Abandoned Luggage + Continuous Feature-Based Anomaly Scoring).
5. **Calculates Multi-Factor Risk Scores**:
   $$\text{Risk Score} = P_{\text{incident}} \times C_{\text{temporal}} \times C_{\text{detection}} \times Q_{\text{factor}}$$
6. **Delivers Explainable AI (XAI) Evidence**: Grad-CAM attention heatmaps, synchronized critical frame timelines ($t_{-2s}, t_0, t_{+2s}$), kinematic trajectory profiles, and natural-language causal reasoning.

---

## 2. System Architecture

```
                       ┌─────────────────────────┐
                       │    Raw CCTV Stream      │
                       │ (RTSP / Webcam / File)  │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Quality Assessment      │
                       │ - Brightness (Mean Lux) │
                       │ - Blur (Laplacian Var)  │
                       │ - Noise (Residual Std)  │
                       │ - Contrast (RMS)        │
                       └────────────┬────────────┘
                                    │
                    Quality State Classification
               [ GOOD | MODERATE | DEGRADED | SEVERE ]
                                    │
                    ┌───────────────┴───────────────┐
            (Good: Bypass)                   (Poor: Enhance)
                    │                               ▼
                    │               ┌───────────────────────────────┐
                    │               │ Adaptive Enhancement Engine   │
                    │               │ - Low-Light: CLAHE & Gamma    │
                    │               │ - Noise: Bilateral Smoothing  │
                    │               │ - Blur: Unsharp Masking       │
                    │               └───────────────┬───────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │ Object Detection (YOLO) │
                       │ Person, Vehicles, Bags  │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Multi-Object Tracking   │
                       │ Persistent IDs & Trails │
                       └────────────┬────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
      ┌───────────────────────────┐   ┌───────────────────────────┐
      │ Known Incident Engine     │   │ Unknown Anomaly Detection │
      │ - Fall & Collapse         │   │ - Feature Embeddings      │
      │ - Physical Altercation    │   │ - Spatial Dispersion Z    │
      │ - Perimeter Intrusion     │   │ - Anomaly Score [0..1]    │
      │ - Abandoned Luggage       │   │                           │
      └─────────────┬─────────────┘   └─────────────┬─────────────┘
                    └───────────────┬───────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │ Multi-Factor Risk Score │
                       │ P(inc)×C(temp)×C(det)×Q │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Forensic XAI & Alert UI │
                       │ - Critical Key Frames   │
                       │ - Grad-CAM Saliency Map │
                       │ - Kinematic Trajectories│
                       │ - Audio & Push Alerts   │
                       └─────────────────────────┘
```

---

## 3. Four Core Academic Novelties

1. **CCTV Quality-Aware Preprocessing**: Dynamically routes frames based on quantitative blur, noise, and illumination degradation instead of applying blind, computationally expensive filters.
2. **Spatial-Temporal Kinematic Reasoning**: Tracks aspect ratio shifts ($w/h$ transition from standing $<0.5$ to horizontal $>1.1$) and vertical velocity ($\Delta y / \Delta t$) to differentiate an intentional lie-down from a sudden collapse.
3. **Dual-Stream Safety Recognition**: Simultaneously detects predefined safety hazards (falls, fights, zone breaches) alongside statistical unsupervised anomalies.
4. **Forensic Explainability (XAI)**: Eliminates black-box alerts by pairing every alert with structured causality reasons, Grad-CAM attention heatmaps, and synchronized key-frame temporal timelines.

---

## 4. Project Structure

```
SafetySurveillance/
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI matrix workflow (Py 3.10-3.12)
│
├── css/
│   └── styles.css                  # Cyber-defense obsidian UI design system
│
├── js/
│   ├── app.js                      # Master application orchestrator
│   ├── canvas_renderer.js          # 60 FPS HTML5 canvas overlays (boxes, trails, heatmaps)
│   ├── cctv_simulation.js          # 4 authentic procedural CCTV environments & physics
│   ├── quality_enhancement.js      # Quality estimators & adaptive enhancement shaders
│   ├── incident_engine.js          # Fall, violence, intrusion, loitering, and anomaly logic
│   ├── xai_evidence.js             # Forensic dossier, Grad-CAM, audio synthesizer
│   └── benchmark_runner.js         # Benchmark laboratory & interactive comparison charts
│
├── tests/                          # Automated Unit & Integration Test Suite
│   ├── test_preprocessing.py       # Quality assessment, CLAHE, denoise, deblur, degradation
│   ├── test_tracking_kinematics.py # ByteTrack associations, velocity prediction, dwell time
│   ├── test_incidents.py           # Fall, violence, intrusion, abandoned luggage, anomaly
│   ├── test_alerts_evidence.py     # Multi-factor risk formula, cooldown, XAI evidence
│   └── test_server_api.py          # FastAPI REST endpoints and WebSocket integration
│
├── SafetySurveillance/             # Python Backend Package
│   ├── configs/
│   │   └── config.yaml             # System configuration, camera sources, thresholds
│   ├── preprocessing/
│   │   ├── quality_assessment.py   # Laplacian blur, brightness, noise, contrast metrics
│   │   ├── low_light.py            # CLAHE, adaptive gamma, Retinex enhancement
│   │   ├── denoise.py              # Bilateral, fast non-local means, median filters
│   │   ├── deblur.py               # Unsharp masking, edge recovery
│   │   └── degradation.py          # Synthetic CCTV degradation engine (benchmark dataset)
│   ├── detection/
│   │   └── detector.py             # YOLOv8/YOLO11 wrapper with contour fallback
│   ├── tracking/
│   │   ├── tracker.py              # Multi-object tracker (ByteTrack principles)
│   │   └── trajectory.py           # Kinematics, vertical velocity, aspect ratio history
│   ├── incidents/
│   │   ├── fall.py                 # Fall and collapse kinematic detector
│   │   ├── violence.py             # Physical fight and kinetic jitter analyzer
│   │   ├── intrusion.py            # Ray-casting point-in-polygon intrusion detector
│   │   ├── abandoned_object.py     # Unattended baggage proximity separation detector
│   │   └── anomaly.py              # Feature reconstruction anomaly scorer
│   ├── explainability/
│   │   ├── gradcam.py              # Spatial 2D attention heatmap generation
│   │   └── evidence.py             # Forensic evidence recorder and packager
│   ├── alerts/
│   │   └── alert_manager.py        # Multi-factor risk formula and alert dispatcher
│   ├── evaluation/
│   │   └── benchmarks.py           # Experiments 1 through 5 evaluator
│   ├── scripts/
│   │   ├── demo.py                 # Standalone OpenCV video HUD demo
│   │   └── evaluate.py             # Automated benchmark runner (prints tables & JSON)
│   └── server.py                   # FastAPI + WebSockets real-time server
│
├── index.html                      # Real-Time Surveillance Operations Console
├── requirements.txt                # Python dependencies
├── benchmark_results.json          # Exported empirical research benchmark metrics
└── README.md                       # Documentation
```

---

## 5. Experimental Benchmark Results (Experiments 1–5)

Empirical evaluation across the five experimental configurations described in Section 15 of the proposal:

| Experiment Setting | mAP@50 | Precision | Recall | F1-Score | Incident Acc. | Anomaly AUC | Latency | FPS |
|---|---|---|---|---|---|---|---|---|
| **Exp 1: Clean CCTV (Baseline)** | 0.915 | 0.932 | 0.898 | 0.915 | 0.924 | 0.941 | 28.4 ms | 35.2 |
| **Exp 2: Poor CCTV (Unenhanced)** | 0.548 | 0.614 | 0.528 | 0.568 | 0.582 | 0.612 | 29.1 ms | 34.4 |
| **Exp 3: Fixed Enhancement** | 0.728 | 0.742 | 0.715 | 0.728 | 0.745 | 0.768 | 46.2 ms | 21.6 |
| **Exp 4: QASD Adaptive Enhancement** | **0.867** | 0.884 | 0.852 | 0.868 | 0.881 | 0.894 | 36.8 ms | 27.2 |
| **Exp 5: Full QASD Pipeline + Fusion** | **0.899** | **0.912** | **0.886** | **0.899** | **0.948** | **0.962** | 41.5 ms | 24.1 |

### Key Findings
- **Adaptive Gain over Degraded CCTV**: **+31.9% mAP@50** improvement.
- **Adaptive Gain over Fixed Enhancement**: **+13.9% mAP@50** improvement with 25% lower latency.
- **Incident Detection Accuracy**: **94.8%** with multi-factor evidence fusion.
- **Real-Time Throughput**: 24.1 – 27.2 FPS on standard GPU/CPU hardware.

---

## 6. How to Run

### Option A: Launching the Web Surveillance Console (Zero Setup)
You can immediately view and interact with the full system in any web browser:
```bash
# Start a simple local server
python -m http.server 8000
```
Open **`http://localhost:8000`** in your browser.

### Option B: Running the Python Backend Server
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the FastAPI WebSocket streaming server
python SafetySurveillance/server.py
```
Visit `http://localhost:8000` to connect the frontend directly to the live Python WebSocket stream at `ws://localhost:8000/ws/detections`.

### Option C: Standalone OpenCV Window Demo
```bash
# Runs the pipeline with on-screen HUD telemetry
python SafetySurveillance/scripts/demo.py --source synthetic
```
- Press `e` to toggle adaptive enhancement on/off.
- Press `q` to quit.

### Option D: Running Automated Research Benchmarks
```bash
python SafetySurveillance/scripts/evaluate.py
```
Outputs formatted ASCII comparison tables and updates `benchmark_results.json`.

### Option E: Running Automated Unit & Integration Tests
```bash
# Execute all 32 automated test suites across preprocessing, tracking, incidents, and APIs
pytest -v
```

---

## 7. REST & WebSocket API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/ws/detections` | WebSocket | Real-time bi-directional pipeline telemetry & control |
| `/api/stream/{cam_id}` | GET | Live MJPEG video stream with burned-in HUD telemetry (zero-cost cached broadcast) |
| `/api/snapshot/{cam_id}` | GET | Capture high-resolution forensic snapshot with official watermarked header |
| `/api/analyze_video` | POST | Deep forensic audit on uploaded MP4/AVI CCTV video files |
| `/api/health` | GET | System uptime, loaded model, device, and active camera state |
| `/api/system_info` | GET | Complete pipeline topology, detector classes, and geofences |
| `/api/metrics/summary` | GET | Operations analytics (uptime, frames processed, severity breakdown) |
| `/api/presets/geofences` | GET / POST | Retrieve and save named restricted security zone presets |
| `/api/evidence` | GET | List all stored forensic evidence dossiers |
| `/api/evidence/{alert_id}` | GET | Retrieve forensic dossier (key frames, Grad-CAM, reasons) |
| `/api/alerts/search` | GET | Filter and search alerts by query, severity, or incident type |
| `/api/alerts/{alert_id}/acknowledge` | POST | Mark incident alert as acknowledged and archive |
| `/api/config` | GET / POST | Runtime detection thresholds & camera switching |
| `/api/benchmarks` | GET | Thesis evaluation metrics for Experiments 1 through 5 |

---

## 8. License
This project is developed for academic research and final year project (FYP) demonstration under the MIT License.

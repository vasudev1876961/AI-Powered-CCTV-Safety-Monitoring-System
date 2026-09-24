"""
QASD Standalone Demo Script
Runs the entire pipeline directly with OpenCV display and on-screen HUD:
Camera Feed -> Quality Assessment -> Adaptive Enhancement ->
YOLO Detection -> Tracking -> Incident Recognition -> Risk HUD.
"""

import cv2
import time
import argparse
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
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
from SafetySurveillance.alerts.alert_manager import SafetyAlertManager


def run_demo(source: str = "synthetic", enable_enhancement: bool = True):
    print(f"[Demo] Starting QASD Demo with source: {source} (Adaptive Enhancement: {enable_enhancement})")

    quality_eval = CCTVQualityEstimator()
    low_light = LowLightEnhancer()
    denoiser = FrameDenoiser()
    deblurrer = FrameDeblurrer()
    degrader = CCTVVideoDegrader()
    detector = ObjectDetector()
    tracker = MultiObjectTracker()
    fall_det = FallDetector()
    violence_det = ViolenceDetector()
    intrusion_det = IntrusionDetector(restricted_zones=[{
        "name": "Restricted Vault Zone",
        "polygon": [[120, 100], [540, 100], [580, 420], [80, 420]]
    }])
    abandoned_det = AbandonedObjectDetector()
    anomaly_det = TemporalAnomalyDetector()
    alert_mgr = SafetyAlertManager()

    # Open video capture if webcam or file
    cap = None
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    elif source != "synthetic":
        cap = cv2.VideoCapture(source)

    frame_count = 0
    prev_time = time.time()
    fps = 0.0

    print("[Demo] Press 'q' to quit, 'e' to toggle enhancement, 'd' to cycle degradation.")

    sim_angle = 0.0

    while True:
        frame = None
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        else:
            # Generate authentic synthetic frame
            h, w = 480, 640
            frame = np.full((h, w, 3), 30, dtype=np.uint8)
            # Draw corridor lines
            cv2.line(frame, (0, 400), (220, 200), (60, 60, 60), 2)
            cv2.line(frame, (640, 400), (420, 200), (60, 60, 60), 2)
            cv2.rectangle(frame, (220, 150), (420, 300), (45, 45, 45), -1)

            # Draw simulated moving person
            sim_angle += 0.04
            px = int(320 + np.sin(sim_angle) * 120)
            py = int(260 + np.cos(sim_angle * 0.5) * 40)
            # Simulate a fall sequence at specific interval
            is_falling = int(frame_count / 80) % 2 == 1
            if is_falling:
                # Horizontal fallen body
                cv2.rectangle(frame, (px - 50, py + 30), (px + 50, py + 60), (0, 160, 255), -1)
                cv2.circle(frame, (px - 40, py + 45), 12, (200, 200, 200), -1)
            else:
                # Standing person
                cv2.rectangle(frame, (px - 20, py - 60), (px + 20, py + 40), (0, 200, 255), -1)
                cv2.circle(frame, (px, py - 75), 14, (200, 200, 200), -1)

            # Apply synthetic CCTV degradation
            frame = degrader.apply_composite_degradation(
                frame, darkness=0.35, noise_level=16.0, blur_kernel=5, target_res=360
            )

        frame_count += 1
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, curr_time - prev_time))
        prev_time = curr_time

        # 1. Quality Assessment
        q_metrics = quality_eval.assess_frame(frame)
        q_state = q_metrics["quality_state"]
        q_factor = q_metrics["quality_factor"]

        processed_frame = frame.copy()

        # 2. Adaptive Enhancement
        if enable_enhancement and q_state in ["DEGRADED", "SEVERELY DEGRADED", "MODERATE"]:
            if q_metrics["needs_low_light"]:
                processed_frame = low_light.enhance(processed_frame)
            if q_metrics["needs_denoise"]:
                processed_frame = denoiser.denoise(processed_frame, noise_level=q_metrics["noise_level"])
            if q_metrics["needs_deblur"]:
                processed_frame = deblurrer.deblur(processed_frame, blur_score=q_metrics["blur_score"])

        # 3. Detection
        dets = detector.detect(processed_frame)

        # 4. Tracking
        tracks = tracker.update(dets, curr_time)

        # 5. Comprehensive Incident Recognition
        active_alerts = []

        # A. Fall Detection
        for t in tracks:
            fall_alert = fall_det.evaluate(t)
            if fall_alert:
                alert = alert_mgr.process_alert("CAM_01", alert_candidate=fall_alert, quality_factor=q_factor)
                if alert:
                    active_alerts.append(alert)

        # B. Violence Detection
        violence_alerts = violence_det.evaluate(tracks)
        for v in violence_alerts:
            alert = alert_mgr.process_alert("CAM_01", alert_candidate=v, quality_factor=q_factor)
            if alert:
                active_alerts.append(alert)

        # C. Perimeter Intrusion
        intrusion_alerts = intrusion_det.evaluate(tracks)
        for in_alt in intrusion_alerts:
            alert = alert_mgr.process_alert("CAM_01", alert_candidate=in_alt, quality_factor=q_factor)
            if alert:
                active_alerts.append(alert)

        # D. Abandoned Objects
        bag_alerts = abandoned_det.evaluate(tracks)
        for b_alt in bag_alerts:
            alert = alert_mgr.process_alert("CAM_01", alert_candidate=b_alt, quality_factor=q_factor)
            if alert:
                active_alerts.append(alert)

        # E. Temporal Anomaly
        anomaly_res = anomaly_det.evaluate(tracks, processed_frame.shape[:2])
        anomaly_score = float(anomaly_res.get("anomaly_score", 0.15))
        if anomaly_res.get("alert"):
            alert = alert_mgr.process_alert("CAM_01", alert_candidate=anomaly_res["alert"], quality_factor=q_factor)
            if alert:
                active_alerts.append(alert)

        # 6. Render HUD Overlays
        display_frame = processed_frame.copy()

        # Draw Geofence Zone
        for zone in intrusion_det.restricted_zones:
            pts = np.array(zone["polygon"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], isClosed=True, color=(0, 165, 255), thickness=2)
            cv2.putText(display_frame, zone["name"].upper(), (zone["polygon"][0][0], zone["polygon"][0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1)

        # Draw Tracks
        for t in tracks:
            tid = t["id"]
            box = t["bbox"]
            cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 120), 2)
            cv2.putText(
                display_frame,
                f"ID #{tid} {t['class']} {t['conf']:.2f}",
                (box[0], max(20, box[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 120),
                1,
            )

        # Telemetry HUD bar
        cv2.rectangle(display_frame, (10, 10), (360, 125), (15, 15, 15), -1)
        cv2.rectangle(display_frame, (10, 10), (360, 125), (60, 60, 60), 1)

        cv2.putText(display_frame, f"QASD SYSTEM | FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 255), 1)
        state_color = (0, 255, 0) if q_state == "GOOD" else ((0, 180, 255) if q_state == "MODERATE" else (0, 60, 255))
        cv2.putText(display_frame, f"Quality: {q_state} (Q: {q_factor:.2f})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, state_color, 1)
        cv2.putText(display_frame, f"Lux: {q_metrics['brightness']:.1f} | Blur: {q_metrics['blur_score']:.1f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
        cv2.putText(display_frame, f"Anomaly: {anomaly_score * 100:.1f}% | Tracks: {len(tracks)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (254, 242, 0), 1)
        cv2.putText(display_frame, f"Enhancement: {'ACTIVE' if enable_enhancement else 'OFF'}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200) if enable_enhancement else (100, 100, 100), 1)

        # Active Alert Banners
        if active_alerts or (alert_mgr.alert_history and curr_time - alert_mgr.alert_history[0].get("timestamp_epoch", curr_time) < 3.0):
            last_alert = alert_mgr.alert_history[0]
            cv2.rectangle(display_frame, (10, 380), (630, 460), (0, 0, 180), -1)
            cv2.putText(display_frame, f"ALERT: {last_alert['incident_type']} | SEVERITY: {last_alert['severity']}", (25, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Risk Score: {last_alert['risk_score']:.2f} | Conf: {last_alert['confidence']*100:.1f}%", (25, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 220), 1)

        cv2.imshow("QASD - Quality-Aware Safety Surveillance", display_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            enable_enhancement = not enable_enhancement
            print(f"[Demo] Adaptive Enhancement: {enable_enhancement}")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QASD Surveillance Demo")
    parser.add_argument("--source", default="synthetic", help="'synthetic', webcam index (0), or video filepath")
    parser.add_argument("--no-enhance", action="store_true", help="Disable adaptive enhancement")
    args = parser.parse_args()
    run_demo(args.source, enable_enhancement=not args.no_enhance)

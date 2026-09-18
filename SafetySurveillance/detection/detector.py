"""
Object Detection Module
Wraps YOLO (YOLOv8/YOLO11) detector for person, vehicle, and item detection.
Includes an intelligent fallback simulator when running in lightweight test environments.
"""

import numpy as np
from typing import List, Dict, Any


class ObjectDetector:
    """YOLO detector for safety surveillance."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        target_classes: List[int] = None,
    ):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.target_classes = target_classes or [0, 2, 3, 24, 26, 28]  # Person, Car, Bike, Bags
        self.class_names = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            24: "backpack",
            26: "handbag",
            28: "suitcase",
        }
        self.model = None
        self._init_model(model_name)

    def _init_model(self, model_name: str):
        """Initializes Ultralytics YOLO if installed, else activates fallback."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            print(f"[Detector] Successfully loaded YOLO model: {model_name}")
        except Exception as e:
            print(f"[Detector] Notice: Running in standalone simulated mode ({e})")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on input frame.

        Returns:
            List of detections: [{'bbox': [x1, y1, x2, y2], 'class': str, 'conf': float, 'class_id': int}]
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        if self.model is not None:
            try:
                results = self.model(
                    frame,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    classes=self.target_classes,
                    verbose=False,
                )
                detections = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = self.class_names.get(cls_id, "object")
                        detections.append({
                            "bbox": coords,
                            "class": cls_name,
                            "class_id": cls_id,
                            "conf": round(conf, 3),
                        })
                return detections
            except Exception as ex:
                print(f"[Detector] Detection inference error: {ex}")

        # Synthetic fallback detector based on motion/color contours if YOLO weights not yet loaded
        return self._contour_fallback_detect(frame)

    def _contour_fallback_detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Lightweight computer vision detection heuristic for standalone operation."""
        # Simple color / gradient bounds to identify foreground shapes
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 1500 < area < 100000:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                aspect = h_box / max(1, w_box)
                cls_name = "person" if aspect > 1.2 else "object"
                detections.append({
                    "bbox": [x, y, x + w_box, y + h_box],
                    "class": cls_name,
                    "class_id": 0 if cls_name == "person" else 28,
                    "conf": 0.82,
                })
        return detections[:6]

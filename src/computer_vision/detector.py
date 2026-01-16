# src/computer_vision/detector.py
from ultralytics import YOLO

VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck (COCO)

class VehicleDetector:
    def __init__(self, model_path="yolov8x.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        result = self.model(frame, conf=self.conf, verbose=False)[0]

        detections = []
        for box in result.boxes:
            cls = int(box.cls)
            if cls in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "class": cls,
                    "conf": float(box.conf)
                })
        return detections

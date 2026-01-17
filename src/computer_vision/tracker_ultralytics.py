from ultralytics import YOLO

VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck


class UltralyticsTracker:
    def __init__(
        self,
        model_path="yolov8x.pt",
        conf=0.3,
        tracker_cfg="bytetrack.yaml"
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.tracker_cfg = tracker_cfg

    def track(self, frame):
        results = self.model.track(
            frame,
            conf=self.conf,
            tracker=self.tracker_cfg,
            persist=True,
            verbose=False
        )[0]

        tracks = []
        if results.boxes.id is None:
            return tracks

        for box, cls, tid, conf in zip(
            results.boxes.xyxy,
            results.boxes.cls,
            results.boxes.id,
            results.boxes.conf
        ):
            if int(cls) not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box)
            tracks.append({
                "track_id": int(tid),
                "bbox": (x1, y1, x2, y2),
                "conf": float(conf),
                "class": int(cls)
            })

        return tracks

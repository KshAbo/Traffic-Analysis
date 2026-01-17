class VehicleTypeCompositionAggregator:
    """
    Computes vehicle type composition ratios per minute
    """

    # COCO class IDs (YOLO)
    BUS_CLASS = 5
    TRUCK_CLASS = 7
    TWO_WHEELER_CLASS = 3  # motorcycle

    def __init__(self, queue_roi=None):
        """
        queue_roi: ROI object or None
        If provided, only vehicles inside queue ROI are considered
        """
        self.queue_roi = queue_roi

        # track_id -> class_id (count once per minute)
        self.seen_tracks = {}

    def update(self, tracks):
        """
        tracks: list of dicts
        Each track must have:
            - track_id
            - bbox
            - class
        """

        for tr in tracks:
            tid = tr["track_id"]
            cls = tr["class"]

            if self.queue_roi is not None:
                if not self.queue_roi.contains_bbox(tr["bbox"]):
                    continue

            # count each vehicle once per minute
            if tid not in self.seen_tracks:
                self.seen_tracks[tid] = cls

    def compute(self):
        """
        Returns per-minute vehicle type ratios
        """

        total = len(self.seen_tracks)
        if total == 0:
            return {
                "bus_ratio": 0.0,
                "truck_ratio": 0.0,
                "two_wheeler_ratio": 0.0
            }

        bus_count = sum(
            1 for cls in self.seen_tracks.values()
            if cls == self.BUS_CLASS
        )

        truck_count = sum(
            1 for cls in self.seen_tracks.values()
            if cls == self.TRUCK_CLASS
        )

        two_wheeler_count = sum(
            1 for cls in self.seen_tracks.values()
            if cls == self.TWO_WHEELER_CLASS
        )

        return {
            "bus_ratio": bus_count / total,
            "truck_ratio": truck_count / total,
            "two_wheeler_ratio": two_wheeler_count / total
        }

    def reset(self):
        """
        Clears per-minute state
        """
        self.seen_tracks.clear()
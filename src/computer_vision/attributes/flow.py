class FlowAggregator:
    def __init__(self, entry_roi, exit_roi):
        self.entry_roi = entry_roi
        self.exit_roi = exit_roi
        self.entry_ids = set()
        self.exit_ids = set()

    def update(self, tracks):
        for tr in tracks:
            tid = tr["track_id"]
            bbox = tr["bbox"]

            if self.entry_roi.contains_bbox(bbox):
                self.entry_ids.add(tid)

            if self.exit_roi.contains_bbox(bbox):
                self.exit_ids.add(tid)

    def compute(self):
        entry_count = len(self.entry_ids)
        exit_count = len(self.exit_ids)

        return {
            "entry_count": entry_count,
            "exit_count": exit_count,
            "flow_imbalance": entry_count - exit_count
        }

    def reset(self):
        self.entry_ids.clear()
        self.exit_ids.clear()

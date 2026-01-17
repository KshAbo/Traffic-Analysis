import numpy as np

class DwellTimeAggregator:
    """
    Computes dwell / waiting time proxies for vehicles inside a queue ROI
    """

    def __init__(self, queue_roi):
        """
        queue_roi: ROI object (from roi_selector.py)
        """
        self.queue_roi = queue_roi

        # track_id -> current continuous dwell length (in frames)
        self.active_dwell = {}

        # completed dwell durations within the minute
        self.completed_dwells = []

    def update(self, tracks):
        """
        tracks: list of dicts
        Each track must have:
            - track_id
            - bbox
        """

        current_ids_in_roi = set()

        for tr in tracks:
            tid = tr["track_id"]
            bbox = tr["bbox"]

            if self.queue_roi.contains_bbox(bbox):
                current_ids_in_roi.add(tid)

                # increment dwell
                if tid in self.active_dwell:
                    self.active_dwell[tid] += 1
                else:
                    self.active_dwell[tid] = 1

        # handle vehicles that left the ROI
        exited_ids = set(self.active_dwell.keys()) - current_ids_in_roi
        for tid in exited_ids:
            self.completed_dwells.append(self.active_dwell[tid])
            del self.active_dwell[tid]

    def compute(self):
        """
        Returns per-minute dwell time attributes (in frames)
        """

        # include ongoing dwell times
        all_dwells = self.completed_dwells + list(self.active_dwell.values())

        if not all_dwells:
            return {
                "avg_wait_time": 0.0,
                "max_wait_time": 0.0
            }

        return {
            "avg_wait_time": float(np.mean(all_dwells)),
            "max_wait_time": float(np.max(all_dwells))
        }

    def reset(self):
        """
        Clears per-minute buffers.
        Keeps active dwell states (important!)
        """
        self.completed_dwells.clear()

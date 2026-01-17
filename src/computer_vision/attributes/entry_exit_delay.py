import numpy as np

class EntryExitDelayAggregator:
    """
    Computes entry-to-exit delay for vehicles passing through the system
    """

    def __init__(self, entry_roi, exit_roi):
        """
        entry_roi: ROI object
        exit_roi: ROI object
        """
        self.entry_roi = entry_roi
        self.exit_roi = exit_roi

        # track_id -> entry frame index
        self.entry_times = {}

        # completed delays (in frames)
        self.completed_delays = []

        # global frame counter (monotonic)
        self.frame_idx = 0

    def update(self, tracks):
        """
        tracks: list of dicts
        Each track must have:
            - track_id
            - bbox
        """

        self.frame_idx += 1

        for tr in tracks:
            tid = tr["track_id"]
            bbox = tr["bbox"]

            # record entry time (only once)
            if tid not in self.entry_times and self.entry_roi.contains_bbox(bbox):
                self.entry_times[tid] = self.frame_idx

            # record exit time (only if entry seen)
            elif tid in self.entry_times and self.exit_roi.contains_bbox(bbox):
                delay = self.frame_idx - self.entry_times[tid]
                self.completed_delays.append(delay)

                # cleanup: ensure one entry-exit per vehicle
                del self.entry_times[tid]

    def compute(self):
        """
        Returns per-minute entry-exit delay attributes (in frames)
        """

        if not self.completed_delays:
            return {
                "entry_to_exit_time_mean": 0.0,
                "entry_to_exit_time_std": 0.0
            }

        return {
            "entry_to_exit_time_mean": float(np.mean(self.completed_delays)),
            "entry_to_exit_time_std": float(np.std(self.completed_delays))
        }

    def reset(self):
        """
        Clears per-minute completed delays.
        Keeps entry_times and frame_idx (important!)
        """
        self.completed_delays.clear()

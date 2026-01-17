# src/computer_vision/attributes/motion.py

import numpy as np
import math

# Tracking only inside the queue
"""
queue_tracks = [
    tr for tr in tracks
    if queue_roi.contains_bbox(tr["bbox"])
]

motion_agg.update(queue_tracks)
"""

class MotionAggregator:
    """
    Computes motion-based congestion proxies using tracked objects
    """

    def __init__(
        self,
        stop_threshold_px=2.0,
        slow_threshold_px=5.0
    ):
        """
        stop_threshold_px: pixel displacement below which a vehicle is considered stopped
        slow_threshold_px: pixel displacement below which a vehicle is considered slow
        """
        self.stop_threshold_px = stop_threshold_px
        self.slow_threshold_px = slow_threshold_px

        # track_id -> last centroid
        self.prev_centroids = {}

        # buffers for one minute
        self.frame_displacements = []
        self.frame_stop_flags = []
        self.frame_slow_flags = []

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, tracks):
        """
        tracks: list of dicts
        Each track must have:
            - track_id
            - bbox
        """

        current_centroids = {}

        for tr in tracks:
            tid = tr["track_id"]
            bbox = tr["bbox"]

            cx, cy = self._centroid(bbox)
            current_centroids[tid] = (cx, cy)

            if tid in self.prev_centroids:
                px, py = self.prev_centroids[tid]
                disp = math.hypot(cx - px, cy - py)

                self.frame_displacements.append(disp)
                self.frame_stop_flags.append(disp < self.stop_threshold_px)
                self.frame_slow_flags.append(disp < self.slow_threshold_px)

        # Update history for next frame
        self.prev_centroids = current_centroids

    def compute(self):
        """
        Returns per-minute motion attributes
        """
        if not self.frame_displacements:
            return {
                "mean_displacement": 0.0,
                "stop_ratio": 0.0,
                "slow_vehicle_fraction": 0.0
            }

        mean_disp = float(np.mean(self.frame_displacements))
        stop_ratio = float(np.mean(self.frame_stop_flags))
        slow_frac = float(np.mean(self.frame_slow_flags))

        return {
            "mean_displacement": mean_disp,
            "stop_ratio": stop_ratio,
            "slow_vehicle_fraction": slow_frac
        }

    def reset(self):
        """
        Clears per-minute buffers.
        Keeps prev_centroids (important!)
        """
        self.frame_displacements.clear()
        self.frame_stop_flags.clear()
        self.frame_slow_flags.clear()

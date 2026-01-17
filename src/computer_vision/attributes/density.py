import numpy as np


class DensityAggregator:
    """
    Computes density-related attributes inside a queue ROI
    """

    def __init__(self, queue_roi):
        """
        queue_roi: ROI object (from roi_selector.py)
        """
        self.queue_roi = queue_roi
        self.frame_densities = []

        # Precompute area once
        self.roi_area = queue_roi.w * queue_roi.h
        if self.roi_area <= 0:
            raise ValueError("Queue ROI has invalid area")

    def update(self, tracks):
        """
        tracks: list of tracked vehicles for current frame
        Each track must have a 'bbox'
        """
        count_in_roi = 0

        for tr in tracks:
            if self.queue_roi.contains_bbox(tr["bbox"]):
                count_in_roi += 1

        density = count_in_roi / self.roi_area
        self.frame_densities.append(density)

    def compute(self):
        """
        Returns per-minute density attributes
        """
        if not self.frame_densities:
            return {
                "avg_density": 0.0,
                "max_density": 0.0
            }

        return {
            "avg_density": float(np.mean(self.frame_densities)),
            "max_density": float(np.max(self.frame_densities))
        }

    def reset(self):
        """
        Clears per-minute buffer
        """
        self.frame_densities.clear()

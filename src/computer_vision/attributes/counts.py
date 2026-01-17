import numpy as np

class VehicleCountAggregator:
    def __init__(self):
        self.frame_counts = []

    def update(self, tracks):
        """
        tracks: list of tracked vehicles for current frame
        """
        self.frame_counts.append(len(tracks))

    def compute(self):
        if not self.frame_counts:
            return {
                "mean_vehicle_count": 0,
                "max_vehicle_count": 0,
                "vehicle_count_std": 0.0
            }

        return {
            "mean_vehicle_count": float(np.mean(self.frame_counts)),
            "max_vehicle_count": int(np.max(self.frame_counts)),
            "vehicle_count_std": float(np.std(self.frame_counts))
        }

    def reset(self):
        self.frame_counts.clear()

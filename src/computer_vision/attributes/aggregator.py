"""
aggregator = MinuteAggregator(
    count_agg,
    flow_agg,
    density_agg,
    motion_agg,
    dwell_agg,
    entry_exit_delay_agg,
    vehicle_composition_agg
)

for frame in frames:
    tracks = tracker.track(frame)
    aggregator.update(tracks)

minute_features = aggregator.finalize_minute()

minute_features["camera_id"] = cam_id
minute_features["minute"] = minute_idx

"""


class MinuteAggregator:
    """
    Orchestrates all per-minute traffic attributes
    """

    def __init__(
        self,
        count_agg,
        flow_agg,
        density_agg,
        motion_agg,
        dwell_agg,
        entry_exit_delay_agg,
        vehicle_composition_agg
    ):
        self.count_agg = count_agg
        self.flow_agg = flow_agg
        self.density_agg = density_agg
        self.motion_agg = motion_agg
        self.dwell_agg = dwell_agg
        self.entry_exit_delay_agg = entry_exit_delay_agg
        self.vehicle_composition_agg = vehicle_composition_agg

    def update(self, tracks):
        """
        Called once per frame
        tracks: list of tracked vehicles
        """

        # Global attributes
        self.count_agg.update(tracks)
        self.flow_agg.update(tracks)

        # Queue-restricted attributes
        self.density_agg.update(tracks)
        self.motion_agg.update(tracks)
        self.dwell_agg.update(tracks)
        self.vehicle_composition_agg.update(tracks)

        # Entry → Exit delay (global, stateful)
        self.entry_exit_delay_agg.update(tracks)

    def finalize_minute(self):
        """
        Called once per minute.
        Returns a flat dict of all attributes.
        """

        features = {}

        # Counts
        features.update(self.count_agg.compute())
        self.count_agg.reset()

        # Flow
        features.update(self.flow_agg.compute())
        self.flow_agg.reset()

        # Density
        features.update(self.density_agg.compute())
        self.density_agg.reset()

        # Motion
        features.update(self.motion_agg.compute())
        self.motion_agg.reset()

        # Dwell / wait
        features.update(self.dwell_agg.compute())
        self.dwell_agg.reset()

        # Entry-exit delay
        features.update(self.entry_exit_delay_agg.compute())
        self.entry_exit_delay_agg.reset()

        # Vehicle composition
        features.update(self.vehicle_composition_agg.compute())
        self.vehicle_composition_agg.reset()

        return features
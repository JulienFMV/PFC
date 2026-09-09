"""Hydro alignment correction without changing the frozen scientific incumbent."""

import numpy as np
import pandas as pd

from pfc_shaping.lt.model.shape_hourly_mlp import ShapeHourlyMLP


class HydroAlignedShapeHourlyMLP(ShapeHourlyMLP):
    """Use Swiss civil weeks and actual hydro timestamps in the native MLP.

    The network, fitting algorithm and assembly interface are inherited. The
    original incumbent remains byte-stable for already frozen comparisons;
    this corrected component has its own explicit identity in local manifests.
    """

    def _setup_hydro(self, hydro_df: pd.DataFrame) -> None:
        local = hydro_df.copy()
        if local.index.tz is not None:
            local.index = local.index.tz_convert("Europe/Zurich")
        super()._setup_hydro(local)

    def _map_hydro_fill(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        if self._hydro_fill_weekly is None:
            return np.full(len(timestamps), 0.5)
        # A Swiss Monday starts at 22:00/23:00 UTC. Daily-midnight alignment
        # would lose these observations and silently retain the neutral value.
        aligned = self._hydro_fill_weekly.reindex(timestamps, method="ffill")
        return aligned.fillna(0.5).to_numpy(dtype=float)

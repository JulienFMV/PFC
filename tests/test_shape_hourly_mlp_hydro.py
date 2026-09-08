import numpy as np
import pandas as pd

from pfc_shaping.lt.model.shape_hourly_mlp_hydro import HydroAlignedShapeHourlyMLP


def _hydro():
    index = pd.date_range("2026-02-23", periods=12, freq="W-MON", tz="Europe/Zurich")
    return pd.DataFrame({"fill_pct": np.arange(20.0, 80.0, 5.0)}, index=index.tz_convert("UTC"))


def test_mlp_maps_real_weekly_hydro_at_observation_time_across_dst():
    hydro = _hydro()
    model = HydroAlignedShapeHourlyMLP()
    model._setup_hydro(hydro)
    timestamps = pd.DatetimeIndex([
        hydro.index[0] - pd.Timedelta(minutes=15),
        hydro.index[0],
        hydro.index[0] + pd.Timedelta(hours=12),
        hydro.index[5] - pd.Timedelta(minutes=15),
        hydro.index[5],
        hydro.index[5] + pd.Timedelta(hours=12),
    ])
    np.testing.assert_allclose(model._map_hydro_fill(timestamps), [0.5, 0.2, 0.2, 0.4, 0.45, 0.45])


def test_mlp_hydro_climatology_uses_swiss_civil_weeks():
    hydro = _hydro()
    model = HydroAlignedShapeHourlyMLP()
    model._setup_hydro(hydro)
    for timestamp, fill in hydro["fill_pct"].items():
        week = timestamp.tz_convert("Europe/Zurich").isocalendar().week
        assert model._climatological_fill.loc[week] == fill / 100.0

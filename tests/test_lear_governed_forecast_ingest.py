from __future__ import annotations

import pandas as pd

from pfc_shaping.data.ingest_entso import (
    _merge_preferred_forecast_frames,
    _merge_preferred_forecast_series,
)


def test_merge_preferred_forecast_series_keeps_dayahead_and_fills_from_weekahead() -> None:
    idx = pd.date_range("2026-05-22 00:00:00+00:00", periods=4, freq="h")
    preferred = pd.Series([1.0, 2.0, None, None], index=idx, name="forecast_load_ch_mw")
    fallback = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx, name="forecast_load_ch_mw")

    merged = _merge_preferred_forecast_series(preferred, fallback, "forecast_load_ch_mw")

    assert list(merged.astype(float)) == [1.0, 2.0, 30.0, 40.0]


def test_merge_preferred_forecast_frames_keeps_preferred_cells_and_fills_gaps() -> None:
    idx = pd.date_range("2026-05-22 00:00:00+00:00", periods=3, freq="h")
    preferred = pd.DataFrame(
        {
            "forecast_solar_ch_mw": [1.0, None, None],
            "forecast_wind_ch_mw": [5.0, 6.0, None],
        },
        index=idx,
    )
    fallback = pd.DataFrame(
        {
            "forecast_solar_ch_mw": [10.0, 20.0, 30.0],
            "forecast_wind_ch_mw": [50.0, 60.0, 70.0],
        },
        index=idx,
    )

    merged = _merge_preferred_forecast_frames(preferred, fallback)

    assert list(merged["forecast_solar_ch_mw"].astype(float)) == [1.0, 20.0, 30.0]
    assert list(merged["forecast_wind_ch_mw"].astype(float)) == [5.0, 6.0, 70.0]

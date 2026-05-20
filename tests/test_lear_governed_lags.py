from __future__ import annotations

import numpy as np
import pandas as pd

from pfc_shaping.model.lear_forecaster import LEARForecaster


def test_governed_feature_block_adds_d1_d7_and_fr_nuclear_columns() -> None:
    idx_15 = pd.date_range("2025-10-01 00:00:00+00:00", periods=24 * 4 * 240, freq="15min")
    epex = pd.DataFrame({"price_eur_mwh": np.linspace(40.0, 120.0, len(idx_15))}, index=idx_15)

    multi_country = pd.DataFrame(
        {
            "forecast_load_ch_mw": np.linspace(5000.0, 7000.0, len(idx_15)),
            "forecast_solar_ch_mw": np.linspace(0.0, 1500.0, len(idx_15)),
        },
        index=idx_15,
    )
    multi_country.index.name = "forecast_for_ts_utc"

    fr_nuclear = pd.DataFrame(
        {
            "forecast_fr_nuclear_available_mw": np.linspace(52000.0, 50000.0, len(idx_15)),
            "forecast_fr_nuclear_unavailability_ratio": np.linspace(0.05, 0.08, len(idx_15)),
        },
        index=idx_15,
    )
    fr_nuclear.index.name = "forecast_for_ts_utc"

    weather_idx = pd.date_range("2025-10-01 00:00:00+00:00", periods=24 * 240, freq="h")
    weather = pd.DataFrame(
        {
            "wx_ch_zurich_shortwave_radiation": np.linspace(0.0, 600.0, len(weather_idx)),
        },
        index=weather_idx,
    )
    weather.index.name = "forecast_for_ts_utc"

    lear = LEARForecaster(use_governed_forecast_features=True, use_governed_d7_features=True)
    lear.fit(
        epex_15min=epex,
        multi_country_forecast=multi_country,
        fr_nuclear_forecast=fr_nuclear,
        weather_forecast=weather,
    )

    X, y = lear._build_features(lear.prices_h_, lear.exog_, target_hour=12)

    assert len(X) == len(y)
    expected_cols = {
        "forecast_load_ch_mw_d-1_h12",
        "forecast_load_ch_mw_d-7_h12",
        "forecast_solar_ch_mw_d-1_h12",
        "forecast_solar_ch_mw_d-7_h12",
        "wx_ch_zurich_shortwave_radiation_d-1_h12",
        "wx_ch_zurich_shortwave_radiation_d-7_h12",
        "forecast_fr_nuclear_available_mw_d-1_h12",
        "forecast_fr_nuclear_available_mw_d-7_h12",
    }
    assert expected_cols.issubset(set(X.columns))


def test_sparse_governed_feature_is_dropped_when_history_is_too_short() -> None:
    idx_15 = pd.date_range("2025-10-01 00:00:00+00:00", periods=24 * 4 * 240, freq="15min")
    epex = pd.DataFrame({"price_eur_mwh": np.linspace(40.0, 120.0, len(idx_15))}, index=idx_15)

    multi_country = pd.DataFrame(
        {"forecast_load_ch_mw": np.linspace(5000.0, 7000.0, len(idx_15))},
        index=idx_15,
    )
    multi_country.index.name = "forecast_for_ts_utc"

    sparse_idx = pd.date_range("2026-05-17 00:00:00+00:00", periods=24 * 4 * 12, freq="15min")
    fr_nuclear = pd.DataFrame(
        {
            "forecast_fr_nuclear_available_mw": np.linspace(52000.0, 50000.0, len(sparse_idx)),
            "forecast_fr_nuclear_unavailability_ratio": np.linspace(0.05, 0.08, len(sparse_idx)),
        },
        index=sparse_idx,
    )
    fr_nuclear.index.name = "forecast_for_ts_utc"

    lear = LEARForecaster(use_governed_forecast_features=True, use_governed_d7_features=True)
    lear.fit(
        epex_15min=epex,
        multi_country_forecast=multi_country,
        fr_nuclear_forecast=fr_nuclear,
    )

    X, _ = lear._build_features(lear.prices_h_, lear.exog_, target_hour=12)

    assert "forecast_load_ch_mw_d-7_h12" in X.columns
    assert "forecast_fr_nuclear_available_mw_d-7_h12" not in X.columns


def test_governed_d7_features_disabled_by_default() -> None:
    idx_15 = pd.date_range("2025-10-01 00:00:00+00:00", periods=24 * 4 * 120, freq="15min")
    epex = pd.DataFrame({"price_eur_mwh": np.linspace(40.0, 120.0, len(idx_15))}, index=idx_15)
    multi_country = pd.DataFrame(
        {"forecast_load_ch_mw": np.linspace(5000.0, 7000.0, len(idx_15))},
        index=idx_15,
    )
    multi_country.index.name = "forecast_for_ts_utc"

    lear = LEARForecaster(use_governed_forecast_features=True)
    lear.fit(epex_15min=epex, multi_country_forecast=multi_country)
    X, _ = lear._build_features(lear.prices_h_, lear.exog_, target_hour=12)

    assert "forecast_load_ch_mw_d-1_h12" in X.columns
    assert "forecast_load_ch_mw_d-7_h12" not in X.columns

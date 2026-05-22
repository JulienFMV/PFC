from __future__ import annotations

import pandas as pd

from pfc_shaping.model.lear_forecaster import LEARForecaster


def test_series_to_forecast_pivot_drops_rows_violating_vintage_cutoff() -> None:
    lear = LEARForecaster(use_governed_forecast_features=True)

    delivery_idx = pd.DatetimeIndex(
        [
            "2026-05-20 12:00:00+00:00",
            "2026-05-20 13:00:00+00:00",
        ],
        tz="UTC",
    )
    series = pd.Series([100.0, 200.0], index=delivery_idx)
    publication_ts = pd.Series(
        pd.to_datetime(
            [
                "2026-05-19 20:00:00+00:00",  # valid: 16h before delivery
                "2026-05-20 03:00:00+00:00",  # invalid: 10h before delivery
            ],
            utc=True,
        ),
        index=delivery_idx,
    )

    pivot = lear._series_to_forecast_pivot(
        series,
        publication_ts=publication_ts,
        source_name="forecast_load_ch_mw",
    )

    assert list(pivot.columns) == [14]  # 12:00 UTC -> 14:00 Europe/Zurich in May
    assert len(pivot.index) == 1
    assert float(pivot.iloc[0, 0]) == 100.0


def test_assess_governed_health_surfaces_vintage_schema_false() -> None:
    lear = LEARForecaster(use_governed_forecast_features=True)
    lear._fitted = True
    idx = pd.date_range("2026-05-01", periods=48, freq="h", tz="Europe/Zurich")
    lear._idx_local = idx
    full_pivot = pd.DataFrame(
        {h: [1.0] for h in range(24)},
        index=[(idx[-1] + pd.Timedelta(days=1)).date()],
    )
    lear._forecast_feature_pivots = {
        col: full_pivot.copy()
        for col in (*lear.GOVERNED_FORECAST_COLUMNS, *lear.GOVERNED_WEATHER_COLUMNS)
    }
    lear._governed_vintage_status = {
        "multi_country_forecast": {"vintage_schema_verified": False, "invalid_rows_dropped": 0, "total_rows_seen": 10},
        "weather_forecast": {"vintage_schema_verified": False, "invalid_rows_dropped": 0, "total_rows_seen": 10},
    }

    health = lear.assess_governed_forecast_feature_health(horizon_days=1, min_coverage_ratio=0.98)

    assert health["enabled_for_run"] is True
    assert health["vintage_schema_verified"] is False


def test_assess_governed_health_drops_only_low_coverage_sources() -> None:
    lear = LEARForecaster(use_governed_forecast_features=True)
    lear._fitted = True
    idx = pd.date_range("2026-05-01", periods=48, freq="h", tz="Europe/Zurich")
    lear._idx_local = idx
    forecast_dates = [
        (idx[-1] + pd.Timedelta(days=1)).date(),
        (idx[-1] + pd.Timedelta(days=2)).date(),
    ]
    full_pivot = pd.DataFrame(
        {h: [1.0, 1.0] for h in range(24)},
        index=forecast_dates,
    )
    weak_pivot = full_pivot.copy()
    weak_pivot.loc[forecast_dates[-1], 23] = None  # 47/48 ~= 0.979 < 0.98

    all_sources = (*lear.GOVERNED_FORECAST_COLUMNS, *lear.GOVERNED_WEATHER_COLUMNS)
    pivots = {source: full_pivot.copy() for source in all_sources}
    weak_source = lear.GOVERNED_WEATHER_COLUMNS[-1]
    pivots[weak_source] = weak_pivot
    lear._forecast_feature_pivots = dict(pivots)
    lear._all_forecast_feature_pivots = dict(pivots)
    lear._governed_vintage_status = {
        "multi_country_forecast": {"vintage_schema_verified": False, "invalid_rows_dropped": 0, "total_rows_seen": 10},
        "weather_forecast": {"vintage_schema_verified": False, "invalid_rows_dropped": 0, "total_rows_seen": 10},
    }

    health = lear.assess_governed_forecast_feature_health(horizon_days=2, min_coverage_ratio=0.98)

    assert health["enabled_for_run"] is True
    assert health["max_supported_horizon_days"] == 2
    assert weak_source in health["missing_sources"]
    assert health["source_coverage"][weak_source] < 0.98
    assert weak_source not in lear._forecast_feature_pivots
    assert lear.GOVERNED_FORECAST_COLUMNS[0] in lear._forecast_feature_pivots


def test_assess_governed_health_disables_run_when_no_sources_meet_threshold() -> None:
    lear = LEARForecaster(use_governed_forecast_features=True)
    lear._fitted = True
    idx = pd.date_range("2026-05-01", periods=48, freq="h", tz="Europe/Zurich")
    lear._idx_local = idx
    forecast_dates = [
        (idx[-1] + pd.Timedelta(days=1)).date(),
        (idx[-1] + pd.Timedelta(days=2)).date(),
    ]
    weak_pivot = pd.DataFrame(
        {h: [1.0, 1.0] for h in range(24)},
        index=forecast_dates,
    )
    weak_pivot.loc[:, :] = None
    weak_pivot.loc[forecast_dates[0], 0] = 1.0

    all_sources = (*lear.GOVERNED_FORECAST_COLUMNS, *lear.GOVERNED_WEATHER_COLUMNS)
    pivots = {source: weak_pivot.copy() for source in all_sources}
    lear._forecast_feature_pivots = dict(pivots)
    lear._all_forecast_feature_pivots = dict(pivots)
    lear._governed_vintage_status = {
        "multi_country_forecast": {"vintage_schema_verified": False, "invalid_rows_dropped": 0, "total_rows_seen": 10},
        "weather_forecast": {"vintage_schema_verified": False, "invalid_rows_dropped": 0, "total_rows_seen": 10},
    }

    health = lear.assess_governed_forecast_feature_health(horizon_days=2, min_coverage_ratio=0.98)

    assert health["enabled_for_run"] is False
    assert health["max_supported_horizon_days"] == 0


def test_assess_governed_health_requires_core_forecast_and_weather_families() -> None:
    lear = LEARForecaster(use_governed_forecast_features=True)
    lear._fitted = True
    idx = pd.date_range("2026-05-01", periods=48, freq="h", tz="Europe/Zurich")
    lear._idx_local = idx
    forecast_dates = [
        (idx[-1] + pd.Timedelta(days=1)).date(),
        (idx[-1] + pd.Timedelta(days=2)).date(),
    ]
    full_pivot = pd.DataFrame(
        {h: [1.0, 1.0] for h in range(24)},
        index=forecast_dates,
    )
    pivots = {
        "forecast_fr_nuclear_available_mw": full_pivot.copy(),
        "forecast_fr_nuclear_unavailability_ratio": full_pivot.copy(),
    }
    lear._forecast_feature_pivots = dict(pivots)
    lear._all_forecast_feature_pivots = dict(pivots)
    lear._governed_vintage_status = {
        "multi_country_forecast": {"vintage_schema_verified": False, "invalid_rows_dropped": 0, "total_rows_seen": 10},
        "weather_forecast": {"vintage_schema_verified": False, "invalid_rows_dropped": 0, "total_rows_seen": 10},
        "fr_nuclear_forecast": {"vintage_schema_verified": False, "invalid_rows_dropped": 0, "total_rows_seen": 10},
    }

    health = lear.assess_governed_forecast_feature_health(horizon_days=2, min_coverage_ratio=0.98)

    assert health["enabled_for_run"] is False
    assert health["max_supported_horizon_days"] == 0


def test_select_effective_l1_ratio_bumps_when_vif_is_dense() -> None:
    lear = LEARForecaster(use_governed_forecast_features=True)
    low = lear._select_effective_l1_ratio({"features_over_5": ["a", "b", "c"]})
    high = lear._select_effective_l1_ratio({"features_over_5": ["a", "b", "c", "d"]})

    assert low == 0.1
    assert high == 0.5

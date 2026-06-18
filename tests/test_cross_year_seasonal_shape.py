from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.model.cross_year_seasonal_shape import apply_cross_year_seasonal_shape_optimizer
from pfc_shaping.lt.model.shape_constraints import eex_peak_mask


def _hourly_frame(timestamps: pd.DatetimeIndex, monthly_levels: dict[tuple[int, int], float]) -> pd.DataFrame:
    values = np.array([monthly_levels[(int(ts.year), int(ts.month))] for ts in timestamps], dtype=float)
    return pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": values - 1.0,
            "price_central_eur_mwh": values,
            "price_fast_eur_mwh": values + 1.0,
            "price_weighted_mean_eur_mwh": values,
            "structural_p10_eur_mwh": values - 1.0,
            "structural_p50_eur_mwh": values,
            "structural_p90_eur_mwh": values + 1.0,
            "structural_width_eur_mwh": [2.0] * len(timestamps),
        }
    )


def _monthly_means(frame: pd.DataFrame, timestamps: pd.DatetimeIndex) -> pd.Series:
    periods = pd.PeriodIndex(timestamps.strftime("%Y-%m"), freq="M")
    return frame.groupby(periods)["price_weighted_mean_eur_mwh"].mean()


def _base_targets(frame: pd.DataFrame, timestamps: pd.DatetimeIndex) -> dict[str, float]:
    month = timestamps.month
    year = timestamps.year
    q1_2028 = (year == 2028) & (month <= 3)
    residual_2028 = (year == 2028) & (month >= 4)
    year_2028 = year == 2028
    year_2029 = year == 2029
    price = frame["price_weighted_mean_eur_mwh"]
    return {
        "2028": float(price.loc[year_2028].mean()),
        "2028-Q1": float(price.loc[q1_2028].mean()),
        "2029": float(price.loc[year_2029].mean()),
        "_2028_residual": float(price.loc[residual_2028].mean()),
    }


def test_cross_year_seasonal_shape_reconciles_residual_vs_next_annual_and_preserves_eex() -> None:
    timestamps = pd.date_range("2028-01-01", "2029-12-31 23:00", freq="h", tz="Europe/Zurich")
    levels: dict[tuple[int, int], float] = {
        (2028, 1): 117.0,
        (2028, 2): 115.0,
        (2028, 3): 98.0,
        (2028, 4): 62.0,
        (2028, 5): 61.0,
        (2028, 6): 62.0,
        (2028, 7): 66.0,
        (2028, 8): 65.0,
        (2028, 9): 72.0,
        (2028, 10): 78.0,
        (2028, 11): 85.0,
        (2028, 12): 85.0,
        (2029, 1): 82.0,
        (2029, 2): 75.0,
        (2029, 3): 68.0,
        (2029, 4): 62.05,
        (2029, 5): 58.0,
        (2029, 6): 57.0,
        (2029, 7): 59.0,
        (2029, 8): 64.0,
        (2029, 9): 72.0,
        (2029, 10): 81.0,
        (2029, 11): 90.0,
        (2029, 12): 99.0,
    }
    hourly = _hourly_frame(timestamps, levels)
    targets = _base_targets(hourly, timestamps)
    base_prices = {"2028": targets["2028"], "2028-Q1": targets["2028-Q1"], "2029": targets["2029"]}
    peak = pd.Series(eex_peak_mask(timestamps.tz_convert("UTC"), country="CH"), index=hourly.index)
    peak_prices = {
        "2028": float(hourly.loc[timestamps.year == 2028].loc[peak[timestamps.year == 2028].to_numpy(), "price_weighted_mean_eur_mwh"].mean()),
        "2028-Q1": float(hourly.loc[(timestamps.year == 2028) & (timestamps.month <= 3)].loc[peak[(timestamps.year == 2028) & (timestamps.month <= 3)].to_numpy(), "price_weighted_mean_eur_mwh"].mean()),
        "2029": float(hourly.loc[timestamps.year == 2029].loc[peak[timestamps.year == 2029].to_numpy(), "price_weighted_mean_eur_mwh"].mean()),
    }
    parent_spread = targets["2029"] - targets["_2028_residual"]
    guide = {f"2028-{month:02d}": levels[(2028, month)] for month in range(4, 13)}
    guide.update({f"2029-{month:02d}": levels[(2028, month)] + parent_spread for month in range(4, 13)})
    guide.update({f"2029-{month:02d}": levels[(2029, month)] for month in range(1, 4)})
    before = _monthly_means(hourly, timestamps)
    before_slope_delta = (before["2029-12"] - before["2029-04"]) - (before["2028-12"] - before["2028-04"])
    before_dec_error = (before["2029-12"] - before["2028-12"]) - parent_spread

    smoothed, audit = apply_cross_year_seasonal_shape_optimizer(
        hourly,
        ts_ch=pd.Series(timestamps, index=hourly.index),
        base_forward_prices=base_prices,
        peak_forward_prices=peak_prices,
        peak_mask=peak,
        guide_month_prices=guide,
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        lambda_cross_month=30.0,
        lambda_cross_slope=60.0,
        max_month_delta_eur_mwh=20.0,
        negative_price_floor=-30.0,
        max_weighted_negative_hours=0,
    )

    after = _monthly_means(smoothed, timestamps)
    after_targets = _base_targets(smoothed, timestamps)
    after_slope_delta = (after["2029-12"] - after["2029-04"]) - (after["2028-12"] - after["2028-04"])
    after_dec_error = (after["2029-12"] - after["2028-12"]) - parent_spread

    assert not audit.empty
    assert after_targets["2028"] == pytest.approx(targets["2028"], abs=1e-6)
    assert after_targets["2028-Q1"] == pytest.approx(targets["2028-Q1"], abs=1e-6)
    assert after_targets["2029"] == pytest.approx(targets["2029"], abs=1e-6)
    assert after_targets["_2028_residual"] == pytest.approx(targets["_2028_residual"], abs=1e-6)
    assert abs(after_dec_error) < abs(before_dec_error) * 0.35
    assert abs(after_slope_delta) < abs(before_slope_delta) * 0.35
    assert float(audit["max_constraint_residual_eur_mwh"].max()) < 1e-6


def test_cross_year_seasonal_shape_uses_guide_shape_not_absolute_level() -> None:
    timestamps = pd.date_range("2028-01-01", "2029-12-31 23:00", freq="h", tz="Europe/Zurich")
    levels: dict[tuple[int, int], float] = {
        (2028, 1): 117.0,
        (2028, 2): 115.0,
        (2028, 3): 98.0,
        (2028, 4): 62.0,
        (2028, 5): 61.0,
        (2028, 6): 62.0,
        (2028, 7): 66.0,
        (2028, 8): 65.0,
        (2028, 9): 72.0,
        (2028, 10): 78.0,
        (2028, 11): 85.0,
        (2028, 12): 85.0,
        (2029, 1): 82.0,
        (2029, 2): 75.0,
        (2029, 3): 68.0,
        (2029, 4): 62.05,
        (2029, 5): 58.0,
        (2029, 6): 57.0,
        (2029, 7): 59.0,
        (2029, 8): 64.0,
        (2029, 9): 72.0,
        (2029, 10): 81.0,
        (2029, 11): 90.0,
        (2029, 12): 99.0,
    }
    hourly = _hourly_frame(timestamps, levels)
    targets = _base_targets(hourly, timestamps)
    base_prices = {"2028": targets["2028"], "2028-Q1": targets["2028-Q1"], "2029": targets["2029"]}
    peak = pd.Series(eex_peak_mask(timestamps.tz_convert("UTC"), country="CH"), index=hourly.index)
    peak_prices = {
        "2028": float(
            hourly.loc[timestamps.year == 2028]
            .loc[peak[timestamps.year == 2028].to_numpy(), "price_weighted_mean_eur_mwh"]
            .mean()
        ),
        "2028-Q1": float(
            hourly.loc[(timestamps.year == 2028) & (timestamps.month <= 3)]
            .loc[peak[(timestamps.year == 2028) & (timestamps.month <= 3)].to_numpy(), "price_weighted_mean_eur_mwh"]
            .mean()
        ),
        "2029": float(
            hourly.loc[timestamps.year == 2029]
            .loc[peak[timestamps.year == 2029].to_numpy(), "price_weighted_mean_eur_mwh"]
            .mean()
        ),
    }
    guide = {f"{year}-{month:02d}": float(levels[(year, month)]) for year in [2028, 2029] for month in range(1, 13)}
    shifted_guide = {key: value + 1000.0 for key, value in guide.items()}

    kwargs = {
        "ts_ch": pd.Series(timestamps, index=hourly.index),
        "base_forward_prices": base_prices,
        "peak_forward_prices": peak_prices,
        "peak_mask": peak,
        "weights": {"slow": 0.25, "central": 0.5, "fast": 0.25},
        "lambda_cross_month": 30.0,
        "lambda_cross_slope": 60.0,
        "max_month_delta_eur_mwh": 20.0,
        "negative_price_floor": -30.0,
        "max_weighted_negative_hours": 0,
    }
    shaped, _ = apply_cross_year_seasonal_shape_optimizer(hourly, guide_month_prices=guide, **kwargs)
    shifted, _ = apply_cross_year_seasonal_shape_optimizer(hourly, guide_month_prices=shifted_guide, **kwargs)

    pd.testing.assert_series_equal(
        _monthly_means(shaped, timestamps),
        _monthly_means(shifted, timestamps),
        check_names=False,
        atol=1e-9,
        rtol=0.0,
    )

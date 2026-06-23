import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.model.seam_nullspace_smoothing import apply_seam_nullspace_smoothing
from pfc_shaping.lt.model.shape_constraints import eex_peak_mask


def _hourly_frame(timestamps: pd.DatetimeIndex, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": values,
            "price_central_eur_mwh": values,
            "price_fast_eur_mwh": values,
            "price_weighted_mean_eur_mwh": values,
            "structural_p10_eur_mwh": values,
            "structural_p50_eur_mwh": values,
            "structural_p90_eur_mwh": values,
            "structural_width_eur_mwh": np.zeros(len(values)),
        }
    )


def test_seam_nullspace_smoothing_reduces_intrabucket_month_cliff_and_preserves_base_mean():
    timestamps = pd.date_range("2030-01-01", "2030-03-31 23:00", freq="h", tz="Europe/Zurich")
    values = np.where(timestamps.month == 1, 80.0, 120.0).astype(float)
    hourly = _hourly_frame(timestamps, values)
    target = float(hourly["price_weighted_mean_eur_mwh"].mean())

    smoothed, audit = apply_seam_nullspace_smoothing(
        hourly,
        ts_ch=pd.Series(timestamps, index=hourly.index),
        base_forward_prices={"2030-Q1": target},
        peak_forward_prices={},
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        seam_window_hours=72,
        target_boundary_jump_eur_mwh=8.0,
        min_boundary_jump_eur_mwh=20.0,
        max_abs_delta_eur_mwh=20.0,
        negative_price_floor=-30.0,
        max_weighted_negative_hours=0,
    )

    before_jump = float(
        hourly.loc[timestamps == pd.Timestamp("2030-02-01 00:00", tz="Europe/Zurich"), "price_weighted_mean_eur_mwh"].iloc[0]
        - hourly.loc[timestamps == pd.Timestamp("2030-01-31 23:00", tz="Europe/Zurich"), "price_weighted_mean_eur_mwh"].iloc[0]
    )
    after_jump = float(
        smoothed.loc[timestamps == pd.Timestamp("2030-02-01 00:00", tz="Europe/Zurich"), "price_weighted_mean_eur_mwh"].iloc[0]
        - smoothed.loc[timestamps == pd.Timestamp("2030-01-31 23:00", tz="Europe/Zurich"), "price_weighted_mean_eur_mwh"].iloc[0]
    )

    assert not audit.empty
    assert abs(after_jump) < abs(before_jump)
    assert float(smoothed["price_weighted_mean_eur_mwh"].mean()) == pytest.approx(target, abs=1e-6)
    assert float(audit["max_constraint_residual_eur_mwh"].max()) == pytest.approx(0.0, abs=1e-8)


def test_seam_nullspace_smoothing_preserves_peak_mean_when_peak_quote_exists():
    timestamps = pd.date_range("2030-01-01", "2030-03-31 23:00", freq="h", tz="Europe/Zurich")
    values = np.where(timestamps.month == 1, 85.0, 125.0).astype(float)
    hourly = _hourly_frame(timestamps, values)
    base_target = float(hourly["price_weighted_mean_eur_mwh"].mean())
    peak_mask = eex_peak_mask(timestamps.tz_convert("UTC"), country="CH")
    peak_target = float(hourly.loc[peak_mask, "price_weighted_mean_eur_mwh"].mean())

    smoothed, _ = apply_seam_nullspace_smoothing(
        hourly,
        ts_ch=pd.Series(timestamps, index=hourly.index),
        base_forward_prices={"2030-Q1": base_target},
        peak_forward_prices={"2030-Q1": peak_target},
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        seam_window_hours=72,
        target_boundary_jump_eur_mwh=8.0,
        min_boundary_jump_eur_mwh=20.0,
        max_abs_delta_eur_mwh=20.0,
        negative_price_floor=-30.0,
        max_weighted_negative_hours=0,
    )

    assert float(smoothed["price_weighted_mean_eur_mwh"].mean()) == pytest.approx(base_target, abs=1e-6)
    assert float(smoothed.loc[peak_mask, "price_weighted_mean_eur_mwh"].mean()) == pytest.approx(
        peak_target,
        abs=1e-6,
    )


def test_seam_nullspace_smoothing_handles_non_default_index_with_peak_constraints():
    timestamps = pd.date_range("2030-01-01", "2030-03-31 23:00", freq="h", tz="Europe/Zurich")
    values = np.where(timestamps.month == 1, 85.0, 125.0).astype(float)
    hourly = _hourly_frame(timestamps, values)
    hourly.index = pd.RangeIndex(1000, 1000 + len(hourly))
    ts_ch = pd.Series(timestamps, index=hourly.index)
    base_target = float(hourly["price_weighted_mean_eur_mwh"].mean())
    peak_mask = eex_peak_mask(timestamps.tz_convert("UTC"), country="CH")
    peak_target = float(hourly.loc[peak_mask, "price_weighted_mean_eur_mwh"].mean())

    smoothed, audit = apply_seam_nullspace_smoothing(
        hourly,
        ts_ch=ts_ch,
        base_forward_prices={"2030-Q1": base_target},
        peak_forward_prices={"2030-Q1": peak_target},
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        seam_window_hours=72,
        target_boundary_jump_eur_mwh=8.0,
        min_boundary_jump_eur_mwh=20.0,
        max_abs_delta_eur_mwh=20.0,
        negative_price_floor=-30.0,
        max_weighted_negative_hours=0,
    )

    assert not audit.empty
    assert float(smoothed["price_weighted_mean_eur_mwh"].mean()) == pytest.approx(base_target, abs=1e-6)
    assert float(smoothed.loc[peak_mask, "price_weighted_mean_eur_mwh"].mean()) == pytest.approx(
        peak_target,
        abs=1e-6,
    )


def test_seam_nullspace_smoothing_handles_dst_months_with_hour_weights():
    timestamps = pd.date_range("2030-03-01", "2030-04-30 23:00", freq="h", tz="Europe/Zurich")
    values = np.where(timestamps.month == 3, 110.0, 70.0).astype(float)
    hourly = _hourly_frame(timestamps, values)
    target = float(hourly["price_weighted_mean_eur_mwh"].mean())

    smoothed, audit = apply_seam_nullspace_smoothing(
        hourly,
        ts_ch=pd.Series(timestamps, index=hourly.index),
        base_forward_prices={"2030": target},
        peak_forward_prices={},
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        seam_window_hours=72,
        target_boundary_jump_eur_mwh=8.0,
        min_boundary_jump_eur_mwh=20.0,
        max_abs_delta_eur_mwh=20.0,
        negative_price_floor=-30.0,
        max_weighted_negative_hours=0,
    )

    assert not audit.empty
    assert float(smoothed["price_weighted_mean_eur_mwh"].mean()) == pytest.approx(target, abs=1e-6)

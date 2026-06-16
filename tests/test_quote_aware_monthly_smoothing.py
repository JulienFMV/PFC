from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.model.quote_aware_monthly_smoothing import apply_quote_aware_monthly_smoothing
from scripts.export_local_test_ch_hourly_csv import _eex_peak_mask


def _hourly_frame(timestamps: pd.DatetimeIndex, monthly_levels: dict[int, float]) -> pd.DataFrame:
    values = np.array([monthly_levels[int(ts.month)] for ts in timestamps], dtype=float)
    return pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": values - 2.0,
            "price_central_eur_mwh": values,
            "price_fast_eur_mwh": values + 2.0,
            "price_weighted_mean_eur_mwh": values,
            "structural_p10_eur_mwh": values - 2.0,
            "structural_p50_eur_mwh": values,
            "structural_p90_eur_mwh": values + 2.0,
            "structural_width_eur_mwh": [4.0] * len(timestamps),
        }
    )


def _monthly_means(frame: pd.DataFrame, timestamps: pd.DatetimeIndex) -> pd.Series:
    months = pd.PeriodIndex(timestamps.strftime("%Y-%m"), freq="M")
    return frame.groupby(months)["price_weighted_mean_eur_mwh"].mean()


def _roughness(values: pd.Series) -> float:
    return float(np.abs(np.diff(values.to_numpy(dtype=float), n=2)).sum())


def test_quote_aware_monthly_smoothing_reduces_intracurve_sawtooth_and_preserves_eex() -> None:
    timestamps = pd.date_range("2027-01-01", "2027-03-31 23:00", freq="h", tz="Europe/Zurich")
    hourly = _hourly_frame(timestamps, {1: 120.0, 2: 150.0, 3: 95.0})
    peak = _eex_peak_mask(pd.Series(timestamps))
    base_target = float(hourly["price_weighted_mean_eur_mwh"].mean())
    peak_target = float(hourly.loc[peak.to_numpy(), "price_weighted_mean_eur_mwh"].mean())

    smoothed, audit = apply_quote_aware_monthly_smoothing(
        hourly,
        ts_ch=pd.Series(timestamps),
        base_forward_prices={"2027-Q1": base_target},
        peak_forward_prices={"2027-Q1": peak_target},
        peak_mask=peak,
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        intensity=1.0,
        smoothness_lambda=20.0,
        negative_price_floor=-30.0,
    )

    old_months = _monthly_means(hourly, timestamps)
    new_months = _monthly_means(smoothed, timestamps)
    assert _roughness(new_months) < _roughness(old_months)
    assert smoothed["price_weighted_mean_eur_mwh"].mean() == pytest.approx(base_target, abs=1e-6)
    assert smoothed.loc[peak.to_numpy(), "price_weighted_mean_eur_mwh"].mean() == pytest.approx(peak_target, abs=1e-6)
    assert not audit.empty
    assert audit["delta_eur_mwh"].abs().max() > 0.0
    assert (smoothed["structural_p10_eur_mwh"] <= smoothed["structural_p50_eur_mwh"]).all()
    assert (smoothed["structural_p50_eur_mwh"] <= smoothed["structural_p90_eur_mwh"]).all()
    assert audit["max_constraint_residual_mwh"].abs().max() < 1e-7
    assert audit["max_peak_constraint_residual_mwh"].abs().max() < 1e-7


def test_quote_aware_monthly_smoothing_keeps_direct_month_quotes_unchanged() -> None:
    timestamps = pd.date_range("2026-07-01", "2026-07-31 23:00", freq="h", tz="Europe/Zurich")
    hourly = _hourly_frame(timestamps, {7: 93.91})

    smoothed, audit = apply_quote_aware_monthly_smoothing(
        hourly,
        ts_ch=pd.Series(timestamps),
        base_forward_prices={"2026-07": 93.91},
        peak_forward_prices={"2026-07": 88.56},
        peak_mask=_eex_peak_mask(pd.Series(timestamps)),
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
    )

    pd.testing.assert_series_equal(
        smoothed["price_weighted_mean_eur_mwh"],
        hourly["price_weighted_mean_eur_mwh"],
        check_names=False,
    )
    assert audit.empty


def test_quote_aware_monthly_smoothing_handles_partial_q1_and_cal_residual() -> None:
    timestamps = pd.date_range("2028-01-01", "2028-12-31 23:00", freq="h", tz="Europe/Zurich")
    levels = {
        1: 105.0,
        2: 123.0,
        3: 100.0,
        4: 52.0,
        5: 44.0,
        6: 40.0,
        7: 63.0,
        8: 47.0,
        9: 77.0,
        10: 86.0,
        11: 113.0,
        12: 106.0,
    }
    hourly = _hourly_frame(timestamps, levels)
    peak = _eex_peak_mask(pd.Series(timestamps))
    q1 = timestamps.quarter == 1
    cal_target = float(hourly["price_weighted_mean_eur_mwh"].mean())
    q1_target = float(hourly.loc[q1, "price_weighted_mean_eur_mwh"].mean())
    q1_peak_target = float(hourly.loc[q1 & peak.to_numpy(), "price_weighted_mean_eur_mwh"].mean())
    cal_peak_target = float(hourly.loc[peak.to_numpy(), "price_weighted_mean_eur_mwh"].mean())
    old = _monthly_means(hourly, timestamps)

    smoothed, audit = apply_quote_aware_monthly_smoothing(
        hourly,
        ts_ch=pd.Series(timestamps),
        base_forward_prices={"2028": cal_target, "2028-Q1": q1_target},
        peak_forward_prices={"2028": cal_peak_target, "2028-Q1": q1_peak_target},
        peak_mask=peak,
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        intensity=1.0,
        smoothness_lambda=20.0,
        negative_price_floor=-30.0,
    )
    new = _monthly_means(smoothed, timestamps)
    residual_mask = timestamps.quarter != 1

    assert smoothed.loc[q1, "price_weighted_mean_eur_mwh"].mean() == pytest.approx(q1_target, abs=1e-6)
    assert smoothed["price_weighted_mean_eur_mwh"].mean() == pytest.approx(cal_target, abs=1e-6)
    assert smoothed.loc[q1 & peak.to_numpy(), "price_weighted_mean_eur_mwh"].mean() == pytest.approx(
        q1_peak_target, abs=1e-6
    )
    assert smoothed.loc[peak.to_numpy(), "price_weighted_mean_eur_mwh"].mean() == pytest.approx(
        cal_peak_target, abs=1e-6
    )
    assert _roughness(new.loc["2028-04":"2028-12"]) < _roughness(old.loc["2028-04":"2028-12"])
    assert smoothed.loc[residual_mask, "price_weighted_mean_eur_mwh"].notna().all()
    assert audit["max_constraint_residual_mwh"].abs().max() < 1e-7


def test_quote_aware_monthly_smoothing_is_position_based_for_nondefault_index() -> None:
    timestamps = pd.date_range("2027-01-01", "2027-03-31 23:00", freq="h", tz="Europe/Zurich")
    hourly = _hourly_frame(timestamps, {1: 120.0, 2: 150.0, 3: 95.0})
    hourly.index = pd.RangeIndex(1000, 1000 + len(hourly))
    peak = _eex_peak_mask(pd.Series(timestamps))

    smoothed, _ = apply_quote_aware_monthly_smoothing(
        hourly,
        ts_ch=pd.Series(timestamps),
        base_forward_prices={"2027-Q1": float(hourly["price_weighted_mean_eur_mwh"].mean())},
        peak_forward_prices={"2027-Q1": float(hourly.loc[peak.to_numpy(), "price_weighted_mean_eur_mwh"].mean())},
        peak_mask=peak,
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        smoothness_lambda=20.0,
    )

    assert smoothed.index.equals(hourly.index)
    assert smoothed["price_weighted_mean_eur_mwh"].notna().all()


def test_quote_aware_monthly_smoothing_rejects_uncovered_rows() -> None:
    timestamps = pd.date_range("2027-01-01", periods=72, freq="h", tz="Europe/Zurich")
    hourly = _hourly_frame(timestamps, {1: 100.0})

    with pytest.raises(ValueError, match="without an EEX BASE bucket"):
        apply_quote_aware_monthly_smoothing(
            hourly,
            ts_ch=pd.Series(timestamps),
            base_forward_prices={},
            peak_forward_prices={},
            peak_mask=_eex_peak_mask(pd.Series(timestamps)),
            weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        )

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


def _month_boundary_jump(frame: pd.DataFrame, timestamps: pd.DatetimeIndex) -> float:
    ts = pd.Series(timestamps)
    jumps: list[float] = []
    for pos in range(1, len(ts)):
        if ts.iloc[pos].month != ts.iloc[pos - 1].month:
            jumps.append(
                abs(
                    float(frame["price_weighted_mean_eur_mwh"].iloc[pos])
                    - float(frame["price_weighted_mean_eur_mwh"].iloc[pos - 1])
                )
            )
    return max(jumps) if jumps else 0.0


def _series_month_boundary_jump(values: pd.Series, timestamps: pd.DatetimeIndex) -> float:
    ts = pd.Series(timestamps)
    jumps: list[float] = []
    for pos in range(1, len(ts)):
        if ts.iloc[pos].month != ts.iloc[pos - 1].month:
            jumps.append(abs(float(values.iloc[pos]) - float(values.iloc[pos - 1])))
    return max(jumps) if jumps else 0.0


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


def test_quote_aware_monthly_smoothing_reduces_month_boundary_jumps() -> None:
    timestamps = pd.date_range("2027-01-01", "2027-03-31 23:00", freq="h", tz="Europe/Zurich")
    hourly = _hourly_frame(timestamps, {1: 120.0, 2: 160.0, 3: 85.0})
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
        smoothness_lambda=25.0,
        negative_price_floor=-30.0,
    )
    stepped = hourly.copy()
    monthly_delta = audit.set_index("month")["target_delta_eur_mwh"].to_dict()
    for month, delta in monthly_delta.items():
        mask = timestamps.month == int(month)
        stepped.loc[mask, "price_weighted_mean_eur_mwh"] += float(delta)

    smoothed_delta = smoothed["price_weighted_mean_eur_mwh"] - hourly["price_weighted_mean_eur_mwh"]
    stepped_delta = stepped["price_weighted_mean_eur_mwh"] - hourly["price_weighted_mean_eur_mwh"]

    assert _series_month_boundary_jump(smoothed_delta, timestamps) < _series_month_boundary_jump(
        stepped_delta,
        timestamps,
    )
    assert smoothed["price_weighted_mean_eur_mwh"].mean() == pytest.approx(base_target, abs=1e-6)
    assert smoothed.loc[peak.to_numpy(), "price_weighted_mean_eur_mwh"].mean() == pytest.approx(peak_target, abs=1e-6)
    assert audit["max_projected_constraint_residual_mwh"].abs().max() < 1e-7


def test_quote_aware_monthly_smoothing_reduces_inter_year_delta_jump() -> None:
    timestamps = pd.date_range("2028-01-01", "2029-12-31 23:00", freq="h", tz="Europe/Zurich")
    hourly = _hourly_frame(
        timestamps,
        {1: 120.0, 2: 100.0, 3: 80.0, 4: 60.0, 5: 50.0, 6: 45.0, 7: 55.0, 8: 60.0, 9: 75.0, 10: 90.0, 11: 110.0, 12: 115.0},
    )
    year_2029 = timestamps.year == 2029
    value_cols = [
        "price_slow_eur_mwh",
        "price_central_eur_mwh",
        "price_fast_eur_mwh",
        "price_weighted_mean_eur_mwh",
        "structural_p10_eur_mwh",
        "structural_p50_eur_mwh",
        "structural_p90_eur_mwh",
    ]
    hourly.loc[year_2029, value_cols] = hourly.loc[year_2029, value_cols] - 10.0
    peak = _eex_peak_mask(pd.Series(timestamps))
    base_targets = {
        "2028": float(hourly.loc[timestamps.year == 2028, "price_weighted_mean_eur_mwh"].mean()),
        "2029": float(hourly.loc[timestamps.year == 2029, "price_weighted_mean_eur_mwh"].mean()),
    }
    peak_targets = {
        "2028": float(hourly.loc[(timestamps.year == 2028) & peak.to_numpy(), "price_weighted_mean_eur_mwh"].mean()),
        "2029": float(hourly.loc[(timestamps.year == 2029) & peak.to_numpy(), "price_weighted_mean_eur_mwh"].mean()),
    }

    smoothed, audit = apply_quote_aware_monthly_smoothing(
        hourly,
        ts_ch=pd.Series(timestamps),
        base_forward_prices=base_targets,
        peak_forward_prices=peak_targets,
        peak_mask=peak,
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        intensity=1.0,
        smoothness_lambda=25.0,
        negative_price_floor=-30.0,
    )
    target_delta = audit.set_index(["year", "month"])["target_delta_eur_mwh"].to_dict()
    stepped_delta = pd.Series(
        [float(target_delta[(int(ts.year), int(ts.month))]) for ts in timestamps],
        index=hourly.index,
        dtype=float,
    )
    smoothed_delta = smoothed["price_weighted_mean_eur_mwh"] - hourly["price_weighted_mean_eur_mwh"]
    year_boundary_pos = int(np.flatnonzero((timestamps.year == 2029) & (timestamps.month == 1))[0])

    stepped_jump = abs(float(stepped_delta.iloc[year_boundary_pos] - stepped_delta.iloc[year_boundary_pos - 1]))
    smoothed_jump = abs(float(smoothed_delta.iloc[year_boundary_pos] - smoothed_delta.iloc[year_boundary_pos - 1]))

    assert smoothed_jump < stepped_jump * 0.10
    assert smoothed["price_weighted_mean_eur_mwh"].groupby(timestamps.year).mean().loc[2028] == pytest.approx(
        base_targets["2028"], abs=1e-6
    )
    assert smoothed["price_weighted_mean_eur_mwh"].groupby(timestamps.year).mean().loc[2029] == pytest.approx(
        base_targets["2029"], abs=1e-6
    )
    assert audit["max_projected_constraint_residual_mwh"].abs().max() < 1e-7


@pytest.mark.parametrize(
    ("intensity", "smoothness_lambda", "max_delta"),
    [
        (0.0, 25.0, 18.0),
        (1.0, 0.0, 18.0),
        (1.0, 25.0, 0.0),
    ],
)
def test_quote_aware_monthly_smoothing_zero_controls_are_noop(
    intensity: float,
    smoothness_lambda: float,
    max_delta: float,
) -> None:
    timestamps = pd.date_range("2027-01-01", "2027-03-31 23:00", freq="h", tz="Europe/Zurich")
    hourly = _hourly_frame(timestamps, {1: 120.0, 2: 160.0, 3: 85.0})
    peak = _eex_peak_mask(pd.Series(timestamps))

    smoothed, audit = apply_quote_aware_monthly_smoothing(
        hourly,
        ts_ch=pd.Series(timestamps),
        base_forward_prices={"2027-Q1": float(hourly["price_weighted_mean_eur_mwh"].mean())},
        peak_forward_prices={"2027-Q1": float(hourly.loc[peak.to_numpy(), "price_weighted_mean_eur_mwh"].mean())},
        peak_mask=peak,
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        intensity=intensity,
        smoothness_lambda=smoothness_lambda,
        max_unquoted_month_delta_eur_mwh=max_delta,
        negative_price_floor=-30.0,
    )

    pd.testing.assert_frame_equal(smoothed, hourly)
    assert audit.empty


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


def test_quote_aware_monthly_smoothing_freezes_direct_month_quotes_inside_mixed_bucket() -> None:
    timestamps = pd.date_range("2028-01-01", "2028-03-31 23:00", freq="h", tz="Europe/Zurich")
    hourly = _hourly_frame(timestamps, {1: 125.0, 2: 95.0, 3: 160.0})
    peak = _eex_peak_mask(pd.Series(timestamps))
    january = timestamps.month == 1
    q1_target = float(hourly["price_weighted_mean_eur_mwh"].mean())
    q1_peak_target = float(hourly.loc[peak.to_numpy(), "price_weighted_mean_eur_mwh"].mean())
    jan_target = float(hourly.loc[january, "price_weighted_mean_eur_mwh"].mean())
    jan_peak_target = float(hourly.loc[january & peak.to_numpy(), "price_weighted_mean_eur_mwh"].mean())

    smoothed, audit = apply_quote_aware_monthly_smoothing(
        hourly,
        ts_ch=pd.Series(timestamps),
        base_forward_prices={"2028-Q1": q1_target, "2028-01": jan_target},
        peak_forward_prices={"2028-Q1": q1_peak_target, "2028-01": jan_peak_target},
        peak_mask=peak,
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        intensity=1.0,
        smoothness_lambda=25.0,
        negative_price_floor=-30.0,
    )

    for column in [
        "price_slow_eur_mwh",
        "price_central_eur_mwh",
        "price_fast_eur_mwh",
        "price_weighted_mean_eur_mwh",
    ]:
        pd.testing.assert_series_equal(smoothed.loc[january, column], hourly.loc[january, column], check_names=False)
    assert smoothed["price_weighted_mean_eur_mwh"].mean() == pytest.approx(q1_target, abs=1e-6)
    assert smoothed.loc[peak.to_numpy(), "price_weighted_mean_eur_mwh"].mean() == pytest.approx(q1_peak_target, abs=1e-6)
    assert audit["max_projected_constraint_residual_mwh"].abs().max() < 1e-7


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

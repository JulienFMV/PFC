from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.compare_epex_shape_lab_ab import compare_ab


def _write_candidate(path, timestamps: pd.DatetimeIndex, delta: np.ndarray | float = 0.0) -> None:
    base = np.where(timestamps.month == 1, 80.0, 90.0).astype(float)
    delta_arr = np.full(len(timestamps), float(delta)) if np.isscalar(delta) else np.asarray(delta, dtype=float)
    values = base + delta_arr
    frame = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%d.%m.%Y %H:%M"),
            "utc_offset_ch": timestamps.strftime("UTC%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%d.%m.%Y %H:%M"),
            "price_slow_eur_mwh": values - 2.0,
            "price_central_eur_mwh": values,
            "price_fast_eur_mwh": values + 2.0,
            "price_weighted_mean_eur_mwh": values,
            "structural_p10_eur_mwh": values - 8.0,
            "structural_p50_eur_mwh": values,
            "structural_p90_eur_mwh": values + 12.0,
            "structural_width_eur_mwh": np.full(len(values), 20.0),
        }
    )
    frame.to_csv(path, index=False)


def test_compare_epex_shape_lab_ab_reports_level_and_shape_deltas(tmp_path) -> None:
    timestamps = pd.date_range("2030-01-01", "2030-02-28 23:00", freq="h", tz="Europe/Zurich")
    delta = np.where(timestamps.weekday >= 5, -1.0, 0.4)
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    output = tmp_path / "comparison"
    _write_candidate(baseline, timestamps)
    _write_candidate(adjusted, timestamps, delta=delta)

    summary = compare_ab(baseline_csv=baseline, adjusted_csv=adjusted, output_dir=output)

    assert summary["benchmark_policy"] == "independent_no_ompex"
    assert summary["ompex_used_in_selection"] is False
    assert summary["n_hours"] == len(timestamps)
    assert summary["quantile_order_adjusted_ok"] is True
    assert summary["max_abs_delta_eur_mwh"] == pytest.approx(1.0)
    assert summary["max_abs_width_delta_eur_mwh"] == pytest.approx(0.0)
    assert summary["max_abs_implied_width_delta_eur_mwh"] == pytest.approx(0.0)
    assert summary["max_abs_reported_minus_implied_width_adjusted_eur_mwh"] == pytest.approx(0.0)
    assert summary["max_abs_peak_offpeak_spread_delta_eur_mwh"] >= 0.0
    assert summary["max_abs_month_hour_mean_delta_eur_mwh"] > 0.0
    assert summary["max_abs_boundary_delta_jump_eur_mwh"] >= 0.0
    assert summary["weighted_negative_cluster_max_hours"] == 0
    assert summary["p10_negative_hours_adjusted"] == 0
    assert (output / "monthly_summary.csv").exists()
    assert (output / "calendar_delta_summary.csv").exists()
    assert (output / "load_type_delta_summary.csv").exists()
    assert (output / "month_hour_delta_summary.csv").exists()
    assert (output / "peak_offpeak_monthly_summary.csv").exists()
    assert (output / "boundary_delta_jumps.csv").exists()
    assert (output / "delta_heatmap_month_hour_2030.png").exists()
    assert (output / "peak_offpeak_spread_delta_by_month.png").exists()
    assert (output / "boundary_delta_jumps.png").exists()

    load_type = pd.read_csv(output / "load_type_delta_summary.csv")
    assert set(load_type["load_type"]) == {"BASE", "PEAK", "OFFPEAK"}
    peak_offpeak = pd.read_csv(output / "peak_offpeak_monthly_summary.csv")
    assert "peak_offpeak_spread_delta_eur_mwh" in set(peak_offpeak.columns)
    month_hour = pd.read_csv(output / "month_hour_delta_summary.csv")
    assert set(["year", "month", "hour", "mean_delta_eur_mwh"]).issubset(month_hour.columns)

    written = json.loads((output / "ab_comparison_summary.json").read_text(encoding="utf-8"))
    assert written["benchmark_policy"] == "independent_no_ompex"
    assert "load_type_delta_summary_csv" in written["outputs"]


def test_compare_epex_shape_lab_ab_reports_stale_width_column(tmp_path) -> None:
    timestamps = pd.date_range("2030-01-01", periods=24, freq="h", tz="Europe/Zurich")
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    _write_candidate(baseline, timestamps)
    _write_candidate(adjusted, timestamps)
    adjusted_frame = pd.read_csv(adjusted)
    adjusted_frame["structural_p90_eur_mwh"] = adjusted_frame["structural_p90_eur_mwh"] + 1.5
    adjusted_frame.to_csv(adjusted, index=False)

    summary = compare_ab(baseline_csv=baseline, adjusted_csv=adjusted, output_dir=tmp_path / "comparison")

    assert summary["max_abs_width_delta_eur_mwh"] == pytest.approx(0.0)
    assert summary["max_abs_implied_width_delta_eur_mwh"] == pytest.approx(1.5)
    assert summary["max_abs_reported_minus_implied_width_adjusted_eur_mwh"] == pytest.approx(1.5)
    monthly = pd.read_csv(tmp_path / "comparison" / "monthly_summary.csv")
    assert "implied_width_delta_mean_eur_mwh" in set(monthly.columns)


def test_compare_epex_shape_lab_ab_rejects_timestamp_mismatch(tmp_path) -> None:
    baseline_ts = pd.date_range("2030-01-01", periods=24, freq="h", tz="Europe/Zurich")
    adjusted_ts = pd.date_range("2030-01-02", periods=24, freq="h", tz="Europe/Zurich")
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    _write_candidate(baseline, baseline_ts)
    _write_candidate(adjusted, adjusted_ts)

    with pytest.raises(ValueError, match="identical timestamp sets"):
        compare_ab(baseline_csv=baseline, adjusted_csv=adjusted, output_dir=tmp_path / "out")

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.backtest_epex_shape_lab_against_spot import (
    _rolling_cutoffs,
    backtest_against_spot,
)


def _write_candidate(path: Path, timestamps: pd.DatetimeIndex, *, shaped: bool) -> None:
    base = np.full(len(timestamps), 80.0)
    shape = _shape(timestamps) if shaped else np.zeros(len(timestamps))
    if shaped:
        month_key = pd.Series(timestamps.strftime("%Y-%m"))
        shape = pd.Series(shape).groupby(month_key).transform(lambda values: values - values.mean()).to_numpy()
    values = base + shape
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
            "structural_p90_eur_mwh": values + 8.0,
            "structural_width_eur_mwh": np.full(len(values), 16.0),
        }
    )
    frame.to_csv(path, index=False)


def _write_spot(path: Path, timestamps_utc: pd.DatetimeIndex) -> None:
    local = timestamps_utc.tz_convert("Europe/Zurich")
    values = 55.0 + _shape(local)
    frame = pd.DataFrame({"price_eur_mwh": values}, index=timestamps_utc)
    frame.to_parquet(path)


def _shape(timestamps: pd.DatetimeIndex) -> np.ndarray:
    hour = timestamps.hour.to_numpy(dtype=float)
    weekend = (timestamps.weekday >= 5).astype(float)
    return 4.0 * np.sin((hour - 7.0) / 24.0 * 2.0 * np.pi) - 2.0 * weekend


def test_backtest_epex_shape_lab_against_spot_reports_no_ompex_lab_only_gain(tmp_path: Path) -> None:
    candidate_ts = pd.date_range("2030-01-01", "2030-01-31 23:00", freq="h", tz="Europe/Zurich")
    spot_ts = pd.date_range("2024-01-01", "2026-03-31 23:00", freq="h", tz="UTC")
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    spot = tmp_path / "spot.parquet"
    _write_candidate(baseline, candidate_ts, shaped=False)
    _write_candidate(adjusted, candidate_ts, shaped=True)
    _write_spot(spot, spot_ts)

    summary = backtest_against_spot(
        baseline_csv=baseline,
        adjusted_csv=adjusted,
        spot_parquet=spot,
        output_dir=tmp_path / "out",
        valuation_timestamp="2026-04-01T00:00:00Z",
        rolling_cutoffs=["2025-01-01T00:00:00Z"],
        lookback_years=1,
        eval_days=31,
        embargo_days=1,
    )

    assert summary["status"] == "DIAGNOSTIC_PASS"
    assert summary["promotion_gate"] is False
    assert summary["production_approved"] is False
    assert summary["benchmark_policy"] == "rolling_origin_epex_spot_no_ompex_lab_only"
    assert summary["ompex_used_in_model"] is False
    assert summary["rolling_metrics"]["mean_profile_mae_improvement_eur_mwh"] > 0.0
    assert summary["rolling_bucket_metrics"]["residual_level:all"]["mean_mae_improvement_eur_mwh"] > 0.0
    assert "residual_level:solar_tail_mar_oct_10_16" in summary["rolling_bucket_metrics"]
    assert "hourly_ramp:all" in summary["rolling_bucket_metrics"]
    assert summary["strict_lab_checks"]["rolling_folds_no_temporal_leak"] is True
    assert (tmp_path / "out" / "rolling_spot_profile_folds.csv").exists()
    assert (tmp_path / "out" / "rolling_spot_bucket_metrics.csv").exists()
    post_csv = tmp_path / "out" / "post_valuation_timestamp_residuals.csv"
    assert summary["output_hashes"]["post_valuation_timestamp_residuals_csv"] == _sha256(post_csv)
    written = json.loads((tmp_path / "out" / "spot_backtest_summary.json").read_text(encoding="utf-8"))
    assert written["strict_lab_gate_pass"] is True
    assert written["output_hashes"] == summary["output_hashes"]
    before = {path.name: path.read_bytes() for path in (tmp_path / "out").iterdir()}
    with pytest.raises(FileExistsError, match="append-only"):
        backtest_against_spot(
            baseline_csv=baseline,
            adjusted_csv=adjusted,
            spot_parquet=spot,
            output_dir=tmp_path / "out",
            valuation_timestamp="2026-04-01T00:00:00Z",
            rolling_cutoffs=["2025-01-01T00:00:00Z"],
            lookback_years=1,
            eval_days=31,
            embargo_days=1,
        )
    assert {path.name: path.read_bytes() for path in (tmp_path / "out").iterdir()} == before


def test_backtest_epex_shape_lab_against_spot_rejects_timestamp_mismatch(tmp_path: Path) -> None:
    baseline_ts = pd.date_range("2030-01-01", periods=48, freq="h", tz="Europe/Zurich")
    adjusted_ts = pd.date_range("2030-01-02", periods=48, freq="h", tz="Europe/Zurich")
    spot_ts = pd.date_range("2024-01-01", periods=24 * 400, freq="h", tz="UTC")
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    spot = tmp_path / "spot.parquet"
    _write_candidate(baseline, baseline_ts, shaped=False)
    _write_candidate(adjusted, adjusted_ts, shaped=True)
    _write_spot(spot, spot_ts)

    with pytest.raises(ValueError, match="identical timestamp sets"):
        backtest_against_spot(
            baseline_csv=baseline,
            adjusted_csv=adjusted,
            spot_parquet=spot,
            output_dir=tmp_path / "out",
            valuation_timestamp="2025-02-01T00:00:00Z",
            rolling_cutoffs=["2025-07-01T00:00:00Z"],
            lookback_years=1,
        )


def test_locked_evaluation_window_is_invariant_to_outside_rows_and_parallel_shift(tmp_path: Path) -> None:
    holdout_start = pd.Timestamp("2026-07-10T00:00:00Z")
    holdout_end = pd.Timestamp("2026-07-12T00:00:00Z")
    candidate_ts = pd.date_range(
        "2026-07-01T00:00:00Z",
        "2026-08-01T00:00:00Z",
        freq="h",
        inclusive="left",
    ).tz_convert("Europe/Zurich")
    spot_ts = pd.date_range("2024-01-01T00:00:00Z", "2026-07-13T00:00:00Z", freq="h")
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    spot_a = tmp_path / "spot-a.parquet"
    spot_b = tmp_path / "spot-b.parquet"
    spot_shifted = tmp_path / "spot-shifted.parquet"
    _write_candidate(baseline, candidate_ts, shaped=False)
    _write_candidate(adjusted, candidate_ts, shaped=True)
    _write_spot(spot_a, spot_ts)
    frame = pd.read_parquet(spot_a)
    contaminated = frame.copy()
    contaminated.loc[pd.Timestamp("2026-07-12T12:00:00Z"), "price_eur_mwh"] = 1_000_000.0
    contaminated.to_parquet(spot_b)
    shifted = frame.copy()
    shifted["price_eur_mwh"] += 777.0
    shifted.to_parquet(spot_shifted)

    def run(spot: Path, name: str) -> dict:
        return backtest_against_spot(
            baseline_csv=baseline,
            adjusted_csv=adjusted,
            spot_parquet=spot,
            output_dir=tmp_path / name,
            valuation_timestamp="2026-07-09T00:00:00Z",
            rolling_cutoffs=["2025-07-01T00:00:00Z"],
            lookback_years=1,
            eval_days=14,
            embargo_days=1,
            min_eval_hours=24,
            evaluation_start_utc=str(holdout_start),
            evaluation_end_utc=str(holdout_end),
        )

    first = run(spot_a, "out-a")
    outside = run(spot_b, "out-b")
    parallel = run(spot_shifted, "out-shifted")

    assert first["strict_lab_gate_pass"] is True
    assert first["evaluation_window"][
        "all_residuals_centered_on_exact_evaluation_overlap"
    ] is True
    assert first["post_valuation_metrics"] == outside["post_valuation_metrics"]
    assert parallel["post_valuation_metrics"] == pytest.approx(first["post_valuation_metrics"], abs=1e-10)
    assert (tmp_path / "out-a" / "post_valuation_timestamp_residuals.csv").read_bytes() == (
        tmp_path / "out-b" / "post_valuation_timestamp_residuals.csv"
    ).read_bytes()
    first_post = pd.read_csv(tmp_path / "out-a" / "post_valuation_timestamp_residuals.csv")
    shifted_post = pd.read_csv(tmp_path / "out-shifted" / "post_valuation_timestamp_residuals.csv")
    assert first_post[[column for column in first_post if column.endswith("eur_mwh")]].to_numpy() == pytest.approx(
        shifted_post[[column for column in shifted_post if column.endswith("eur_mwh")]].to_numpy(), abs=1e-10
    )


def test_auto_rolling_cutoffs_do_not_duplicate_a_single_admissible_origin() -> None:
    index = pd.date_range("2024-01-01T00:00:00Z", "2026-01-16T00:00:00Z", freq="h")
    spot = pd.DataFrame({"price_eur_mwh": np.ones(len(index))}, index=index)

    cutoffs = _rolling_cutoffs(
        spot,
        valuation_utc=pd.Timestamp("2026-02-01T00:00:00Z"),
        rolling_cutoffs=None,
        lookback_years=2,
        eval_days=14,
        embargo_days=1,
        max_auto_folds=12,
    )

    assert cutoffs == [pd.Timestamp("2026-01-01T00:00:00Z")]


def test_explicit_rolling_cutoffs_reject_duplicates() -> None:
    index = pd.date_range("2024-01-01T00:00:00Z", periods=48, freq="h")
    spot = pd.DataFrame({"price_eur_mwh": np.ones(len(index))}, index=index)

    with pytest.raises(ValueError, match="must be unique"):
        _rolling_cutoffs(
            spot,
            valuation_utc=pd.Timestamp("2026-01-01T00:00:00Z"),
            rolling_cutoffs=["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            lookback_years=1,
            eval_days=14,
            embargo_days=1,
            max_auto_folds=12,
        )


def test_auto_rolling_cutoffs_are_daily_and_non_overlapping() -> None:
    index = pd.date_range("2023-01-01T00:00:00Z", "2026-01-01T00:00:00Z", freq="h")
    spot = pd.DataFrame({"price_eur_mwh": np.ones(len(index))}, index=index)

    cutoffs = _rolling_cutoffs(
        spot,
        valuation_utc=pd.Timestamp("2026-01-01T00:00:00Z"),
        rolling_cutoffs=None,
        lookback_years=2,
        eval_days=14,
        embargo_days=1,
        max_auto_folds=12,
    )

    assert 1 < len(cutoffs) <= 12
    assert all(cutoff == cutoff.floor("D") for cutoff in cutoffs)
    assert all(
        cutoffs[index] - cutoffs[index - 1] >= pd.Timedelta(days=14)
        for index in range(1, len(cutoffs))
    )


def test_rolling_cutoffs_reject_more_than_twelve_origins() -> None:
    index = pd.date_range("2023-01-01T00:00:00Z", "2026-01-01T00:00:00Z", freq="h")
    spot = pd.DataFrame({"price_eur_mwh": np.ones(len(index))}, index=index)

    with pytest.raises(ValueError, match="between 1 and 12"):
        _rolling_cutoffs(
            spot,
            valuation_utc=pd.Timestamp("2026-01-01T00:00:00Z"),
            rolling_cutoffs=None,
            lookback_years=1,
            eval_days=1,
            embargo_days=1,
            max_auto_folds=13,
        )
    explicit = [str(value) for value in pd.date_range("2025-01-01", periods=13, freq="14D", tz="UTC")]
    with pytest.raises(ValueError, match="cannot exceed 12"):
        _rolling_cutoffs(
            spot,
            valuation_utc=pd.Timestamp("2026-01-01T00:00:00Z"),
            rolling_cutoffs=explicit,
            lookback_years=1,
            eval_days=14,
            embargo_days=1,
            max_auto_folds=12,
        )


def test_auto_rolling_cutoffs_do_not_emit_non_daily_fallback() -> None:
    index = pd.date_range(
        "2024-01-01T12:00:00Z",
        "2025-01-03T12:00:00Z",
        freq="h",
    )
    spot = pd.DataFrame({"price_eur_mwh": np.ones(len(index))}, index=index)

    cutoffs = _rolling_cutoffs(
        spot,
        valuation_utc=pd.Timestamp("2025-02-01T00:00:00Z"),
        rolling_cutoffs=None,
        lookback_years=1,
        eval_days=1,
        embargo_days=1,
        max_auto_folds=12,
    )

    assert cutoffs == []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

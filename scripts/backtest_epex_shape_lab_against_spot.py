"""Backtest EPEX shape-lab candidates against realized EPEX spot, without OMPEX.

This diagnostic is lab-only.  It compares baseline and adjusted HPFC shapes
against realized EPEX spot profiles, records anti-leakage checks for rolling
spot folds, and never approves production promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.durable_artifact import write_durable_exact_bytes  # noqa: E402
from scripts.export_local_test_ch_hourly_csv import _parse_timestamp_ch  # noqa: E402

PRICE = "price_weighted_mean_eur_mwh"
SPOT_PRICE = "price_eur_mwh"
PRICE_COLUMNS = [
    "price_slow_eur_mwh",
    "price_central_eur_mwh",
    "price_fast_eur_mwh",
    PRICE,
    "structural_p10_eur_mwh",
    "structural_p50_eur_mwh",
    "structural_p90_eur_mwh",
    "structural_width_eur_mwh",
]
PROFILE_KEYS = ["month", "hour", "is_weekend"]
_MAX_ROLLING_FOLDS = 12


def backtest_against_spot(
    *,
    baseline_csv: Path,
    adjusted_csv: Path,
    spot_parquet: Path,
    output_dir: Path,
    valuation_timestamp: str,
    rolling_cutoffs: list[str] | None = None,
    lookback_years: int = 2,
    eval_days: int = 30,
    embargo_days: int = 1,
    max_auto_folds: int = 12,
    min_eval_hours: int = 24,
    evaluation_start_utc: str | None = None,
    evaluation_end_utc: str | None = None,
    spot_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if lookback_years <= 0:
        raise ValueError("lookback_years must be positive")
    if eval_days <= 0:
        raise ValueError("eval_days must be positive")
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative")
    if max_auto_folds <= 0 or max_auto_folds > _MAX_ROLLING_FOLDS:
        raise ValueError(f"max_auto_folds must be between 1 and {_MAX_ROLLING_FOLDS}")

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_csv = output_dir / "rolling_spot_profile_folds.csv"
    bucket_csv = output_dir / "rolling_spot_bucket_metrics.csv"
    profile_csv = output_dir / "candidate_month_hour_profiles.csv"
    post_csv = output_dir / "post_valuation_timestamp_residuals.csv"
    summary_path = output_dir / "spot_backtest_summary.json"
    output_paths = (fold_csv, bucket_csv, profile_csv, post_csv, summary_path)
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"backtest evidence is append-only; outputs already exist: {existing}")

    valuation_utc = _to_utc_timestamp(valuation_timestamp)
    evaluation_start, evaluation_end = _validated_evaluation_window(
        evaluation_start_utc=evaluation_start_utc,
        evaluation_end_utc=evaluation_end_utc,
        valuation_utc=valuation_utc,
        embargo_days=embargo_days,
    )
    baseline = _load_candidate(baseline_csv, label="baseline")
    adjusted = _load_candidate(adjusted_csv, label="adjusted")
    spot = _load_spot(spot_parquet)
    joined = _join_candidates(baseline, adjusted)

    candidate_profiles = _candidate_profiles(joined)
    cutoffs = _rolling_cutoffs(
        spot,
        valuation_utc=valuation_utc,
        rolling_cutoffs=rolling_cutoffs,
        lookback_years=lookback_years,
        eval_days=eval_days,
        embargo_days=embargo_days,
        max_auto_folds=max_auto_folds,
    )
    fold_rows = [
        _evaluate_fold(
            spot,
            candidate_profiles,
            cutoff=cutoff,
            lookback_years=lookback_years,
            eval_days=eval_days,
            embargo_days=embargo_days,
            valuation_utc=valuation_utc,
            min_eval_hours=min_eval_hours,
        )
        for cutoff in cutoffs
    ]
    bucket_rows = [
        row
        for cutoff in cutoffs
        for row in _fold_bucket_rows(
            spot,
            candidate_profiles,
            cutoff=cutoff,
            eval_days=eval_days,
            embargo_days=embargo_days,
        )
    ]
    folds = pd.DataFrame(fold_rows)
    buckets = pd.DataFrame(bucket_rows)
    post_valuation = _post_valuation_overlap(
        joined,
        spot,
        valuation_utc=valuation_utc,
        embargo_days=embargo_days,
        evaluation_start_utc=evaluation_start,
        evaluation_end_utc=evaluation_end,
    )
    strict_checks = _strict_lab_checks(
        joined,
        folds,
        post_valuation,
        min_eval_hours=min_eval_hours,
        evaluation_start_utc=evaluation_start,
        evaluation_end_utc=evaluation_end,
    )

    folds.to_csv(fold_csv, index=False, mode="x")
    buckets.to_csv(bucket_csv, index=False, mode="x")
    candidate_profiles.to_csv(profile_csv, index=False, mode="x")
    post_valuation.to_csv(post_csv, index=False, mode="x")

    passed_folds = folds[folds["fold_eligible"].astype(bool)] if not folds.empty else folds
    summary = {
        "schema_version": "epex_shape_lab_spot_backtest.v3",
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "independent_production_evidence": False,
        "benchmark_policy": "rolling_origin_epex_spot_no_ompex_lab_only",
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "baseline_csv": str(baseline_csv),
        "adjusted_csv": str(adjusted_csv),
        "spot_parquet": str(spot_parquet),
        "source_hashes": {
            "baseline_csv": _sha256(baseline_csv),
            "adjusted_csv": _sha256(adjusted_csv),
            "spot_parquet": _sha256(spot_parquet),
        },
        "spot_provenance": spot_provenance,
        "valuation_timestamp_utc": str(valuation_utc),
        "lookback_years": int(lookback_years),
        "eval_days": int(eval_days),
        "embargo_days": int(embargo_days),
        "candidate_hours": int(len(joined)),
        "spot_hours": int(len(spot)),
        "evaluation_spot_hours": int(
            len(_spot_evaluation_window(spot, start_utc=evaluation_start, end_utc=evaluation_end))
        ),
        "evaluation_window": {
            "start_utc": _utc_text_or_none(evaluation_start),
            "end_utc": _utc_text_or_none(evaluation_end),
            "end_exclusive": True,
            "spot_filtered_before_residual_centering": True,
            "all_residuals_centered_on_exact_evaluation_overlap": True,
        },
        "rolling_fold_count": int(len(folds)),
        "eligible_rolling_fold_count": int(len(passed_folds)),
        "post_valuation_overlap_hours": int(len(post_valuation)),
        "strict_lab_gate_pass": bool(all(strict_checks.values())),
        "strict_lab_checks": strict_checks,
        "rolling_metrics": _aggregate_fold_metrics(passed_folds),
        "rolling_bucket_metrics": _aggregate_bucket_metrics(buckets),
        "post_valuation_metrics": _post_valuation_metrics(post_valuation),
        "outputs": {
            "rolling_spot_profile_folds_csv": str(fold_csv),
            "rolling_spot_bucket_metrics_csv": str(bucket_csv),
            "candidate_month_hour_profiles_csv": str(profile_csv),
            "post_valuation_timestamp_residuals_csv": str(post_csv),
        },
        "output_hashes": {
            "rolling_spot_profile_folds_csv": _sha256(fold_csv),
            "rolling_spot_bucket_metrics_csv": _sha256(bucket_csv),
            "candidate_month_hour_profiles_csv": _sha256(profile_csv),
            "post_valuation_timestamp_residuals_csv": _sha256(post_csv),
        },
        "status": "DIAGNOSTIC_PASS" if all(strict_checks.values()) else "DIAGNOSTIC_FAIL",
        "note": (
            "Lab-only no-OMPEX spot diagnostic. Historical rolling folds test "
            "profile stability and relative shape fit, but they do not approve "
            "production because the candidate may have been selected after "
            "historical spot rows were known. A configured locked evaluation "
            "window is isolated before residual centering."
        ),
    }
    payload = (json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_durable_exact_bytes(summary_path, payload, label="spot backtest summary")
    return summary


def _load_candidate(path: Path, *, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(["timestamp_ch", *PRICE_COLUMNS]) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} candidate missing required columns: {missing}")
    ts = _parse_timestamp_ch(frame["timestamp_ch"], frame.get("utc_offset_ch"))
    out = frame.copy()
    for column in PRICE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="raise")
    out.index = pd.DatetimeIndex(ts)
    if out.index.has_duplicates:
        raise ValueError(f"{label} candidate has duplicate timestamps")
    out = out.sort_index()
    out.index.name = "timestamp_ch"
    return out[PRICE_COLUMNS]


def _load_spot(path: Path) -> pd.DataFrame:
    spot = pd.read_parquet(path)
    if SPOT_PRICE not in spot.columns:
        raise ValueError(f"spot parquet missing {SPOT_PRICE!r}")
    if not isinstance(spot.index, pd.DatetimeIndex):
        raise ValueError("spot parquet must use a DatetimeIndex")
    idx = spot.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out = pd.DataFrame({SPOT_PRICE: pd.to_numeric(spot[SPOT_PRICE], errors="raise").to_numpy(dtype=float)}, index=idx)
    out = out[np.isfinite(out[SPOT_PRICE].to_numpy(dtype=float))].sort_index()
    if out.index.has_duplicates:
        raise ValueError("spot parquet has duplicate timestamps")
    return out


def _join_candidates(baseline: pd.DataFrame, adjusted: pd.DataFrame) -> pd.DataFrame:
    joined = baseline.join(adjusted, how="inner", lsuffix="_baseline", rsuffix="_adjusted")
    if len(joined) != len(baseline) or len(joined) != len(adjusted):
        raise ValueError(
            "baseline and adjusted candidates must have identical timestamp sets: "
            f"baseline={len(baseline)}, adjusted={len(adjusted)}, overlap={len(joined)}"
        )
    joined["timestamp_utc"] = joined.index.tz_convert("UTC")
    joined = _add_local_calendar(joined)
    for suffix in ("baseline", "adjusted"):
        price = joined[f"{PRICE}_{suffix}"]
        monthly_mean = price.groupby(joined["year_month"]).transform("mean")
        joined[f"shape_residual_{suffix}_eur_mwh"] = price - monthly_mean
    joined["weighted_delta_eur_mwh"] = joined[f"{PRICE}_adjusted"] - joined[f"{PRICE}_baseline"]
    joined["implied_width_delta_eur_mwh"] = (
        joined["structural_p90_eur_mwh_adjusted"]
        - joined["structural_p10_eur_mwh_adjusted"]
        - joined["structural_p90_eur_mwh_baseline"]
        + joined["structural_p10_eur_mwh_baseline"]
    )
    return joined


def _candidate_profiles(joined: pd.DataFrame) -> pd.DataFrame:
    grouped = joined.groupby(PROFILE_KEYS, sort=True)
    rows = []
    for key, group in grouped:
        month, hour, is_weekend = key
        rows.append(
            {
                "month": int(month),
                "hour": int(hour),
                "is_weekend": bool(is_weekend),
                "candidate_hours": int(len(group)),
                "baseline_shape_residual_eur_mwh": float(group["shape_residual_baseline_eur_mwh"].mean()),
                "adjusted_shape_residual_eur_mwh": float(group["shape_residual_adjusted_eur_mwh"].mean()),
                "adjusted_minus_baseline_eur_mwh": float(group["weighted_delta_eur_mwh"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_fold(
    spot: pd.DataFrame,
    candidate_profiles: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    lookback_years: int,
    eval_days: int,
    embargo_days: int,
    valuation_utc: pd.Timestamp,
    min_eval_hours: int,
) -> dict[str, Any]:
    cutoff = _to_utc_timestamp(cutoff)
    train_start = cutoff - pd.DateOffset(years=int(lookback_years))
    eval_start = cutoff + pd.Timedelta(days=int(embargo_days))
    eval_end = eval_start + pd.Timedelta(days=int(eval_days))
    train = spot[(spot.index >= train_start) & (spot.index < cutoff)]
    eval_frame = spot[(spot.index >= eval_start) & (spot.index < eval_end)]
    train_profile = _spot_profile(train)
    eval_profile = _spot_profile(eval_frame)
    merged = candidate_profiles.merge(eval_profile, on=PROFILE_KEYS, how="inner", validate="one_to_one")
    train_eval = train_profile.merge(eval_profile, on=PROFILE_KEYS, how="inner", suffixes=("_train", "_eval"))
    train_eval_diff = (
        train_eval["spot_shape_residual_eur_mwh_eval"] - train_eval["spot_shape_residual_eur_mwh_train"]
        if not train_eval.empty
        else pd.Series(dtype=float)
    )
    baseline_error = (
        merged["baseline_shape_residual_eur_mwh"] - merged["spot_shape_residual_eur_mwh"]
        if not merged.empty
        else pd.Series(dtype=float)
    )
    adjusted_error = (
        merged["adjusted_shape_residual_eur_mwh"] - merged["spot_shape_residual_eur_mwh"]
        if not merged.empty
        else pd.Series(dtype=float)
    )
    no_temporal_leak = bool(train.empty or eval_frame.empty or train.index.max() < eval_frame.index.min())
    row = {
        "cutoff_utc": str(cutoff),
        "train_start_utc": str(train_start),
        "train_end_exclusive_utc": str(cutoff),
        "eval_start_utc": str(eval_start),
        "eval_end_exclusive_utc": str(eval_end),
        "valuation_timestamp_utc": str(valuation_utc),
        "train_hours": int(len(train)),
        "eval_hours": int(len(eval_frame)),
        "common_profile_cells": int(len(merged)),
        "train_eval_common_profile_cells": int(len(train_eval)),
        "no_temporal_leak": no_temporal_leak,
        "eval_after_current_valuation": bool(eval_start >= valuation_utc),
        "historical_fold_not_independent_of_current_candidate_fit": bool(eval_end <= valuation_utc),
        "baseline_profile_mae_eur_mwh": _mae(baseline_error),
        "adjusted_profile_mae_eur_mwh": _mae(adjusted_error),
        "profile_mae_improvement_eur_mwh": _mae(baseline_error) - _mae(adjusted_error),
        "baseline_profile_rmse_eur_mwh": _rmse(baseline_error),
        "adjusted_profile_rmse_eur_mwh": _rmse(adjusted_error),
        "profile_rmse_improvement_eur_mwh": _rmse(baseline_error) - _rmse(adjusted_error),
        "baseline_profile_correlation": _corr(merged["baseline_shape_residual_eur_mwh"], merged["spot_shape_residual_eur_mwh"])
        if not merged.empty
        else np.nan,
        "adjusted_profile_correlation": _corr(merged["adjusted_shape_residual_eur_mwh"], merged["spot_shape_residual_eur_mwh"])
        if not merged.empty
        else np.nan,
        "train_eval_profile_mae_eur_mwh": _mae(train_eval_diff),
        "train_eval_profile_correlation": _corr(
            train_eval["spot_shape_residual_eur_mwh_train"],
            train_eval["spot_shape_residual_eur_mwh_eval"],
        )
        if not train_eval.empty
        else np.nan,
    }
    row["fold_eligible"] = bool(
        row["no_temporal_leak"]
        and row["eval_hours"] >= min_eval_hours
        and row["common_profile_cells"] > 0
        and np.isfinite(row["baseline_profile_mae_eur_mwh"])
        and np.isfinite(row["adjusted_profile_mae_eur_mwh"])
    )
    return row


def _fold_bucket_rows(
    spot: pd.DataFrame,
    candidate_profiles: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    eval_days: int,
    embargo_days: int,
) -> list[dict[str, Any]]:
    cutoff = _to_utc_timestamp(cutoff)
    eval_start = cutoff + pd.Timedelta(days=int(embargo_days))
    eval_end = eval_start + pd.Timedelta(days=int(eval_days))
    eval_frame = spot[(spot.index >= eval_start) & (spot.index < eval_end)]
    merged = _timestamp_spot_candidate_residuals(eval_frame, candidate_profiles)
    if merged.empty:
        return []
    rows = []
    for bucket, mask in _bucket_masks(merged).items():
        rows.append(_bucket_metric_row(cutoff, bucket, "residual_level", merged.loc[mask]))
    ramp = merged.sort_index().copy()
    ramp["spot_ramp_eur_mwh"] = ramp["spot_residual_eur_mwh"].diff()
    ramp["baseline_ramp_eur_mwh"] = ramp["baseline_shape_residual_eur_mwh"].diff()
    ramp["adjusted_ramp_eur_mwh"] = ramp["adjusted_shape_residual_eur_mwh"].diff()
    ramp = ramp.dropna(
        subset=[
            "spot_ramp_eur_mwh",
            "baseline_ramp_eur_mwh",
            "adjusted_ramp_eur_mwh",
        ]
    )
    if not ramp.empty:
        rows.append(_bucket_metric_row(cutoff, "all", "hourly_ramp", ramp))
    return rows


def _timestamp_spot_candidate_residuals(
    eval_frame: pd.DataFrame,
    candidate_profiles: pd.DataFrame,
) -> pd.DataFrame:
    if eval_frame.empty:
        return pd.DataFrame()
    local = eval_frame.index.tz_convert("Europe/Zurich")
    frame = pd.DataFrame(
        {
            SPOT_PRICE: eval_frame[SPOT_PRICE].to_numpy(dtype=float),
            "month": local.month.astype(int),
            "hour": local.hour.astype(int),
            "weekday": local.weekday.astype(int),
            "is_weekend": (local.weekday >= 5).astype(bool),
            "year_month": pd.PeriodIndex(local.strftime("%Y-%m"), freq="M").astype(str),
        },
        index=eval_frame.index,
    )
    frame["spot_residual_eur_mwh"] = frame[SPOT_PRICE] - frame.groupby("year_month")[SPOT_PRICE].transform("mean")
    frame["is_peak_like"] = (frame["weekday"] < 5) & frame["hour"].between(8, 19)
    frame["is_solar_tail"] = frame["month"].between(3, 10) & frame["hour"].between(10, 16)
    frame["is_midday"] = frame["hour"].between(11, 15)
    frame["is_evening_ramp"] = frame["hour"].between(17, 21)
    frame["is_night"] = frame["hour"].between(0, 5)
    merged = frame.reset_index(names="timestamp_utc").merge(
        candidate_profiles,
        on=PROFILE_KEYS,
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        return pd.DataFrame()
    merged = merged.set_index(pd.DatetimeIndex(merged["timestamp_utc"])).sort_index()
    return merged


def _bucket_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "all": pd.Series(True, index=frame.index),
        "weekday": ~frame["is_weekend"].astype(bool),
        "weekend": frame["is_weekend"].astype(bool),
        "peak_like_weekday_08_19": frame["is_peak_like"].astype(bool),
        "offpeak_like": ~frame["is_peak_like"].astype(bool),
        "solar_tail_mar_oct_10_16": frame["is_solar_tail"].astype(bool),
        "midday_11_15": frame["is_midday"].astype(bool),
        "evening_ramp_17_21": frame["is_evening_ramp"].astype(bool),
        "night_00_05": frame["is_night"].astype(bool),
    }


def _bucket_metric_row(
    cutoff: pd.Timestamp,
    bucket: str,
    metric_type: str,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    if metric_type == "hourly_ramp":
        baseline_error = frame["baseline_ramp_eur_mwh"] - frame["spot_ramp_eur_mwh"]
        adjusted_error = frame["adjusted_ramp_eur_mwh"] - frame["spot_ramp_eur_mwh"]
    else:
        baseline_error = frame["baseline_shape_residual_eur_mwh"] - frame["spot_residual_eur_mwh"]
        adjusted_error = frame["adjusted_shape_residual_eur_mwh"] - frame["spot_residual_eur_mwh"]
    return {
        "cutoff_utc": str(_to_utc_timestamp(cutoff)),
        "bucket": bucket,
        "metric_type": metric_type,
        "hours": int(len(frame)),
        "baseline_mae_eur_mwh": _mae(baseline_error),
        "adjusted_mae_eur_mwh": _mae(adjusted_error),
        "mae_improvement_eur_mwh": _mae(baseline_error) - _mae(adjusted_error),
        "baseline_rmse_eur_mwh": _rmse(baseline_error),
        "adjusted_rmse_eur_mwh": _rmse(adjusted_error),
        "rmse_improvement_eur_mwh": _rmse(baseline_error) - _rmse(adjusted_error),
        "baseline_error_p95_eur_mwh": _abs_quantile(baseline_error, 0.95),
        "adjusted_error_p95_eur_mwh": _abs_quantile(adjusted_error, 0.95),
    }


def _post_valuation_overlap(
    joined: pd.DataFrame,
    spot: pd.DataFrame,
    *,
    valuation_utc: pd.Timestamp,
    embargo_days: int,
    evaluation_start_utc: pd.Timestamp | None = None,
    evaluation_end_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    eval_start = valuation_utc + pd.Timedelta(days=int(embargo_days))
    if evaluation_start_utc is not None:
        eval_start = max(eval_start, evaluation_start_utc)
    bounded_spot = _spot_evaluation_window(
        spot,
        start_utc=eval_start,
        end_utc=evaluation_end_utc,
    )
    candidate = joined.copy()
    candidate.index = candidate["timestamp_utc"]
    overlap = candidate.join(bounded_spot, how="inner")
    if overlap.empty:
        return pd.DataFrame(
            columns=[
                "timestamp_utc",
                "timestamp_ch",
                "spot_residual_eur_mwh",
                "baseline_residual_eur_mwh",
                "adjusted_residual_eur_mwh",
                "baseline_abs_error_eur_mwh",
                "adjusted_abs_error_eur_mwh",
            ]
        )
    local_month = overlap.index.tz_convert("Europe/Zurich").strftime("%Y-%m")
    overlap["spot_residual_eur_mwh"] = overlap[SPOT_PRICE] - overlap.groupby(local_month)[SPOT_PRICE].transform(
        "mean"
    )
    for suffix in ("baseline", "adjusted"):
        price_column = f"{PRICE}_{suffix}"
        overlap[f"evaluation_residual_{suffix}_eur_mwh"] = (
            overlap[price_column]
            - overlap.groupby(local_month)[price_column].transform("mean")
        )
    overlap["baseline_abs_error_eur_mwh"] = (
        overlap["evaluation_residual_baseline_eur_mwh"]
        - overlap["spot_residual_eur_mwh"]
    ).abs()
    overlap["adjusted_abs_error_eur_mwh"] = (
        overlap["evaluation_residual_adjusted_eur_mwh"]
        - overlap["spot_residual_eur_mwh"]
    ).abs()
    out = pd.DataFrame(
        {
            "timestamp_utc": overlap.index.astype(str),
            "timestamp_ch": overlap.index.tz_convert("Europe/Zurich").astype(str),
            "spot_residual_eur_mwh": overlap["spot_residual_eur_mwh"].to_numpy(dtype=float),
            "baseline_residual_eur_mwh": overlap[
                "evaluation_residual_baseline_eur_mwh"
            ].to_numpy(dtype=float),
            "adjusted_residual_eur_mwh": overlap[
                "evaluation_residual_adjusted_eur_mwh"
            ].to_numpy(dtype=float),
            "baseline_abs_error_eur_mwh": overlap["baseline_abs_error_eur_mwh"].to_numpy(dtype=float),
            "adjusted_abs_error_eur_mwh": overlap["adjusted_abs_error_eur_mwh"].to_numpy(dtype=float),
        }
    )
    return out


def _strict_lab_checks(
    joined: pd.DataFrame,
    folds: pd.DataFrame,
    post_valuation: pd.DataFrame,
    *,
    min_eval_hours: int,
    evaluation_start_utc: pd.Timestamp | None,
    evaluation_end_utc: pd.Timestamp | None,
) -> dict[str, bool]:
    quantile_order = (
        (joined["structural_p10_eur_mwh_adjusted"] <= joined["structural_p50_eur_mwh_adjusted"])
        & (joined["structural_p50_eur_mwh_adjusted"] <= joined["structural_p90_eur_mwh_adjusted"])
    ).all()
    monthly_drift = joined.groupby("year_month")["weighted_delta_eur_mwh"].mean().abs().max()
    finite_adjusted = np.isfinite(
        joined[[f"{column}_adjusted" for column in PRICE_COLUMNS]].to_numpy(dtype=float)
    ).all()
    no_temporal_leak = bool(not folds.empty and folds["no_temporal_leak"].astype(bool).all())
    fold_cutoffs = (
        pd.DatetimeIndex([_to_utc_timestamp(value) for value in folds["cutoff_utc"]])
        if not folds.empty and "cutoff_utc" in folds
        else pd.DatetimeIndex([])
    )
    eval_starts = (
        pd.DatetimeIndex([_to_utc_timestamp(value) for value in folds["eval_start_utc"]])
        if not folds.empty and "eval_start_utc" in folds
        else pd.DatetimeIndex([])
    )
    eval_ends = (
        pd.DatetimeIndex([_to_utc_timestamp(value) for value in folds["eval_end_exclusive_utc"]])
        if not folds.empty and "eval_end_exclusive_utc" in folds
        else pd.DatetimeIndex([])
    )
    unique_cutoffs = bool(
        len(fold_cutoffs) > 0
        and not fold_cutoffs.isna().any()
        and not fold_cutoffs.duplicated().any()
        and fold_cutoffs.is_monotonic_increasing
    )
    non_overlapping_evaluations = bool(
        len(eval_starts) > 0
        and len(eval_starts) == len(eval_ends)
        and eval_starts.is_monotonic_increasing
        and all(eval_starts[index] >= eval_ends[index - 1] for index in range(1, len(eval_starts)))
    )
    eligible_folds = bool(not folds.empty and int(folds["fold_eligible"].sum()) > 0)
    locked_window = evaluation_start_utc is not None and evaluation_end_utc is not None
    post_timestamps = (
        pd.to_datetime(post_valuation["timestamp_utc"], utc=True)
        if not post_valuation.empty
        else pd.DatetimeIndex([])
    )
    evaluation_window_isolated = bool(
        not locked_window
        or (
            len(post_timestamps) > 0
            and (post_timestamps >= evaluation_start_utc).all()
            and (post_timestamps < evaluation_end_utc).all()
            and not post_timestamps.duplicated().any()
        )
    )
    return {
        "finite_adjusted_ok": bool(finite_adjusted),
        "quantile_order_adjusted_ok": bool(quantile_order),
        "monthly_weighted_mean_drift_ok": bool(float(monthly_drift) <= 1e-5),
        "implied_width_drift_ok": bool(float(joined["implied_width_delta_eur_mwh"].abs().max()) <= 1e-9),
        "weighted_negative_hours_ok": bool(int((joined[f"{PRICE}_adjusted"] < 0.0).sum()) == 0),
        "rolling_folds_present": eligible_folds,
        "rolling_folds_no_temporal_leak": no_temporal_leak,
        "rolling_folds_unique_ordered_cutoffs": unique_cutoffs,
        "rolling_folds_non_overlapping_evaluations": non_overlapping_evaluations,
        "post_valuation_overlap_recorded": bool(
            len(post_valuation) >= min_eval_hours if locked_window else True
        ),
        "evaluation_window_isolated_before_centering": evaluation_window_isolated,
        "min_eval_hours_positive": bool(min_eval_hours > 0),
    }


def _validated_evaluation_window(
    *,
    evaluation_start_utc: str | None,
    evaluation_end_utc: str | None,
    valuation_utc: pd.Timestamp,
    embargo_days: int,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if (evaluation_start_utc is None) != (evaluation_end_utc is None):
        raise ValueError("evaluation_start_utc and evaluation_end_utc must be provided together")
    if evaluation_start_utc is None:
        return None, None
    start = _to_utc_timestamp(evaluation_start_utc)
    end = _to_utc_timestamp(evaluation_end_utc)
    if end <= start:
        raise ValueError("evaluation_end_utc must be after evaluation_start_utc")
    minimum_start = valuation_utc + pd.Timedelta(days=int(embargo_days))
    if start < minimum_start:
        raise ValueError("evaluation_start_utc must respect the post-valuation embargo")
    return start, end


def _spot_evaluation_window(
    spot: pd.DataFrame,
    *,
    start_utc: pd.Timestamp | None,
    end_utc: pd.Timestamp | None,
) -> pd.DataFrame:
    if start_utc is None and end_utc is None:
        return spot
    mask = pd.Series(True, index=spot.index)
    if start_utc is not None:
        mask &= spot.index >= start_utc
    if end_utc is not None:
        mask &= spot.index < end_utc
    return spot.loc[mask.to_numpy()].copy()


def _utc_text_or_none(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _spot_profile(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*PROFILE_KEYS, "spot_shape_residual_eur_mwh", "spot_hours"])
    local = frame.index.tz_convert("Europe/Zurich")
    out = pd.DataFrame(
        {
            SPOT_PRICE: frame[SPOT_PRICE].to_numpy(dtype=float),
            "month": local.month.astype(int),
            "hour": local.hour.astype(int),
            "is_weekend": (local.weekday >= 5).astype(bool),
            "year_month": pd.PeriodIndex(local.strftime("%Y-%m"), freq="M").astype(str),
        },
        index=frame.index,
    )
    out["spot_residual"] = out[SPOT_PRICE] - out.groupby("year_month")[SPOT_PRICE].transform("mean")
    grouped = out.groupby(PROFILE_KEYS, sort=True)
    return grouped["spot_residual"].agg(spot_shape_residual_eur_mwh="mean", spot_hours="size").reset_index()


def _add_local_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    idx = out.index
    out["year_month"] = pd.PeriodIndex(idx.strftime("%Y-%m"), freq="M").astype(str)
    out["month"] = idx.month.astype(int)
    out["hour"] = idx.hour.astype(int)
    out["is_weekend"] = idx.weekday >= 5
    return out


def _rolling_cutoffs(
    spot: pd.DataFrame,
    *,
    valuation_utc: pd.Timestamp,
    rolling_cutoffs: list[str] | None,
    lookback_years: int,
    eval_days: int,
    embargo_days: int,
    max_auto_folds: int,
) -> list[pd.Timestamp]:
    if max_auto_folds <= 0 or max_auto_folds > _MAX_ROLLING_FOLDS:
        raise ValueError(f"max_auto_folds must be between 1 and {_MAX_ROLLING_FOLDS}")
    if rolling_cutoffs:
        explicit = [_to_utc_timestamp(value) for value in rolling_cutoffs]
        if len(explicit) > _MAX_ROLLING_FOLDS:
            raise ValueError(f"rolling_cutoffs cannot exceed {_MAX_ROLLING_FOLDS} origins")
        if len(set(explicit)) != len(explicit):
            raise ValueError("rolling_cutoffs must be unique")
        if explicit != sorted(explicit):
            raise ValueError("rolling_cutoffs must be strictly increasing")
        return explicit
    min_cutoff = spot.index.min() + pd.DateOffset(years=int(lookback_years))
    max_cutoff = spot.index.max() - pd.Timedelta(days=int(eval_days + embargo_days))
    max_cutoff = min(max_cutoff, valuation_utc - pd.Timedelta(days=int(eval_days + embargo_days)))
    if min_cutoff > max_cutoff:
        return []
    first_daily_cutoff = min_cutoff.ceil("D")
    last_daily_cutoff = max_cutoff.floor("D")
    if first_daily_cutoff > last_daily_cutoff:
        return []
    candidates = pd.date_range(
        first_daily_cutoff,
        last_daily_cutoff,
        freq=pd.Timedelta(days=int(eval_days)),
    )
    if len(candidates) > int(max_auto_folds):
        positions = np.linspace(0, len(candidates) - 1, num=int(max_auto_folds))
        positions = np.unique(np.rint(positions).astype(int))
        candidates = candidates[positions]
    result = [_to_utc_timestamp(cutoff) for cutoff in candidates]
    if len(set(result)) != len(result):
        raise RuntimeError("automatic rolling cutoffs are not unique")
    return result


def _aggregate_fold_metrics(folds: pd.DataFrame) -> dict[str, Any]:
    if folds.empty:
        return {}
    numeric = folds.copy()
    return {
        "mean_baseline_profile_mae_eur_mwh": _mean(numeric["baseline_profile_mae_eur_mwh"]),
        "mean_adjusted_profile_mae_eur_mwh": _mean(numeric["adjusted_profile_mae_eur_mwh"]),
        "mean_profile_mae_improvement_eur_mwh": _mean(numeric["profile_mae_improvement_eur_mwh"]),
        "median_profile_mae_improvement_eur_mwh": _median(numeric["profile_mae_improvement_eur_mwh"]),
        "positive_mae_improvement_fold_count": int((numeric["profile_mae_improvement_eur_mwh"] > 0.0).sum()),
        "mean_baseline_profile_correlation": _mean(numeric["baseline_profile_correlation"]),
        "mean_adjusted_profile_correlation": _mean(numeric["adjusted_profile_correlation"]),
        "mean_train_eval_profile_mae_eur_mwh": _mean(numeric["train_eval_profile_mae_eur_mwh"]),
        "historical_fold_not_independent_count": int(
            numeric["historical_fold_not_independent_of_current_candidate_fit"].astype(bool).sum()
        ),
        "post_valuation_fold_count": int(numeric["eval_after_current_valuation"].astype(bool).sum()),
    }


def _aggregate_bucket_metrics(buckets: pd.DataFrame) -> dict[str, Any]:
    if buckets.empty:
        return {}
    out: dict[str, Any] = {}
    for (bucket, metric_type), group in buckets.groupby(["bucket", "metric_type"], sort=True):
        key = f"{metric_type}:{bucket}"
        out[key] = {
            "fold_count": int(len(group)),
            "total_hours": int(group["hours"].sum()),
            "mean_baseline_mae_eur_mwh": _mean(group["baseline_mae_eur_mwh"]),
            "mean_adjusted_mae_eur_mwh": _mean(group["adjusted_mae_eur_mwh"]),
            "mean_mae_improvement_eur_mwh": _mean(group["mae_improvement_eur_mwh"]),
            "positive_mae_improvement_fold_count": int((group["mae_improvement_eur_mwh"] > 0.0).sum()),
            "mean_baseline_rmse_eur_mwh": _mean(group["baseline_rmse_eur_mwh"]),
            "mean_adjusted_rmse_eur_mwh": _mean(group["adjusted_rmse_eur_mwh"]),
            "mean_rmse_improvement_eur_mwh": _mean(group["rmse_improvement_eur_mwh"]),
            "mean_adjusted_error_p95_eur_mwh": _mean(group["adjusted_error_p95_eur_mwh"]),
        }
    return out


def _post_valuation_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"hours": 0}
    return {
        "hours": int(len(frame)),
        "baseline_residual_mae_eur_mwh": _mean(frame["baseline_abs_error_eur_mwh"]),
        "adjusted_residual_mae_eur_mwh": _mean(frame["adjusted_abs_error_eur_mwh"]),
        "residual_mae_improvement_eur_mwh": _mean(frame["baseline_abs_error_eur_mwh"])
        - _mean(frame["adjusted_abs_error_eur_mwh"]),
    }


def _to_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mae(values: pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(np.abs(arr)))


def _rmse(values: pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(arr))))


def _abs_quantile(values: pd.Series, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = np.abs(arr[np.isfinite(arr)])
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _corr(left: pd.Series, right: pd.Series) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    if float(np.std(x[mask])) == 0.0 or float(np.std(y[mask])) == 0.0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _mean(values: pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _median(values: pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--adjusted-csv", type=Path, required=True)
    parser.add_argument("--spot-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--valuation-timestamp", required=True)
    parser.add_argument("--rolling-cutoff", action="append", dest="rolling_cutoffs")
    parser.add_argument("--lookback-years", type=int, default=2)
    parser.add_argument("--eval-days", type=int, default=30)
    parser.add_argument("--embargo-days", type=int, default=1)
    parser.add_argument("--max-auto-folds", type=int, default=12)
    parser.add_argument("--min-eval-hours", type=int, default=24)
    parser.add_argument("--evaluation-start-utc")
    parser.add_argument("--evaluation-end-utc")
    args = parser.parse_args(argv)
    summary = backtest_against_spot(
        baseline_csv=args.baseline_csv,
        adjusted_csv=args.adjusted_csv,
        spot_parquet=args.spot_parquet,
        output_dir=args.output_dir,
        valuation_timestamp=args.valuation_timestamp,
        rolling_cutoffs=args.rolling_cutoffs,
        lookback_years=args.lookback_years,
        eval_days=args.eval_days,
        embargo_days=args.embargo_days,
        max_auto_folds=args.max_auto_folds,
        min_eval_hours=args.min_eval_hours,
        evaluation_start_utc=args.evaluation_start_utc,
        evaluation_end_utc=args.evaluation_end_utc,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["status"] == "DIAGNOSTIC_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

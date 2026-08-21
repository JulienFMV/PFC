"""
quality_gate.py
---------------
Production quality checks for the PFC pipeline.

Hard-fails on critical data quality issues to avoid publishing corrupted outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class QualityGateError(RuntimeError):
    """Raised when a critical quality gate fails."""


@dataclass
class QualityReport:
    dataset: str
    checks_passed: int
    warnings: list[str]
    errors: list[str]
    metrics: dict[str, object] = field(default_factory=dict)


def input_grid_errors(
    df: pd.DataFrame,
    *,
    name: str,
    expected_cadence_seconds: int,
    expected_phase_offset_seconds: int = 0,
    min_grid_coverage: float = 0.95,
) -> list[str]:
    """Return fail-closed cadence and absolute UTC phase violations."""

    try:
        index = pd.DatetimeIndex(df.index)
    except (TypeError, ValueError):
        return [f"{name}.index_not_datetime"]
    if index.tz is None:
        return [f"{name}.index_timezone_naive"]
    index = index.tz_convert("UTC").sort_values().unique()
    if index.hasnans:
        return [f"{name}.index_contains_nat"]
    if len(index) < 2:
        return [f"{name}.cadence_insufficient_rows"]
    cadence_seconds = int(expected_cadence_seconds)
    elapsed_seconds = (index[-1] - index[0]).total_seconds()
    expected_rows = int(math.floor(elapsed_seconds / cadence_seconds)) + 1
    coverage = len(index) / expected_rows if expected_rows > 0 else 0.0
    errors: list[str] = []
    if coverage < float(min_grid_coverage):
        errors.append(f"{name}.grid_coverage={coverage:.6f}")
    cadence_ns = cadence_seconds * 1_000_000_000
    phase_ns = int(expected_phase_offset_seconds) * 1_000_000_000
    if phase_ns < 0 or phase_ns >= cadence_ns:
        errors.append(f"{name}.grid_phase_policy_invalid")
    elif any((int(timestamp.value) - phase_ns) % cadence_ns != 0 for timestamp in index):
        errors.append(f"{name}.grid_phase_mismatch")
    deltas = pd.Series(index[1:] - index[:-1]).dt.total_seconds().to_numpy()
    if any(
        abs((float(delta) / cadence_seconds) - round(float(delta) / cadence_seconds)) > 1e-9
        for delta in deltas
    ):
        errors.append(f"{name}.off_grid_timestamps")
    return errors


def weekly_local_grid_errors(
    df: pd.DataFrame,
    *,
    name: str,
    timezone: str,
    weekday: int,
    min_grid_coverage: float = 0.95,
) -> list[str]:
    """Validate one weekly civil-time grid without imposing a false UTC DST phase."""

    try:
        index = pd.DatetimeIndex(df.index)
    except (TypeError, ValueError):
        return [f"{name}.index_not_datetime"]
    if index.tz is None:
        return [f"{name}.index_timezone_naive"]
    index = index.tz_convert("UTC").sort_values().unique()
    if index.hasnans:
        return [f"{name}.index_contains_nat"]
    if len(index) < 2:
        return [f"{name}.cadence_insufficient_rows"]
    local = index.tz_convert(timezone)
    errors: list[str] = []
    if any(
        timestamp.weekday() != weekday
        or timestamp.hour != 0
        or timestamp.minute != 0
        or timestamp.second != 0
        or timestamp.microsecond != 0
        for timestamp in local
    ):
        errors.append(f"{name}.local_grid_phase_mismatch")
    local_naive = local.tz_localize(None)
    elapsed_days = (local_naive[-1] - local_naive[0]).days
    expected_rows = elapsed_days // 7 + 1
    coverage = len(local_naive) / expected_rows if expected_rows > 0 else 0.0
    if coverage < float(min_grid_coverage):
        errors.append(f"{name}.grid_coverage={coverage:.6f}")
    deltas = local_naive[1:] - local_naive[:-1]
    if any(delta % pd.Timedelta(days=7) != pd.Timedelta(0) for delta in deltas):
        errors.append(f"{name}.off_grid_timestamps")
    return errors


def _is_tz_aware_index(df: pd.DataFrame) -> bool:
    return isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None


def validate_input_frame(
    df: pd.DataFrame | None,
    *,
    name: str,
    required_columns: list[str],
    min_rows: int,
    max_age_days: float,
    reference_timestamp: str | pd.Timestamp | None = None,
    fail_on_stale: bool = False,
    min_finite_fraction: float = 0.0,
    recent_window_days: float | None = None,
    min_recent_finite_fraction: float = 0.0,
    max_finite_gap_days: float | None = None,
) -> QualityReport:
    warnings: list[str] = []
    errors: list[str] = []
    checks_passed = 0
    metrics: dict[str, object] = {}

    if df is None:
        raise QualityGateError(f"{name}: dataframe is None")
    if df.empty:
        raise QualityGateError(f"{name}: dataframe is empty")
    checks_passed += 1

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise QualityGateError(f"{name}: missing required columns {missing}")
    checks_passed += 1

    if len(df) < min_rows:
        raise QualityGateError(f"{name}: not enough rows ({len(df)} < {min_rows})")
    checks_passed += 1

    if not _is_tz_aware_index(df):
        raise QualityGateError(f"{name}: DatetimeIndex must be timezone-aware")
    checks_passed += 1

    if df.index.has_duplicates:
        errors.append(f"{name}: DatetimeIndex contains duplicate timestamps")
    else:
        checks_passed += 1

    if not df.index.is_monotonic_increasing:
        errors.append(f"{name}: index is not monotonic increasing")
    else:
        checks_passed += 1

    latest_ts = df.index.max()
    reference = pd.Timestamp(reference_timestamp) if reference_timestamp is not None else pd.Timestamp.now("UTC")
    if reference.tzinfo is None:
        reference = reference.tz_localize("UTC")
    else:
        reference = reference.tz_convert("UTC")
    latest_utc = latest_ts.tz_convert("UTC")
    if latest_utc > reference:
        errors.append(
            f"{name}: latest point {latest_utc.isoformat()} is after reference timestamp "
            f"{reference.isoformat()}"
        )
    for column in required_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        finite_mask = pd.Series(np.isfinite(values.to_numpy(dtype=float)), index=df.index)
        finite_fraction = float(finite_mask.mean())
        metrics[f"{column}.finite_fraction"] = finite_fraction
        if finite_fraction < float(min_finite_fraction):
            errors.append(
                f"{name}.{column}: finite coverage is too low "
                f"({finite_fraction:.4f} < {float(min_finite_fraction):.4f})"
            )
        else:
            checks_passed += 1
        if not finite_mask.any():
            errors.append(f"{name}.{column}: no finite observations")
            continue
        finite_index_utc = df.index[finite_mask.to_numpy()].tz_convert("UTC")
        if recent_window_days is not None:
            window_start = reference - pd.Timedelta(days=float(recent_window_days))
            index_utc = df.index.tz_convert("UTC")
            recent_rows = (index_utc >= window_start) & (index_utc <= reference)
            recent_fraction = (
                float(finite_mask.to_numpy()[recent_rows].mean())
                if bool(recent_rows.any())
                else 0.0
            )
            metrics[f"{column}.recent_finite_fraction"] = recent_fraction
            metrics[f"{column}.recent_window_days"] = float(recent_window_days)
            if recent_fraction < float(min_recent_finite_fraction):
                errors.append(
                    f"{name}.{column}: recent finite coverage is too low "
                    f"({recent_fraction:.4f} < {float(min_recent_finite_fraction):.4f})"
                )
            else:
                checks_passed += 1
        if max_finite_gap_days is not None:
            gap_window_start = (
                reference - pd.Timedelta(days=float(recent_window_days))
                if recent_window_days is not None
                else finite_index_utc.min()
            )
            recent_finite = finite_index_utc[
                (finite_index_utc >= gap_window_start) & (finite_index_utc <= reference)
            ]
            gap_points = list(recent_finite)
            max_gap_days = float("inf")
            if gap_points:
                boundaries = pd.DatetimeIndex([gap_window_start, *gap_points, reference]).sort_values()
                max_gap_days = float(boundaries.to_series().diff().dt.total_seconds().max() / 86400.0)
            metrics[f"{column}.max_finite_gap_days"] = max_gap_days
            if max_gap_days > float(max_finite_gap_days):
                errors.append(
                    f"{name}.{column}: finite observation gap is too large "
                    f"({max_gap_days:.3f} days > {float(max_finite_gap_days):.3f} days)"
                )
            else:
                checks_passed += 1
        latest_finite = df.index[finite_mask.to_numpy()].max().tz_convert("UTC")
        metrics[f"{column}.latest_finite_timestamp"] = latest_finite.isoformat()
        if latest_finite > reference:
            errors.append(
                f"{name}.{column}: latest finite point {latest_finite.isoformat()} "
                f"is after reference timestamp {reference.isoformat()}"
            )
            continue
        age_days = (reference - latest_finite).total_seconds() / 86400.0
        metrics[f"{column}.age_days"] = age_days
        if age_days > max_age_days:
            message = (
                f"{name}.{column}: latest finite point is stale "
                f"({age_days:.1f} days > {max_age_days} days)"
            )
            if fail_on_stale:
                errors.append(message)
            else:
                warnings.append(message)
        else:
            checks_passed += 1

    if errors:
        raise QualityGateError("; ".join(errors))
    return QualityReport(name, checks_passed, warnings, errors, metrics)


def validate_pfc_output(
    pfc_df: pd.DataFrame,
    *,
    expected_min_rows: int,
) -> QualityReport:
    warnings: list[str] = []
    errors: list[str] = []
    checks_passed = 0

    if pfc_df is None or pfc_df.empty:
        raise QualityGateError("PFC output: empty dataframe")
    checks_passed += 1

    required = ["price_shape", "profile_type", "confidence"]
    missing = [c for c in required if c not in pfc_df.columns]
    if missing:
        raise QualityGateError(f"PFC output: missing required columns {missing}")
    checks_passed += 1

    if len(pfc_df) < expected_min_rows:
        raise QualityGateError(f"PFC output: not enough rows ({len(pfc_df)} < {expected_min_rows})")
    checks_passed += 1

    if pfc_df["price_shape"].isna().any():
        errors.append("PFC output: NaN in price_shape")
    else:
        checks_passed += 1

    finite = pd.Series(pd.to_numeric(pfc_df["price_shape"], errors="coerce")).dropna()
    if finite.empty:
        raise QualityGateError("PFC output: no finite prices")
    if (finite.abs() > 10000).any():
        errors.append("PFC output: implausible absolute price above 10000 EUR/MWh")
    else:
        checks_passed += 1

    if "p10" in pfc_df.columns and "p90" in pfc_df.columns:
        bad_band = (pfc_df["p10"] > pfc_df["p90"]).sum()
        if bad_band > 0:
            errors.append(f"PFC output: {bad_band} rows with p10 > p90")
        else:
            checks_passed += 1

    if "confidence" in pfc_df.columns:
        c = pd.to_numeric(pfc_df["confidence"], errors="coerce")
        if ((c < 0) | (c > 1)).any():
            warnings.append("PFC output: confidence out of [0,1] on some rows")
        else:
            checks_passed += 1

    if errors:
        raise QualityGateError("; ".join(errors))
    return QualityReport("PFC output", checks_passed, warnings, errors)

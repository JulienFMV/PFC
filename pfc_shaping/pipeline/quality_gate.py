"""
quality_gate.py
---------------
Production quality checks for the PFC pipeline.

Hard-fails on critical data quality issues to avoid publishing corrupted outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class QualityGateError(RuntimeError):
    """Raised when a critical quality gate fails."""


@dataclass
class QualityReport:
    dataset: str
    checks_passed: int
    warnings: list[str]
    errors: list[str]


def _is_tz_aware_index(df: pd.DataFrame) -> bool:
    return isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None


def validate_input_frame(
    df: pd.DataFrame | None,
    *,
    name: str,
    required_columns: list[str],
    min_rows: int,
    max_age_days: int,
) -> QualityReport:
    warnings: list[str] = []
    errors: list[str] = []
    checks_passed = 0

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

    if not df.index.is_monotonic_increasing:
        errors.append(f"{name}: index is not monotonic increasing")
    else:
        checks_passed += 1

    latest_ts = df.index.max()
    now = pd.Timestamp.utcnow()
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    age_days = (now - latest_ts.tz_convert("UTC")).total_seconds() / 86400.0
    if age_days > max_age_days:
        warnings.append(f"{name}: latest point is stale ({age_days:.1f} days)")
    else:
        checks_passed += 1

    if errors:
        raise QualityGateError("; ".join(errors))
    return QualityReport(name, checks_passed, warnings, errors)


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

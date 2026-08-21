"""Dormant solver-neutral contract for future EEX short-tenor shape signals.

The module performs no quote extraction, fitting, activation, or production
assembly.  It only validates a pre-built additive signal and projects it into
the nullspace of the active CH BASE/PEAK/OFFPEAK constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

import numpy as np
import pandas as pd

from pfc_shaping.lt.model.shape_constraints import (
    build_base_peak_offpeak_constraint_system,
    interval_hours,
    validate_utc_index,
)


SHORT_TENOR_SHAPE_CONTRACT_VERSION = "fmv-eex-ch-short-tenor-shape-contract-v1"
SHORT_TENOR_SHAPE_PROJECTION_SCHEMA_VERSION = "fmv_eex_ch_short_tenor_shape_projection.v1"
LOCAL_TIMEZONE = "Europe/Zurich"
MAX_ABS_RESIDUAL_EUR_MWH = 1e-9
MAX_BLOCK_DISPERSION_EUR_MWH = 1e-10
_CONTENT_ID_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ShortTenorShapeProjection:
    """Validated additive delta and non-price-bearing audit evidence."""

    delta: pd.Series
    constraint_residuals: pd.DataFrame
    monthly_residuals: pd.DataFrame
    audit: Mapping[str, object]


def project_short_tenor_shape_delta(
    raw_delta: pd.Series | np.ndarray,
    *,
    monthly_base_levels: Mapping[str, float],
    source_normalization_content_id: str,
    source_audit_content_id: str,
    valuation_timestamp_utc: str | pd.Timestamp,
    index: pd.DatetimeIndex | None = None,
    peak_forward_prices: Mapping[str, float] | None = None,
    country: str = "CH",
) -> ShortTenorShapeProjection:
    """Validate and project a dormant DAY/WEEK/WEEKEND additive shape signal.

    The supplied monthly levels are used only to construct the same active
    constraint rows as the solver.  Their numeric targets are deliberately
    absent from the returned audit objects.
    """

    if str(country).upper() != "CH":
        raise ValueError("short-tenor shape contract currently supports CH only")
    _validate_content_id(source_normalization_content_id, "source_normalization_content_id")
    _validate_content_id(source_audit_content_id, "source_audit_content_id")

    values, utc = _coerce_signal_and_index(raw_delta, index=index)
    hours = interval_hours(utc)
    cadence_hours = float(hours[0])
    cadence = "1h" if np.isclose(cadence_hours, 1.0) else "15min"
    local = utc.tz_convert(LOCAL_TIMEZONE)
    month_keys = _require_complete_local_months(utc, cadence_hours=cadence_hours)
    base_levels = _validate_monthly_levels(monthly_base_levels, month_keys)
    peak_levels = _validate_peak_levels(peak_forward_prices or {})
    valuation = _validate_valuation_timestamp(valuation_timestamp_utc, first_delivery=utc[0])

    raw_block_dispersion = _maximum_short_block_dispersion(values, local)
    if raw_block_dispersion > MAX_BLOCK_DISPERSION_EUR_MWH:
        raise ValueError(
            "raw short-tenor signal contains unauthorized within-day/block structure: "
            f"maximum dispersion={raw_block_dispersion:.12g} EUR/MWh"
        )

    system = build_base_peak_offpeak_constraint_system(
        utc,
        base_levels,
        peak_levels,
        country="CH",
    )
    matrix = system.matrix
    projected = _project_to_nullspace(values, matrix)
    constraint_values = matrix @ projected if matrix.size else np.zeros(0, dtype=float)
    max_constraint_residual = (
        float(np.max(np.abs(constraint_values))) if len(constraint_values) else 0.0
    )
    if max_constraint_residual > MAX_ABS_RESIDUAL_EUR_MWH:
        raise RuntimeError(
            "short-tenor nullspace projection failed its constraint tolerance: "
            f"{max_constraint_residual:.12g} EUR/MWh"
        )

    projected_block_dispersion = _maximum_short_block_dispersion(projected, local)
    if projected_block_dispersion > MAX_BLOCK_DISPERSION_EUR_MWH:
        raise RuntimeError(
            "constraint projection introduced unauthorized within-day/block structure: "
            f"maximum dispersion={projected_block_dispersion:.12g} EUR/MWh"
        )

    monthly_residuals = _monthly_weighted_residuals(projected, local, hours)
    max_monthly_residual = float(monthly_residuals["abs_error"].max())
    if max_monthly_residual > MAX_ABS_RESIDUAL_EUR_MWH:
        raise RuntimeError(
            "short-tenor projection rewrote a solver-authoritative monthly mean: "
            f"{max_monthly_residual:.12g} EUR/MWh"
        )

    constraint_residuals = pd.DataFrame(
        {
            "name": system.names,
            "kind": [row.kind for row in system.rows],
            "achieved_delta_mean_eur_mwh": constraint_values,
            "abs_error": np.abs(constraint_values),
            "delivery_hours": [float(row.metadata.get("hours", np.nan)) for row in system.rows],
        }
    )
    raw_norm = float(np.linalg.norm(values))
    projected_norm = float(np.linalg.norm(projected))
    audit: dict[str, object] = {
        "schema_version": SHORT_TENOR_SHAPE_PROJECTION_SCHEMA_VERSION,
        "contract_version": SHORT_TENOR_SHAPE_CONTRACT_VERSION,
        "status": "PASS_LOCAL_MATHEMATICAL_CONTRACT_ONLY_NO_MODEL_AUTHORITY",
        "country": "CH",
        "delivery_timezone": LOCAL_TIMEZONE,
        "cadence": cadence,
        "interval_count": int(len(utc)),
        "local_months": list(month_keys),
        "constraint_count": int(len(system.rows)),
        "constraint_kinds": sorted({row.kind for row in system.rows}),
        "max_abs_raw_delta_eur_mwh": float(np.max(np.abs(values))),
        "max_abs_projected_delta_eur_mwh": float(np.max(np.abs(projected))),
        "l2_retention_ratio": projected_norm / raw_norm if raw_norm > 0.0 else 0.0,
        "max_raw_block_dispersion_eur_mwh": raw_block_dispersion,
        "max_projected_block_dispersion_eur_mwh": projected_block_dispersion,
        "max_constraint_residual_eur_mwh": max_constraint_residual,
        "max_monthly_mean_residual_eur_mwh": max_monthly_residual,
        "source_normalization_content_id": source_normalization_content_id,
        "source_audit_content_id": source_audit_content_id,
        "valuation_timestamp_utc": valuation.isoformat(),
        "point_in_time_availability_proven": False,
        "model_input_authorized": False,
        "candidate_assembly_authorized": False,
        "production_authorized": False,
        "ompex_used": False,
        "afry_numeric_values_used": False,
    }
    delta = pd.Series(projected, index=utc, name="short_tenor_shape_delta_eur_mwh")
    return ShortTenorShapeProjection(
        delta=delta,
        constraint_residuals=constraint_residuals,
        monthly_residuals=monthly_residuals,
        audit=audit,
    )


def _coerce_signal_and_index(
    raw_delta: pd.Series | np.ndarray,
    *,
    index: pd.DatetimeIndex | None,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    if isinstance(raw_delta, pd.Series):
        if not isinstance(raw_delta.index, pd.DatetimeIndex):
            raise TypeError("raw_delta Series must use a DatetimeIndex")
        series_utc = validate_utc_index(raw_delta.index)
        if index is not None:
            explicit_utc = validate_utc_index(index)
            if not series_utc.equals(explicit_utc):
                raise ValueError("raw_delta Series index does not match explicit delivery index")
        utc = series_utc
        values = raw_delta.to_numpy(dtype=float)
    else:
        if index is None:
            raise ValueError("index is required when raw_delta is not a Series")
        utc = validate_utc_index(index)
        values = np.asarray(raw_delta, dtype=float)
    if values.ndim != 1:
        raise ValueError("raw_delta must be one-dimensional")
    if len(values) != len(utc):
        raise ValueError(f"raw_delta length mismatch: {len(values)} != {len(utc)}")
    if len(values) < 2:
        raise ValueError("raw_delta must cover at least two delivery intervals")
    if not np.isfinite(values).all():
        raise ValueError("raw_delta must contain only finite values")
    return values.astype(float, copy=True), utc


def _require_complete_local_months(
    utc: pd.DatetimeIndex,
    *,
    cadence_hours: float,
) -> tuple[str, ...]:
    local = utc.tz_convert(LOCAL_TIMEZONE)
    first_key = str(local[0].strftime("%Y-%m"))
    last_key = str(local[-1].strftime("%Y-%m"))
    first_local = pd.Timestamp(f"{first_key}-01", tz=LOCAL_TIMEZONE)
    last_local = pd.Timestamp(f"{last_key}-01", tz=LOCAL_TIMEZONE) + pd.offsets.MonthBegin(1)
    frequency = "1h" if np.isclose(cadence_hours, 1.0) else "15min"
    expected = pd.date_range(
        first_local.tz_convert("UTC"),
        last_local.tz_convert("UTC"),
        freq=frequency,
        inclusive="left",
    )
    if not utc.equals(expected):
        raise ValueError("delivery index must contain contiguous complete local calendar months")
    return tuple(dict.fromkeys(local.strftime("%Y-%m").tolist()))


def _validate_monthly_levels(
    levels: Mapping[str, float],
    month_keys: tuple[str, ...],
) -> dict[str, float]:
    normalized = {str(key): float(value) for key, value in levels.items()}
    expected = set(month_keys)
    actual = set(normalized)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"monthly_base_levels must exactly cover delivered months; missing={missing}, extra={extra}")
    if not all(np.isfinite(value) for value in normalized.values()):
        raise ValueError("monthly_base_levels must contain only finite values")
    return normalized


def _validate_peak_levels(
    levels: Mapping[str, float],
) -> dict[str, float]:
    normalized = {str(key): float(value) for key, value in levels.items()}
    if not all(np.isfinite(value) for value in normalized.values()):
        raise ValueError("peak_forward_prices must contain only finite values")
    return normalized


def _validate_content_id(value: str, field: str) -> None:
    if not _CONTENT_ID_RE.fullmatch(str(value)):
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256 content id")


def _validate_valuation_timestamp(
    value: str | pd.Timestamp,
    *,
    first_delivery: pd.Timestamp,
) -> pd.Timestamp:
    valuation = pd.Timestamp(value)
    if valuation.tzinfo is None:
        raise ValueError("valuation_timestamp_utc must be timezone-aware")
    valuation = valuation.tz_convert("UTC")
    if valuation >= first_delivery:
        raise ValueError("valuation_timestamp_utc must precede the first delivery interval")
    return valuation


def _maximum_short_block_dispersion(values: np.ndarray, local: pd.DatetimeIndex) -> float:
    block = np.where((local.hour >= 8) & (local.hour < 20), "PEAK", "OFFPEAK")
    frame = pd.DataFrame(
        {
            "value": np.asarray(values, dtype=float),
            "day": local.strftime("%Y-%m-%d"),
            "block": block,
        }
    )
    ranges = frame.groupby(["day", "block"], sort=False)["value"].agg(lambda x: float(x.max() - x.min()))
    return float(ranges.max()) if len(ranges) else 0.0


def _project_to_nullspace(values: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    projected = np.asarray(values, dtype=float).copy()
    if matrix.size == 0:
        return projected
    gram = matrix @ matrix.T
    for _ in range(3):
        residual = matrix @ projected
        if float(np.max(np.abs(residual))) <= 1e-12:
            break
        multipliers = np.linalg.lstsq(gram, residual, rcond=None)[0]
        projected -= matrix.T @ multipliers
    return projected


def _monthly_weighted_residuals(
    values: np.ndarray,
    local: pd.DatetimeIndex,
    hours: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "month": local.strftime("%Y-%m"),
            "value": np.asarray(values, dtype=float),
            "hours": np.asarray(hours, dtype=float),
        }
    )
    rows = []
    for month, group in frame.groupby("month", sort=True):
        delivery_hours = float(group["hours"].sum())
        achieved = float(np.dot(group["value"], group["hours"]) / delivery_hours)
        rows.append(
            {
                "month": str(month),
                "achieved_delta_mean_eur_mwh": achieved,
                "abs_error": abs(achieved),
                "delivery_hours": delivery_hours,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "LOCAL_TIMEZONE",
    "MAX_ABS_RESIDUAL_EUR_MWH",
    "MAX_BLOCK_DISPERSION_EUR_MWH",
    "SHORT_TENOR_SHAPE_CONTRACT_VERSION",
    "SHORT_TENOR_SHAPE_PROJECTION_SCHEMA_VERSION",
    "ShortTenorShapeProjection",
    "project_short_tenor_shape_delta",
]

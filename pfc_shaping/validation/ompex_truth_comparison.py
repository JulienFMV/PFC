"""Outcome-blind descriptive comparison of CH LT and OMPEX against truth.

This module deliberately does not decide superiority.  It enforces the exact
common hourly grid and computes paired forecast errors against the same truth.
Scientific admission still requires authenticated origins, an independent
truth receipt, preregistered margins and dependence-aware multi-origin
inference under the OMPEX benchmark contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

SCHEMA_VERSION = "fmv_ompex_truth_comparison.v1"
LOCAL_TIMEZONE = "Europe/Zurich"
UTC = "UTC"
HOURLY = pd.Timedelta(hours=1)


class OmpexTruthComparisonError(ValueError):
    """Raised when a paired comparison would be ambiguous or biased."""


@dataclass(frozen=True)
class OmpexTruthComparison:
    """Exact aligned data and price-error diagnostics."""

    aligned: pd.DataFrame
    subgroup_metrics: pd.DataFrame
    summary: Mapping[str, object]


def compare_hourly_forecasts_against_truth(
    *,
    candidate: pd.Series,
    ompex: pd.Series,
    truth: pd.Series,
    origin_utc: pd.Timestamp,
    target_start_utc: pd.Timestamp,
    target_end_utc: pd.Timestamp,
    timestamp_semantics_authenticated: bool = False,
    truth_independence_authenticated: bool = False,
) -> OmpexTruthComparison:
    """Score both forecasts against one exact hourly truth window.

    The target interval is half-open: ``[target_start_utc, target_end_utc)``.
    No inner join, imputation, duplicate aggregation or automatic timestamp
    shift is allowed.  A negative MAE delta means that the candidate has lower
    descriptive MAE than OMPEX on this exact window.
    """

    origin = _utc_timestamp(origin_utc, label="origin")
    target_start = _utc_hour(target_start_utc, label="target start")
    target_end = _utc_hour(target_end_utc, label="target end")
    if target_end <= target_start:
        raise OmpexTruthComparisonError("target end must be after target start")
    if target_start < origin:
        raise OmpexTruthComparisonError("target starts before the forecast origin")

    expected = pd.date_range(
        target_start,
        target_end,
        freq=HOURLY,
        inclusive="left",
        name="delivery_start_utc",
    )
    if expected.empty:
        raise OmpexTruthComparisonError("target window has no native hours")

    candidate_window = _exact_window(candidate, expected=expected, label="candidate")
    ompex_window = _exact_window(ompex, expected=expected, label="OMPEX")
    truth_window = _exact_window(truth, expected=expected, label="truth")

    aligned = pd.DataFrame(
        {
            "candidate_eur_mwh": candidate_window,
            "ompex_eur_mwh": ompex_window,
            "truth_eur_mwh": truth_window,
        },
        index=expected,
    )
    aligned["candidate_error_eur_mwh"] = (
        aligned["candidate_eur_mwh"] - aligned["truth_eur_mwh"]
    )
    aligned["ompex_error_eur_mwh"] = (
        aligned["ompex_eur_mwh"] - aligned["truth_eur_mwh"]
    )
    aligned["candidate_absolute_error_eur_mwh"] = aligned[
        "candidate_error_eur_mwh"
    ].abs()
    aligned["ompex_absolute_error_eur_mwh"] = aligned[
        "ompex_error_eur_mwh"
    ].abs()
    aligned["paired_absolute_loss_delta_eur_mwh"] = (
        aligned["candidate_absolute_error_eur_mwh"]
        - aligned["ompex_absolute_error_eur_mwh"]
    )

    candidate_metrics = _point_metrics(candidate_window, truth_window)
    ompex_metrics = _point_metrics(ompex_window, truth_window)
    paired = _paired_metrics(candidate_window, ompex_window, truth_window)
    ramp = _ramp_metrics(candidate_window, ompex_window, truth_window)
    subgroup_metrics = _subgroup_metrics(
        aligned,
        origin_utc=origin,
    )

    blockers = [
        "candidate freeze chronology before OMPEX access is not authenticated here",
        "OMPEX vintage availability at the asserted origin is not authenticated here",
        "authenticated countable forecast origins are not supplied by this descriptive scorer",
        "numeric superiority and subgroup non-inferiority margins are not preregistered here",
        "dependence-aware multi-origin inference and multiplicity control are not run here",
        "monthly solver constraint integrity requires its separate manifest-backed gate",
    ]
    if not timestamp_semantics_authenticated:
        blockers.append("OMPEX hour-ending timestamp semantics are not authenticated")
    if not truth_independence_authenticated:
        blockers.append("independence and publication timing of realised truth are not authenticated")

    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "DESCRIPTIVE_ONLY_NO_SUPERIORITY_DECISION",
        "origin_utc": origin.isoformat(),
        "target_start_utc": target_start.isoformat(),
        "target_end_utc_exclusive": target_end.isoformat(),
        "native_resolution": "PT1H",
        "currency_unit": "EUR/MWh",
        "n_native_hours": len(expected),
        "same_target_grid": True,
        "same_missingness_mask": True,
        "automatic_alignment_selection_used": False,
        "timestamp_semantics_authenticated": timestamp_semantics_authenticated,
        "truth_independence_authenticated": truth_independence_authenticated,
        "effective_sample_size_unit": "forecast_origin_and_native_hour",
        "countable_origin_count": 0,
        "candidate": candidate_metrics,
        "ompex": ompex_metrics,
        "paired": paired,
        "ramp": ramp,
        "subgroup_count": int(len(subgroup_metrics)),
        "superiority_claim_authorized": False,
        "model_selection_authorized": False,
        "promotion_authorized": False,
        "production_authorized": False,
        "blockers": blockers,
    }
    return OmpexTruthComparison(
        aligned=aligned,
        subgroup_metrics=subgroup_metrics,
        summary=summary,
    )


def _utc_timestamp(value: pd.Timestamp, *, label: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        raise OmpexTruthComparisonError(f"{label} must be timezone-aware")
    return parsed.tz_convert(UTC)


def _utc_hour(value: pd.Timestamp, *, label: str) -> pd.Timestamp:
    parsed = _utc_timestamp(value, label=label)
    if parsed != parsed.floor("h"):
        raise OmpexTruthComparisonError(f"{label} must align to a native hour")
    return parsed


def _exact_window(
    series: pd.Series, *, expected: pd.DatetimeIndex, label: str
) -> pd.Series:
    if not isinstance(series, pd.Series) or series.empty:
        raise OmpexTruthComparisonError(f"{label} must be a non-empty Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise OmpexTruthComparisonError(f"{label} index must be a DatetimeIndex")
    if series.index.tz is None:
        raise OmpexTruthComparisonError(f"{label} timestamps must be timezone-aware")

    index = series.index.tz_convert(UTC).rename("delivery_start_utc")
    numeric = pd.to_numeric(series, errors="coerce")
    normalized = pd.Series(numeric.to_numpy(dtype="float64"), index=index, name=label)
    if normalized.index.has_duplicates:
        raise OmpexTruthComparisonError(f"{label} timestamps must be unique")
    if not normalized.index.is_monotonic_increasing:
        raise OmpexTruthComparisonError(f"{label} timestamps must be sorted")
    if not bool(np.isfinite(normalized.to_numpy()).all()):
        raise OmpexTruthComparisonError(f"{label} values must be finite and complete")

    window = normalized.loc[
        (normalized.index >= expected[0])
        & (normalized.index < expected[-1] + HOURLY)
    ]
    if not window.index.equals(expected):
        missing = expected.difference(window.index)
        extra = window.index.difference(expected)
        raise OmpexTruthComparisonError(
            f"{label} does not exactly cover the target grid "
            f"(missing={len(missing)}, extra={len(extra)})"
        )
    return window


def _point_metrics(forecast: pd.Series, truth: pd.Series) -> dict[str, float | int]:
    error = forecast.to_numpy() - truth.to_numpy()
    absolute = np.abs(error)
    return {
        "n_native_hours": int(len(error)),
        "mae_eur_mwh": float(np.mean(absolute)),
        "rmse_eur_mwh": float(np.sqrt(np.mean(np.square(error)))),
        "bias_eur_mwh": float(np.mean(error)),
        "median_absolute_error_eur_mwh": float(np.median(absolute)),
        "p95_absolute_error_eur_mwh": float(np.quantile(absolute, 0.95)),
        "max_absolute_error_eur_mwh": float(np.max(absolute)),
    }


def _paired_metrics(
    candidate: pd.Series, ompex: pd.Series, truth: pd.Series
) -> dict[str, float | int | None]:
    candidate_loss = np.abs(candidate.to_numpy() - truth.to_numpy())
    ompex_loss = np.abs(ompex.to_numpy() - truth.to_numpy())
    delta = candidate_loss - ompex_loss
    ompex_mae = float(np.mean(ompex_loss))
    relative = None if ompex_mae == 0.0 else float(-np.mean(delta) / ompex_mae)
    tolerance = 1e-12
    return {
        "mae_delta_candidate_minus_ompex_eur_mwh": float(np.mean(delta)),
        "relative_mae_improvement_vs_ompex": relative,
        "candidate_better_hour_count": int(np.sum(delta < -tolerance)),
        "equal_absolute_error_hour_count": int(np.sum(np.abs(delta) <= tolerance)),
        "candidate_worse_hour_count": int(np.sum(delta > tolerance)),
        "confidence_interval": None,
        "p_value": None,
        "decision": "NOT_EVALUATED",
    }


def _ramp_metrics(
    candidate: pd.Series, ompex: pd.Series, truth: pd.Series
) -> dict[str, float | int | None]:
    if len(truth) < 2:
        return {
            "n_native_ramps": 0,
            "candidate_ramp_mae_eur_mwh": None,
            "ompex_ramp_mae_eur_mwh": None,
            "ramp_mae_delta_candidate_minus_ompex_eur_mwh": None,
        }
    truth_ramp = np.diff(truth.to_numpy())
    candidate_error = np.diff(candidate.to_numpy()) - truth_ramp
    ompex_error = np.diff(ompex.to_numpy()) - truth_ramp
    candidate_mae = float(np.mean(np.abs(candidate_error)))
    ompex_mae = float(np.mean(np.abs(ompex_error)))
    return {
        "n_native_ramps": int(len(truth_ramp)),
        "candidate_ramp_mae_eur_mwh": candidate_mae,
        "ompex_ramp_mae_eur_mwh": ompex_mae,
        "ramp_mae_delta_candidate_minus_ompex_eur_mwh": candidate_mae - ompex_mae,
    }


def _subgroup_metrics(
    aligned: pd.DataFrame, *, origin_utc: pd.Timestamp
) -> pd.DataFrame:
    index = aligned.index
    local = index.tz_convert(LOCAL_TIMEZONE)
    masks: list[tuple[str, str, np.ndarray]] = [
        ("overall", "all", np.ones(len(index), dtype=bool)),
    ]

    peak = (local.dayofweek < 5) & (local.hour >= 8) & (local.hour < 20)
    weekend = local.dayofweek >= 5
    masks.extend(
        [
            ("delivery_block", "weekday_peak_08_20", np.asarray(peak)),
            ("delivery_block", "weekday_offpeak", np.asarray(~peak & ~weekend)),
            ("delivery_block", "weekend", np.asarray(weekend)),
        ]
    )

    season_by_month = {
        "winter": {12, 1, 2},
        "spring": {3, 4, 5},
        "summer": {6, 7, 8},
        "autumn": {9, 10, 11},
    }
    months = np.asarray(local.month)
    for name, values in season_by_month.items():
        masks.append(("season", name, np.isin(months, list(values))))

    truth = aligned["truth_eur_mwh"].to_numpy()
    lower, upper = np.quantile(truth, [0.05, 0.95])
    masks.extend(
        [
            ("truth_sign", "negative", truth < 0.0),
            ("truth_sign", "nonnegative", truth >= 0.0),
            ("truth_tail", "lower_5pct", truth <= lower),
            ("truth_tail", "central_90pct", (truth > lower) & (truth < upper)),
            ("truth_tail", "upper_5pct", truth >= upper),
        ]
    )

    horizon_days = np.asarray((index - origin_utc).total_seconds() / 86_400.0)
    masks.extend(
        [
            ("horizon", "0_31_days", horizon_days <= 31.0),
            ("horizon", "gt31_365_days", (horizon_days > 31.0) & (horizon_days <= 365.0)),
            ("horizon", "gt365_days", horizon_days > 365.0),
        ]
    )

    offsets = np.asarray(
        [timestamp.utcoffset().total_seconds() for timestamp in local],
        dtype="float64",
    )
    transition = np.zeros(len(offsets), dtype=bool)
    if len(offsets) > 1:
        changes = np.flatnonzero(offsets[1:] != offsets[:-1]) + 1
        for position in changes:
            transition[max(0, position - 1) : min(len(transition), position + 1)] = True
    masks.extend(
        [
            ("dst", "transition_adjacent_hours", transition),
            ("dst", "ordinary_hours", ~transition),
        ]
    )

    rows: list[dict[str, object]] = []
    for family, subgroup, mask in masks:
        count = int(np.sum(mask))
        if count == 0:
            rows.append(
                {
                    "family": family,
                    "subgroup": subgroup,
                    "n_native_hours": 0,
                    "status": "UNSUPPORTED_EMPTY",
                    "candidate_mae_eur_mwh": math.nan,
                    "ompex_mae_eur_mwh": math.nan,
                    "mae_delta_candidate_minus_ompex_eur_mwh": math.nan,
                    "candidate_rmse_eur_mwh": math.nan,
                    "ompex_rmse_eur_mwh": math.nan,
                }
            )
            continue
        selected = aligned.loc[mask]
        candidate_metrics = _point_metrics(
            selected["candidate_eur_mwh"], selected["truth_eur_mwh"]
        )
        ompex_metrics = _point_metrics(
            selected["ompex_eur_mwh"], selected["truth_eur_mwh"]
        )
        rows.append(
            {
                "family": family,
                "subgroup": subgroup,
                "n_native_hours": count,
                "status": "DESCRIPTIVE_ONLY",
                "candidate_mae_eur_mwh": candidate_metrics["mae_eur_mwh"],
                "ompex_mae_eur_mwh": ompex_metrics["mae_eur_mwh"],
                "mae_delta_candidate_minus_ompex_eur_mwh": (
                    candidate_metrics["mae_eur_mwh"] - ompex_metrics["mae_eur_mwh"]
                ),
                "candidate_rmse_eur_mwh": candidate_metrics["rmse_eur_mwh"],
                "ompex_rmse_eur_mwh": ompex_metrics["rmse_eur_mwh"],
            }
        )
    return pd.DataFrame(rows)

"""Authority-negative synthetic scoring for the frozen LT evaluation protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from pfc_shaping.lt.evaluation_protocol import (
    LEAD_MONTH_BUCKETS,
    PRIMARY_METRIC,
    SECONDARY_METRICS,
    EvaluationAuthority,
    default_evaluation_protocol,
)
from pfc_shaping.lt.model.shape_constraints import interval_hours, validate_utc_index

SYNTHETIC_EVALUATION_SCHEMA = "fmv-lt-synthetic-evaluation.v1"
SYNTHETIC_PROVENANCE = "SYNTHETIC_FIXTURE_ONLY"
_COMPUTED_METRICS = (
    PRIMARY_METRIC,
    "MONTHLY_LEVEL_NEUTRALIZED_RMSE_EUR_MWH",
    "MONTHLY_LEVEL_NEUTRALIZED_BIAS_EUR_MWH",
    "P95_ABSOLUTE_ERROR_EUR_MWH",
)
_UNSUPPORTED_REASONS = {
    "BASE_PEAK_OFFPEAK_WEIGHTED_ERROR_EUR_MWH": (
        "FROZEN_PRODUCT_WEIGHT_FORMULA_AND_MARKET_GATE_INPUTS_ABSENT"
    ),
    "FIXED_PROFILE_CAPTURE_PRICE_ERROR_EUR_MWH": "FROZEN_PROFILE_AND_FULL_PRICE_INPUTS_ABSENT",
    "BLOC_13_CONTRACT_PAYOFF_ERROR_CHF": "FROZEN_CONTRACT_AND_FX_INPUTS_ABSENT",
    "HYDRO_DISPATCH_REALIZED_NET_VALUE_OR_REGRET_CHF": (
        "FROZEN_DISPATCH_POLICY_REALIZED_PRICES_AND_FX_INPUTS_ABSENT"
    ),
}


class EvaluationEngineError(ValueError):
    """Raised when a synthetic evaluation request violates the protocol."""


class MetricStatus(str, Enum):
    COMPUTED_SYNTHETIC = "COMPUTED_SYNTHETIC_NON_SCIENTIFIC"
    UNSUPPORTED = "UNSUPPORTED_NEVER_PASS"


@dataclass(frozen=True, slots=True)
class SyntheticEvaluationSet:
    """One immutable prediction fixture on a common delivery index."""

    fixture_id: str
    origin_slot_id: str
    origin_as_of_utc: datetime
    delivery_at_utc: pd.DatetimeIndex
    truth_eur_mwh: np.ndarray
    predictions_eur_mwh: Mapping[str, np.ndarray]
    provenance: str = field(default=SYNTHETIC_PROVENANCE, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.fixture_id, str) or not self.fixture_id.startswith("synthetic-"):
            raise EvaluationEngineError("fixture_id must identify a synthetic fixture")
        protocol = default_evaluation_protocol()
        slot = next(
            (item for item in protocol.holdout.origin_slots if item.slot_id == self.origin_slot_id),
            None,
        )
        if slot is None:
            raise EvaluationEngineError("origin_slot_id is not in the frozen prospective cohort")
        origin = _require_utc_datetime(self.origin_as_of_utc, "origin_as_of_utc")
        if origin != slot.origin_as_of_utc:
            raise EvaluationEngineError("origin timestamp does not match the frozen origin slot")
        try:
            delivery = validate_utc_index(self.delivery_at_utc)
            interval_hours(delivery)
        except (TypeError, ValueError) as exc:
            raise EvaluationEngineError(str(exc)) from exc
        if len(delivery) == 0:
            raise EvaluationEngineError("delivery fixture cannot be empty")
        if bool((delivery < pd.Timestamp(origin)).any()):
            raise EvaluationEngineError("delivery rows cannot precede the frozen origin")
        leads = _lead_month_numbers(delivery, origin)
        if bool(((leads < 1) | (leads > 36)).any()):
            raise EvaluationEngineError(
                "delivery rows must remain within frozen lead months 1 through 36"
            )

        truth = _readonly_vector(self.truth_eur_mwh, len(delivery), "truth_eur_mwh")
        expected_ids = tuple(item.candidate_id for item in protocol.candidates)
        if not isinstance(self.predictions_eur_mwh, Mapping):
            raise EvaluationEngineError("predictions_eur_mwh must be a mapping")
        if set(self.predictions_eur_mwh) != set(expected_ids):
            raise EvaluationEngineError(
                "predictions must contain the exact frozen candidate inventory"
            )
        predictions = {
            candidate_id: _readonly_vector(
                self.predictions_eur_mwh[candidate_id],
                len(delivery),
                f"prediction {candidate_id}",
            )
            for candidate_id in expected_ids
        }
        object.__setattr__(self, "origin_as_of_utc", origin)
        object.__setattr__(self, "delivery_at_utc", delivery)
        object.__setattr__(self, "truth_eur_mwh", truth)
        object.__setattr__(self, "predictions_eur_mwh", MappingProxyType(predictions))


@dataclass(frozen=True, slots=True)
class MetricResult:
    metric_id: str
    status: MetricStatus
    value: float | None
    reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, MetricStatus):
            raise TypeError("status must be a MetricStatus")
        if self.status is MetricStatus.COMPUTED_SYNTHETIC:
            if self.value is None or not np.isfinite(self.value) or self.reason is not None:
                raise EvaluationEngineError(
                    "computed metrics require one finite value and no reason"
                )
        elif self.value is not None or not self.reason:
            raise EvaluationEngineError("unsupported metrics require a reason and no value")

    def to_manifest(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "status": self.status.value,
            "value": self.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class HorizonScore:
    bucket_id: str
    common_row_count: int
    total_row_count: int
    common_energy_hours: float
    metrics: Mapping[str, MetricResult]

    def __post_init__(self) -> None:
        _validate_metric_mapping(self.metrics, label="horizon")
        if self.bucket_id not in LEAD_MONTH_BUCKETS:
            raise EvaluationEngineError("unknown lead-month bucket")
        if self.common_row_count < 0 or self.total_row_count < self.common_row_count:
            raise EvaluationEngineError("horizon row counts are inconsistent")
        if not np.isfinite(self.common_energy_hours) or self.common_energy_hours < 0.0:
            raise EvaluationEngineError("horizon common energy must be finite and non-negative")
        if (self.common_row_count == 0) != (self.common_energy_hours == 0.0):
            raise EvaluationEngineError("empty horizon rows and energy must agree")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    def to_manifest(self) -> dict[str, object]:
        return {
            "bucket_id": self.bucket_id,
            "common_row_count": self.common_row_count,
            "total_row_count": self.total_row_count,
            "common_energy_hours": self.common_energy_hours,
            "metrics": [metric.to_manifest() for metric in self.metrics.values()],
        }


@dataclass(frozen=True, slots=True)
class CandidateScore:
    candidate_id: str
    metrics: Mapping[str, MetricResult]
    common_row_count: int
    total_row_count: int
    common_energy_hours: float
    horizon_scores: tuple[HorizonScore, ...]

    def __post_init__(self) -> None:
        _validate_metric_mapping(self.metrics, label="candidate")
        if not 0 < self.common_row_count <= self.total_row_count:
            raise EvaluationEngineError("candidate row counts are inconsistent")
        if not np.isfinite(self.common_energy_hours) or self.common_energy_hours <= 0.0:
            raise EvaluationEngineError("candidate common energy must be positive and finite")
        if not isinstance(self.horizon_scores, tuple) or any(
            not isinstance(score, HorizonScore) for score in self.horizon_scores
        ):
            raise TypeError("horizon_scores must be a tuple of HorizonScore values")
        if tuple(score.bucket_id for score in self.horizon_scores) != LEAD_MONTH_BUCKETS:
            raise EvaluationEngineError("horizon-score inventory differs from the frozen estimand")
        if sum(score.common_row_count for score in self.horizon_scores) != self.common_row_count:
            raise EvaluationEngineError("horizon common rows do not reconcile to the candidate")
        if sum(score.total_row_count for score in self.horizon_scores) != self.total_row_count:
            raise EvaluationEngineError("horizon total rows do not reconcile to the candidate")
        horizon_energy = sum(score.common_energy_hours for score in self.horizon_scores)
        if not np.isclose(horizon_energy, self.common_energy_hours, rtol=0.0, atol=1e-12):
            raise EvaluationEngineError("horizon energy does not reconcile to the candidate")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    def to_manifest(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "common_row_count": self.common_row_count,
            "total_row_count": self.total_row_count,
            "common_energy_hours": self.common_energy_hours,
            "metrics": [metric.to_manifest() for metric in self.metrics.values()],
            "horizon_scores": [score.to_manifest() for score in self.horizon_scores],
        }


@dataclass(frozen=True, slots=True)
class SyntheticEvaluationReport:
    fixture_id: str
    origin_slot_id: str
    protocol_semantic_sha256: str
    common_row_count: int
    total_row_count: int
    scores: tuple[CandidateScore, ...]
    authority: EvaluationAuthority = field(default_factory=EvaluationAuthority, init=False)
    schema_version: str = field(default=SYNTHETIC_EVALUATION_SCHEMA, init=False)
    provenance: str = field(default=SYNTHETIC_PROVENANCE, init=False)

    def __post_init__(self) -> None:
        protocol = default_evaluation_protocol()
        expected_ids = tuple(item.candidate_id for item in protocol.candidates)
        if self.protocol_semantic_sha256 != protocol.semantic_sha256():
            raise EvaluationEngineError("report protocol hash differs from the canonical protocol")
        if not isinstance(self.scores, tuple) or any(
            not isinstance(score, CandidateScore) for score in self.scores
        ):
            raise TypeError("scores must be a tuple of CandidateScore values")
        if tuple(score.candidate_id for score in self.scores) != expected_ids:
            raise EvaluationEngineError("report candidate inventory differs from the protocol")
        if not 0 < self.common_row_count <= self.total_row_count:
            raise EvaluationEngineError("report row counts are inconsistent")
        if any(
            score.common_row_count != self.common_row_count
            or score.total_row_count != self.total_row_count
            for score in self.scores
        ):
            raise EvaluationEngineError("candidate and report row counts differ")

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "provenance": self.provenance,
            "fixture_id": self.fixture_id,
            "origin_slot_id": self.origin_slot_id,
            "protocol_semantic_sha256": self.protocol_semantic_sha256,
            "common_row_count": self.common_row_count,
            "total_row_count": self.total_row_count,
            "comparison_rows": "SAME_COMPLETE_CASE_INTERSECTION_FOR_ALL_CANDIDATES",
            "monthly_level_authority": "CH_MONTHLY_BASE_SOLVER_UNCHANGED",
            "horizon_buckets": list(LEAD_MONTH_BUCKETS),
            "unsupported_layer_cannot_be_hidden_by_aggregate_pass": True,
            "countable_origin": False,
            "real_truth_opened": False,
            "ranking_or_selection_performed": False,
            "scores": [score.to_manifest() for score in self.scores],
            "authority": self.authority.to_manifest(),
        }


def evaluate_synthetic_predictions(batch: SyntheticEvaluationSet) -> SyntheticEvaluationReport:
    """Score one synthetic fixture without ranking or granting any authority."""

    if not isinstance(batch, SyntheticEvaluationSet):
        raise TypeError("batch must be a SyntheticEvaluationSet")
    protocol = default_evaluation_protocol()
    weights = interval_hours(batch.delivery_at_utc)
    total_leads = _lead_month_numbers(batch.delivery_at_utc, batch.origin_as_of_utc)
    common = np.isfinite(batch.truth_eur_mwh)
    for prediction in batch.predictions_eur_mwh.values():
        common &= np.isfinite(prediction)
    if not bool(common.any()):
        raise EvaluationEngineError("the common complete-case intersection is empty")

    eligible_index = batch.delivery_at_utc[common]
    eligible_weights = weights[common]
    truth_centered = _monthly_center(
        batch.truth_eur_mwh[common],
        eligible_index,
        eligible_weights,
    )
    scores = tuple(
        _score_candidate(
            candidate_id,
            prediction[common],
            truth_centered,
            eligible_index,
            eligible_weights,
            eligible_leads=total_leads[common],
            total_leads=total_leads,
            total_row_count=len(batch.delivery_at_utc),
        )
        for candidate_id, prediction in batch.predictions_eur_mwh.items()
    )
    return SyntheticEvaluationReport(
        fixture_id=batch.fixture_id,
        origin_slot_id=batch.origin_slot_id,
        protocol_semantic_sha256=protocol.semantic_sha256(),
        common_row_count=int(common.sum()),
        total_row_count=len(common),
        scores=scores,
    )


def _score_candidate(
    candidate_id: str,
    prediction: np.ndarray,
    truth_centered: np.ndarray,
    index: pd.DatetimeIndex,
    weights: np.ndarray,
    *,
    eligible_leads: np.ndarray,
    total_leads: np.ndarray,
    total_row_count: int,
) -> CandidateScore:
    prediction_centered = _monthly_center(prediction, index, weights)
    error = prediction_centered - truth_centered
    horizon_scores = tuple(
        _score_horizon_bucket(bucket_id, error, weights, eligible_leads, total_leads)
        for bucket_id in LEAD_MONTH_BUCKETS
    )
    return CandidateScore(
        candidate_id=candidate_id,
        metrics=_computed_and_unsupported_metrics(error, weights),
        common_row_count=len(error),
        total_row_count=total_row_count,
        common_energy_hours=float(weights.sum()),
        horizon_scores=horizon_scores,
    )


def _computed_and_unsupported_metrics(
    error: np.ndarray,
    weights: np.ndarray,
) -> dict[str, MetricResult]:
    absolute_error = np.abs(error)
    weight_sum = float(weights.sum())
    values = {
        PRIMARY_METRIC: float(np.dot(weights, absolute_error) / weight_sum),
        "MONTHLY_LEVEL_NEUTRALIZED_RMSE_EUR_MWH": float(
            np.sqrt(np.dot(weights, np.square(error)) / weight_sum)
        ),
        "MONTHLY_LEVEL_NEUTRALIZED_BIAS_EUR_MWH": float(np.dot(weights, error) / weight_sum),
        "P95_ABSOLUTE_ERROR_EUR_MWH": _weighted_quantile(
            absolute_error,
            weights,
            0.95,
        ),
    }
    metrics: dict[str, MetricResult] = {}
    for metric_id in (PRIMARY_METRIC, *SECONDARY_METRICS):
        if metric_id in _COMPUTED_METRICS:
            metrics[metric_id] = MetricResult(
                metric_id=metric_id,
                status=MetricStatus.COMPUTED_SYNTHETIC,
                value=values[metric_id],
                reason=None,
            )
        else:
            metrics[metric_id] = MetricResult(
                metric_id=metric_id,
                status=MetricStatus.UNSUPPORTED,
                value=None,
                reason=_UNSUPPORTED_REASONS[metric_id],
            )
    return metrics


def _score_horizon_bucket(
    bucket_id: str,
    error: np.ndarray,
    weights: np.ndarray,
    eligible_leads: np.ndarray,
    total_leads: np.ndarray,
) -> HorizonScore:
    first, last = _bucket_bounds(bucket_id)
    common = (eligible_leads >= first) & (eligible_leads <= last)
    total = (total_leads >= first) & (total_leads <= last)
    if not bool(common.any()):
        reason = "NO_COMMON_SYNTHETIC_ROWS_FOR_HORIZON_BUCKET"
        metrics = {
            metric_id: MetricResult(
                metric_id=metric_id,
                status=MetricStatus.UNSUPPORTED,
                value=None,
                reason=reason,
            )
            for metric_id in (PRIMARY_METRIC, *SECONDARY_METRICS)
        }
        return HorizonScore(
            bucket_id=bucket_id,
            common_row_count=0,
            total_row_count=int(total.sum()),
            common_energy_hours=0.0,
            metrics=metrics,
        )
    selected_weights = weights[common]
    return HorizonScore(
        bucket_id=bucket_id,
        common_row_count=int(common.sum()),
        total_row_count=int(total.sum()),
        common_energy_hours=float(selected_weights.sum()),
        metrics=_computed_and_unsupported_metrics(error[common], selected_weights),
    )


def _monthly_center(
    values: np.ndarray,
    index: pd.DatetimeIndex,
    weights: np.ndarray,
) -> np.ndarray:
    local = index.tz_convert("Europe/Zurich")
    month_codes = local.year.to_numpy(dtype=np.int64) * 12 + local.month.to_numpy(dtype=np.int64)
    centered = np.empty(len(values), dtype=float)
    for month_code in np.unique(month_codes):
        mask = month_codes == month_code
        month_weights = weights[mask]
        weight_sum = float(month_weights.sum())
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            raise EvaluationEngineError("monthly energy weights must have a positive finite sum")
        mean = float(np.dot(month_weights, values[mask]) / weight_sum)
        centered[mask] = values[mask] - mean
    if not np.isfinite(centered).all():
        raise EvaluationEngineError("monthly centering returned non-finite values")
    return centered


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise EvaluationEngineError("quantile must be between zero and one")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    threshold = quantile * float(sorted_weights.sum())
    index = int(np.searchsorted(np.cumsum(sorted_weights), threshold, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _lead_month_numbers(index: pd.DatetimeIndex, origin: datetime) -> np.ndarray:
    local = index.tz_convert("Europe/Zurich")
    local_origin = pd.Timestamp(origin).tz_convert("Europe/Zurich")
    origin_ordinal = local_origin.year * 12 + local_origin.month
    return (
        local.year.to_numpy(dtype=np.int64) * 12
        + local.month.to_numpy(dtype=np.int64)
        - origin_ordinal
    )


def _bucket_bounds(bucket_id: str) -> tuple[int, int]:
    try:
        return int(bucket_id[1:3]), int(bucket_id[5:7])
    except (TypeError, ValueError) as exc:  # pragma: no cover - frozen constant guard
        raise EvaluationEngineError("lead-month bucket identifier is malformed") from exc


def _validate_metric_mapping(metrics: Mapping[str, MetricResult], *, label: str) -> None:
    expected = (PRIMARY_METRIC, *SECONDARY_METRICS)
    if tuple(metrics) != expected:
        raise EvaluationEngineError(f"{label} metric inventory differs from the frozen estimand")
    if any(
        not isinstance(metric, MetricResult) or metric.metric_id != metric_id
        for metric_id, metric in metrics.items()
    ):
        raise EvaluationEngineError(f"{label} metric mapping keys and values are inconsistent")


def _readonly_vector(value: object, length: int, label: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluationEngineError(f"{label} must be numeric") from exc
    if array.shape != (length,):
        raise EvaluationEngineError(f"{label} must contain exactly {length} rows")
    array.setflags(write=False)
    return array


def _require_utc_datetime(value: object, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EvaluationEngineError(f"{label} must be a timezone-aware datetime")
    if value.utcoffset() != timedelta(0):
        raise EvaluationEngineError(f"{label} must use UTC")
    return pd.Timestamp(value).tz_convert("UTC").to_pydatetime()


__all__ = [
    "CandidateScore",
    "EvaluationEngineError",
    "HorizonScore",
    "MetricResult",
    "MetricStatus",
    "SYNTHETIC_EVALUATION_SCHEMA",
    "SYNTHETIC_PROVENANCE",
    "SyntheticEvaluationReport",
    "SyntheticEvaluationSet",
    "evaluate_synthetic_predictions",
]

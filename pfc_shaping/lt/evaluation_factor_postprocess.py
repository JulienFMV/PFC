"""Pure incumbent-equivalent post-processing for LT challenger factors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from pfc_shaping.lt.evaluation_protocol import (
    CandidateRole,
    default_evaluation_protocol,
)

FACTOR_BATCH_SCHEMA = "fmv-lt-postprocessed-challenger-factors.v1"


class FactorPostprocessError(ValueError):
    """Raised when raw factors violate the challenger post-processing contract."""


@dataclass(frozen=True, slots=True)
class FactorPostprocessAuthority:
    """Non-overridable negative authority for transformed predictions."""

    model_training_authorized: bool = field(default=False, init=False)
    truth_open_authorized: bool = field(default=False, init=False)
    model_selection_authorized: bool = field(default=False, init=False)
    monthly_level_change_authorized: bool = field(default=False, init=False)
    publication_authorized: bool = field(default=False, init=False)
    production_authorized: bool = field(default=False, init=False)

    def to_manifest(self) -> dict[str, bool]:
        return {
            "model_training_authorized": self.model_training_authorized,
            "truth_open_authorized": self.truth_open_authorized,
            "model_selection_authorized": self.model_selection_authorized,
            "monthly_level_change_authorized": self.monthly_level_change_authorized,
            "publication_authorized": self.publication_authorized,
            "production_authorized": self.production_authorized,
        }


@dataclass(frozen=True, slots=True)
class PostprocessedChallengerFactors:
    """Detached challenger factors carrying no execution or selection authority."""

    candidate_id: str
    origin_slot_id: str
    origin_as_of_utc: datetime
    delivery_at_utc: pd.DatetimeIndex
    values: np.ndarray
    audit: Mapping[str, object]
    authority: FactorPostprocessAuthority = field(
        default_factory=FactorPostprocessAuthority,
        init=False,
    )

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": FACTOR_BATCH_SCHEMA,
            "status": "PASS_POSTPROCESSED_CHALLENGER_FACTORS_NO_MODEL_AUTHORITY",
            "candidate_id": self.candidate_id,
            "origin_slot_id": self.origin_slot_id,
            "origin_as_of_utc": self.origin_as_of_utc.isoformat(),
            "output_name": "F_H",
            "delivery_grain": "NATIVE_QUARTER_HOUR_UTC_GRID",
            "incumbent_allowed": False,
            "postprocessing": [
                "POSITIVE_FLOOR_0_1",
                "SWISS_LOCAL_DAY_ARITHMETIC_MEAN_NORMALIZATION",
                "FINAL_CLIP_0_4_2_0",
            ],
            "postprocessing_applied_exactly_once": True,
            **dict(self.audit),
            "authority": self.authority.to_manifest(),
        }


def postprocess_challenger_factors(
    raw_factors: np.ndarray,
    delivery_at_utc: pd.DatetimeIndex,
    *,
    candidate_id: str,
    origin_slot_id: str,
    origin_as_of_utc: str | datetime | pd.Timestamp,
) -> PostprocessedChallengerFactors:
    """Apply the native incumbent's factor rules once to one challenger."""

    protocol = default_evaluation_protocol()
    candidate = next(
        (item for item in protocol.candidates if item.candidate_id == candidate_id),
        None,
    )
    if candidate is None:
        raise FactorPostprocessError("candidate_id is not in the frozen protocol")
    if candidate.role is not CandidateRole.CHALLENGER:
        raise FactorPostprocessError("incumbent factors are already postprocessed natively")

    slot = next(
        (item for item in protocol.holdout.origin_slots if item.slot_id == origin_slot_id),
        None,
    )
    if slot is None:
        raise FactorPostprocessError("origin_slot_id is not a frozen protocol slot")
    origin = _utc_scalar(origin_as_of_utc, label="origin")
    if origin.to_pydatetime() != slot.origin_as_of_utc:
        raise FactorPostprocessError("origin timestamp differs from the frozen protocol slot")

    delivery = _utc_index(delivery_at_utc)
    if len(delivery) == 0:
        raise FactorPostprocessError("delivery cannot be empty")
    if delivery.has_duplicates:
        raise FactorPostprocessError("delivery timestamps are duplicated")
    if bool((delivery < origin).any()):
        raise FactorPostprocessError("prediction delivery cannot precede origin")
    aligned = (
        (delivery.minute.to_numpy() % 15 == 0)
        & (delivery.second.to_numpy() == 0)
        & (delivery.microsecond.to_numpy() == 0)
    )
    if not bool(aligned.all()):
        raise FactorPostprocessError("delivery timestamps are not on the quarter-hour grid")

    raw = np.asarray(raw_factors, dtype=float)
    if raw.shape != (len(delivery),) or not np.isfinite(raw).all():
        raise FactorPostprocessError("raw factors must be one finite value per delivery row")
    floored = np.maximum(raw, 0.1)
    local = delivery.tz_convert("Europe/Zurich")
    day_keys = np.asarray([timestamp.strftime("%Y-%m-%d") for timestamp in local])
    daily_mean = pd.Series(floored).groupby(day_keys).transform("mean").to_numpy(dtype=float)
    values = _readonly(np.clip(floored / daily_mean, 0.4, 2.0))
    audit = MappingProxyType(
        {
            "row_count": len(values),
            "raw_value_sha256": _array_hash(raw),
            "delivery_sha256": _timestamp_hash(delivery),
            "value_sha256": _array_hash(values),
            "model_fit_performed": False,
            "real_truth_opened": False,
            "ranking_or_selection_performed": False,
        }
    )
    return PostprocessedChallengerFactors(
        candidate_id=candidate_id,
        origin_slot_id=origin_slot_id,
        origin_as_of_utc=slot.origin_as_of_utc,
        delivery_at_utc=delivery,
        values=values,
        audit=audit,
    )


def _utc_scalar(value: object, *, label: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise FactorPostprocessError(f"{label} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise FactorPostprocessError(f"{label} timestamp must be timezone-aware")
    return parsed.tz_convert("UTC")


def _utc_index(value: object) -> pd.DatetimeIndex:
    if not isinstance(value, pd.DatetimeIndex) or value.tz is None:
        raise FactorPostprocessError("delivery must be a timezone-aware DatetimeIndex")
    return value.tz_convert("UTC").copy()


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True, order="C")
    result.setflags(write=False)
    return result


def _array_hash(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    header = json.dumps(
        {"dtype": "<f8", "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + canonical.tobytes(order="C")).hexdigest()


def _timestamp_hash(values: pd.DatetimeIndex) -> str:
    payload = json.dumps(
        [value.isoformat() for value in values],
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "FACTOR_BATCH_SCHEMA",
    "FactorPostprocessAuthority",
    "FactorPostprocessError",
    "PostprocessedChallengerFactors",
    "postprocess_challenger_factors",
]

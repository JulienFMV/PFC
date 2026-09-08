"""Pure incumbent-equivalent target construction for LT hourly challengers.

The source-bound incumbent learns a clipped quarter-hour price ratio aggregated
by Swiss-local date and clock hour.  This module reproduces only that target
transformation from already materialized CH observations.  It performs no I/O,
feature construction, fitting, prediction, scoring, or authority transition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from pfc_shaping.lt.evaluation_protocol import default_evaluation_protocol

TARGET_BATCH_SCHEMA = "fmv-lt-incumbent-equivalent-hourly-targets.v1"
_SOURCE_COLUMNS = (
    "delivery_at_utc",
    "price_available_at_utc",
    "price_eur_mwh",
)
_HALF_LIFE_DAYS = 180.0


class HourlyTargetBuildError(ValueError):
    """Raised when source rows cannot reproduce the incumbent target."""


@dataclass(frozen=True, slots=True)
class HourlyTargetBuildAuthority:
    """Non-overridable negative authority for constructed target values."""

    data_acquisition_authorized: bool = field(default=False, init=False)
    target_materialization_authorized: bool = field(default=False, init=False)
    model_training_authorized: bool = field(default=False, init=False)
    truth_open_authorized: bool = field(default=False, init=False)
    model_selection_authorized: bool = field(default=False, init=False)
    monthly_level_change_authorized: bool = field(default=False, init=False)
    production_authorized: bool = field(default=False, init=False)

    def to_manifest(self) -> dict[str, bool]:
        return {
            "data_acquisition_authorized": self.data_acquisition_authorized,
            "target_materialization_authorized": self.target_materialization_authorized,
            "model_training_authorized": self.model_training_authorized,
            "truth_open_authorized": self.truth_open_authorized,
            "model_selection_authorized": self.model_selection_authorized,
            "monthly_level_change_authorized": self.monthly_level_change_authorized,
            "production_authorized": self.production_authorized,
        }


@dataclass(frozen=True, slots=True)
class ConstructedHourlyTargets:
    """Detached target rows ready for the common input assembly boundary."""

    origin_slot_id: str
    origin_as_of_utc: datetime
    row_ids: tuple[str, ...]
    delivery_at_utc: pd.DatetimeIndex
    target_available_at_utc: pd.DatetimeIndex
    values: np.ndarray
    audit: Mapping[str, object]
    authority: HourlyTargetBuildAuthority = field(
        default_factory=HourlyTargetBuildAuthority,
        init=False,
    )

    def to_metadata_frame(self) -> pd.DataFrame:
        """Return a detached frame accepted by the common input assembler."""

        return pd.DataFrame(
            {
                "row_id": self.row_ids,
                "delivery_at_utc": self.delivery_at_utc.copy(),
                "target_available_at_utc": self.target_available_at_utc.copy(),
                "target_f_h": np.array(self.values, copy=True),
            }
        )

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": TARGET_BATCH_SCHEMA,
            "status": "PASS_CONSTRUCTED_TARGETS_NO_MODEL_OR_TRUTH_AUTHORITY",
            "origin_slot_id": self.origin_slot_id,
            "origin_as_of_utc": self.origin_as_of_utc.isoformat(),
            "target_name": "INCUMBENT_EQUIVALENT_HOURLY_F_H",
            "target_column": "target_f_h",
            "direct_raw_price_fit_allowed": False,
            "source_market": "CH",
            "source_resolution": "QUARTER_HOUR",
            "local_day_timezone": "Europe/Zurich",
            "daily_mean_min_exclusive_eur_mwh": 5.0,
            "quarter_hour_ratio_clip": [0.2, 3.0],
            "half_life_days": _HALF_LIFE_DAYS,
            "training_group": "SWISS_LOCAL_DATE_AND_CLOCK_HOUR",
            "repeated_fallback_hour_policy": "MERGE_TO_MATCH_SOURCE_BOUND_INCUMBENT",
            **dict(self.audit),
            "authority": self.authority.to_manifest(),
        }


def construct_incumbent_equivalent_hourly_targets(
    source: pd.DataFrame,
    *,
    origin_slot_id: str,
    origin_as_of_utc: str | datetime | pd.Timestamp,
) -> ConstructedHourlyTargets:
    """Construct the frozen incumbent's hourly ``f_H`` training target."""

    origin = _frozen_origin(origin_slot_id, origin_as_of_utc)
    frame = _exact_source(source)
    delivery = _utc_index(frame["delivery_at_utc"], label="delivery")
    available = _utc_index(frame["price_available_at_utc"], label="price availability")
    _validate_times(delivery, available, pd.Timestamp(origin))

    price = pd.to_numeric(frame["price_eur_mwh"], errors="coerce").to_numpy(
        dtype=float,
        copy=True,
    )
    if bool(np.isinf(price).any()):
        raise HourlyTargetBuildError("price_eur_mwh cannot contain infinity")
    finite = np.isfinite(price)
    if not bool(finite.any()):
        raise HourlyTargetBuildError("price_eur_mwh has no finite observation")

    order = np.argsort(delivery.asi8, kind="stable")
    delivery = delivery[order]
    available = available[order]
    price = price[order]
    finite = finite[order]
    local = delivery.tz_convert("Europe/Zurich")
    day_keys = np.asarray([timestamp.strftime("%Y-%m-%d") for timestamp in local])
    hour_keys = np.asarray(
        [f"{day}-{timestamp.hour}" for day, timestamp in zip(day_keys, local, strict=True)]
    )

    daily_mean = pd.Series(price).groupby(day_keys).transform("mean").to_numpy(dtype=float)
    eligible = finite & np.isfinite(daily_mean) & (daily_mean > 5.0)
    if not bool(eligible.any()):
        raise HourlyTargetBuildError("no source row passes the incumbent daily-mean rule")

    ratios = np.full(len(price), np.nan, dtype=float)
    ratios[eligible] = np.clip(price[eligible] / daily_mean[eligible], 0.2, 3.0)
    t_max = delivery[finite].max()
    age_days = (t_max - delivery).total_seconds().to_numpy(dtype=float) / 86_400.0
    weights = np.exp(-np.log(2.0) * age_days / _HALF_LIFE_DAYS)

    groups = sorted(set(hour_keys[eligible]), key=lambda key: delivery[hour_keys == key].min())
    row_ids: list[str] = []
    target_delivery: list[pd.Timestamp] = []
    target_available: list[pd.Timestamp] = []
    targets: list[float] = []
    repeated_fallback_groups = 0
    for key in groups:
        members = eligible & (hour_keys == key)
        member_weights = weights[members]
        row_ids.append(f"fh-{key.replace('-', '')}")
        target_delivery.append(delivery[members].min())
        target_available.append(available[members].max())
        targets.append(float(np.dot(member_weights, ratios[members]) / member_weights.sum()))
        if int(members.sum()) > 4:
            repeated_fallback_groups += 1

    delivery_result = pd.DatetimeIndex(target_delivery)
    available_result = pd.DatetimeIndex(target_available)
    values = _readonly(np.asarray(targets, dtype=float))
    audit = MappingProxyType(
        {
            "source_row_count": len(frame),
            "finite_source_row_count": int(finite.sum()),
            "eligible_quarter_hour_row_count": int(eligible.sum()),
            "excluded_quarter_hour_row_count": int((~eligible).sum()),
            "hourly_target_row_count": len(values),
            "repeated_fallback_group_count": repeated_fallback_groups,
            "row_identity_sha256": _identity_hash(tuple(row_ids), delivery_result),
            "delivery_sha256": _timestamp_hash(delivery_result),
            "availability_sha256": _timestamp_hash(available_result),
            "value_sha256": _array_hash(values),
            "real_data_training_performed": False,
            "real_truth_opened": False,
            "ranking_or_selection_performed": False,
        }
    )
    return ConstructedHourlyTargets(
        origin_slot_id=origin_slot_id,
        origin_as_of_utc=origin,
        row_ids=tuple(row_ids),
        delivery_at_utc=delivery_result,
        target_available_at_utc=available_result,
        values=values,
        audit=audit,
    )


def _frozen_origin(
    origin_slot_id: str,
    origin_as_of_utc: str | datetime | pd.Timestamp,
) -> datetime:
    protocol = default_evaluation_protocol()
    slot = next(
        (item for item in protocol.holdout.origin_slots if item.slot_id == origin_slot_id),
        None,
    )
    if slot is None:
        raise HourlyTargetBuildError("origin_slot_id is not a frozen protocol slot")
    origin = _utc_scalar(origin_as_of_utc, label="origin")
    if origin.to_pydatetime() != slot.origin_as_of_utc:
        raise HourlyTargetBuildError("origin timestamp differs from the frozen protocol slot")
    return slot.origin_as_of_utc


def _exact_source(source: object) -> pd.DataFrame:
    if not isinstance(source, pd.DataFrame):
        raise TypeError("source must be a pandas DataFrame")
    if tuple(source.columns) != _SOURCE_COLUMNS:
        raise HourlyTargetBuildError("source columns are not exact")
    if len(source) == 0:
        raise HourlyTargetBuildError("source cannot be empty")
    return source.copy(deep=True)


def _utc_scalar(value: object, *, label: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise HourlyTargetBuildError(f"{label} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise HourlyTargetBuildError(f"{label} timestamp must be timezone-aware")
    return parsed.tz_convert("UTC")


def _utc_index(values: pd.Series, *, label: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([_utc_scalar(value, label=label) for value in values])


def _validate_times(
    delivery: pd.DatetimeIndex,
    available: pd.DatetimeIndex,
    origin: pd.Timestamp,
) -> None:
    if delivery.has_duplicates:
        raise HourlyTargetBuildError("delivery timestamps are duplicated")
    aligned = (
        (delivery.minute.to_numpy() % 15 == 0)
        & (delivery.second.to_numpy() == 0)
        & (delivery.microsecond.to_numpy() == 0)
    )
    if not bool(aligned.all()):
        raise HourlyTargetBuildError("delivery timestamps are not on the quarter-hour grid")
    if bool((delivery >= origin).any()):
        raise HourlyTargetBuildError("training delivery must precede origin")
    if bool((available >= origin).any()):
        raise HourlyTargetBuildError("price availability must be strictly before origin")


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


def _identity_hash(row_ids: tuple[str, ...], delivery: pd.DatetimeIndex) -> str:
    payload = [
        {"row_id": row_id, "delivery_at_utc": timestamp.isoformat()}
        for row_id, timestamp in zip(row_ids, delivery, strict=True)
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "TARGET_BATCH_SCHEMA",
    "ConstructedHourlyTargets",
    "HourlyTargetBuildAuthority",
    "HourlyTargetBuildError",
    "construct_incumbent_equivalent_hourly_targets",
]

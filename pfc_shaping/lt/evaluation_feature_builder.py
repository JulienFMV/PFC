"""Pure constructor for the frozen LT hourly evaluation features.

Inputs are already materialized and origin-safe. This module computes only the
Swiss-local calendar encodings and origin-relative maturity used by the
source-bound incumbent. It never loads, fills, forecasts, fits, or scores data.
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

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.evaluation_feature_inventory import (
    CANONICAL_FEATURE_INVENTORY_SEMANTIC_SHA256,
    CANONICAL_FEATURE_NAMES,
)
from pfc_shaping.lt.evaluation_protocol import default_evaluation_protocol
from pfc_shaping.lt.model.shape_hourly_mlp import _encode_features

FEATURE_BATCH_SCHEMA = "fmv-lt-hourly-feature-batch.v1"
_INPUT_COLUMNS = ("delivery_at_utc", "hydro_available_at_utc", "hydro_fill")
_SPLIT_ROLE = {
    "training": "REALIZED_ACTUAL",
    "prediction": "ORIGIN_FROZEN_CLIMATOLOGY",
}
_SECONDS_PER_YEAR = 365.25 * 86_400.0


class HourlyFeatureBuildError(ValueError):
    """Raised when a feature batch violates the frozen causal contract."""


@dataclass(frozen=True, slots=True)
class HourlyFeatureBuildAuthority:
    """Non-overridable negative authority for constructed feature values."""

    data_acquisition_authorized: bool = field(default=False, init=False)
    feature_forecast_authorized: bool = field(default=False, init=False)
    missing_value_fill_authorized: bool = field(default=False, init=False)
    model_training_authorized: bool = field(default=False, init=False)
    truth_open_authorized: bool = field(default=False, init=False)
    model_selection_authorized: bool = field(default=False, init=False)
    monthly_level_change_authorized: bool = field(default=False, init=False)
    production_authorized: bool = field(default=False, init=False)

    def to_manifest(self) -> dict[str, bool]:
        return {
            "data_acquisition_authorized": self.data_acquisition_authorized,
            "feature_forecast_authorized": self.feature_forecast_authorized,
            "missing_value_fill_authorized": self.missing_value_fill_authorized,
            "model_training_authorized": self.model_training_authorized,
            "truth_open_authorized": self.truth_open_authorized,
            "model_selection_authorized": self.model_selection_authorized,
            "monthly_level_change_authorized": self.monthly_level_change_authorized,
            "production_authorized": self.production_authorized,
        }


@dataclass(frozen=True, slots=True)
class ConstructedHourlyFeatures:
    """Detached read-only feature values for one split and frozen origin."""

    origin_slot_id: str
    origin_as_of_utc: datetime
    split: str
    hydro_information_role: str
    hydro_training_cutoff_utc: datetime | None
    delivery_at_utc: pd.DatetimeIndex
    hydro_available_at_utc: pd.DatetimeIndex
    feature_names: tuple[str, ...]
    values: np.ndarray
    audit: Mapping[str, object]
    authority: HourlyFeatureBuildAuthority = field(
        default_factory=HourlyFeatureBuildAuthority,
        init=False,
    )

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": FEATURE_BATCH_SCHEMA,
            "status": "PASS_CONSTRUCTED_HOURLY_FEATURES_NO_EXECUTION_AUTHORITY",
            "origin_slot_id": self.origin_slot_id,
            "origin_as_of_utc": self.origin_as_of_utc.isoformat(),
            "split": self.split,
            "hydro_information_role": self.hydro_information_role,
            "hydro_training_cutoff_utc": (
                None
                if self.hydro_training_cutoff_utc is None
                else self.hydro_training_cutoff_utc.isoformat()
            ),
            "feature_names": list(self.feature_names),
            "feature_inventory_semantic_sha256": (CANONICAL_FEATURE_INVENTORY_SEMANTIC_SHA256),
            "hydro_unit": "FRACTION_0_1",
            "calendar_timezone": "Europe/Zurich",
            "missingness_policy": "PRESERVE_NULL_FOR_COMMON_DOWNSTREAM_MASK",
            **dict(self.audit),
            "authority": self.authority.to_manifest(),
        }


def construct_hourly_features(
    source: pd.DataFrame,
    *,
    origin_slot_id: str,
    origin_as_of_utc: str | datetime | pd.Timestamp,
    split: str,
    hydro_information_role: str,
    hydro_training_cutoff_utc: str | datetime | pd.Timestamp | None = None,
) -> ConstructedHourlyFeatures:
    """Construct the exact nine-column hourly matrix from admitted values."""

    origin = _frozen_origin(origin_slot_id, origin_as_of_utc)
    split_name = _split(split, hydro_information_role)
    frame = _exact_source(source)
    delivery = _utc_index(frame["delivery_at_utc"], label="delivery")
    available = _utc_index(
        frame["hydro_available_at_utc"],
        label="hydro availability",
    )
    if delivery.has_duplicates:
        raise HourlyFeatureBuildError("delivery timestamps are duplicated")

    origin_timestamp = pd.Timestamp(origin)
    cutoff = _validate_timing(
        split=split_name,
        delivery=delivery,
        available=available,
        origin=origin_timestamp,
        hydro_training_cutoff_utc=hydro_training_cutoff_utc,
    )
    hydro_fill = _hydro_fill(frame["hydro_fill"])
    local = delivery.tz_convert("Europe/Zurich")
    calendar = enrich_15min_index(delivery, country="CH")
    is_holiday = calendar["type_jour"].isin(["Ferie_CH", "Ferie_DE"]).to_numpy(dtype=float)
    years_ahead = np.maximum(
        (delivery - origin_timestamp).total_seconds().to_numpy(dtype=float) / _SECONDS_PER_YEAR,
        0.0,
    )
    encoded = _encode_features(
        local.hour.to_numpy(dtype=float),
        local.month.to_numpy(dtype=float),
        local.dayofweek.to_numpy(dtype=float),
        is_holiday,
        hydro_fill,
        years_ahead,
    )[:, : len(CANONICAL_FEATURE_NAMES)]
    values = np.array(encoded, dtype=float, copy=True, order="C")
    values.setflags(write=False)
    audit = MappingProxyType(
        {
            "row_count": len(frame),
            "missing_hydro_row_count": int(np.isnan(hydro_fill).sum()),
            "value_sha256": _array_hash(values),
            "delivery_sha256": _timestamp_hash(delivery),
            "raw_entsoe_actual_column_count": 0,
            "outage_feature_column_count": 0,
            "warehouse_start_count": 0,
            "gpu_execution_count": 0,
        }
    )
    return ConstructedHourlyFeatures(
        origin_slot_id=origin_slot_id,
        origin_as_of_utc=origin,
        split=split_name,
        hydro_information_role=hydro_information_role,
        hydro_training_cutoff_utc=cutoff,
        delivery_at_utc=delivery,
        hydro_available_at_utc=available,
        feature_names=CANONICAL_FEATURE_NAMES,
        values=values,
        audit=audit,
    )


def _frozen_origin(
    origin_slot_id: str,
    origin_as_of_utc: str | datetime | pd.Timestamp,
) -> datetime:
    slot = next(
        (
            item
            for item in default_evaluation_protocol().holdout.origin_slots
            if item.slot_id == origin_slot_id
        ),
        None,
    )
    if slot is None:
        raise HourlyFeatureBuildError("origin_slot_id is not a frozen protocol slot")
    origin = _utc_scalar(origin_as_of_utc, label="origin")
    if origin.to_pydatetime() != slot.origin_as_of_utc:
        raise HourlyFeatureBuildError("origin timestamp differs from the frozen protocol slot")
    return slot.origin_as_of_utc


def _split(split: object, hydro_information_role: object) -> str:
    if split not in _SPLIT_ROLE:
        raise HourlyFeatureBuildError("split must be training or prediction")
    expected_role = _SPLIT_ROLE[str(split)]
    if hydro_information_role != expected_role:
        raise HourlyFeatureBuildError("hydro information role differs from split")
    return str(split)


def _exact_source(source: object) -> pd.DataFrame:
    if not isinstance(source, pd.DataFrame) or source.empty:
        raise HourlyFeatureBuildError("source must be a non-empty DataFrame")
    if source.columns.has_duplicates or tuple(source.columns) != _INPUT_COLUMNS:
        raise HourlyFeatureBuildError("source columns are not exact")
    return source.copy(deep=True)


def _validate_timing(
    *,
    split: str,
    delivery: pd.DatetimeIndex,
    available: pd.DatetimeIndex,
    origin: pd.Timestamp,
    hydro_training_cutoff_utc: object,
) -> datetime | None:
    if split == "training":
        if bool((delivery >= origin).any()):
            raise HourlyFeatureBuildError("training delivery must precede origin")
        if bool((available >= origin).any()):
            raise HourlyFeatureBuildError("training hydro must be available before origin")
        if hydro_training_cutoff_utc is not None:
            raise HourlyFeatureBuildError("training actual hydro cannot have a climatology cutoff")
        return None

    if bool((delivery < origin).any()):
        raise HourlyFeatureBuildError("prediction delivery cannot precede origin")
    if bool((available > origin).any()):
        raise HourlyFeatureBuildError("prediction hydro must be available by origin")
    if hydro_training_cutoff_utc is None:
        raise HourlyFeatureBuildError("prediction hydro requires a climatology cutoff")
    cutoff = _utc_scalar(hydro_training_cutoff_utc, label="hydro training cutoff")
    if cutoff >= origin:
        raise HourlyFeatureBuildError("hydro climatology cutoff must precede origin")
    if bool((available < cutoff).any()):
        raise HourlyFeatureBuildError("hydro climatology cannot be available before its cutoff")
    return cutoff.to_pydatetime()


def _utc_scalar(value: object, *, label: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise HourlyFeatureBuildError(f"{label} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise HourlyFeatureBuildError(f"{label} timestamp must be timezone-aware")
    return parsed.tz_convert("UTC")


def _utc_index(values: pd.Series, *, label: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([_utc_scalar(value, label=label) for value in values])


def _hydro_fill(values: pd.Series) -> np.ndarray:
    try:
        numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise HourlyFeatureBuildError("hydro_fill must be numeric or null") from exc
    finite = numeric[np.isfinite(numeric)]
    if bool(((finite < 0.0) | (finite > 1.0)).any()):
        raise HourlyFeatureBuildError("hydro_fill must use fraction unit in [0, 1]")
    if bool(np.isinf(numeric).any()):
        raise HourlyFeatureBuildError("hydro_fill cannot contain infinity")
    return numeric


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
    "FEATURE_BATCH_SCHEMA",
    "ConstructedHourlyFeatures",
    "HourlyFeatureBuildAuthority",
    "HourlyFeatureBuildError",
    "construct_hourly_features",
]

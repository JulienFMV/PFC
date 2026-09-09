"""Pure common-input boundary for the governed LT model comparison.

This module validates already materialized PRD frames. It has no connector,
feature invention, model fitting, truth-opening, scoring, or authority path.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from pfc_shaping.lt.evaluation_feature_builder import ConstructedHourlyFeatures
from pfc_shaping.lt.evaluation_feature_inventory import (
    CANONICAL_FEATURE_NAMES,
    default_hourly_feature_inventory,
)
from pfc_shaping.lt.evaluation_protocol import default_evaluation_protocol

EVALUATION_INPUT_SCHEMA = "fmv-lt-prd-origin-inputs.v1"
_TRAINING_METADATA = (
    "row_id",
    "delivery_at_utc",
    "available_at_utc",
    "target_f_h",
)
_PREDICTION_METADATA = ("row_id", "delivery_at_utc", "available_at_utc")
_TRAINING_ASSEMBLY_METADATA = (
    "row_id",
    "delivery_at_utc",
    "target_available_at_utc",
    "target_f_h",
)
_PREDICTION_ASSEMBLY_METADATA = ("row_id", "delivery_at_utc")
_NAME = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EvaluationInputError(ValueError):
    """Raised when one prepared evaluation input violates the common contract."""


@dataclass(frozen=True, slots=True)
class EvaluationInputAuthority:
    """Non-overridable negative authority for prepared inputs."""

    data_acquisition_authorized: bool = field(default=False, init=False)
    model_training_authorized: bool = field(default=False, init=False)
    truth_open_authorized: bool = field(default=False, init=False)
    model_selection_authorized: bool = field(default=False, init=False)
    monthly_level_change_authorized: bool = field(default=False, init=False)
    publication_authorized: bool = field(default=False, init=False)
    production_authorized: bool = field(default=False, init=False)

    def to_manifest(self) -> dict[str, bool]:
        return {
            "data_acquisition_authorized": self.data_acquisition_authorized,
            "model_training_authorized": self.model_training_authorized,
            "truth_open_authorized": self.truth_open_authorized,
            "model_selection_authorized": self.model_selection_authorized,
            "monthly_level_change_authorized": self.monthly_level_change_authorized,
            "publication_authorized": self.publication_authorized,
            "production_authorized": self.production_authorized,
        }


@dataclass(frozen=True, slots=True)
class PreparedOriginInputs:
    """Detached complete-case arrays shared by every model and reference."""

    origin_slot_id: str
    origin_as_of_utc: datetime
    feature_names: tuple[str, ...]
    source_snapshot_sha256: Mapping[str, str]
    training_row_ids: tuple[str, ...]
    training_delivery_at_utc: pd.DatetimeIndex
    training_available_at_utc: pd.DatetimeIndex
    training_features: np.ndarray
    training_target: np.ndarray
    prediction_row_ids: tuple[str, ...]
    prediction_delivery_at_utc: pd.DatetimeIndex
    prediction_available_at_utc: pd.DatetimeIndex
    prediction_features: np.ndarray
    audit: Mapping[str, object]
    authority: EvaluationInputAuthority = field(
        default_factory=EvaluationInputAuthority, init=False
    )

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": EVALUATION_INPUT_SCHEMA,
            "status": "PASS_PREPARED_PRD_INPUTS_NO_MODEL_OR_TRUTH_AUTHORITY",
            "source_environment": "PRD_DATABRICKS",
            "origin_slot_id": self.origin_slot_id,
            "origin_as_of_utc": self.origin_as_of_utc.isoformat(),
            "feature_names": list(self.feature_names),
            "feature_schema_policy": "EXACT_FROZEN_HOURLY_ORDER_SHARED_BY_ALL_MODELS",
            "complete_case_policy": "ONE_COMMON_MASK_BEFORE_ANY_MODEL_EXECUTION",
            "training_target": "INCUMBENT_EQUIVALENT_HOURLY_F_H",
            "direct_raw_price_fit_allowed": False,
            "prediction_truth_present": False,
            "source_snapshot_sha256": dict(self.source_snapshot_sha256),
            **dict(self.audit),
            "authority": self.authority.to_manifest(),
        }


def prepare_prd_origin_inputs(
    training_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    *,
    origin_slot_id: str,
    origin_as_of_utc: str | datetime | pd.Timestamp,
    feature_names: tuple[str, ...],
    source_snapshot_sha256: Mapping[str, str],
) -> PreparedOriginInputs:
    """Validate and detach one origin's common model/reference input arrays."""

    origin = _frozen_origin(origin_slot_id, origin_as_of_utc)
    features = _feature_names(feature_names)
    inventory = default_hourly_feature_inventory()
    if features != inventory.feature_names:
        raise EvaluationInputError("feature_names differ from the frozen hourly inventory")
    sources = _source_hashes(source_snapshot_sha256)
    training = _exact_frame(
        training_frame,
        (*_TRAINING_METADATA, *features),
        label="training",
    )
    prediction = _exact_frame(
        prediction_frame,
        (*_PREDICTION_METADATA, *features),
        label="prediction",
    )

    training_ids = _row_ids(training["row_id"], label="training")
    prediction_ids = _row_ids(prediction["row_id"], label="prediction")
    if set(training_ids) & set(prediction_ids):
        raise EvaluationInputError("training and prediction row IDs overlap")

    training_delivery = _utc_index(training["delivery_at_utc"], label="training delivery")
    training_available = _utc_index(training["available_at_utc"], label="training availability")
    prediction_delivery = _utc_index(prediction["delivery_at_utc"], label="prediction delivery")
    prediction_available = _utc_index(
        prediction["available_at_utc"], label="prediction availability"
    )
    origin_timestamp = pd.Timestamp(origin)
    if bool((training_delivery >= origin_timestamp).any()):
        raise EvaluationInputError("training delivery must precede origin")
    if bool((prediction_delivery < origin_timestamp).any()):
        raise EvaluationInputError("prediction delivery cannot precede origin")
    if bool((training_available >= origin_timestamp).any()):
        raise EvaluationInputError(
            "training features and target must be available strictly before origin"
        )
    if bool((prediction_available > origin_timestamp).any()):
        raise EvaluationInputError("prediction features must be available at or before origin")
    _unique_timestamps(training_delivery, label="training delivery")
    _unique_timestamps(prediction_delivery, label="prediction delivery")

    training_values = _numeric_matrix(training, features)
    prediction_values = _numeric_matrix(prediction, features)
    target = _numeric_vector(training["target_f_h"], label="training target")
    training_mask = np.isfinite(training_values).all(axis=1) & np.isfinite(target)
    prediction_mask = np.isfinite(prediction_values).all(axis=1)
    if int(training_mask.sum()) < 2:
        raise EvaluationInputError("training complete-case mask contains fewer than two rows")
    if not bool(prediction_mask.any()):
        raise EvaluationInputError("prediction complete-case mask is empty")

    training_order = _eligible_order(training_delivery, training_ids, training_mask)
    prediction_order = _eligible_order(prediction_delivery, prediction_ids, prediction_mask)
    prepared_training_features = _readonly(training_values[training_order])
    prepared_target = _readonly(target[training_order])
    prepared_prediction_features = _readonly(prediction_values[prediction_order])
    prepared_training_ids = tuple(training_ids[index] for index in training_order)
    prepared_prediction_ids = tuple(prediction_ids[index] for index in prediction_order)
    prepared_training_delivery = training_delivery[training_order]
    prepared_training_available = training_available[training_order]
    prepared_prediction_delivery = prediction_delivery[prediction_order]
    prepared_prediction_available = prediction_available[prediction_order]

    audit = MappingProxyType(
        {
            "training_rows": {
                "source_count": len(training),
                "eligible_count": len(training_order),
                "excluded_incomplete_count": len(training) - len(training_order),
                "row_identity_sha256": _identity_hash(
                    prepared_training_ids,
                    prepared_training_delivery,
                ),
                "feature_values_sha256": _array_hash(prepared_training_features),
                "target_values_sha256": _array_hash(prepared_target),
            },
            "prediction_rows": {
                "source_count": len(prediction),
                "eligible_count": len(prediction_order),
                "excluded_incomplete_count": len(prediction) - len(prediction_order),
                "row_identity_sha256": _identity_hash(
                    prepared_prediction_ids,
                    prepared_prediction_delivery,
                ),
                "feature_values_sha256": _array_hash(prepared_prediction_features),
            },
            "real_data_training_performed": False,
            "real_truth_opened": False,
            "ranking_or_selection_performed": False,
            "feature_inventory_semantic_sha256": inventory.semantic_sha256(),
            "feature_inventory_selection_authorized": False,
        }
    )
    return PreparedOriginInputs(
        origin_slot_id=origin_slot_id,
        origin_as_of_utc=origin,
        feature_names=features,
        source_snapshot_sha256=sources,
        training_row_ids=prepared_training_ids,
        training_delivery_at_utc=prepared_training_delivery,
        training_available_at_utc=prepared_training_available,
        training_features=prepared_training_features,
        training_target=prepared_target,
        prediction_row_ids=prepared_prediction_ids,
        prediction_delivery_at_utc=prepared_prediction_delivery,
        prediction_available_at_utc=prepared_prediction_available,
        prediction_features=prepared_prediction_features,
        audit=audit,
    )


def prepare_constructed_prd_origin_inputs(
    training_metadata: pd.DataFrame,
    prediction_metadata: pd.DataFrame,
    training_features: ConstructedHourlyFeatures,
    prediction_features: ConstructedHourlyFeatures,
    *,
    source_snapshot_sha256: Mapping[str, str],
) -> PreparedOriginInputs:
    """Align constructed features to row metadata and run the common validator."""

    _constructed_pair(training_features, prediction_features)
    training = _exact_frame(
        training_metadata,
        _TRAINING_ASSEMBLY_METADATA,
        label="training assembly metadata",
    )
    prediction = _exact_frame(
        prediction_metadata,
        _PREDICTION_ASSEMBLY_METADATA,
        label="prediction assembly metadata",
    )
    training_delivery = _utc_index(
        training["delivery_at_utc"],
        label="training assembly delivery",
    )
    prediction_delivery = _utc_index(
        prediction["delivery_at_utc"],
        label="prediction assembly delivery",
    )
    training_order = _batch_alignment(
        training_features,
        training_delivery,
        label="training",
    )
    prediction_order = _batch_alignment(
        prediction_features,
        prediction_delivery,
        label="prediction",
    )
    target_available = _utc_index(
        training["target_available_at_utc"],
        label="training target availability",
    )
    aligned_training_hydro_available = training_features.hydro_available_at_utc[training_order]
    training_available = pd.DatetimeIndex(
        pd.to_datetime(
            np.maximum(
                target_available.asi8,
                aligned_training_hydro_available.asi8,
            ),
            utc=True,
        )
    )
    aligned_prediction_hydro_available = prediction_features.hydro_available_at_utc[
        prediction_order
    ]
    training_frame = _assembled_training_frame(
        training,
        training_delivery,
        training_available,
        training_features.values[training_order],
    )
    prediction_frame = _assembled_prediction_frame(
        prediction,
        prediction_delivery,
        aligned_prediction_hydro_available,
        prediction_features.values[prediction_order],
    )
    prepared = prepare_prd_origin_inputs(
        training_frame,
        prediction_frame,
        origin_slot_id=training_features.origin_slot_id,
        origin_as_of_utc=training_features.origin_as_of_utc,
        feature_names=training_features.feature_names,
        source_snapshot_sha256=source_snapshot_sha256,
    )
    audit = {
        **dict(prepared.audit),
        "constructed_feature_batches": {
            "training_value_sha256": training_features.audit["value_sha256"],
            "training_delivery_sha256": training_features.audit["delivery_sha256"],
            "prediction_value_sha256": prediction_features.audit["value_sha256"],
            "prediction_delivery_sha256": prediction_features.audit["delivery_sha256"],
            "alignment_policy": "EXACT_TIMESTAMP_SET_THEN_METADATA_ORDER",
        },
    }
    return replace(prepared, audit=MappingProxyType(audit))


def _constructed_pair(
    training: object,
    prediction: object,
) -> None:
    if not isinstance(training, ConstructedHourlyFeatures) or not isinstance(
        prediction, ConstructedHourlyFeatures
    ):
        raise EvaluationInputError("constructed feature batches have invalid types")
    if training.split != "training" or prediction.split != "prediction":
        raise EvaluationInputError("constructed feature batch splits differ")
    _validate_constructed_batch(training, label="training")
    _validate_constructed_batch(prediction, label="prediction")
    if (
        training.origin_slot_id != prediction.origin_slot_id
        or training.origin_as_of_utc != prediction.origin_as_of_utc
    ):
        raise EvaluationInputError("constructed feature batch origins differ")
    if (
        training.feature_names != CANONICAL_FEATURE_NAMES
        or prediction.feature_names != CANONICAL_FEATURE_NAMES
    ):
        raise EvaluationInputError("constructed feature inventories differ")


def _validate_constructed_batch(
    batch: ConstructedHourlyFeatures,
    *,
    label: str,
) -> None:
    row_count = len(batch.delivery_at_utc)
    if (
        not isinstance(batch.delivery_at_utc, pd.DatetimeIndex)
        or batch.delivery_at_utc.tz is None
        or batch.delivery_at_utc.has_duplicates
        or not isinstance(batch.hydro_available_at_utc, pd.DatetimeIndex)
        or batch.hydro_available_at_utc.tz is None
        or len(batch.hydro_available_at_utc) != row_count
    ):
        raise EvaluationInputError(f"{label} constructed timestamps are invalid")
    if (
        not isinstance(batch.values, np.ndarray)
        or batch.values.shape != (row_count, len(CANONICAL_FEATURE_NAMES))
        or batch.values.flags.writeable
    ):
        raise EvaluationInputError(f"{label} constructed values are invalid")
    if not np.isfinite(batch.values[:, :7]).all() or not np.isfinite(batch.values[:, 8]).all():
        raise EvaluationInputError(f"{label} constructed deterministic features are invalid")
    hydro = batch.values[:, 7]
    finite_hydro = hydro[np.isfinite(hydro)]
    if bool(np.isinf(hydro).any()) or bool(((finite_hydro < 0.0) | (finite_hydro > 1.0)).any()):
        raise EvaluationInputError(f"{label} constructed hydro values are invalid")
    if not isinstance(batch.audit, Mapping):
        raise EvaluationInputError(f"{label} constructed audit is invalid")
    try:
        expected_value_hash = batch.audit["value_sha256"]
        expected_delivery_hash = batch.audit["delivery_sha256"]
    except KeyError as exc:
        raise EvaluationInputError(f"{label} constructed audit is incomplete") from exc
    if expected_value_hash != _array_hash(batch.values) or expected_delivery_hash != (
        _timestamp_hash(batch.delivery_at_utc)
    ):
        raise EvaluationInputError(f"{label} constructed batch hash differs")
    if any(batch.authority.to_manifest().values()):
        raise EvaluationInputError(f"{label} constructed authority is not negative")


def _batch_alignment(
    batch: ConstructedHourlyFeatures,
    delivery: pd.DatetimeIndex,
    *,
    label: str,
) -> np.ndarray:
    _unique_timestamps(delivery, label=f"{label} assembly delivery")
    source_positions = {
        timestamp.value: position for position, timestamp in enumerate(batch.delivery_at_utc)
    }
    if len(delivery) != len(source_positions) or set(delivery.asi8) != set(source_positions):
        raise EvaluationInputError(
            f"{label} metadata delivery population differs from constructed features"
        )
    return np.asarray(
        [source_positions[timestamp.value] for timestamp in delivery],
        dtype=np.int64,
    )


def _assembled_training_frame(
    metadata: pd.DataFrame,
    delivery: pd.DatetimeIndex,
    available: pd.DatetimeIndex,
    values: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row_id": metadata["row_id"].to_numpy(copy=True),
            "delivery_at_utc": delivery,
            "available_at_utc": available,
            "target_f_h": metadata["target_f_h"].to_numpy(copy=True),
        }
    )
    for position, feature_name in enumerate(CANONICAL_FEATURE_NAMES):
        frame[feature_name] = values[:, position]
    return frame


def _assembled_prediction_frame(
    metadata: pd.DataFrame,
    delivery: pd.DatetimeIndex,
    available: pd.DatetimeIndex,
    values: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row_id": metadata["row_id"].to_numpy(copy=True),
            "delivery_at_utc": delivery,
            "available_at_utc": available,
        }
    )
    for position, feature_name in enumerate(CANONICAL_FEATURE_NAMES):
        frame[feature_name] = values[:, position]
    return frame


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
        raise EvaluationInputError("origin_slot_id is not a frozen protocol slot")
    origin = _utc_scalar(origin_as_of_utc, label="origin")
    if origin.to_pydatetime() != slot.origin_as_of_utc:
        raise EvaluationInputError("origin timestamp differs from the frozen protocol slot")
    return slot.origin_as_of_utc


def _feature_names(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise EvaluationInputError("feature_names must be a non-empty tuple")
    if any(not isinstance(name, str) or not _NAME.fullmatch(name) for name in value):
        raise EvaluationInputError("feature_names contain an invalid identifier")
    if len(set(value)) != len(value):
        raise EvaluationInputError("feature_names must be unique")
    if set(value) & set((*_TRAINING_METADATA, *_PREDICTION_METADATA)):
        raise EvaluationInputError("feature_names overlap reserved metadata")
    return value


def _source_hashes(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise EvaluationInputError("source snapshot hashes must be a non-empty mapping")
    result: dict[str, str] = {}
    for role, digest in value.items():
        if not isinstance(role, str) or not _NAME.fullmatch(role):
            raise EvaluationInputError("source snapshot role is invalid")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise EvaluationInputError("source snapshot digest must be a lowercase SHA-256")
        result[role] = digest
    return MappingProxyType(dict(sorted(result.items())))


def _exact_frame(frame: object, columns: tuple[str, ...], *, label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise EvaluationInputError(f"{label} frame must be a non-empty DataFrame")
    if frame.columns.has_duplicates or tuple(frame.columns) != columns:
        raise EvaluationInputError(f"{label} columns are not exact")
    return frame.copy(deep=True)


def _row_ids(values: pd.Series, *, label: str) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value or len(value) > 128:
            raise EvaluationInputError(f"{label} row ID is invalid")
        result.append(value)
    if len(set(result)) != len(result):
        raise EvaluationInputError(f"{label} row IDs are duplicated")
    return tuple(result)


def _utc_scalar(value: object, *, label: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationInputError(f"{label} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise EvaluationInputError(f"{label} timestamp must be timezone-aware")
    return parsed.tz_convert("UTC")


def _utc_index(values: pd.Series, *, label: str) -> pd.DatetimeIndex:
    parsed = [_utc_scalar(value, label=label) for value in values]
    return pd.DatetimeIndex(parsed)


def _unique_timestamps(values: pd.DatetimeIndex, *, label: str) -> None:
    if values.has_duplicates:
        raise EvaluationInputError(f"{label} timestamps are duplicated")


def _numeric_matrix(frame: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    converted = frame.loc[:, list(names)].apply(pd.to_numeric, errors="coerce")
    return converted.to_numpy(dtype=float, copy=True)


def _numeric_vector(values: pd.Series, *, label: str) -> np.ndarray:
    converted = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, copy=True)
    if converted.ndim != 1:
        raise EvaluationInputError(f"{label} is not one-dimensional")
    return converted


def _eligible_order(
    delivery: pd.DatetimeIndex,
    row_ids: tuple[str, ...],
    mask: np.ndarray,
) -> np.ndarray:
    eligible = np.flatnonzero(mask)
    return np.asarray(
        sorted(eligible, key=lambda index: (delivery[index].value, row_ids[index])),
        dtype=np.int64,
    )


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


def _identity_hash(row_ids: tuple[str, ...], delivery: pd.DatetimeIndex) -> str:
    payload = [
        {"row_id": row_id, "delivery_at_utc": timestamp.isoformat()}
        for row_id, timestamp in zip(row_ids, delivery, strict=True)
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _timestamp_hash(values: pd.DatetimeIndex) -> str:
    payload = json.dumps(
        [value.isoformat() for value in values],
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CANONICAL_FEATURE_NAMES",
    "EVALUATION_INPUT_SCHEMA",
    "EvaluationInputAuthority",
    "EvaluationInputError",
    "PreparedOriginInputs",
    "prepare_constructed_prd_origin_inputs",
    "prepare_prd_origin_inputs",
]

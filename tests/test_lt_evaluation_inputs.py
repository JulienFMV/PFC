from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.evaluation_feature_builder import construct_hourly_features
from pfc_shaping.lt.evaluation_inputs import (
    CANONICAL_FEATURE_NAMES,
    EvaluationInputAuthority,
    EvaluationInputError,
    prepare_constructed_prd_origin_inputs,
    prepare_prd_origin_inputs,
)
from pfc_shaping.lt.evaluation_protocol import default_evaluation_protocol

ROOT = Path(__file__).resolve().parents[1]
FEATURES = CANONICAL_FEATURE_NAMES
SNAPSHOTS = {
    "calendar": "1" * 64,
    "entsoe_pit": "2" * 64,
    "spot_truth": "3" * 64,
}


def _origin():
    return default_evaluation_protocol().holdout.origin_slots[0]


def _training_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row_id": ["train-1", "train-2", "train-incomplete"],
            "delivery_at_utc": [
                "2026-09-01T00:00:00Z",
                "2026-09-01T01:00:00Z",
                "2026-09-01T02:00:00Z",
            ],
            "available_at_utc": [
                "2026-09-02T00:00:00Z",
                "2026-09-02T01:00:00Z",
                "2026-09-02T02:00:00Z",
            ],
            "target_f_h": [0.9, 1.0, 1.1],
        }
    )
    for position, feature in enumerate(FEATURES):
        frame[feature] = [position, position + 0.25, position + 0.5]
    frame.loc[2, FEATURES[0]] = np.nan
    return frame.loc[:, ["row_id", "delivery_at_utc", "available_at_utc", "target_f_h", *FEATURES]]


def _prediction_frame() -> pd.DataFrame:
    origin = _origin().origin_as_of_utc
    frame = pd.DataFrame(
        {
            "row_id": ["predict-1", "predict-incomplete"],
            "delivery_at_utc": [
                "2026-11-01T00:00:00Z",
                "2026-11-01T01:00:00Z",
            ],
            "available_at_utc": [origin.isoformat(), origin.isoformat()],
        }
    )
    for position, feature in enumerate(FEATURES):
        frame[feature] = [position + 0.5, position + 0.75]
    frame.loc[1, FEATURES[-1]] = np.nan
    return frame.loc[:, ["row_id", "delivery_at_utc", "available_at_utc", *FEATURES]]


def _prepare(
    training: pd.DataFrame | None = None,
    prediction: pd.DataFrame | None = None,
):
    origin = _origin()
    return prepare_prd_origin_inputs(
        _training_frame() if training is None else training,
        _prediction_frame() if prediction is None else prediction,
        origin_slot_id=origin.slot_id,
        origin_as_of_utc=origin.origin_as_of_utc,
        feature_names=FEATURES,
        source_snapshot_sha256=SNAPSHOTS,
    )


def test_adapter_builds_one_shared_explicit_feature_schema_and_masks_once() -> None:
    prepared = _prepare()
    manifest = prepared.to_manifest()

    assert prepared.feature_names == FEATURES
    assert prepared.training_row_ids == ("train-1", "train-2")
    assert prepared.prediction_row_ids == ("predict-1",)
    assert prepared.training_features.shape == (2, 9)
    assert prepared.prediction_features.shape == (1, 9)
    assert manifest["source_environment"] == "PRD_DATABRICKS"
    assert manifest["feature_schema_policy"] == ("EXACT_FROZEN_HOURLY_ORDER_SHARED_BY_ALL_MODELS")
    assert manifest["complete_case_policy"] == "ONE_COMMON_MASK_BEFORE_ANY_MODEL_EXECUTION"
    assert manifest["training_rows"]["source_count"] == 3
    assert manifest["training_rows"]["eligible_count"] == 2
    assert manifest["prediction_rows"]["source_count"] == 2
    assert manifest["prediction_rows"]["eligible_count"] == 1
    assert manifest["prediction_truth_present"] is False


def test_adapter_requires_exact_frame_columns_and_never_accepts_future_truth() -> None:
    prediction = _prediction_frame()
    prediction["target_f_h"] = [0.9, 1.1]
    with pytest.raises(EvaluationInputError, match="prediction columns are not exact"):
        _prepare(prediction=prediction)

    reordered = _training_frame()[
        [
            "row_id",
            "delivery_at_utc",
            "available_at_utc",
            "target_f_h",
            *reversed(FEATURES),
        ]
    ]
    with pytest.raises(EvaluationInputError, match="training columns are not exact"):
        _prepare(training=reordered)


@pytest.mark.parametrize("split", ["training", "prediction"])
def test_adapter_rejects_any_feature_availability_after_origin(split: str) -> None:
    origin = _origin().origin_as_of_utc
    training = _training_frame()
    prediction = _prediction_frame()
    if split == "training":
        training.loc[0, "available_at_utc"] = origin.isoformat()
    else:
        prediction.loc[0, "available_at_utc"] = (
            pd.Timestamp(origin) + pd.Timedelta(seconds=1)
        ).isoformat()

    message = (
        "available strictly before origin"
        if split == "training"
        else "available at or before origin"
    )
    with pytest.raises(EvaluationInputError, match=message):
        _prepare(training=training, prediction=prediction)


def test_adapter_rejects_delivery_on_the_wrong_side_of_origin() -> None:
    origin = _origin().origin_as_of_utc
    training = _training_frame()
    training.loc[0, "delivery_at_utc"] = origin.isoformat()
    with pytest.raises(EvaluationInputError, match="training delivery must precede origin"):
        _prepare(training=training)

    prediction = _prediction_frame()
    prediction.loc[0, "delivery_at_utc"] = "2026-09-01T00:00:00Z"
    with pytest.raises(EvaluationInputError, match="prediction delivery cannot precede origin"):
        _prepare(prediction=prediction)


def test_adapter_binds_a_real_frozen_origin_and_hash_only_source_provenance() -> None:
    origin = _origin()
    with pytest.raises(EvaluationInputError, match="frozen protocol slot"):
        prepare_prd_origin_inputs(
            _training_frame(),
            _prediction_frame(),
            origin_slot_id="origin-unknown",
            origin_as_of_utc=origin.origin_as_of_utc,
            feature_names=FEATURES,
            source_snapshot_sha256=SNAPSHOTS,
        )
    with pytest.raises(EvaluationInputError, match="lowercase SHA-256"):
        prepare_prd_origin_inputs(
            _training_frame(),
            _prediction_frame(),
            origin_slot_id=origin.slot_id,
            origin_as_of_utc=origin.origin_as_of_utc,
            feature_names=FEATURES,
            source_snapshot_sha256={"entsoe_pit": "invalid"},
        )


def test_prepared_arrays_are_detached_read_only_and_authority_negative() -> None:
    training = _training_frame()
    prepared = _prepare(training=training)
    training.loc[0, FEATURES[0]] = 99.0

    assert prepared.training_features[0, 0] == 0.0
    assert prepared.training_features.flags.writeable is False
    assert prepared.training_target.flags.writeable is False
    assert prepared.prediction_features.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        prepared.training_features[0, 0] = 1.0

    authority = prepared.authority
    assert authority == EvaluationInputAuthority()
    assert not any(authority.to_manifest().values())
    with pytest.raises(FrozenInstanceError):
        authority.model_training_authorized = True  # type: ignore[misc]


def test_adapter_output_is_stable_under_source_row_order() -> None:
    first = _prepare()
    second = _prepare(
        training=_training_frame().sample(frac=1.0, random_state=17),
        prediction=_prediction_frame().sample(frac=1.0, random_state=19),
    )

    assert first.training_row_ids == second.training_row_ids
    assert first.prediction_row_ids == second.prediction_row_ids
    np.testing.assert_array_equal(first.training_features, second.training_features)
    np.testing.assert_array_equal(first.training_target, second.training_target)
    np.testing.assert_array_equal(first.prediction_features, second.prediction_features)
    assert first.to_manifest() == second.to_manifest()


def test_adapter_rejects_a_valid_but_noncanonical_feature_inventory() -> None:
    origin = _origin()
    with pytest.raises(EvaluationInputError, match="frozen hourly inventory"):
        prepare_prd_origin_inputs(
            _training_frame(),
            _prediction_frame(),
            origin_slot_id=origin.slot_id,
            origin_as_of_utc=origin.origin_as_of_utc,
            feature_names=tuple(reversed(FEATURES)),
            source_snapshot_sha256=SNAPSHOTS,
        )


def test_adapter_has_no_io_ct_fit_or_gpu_execution_path() -> None:
    source = (ROOT / "pfc_shaping/lt/evaluation_inputs.py").read_text(encoding="utf-8").lower()
    forbidden = (
        "pfc_shaping.ct",
        "read_parquet",
        "read_csv",
        "to_parquet",
        "to_csv",
        "requests.",
        ".fit(",
        "cuda",
    )
    assert not any(fragment in source for fragment in forbidden)


def _constructed_batches():
    origin = _origin()
    origin_timestamp = pd.Timestamp(origin.origin_as_of_utc)
    training = construct_hourly_features(
        pd.DataFrame(
            {
                "delivery_at_utc": [
                    "2026-09-01T00:00:00Z",
                    "2026-09-01T01:00:00Z",
                ],
                "hydro_available_at_utc": [
                    (origin_timestamp - pd.Timedelta(days=2)).isoformat(),
                    (origin_timestamp - pd.Timedelta(days=1)).isoformat(),
                ],
                "hydro_fill": [0.4, 0.5],
            }
        ),
        origin_slot_id=origin.slot_id,
        origin_as_of_utc=origin.origin_as_of_utc,
        split="training",
        hydro_information_role="REALIZED_ACTUAL",
    )
    prediction = construct_hourly_features(
        pd.DataFrame(
            {
                "delivery_at_utc": [
                    "2026-11-01T00:00:00Z",
                    "2026-11-01T01:00:00Z",
                ],
                "hydro_available_at_utc": [origin_timestamp.isoformat()] * 2,
                "hydro_fill": [0.6, 0.7],
            }
        ),
        origin_slot_id=origin.slot_id,
        origin_as_of_utc=origin.origin_as_of_utc,
        split="prediction",
        hydro_information_role="ORIGIN_FROZEN_CLIMATOLOGY",
        hydro_training_cutoff_utc=origin_timestamp - pd.Timedelta(seconds=1),
    )
    return training, prediction


def _assembly_metadata():
    origin_timestamp = pd.Timestamp(_origin().origin_as_of_utc)
    training = pd.DataFrame(
        {
            "row_id": ["train-2", "train-1"],
            "delivery_at_utc": [
                "2026-09-01T01:00:00Z",
                "2026-09-01T00:00:00Z",
            ],
            "target_available_at_utc": [
                (origin_timestamp - pd.Timedelta(hours=12)).isoformat(),
                (origin_timestamp - pd.Timedelta(hours=18)).isoformat(),
            ],
            "target_f_h": [1.1, 0.9],
        }
    )
    prediction = pd.DataFrame(
        {
            "row_id": ["predict-2", "predict-1"],
            "delivery_at_utc": [
                "2026-11-01T01:00:00Z",
                "2026-11-01T00:00:00Z",
            ],
        }
    )
    return training, prediction


def _prepare_constructed(
    training_metadata: pd.DataFrame | None = None,
    prediction_metadata: pd.DataFrame | None = None,
):
    training_features, prediction_features = _constructed_batches()
    default_training, default_prediction = _assembly_metadata()
    return prepare_constructed_prd_origin_inputs(
        default_training if training_metadata is None else training_metadata,
        default_prediction if prediction_metadata is None else prediction_metadata,
        training_features,
        prediction_features,
        source_snapshot_sha256=SNAPSHOTS,
    )


def test_constructed_adapter_aligns_by_timestamp_and_delegates_common_validation() -> None:
    prepared = _prepare_constructed()
    origin_timestamp = pd.Timestamp(_origin().origin_as_of_utc)

    assert prepared.training_row_ids == ("train-1", "train-2")
    assert prepared.prediction_row_ids == ("predict-1", "predict-2")
    np.testing.assert_array_equal(prepared.training_features[:, 7], [0.4, 0.5])
    np.testing.assert_array_equal(prepared.prediction_features[:, 7], [0.6, 0.7])
    assert prepared.training_available_at_utc.equals(
        pd.DatetimeIndex(
            [
                origin_timestamp - pd.Timedelta(hours=18),
                origin_timestamp - pd.Timedelta(hours=12),
            ]
        )
    )
    assert prepared.prediction_available_at_utc.equals(
        pd.DatetimeIndex([origin_timestamp, origin_timestamp])
    )
    assert prepared.audit["constructed_feature_batches"]["alignment_policy"] == (
        "EXACT_TIMESTAMP_SET_THEN_METADATA_ORDER"
    )


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate"])
def test_constructed_adapter_rejects_nonidentical_delivery_population(kind: str) -> None:
    training, prediction = _assembly_metadata()
    if kind == "missing":
        training = training.iloc[:1].copy()
    elif kind == "extra":
        extra = training.iloc[[0]].copy()
        extra["row_id"] = "train-extra"
        extra["delivery_at_utc"] = "2026-09-01T02:00:00Z"
        training = pd.concat([training, extra], ignore_index=True)
    else:
        training.loc[1, "delivery_at_utc"] = training.loc[0, "delivery_at_utc"]

    message = "timestamps are duplicated" if kind == "duplicate" else "population differs"
    with pytest.raises(EvaluationInputError, match=message):
        _prepare_constructed(training_metadata=training, prediction_metadata=prediction)


def test_constructed_adapter_rejects_extra_metadata_and_future_target_availability() -> None:
    training, prediction = _assembly_metadata()
    training["load_mw"] = 8_000.0
    with pytest.raises(EvaluationInputError, match="columns are not exact"):
        _prepare_constructed(training_metadata=training, prediction_metadata=prediction)

    training, prediction = _assembly_metadata()
    training.loc[0, "target_available_at_utc"] = _origin().origin_as_of_utc.isoformat()
    with pytest.raises(EvaluationInputError, match="available strictly before origin"):
        _prepare_constructed(training_metadata=training, prediction_metadata=prediction)


def test_constructed_adapter_rejects_swapped_splits_or_different_origins() -> None:
    training_features, prediction_features = _constructed_batches()
    training, prediction = _assembly_metadata()
    with pytest.raises(EvaluationInputError, match="splits differ"):
        prepare_constructed_prd_origin_inputs(
            training,
            prediction,
            prediction_features,
            training_features,
            source_snapshot_sha256=SNAPSHOTS,
        )

    second_origin = default_evaluation_protocol().holdout.origin_slots[1]
    second_origin_timestamp = pd.Timestamp(second_origin.origin_as_of_utc)
    different_prediction = construct_hourly_features(
        pd.DataFrame(
            {
                "delivery_at_utc": [
                    second_origin_timestamp + pd.Timedelta(days=1),
                    second_origin_timestamp + pd.Timedelta(days=1, hours=1),
                ],
                "hydro_available_at_utc": [second_origin_timestamp] * 2,
                "hydro_fill": [0.6, 0.7],
            }
        ),
        origin_slot_id=second_origin.slot_id,
        origin_as_of_utc=second_origin.origin_as_of_utc,
        split="prediction",
        hydro_information_role="ORIGIN_FROZEN_CLIMATOLOGY",
        hydro_training_cutoff_utc=second_origin_timestamp - pd.Timedelta(seconds=1),
    )
    with pytest.raises(EvaluationInputError, match="origins differ"):
        prepare_constructed_prd_origin_inputs(
            training,
            prediction,
            training_features,
            different_prediction,
            source_snapshot_sha256=SNAPSHOTS,
        )


def test_constructed_adapter_output_remains_read_only_and_authority_negative() -> None:
    prepared = _prepare_constructed()

    assert prepared.training_features.flags.writeable is False
    assert prepared.prediction_features.flags.writeable is False
    assert prepared.training_target.flags.writeable is False
    assert not any(prepared.authority.to_manifest().values())


def test_constructed_adapter_reverifies_batch_values_against_its_audit_hash() -> None:
    training_features, prediction_features = _constructed_batches()
    training, prediction = _assembly_metadata()
    tampered_values = np.array(training_features.values, copy=True)
    tampered_values[0, 7] = 0.9
    tampered_values.setflags(write=False)
    tampered = replace(training_features, values=tampered_values)

    with pytest.raises(EvaluationInputError, match="batch hash differs"):
        prepare_constructed_prd_origin_inputs(
            training,
            prediction,
            tampered,
            prediction_features,
            source_snapshot_sha256=SNAPSHOTS,
        )

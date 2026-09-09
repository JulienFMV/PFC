from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pfc_shaping.lt.model.shape_hourly_mlp as incumbent_module
from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.evaluation_feature_builder import construct_hourly_features
from pfc_shaping.lt.evaluation_inputs import prepare_constructed_prd_origin_inputs
from pfc_shaping.lt.evaluation_protocol import default_evaluation_protocol
from pfc_shaping.lt.evaluation_target_builder import (
    HourlyTargetBuildAuthority,
    HourlyTargetBuildError,
    construct_incumbent_equivalent_hourly_targets,
)

ROOT = Path(__file__).resolve().parents[1]


def _origin():
    return default_evaluation_protocol().holdout.origin_slots[1]


def _source(*, fallback: bool = False) -> pd.DataFrame:
    start = "2026-10-25T00:00:00Z" if fallback else "2026-10-20T00:00:00Z"
    periods = 100 if fallback else 192
    delivery = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    local = delivery.tz_convert("Europe/Zurich")
    price = 50.0 + local.hour.to_numpy(dtype=float) + local.minute.to_numpy(dtype=float) / 60.0
    available = pd.Timestamp(_origin().origin_as_of_utc) - pd.Timedelta(days=1)
    return pd.DataFrame(
        {
            "delivery_at_utc": delivery,
            "price_available_at_utc": [available] * len(delivery),
            "price_eur_mwh": price,
        }
    )


def _build(source: pd.DataFrame | None = None):
    origin = _origin()
    return construct_incumbent_equivalent_hourly_targets(
        _source() if source is None else source,
        origin_slot_id=origin.slot_id,
        origin_as_of_utc=origin.origin_as_of_utc,
    )


def test_builder_matches_the_hash_bound_incumbent_target_and_features(monkeypatch) -> None:
    source = _source(fallback=True)
    captured: dict[str, np.ndarray] = {}

    class CaptureRegressor:
        def __init__(self, **_kwargs) -> None:
            self.n_iter_ = 1
            self.loss_ = 0.0

        def fit(self, features, target):
            captured["features"] = np.array(features, copy=True)
            captured["target"] = np.array(target, copy=True)
            return self

        def predict(self, features):
            return np.ones(len(features), dtype=float)

    monkeypatch.setattr(incumbent_module, "MLPRegressor", CaptureRegressor)
    delivery = pd.DatetimeIndex(source["delivery_at_utc"])
    calendar = enrich_15min_index(delivery, country="CH")
    incumbent_module.ShapeHourlyMLP().fit(
        pd.DataFrame(
            {"price_eur_mwh": source["price_eur_mwh"].to_numpy()},
            index=delivery,
        ),
        calendar,
    )

    built = _build(source)
    feature_batch = construct_hourly_features(
        pd.DataFrame(
            {
                "delivery_at_utc": built.delivery_at_utc,
                "hydro_available_at_utc": built.target_available_at_utc,
                "hydro_fill": np.full(len(built.values), 0.5),
            }
        ),
        origin_slot_id=_origin().slot_id,
        origin_as_of_utc=_origin().origin_as_of_utc,
        split="training",
        hydro_information_role="REALIZED_ACTUAL",
    )

    local = delivery.tz_convert("Europe/Zurich")
    native_keys = np.asarray([f"{timestamp:%Y-%m-%d}-{timestamp.hour}" for timestamp in local])
    unique_native_keys = np.unique(native_keys)
    native_position = {key: position for position, key in enumerate(unique_native_keys)}
    built_keys = [
        f"{timestamp:%Y-%m-%d}-{timestamp.hour}"
        for timestamp in built.delivery_at_utc.tz_convert("Europe/Zurich")
    ]
    native_order = np.asarray([native_position[key] for key in built_keys])

    np.testing.assert_allclose(built.values, captured["target"][native_order], atol=1e-15)
    np.testing.assert_allclose(
        feature_batch.values,
        captured["features"][native_order, :9],
        atol=1e-15,
    )
    assert built.audit["repeated_fallback_group_count"] == 1
    assert len(built.values) == 24


def test_builder_applies_daily_threshold_then_ratio_clip_before_aggregation() -> None:
    source = _source().iloc[:8].copy()
    source.loc[source.index[:4], "delivery_at_utc"] = pd.date_range(
        "2026-10-20T00:00:00Z", periods=4, freq="15min", tz="UTC"
    )
    source.loc[source.index[4:], "delivery_at_utc"] = pd.date_range(
        "2026-10-21T00:00:00Z", periods=4, freq="15min", tz="UTC"
    )
    source.loc[source.index[:4], "price_eur_mwh"] = 4.0
    source.loc[source.index[4:], "price_eur_mwh"] = [1.0, 10.0, 100.0, 100.0]

    built = _build(source)

    assert len(built.values) == 1
    raw = np.asarray([1.0, 10.0, 100.0, 100.0]) / 52.75
    expected = np.average(
        np.clip(raw, 0.2, 3.0),
        weights=np.exp(-np.log(2.0) * np.asarray([0.75, 0.5, 0.25, 0.0]) / 24 / 180),
    )
    assert built.values[0] == pytest.approx(expected)
    assert built.audit["excluded_quarter_hour_row_count"] == 4


def test_builder_metadata_is_exact_detached_and_authority_negative() -> None:
    source = _source()
    built = _build(source)
    metadata = built.to_metadata_frame()
    source.loc[0, "price_eur_mwh"] = 999.0
    metadata.loc[0, "target_f_h"] = 999.0

    assert tuple(metadata.columns) == (
        "row_id",
        "delivery_at_utc",
        "target_available_at_utc",
        "target_f_h",
    )
    assert built.values.flags.writeable is False
    assert built.values[0] != 999.0
    assert built.to_manifest()["direct_raw_price_fit_allowed"] is False
    assert built.authority == HourlyTargetBuildAuthority()
    assert not any(built.authority.to_manifest().values())
    with pytest.raises(FrozenInstanceError):
        built.authority.model_training_authorized = True  # type: ignore[misc]


def test_built_targets_feed_the_common_assembly_without_relabelling_price() -> None:
    built = _build()
    training_features = construct_hourly_features(
        pd.DataFrame(
            {
                "delivery_at_utc": built.delivery_at_utc,
                "hydro_available_at_utc": built.target_available_at_utc,
                "hydro_fill": np.full(len(built.values), 0.5),
            }
        ),
        origin_slot_id=_origin().slot_id,
        origin_as_of_utc=_origin().origin_as_of_utc,
        split="training",
        hydro_information_role="REALIZED_ACTUAL",
    )
    prediction_delivery = pd.date_range(
        "2026-12-01T00:00:00Z",
        periods=2,
        freq="15min",
        tz="UTC",
    )
    prediction_features = construct_hourly_features(
        pd.DataFrame(
            {
                "delivery_at_utc": prediction_delivery,
                "hydro_available_at_utc": [pd.Timestamp(_origin().origin_as_of_utc)] * 2,
                "hydro_fill": [0.5, 0.5],
            }
        ),
        origin_slot_id=_origin().slot_id,
        origin_as_of_utc=_origin().origin_as_of_utc,
        split="prediction",
        hydro_information_role="ORIGIN_FROZEN_CLIMATOLOGY",
        hydro_training_cutoff_utc=(
            pd.Timestamp(_origin().origin_as_of_utc) - pd.Timedelta(seconds=1)
        ),
    )
    prepared = prepare_constructed_prd_origin_inputs(
        built.to_metadata_frame(),
        pd.DataFrame(
            {
                "row_id": ["predict-1", "predict-2"],
                "delivery_at_utc": prediction_delivery,
            }
        ),
        training_features,
        prediction_features,
        source_snapshot_sha256={"spot_truth": "1" * 64, "hydro": "2" * 64},
    )

    np.testing.assert_array_equal(prepared.training_target, built.values)
    assert prepared.to_manifest()["training_target"] == ("INCUMBENT_EQUIVALENT_HOURLY_F_H")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra", "columns are not exact"),
        ("duplicate", "timestamps are duplicated"),
        ("off_grid", "quarter-hour grid"),
        ("future_delivery", "delivery must precede origin"),
        ("future_availability", "availability must be strictly before origin"),
        ("infinite", "cannot contain infinity"),
    ],
)
def test_builder_fails_closed_on_invalid_source(mutation: str, message: str) -> None:
    source = _source()
    if mutation == "extra":
        source["market"] = "CH"
    elif mutation == "duplicate":
        source.loc[1, "delivery_at_utc"] = source.loc[0, "delivery_at_utc"]
    elif mutation == "off_grid":
        source.loc[0, "delivery_at_utc"] = pd.Timestamp(
            source.loc[0, "delivery_at_utc"]
        ) + pd.Timedelta(minutes=1)
    elif mutation == "future_delivery":
        source.loc[0, "delivery_at_utc"] = _origin().origin_as_of_utc
    elif mutation == "future_availability":
        source.loc[0, "price_available_at_utc"] = _origin().origin_as_of_utc
    else:
        source.loc[0, "price_eur_mwh"] = np.inf

    with pytest.raises(HourlyTargetBuildError, match=message):
        _build(source)


def test_builder_rejects_unknown_or_mismatched_origin() -> None:
    origin = _origin()
    with pytest.raises(HourlyTargetBuildError, match="frozen protocol slot"):
        construct_incumbent_equivalent_hourly_targets(
            _source(),
            origin_slot_id="origin-unknown",
            origin_as_of_utc=origin.origin_as_of_utc,
        )
    with pytest.raises(HourlyTargetBuildError, match="timestamp differs"):
        construct_incumbent_equivalent_hourly_targets(
            _source(),
            origin_slot_id=origin.slot_id,
            origin_as_of_utc=pd.Timestamp(origin.origin_as_of_utc) - pd.Timedelta(seconds=1),
        )


def test_builder_has_no_io_fit_ct_or_gpu_execution_path() -> None:
    source = (
        (ROOT / "pfc_shaping/lt/evaluation_target_builder.py").read_text(encoding="utf-8").lower()
    )
    forbidden = (
        "pfc_shaping.ct",
        "read_parquet",
        "read_csv",
        "to_parquet",
        "to_csv",
        "requests.",
        ".fit(",
        ".predict(",
        "cuda",
    )
    assert not any(fragment in source for fragment in forbidden)

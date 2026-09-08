from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.evaluation_feature_builder import (
    HourlyFeatureBuildAuthority,
    HourlyFeatureBuildError,
    construct_hourly_features,
)
from pfc_shaping.lt.evaluation_feature_inventory import (
    CANONICAL_FEATURE_INVENTORY_SEMANTIC_SHA256,
    CANONICAL_FEATURE_NAMES,
)
from pfc_shaping.lt.evaluation_protocol import default_evaluation_protocol
from pfc_shaping.lt.model.shape_hourly_mlp import _encode_features

ROOT = Path(__file__).resolve().parents[1]


def _origin():
    return default_evaluation_protocol().holdout.origin_slots[0]


def _source(*, prediction: bool = False) -> pd.DataFrame:
    origin = pd.Timestamp(_origin().origin_as_of_utc)
    delivery = (
        ["2026-12-25T00:00:00Z", "2027-03-28T01:00:00Z"]
        if prediction
        else ["2026-09-01T00:00:00Z", "2026-09-01T01:00:00Z"]
    )
    available = origin if prediction else origin - pd.Timedelta(days=1)
    return pd.DataFrame(
        {
            "delivery_at_utc": delivery,
            "hydro_available_at_utc": [available.isoformat()] * 2,
            "hydro_fill": [0.45, 0.55],
        }
    )


def _build(source: pd.DataFrame | None = None, *, prediction: bool = False):
    origin = _origin()
    return construct_hourly_features(
        _source(prediction=prediction) if source is None else source,
        origin_slot_id=origin.slot_id,
        origin_as_of_utc=origin.origin_as_of_utc,
        split="prediction" if prediction else "training",
        hydro_information_role=("ORIGIN_FROZEN_CLIMATOLOGY" if prediction else "REALIZED_ACTUAL"),
        hydro_training_cutoff_utc=(
            pd.Timestamp(origin.origin_as_of_utc) - pd.Timedelta(seconds=1) if prediction else None
        ),
    )


@pytest.mark.parametrize("prediction", [False, True])
def test_builder_matches_the_incumbent_nine_feature_encoding(prediction: bool) -> None:
    source = _source(prediction=prediction)
    built = _build(source, prediction=prediction)
    delivery = pd.DatetimeIndex(pd.to_datetime(source["delivery_at_utc"], utc=True))
    local = delivery.tz_convert("Europe/Zurich")
    calendar = enrich_15min_index(delivery, country="CH")
    origin = pd.Timestamp(_origin().origin_as_of_utc)
    years_ahead = np.maximum(
        (delivery - origin).total_seconds().to_numpy(dtype=float) / (365.25 * 86_400.0),
        0.0,
    )
    expected = _encode_features(
        local.hour.to_numpy(dtype=float),
        local.month.to_numpy(dtype=float),
        local.dayofweek.to_numpy(dtype=float),
        calendar["type_jour"].isin(["Ferie_CH", "Ferie_DE"]).to_numpy(dtype=float),
        source["hydro_fill"].to_numpy(dtype=float),
        years_ahead,
    )[:, :9]

    assert built.feature_names == CANONICAL_FEATURE_NAMES
    assert built.to_manifest()["feature_inventory_semantic_sha256"] == (
        CANONICAL_FEATURE_INVENTORY_SEMANTIC_SHA256
    )
    np.testing.assert_array_equal(built.values, expected)
    assert not set(("load_mw", "solar_mw", "wind_mw")).intersection(built.feature_names)


def test_builder_uses_swiss_local_time_and_valais_or_german_holidays() -> None:
    source = _source(prediction=True)
    built = _build(source, prediction=True)

    assert built.values[0, 0] == pytest.approx(np.sin(2 * np.pi * 1 / 24))
    assert built.values[0, 6] == 1.0
    assert built.values[1, 0] == pytest.approx(np.sin(2 * np.pi * 3 / 24))


def test_builder_preserves_missing_hydro_for_the_downstream_common_mask() -> None:
    source = _source()
    source.loc[1, "hydro_fill"] = np.nan
    built = _build(source)

    assert np.isnan(built.values[1, 7])
    assert built.to_manifest()["missing_hydro_row_count"] == 1


@pytest.mark.parametrize("value", [-0.01, 1.01, np.inf, "not-numeric"])
def test_builder_rejects_wrong_hydro_unit_or_non_numeric_values(value: object) -> None:
    source = _source()
    if isinstance(value, str):
        source["hydro_fill"] = source["hydro_fill"].astype(object)
    source.loc[0, "hydro_fill"] = value
    with pytest.raises(HourlyFeatureBuildError, match="hydro_fill"):
        _build(source)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("load_mw", 8_000.0),
        ("unavailable_nuclear", 0.0),
    ],
)
def test_builder_rejects_raw_entsoe_or_outage_columns(column: str, value: float) -> None:
    source = _source()
    source[column] = value
    with pytest.raises(HourlyFeatureBuildError, match="source columns are not exact"):
        _build(source)


def test_builder_enforces_split_role_delivery_availability_and_cutoff() -> None:
    origin = _origin()
    with pytest.raises(HourlyFeatureBuildError, match="role differs"):
        construct_hourly_features(
            _source(),
            origin_slot_id=origin.slot_id,
            origin_as_of_utc=origin.origin_as_of_utc,
            split="training",
            hydro_information_role="ORIGIN_FROZEN_CLIMATOLOGY",
        )

    training = _source()
    training.loc[0, "delivery_at_utc"] = origin.origin_as_of_utc.isoformat()
    with pytest.raises(HourlyFeatureBuildError, match="training delivery"):
        _build(training)

    prediction = _source(prediction=True)
    prediction.loc[0, "hydro_available_at_utc"] = (
        pd.Timestamp(origin.origin_as_of_utc) + pd.Timedelta(seconds=1)
    ).isoformat()
    with pytest.raises(HourlyFeatureBuildError, match="available by origin"):
        _build(prediction, prediction=True)

    with pytest.raises(HourlyFeatureBuildError, match="cutoff must precede origin"):
        construct_hourly_features(
            _source(prediction=True),
            origin_slot_id=origin.slot_id,
            origin_as_of_utc=origin.origin_as_of_utc,
            split="prediction",
            hydro_information_role="ORIGIN_FROZEN_CLIMATOLOGY",
            hydro_training_cutoff_utc=origin.origin_as_of_utc,
        )


def test_builder_is_detached_read_only_deterministic_and_authority_negative() -> None:
    source = _source()
    first = _build(source)
    source.loc[0, "hydro_fill"] = 0.99
    second = _build(_source())

    assert first.values.flags.writeable is False
    assert first.values[0, 7] == 0.45
    assert first.to_manifest() == second.to_manifest()
    assert first.authority == HourlyFeatureBuildAuthority()
    assert not any(first.authority.to_manifest().values())
    with pytest.raises(ValueError, match="read-only"):
        first.values[0, 0] = 0.0
    with pytest.raises(FrozenInstanceError):
        first.authority.model_training_authorized = True  # type: ignore[misc]


def test_builder_rejects_unknown_or_mismatched_frozen_origin() -> None:
    origin = _origin()
    with pytest.raises(HourlyFeatureBuildError, match="frozen protocol slot"):
        construct_hourly_features(
            _source(),
            origin_slot_id="origin-unknown",
            origin_as_of_utc=origin.origin_as_of_utc,
            split="training",
            hydro_information_role="REALIZED_ACTUAL",
        )


def test_builder_has_no_io_fit_ct_or_gpu_execution_path() -> None:
    source = (
        (ROOT / "pfc_shaping/lt/evaluation_feature_builder.py").read_text(encoding="utf-8").lower()
    )
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

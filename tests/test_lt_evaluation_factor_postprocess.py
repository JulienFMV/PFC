from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.evaluation_factor_postprocess import (
    FactorPostprocessAuthority,
    FactorPostprocessError,
    postprocess_challenger_factors,
)
from pfc_shaping.lt.evaluation_protocol import default_evaluation_protocol
from pfc_shaping.lt.model.shape_hourly_mlp import ShapeHourlyMLP

ROOT = Path(__file__).resolve().parents[1]


def _origin():
    return default_evaluation_protocol().holdout.origin_slots[1]


def _delivery() -> pd.DatetimeIndex:
    return pd.date_range("2026-11-04T00:00:00Z", periods=12, freq="15min", tz="UTC")


def _postprocess(raw: np.ndarray, delivery: pd.DatetimeIndex | None = None, **overrides):
    origin = _origin()
    return postprocess_challenger_factors(
        raw,
        _delivery() if delivery is None else delivery,
        candidate_id=overrides.get("candidate_id", "ridge-linear"),
        origin_slot_id=overrides.get("origin_slot_id", origin.slot_id),
        origin_as_of_utc=overrides.get("origin_as_of_utc", origin.origin_as_of_utc),
    )


def test_challenger_postprocessing_matches_native_incumbent_apply() -> None:
    delivery = _delivery()
    raw = np.asarray([-1.0, 0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0])

    class FixedPredictor:
        def predict(self, _features):
            return np.array(raw, copy=True)

    incumbent = ShapeHourlyMLP()
    incumbent.mlp_ = FixedPredictor()  # type: ignore[assignment]
    expected = incumbent.apply(
        delivery,
        enrich_15min_index(delivery, country="CH"),
        reference_date=pd.Timestamp(_origin().origin_as_of_utc),
    )
    observed = _postprocess(raw, delivery)

    np.testing.assert_array_equal(observed.values, expected.to_numpy(dtype=float))
    assert observed.to_manifest()["postprocessing_applied_exactly_once"] is True


def test_postprocessor_rejects_incumbent_to_prevent_double_application() -> None:
    with pytest.raises(FactorPostprocessError, match="already postprocessed natively"):
        _postprocess(np.ones(len(_delivery())), candidate_id="current-unweighted-mlp")


def test_postprocessed_factors_are_detached_hashed_and_authority_negative() -> None:
    raw = np.linspace(0.5, 1.5, len(_delivery()))
    observed = _postprocess(raw)
    raw[0] = 999.0

    assert observed.values.flags.writeable is False
    assert observed.values[0] != 999.0
    assert observed.audit["row_count"] == len(_delivery())
    assert observed.authority == FactorPostprocessAuthority()
    assert not any(observed.authority.to_manifest().values())
    with pytest.raises(ValueError, match="read-only"):
        observed.values[0] = 1.0
    with pytest.raises(FrozenInstanceError):
        observed.authority.production_authorized = True  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unknown_candidate", "not in the frozen protocol"),
        ("unknown_origin", "not a frozen protocol slot"),
        ("mismatched_origin", "timestamp differs"),
        ("naive_delivery", "timezone-aware"),
        ("duplicate", "timestamps are duplicated"),
        ("off_grid", "quarter-hour grid"),
        ("past_delivery", "cannot precede origin"),
        ("nonfinite", "one finite value"),
        ("wrong_shape", "one finite value"),
    ],
)
def test_postprocessor_fails_closed_on_invalid_requests(mutation: str, message: str) -> None:
    origin = _origin()
    delivery = _delivery()
    raw = np.ones(len(delivery))
    overrides = {}
    if mutation == "unknown_candidate":
        overrides["candidate_id"] = "unknown-candidate"
    elif mutation == "unknown_origin":
        overrides["origin_slot_id"] = "origin-unknown"
    elif mutation == "mismatched_origin":
        overrides["origin_as_of_utc"] = pd.Timestamp(origin.origin_as_of_utc) + pd.Timedelta(
            seconds=1
        )
    elif mutation == "naive_delivery":
        delivery = delivery.tz_localize(None)
    elif mutation == "duplicate":
        delivery = delivery.insert(1, delivery[0])[:-1]
    elif mutation == "off_grid":
        delivery = pd.DatetimeIndex([delivery[0] + pd.Timedelta(minutes=1), *delivery[1:]])
    elif mutation == "past_delivery":
        delivery = pd.DatetimeIndex(
            [pd.Timestamp(origin.origin_as_of_utc) - pd.Timedelta(minutes=15), *delivery[1:]]
        )
    elif mutation == "nonfinite":
        raw[0] = np.nan
    else:
        raw = raw[:-1]

    with pytest.raises(FactorPostprocessError, match=message):
        _postprocess(raw, delivery, **overrides)


def test_postprocessor_has_no_io_fit_ct_or_gpu_execution_path() -> None:
    source = (
        (ROOT / "pfc_shaping/lt/evaluation_factor_postprocess.py")
        .read_text(encoding="utf-8")
        .lower()
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

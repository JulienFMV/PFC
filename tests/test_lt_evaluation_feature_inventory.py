from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pfc_shaping.lt.evaluation_feature_inventory import (
    CANONICAL_FEATURE_INVENTORY_SEMANTIC_SHA256,
    CANONICAL_FEATURE_NAMES,
    FeatureInventoryAuthority,
    default_hourly_feature_inventory,
)

ROOT = Path(__file__).resolve().parents[1]


def test_inventory_freezes_the_exact_nine_incumbent_varying_inputs() -> None:
    inventory = default_hourly_feature_inventory()
    manifest = inventory.to_manifest()

    assert inventory.feature_names == (
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "is_holiday",
        "hydro_fill",
        "years_ahead",
    )
    assert tuple(manifest["feature_names"]) == CANONICAL_FEATURE_NAMES
    assert manifest["scope"] == "LT_HOURLY_CANDIDATE_MATRIX_ONLY"
    assert manifest["evidence_bindings"] == {
        "incumbent_source_normalized_lf_sha256": (
            "8f199d8075d1cd0e1d0231c3999a819d2663c4ca0a6af8ebb1178b1e6bab4aef"
        ),
        "incumbent_config_normalized_lf_sha256": (
            "f06bb9d101289e2750f72eae10cd8726aed525645455bf23b4ae524b1f1e972d"
        ),
        "entsoe_feature_availability_contract_content_id": (
            "c7826b4ad2fa5cdb6baff5d077f00ee5fd8d98108cebef9920c147d787df2ab0"
        ),
    }
    assert manifest["missingness_policy"] == ("PRESERVE_NULL_THEN_ONE_COMMON_COMPLETE_CASE_MASK")
    features = {item["feature_name"]: item for item in manifest["features"]}
    assert features["hydro_fill"]["unit"] == "FRACTION_0_1"
    assert features["years_ahead"]["unit"] == "365_25_DAY_YEARS"


def test_outages_are_explicit_disabled_constants_not_zero_filled_features() -> None:
    manifest = default_hourly_feature_inventory().to_manifest()
    constants = manifest["incumbent_compatibility_constants"]

    assert constants == {
        "unavailable_nuclear": 0.0,
        "unavailable_hydro": 0.0,
        "unavailable_thermal": 0.0,
        "meaning": "FEATURE_DISABLED_BY_GOVERNED_CONFIG_NOT_MISSING_VALUE_FILL",
        "candidate_matrix_membership": False,
    }
    assert not set(constants).intersection(CANONICAL_FEATURE_NAMES)


def test_raw_entsoe_actuals_are_excluded_from_future_candidate_inputs() -> None:
    manifest = default_hourly_feature_inventory().to_manifest()
    excluded = manifest["excluded_candidate_features"]
    downstream = manifest["shared_downstream_context"]

    assert excluded["raw_entsoe_actuals"] == [
        "load_mw",
        "solar_mw",
        "wind_mw",
        "cross_border_mw",
    ]
    assert excluded["raw_actual_reason"] == "NO_SAME_TARGET_FUTURE_ACTUAL_AT_ORIGIN"
    assert downstream["candidate_matrix_membership"] is False
    assert downstream["origin_frozen_climatology_features"] == [
        "solar_regime",
        "load_deviation",
        "flow_deviation",
    ]


def test_inventory_is_hash_frozen_and_authority_negative() -> None:
    inventory = default_hourly_feature_inventory()
    assert inventory.semantic_sha256() == CANONICAL_FEATURE_INVENTORY_SEMANTIC_SHA256
    assert inventory.authority == FeatureInventoryAuthority()
    assert not any(inventory.authority.to_manifest().values())
    with pytest.raises(FrozenInstanceError):
        inventory.authority.model_training_authorized = True  # type: ignore[misc]


def test_inventory_has_no_io_fit_ct_or_gpu_execution_path() -> None:
    source = (
        (ROOT / "pfc_shaping/lt/evaluation_feature_inventory.py")
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
        "cuda",
    )
    assert not any(fragment in source for fragment in forbidden)

from __future__ import annotations

import hashlib
import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pfc_shaping.lt.evaluation_execution_contract import (
    CANONICAL_EXECUTION_CONTRACT_SEMANTIC_SHA256,
    CandidateExecutionAuthority,
    CandidateExecutionContract,
    default_candidate_execution_contract,
)
from pfc_shaping.lt.evaluation_protocol import default_evaluation_protocol
from pfc_shaping.lt.model.shape_hourly_mlp import ShapeHourlyMLP

ROOT = Path(__file__).resolve().parents[1]


def _unfrozen_manifest() -> dict[str, object]:
    return CandidateExecutionContract().to_manifest()


def test_contract_selects_native_incumbent_without_a_generic_surrogate() -> None:
    manifest = _unfrozen_manifest()

    assert manifest["execution_path"] == (
        "NATIVE_INCUMBENT_REPLAY_WITH_COMMON_CHALLENGER_OBSERVATIONS"
    )
    incumbent = manifest["incumbent_adapter"]
    assert incumbent == {
        "candidate_id": "current-unweighted-mlp",
        "fit_interface": "NATIVE_SHAPE_HOURLY_MLP_FIT",
        "prediction_interface": "NATIVE_SHAPE_HOURLY_MLP_APPLY",
        "generic_surrogate_allowed": False,
        "source_hash_reverification_required": True,
    }
    assert manifest["challenger_adapter"]["synthetic_lab_is_real_data_runner"] is False


def test_contract_separates_learning_target_from_final_price_scoring() -> None:
    manifest = _unfrozen_manifest()
    learning = manifest["learning_estimand"]
    prediction = manifest["prediction_estimand"]
    scoring = manifest["scoring_boundary"]

    assert learning["name"] == "INCUMBENT_EQUIVALENT_HOURLY_F_H"
    assert learning["target_column"] == "target_f_h"
    assert learning["direct_raw_price_fit_allowed"] is False
    assert learning["quarter_hour_clip"] == [0.2, 3.0]
    assert learning["repeated_fallback_hour_policy"] == ("MERGE_TO_MATCH_SOURCE_BOUND_INCUMBENT")
    assert prediction["raw_model_output"] == "F_H"
    assert prediction["delivery_rows"] == "NATIVE_QUARTER_HOUR_UTC_GRID"
    assert scoring["candidate_output_before_assembly"] == "F_H_NOT_EUR_MWH"
    assert scoring["scoring_input"] == "FULL_PRICE_EUR_MWH"
    assert scoring["monthly_level_authority"] == "CH_MONTHLY_BASE_SOLVER_UNCHANGED"


def test_contract_binds_the_frozen_protocol_and_source_incumbent() -> None:
    protocol = default_evaluation_protocol()
    bindings = _unfrozen_manifest()["evidence_bindings"]

    assert bindings == {
        "evaluation_protocol_semantic_sha256": protocol.semantic_sha256(),
        "feature_inventory_semantic_sha256": (
            "83fabacc804de4201560877400fefa6974076f64d14f9738306a86bcf815a6fd"
        ),
        "incumbent_source_normalized_lf_sha256": (
            "8f199d8075d1cd0e1d0231c3999a819d2663c4ca0a6af8ebb1178b1e6bab4aef"
        ),
        "incumbent_config_normalized_lf_sha256": (
            "f06bb9d101289e2750f72eae10cd8726aed525645455bf23b4ae524b1f1e972d"
        ),
    }


def test_bound_incumbent_really_has_a_distinct_native_interface() -> None:
    protocol = default_evaluation_protocol()
    source_path = ROOT / "pfc_shaping/lt/model/shape_hourly_mlp.py"
    normalized = source_path.read_text(encoding="utf-8").replace("\r\n", "\n")

    assert hashlib.sha256(normalized.encode("utf-8")).hexdigest() == (
        protocol.bindings.incumbent_source_normalized_lf_sha256
    )
    assert tuple(inspect.signature(ShapeHourlyMLP.fit).parameters) == (
        "self",
        "epex_df",
        "calendar_df",
        "hydro_df",
        "outages_df",
    )
    assert tuple(inspect.signature(ShapeHourlyMLP.apply).parameters) == (
        "self",
        "timestamps",
        "calendar_df",
        "reference_date",
        "outages_forecast",
    )


def test_contract_is_hash_frozen_and_authority_negative() -> None:
    contract = default_candidate_execution_contract()

    assert contract.semantic_sha256() == CANONICAL_EXECUTION_CONTRACT_SEMANTIC_SHA256
    assert contract.authority == CandidateExecutionAuthority()
    assert not any(contract.authority.to_manifest().values())
    with pytest.raises(FrozenInstanceError):
        contract.authority.model_training_authorized = True  # type: ignore[misc]


def test_contract_has_no_io_fit_ct_or_gpu_execution_path() -> None:
    source = (
        (ROOT / "pfc_shaping/lt/evaluation_execution_contract.py")
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

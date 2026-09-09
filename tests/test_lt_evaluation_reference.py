from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pfc_shaping.lt.evaluation_protocol import default_evaluation_protocol
from pfc_shaping.lt.evaluation_reference import (
    CANONICAL_REFERENCE_SEMANTIC_SHA256,
    REFERENCE_BENCHMARK_SCHEMA,
    ReferenceAuthority,
    default_primary_reference_benchmark,
)

ROOT = Path(__file__).resolve().parents[1]


def test_seasonal_reference_is_primary_but_outside_candidate_selection() -> None:
    reference = default_primary_reference_benchmark()
    protocol = default_evaluation_protocol()
    manifest = reference.to_manifest()

    assert manifest["schema_version"] == REFERENCE_BENCHMARK_SCHEMA
    assert manifest["benchmark_id"] == "market-constrained-seasonal-reference-v1"
    assert manifest["role"] == "PRIMARY_PROMOTION_REFERENCE"
    assert manifest["candidate_inventory_membership"] is False
    assert manifest["candidate_selection_eligible"] is False
    assert reference.benchmark_id not in {item.candidate_id for item in protocol.candidates}
    assert manifest["comparison_population"] == "SAME_ORIGINS_ROWS_MASKS_AND_WEIGHTS_AS_CANDIDATES"
    assert manifest["primary_metric"] == protocol.primary_metric
    assert manifest["secondary_metrics"] == list(protocol.secondary_metrics)
    assert manifest["lead_month_buckets"] == list(protocol.holdout.lead_month_buckets)


def test_seasonal_reference_preserves_solver_level_and_has_no_tuning() -> None:
    manifest = default_primary_reference_benchmark().to_manifest()

    assert manifest["level_authority"] == "CH_MONTHLY_BASE_SOLVER_ONLY"
    assert manifest["shape_constraint"] == "ENERGY_WEIGHTED_ZERO_MEAN_WITHIN_SOLVER_MONTH"
    assert manifest["fit_policy"] == "FROZEN_TRANSPARENT_FORMULA_NO_HYPERPARAMETER_TUNING"
    assert manifest["promotion_target"] == (
        "CANDIDATE_WEIGHTED_MAE_AND_RMSE_IMPROVEMENT_AT_LEAST_2_PERCENT"
    )
    assert manifest["scoring_lane"] == "SEPARATE_REFERENCE_SCORE_BEFORE_CANDIDATE_RANKING"


def test_gpu_is_remembered_without_changing_the_reference_or_cpu_oracle() -> None:
    runtime = default_primary_reference_benchmark().to_manifest()["runtime_policy"]

    assert runtime == {
        "reference_backend": "CPU_FLOAT64",
        "gpu_acceleration_applicable_to_reference": False,
        "canonical_lightgbm_backend": "CPU_DETERMINISTIC",
        "gpu_available_for_qualified_nonlinear_work": True,
        "gpu_fit_or_model_selection_status": "REQUIRES_SEPARATE_CPU_GPU_PARITY_QUALIFICATION",
        "gpu_does_not_replace_cpu_hard_gate_oracle": True,
    }


def test_reference_authority_is_immutable_and_entirely_negative() -> None:
    authority = default_primary_reference_benchmark().authority
    manifest = authority.to_manifest()

    assert authority == ReferenceAuthority()
    assert manifest and not any(manifest.values())
    with pytest.raises(FrozenInstanceError):
        authority.production_authorized = True  # type: ignore[misc]


def test_reference_manifest_hash_is_stable_and_contains_no_real_values() -> None:
    reference = default_primary_reference_benchmark()
    manifest = reference.to_manifest()

    assert reference.semantic_sha256() == CANONICAL_REFERENCE_SEMANTIC_SHA256
    encoded = json.dumps(manifest, sort_keys=True).lower()
    assert "real_data_training_performed" in encoded
    assert manifest["real_data_training_performed"] is False
    assert manifest["real_truth_opened"] is False


def test_reference_contract_has_no_ct_or_data_access_path() -> None:
    source = (ROOT / "pfc_shaping/lt/evaluation_reference.py").read_text(encoding="utf-8")
    forbidden = (
        "pfc_shaping.ct",
        "databricks",
        "read_parquet",
        "read_csv",
        "to_parquet",
        "to_csv",
        "requests.",
    )
    assert not any(fragment in source.lower() for fragment in forbidden)

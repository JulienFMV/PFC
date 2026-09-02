from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from pfc_shaping.lt.evaluation_protocol import (
    CANONICAL_PROTOCOL_SEMANTIC_SHA256,
    EVALUATION_PROTOCOL_VERSION,
    EVALUATION_STATUS,
    LEAD_MONTH_BUCKETS,
    PRIMARY_METRIC,
    SECONDARY_METRICS,
    CandidateRole,
    CandidateSpec,
    EvaluationAuthority,
    FitPolicy,
    ModelFamily,
    OriginSlot,
    default_evaluation_protocol,
)
from pfc_shaping.validation import ch_lt_dependence_power_design as power_design
from pfc_shaping.validation import ch_lt_estimand_contract as estimand
from pfc_shaping.validation import ch_lt_origin_registry_protocol as origin_registry

ROOT = Path(__file__).resolve().parents[1]


def _normalized_lf_sha256(relative_path: str) -> str:
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode()).hexdigest()


def test_default_protocol_freezes_exact_candidate_inventory() -> None:
    protocol = default_evaluation_protocol()
    manifest = protocol.to_manifest()

    assert manifest["contract_version"] == EVALUATION_PROTOCOL_VERSION
    assert [item.family for item in protocol.candidates] == list(ModelFamily)
    assert [item.role for item in protocol.candidates].count(CandidateRole.INCUMBENT) == 1
    assert manifest["future_holdout_tuning"] == "FORBIDDEN"
    assert manifest["monthly_level_authority"] == "CH_MONTHLY_BASE_SOLVER_UNCHANGED"
    assert all(candidate["training_authorized"] is False for candidate in manifest["candidates"])


def test_protocol_authority_is_immutable_and_entirely_negative() -> None:
    authority = default_evaluation_protocol().authority
    manifest = authority.to_manifest()

    assert authority == EvaluationAuthority()
    assert manifest["status"] == EVALUATION_STATUS
    assert not any(value for key, value in manifest.items() if key != "status")
    with pytest.raises(TypeError):
        EvaluationAuthority(model_training_authorized=True)  # type: ignore[call-arg]
    with pytest.raises(FrozenInstanceError):
        authority.production_authorized = True  # type: ignore[misc]


def test_metrics_and_lead_buckets_reuse_the_existing_estimand() -> None:
    protocol = default_evaluation_protocol()

    assert PRIMARY_METRIC == estimand.EXPECTED_METRIC_POLICY["primary_metric"]
    assert SECONDARY_METRICS == tuple(estimand.EXPECTED_METRIC_POLICY["secondary_metrics"])
    assert protocol.holdout.lead_month_buckets == tuple(
        bucket["bucket_id"] for bucket in estimand.EXPECTED_HORIZON_POLICY["buckets"]
    )
    assert protocol.holdout.lead_month_buckets == LEAD_MONTH_BUCKETS


def test_protocol_bindings_match_exact_local_bytes() -> None:
    bindings = default_evaluation_protocol().bindings

    assert bindings.estimand_sha256 == estimand.DOCUMENT_SHA256
    assert bindings.origin_registry_sha256 == origin_registry.PROTOCOL_SHA256
    assert bindings.dependence_power_design_sha256 == power_design.DESIGN_SHA256
    assert bindings.incumbent_source_normalized_lf_sha256 == _normalized_lf_sha256(
        "pfc_shaping/lt/model/shape_hourly_mlp.py"
    )
    assert bindings.incumbent_config_normalized_lf_sha256 == _normalized_lf_sha256(
        "pfc_shaping/config.yaml"
    )
    assert bindings.challenger_source_normalized_lf_sha256 == _normalized_lf_sha256(
        "pfc_shaping/lt/evaluation_challengers.py"
    )
    assert bindings.evaluation_engine_normalized_lf_sha256 == _normalized_lf_sha256(
        "pfc_shaping/lt/evaluation_engine.py"
    )
    assert bindings.origin_registration_envelope_normalized_lf_sha256 == _normalized_lf_sha256(
        "pfc_shaping/lt/origin_registration_envelope.py"
    )
    assert bindings.package_contract_normalized_lf_sha256 == _normalized_lf_sha256(
        "pfc_shaping/package_contract.py"
    )
    assert bindings.runtime_spec_normalized_lf_sha256 == _normalized_lf_sha256("pyproject.toml")
    assert all(
        item.implementation_normalized_lf_sha256
        for item in default_evaluation_protocol().candidates
    )


def test_future_cohort_is_explicit_but_has_zero_countable_origins() -> None:
    protocol = default_evaluation_protocol()
    holdout = protocol.holdout
    manifest = holdout.to_manifest()

    assert len(holdout.origin_slots) == 12
    assert holdout.origin_slots[0].slot_id == "origin-2026-10"
    assert holdout.origin_slots[-1].slot_id == "origin-2027-09"
    assert holdout.origin_slots[0].first_delivery_month == "2026-11"
    assert holdout.origin_slots[0].last_delivery_month == "2029-10"
    assert manifest["external_registration_status"] == "PENDING"
    assert manifest["scheduled_origin_count"] == 12
    assert manifest["countable_origin_count"] == 0
    assert manifest["truth_open_authorized"] is False
    assert manifest["holdout_consumed"] is False
    assert "t057" not in json.dumps(protocol.to_manifest(), sort_keys=True).lower()


def test_candidate_contract_rejects_role_and_selection_leakage() -> None:
    incumbent = default_evaluation_protocol().candidates[0]
    with pytest.raises(ValueError, match="incumbent must"):
        replace(incumbent, family=ModelFamily.RIDGE)
    with pytest.raises(ValueError, match="cannot tune"):
        replace(incumbent, tuning_grid=(("alpha", ("1",)),))

    challenger = default_evaluation_protocol().candidates[1]
    with pytest.raises(ValueError, match="nested-origin"):
        replace(challenger, fit_policy=FitPolicy.FROZEN_IMPLEMENTATION_PER_ORIGIN)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(challenger, implementation_normalized_lf_sha256="invalid")


def test_parameter_and_tuning_tables_must_be_canonical() -> None:
    with pytest.raises(ValueError, match="unique and sorted"):
        CandidateSpec(
            candidate_id="invalid-order",
            role=CandidateRole.CHALLENGER,
            family=ModelFamily.RIDGE,
            fit_policy=FitPolicy.NESTED_ORIGIN_SELECTION,
            fixed_parameters=(("zeta", "1"), ("alpha", "1")),
        )
    with pytest.raises(ValueError, match="requires a valid name and choices"):
        CandidateSpec(
            candidate_id="invalid-grid",
            role=CandidateRole.CHALLENGER,
            family=ModelFamily.RIDGE,
            fit_policy=FitPolicy.NESTED_ORIGIN_SELECTION,
            fixed_parameters=(("alpha", "1"),),
            tuning_grid=(("alpha", ()),),
        )


def test_origin_slot_requires_exact_lead_support() -> None:
    slot = default_evaluation_protocol().holdout.origin_slots[0]
    with pytest.raises(ValueError, match="lead month 1"):
        replace(slot, first_delivery_month="2026-12")
    with pytest.raises(ValueError, match="lead month 36"):
        replace(slot, last_delivery_month="2029-09")
    with pytest.raises(ValueError, match="UTC origin month"):
        OriginSlot(
            slot_id="origin-2026-11",
            origin_as_of_utc=slot.origin_as_of_utc,
            first_delivery_month=slot.first_delivery_month,
            last_delivery_month=slot.last_delivery_month,
        )


def test_protocol_manifest_hash_is_stable() -> None:
    assert default_evaluation_protocol().semantic_sha256() == CANONICAL_PROTOCOL_SEMANTIC_SHA256


def test_lt_protocol_has_no_ct_import_or_runtime_model_dependency() -> None:
    source = (ROOT / "pfc_shaping/lt/evaluation_protocol.py").read_text(encoding="utf-8")

    assert "pfc_shaping.ct" not in source
    assert "shape_hourly_mlp import" not in source
    assert "lightgbm import" not in source


def test_synthetic_evaluation_modules_have_no_data_access_path() -> None:
    forbidden = (
        "pfc_shaping.ct",
        "databricks",
        "read_parquet",
        "read_csv",
        "to_parquet",
        "to_csv",
        "requests.",
    )
    for relative_path in (
        "pfc_shaping/lt/evaluation_challengers.py",
        "pfc_shaping/lt/evaluation_engine.py",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8").lower()
        assert not any(fragment in source for fragment in forbidden)

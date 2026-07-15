from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import numpy as np
import pandas as pd
import pytest

from pfc_shaping.calibration import tier2_monthly_eex_evaluation as tier2
from pfc_shaping.calibration.cascading import count_hours
from pfc_shaping.pipeline.model_governance_contract import (
    MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV,
    sign_model_governance_artifact_receipt,
)


def _fold_spec(
    campaign: str,
    number: int,
    origin_date: str,
    bucket: str,
) -> dict[str, object] | None:
    tenor, horizon = bucket.split("_", maxsplit=1)
    origin_year = int(origin_date[:4])
    origin_month = int(origin_date[5:7])
    horizon_years = {"H0": 0, "H1": 1, "H2": 2, "H3_PLUS": 3}[horizon]
    delivery_year = origin_year + horizon_years
    if horizon == "H0" and tenor == "MONTH":
        if origin_month == 12:
            return None
        target_product = f"{delivery_year}-{origin_month + 1:02d}"
    elif horizon == "H0":
        current_quarter = (origin_month - 1) // 3 + 1
        if current_quarter == 4:
            return None
        target_product = f"{delivery_year}-Q{current_quarter + 1}"
    else:
        target_product = {
            "MONTH": f"{delivery_year}-01",
            "QUARTER": f"{delivery_year}-Q1",
        }[tenor]
    fold_as_of_utc = f"{origin_date}T08:00:00+00:00"
    fold: dict[str, object] = {
        "fold_id": f"{campaign.lower()}-{number:03d}-{bucket}",
        "campaign_type": campaign,
        "origin_id": tier2._planned_origin_id(campaign, fold_as_of_utc),
        "fold_as_of_utc": fold_as_of_utc,
        "tenor_horizon_bucket": bucket,
        "target_product": target_product,
        "target_load_type": "BASE",
        "target_rule": "MASK_ONE_IDENTIFIABLE_SAME_VINTAGE_EEX_QUOTE",
        "historical_context_rule": "EXPANDING_PIT_AVAILABLE_AT_LE_FOLD_AS_OF_NO_RETUNING",
    }
    fold["fold_spec_hash"] = tier2._sha256_json(fold)
    return fold


def _campaign_folds(campaign: str, *, origins: int, first_year: int) -> list[dict[str, object]]:
    folds: list[dict[str, object]] = []
    origin_offset = 0 if campaign == "SELECTION" else 1_000
    for index in range(1, origins + 1):
        year = first_year + (index - 1) // 12
        month = (index - 1) % 12 + 1
        for bucket in tier2.MANDATORY_TENOR_HORIZON_BUCKETS:
            fold = _fold_spec(
                campaign,
                origin_offset + index,
                f"{year}-{month:02d}-15",
                bucket,
            )
            if fold is not None:
                folds.append(fold)
    return folds


def _plan(*, schema: str = tier2.PLAN_SCHEMA) -> dict[str, object]:
    selection_eex_root: dict[str, object] = {
        "catalog_id": "d" * 64,
        "catalog_sha256": "e" * 64,
        "history_sha256": "f" * 64,
    }
    selection_eex_root["root_sha256"] = tier2._sha256_json(
        {
            "schema_version": "tier2_monthly_eex_catalog_root.v1",
            **selection_eex_root,
        }
    )
    plan: dict[str, object] = {
        "schema_version": schema,
        "scope": tier2.SCOPE,
        "full_tier2_pfc_status": tier2.FULL_TIER2_STATUS,
        "production_approved": False,
        "market": "CH",
        "timezone": "Europe/Zurich",
        "estimand": {
            "unit": "ONE_ECONOMIC_SAME_VINTAGE_EEX_QUOTE",
            "target_removed_from_all_inputs": True,
            "algebraically_revealing_quotes_removed": True,
            "retained_hard_quotes_repriced": True,
            "later_quotes_namespace": "DIAGNOSTIC_EXCLUDED",
            "allowed_sources": ["EEX"],
            "forbidden_sources": ["OMPEX", "PROXY", "BENCHMARK"],
        },
        "baseline": dict(tier2._EXPECTED_BASELINE),
        "secondary_challengers": ["parent_flat.v1", "last_visible_quote.v1"],
        "coverage": dict(tier2._EXPECTED_COVERAGE),
        "performance": dict(tier2._EXPECTED_PERFORMANCE),
        "bootstrap": dict(tier2._EXPECTED_BOOTSTRAP),
        "identifiability": dict(tier2._EXPECTED_IDENTIFIABILITY),
        "status_precedence": [
            "ERROR",
            "FAIL",
            "UNSUPPORTED",
            tier2.PASS_STATUS,
        ],
        "candidate_grid_sha256": "a" * 64,
        "candidate_grid_contract": dict(tier2._EXPECTED_CANDIDATE_GRID_CONTRACT),
        "candidate_config_sha256_allowlist": ["1" * 64, "2" * 64],
        "selection_rule": "MIN_SELECTION_MAE_THEN_RMSE_LEXICOGRAPHIC_FIXED_GRID",
        "selection_eex_root": selection_eex_root,
        "fold_source_policy": dict(tier2._EXPECTED_FOLD_SOURCE_POLICY),
        "fold_context_policy": dict(tier2._EXPECTED_FOLD_CONTEXT_POLICY),
        "fold_weight_policy": dict(tier2._EXPECTED_FOLD_WEIGHT_POLICY),
        "fold_claim_policy": dict(tier2._EXPECTED_FOLD_CLAIM_POLICY),
        "evaluation_source_closure_sha256": "b" * 64,
        "runtime_lock_sha256": "c" * 64,
        "mandatory_tenor_horizon_buckets": list(
            tier2.MANDATORY_TENOR_HORIZON_BUCKETS
        ),
        "selection_folds": _campaign_folds(
            "SELECTION", origins=tier2.MIN_SELECTION_ORIGINS, first_year=2023
        ),
        "holdout_folds": _campaign_folds(
            "HOLDOUT", origins=tier2.MIN_HOLDOUT_ORIGINS, first_year=2026
        ),
    }
    if schema == tier2.SELECTION_INPUT_PLAN_SCHEMA:
        plan["selection_input_resource_profile"] = dict(
            tier2.EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE
        )
    plan["plan_id"] = tier2._sha256_json(plan)
    return plan


def _repin_plan(plan: dict[str, object]) -> None:
    unsigned = dict(plan)
    unsigned.pop("plan_id", None)
    plan["plan_id"] = tier2._sha256_json(unsigned)


def _repin_fold(fold: dict[str, object]) -> None:
    unsigned = dict(fold)
    unsigned.pop("fold_spec_hash", None)
    fold["fold_spec_hash"] = tier2._sha256_json(unsigned)


def _freeze(plan: dict[str, object]) -> dict[str, object]:
    freeze: dict[str, object] = {
        "schema_version": tier2.CANDIDATE_FREEZE_SCHEMA,
        "scope": tier2.SCOPE,
        "full_tier2_pfc_status": tier2.FULL_TIER2_STATUS,
        "production_approved": False,
        "plan_sha256": tier2._sha256_json(plan),
        "selection_campaign_sha256": "1" * 64,
        "selected_config_sha256": "2" * 64,
        "source_closure_sha256": "3" * 64,
        "runtime_fingerprint_sha256": "4" * 64,
        "bootstrap_block_length_origins": 3,
        "frozen_at_utc": "2026-12-31T12:00:00Z",
    }
    freeze["freeze_id"] = tier2._sha256_json(freeze)
    return freeze


def _fold_result(
    spec: dict[str, object],
    *,
    candidate_error: float = 8.0,
    baseline_error: float = 10.0,
    status: str = tier2.FOLD_EVALUATED,
) -> dict[str, object]:
    evaluated = status == tier2.FOLD_EVALUATED
    return {
        "schema_version": tier2.FOLD_SCHEMA,
        "fold_id": spec["fold_id"],
        "fold_spec_hash": spec["fold_spec_hash"],
        "campaign_type": spec["campaign_type"],
        "origin_id": spec["origin_id"],
        "origin_date": spec["origin_date"],
        "tenor_horizon_bucket": spec["tenor_horizon_bucket"],
        "status": status,
        "included_in_complete_case": evaluated,
        "target_weight": 1.0,
        "candidate_abs_error": candidate_error,
        "candidate_squared_error": candidate_error**2,
        "baseline_abs_error": baseline_error,
        "baseline_squared_error": baseline_error**2,
        "repricing_max_abs_error": 0.0,
        "conservation_max_abs_error": 0.0,
        "fold_as_of_utc": "2027-12-31T18:00:00Z",
        "selection_training_context_max_available_at_utc": "2026-12-15T08:00:00Z",
        "fold_historical_context_max_available_at_utc": "2026-12-30T08:00:00Z",
        "evaluation_snapshot_received_at_utc": "2027-01-02T08:00:00Z",
        "masked_target_available_at_utc": "2027-01-02T08:00:00Z",
        "selection_identity_hashes": {
            "document": ["1" * 64],
            "snapshot": ["2" * 64],
            "revision": ["3" * 64],
            "quote": ["4" * 64],
        },
        "evaluation_identity_hashes": {
            "document": [hashlib.sha256(f"document-{spec['fold_id']}".encode()).hexdigest()],
            "snapshot": [hashlib.sha256(f"snapshot-{spec['fold_id']}".encode()).hexdigest()],
            "revision": [hashlib.sha256(f"revision-{spec['fold_id']}".encode()).hexdigest()],
            "quote": [hashlib.sha256(f"quote-{spec['fold_id']}".encode()).hexdigest()],
        },
        "masked_target_quote_id": f"target-{spec['fold_id']}",
        "retained_quote_ids": [f"retained-{spec['fold_id']}"],
        "revealing_quote_ids": [],
        "later_quote_namespace": "DIAGNOSTIC_EXCLUDED",
    }


def _campaign(
    plan: dict[str, object],
    freeze: dict[str, object],
    results: list[dict[str, object]],
) -> dict[str, object]:
    block_length = int(freeze["bootstrap_block_length_origins"])
    summary = tier2._campaign_performance(
        results,
        campaign_type="HOLDOUT",
        block_length=block_length,
    )
    campaign: dict[str, object] = {
        "schema_version": tier2.CAMPAIGN_SCHEMA,
        "scope": tier2.SCOPE,
        "full_tier2_pfc_status": tier2.FULL_TIER2_STATUS,
        "production_approved": False,
        "campaign_type": "HOLDOUT",
        "plan_sha256": tier2._sha256_json(plan),
        "candidate_freeze_sha256": tier2._sha256_json(freeze),
        "expected_fold_hashes": {
            str(item["fold_id"]): str(item["fold_spec_hash"])
            for item in plan["holdout_folds"]
        },
        "fold_results": results,
        "bootstrap_block_length_origins": block_length,
        "reported_effective_block_count": int(np.ceil(len(results) / block_length)),
        "final_status": tier2._campaign_status(
            results,
            summary,
            campaign_type="HOLDOUT",
        ),
        "performance_summary": summary,
    }
    campaign["campaign_id"] = tier2._sha256_json(campaign)
    return campaign


def test_atomic_basis_does_not_treat_base_as_spanned_by_peak() -> None:
    assessment = tier2.assess_target_identifiability(
        target=tier2.ProductConstraint("2027-01", "BASE"),
        retained=[tier2.ProductConstraint("2027-01", "PEAK")],
    )

    assert assessment.retained_rank == 1
    assert assessment.augmented_rank == 2
    assert assessment.status == "IDENTIFIABLE"


def test_base_is_spanned_by_peak_and_offpeak_on_same_delivery_month() -> None:
    assessment = tier2.assess_target_identifiability(
        target=tier2.ProductConstraint("2027-01", "BASE"),
        retained=[
            tier2.ProductConstraint("2027-01", "PEAK"),
            tier2.ProductConstraint("2027-01", "OFFPEAK"),
        ],
    )

    assert assessment.retained_rank == assessment.augmented_rank == 2
    assert assessment.status == "UNSUPPORTED_TARGET_SPANNED_BY_RETAINED_QUOTES"


def test_product_vector_uses_real_dst_hour_counts() -> None:
    vector = tier2.product_constraint_vector(
        tier2.ProductConstraint("2027-Q1", "BASE"),
        delivery_months=["2027-01", "2027-02", "2027-03"],
    )
    march_total, _, _ = count_hours(2027, 3, 3)
    total = sum(count_hours(2027, month, month)[0] for month in (1, 2, 3))

    assert march_total == 743
    assert vector[4] + vector[5] == pytest.approx(march_total / total)
    assert vector.sum() == pytest.approx(1.0)


def test_redundant_constraints_do_not_create_infinite_condition_number() -> None:
    assessment = tier2.assess_target_identifiability(
        target=tier2.ProductConstraint("2027-01", "OFFPEAK"),
        retained=[
            tier2.ProductConstraint("2027-01", "PEAK"),
            tier2.ProductConstraint("2027-01", "PEAK"),
        ],
    )

    assert assessment.retained_condition_number == pytest.approx(1.0)
    assert assessment.status == "IDENTIFIABLE"


def test_plan_cannot_claim_hourly_or_production_approval() -> None:
    plan = _plan()
    plan["production_approved"] = True
    _repin_plan(plan)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="cannot approve production"):
        tier2._validate_evaluation_plan(plan)


def test_selection_input_plan_v2_precommits_exact_resource_profile() -> None:
    plan = _plan(schema=tier2.SELECTION_INPUT_PLAN_SCHEMA)

    verified = tier2._validate_evaluation_plan(plan)

    assert verified["selection_input_resource_profile"] == (
        tier2.EXPECTED_SELECTION_INPUT_RESOURCE_PROFILE
    )


def test_selection_input_plan_v2_rejects_changed_resource_profile() -> None:
    plan = _plan(schema=tier2.SELECTION_INPUT_PLAN_SCHEMA)
    plan["selection_input_resource_profile"]["max_selection_folds"] = 513
    _repin_plan(plan)

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError,
        match="selection input resource profile",
    ):
        tier2._validate_evaluation_plan(plan)


def test_plan_rejects_weakened_holdout_threshold() -> None:
    plan = _plan()
    plan["performance"]["mae_min_relative_improvement"] = 0.0
    _repin_plan(plan)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="performance contract"):
        tier2._validate_evaluation_plan(plan)


def test_plan_numeric_contract_rejects_boolean_substitution() -> None:
    plan = _plan()
    plan["baseline"]["shape_energy_weighted_mean"] = False
    _repin_plan(plan)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="primary baseline"):
        tier2._validate_evaluation_plan(plan)


def test_plan_rejects_unpinned_selection_eex_root() -> None:
    plan = _plan()
    plan["selection_eex_root"]["history_sha256"] = "0" * 64
    _repin_plan(plan)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="root hash"):
        tier2._validate_evaluation_plan(plan)


def test_plan_rejects_target_dependent_revision_fallback() -> None:
    plan = _plan()
    plan["fold_source_policy"]["revision_fallback_allowed"] = True
    _repin_plan(plan)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="source policy"):
        tier2._validate_evaluation_plan(plan)


@pytest.mark.parametrize(
    ("section", "field", "replacement", "message"),
    [
        (
            "fold_source_policy",
            "snapshot_selection",
            "ANY_CONVENIENT_SNAPSHOT",
            "source policy",
        ),
        (
            "fold_source_policy",
            "revealing_overlap_policy",
            "TARGET_ONLY",
            "source policy",
        ),
        ("fold_context_policy", "lookback_months", 12, "context policy"),
        ("fold_context_policy", "embargo_months", 0, "context policy"),
        (
            "fold_context_policy",
            "pit_time_axis",
            "date",
            "context policy",
        ),
        ("fold_weight_policy", "formula", "UNIT_WEIGHT", "weight policy"),
        (
            "fold_claim_policy",
            "campaign_eligible",
            True,
            "claim policy",
        ),
    ],
)
def test_plan_rejects_weakened_fold_policies(
    section: str,
    field: str,
    replacement: object,
    message: str,
) -> None:
    plan = _plan()
    plan[section][field] = replacement
    _repin_plan(plan)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match=message):
        tier2._validate_evaluation_plan(plan)


def test_plan_rejects_reordered_candidate_config_allowlist() -> None:
    plan = _plan()
    plan["candidate_config_sha256_allowlist"] = list(
        reversed(plan["candidate_config_sha256_allowlist"])
    )
    _repin_plan(plan)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="sorted and unique"):
        tier2._validate_evaluation_plan(plan)


def test_plan_rejects_noncanonical_lowercase_load_type() -> None:
    plan = _plan()
    fold = plan["selection_folds"][0]
    fold["target_load_type"] = "base"
    _repin_fold(fold)
    _repin_plan(plan)

    with pytest.raises(
        tier2.Tier2MonthlyEexEvaluationError,
        match="canonical uppercase",
    ):
        tier2._validate_evaluation_plan(plan)


def test_plan_rejects_same_temporal_origin_under_different_ids() -> None:
    plan = _plan()
    selection = plan["selection_folds"][0]
    holdout = plan["holdout_folds"][0]
    holdout["fold_as_of_utc"] = selection["fold_as_of_utc"]
    holdout["origin_id"] = tier2._planned_origin_id(
        "HOLDOUT", str(selection["fold_as_of_utc"])
    )
    holdout["target_product"] = selection["target_product"]
    _repin_fold(holdout)
    _repin_plan(plan)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="origin times overlap"):
        tier2._validate_evaluation_plan(plan)


def test_plan_rejects_target_already_in_delivery() -> None:
    plan = _plan()
    fold = next(
        item
        for item in plan["holdout_folds"]
        if item["tenor_horizon_bucket"] == "MONTH_H0"
    )
    fold["target_product"] = str(fold["fold_as_of_utc"])[:7]
    _repin_fold(fold)
    _repin_plan(plan)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="strictly after"):
        tier2._validate_evaluation_plan(plan)


def test_plan_requires_every_mandatory_bucket_at_preregistered_depth() -> None:
    plan = _plan()
    plan["holdout_folds"] = [
        fold
        for fold in plan["holdout_folds"]
        if fold["tenor_horizon_bucket"] != "QUARTER_H3_PLUS"
    ]
    _repin_plan(plan)

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="QUARTER_H3_PLUS"):
        tier2._validate_evaluation_plan(plan)


@pytest.mark.parametrize(
    ("validator", "args"),
    [
        (tier2.validate_candidate_freeze, ({},)),
        (
            tier2.validate_fold_result,
            ({},),
        ),
        (tier2.validate_campaign, ({},)),
        (tier2.validate_evaluation_index, ({},)),
    ],
)
def test_downstream_artifacts_fail_closed_until_row_level_replay_exists(
    validator: object,
    args: tuple[object, ...],
) -> None:
    kwargs: dict[str, object] = {}
    if validator is tier2.validate_fold_result:
        kwargs["campaign_type"] = "HOLDOUT"
    if validator is tier2.validate_campaign:
        kwargs["plan"] = _plan()
    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="UNSUPPORTED"):
        validator(*args, **kwargs)


def test_verified_plan_proof_cannot_be_constructed_directly() -> None:
    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="path-based"):
        tier2.VerifiedEvaluationPlan(
            plan_id="1" * 64,
            artifact_sha256="2" * 64,
            governance_key_id="3" * 64,
            timestamp_key_id="4" * 64,
            trusted_time_utc=pd.Timestamp("2026-01-01T00:00:00Z"),
            governance_receipt_sha256="5" * 64,
            timestamp_receipt_sha256="6" * 64,
            payload={},
        )


def test_exact_plan_bytes_require_distinct_governance_and_timestamp_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    governance_private, governance_public = _write_keypair(tmp_path, "governance")
    timestamp_private, timestamp_public = _write_keypair(tmp_path, "timestamp")
    monkeypatch.setenv(MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV, str(governance_public))
    monkeypatch.setenv(tier2.TIMESTAMP_PUBLIC_KEY_ENV, str(timestamp_public))
    monkeypatch.setenv(tier2.TIMESTAMP_JOURNAL_ID_ENV, "journal-test")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(_plan(), sort_keys=True), encoding="utf-8")
    governance = sign_model_governance_artifact_receipt(
        path,
        artifact_role="tier2_monthly_eex_evaluation_plan",
        authority_id="MODEL-RISK",
        issued_at_utc="2026-01-01T08:00:00Z",
        data_cutoff_utc="2025-12-31T23:59:59Z",
        private_key_path=governance_private,
    )
    timestamp = _timestamp_receipt(path, timestamp_private)
    governance_path = tmp_path / "governance-receipt.json"
    timestamp_path = tmp_path / "timestamp-receipt.json"
    governance_path.write_text(json.dumps(governance, sort_keys=True), encoding="utf-8")
    timestamp_path.write_text(json.dumps(timestamp, sort_keys=True), encoding="utf-8")

    verified = tier2.verify_governed_plan_artifact(
        path,
        governance_receipt_path=governance_path,
        timestamp_receipt_path=timestamp_path,
    )

    assert verified.payload["scope"] == tier2.SCOPE
    assert verified.timestamp_key_id != verified.governance_key_id
    with pytest.raises(TypeError):
        verified.payload["scope"] = "MUTATED"

    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="artifact hash"):
        tier2.verify_governed_plan_artifact(
            path,
            governance_receipt_path=governance_path,
            timestamp_receipt_path=timestamp_path,
        )


def test_exact_selection_input_plan_v2_uses_its_signed_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    governance_private, governance_public = _write_keypair(tmp_path, "governance-v2")
    timestamp_private, timestamp_public = _write_keypair(tmp_path, "timestamp-v2")
    monkeypatch.setenv(MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV, str(governance_public))
    monkeypatch.setenv(tier2.TIMESTAMP_PUBLIC_KEY_ENV, str(timestamp_public))
    monkeypatch.setenv(tier2.TIMESTAMP_JOURNAL_ID_ENV, "journal-test")
    path = tmp_path / "plan-v2.json"
    path.write_text(
        json.dumps(_plan(schema=tier2.SELECTION_INPUT_PLAN_SCHEMA), sort_keys=True),
        encoding="utf-8",
    )
    governance = sign_model_governance_artifact_receipt(
        path,
        artifact_role="tier2_monthly_eex_evaluation_plan",
        authority_id="MODEL-RISK",
        issued_at_utc="2026-01-01T08:00:00Z",
        data_cutoff_utc="2025-12-31T23:59:59Z",
        private_key_path=governance_private,
    )
    timestamp = _timestamp_receipt(path, timestamp_private)
    governance_path = tmp_path / "governance-v2.json"
    timestamp_path = tmp_path / "timestamp-v2.json"
    governance_path.write_text(json.dumps(governance, sort_keys=True), encoding="utf-8")
    timestamp_path.write_text(json.dumps(timestamp, sort_keys=True), encoding="utf-8")

    verified = tier2.verify_governed_plan_artifact(
        path,
        governance_receipt_path=governance_path,
        timestamp_receipt_path=timestamp_path,
    )

    assert verified.payload["schema_version"] == tier2.SELECTION_INPUT_PLAN_SCHEMA


def test_governed_plan_rejects_duplicate_receipt_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    governance_private, governance_public = _write_keypair(tmp_path, "governance")
    timestamp_private, timestamp_public = _write_keypair(tmp_path, "timestamp")
    monkeypatch.setenv(MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV, str(governance_public))
    monkeypatch.setenv(tier2.TIMESTAMP_PUBLIC_KEY_ENV, str(timestamp_public))
    monkeypatch.setenv(tier2.TIMESTAMP_JOURNAL_ID_ENV, "journal-test")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan(), sort_keys=True), encoding="utf-8")
    governance = sign_model_governance_artifact_receipt(
        plan_path,
        artifact_role="tier2_monthly_eex_evaluation_plan",
        authority_id="MODEL-RISK",
        issued_at_utc="2026-01-01T08:00:00Z",
        data_cutoff_utc="2025-12-31T23:59:59Z",
        private_key_path=governance_private,
    )
    governance_path = tmp_path / "governance.json"
    original = json.dumps(governance, sort_keys=True)
    governance_path.write_text(
        '{"schema_version":"duplicate",' + original[1:], encoding="utf-8"
    )
    timestamp_path = tmp_path / "timestamp.json"
    timestamp_path.write_text(
        json.dumps(_timestamp_receipt(plan_path, timestamp_private), sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="duplicate JSON key"):
        tier2.verify_governed_plan_artifact(
            plan_path,
            governance_receipt_path=governance_path,
            timestamp_receipt_path=timestamp_path,
        )


def test_governed_plan_rejects_shared_governance_and_timestamp_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_private, shared_public = _write_keypair(tmp_path, "shared")
    monkeypatch.setenv(MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV, str(shared_public))
    monkeypatch.setenv(tier2.TIMESTAMP_PUBLIC_KEY_ENV, str(shared_public))
    monkeypatch.setenv(tier2.TIMESTAMP_JOURNAL_ID_ENV, "journal-test")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan(), sort_keys=True), encoding="utf-8")
    governance = sign_model_governance_artifact_receipt(
        plan_path,
        artifact_role="tier2_monthly_eex_evaluation_plan",
        authority_id="MODEL-RISK",
        issued_at_utc="2026-01-01T08:00:00Z",
        data_cutoff_utc="2025-12-31T23:59:59Z",
        private_key_path=shared_private,
    )
    governance_path = tmp_path / "governance.json"
    timestamp_path = tmp_path / "timestamp.json"
    governance_path.write_text(json.dumps(governance, sort_keys=True), encoding="utf-8")
    timestamp_path.write_text(
        json.dumps(_timestamp_receipt(plan_path, shared_private), sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="not independent"):
        tier2.verify_governed_plan_artifact(
            plan_path,
            governance_receipt_path=governance_path,
            timestamp_receipt_path=timestamp_path,
        )


def test_governed_plan_requires_holdout_after_trusted_preregistration_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    governance_private, governance_public = _write_keypair(tmp_path, "governance")
    timestamp_private, timestamp_public = _write_keypair(tmp_path, "timestamp")
    monkeypatch.setenv(MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_ENV, str(governance_public))
    monkeypatch.setenv(tier2.TIMESTAMP_PUBLIC_KEY_ENV, str(timestamp_public))
    monkeypatch.setenv(tier2.TIMESTAMP_JOURNAL_ID_ENV, "journal-test")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan(), sort_keys=True), encoding="utf-8")
    governance = sign_model_governance_artifact_receipt(
        plan_path,
        artifact_role="tier2_monthly_eex_evaluation_plan",
        authority_id="MODEL-RISK",
        issued_at_utc="2026-01-01T08:00:00Z",
        data_cutoff_utc="2025-12-31T23:59:59Z",
        private_key_path=governance_private,
    )
    governance_path = tmp_path / "governance.json"
    timestamp_path = tmp_path / "timestamp.json"
    governance_path.write_text(json.dumps(governance, sort_keys=True), encoding="utf-8")
    timestamp_path.write_text(
        json.dumps(
            _timestamp_receipt(
                plan_path,
                timestamp_private,
                received_at_utc="2028-01-01T08:00:00Z",
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="post-preregistration"):
        tier2.verify_governed_plan_artifact(
            plan_path,
            governance_receipt_path=governance_path,
            timestamp_receipt_path=timestamp_path,
        )


def test_trust_anchor_must_be_absolute_and_not_reparse_traversed(
    tmp_path: Path,
) -> None:
    _, public_path = _write_keypair(tmp_path, "trusted")
    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="must be absolute"):
        tier2._load_public_key(Path("trusted.public.pem"))

    link = tmp_path / "trusted-link.pem"
    try:
        link.symlink_to(public_path)
    except OSError:
        pytest.skip("Windows symlink creation is not enabled")
    with pytest.raises(tier2.Tier2MonthlyEexEvaluationError, match="symlink or junction"):
        tier2._load_public_key(link)


def _write_keypair(tmp_path: Path, stem: str) -> tuple[Path, Path]:
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / f"{stem}.private.pem"
    public_path = tmp_path / f"{stem}.public.pem"
    private_path.write_bytes(
        private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path


def _timestamp_receipt(
    path: Path,
    private_path: Path,
    *,
    received_at_utc: str = "2026-01-01T08:00:01Z",
) -> dict[str, object]:
    private = serialization.load_pem_private_key(private_path.read_bytes(), password=None)
    public_raw = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    artifact_schema = json.loads(path.read_text(encoding="utf-8"))["schema_version"]
    unsigned: dict[str, object] = {
        "schema_version": tier2.TIMESTAMP_RECEIPT_SCHEMA,
        "artifact_role": "tier2_monthly_eex_evaluation_plan",
        "artifact_schema_version": artifact_schema,
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "artifact_size_bytes": path.stat().st_size,
        "authority_id": "TRUSTED-TIME",
        "received_at_utc": received_at_utc,
        "journal_id": "journal-test",
        "journal_sequence": 0,
        "previous_receipt_id": "",
    }
    unsigned["receipt_id"] = tier2._sha256_json(unsigned)
    payload = tier2._canonical_json_bytes(unsigned)
    signed = dict(unsigned)
    signed["trusted_time_attestation"] = {
        "algorithm": "Ed25519",
        "key_id": hashlib.sha256(public_raw).hexdigest(),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "value_base64": base64.b64encode(private.sign(payload)).decode("ascii"),
    }
    return signed

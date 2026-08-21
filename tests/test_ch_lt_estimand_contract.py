from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import pfc_shaping.cli.audit_ch_lt_estimand_contract as cli_module
from pfc_shaping.cli.audit_ch_lt_estimand_contract import (
    _workspace_output,
    audit_estimand_contract,
    main,
)
from pfc_shaping.pipeline.governed_release_cli_contract import ReleaseCliIdentityError
from pfc_shaping.validation.ch_lt_estimand_contract import (
    STATUS,
    ChLtEstimandContractError,
    assess_estimand_contract,
    compute_contract_id,
    compute_validator_policy_sha256,
    load_estimand_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DRAFT = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "CH-LT-ESTIMAND-AND-ECONOMIC-DESIGN-DRAFT-20260724.json"
)
CANONICAL_SHA256 = "4209931e28a7c1cf2a4224d779f73648c4c9c5eac55df0a7ba1ad872226e2931"
CANONICAL_SEMANTIC_SHA256 = (
    "41ce07d1cf04e77a6936dc2d6f6fece387cbb415aa4588fa40406072e741b384"
)
VALIDATOR_POLICY_SHA256 = (
    "52c90167c51779724509d8a69ecc368c77a547f2eb2f3f55dbac10b98185a276"
)


def _draft() -> dict[str, object]:
    return json.loads(CANONICAL_DRAFT.read_text(encoding="utf-8"))


def _rehash(document: dict[str, object]) -> dict[str, object]:
    document["contract_id"] = compute_contract_id(document)
    return document


def _layer(document: dict[str, object], layer_id: str) -> dict[str, object]:
    layers = document["target_layers"]
    assert isinstance(layers, list)
    return next(
        layer
        for layer in layers
        if isinstance(layer, dict) and layer.get("layer_id") == layer_id
    )


def test_canonical_draft_is_hash_bound_and_fail_closed() -> None:
    payload = CANONICAL_DRAFT.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == CANONICAL_SHA256
    document, loaded_payload = load_estimand_contract(
        CANONICAL_DRAFT,
        expected_sha256=CANONICAL_SHA256,
    )

    assessment = assess_estimand_contract(document, document_bytes=loaded_payload)

    assert assessment.status == STATUS
    assert assessment.contract_id == compute_contract_id(document)
    assert assessment.source_document_sha256 == CANONICAL_SHA256
    assert assessment.canonical_semantic_sha256 == CANONICAL_SEMANTIC_SHA256
    assert assessment.validator_policy_sha256 == VALIDATOR_POLICY_SHA256
    assert compute_validator_policy_sha256() == VALIDATOR_POLICY_SHA256
    assert assessment.execution_authorized is False
    assert assessment.scientific_admission is False
    assert assessment.production_authorization is False
    assert assessment.promotion_gate is False
    assert set(assessment.blockers) == {
        "ORIGIN_AVAILABLE_CH_EEX_VINTAGE_MANIFEST_ABSENT",
        "MONTHLY_SOLVER_CONFIGURATION_MANIFEST_ABSENT",
        "FROZEN_CANDIDATE_BASELINE_IDENTITY_MANIFEST_ABSENT",
        "DIRECT_CH_HOURLY_TRUTH_MANIFEST_ABSENT",
        "DIRECT_CH_NATIVE_QUARTER_HOUR_TRUTH_MANIFEST_ABSENT",
        "EXACT_STATISTICAL_DESIGN_MANIFEST_ABSENT",
        "VERSIONED_CALENDAR_AND_STRATA_MANIFEST_ABSENT",
        "PROBABILISTIC_SCENARIO_DESIGN_MANIFEST_ABSENT",
        "EXACT_FMV_PROFILE_POPULATION_MANIFEST_ABSENT",
        "HYDRO_DISPATCH_POLICY_MANIFEST_ABSENT",
        "ECONOMIC_VALUATION_AND_FX_CONVENTION_MANIFEST_ABSENT",
        "FMV_RISK_ECONOMIC_MDE_APPROVAL_ABSENT",
        "EXACT_ORIGIN_TARGET_MASK_INVENTORY_ABSENT",
        "POST_EVALUATION_MARKET_CONSISTENCY_AUDIT_MANIFEST_ABSENT",
        "EXTERNAL_ESTIMAND_ADMISSION_ENVELOPE_NOT_IMPLEMENTED",
    }


def test_fake_local_evidence_hashes_still_cannot_authorize_evaluation() -> None:
    document = copy.deepcopy(_draft())
    evidence = document["evidence_bindings"]
    assert isinstance(evidence, dict)
    for index, field in enumerate(evidence):
        evidence[field] = format(index + 1, "x") * 64
    _rehash(document)

    assessment = assess_estimand_contract(document)

    assert assessment.execution_authorized is False
    assert assessment.scientific_admission is False
    assert assessment.blockers == (
        "EXTERNAL_ESTIMAND_ADMISSION_ENVELOPE_NOT_IMPLEMENTED",
    )


def test_weakened_horizon_is_rejected_even_when_rehashed() -> None:
    document = _draft()
    horizon = document["horizon_policy"]
    assert isinstance(horizon, dict)
    horizon["eligible_lead_month_max"] = 35
    _rehash(document)

    with pytest.raises(ChLtEstimandContractError, match="horizon policy"):
        assess_estimand_contract(document)


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("top", "production_authorization", True, "top-level field inventory"),
        ("lifecycle", "shadow_approval", "GO", "lifecycle field inventory"),
    ],
)
def test_unknown_authority_fields_are_rejected_even_when_rehashed(
    target: str, field: str, value: object, message: str
) -> None:
    document = _draft()
    selected = document if target == "top" else document["lifecycle"]
    assert isinstance(selected, dict)
    selected[field] = value
    _rehash(document)

    with pytest.raises(ChLtEstimandContractError, match=message):
        assess_estimand_contract(document)


def test_hourly_duplication_cannot_be_relabelled_as_quarter_hour_truth() -> None:
    document = _draft()
    layer = _layer(document, "QUARTER_HOURLY_SHAPE")
    layer["hourly_duplication_or_interpolation"] = "ALLOWED"
    _rehash(document)

    with pytest.raises(ChLtEstimandContractError, match="target layers"):
        assess_estimand_contract(document)


def test_proxy_truth_cannot_be_relabelled_as_direct_ch_truth() -> None:
    document = _draft()
    truth = document["truth_policy"]
    assert isinstance(truth, dict)
    truth["proxy_or_neighbor_truth_for_ch"] = "ALLOWED"
    _rehash(document)

    with pytest.raises(ChLtEstimandContractError, match="truth policy"):
        assess_estimand_contract(document)


def test_quarter_hour_product_cannot_be_selected_after_model_development() -> None:
    document = _draft()
    truth = document["truth_policy"]
    assert isinstance(truth, dict)
    truth["truth_product_id_freeze_boundary"] = "BEFORE_SCORING"
    _rehash(document)

    with pytest.raises(ChLtEstimandContractError, match="truth policy"):
        assess_estimand_contract(document)


def test_statistical_design_cannot_drop_dependence_or_mde() -> None:
    document = _draft()
    policy = document["statistical_decision_policy"]
    assert isinstance(policy, dict)
    required = policy["design_manifest_required_fields"]
    assert isinstance(required, list)
    required.remove("dependence_cluster_and_overlap_model")
    _rehash(document)

    with pytest.raises(ChLtEstimandContractError, match="statistical decision policy"):
        assess_estimand_contract(document)


def test_probabilistic_ensemble_must_preserve_solver_forward_expectation() -> None:
    document = _draft()
    policy = document["probabilistic_policy"]
    assert isinstance(policy, dict)
    policy["ensemble_monthly_forward_consistency"] = "PATHWISE_FIXED_LEVEL_ONLY"
    _rehash(document)

    with pytest.raises(ChLtEstimandContractError, match="probabilistic policy"):
        assess_estimand_contract(document)


def test_pfc_flavors_defaults_cannot_become_economic_evidence() -> None:
    document = _draft()
    profiles = document["profile_policy"]
    assert isinstance(profiles, dict)
    profiles["pfc_flavors_default_capture_premium_role"] = "MDE_EVIDENCE"
    _rehash(document)

    with pytest.raises(ChLtEstimandContractError, match="profile policy"):
        assess_estimand_contract(document)


def test_bloc_13_cannot_be_reduced_to_a_generic_fixed_profile() -> None:
    document = _draft()
    profiles = document["profile_policy"]
    assert isinstance(profiles, dict)
    fixed_roles = profiles["fixed_profile_roles"]
    assert isinstance(fixed_roles, list)
    fixed_roles.append("BLOC_13")
    _rehash(document)

    with pytest.raises(ChLtEstimandContractError, match="profile policy"):
        assess_estimand_contract(document)


def test_dispatch_capture_premium_shortcut_is_rejected() -> None:
    document = _draft()
    dispatch = document["dispatch_policy"]
    assert isinstance(dispatch, dict)
    dispatch["price_capture_premium_shortcut"] = "ALLOWED"
    _rehash(document)

    with pytest.raises(ChLtEstimandContractError, match="dispatch policy"):
        assess_estimand_contract(document)


def test_python_bool_integer_equivalence_cannot_bypass_controls() -> None:
    document = _draft()
    controls = document["execution_controls"]
    assert isinstance(controls, dict)
    controls["future_holdout_consumption"] = 0
    _rehash(document)

    with pytest.raises(ChLtEstimandContractError, match="execution controls"):
        assess_estimand_contract(document)


def test_mapping_and_document_bytes_must_be_identical() -> None:
    payload = CANONICAL_DRAFT.read_bytes()
    document = _draft()
    document["market"] = "DE"

    with pytest.raises(ChLtEstimandContractError, match="mapping/bytes mismatch"):
        assess_estimand_contract(document, document_bytes=payload)


def test_mapping_only_assessment_never_claims_a_source_document_hash() -> None:
    assessment = assess_estimand_contract(_draft())

    assert assessment.source_document_sha256 is None
    assert assessment.canonical_semantic_sha256 == CANONICAL_SEMANTIC_SHA256


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":"a","schema_version":"b"}',
        b'{"schema_version":NaN}',
    ],
)
def test_duplicate_keys_and_nonfinite_json_are_rejected(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "invalid-contract.json"
    path.write_bytes(payload)

    with pytest.raises(ChLtEstimandContractError, match="invalid JSON"):
        load_estimand_contract(path)


def test_audit_api_requires_exact_document_sha() -> None:
    result = audit_estimand_contract(
        contract_path=CANONICAL_DRAFT,
        expected_contract_sha256=CANONICAL_SHA256,
    )
    assert result["status"] == STATUS
    assert result["execution_authorized"] is False

    with pytest.raises(ChLtEstimandContractError, match="SHA-256 mismatch"):
        audit_estimand_contract(
            contract_path=CANONICAL_DRAFT,
            expected_contract_sha256="0" * 64,
        )


def test_cli_validates_draft_but_default_admission_fails_closed(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "assert_installed_runtime_sealed", lambda: None)
    common = [
        "--contract",
        str(CANONICAL_DRAFT),
        "--expected-contract-sha256",
        CANONICAL_SHA256,
    ]

    assert main([*common, "--mode", "validate-draft"]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["execution_authorized"] is False
    assert validated["audit_mode"] == "validate-draft"
    assert validated["command_exit_code"] == 0
    assert validated["runtime_identity"]["implementation"] == "CPython"

    assert main(common) == 3
    admission = json.loads(capsys.readouterr().out)
    assert admission["scientific_admission"] is False
    assert admission["audit_mode"] == "admit-evaluation"
    assert admission["command_exit_code"] == 3

    invalid = common.copy()
    invalid[-1] = "0" * 64
    assert main(invalid) == 2
    assert json.loads(capsys.readouterr().err)["status"] == "INVALID_ESTIMAND_CONTRACT"


def test_checkout_module_form_fails_closed_without_isolated_runtime() -> None:
    common = [
        sys.executable,
        "-B",
        "-m",
        "pfc_shaping.cli.audit_ch_lt_estimand_contract",
        "--contract",
        str(CANONICAL_DRAFT),
        "--expected-contract-sha256",
        CANONICAL_SHA256,
    ]

    rejected = subprocess.run(
        [*common, "--mode", "validate-draft"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert rejected.returncode == 2
    assert rejected.stdout == ""
    failure = json.loads(rejected.stderr)
    assert failure["status"] == "INVALID_GOVERNED_LT_RUNTIME"
    assert failure["production_authorization"] is False


def test_estimand_cli_rejects_unsealed_runtime_before_argument_parsing(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject() -> None:
        raise ReleaseCliIdentityError("runtime is not isolated")

    monkeypatch.setattr(cli_module, "assert_installed_runtime_sealed", reject)

    assert main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    failure = json.loads(captured.err)
    assert failure["status"] == "INVALID_GOVERNED_LT_RUNTIME"
    assert failure["command_id"] == "audit_ch_lt_estimand_contract"
    assert failure["production_authorization"] is False


def test_output_is_json_namespaced_durable_and_no_overwrite(
    tmp_path: Path,
) -> None:
    audit_root = tmp_path / "repo" / "output"
    namespace = (
        audit_root / "phase14" / "ch_lt_estimand_contract_audits"
    )
    namespace.mkdir(parents=True)
    output = namespace / "audit-001.json"

    result = audit_estimand_contract(
        contract_path=CANONICAL_DRAFT,
        expected_contract_sha256=CANONICAL_SHA256,
        mode="validate-draft",
        output_path=output,
        audit_root=audit_root,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert result["audit_operation_id"]
    with pytest.raises(ValueError, match="already exists"):
        audit_estimand_contract(
            contract_path=CANONICAL_DRAFT,
            expected_contract_sha256=CANONICAL_SHA256,
            mode="validate-draft",
            output_path=output,
            audit_root=audit_root,
        )
    with pytest.raises(ValueError, match=".json suffix"):
        _workspace_output(namespace / "not-json.txt", audit_root=audit_root)
    with pytest.raises(ValueError, match="canonical audit namespace"):
        _workspace_output(tmp_path / "outside.json", audit_root=audit_root)
    with pytest.raises(ValueError, match="--audit-root is required"):
        audit_estimand_contract(
            contract_path=CANONICAL_DRAFT,
            expected_contract_sha256=CANONICAL_SHA256,
            output_path=namespace / "missing-root.json",
        )


def test_hardlinked_contract_input_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "contract.json"
    second = tmp_path / "contract-hardlink.json"
    first.write_bytes(CANONICAL_DRAFT.read_bytes())
    try:
        os.link(first, second)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="single-link regular file"):
        load_estimand_contract(first, expected_sha256=CANONICAL_SHA256)

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import pfc_shaping.cli.audit_ch_lt_compute_runtime as cli_module
from pfc_shaping.cli.audit_ch_lt_compute_runtime import main
from pfc_shaping.pipeline.governed_release_cli_contract import ReleaseCliIdentityError
from pfc_shaping.validation.ch_lt_compute_runtime import (
    BLOCKERS,
    STATUS,
    ChLtComputeRuntimeError,
    assess_compute_runtime,
    compute_contract_id,
    derive_standardized_shock_seed,
    load_compute_runtime_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "CH-LT-COMPUTE-RUNTIME-DRAFT-20260727.json"
)


def _document() -> dict[str, object]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _rehash(document: dict[str, object]) -> None:
    document["contract_id"] = compute_contract_id(document)


def test_canonical_compute_contract_is_non_authoritative_and_hash_bound() -> None:
    payload = CONTRACT.read_bytes()
    document = json.loads(payload)

    result = assess_compute_runtime(document, document_bytes=payload)

    assert result.contract_id == compute_contract_id(document)
    assert result.document_sha256 == hashlib.sha256(payload).hexdigest()
    assert result.blockers == BLOCKERS
    audit = result.to_dict()
    assert audit["status"] == STATUS
    assert audit["execution_authorized"] is False
    assert audit["production_authorization"] is False
    assert audit["promotion_gate"] is False


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    [
        ("authority_boundary", "scenario_level_policy", "PATHWISE_FIXED", "authority"),
        ("rng_policy", "cuda_rng_forbidden_v1", False, "RNG policy"),
        ("determinism_policy", "tf32", True, "determinism"),
        ("parity_policy", "price_and_scenario_max_abs_eur_mwh", 1.0, "parity"),
        ("fallback_policy", "any_gpu_fit_attempt_v1", "FALLBACK_CPU", "fallback"),
        ("anti_leakage_policy", "future_data_injection_test_required", False, "anti-leakage"),
    ],
)
def test_policy_weakening_fails_even_when_rehashed(
    section: str,
    field: str,
    value: object,
    match: str,
) -> None:
    document = copy.deepcopy(_document())
    nested = document[section]
    assert isinstance(nested, dict)
    nested[field] = value
    _rehash(document)

    with pytest.raises(ChLtComputeRuntimeError, match=match):
        assess_compute_runtime(document)


def test_unknown_top_level_authority_field_is_rejected() -> None:
    document = _document()
    document["gpu_authorized"] = True
    _rehash(document)

    with pytest.raises(ChLtComputeRuntimeError, match="top-level keys"):
        assess_compute_runtime(document)


def test_bool_integer_equivalence_cannot_weaken_authority() -> None:
    document = _document()
    document["execution_authorized"] = 0
    _rehash(document)

    with pytest.raises(ChLtComputeRuntimeError, match="execution_authorized"):
        assess_compute_runtime(document)


def test_document_bytes_must_rebind_to_the_assessed_mapping() -> None:
    document = _document()

    with pytest.raises(ChLtComputeRuntimeError, match="do not bind"):
        assess_compute_runtime(document, document_bytes=b"{}")


def test_standardized_shock_seed_is_candidate_independent_and_domain_separated() -> None:
    common = {
        "plan_id": "a" * 64,
        "origin_id": "origin-1",
        "shock_stream_id": "comparison-family-1",
        "scenario_id": "scenario-7",
    }
    first = derive_standardized_shock_seed(**common, domain="standard-normal-v1")
    repeated = derive_standardized_shock_seed(**common, domain="standard-normal-v1")
    distinct = derive_standardized_shock_seed(**common, domain="bootstrap-v1")

    assert first == repeated
    assert 0 <= first < 2**128
    assert distinct != first


def test_zero_reference_and_zero_margin_semantics_are_explicit() -> None:
    document = _document()
    parity = document["parity_policy"]
    monte_carlo = document["monte_carlo_policy"]
    assert isinstance(parity, dict)
    assert isinstance(monte_carlo, dict)

    assert parity["below_relative_epsilon"] == "ABSOLUTE_CRITERION_ONLY"
    assert "ONE_SIDED_95_MC_CI" in str(monte_carlo["zero_margin_noninferiority_rule"])
    assert monte_carlo["missing_positive_reference_scale"] == "UNSUPPORTED"


def test_loader_rejects_wrong_caller_held_hash() -> None:
    with pytest.raises(ChLtComputeRuntimeError, match="SHA-256 mismatch"):
        load_compute_runtime_contract(CONTRACT, expected_sha256="0" * 64)


def test_cli_validates_structure_but_never_claims_execution(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "assert_installed_runtime_sealed", lambda: None)
    payload = CONTRACT.read_bytes()
    code = main(
        [
            "--contract",
            str(CONTRACT),
            "--expected-contract-sha256",
            hashlib.sha256(payload).hexdigest(),
        ]
    )

    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == STATUS
    assert result["execution_authorized"] is False


def test_cli_rejects_unsealed_runtime_before_argument_parsing(
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
    assert failure["command_id"] == "audit_ch_lt_compute_runtime"
    assert failure["production_authorization"] is False

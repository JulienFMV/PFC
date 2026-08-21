from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import pfc_shaping.validation.ch_lt_successor_candidate_core_v2 as candidate_core_v2
import pfc_shaping.validation.ch_lt_successor_candidate_core_v3 as candidate_core
import pfc_shaping.validation.ch_lt_successor_readiness as readiness
from pfc_shaping.validation.ch_lt_successor_readiness import (
    CONTRACT_ID,
    CONTRACT_RELATIVE_PATH,
    CONTRACT_SHA256,
    REQUIRED_BEFORE_CREATION,
    STATUS,
    ChLtSuccessorReadinessError,
    assess_successor_readiness,
    compute_contract_id,
    verify_successor_readiness,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT.joinpath(*CONTRACT_RELATIVE_PATH.split("/"))
ESTIMAND = ROOT / readiness.ESTIMAND_RELATIVE_PATH


def _document() -> dict[str, object]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _assessment_bytes(
    document: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> bytes:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    monkeypatch.setattr(readiness, "CONTRACT_SHA256", hashlib.sha256(payload).hexdigest())
    return payload


def test_canonical_readiness_contract_is_valid_but_successor_remains_uncreated() -> None:
    result = verify_successor_readiness(
        repo_root=ROOT,
        contract_path=CONTRACT,
        expected_contract_sha256=CONTRACT_SHA256,
    )

    assert hashlib.sha256(CONTRACT.read_bytes()).hexdigest() == CONTRACT_SHA256
    assert result["status"] == STATUS
    assert result["contract_id"] == CONTRACT_ID
    assert result["successor_exists"] is False
    assert result["candidate_core_authored"] is True
    assert result["candidate_core_hash_closed_locally"] is True
    assert result["candidate_core_frozen"] is False
    assert result["candidate_core_externally_frozen"] is False
    assert result["candidate_core_admitted"] is False
    assert result["readiness_complete"] is False
    assert result["blockers"] == list(REQUIRED_BEFORE_CREATION)
    assert result["future_holdout_consumed"] is False
    assert result["t057_consumed"] is False
    assert result["scientific_admission"] is False
    assert result["execution_authorized"] is False
    assert result["production_authorization"] is False
    assert result["promotion_gate"] is False


def test_contract_id_is_exact_semantic_hash() -> None:
    document = _document()

    assert compute_contract_id(document) == CONTRACT_ID


def test_economic_population_ids_match_the_bound_estimand_exactly() -> None:
    contract = _document()
    estimand = json.loads(ESTIMAND.read_text(encoding="utf-8"))

    assert contract["economic_materiality"]["required_simultaneous_populations"] == (
        estimand["profile_policy"]["required_population_ids"]
    )
    assert estimand["dispatch_policy"]["profile_id"] == "HYDRO_DISPATCH"


def test_verifier_never_reads_t057_or_future_truth_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_readiness = readiness.read_stable_single_link_file
    original_candidate_core = candidate_core.read_stable_single_link_file
    original_candidate_core_v2 = candidate_core_v2.read_stable_single_link_file
    observed: list[str] = []

    def _readiness_reader(path: Path, **kwargs: object) -> bytes:
        observed.append(str(path))
        return original_readiness(path, **kwargs)

    def _candidate_core_reader(path: Path, **kwargs: object) -> bytes:
        observed.append(str(path))
        return original_candidate_core(path, **kwargs)

    def _candidate_core_v2_reader(path: Path, **kwargs: object) -> bytes:
        observed.append(str(path))
        return original_candidate_core_v2(path, **kwargs)

    monkeypatch.setattr(readiness, "read_stable_single_link_file", _readiness_reader)
    monkeypatch.setattr(
        candidate_core, "read_stable_single_link_file", _candidate_core_reader
    )
    monkeypatch.setattr(
        candidate_core_v2, "read_stable_single_link_file", _candidate_core_v2_reader
    )
    verify_successor_readiness(
        repo_root=ROOT,
        contract_path=CONTRACT,
        expected_contract_sha256=CONTRACT_SHA256,
    )

    assert len(observed) == 17
    t057_reads = [path for path in observed if "T057" in path.upper()]
    assert len(t057_reads) == 1
    assert t057_reads[0].endswith("T057-OUTCOME-BLIND-TOMBSTONE-20260730.json")
    assert all("T057-EVIDENCE-SUPERSESSION-REGISTRY.json" not in path for path in observed)
    assert all("output\\phase14\\t057" not in path.lower() for path in observed)
    assert all("holdout_runner" not in path.lower() for path in observed)


def test_wrong_caller_held_hash_fails_closed() -> None:
    with pytest.raises(ChLtSuccessorReadinessError, match="frozen canonical identity"):
        verify_successor_readiness(
            repo_root=ROOT,
            contract_path=CONTRACT,
            expected_contract_sha256="0" * 64,
        )


def test_noncanonical_contract_path_fails_closed(tmp_path: Path) -> None:
    shadow = tmp_path / CONTRACT.name
    shadow.write_bytes(CONTRACT.read_bytes())

    with pytest.raises(ChLtSuccessorReadinessError, match="path is not canonical"):
        verify_successor_readiness(
            repo_root=ROOT,
            contract_path=shadow,
            expected_contract_sha256=CONTRACT_SHA256,
        )


def test_unknown_top_level_authority_field_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document()
    document["successor_approved"] = True
    payload = _assessment_bytes(document, monkeypatch)

    with pytest.raises(ChLtSuccessorReadinessError, match="keys are not exact"):
        assess_successor_readiness(document, document_bytes=payload)


def test_mapping_without_exact_byte_provenance_cannot_be_assessed() -> None:
    with pytest.raises(TypeError, match="document_bytes"):
        assess_successor_readiness(_document())  # type: ignore[call-arg]


def test_mapping_and_bytes_must_match(monkeypatch: pytest.MonkeyPatch) -> None:
    document = _document()
    other = _document()
    other["production_authorization"] = True
    payload = _assessment_bytes(other, monkeypatch)

    with pytest.raises(ChLtSuccessorReadinessError, match="document mapping/bytes"):
        assess_successor_readiness(document, document_bytes=payload)


def test_claiming_t057_consumption_cannot_be_rehashed_into_validity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document()
    holdout = document["holdout_non_consumption"]
    assert isinstance(holdout, dict)
    holdout["t057_consumed"] = True
    mutated_id = compute_contract_id(document)
    document["contract_id"] = mutated_id
    monkeypatch.setattr(readiness, "CONTRACT_ID", mutated_id)
    payload = _assessment_bytes(document, monkeypatch)

    with pytest.raises(ChLtSuccessorReadinessError, match="holdout non-consumption"):
        assess_successor_readiness(document, document_bytes=payload)


def test_satisfying_a_requirement_inside_structure_only_contract_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document()
    gate = document["successor_creation_gate"]
    assert isinstance(gate, dict)
    evidence = gate["current_requirement_evidence"]
    assert isinstance(evidence, list) and isinstance(evidence[0], dict)
    evidence[0]["status"] = "SATISFIED"
    evidence[0]["path"] = "forged.json"
    evidence[0]["sha256"] = "a" * 64
    mutated_id = compute_contract_id(document)
    document["contract_id"] = mutated_id
    monkeypatch.setattr(readiness, "CONTRACT_ID", mutated_id)
    payload = _assessment_bytes(document, monkeypatch)

    with pytest.raises(ChLtSuccessorReadinessError, match="requirement evidence"):
        assess_successor_readiness(document, document_bytes=payload)


def test_monthly_solver_expectation_authority_cannot_be_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document()
    policy = document["probabilistic_scenario_and_monte_carlo"]
    assert isinstance(policy, dict)
    policy["ensemble_monthly_forward_consistency"] = "PATHWISE_FIXED"
    mutated_id = compute_contract_id(document)
    document["contract_id"] = mutated_id
    monkeypatch.setattr(readiness, "CONTRACT_ID", mutated_id)
    payload = _assessment_bytes(document, monkeypatch)

    with pytest.raises(ChLtSuccessorReadinessError, match="probabilistic/scenario/MC"):
        assess_successor_readiness(document, document_bytes=payload)


@pytest.mark.parametrize(
    ("section", "field", "replacement", "error"),
    [
        (
            "dependence_power_and_multiplicity",
            "cross_fitting_substitutes_for_independent_confirmation",
            True,
            "dependence/power/multiplicity",
        ),
        (
            "deterministic_candidate_and_market_gates",
            "monthly_level_authority",
            "HOURLY_SHAPER",
            "deterministic candidate/market gate",
        ),
        (
            "probabilistic_scenario_and_monte_carlo",
            "insufficient_effective_calibration_cells",
            "PASS",
            "probabilistic/scenario/MC",
        ),
        (
            "economic_materiality",
            "required_simultaneous_populations",
            ["FIL", "ACC"],
            "economic materiality",
        ),
        (
            "compute_runtime_qualification",
            "solver_repricing_and_hard_market_gates_on_gpu",
            "ALLOWED",
            "compute runtime qualification",
        ),
    ],
)
def test_scientific_readiness_policies_are_closed(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    replacement: object,
    error: str,
) -> None:
    document = _document()
    policy = document[section]
    assert isinstance(policy, dict)
    policy[field] = replacement
    mutated_id = compute_contract_id(document)
    document["contract_id"] = mutated_id
    monkeypatch.setattr(readiness, "CONTRACT_ID", mutated_id)
    payload = _assessment_bytes(document, monkeypatch)

    with pytest.raises(ChLtSuccessorReadinessError, match=error):
        assess_successor_readiness(document, document_bytes=payload)

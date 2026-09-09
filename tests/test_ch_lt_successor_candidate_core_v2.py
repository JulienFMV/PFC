from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import pfc_shaping.cli.audit_ch_lt_successor_candidate_core as packaged_cli
import pfc_shaping.validation.ch_lt_successor_candidate_core_v2 as candidate_core
from pfc_shaping.pipeline.governed_release_cli_contract import ReleaseCliIdentityError
from pfc_shaping.validation.ch_lt_successor_candidate_core_v2 import (
    CORE_ID,
    CORE_RELATIVE_PATH,
    CORE_SHA256,
    REQUIRED_EVIDENCE,
    STATUS,
    ChLtSuccessorCandidateCoreV2Error,
    assess_candidate_core_v2,
    compute_core_id,
    verify_candidate_core_v2,
)
from scripts import audit_ch_lt_successor_candidate_core as audit_script

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.joinpath(*CORE_RELATIVE_PATH.split("/"))


def _document() -> dict[str, object]:
    return json.loads(CORE.read_text(encoding="utf-8"))


def _mutated_bytes(
    document: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> bytes:
    mutated_id = compute_core_id(document)
    document["core_id"] = mutated_id
    monkeypatch.setattr(candidate_core, "CORE_ID", mutated_id)
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    monkeypatch.setattr(candidate_core, "CORE_SHA256", hashlib.sha256(payload).hexdigest())
    return payload


def test_canonical_v2_core_is_outcome_blind_and_non_authoritative() -> None:
    result = verify_candidate_core_v2(
        repo_root=ROOT,
        core_path=CORE,
        expected_core_sha256=CORE_SHA256,
    )

    assert hashlib.sha256(CORE.read_bytes()).hexdigest() == CORE_SHA256
    assert compute_core_id(_document()) == CORE_ID
    assert result["status"] == STATUS
    assert result["core_id"] == CORE_ID
    assert result["candidate_core_authored"] is True
    assert result["candidate_core_hash_closed_locally"] is True
    assert result["candidate_core_frozen"] is False
    assert result["candidate_core_externally_frozen"] is False
    assert result["candidate_core_admitted"] is False
    assert result["successor_exists"] is False
    assert result["missing_evidence"] == list(REQUIRED_EVIDENCE)
    assert result["legacy_t057_outcome_metadata_exposed"] is True
    assert result["legacy_t057_confirmation_reuse"] is False
    for field in (
        "future_holdout_consumed",
        "t057_consumed",
        "scientific_admission",
        "execution_authorized",
        "production_authorization",
        "promotion_gate",
    ):
        assert result[field] is False


def test_v2_verifier_reads_only_one_outcome_blind_t057_tombstone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = candidate_core.read_stable_single_link_file
    observed: list[str] = []

    def _reader(path: Path, **kwargs: object) -> bytes:
        observed.append(path.as_posix())
        return original(path, **kwargs)

    monkeypatch.setattr(candidate_core, "read_stable_single_link_file", _reader)
    verify_candidate_core_v2(
        repo_root=ROOT,
        core_path=CORE,
        expected_core_sha256=CORE_SHA256,
    )

    assert len(observed) == 10
    t057_reads = [path for path in observed if "T057" in path.upper()]
    assert len(t057_reads) == 1
    assert t057_reads[0].endswith("T057-OUTCOME-BLIND-TOMBSTONE-20260730.json")
    assert all("T057-EVIDENCE-SUPERSESSION-REGISTRY.json" not in path for path in observed)
    assert all("holdout" not in path.lower() or "tombstone" in path.lower() for path in observed)


def test_wrong_caller_hash_and_noncanonical_path_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ChLtSuccessorCandidateCoreV2Error, match="frozen canonical identity"):
        verify_candidate_core_v2(
            repo_root=ROOT,
            core_path=CORE,
            expected_core_sha256="0" * 64,
        )
    shadow = tmp_path / CORE.name
    shadow.write_bytes(CORE.read_bytes())
    with pytest.raises(ChLtSuccessorCandidateCoreV2Error, match="path is not canonical"):
        verify_candidate_core_v2(
            repo_root=ROOT,
            core_path=shadow,
            expected_core_sha256=CORE_SHA256,
        )


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("lifecycle", "externally_frozen", True),
        ("outcome_blind_firewall", "outcome_bearing_registry_bytes", "ALLOWED"),
        ("overlap_aware_dependence_and_power", "minimum_resampling_block_length_origins", 12),
        ("monte_carlo_design_correction", "zero_margin_noninferiority_rule", "ZERO_WIDTH"),
        ("selection_and_hypothesis_freeze", "within_family_rule", "BEST_IN_SAMPLE"),
    ],
)
def test_rehashed_v2_policy_mutations_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    replacement: object,
) -> None:
    document = _document()
    policy = document[section]
    assert isinstance(policy, dict)
    policy[field] = replacement
    payload = _mutated_bytes(document, monkeypatch)

    with pytest.raises(ChLtSuccessorCandidateCoreV2Error, match="semantic section mismatch"):
        assess_candidate_core_v2(document, document_bytes=payload)


def test_current_local_cli_rejects_superseded_v2_core(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = audit_script.main(
        [
            "--core",
            str(CORE),
            "--expected-core-sha256",
            CORE_SHA256,
        ]
    )
    captured = capsys.readouterr()
    result = json.loads(captured.err)

    assert exit_code == 2
    assert captured.out == ""
    assert result["status"] == "INVALID_CANDIDATE_CORE_NO_GO"
    assert result["candidate_core_admitted"] is False
    assert result["execution_authorized"] is False


def test_packaged_cli_is_sealed_and_reports_structured_failures(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(packaged_cli, "assert_installed_runtime_sealed", lambda: None)
    assert packaged_cli.main(
        [
            "--evidence-root",
            str(ROOT),
            "--core",
            str(CORE),
            "--expected-core-sha256",
            "0" * 64,
        ]
    ) == 2
    captured = capsys.readouterr()
    failure = json.loads(captured.err)
    assert captured.out == ""
    assert failure["status"] == "INVALID_CANDIDATE_CORE_NO_GO"
    assert failure["candidate_core_admitted"] is False
    assert failure["successor_exists"] is False


def test_packaged_cli_rejects_unsealed_checkout_before_parsing(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _reject() -> None:
        raise ReleaseCliIdentityError("checkout runtime is not sealed")

    monkeypatch.setattr(packaged_cli, "assert_installed_runtime_sealed", _reject)
    assert packaged_cli.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["status"] == "INVALID_GOVERNED_LT_RUNTIME"

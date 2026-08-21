from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import pfc_shaping.validation.ch_lt_successor_candidate_core as candidate_core
from pfc_shaping.validation.ch_lt_successor_candidate_core import (
    CORE_ID,
    CORE_RELATIVE_PATH,
    CORE_SHA256,
    ChLtSuccessorCandidateCoreError,
    assess_candidate_core,
    compute_core_id,
    verify_candidate_core,
)

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


def test_v1_policy_bytes_remain_semantically_identified() -> None:
    document = _document()

    assert hashlib.sha256(CORE.read_bytes()).hexdigest() == CORE_SHA256
    assert compute_core_id(document) == CORE_ID
    result = assess_candidate_core(document, document_bytes=CORE.read_bytes())
    assert result["candidate_core_admitted"] is False
    assert result["execution_authorized"] is False
    assert result["production_authorization"] is False


def test_v1_verifier_is_permanently_disabled_before_any_artifact_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[Path] = []

    def _forbidden_reader(path: Path, **kwargs: object) -> bytes:
        observed.append(path)
        raise AssertionError("v1 verifier must not read artifacts")

    monkeypatch.setattr(candidate_core, "read_stable_single_link_file", _forbidden_reader)

    with pytest.raises(ChLtSuccessorCandidateCoreError, match="outcome-blind v2"):
        verify_candidate_core(
            repo_root=ROOT,
            core_path=CORE,
            expected_core_sha256=CORE_SHA256,
        )

    assert observed == []


def test_v1_wrong_caller_hash_still_fails_before_supersession_message() -> None:
    with pytest.raises(ChLtSuccessorCandidateCoreError, match="frozen canonical identity"):
        verify_candidate_core(
            repo_root=ROOT,
            core_path=CORE,
            expected_core_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        (
            "deterministic_market_and_complexity_gates",
            "unsupported_or_unstable_complexity_can_pass",
            True,
        ),
        (
            "dependence_power_and_multiplicity_design",
            "insufficient_effective_clusters_or_power",
            "PASS",
        ),
        (
            "economic_decision_design",
            "ompex_as_mde_or_population_truth",
            "ALLOWED",
        ),
        (
            "compute_runtime_design",
            "gpu_allowed",
            "GPU_FIT_AND_SELECTION_ALLOWED",
        ),
        (
            "origin_fold_and_truth_design",
            "future_episode_policy",
            "REUSABLE_AFTER_TRUTH",
        ),
    ],
)
def test_v1_rehashed_security_policy_mutations_are_rejected(
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

    with pytest.raises(ChLtSuccessorCandidateCoreError, match="semantic section mismatch"):
        assess_candidate_core(document, document_bytes=payload)

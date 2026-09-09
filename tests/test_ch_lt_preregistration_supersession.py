from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import pfc_shaping.cli.verify_ch_lt_preregistration_supersession as packaged_cli
import pfc_shaping.validation.ch_lt_preregistration_supersession as supersession
from pfc_shaping.pipeline.governed_release_cli_contract import ReleaseCliIdentityError
from pfc_shaping.validation.ch_lt_preregistration_supersession import (
    REGISTRY_RELATIVE_PATH,
    REGISTRY_SHA256,
    STATUS,
    ChLtPreregistrationSupersessionError,
    verify_preregistration_supersession,
)
from scripts.verify_ch_lt_preregistration_supersession import main

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT.joinpath(*REGISTRY_RELATIVE_PATH.split("/"))


def test_canonical_supersession_binds_all_documents_and_stays_no_go() -> None:
    result = verify_preregistration_supersession(
        repo_root=ROOT,
        registry_path=REGISTRY,
        expected_registry_sha256=REGISTRY_SHA256,
    )

    assert result["status"] == STATUS
    assert result["registry_sha256"] == hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
    assert result["successor_exists"] is False
    assert result["readiness_status"] == (
        "CANDIDATE_CORE_V3_CONTRAST_AWARE_LOCAL_HASH_CLOSED_NOT_EXTERNALLY_FROZEN_"
        "NOT_ADMITTED_SUCCESSOR_NOT_CREATED"
    )
    assert result["candidate_core_authored"] is True
    assert result["candidate_core_hash_closed_locally"] is True
    assert result["candidate_core_frozen"] is False
    assert result["candidate_core_externally_frozen"] is False
    assert result["candidate_core_admitted"] is False
    assert result["readiness_complete"] is False
    assert len(result["readiness_blockers"]) == 10
    assert result["superseded_v1_usable"] is False
    assert result["scientific_admission"] is False
    assert result["execution_authorized"] is False
    assert result["production_authorization"] is False
    assert result["promotion_gate"] is False


def test_wrong_caller_held_registry_hash_fails_closed() -> None:
    with pytest.raises(
        ChLtPreregistrationSupersessionError, match="frozen canonical identity"
    ):
        verify_preregistration_supersession(
            repo_root=ROOT,
            registry_path=REGISTRY,
            expected_registry_sha256="0" * 64,
        )


def test_rehashed_registry_cannot_replace_canonical_bytes_even_if_semantics_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    mutated = json.dumps(registry, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(mutated).hexdigest() != REGISTRY_SHA256
    with pytest.raises(
        ChLtPreregistrationSupersessionError, match="frozen canonical identity"
    ):
        verify_preregistration_supersession(
            repo_root=ROOT,
            registry_path=REGISTRY,
            expected_registry_sha256=hashlib.sha256(mutated).hexdigest(),
        )


def test_bound_estimand_policy_mutation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_reader = supersession.read_stable_single_link_file
    estimand_path = ROOT.joinpath(*supersession.ESTIMAND_RELATIVE_PATH.split("/"))
    document = json.loads(estimand_path.read_bytes())
    document["probabilistic_policy"]["ensemble_monthly_forward_consistency"] = (
        "PATHWISE_FIXED"
    )
    mutated = json.dumps(document, separators=(",", ":")).encode("utf-8")

    def _reader(path: Path, **kwargs: object) -> bytes:
        if Path(path) == estimand_path:
            return mutated
        return original_reader(path, **kwargs)

    monkeypatch.setattr(supersession, "read_stable_single_link_file", _reader)
    monkeypatch.setattr(
        supersession,
        "ESTIMAND_SHA256",
        hashlib.sha256(mutated).hexdigest(),
    )
    with pytest.raises(ChLtPreregistrationSupersessionError, match="clarification bindings"):
        verify_preregistration_supersession(
            repo_root=ROOT,
            registry_path=REGISTRY,
            expected_registry_sha256=REGISTRY_SHA256,
        )


def test_cli_reports_verified_supersession_without_authority(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "--registry",
                str(REGISTRY),
                "--expected-registry-sha256",
                REGISTRY_SHA256,
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == STATUS
    assert result["execution_authorized"] is False


def test_packaged_cli_reports_verified_supersession_without_authority(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(packaged_cli, "assert_installed_runtime_sealed", lambda: None)

    assert (
        packaged_cli.main(
            [
                "--evidence-root",
                str(ROOT),
                "--registry",
                str(REGISTRY),
                "--expected-registry-sha256",
                REGISTRY_SHA256,
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == STATUS
    assert result["execution_authorized"] is False
    assert result["production_authorization"] is False


def test_packaged_cli_rejects_unsealed_checkout_before_parsing(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject() -> None:
        raise ReleaseCliIdentityError("checkout runtime is not sealed")

    monkeypatch.setattr(packaged_cli, "assert_installed_runtime_sealed", reject)

    assert packaged_cli.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    failure = json.loads(captured.err)
    assert failure["status"] == "INVALID_GOVERNED_LT_RUNTIME"
    assert failure["execution_authorized"] is False
    assert failure["production_authorization"] is False

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import pfc_shaping.pipeline.atomic_promotion as atomic_promotion
import pfc_shaping.pipeline.governed_release as governed_release
import pfc_shaping.pipeline.promotion_contract as promotion_contract
from pfc_shaping.calibration.monthly_curve_lambda_calibration import config_hash
from pfc_shaping.pipeline.atomic_promotion import (
    EVENT_SIGNING_PRIVATE_KEY_ENV,
    PromotionBusyError,
    PromotionConflictError,
    PromotionProjectionRepairRequired,
    create_candidate_staging,
    finalize_candidate_staging,
)
from pfc_shaping.pipeline.candidate_evidence import (
    GOVERNED_RUN_SPEC_SCHEMA,
    REQUIRED_EVIDENCE_ROLES,
    EvidenceArtifactInput,
    _canonical_mapping_sha256,
    _run_spec_artifact_bindings,
    seal_candidate_evidence,
)
from pfc_shaping.pipeline.governed_release import (
    GovernedReleaseError,
    release_status,
)
from pfc_shaping.pipeline.governed_release import (
    audit_release_request as _audit_release_request,
)
from pfc_shaping.pipeline.governed_release import (
    promote_release_request as _promote_release_request,
)
from pfc_shaping.pipeline.model_governance_contract import (
    sign_model_governance_artifact_receipt,
)
from pfc_shaping.pipeline.promotion_contract import (
    PROMOTION_POLICY_VERSION,
    REQUIRED_PROMOTION_GATE_IDS,
    sign_promotion_receipt,
)
from pfc_shaping.version import __version__


def audit_release_request(**kwargs):
    _supply_only_request_id(kwargs)
    return _audit_release_request(**kwargs)


def promote_release_request(**kwargs):
    _supply_only_request_id(kwargs)
    return _promote_release_request(**kwargs)


def register_release_request(**kwargs):
    request_nonce = kwargs.pop("request_nonce", None)
    release = governed_release._absolute_root(
        kwargs["release_root"], name="release_root", must_exist=False
    )
    workflow = governed_release._external_workflow_root(
        kwargs["workflow_root"], release_root=release
    )
    workflow, workflow_domain_sha256 = governed_release.ensure_workflow_domain(
        workflow, create=True
    )
    evidence = governed_release._absolute_root(
        kwargs["evidence_root"], name="evidence_root", must_exist=True
    )
    bundle = governed_release.verify_candidate_bundle(release, run_id=kwargs["run_id"])
    captured = governed_release._capture_artifacts(evidence, kwargs["artifacts"])
    signing_key = os.environ[
        "PFC_TEST_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH"
    ]
    os.environ["PFC_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH"] = signing_key
    try:
        return governed_release._persist_release_request(
            workflow=workflow,
            workflow_domain_sha256=workflow_domain_sha256,
            release_root_sha256=governed_release._release_root_sha256(release),
            run_id=kwargs["run_id"],
            candidate_bundle_manifest_sha256=bundle.manifest_sha256,
            captured=captured,
            registration_contract="assembled_candidate_seal.v1",
            expected_current_event_id=kwargs["expected_current_event_id"],
            request_nonce=request_nonce,
        )
    finally:
        os.environ.pop("PFC_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH", None)


def _supply_only_request_id(kwargs: dict[str, object]) -> None:
    if "request_id" in kwargs:
        return
    workflow = Path(str(kwargs["workflow_root"]))
    run_id = str(kwargs["run_id"])
    matches = sorted((workflow / run_id / "requests").glob("*/release_request.json"))
    if len(matches) == 1:
        kwargs["request_id"] = json.loads(matches[0].read_text(encoding="utf-8"))[
            "request_id"
        ]


@pytest.fixture(autouse=True)
def _governance_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    private, public = _key_pair(tmp_path, "governance")
    monkeypatch.setenv(
        "PFC_TEST_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH",
        str(private),
    )
    monkeypatch.setenv("PFC_MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_PATH", str(public))
    monkeypatch.setenv(
        "PFC_PROMOTION_RELEASE_DOMAIN_ID",
        "123e4567-e89b-42d3-a456-426614174001",
    )
    register_private, register_public = _key_pair(tmp_path, "register")
    monkeypatch.setenv(
        "PFC_TEST_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH", str(register_private)
    )
    monkeypatch.setenv(
        "PFC_RELEASE_REQUEST_TRUSTED_PUBLIC_KEY_PATH", str(register_public)
    )
    monkeypatch.setenv(
        "PFC_RELEASE_WORKFLOW_DOMAIN_ID",
        "123e4567-e89b-42d3-a456-426614174002",
    )
    monkeypatch.setattr(
        governed_release,
        "_now_utc",
        lambda: "2026-07-13T00:03:00.123456+00:00",
    )
    monkeypatch.setattr(
        atomic_promotion,
        "verify_assembled_candidate_evidence",
        lambda *args, **kwargs: {"status": "TEST_UNIT_SEAL"},
    )
    monkeypatch.setattr(
        governed_release,
        "verify_assembled_candidate_evidence",
        lambda *args, **kwargs: {"status": "TEST_UNIT_SEAL"},
    )


@pytest.fixture
def release_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    receipt_private, receipt_public = _key_pair(tmp_path, "receipt")
    event_private, event_public = _key_pair(tmp_path, "event")
    monkeypatch.setenv("PFC_TEST_PROMOTION_SIGNING_PRIVATE_KEY_PATH", str(receipt_private))
    monkeypatch.setenv("PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH", str(receipt_public))
    monkeypatch.setenv("PFC_TEST_PROMOTION_EVENT_SIGNING_PRIVATE_KEY_PATH", str(event_private))
    monkeypatch.setenv("PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_PATH", str(event_public))
    monkeypatch.setenv("PFC_PROMOTION_JOURNAL_ROOT", str(tmp_path / "journal"))
    monkeypatch.setattr(
        atomic_promotion,
        "_promotion_now_utc",
        lambda: datetime(2026, 7, 13, 0, 5, tzinfo=timezone.utc),
    )
    return {
        "receipt_private": receipt_private,
        "receipt_public": receipt_public,
        "event_private": event_private,
        "event_public": event_public,
    }


def test_governed_release_audits_and_promotes_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_keys: dict[str, Path],
) -> None:
    release, workflow, evidence, data_root, bundle, artifacts = _fixture(tmp_path)
    request = register_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        artifacts=artifacts,
        expected_current_event_id=None,
    )
    assert request["created_at_utc"] == "2026-07-13T00:03:00+00:00"
    _install_capstone(monkeypatch, bundle=bundle, approved=True)
    monkeypatch.delenv(EVENT_SIGNING_PRIVATE_KEY_ENV, raising=False)

    audited = audit_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        data_root=data_root,
        run_id=bundle.run_id,
        signing_private_key=release_keys["receipt_private"],
    )
    resumed_audit = audit_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        data_root=data_root,
        run_id=bundle.run_id,
        signing_private_key=release_keys["receipt_private"],
    )
    assert audited == resumed_audit
    assert audited["approved"] is True
    assert audited["request_id"] == request["request_id"]
    request_slot = workflow / bundle.run_id / "requests" / str(request["request_id"])[:32]
    audit_slot = workflow / bundle.run_id / "audit-results" / str(request["request_id"])[:32]
    assert {path.name for path in request_slot.iterdir()} == {"release_request.json"}
    assert (audit_slot / "promotion_receipt.json").is_file()
    assert (audit_slot / "audit_result.json").is_file()

    monkeypatch.setenv(EVENT_SIGNING_PRIVATE_KEY_ENV, str(release_keys["event_private"]))
    monkeypatch.delenv("PFC_PROMOTION_SIGNING_PRIVATE_KEY_PATH", raising=False)
    promoted = promote_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
    )
    receipt_keyring = tmp_path / "receipt-keyring"
    receipt_keyring.mkdir()
    (receipt_keyring / "historical.pem").write_bytes(
        release_keys["receipt_public"].read_bytes()
    )
    _, new_receipt_public = _key_pair(tmp_path, "receipt-new")
    monkeypatch.setenv(
        "PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH",
        str(new_receipt_public),
    )
    monkeypatch.setenv(
        "PFC_PROMOTION_TRUSTED_PUBLIC_KEY_DIR",
        str(receipt_keyring),
    )
    resumed_promotion = promote_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
    )

    assert promoted == resumed_promotion
    assert promoted["status"] == "PROMOTED"
    promotion_result = (
        workflow
        / bundle.run_id
        / "promotion-results"
        / str(request["request_id"])[:32]
        / "promotion_result.json"
    )
    assert promotion_result.is_file()
    assert {path.name for path in request_slot.iterdir()} == {"release_request.json"}
    monkeypatch.delenv(EVENT_SIGNING_PRIVATE_KEY_ENV, raising=False)
    next_policy = "lt_promotion_policy.deterministic_only.v2"
    monkeypatch.setattr(
        promotion_contract,
        "PROMOTION_GATE_IDS_BY_POLICY",
        {
            **promotion_contract.PROMOTION_GATE_IDS_BY_POLICY,
            next_policy: promotion_contract.PROMOTION_GATE_IDS_BY_POLICY[
                PROMOTION_POLICY_VERSION
            ],
        },
    )
    monkeypatch.setattr(promotion_contract, "PROMOTION_POLICY_VERSION", next_policy)
    rotated_audit = audit_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        data_root=data_root,
        run_id=bundle.run_id,
        signing_private_key=None,
    )
    assert rotated_audit == audited
    monkeypatch.setenv(EVENT_SIGNING_PRIVATE_KEY_ENV, str(release_keys["event_private"]))
    assert (
        promote_release_request(
            release_root=release,
            workflow_root=workflow,
            evidence_root=evidence,
            run_id=bundle.run_id,
        )
        == promoted
    )
    monkeypatch.delenv(EVENT_SIGNING_PRIVATE_KEY_ENV, raising=False)
    tree_before_status = _tree_bytes(tmp_path)
    status = release_status(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
    )
    assert _tree_bytes(tmp_path) == tree_before_status
    assert status["audit_approved"] is True
    assert status["promoted_current"] is True
    assert status["state"] == "PROMOTED_CURRENT"
    (release / "current.json").unlink()
    repair_status = release_status(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        request_id=str(request["request_id"]),
    )
    assert repair_status["state"] == "POINTER_REPAIR_REQUIRED"
    assert repair_status["projection_repair_required"] is True


def test_promotion_result_failure_is_reported_as_committed_projection_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_keys: dict[str, Path],
) -> None:
    release, workflow, evidence, data_root, bundle, artifacts = _fixture(tmp_path)
    request = register_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        artifacts=artifacts,
        expected_current_event_id=None,
    )
    _install_capstone(monkeypatch, bundle=bundle, approved=True)
    audit_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        data_root=data_root,
        run_id=bundle.run_id,
        signing_private_key=release_keys["receipt_private"],
    )
    monkeypatch.setenv(EVENT_SIGNING_PRIVATE_KEY_ENV, str(release_keys["event_private"]))
    original_write = governed_release._write_result_idempotent

    def fail_promotion_projection(path: Path, payload: dict[str, object]):
        if path.name == "promotion_result.json":
            raise OSError("promotion result namespace is unavailable")
        return original_write(path, payload)

    monkeypatch.setattr(
        governed_release,
        "_write_result_idempotent",
        fail_promotion_projection,
    )
    with pytest.raises(PromotionProjectionRepairRequired) as raised:
        _promote_release_request(
            release_root=release,
            workflow_root=workflow,
            evidence_root=evidence,
            run_id=bundle.run_id,
            request_id=str(request["request_id"]),
        )

    assert raised.value.commit_status == "COMMITTED"
    assert (release / "current.json").is_file()
    assert len(atomic_promotion._load_immutable_journal_history(release.resolve())) == 1

    monkeypatch.setattr(
        governed_release,
        "_write_result_idempotent",
        original_write,
    )
    retried = _promote_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        request_id=str(request["request_id"]),
    )
    assert retried["status"] == "PROMOTED"
    assert retried["event_id"] == raised.value.event_id


def test_promotion_result_uses_receipt_hash_returned_by_atomic_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_keys: dict[str, Path],
) -> None:
    release, workflow, evidence, data_root, bundle, artifacts = _fixture(tmp_path)
    request = register_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        artifacts=artifacts,
        expected_current_event_id=None,
    )
    _install_capstone(monkeypatch, bundle=bundle, approved=True)
    audit_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        data_root=data_root,
        run_id=bundle.run_id,
        signing_private_key=release_keys["receipt_private"],
    )
    monkeypatch.setenv(EVENT_SIGNING_PRIVATE_KEY_ENV, str(release_keys["event_private"]))
    committed_receipt_sha256 = "c" * 64
    monkeypatch.setattr(
        governed_release,
        "promote_candidate",
        lambda *args, **kwargs: {
            "event_id": "e" * 64,
            "promotion_receipt_sha256": committed_receipt_sha256,
        },
    )

    result = _promote_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        request_id=str(request["request_id"]),
    )

    assert result["promotion_receipt_sha256"] == committed_receipt_sha256


def test_post_commit_timestamp_failure_is_reported_as_committed_projection_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_keys: dict[str, Path],
) -> None:
    release, workflow, evidence, data_root, bundle, artifacts = _fixture(tmp_path)
    request = register_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        artifacts=artifacts,
        expected_current_event_id=None,
    )
    _install_capstone(monkeypatch, bundle=bundle, approved=True)
    audit_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        data_root=data_root,
        run_id=bundle.run_id,
        signing_private_key=release_keys["receipt_private"],
    )
    monkeypatch.setenv(EVENT_SIGNING_PRIVATE_KEY_ENV, str(release_keys["event_private"]))
    monkeypatch.setattr(
        governed_release,
        "_now_utc",
        lambda: (_ for _ in ()).throw(OSError("clock unavailable after commit")),
    )

    with pytest.raises(PromotionProjectionRepairRequired) as raised:
        _promote_release_request(
            release_root=release,
            workflow_root=workflow,
            evidence_root=evidence,
            run_id=bundle.run_id,
            request_id=str(request["request_id"]),
        )

    assert raised.value.commit_status == "COMMITTED"
    assert (release / "current.json").is_file()
    assert len(atomic_promotion._load_immutable_journal_history(release.resolve())) == 1


def test_read_only_receipt_lookup_does_not_create_audit_namespace(
    tmp_path: Path,
) -> None:
    workflow = tmp_path / "workflow"
    workflow.mkdir()

    path = governed_release._receipt_path(
        workflow,
        "run-001",
        request_id="a" * 64,
        read_only=True,
    )

    assert not path.exists()
    assert not (workflow / "run-001").exists()


def test_audit_request_binding_never_recovers_release_domain_hardlink(
    tmp_path: Path,
    release_keys: dict[str, Path],
) -> None:
    release, workflow, evidence, data_root, bundle, artifacts = _fixture(tmp_path)
    request = register_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        artifacts=artifacts,
        expected_current_event_id=None,
    )
    marker = release / atomic_promotion.RELEASE_DOMAIN_MARKER
    temporary = release / ".pfc-999-abcdef123456.tmp"
    os.link(marker, temporary)

    with pytest.raises(atomic_promotion.PromotionError, match="forbidden hard links"):
        _audit_release_request(
            release_root=release,
            workflow_root=workflow,
            evidence_root=evidence,
            data_root=data_root,
            run_id=bundle.run_id,
            request_id=str(request["request_id"]),
            signing_private_key=release_keys["receipt_private"],
        )

    assert temporary.exists()
    assert marker.stat().st_nlink == 2


def test_release_status_missing_run_is_strictly_read_only(tmp_path: Path) -> None:
    release = tmp_path / "missing-release"
    workflow = tmp_path / "missing-workflow"
    evidence = tmp_path / "missing-evidence"

    status = release_status(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id="run-001",
    )

    assert status["schema_version"] == "governed_lt_release_status.v2"
    assert status["state"] == "NOT_FOUND"
    assert status["candidate_verified"] is False
    assert not release.exists()
    assert not workflow.exists()
    assert not evidence.exists()


def test_current_resolution_does_not_create_missing_release_domain_marker(
    tmp_path: Path,
    release_keys: dict[str, Path],
) -> None:
    release = tmp_path / "release"
    release.mkdir()

    with pytest.raises(atomic_promotion.PromotionError, match="domain marker is unavailable"):
        atomic_promotion.resolve_current_candidate(release)

    assert not (release / atomic_promotion.RELEASE_DOMAIN_MARKER).exists()


def test_current_resolution_does_not_recreate_missing_release_root(
    tmp_path: Path,
    release_keys: dict[str, Path],
) -> None:
    release = tmp_path / "missing-release"

    with pytest.raises(atomic_promotion.PromotionError, match="release root is unavailable"):
        atomic_promotion.resolve_current_candidate(release)

    assert not release.exists()


def test_mutating_public_api_rejects_unsealed_checkout_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pfc_shaping.pipeline.governed_release_cli_contract as cli_contract

    monkeypatch.setattr(cli_contract, "SOURCE_REVISION", None)

    with pytest.raises(GovernedReleaseError, match="installed runtime"):
        governed_release.register_release_request(
            release_root=tmp_path / "release",
            workflow_root=tmp_path / "workflow",
            evidence_root=tmp_path / "evidence",
            run_id="run-001",
            artifacts={},
            expected_current_event_id=None,
        )
    assert not (tmp_path / "release").exists()
    assert not (tmp_path / "workflow").exists()
    assert not (tmp_path / "evidence").exists()


def test_rollback_public_api_rejects_relative_paths_before_io(tmp_path: Path) -> None:
    with pytest.raises(GovernedReleaseError, match="release_root must be absolute"):
        governed_release.rollback_release(
            release_root="relative-release",
            rollback_authorization_path=tmp_path / "authorization.json",
        )

    assert not (Path.cwd() / "relative-release").exists()


def test_public_legacy_registration_api_is_disabled(
    tmp_path: Path,
    release_keys: dict[str, Path],
) -> None:
    release, workflow, evidence, data_root, bundle, artifacts = _fixture(tmp_path)
    del data_root, release_keys

    with pytest.raises(GovernedReleaseError, match="permanently disabled"):
        governed_release.register_release_request(
            release_root=release,
            workflow_root=workflow,
            evidence_root=evidence,
            run_id=bundle.run_id,
            artifacts=artifacts,
            expected_current_event_id=None,
        )


def test_workflow_json_publish_failure_leaves_no_partial_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "workflow" / "request.json"

    def fail_link(*args) -> None:
        raise OSError("injected")

    with monkeypatch.context() as failure:
        failure.setattr(governed_release.os, "link", fail_link)
        with pytest.raises(OSError, match="injected"):
            governed_release._write_json_exclusive(path, {"run_id": "run-001"})

    assert not path.exists()
    assert not list(path.parent.glob(".pfc-*.tmp"))
    governed_release._write_json_exclusive(path, {"run_id": "run-001"})
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "run-001"


def test_audit_identity_rejects_event_signing_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_keys: dict[str, Path],
) -> None:
    release, workflow, evidence, data_root, bundle, artifacts = _fixture(tmp_path)
    _register(release, workflow, evidence, bundle.run_id, artifacts)
    called = False

    def forbidden(_: list[str]) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(governed_release, "_run_capstone", forbidden)
    monkeypatch.setenv(EVENT_SIGNING_PRIVATE_KEY_ENV, str(release_keys["event_private"]))

    with pytest.raises(GovernedReleaseError, match="audit identity exposes forbidden"):
        audit_release_request(
            release_root=release,
            workflow_root=workflow,
            evidence_root=evidence,
            data_root=data_root,
            run_id=bundle.run_id,
            signing_private_key=release_keys["receipt_private"],
        )
    assert called is False


def test_audit_api_rejects_relative_private_key_before_workflow_io(
    tmp_path: Path,
) -> None:
    with pytest.raises(GovernedReleaseError, match="must be absolute"):
        governed_release.audit_release_request(
            release_root=tmp_path / "release",
            workflow_root=tmp_path / "workflow",
            evidence_root=tmp_path / "evidence",
            data_root=tmp_path / "data",
            run_id="run-001",
            request_id="a" * 64,
            signing_private_key="relative/receipt-private.pem",
        )

    assert not (tmp_path / "release").exists()
    assert not (tmp_path / "workflow").exists()
    assert not (tmp_path / "evidence").exists()


def test_audit_api_rejects_linked_private_key_before_workflow_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_keys: dict[str, Path],
) -> None:
    key = release_keys["receipt_private"]
    original = governed_release.path_is_link
    monkeypatch.setattr(
        governed_release,
        "path_is_link",
        lambda path: path == key or original(path),
    )

    with pytest.raises(GovernedReleaseError, match="cannot contain a link"):
        governed_release.audit_release_request(
            release_root=tmp_path / "release",
            workflow_root=tmp_path / "workflow",
            evidence_root=tmp_path / "evidence",
            data_root=tmp_path / "data",
            run_id="run-001",
            request_id="a" * 64,
            signing_private_key=key,
        )

    assert not (tmp_path / "release").exists()


def test_audit_api_rejects_hardlinked_private_key_before_workflow_io(
    tmp_path: Path,
    release_keys: dict[str, Path],
) -> None:
    key = release_keys["receipt_private"]
    linked = tmp_path / "receipt-private-linked.pem"
    try:
        os.link(key, linked)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable: {exc}")

    with pytest.raises(GovernedReleaseError, match="exactly one filesystem link"):
        governed_release.audit_release_request(
            release_root=tmp_path / "release",
            workflow_root=tmp_path / "workflow",
            evidence_root=tmp_path / "evidence",
            data_root=tmp_path / "data",
            run_id="run-001",
            request_id="a" * 64,
            signing_private_key=linked,
        )

    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize(
    "raw",
    [
        '{"status": NaN}',
        '{"status": Infinity}',
        '{"status": "first", "status": "second"}',
        ("[" * 130) + "0" + ("]" * 130),
    ],
)
def test_workflow_json_loader_rejects_ambiguous_documents(
    tmp_path: Path,
    raw: str,
) -> None:
    path = tmp_path / "workflow.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(GovernedReleaseError, match="cannot read workflow JSON"):
        governed_release._load_json(path)


def test_promotion_identity_rejects_receipt_private_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_keys: dict[str, Path],
) -> None:
    release, workflow, evidence, data_root, bundle, artifacts = _fixture(tmp_path)
    _register(release, workflow, evidence, bundle.run_id, artifacts)
    _install_capstone(monkeypatch, bundle=bundle, approved=True)
    monkeypatch.delenv(EVENT_SIGNING_PRIVATE_KEY_ENV, raising=False)
    audit_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        data_root=data_root,
        run_id=bundle.run_id,
        signing_private_key=release_keys["receipt_private"],
    )
    monkeypatch.setenv(EVENT_SIGNING_PRIVATE_KEY_ENV, str(release_keys["event_private"]))
    monkeypatch.setenv(
        "PFC_PROMOTION_SIGNING_PRIVATE_KEY_PATH",
        str(release_keys["receipt_private"]),
    )

    with pytest.raises(GovernedReleaseError, match="promote identity exposes forbidden"):
        promote_release_request(
            release_root=release,
            workflow_root=workflow,
            evidence_root=evidence,
            run_id=bundle.run_id,
        )
    assert not (release / "current.json").exists()


def test_evidence_mutation_blocks_audit_before_capstone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_keys: dict[str, Path],
) -> None:
    release, workflow, evidence, data_root, bundle, artifacts = _fixture(tmp_path)
    _register(release, workflow, evidence, bundle.run_id, artifacts)
    Path(artifacts["audit_gates"]).write_text("mutated", encoding="utf-8")
    monkeypatch.delenv(EVENT_SIGNING_PRIVATE_KEY_ENV, raising=False)
    called = False

    def forbidden(_: list[str]) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(governed_release, "_run_capstone", forbidden)

    with pytest.raises(GovernedReleaseError, match="artifact (size|hash) mismatch"):
        audit_release_request(
            release_root=release,
            workflow_root=workflow,
            evidence_root=evidence,
            data_root=data_root,
            run_id=bundle.run_id,
            signing_private_key=release_keys["receipt_private"],
        )
    assert called is False


@pytest.mark.parametrize(
    ("receipt_override", "message"),
    [
        ({"release_request_id": "different-request"}, "release_request_id mismatch"),
        (
            {"evidence_inventory_sha256": "0" * 64},
            "evidence inventory mismatch",
        ),
        (
            {"artifact_hash_role": "audit_gates"},
            "evidence artifact hash mismatch: audit_gates",
        ),
    ],
)
def test_audit_rejects_signed_receipt_not_bound_to_registered_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_keys: dict[str, Path],
    receipt_override: dict[str, str],
    message: str,
) -> None:
    release, workflow, evidence, data_root, bundle, artifacts = _fixture(tmp_path)
    _register(release, workflow, evidence, bundle.run_id, artifacts)
    _install_capstone(
        monkeypatch,
        bundle=bundle,
        approved=True,
        receipt_override=receipt_override,
    )
    monkeypatch.delenv(EVENT_SIGNING_PRIVATE_KEY_ENV, raising=False)

    with pytest.raises(GovernedReleaseError, match=message):
        audit_release_request(
            release_root=release,
            workflow_root=workflow,
            evidence_root=evidence,
            data_root=data_root,
            run_id=bundle.run_id,
            signing_private_key=release_keys["receipt_private"],
        )


def test_rejected_signed_receipt_is_resumable_but_never_promotable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_keys: dict[str, Path],
) -> None:
    release, workflow, evidence, data_root, bundle, artifacts = _fixture(tmp_path)
    _register(release, workflow, evidence, bundle.run_id, artifacts)
    _install_capstone(monkeypatch, bundle=bundle, approved=False)
    monkeypatch.delenv(EVENT_SIGNING_PRIVATE_KEY_ENV, raising=False)

    result = audit_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        data_root=data_root,
        run_id=bundle.run_id,
        signing_private_key=release_keys["receipt_private"],
    )
    assert result["status"] == "REJECTED"
    monkeypatch.setenv(EVENT_SIGNING_PRIVATE_KEY_ENV, str(release_keys["event_private"]))
    monkeypatch.delenv("PFC_PROMOTION_SIGNING_PRIVATE_KEY_PATH", raising=False)

    with pytest.raises(GovernedReleaseError, match="receipt is not approved"):
        promote_release_request(
            release_root=release,
            workflow_root=workflow,
            evidence_root=evidence,
            run_id=bundle.run_id,
        )
    assert not (release / "current.json").exists()


def test_audit_retry_recovers_result_without_private_key_after_receipt_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_keys: dict[str, Path],
) -> None:
    release, workflow, evidence, data_root, bundle, artifacts = _fixture(tmp_path)
    _register(release, workflow, evidence, bundle.run_id, artifacts)
    _install_capstone(monkeypatch, bundle=bundle, approved=True)
    monkeypatch.delenv(EVENT_SIGNING_PRIVATE_KEY_ENV, raising=False)
    expected = audit_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        data_root=data_root,
        run_id=bundle.run_id,
        signing_private_key=release_keys["receipt_private"],
    )
    result_path = (
        workflow
        / bundle.run_id
        / "audit-results"
        / str(expected["request_id"])[:32]
        / "audit_result.json"
    )
    result_path.unlink()
    release_keys["receipt_private"].unlink()

    recovered = audit_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        data_root=tmp_path / "missing-data-root",
        run_id=bundle.run_id,
        signing_private_key=None,
    )

    assert recovered == expected
    assert json.loads(result_path.read_text(encoding="utf-8"))["approved"] is True


def test_workflow_root_must_be_external_to_release_root(tmp_path: Path) -> None:
    release, _, evidence, _, bundle, artifacts = _fixture(tmp_path)

    with pytest.raises(GovernedReleaseError, match="external to release_root"):
        _register(
            release,
            release / "workflow",
            evidence,
            bundle.run_id,
            artifacts,
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows MAX_PATH contract")
def test_request_workflow_path_fails_before_windows_max_path(tmp_path: Path) -> None:
    workflow = tmp_path / ("w" * 120)

    with pytest.raises(GovernedReleaseError, match="MAX_PATH safety budget"):
        governed_release._request_dir(
            workflow.resolve(),
            "run-001",
            "a" * 64,
            create=True,
        )
    assert not (workflow / "run-001").exists()


def test_registration_seals_expected_head_and_keeps_requests_immutable(
    tmp_path: Path,
) -> None:
    release, workflow, evidence, _, bundle, artifacts = _fixture(tmp_path)
    first = register_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        artifacts=artifacts,
        expected_current_event_id=None,
    )
    retry = register_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        artifacts=artifacts,
        expected_current_event_id=None,
    )
    rebased = register_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        artifacts=artifacts,
        expected_current_event_id="b" * 64,
    )
    renewed = register_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        artifacts=artifacts,
        expected_current_event_id=None,
        request_nonce="receipt-renewal-001",
    )

    assert retry == first
    assert first["expected_current"] == {"kind": "NONE"}
    assert rebased["expected_current"] == {"kind": "EVENT", "event_id": "b" * 64}
    assert rebased["request_id"] != first["request_id"]
    assert renewed["request_nonce"] == "receipt-renewal-001"
    assert renewed["request_id"] != first["request_id"]
    for request in (first, rebased, renewed):
        path = (
            workflow
            / bundle.run_id
            / "requests"
            / str(request["request_id"])[:32]
            / "release_request.json"
        )
        assert json.loads(path.read_text(encoding="utf-8")) == request


def test_multiple_requests_require_explicit_request_id_for_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, workflow, evidence, data_root, bundle, artifacts = _fixture(tmp_path)
    register_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        artifacts=artifacts,
        expected_current_event_id=None,
    )
    register_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        artifacts=artifacts,
        expected_current_event_id="c" * 64,
    )
    monkeypatch.delenv(EVENT_SIGNING_PRIVATE_KEY_ENV, raising=False)

    with pytest.raises(TypeError, match="request_id"):
        audit_release_request(
            release_root=release,
            workflow_root=workflow,
            evidence_root=evidence,
            data_root=data_root,
            run_id=bundle.run_id,
            signing_private_key=None,
        )


def test_request_lookup_rejects_same_prefix_with_different_full_id(
    tmp_path: Path,
) -> None:
    release, workflow, evidence, data_root, bundle, artifacts = _fixture(tmp_path)
    request = register_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        artifacts=artifacts,
        expected_current_event_id=None,
    )
    forged = str(request["request_id"])[:32] + ("f" * 32)
    if forged == request["request_id"]:
        forged = str(request["request_id"])[:32] + ("e" * 32)

    with pytest.raises(GovernedReleaseError, match="full request_id mismatch"):
        _audit_release_request(
            release_root=release,
            workflow_root=workflow,
            evidence_root=evidence,
            data_root=data_root,
            run_id=bundle.run_id,
            request_id=forged,
            signing_private_key=None,
        )
    with pytest.raises(GovernedReleaseError, match="full request_id mismatch"):
        _promote_release_request(
            release_root=release,
            workflow_root=workflow,
            evidence_root=evidence,
            run_id=bundle.run_id,
            request_id=forged,
        )
    with pytest.raises(GovernedReleaseError, match="full request_id mismatch"):
        release_status(
            release_root=release,
            workflow_root=workflow,
            evidence_root=evidence,
            run_id=bundle.run_id,
            request_id=forged,
        )


@pytest.mark.parametrize(
    "primary_error",
    [
        PromotionConflictError("conflict"),
        PromotionBusyError("busy"),
        PromotionProjectionRepairRequired("a" * 64),
    ],
    ids=["conflict", "busy", "projection-repair"],
)
def test_failure_telemetry_cannot_mask_transition_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_keys: dict[str, Path],
    primary_error: Exception,
) -> None:
    release, workflow, evidence, data_root, bundle, artifacts = _fixture(tmp_path)
    request = register_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        artifacts=artifacts,
        expected_current_event_id=None,
    )
    _install_capstone(monkeypatch, bundle=bundle, approved=True)
    audit_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        data_root=data_root,
        run_id=bundle.run_id,
        request_id=str(request["request_id"]),
        signing_private_key=release_keys["receipt_private"],
    )
    monkeypatch.setenv(EVENT_SIGNING_PRIVATE_KEY_ENV, str(release_keys["event_private"]))

    def fail_transition(*args, **kwargs):
        raise primary_error

    def fail_telemetry(*args, **kwargs):
        raise OSError("workflow volume unavailable")

    monkeypatch.setattr(governed_release, "promote_candidate", fail_transition)
    monkeypatch.setattr(governed_release, "_record_failure", fail_telemetry)

    with pytest.raises(type(primary_error), match=str(primary_error)):
        _promote_release_request(
            release_root=release,
            workflow_root=workflow,
            evidence_root=evidence,
            run_id=bundle.run_id,
            request_id=str(request["request_id"]),
        )


def test_status_is_bound_to_request_not_only_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_keys: dict[str, Path],
) -> None:
    release, workflow, evidence, data_root, bundle, artifacts = _fixture(tmp_path)
    first = register_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        artifacts=artifacts,
        expected_current_event_id=None,
    )
    _install_capstone(monkeypatch, bundle=bundle, approved=True)
    audit_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        data_root=data_root,
        run_id=bundle.run_id,
        request_id=str(first["request_id"]),
        signing_private_key=release_keys["receipt_private"],
    )
    monkeypatch.setenv(EVENT_SIGNING_PRIVATE_KEY_ENV, str(release_keys["event_private"]))
    first_result = _promote_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        request_id=str(first["request_id"]),
    )
    monkeypatch.delenv(EVENT_SIGNING_PRIVATE_KEY_ENV, raising=False)
    second = register_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        artifacts=artifacts,
        expected_current_event_id=str(first_result["event_id"]),
        request_nonce="second-request",
    )
    audit_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        data_root=data_root,
        run_id=bundle.run_id,
        request_id=str(second["request_id"]),
        signing_private_key=release_keys["receipt_private"],
    )
    monkeypatch.setenv(EVENT_SIGNING_PRIVATE_KEY_ENV, str(release_keys["event_private"]))
    _promote_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        request_id=str(second["request_id"]),
    )
    monkeypatch.delenv(EVENT_SIGNING_PRIVATE_KEY_ENV, raising=False)

    old_status = release_status(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        request_id=str(first["request_id"]),
    )
    current_status = release_status(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=bundle.run_id,
        request_id=str(second["request_id"]),
    )

    assert old_status["state"] == "PROMOTED_SUPERSEDED"
    assert old_status["promoted_current"] is False
    assert current_status["state"] == "PROMOTED_CURRENT"
    assert current_status["promoted_current"] is True


def _fixture(tmp_path: Path):
    release = tmp_path / "releases"
    workflow = tmp_path / "workflow"
    evidence = tmp_path / "evidence"
    data_root = tmp_path / "data"
    evidence.mkdir()
    data_root.mkdir()
    bundle = _candidate(release, "run-001")
    artifacts: dict[str, Path] = {}
    for role in governed_release.REQUIRED_ARTIFACT_ROLES:
        path = evidence / f"{role}.json"
        path.write_text(json.dumps({"role": role}), encoding="utf-8")
        artifacts[role] = path
    return release, workflow, evidence, data_root, bundle, artifacts


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _register(
    release: Path,
    workflow: Path,
    evidence: Path,
    run_id: str,
    artifacts: dict[str, Path],
) -> dict[str, object]:
    return register_release_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=run_id,
        artifacts=artifacts,
        expected_current_event_id=None,
    )


def _candidate(root: Path, run_id: str):
    staging = create_candidate_staging(root, run_id=run_id)
    (staging / "payload.txt").write_text("candidate", encoding="utf-8")
    market = staging / "markets" / "CH" / "pfc.csv"
    model = staging / "models" / "shared" / "shape.bin"
    market.parent.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    market.write_text("candidate", encoding="utf-8")
    model.write_bytes(b"deterministic-model")
    deterministic_artifacts = {
        path.relative_to(staging).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (market, model)
    }
    deterministic_sha256 = hashlib.sha256(
        json.dumps(
            deterministic_artifacts,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = staging / "manifests" / "candidate_run_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "lt_candidate_run.v1",
                "run_id": run_id,
                "pipeline_scope": "LT_ONLY",
                "promotion_eligible": False,
                "reference_timestamp": "2026-07-13T00:00:00+00:00",
                "reference_timestamp_is_explicit": True,
                "candidate_serialized_at_utc": "2026-07-13T00:01:00+00:00",
                "deterministic_artifacts": deterministic_artifacts,
                "deterministic_artifacts_sha256": deterministic_sha256,
            }
        ),
        encoding="utf-8",
    )
    _seal_evidence(staging, run_id)
    return finalize_candidate_staging(root, run_id=run_id, staging_path=staging)


def _seal_evidence(staging: Path, run_id: str) -> None:
    evidence = staging / "evidence" / "promotion"
    evidence.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, EvidenceArtifactInput] = {}
    for role in sorted(REQUIRED_EVIDENCE_ROLES):
        artifact = evidence / f"{role}.json"
        if role == "historical_thresholds":
            _write_historical_threshold_fixture(artifact)
        elif role == "selected_lambda_decision":
            canonical = {"lambda_prior": 1.0}
            artifact.write_text(
                json.dumps(
                    {
                        "schema_version": "monthly_curve_selected_lambda_decision.v1",
                        "decision_id": "decision-test",
                        "calibration_campaign_id": "campaign-test",
                        "selection_reason": "test fixture",
                        "selection_status": "PRODUCTION_APPROVED",
                        "production_approved": True,
                        "canonical_config": canonical,
                        "config_hash": config_hash(canonical),
                    }
                ),
                encoding="utf-8",
            )
        else:
            artifact.write_text(json.dumps({"role": role}), encoding="utf-8")
        receipt = None
        if role in {"historical_thresholds", "selected_lambda_decision"}:
            receipt = evidence / f"{role}.authority.json"
            receipt.write_text(
                json.dumps(
                    sign_model_governance_artifact_receipt(
                        artifact,
                        artifact_role=role,
                        authority_id="FMV_MODEL_GOVERNANCE_TEST",
                        issued_at_utc="2026-07-12T00:00:00Z",
                        data_cutoff_utc="2026-07-11T00:00:00Z",
                        private_key_path=os.environ[
                            "PFC_TEST_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH"
                        ],
                    )
                ),
                encoding="utf-8",
            )
        artifacts[role] = EvidenceArtifactInput(
            path=artifact,
            producer_contract="test.fixture.v1",
            authority_receipt_path=receipt,
        )
    _bind_pre_run_fixture(staging, run_id, artifacts)
    seal_candidate_evidence(staging, run_id=run_id, artifacts=artifacts)


def _write_historical_threshold_fixture(path: Path) -> None:
    columns = [
        "gate_id", "metric", "market", "delivery_bucket", "lookback_start",
        "lookback_end", "n_snapshots", "min_required_n", "p50", "p90",
        "p975", "max_observed", "regime_filter", "status",
    ]
    lines = [",".join(columns)]
    for gate, metric in (
        ("same_month_rank_consistency", "same_month_shape_delta_abs_eur_mwh"),
        ("residual_vs_implied_comparable_block", "comparable_block_shape_delta_abs_eur_mwh"),
    ):
        for bucket in ["all", *(f"month_{month:02d}" for month in range(1, 13))]:
            lines.append(
                f"{gate},{metric},CH,{bucket},,,0,24,,,,,test_fixture,UNSUPPORTED"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bind_pre_run_fixture(
    staging: Path,
    run_id: str,
    artifacts: dict[str, EvidenceArtifactInput],
) -> None:
    run_path = staging / "manifests" / "candidate_run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    entries: dict[str, object] = {}
    for role in ("historical_thresholds", "selected_lambda_decision"):
        item = artifacts[role]
        artifact = Path(item.path)
        receipt = Path(item.authority_receipt_path)
        receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
        entries[role] = {
            "path": artifact.relative_to(staging).as_posix(),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "size_bytes": artifact.stat().st_size,
            "authority_receipt": {
                "path": receipt.relative_to(staging).as_posix(),
                "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                "receipt_id": receipt_payload["receipt_id"],
                "authority_id": receipt_payload["authority_id"],
                "issued_at_utc": receipt_payload["issued_at_utc"],
                "data_cutoff_utc": receipt_payload["data_cutoff_utc"],
            },
        }
    config_path = staging / "evidence" / "independent" / "runtime_config" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("model: {}\n", encoding="utf-8")
    config_payload = {"model": {}}
    entries["runtime_config"] = {
        "path": config_path.relative_to(staging).as_posix(),
        "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "size_bytes": config_path.stat().st_size,
        "semantic_sha256": hashlib.sha256(
            json.dumps(
                config_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
    }
    run_spec = {
        "schema_version": GOVERNED_RUN_SPEC_SCHEMA,
        "run_id": run_id,
        "as_of_utc": run["reference_timestamp"],
        "build_timestamp_utc": "2026-07-13T00:01:00+00:00",
        "runtime": {
            "distribution": "fmv-pfc-lt",
            "version": __version__,
            "source_revision": "1" * 40,
        },
        "config": {
            key: entries["runtime_config"][key]
            for key in ("path", "sha256", "semantic_sha256")
        },
        "input_snapshot": {
            "generation_id": "generation-001",
            "contract_logical_path": "snapshots/generation-001/lt_input_snapshot.json",
            "contract_sha256": "2" * 64,
            "pointer_logical_path": "views/pfc_lt/current.json",
            "pointer_sha256": "3" * 64,
        },
        "model_controls": {
            "peak_source_policy": "same_first",
            "use_seasonal_hourly_shape": True,
        },
        "pre_run_artifacts": _run_spec_artifact_bindings(entries),
    }
    pre_path = staging / "manifests" / "pre_run_governance_manifest.json"
    pre_path.write_text(
        json.dumps(
            {
                "schema_version": "pre_run_model_governance.v2",
                "run_id": run_id,
                "valuation_timestamp": run["reference_timestamp"],
                "captured_at_utc": "2026-07-13T00:01:00+00:00",
                "artifacts": entries,
                "governed_run_spec": run_spec,
                "governed_run_spec_sha256": _canonical_mapping_sha256(run_spec),
                "promotion_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    run["pre_run_governance_manifest"] = "manifests/pre_run_governance_manifest.json"
    run["pre_run_governance_manifest_sha256"] = hashlib.sha256(
        pre_path.read_bytes()
    ).hexdigest()
    run_path.write_text(json.dumps(run), encoding="utf-8")


def _install_capstone(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bundle,
    approved: bool,
    receipt_override: dict[str, str] | None = None,
) -> None:
    def fake_capstone(argv: list[str]) -> int:
        output = Path(argv[argv.index("--output") + 1])
        private_key = Path(argv[argv.index("--signing-private-key") + 1])
        request_path = Path(argv[argv.index("--registered-release-request") + 1])
        registered_request = json.loads(request_path.read_text(encoding="utf-8"))
        statuses = {
            gate_id: "PASS" for gate_id in sorted(REQUIRED_PROMOTION_GATE_IDS)
        }
        if not approved:
            statuses[sorted(REQUIRED_PROMOTION_GATE_IDS)[0]] = "CRITICAL"
        payload = {
            "schema_version": "promotion_receipt.v2",
            "promotion_policy_version": PROMOTION_POLICY_VERSION,
            "authorization_mode": "REGISTERED_PROMOTION",
            "run_id": bundle.run_id,
            "approved": approved,
            "valuation_timestamp": "2026-07-13T00:00:00+00:00",
            "promotion_evaluated_at": "2026-07-13T00:04:00+00:00",
            "candidate_bundle_manifest_sha256": bundle.manifest_sha256,
            "candidate_run_manifest_sha256": hashlib.sha256(
                (bundle.path / "manifests" / "candidate_run_manifest.json").read_bytes()
            ).hexdigest(),
            "release_request_id": argv[argv.index("--release-request-id") + 1],
            "release_operation_id": argv[argv.index("--release-operation-id") + 1],
            "expected_current_event_id": (
                None
                if argv[argv.index("--expected-current-event-id") + 1] == "NONE"
                else argv[argv.index("--expected-current-event-id") + 1]
            ),
            "operation_created_at_utc": argv[
                argv.index("--operation-created-at-utc") + 1
            ],
            "release_root_sha256": argv[argv.index("--release-root-sha256") + 1],
            "evidence_inventory_sha256": argv[
                argv.index("--evidence-inventory-sha256") + 1
            ],
            "workflow_domain_sha256": registered_request[
                "workflow_domain_sha256"
            ],
            "release_request_document_sha256": hashlib.sha256(
                request_path.read_bytes()
            ).hexdigest(),
            "input_artifacts": {
                role: {
                    "path": str(Path(argv[argv.index(flag) + 1]).resolve()),
                    "sha256": hashlib.sha256(
                        Path(argv[argv.index(flag) + 1]).read_bytes()
                    ).hexdigest(),
                    "present": True,
                }
                for role, flag in {
                    "audit_gates": "--audit-gates",
                    "historical_thresholds": "--historical-thresholds",
                    "production_manifest": "--production-manifest",
                    "export_manifest": "--export-manifest",
                    "selected_config_artifact": "--selected-config-artifact",
                    "product_normalization_summary": "--product-normalization-summary",
                }.items()
            },
            "decision_summary": {
                "status": "PROMOTION_EVIDENCE_PASS" if approved else "BLOCKED",
                "approved": approved,
                "blocking_count": 0 if approved else 1,
            },
            "governance_gates": [
                {"gate_id": gate_id, "status": status}
                for gate_id, status in statuses.items()
            ],
        }
        override = dict(receipt_override or {})
        artifact_hash_role = override.pop("artifact_hash_role", None)
        payload.update(override)
        if artifact_hash_role is not None:
            payload["input_artifacts"][artifact_hash_role]["sha256"] = "0" * 64
        signed = sign_promotion_receipt(payload, private_key_path=private_key)
        output.write_text(json.dumps(signed), encoding="utf-8")
        return 0 if approved else 1

    monkeypatch.setattr(governed_release, "_run_capstone", fake_capstone)


def _key_pair(root: Path, name: str) -> tuple[Path, Path]:
    private_key = Ed25519PrivateKey.generate()
    private_path = root / f"{name}-private.pem"
    public_path = root / f"{name}-public.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path

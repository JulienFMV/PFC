from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import pfc_shaping.pipeline.atomic_promotion as atomic_promotion
import pfc_shaping.pipeline.governed_release_cli_contract as release_cli_contract
from pfc_shaping.calibration.monthly_curve_lambda_calibration import config_hash
from pfc_shaping.pipeline.atomic_promotion import (
    PromotionBusyError,
    PromotionConflictError,
    PromotionError,
    PromotionProjectionRepairRequired,
    create_candidate_staging,
    finalize_candidate_staging,
    resolve_current_candidate,
    rollback_to_candidate,
    verify_candidate_bundle,
)
from pfc_shaping.pipeline.atomic_promotion import (
    promote_candidate as _promote_candidate,
)
from pfc_shaping.pipeline.candidate_evidence import (
    GOVERNED_RUN_SPEC_SCHEMA,
    REQUIRED_EVIDENCE_ROLES,
    EvidenceArtifactInput,
    _canonical_mapping_sha256,
    _run_spec_artifact_bindings,
    seal_candidate_evidence,
)
from pfc_shaping.pipeline.model_governance_contract import (
    sign_model_governance_artifact_receipt,
)
from pfc_shaping.pipeline.promotion_contract import (
    PROMOTION_POLICY_VERSION,
    REQUIRED_PROMOTION_GATE_IDS,
    sign_promotion_event,
    sign_promotion_head,
    sign_promotion_receipt,
    sign_rollback_authorization,
)
from pfc_shaping.version import __version__

_EXPECTED_FROM_HISTORY = object()


def promote_candidate(*args, **kwargs):
    """Unit-test promotion mechanics behind a simulated assembled-candidate seal."""

    kwargs["require_assembled_evidence"] = True
    receipt = json.loads(Path(kwargs["promotion_receipt_path"]).read_text(encoding="utf-8"))
    contract = {
        "expected_current_event_id": receipt["expected_current_event_id"],
        "operation_id": receipt["release_operation_id"],
        "release_request_id": receipt["release_request_id"],
        "operation_created_at_utc": receipt["operation_created_at_utc"],
    }
    for key, value in contract.items():
        kwargs.setdefault(key, value)
    receipt_path = Path(kwargs["promotion_receipt_path"])
    kwargs.setdefault("registered_release_request_path", receipt_path.with_name("release_request.json"))
    kwargs.setdefault("workflow_root", receipt_path.parent)
    request = {
        "schema_version": "governed_lt_release_request.v3",
        "run_id": kwargs["run_id"],
        "request_id": kwargs["release_request_id"],
        "created_at_utc": kwargs["operation_created_at_utc"],
        "release_root_sha256": receipt["release_root_sha256"],
        "workflow_domain_sha256": receipt["workflow_domain_sha256"],
        "registration_contract": "assembled_candidate_seal.v1",
        "candidate_bundle_manifest_sha256": receipt[
            "candidate_bundle_manifest_sha256"
        ],
        "expected_current": (
            {"kind": "NONE"}
            if kwargs["expected_current_event_id"] is None
            else {"kind": "EVENT", "event_id": kwargs["expected_current_event_id"]}
        ),
    }
    with patch.object(
        atomic_promotion,
        "verify_assembled_candidate_evidence",
        return_value={"status": "TEST_UNIT_SEAL"},
    ), patch.object(
        atomic_promotion,
        "load_registered_release_request_evidence",
        return_value=(request, receipt["release_request_document_sha256"]),
    ):
        return _promote_candidate(*args, **kwargs)


def _contract_from_receipt(path: Path) -> dict[str, object]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    return {
        "expected_current_event_id": receipt["expected_current_event_id"],
        "operation_id": receipt["release_operation_id"],
        "release_request_id": receipt["release_request_id"],
        "operation_created_at_utc": receipt["operation_created_at_utc"],
    }


def _rollback_authorization(
    path: Path,
    *,
    expected_current_event_id: str,
    target_promotion_event_id: str,
    target_run_id: str,
    target_bundle_manifest_sha256: str,
    operation_id: str = "rollback-operation-001",
) -> Path:
    signed = sign_rollback_authorization(
        {
            "schema_version": "rollback_authorization.v1",
            "operation_id": operation_id,
            "release_root_sha256": atomic_promotion._release_root_id(
                path.parent.joinpath("releases").resolve()
            ),
            "expected_current_event_id": expected_current_event_id,
            "target_promotion_event_id": target_promotion_event_id,
            "target_run_id": target_run_id,
            "target_bundle_manifest_sha256": target_bundle_manifest_sha256,
            "reason_code": "QUALITY_REGRESSION",
            "incident_id": "INC-2026-001",
            "authorized_at_utc": "2026-07-13T00:05:00+00:00",
            "expires_at_utc": "2026-07-13T00:20:00+00:00",
        },
        private_key_path=Path(
            os.environ["PFC_TEST_ROLLBACK_AUTHORIZATION_SIGNING_PRIVATE_KEY_PATH"]
        ),
    )
    path.write_text(json.dumps(signed), encoding="utf-8")
    return path


def _file_tree(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(autouse=True)
def _promotion_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    private_key = Ed25519PrivateKey.generate()
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
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
    monkeypatch.setenv("PFC_TEST_PROMOTION_SIGNING_PRIVATE_KEY_PATH", str(private_path))
    monkeypatch.setenv("PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH", str(public_path))
    register_key = Ed25519PrivateKey.generate()
    register_public_path = tmp_path / "register-public.pem"
    register_public_path.write_bytes(
        register_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv(
        "PFC_RELEASE_REQUEST_TRUSTED_PUBLIC_KEY_PATH", str(register_public_path)
    )
    event_key = Ed25519PrivateKey.generate()
    event_private_path = tmp_path / "event-private.pem"
    event_public_path = tmp_path / "event-public.pem"
    event_private_path.write_bytes(
        event_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    event_public_path.write_bytes(
        event_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv("PFC_PROMOTION_EVENT_SIGNING_PRIVATE_KEY_PATH", str(event_private_path))
    monkeypatch.setenv("PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_PATH", str(event_public_path))
    monkeypatch.setenv("PFC_PROMOTION_JOURNAL_ROOT", str(tmp_path / "promotion-journal"))
    monkeypatch.setenv(
        "PFC_PROMOTION_RELEASE_DOMAIN_ID",
        "123e4567-e89b-42d3-a456-426614174000",
    )
    rollback_key = Ed25519PrivateKey.generate()
    rollback_private = tmp_path / "rollback-private.pem"
    rollback_public = tmp_path / "rollback-public.pem"
    rollback_private.write_bytes(
        rollback_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    rollback_public.write_bytes(
        rollback_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv(
        "PFC_TEST_ROLLBACK_AUTHORIZATION_SIGNING_PRIVATE_KEY_PATH",
        str(rollback_private),
    )
    monkeypatch.setenv(
        "PFC_ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_PATH",
        str(rollback_public),
    )
    governance_key = Ed25519PrivateKey.generate()
    governance_private = tmp_path / "governance-private.pem"
    governance_public = tmp_path / "governance-public.pem"
    governance_private.write_bytes(
        governance_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    governance_public.write_bytes(
        governance_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv(
        "PFC_TEST_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH",
        str(governance_private),
    )
    monkeypatch.setenv("PFC_MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_PATH", str(governance_public))
    monkeypatch.setattr(
        atomic_promotion,
        "_promotion_now_utc",
        lambda: datetime(2026, 7, 13, 0, 5, tzinfo=timezone.utc),
    )


def _candidate(
    root: Path,
    run_id: str,
    content: str,
    *,
    metadata: dict[str, object] | None = None,
):
    staging = create_candidate_staging(root, run_id=run_id)
    (staging / "markets" / "ch").mkdir(parents=True)
    (staging / "markets" / "ch" / "pfc.csv").write_text(content, encoding="utf-8")
    (staging / "models" / "shared").mkdir(parents=True)
    (staging / "models" / "shared" / "shape.bin").write_bytes(b"deterministic-model")
    deterministic_artifacts = {
        path.relative_to(staging).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for root_name in ("markets", "models")
        for path in sorted((staging / root_name).rglob("*"))
        if path.is_file()
    }
    deterministic_sha256 = hashlib.sha256(
        json.dumps(
            deterministic_artifacts,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path = staging / "manifests" / "candidate_run_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
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
    return finalize_candidate_staging(
        root,
        run_id=run_id,
        staging_path=staging,
        metadata=metadata,
    )


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


def _receipt(
    path: Path,
    *,
    bundle,
    approved: bool = True,
    gate_ids: list[str] | None = None,
    evaluated_at: str = "2026-07-13T00:04:00+00:00",
    release_request_id: str | None = None,
    expected_current_event_id: str | None | object = _EXPECTED_FROM_HISTORY,
    operation_id: str | None = None,
    operation_created_at_utc: str = "2026-07-13T00:05:00+00:00",
    authorization_mode: str = "REGISTERED_PROMOTION",
) -> Path:
    root = bundle.path.parents[1]
    atomic_promotion._release_root_id(root)
    history = atomic_promotion._load_immutable_journal_history(root.resolve())
    current_event_id = str(history[-1]["event_id"]) if history else None
    expected_event_id = (
        current_event_id
        if expected_current_event_id is _EXPECTED_FROM_HISTORY
        else expected_current_event_id
    )
    request_id = release_request_id or f"test-request-{bundle.run_id}"
    payload = {
        "schema_version": "promotion_receipt.v2",
        "promotion_policy_version": PROMOTION_POLICY_VERSION,
        "authorization_mode": authorization_mode,
        "run_id": bundle.run_id,
        "approved": approved,
        "valuation_timestamp": "2026-07-13T00:00:00+00:00",
        "promotion_evaluated_at": evaluated_at,
        "candidate_bundle_manifest_sha256": bundle.manifest_sha256,
        "release_request_id": request_id,
        "release_operation_id": operation_id or request_id,
        "expected_current_event_id": expected_event_id,
        "operation_created_at_utc": operation_created_at_utc,
        "release_root_sha256": atomic_promotion._release_root_id(root.resolve()),
        "workflow_domain_sha256": "a" * 64,
        "release_request_document_sha256": "b" * 64,
        "candidate_run_manifest_sha256": hashlib.sha256(
            (bundle.path / "manifests" / "candidate_run_manifest.json").read_bytes()
        ).hexdigest(),
        "decision_summary": {
            "status": "PROMOTION_EVIDENCE_PASS" if approved else "BLOCKED",
            "approved": approved,
            "blocking_count": 0 if approved else 1,
        },
        "governance_gates": [
            {"gate_id": gate_id, "status": "PASS"}
            for gate_id in (
                gate_ids if gate_ids is not None else sorted(REQUIRED_PROMOTION_GATE_IDS)
            )
        ],
    }
    payload = sign_promotion_receipt(
        payload,
        private_key_path=os.environ["PFC_TEST_PROMOTION_SIGNING_PRIVATE_KEY_PATH"],
    )
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return path


def test_promote_candidate_atomically_replaces_current_pointer(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "timestamp;price\n1;80\n")
    receipt = _receipt(tmp_path / "receipt.json", bundle=bundle)

    pointer = promote_candidate(root, run_id="run-001", promotion_receipt_path=receipt)

    persisted = json.loads((root / "current.json").read_text(encoding="utf-8"))
    assert pointer == persisted
    assert persisted["run_id"] == "run-001"
    assert persisted["candidate_bundle_manifest_sha256"] == bundle.manifest_sha256
    assert not (root / ".promotion.lock").exists()


def test_promotion_conflict_mutates_no_release_or_journal_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    first = _candidate(root, "run-001", "first")
    first_receipt = _receipt(tmp_path / "first.json", bundle=first)
    first_contract = _contract_from_receipt(first_receipt)
    first_pointer = promote_candidate(
        root,
        run_id=first.run_id,
        promotion_receipt_path=first_receipt,
        require_assembled_evidence=True,
        **first_contract,
    )
    second = _candidate(root, "run-002", "second")
    second_receipt = _receipt(tmp_path / "second.json", bundle=second)
    release_before = _file_tree(root)
    journal_before = _file_tree(tmp_path / "promotion-journal")

    with pytest.raises(PromotionConflictError, match="compare-and-swap conflict"):
        promote_candidate(
            root,
            run_id=second.run_id,
            promotion_receipt_path=second_receipt,
            expected_current_event_id=None,
            operation_id=f"test-request-{second.run_id}",
            release_request_id=f"test-request-{second.run_id}",
            operation_created_at_utc="2026-07-13T00:05:00+00:00",
            require_assembled_evidence=True,
        )

    assert _file_tree(root) == release_before
    assert _file_tree(tmp_path / "promotion-journal") == journal_before
    assert resolve_current_candidate(root).event["event_id"] == first_pointer["event_id"]


def test_promotion_receipt_must_bind_release_request_id(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt = _receipt(
        tmp_path / "receipt.json",
        bundle=bundle,
        release_request_id="request-from-receipt",
    )

    with pytest.raises(PromotionError, match="receipt operation binding mismatch"):
        promote_candidate(
            root,
            run_id=bundle.run_id,
            promotion_receipt_path=receipt,
            expected_current_event_id=None,
            operation_id="different-request",
            release_request_id="different-request",
            operation_created_at_utc="2026-07-13T00:05:00+00:00",
            require_assembled_evidence=True,
        )

    assert not (root / "promotion_receipts").exists()
    assert atomic_promotion._load_immutable_journal_history(root.resolve()) == []
    assert not (root / "current.json").exists()


def test_promotion_rejects_future_operation_timestamp(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt = _receipt(
        tmp_path / "receipt.json",
        bundle=bundle,
        release_request_id="request-future",
    )

    with pytest.raises(PromotionError, match="operation_created_at_utc is in the future"):
        promote_candidate(
            root,
            run_id=bundle.run_id,
            promotion_receipt_path=receipt,
            expected_current_event_id=None,
            operation_id="request-future",
            release_request_id="request-future",
            operation_created_at_utc="2026-07-13T00:07:00+00:00",
            require_assembled_evidence=True,
        )

    assert not (root / "current.json").exists()


def test_direct_promotion_api_rejects_colocated_receipt_private_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt = _receipt(tmp_path / "receipt.json", bundle=bundle)
    monkeypatch.setenv(
        "PFC_PROMOTION_SIGNING_PRIVATE_KEY_PATH",
        os.environ["PFC_TEST_PROMOTION_SIGNING_PRIVATE_KEY_PATH"],
    )

    with pytest.raises(PromotionError, match="forbidden private keys"):
        promote_candidate(
            root,
            run_id=bundle.run_id,
            promotion_receipt_path=receipt,
            require_assembled_evidence=True,
            **_contract_from_receipt(receipt),
        )


def test_transition_rejects_reused_receipt_and_rollback_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt = _receipt(tmp_path / "receipt.json", bundle=bundle)
    monkeypatch.setenv(
        "PFC_ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_PATH",
        os.environ["PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH"],
    )

    with pytest.raises(PromotionError, match="authority keyrings overlap"):
        promote_candidate(
            root,
            run_id=bundle.run_id,
            promotion_receipt_path=receipt,
            require_assembled_evidence=True,
            **_contract_from_receipt(receipt),
        )


def test_transition_rejects_cross_authority_historical_key_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt = _receipt(tmp_path / "receipt.json", bundle=bundle)
    receipt_keyring = tmp_path / "receipt-keyring"
    receipt_keyring.mkdir()
    (receipt_keyring / "event-key.pem").write_bytes(
        Path(os.environ["PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_PATH"]).read_bytes()
    )
    monkeypatch.setenv("PFC_PROMOTION_TRUSTED_PUBLIC_KEY_DIR", str(receipt_keyring))

    with pytest.raises(PromotionError, match="authority keyrings overlap"):
        promote_candidate(
            root,
            run_id=bundle.run_id,
            promotion_receipt_path=receipt,
            require_assembled_evidence=True,
            **_contract_from_receipt(receipt),
        )


def test_exact_retry_repairs_missing_projections_without_new_event(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt = _receipt(tmp_path / "receipt.json", bundle=bundle)
    contract = _contract_from_receipt(receipt)
    first = promote_candidate(
        root,
        run_id=bundle.run_id,
        promotion_receipt_path=receipt,
        require_assembled_evidence=True,
        **contract,
    )
    mutable_head = next((tmp_path / "promotion-journal" / "heads").glob("*.json"))
    mutable_head.unlink()
    (root / "current.json").unlink()

    authoritative = resolve_current_candidate(root)
    assert authoritative.event["event_id"] == first["event_id"]

    retried = promote_candidate(
        root,
        run_id=bundle.run_id,
        promotion_receipt_path=receipt,
        require_assembled_evidence=True,
        **contract,
    )

    assert retried["event_id"] == first["event_id"]
    assert len(atomic_promotion._load_immutable_journal_history(root.resolve())) == 1
    assert mutable_head.is_file()
    assert json.loads((root / "current.json").read_text(encoding="utf-8")) == retried


def test_exact_promotion_retry_repair_failure_preserves_committed_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt = _receipt(tmp_path / "receipt.json", bundle=bundle)
    contract = _contract_from_receipt(receipt)
    first = promote_candidate(
        root,
        run_id=bundle.run_id,
        promotion_receipt_path=receipt,
        require_assembled_evidence=True,
        **contract,
    )
    monkeypatch.setattr(
        atomic_promotion,
        "_repair_journal_projections",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("projection unavailable")),
    )

    with pytest.raises(PromotionProjectionRepairRequired) as exc:
        promote_candidate(
            root,
            run_id=bundle.run_id,
            promotion_receipt_path=receipt,
            require_assembled_evidence=True,
            **contract,
        )

    assert exc.value.event_id == first["event_id"]
    assert exc.value.commit_status == "COMMITTED"


def test_exact_retry_rejects_a_different_signed_receipt(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt = _receipt(tmp_path / "receipt.json", bundle=bundle)
    contract = _contract_from_receipt(receipt)
    promote_candidate(
        root,
        run_id=bundle.run_id,
        promotion_receipt_path=receipt,
        require_assembled_evidence=True,
        **contract,
    )
    substituted = _receipt(
        tmp_path / "substituted.json",
        bundle=bundle,
        evaluated_at="2026-07-13T00:03:00+00:00",
        release_request_id=str(contract["release_request_id"]),
        operation_id=str(contract["operation_id"]),
        expected_current_event_id=None,
        operation_created_at_utc=str(contract["operation_created_at_utc"]),
    )

    with pytest.raises(PromotionConflictError, match="different receipt"):
        promote_candidate(
            root,
            run_id=bundle.run_id,
            promotion_receipt_path=substituted,
            require_assembled_evidence=True,
            **contract,
        )


def test_signed_receipt_cannot_be_rebased_to_a_new_head(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    first = _candidate(root, "run-001", "first")
    second = _candidate(root, "run-002", "second")
    stale_receipt = _receipt(tmp_path / "first.json", bundle=first)
    first_pointer = promote_candidate(
        root,
        run_id=first.run_id,
        promotion_receipt_path=stale_receipt,
    )
    second_receipt = _receipt(tmp_path / "second.json", bundle=second)
    second_pointer = promote_candidate(
        root,
        run_id=second.run_id,
        promotion_receipt_path=second_receipt,
    )

    with pytest.raises(PromotionConflictError, match="operation_id is already committed"):
        promote_candidate(
            root,
            run_id=first.run_id,
            promotion_receipt_path=stale_receipt,
            expected_current_event_id=str(second_pointer["event_id"]),
            operation_id=str(
                json.loads(stale_receipt.read_text(encoding="utf-8"))[
                    "release_request_id"
                ]
            ),
            release_request_id=str(
                json.loads(stale_receipt.read_text(encoding="utf-8"))[
                    "release_request_id"
                ]
            ),
            operation_created_at_utc="2026-07-13T00:05:00+00:00",
            require_assembled_evidence=True,
        )

    assert resolve_current_candidate(root).event["event_id"] == second_pointer["event_id"]
    assert first_pointer["event_id"] != second_pointer["event_id"]


def test_original_operation_cannot_retry_across_a_b_a_history(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    first = _candidate(root, "run-001", "first")
    second = _candidate(root, "run-002", "second")
    first_receipt = _receipt(tmp_path / "first.json", bundle=first)
    operation_a = _contract_from_receipt(first_receipt)
    pointer_a = promote_candidate(
        root,
        run_id=first.run_id,
        promotion_receipt_path=first_receipt,
        require_assembled_evidence=True,
        **operation_a,
    )
    second_receipt = _receipt(tmp_path / "second.json", bundle=second)
    operation_b = _contract_from_receipt(second_receipt)
    pointer_b = promote_candidate(
        root,
        run_id=second.run_id,
        promotion_receipt_path=second_receipt,
        require_assembled_evidence=True,
        **operation_b,
    )
    first_receipt_rebased = _receipt(
        tmp_path / "first-rebased.json",
        bundle=first,
        release_request_id="test-request-run-001-rebased",
    )
    operation_a2 = _contract_from_receipt(first_receipt_rebased)
    pointer_a2 = promote_candidate(
        root,
        run_id=first.run_id,
        promotion_receipt_path=first_receipt_rebased,
        require_assembled_evidence=True,
        **operation_a2,
    )

    with pytest.raises(PromotionConflictError, match="compare-and-swap conflict"):
        promote_candidate(
            root,
            run_id=first.run_id,
            promotion_receipt_path=first_receipt,
            require_assembled_evidence=True,
            **operation_a,
        )

    assert pointer_a["event_id"] != pointer_a2["event_id"]
    assert pointer_b["event_id"] == operation_a2["expected_current_event_id"]
    assert resolve_current_candidate(root).event["event_id"] == pointer_a2["event_id"]
    assert len(atomic_promotion._load_immutable_journal_history(root.resolve())) == 3


@pytest.mark.parametrize(
    "boundary",
    [
        "_archive_promotion_receipt",
        "_write_promotion_event",
        "_commit_immutable_journal_head",
        "_write_mutable_head_projection",
        "_write_current_pointer_projection",
    ],
)
def test_promotion_retry_recovers_each_crash_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt = _receipt(tmp_path / "receipt.json", bundle=bundle)
    contract = _contract_from_receipt(receipt)
    original = getattr(atomic_promotion, boundary)

    def fail_after_boundary(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError(f"crash after {boundary}")

    with monkeypatch.context() as crash:
        crash.setattr(atomic_promotion, boundary, fail_after_boundary)
        expected_error = (
            PromotionProjectionRepairRequired
            if boundary
            in {
                "_commit_immutable_journal_head",
                "_write_mutable_head_projection",
                "_write_current_pointer_projection",
            }
            else RuntimeError
        )
        with pytest.raises(expected_error):
            promote_candidate(
                root,
                run_id=bundle.run_id,
                promotion_receipt_path=receipt,
                require_assembled_evidence=True,
                **contract,
            )

    recovered = promote_candidate(
        root,
        run_id=bundle.run_id,
        promotion_receipt_path=receipt,
        require_assembled_evidence=True,
        **contract,
    )

    history = atomic_promotion._load_immutable_journal_history(root.resolve())
    assert len(history) == 1
    assert recovered["event_id"] == history[0]["event_id"]
    assert resolve_current_candidate(root).event["event_id"] == recovered["event_id"]


def test_exact_retry_repairs_stale_valid_projections(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    first = _candidate(root, "run-001", "first")
    first_receipt = _receipt(tmp_path / "first.json", bundle=first)
    first_contract = _contract_from_receipt(first_receipt)
    first_pointer = promote_candidate(
        root,
        run_id=first.run_id,
        promotion_receipt_path=first_receipt,
        require_assembled_evidence=True,
        **first_contract,
    )
    mutable_head = next((tmp_path / "promotion-journal" / "heads").glob("*.json"))
    first_head = mutable_head.read_bytes()
    second = _candidate(root, "run-002", "second")
    second_receipt = _receipt(tmp_path / "second.json", bundle=second)
    second_contract = _contract_from_receipt(second_receipt)
    second_pointer = promote_candidate(
        root,
        run_id=second.run_id,
        promotion_receipt_path=second_receipt,
        require_assembled_evidence=True,
        **second_contract,
    )
    (root / "current.json").write_text(json.dumps(first_pointer), encoding="utf-8")
    mutable_head.write_bytes(first_head)

    retried = promote_candidate(
        root,
        run_id=second.run_id,
        promotion_receipt_path=second_receipt,
        require_assembled_evidence=True,
        **second_contract,
    )

    assert retried == second_pointer
    assert json.loads((root / "current.json").read_text(encoding="utf-8")) == second_pointer
    latest_head = atomic_promotion._load_immutable_journal_history(root.resolve())[-1]
    assert json.loads(mutable_head.read_text(encoding="utf-8")) == latest_head


def test_generic_finalizer_cannot_claim_assembled_evidence_contract(tmp_path: Path) -> None:
    root = tmp_path / "releases"

    with pytest.raises(PromotionError, match="evidence_contract metadata is reserved"):
        _candidate(
            root,
            "run-001",
            "generic",
            metadata={"evidence_contract": "assembled_lt_candidate.v1"},
        )

    assert (root / "candidates" / ".run-001.staging").is_dir()
    assert not (root / "candidates" / "run-001").exists()


def test_promote_candidate_requires_assembled_evidence_by_default(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "generic")
    receipt = _receipt(tmp_path / "receipt.json", bundle=bundle)

    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    request = {
        "run_id": "run-001",
        "request_id": receipt_payload["release_request_id"],
        "created_at_utc": receipt_payload["operation_created_at_utc"],
        "release_root_sha256": receipt_payload["release_root_sha256"],
        "workflow_domain_sha256": receipt_payload["workflow_domain_sha256"],
        "registration_contract": "assembled_candidate_seal.v1",
        "candidate_bundle_manifest_sha256": receipt_payload[
            "candidate_bundle_manifest_sha256"
        ],
        "expected_current": {"kind": "NONE"},
    }
    with patch.object(
        atomic_promotion,
        "load_registered_release_request_evidence",
        return_value=(request, receipt_payload["release_request_document_sha256"]),
    ), pytest.raises(PromotionError, match="assembled candidate replay failed"):
        atomic_promotion.promote_candidate(
            root,
            run_id="run-001",
            promotion_receipt_path=receipt,
            registered_release_request_path=tmp_path / "release_request.json",
            workflow_root=tmp_path,
            **_contract_from_receipt(receipt),
        )

    assert not (root / "current.json").exists()


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("registration_contract", "generic_candidate.v1"),
        ("candidate_bundle_manifest_sha256", "f" * 64),
    ],
)
def test_promotion_rejects_REGISTER_candidate_binding_mismatch(
    tmp_path: Path,
    field: str,
    invalid_value: str,
) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt = _receipt(tmp_path / "receipt.json", bundle=bundle)
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    request = {
        "run_id": bundle.run_id,
        "request_id": receipt_payload["release_request_id"],
        "created_at_utc": receipt_payload["operation_created_at_utc"],
        "release_root_sha256": receipt_payload["release_root_sha256"],
        "workflow_domain_sha256": receipt_payload["workflow_domain_sha256"],
        "registration_contract": "assembled_candidate_seal.v1",
        "candidate_bundle_manifest_sha256": bundle.manifest_sha256,
        "expected_current": {"kind": "NONE"},
    }
    request[field] = invalid_value

    with patch.object(
        atomic_promotion,
        "load_registered_release_request_evidence",
        return_value=(request, receipt_payload["release_request_document_sha256"]),
    ), pytest.raises(PromotionError, match="request candidate binding mismatch"):
        _promote_candidate(
            root,
            run_id=bundle.run_id,
            promotion_receipt_path=receipt,
            registered_release_request_path=tmp_path / "release_request.json",
            workflow_root=tmp_path,
            require_assembled_evidence=True,
            **_contract_from_receipt(receipt),
        )

    assert not (root / "current.json").exists()


@pytest.mark.parametrize("mutated_read", [2, 3])
def test_promotion_rechecks_REGISTER_binding_after_each_bundle_replay(
    tmp_path: Path,
    mutated_read: int,
) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt = _receipt(tmp_path / "receipt.json", bundle=bundle)
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    request = {
        "run_id": bundle.run_id,
        "request_id": receipt_payload["release_request_id"],
        "created_at_utc": receipt_payload["operation_created_at_utc"],
        "release_root_sha256": receipt_payload["release_root_sha256"],
        "workflow_domain_sha256": receipt_payload["workflow_domain_sha256"],
        "registration_contract": "assembled_candidate_seal.v1",
        "candidate_bundle_manifest_sha256": bundle.manifest_sha256,
        "expected_current": {"kind": "NONE"},
    }
    changed = atomic_promotion.CandidateBundle(
        run_id=bundle.run_id,
        path=bundle.path,
        manifest_path=bundle.manifest_path,
        manifest_sha256="f" * 64,
    )
    bundle_reads = [bundle] * (mutated_read - 1) + [changed]

    with patch.object(
        atomic_promotion,
        "load_registered_release_request_evidence",
        return_value=(request, receipt_payload["release_request_document_sha256"]),
    ), patch.object(
        atomic_promotion,
        "verify_candidate_bundle",
        side_effect=bundle_reads,
    ), patch.object(
        atomic_promotion,
        "verify_assembled_candidate_evidence",
        return_value={"status": "TEST_UNIT_SEAL"},
    ), pytest.raises(PromotionError, match="request candidate binding mismatch"):
        _promote_candidate(
            root,
            run_id=bundle.run_id,
            promotion_receipt_path=receipt,
            registered_release_request_path=tmp_path / "release_request.json",
            workflow_root=tmp_path,
            require_assembled_evidence=True,
            **_contract_from_receipt(receipt),
        )

    assert not (root / "current.json").exists()


def test_direct_promotion_rejects_unsealed_checkout_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "generic")
    receipt = _receipt(tmp_path / "receipt.json", bundle=bundle)
    monkeypatch.setattr(release_cli_contract, "SOURCE_REVISION", None)

    with pytest.raises(PromotionError, match="installed runtime with a sealed identity"):
        atomic_promotion.promote_candidate(
            root,
            run_id=bundle.run_id,
            promotion_receipt_path=receipt,
            **_contract_from_receipt(receipt),
        )

    assert not (root / "current.json").exists()


def test_diagnostic_receipt_cannot_authorize_atomic_promotion(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "generic")
    receipt = _receipt(
        tmp_path / "receipt.json",
        bundle=bundle,
        authorization_mode="DIAGNOSTIC_ONLY",
    )

    with pytest.raises(PromotionError, match="registered promotion authorization"):
        promote_candidate(
            root,
            run_id=bundle.run_id,
            promotion_receipt_path=receipt,
        )

    assert not (root / "current.json").exists()


def test_failed_receipt_leaves_current_pointer_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    first = _candidate(root, "run-001", "first")
    promote_candidate(
        root,
        run_id="run-001",
        promotion_receipt_path=_receipt(tmp_path / "first.json", bundle=first),
    )
    before = (root / "current.json").read_bytes()
    second = _candidate(root, "run-002", "second")

    with pytest.raises(PromotionError, match="not an approved"):
        promote_candidate(
            root,
            run_id="run-002",
            promotion_receipt_path=_receipt(
                tmp_path / "second.json",
                bundle=second,
                approved=False,
            ),
        )

    assert (root / "current.json").read_bytes() == before


def test_bundle_tamper_blocks_promotion(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "original")
    (bundle.path / "markets" / "ch" / "pfc.csv").write_text("tampered", encoding="utf-8")

    with pytest.raises(PromotionError, match="deterministic artifacts are invalid"):
        promote_candidate(
            root,
            run_id="run-001",
            promotion_receipt_path=_receipt(tmp_path / "receipt.json", bundle=bundle),
        )

    assert not (root / "current.json").exists()


def test_existing_lock_blocks_concurrent_promotion(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt = _receipt(tmp_path / "receipt.json", bundle=bundle)
    (root / ".promotion.lock").write_text("held", encoding="utf-8")

    with pytest.raises(PromotionBusyError, match="lock is busy"):
        promote_candidate(root, run_id="run-001", promotion_receipt_path=receipt)

    assert not (root / "current.json").exists()


def test_immutable_publication_exposes_only_complete_final_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "immutable.json"
    payload = b'{"complete":true}\n'

    def inspect_before_publish(source: Path, target: Path) -> None:
        assert not target.exists()
        assert source.read_bytes() == payload
        raise RuntimeError("stop before atomic link")

    monkeypatch.setattr(atomic_promotion.os, "link", inspect_before_publish)
    with pytest.raises(RuntimeError, match="stop before atomic link"):
        atomic_promotion._write_bytes_exclusive_fsync(destination, payload)

    assert not destination.exists()
    assert not list(tmp_path.glob(".pfc-*.tmp"))


def test_durable_directory_creation_syncs_new_entries_and_parents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "journal" / "h" / "release"
    synced: list[Path] = []
    monkeypatch.setattr(atomic_promotion, "_fsync_directory", synced.append)

    atomic_promotion._mkdir_durable(target)

    assert target.is_dir()
    for directory in (tmp_path / "journal", tmp_path / "journal" / "h", target):
        assert directory in synced
        assert directory.parent in synced


def test_atomic_json_replacement_failure_cleans_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "projection.json"
    monkeypatch.setattr(
        atomic_promotion,
        "_replace_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        atomic_promotion._atomic_write_json(destination, {"state": "pending"})

    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_candidate_finalization_flushes_tree_before_rename_and_parent_after(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    synced: list[Path] = []
    original_replace = atomic_promotion._replace_directory

    monkeypatch.setattr(
        atomic_promotion,
        "_fsync_tree",
        lambda root: order.append(f"tree:{root.name}"),
    )

    def replace_after_flush(source: Path, destination: Path) -> None:
        assert order == [f"tree:{source.name}"]
        order.append(f"replace:{destination.name}")
        original_replace(source, destination)

    monkeypatch.setattr(atomic_promotion, "_replace_directory", replace_after_flush)
    monkeypatch.setattr(atomic_promotion, "_fsync_directory", synced.append)

    bundle = _candidate(tmp_path / "releases", "run-001", "candidate")

    assert order == ["tree:.run-001.staging", "replace:run-001"]
    assert bundle.path.parent in synced


def test_prepositioned_hardlink_receipt_archive_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "release"
    root.mkdir()
    receipt_id = "a" * 64
    payload = b'{"receipt":"same-bytes"}\n'
    external = tmp_path / "external.json"
    external.write_bytes(payload)
    archive = root / "promotion_receipts"
    archive.mkdir()
    os.link(external, archive / f"{receipt_id}.json")

    with pytest.raises(PromotionError, match="forbidden hard links"):
        atomic_promotion._archive_promotion_receipt(
            root.resolve(),
            receipt={"receipt_id": receipt_id},
            receipt_bytes=payload,
        )


@pytest.mark.parametrize(
    "relative",
    [Path("markets/ch/pfc.csv"), Path("candidate_bundle_manifest.json")],
    ids=["payload", "root-manifest"],
)
def test_candidate_bundle_rejects_external_hardlinked_payload(
    tmp_path: Path,
    relative: Path,
) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    payload = bundle.path / relative
    original = payload.read_bytes()
    external = tmp_path / "external-pfc.csv"
    external.write_bytes(original)
    payload.unlink()
    os.link(external, payload)

    with pytest.raises(PromotionError, match="hardlinks are forbidden"):
        verify_candidate_bundle(root, run_id=bundle.run_id)


def test_two_process_promotions_on_same_expected_head_commit_exactly_one(
    tmp_path: Path,
) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt_a = _receipt(
        tmp_path / "receipt-a.json",
        bundle=bundle,
        release_request_id="request-A",
    )
    receipt_b = _receipt(
        tmp_path / "receipt-b.json",
        bundle=bundle,
        release_request_id="request-B",
    )
    code = """
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from pfc_shaping.pipeline import atomic_promotion as ap
from pfc_shaping.pipeline import candidate_evidence as ce
from pfc_shaping.pipeline import governed_release_cli_contract as grc
ce.SOURCE_REVISION = "1111111111111111111111111111111111111111"
grc.SOURCE_REVISION = "1111111111111111111111111111111111111111111111111111111111111111"
grc._installed_runtime_source_revision = lambda: grc.SOURCE_REVISION
ap._promotion_now_utc = lambda: datetime(2026, 7, 13, 0, 5, tzinfo=timezone.utc)
ap.verify_assembled_candidate_evidence = lambda *_args, **_kwargs: {"status": "TEST_UNIT_SEAL"}
receipt = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
ap.load_registered_release_request_evidence = lambda *_args, **_kwargs: ({
    "run_id": "run-001",
    "request_id": sys.argv[3],
    "created_at_utc": "2026-07-13T00:05:00+00:00",
    "release_root_sha256": receipt["release_root_sha256"],
    "workflow_domain_sha256": receipt["workflow_domain_sha256"],
    "registration_contract": "assembled_candidate_seal.v1",
    "candidate_bundle_manifest_sha256": receipt["candidate_bundle_manifest_sha256"],
    "expected_current": {"kind": "NONE"},
}, receipt["release_request_document_sha256"])
try:
    ap.promote_candidate(
        sys.argv[1],
        run_id="run-001",
        promotion_receipt_path=sys.argv[2],
        expected_current_event_id=None,
        operation_id=sys.argv[3],
        release_request_id=sys.argv[3],
        operation_created_at_utc="2026-07-13T00:05:00+00:00",
        registered_release_request_path=Path(sys.argv[2]).with_name("release_request.json"),
        workflow_root=Path(sys.argv[2]).parent,
        require_assembled_evidence=True,
    )
except ap.PromotionConflictError:
    raise SystemExit(40)
except ap.PromotionBusyError:
    raise SystemExit(41)
"""
    env = os.environ.copy()
    env["PFC_PROMOTION_LOCK_WAIT_SECONDS"] = "10"
    cwd = Path(__file__).resolve().parents[1]
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(root), str(receipt), request_id],
            cwd=cwd,
            env=env,
        )
        for receipt, request_id in ((receipt_a, "request-A"), (receipt_b, "request-B"))
    ]
    return_codes = {process.wait(timeout=30) for process in processes}

    assert return_codes == {0, 40}
    assert len(atomic_promotion._load_immutable_journal_history(root.resolve())) == 1


def test_abandoned_well_formed_lock_requires_operator_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    root.mkdir()
    lock = root / ".promotion.lock"
    lock.write_text(
        "schema=promotion_lock.v2\n"
        f"host={atomic_promotion._lock_host_id()}\n"
        "pid=999999\nstart=unknown\n"
        f"token={'a' * 32}\n",
        encoding="ascii",
    )
    monkeypatch.setattr(atomic_promotion, "_pid_is_alive", lambda pid: False)

    with pytest.raises(PromotionBusyError, match="lock is busy"):
        with atomic_promotion._promotion_lock(root):
            raise AssertionError("stale lock must remain authoritative")

    assert lock.is_file()


def test_foreign_host_lock_is_never_recovered_automatically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    root.mkdir()
    lock = root / ".promotion.lock"
    lock.write_text(
        "schema=promotion_lock.v2\n"
        "host=other-host\n"
        "pid=999999\nstart=unknown\n"
        f"token={'c' * 32}\n",
        encoding="ascii",
    )
    monkeypatch.setattr(atomic_promotion, "_pid_is_alive", lambda pid: False)

    with pytest.raises(PromotionBusyError, match="lock is busy"):
        with atomic_promotion._promotion_lock(root):
            raise AssertionError("foreign lock must remain authoritative")

    assert lock.is_file()


@pytest.mark.parametrize("value", ["NaN", "inf", "-inf"])
def test_lock_wait_rejects_nonfinite_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    root = tmp_path / "releases"
    root.mkdir()
    monkeypatch.setenv("PFC_PROMOTION_LOCK_WAIT_SECONDS", value)

    with pytest.raises(PromotionError, match="out of range"):
        with atomic_promotion._promotion_lock(root):
            raise AssertionError("invalid wait must fail before lock acquisition")

    assert not (root / ".promotion.lock").exists()


def test_lock_with_reused_live_pid_requires_operator_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    root.mkdir()
    lock = root / ".promotion.lock"
    lock.write_text(
        "schema=promotion_lock.v2\n"
        f"host={atomic_promotion._lock_host_id()}\n"
        f"pid=424242\nstart={'a' * 16}\n"
        f"token={'b' * 32}\n",
        encoding="ascii",
    )
    monkeypatch.setattr(atomic_promotion, "_pid_is_alive", lambda pid: True)
    monkeypatch.setattr(atomic_promotion, "process_start_token", lambda pid: "b" * 16)

    with pytest.raises(PromotionBusyError, match="lock is busy"):
        with atomic_promotion._promotion_lock(root):
            raise AssertionError("stale lock must remain authoritative")

    assert lock.is_file()


def test_candidate_temporary_recovery_is_pid_aware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / ".run-001.staging"
    staging.mkdir()
    orphan = staging / f".pfc-424242-{('a' * 16)}-{('b' * 32)}.tmp"
    orphan.write_text("partial", encoding="utf-8")
    monkeypatch.setattr(atomic_promotion, "_pid_is_alive", lambda pid: False)

    atomic_promotion._recover_abandoned_candidate_temporaries(staging)

    assert not orphan.exists()

    active = staging / f".pfc-424242-{('c' * 16)}-{('d' * 32)}.tmp"
    active.write_text("partial", encoding="utf-8")
    monkeypatch.setattr(atomic_promotion, "_pid_is_alive", lambda pid: True)
    monkeypatch.setattr(atomic_promotion, "process_start_token", lambda pid: "c" * 16)
    with pytest.raises(PromotionError, match="write is still active"):
        atomic_promotion._recover_abandoned_candidate_temporaries(staging)

    reused = staging / f".pfc-424242-{('e' * 16)}-{('f' * 32)}.tmp"
    reused.write_text("partial", encoding="utf-8")
    monkeypatch.setattr(atomic_promotion, "process_start_token", lambda pid: "1" * 16)
    active.unlink()
    atomic_promotion._recover_abandoned_candidate_temporaries(staging)
    assert not reused.exists()


def test_candidate_temporary_recovery_rejects_ambiguous_files(tmp_path: Path) -> None:
    staging = tmp_path / ".run-001.staging"
    staging.mkdir()
    (staging / "worker.tmp").write_text("partial", encoding="utf-8")

    with pytest.raises(PromotionError, match="unrecognized candidate temporary"):
        atomic_promotion._recover_abandoned_candidate_temporaries(staging)


def test_rollback_repoints_to_verified_prior_candidate(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    first = _candidate(root, "run-001", "first")
    second = _candidate(root, "run-002", "second")
    first_pointer = promote_candidate(
        root,
        run_id="run-001",
        promotion_receipt_path=_receipt(tmp_path / "first.json", bundle=first),
    )
    second_pointer = promote_candidate(
        root,
        run_id="run-002",
        promotion_receipt_path=_receipt(tmp_path / "second.json", bundle=second),
    )

    authorization = _rollback_authorization(
        tmp_path / "rollback.json",
        expected_current_event_id=str(second_pointer["event_id"]),
        target_promotion_event_id=str(first_pointer["event_id"]),
        target_run_id=first.run_id,
        target_bundle_manifest_sha256=first.manifest_sha256,
    )
    pointer = rollback_to_candidate(
        root,
        rollback_authorization_path=authorization,
    )

    assert pointer["run_id"] == "run-001"
    assert pointer["rollback"] is True
    assert pointer["previous_run_id"] == "run-002"


def test_rollback_exact_retry_after_key_rotation_repairs_projections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    first = _candidate(root, "run-001", "first")
    second = _candidate(root, "run-002", "second")
    first_pointer = promote_candidate(
        root,
        run_id=first.run_id,
        promotion_receipt_path=_receipt(tmp_path / "first.json", bundle=first),
    )
    second_pointer = promote_candidate(
        root,
        run_id=second.run_id,
        promotion_receipt_path=_receipt(tmp_path / "second.json", bundle=second),
    )
    authorization = _rollback_authorization(
        tmp_path / "rollback.json",
        expected_current_event_id=str(second_pointer["event_id"]),
        target_promotion_event_id=str(first_pointer["event_id"]),
        target_run_id=first.run_id,
        target_bundle_manifest_sha256=first.manifest_sha256,
    )
    expected = rollback_to_candidate(root, rollback_authorization_path=authorization)
    history_before = atomic_promotion._load_immutable_journal_history(root.resolve())
    (root / "current.json").unlink()
    journal_root = Path(os.environ["PFC_PROMOTION_JOURNAL_ROOT"]).resolve()
    atomic_promotion._journal_head_path(
        journal_root,
        root.resolve(),
        create=False,
    ).unlink()
    old_public = Path(
        os.environ["PFC_ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_PATH"]
    )
    keyring = tmp_path / "rollback-keyring"
    keyring.mkdir()
    (keyring / "historical.pem").write_bytes(old_public.read_bytes())
    new_key = Ed25519PrivateKey.generate()
    new_public = tmp_path / "rollback-new-public.pem"
    new_public.write_bytes(
        new_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv(
        "PFC_ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_PATH",
        str(new_public),
    )
    monkeypatch.setenv(
        "PFC_ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_DIR",
        str(keyring),
    )

    recovered = rollback_to_candidate(root, rollback_authorization_path=authorization)

    assert recovered == expected
    assert json.loads((root / "current.json").read_text(encoding="utf-8")) == expected
    assert atomic_promotion._load_immutable_journal_history(root.resolve()) == history_before


def test_exact_rollback_retry_repair_failure_preserves_committed_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    first = _candidate(root, "run-001", "first")
    second = _candidate(root, "run-002", "second")
    first_pointer = promote_candidate(
        root,
        run_id=first.run_id,
        promotion_receipt_path=_receipt(tmp_path / "first.json", bundle=first),
    )
    second_pointer = promote_candidate(
        root,
        run_id=second.run_id,
        promotion_receipt_path=_receipt(tmp_path / "second.json", bundle=second),
    )
    authorization = _rollback_authorization(
        tmp_path / "rollback.json",
        expected_current_event_id=str(second_pointer["event_id"]),
        target_promotion_event_id=str(first_pointer["event_id"]),
        target_run_id=first.run_id,
        target_bundle_manifest_sha256=first.manifest_sha256,
    )
    committed = rollback_to_candidate(root, rollback_authorization_path=authorization)
    monkeypatch.setattr(
        atomic_promotion,
        "_repair_journal_projections",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("projection unavailable")),
    )

    with pytest.raises(PromotionProjectionRepairRequired) as exc:
        rollback_to_candidate(root, rollback_authorization_path=authorization)

    assert exc.value.event_id == committed["event_id"]
    assert exc.value.commit_status == "COMMITTED"


def test_rollback_operation_id_cannot_authorize_a_different_transition(
    tmp_path: Path,
) -> None:
    root = tmp_path / "releases"
    first = _candidate(root, "run-001", "first")
    second = _candidate(root, "run-002", "second")
    first_pointer = promote_candidate(
        root,
        run_id=first.run_id,
        promotion_receipt_path=_receipt(tmp_path / "first.json", bundle=first),
    )
    second_pointer = promote_candidate(
        root,
        run_id=second.run_id,
        promotion_receipt_path=_receipt(tmp_path / "second.json", bundle=second),
    )
    operation_id = "rollback-operation-reuse"
    first_authorization = _rollback_authorization(
        tmp_path / "rollback-first.json",
        expected_current_event_id=str(second_pointer["event_id"]),
        target_promotion_event_id=str(first_pointer["event_id"]),
        target_run_id=first.run_id,
        target_bundle_manifest_sha256=first.manifest_sha256,
        operation_id=operation_id,
    )
    rollback_pointer = rollback_to_candidate(
        root,
        rollback_authorization_path=first_authorization,
    )
    reused_authorization = _rollback_authorization(
        tmp_path / "rollback-reused-operation.json",
        expected_current_event_id=str(rollback_pointer["event_id"]),
        target_promotion_event_id=str(second_pointer["event_id"]),
        target_run_id=second.run_id,
        target_bundle_manifest_sha256=second.manifest_sha256,
        operation_id=operation_id,
    )
    before = _file_tree(root)

    with pytest.raises(PromotionError, match="operation_id is already committed"):
        rollback_to_candidate(
            root,
            rollback_authorization_path=reused_authorization,
        )

    assert _file_tree(root) == before


def test_rollback_stale_expected_head_is_conflict_without_mutation(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    first = _candidate(root, "run-001", "first")
    second = _candidate(root, "run-002", "second")
    first_pointer = promote_candidate(
        root,
        run_id=first.run_id,
        promotion_receipt_path=_receipt(tmp_path / "first.json", bundle=first),
    )
    promote_candidate(
        root,
        run_id=second.run_id,
        promotion_receipt_path=_receipt(tmp_path / "second.json", bundle=second),
    )
    before_pointer = (root / "current.json").read_bytes()
    before_history = atomic_promotion._load_immutable_journal_history(root.resolve())
    authorization = _rollback_authorization(
        tmp_path / "stale.json",
        expected_current_event_id=str(first_pointer["event_id"]),
        target_promotion_event_id=str(first_pointer["event_id"]),
        target_run_id=first.run_id,
        target_bundle_manifest_sha256=first.manifest_sha256,
    )

    with pytest.raises(PromotionConflictError, match="rollback compare-and-swap"):
        rollback_to_candidate(root, rollback_authorization_path=authorization)

    assert (root / "current.json").read_bytes() == before_pointer
    assert atomic_promotion._load_immutable_journal_history(root.resolve()) == before_history
    assert not (root / "rollback_authorizations").exists()


def test_rollback_rejects_expired_fresh_authorization_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    first = _candidate(root, "run-001", "first")
    second = _candidate(root, "run-002", "second")
    first_pointer = promote_candidate(
        root,
        run_id=first.run_id,
        promotion_receipt_path=_receipt(tmp_path / "first.json", bundle=first),
    )
    second_pointer = promote_candidate(
        root,
        run_id=second.run_id,
        promotion_receipt_path=_receipt(tmp_path / "second.json", bundle=second),
    )
    authorization = _rollback_authorization(
        tmp_path / "expired.json",
        expected_current_event_id=str(second_pointer["event_id"]),
        target_promotion_event_id=str(first_pointer["event_id"]),
        target_run_id=first.run_id,
        target_bundle_manifest_sha256=first.manifest_sha256,
    )
    before = (root / "current.json").read_bytes()
    monkeypatch.setattr(
        atomic_promotion,
        "_promotion_now_utc",
        lambda: datetime(2026, 7, 13, 0, 21, tzinfo=timezone.utc),
    )

    with pytest.raises(PromotionError, match="expired"):
        rollback_to_candidate(root, rollback_authorization_path=authorization)

    assert (root / "current.json").read_bytes() == before
    assert not (root / "rollback_authorizations").exists()


@pytest.mark.parametrize(
    "boundary",
    [
        "_archive_rollback_authorization",
        "_write_promotion_event",
        "_commit_immutable_journal_head",
        "_write_mutable_head_projection",
        "_write_current_pointer_projection",
    ],
)
def test_rollback_retry_recovers_each_crash_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    root = tmp_path / "releases"
    first = _candidate(root, "run-001", "first")
    second = _candidate(root, "run-002", "second")
    first_pointer = promote_candidate(
        root,
        run_id=first.run_id,
        promotion_receipt_path=_receipt(tmp_path / "first.json", bundle=first),
    )
    second_pointer = promote_candidate(
        root,
        run_id=second.run_id,
        promotion_receipt_path=_receipt(tmp_path / "second.json", bundle=second),
    )
    authorization = _rollback_authorization(
        tmp_path / "rollback.json",
        expected_current_event_id=str(second_pointer["event_id"]),
        target_promotion_event_id=str(first_pointer["event_id"]),
        target_run_id=first.run_id,
        target_bundle_manifest_sha256=first.manifest_sha256,
    )
    original = getattr(atomic_promotion, boundary)

    def fail_after_boundary(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError(f"crash after {boundary}")

    with monkeypatch.context() as crash:
        crash.setattr(atomic_promotion, boundary, fail_after_boundary)
        expected_error = (
            PromotionProjectionRepairRequired
            if boundary
            in {
                "_commit_immutable_journal_head",
                "_write_mutable_head_projection",
                "_write_current_pointer_projection",
            }
            else RuntimeError
        )
        with pytest.raises(expected_error):
            rollback_to_candidate(root, rollback_authorization_path=authorization)

    recovered = rollback_to_candidate(root, rollback_authorization_path=authorization)

    assert recovered["run_id"] == first.run_id
    assert recovered["rollback"] is True
    assert resolve_current_candidate(root).event["event_id"] == recovered["event_id"]
    assert len(atomic_promotion._load_immutable_journal_history(root.resolve())) == 3


def test_run_id_cannot_escape_release_root(tmp_path: Path) -> None:
    with pytest.raises(PromotionError, match="invalid run_id"):
        create_candidate_staging(tmp_path / "releases", run_id="../escape")


def test_release_domain_is_independent_from_mount_alias(tmp_path: Path) -> None:
    first_alias = tmp_path / "short-host" / "release"
    second_alias = tmp_path / "fqdn-host" / "release"

    assert atomic_promotion._release_root_id(
        first_alias
    ) == atomic_promotion._release_root_id(second_alias)


def test_pinned_release_domain_rejects_drift_after_projection_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    promote_candidate(
        root,
        run_id=bundle.run_id,
        promotion_receipt_path=_receipt(tmp_path / "receipt.json", bundle=bundle),
    )
    (root / "current.json").unlink()
    monkeypatch.setenv(
        "PFC_PROMOTION_RELEASE_DOMAIN_ID",
        "123e4567-e89b-42d3-a456-426614174099",
    )

    with pytest.raises(PromotionError, match="does not match pinned marker"):
        resolve_current_candidate(root)


def test_release_domain_recovers_exact_crash_leftover_hardlink(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    expected = atomic_promotion._release_root_id(root)
    marker = root / atomic_promotion.RELEASE_DOMAIN_MARKER
    temporary = root / ".pfc-999-abcdef123456.tmp"
    os.link(marker, temporary)
    assert marker.stat().st_nlink == 2

    assert atomic_promotion._release_root_id(root) == expected
    assert not temporary.exists()
    assert marker.stat().st_nlink == 1


def test_release_domain_read_only_validation_never_recovers_crash_link(
    tmp_path: Path,
) -> None:
    root = tmp_path / "releases"
    atomic_promotion._release_root_id(root)
    marker = root / atomic_promotion.RELEASE_DOMAIN_MARKER
    temporary = root / ".pfc-999-abcdef123456.tmp"
    os.link(marker, temporary)

    with pytest.raises(PromotionError, match="forbidden hard links"):
        atomic_promotion._release_root_id(root, create=False)

    assert temporary.exists()
    assert marker.stat().st_nlink == 2


def test_candidate_verification_does_not_create_missing_release_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "missing-release"

    with pytest.raises(PromotionError, match="release root is unavailable"):
        verify_candidate_bundle(root, run_id="run-001")

    assert not root.exists()


def test_external_journal_identity_error_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = tmp_path / "release"
    monkeypatch.setenv("PFC_PROMOTION_JOURNAL_ROOT", str(tmp_path / "journal"))
    monkeypatch.setattr(
        atomic_promotion,
        "paths_overlap_by_identity",
        lambda *_args: (_ for _ in ()).throw(ValueError("SMB identity unavailable")),
    )

    with pytest.raises(PromotionError, match="journal root identity"):
        atomic_promotion._promotion_journal_root(release)

    assert not release.exists()
    assert not (tmp_path / "journal").exists()


def test_external_journal_reparse_validation_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = tmp_path / "release"
    journal = tmp_path / "journal"
    monkeypatch.setenv("PFC_PROMOTION_JOURNAL_ROOT", str(journal))
    original = atomic_promotion.assert_absolute_path_has_no_links

    def reject_journal(path: str | Path) -> Path:
        if Path(path) == journal:
            raise ValueError("simulated journal junction")
        return original(path)

    monkeypatch.setattr(
        atomic_promotion,
        "assert_absolute_path_has_no_links",
        reject_journal,
    )

    with pytest.raises(PromotionError, match="journal root identity"):
        atomic_promotion._promotion_journal_root(release)

    assert not journal.exists()


def test_lock_publication_cleanup_failure_does_not_abandon_owned_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = tmp_path / ".promotion.lock"
    original_unlink = Path.unlink

    def reject_temporary(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.startswith(".pfc-lock-"):
            raise OSError("injected temporary cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", reject_temporary)

    with pytest.raises(PromotionError, match="cleanup requires repair"):
        atomic_promotion._publish_lock_exclusive(lock, b"owner")

    assert not lock.exists()


def test_immutable_directory_never_deletes_live_writer_temporary(tmp_path: Path) -> None:
    directory = tmp_path / "failures"
    directory.mkdir()
    temporary = directory / f".pfc-{os.getpid()}-abcdef123456.tmp"
    temporary.write_bytes(b"in-flight")

    with pytest.raises(PromotionBusyError, match="active concurrent writer"):
        atomic_promotion.prepare_immutable_directory(
            directory,
            label="failure manifest",
        )

    assert temporary.read_bytes() == b"in-flight"


def test_immutable_directory_never_deletes_remote_writer_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "failures"
    directory.mkdir()
    monkeypatch.setattr(
        atomic_promotion,
        "_exclusive_writer_host_hash",
        lambda: "b" * 12,
    )
    temporary = directory / f".pfc-v2-{'a' * 12}-424242-{'c' * 16}-{'d' * 12}.tmp"
    temporary.write_bytes(b"remote-in-flight")

    with pytest.raises(PromotionBusyError, match="active concurrent writer"):
        atomic_promotion.prepare_immutable_directory(
            directory,
            label="failure manifest",
        )

    assert temporary.read_bytes() == b"remote-in-flight"


def test_immutable_directory_recovers_proven_dead_local_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "failures"
    directory.mkdir()
    monkeypatch.setattr(
        atomic_promotion,
        "_exclusive_writer_host_hash",
        lambda: "b" * 12,
    )
    monkeypatch.setattr(atomic_promotion, "_pid_is_alive", lambda _pid: False)
    temporary = directory / f".pfc-v2-{'b' * 12}-424242-{'c' * 16}-{'d' * 12}.tmp"
    temporary.write_bytes(b"abandoned")

    atomic_promotion.prepare_immutable_directory(
        directory,
        label="failure manifest",
    )

    assert not temporary.exists()


def test_operator_interrupt_after_journal_commit_requires_projection_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_id = "e" * 64
    monkeypatch.setattr(
        atomic_promotion,
        "_commit_immutable_journal_head",
        lambda *_args, **_kwargs: {"event_id": event_id},
    )
    monkeypatch.setattr(
        atomic_promotion,
        "_write_mutable_head_projection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(
        atomic_promotion,
        "_load_immutable_journal_history",
        lambda *_args, **_kwargs: [{"event_id": event_id}],
    )

    with pytest.raises(PromotionProjectionRepairRequired) as exc:
        atomic_promotion._commit_and_publish_transition(
            tmp_path / "release",
            event={"event_id": event_id},
            pointer={},
            updated_at_utc="2026-07-14T12:00:00+00:00",
            trusted_public_key_path=tmp_path / "trusted.pem",
        )

    assert exc.value.event_id == event_id
    assert exc.value.commit_status == "COMMITTED"


def test_returned_journal_commit_does_not_depend_on_smb_reverification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_id = "e" * 64
    monkeypatch.setattr(
        atomic_promotion,
        "_commit_immutable_journal_head",
        lambda *_args, **_kwargs: {"event_id": event_id},
    )
    monkeypatch.setattr(
        atomic_promotion,
        "_write_mutable_head_projection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("projection unavailable")),
    )
    monkeypatch.setattr(
        atomic_promotion,
        "_load_immutable_journal_history",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("SMB unavailable")),
    )

    with pytest.raises(PromotionProjectionRepairRequired) as exc:
        atomic_promotion._commit_and_publish_transition(
            tmp_path / "release",
            event={"event_id": event_id},
            pointer={},
            updated_at_utc="2026-07-15T08:00:00+00:00",
            trusted_public_key_path=tmp_path / "trusted.pem",
        )

    assert exc.value.event_id == event_id
    assert exc.value.commit_status == "COMMITTED"


def test_failed_journal_write_with_unavailable_replay_is_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_id = "e" * 64
    monkeypatch.setattr(
        atomic_promotion,
        "_commit_immutable_journal_head",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("write result unknown")),
    )
    monkeypatch.setattr(
        atomic_promotion,
        "_load_immutable_journal_history",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("SMB unavailable")),
    )

    with pytest.raises(PromotionProjectionRepairRequired) as exc:
        atomic_promotion._commit_and_publish_transition(
            tmp_path / "release",
            event={"event_id": event_id},
            pointer={},
            updated_at_utc="2026-07-15T08:00:00+00:00",
            trusted_public_key_path=tmp_path / "trusted.pem",
        )

    assert exc.value.event_id == event_id
    assert exc.value.commit_status == "INDETERMINATE"


def test_projection_repair_status_survives_lock_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_id = "e" * 64
    monkeypatch.setattr(atomic_promotion, "_publish_lock_exclusive", lambda *_: True)
    monkeypatch.setattr(
        atomic_promotion,
        "_remove_owned_lock",
        lambda *_: (_ for _ in ()).throw(OSError("persistent sharing violation")),
    )

    with pytest.raises(PromotionProjectionRepairRequired) as exc:
        with atomic_promotion._promotion_lock(tmp_path):
            raise PromotionProjectionRepairRequired(event_id)

    assert exc.value.event_id == event_id
    assert exc.value.lock_cleanup_status == "FAIL"
    assert exc.value.lock_cleanup_error_type == "OSError"


def test_primary_conflict_exposes_lock_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(atomic_promotion, "_publish_lock_exclusive", lambda *_: True)
    monkeypatch.setattr(
        atomic_promotion,
        "_remove_owned_lock",
        lambda *_: (_ for _ in ()).throw(OSError("persistent sharing violation")),
    )

    with pytest.raises(PromotionConflictError) as exc:
        with atomic_promotion._promotion_lock(tmp_path):
            raise PromotionConflictError("CAS mismatch")

    assert exc.value.lock_cleanup_status == "FAIL"
    assert exc.value.lock_cleanup_error_type == "OSError"


@pytest.mark.parametrize("cleanup_error", [KeyboardInterrupt(), SystemExit(23)])
def test_projection_repair_survives_base_exception_during_lock_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_error: BaseException,
) -> None:
    event_id = "e" * 64
    monkeypatch.setattr(atomic_promotion, "_publish_lock_exclusive", lambda *_: True)

    def interrupted_cleanup(*_args: object) -> None:
        raise cleanup_error

    monkeypatch.setattr(atomic_promotion, "_remove_owned_lock", interrupted_cleanup)

    with pytest.raises(PromotionProjectionRepairRequired) as exc:
        with atomic_promotion._promotion_lock(tmp_path):
            raise PromotionProjectionRepairRequired(event_id)

    assert exc.value.event_id == event_id
    assert exc.value.commit_status == "COMMITTED"
    assert exc.value.lock_cleanup_status == "FAIL"
    assert exc.value.lock_cleanup_error_type == type(cleanup_error).__name__


@pytest.mark.parametrize("cleanup_error", [KeyboardInterrupt(), SystemExit(23)])
def test_committed_body_cleanup_base_exception_becomes_projection_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_error: BaseException,
) -> None:
    event_id = "e" * 64
    monkeypatch.setattr(atomic_promotion, "_publish_lock_exclusive", lambda *_: True)

    def interrupted_cleanup(*_args: object) -> None:
        raise cleanup_error

    monkeypatch.setattr(atomic_promotion, "_remove_owned_lock", interrupted_cleanup)
    monkeypatch.setattr(
        atomic_promotion,
        "_load_immutable_journal_history",
        lambda *_args, **_kwargs: [{"event_id": event_id}],
    )

    with pytest.raises(PromotionProjectionRepairRequired) as exc:
        with atomic_promotion._promotion_lock(tmp_path) as lock_state:
            lock_state.bind_journal_event(event_id)
            pass

    assert exc.value.event_id == event_id
    assert exc.value.commit_status == "COMMITTED"
    assert exc.value.lock_cleanup_status == "FAIL"
    assert exc.value.lock_cleanup_error_type == type(cleanup_error).__name__


def test_successful_body_with_unknown_lock_cleanup_is_indeterminate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_id = "e" * 64
    monkeypatch.setattr(atomic_promotion, "_publish_lock_exclusive", lambda *_: True)
    monkeypatch.setattr(
        atomic_promotion,
        "_remove_owned_lock",
        lambda *_: (_ for _ in ()).throw(OSError("persistent sharing violation")),
    )
    monkeypatch.setattr(
        atomic_promotion,
        "_load_immutable_journal_history",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("SMB unavailable")),
    )

    with pytest.raises(PromotionProjectionRepairRequired) as exc:
        with atomic_promotion._promotion_lock(tmp_path) as lock_state:
            lock_state.bind_journal_event(event_id)
            pass

    assert exc.value.event_id == event_id
    assert exc.value.commit_status == "INDETERMINATE"
    assert exc.value.lock_cleanup_status == "FAIL"


def test_lock_cleanup_never_maps_a_prior_event_to_the_attempted_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_event_id = "e" * 64
    prior_event_id = "a" * 64
    monkeypatch.setattr(atomic_promotion, "_publish_lock_exclusive", lambda *_: True)
    monkeypatch.setattr(
        atomic_promotion,
        "_remove_owned_lock",
        lambda *_: (_ for _ in ()).throw(OSError("persistent sharing violation")),
    )
    monkeypatch.setattr(
        atomic_promotion,
        "_load_immutable_journal_history",
        lambda *_args, **_kwargs: [{"event_id": prior_event_id}],
    )

    with pytest.raises(PromotionProjectionRepairRequired) as exc:
        with atomic_promotion._promotion_lock(tmp_path) as lock_state:
            lock_state.bind_journal_event(expected_event_id)

    assert exc.value.event_id == expected_event_id
    assert exc.value.commit_status == "INDETERMINATE"


def test_non_journal_finalization_cleanup_failure_never_claims_prior_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "release"
    staging = root / "candidates" / ".run-001.staging"
    staging.mkdir(parents=True)
    bundle = atomic_promotion.CandidateBundle(
        run_id="run-001",
        path=root / "candidates" / "run-001",
        manifest_path=root / "candidates" / "run-001" / "bundle_manifest.json",
        manifest_sha256="b" * 64,
    )
    monkeypatch.setattr(atomic_promotion, "_publish_lock_exclusive", lambda *_: True)
    monkeypatch.setattr(
        atomic_promotion,
        "_remove_owned_lock",
        lambda *_: (_ for _ in ()).throw(OSError("persistent sharing violation")),
    )
    monkeypatch.setattr(
        atomic_promotion,
        "_load_immutable_journal_history",
        lambda *_args, **_kwargs: [{"event_id": "a" * 64}],
    )
    monkeypatch.setattr(
        atomic_promotion,
        "verify_assembled_candidate_evidence",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        atomic_promotion,
        "_finalize_candidate_staging",
        lambda *_args, **_kwargs: bundle,
    )

    with pytest.raises(PromotionError, match="non-journal operation") as exc:
        atomic_promotion.finalize_assembled_candidate_staging(root, run_id="run-001")

    assert not isinstance(exc.value, PromotionProjectionRepairRequired)


def test_candidates_root_link_cannot_escape_release_root(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    try:
        (root / "candidates").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises(PromotionError, match="candidates root cannot be a link"):
        create_candidate_staging(root, run_id="run-001")

    assert list(outside.iterdir()) == []


def test_candidates_root_junction_is_rejected_before_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "releases").resolve()
    root.mkdir()
    candidates = root / "candidates"
    original_is_link = atomic_promotion._path_is_link

    def reports_candidates_as_junction(path: Path) -> bool:
        return path == candidates or original_is_link(path)

    monkeypatch.setattr(atomic_promotion, "_path_is_link", reports_candidates_as_junction)

    with pytest.raises(PromotionError, match="candidates root cannot be a link"):
        create_candidate_staging(root, run_id="run-001")

    assert not candidates.exists()


def test_finalization_reverifies_and_quarantines_hash_rename_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    staging = create_candidate_staging(root, run_id="run-001")
    payload = staging / "payload.txt"
    payload.write_text("original", encoding="utf-8")
    deterministic_path = staging / "markets" / "CH" / "pfc.csv"
    deterministic_path.parent.mkdir(parents=True)
    deterministic_path.write_text("original", encoding="utf-8")
    deterministic_artifacts = {
        deterministic_path.relative_to(staging).as_posix(): hashlib.sha256(
            deterministic_path.read_bytes()
        ).hexdigest()
    }
    deterministic_sha256 = hashlib.sha256(
        json.dumps(
            deterministic_artifacts,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path = staging / "manifests" / "candidate_run_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "lt_candidate_run.v1",
                "run_id": "run-001",
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
    _seal_evidence(staging, "run-001")
    original_replace = atomic_promotion._replace_directory

    def mutate_after_rename(source: Path, destination: Path) -> None:
        original_replace(source, destination)
        (destination / "payload.txt").write_text("mutated", encoding="utf-8")

    monkeypatch.setattr(atomic_promotion, "_replace_directory", mutate_after_rename)

    with pytest.raises(PromotionError, match="failed verification after finalization"):
        finalize_candidate_staging(root, run_id="run-001", staging_path=staging)

    assert not (root / "candidates" / "run-001").exists()
    quarantined = list((root / "candidates").glob(".run-001.failed-*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "payload.txt").read_text(encoding="utf-8") == "mutated"


def test_unsigned_self_hashed_receipt_cannot_authorize_promotion(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt_path = _receipt(tmp_path / "receipt.json", bundle=bundle)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("signature")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(PromotionError, match="signature is missing"):
        promote_candidate(root, run_id="run-001", promotion_receipt_path=receipt_path)
    assert not (root / "current.json").exists()


def test_receipt_tamper_after_signature_cannot_authorize_promotion(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt_path = _receipt(tmp_path / "receipt.json", bundle=bundle)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["decision_summary"]["status"] = "FORGED"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(PromotionError, match="receipt_id|signature is invalid"):
        promote_candidate(root, run_id="run-001", promotion_receipt_path=receipt_path)
    assert not (root / "current.json").exists()


def test_duplicate_json_keys_are_rejected_before_signature_replay(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt = _receipt(tmp_path / "receipt.json", bundle=bundle)
    raw = receipt.read_text(encoding="utf-8")
    receipt.write_text('{"approved":false,' + raw[1:], encoding="utf-8")

    with pytest.raises(PromotionError, match="duplicate JSON key"):
        promote_candidate(root, run_id=bundle.run_id, promotion_receipt_path=receipt)


def test_nonfinite_json_is_rejected_before_signature_replay(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt = _receipt(tmp_path / "receipt.json", bundle=bundle)
    raw = receipt.read_text(encoding="utf-8")
    receipt.write_text(raw.replace('"approved": true', '"approved": NaN'), encoding="utf-8")

    with pytest.raises(PromotionError, match="non-finite JSON number"):
        promote_candidate(root, run_id=bundle.run_id, promotion_receipt_path=receipt)


def test_signed_journal_head_rejects_boolean_sequence(tmp_path: Path) -> None:
    root = (tmp_path / "release").resolve()
    root.mkdir()
    head = sign_promotion_head(
        {
            "schema_version": "promotion_head.v1",
            "release_root_sha256": atomic_promotion._release_root_id(root),
            "event_id": "e" * 64,
            "previous_event_id": None,
            "sequence": True,
            "updated_at_utc": "2026-07-15T08:00:00+00:00",
        },
        private_key_path=os.environ["PFC_PROMOTION_EVENT_SIGNING_PRIVATE_KEY_PATH"],
    )
    journal_root = atomic_promotion._promotion_journal_root(root)
    history_path = atomic_promotion._journal_history_path(journal_root, root, head)
    history_path.write_text(json.dumps(head), encoding="utf-8")

    with pytest.raises(PromotionError, match="sequence is invalid"):
        atomic_promotion._load_immutable_journal_history(root)


def test_candidate_bundle_manifest_rejects_boolean_run_id(tmp_path: Path) -> None:
    manifest = tmp_path / "candidate_bundle_manifest.json"
    manifest.write_text(json.dumps({"run_id": True}), encoding="utf-8")

    with pytest.raises(PromotionError, match="manifest run_id is invalid"):
        atomic_promotion.verify_candidate_bundle_manifest(manifest)


def test_replay_rejects_event_signed_for_different_receipt_operation(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    pointer = promote_candidate(
        root,
        run_id=bundle.run_id,
        promotion_receipt_path=_receipt(tmp_path / "receipt.json", bundle=bundle),
    )
    original = atomic_promotion._load_promotion_event(root.resolve(), str(pointer["event_id"]))
    forged_payload = {
        key: value
        for key, value in original.items()
        if key not in {"event_id", "signature"}
    }
    forged_payload["operation_id"] = "event-authority-rebased-operation"
    forged = atomic_promotion._write_promotion_event(root.resolve(), forged_payload)

    with pytest.raises(PromotionError, match="receipt operation binding mismatch"):
        atomic_promotion._replay_promotion_event_chain(
            root.resolve(),
            start_event_id=str(forged["event_id"]),
        )


def test_validly_signed_event_rejects_numeric_operation_id(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    pointer = promote_candidate(
        root,
        run_id=bundle.run_id,
        promotion_receipt_path=_receipt(tmp_path / "receipt.json", bundle=bundle),
    )
    original = atomic_promotion._load_promotion_event(
        root.resolve(), str(pointer["event_id"])
    )
    forged_payload = {
        key: value for key, value in original.items() if key != "event_id"
    }
    forged_payload["operation_id"] = 7
    forged = sign_promotion_event(
        forged_payload,
        private_key_path=os.environ["PFC_PROMOTION_EVENT_SIGNING_PRIVATE_KEY_PATH"],
    )
    forged_path = root / "promotion_events" / f"{forged['event_id']}.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(PromotionError, match="operation_id must be an exact string"):
        atomic_promotion._load_promotion_event(
            root.resolve(), str(forged["event_id"])
        )


def test_validly_signed_receipt_rejects_boolean_run_id(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "True", "candidate")
    receipt_path = _receipt(tmp_path / "receipt.json", bundle=bundle)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["run_id"] = True
    forged = sign_promotion_receipt(
        payload,
        private_key_path=os.environ["PFC_TEST_PROMOTION_SIGNING_PRIVATE_KEY_PATH"],
    )
    receipt_path.write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(PromotionError, match="run_id must be an exact string"):
        promote_candidate(
            root,
            run_id=bundle.run_id,
            promotion_receipt_path=receipt_path,
            require_assembled_evidence=True,
            **_contract_from_receipt(receipt_path),
        )


def test_event_key_rotation_replays_history_from_public_keyring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    first = _candidate(root, "run-001", "first")
    promote_candidate(
        root,
        run_id=first.run_id,
        promotion_receipt_path=_receipt(tmp_path / "first.json", bundle=first),
    )
    old_public = Path(os.environ["PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_PATH"])
    keyring = tmp_path / "event-keyring"
    keyring.mkdir()
    (keyring / "historical.pem").write_bytes(old_public.read_bytes())
    new_key = Ed25519PrivateKey.generate()
    new_private = tmp_path / "event-new-private.pem"
    new_public = tmp_path / "event-new-public.pem"
    new_private.write_bytes(
        new_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    new_public.write_bytes(
        new_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv("PFC_PROMOTION_EVENT_SIGNING_PRIVATE_KEY_PATH", str(new_private))
    monkeypatch.setenv("PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_PATH", str(new_public))
    monkeypatch.setenv("PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_DIR", str(keyring))
    second = _candidate(root, "run-002", "second")

    pointer = promote_candidate(
        root,
        run_id=second.run_id,
        promotion_receipt_path=_receipt(tmp_path / "second.json", bundle=second),
    )

    assert resolve_current_candidate(root).event["event_id"] == pointer["event_id"]


def test_receipt_key_rotation_replays_archived_receipt_from_public_keyring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    pointer = promote_candidate(
        root,
        run_id=bundle.run_id,
        promotion_receipt_path=_receipt(tmp_path / "receipt.json", bundle=bundle),
    )
    old_public = Path(os.environ["PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH"])
    keyring = tmp_path / "receipt-keyring"
    keyring.mkdir()
    (keyring / "historical.pem").write_bytes(old_public.read_bytes())
    new_key = Ed25519PrivateKey.generate()
    new_public = tmp_path / "receipt-new-public.pem"
    new_public.write_bytes(
        new_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv("PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH", str(new_public))
    monkeypatch.setenv("PFC_PROMOTION_TRUSTED_PUBLIC_KEY_DIR", str(keyring))

    assert resolve_current_candidate(root).event["event_id"] == pointer["event_id"]


def test_historical_receipt_key_cannot_authorize_new_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    first = _candidate(root, "run-001", "first")
    promote_candidate(
        root,
        run_id=first.run_id,
        promotion_receipt_path=_receipt(tmp_path / "first.json", bundle=first),
    )
    old_public = Path(os.environ["PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH"])
    keyring = tmp_path / "receipt-keyring"
    keyring.mkdir()
    (keyring / "historical.pem").write_bytes(old_public.read_bytes())
    new_key = Ed25519PrivateKey.generate()
    new_public = tmp_path / "receipt-new-public.pem"
    new_public.write_bytes(
        new_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv("PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH", str(new_public))
    monkeypatch.setenv("PFC_PROMOTION_TRUSTED_PUBLIC_KEY_DIR", str(keyring))
    second = _candidate(root, "run-002", "second")
    historical_receipt = _receipt(tmp_path / "second.json", bundle=second)

    with pytest.raises(PromotionError, match="signing key is not trusted"):
        promote_candidate(
            root,
            run_id=second.run_id,
            promotion_receipt_path=historical_receipt,
        )

    assert resolve_current_candidate(root).bundle.run_id == first.run_id


def test_historical_rollback_key_cannot_authorize_new_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    first = _candidate(root, "run-001", "first")
    first_pointer = promote_candidate(
        root,
        run_id=first.run_id,
        promotion_receipt_path=_receipt(tmp_path / "first.json", bundle=first),
    )
    second = _candidate(root, "run-002", "second")
    second_pointer = promote_candidate(
        root,
        run_id=second.run_id,
        promotion_receipt_path=_receipt(tmp_path / "second.json", bundle=second),
    )
    old_public = Path(
        os.environ["PFC_ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_PATH"]
    )
    keyring = tmp_path / "rollback-keyring"
    keyring.mkdir()
    (keyring / "historical.pem").write_bytes(old_public.read_bytes())
    new_key = Ed25519PrivateKey.generate()
    new_public = tmp_path / "rollback-new-public.pem"
    new_public.write_bytes(
        new_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv(
        "PFC_ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_PATH",
        str(new_public),
    )
    monkeypatch.setenv(
        "PFC_ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_DIR",
        str(keyring),
    )
    authorization = _rollback_authorization(
        tmp_path / "rollback.json",
        expected_current_event_id=str(second_pointer["event_id"]),
        target_promotion_event_id=str(first_pointer["event_id"]),
        target_run_id=first.run_id,
        target_bundle_manifest_sha256=first.manifest_sha256,
    )

    with pytest.raises(PromotionError, match="signing key is not trusted"):
        rollback_to_candidate(root, rollback_authorization_path=authorization)

    assert resolve_current_candidate(root).bundle.run_id == second.run_id


def test_receipt_signed_by_untrusted_key_cannot_authorize_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt_path = _receipt(tmp_path / "receipt.json", bundle=bundle)
    other_key = Ed25519PrivateKey.generate().public_key()
    other_public = tmp_path / "other-public.pem"
    other_public.write_bytes(
        other_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    monkeypatch.setenv("PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH", str(other_public))
    with pytest.raises(PromotionError, match="signing key is not trusted"):
        promote_candidate(root, run_id="run-001", promotion_receipt_path=receipt_path)
    assert not (root / "current.json").exists()


def test_primary_receipt_trust_anchor_link_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt_path = _receipt(tmp_path / "receipt.json", bundle=bundle)
    trusted = Path(os.environ["PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH"])
    original_is_link = atomic_promotion._path_is_link

    def reports_trust_anchor_as_link(path: Path) -> bool:
        return path == trusted or original_is_link(path)

    monkeypatch.setattr(atomic_promotion, "_path_is_link", reports_trust_anchor_as_link)

    with pytest.raises(PromotionError, match="cannot contain a link"):
        promote_candidate(root, run_id=bundle.run_id, promotion_receipt_path=receipt_path)


def test_relative_primary_receipt_trust_anchor_is_rejected_before_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH",
        "relative/receipt-public.pem",
    )

    with pytest.raises(PromotionError, match="must be absolute"):
        atomic_promotion._trusted_public_key_path()


def test_real_linked_primary_receipt_trust_anchor_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt_path = _receipt(tmp_path / "receipt.json", bundle=bundle)
    target = Path(os.environ["PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH"])
    linked = tmp_path / "linked-receipt-public.pem"
    try:
        linked.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")
    monkeypatch.setenv("PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH", str(linked))

    with pytest.raises(PromotionError, match="cannot contain a link"):
        promote_candidate(root, run_id=bundle.run_id, promotion_receipt_path=receipt_path)


def test_promoter_api_does_not_accept_operator_supplied_trust_anchor(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt_path = _receipt(tmp_path / "receipt.json", bundle=bundle)

    with pytest.raises(TypeError, match="trusted_public_key_path"):
        promote_candidate(
            root,
            run_id="run-001",
            promotion_receipt_path=receipt_path,
            **{"trusted_public_key_path": tmp_path / "attacker.pem"},
        )


@pytest.mark.parametrize(
    "gate_ids",
    [
        sorted(REQUIRED_PROMOTION_GATE_IDS)[1:],
        [*sorted(REQUIRED_PROMOTION_GATE_IDS), "unreviewed_extra_gate"],
        [*sorted(REQUIRED_PROMOTION_GATE_IDS), sorted(REQUIRED_PROMOTION_GATE_IDS)[0]],
    ],
)
def test_signed_receipt_requires_exact_unique_gate_set(
    tmp_path: Path,
    gate_ids: list[str],
) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "candidate")
    receipt_path = _receipt(
        tmp_path / "receipt.json",
        bundle=bundle,
        gate_ids=gate_ids,
    )

    with pytest.raises(PromotionError, match="gate set mismatch|duplicate governance gates"):
        promote_candidate(root, run_id="run-001", promotion_receipt_path=receipt_path)
    assert not (root / "current.json").exists()


def test_rollback_rejects_candidate_never_approved(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    approved = _candidate(root, "run-001", "approved")
    never_approved = _candidate(root, "run-never-approved", "not approved")
    current = promote_candidate(
        root,
        run_id="run-001",
        promotion_receipt_path=_receipt(tmp_path / "approved.json", bundle=approved),
    )
    before = (root / "current.json").read_bytes()

    authorization = _rollback_authorization(
        tmp_path / "rollback-never-approved.json",
        expected_current_event_id=str(current["event_id"]),
        target_promotion_event_id="f" * 64,
        target_run_id=never_approved.run_id,
        target_bundle_manifest_sha256=never_approved.manifest_sha256,
    )
    with pytest.raises(PromotionError, match="exact committed PROMOTE"):
        rollback_to_candidate(
            root,
            rollback_authorization_path=authorization,
        )
    assert (root / "current.json").read_bytes() == before


def test_current_reader_reverifies_bundle_event_and_archived_receipt(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "approved")
    external_receipt = _receipt(tmp_path / "external.json", bundle=bundle)
    pointer = promote_candidate(
        root,
        run_id="run-001",
        promotion_receipt_path=external_receipt,
    )
    external_receipt.unlink()

    current = resolve_current_candidate(root)
    assert current.bundle.run_id == "run-001"
    assert current.event["event_id"] == pointer["event_id"]
    assert str(current.receipt["receipt_id"])

    (bundle.path / "markets" / "ch" / "pfc.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(PromotionError, match="deterministic artifacts are invalid"):
        resolve_current_candidate(root)


def test_current_reader_rejects_tampered_promotion_event(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "approved")
    pointer = promote_candidate(
        root,
        run_id="run-001",
        promotion_receipt_path=_receipt(tmp_path / "external.json", bundle=bundle),
    )
    event_path = root / str(pointer["event_path"])
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["run_id"] = "forged-run"
    event_path.write_text(json.dumps(event), encoding="utf-8")

    with pytest.raises(PromotionError, match="event authentication failed"):
        resolve_current_candidate(root)


def test_current_reader_rejects_coherently_hashed_attacker_event(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "approved")
    pointer = promote_candidate(
        root,
        run_id="run-001",
        promotion_receipt_path=_receipt(tmp_path / "external.json", bundle=bundle),
    )
    attacker_key = Ed25519PrivateKey.generate()
    attacker_private = tmp_path / "attacker-event-private.pem"
    attacker_private.write_bytes(
        attacker_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    forged = sign_promotion_event(
        {
            "schema_version": "promotion_event.v2",
            "event_type": "PROMOTE",
            "run_id": "run-001",
            "candidate_bundle_manifest_sha256": pointer["candidate_bundle_manifest_sha256"],
            "candidate_run_manifest_sha256": pointer["candidate_run_manifest_sha256"],
            "promotion_receipt_path": pointer["promotion_receipt_path"],
            "promotion_receipt_sha256": pointer["promotion_receipt_sha256"],
            "receipt_id": json.loads(
                (root / str(pointer["promotion_receipt_path"])).read_text(encoding="utf-8")
            )["receipt_id"],
            "recorded_at_utc": "2026-07-13T00:06:00+00:00",
            "previous_event_id": pointer["event_id"],
        },
        private_key_path=attacker_private,
    )
    forged_path = root / "promotion_events" / f"{forged['event_id']}.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")
    current = json.loads((root / "current.json").read_text(encoding="utf-8"))
    current["event_id"] = forged["event_id"]
    current["event_path"] = f"promotion_events/{forged['event_id']}.json"
    (root / "current.json").write_text(json.dumps(current), encoding="utf-8")

    with pytest.raises(PromotionError, match="uncommitted promotion event"):
        resolve_current_candidate(root)


def test_current_reader_ignores_stale_valid_pointer_projection(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    first = _candidate(root, "run-001", "first")
    first_pointer = promote_candidate(
        root,
        run_id="run-001",
        promotion_receipt_path=_receipt(tmp_path / "first.json", bundle=first),
    )
    second = _candidate(root, "run-002", "second")
    second_pointer = promote_candidate(
        root,
        run_id="run-002",
        promotion_receipt_path=_receipt(tmp_path / "second.json", bundle=second),
    )

    (root / "current.json").write_text(json.dumps(first_pointer), encoding="utf-8")

    current = resolve_current_candidate(root)

    assert current.event["event_id"] == second_pointer["event_id"]
    assert current.bundle.run_id == "run-002"


def test_current_reader_ignores_stale_valid_pointer_and_mutable_head(
    tmp_path: Path,
) -> None:
    root = tmp_path / "releases"
    first = _candidate(root, "run-001", "first")
    first_pointer = promote_candidate(
        root,
        run_id="run-001",
        promotion_receipt_path=_receipt(tmp_path / "first.json", bundle=first),
    )
    mutable_head = next((tmp_path / "promotion-journal" / "heads").glob("*.json"))
    first_head = mutable_head.read_bytes()
    second = _candidate(root, "run-002", "second")
    second_pointer = promote_candidate(
        root,
        run_id="run-002",
        promotion_receipt_path=_receipt(tmp_path / "second.json", bundle=second),
    )

    (root / "current.json").write_text(json.dumps(first_pointer), encoding="utf-8")
    mutable_head.write_bytes(first_head)

    current = resolve_current_candidate(root)

    assert current.event["event_id"] == second_pointer["event_id"]
    assert current.bundle.run_id == "run-002"


def test_finalizer_requires_canonical_candidate_run_manifest(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    staging = create_candidate_staging(root, run_id="run-001")
    (staging / "payload.txt").write_text("payload", encoding="utf-8")

    with pytest.raises(PromotionError, match="candidate run manifest is missing"):
        finalize_candidate_staging(root, run_id="run-001", staging_path=staging)


def test_nested_bundle_manifest_filename_is_hash_bound(tmp_path: Path) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "approved")
    nested = bundle.path / "evidence" / "candidate_bundle_manifest.json"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_text("not the root manifest", encoding="utf-8")

    with pytest.raises(PromotionError, match="file set/hash mismatch"):
        verify_candidate_bundle(root, run_id="run-001")


@pytest.mark.parametrize(
    ("evaluated_at", "message"),
    [
        ("2026-07-12T23:00:00+00:00", "too old"),
        ("2026-07-13T00:10:00+00:00", "in the future"),
    ],
)
def test_new_promotion_requires_recent_runtime_evaluation(
    tmp_path: Path,
    evaluated_at: str,
    message: str,
) -> None:
    root = tmp_path / "releases"
    bundle = _candidate(root, "run-001", "approved")
    receipt = _receipt(
        tmp_path / "receipt.json",
        bundle=bundle,
        evaluated_at=evaluated_at,
    )

    with pytest.raises(PromotionError, match=message):
        promote_candidate(root, run_id="run-001", promotion_receipt_path=receipt)
    assert not (root / "current.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing retry contract")
def test_directory_replace_retries_transient_windows_sharing_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "staging"
    destination = tmp_path / "candidate"
    source.mkdir()
    real_replace = atomic_promotion.os.replace
    attempts = 0

    def flaky_replace(first: Path, second: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 7:
            raise PermissionError(5, "sharing violation")
        real_replace(first, second)

    monkeypatch.setattr(atomic_promotion.os, "replace", flaky_replace)
    monkeypatch.setattr(atomic_promotion.time, "sleep", lambda _seconds: None)

    atomic_promotion._replace_directory(source, destination)

    assert destination.is_dir()
    assert not source.exists()
    assert attempts == 7


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing retry contract")
def test_atomic_json_replace_retries_transient_windows_sharing_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "pointer.json"
    target.write_text('{"generation":"old"}\n', encoding="utf-8")
    real_replace = atomic_promotion.os.replace
    attempts = 0

    def flaky_replace(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "sharing violation")
        real_replace(source, destination)

    monkeypatch.setattr(atomic_promotion.os, "replace", flaky_replace)

    atomic_promotion._atomic_write_json(target, {"generation": "new"})

    assert atomic_promotion._load_json(target) == {"generation": "new"}
    assert attempts == 3


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing retry contract")
def test_atomic_json_reader_retries_transient_windows_sharing_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "pointer.json"
    target.write_text('{"generation":"stable"}\n', encoding="utf-8")
    real_read_bytes = Path.read_bytes
    attempts = 0

    def flaky_read_bytes(path: Path) -> bytes:
        nonlocal attempts
        if path == target:
            attempts += 1
            if attempts < 3:
                raise PermissionError(5, "sharing violation")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", flaky_read_bytes)

    assert atomic_promotion._load_json(target) == {"generation": "stable"}
    assert attempts == 3


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing retry contract")
def test_atomic_json_replace_deadline_preserves_old_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "pointer.json"
    original = b'{"generation":"old"}\n'
    target.write_bytes(original)

    def always_busy(_source: Path, _destination: Path) -> None:
        raise PermissionError(5, "sharing violation")

    monkeypatch.setattr(atomic_promotion.os, "replace", always_busy)

    with pytest.raises(TimeoutError, match="deadline"):
        atomic_promotion._atomic_write_json(
            target,
            {"generation": "new"},
            deadline=time.monotonic() + 0.01,
        )

    assert target.read_bytes() == original

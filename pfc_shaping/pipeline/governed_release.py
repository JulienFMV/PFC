"""Durable controller for governed LT audit and promotion transitions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from pfc_shaping.path_safety import path_is_link
from pfc_shaping.pipeline.atomic_promotion import (
    BUNDLE_MANIFEST,
    EVENT_SIGNING_PRIVATE_KEY_ENV,
    RECEIPT_TRUSTED_PUBLIC_KEY_DIR_ENV,
    PromotionError,
    PromotionProjectionRepairRequired,
    _matching_trusted_public_key,
    _release_root_id,
    _required_env_key_path,
    _trusted_public_key_identities,
    promote_candidate,
    resolve_current_candidate,
    rollback_to_candidate,
    verify_candidate_bundle,
)
from pfc_shaping.pipeline.candidate_evidence import verify_assembled_candidate_evidence
from pfc_shaping.pipeline.candidate_evidence_assembler import (
    verify_candidate_derived_evidence,
)
from pfc_shaping.pipeline.candidate_product_evidence import (
    PRODUCT_EVIDENCE_MANIFEST,
    verify_candidate_product_evidence,
)
from pfc_shaping.pipeline.governed_release_cli_contract import (
    ReleaseCliIdentityError,
    assert_installed_runtime_sealed,
    assert_phase_private_key_scope,
)
from pfc_shaping.pipeline.process_identity import candidate_temporary_name
from pfc_shaping.pipeline.promotion_contract import (
    ReceiptAuthenticationError,
    public_key_id_from_private_key_path,
    public_key_id_from_public_key_path,
    validate_promotion_receipt_decision,
    validate_required_gate_set,
    verify_promotion_receipt_signature,
)
from pfc_shaping.pipeline.release_request_contract import (
    REGISTER_SIGNING_PRIVATE_KEY_ENV,
    REGISTER_TRUSTED_PUBLIC_KEY_DIR_ENV,
    REGISTER_TRUSTED_PUBLIC_KEY_ENV,
    RELEASE_REQUEST_SCHEMA,
    ReleaseRequestAuthenticationError,
    ensure_workflow_domain,
    load_registered_release_request,
    release_request_document_sha256,
    sign_release_request,
)
from pfc_shaping.pipeline.strict_structured_data import (
    StrictStructuredDataError,
    load_strict_json,
)

REQUEST_SCHEMA = RELEASE_REQUEST_SCHEMA
AUDIT_RESULT_SCHEMA = "governed_lt_audit_result.v1"
PROMOTION_RESULT_SCHEMA = "governed_lt_promotion_result.v1"
ROLLBACK_RESULT_SCHEMA = "governed_lt_rollback_result.v1"
RECEIPT_PRIVATE_KEY_ENV = "PFC_PROMOTION_SIGNING_PRIVATE_KEY_PATH"
RECEIPT_TRUSTED_KEY_ENV = "PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH"
REQUIRED_ARTIFACT_ROLES = (
    "audit_gates",
    "historical_thresholds",
    "production_manifest",
    "export_manifest",
    "selected_config_artifact",
    "product_normalization_summary",
)
OPTIONAL_ARTIFACT_ROLES = ("audit_manifest",)


class GovernedReleaseError(RuntimeError):
    """Raised when a governed release transition cannot be proven safe."""


def register_release_request(
    *,
    release_root: str | Path,
    workflow_root: str | Path,
    evidence_root: str | Path,
    run_id: str,
    artifacts: Mapping[str, str | Path],
    expected_current_event_id: str | None,
    request_nonce: str | None = None,
) -> dict[str, object]:
    """Reject the retired external-artifact registration contract."""

    _assert_phase_identity("register")
    del (
        release_root,
        workflow_root,
        evidence_root,
        run_id,
        artifacts,
        expected_current_event_id,
        request_nonce,
    )
    raise GovernedReleaseError(
        "legacy external-artifact registration is permanently disabled"
    )


def _persist_release_request(
    *,
    workflow: Path,
    workflow_domain_sha256: str,
    release_root_sha256: str,
    run_id: str,
    candidate_bundle_manifest_sha256: str,
    captured: Mapping[str, object],
    registration_contract: str,
    expected_current_event_id: str | None,
    request_nonce: str | None,
) -> dict[str, object]:
    expected_current = _expected_current_contract(expected_current_event_id)
    stable_request: dict[str, object] = {
        "schema_version": REQUEST_SCHEMA,
        "run_id": run_id,
        "candidate_bundle_manifest_sha256": candidate_bundle_manifest_sha256,
        "registration_contract": str(registration_contract),
        "release_root_sha256": release_root_sha256,
        "workflow_domain_sha256": workflow_domain_sha256,
        "expected_current": expected_current,
        "artifacts": captured,
        "evidence_inventory_sha256": _sha256_json(captured),
    }
    if request_nonce is not None:
        stable_request["request_nonce"] = _request_nonce(request_nonce)
    existing = _find_matching_request(workflow, run_id, stable_request)
    if existing is not None:
        return existing
    request_unsigned = {
        **stable_request,
        "created_at_utc": _canonical_utc_seconds(_now_utc()),
    }
    signing_key = str(os.environ.get(REGISTER_SIGNING_PRIVATE_KEY_ENV, "")).strip()
    if not signing_key:
        raise GovernedReleaseError(f"{REGISTER_SIGNING_PRIVATE_KEY_ENV} is required")
    _assert_register_signing_authority(signing_key)
    try:
        request = sign_release_request(
            request_unsigned,
            private_key_path=signing_key,
        )
    except ReleaseRequestAuthenticationError as exc:
        raise GovernedReleaseError(f"release request signing failed: {exc}") from exc
    path = _request_path(workflow, run_id, request_id=str(request["request_id"]))
    if path.exists():
        existing = _load_json(path)
        try:
            existing = load_registered_release_request(
                path,
                workflow_root=workflow,
                expected_run_id=run_id,
                allow_historical_key=True,
            )
        except ReleaseRequestAuthenticationError as exc:
            raise GovernedReleaseError(f"release request authentication failed: {exc}") from exc
        _verify_request_document(existing, expected_run_id=run_id)
        if existing != request:
            raise GovernedReleaseError("release request identity collision")
        return existing
    _write_json_exclusive(path, request)
    return request


def _find_matching_request(
    workflow: Path,
    run_id: str,
    stable_request: Mapping[str, object],
) -> dict[str, object] | None:
    requests = _run_workflow_path(workflow, run_id) / "requests"
    if not requests.is_dir():
        return None
    for path in sorted(requests.glob("*/release_request.json")):
        try:
            existing = load_registered_release_request(
                path,
                workflow_root=workflow,
                expected_run_id=run_id,
                allow_historical_key=True,
            )
        except ReleaseRequestAuthenticationError as exc:
            raise GovernedReleaseError(f"release request authentication failed: {exc}") from exc
        _verify_request_document(existing, expected_run_id=run_id)
        comparable = {
            key: value
            for key, value in existing.items()
            if key not in {"request_id", "created_at_utc", "register_signature"}
        }
        if comparable == dict(stable_request):
            return existing
    return None


def register_assembled_release_request(
    *,
    release_root: str | Path,
    workflow_root: str | Path,
    evidence_root: str | Path,
    run_id: str,
    expected_current_event_id: str | None,
    request_nonce: str | None = None,
) -> dict[str, object]:
    """Register only artifacts derived from the canonical finalized candidate seal."""

    _assert_phase_identity("register")
    release = _absolute_root(release_root, name="release_root", must_exist=True)
    evidence = _absolute_root(evidence_root, name="evidence_root", must_exist=True)
    bundle = verify_candidate_bundle(release, run_id=run_id)
    sealed = verify_assembled_candidate_evidence(
        bundle.path,
        expected_run_id=run_id,
    )
    entries = sealed.get("artifacts")
    if not isinstance(entries, Mapping):
        raise GovernedReleaseError("canonical candidate evidence inventory is invalid")
    role_map = {
        "audit_gates": "audit_gates",
        "historical_thresholds": "historical_thresholds",
        "production_manifest": "production_manifest",
        "export_manifest": "export_manifest",
        "selected_config_artifact": "selected_config_run_parity",
        "product_normalization_summary": "product_normalization_summary",
    }
    captured: dict[str, object] = {}
    destination = evidence / run_id
    destination.mkdir(parents=True, exist_ok=True)
    for release_role, seal_role in role_map.items():
        entry = entries.get(seal_role)
        if not isinstance(entry, Mapping):
            raise GovernedReleaseError(f"canonical seal role is missing: {seal_role}")
        source = (bundle.path / str(entry.get("path", ""))).resolve()
        try:
            source.relative_to(bundle.path)
        except ValueError as exc:
            raise GovernedReleaseError(f"canonical seal role escapes bundle: {seal_role}") from exc
        if not source.is_file() or source.is_symlink():
            raise GovernedReleaseError(f"canonical seal role is unavailable: {seal_role}")
        try:
            payload = source.read_bytes()
        except OSError as exc:
            raise GovernedReleaseError(
                f"canonical seal role cannot be captured: {seal_role}"
            ) from exc
        expected_sha256 = str(entry.get("sha256", ""))
        expected_size = int(entry.get("size_bytes", -1))
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise GovernedReleaseError(f"canonical seal role hash mismatch: {seal_role}")
        if len(payload) != expected_size:
            raise GovernedReleaseError(f"canonical seal role size mismatch: {seal_role}")
        target = destination / f"{release_role}{source.suffix}"
        _write_bytes_idempotent(target, payload)
        captured[release_role] = {
            "path": target.relative_to(evidence).as_posix(),
            "sha256": expected_sha256,
            "size_bytes": expected_size,
        }
    workflow = _external_workflow_path(workflow_root, release_root=release)
    if not workflow.is_dir():
        raise GovernedReleaseError(f"workflow_root is unavailable: {workflow}")
    try:
        workflow, workflow_domain_sha256 = ensure_workflow_domain(workflow, create=False)
    except ReleaseRequestAuthenticationError as exc:
        raise GovernedReleaseError(f"workflow domain authentication failed: {exc}") from exc
    return _persist_release_request(
        workflow=workflow,
        workflow_domain_sha256=workflow_domain_sha256,
        release_root_sha256=_release_root_sha256(release, create=False),
        run_id=run_id,
        candidate_bundle_manifest_sha256=bundle.manifest_sha256,
        captured=captured,
        registration_contract="assembled_candidate_seal.v1",
        expected_current_event_id=expected_current_event_id,
        request_nonce=request_nonce,
    )


def audit_release_request(
    *,
    release_root: str | Path,
    workflow_root: str | Path,
    evidence_root: str | Path,
    data_root: str | Path,
    run_id: str,
    request_id: str,
    signing_private_key: str | Path | None,
    require_assembled_registration: bool = True,
) -> dict[str, object]:
    if require_assembled_registration is not True:
        raise GovernedReleaseError(
            "audit without assembled candidate registration is permanently disabled"
        )
    _assert_phase_identity("audit")
    if os.environ.get(EVENT_SIGNING_PRIVATE_KEY_ENV):
        raise GovernedReleaseError(
            f"audit identity must not expose {EVENT_SIGNING_PRIVATE_KEY_ENV}"
        )
    key = (
        _authority_private_key_path(signing_private_key)
        if signing_private_key is not None
        else None
    )
    if key is not None:
        _assert_audit_signing_authority(key)
    release, workflow, evidence, request = _load_bound_request(
        release_root=release_root,
        workflow_root=workflow_root,
        evidence_root=evidence_root,
        run_id=run_id,
        request_id=request_id,
        require_assembled_registration=require_assembled_registration,
    )
    receipt_path = _receipt_path(
        workflow,
        run_id,
        request_id=str(request["request_id"]),
    )
    registered_request_path = _request_path(
        workflow,
        run_id,
        request_id=str(request["request_id"]),
        read_only=True,
    )
    if receipt_path.exists():
        result = _derive_audit_result(
            request,
            receipt_path=receipt_path,
            registered_request_path=registered_request_path,
            allow_historical_key=True,
        )
        _write_result_idempotent(
            _audit_result_dir(workflow, run_id, str(request["request_id"]))
            / "audit_result.json",
            result,
        )
        return result
    data = _absolute_root(data_root, name="data_root", must_exist=True)
    if key is None:
        raise GovernedReleaseError(
            "receipt signing private key is required when no receipt exists"
        )
    artifacts = _artifact_paths(request, evidence_root=evidence)
    run_dir = _audit_result_dir(workflow, run_id, str(request["request_id"]))
    expected_current = request.get("expected_current")
    if not isinstance(expected_current, Mapping):
        raise GovernedReleaseError("release request expected_current is invalid")
    capstone_args = [
        "--audit-gates",
        str(artifacts["audit_gates"]),
        "--historical-thresholds",
        str(artifacts["historical_thresholds"]),
        "--production-manifest",
        str(artifacts["production_manifest"]),
        "--export-manifest",
        str(artifacts["export_manifest"]),
        "--selected-config-artifact",
        str(artifacts["selected_config_artifact"]),
        "--candidate-bundle-manifest",
        str(release / "candidates" / run_id / BUNDLE_MANIFEST),
        "--data-root",
        str(data),
        "--product-normalization-summary",
        str(artifacts["product_normalization_summary"]),
        "--signing-private-key",
        str(key),
        "--release-request-id",
        str(request["request_id"]),
        "--release-operation-id",
        str(request["request_id"]),
        "--expected-current-event-id",
        (
            str(expected_current["event_id"])
            if expected_current.get("kind") == "EVENT"
            else "NONE"
        ),
        "--operation-created-at-utc",
        str(request["created_at_utc"]),
        "--release-root-sha256",
        str(request["release_root_sha256"]),
        "--evidence-inventory-sha256",
        str(request["evidence_inventory_sha256"]),
        "--registered-release-request",
        str(_request_path(workflow, run_id, request_id=str(request["request_id"]))),
        "--workflow-root",
        str(workflow),
        "--augmented-audit-gates",
        str(run_dir / "augmented_audit_gates.csv"),
        "--details-output",
        str(run_dir / "promotion_details.csv"),
        "--output",
        str(receipt_path),
    ]
    if "audit_manifest" in artifacts:
        capstone_args.extend(["--manifest", str(artifacts["audit_manifest"])])
    try:
        return_code = _run_capstone(capstone_args)
        result = _derive_audit_result(
            request,
            receipt_path=receipt_path,
            registered_request_path=registered_request_path,
        )
        if (return_code == 0) != bool(result["approved"]):
            raise GovernedReleaseError("capstone return code and signed receipt disagree")
        _write_result_idempotent(run_dir / "audit_result.json", result)
        return result
    except Exception as exc:
        _record_failure_best_effort(
            workflow,
            run_id=run_id,
            phase="AUDIT",
            request=request,
            exc=exc,
        )
        raise


def promote_release_request(
    *,
    release_root: str | Path,
    workflow_root: str | Path,
    evidence_root: str | Path,
    run_id: str,
    request_id: str,
    require_assembled_registration: bool = True,
) -> dict[str, object]:
    if require_assembled_registration is not True:
        raise GovernedReleaseError(
            "promotion without assembled candidate registration is permanently disabled"
        )
    _assert_phase_identity("promote")
    if os.environ.get(RECEIPT_PRIVATE_KEY_ENV):
        raise GovernedReleaseError(
            f"promotion identity must not expose {RECEIPT_PRIVATE_KEY_ENV}"
        )
    release, workflow, _, request = _load_bound_request(
        release_root=release_root,
        workflow_root=workflow_root,
        evidence_root=evidence_root,
        run_id=run_id,
        request_id=request_id,
        require_assembled_registration=require_assembled_registration,
        allow_historical_request_key=True,
    )
    receipt_path = _receipt_path(
        workflow,
        run_id,
        request_id=str(request["request_id"]),
        read_only=True,
    )
    audit_result = _derive_audit_result(
        request,
        receipt_path=receipt_path,
        registered_request_path=_request_path(
            workflow,
            run_id,
            request_id=str(request["request_id"]),
            read_only=True,
        ),
        allow_historical_key=True,
    )
    if audit_result["approved"] is not True:
        raise GovernedReleaseError("signed capstone receipt is not approved")
    try:
        expected_current = request["expected_current"]
        if not isinstance(expected_current, Mapping):
            raise GovernedReleaseError("release request expected_current is invalid")
        pointer = promote_candidate(
            release,
            run_id=run_id,
            promotion_receipt_path=receipt_path,
            require_assembled_evidence=require_assembled_registration,
            expected_current_event_id=(
                str(expected_current["event_id"])
                if expected_current.get("kind") == "EVENT"
                else None
            ),
            operation_id=str(request["request_id"]),
            release_request_id=str(request["request_id"]),
            operation_created_at_utc=str(request["created_at_utc"]),
            registered_release_request_path=_request_path(
                workflow,
                run_id,
                request_id=str(request["request_id"]),
                read_only=True,
            ),
            workflow_root=workflow,
        )
        try:
            result = {
                "schema_version": PROMOTION_RESULT_SCHEMA,
                "request_id": request["request_id"],
                "run_id": run_id,
                "status": "PROMOTED",
                "promotion_receipt_sha256": pointer["promotion_receipt_sha256"],
                "event_id": pointer["event_id"],
                "recorded_at_utc": _now_utc(),
            }
            result = _write_result_idempotent(
                _promotion_result_path(
                    workflow,
                    run_id,
                    str(request["request_id"]),
                ),
                result,
            )
        except BaseException as exc:
            raise PromotionProjectionRepairRequired(
                str(pointer["event_id"]),
                commit_status="COMMITTED",
            ) from exc
        return result
    except Exception as exc:
        _record_failure_best_effort(
            workflow,
            run_id=run_id,
            phase="PROMOTE",
            request=request,
            exc=exc,
        )
        raise


def rollback_release(
    *,
    release_root: str | Path,
    rollback_authorization_path: str | Path,
) -> dict[str, object]:
    """Apply one independently authorized rollback to an exact PROMOTE event."""

    _assert_phase_identity("rollback")
    if os.environ.get(RECEIPT_PRIVATE_KEY_ENV):
        raise GovernedReleaseError(
            f"rollback identity must not expose {RECEIPT_PRIVATE_KEY_ENV}"
        )
    release = _absolute_path(release_root, name="release_root")
    authorization = _absolute_regular_input_path(
        rollback_authorization_path,
        name="rollback_authorization_path",
    )
    pointer = rollback_to_candidate(
        release,
        rollback_authorization_path=authorization,
    )
    return {
        "schema_version": ROLLBACK_RESULT_SCHEMA,
        "status": "ROLLED_BACK",
        "run_id": pointer["run_id"],
        "event_id": pointer["event_id"],
        "previous_run_id": pointer.get("previous_run_id"),
        "rollback_reason": pointer.get("rollback_reason"),
    }


def release_status(
    *,
    release_root: str | Path,
    workflow_root: str | Path,
    evidence_root: str | Path,
    run_id: str,
    request_id: str | None = None,
) -> dict[str, object]:
    _assert_phase_identity("status")
    release = _absolute_path(release_root, name="release_root")
    workflow = _external_workflow_path(workflow_root, release_root=release)
    evidence = _absolute_path(evidence_root, name="evidence_root")
    run_dir = _run_workflow_path(workflow, run_id)
    staging = release / "candidates" / f".{run_id}.staging"
    final = release / "candidates" / run_id
    if staging.exists() and final.exists():
        return _status_payload(run_id=run_id, state="INVALID")
    if staging.is_dir():
        try:
            verify_candidate_derived_evidence(staging, expected_run_id=run_id)
            if (staging / PRODUCT_EVIDENCE_MANIFEST).is_file():
                verify_candidate_product_evidence(staging, expected_run_id=run_id)
                state = "STAGED_FINALIZATION_PENDING"
            else:
                state = "STAGED_POLICY_REQUIRED"
        except RuntimeError:
            state = "INVALID"
        return _status_payload(run_id=run_id, state=state)
    if not final.is_dir():
        return _status_payload(run_id=run_id, state="NOT_FOUND")
    requests_dir = run_dir / "requests"
    if not requests_dir.is_dir() or not any(requests_dir.glob("*/release_request.json")):
        request_path = requests_dir / "missing" / "release_request.json"
    else:
        try:
            request_path = _request_path(
                workflow,
                run_id,
                request_id=request_id,
                read_only=True,
            )
        except GovernedReleaseError:
            return _status_payload(run_id=run_id, state="INVALID")
    try:
        bundle = verify_candidate_bundle(release, run_id=run_id)
        if not request_path.is_file():
            verify_assembled_candidate_evidence(bundle.path, expected_run_id=run_id)
    except RuntimeError:
        return _status_payload(run_id=run_id, state="INVALID")
    if not request_path.is_file():
        return _status_payload(
            run_id=run_id,
            state="FINALIZED",
            candidate_verified=True,
        )
    _, _, _, request = _load_bound_request(
        release_root=release,
        workflow_root=workflow,
        evidence_root=evidence,
        run_id=run_id,
        request_id=request_id,
        read_only=True,
    )
    if request.get("registration_contract") == "assembled_candidate_seal.v1":
        try:
            verify_assembled_candidate_evidence(bundle.path, expected_run_id=run_id)
        except RuntimeError:
            return _status_payload(run_id=run_id, state="INVALID")
    receipt_path = _receipt_path(
        workflow,
        run_id,
        request_id=str(request["request_id"]),
        read_only=True,
    )
    audit: dict[str, object] | None = None
    if receipt_path.exists():
        audit = _derive_audit_result(
            request,
            receipt_path=receipt_path,
            registered_request_path=request_path,
            allow_historical_key=True,
        )
    state = "REGISTERED"
    if audit is not None:
        state = "AUDIT_APPROVED" if audit.get("approved") is True else "AUDIT_REJECTED"
    promoted = False
    event_id = None
    current_event_type = None
    projection_repair_required = False
    promotion_result_path = _promotion_result_path(
        workflow,
        run_id,
        str(request["request_id"]),
        read_only=True,
    )
    try:
        current = resolve_current_candidate(release)
        promoted = (
            current.bundle.run_id == run_id
            and str(current.event.get("release_request_id", ""))
            == str(request["request_id"])
        )
        current_event_type = str(current.event.get("event_type", ""))
        projection_repair_required = current.projection_repair_required
        event_id = current.event.get("event_id") if promoted else None
        if projection_repair_required:
            state = "POINTER_REPAIR_REQUIRED"
        elif promoted and current_event_type == "ROLLBACK":
            state = "ROLLBACK_CURRENT"
        elif promoted:
            state = "PROMOTED_CURRENT"
        elif promotion_result_path.is_file():
            state = "PROMOTED_SUPERSEDED"
    except PromotionError as exc:
        if "no committed history" not in str(exc):
            return _status_payload(run_id=run_id, state="INVALID")
    return _status_payload(
        run_id=run_id,
        state=state,
        request_id=str(request["request_id"]),
        candidate_verified=True,
        audit_receipt_present=audit is not None,
        audit_approved=audit.get("approved") if audit else None,
        promoted_current=promoted,
        event_id=str(event_id) if event_id is not None else None,
        current_event_type=current_event_type,
        projection_repair_required=projection_repair_required,
    )


def _status_payload(
    *,
    run_id: str,
    state: str,
    request_id: str | None = None,
    candidate_verified: bool = False,
    audit_receipt_present: bool = False,
    audit_approved: bool | None = None,
    promoted_current: bool = False,
    event_id: str | None = None,
    current_event_type: str | None = None,
    projection_repair_required: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": "governed_lt_release_status.v2",
        "state": state,
        "request_id": request_id,
        "run_id": run_id,
        "candidate_verified": candidate_verified,
        "audit_receipt_present": audit_receipt_present,
        "audit_approved": audit_approved,
        "promoted_current": promoted_current,
        "event_id": event_id,
        "current_event_type": current_event_type,
        "projection_repair_required": projection_repair_required,
    }


def _load_bound_request(
    *,
    release_root: str | Path,
    workflow_root: str | Path,
    evidence_root: str | Path,
    run_id: str,
    request_id: str | None = None,
    require_assembled_registration: bool = False,
    read_only: bool = False,
    allow_historical_request_key: bool = False,
) -> tuple[Path, Path, Path, dict[str, object]]:
    release = _absolute_root(release_root, name="release_root", must_exist=True)
    workflow = _external_workflow_path(workflow_root, release_root=release)
    if not workflow.is_dir():
        raise GovernedReleaseError(f"workflow_root is unavailable: {workflow}")
    evidence = _absolute_root(evidence_root, name="evidence_root", must_exist=True)
    request_path = _request_path(
        workflow,
        run_id,
        request_id=request_id,
        read_only=True,
    )
    try:
        request = load_registered_release_request(
            request_path,
            workflow_root=workflow,
            expected_run_id=run_id,
            allow_historical_key=(read_only or allow_historical_request_key),
        )
    except ReleaseRequestAuthenticationError as exc:
        raise GovernedReleaseError(f"release request authentication failed: {exc}") from exc
    _verify_request_document(request, expected_run_id=run_id)
    if request_id is not None and str(request.get("request_id", "")) != str(request_id):
        raise GovernedReleaseError("release request full request_id mismatch")
    if request.get("release_root_sha256") != _release_root_sha256(
        release,
        create=False,
    ):
        raise GovernedReleaseError("release request release root mismatch")
    if (
        require_assembled_registration
        and request.get("registration_contract") != "assembled_candidate_seal.v1"
    ):
        raise GovernedReleaseError("release request was not derived from the assembled candidate seal")
    bundle = verify_candidate_bundle(release, run_id=run_id)
    if bundle.manifest_sha256 != request.get("candidate_bundle_manifest_sha256"):
        raise GovernedReleaseError("release request candidate bundle hash mismatch")
    _artifact_paths(request, evidence_root=evidence)
    return release, workflow, evidence, request


def _capture_artifacts(
    evidence_root: Path,
    artifacts: Mapping[str, str | Path],
) -> dict[str, object]:
    missing = sorted(set(REQUIRED_ARTIFACT_ROLES).difference(artifacts))
    unknown = sorted(
        set(artifacts).difference((*REQUIRED_ARTIFACT_ROLES, *OPTIONAL_ARTIFACT_ROLES))
    )
    if missing:
        raise GovernedReleaseError(f"release request missing artifacts: {missing}")
    if unknown:
        raise GovernedReleaseError(f"release request has unknown artifacts: {unknown}")
    captured: dict[str, object] = {}
    for role, selected in sorted(artifacts.items()):
        path, relative = _resolve_regular_evidence_file(
            evidence_root,
            selected,
            role=str(role),
        )
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise GovernedReleaseError(f"cannot capture release artifact: {role}") from exc
        captured[role] = {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    return captured


def _artifact_paths(
    request: Mapping[str, object],
    *,
    evidence_root: Path,
) -> dict[str, Path]:
    raw = request.get("artifacts")
    if not isinstance(raw, Mapping):
        raise GovernedReleaseError("release request artifacts are invalid")
    paths: dict[str, Path] = {}
    for role in REQUIRED_ARTIFACT_ROLES:
        if role not in raw:
            raise GovernedReleaseError(f"release request artifact is missing: {role}")
    for role, entry in raw.items():
        if not isinstance(entry, Mapping):
            raise GovernedReleaseError(f"release request artifact is invalid: {role}")
        path, _ = _resolve_regular_evidence_file(
            evidence_root,
            str(entry.get("path", "")),
            role=str(role),
        )
        if path.stat().st_size != int(entry.get("size_bytes", -1)):
            raise GovernedReleaseError(f"release request artifact size mismatch: {role}")
        if _sha256_file(path) != str(entry.get("sha256", "")):
            raise GovernedReleaseError(f"release request artifact hash mismatch: {role}")
        paths[str(role)] = path
    return paths


def _derive_audit_result(
    request: Mapping[str, object],
    *,
    receipt_path: Path,
    registered_request_path: Path,
    allow_historical_key: bool = False,
) -> dict[str, object]:
    try:
        receipt_document = _load_json(receipt_path)
        selected_public_key = _matching_trusted_public_key(
            receipt_document,
            primary_path=_required_env_key_path(RECEIPT_TRUSTED_KEY_ENV),
            keyring_directory_env=RECEIPT_TRUSTED_PUBLIC_KEY_DIR_ENV,
            label="promotion receipt",
            allow_historical=allow_historical_key,
        )
        receipt = verify_promotion_receipt_signature(
            receipt_document,
            trusted_public_key_path=selected_public_key,
        )
        approved = validate_promotion_receipt_decision(receipt)
        validate_required_gate_set(
            receipt,
            require_pass=approved,
            allow_historical_policy=allow_historical_key,
        )
    except (PromotionError, ReceiptAuthenticationError) as exc:
        raise GovernedReleaseError(f"promotion receipt authentication failed: {exc}") from exc
    if str(receipt.get("run_id", "")) != str(request.get("run_id", "")):
        raise GovernedReleaseError("promotion receipt run_id mismatch")
    if receipt.get("authorization_mode") != "REGISTERED_PROMOTION":
        raise GovernedReleaseError("promotion receipt is diagnostic-only")
    if receipt.get("candidate_bundle_manifest_sha256") != request.get(
        "candidate_bundle_manifest_sha256"
    ):
        raise GovernedReleaseError("promotion receipt candidate hash mismatch")
    if str(receipt.get("release_request_id", "")) != str(request.get("request_id", "")):
        raise GovernedReleaseError("promotion receipt release_request_id mismatch")
    expected_current = request.get("expected_current")
    if not isinstance(expected_current, Mapping):
        raise GovernedReleaseError("release request expected_current is invalid")
    expected_event_id = (
        str(expected_current["event_id"])
        if expected_current.get("kind") == "EVENT"
        else None
    )
    operation_bindings = {
        "release_operation_id": request.get("request_id"),
        "expected_current_event_id": expected_event_id,
        "operation_created_at_utc": request.get("created_at_utc"),
        "release_root_sha256": request.get("release_root_sha256"),
    }
    if any(receipt.get(key) != value for key, value in operation_bindings.items()):
        raise GovernedReleaseError("promotion receipt operation binding mismatch")
    if str(receipt.get("evidence_inventory_sha256", "")) != str(
        request.get("evidence_inventory_sha256", "")
    ):
        raise GovernedReleaseError("promotion receipt evidence inventory mismatch")
    if receipt.get("workflow_domain_sha256") != request.get(
        "workflow_domain_sha256"
    ):
        raise GovernedReleaseError("promotion receipt workflow domain mismatch")
    try:
        request_document_sha256 = release_request_document_sha256(
            registered_request_path
        )
    except ReleaseRequestAuthenticationError as exc:
        raise GovernedReleaseError(f"release request hash failed: {exc}") from exc
    if receipt.get("release_request_document_sha256") != request_document_sha256:
        raise GovernedReleaseError("promotion receipt REGISTER document hash mismatch")
    receipt_artifacts = receipt.get("input_artifacts")
    request_artifacts = request.get("artifacts")
    if not isinstance(receipt_artifacts, Mapping) or not isinstance(
        request_artifacts, Mapping
    ):
        raise GovernedReleaseError("promotion receipt evidence artifacts are missing")
    for role, expected in request_artifacts.items():
        actual = receipt_artifacts.get(role)
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            raise GovernedReleaseError(
                f"promotion receipt evidence artifact is missing: {role}"
            )
        if str(actual.get("sha256", "")) != str(expected.get("sha256", "")):
            raise GovernedReleaseError(
                f"promotion receipt evidence artifact hash mismatch: {role}"
            )
    if receipt.get("schema_version") != "promotion_receipt.v2":
        raise GovernedReleaseError("promotion receipt schema mismatch")
    return {
        "schema_version": AUDIT_RESULT_SCHEMA,
        "request_id": request["request_id"],
        "run_id": request["run_id"],
        "status": "APPROVED" if approved else "REJECTED",
        "approved": approved,
        "promotion_receipt_sha256": _sha256_file(receipt_path),
        "receipt_id": receipt.get("receipt_id"),
        "promotion_evaluated_at": receipt.get("promotion_evaluated_at"),
    }


def _verify_request_document(
    request: Mapping[str, object],
    *,
    expected_run_id: str,
) -> None:
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise GovernedReleaseError("release request schema mismatch")
    if str(request.get("run_id", "")) != expected_run_id:
        raise GovernedReleaseError("release request run_id mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", str(request.get("release_root_sha256", ""))):
        raise GovernedReleaseError("release request release_root_sha256 is invalid")
    artifacts = request.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise GovernedReleaseError("release request artifacts are invalid")
    if str(request.get("evidence_inventory_sha256", "")) != _sha256_json(artifacts):
        raise GovernedReleaseError("release request evidence inventory hash mismatch")
    expected_current = request.get("expected_current")
    if not isinstance(expected_current, Mapping):
        raise GovernedReleaseError("release request expected_current is invalid")
    kind = expected_current.get("kind")
    if kind == "NONE":
        if set(expected_current) != {"kind"}:
            raise GovernedReleaseError("release request expect-none contract is invalid")
    elif kind == "EVENT":
        if set(expected_current) != {"kind", "event_id"} or not re.fullmatch(
            r"[0-9a-f]{64}", str(expected_current.get("event_id", ""))
        ):
            raise GovernedReleaseError("release request expected event_id is invalid")
    else:
        raise GovernedReleaseError("release request expected_current kind is invalid")
    if "request_nonce" in request:
        _request_nonce(str(request["request_nonce"]))
    unsigned = dict(request)
    unsigned.pop("register_signature", None)
    request_id = str(unsigned.pop("request_id", ""))
    if request_id != _sha256_json(unsigned):
        raise GovernedReleaseError("release request_id mismatch")


def _expected_current_contract(event_id: str | None) -> dict[str, str]:
    if event_id is None:
        return {"kind": "NONE"}
    canonical = str(event_id).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", canonical):
        raise GovernedReleaseError("expected current event_id is invalid")
    return {"kind": "EVENT", "event_id": canonical}


def _request_nonce(value: str) -> str:
    selected = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", selected):
        raise GovernedReleaseError("request_nonce is invalid")
    return selected


def _resolve_regular_evidence_file(
    root: Path,
    selected: str | Path,
    *,
    role: str,
) -> tuple[Path, str]:
    raw = Path(selected).expanduser()
    lexical = raw if raw.is_absolute() else root / raw
    lexical = Path(os.path.abspath(lexical))
    try:
        relative_path = lexical.relative_to(root)
    except ValueError as exc:
        raise GovernedReleaseError(f"{role} escapes evidence_root: {lexical}") from exc
    cursor = root
    for part in relative_path.parts:
        cursor = cursor / part
        if path_is_link(cursor):
            raise GovernedReleaseError(f"{role} traverses a forbidden link: {cursor}")
    resolved = lexical.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise GovernedReleaseError(f"{role} escapes evidence_root: {resolved}") from exc
    if not resolved.is_file():
        raise GovernedReleaseError(f"{role} is not a regular evidence file: {resolved}")
    return resolved, relative


def _request_path(
    workflow_root: Path,
    run_id: str,
    *,
    request_id: str | None = None,
    read_only: bool = False,
) -> Path:
    if request_id is not None:
        return _request_dir(
            workflow_root,
            run_id,
            request_id,
            create=not read_only,
        ) / "release_request.json"
    requests = _run_workflow_path(workflow_root, run_id) / "requests"
    if not requests.is_dir():
        raise GovernedReleaseError("release request is unavailable")
    matches = sorted(requests.glob("*/release_request.json"))
    if len(matches) != 1:
        raise GovernedReleaseError(
            "request_id is required when the run does not have exactly one request"
        )
    return matches[0]


def _receipt_path(
    workflow_root: Path,
    run_id: str,
    *,
    request_id: str,
    read_only: bool = False,
) -> Path:
    return _audit_result_dir(
        workflow_root,
        run_id,
        request_id,
        create=not read_only,
    ) / "promotion_receipt.json"


def _audit_result_dir(
    workflow_root: Path,
    run_id: str,
    request_id: str,
    *,
    create: bool = True,
) -> Path:
    return _phase_result_dir(
        workflow_root,
        run_id,
        request_id,
        phase="audit-results",
        create=create,
    )


def _promotion_result_path(
    workflow_root: Path,
    run_id: str,
    request_id: str,
    *,
    read_only: bool = False,
) -> Path:
    return _phase_result_dir(
        workflow_root,
        run_id,
        request_id,
        phase="promotion-results",
        create=not read_only,
    ) / "promotion_result.json"


def _phase_result_dir(
    workflow_root: Path,
    run_id: str,
    request_id: str,
    *,
    phase: str,
    create: bool,
) -> Path:
    if phase not in {"audit-results", "promotion-results"}:
        raise GovernedReleaseError("workflow result phase is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", str(request_id)):
        raise GovernedReleaseError("request_id is invalid")
    path = (
        _run_workflow_path(workflow_root, run_id)
        / phase
        / str(request_id)[:32]
    ).resolve()
    try:
        path.relative_to(workflow_root)
    except ValueError as exc:
        raise GovernedReleaseError("workflow result directory escapes root") from exc
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _request_dir(
    workflow_root: Path,
    run_id: str,
    request_id: str,
    *,
    create: bool = True,
) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", str(request_id)):
        raise GovernedReleaseError("request_id is invalid")
    run_dir = _run_workflow_path(workflow_root, run_id)
    path = (run_dir / "requests" / str(request_id)[:32]).resolve()
    if os.name == "nt":
        longest_phase_temp = path / (
            ".promotion_receipt.json." + ("9" * 10) + "." + ("f" * 32) + ".tmp"
        )
        if len(str(longest_phase_temp)) >= 260:
            raise GovernedReleaseError(
                "request workflow path exceeds the Windows MAX_PATH safety budget"
            )
    try:
        path.relative_to(workflow_root)
    except ValueError as exc:
        raise GovernedReleaseError("request workflow directory escapes root") from exc
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _run_workflow_dir(workflow_root: Path, run_id: str) -> Path:
    path = _run_workflow_path(workflow_root, run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_workflow_path(workflow_root: Path, run_id: str) -> Path:
    if not run_id or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-" for char in run_id):
        raise GovernedReleaseError("run_id is invalid")
    path = (workflow_root / run_id).resolve()
    try:
        path.relative_to(workflow_root)
    except ValueError as exc:
        raise GovernedReleaseError("run workflow directory escapes root") from exc
    return path


def _external_workflow_root(value: str | Path, *, release_root: Path) -> Path:
    root = _external_workflow_path(value, release_root=release_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _external_workflow_path(value: str | Path, *, release_root: Path) -> Path:
    root = _absolute_path(value, name="workflow_root")
    if root == release_root or root in release_root.parents or release_root in root.parents:
        raise GovernedReleaseError("workflow_root must be external to release_root")
    return root


def _absolute_root(value: str | Path, *, name: str, must_exist: bool) -> Path:
    root = _absolute_path(value, name=name)
    if must_exist and not root.is_dir():
        raise GovernedReleaseError(f"{name} is unavailable: {root}")
    if not must_exist:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _absolute_path(value: str | Path, *, name: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise GovernedReleaseError(f"{name} must be absolute")
    return candidate.resolve()


def _authority_private_key_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise GovernedReleaseError("receipt signing private key path must be absolute")
    lexical = Path(os.path.abspath(path))
    for component in (lexical, *lexical.parents):
        try:
            is_link = path_is_link(component)
        except OSError as exc:
            raise GovernedReleaseError(
                "cannot validate receipt signing private key path"
            ) from exc
        if is_link:
            raise GovernedReleaseError(
                "receipt signing private key path cannot contain a link"
            )
    if not lexical.is_file():
        raise GovernedReleaseError(
            f"receipt signing private key is unavailable: {lexical}"
        )
    try:
        link_count = int(lexical.stat().st_nlink)
    except (AttributeError, OSError) as exc:
        raise GovernedReleaseError(
            "cannot validate receipt signing private key file"
        ) from exc
    if link_count != 1:
        raise GovernedReleaseError(
            "receipt signing private key must have exactly one filesystem link"
        )
    return lexical


def _assert_register_signing_authority(private_key_path: str | Path) -> None:
    try:
        trusted = _required_env_key_path(REGISTER_TRUSTED_PUBLIC_KEY_ENV)
        signing_id = public_key_id_from_private_key_path(private_key_path)
        if public_key_id_from_public_key_path(trusted) != signing_id:
            raise GovernedReleaseError(
                "REGISTER private key does not match its active trusted public key"
            )
        receipt_selected = str(os.environ.get(RECEIPT_TRUSTED_KEY_ENV, "")).strip()
        if receipt_selected and signing_id in _trusted_public_key_identities(
            _required_env_key_path(RECEIPT_TRUSTED_KEY_ENV),
            keyring_directory_env=RECEIPT_TRUSTED_PUBLIC_KEY_DIR_ENV,
            label="promotion receipt",
        ):
            raise GovernedReleaseError("REGISTER and AUDIT authority keyrings overlap")
    except (PromotionError, ReceiptAuthenticationError) as exc:
        raise GovernedReleaseError(f"REGISTER authority validation failed: {exc}") from exc


def _assert_audit_signing_authority(private_key_path: str | Path) -> None:
    try:
        signing_id = public_key_id_from_private_key_path(private_key_path)
        if public_key_id_from_public_key_path(
            _required_env_key_path(RECEIPT_TRUSTED_KEY_ENV)
        ) != signing_id:
            raise GovernedReleaseError(
                "AUDIT private key does not match its active trusted public key"
            )
        register_ids = _trusted_public_key_identities(
            _required_env_key_path(REGISTER_TRUSTED_PUBLIC_KEY_ENV),
            keyring_directory_env=REGISTER_TRUSTED_PUBLIC_KEY_DIR_ENV,
            label="REGISTER request",
        )
        if signing_id in register_ids:
            raise GovernedReleaseError("AUDIT and REGISTER authority keyrings overlap")
    except (PromotionError, ReceiptAuthenticationError) as exc:
        raise GovernedReleaseError(f"AUDIT authority validation failed: {exc}") from exc


def _absolute_regular_input_path(value: str | Path, *, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise GovernedReleaseError(f"{name} must be absolute")
    lexical = Path(os.path.abspath(path))
    for component in (lexical, *lexical.parents):
        if path_is_link(component):
            raise GovernedReleaseError(f"{name} cannot contain a link")
    if not lexical.is_file():
        raise GovernedReleaseError(f"{name} is unavailable: {lexical}")
    return lexical


def _record_failure(
    workflow_root: Path,
    *,
    run_id: str,
    phase: str,
    request: Mapping[str, object],
    exc: Exception,
) -> None:
    failure = {
        "schema_version": "governed_lt_failure.v1",
        "request_id": request.get("request_id"),
        "run_id": run_id,
        "phase": phase,
        "failed_at_utc": _now_utc(),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    path = _run_workflow_dir(workflow_root, run_id) / "failures" / (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex}.json"
    )
    _write_json_exclusive(path, failure)


def _record_failure_best_effort(
    workflow_root: Path,
    *,
    run_id: str,
    phase: str,
    request: Mapping[str, object],
    exc: Exception,
) -> None:
    try:
        _record_failure(
            workflow_root,
            run_id=run_id,
            phase=phase,
            request=request,
            exc=exc,
        )
    except Exception:
        return


def _write_result_idempotent(
    path: Path,
    payload: Mapping[str, object],
) -> dict[str, object]:
    if path.exists():
        existing = _load_json(path)
        comparable = {key: value for key, value in payload.items() if key != "recorded_at_utc"}
        existing_comparable = {
            key: value for key, value in existing.items() if key != "recorded_at_utc"
        }
        if existing_comparable != comparable:
            raise GovernedReleaseError(f"existing result differs: {path}")
        return existing
    _write_json_exclusive(path, payload)
    return dict(payload)


def _run_capstone(argv: Sequence[str]) -> int:
    from pfc_shaping.calibration.monthly_curve_capstone import main

    return int(main(list(argv)))


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    _write_bytes_exclusive(path, raw)


def _write_bytes_idempotent(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise GovernedReleaseError(f"evidence capture collision: {path}")
        return
    temporary = path.with_name(candidate_temporary_name(suffix="tmp"))
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise GovernedReleaseError(f"evidence capture collision: {path}")
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    temporary = path.with_name(candidate_temporary_name(suffix="tmp"))
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise GovernedReleaseError(
                f"immutable workflow artifact already exists: {path}"
            ) from exc
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _load_json(path: Path) -> dict[str, object]:
    try:
        raw = path.read_text(encoding="utf-8")
        payload = load_strict_json(raw)
    except (OSError, UnicodeError, StrictStructuredDataError, RecursionError) as exc:
        raise GovernedReleaseError(f"cannot read workflow JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise GovernedReleaseError(f"workflow JSON must be an object: {path}")
    return payload


def _assert_phase_identity(command: str) -> None:
    try:
        if command != "status":
            assert_installed_runtime_sealed()
        assert_phase_private_key_scope(command)
    except ReleaseCliIdentityError as exc:
        raise GovernedReleaseError(str(exc)) from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, object]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _release_root_sha256(release_root: Path, *, create: bool = True) -> str:
    return _release_root_id(release_root, create=create)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_utc_seconds(value: str) -> str:
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise GovernedReleaseError("request timestamp is invalid") from exc
    if timestamp.tzinfo is None:
        raise GovernedReleaseError("request timestamp is timezone-naive")
    return timestamp.astimezone(timezone.utc).replace(microsecond=0).isoformat()

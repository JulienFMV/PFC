"""Immutable candidate bundles and atomic production-pointer promotion."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Mapping

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    paths_overlap_by_identity,
    process_immutable_directory_entries,
    process_immutable_file_bytes,
)
from pfc_shaping.path_safety import (
    path_is_link as _path_is_link,
)
from pfc_shaping.pipeline.candidate_evidence import (
    verify_assembled_candidate_evidence,
    verify_candidate_evidence,
)
from pfc_shaping.pipeline.governed_release_cli_contract import (
    ReleaseCliIdentityError,
    assert_production_transition_runtime_authorized,
)
from pfc_shaping.pipeline.process_identity import process_start_token
from pfc_shaping.pipeline.promotion_contract import (
    EventAuthenticationError,
    ReceiptAuthenticationError,
    RollbackAuthorizationAuthenticationError,
    public_key_id_from_private_key_path,
    public_key_id_from_public_key_path,
    sign_promotion_event,
    sign_promotion_head,
    validate_promotion_receipt_decision,
    validate_required_gate_set,
    verify_promotion_event_signature,
    verify_promotion_head_signature,
    verify_promotion_receipt_signature,
    verify_rollback_authorization_signature,
)
from pfc_shaping.pipeline.release_request_contract import (
    REGISTER_TRUSTED_PUBLIC_KEY_DIR_ENV,
    REGISTER_TRUSTED_PUBLIC_KEY_ENV,
    ReleaseRequestAuthenticationError,
    load_registered_release_request_evidence,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
BUNDLE_MANIFEST = "candidate_bundle_manifest.json"
CURRENT_POINTER = "current.json"
LOCK_FILE = ".promotion.lock"
RELEASE_DOMAIN_MARKER = "release_domain.json"
CANDIDATE_RUN_MANIFEST = Path("manifests") / "candidate_run_manifest.json"
PROMOTION_EVENTS_DIR = "promotion_events"
PROMOTION_RECEIPTS_DIR = "promotion_receipts"
ROLLBACK_AUTHORIZATIONS_DIR = "rollback_authorizations"
MAX_NEW_RECEIPT_AGE = 30 * 60
MAX_CLOCK_SKEW = 60
EVENT_SIGNING_PRIVATE_KEY_ENV = "PFC_PROMOTION_EVENT_SIGNING_PRIVATE_KEY_PATH"
EVENT_TRUSTED_PUBLIC_KEY_ENV = "PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_PATH"
EVENT_TRUSTED_PUBLIC_KEY_DIR_ENV = "PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_DIR"
RELEASE_DOMAIN_ID_ENV = "PFC_PROMOTION_RELEASE_DOMAIN_ID"
RECEIPT_TRUSTED_PUBLIC_KEY_DIR_ENV = "PFC_PROMOTION_TRUSTED_PUBLIC_KEY_DIR"
PROMOTION_JOURNAL_ROOT_ENV = "PFC_PROMOTION_JOURNAL_ROOT"
PROMOTION_LOCK_WAIT_SECONDS_ENV = "PFC_PROMOTION_LOCK_WAIT_SECONDS"
ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_ENV = (
    "PFC_ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_PATH"
)
ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_DIR_ENV = (
    "PFC_ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_DIR"
)
FORBIDDEN_TRANSITION_PRIVATE_KEY_ENVS = (
    "PFC_PROMOTION_SIGNING_PRIVATE_KEY_PATH",
    "PFC_ROLLBACK_AUTHORIZATION_SIGNING_PRIVATE_KEY_PATH",
    "PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH",
    "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH",
    "PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH",
    "PFC_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH",
)
_CANDIDATE_TEMP_PATTERNS = (
    re.compile(
        r"^\.pfc-(?P<pid>[1-9][0-9]*)-(?P<start>[0-9a-f]{16,32}|unknown)-"
        r"[0-9a-f]{32}\.tmp$"
    ),
    re.compile(
        r"^\.pfc-(?P<pid>[1-9][0-9]*)-(?P<start>[0-9a-f]{16,32}|unknown)-"
        r"[0-9a-f]{32}\.pending\.json$"
    ),
)
_EXCLUSIVE_WRITE_TEMP_PATTERN = re.compile(
    r"^\.pfc-(?:"
    r"v2-(?P<host>[0-9a-f]{12})-(?P<pid>[1-9][0-9]*)-"
    r"(?P<start>[0-9a-f]{16,32}|unknown)-(?P<nonce>[0-9a-f]{12})"
    r"|(?P<legacy_pid>[1-9][0-9]*)-(?P<legacy_nonce>[0-9a-f]{12})"
    r")\.tmp$"
)


class PromotionError(RuntimeError):
    """Raised when staging, verification, promotion or rollback fails closed."""


class PromotionConflictError(PromotionError):
    """Raised when a promotion request no longer matches committed history."""


class PromotionBusyError(PromotionError):
    """Raised when another live transition holds the lock past the wait budget."""


class ImmutableArtifactExistsError(PromotionError):
    """Raised only when an exclusive immutable destination already exists."""


class PromotionProjectionRepairRequired(PromotionError):
    """Raised when a commit is durable or indeterminate and needs reconciliation."""

    def __init__(
        self,
        event_id: str | None,
        *,
        commit_status: str = "COMMITTED",
    ) -> None:
        if commit_status not in {"COMMITTED", "INDETERMINATE"}:
            raise ValueError("invalid repair-required commit status")
        self.event_id = event_id
        self.commit_status = commit_status
        transition = event_id or "unknown-event"
        qualifier = "is committed" if commit_status == "COMMITTED" else "may be committed"
        super().__init__(
            f"transition {transition} {qualifier}; reconciliation and projection repair required"
        )


@dataclass(frozen=True)
class CandidateBundle:
    run_id: str
    path: Path
    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True)
class CurrentRelease:
    bundle: CandidateBundle
    pointer: Mapping[str, object]
    event: Mapping[str, object]
    receipt: Mapping[str, object]
    projection_repair_required: bool = False


@dataclass
class _PromotionLockState:
    expected_journal_event_id: str | None = None

    def bind_journal_event(self, event_id: object) -> None:
        value = str(event_id)
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise PromotionError("lock journal event ID is invalid")
        if self.expected_journal_event_id not in {None, value}:
            raise PromotionError("lock cannot bind more than one journal event")
        self.expected_journal_event_id = value


def create_candidate_staging(release_root: str | Path, *, run_id: str) -> Path:
    root = _release_root(release_root)
    _validate_run_id(run_id)
    candidates = _candidates_root(root, create=True)
    staging = candidates / f".{run_id}.staging"
    final = candidates / run_id
    if staging.exists() or final.exists():
        raise PromotionError(f"candidate run_id already exists: {run_id}")
    staging.mkdir()
    return staging


def validate_release_run_id(run_id: str) -> str:
    """Validate a release identifier before callers derive any filesystem path."""

    _validate_run_id(run_id)
    return str(run_id)


def finalize_candidate_staging(
    release_root: str | Path,
    *,
    run_id: str,
    staging_path: str | Path,
    metadata: Mapping[str, object] | None = None,
) -> CandidateBundle:
    """Finalize a generic candidate without allowing reserved contract claims."""

    bundle_metadata = dict(metadata or {})
    if "evidence_contract" in bundle_metadata:
        raise PromotionError("evidence_contract metadata is reserved")
    return _finalize_candidate_staging(
        release_root,
        run_id=run_id,
        staging_path=staging_path,
        metadata=bundle_metadata,
    )


def _finalize_candidate_staging(
    release_root: str | Path,
    *,
    run_id: str,
    staging_path: str | Path,
    metadata: Mapping[str, object],
) -> CandidateBundle:
    root = _release_root(release_root)
    _validate_run_id(run_id)
    candidates = _candidates_root(root, create=False)
    staging = _resolve_within(candidates, staging_path)
    expected_staging = (candidates / f".{run_id}.staging").resolve()
    if staging != expected_staging or not staging.is_dir():
        raise PromotionError(f"invalid candidate staging path: {staging}")
    _validate_candidate_run_manifest(staging, expected_run_id=run_id)
    try:
        verify_candidate_evidence(staging, expected_run_id=run_id)
    except RuntimeError as exc:
        raise PromotionError("candidate evidence is incomplete or invalid") from exc
    files = _hash_bundle_files(staging)
    if not files:
        raise PromotionError("candidate bundle is empty")
    candidate_finalized_at_utc = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, object] = {
        "schema_version": "candidate_bundle.v1",
        "run_id": run_id,
        "created_at_utc": candidate_finalized_at_utc,
        "candidate_finalized_at_utc": candidate_finalized_at_utc,
        "files": files,
        "metadata": dict(metadata),
    }
    manifest["content_sha256"] = _sha256_json(
        {"run_id": run_id, "files": files, "metadata": manifest["metadata"]}
    )
    manifest_path = staging / BUNDLE_MANIFEST
    _write_json_fsync(manifest_path, manifest)
    _fsync_tree(staging)
    final = candidates / run_id
    _replace_directory(staging, final)
    _fsync_directory(final.parent)
    final_manifest = final / BUNDLE_MANIFEST
    try:
        verified = verify_candidate_bundle_manifest(final_manifest)
    except Exception as exc:
        quarantine = final.with_name(f".{run_id}.failed-{time.time_ns()}")
        try:
            os.replace(final, quarantine)
            _fsync_directory(final.parent)
        except OSError as quarantine_exc:
            raise PromotionError(
                "candidate failed verification after finalization and could not be quarantined"
            ) from quarantine_exc
        raise PromotionError(
            f"candidate failed verification after finalization; quarantined at {quarantine}"
        ) from exc
    if verified.run_id != run_id or verified.path != final.resolve():
        raise PromotionError("finalized candidate identity changed during verification")
    return verified


def finalize_assembled_candidate_staging(
    release_root: str | Path,
    *,
    run_id: str,
    metadata: Mapping[str, object] | None = None,
) -> CandidateBundle:
    """Finalize the canonical staged LT candidate under the release lock."""

    root = _release_root(release_root)
    _validate_run_id(run_id)
    bundle_metadata = dict(metadata or {})
    if "evidence_contract" in bundle_metadata:
        raise PromotionError("evidence_contract metadata is reserved")
    with _promotion_lock(root):
        candidates = _candidates_root(root, create=False)
        staging = candidates / f".{run_id}.staging"
        final = candidates / run_id
        if final.is_dir():
            if staging.exists():
                raise PromotionError("candidate has both staging and final directories")
            bundle = verify_candidate_bundle(root, run_id=run_id)
            manifest = _load_json(bundle.manifest_path)
            metadata = manifest.get("metadata")
            if (
                not isinstance(metadata, Mapping)
                or metadata.get("evidence_contract") != "assembled_lt_candidate.v1"
            ):
                raise PromotionError("existing candidate is not an assembled LT candidate")
            try:
                verify_assembled_candidate_evidence(bundle.path, expected_run_id=run_id)
            except RuntimeError as exc:
                raise PromotionError("existing assembled candidate replay failed") from exc
            return bundle
        _recover_abandoned_candidate_temporaries(staging)
        try:
            verify_assembled_candidate_evidence(staging, expected_run_id=run_id)
        except RuntimeError as exc:
            raise PromotionError("assembled candidate evidence is incomplete or invalid") from exc
        bundle = _finalize_candidate_staging(
            root,
            run_id=run_id,
            staging_path=staging,
            metadata={
                **bundle_metadata,
                "evidence_contract": "assembled_lt_candidate.v1",
            },
        )
        try:
            verify_assembled_candidate_evidence(bundle.path, expected_run_id=run_id)
        except RuntimeError as exc:
            quarantine = bundle.path.with_name(f".{run_id}.failed-{time.time_ns()}")
            try:
                os.replace(bundle.path, quarantine)
            except OSError as quarantine_exc:
                raise PromotionError(
                    "final assembled candidate replay failed and quarantine failed"
                ) from quarantine_exc
            raise PromotionError(
                f"final assembled candidate replay failed; quarantined at {quarantine}"
            ) from exc
        return bundle


def verify_candidate_bundle(release_root: str | Path, *, run_id: str) -> CandidateBundle:
    root = _release_root(release_root, create=False)
    _validate_run_id(run_id)
    candidates = _candidates_root(root, create=False)
    bundle = _resolve_within(candidates, candidates / run_id)
    manifest_path = bundle / BUNDLE_MANIFEST
    verified = verify_candidate_bundle_manifest(manifest_path)
    if verified.run_id != run_id or verified.path != bundle:
        raise PromotionError("candidate bundle path/run_id mismatch")
    return verified


def verify_candidate_bundle_manifest(manifest_path: str | Path) -> CandidateBundle:
    """Verify a finalized bundle directly from its root manifest path."""

    manifest_path = Path(manifest_path).resolve()
    bundle = manifest_path.parent
    if not manifest_path.is_file():
        raise PromotionError(f"candidate bundle manifest is missing: {manifest_path}")
    manifest = _load_json(manifest_path)
    run_id = manifest.get("run_id")
    if type(run_id) is not str:
        raise PromotionError("candidate bundle manifest run_id is invalid")
    _validate_run_id(run_id)
    if manifest.get("schema_version") != "candidate_bundle.v1":
        raise PromotionError("candidate bundle manifest schema mismatch")
    _validate_candidate_run_manifest(bundle, expected_run_id=run_id)
    try:
        verify_candidate_evidence(bundle, expected_run_id=run_id)
    except RuntimeError as exc:
        raise PromotionError("candidate evidence is incomplete or invalid") from exc
    expected_files = manifest.get("files")
    if not isinstance(expected_files, Mapping):
        raise PromotionError("candidate bundle manifest files are invalid")
    actual_files = _hash_bundle_files(bundle)
    if dict(expected_files) != actual_files:
        raise PromotionError("candidate bundle file set/hash mismatch")
    expected_content = _sha256_json(
        {"run_id": run_id, "files": actual_files, "metadata": manifest.get("metadata", {})}
    )
    if manifest.get("content_sha256") != expected_content:
        raise PromotionError("candidate bundle content hash mismatch")
    return CandidateBundle(run_id, bundle, manifest_path, _sha256_file(manifest_path))


def _assert_sealed_transition_runtime() -> None:
    try:
        assert_production_transition_runtime_authorized()
    except ReleaseCliIdentityError as exc:
        raise PromotionError(str(exc)) from exc


def promote_candidate(
    release_root: str | Path,
    *,
    run_id: str,
    promotion_receipt_path: str | Path,
    expected_current_event_id: str | None,
    operation_id: str,
    release_request_id: str,
    operation_created_at_utc: str,
    registered_release_request_path: str | Path | None = None,
    workflow_root: str | Path | None = None,
    require_assembled_evidence: bool = True,
) -> dict[str, object]:
    if require_assembled_evidence is not True:
        raise PromotionError(
            "promotion without assembled candidate evidence is permanently disabled"
        )
    _assert_sealed_transition_runtime()
    root = _release_root(release_root)
    _assert_transition_authority_separation(require_rollback_authority=False)
    _validate_expected_event_id(expected_current_event_id)
    operation_id = _required_operation_value(operation_id, label="operation_id")
    release_request_id = _required_operation_value(
        release_request_id,
        label="release_request_id",
    )
    operation_created_at_utc = _canonical_operation_timestamp(operation_created_at_utc)
    if (
        _aware_datetime(operation_created_at_utc, label="operation_created_at_utc")
        - _promotion_now_utc()
    ).total_seconds() > MAX_CLOCK_SKEW:
        raise PromotionError("operation_created_at_utc is in the future")
    receipt_path = Path(promotion_receipt_path).resolve()
    if registered_release_request_path is None or workflow_root is None:
        raise PromotionError(
            "promotion requires the signed REGISTER request and governed workflow root"
        )
    try:
        release_request, release_request_sha256 = load_registered_release_request_evidence(
            registered_release_request_path,
            workflow_root=workflow_root,
            expected_run_id=run_id,
            allow_historical_key=True,
        )
    except ReleaseRequestAuthenticationError as exc:
        raise PromotionError(f"REGISTER authority replay failed: {exc}") from exc
    _validate_registered_release_request_binding(
        release_request,
        root=root,
        run_id=run_id,
        expected_current_event_id=expected_current_event_id,
        operation_id=operation_id,
        release_request_id=release_request_id,
        operation_created_at_utc=operation_created_at_utc,
    )
    key_path = _trusted_public_key_path()
    with _promotion_lock(root) as lock_state:
        history, chain = _load_authoritative_journal(root)
        _validate_journal_projections(
            root,
            history=history,
            chain=chain,
            trusted_public_key_path=key_path,
        )
        current_event = chain[-1] if chain else None
        bundle = verify_candidate_bundle(root, run_id=run_id)
        _validate_registered_release_request_candidate_binding(
            release_request,
            candidate_bundle_manifest_sha256=bundle.manifest_sha256,
        )
        try:
            verify_assembled_candidate_evidence(bundle.path, expected_run_id=run_id)
        except RuntimeError as exc:
            raise PromotionError("assembled candidate replay failed at promotion") from exc
        if _is_exact_promotion_retry(
            current_event,
            run_id=run_id,
            candidate_bundle_manifest_sha256=bundle.manifest_sha256,
            expected_current_event_id=expected_current_event_id,
            operation_id=operation_id,
            release_request_id=release_request_id,
            operation_created_at_utc=operation_created_at_utc,
        ):
            try:
                retry_receipt_bytes = receipt_path.read_bytes()
            except OSError as exc:
                raise PromotionError(
                    f"cannot read promotion receipt: {receipt_path}"
                ) from exc
            retry_receipt = _verify_authorized_receipt(
                bundle,
                receipt_bytes=retry_receipt_bytes,
                receipt_path=receipt_path,
                trusted_public_key_path=key_path,
                allow_historical_key=True,
            )
            _validate_receipt_operation_binding(
                retry_receipt,
                root=root,
                expected_current_event_id=expected_current_event_id,
                operation_id=operation_id,
                release_request_id=release_request_id,
                operation_created_at_utc=operation_created_at_utc,
                workflow_domain_sha256=str(
                    release_request["workflow_domain_sha256"]
                ),
                release_request_document_sha256=release_request_sha256,
            )
            if (
                str(current_event.get("receipt_id", ""))
                != str(retry_receipt.get("receipt_id", ""))
                or str(current_event.get("promotion_receipt_sha256", ""))
                != hashlib.sha256(retry_receipt_bytes).hexdigest()
            ):
                raise PromotionConflictError(
                    "promotion operation is already committed with a different receipt"
                )
            current = _current_release_from_event(
                root,
                event=current_event,
                trusted_public_key_path=key_path,
            )
            lock_state.bind_journal_event(current_event["event_id"])
            try:
                _repair_journal_projections(root, head=history[-1], pointer=current.pointer)
            except BaseException as exc:
                raise PromotionProjectionRepairRequired(
                    str(current_event["event_id"]),
                    commit_status="COMMITTED",
                ) from exc
            return dict(current.pointer)

        actual_event_id = str(current_event["event_id"]) if current_event else None
        if actual_event_id != expected_current_event_id:
            raise PromotionConflictError(
                "promotion compare-and-swap conflict: "
                f"expected={expected_current_event_id!r}, actual={actual_event_id!r}"
            )
        if any(event.get("operation_id") == operation_id for event in chain):
            raise PromotionConflictError(
                f"promotion operation_id is already committed: {operation_id}"
            )

        bundle = verify_candidate_bundle(root, run_id=run_id)
        _validate_registered_release_request_candidate_binding(
            release_request,
            candidate_bundle_manifest_sha256=bundle.manifest_sha256,
        )
        try:
            verify_assembled_candidate_evidence(bundle.path, expected_run_id=run_id)
        except RuntimeError as exc:
            raise PromotionError("assembled candidate replay failed at promotion") from exc
        try:
            receipt_bytes = receipt_path.read_bytes()
        except OSError as exc:
            raise PromotionError(f"cannot read promotion receipt: {receipt_path}") from exc
        receipt = _verify_authorized_receipt(
            bundle,
            receipt_bytes=receipt_bytes,
            receipt_path=receipt_path,
            trusted_public_key_path=key_path,
            require_recent_evaluation=True,
        )
        _validate_receipt_operation_binding(
            receipt,
            root=root,
            expected_current_event_id=expected_current_event_id,
            operation_id=operation_id,
            release_request_id=release_request_id,
            operation_created_at_utc=operation_created_at_utc,
            workflow_domain_sha256=str(release_request["workflow_domain_sha256"]),
            release_request_document_sha256=release_request_sha256,
        )
        bundle = verify_candidate_bundle(root, run_id=run_id)
        _validate_registered_release_request_candidate_binding(
            release_request,
            candidate_bundle_manifest_sha256=bundle.manifest_sha256,
        )
        try:
            verify_assembled_candidate_evidence(bundle.path, expected_run_id=run_id)
        except RuntimeError as exc:
            raise PromotionError(
                "assembled candidate changed before pointer promotion"
            ) from exc
        if current_event is not None:
            previous_recorded_at = _aware_datetime(
                current_event.get("recorded_at_utc"),
                label="current event recorded_at_utc",
            )
            if _aware_datetime(
                operation_created_at_utc,
                label="operation_created_at_utc",
            ) < previous_recorded_at:
                raise PromotionError("operation_created_at_utc predates committed history")
        archived_receipt_path = _archive_promotion_receipt(
            root,
            receipt=receipt,
            receipt_bytes=receipt_bytes,
        )
        event = _write_promotion_event(
            root,
            {
                "schema_version": "promotion_event.v2",
                "event_type": "PROMOTE",
                "release_root_sha256": _release_root_id(root),
                "run_id": run_id,
                "candidate_bundle_manifest_sha256": bundle.manifest_sha256,
                "candidate_run_manifest_sha256": receipt["candidate_run_manifest_sha256"],
                "promotion_receipt_path": archived_receipt_path.relative_to(root).as_posix(),
                "promotion_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "receipt_id": receipt["receipt_id"],
                "operation_id": operation_id,
                "release_request_id": release_request_id,
                "operation_created_at_utc": operation_created_at_utc,
                "expected_current_event_id": expected_current_event_id,
                "recorded_at_utc": operation_created_at_utc,
                "previous_event_id": actual_event_id,
            },
        )
        lock_state.bind_journal_event(event["event_id"])
        pointer = _pointer_for_event(root, event=event)
        _commit_and_publish_transition(
            root,
            event=event,
            pointer=pointer,
            updated_at_utc=operation_created_at_utc,
            trusted_public_key_path=key_path,
        )
    return dict(pointer)


def rollback_to_candidate(
    release_root: str | Path,
    *,
    rollback_authorization_path: str | Path,
) -> dict[str, object]:
    _assert_sealed_transition_runtime()
    root = _release_root(release_root)
    _assert_transition_authority_separation(require_rollback_authority=True)
    key_path = _trusted_public_key_path()
    authorization_path = _absolute_lexical_path(rollback_authorization_path)
    _assert_path_has_no_links(
        authorization_path,
        label="rollback authorization",
    )
    if not authorization_path.is_file():
        raise PromotionError(
            f"rollback authorization is unavailable: {authorization_path}"
        )
    with _promotion_lock(root) as lock_state:
        history, chain = _load_authoritative_journal(root)
        _validate_journal_projections(
            root,
            history=history,
            chain=chain,
            trusted_public_key_path=key_path,
        )
        if not chain:
            raise PromotionError("cannot rollback without a current promoted release")
        try:
            authorization_bytes = authorization_path.read_bytes()
        except OSError as exc:
            raise PromotionError(
                f"cannot read rollback authorization: {authorization_path}"
            ) from exc
        current_event = chain[-1]
        authorization_sha256 = hashlib.sha256(authorization_bytes).hexdigest()
        committed_exact_bytes = (
            current_event.get("event_type") == "ROLLBACK"
            and current_event.get("rollback_authorization_sha256")
            == authorization_sha256
        )
        authorization = _verify_rollback_authorization(
            authorization_bytes,
            authorization_path=authorization_path,
            allow_historical_key=committed_exact_bytes,
        )
        if authorization.get("release_root_sha256") != _release_root_id(root):
            raise PromotionError("rollback authorization release root mismatch")
        if _is_exact_rollback_retry(current_event, authorization=authorization):
            current = _current_release_from_event(
                root,
                event=current_event,
                trusted_public_key_path=key_path,
            )
            lock_state.bind_journal_event(current_event["event_id"])
            try:
                _repair_journal_projections(root, head=history[-1], pointer=current.pointer)
            except BaseException as exc:
                raise PromotionProjectionRepairRequired(
                    str(current_event["event_id"]),
                    commit_status="COMMITTED",
                ) from exc
            return dict(current.pointer)
        if committed_exact_bytes:
            raise PromotionError("committed rollback authorization binding mismatch")
        actual_event_id = str(current_event["event_id"])
        expected_event_id = str(authorization["expected_current_event_id"])
        if actual_event_id != expected_event_id:
            raise PromotionConflictError(
                "rollback compare-and-swap conflict: "
                f"expected={expected_event_id!r}, actual={actual_event_id!r}"
            )
        if any(
            event.get("operation_id") == authorization["operation_id"]
            for event in chain
        ):
            raise PromotionError(
                f"rollback operation_id is already committed: {authorization['operation_id']}"
            )
        _validate_rollback_authorization_freshness(
            authorization,
            current_event=current_event,
        )
        target_event_id = str(authorization["target_promotion_event_id"])
        approved_event = next(
            (event for event in chain if str(event["event_id"]) == target_event_id),
            None,
        )
        if approved_event is None or approved_event.get("event_type") != "PROMOTE":
            raise PromotionError("rollback target is not an exact committed PROMOTE event")
        run_id = str(authorization["target_run_id"])
        if str(approved_event.get("run_id", "")) != run_id:
            raise PromotionError("rollback authorization target run_id mismatch")
        if str(approved_event.get("candidate_bundle_manifest_sha256", "")) != str(
            authorization["target_bundle_manifest_sha256"]
        ):
            raise PromotionError("rollback authorization target bundle mismatch")
        if target_event_id == actual_event_id:
            raise PromotionError("rollback target promotion event is already current")
        bundle = verify_candidate_bundle(root, run_id=run_id)
        if bundle.manifest_sha256 != approved_event.get("candidate_bundle_manifest_sha256"):
            raise PromotionError("rollback candidate no longer matches its approved promotion event")
        receipt_rel = str(approved_event["promotion_receipt_path"])
        receipt_path = _resolve_within(root, root / receipt_rel)
        receipt_bytes = receipt_path.read_bytes()
        if hashlib.sha256(receipt_bytes).hexdigest() != approved_event.get(
            "promotion_receipt_sha256"
        ):
            raise PromotionError("rollback promotion receipt hash mismatch")
        receipt = _verify_authorized_receipt(
            bundle,
            receipt_bytes=receipt_bytes,
            receipt_path=receipt_path,
            trusted_public_key_path=key_path,
            allow_historical_key=True,
        )
        archived_authorization = _archive_rollback_authorization(
            root,
            authorization=authorization,
            authorization_bytes=authorization_bytes,
        )
        event = _write_promotion_event(
            root,
            {
                "schema_version": "promotion_event.v2",
                "event_type": "ROLLBACK",
                "release_root_sha256": _release_root_id(root),
                "run_id": run_id,
                "candidate_bundle_manifest_sha256": bundle.manifest_sha256,
                "candidate_run_manifest_sha256": receipt["candidate_run_manifest_sha256"],
                "promotion_receipt_path": receipt_rel,
                "promotion_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "receipt_id": receipt["receipt_id"],
                "operation_id": authorization["operation_id"],
                "expected_current_event_id": expected_event_id,
                "target_promotion_event_id": target_event_id,
                "rollback_authorization_id": authorization["authorization_id"],
                "rollback_authorization_path": archived_authorization.relative_to(root).as_posix(),
                "rollback_authorization_sha256": hashlib.sha256(
                    authorization_bytes
                ).hexdigest(),
                "release_request_id": approved_event["release_request_id"],
                "rollback_reason": authorization["reason_code"],
                "rollback_incident_id": authorization["incident_id"],
                "recorded_at_utc": _canonical_utc_timestamp(
                    authorization["authorized_at_utc"],
                    label="rollback authorized_at_utc",
                ),
                "previous_event_id": actual_event_id,
            },
        )
        lock_state.bind_journal_event(event["event_id"])
        pointer = _pointer_for_event(root, event=event)
        _commit_and_publish_transition(
            root,
            event=event,
            pointer=pointer,
            updated_at_utc=str(event["recorded_at_utc"]),
            trusted_public_key_path=key_path,
        )
    return dict(pointer)


def resolve_current_candidate(
    release_root: str | Path,
) -> CurrentRelease:
    """Resolve the current pointer only after replaying every authorization binding."""

    root = _release_root(release_root, create=False)
    return _resolve_current_release(
        root,
        trusted_public_key_path=_trusted_public_key_path(),
    )


def _resolve_current_release(
    root: Path,
    *,
    trusted_public_key_path: Path,
) -> CurrentRelease:
    history, chain = _load_authoritative_journal(root)
    if not chain:
        raise PromotionError("promotion journal has no committed history")
    projection_repair_required = _validate_journal_projections(
        root,
        history=history,
        chain=chain,
        trusted_public_key_path=trusted_public_key_path,
    )
    return _current_release_from_event(
        root,
        event=chain[-1],
        trusted_public_key_path=trusted_public_key_path,
        projection_repair_required=projection_repair_required,
    )


def _load_authoritative_journal(
    root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Load only immutable signed heads, then verify their complete event chain."""

    history = _load_immutable_journal_history(root)
    if not history:
        return history, []
    chain = _replay_promotion_event_chain(
        root,
        start_event_id=str(history[-1]["event_id"]),
    )
    committed_ids = [str(head["event_id"]) for head in history]
    event_ids = [str(event["event_id"]) for event in chain]
    if committed_ids != event_ids:
        raise PromotionError("immutable promotion heads do not match the event chain")
    return history, chain


def _validate_journal_projections(
    root: Path,
    *,
    history: list[dict[str, object]],
    chain: list[dict[str, object]],
    trusted_public_key_path: Path,
) -> bool:
    """Accept absent or stale committed projections, but reject forged projections."""

    journal_root = _promotion_journal_root(root, create=False)
    mutable_path = _journal_head_path(journal_root, root, create=False)
    repair_required = False
    if mutable_path.exists():
        mutable = _verify_journal_head(root, _load_json(mutable_path))
        if not any(mutable == committed for committed in history):
            raise PromotionError("mutable promotion head is not a committed immutable head")
        repair_required = bool(history and mutable != history[-1])
    elif not history and journal_root.exists():
        # An empty journal may legitimately have its directory pre-created.
        pass
    elif history:
        repair_required = True

    pointer_path = root / CURRENT_POINTER
    if not pointer_path.exists():
        return bool(history) or repair_required
    if not chain:
        raise PromotionError("current pointer exists without committed promotion history")
    pointer = _load_json(pointer_path)
    event_id = str(pointer.get("event_id", ""))
    committed = {str(event["event_id"]): event for event in chain}
    event = committed.get(event_id)
    if event is None:
        raise PromotionError("current pointer references an uncommitted promotion event")
    expected = _current_release_from_event(
        root,
        event=event,
        trusted_public_key_path=trusted_public_key_path,
    ).pointer
    if dict(pointer) != dict(expected):
        raise PromotionError("current pointer projection is tampered or invalid")
    if chain and event_id != str(chain[-1]["event_id"]):
        repair_required = True
    return repair_required


def _current_release_from_event(
    root: Path,
    *,
    event: Mapping[str, object],
    trusted_public_key_path: Path,
    projection_repair_required: bool = False,
) -> CurrentRelease:
    run_id = str(event.get("run_id", ""))
    bundle = verify_candidate_bundle(root, run_id=run_id)
    if bundle.manifest_sha256 != event.get("candidate_bundle_manifest_sha256"):
        raise PromotionError("promotion event candidate bundle hash mismatch")
    candidate_run_hash = _sha256_file(bundle.path / CANDIDATE_RUN_MANIFEST)
    if candidate_run_hash != event.get("candidate_run_manifest_sha256"):
        raise PromotionError("promotion event candidate run hash mismatch")
    receipt_path = _resolve_within(
        root,
        root / str(event.get("promotion_receipt_path", "")),
    )
    try:
        receipt_bytes = receipt_path.read_bytes()
    except OSError as exc:
        raise PromotionError("current promotion receipt is unavailable") from exc
    if hashlib.sha256(receipt_bytes).hexdigest() != event.get("promotion_receipt_sha256"):
        raise PromotionError("current promotion receipt hash mismatch")
    receipt = _verify_authorized_receipt(
        bundle,
        receipt_bytes=receipt_bytes,
        receipt_path=receipt_path,
        trusted_public_key_path=trusted_public_key_path,
        allow_historical_key=True,
    )
    if str(receipt.get("receipt_id", "")) != str(event.get("receipt_id", "")):
        raise PromotionError("promotion event receipt_id mismatch")
    return CurrentRelease(
        bundle=bundle,
        pointer=_pointer_for_event(root, event=event),
        event=dict(event),
        receipt=receipt,
        projection_repair_required=projection_repair_required,
    )


def _pointer_for_event(root: Path, *, event: Mapping[str, object]) -> dict[str, object]:
    event_id = str(event["event_id"])
    run_id = str(event["run_id"])
    previous_event_id = event.get("previous_event_id")
    previous_run_id = None
    if previous_event_id:
        previous_run_id = str(
            _load_promotion_event(root, str(previous_event_id)).get("run_id", "")
        )
    pointer: dict[str, object] = {
        "schema_version": "current_pointer.v2",
        "run_id": run_id,
        "candidate_path": (Path("candidates") / run_id).as_posix(),
        "candidate_bundle_manifest_sha256": event["candidate_bundle_manifest_sha256"],
        "candidate_run_manifest_sha256": event["candidate_run_manifest_sha256"],
        "promotion_receipt_path": event["promotion_receipt_path"],
        "promotion_receipt_sha256": event["promotion_receipt_sha256"],
        "previous_run_id": previous_run_id,
        "event_id": event_id,
        "event_path": (Path(PROMOTION_EVENTS_DIR) / f"{event_id}.json").as_posix(),
    }
    if event.get("event_type") == "ROLLBACK":
        pointer.update(
            {
                "rollback": True,
                "rollback_reason": str(event.get("rollback_reason", "")),
                "rolled_back_at_utc": event["recorded_at_utc"],
            }
        )
    else:
        pointer["promoted_at_utc"] = event["recorded_at_utc"]
    return pointer


def _is_exact_promotion_retry(
    event: Mapping[str, object] | None,
    *,
    run_id: str,
    candidate_bundle_manifest_sha256: str,
    expected_current_event_id: str | None,
    operation_id: str,
    release_request_id: str,
    operation_created_at_utc: str,
) -> bool:
    if event is None or event.get("event_type") != "PROMOTE":
        return False
    expected = {
        "run_id": run_id,
        "candidate_bundle_manifest_sha256": candidate_bundle_manifest_sha256,
        "expected_current_event_id": expected_current_event_id,
        "operation_id": operation_id,
        "release_request_id": release_request_id,
        "operation_created_at_utc": operation_created_at_utc,
    }
    return all(event.get(key) == value for key, value in expected.items())


def _validate_receipt_operation_binding(
    receipt: Mapping[str, object],
    *,
    root: Path,
    expected_current_event_id: str | None,
    operation_id: str,
    release_request_id: str,
    operation_created_at_utc: str,
    workflow_domain_sha256: str,
    release_request_document_sha256: str,
) -> None:
    expected = {
        "release_request_id": release_request_id,
        "release_operation_id": operation_id,
        "expected_current_event_id": expected_current_event_id,
        "operation_created_at_utc": operation_created_at_utc,
        "release_root_sha256": _release_root_id(root),
        "workflow_domain_sha256": workflow_domain_sha256,
        "release_request_document_sha256": release_request_document_sha256,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise PromotionError("promotion receipt operation binding mismatch")


def _validate_registered_release_request_binding(
    request: Mapping[str, object],
    *,
    root: Path,
    run_id: str,
    expected_current_event_id: str | None,
    operation_id: str,
    release_request_id: str,
    operation_created_at_utc: str,
) -> None:
    expected_current = (
        {"kind": "NONE"}
        if expected_current_event_id is None
        else {"kind": "EVENT", "event_id": expected_current_event_id}
    )
    expected = {
        "run_id": run_id,
        "request_id": release_request_id,
        "created_at_utc": operation_created_at_utc,
        "release_root_sha256": _release_root_id(root),
        "expected_current": expected_current,
    }
    if operation_id != release_request_id or any(
        request.get(key) != value for key, value in expected.items()
    ):
        raise PromotionError("registered release request operation binding mismatch")


def _validate_registered_release_request_candidate_binding(
    request: Mapping[str, object],
    *,
    candidate_bundle_manifest_sha256: str,
) -> None:
    expected = {
        "registration_contract": "assembled_candidate_seal.v1",
        "candidate_bundle_manifest_sha256": candidate_bundle_manifest_sha256,
    }
    if any(request.get(key) != value for key, value in expected.items()):
        raise PromotionError("registered release request candidate binding mismatch")


def _is_exact_rollback_retry(
    event: Mapping[str, object],
    *,
    authorization: Mapping[str, object],
) -> bool:
    if event.get("event_type") != "ROLLBACK":
        return False
    expected = {
        "operation_id": authorization["operation_id"],
        "expected_current_event_id": authorization["expected_current_event_id"],
        "target_promotion_event_id": authorization["target_promotion_event_id"],
        "run_id": authorization["target_run_id"],
        "candidate_bundle_manifest_sha256": authorization[
            "target_bundle_manifest_sha256"
        ],
        "rollback_authorization_id": authorization["authorization_id"],
    }
    return all(event.get(key) == value for key, value in expected.items())


def _verify_rollback_authorization(
    authorization_bytes: bytes,
    *,
    authorization_path: Path,
    allow_historical_key: bool = False,
) -> dict[str, object]:
    document = _load_json_bytes(authorization_bytes, authorization_path)
    selected_public_key = _matching_trusted_public_key(
        document,
        primary_path=_required_env_key_path(
            ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_ENV
        ),
        keyring_directory_env=ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_DIR_ENV,
        label="rollback authorization",
        allow_historical=allow_historical_key,
    )
    try:
        return verify_rollback_authorization_signature(
            document,
            trusted_public_key_path=selected_public_key,
        )
    except RollbackAuthorizationAuthenticationError as exc:
        raise PromotionError(
            f"rollback authorization authentication failed: {exc}"
        ) from exc


def _validate_rollback_authorization_freshness(
    authorization: Mapping[str, object],
    *,
    current_event: Mapping[str, object],
) -> None:
    authorized_at = _aware_datetime(
        authorization.get("authorized_at_utc"),
        label="rollback authorized_at_utc",
    )
    expires_at = _aware_datetime(
        authorization.get("expires_at_utc"),
        label="rollback expires_at_utc",
    )
    if expires_at <= authorized_at:
        raise PromotionError("rollback authorization expiry is not after authorization")
    if (expires_at - authorized_at).total_seconds() > MAX_NEW_RECEIPT_AGE:
        raise PromotionError("rollback authorization validity window is too long")
    now = _promotion_now_utc()
    if (authorized_at - now).total_seconds() > MAX_CLOCK_SKEW:
        raise PromotionError("rollback authorization timestamp is in the future")
    if now >= expires_at:
        raise PromotionError("rollback authorization is expired")
    current_recorded_at = _aware_datetime(
        current_event.get("recorded_at_utc"),
        label="current event recorded_at_utc",
    )
    if authorized_at < current_recorded_at:
        raise PromotionError("rollback authorization predates the expected current event")


def _canonical_utc_timestamp(value: object, *, label: str) -> str:
    return _aware_datetime(value, label=label).replace(microsecond=0).isoformat()


def _validate_expected_event_id(event_id: str | None) -> None:
    if event_id is not None and not re.fullmatch(r"[0-9a-f]{64}", str(event_id)):
        raise PromotionError("expected_current_event_id is invalid")


def _required_operation_value(value: str, *, label: str) -> str:
    selected = str(value).strip()
    if not selected or len(selected) > 256:
        raise PromotionError(f"{label} is invalid")
    return selected


def _canonical_operation_timestamp(value: str) -> str:
    timestamp = _aware_datetime(value, label="operation_created_at_utc")
    return timestamp.replace(microsecond=0).isoformat()


def _verify_authorized_receipt(
    bundle: CandidateBundle,
    *,
    receipt_bytes: bytes,
    receipt_path: Path,
    trusted_public_key_path: Path,
    require_recent_evaluation: bool = False,
    allow_historical_key: bool = False,
) -> dict[str, object]:
    receipt_document = _load_json_bytes(receipt_bytes, receipt_path)
    selected_public_key = _matching_trusted_public_key(
        receipt_document,
        primary_path=trusted_public_key_path,
        keyring_directory_env=RECEIPT_TRUSTED_PUBLIC_KEY_DIR_ENV,
        label="promotion receipt",
        allow_historical=allow_historical_key,
    )
    try:
        receipt = verify_promotion_receipt_signature(
            receipt_document,
            trusted_public_key_path=selected_public_key,
        )
        validate_required_gate_set(
            receipt,
            allow_historical_policy=allow_historical_key,
        )
        approved = validate_promotion_receipt_decision(receipt)
    except ReceiptAuthenticationError as exc:
        raise PromotionError(f"promotion receipt authentication failed: {exc}") from exc
    if receipt.get("schema_version") != "promotion_receipt.v2" or approved is not True:
        raise PromotionError("promotion receipt is not an approved promotion_receipt.v2")
    if receipt.get("authorization_mode") != "REGISTERED_PROMOTION":
        raise PromotionError("promotion receipt is not a registered promotion authorization")
    _require_exact_string_fields(
        receipt,
        (
            "run_id",
            "valuation_timestamp",
            "promotion_evaluated_at",
            "candidate_bundle_manifest_sha256",
            "candidate_run_manifest_sha256",
            "receipt_id",
            "release_request_id",
            "release_operation_id",
            "operation_created_at_utc",
            "release_root_sha256",
            "workflow_domain_sha256",
            "release_request_document_sha256",
        ),
        label="promotion receipt",
    )
    expected_current = receipt.get("expected_current_event_id")
    if expected_current is not None and type(expected_current) is not str:
        raise PromotionError(
            "promotion receipt expected_current_event_id must be an exact string or null"
        )
    if receipt["run_id"] != bundle.run_id:
        raise PromotionError("promotion receipt run_id is not bound to candidate")
    if receipt.get("candidate_bundle_manifest_sha256") != bundle.manifest_sha256:
        raise PromotionError("promotion receipt is not bound to the candidate bundle manifest")
    if receipt.get("candidate_run_manifest_sha256") != _sha256_file(
        bundle.path / CANDIDATE_RUN_MANIFEST
    ):
        raise PromotionError("promotion receipt is not bound to candidate run manifest")
    if require_recent_evaluation:
        try:
            evaluated_at = datetime.fromisoformat(receipt["promotion_evaluated_at"])
        except ValueError as exc:
            raise PromotionError("promotion receipt evaluated_at timestamp is invalid") from exc
        if evaluated_at.tzinfo is None:
            raise PromotionError("promotion receipt evaluated_at timestamp is timezone-naive")
        now = _promotion_now_utc()
        age_seconds = (now - evaluated_at.astimezone(timezone.utc)).total_seconds()
        if age_seconds < -MAX_CLOCK_SKEW:
            raise PromotionError("promotion receipt evaluation timestamp is in the future")
        if age_seconds > MAX_NEW_RECEIPT_AGE:
            raise PromotionError("promotion receipt evaluation is too old for a new promotion")
    return receipt


def _archive_promotion_receipt(
    root: Path,
    *,
    receipt: Mapping[str, object],
    receipt_bytes: bytes,
) -> Path:
    receipt_id = str(receipt.get("receipt_id", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", receipt_id):
        raise PromotionError("promotion receipt_id is not a canonical SHA-256")
    directory = root / PROMOTION_RECEIPTS_DIR
    _mkdir_durable(directory)
    directory = _resolve_within(root, directory)
    destination = directory / f"{receipt_id}.json"
    if destination.exists():
        _validate_existing_immutable_file(destination, label="promotion receipt archive")
        if destination.read_bytes() != receipt_bytes:
            raise PromotionError("promotion receipt archive collision")
        return destination
    _write_bytes_exclusive_fsync(destination, receipt_bytes)
    return destination


def _archive_rollback_authorization(
    root: Path,
    *,
    authorization: Mapping[str, object],
    authorization_bytes: bytes,
) -> Path:
    authorization_id = str(authorization.get("authorization_id", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", authorization_id):
        raise PromotionError("rollback authorization_id is not a canonical SHA-256")
    directory = root / ROLLBACK_AUTHORIZATIONS_DIR
    _mkdir_durable(directory)
    directory = _resolve_within(root, directory)
    destination = directory / f"{authorization_id}.json"
    if destination.exists():
        _validate_existing_immutable_file(
            destination,
            label="rollback authorization archive",
        )
        if destination.read_bytes() != authorization_bytes:
            raise PromotionError("rollback authorization archive collision")
        return destination
    _write_bytes_exclusive_fsync(destination, authorization_bytes)
    return destination


def _write_promotion_event(root: Path, payload: Mapping[str, object]) -> dict[str, object]:
    event = sign_promotion_event(
        payload,
        private_key_path=_required_env_key_path(EVENT_SIGNING_PRIVATE_KEY_ENV),
    )
    event_id = str(event["event_id"])
    directory = root / PROMOTION_EVENTS_DIR
    _mkdir_durable(directory)
    directory = _resolve_within(root, directory)
    destination = directory / f"{event_id}.json"
    if destination.exists():
        existing = _load_promotion_event(root, event_id)
        if existing != {key: value for key, value in event.items() if key != "signature"}:
            raise PromotionError("promotion event collision")
        return event
    raw = (json.dumps(event, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    _write_bytes_exclusive_fsync(destination, raw)
    return event


def _load_promotion_event(root: Path, event_id: str) -> dict[str, object]:
    if not re.fullmatch(r"[0-9a-f]{64}", event_id):
        raise PromotionError("promotion event_id is invalid")
    path = _resolve_within(
        root,
        root / PROMOTION_EVENTS_DIR / f"{event_id}.json",
    )
    document = _load_json(path)
    selected_public_key = _matching_trusted_public_key(
        document,
        primary_path=_required_env_key_path(EVENT_TRUSTED_PUBLIC_KEY_ENV),
        keyring_directory_env=EVENT_TRUSTED_PUBLIC_KEY_DIR_ENV,
        label="promotion event",
        allow_historical=True,
    )
    try:
        event = verify_promotion_event_signature(
            document,
            trusted_public_key_path=selected_public_key,
        )
    except EventAuthenticationError as exc:
        raise PromotionError(f"promotion event authentication failed: {exc}") from exc
    if event.get("event_id") != event_id or type(event.get("event_id")) is not str:
        raise PromotionError("promotion event path/id mismatch")
    if event.get("schema_version") != "promotion_event.v2":
        raise PromotionError("promotion event schema mismatch")
    if event.get("event_type") not in {"PROMOTE", "ROLLBACK"}:
        raise PromotionError("promotion event type is invalid")
    if event.get("release_root_sha256") != _release_root_id(root, create=False):
        raise PromotionError("promotion event release root mismatch")
    _validate_authenticated_event_types(event)
    return event


def _validate_authenticated_event_types(event: Mapping[str, object]) -> None:
    common_fields = (
        "event_id",
        "release_root_sha256",
        "run_id",
        "candidate_bundle_manifest_sha256",
        "candidate_run_manifest_sha256",
        "promotion_receipt_path",
        "promotion_receipt_sha256",
        "receipt_id",
        "operation_id",
        "release_request_id",
        "recorded_at_utc",
    )
    _require_exact_string_fields(event, common_fields, label="promotion event")
    for field in ("expected_current_event_id", "previous_event_id"):
        value = event.get(field)
        if value is not None and type(value) is not str:
            raise PromotionError(
                f"promotion event {field} must be an exact string or null"
            )
    event_type = event.get("event_type")
    if event_type == "PROMOTE":
        _require_exact_string_fields(
            event,
            ("operation_created_at_utc",),
            label="promotion event",
        )
    elif event_type == "ROLLBACK":
        _require_exact_string_fields(
            event,
            (
                "target_promotion_event_id",
                "rollback_authorization_id",
                "rollback_authorization_path",
                "rollback_authorization_sha256",
                "rollback_reason",
                "rollback_incident_id",
            ),
            label="promotion event",
        )


def _require_exact_string_fields(
    document: Mapping[str, object],
    fields: tuple[str, ...],
    *,
    label: str,
) -> None:
    for field in fields:
        if type(document.get(field)) is not str:
            raise PromotionError(f"{label} {field} must be an exact string")


def _replay_promotion_event_chain(
    root: Path,
    *,
    start_event_id: str,
) -> list[dict[str, object]]:
    seen: set[str] = set()
    reverse_chain: list[dict[str, object]] = []
    event_id: str | None = start_event_id
    while event_id:
        if event_id in seen:
            raise PromotionError("promotion event chain contains a cycle")
        seen.add(event_id)
        event = _load_promotion_event(root, event_id)
        reverse_chain.append(event)
        previous = event.get("previous_event_id")
        event_id = str(previous) if previous else None
    chain = list(reversed(reverse_chain))
    if not chain or chain[0].get("event_type") != "PROMOTE":
        raise PromotionError("promotion event chain must start with PROMOTE")
    promoted_events: dict[str, dict[str, object]] = {}
    previous_id: str | None = None
    previous_time: datetime | None = None
    for event in chain:
        if event.get("previous_event_id") != previous_id:
            raise PromotionError("promotion event previous_event_id mismatch")
        run_id = str(event.get("run_id", ""))
        _validate_run_id(run_id)
        for key in (
            "candidate_bundle_manifest_sha256",
            "candidate_run_manifest_sha256",
            "promotion_receipt_sha256",
            "receipt_id",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", str(event.get(key, ""))):
                raise PromotionError(f"promotion event {key} is invalid")
        historical_bundle = verify_candidate_bundle(root, run_id=run_id)
        if historical_bundle.manifest_sha256 != event.get(
            "candidate_bundle_manifest_sha256"
        ):
            raise PromotionError("historical promotion event bundle hash mismatch")
        historical_receipt_path = _resolve_within(
            root,
            root / str(event.get("promotion_receipt_path", "")),
        )
        try:
            historical_receipt_bytes = historical_receipt_path.read_bytes()
        except OSError as exc:
            raise PromotionError("historical promotion receipt is unavailable") from exc
        if hashlib.sha256(historical_receipt_bytes).hexdigest() != event.get(
            "promotion_receipt_sha256"
        ):
            raise PromotionError("historical promotion receipt hash mismatch")
        historical_receipt = _verify_authorized_receipt(
            historical_bundle,
            receipt_bytes=historical_receipt_bytes,
            receipt_path=historical_receipt_path,
            trusted_public_key_path=_trusted_public_key_path(),
            allow_historical_key=True,
        )
        if str(historical_receipt.get("receipt_id", "")) != str(event.get("receipt_id", "")):
            raise PromotionError("historical promotion event receipt_id mismatch")
        recorded_at = _aware_datetime(event.get("recorded_at_utc"), label="event recorded_at_utc")
        if previous_time is not None and recorded_at < previous_time:
            raise PromotionError("promotion event timestamps are not monotonic")
        if event.get("event_type") == "PROMOTE":
            promoted_events[str(event["event_id"])] = event
            operation_fields = (
                "operation_id",
                "release_request_id",
                "operation_created_at_utc",
                "expected_current_event_id",
            )
            if not all(field in event for field in operation_fields):
                raise PromotionError("promotion event operation binding is incomplete")
            _required_operation_value(str(event["operation_id"]), label="operation_id")
            _required_operation_value(
                str(event["release_request_id"]),
                label="release_request_id",
            )
            expected_current = event.get("expected_current_event_id")
            _validate_expected_event_id(
                str(expected_current) if expected_current is not None else None
            )
            if expected_current != event.get("previous_event_id"):
                raise PromotionError("promotion event CAS binding mismatch")
            operation_created = _canonical_operation_timestamp(
                str(event["operation_created_at_utc"])
            )
            if operation_created != event.get("recorded_at_utc"):
                raise PromotionError("promotion event operation timestamp mismatch")
            _validate_receipt_operation_binding(
                historical_receipt,
                root=root,
                expected_current_event_id=(
                    str(expected_current) if expected_current is not None else None
                ),
                operation_id=str(event["operation_id"]),
                release_request_id=str(event["release_request_id"]),
                operation_created_at_utc=operation_created,
                workflow_domain_sha256=str(
                    historical_receipt["workflow_domain_sha256"]
                ),
                release_request_document_sha256=str(
                    historical_receipt["release_request_document_sha256"]
                ),
            )
        else:
            _validate_committed_rollback_event(
                root,
                event=event,
                promoted_events=promoted_events,
            )
        previous_id = str(event["event_id"])
        previous_time = recorded_at
    return chain


def _validate_committed_rollback_event(
    root: Path,
    *,
    event: Mapping[str, object],
    promoted_events: Mapping[str, Mapping[str, object]],
) -> None:
    operation_id = _required_operation_value(
        str(event.get("operation_id", "")),
        label="rollback operation_id",
    )
    expected_current = str(event.get("expected_current_event_id", ""))
    _validate_expected_event_id(expected_current)
    if expected_current != event.get("previous_event_id"):
        raise PromotionError("rollback event CAS binding mismatch")
    target_event_id = str(event.get("target_promotion_event_id", ""))
    _validate_expected_event_id(target_event_id)
    target = promoted_events.get(target_event_id)
    if target is None:
        raise PromotionError("rollback event does not target an earlier PROMOTE event")
    for key in (
        "run_id",
        "candidate_bundle_manifest_sha256",
        "candidate_run_manifest_sha256",
        "promotion_receipt_path",
        "promotion_receipt_sha256",
        "receipt_id",
        "release_request_id",
        "release_root_sha256",
    ):
        if event.get(key) != target.get(key):
            raise PromotionError(f"rollback event target {key} mismatch")
    authorization_rel = str(event.get("rollback_authorization_path", ""))
    authorization_path = _resolve_within(root, root / authorization_rel)
    try:
        authorization_bytes = authorization_path.read_bytes()
    except OSError as exc:
        raise PromotionError("historical rollback authorization is unavailable") from exc
    if hashlib.sha256(authorization_bytes).hexdigest() != event.get(
        "rollback_authorization_sha256"
    ):
        raise PromotionError("historical rollback authorization hash mismatch")
    authorization = _verify_rollback_authorization(
        authorization_bytes,
        authorization_path=authorization_path,
        allow_historical_key=True,
    )
    expected = {
        "authorization_id": event.get("rollback_authorization_id"),
        "operation_id": operation_id,
        "release_root_sha256": _release_root_id(root),
        "expected_current_event_id": expected_current,
        "target_promotion_event_id": target_event_id,
        "target_run_id": event.get("run_id"),
        "target_bundle_manifest_sha256": event.get(
            "candidate_bundle_manifest_sha256"
        ),
        "reason_code": event.get("rollback_reason"),
        "incident_id": event.get("rollback_incident_id"),
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise PromotionError("rollback event authorization binding mismatch")
    if _canonical_utc_timestamp(
        authorization.get("authorized_at_utc"),
        label="rollback authorized_at_utc",
    ) != event.get("recorded_at_utc"):
        raise PromotionError("rollback event authorization timestamp mismatch")


def _commit_immutable_journal_head(
    root: Path,
    *,
    event_id: str,
    updated_at_utc: str,
) -> dict[str, object]:
    history = _load_immutable_journal_history(root)
    previous_event_id = str(history[-1]["event_id"]) if history else None
    head = sign_promotion_head(
        {
            "schema_version": "promotion_head.v1",
            "release_root_sha256": _release_root_id(root),
            "event_id": event_id,
            "previous_event_id": previous_event_id,
            "sequence": len(history) + 1,
            "updated_at_utc": updated_at_utc,
        },
        private_key_path=_required_env_key_path(EVENT_SIGNING_PRIVATE_KEY_ENV),
    )
    journal_root = _promotion_journal_root(root, create=True)
    history_path = _journal_history_path(journal_root, root, head)
    raw = (json.dumps(head, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    _write_bytes_exclusive_fsync(history_path, raw)
    return head


def _commit_and_publish_transition(
    root: Path,
    *,
    event: Mapping[str, object],
    pointer: Mapping[str, object],
    updated_at_utc: str,
    trusted_public_key_path: Path,
) -> None:
    event_id = str(event["event_id"])
    head_committed = False
    try:
        head = _commit_immutable_journal_head(
            root,
            event_id=event_id,
            updated_at_utc=updated_at_utc,
        )
        head_committed = True
        _write_mutable_head_projection(root, head)
        _write_current_pointer_projection(root, pointer)
        _resolve_current_release(root, trusted_public_key_path=trusted_public_key_path)
    except BaseException as exc:
        if head_committed:
            raise PromotionProjectionRepairRequired(event_id) from exc
        try:
            history = _load_immutable_journal_history(root)
            committed = bool(history and str(history[-1]["event_id"]) == event_id)
        except BaseException as verification_error:
            raise PromotionProjectionRepairRequired(
                event_id,
                commit_status="INDETERMINATE",
            ) from verification_error
        if committed:
            raise PromotionProjectionRepairRequired(event_id) from exc
        raise


def _write_mutable_head_projection(root: Path, head: Mapping[str, object]) -> None:
    journal_root = _promotion_journal_root(root, create=True)
    head_path = _journal_head_path(journal_root, root, create=True)
    _atomic_write_json(head_path, head)


def _write_current_pointer_projection(
    root: Path,
    pointer: Mapping[str, object],
) -> None:
    _atomic_write_json(root / CURRENT_POINTER, pointer)


def _repair_journal_projections(
    root: Path,
    *,
    head: Mapping[str, object],
    pointer: Mapping[str, object],
) -> None:
    _write_mutable_head_projection(root, head)
    _write_current_pointer_projection(root, pointer)


def _load_journal_head(root: Path) -> str:
    history = _load_immutable_journal_history(root)
    if not history:
        raise PromotionError("promotion journal has no immutable signed head history")
    return str(history[-1]["event_id"])


def _load_immutable_journal_history(root: Path) -> list[dict[str, object]]:
    journal_root = _promotion_journal_root(root, create=False)
    history_dir = _journal_history_dir(journal_root, root, create=False)
    if not history_dir.is_dir():
        return []
    history_files = sorted(history_dir.glob("*.json"))
    if not history_files:
        return []
    heads: list[dict[str, object]] = []
    previous_event_id: str | None = None
    for expected_sequence, path in enumerate(history_files, start=1):
        head = _verify_journal_head(root, _load_json(path))
        sequence = head.get("sequence")
        if type(sequence) is not int:
            raise PromotionError("promotion journal head sequence is invalid")
        if sequence != expected_sequence:
            raise PromotionError("promotion journal head sequence is not contiguous")
        expected_name = f"{sequence:012d}-{str(head['head_id'])[:32]}.json"
        if path.name != expected_name:
            raise PromotionError("promotion journal immutable head filename mismatch")
        if head.get("previous_event_id") != previous_event_id:
            raise PromotionError("promotion journal immutable head chain mismatch")
        previous_event_id = str(head["event_id"])
        heads.append(head)
    return heads


def _load_journal_history(root: Path) -> list[dict[str, object]]:
    """Compatibility alias for callers that need authoritative committed history."""

    return _load_immutable_journal_history(root)


def _verify_journal_head(root: Path, document: Mapping[str, object]) -> dict[str, object]:
    selected_public_key = _matching_trusted_public_key(
        document,
        primary_path=_required_env_key_path(EVENT_TRUSTED_PUBLIC_KEY_ENV),
        keyring_directory_env=EVENT_TRUSTED_PUBLIC_KEY_DIR_ENV,
        label="promotion journal head",
        allow_historical=True,
    )
    try:
        head = verify_promotion_head_signature(
            document,
            trusted_public_key_path=selected_public_key,
        )
    except EventAuthenticationError as exc:
        raise PromotionError(f"promotion journal head authentication failed: {exc}") from exc
    if head.get("schema_version") != "promotion_head.v1":
        raise PromotionError("promotion journal head schema mismatch")
    release_root_sha256 = head.get("release_root_sha256")
    if (
        type(release_root_sha256) is not str
        or release_root_sha256 != _release_root_id(root, create=False)
    ):
        raise PromotionError("promotion journal head release root mismatch")
    event_id = head.get("event_id")
    if type(event_id) is not str or not re.fullmatch(r"[0-9a-f]{64}", event_id):
        raise PromotionError("promotion journal head event_id is invalid")
    head_id = head.get("head_id")
    if type(head_id) is not str or not re.fullmatch(r"[0-9a-f]{64}", head_id):
        raise PromotionError("promotion journal head head_id is invalid")
    return dict(document)


def _promotion_journal_root(release_root: Path, *, create: bool = True) -> Path:
    selected = os.environ.get(PROMOTION_JOURNAL_ROOT_ENV)
    if not selected:
        raise PromotionError(f"{PROMOTION_JOURNAL_ROOT_ENV} is required")
    try:
        journal_root = assert_absolute_path_has_no_links(Path(selected).expanduser())
        release = assert_absolute_path_has_no_links(release_root)
        overlaps = paths_overlap_by_identity(journal_root, release)
    except (OSError, ValueError) as exc:
        raise PromotionError("cannot validate promotion journal root identity") from exc
    if overlaps:
        raise PromotionError("promotion journal root must be external to release root")
    if create:
        _mkdir_durable(journal_root)
        try:
            journal_root = assert_absolute_path_has_no_links(journal_root)
            if paths_overlap_by_identity(journal_root, release):
                raise PromotionError(
                    "promotion journal root must be external to release root"
                )
        except (OSError, ValueError) as exc:
            raise PromotionError("cannot revalidate promotion journal root identity") from exc
    return journal_root


def _journal_head_path(
    journal_root: Path,
    release_root: Path,
    *,
    create: bool = True,
) -> Path:
    heads = journal_root / "heads"
    if create:
        _mkdir_durable(heads)
    try:
        heads = assert_absolute_path_has_no_links(heads)
    except (OSError, ValueError) as exc:
        raise PromotionError("cannot validate promotion journal heads") from exc
    if journal_root != heads and journal_root not in heads.parents:
        raise PromotionError(f"path escapes release root: {heads}")
    return heads / f"{_release_root_id(release_root, create=create)}.json"


def _journal_history_dir(
    journal_root: Path,
    release_root: Path,
    *,
    create: bool = True,
) -> Path:
    history = journal_root / "h" / _release_root_id(
        release_root,
        create=create,
    )[:32]
    if create:
        _mkdir_durable(history)
    try:
        history = assert_absolute_path_has_no_links(history)
    except (OSError, ValueError) as exc:
        raise PromotionError("cannot validate promotion journal history") from exc
    if journal_root != history and journal_root not in history.parents:
        raise PromotionError(f"path escapes release root: {history}")
    return history


def _journal_history_path(
    journal_root: Path,
    release_root: Path,
    head: Mapping[str, object],
) -> Path:
    sequence = int(head["sequence"])
    head_id = str(head["head_id"])
    return _journal_history_dir(journal_root, release_root) / (
        f"{sequence:012d}-{head_id[:32]}.json"
    )


def _release_root_id(root: Path, *, create: bool = True) -> str:
    selected = str(os.environ.get(RELEASE_DOMAIN_ID_ENV, "")).strip().lower()
    if not selected:
        raise PromotionError(f"{RELEASE_DOMAIN_ID_ENV} is required")
    try:
        domain_id = str(uuid.UUID(selected))
    except ValueError as exc:
        raise PromotionError(f"{RELEASE_DOMAIN_ID_ENV} must be a canonical UUID") from exc
    if selected != domain_id:
        raise PromotionError(f"{RELEASE_DOMAIN_ID_ENV} must be a canonical UUID")
    domain_sha256 = hashlib.sha256(
        f"pfc_lt_release_domain.v1:{domain_id}".encode("ascii")
    ).hexdigest()
    marker_payload: dict[str, object] = {
        "schema_version": "pfc_lt_release_domain.v1",
        "release_domain_id": domain_id,
        "release_domain_sha256": domain_sha256,
    }
    root = Path(root).resolve()
    if create:
        _mkdir_durable(root)
    elif not root.is_dir():
        raise PromotionError("release root is unavailable")
    marker = root / RELEASE_DOMAIN_MARKER
    marker_is_link = _path_is_link(marker)
    if create and not marker.exists() and not marker_is_link:
        if any(
            (root / name).exists()
            for name in (
                CURRENT_POINTER,
                PROMOTION_EVENTS_DIR,
                ROLLBACK_AUTHORIZATIONS_DIR,
            )
        ):
            raise PromotionError("release domain marker is missing after transition activity")
        raw = (json.dumps(marker_payload, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        try:
            _write_bytes_exclusive_fsync(marker, raw)
        except PromotionError:
            if not marker.exists():
                raise
    if not marker.exists() and not marker_is_link:
        raise PromotionError("release domain marker is unavailable")
    _validate_existing_immutable_file(
        marker,
        label="release domain marker",
        recover_crash_links=create,
    )
    if _load_json(marker) != marker_payload:
        raise PromotionError("configured release domain does not match pinned marker")
    return domain_sha256


def _find_approved_promotion_event(
    root: Path,
    *,
    start_event_id: str,
    run_id: str,
) -> dict[str, object]:
    seen: set[str] = set()
    event_id: str | None = start_event_id
    while event_id:
        if event_id in seen:
            raise PromotionError("promotion event chain contains a cycle")
        seen.add(event_id)
        event = _load_promotion_event(root, event_id)
        if event.get("event_type") == "PROMOTE" and str(event.get("run_id", "")) == run_id:
            return event
        previous = event.get("previous_event_id")
        event_id = str(previous) if previous else None
    raise PromotionError(f"rollback target was never approved in promotion history: {run_id}")


def _trusted_public_key_path() -> Path:
    selected = os.environ.get("PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH")
    if not selected:
        raise PromotionError("trusted promotion public key path is required")
    path = _absolute_lexical_path(selected)
    _assert_path_has_no_links(path, label="trusted promotion public key")
    if not path.is_file():
        raise PromotionError(f"trusted promotion public key is unavailable: {path}")
    return path


def _assert_transition_authority_separation(
    *,
    require_rollback_authority: bool,
) -> None:
    exposed = sorted(
        name
        for name in FORBIDDEN_TRANSITION_PRIVATE_KEY_ENVS
        if str(os.environ.get(name, "")).strip()
    )
    if exposed:
        raise PromotionError(f"transition identity exposes forbidden private keys: {exposed}")
    event_private = _required_env_key_path(EVENT_SIGNING_PRIVATE_KEY_ENV)
    event_public = _required_env_key_path(EVENT_TRUSTED_PUBLIC_KEY_ENV)
    receipt_public = _trusted_public_key_path()
    event_id = public_key_id_from_private_key_path(event_private)
    if public_key_id_from_public_key_path(event_public) != event_id:
        raise PromotionError("promotion event private/public key identity mismatch")
    authority_key_ids = {
        "event": set(
            _trusted_public_key_identities(
                event_public,
                keyring_directory_env=EVENT_TRUSTED_PUBLIC_KEY_DIR_ENV,
                label="promotion event",
            )
        ),
        "receipt": set(
            _trusted_public_key_identities(
                receipt_public,
                keyring_directory_env=RECEIPT_TRUSTED_PUBLIC_KEY_DIR_ENV,
                label="promotion receipt",
            )
        ),
        "register": set(
            _trusted_public_key_identities(
                _required_env_key_path(REGISTER_TRUSTED_PUBLIC_KEY_ENV),
                keyring_directory_env=REGISTER_TRUSTED_PUBLIC_KEY_DIR_ENV,
                label="REGISTER request",
            )
        ),
    }
    rollback_selected = os.environ.get(ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_ENV)
    if require_rollback_authority and not rollback_selected:
        raise PromotionError(
            f"{ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_ENV} is required"
        )
    if rollback_selected:
        authority_key_ids["rollback_authorization"] = set(
            _trusted_public_key_identities(
                _required_env_key_path(
                    ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_ENV
                ),
                keyring_directory_env=(
                    ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_DIR_ENV
                ),
                label="rollback authorization",
            )
        )
    labels = sorted(authority_key_ids)
    collisions = [
        f"{left}/{right}"
        for index, left in enumerate(labels)
        for right in labels[index + 1 :]
        if authority_key_ids[left].intersection(authority_key_ids[right])
    ]
    if collisions:
        raise PromotionError(f"promotion authority keyrings overlap: {collisions}")


def _validate_existing_immutable_file(
    path: Path,
    *,
    label: str,
    recover_crash_links: bool = True,
) -> None:
    if _path_is_link(path) or not path.is_file():
        raise PromotionError(f"{label} is not a regular file")
    if recover_crash_links:
        _recover_exclusive_write_links(path)
    try:
        link_count = int(path.stat().st_nlink)
    except (AttributeError, OSError) as exc:
        raise PromotionError(f"cannot validate {label}") from exc
    if link_count != 1:
        raise PromotionError(f"{label} has forbidden hard links")


def validate_existing_immutable_file(path: Path, *, label: str) -> None:
    """Validate one published immutable file, including crash-link recovery."""

    _validate_existing_immutable_file(path, label=label)


def prepare_immutable_directory(directory: Path, *, label: str) -> None:
    """Recover exact crash links and reject ambiguous directory residue."""

    if _path_is_link(directory) or not directory.is_dir():
        raise PromotionError(f"{label} is not a regular directory")
    entries = list(directory.iterdir())
    changed = False
    for temporary in entries:
        temporary_match = _EXCLUSIVE_WRITE_TEMP_PATTERN.fullmatch(temporary.name)
        if not temporary_match:
            continue
        if _path_is_link(temporary) or not temporary.is_file():
            raise PromotionError(f"{label} contains an invalid temporary")
        temporary_stat = temporary.stat()
        matching: list[Path] = []
        for candidate in entries:
            if candidate == temporary or _EXCLUSIVE_WRITE_TEMP_PATTERN.fullmatch(
                candidate.name
            ):
                continue
            if _path_is_link(candidate) or not candidate.is_file():
                continue
            candidate_stat = candidate.stat()
            if (
                candidate_stat.st_dev == temporary_stat.st_dev
                and candidate_stat.st_ino == temporary_stat.st_ino
            ):
                matching.append(candidate)
        if not matching and int(temporary_stat.st_nlink) == 1:
            if _exclusive_writer_may_be_alive(temporary_match):
                raise PromotionBusyError(f"{label} has an active concurrent writer")
            _unlink_with_retry(temporary)
            changed = True
            continue
        if len(matching) != 1 or int(temporary_stat.st_nlink) != 2:
            raise PromotionError(f"{label} contains ambiguous hardlink residue")
        _unlink_with_retry(temporary)
        changed = True
        _validate_existing_immutable_file(matching[0], label=label)
    if changed:
        _fsync_directory(directory)
    for artifact in directory.iterdir():
        if _EXCLUSIVE_WRITE_TEMP_PATTERN.fullmatch(artifact.name):
            raise PromotionError(f"{label} contains unrecovered temporary residue")
        if artifact.suffix != ".json":
            raise PromotionError(f"{label} contains an unexpected entry")
        _validate_existing_immutable_file(artifact, label=label)


def _recover_exclusive_write_links(path: Path) -> None:
    try:
        destination_stat = path.stat()
    except OSError as exc:
        raise PromotionError(f"cannot inspect immutable artifact: {path}") from exc
    recovered = False
    for temporary in path.parent.glob(".pfc-*.tmp"):
        if not _EXCLUSIVE_WRITE_TEMP_PATTERN.fullmatch(temporary.name):
            continue
        if _path_is_link(temporary) or not temporary.is_file():
            continue
        try:
            temporary_stat = temporary.stat()
        except OSError:
            continue
        if (
            temporary_stat.st_dev != destination_stat.st_dev
            or temporary_stat.st_ino != destination_stat.st_ino
        ):
            continue
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            raise PromotionError(
                f"cannot recover immutable artifact temporary: {temporary}"
            ) from exc
        recovered = True
    if recovered:
        _fsync_directory(path.parent)


def _required_env_key_path(name: str) -> Path:
    selected = os.environ.get(name)
    if not selected:
        raise PromotionError(f"{name} is required")
    path = _absolute_lexical_path(selected)
    _assert_path_has_no_links(path, label=f"configured key for {name}")
    if not path.is_file():
        raise PromotionError(f"configured key is unavailable for {name}: {path}")
    return path


def _matching_trusted_public_key(
    document: Mapping[str, object],
    *,
    primary_path: Path,
    keyring_directory_env: str,
    label: str,
    allow_historical: bool = False,
) -> Path:
    signature = document.get("signature")
    if not isinstance(signature, Mapping):
        raise PromotionError(f"{label} signature is missing")
    key_id = str(signature.get("key_id", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", key_id):
        raise PromotionError(f"{label} signing key_id is invalid")
    trusted = _trusted_public_key_identities(
        primary_path,
        keyring_directory_env=keyring_directory_env,
        label=label,
    )
    if not allow_historical:
        primary_id = public_key_id_from_public_key_path(primary_path.resolve())
        trusted = {primary_id: trusted[primary_id]}
    selected = trusted.get(key_id)
    if selected is None:
        raise PromotionError(f"{label} signing key is not trusted by the configured keyring")
    return selected


def _trusted_public_key_identities(
    primary_path: Path,
    *,
    keyring_directory_env: str,
    label: str,
) -> dict[str, Path]:
    candidates = [primary_path]
    configured_directory = str(os.environ.get(keyring_directory_env, "")).strip()
    if configured_directory:
        directory = _absolute_lexical_path(configured_directory)
        immutable_entries = process_immutable_directory_entries(directory)
        if immutable_entries is None:
            _assert_path_has_no_links(
                directory, label=f"{keyring_directory_env} directory"
            )
            if not directory.is_absolute() or not directory.is_dir():
                raise PromotionError(f"{keyring_directory_env} is unavailable")
            candidates.extend(sorted(directory.glob("*.pem")))
        else:
            candidates.extend(immutable_entries)
    trusted: dict[str, Path] = {}
    for candidate in candidates:
        frozen = process_immutable_file_bytes(candidate)
        if frozen is None:
            _assert_path_has_no_links(candidate, label=f"{label} keyring entry")
            resolved = candidate.resolve()
            if not resolved.is_file():
                raise PromotionError(f"{label} keyring entry is not a regular file")
        else:
            resolved = candidate
        candidate_id = public_key_id_from_public_key_path(resolved)
        trusted.setdefault(candidate_id, resolved)
    return trusted


def _absolute_lexical_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise PromotionError("authority key paths must be absolute")
    return Path(os.path.abspath(path))


def _assert_path_has_no_links(path: Path, *, label: str) -> None:
    for component in (path, *path.parents):
        try:
            is_link = _path_is_link(component)
        except OSError as exc:
            raise PromotionError(f"cannot validate {label}") from exc
        if is_link:
            raise PromotionError(f"{label} cannot contain a link")


def _aware_datetime(value: object, *, label: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise PromotionError(f"{label} is invalid") from exc
    if timestamp.tzinfo is None:
        raise PromotionError(f"{label} is timezone-naive")
    return timestamp.astimezone(timezone.utc)


def _promotion_now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _write_bytes_exclusive_fsync(path: Path, payload: bytes) -> None:
    _write_bytes_exclusive_fsync_with_post_link(path, payload)


def write_immutable_bytes(path: Path, payload: bytes) -> None:
    """Publish exact bytes and prove that the destination has one link."""

    _write_bytes_exclusive_fsync_with_post_link(path, payload)
    _validate_existing_immutable_file(path, label="immutable artifact")


def _write_bytes_exclusive_fsync_with_post_link(
    path: Path,
    payload: bytes,
    *,
    post_link: Callable[[Path, Path], None] | None = None,
) -> None:
    """Publish immutable bytes, with an internal deterministic crash-test seam."""

    _mkdir_durable(path.parent)
    pid = os.getpid()
    start = process_start_token(pid) or "unknown"
    if not re.fullmatch(r"[0-9a-f]{16,32}|unknown", start):
        start = hashlib.sha256(start.encode("utf-8")).hexdigest()[:16]
    temporary = path.with_name(
        f".pfc-v2-{_exclusive_writer_host_hash()}-{pid}-{start}-"
        f"{uuid.uuid4().hex[:12]}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ImmutableArtifactExistsError(
                f"immutable artifact already exists: {path}"
            ) from exc
        _fsync_directory(path.parent)
        if post_link is not None:
            post_link(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            raise PromotionError(
                f"cannot remove immutable artifact temporary: {temporary}"
            ) from exc
        else:
            _fsync_directory(temporary.parent)


def _replace_directory(source: Path, destination: Path) -> None:
    for attempt in range(20):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 19:
                raise
            time.sleep(min(0.025 * (attempt + 1), 0.25))


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            with path.open("rb+") as handle:
                os.fsync(handle.fileno())
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


@contextmanager
def _promotion_lock(root: Path) -> Iterator[_PromotionLockState]:
    lock_path = root / LOCK_FILE
    try:
        wait_seconds = float(os.environ.get(PROMOTION_LOCK_WAIT_SECONDS_ENV, "0"))
    except ValueError as exc:
        raise PromotionError(f"{PROMOTION_LOCK_WAIT_SECONDS_ENV} is invalid") from exc
    if not math.isfinite(wait_seconds) or wait_seconds < 0 or wait_seconds > 300:
        raise PromotionError(f"{PROMOTION_LOCK_WAIT_SECONDS_ENV} is out of range")
    host_id = _lock_host_id()
    pid = os.getpid()
    owner = (
        "schema=promotion_lock.v2\n"
        f"host={host_id}\n"
        f"pid={pid}\n"
        f"start={process_start_token(pid) or 'unknown'}\n"
        f"token={uuid.uuid4().hex}\n"
    ).encode("ascii")
    deadline = time.monotonic() + wait_seconds
    while not _publish_lock_exclusive(lock_path, owner):
        if time.monotonic() >= deadline:
            raise PromotionBusyError(f"promotion lock is busy: {lock_path}")
        time.sleep(0.05)
    lock_state = _PromotionLockState()
    primary_error: BaseException | None = None
    try:
        yield lock_state
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            _remove_owned_lock(lock_path, owner)
        except BaseException as cleanup_error:
            if isinstance(primary_error, PromotionProjectionRepairRequired):
                expected_event_id = lock_state.expected_journal_event_id
                if (
                    expected_event_id is not None
                    and primary_error.event_id != expected_event_id
                ):
                    raise PromotionError(
                        "repair-required event does not match lock-bound journal event"
                    ) from primary_error
                primary_error.lock_cleanup_status = "FAIL"
                primary_error.lock_cleanup_error_type = type(cleanup_error).__name__
            elif primary_error is None:
                event_id = lock_state.expected_journal_event_id
                if event_id is None:
                    raise PromotionError(
                        "promotion lock cleanup failed after a non-journal operation"
                    ) from cleanup_error
                commit_status = "INDETERMINATE"
                try:
                    history = _load_immutable_journal_history(root)
                    if history and str(history[-1]["event_id"]) == event_id:
                        commit_status = "COMMITTED"
                except BaseException:
                    pass
                repair = PromotionProjectionRepairRequired(
                    event_id,
                    commit_status=commit_status,
                )
                repair.lock_cleanup_status = "FAIL"
                repair.lock_cleanup_error_type = type(cleanup_error).__name__
                raise repair from cleanup_error
            else:
                primary_error.lock_cleanup_status = "FAIL"
                primary_error.lock_cleanup_error_type = type(cleanup_error).__name__
                if hasattr(primary_error, "add_note"):
                    primary_error.add_note(
                        "promotion lock cleanup also failed with "
                        f"{type(cleanup_error).__name__}"
                    )


def _publish_lock_exclusive(lock_path: Path, owner: bytes) -> bool:
    temporary = lock_path.with_name(
        f".pfc-lock-{os.getpid()}-{uuid.uuid4().hex[:12]}.tmp"
    )
    published = False
    try:
        with temporary.open("xb") as handle:
            handle.write(owner)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, lock_path)
        except FileExistsError:
            _unlink_with_retry(temporary, missing_ok=True)
            return False
        published = True
        _fsync_directory(lock_path.parent)
        _unlink_with_retry(temporary, missing_ok=True)
        return True
    except BaseException as exc:
        cleanup_errors: list[Exception] = []
        try:
            _unlink_with_retry(temporary, missing_ok=True)
        except Exception as cleanup_exc:
            cleanup_errors.append(cleanup_exc)
        if published:
            try:
                _remove_owned_lock(lock_path, owner)
            except Exception as cleanup_exc:
                cleanup_errors.append(cleanup_exc)
        if cleanup_errors:
            raise PromotionError(
                "promotion lock publication failed and cleanup requires repair"
            ) from exc
        raise


def _remove_owned_lock(lock_path: Path, owner: bytes) -> None:
    try:
        current_owner = _read_bytes_with_retry(lock_path)
    except OSError as exc:
        raise PromotionError("promotion lock disappeared while held") from exc
    if current_owner != owner:
        raise PromotionError("promotion lock ownership changed while held")
    try:
        _unlink_with_retry(lock_path)
    except OSError as exc:
        raise PromotionError("cannot remove owned promotion lock") from exc
    _fsync_directory(lock_path.parent)


def _lock_host_id() -> str:
    selected = os.environ.get("PFC_PROMOTION_LOCK_HOST_ID", socket.gethostname())
    canonical = str(selected).strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,127}", canonical):
        raise PromotionError("promotion lock host identity is invalid")
    return canonical


def _exclusive_writer_host_hash() -> str:
    return hashlib.sha256(_lock_host_id().encode("utf-8")).hexdigest()[:12]


def _exclusive_writer_may_be_alive(match: re.Match[str]) -> bool:
    host = match.group("host")
    if host is None:
        return True
    if host != _exclusive_writer_host_hash():
        return True
    pid = int(match.group("pid"))
    if not _pid_is_alive(pid):
        return False
    recorded_start = str(match.group("start"))
    current_start = process_start_token(pid)
    if recorded_start == "unknown" or current_start is None:
        return True
    return current_start == recorded_start


def _pid_is_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            error = int(ctypes.windll.kernel32.GetLastError())  # type: ignore[attr-defined]
            return error != 87  # ERROR_INVALID_PARAMETER means the PID is absent.
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(  # type: ignore[attr-defined]
                handle,
                ctypes.byref(exit_code),
            ):
                return True
            return int(exit_code.value) == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _recover_abandoned_candidate_temporaries(staging: Path) -> None:
    """Remove only recognized atomic-write remnants owned by dead processes."""

    if not staging.is_dir():
        raise PromotionError(f"candidate staging directory is missing: {staging}")
    for path in sorted(staging.rglob("*")):
        if not path.is_file():
            continue
        match = None
        for pattern in _CANDIDATE_TEMP_PATTERNS:
            match = pattern.fullmatch(path.name)
            if match is not None:
                break
        if match is None:
            if ".tmp" in path.name or path.name.endswith(".pending.json"):
                raise PromotionError(f"unrecognized candidate temporary file: {path}")
            continue
        pid = int(match.group("pid"))
        if _pid_is_alive(pid):
            recorded_start = match.group("start")
            active_start = process_start_token(pid)
            if (
                recorded_start == "unknown"
                or active_start is None
                or active_start == recorded_start
            ):
                raise PromotionError(f"candidate write is still active: {path}")
        try:
            path.unlink()
        except OSError as exc:
            raise PromotionError(f"cannot remove abandoned candidate temporary: {path}") from exc


def _hash_bundle_files(bundle: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    root_manifest = (bundle / BUNDLE_MANIFEST).resolve()
    for path in sorted(bundle.rglob("*")):
        if _path_is_link(path):
            raise PromotionError(f"links are forbidden in candidate bundles: {path}")
        resolved = path.resolve()
        if resolved != bundle.resolve() and bundle.resolve() not in resolved.parents:
            raise PromotionError(f"candidate bundle entry escapes bundle root: {path}")
        if not path.is_file():
            continue
        try:
            link_count = int(path.stat().st_nlink)
        except (AttributeError, OSError) as exc:
            raise PromotionError(f"cannot validate candidate bundle file: {path}") from exc
        if link_count != 1:
            raise PromotionError(f"hardlinks are forbidden in candidate bundles: {path}")
        if path.resolve() == root_manifest:
            continue
        relative = path.relative_to(bundle).as_posix()
        files[relative] = _sha256_file(path)
    return files


def _validate_candidate_run_manifest(bundle: Path, *, expected_run_id: str) -> dict[str, object]:
    path = bundle / CANDIDATE_RUN_MANIFEST
    if not path.is_file():
        raise PromotionError(f"candidate run manifest is missing: {path}")
    manifest = _load_json(path)
    errors: list[str] = []
    if manifest.get("schema_version") != "lt_candidate_run.v1":
        errors.append("schema_version")
    run_id = manifest.get("run_id")
    if type(run_id) is not str or run_id != expected_run_id:
        errors.append("run_id")
    if manifest.get("pipeline_scope") != "LT_ONLY":
        errors.append("pipeline_scope")
    if manifest.get("promotion_eligible") is not False:
        errors.append("candidate_must_not_self_approve")
    if manifest.get("reference_timestamp_is_explicit") is not True:
        errors.append("reference_timestamp_is_explicit")
    for key in (
        "reference_timestamp",
        "run_started_at_utc",
        "candidate_serialized_at_utc",
    ):
        value = manifest.get(key)
        if type(value) is not str:
            errors.append(key)
            continue
        try:
            timestamp = datetime.fromisoformat(value)
        except ValueError:
            errors.append(key)
        else:
            if timestamp.tzinfo is None:
                errors.append(key)
    if errors:
        raise PromotionError(
            "candidate run manifest contract mismatch: " + ",".join(sorted(set(errors)))
        )
    try:
        from pfc_shaping.pipeline.candidate_bundle import (
            verify_deterministic_artifact_inventory,
        )

        verify_deterministic_artifact_inventory(bundle, manifest)
    except RuntimeError as exc:
        raise PromotionError("candidate deterministic artifacts are invalid") from exc
    return manifest


def _release_root(value: str | Path, *, create: bool = True) -> Path:
    selected = Path(value).expanduser()
    if not selected.is_absolute():
        raise PromotionError("release root must be absolute")
    root = Path(os.path.abspath(selected))
    _assert_path_has_no_links(root, label="release root")
    if create:
        _mkdir_durable(root)
    elif not root.is_dir():
        raise PromotionError("release root is unavailable")
    _assert_path_has_no_links(root, label="release root")
    return root


def _candidates_root(root: Path, *, create: bool) -> Path:
    candidates = root / "candidates"
    if _path_is_link(candidates):
        raise PromotionError("release candidates root cannot be a link")
    if create:
        _mkdir_durable(candidates)
    if not candidates.is_dir():
        raise PromotionError("release candidates root is unavailable")
    resolved = candidates.resolve()
    if resolved != root / "candidates":
        raise PromotionError("release candidates root escapes release root")
    return resolved


def _resolve_within(parent: Path, value: str | Path) -> Path:
    parent = parent.resolve()
    resolved = Path(value).resolve()
    if resolved != parent and parent not in resolved.parents:
        raise PromotionError(f"path escapes release root: {resolved}")
    return resolved


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_RE.fullmatch(str(run_id)):
        raise PromotionError(f"invalid run_id: {run_id!r}")


def _load_json(path: Path) -> dict[str, object]:
    try:
        raw = _read_bytes_with_retry(path)
    except OSError as exc:
        raise PromotionError(f"cannot read JSON artifact {path}: {exc}") from exc
    return _load_json_bytes(raw, path)


def _read_bytes_with_retry(path: Path) -> bytes:
    """Retry transient Windows sharing failures around atomic replacement."""

    for attempt in range(40):
        try:
            return path.read_bytes()
        except PermissionError:
            if os.name != "nt" or attempt == 39:
                raise
            time.sleep(min(0.0025 * (attempt + 1), 0.05))
    raise AssertionError("unreachable")


def _unlink_with_retry(path: Path, *, missing_ok: bool = False) -> None:
    """Retry transient Windows sharing failures without losing ownership checks."""

    for attempt in range(40):
        try:
            path.unlink(missing_ok=missing_ok)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 39:
                raise
            time.sleep(min(0.0025 * (attempt + 1), 0.05))
    raise AssertionError("unreachable")


def unlink_with_retry(path: Path, *, missing_ok: bool = False) -> None:
    """Public runtime wrapper for bounded Windows-safe cleanup."""

    _unlink_with_retry(path, missing_ok=missing_ok)


def _load_json_bytes(raw: bytes, path: Path) -> dict[str, object]:
    try:
        value = load_strict_json(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise PromotionError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PromotionError(f"JSON artifact is not an object: {path}")
    return value


def _atomic_write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    deadline: float | None = None,
) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        _write_json_fsync(temp, payload)
        _replace_file(temp, path, deadline=deadline)
        _fsync_directory(path.parent)
    finally:
        _unlink_with_retry(temp, missing_ok=True)


def _replace_file(
    source: Path,
    destination: Path,
    *,
    deadline: float | None = None,
) -> None:
    """Retry transient Windows sharing violations without exposing partial files."""

    for attempt in range(40):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 39:
                raise
            delay = min(0.0025 * (attempt + 1), 0.05)
            if deadline is not None and time.monotonic() + delay >= deadline:
                raise TimeoutError("atomic file replacement exceeded its deadline")
            time.sleep(delay)


def _write_json_fsync(path: Path, payload: Mapping[str, object]) -> None:
    _mkdir_durable(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            default=str,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    for directory in reversed(missing):
        if not directory.is_dir():
            raise PromotionError(f"durable directory creation failed: {directory}")
        _fsync_directory(directory)
        _fsync_directory(directory.parent)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

"""Fail-closed publication of immutable governed LT input snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from pfc_shaping.data.acquisition_contract import (
    PROVIDER_RAW_LT_INPUT_SNAPSHOT_SCHEMA,
    REPLAY_GOVERNED_LT_INPUT_ROLES,
)
from pfc_shaping.data.lt_input_sources import (
    validate_governed_lt_snapshot_bundle,
    validate_lt_input_contract_semantics,
)
from pfc_shaping.data.shared_data_root import (
    is_link_or_junction,
    resolve_confined_path,
    resolve_fmv_data_root,
)
from pfc_shaping.data.snapshot_publication_contract import (
    BOOTSTRAP_REQUIRED_SOURCE_CLASS,
    BOOTSTRAP_TRANSITION_TYPE,
    GOVERNED_AUTHORITY_TRUST_ENVIRONMENTS,
    assert_disjoint_publication_authority_ids,
    configured_publication_bootstrap_keyring_directory,
    publication_domain_sha256,
    publication_private_key_id,
    publication_public_key_id,
    publication_trusted_key_ids,
    required_publication_bootstrap_public_key_path,
    required_publication_private_key_path,
    required_publication_public_key_path,
    required_publication_request_private_key_path,
    required_publication_request_public_key_path,
    sign_snapshot_publication_intent,
    verify_snapshot_bootstrap_intent_authorization,
)
from pfc_shaping.data.snapshot_publication_state import (
    EXTERNAL_POINTER_SCHEMA,
    _legacy_load_publication_anchor_history_for_tests,
    _legacy_pointer_mapping_from_event_for_tests,
    _legacy_publication_anchor_directory_for_tests,
    _legacy_read_publication_event_for_tests,
    _legacy_verify_pointer_against_anchor_for_tests,
    _legacy_verify_publication_projection_for_tests,
    canonical_json,
    pointer_mapping_from_external_receipt,
    verify_external_publication_evidence,
)
from pfc_shaping.durable_artifact import (
    DurableArtifactExistsError as ImmutableArtifactExistsError,
)
from pfc_shaping.durable_artifact import (
    DurableArtifactRepairRequired,
    durable_artifact_durability_classification,
    write_durable_exact_bytes,
)
from pfc_shaping.durable_artifact import (
    prepare_durable_json_directory as prepare_immutable_directory,
)
from pfc_shaping.durable_artifact import (
    validate_durable_file as validate_existing_immutable_file,
)
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    freeze_process_immutable_directory,
    read_stable_single_link_file,
    register_process_immutable_file_bytes,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PORTABLE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_OTHER_AUTHORITY_TRUST_ENVS = {
    role: environment
    for role, environment in GOVERNED_AUTHORITY_TRUST_ENVIRONMENTS.items()
    if not role.startswith("publication_")
}
_CAPTURED_TRUST_SNAPSHOTS: list[tempfile.TemporaryDirectory[str]] = []
_FORBIDDEN_PUBLISHER_PRIVATE_KEY_ENVS = (
    "PFC_DATA_PUBLICATION_BOOTSTRAP_SIGNING_PRIVATE_KEY_PATH",
    "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH",
    "PFC_DATA_TIMESTAMP_SIGNING_PRIVATE_KEY_PATH",
    "PFC_DATA_JOURNAL_SIGNING_PRIVATE_KEY_PATH",
    "PFC_PROMOTION_SIGNING_PRIVATE_KEY_PATH",
    "PFC_PROMOTION_EVENT_SIGNING_PRIVATE_KEY_PATH",
    "PFC_ROLLBACK_AUTHORIZATION_SIGNING_PRIVATE_KEY_PATH",
    "PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH",
    "PFC_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH",
    "PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH",
)


class SnapshotPublicationConflict(RuntimeError):
    """Raised when immutable state or the expected current pointer diverges."""


class SnapshotPublicationBusy(RuntimeError):
    """Raised when another publisher owns the LT publication lock."""


class SnapshotPublicationRepairRequired(RuntimeError):
    """Raised after external commit when the local projection still needs repair."""

    def __init__(self, result: Mapping[str, object]) -> None:
        self.result = dict(result)
        super().__init__(
            "external CAS is committed but local publication evidence requires repair"
        )


class SnapshotPublicationRetired(RuntimeError):
    """Raised when the rejected filesystem publication API is invoked."""


GENERATION_INVENTORY_SCHEMA = "lt_snapshot_generation_inventory.v1"
_EXTERNAL_ANCHOR_PRIVATE_KEY_ENV = "PFC_DATA_PUBLICATION_ANCHOR_SIGNING_PRIVATE_KEY_PATH"


def committed_snapshot_repair_result(
    *,
    command: str,
    intent: Mapping[str, object],
    receipt: Mapping[str, object],
    receipt_payload: bytes,
    receipt_path: Path,
    projection_status: str,
    retry_disposition: str,
    error: Exception,
) -> dict[str, object]:
    """Build the canonical operational result after an externally proven commit."""

    result: dict[str, object] = {
        "schema_version": "lt_snapshot_publication_cli_result.v1",
        "status": "COMMITTED_REPAIR_REQUIRED",
        "command": command,
        "commit_status": "COMMITTED",
        "projection_status": projection_status,
        "retry_disposition": retry_disposition,
        "operation_id": intent["operation_id"],
        "intent_id": intent["intent_id"],
        "anchor_receipt_id": receipt["receipt_id"],
        "anchor_sequence": receipt["sequence"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "local_durability": durable_artifact_durability_classification(),
        "production_authorization": False,
    }
    if isinstance(error, DurableArtifactRepairRequired):
        result["local_repair"] = error.result
    return result


def prepare_governed_lt_snapshot(
    *,
    source_bundle: str | Path | None,
    data_root: str | Path,
    operation_id: str,
    operation_created_at_utc: str,
    expected_anchor_receipt_id: str | None,
    expected_anchor_sequence: int,
    adopt_current_legacy: bool = False,
    expected_legacy_pointer_sha256: str | None = None,
    allow_empty_genesis: bool = False,
    bootstrap_authorization_path: str | Path | None = None,
) -> dict[str, object]:
    """Prepare a generation and intent without changing the live pointer."""

    operation_id = _canonical_operation_id(operation_id)
    if adopt_current_legacy:
        raise SnapshotPublicationConflict(
            "ADOPT_LEGACY is forbidden; use an independently authorized BOOTSTRAP"
        )
    if allow_empty_genesis:
        raise SnapshotPublicationConflict(
            "normal publication cannot create genesis; an IT-authorized BOOTSTRAP is required"
        )
    root = _publication_data_root(data_root)
    request_private = required_publication_request_private_key_path()
    request_public = required_publication_request_public_key_path()
    if publication_private_key_id(request_private) != publication_public_key_id(request_public):
        raise ValueError("publication request private key does not match trusted public key")
    _assert_external_request_authority_isolated(request_public)
    expected_anchor_receipt_id = _validate_anchor_expectation(
        expected_anchor_receipt_id,
        expected_anchor_sequence,
    )
    view = _publication_view(root)
    current = view / "current.json"
    current_payload = (
        _read_stable_regular_file(current, label="current LT pointer") if current.exists() else None
    )
    current_pointer = _strict_json_mapping(current_payload, label="current LT pointer")
    legacy_pointer_sha256: str | None = None
    bootstrap_authorization: dict[str, object] | None = None
    bootstrap_authorization_payload: bytes | None = None
    bootstrap_authorization_id: str | None = None
    bootstrap_authorization_sha256: str | None = None

    if bootstrap_authorization_path is not None:
        if source_bundle is not None:
            raise ValueError("legacy bootstrap cannot consume a source bundle")
        if current_payload is None or current_pointer is None:
            raise ValueError("legacy bootstrap requires an existing LT pointer")
        if (
            current_pointer.get("schema_version") != "lt_data_pointer.v1"
            or current_pointer.get("publication_event_id") is not None
        ):
            raise ValueError("legacy bootstrap requires an ungoverned v1 pointer")
        legacy_pointer_sha256 = hashlib.sha256(current_payload).hexdigest()
        if expected_legacy_pointer_sha256 is not None and legacy_pointer_sha256 != (
            _validated_sha256(
                expected_legacy_pointer_sha256,
                label="expected legacy pointer SHA-256",
            )
        ):
            raise SnapshotPublicationConflict("legacy bootstrap pointer SHA-256 diverged")
        if expected_anchor_sequence != 0 or expected_anchor_receipt_id is not None:
            raise SnapshotPublicationConflict("legacy bootstrap requires an empty external head")
        generation_id, contract_payload, contract, final, bound = _legacy_generation(
            root,
            current_pointer,
        )
        if (
            contract.get("source_class") != BOOTSTRAP_REQUIRED_SOURCE_CLASS
            or contract.get("calibration_eligible") is not False
        ):
            raise ValueError(
                "legacy bootstrap must preserve MIGRATED_UNVERIFIED and "
                "calibration_eligible=false"
            )
        bootstrap_authorization_payload = _stable_external_file(
            bootstrap_authorization_path,
            label="IT bootstrap authorization",
        )
        bootstrap_authorization = _strict_json_mapping(
            bootstrap_authorization_payload,
            label="IT bootstrap authorization",
        )
        assert bootstrap_authorization is not None
        transition_type = BOOTSTRAP_TRANSITION_TYPE
        resumed = True
    else:
        if source_bundle is None:
            raise ValueError("normal publication requires a source bundle")
        _validate_normal_prepare_head(
            current_pointer,
            expected_anchor_receipt_id=expected_anchor_receipt_id,
            expected_anchor_sequence=expected_anchor_sequence,
        )
        source = _absolute_unlinked_directory(source_bundle, label="source bundle")
        contract_payload = _read_stable_regular_file(
            source / "lt_input_snapshot.json",
            label="source contract",
        )
        contract = _strict_json_mapping(contract_payload, label="source contract")
        assert contract is not None
        validate_governed_lt_snapshot_bundle(source, contract)
        generation_id = _portable_id(contract.get("generation_id"), label="generation_id")
        bound = _bound_files(source, contract, contract_payload)
        _validate_exact_bundle_inventory(source, bound)
        final, resumed = _install_prepared_generation(
            root,
            source=source,
            generation_id=generation_id,
            bound=bound,
            operation_id=operation_id,
        )
        transition_type = "PUBLISH"

    _, inventory_payload = _generation_inventory(
        final,
        generation_id=generation_id,
        bound=bound,
    )
    inventory_sha256 = hashlib.sha256(inventory_payload).hexdigest()
    inventory_path = view / "inventories" / f"{inventory_sha256}.json"
    contract_sha256 = hashlib.sha256(contract_payload).hexdigest()
    if bootstrap_authorization is not None:
        bootstrap_authorization_id = _validated_sha256(
            bootstrap_authorization.get("authorization_id"),
            label="bootstrap authorization id",
        )
        assert bootstrap_authorization_payload is not None
        if canonical_json(bootstrap_authorization) != bootstrap_authorization_payload:
            raise ValueError("IT bootstrap authorization must be canonical JSON")
        bootstrap_authorization_sha256 = hashlib.sha256(
            bootstrap_authorization_payload
        ).hexdigest()
    intent = sign_snapshot_publication_intent(
        {
            "schema_version": "lt_snapshot_publication_intent.v1",
            "publication_domain_sha256": publication_domain_sha256(),
            "operation_id": operation_id,
            "operation_created_at_utc": operation_created_at_utc,
            "transition_type": transition_type,
            "expected_anchor_receipt_id": expected_anchor_receipt_id,
            "expected_anchor_sequence": expected_anchor_sequence,
            "generation_id": generation_id,
            "contract_path": (
                Path("snapshots") / generation_id / "lt_input_snapshot.json"
            ).as_posix(),
            "contract_sha256": contract_sha256,
            "generation_inventory_sha256": inventory_sha256,
            "legacy_pointer_sha256": legacy_pointer_sha256,
            "bootstrap_authorization": bootstrap_authorization,
            "bootstrap_authorization_id": bootstrap_authorization_id,
            "bootstrap_authorization_sha256": bootstrap_authorization_sha256,
        },
        private_key_path=request_private,
        bootstrap_trusted_public_key_path=(
            required_publication_bootstrap_public_key_path()
            if bootstrap_authorization is not None
            else None
        ),
        bootstrap_trusted_public_key_dir=(
            configured_publication_bootstrap_keyring_directory()
            if bootstrap_authorization is not None
            else None
        ),
    )
    if bootstrap_authorization is not None:
        verify_snapshot_bootstrap_intent_authorization(
            intent,
            trusted_public_key_path=required_publication_bootstrap_public_key_path(),
            trusted_public_key_dir=configured_publication_bootstrap_keyring_directory(),
            admitted_at_utc=_reference_utc(),
        )
    intent_payload = canonical_json(intent)
    intent_id = str(intent["intent_id"])
    intent_path = view / "intents" / f"{intent_id}.json"
    _write_immutable_or_identical(inventory_path, inventory_payload)
    _write_immutable_or_identical(intent_path, intent_payload)
    if bootstrap_authorization_payload is not None:
        _write_immutable_or_identical(
            view
            / "bootstrap-authorizations"
            / f"{bootstrap_authorization_id}.json",
            bootstrap_authorization_payload,
        )
    operation_path = view / "operations" / f"{operation_id}.json"
    _write_immutable_or_identical(
        operation_path,
        canonical_json(
            {
                "schema_version": "lt_snapshot_publication_operation.v1",
                "operation_id": operation_id,
                "intent_id": intent_id,
                "intent_sha256": hashlib.sha256(intent_payload).hexdigest(),
                "bootstrap_authorization_id": bootstrap_authorization_id,
                "bootstrap_authorization_sha256": bootstrap_authorization_sha256,
            }
        ),
    )
    return {
        "schema_version": "lt_snapshot_publication_cli_result.v1",
        "status": "PREPARED_NOT_COMMITTED",
        "command": "prepare",
        "commit_status": "NOT_COMMITTED",
        "projection_status": "GENERATION_AND_INTENT_PREPARED",
        "retry_disposition": "SUBMIT_EXACT_INTENT_TO_CAS",
        "operation_id": operation_id,
        "intent_id": intent_id,
        "intent_path": str(intent_path),
        "intent_sha256": hashlib.sha256(intent_payload).hexdigest(),
        "generation_id": generation_id,
        "generation_path": str(final),
        "generation_inventory_path": str(inventory_path),
        "generation_inventory_sha256": inventory_sha256,
        "resumed_existing_generation": resumed,
        "bootstrap_authorization_id": bootstrap_authorization_id,
        "local_durability": durable_artifact_durability_classification(),
        "production_authorization": False,
    }


def finalize_governed_lt_snapshot(
    *,
    data_root: str | Path,
    intent_path: str | Path,
    anchor_receipt_path: str | Path,
    head_observation_path: str | Path,
    expected_head_challenge_nonce: str,
) -> dict[str, object]:
    """Project current.json only after external CAS and fresh HEAD evidence."""

    root = _publication_data_root(data_root)
    view = _publication_view(root)
    intent_payload = _stable_external_file(intent_path, label="publication intent")
    receipt_payload = _stable_external_file(
        anchor_receipt_path,
        label="external anchor receipt",
    )
    observation_payload = _stable_external_file(
        head_observation_path,
        label="external anchor head observation",
    )
    intent = _strict_json_mapping(intent_payload, label="publication intent")
    receipt = _strict_json_mapping(receipt_payload, label="external anchor receipt")
    assert intent is not None and receipt is not None
    pointer = pointer_mapping_from_external_receipt(
        intent,
        receipt,
        receipt_sha256=hashlib.sha256(receipt_payload).hexdigest(),
    )
    evidence = verify_external_publication_evidence(
        intent_payload=intent_payload,
        receipt_payload=receipt_payload,
        observation_payload=observation_payload,
        pointer=pointer,
        require_current=True,
        expected_challenge_nonce=expected_head_challenge_nonce,
    )
    try:
        return _finalize_verified_external_publication(
            root=root,
            view=view,
            intent=intent,
            receipt=receipt,
            pointer=pointer,
            evidence=evidence,
            intent_payload=intent_payload,
            receipt_payload=receipt_payload,
            observation_payload=observation_payload,
        )
    except SnapshotPublicationRepairRequired:
        raise
    except Exception as exc:
        raise SnapshotPublicationRepairRequired(
            committed_snapshot_repair_result(
                command="finalize",
                intent=intent,
                receipt=receipt,
                receipt_payload=receipt_payload,
                receipt_path=Path(anchor_receipt_path).expanduser(),
                projection_status="LOCAL_PROJECTION_REPAIR_REQUIRED",
                retry_disposition="RETRY_FINALIZE_WITH_FRESH_HEAD",
                error=exc,
            )
        ) from exc


def _finalize_verified_external_publication(
    *,
    root: Path,
    view: Path,
    intent: Mapping[str, object],
    receipt: Mapping[str, object],
    pointer: Mapping[str, object],
    evidence: Mapping[str, object],
    intent_payload: bytes,
    receipt_payload: bytes,
    observation_payload: bytes,
) -> dict[str, object]:
    generation_id = str(evidence["generation_id"])
    inventory_sha256 = str(pointer["generation_inventory_sha256"])
    inventory_path = view / "inventories" / f"{inventory_sha256}.json"
    inventory_payload = _read_stable_regular_file(
        inventory_path,
        label="generation inventory",
    )
    if hashlib.sha256(inventory_payload).hexdigest() != inventory_sha256:
        raise SnapshotPublicationConflict("generation inventory hash mismatch")
    bound = _bound_from_inventory(
        _strict_json_mapping(inventory_payload, label="generation inventory"),
        expected_generation_id=generation_id,
    )
    final = root / "snapshots" / generation_id
    _verify_existing_generation(final, bound)
    contract_payload = _read_stable_regular_file(
        final / "lt_input_snapshot.json",
        label="prepared LT contract",
    )
    if hashlib.sha256(contract_payload).hexdigest() != pointer["contract_sha256"]:
        raise SnapshotPublicationConflict("prepared LT contract hash mismatch")
    if intent.get("transition_type") == BOOTSTRAP_TRANSITION_TYPE:
        contract = _strict_json_mapping(contract_payload, label="bootstrapped LT contract")
        assert contract is not None
        if (
            contract.get("source_class") != BOOTSTRAP_REQUIRED_SOURCE_CLASS
            or contract.get("calibration_eligible") is not False
        ):
            raise SnapshotPublicationConflict(
                "bootstrapped generation became calibration eligible or changed source class"
            )
        authorization = intent.get("bootstrap_authorization")
        if not isinstance(authorization, Mapping):
            raise SnapshotPublicationConflict("bootstrap authorization is unavailable")
        authorization_payload = canonical_json(authorization)
        if hashlib.sha256(authorization_payload).hexdigest() != intent.get(
            "bootstrap_authorization_sha256"
        ):
            raise SnapshotPublicationConflict("bootstrap authorization hash mismatch")
        _write_immutable_or_identical(
            view
            / "bootstrap-authorizations"
            / f"{intent['bootstrap_authorization_id']}.json",
            authorization_payload,
        )

    _write_immutable_or_identical(
        view / "intents" / f"{intent['intent_id']}.json",
        intent_payload,
    )
    _write_immutable_or_identical(
        view / "receipts" / f"{receipt['receipt_id']}.json",
        receipt_payload,
    )
    _write_immutable_or_identical(
        view / "observations" / f"{evidence['observation_id']}.json",
        observation_payload,
    )
    pointer_payload = canonical_json(pointer)
    current = view / "current.json"
    lock_handle = None
    try:
        lock_handle = _acquire_file_lock(view / ".publish.lock")
        current_payload = (
            _read_stable_regular_file(current, label="current LT pointer")
            if current.exists()
            else None
        )
        current_pointer = _strict_json_mapping(current_payload, label="current LT pointer")
        if current_payload == pointer_payload:
            status = "ALREADY_FINALIZED"
        else:
            _validate_finalize_predecessor(
                current_payload=current_payload,
                current_pointer=current_pointer,
                intent=intent,
                receipt=receipt,
            )
            _write_atomic(current, pointer_payload)
            if _read_stable_regular_file(current, label="finalized LT pointer") != pointer_payload:
                raise RuntimeError("finalized LT pointer bytes diverged")
            status = "FINALIZED"
    finally:
        _release_file_lock(lock_handle)
    return {
        "schema_version": "lt_snapshot_publication_cli_result.v1",
        "status": status,
        "command": "finalize",
        "commit_status": "COMMITTED",
        "projection_status": status,
        "retry_disposition": "NONE",
        "operation_id": intent["operation_id"],
        "intent_id": intent["intent_id"],
        "anchor_receipt_id": receipt["receipt_id"],
        "anchor_sequence": receipt["sequence"],
        "pointer_path": str(current),
        "pointer_sha256": hashlib.sha256(pointer_payload).hexdigest(),
        "bootstrap_authorization_id": intent.get("bootstrap_authorization_id"),
        "local_durability": durable_artifact_durability_classification(),
        "production_authorization": False,
    }


def publish_governed_lt_snapshot(
    *,
    source_bundle: str | Path,
    data_root: str | Path,
    expected_current_event_id: str | None,
    operation_id: str,
) -> dict[str, object]:
    """Reject the retired filesystem publication prototype."""

    del source_bundle, data_root, expected_current_event_id, operation_id
    raise SnapshotPublicationRetired(
        "filesystem publication is retired; use PREPARE -> external CAS -> FINALIZE"
    )


def _publish_filesystem_prototype_for_tests(
    *,
    source_bundle: str | Path,
    data_root: str | Path,
    expected_current_event_id: str | None,
    operation_id: str,
) -> dict[str, object]:
    """Verify, copy and CAS-publish one pre-signed v2 acquisition bundle.

    ``None`` means the caller explicitly expects no committed event. The
    publisher never signs acquisition evidence and therefore needs no private
    acquisition or trusted-time key.
    """

    from pfc_shaping.data.snapshot_legacy_publication_signing import (
        sign_snapshot_publication_anchor,
        sign_snapshot_publication_event,
    )

    source = _absolute_unlinked_directory(source_bundle, label="source bundle")
    lexical_data_root = assert_absolute_path_has_no_links(Path(data_root).expanduser())
    root = resolve_fmv_data_root(lexical_data_root, require_exists=True)
    if is_link_or_junction(root):
        raise ValueError(f"FMV data_root cannot be a link or junction: {root}")
    if expected_current_event_id is not None:
        expected_current_event_id = _validated_sha256(
            expected_current_event_id,
            label="expected current publication event id",
        )
    raw_operation_id = str(operation_id).strip().lower()
    try:
        operation_id = str(uuid.UUID(raw_operation_id))
    except ValueError as exc:
        raise ValueError("publication operation_id must be a canonical UUID") from exc
    if raw_operation_id != operation_id:
        raise ValueError("publication operation_id must be a canonical UUID")
    publication_private_key = required_publication_private_key_path()
    publication_public_key = required_publication_public_key_path()
    if publication_private_key_id(publication_private_key) != publication_public_key_id(
        publication_public_key
    ):
        raise ValueError("publication private key does not match trusted public key")
    _assert_publication_authority_isolated(publication_public_key)
    anchor_directory = _legacy_publication_anchor_directory_for_tests(root, create=True)
    prepare_immutable_directory(anchor_directory, label="publication anchor journal")

    contract_path = source / "lt_input_snapshot.json"
    contract_payload = _read_stable_regular_file(contract_path, label="source contract")
    try:
        contract = load_strict_json(contract_payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("source contract is not strict UTF-8 JSON") from exc
    if not isinstance(contract, Mapping):
        raise ValueError("source contract must be a mapping")
    validate_governed_lt_snapshot_bundle(source, contract)
    generation_id = _portable_id(contract.get("generation_id"), label="generation_id")
    bound = _bound_files(source, contract, contract_payload)
    _validate_exact_bundle_inventory(source, bound)

    snapshots = resolve_confined_path(root, "snapshots", label="snapshots directory")
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshots = resolve_confined_path(
        root,
        snapshots,
        label="snapshots directory",
        require_exists=True,
    )
    final = snapshots / generation_id
    staging = snapshots / f".{generation_id}.staging-{uuid.uuid4().hex}"
    resumed = False
    if final.exists():
        _verify_existing_generation(final, bound)
        resumed = True
    else:
        staging.mkdir()
        try:
            _copy_bound_bundle(source, staging, bound)
            copied_contract = load_strict_json(
                (staging / "lt_input_snapshot.json").read_text(encoding="utf-8")
            )
            if not isinstance(copied_contract, Mapping):
                raise ValueError("copied LT contract must be a mapping")
            validate_governed_lt_snapshot_bundle(staging, copied_contract)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    view = resolve_confined_path(root, "views/pfc_lt", label="LT view directory")
    view.mkdir(parents=True, exist_ok=True)
    view = resolve_confined_path(root, view, label="LT view directory", require_exists=True)
    lock = view / ".publish.lock"
    lock_handle = None
    already_current = False
    projection_repaired = False
    try:
        lock_handle = _acquire_file_lock(lock)
        try:
            current = view / "current.json"
            events = view / "events"
            events.mkdir(exist_ok=True)
            events = resolve_confined_path(
                root,
                events,
                label="LT publication events directory",
                require_exists=True,
            )
            current_sha, current_pointer = _read_current_pointer(current)
            contract_sha = hashlib.sha256(contract_payload).hexdigest()
            anchor_history = _legacy_load_publication_anchor_history_for_tests(root)
            current_revision = int((current_pointer or {}).get("revision", 0))
            if len(anchor_history) not in {current_revision, current_revision + 1}:
                raise SnapshotPublicationConflict(
                    "external publication anchor is not aligned with current pointer"
                )
            if current_pointer is not None:
                if current_revision < 1 or len(anchor_history) < current_revision:
                    raise SnapshotPublicationConflict(
                        "current pointer has no external publication anchor"
                    )
                _legacy_verify_pointer_against_anchor_for_tests(
                    root,
                    current_pointer,
                    pointer_sha256=str(current_sha),
                    anchor=anchor_history[current_revision - 1],
                )
            committed_operation = _committed_operation_event(
                root,
                anchor_history,
                operation_id=operation_id,
            )
            if committed_operation is not None:
                if str(anchor_history[-1]["event_id"]) != str(committed_operation["event_id"]):
                    raise SnapshotPublicationConflict(
                        "publication operation_id is already committed historically"
                    )
                if (
                    committed_operation.get("previous_event_id") != expected_current_event_id
                    or committed_operation.get("generation_id") != generation_id
                    or committed_operation.get("contract_sha256") != contract_sha
                ):
                    raise SnapshotPublicationConflict(
                        "publication operation_id is already committed with different inputs"
                    )
                if not resumed:
                    raise SnapshotPublicationConflict(
                        "committed publication targets a missing immutable generation"
                    )
                retry_pointer = _legacy_pointer_mapping_from_event_for_tests(
                    committed_operation
                )
                retry_payload = canonical_json(retry_pointer)
                retry_sha = hashlib.sha256(retry_payload).hexdigest()
                if dict(current_pointer or {}) != retry_pointer or current_sha != retry_sha:
                    _write_atomic(current, retry_payload)
                    projection_repaired = True
                _legacy_verify_publication_projection_for_tests(
                    root,
                    retry_pointer,
                    pointer_sha256=retry_sha,
                )
                return {
                    "generation_id": generation_id,
                    "snapshot_path": str(final),
                    "pointer_path": str(current),
                    "pointer_sha256": retry_sha,
                    "publication_event_id": committed_operation["event_id"],
                    "resumed_existing_generation": True,
                    "already_current": True,
                    "exact_retry": True,
                    "projection_repaired": projection_repaired,
                    "commit_status": "COMMITTED",
                    "published_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            if len(anchor_history) == current_revision + 1:
                committed_anchor = anchor_history[-1]
                committed_event_id = str(committed_anchor["event_id"])
                committed_event = _legacy_read_publication_event_for_tests(
                    root, committed_event_id
                )
                if (
                    committed_event.get("operation_id") != operation_id
                    or committed_event.get("previous_event_id") != expected_current_event_id
                    or committed_event.get("revision") != current_revision + 1
                    or committed_event.get("generation_id") != generation_id
                    or committed_event.get("contract_sha256") != contract_sha
                ):
                    raise SnapshotPublicationConflict(
                        "committed publication requires a different projection repair"
                    )
                if not resumed:
                    raise SnapshotPublicationConflict(
                        "committed publication targets a missing immutable generation"
                    )
                repaired_pointer = _legacy_pointer_mapping_from_event_for_tests(
                    committed_event
                )
                repaired_payload = canonical_json(repaired_pointer)
                repaired_sha = hashlib.sha256(repaired_payload).hexdigest()
                if committed_anchor.get("pointer_sha256") != repaired_sha:
                    raise SnapshotPublicationConflict(
                        "committed publication anchor/pointer binding mismatch"
                    )
                _write_atomic(current, repaired_payload)
                _legacy_verify_publication_projection_for_tests(
                    root,
                    repaired_pointer,
                    pointer_sha256=repaired_sha,
                )
                pointer_sha = repaired_sha
                event_id = committed_event_id
                projection_repaired = True
            elif (current_pointer or {}).get("publication_event_id") != expected_current_event_id:
                raise SnapshotPublicationConflict(
                    "current committed publication event does not match caller expectation"
                )
            elif (
                current_pointer is not None
                and str(current_pointer.get("generation_id", "")) == generation_id
                and str(current_pointer.get("contract_sha256", "")) == contract_sha
            ):
                if not resumed:
                    raise SnapshotPublicationConflict(
                        "current LT pointer targets a missing immutable generation"
                    )
                pointer_sha = str(current_sha)
                event_id = str(current_pointer.get("publication_event_id", "")) or None
                already_current = True
            else:
                if not resumed:
                    _verify_existing_generation(staging, bound)
                    _replace_directory(staging, final)
                    _verify_existing_generation(final, bound)
                    _fsync_directory(final.parent)
                previous_revision = current_revision
                previous_event_id = (current_pointer or {}).get("publication_event_id")
                event = sign_snapshot_publication_event(
                    {
                        "schema_version": "lt_pointer_publication_event.v3",
                        "publication_domain_sha256": publication_domain_sha256(),
                        "operation_id": operation_id,
                        "revision": previous_revision + 1,
                        "previous_event_id": previous_event_id,
                        "previous_pointer_sha256": current_sha,
                        "generation_id": generation_id,
                        "contract_sha256": contract_sha,
                    },
                    private_key_path=publication_private_key,
                )
                event_id = str(event["event_id"])
                pointer = _legacy_pointer_mapping_from_event_for_tests(event)
                pointer_payload = canonical_json(pointer)
                pointer_sha = hashlib.sha256(pointer_payload).hexdigest()
                event_payload = canonical_json(event)
                _write_immutable_or_identical(events / f"{event_id}.json", event_payload)
                previous_anchor_id = (
                    str(anchor_history[previous_revision - 1]["anchor_id"])
                    if previous_revision
                    else None
                )
                anchor = sign_snapshot_publication_anchor(
                    {
                        "schema_version": "lt_pointer_publication_anchor.v1",
                        "publication_domain_sha256": publication_domain_sha256(),
                        "sequence": previous_revision + 1,
                        "event_id": event_id,
                        "pointer_sha256": pointer_sha,
                        "previous_anchor_id": previous_anchor_id,
                    },
                    private_key_path=publication_private_key,
                )
                anchor_payload = canonical_json(anchor)
                anchor_path = _legacy_publication_anchor_directory_for_tests(
                    root, create=False
                ) / (
                    f"{previous_revision + 1:012d}-{anchor['anchor_id']}.json"
                )
                if len(anchor_history) == previous_revision + 1:
                    existing_anchor = anchor_history[-1]
                    if existing_anchor != {
                        key: value for key, value in anchor.items() if key != "signature"
                    }:
                        raise SnapshotPublicationConflict(
                            "external publication anchor collision during recovery"
                        )
                else:
                    _write_immutable_or_identical(anchor_path, anchor_payload)
                _write_atomic(current, pointer_payload)
                if (
                    _read_stable_regular_file(current, label="published LT pointer")
                    != pointer_payload
                ):
                    raise RuntimeError("published LT pointer bytes diverged after atomic replace")
                _legacy_verify_publication_projection_for_tests(
                    root,
                    pointer,
                    pointer_sha256=pointer_sha,
                )
        finally:
            _release_file_lock(lock_handle)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "generation_id": generation_id,
        "snapshot_path": str(final),
        "pointer_path": str(view / "current.json"),
        "pointer_sha256": pointer_sha,
        "publication_event_id": event_id,
        "resumed_existing_generation": resumed,
        "already_current": already_current,
        "exact_retry": False,
        "projection_repaired": projection_repaired,
        "commit_status": "NO_CHANGE" if already_current else "COMMITTED",
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _committed_operation_event(
    root: Path,
    anchor_history: list[Mapping[str, object]],
    *,
    operation_id: str,
) -> dict[str, object] | None:
    matches = [
        event
        for anchor in anchor_history
        if (
            event := _legacy_read_publication_event_for_tests(
                root, str(anchor["event_id"])
            )
        ).get("operation_id")
        == operation_id
    ]
    if len(matches) > 1:
        raise SnapshotPublicationConflict(
            "publication operation_id appears more than once in committed history"
        )
    return matches[0] if matches else None


def _assert_publication_authority_isolated(public_key: Path) -> None:
    exposed = sorted(
        name
        for name in _FORBIDDEN_PUBLISHER_PRIVATE_KEY_ENVS
        if str(os.environ.get(name, "")).strip()
    )
    if exposed:
        raise ValueError(f"publisher process exposes forbidden private keys: {exposed}")
    publication_id = publication_public_key_id(public_key)
    collisions = []
    for primary_env, directory_env in _OTHER_AUTHORITY_TRUST_ENVS.values():
        value = str(os.environ.get(primary_env, "")).strip()
        directory = (
            None
            if directory_env is None
            else str(os.environ.get(directory_env, "")).strip() or None
        )
        if value and publication_id in publication_trusted_key_ids(
            value,
            keyring_directory=directory,
        ):
            collisions.append(primary_env)
    if collisions:
        raise ValueError(
            f"publication authority overlaps another governed authority: {sorted(collisions)}"
        )


def _canonical_operation_id(value: object) -> str:
    raw = str(value or "").strip().lower()
    try:
        canonical = str(uuid.UUID(raw))
    except ValueError as exc:
        raise ValueError("publication operation_id must be a canonical UUID") from exc
    if raw != canonical:
        raise ValueError("publication operation_id must be a canonical UUID")
    return canonical


def _reference_utc() -> datetime:
    return datetime.now(timezone.utc)


def _publication_data_root(value: str | Path) -> Path:
    lexical = assert_absolute_path_has_no_links(Path(value).expanduser())
    root = resolve_fmv_data_root(lexical, require_exists=True)
    if is_link_or_junction(root):
        raise ValueError(f"FMV data_root cannot be a link or junction: {root}")
    return root


def _publication_view(root: Path) -> Path:
    view = resolve_confined_path(root, "views/pfc_lt", label="LT view directory")
    view.mkdir(parents=True, exist_ok=True)
    return resolve_confined_path(
        root,
        view,
        label="LT view directory",
        require_exists=True,
    )


def assert_snapshot_publisher_runtime_identity(*, phase: str) -> None:
    """Validate the phase identity and capture one globally disjoint trust registry."""

    if phase not in {"prepare", "cas", "finalize"}:
        raise ValueError("snapshot publisher phase is invalid")
    request_private_env = "PFC_DATA_PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_PATH"
    additionally_forbidden = (
        [request_private_env]
        if phase in {"cas", "finalize"}
        and str(os.environ.get(request_private_env, "")).strip()
        else []
    )
    _assert_external_request_authority_isolated(
        required_publication_request_public_key_path(),
        additionally_forbidden_private_key_envs=additionally_forbidden,
    )


def _assert_external_request_authority_isolated(
    request_public_key: Path,
    *,
    additionally_forbidden_private_key_envs: list[str] | None = None,
) -> None:
    obsolete_publication_private_key_env = "PFC_DATA_PUBLICATION_SIGNING_PRIVATE_KEY_PATH"
    exposed = sorted(
        [
            *(
                name
                for name in _FORBIDDEN_PUBLISHER_PRIVATE_KEY_ENVS
                if str(os.environ.get(name, "")).strip()
            ),
            *(
                [_EXTERNAL_ANCHOR_PRIVATE_KEY_ENV]
                if str(os.environ.get(_EXTERNAL_ANCHOR_PRIVATE_KEY_ENV, "")).strip()
                else []
            ),
            *(
                [obsolete_publication_private_key_env]
                if str(os.environ.get(obsolete_publication_private_key_env, "")).strip()
                else []
            ),
            *(additionally_forbidden_private_key_envs or []),
        ]
    )
    if exposed:
        raise ValueError(f"publication request process exposes forbidden private keys: {exposed}")
    if publication_public_key_id(request_public_key) != publication_public_key_id(
        required_publication_request_public_key_path()
    ):
        raise ValueError("publication request trust changed before registry capture")
    trust_environments = GOVERNED_AUTHORITY_TRUST_ENVIRONMENTS
    missing_authorities = sorted(
        primary_env
        for primary_env, _ in trust_environments.values()
        if not str(os.environ.get(primary_env, "")).strip()
    )
    if missing_authorities:
        raise ValueError(
            "publisher trust registry is incomplete: "
            f"{missing_authorities}"
        )
    snapshot = tempfile.TemporaryDirectory(prefix="fmv-pfc-trust-")
    snapshot_root = Path(snapshot.name)
    captured: dict[str, tuple[str, Path, str | None, Path | None]] = {}
    authority_ids: dict[str, frozenset[str]] = {}
    try:
        for role, (primary_env, directory_env) in sorted(trust_environments.items()):
            role_root = snapshot_root / role
            role_root.mkdir(mode=0o700)
            primary = _capture_public_trust_file(
                Path(str(os.environ[primary_env]).strip()),
                role_root / "active.pem",
                label=f"{role} active public key",
            )
            captured_directory = None
            if directory_env is not None:
                source_directory = str(os.environ.get(directory_env, "")).strip()
                if source_directory:
                    captured_directory = _capture_public_trust_keyring(
                        Path(source_directory),
                        role_root / "keyring",
                        label=f"{role} historical public keyring",
                    )
            authority_ids[role] = publication_trusted_key_ids(
                primary,
                keyring_directory=captured_directory,
            )
            captured[role] = (
                primary_env,
                primary,
                directory_env,
                captured_directory,
            )
        assert_disjoint_publication_authority_ids(authority_ids)
    except ValueError as exc:
        snapshot.cleanup()
        raise ValueError("publisher trust registry contains overlapping authorities") from exc
    except BaseException:
        snapshot.cleanup()
        raise
    for primary_env, primary, directory_env, directory in captured.values():
        os.environ[primary_env] = str(primary)
        if directory_env is not None:
            if directory is None:
                os.environ.pop(directory_env, None)
            else:
                os.environ[directory_env] = str(directory)
    _CAPTURED_TRUST_SNAPSHOTS.append(snapshot)


def _capture_public_trust_file(source: Path, target: Path, *, label: str) -> Path:
    payload = read_stable_single_link_file(
        assert_absolute_path_has_no_links(source.expanduser()),
        label=label,
    )
    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return register_process_immutable_file_bytes(target, payload, label=label)


def _capture_public_trust_keyring(source: Path, target: Path, *, label: str) -> Path:
    source = assert_absolute_path_has_no_links(source.expanduser())
    if not source.is_dir():
        raise ValueError(f"{label} is not a directory")
    source_entries = freeze_process_immutable_directory(source, label=label)
    target.mkdir(mode=0o700)
    for index, entry in enumerate(source_entries):
        _capture_public_trust_file(
            entry,
            target / f"{index:04d}.pem",
            label=f"{label} entry",
        )
    freeze_process_immutable_directory(target, label=label)
    return target


def _validate_anchor_expectation(
    receipt_id: str | None,
    sequence: int,
) -> str | None:
    if type(sequence) is not int or sequence < 0:
        raise ValueError("expected_anchor_sequence must be a non-negative integer")
    if sequence == 0:
        if receipt_id is not None:
            raise ValueError("genesis expectation cannot include an anchor receipt")
        return None
    return _validated_sha256(receipt_id, label="expected anchor receipt id")


def _strict_json_mapping(payload: bytes | None, *, label: str) -> dict[str, object] | None:
    if payload is None:
        return None
    try:
        document = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a mapping")
    return document


def _validate_normal_prepare_head(
    current: Mapping[str, object] | None,
    *,
    expected_anchor_receipt_id: str | None,
    expected_anchor_sequence: int,
) -> None:
    if current is None:
        raise ValueError("normal publication requires an existing external predecessor")
    if current.get("schema_version") != EXTERNAL_POINTER_SCHEMA:
        raise ValueError("existing legacy pointer requires an IT-authorized BOOTSTRAP")
    if (
        current.get("anchor_receipt_id") != expected_anchor_receipt_id
        or current.get("anchor_sequence") != expected_anchor_sequence
    ):
        raise SnapshotPublicationConflict(
            "local pointer does not match expected external anchor head"
        )


def _legacy_generation(
    root: Path,
    pointer: Mapping[str, object],
) -> tuple[str, bytes, dict[str, object], Path, dict[str, tuple[str, int | None]]]:
    generation_id = _portable_id(pointer.get("generation_id"), label="legacy generation_id")
    expected_path = (Path("snapshots") / generation_id / "lt_input_snapshot.json").as_posix()
    if pointer.get("contract_path") != expected_path:
        raise ValueError("legacy pointer contract path is not canonical")
    contract_path = resolve_confined_path(
        root,
        expected_path,
        label="legacy LT contract",
        require_exists=True,
        require_file=True,
    )
    contract_payload = _read_stable_regular_file(contract_path, label="legacy LT contract")
    if hashlib.sha256(contract_payload).hexdigest() != pointer.get("contract_sha256"):
        raise SnapshotPublicationConflict("legacy pointer contract hash mismatch")
    contract = _strict_json_mapping(contract_payload, label="legacy LT contract")
    assert contract is not None
    validate_lt_input_contract_semantics(contract)
    if contract.get("generation_id") != generation_id:
        raise ValueError("legacy pointer/contract generation mismatch")
    final = contract_path.parent
    bound = _discover_bound_files(final)
    return generation_id, contract_payload, contract, final, bound


def _install_prepared_generation(
    root: Path,
    *,
    source: Path,
    generation_id: str,
    bound: Mapping[str, tuple[str, int | None]],
    operation_id: str,
) -> tuple[Path, bool]:
    snapshots = resolve_confined_path(root, "snapshots", label="snapshots directory")
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshots = resolve_confined_path(
        root,
        snapshots,
        label="snapshots directory",
        require_exists=True,
    )
    final = snapshots / generation_id
    if final.exists():
        _verify_existing_generation(final, bound)
        return final, True
    staging = snapshots / f".{generation_id}.prepare-{operation_id}"
    if staging.exists():
        try:
            _verify_existing_generation(staging, bound)
        except Exception as exc:
            raise SnapshotPublicationConflict(
                "existing prepare staging directory is invalid"
            ) from exc
    else:
        staging.mkdir()
        try:
            _copy_bound_bundle(source, staging, bound)
            copied = _strict_json_mapping(
                _read_stable_regular_file(
                    staging / "lt_input_snapshot.json",
                    label="copied LT contract",
                ),
                label="copied LT contract",
            )
            assert copied is not None
            validate_governed_lt_snapshot_bundle(staging, copied)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    view = _publication_view(root)
    lock_handle = None
    try:
        lock_handle = _acquire_file_lock(view / ".publish.lock")
        if final.exists():
            _verify_existing_generation(final, bound)
            shutil.rmtree(staging, ignore_errors=True)
            return final, True
        _verify_existing_generation(staging, bound)
        _replace_directory(staging, final)
        _verify_existing_generation(final, bound)
        _fsync_directory(final.parent)
    finally:
        _release_file_lock(lock_handle)
    return final, False


def _discover_bound_files(root: Path) -> dict[str, tuple[str, int | None]]:
    bound: dict[str, tuple[str, int | None]] = {}
    for directory, directories, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in directories:
            if is_link_or_junction(parent / name):
                raise ValueError("legacy generation contains a linked directory")
        for name in files:
            path = parent / name
            payload = read_stable_single_link_file(path, label="legacy generation artifact")
            bound[path.relative_to(root).as_posix()] = (
                hashlib.sha256(payload).hexdigest(),
                len(payload),
            )
    if not bound:
        raise ValueError("legacy generation is empty")
    return bound


def _generation_inventory(
    root: Path,
    *,
    generation_id: str,
    bound: Mapping[str, tuple[str, int | None]],
) -> tuple[dict[str, object], bytes]:
    _verify_existing_generation(root, bound)
    files = []
    for path, (digest, expected_size) in sorted(bound.items()):
        actual_size = (root / Path(path)).stat(follow_symlinks=False).st_size
        if expected_size is not None and actual_size != expected_size:
            raise SnapshotPublicationConflict("generation inventory size mismatch")
        files.append({"path": path, "sha256": digest, "size_bytes": actual_size})
    inventory = {
        "schema_version": GENERATION_INVENTORY_SCHEMA,
        "generation_id": generation_id,
        "files": files,
    }
    return inventory, canonical_json(inventory)


def _bound_from_inventory(
    inventory: Mapping[str, object] | None,
    *,
    expected_generation_id: str,
) -> dict[str, tuple[str, int | None]]:
    if inventory is None or set(inventory) != {"schema_version", "generation_id", "files"}:
        raise ValueError("generation inventory schema is invalid")
    if (
        inventory.get("schema_version") != GENERATION_INVENTORY_SCHEMA
        or inventory.get("generation_id") != expected_generation_id
        or not isinstance(inventory.get("files"), list)
    ):
        raise ValueError("generation inventory identity is invalid")
    bound: dict[str, tuple[str, int | None]] = {}
    for entry in inventory["files"]:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256", "size_bytes"}:
            raise ValueError("generation inventory entry is invalid")
        path = _portable_relative_path(entry["path"], label="inventory artifact")
        if path in bound:
            raise ValueError("generation inventory repeats an artifact")
        size = entry["size_bytes"]
        if type(size) is not int or size < 0:
            raise ValueError("generation inventory artifact size is invalid")
        bound[path] = (
            _validated_sha256(entry["sha256"], label="inventory artifact SHA-256"),
            size,
        )
    if not bound or "lt_input_snapshot.json" not in bound:
        raise ValueError("generation inventory is incomplete")
    return bound


def _stable_external_file(value: str | Path, *, label: str) -> bytes:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    return read_stable_single_link_file(path, label=label)


def _validate_finalize_predecessor(
    *,
    current_payload: bytes | None,
    current_pointer: Mapping[str, object] | None,
    intent: Mapping[str, object],
    receipt: Mapping[str, object],
) -> None:
    if intent.get("transition_type") == BOOTSTRAP_TRANSITION_TYPE:
        if current_payload is None or current_pointer is None:
            raise SnapshotPublicationConflict("bootstrap legacy pointer disappeared before finalize")
        if current_pointer.get("schema_version") != "lt_data_pointer.v1":
            raise SnapshotPublicationConflict("bootstrap legacy pointer was already replaced")
        if hashlib.sha256(current_payload).hexdigest() != intent.get("legacy_pointer_sha256"):
            raise SnapshotPublicationConflict("bootstrap legacy pointer changed before finalize")
        return
    expected_id = intent.get("expected_anchor_receipt_id")
    expected_sequence = intent.get("expected_anchor_sequence")
    if expected_sequence == 0:
        if current_payload is not None:
            raise SnapshotPublicationConflict("genesis pointer appeared before finalize")
        return
    if current_pointer is None or current_pointer.get("schema_version") != EXTERNAL_POINTER_SCHEMA:
        raise SnapshotPublicationConflict("external predecessor pointer is unavailable")
    if (
        current_pointer.get("anchor_receipt_id") != expected_id
        or current_pointer.get("anchor_sequence") != expected_sequence
        or receipt.get("previous_receipt_id") != expected_id
    ):
        raise SnapshotPublicationConflict("external predecessor pointer diverged")


def _bound_files(
    source: Path,
    contract: Mapping[str, object],
    contract_payload: bytes,
) -> dict[str, tuple[str, int | None]]:
    bound = {
        "lt_input_snapshot.json": (
            hashlib.sha256(contract_payload).hexdigest(),
            len(contract_payload),
        )
    }
    roles = contract.get("files")
    assert isinstance(roles, Mapping)
    for role, raw_entry in roles.items():
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"LT input contract role {role} must be a mapping")
        _add_binding(
            bound,
            raw_entry.get("path"),
            raw_entry.get("sha256"),
            raw_entry.get("size_bytes"),
            label=f"{role}",
        )
        raw = raw_entry.get("raw_artifact")
        derivation = raw_entry.get("derivation")
        quality = raw_entry.get("quality_evidence")
        assert isinstance(raw, Mapping)
        assert isinstance(derivation, Mapping)
        assert isinstance(quality, Mapping)
        _add_binding(
            bound,
            raw.get("path"),
            raw.get("sha256"),
            raw.get("size_bytes"),
            label=f"{role}.raw",
        )
        if (
            contract.get("schema_version") == PROVIDER_RAW_LT_INPUT_SNAPSHOT_SCHEMA
            and str(role) in REPLAY_GOVERNED_LT_INPUT_ROLES
        ):
            provider_raw = raw_entry.get("provider_raw_artifact")
            provider_derivation = raw_entry.get("provider_derivation")
            if not isinstance(provider_raw, Mapping):
                raise ValueError(f"{role}.provider_raw_artifact binding is missing")
            if not isinstance(provider_derivation, Mapping):
                raise ValueError(f"{role}.provider_derivation binding is missing")
            _add_binding(
                bound,
                provider_raw.get("path"),
                provider_raw.get("sha256"),
                provider_raw.get("size_bytes"),
                label=f"{role}.provider_raw",
            )
            _add_binding(
                bound,
                provider_derivation.get("parser_code_path"),
                provider_derivation.get("parser_code_sha256"),
                None,
                label=f"{role}.provider_parser_code",
            )
            _add_binding(
                bound,
                provider_derivation.get("parser_config_path"),
                provider_derivation.get("parser_config_sha256"),
                None,
                label=f"{role}.provider_parser_config",
            )
        _add_binding(
            bound,
            derivation.get("parser_code_path"),
            derivation.get("parser_code_sha256"),
            None,
            label=f"{role}.parser_code",
        )
        _add_binding(
            bound,
            derivation.get("parser_config_path"),
            derivation.get("parser_config_sha256"),
            None,
            label=f"{role}.parser_config",
        )
        _add_binding(
            bound,
            quality.get("report_path"),
            quality.get("report_sha256"),
            None,
            label=f"{role}.quality_report",
        )
        if role == "eex_forwards_history":
            catalog_binding = raw_entry.get("vintage_catalog")
            if not isinstance(catalog_binding, Mapping):
                raise ValueError(
                    "governed EEX forward history requires a signed vintage_catalog binding"
                )
            _add_binding(
                bound,
                catalog_binding.get("path"),
                catalog_binding.get("sha256"),
                catalog_binding.get("size_bytes"),
                label="eex_forwards_history.vintage_catalog",
            )
            _add_eex_vintage_catalog_closure(
                bound,
                source=source,
                binding=catalog_binding,
            )
    return bound


def _add_eex_vintage_catalog_closure(
    bound: dict[str, tuple[str, int | None]],
    *,
    source: Path,
    binding: Mapping[str, object],
) -> None:
    catalog_logical = _portable_relative_path(
        binding.get("path"),
        label="eex_forwards_history.vintage_catalog",
    )
    catalog_path = resolve_confined_path(
        source,
        catalog_logical,
        label="EEX vintage catalog",
        require_exists=True,
        require_file=True,
    )
    try:
        catalog_payload = _read_stable_regular_file(
            catalog_path,
            label="EEX vintage catalog closure source",
        )
        expected_hash = _validated_sha256(
            binding.get("sha256"),
            label="eex_forwards_history.vintage_catalog SHA-256",
        )
        expected_size = binding.get("size_bytes")
        if (
            hashlib.sha256(catalog_payload).hexdigest() != expected_hash
            or not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or len(catalog_payload) != expected_size
        ):
            raise ValueError("EEX vintage catalog closure binding mismatch")
        catalog = load_strict_json(catalog_payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("EEX vintage catalog is unreadable") from exc
    if not isinstance(catalog, Mapping):
        raise ValueError("EEX vintage catalog must be a mapping")

    history = catalog.get("history_parquet")
    documents = catalog.get("source_documents")
    if not isinstance(history, Mapping) or not isinstance(documents, list) or not documents:
        raise ValueError("EEX vintage catalog artifact inventory is incomplete")
    _add_catalog_relative_binding(
        bound,
        source=source,
        catalog_path=catalog_path,
        path_value=history.get("path"),
        hash_value=history.get("sha256"),
        size_value=history.get("size_bytes"),
        label="EEX vintage history parquet",
    )
    for index, raw_document in enumerate(documents):
        if not isinstance(raw_document, Mapping):
            raise ValueError("EEX vintage source document entry is invalid")
        _add_catalog_relative_binding(
            bound,
            source=source,
            catalog_path=catalog_path,
            path_value=raw_document.get("path"),
            hash_value=raw_document.get("sha256"),
            size_value=raw_document.get("size_bytes"),
            label=f"EEX vintage source document {index}",
        )
        _add_catalog_relative_binding(
            bound,
            source=source,
            catalog_path=catalog_path,
            path_value=raw_document.get("parser_code_path"),
            hash_value=raw_document.get("parser_code_sha256"),
            size_value=None,
            label=f"EEX vintage parser code {index}",
        )


def _add_catalog_relative_binding(
    bound: dict[str, tuple[str, int | None]],
    *,
    source: Path,
    catalog_path: Path,
    path_value: object,
    hash_value: object,
    size_value: object,
    label: str,
) -> None:
    relative = Path(str(path_value or ""))
    if not str(path_value or "") or relative.is_absolute() or relative.drive:
        raise ValueError(f"{label} path must be relative to the catalog")
    candidate = (catalog_path.parent / relative).resolve()
    try:
        logical = candidate.relative_to(source).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the source bundle") from exc
    _add_binding(
        bound,
        logical,
        hash_value,
        size_value,
        label=label,
    )


def _add_binding(
    bound: dict[str, tuple[str, int | None]],
    path_value: object,
    hash_value: object,
    size_value: object,
    *,
    label: str,
) -> None:
    logical = _portable_relative_path(path_value, label=label)
    digest = _validated_sha256(hash_value, label=f"{label} SHA-256")
    size: int | None
    if size_value is None:
        size = None
    elif not isinstance(size_value, int) or isinstance(size_value, bool) or size_value < 0:
        raise ValueError(f"{label} size must be a non-negative integer")
    else:
        size = size_value
    existing = bound.get(logical)
    if existing is not None and existing[0] != digest:
        raise ValueError(f"conflicting LT bundle bindings for {logical}")
    if (
        existing is not None
        and existing[1] is not None
        and size is not None
        and existing[1] != size
    ):
        raise ValueError(f"conflicting LT bundle sizes for {logical}")
    bound[logical] = (digest, existing[1] if existing and existing[1] is not None else size)


def _validate_exact_bundle_inventory(
    source: Path,
    bound: Mapping[str, tuple[str, int | None]],
) -> None:
    discovered: set[str] = set()
    for directory, directories, files in os.walk(source, followlinks=False):
        directory_path = Path(directory)
        for name in directories:
            candidate = directory_path / name
            if is_link_or_junction(candidate):
                raise ValueError(f"source bundle contains linked directory: {candidate}")
        for name in files:
            candidate = directory_path / name
            if is_link_or_junction(candidate):
                raise ValueError(f"source bundle contains linked file: {candidate}")
            discovered.add(candidate.relative_to(source).as_posix())
    if discovered != set(bound):
        missing = sorted(set(bound).difference(discovered))
        extra = sorted(discovered.difference(bound))
        raise ValueError(
            f"source bundle file set is not fully bound; missing={missing}; extra={extra}"
        )


def _copy_bound_bundle(
    source: Path,
    destination: Path,
    bound: Mapping[str, tuple[str, int | None]],
) -> None:
    for logical, (expected_hash, expected_size) in bound.items():
        path = resolve_confined_path(
            source,
            logical,
            label=f"bound source artifact {logical}",
            require_exists=True,
            require_file=True,
        )
        target = destination / Path(logical)
        target.parent.mkdir(parents=True, exist_ok=True)
        _stream_verify_bound_file(
            path,
            expected_hash=expected_hash,
            expected_size=expected_size,
            label=f"bound source artifact {logical}",
            destination=target,
        )


def _verify_existing_generation(
    final: Path,
    bound: Mapping[str, tuple[str, int | None]],
) -> None:
    if is_link_or_junction(final) or not final.is_dir():
        raise SnapshotPublicationConflict("existing generation is not an immutable directory")
    _validate_exact_bundle_inventory(final, bound)
    for logical, (expected_hash, expected_size) in bound.items():
        path = resolve_confined_path(
            final,
            logical,
            label=f"existing generation artifact {logical}",
            require_exists=True,
            require_file=True,
        )
        _stream_verify_bound_file(
            path,
            expected_hash=expected_hash,
            expected_size=expected_size,
            label=f"existing generation artifact {logical}",
        )


def _stream_verify_bound_file(
    path: Path,
    *,
    expected_hash: str,
    expected_size: int | None,
    label: str,
    destination: Path | None = None,
) -> None:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1 or is_link_or_junction(path):
        raise ValueError(f"{label} must be a single-link regular file")
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor: int | None = None
    target_handle = None
    digest = hashlib.sha256()
    size = 0
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(opened):
            raise ValueError(f"{label} identity changed before streaming")
        if destination is not None:
            target_handle = destination.open("xb")
        with os.fdopen(descriptor, "rb") as source_handle:
            descriptor = None
            while True:
                chunk = source_handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                if target_handle is not None:
                    target_handle.write(chunk)
        if target_handle is not None:
            target_handle.flush()
            os.fsync(target_handle.fileno())
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if target_handle is not None:
            target_handle.close()
    after = path.stat(follow_symlinks=False)
    if _file_identity(before) != _file_identity(after) or size != int(after.st_size):
        raise ValueError(f"{label} changed while streaming")
    if digest.hexdigest() != expected_hash:
        raise ValueError(f"{label} hash mismatch")
    if expected_size is not None and size != expected_size:
        raise ValueError(f"{label} size mismatch")
    if destination is not None:
        copied = destination.stat(follow_symlinks=False)
        if not stat.S_ISREG(copied.st_mode) or int(copied.st_nlink) != 1:
            raise ValueError(f"{label} destination is not a single-link regular file")


def _read_current_pointer(path: Path) -> tuple[str | None, Mapping[str, object] | None]:
    if not path.exists():
        return None, None
    payload = _read_stable_regular_file(path, label="current LT pointer")
    try:
        pointer = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SnapshotPublicationConflict("current LT pointer is not strict JSON") from exc
    if not isinstance(pointer, Mapping) or pointer.get("schema_version") != "lt_data_pointer.v1":
        raise SnapshotPublicationConflict("current LT pointer schema is invalid")
    revision = pointer.get("revision", 0)
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        raise SnapshotPublicationConflict("current LT pointer revision is invalid")
    return hashlib.sha256(payload).hexdigest(), pointer


def _acquire_file_lock(path: Path):
    handle = None
    descriptor: int | None = None
    try:
        flags = os.O_RDWR | int(getattr(os, "O_BINARY", 0))
        flags |= int(getattr(os, "O_NOINHERIT", 0))
        flags |= int(getattr(os, "O_NOFOLLOW", 0))
        try:
            descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            before = path.lstat()
            if (
                not stat.S_ISREG(before.st_mode)
                or int(before.st_nlink) != 1
                or is_link_or_junction(path)
            ):
                raise ValueError("publication lock is not a single-link regular file")
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            after = path.lstat()
            if _file_identity(before) != _file_identity(after) or _lock_identity(
                opened
            ) != _lock_identity(after):
                raise ValueError("publication lock changed while opening")
        opened = os.fstat(descriptor)
        lexical = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or int(opened.st_size) not in {0, 1}
            or is_link_or_junction(path)
            or _lock_identity(opened) != _lock_identity(lexical)
        ):
            raise ValueError("publication lock descriptor is not safe")
        handle = os.fdopen(descriptor, "r+b", closefd=True)
        descriptor = None
        if opened.st_size == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = os.fstat(handle.fileno())
        current = path.lstat()
        if (
            is_link_or_junction(path)
            or int(locked.st_nlink) != 1
            or _lock_identity(locked) != _lock_identity(current)
        ):
            raise ValueError("publication lock path changed after locking")
        return handle
    except (OSError, ValueError) as exc:
        if handle is not None:
            handle.close()
        if descriptor is not None:
            os.close(descriptor)
        raise SnapshotPublicationBusy(f"LT snapshot publication lock is busy: {path}") from exc


def _release_file_lock(handle) -> None:
    if handle is None:
        return
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (OSError, ValueError):
        pass
    finally:
        try:
            handle.close()
        except OSError:
            pass


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_immutable_or_identical(path: Path, payload: bytes) -> None:
    path.parent.mkdir(exist_ok=True)
    prepare_immutable_directory(path.parent, label="publication journal")
    try:
        write_durable_exact_bytes(
            path,
            payload,
            label="publication journal artifact",
        )
    except ImmutableArtifactExistsError:
        validate_existing_immutable_file(path, label="publication journal artifact")
        if _read_stable_regular_file(path, label="publication journal artifact") != payload:
            raise SnapshotPublicationConflict("publication journal artifact collision")


def _write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        _write_exclusive(temporary, payload)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _replace_directory(source: Path, destination: Path) -> None:
    for attempt in range(20):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 19:
                raise
            time.sleep(min(0.025 * (attempt + 1), 0.25))


def _read_stable_regular_file(path: Path, *, label: str) -> bytes:
    return read_stable_single_link_file(path, label=label)


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def _lock_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (int(value.st_dev), int(value.st_ino), int(value.st_nlink))


def _absolute_unlinked_directory(value: str | Path, *, label: str) -> Path:
    lexical = assert_absolute_path_has_no_links(Path(value).expanduser())
    resolved = lexical.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} must be an existing unlinked directory")
    return resolved


def _portable_relative_path(value: object, *, label: str) -> str:
    raw = str(value or "").strip()
    path = PurePosixPath(raw)
    components = raw.split("/")
    if (
        not raw
        or path.is_absolute()
        or "\\" in raw
        or any(not _portable_component(component) for component in components)
    ):
        raise ValueError(f"{label} path must be portable and relative")
    return path.as_posix()


def _portable_id(value: object, *, label: str) -> str:
    raw = str(value or "").strip()
    if not _portable_component(raw):
        raise ValueError(f"{label} is not a portable identifier")
    return raw


def _portable_component(value: str) -> bool:
    return bool(
        _PORTABLE_COMPONENT.fullmatch(value)
        and value not in {".", ".."}
        and not value.endswith((".", " "))
        and value.split(".", 1)[0].upper() not in _WINDOWS_RESERVED
    )


def _validated_sha256(value: object, *, label: str) -> str:
    raw = str(value or "").strip().lower()
    if not _SHA256.fullmatch(raw):
        raise ValueError(f"{label} is invalid")
    return raw


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

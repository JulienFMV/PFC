"""Read-only verification of signed LT snapshot publication state."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pfc_shaping.data.shared_data_root import resolve_confined_path
from pfc_shaping.data.snapshot_publication_contract import (
    PUBLICATION_ANCHOR_ROOT_ENV,
    configured_publication_anchor_keyring_directory,
    configured_publication_bootstrap_keyring_directory,
    configured_publication_request_keyring_directory,
    publication_authority_registry_from_environment,
    publication_domain_sha256,
    publication_signature_key_id,
    required_publication_anchor_public_key_path,
    required_publication_bootstrap_public_key_path,
    required_publication_public_key_path,
    required_publication_request_public_key_path,
    verify_snapshot_anchor_head_observation,
    verify_snapshot_anchor_receipt,
    verify_snapshot_publication_anchor,
    verify_snapshot_publication_event,
    verify_snapshot_publication_intent,
)
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    paths_overlap_by_identity,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json


class SnapshotPublicationStateError(ValueError):
    """Raised when the pointer, signed journal, or external anchor diverges."""


EXTERNAL_POINTER_SCHEMA = "lt_data_pointer.v2"
HEAD_CHALLENGE_SCHEMA = "lt_snapshot_head_challenge.v1"
_EXTERNAL_POINTER_FIELDS = frozenset(
    {
        "schema_version",
        "anchor_receipt_id",
        "anchor_receipt_sha256",
        "anchor_sequence",
        "publication_intent_id",
        "operation_id",
        "generation_id",
        "contract_path",
        "contract_sha256",
        "generation_inventory_sha256",
    }
)
_MAX_OBSERVATION_CLOCK_SKEW = timedelta(seconds=30)


def pointer_mapping_from_external_receipt(
    intent: Mapping[str, object],
    receipt: Mapping[str, object],
    *,
    receipt_sha256: str,
) -> dict[str, object]:
    generation_id = str(receipt.get("generation_id", ""))
    return {
        "schema_version": EXTERNAL_POINTER_SCHEMA,
        "anchor_receipt_id": receipt.get("receipt_id"),
        "anchor_receipt_sha256": receipt_sha256,
        "anchor_sequence": receipt.get("sequence"),
        "publication_intent_id": intent.get("intent_id"),
        "operation_id": receipt.get("operation_id"),
        "generation_id": generation_id,
        "contract_path": (Path("snapshots") / generation_id / "lt_input_snapshot.json").as_posix(),
        "contract_sha256": receipt.get("contract_sha256"),
        "generation_inventory_sha256": receipt.get("generation_inventory_sha256"),
    }


def external_head_challenge_sha256(
    pointer: Mapping[str, object],
    *,
    challenge_nonce: str,
) -> str:
    if not _is_sha256(challenge_nonce):
        raise SnapshotPublicationStateError("HEAD challenge nonce is invalid")
    return hashlib.sha256(
        canonical_json(
            {
                "schema_version": HEAD_CHALLENGE_SCHEMA,
                "pointer_sha256": hashlib.sha256(canonical_json(pointer)).hexdigest(),
                "anchor_receipt_id": pointer.get("anchor_receipt_id"),
                "anchor_receipt_sha256": pointer.get("anchor_receipt_sha256"),
                "challenge_nonce": challenge_nonce,
            }
        )
    ).hexdigest()


def verify_external_publication_evidence(
    *,
    intent_payload: bytes,
    receipt_payload: bytes,
    observation_payload: bytes,
    pointer: Mapping[str, object],
    require_current: bool,
    expected_challenge_nonce: str | None = None,
) -> dict[str, object]:
    """Verify the full publisher-intent/external-CAS/HEAD-observation closure."""

    intent_document = _strict_mapping(intent_payload, label="publication intent")
    receipt_document = _strict_mapping(receipt_payload, label="anchor receipt")
    observation_document = _strict_mapping(
        observation_payload,
        label="anchor head observation",
    )
    try:
        intent = verify_snapshot_publication_intent(
            intent_document,
            trusted_public_key_path=required_publication_request_public_key_path(),
            trusted_public_key_dir=configured_publication_request_keyring_directory(),
            bootstrap_trusted_public_key_path=(
                required_publication_bootstrap_public_key_path()
                if intent_document.get("transition_type") == "BOOTSTRAP"
                else None
            ),
            bootstrap_trusted_public_key_dir=(
                configured_publication_bootstrap_keyring_directory()
                if intent_document.get("transition_type") == "BOOTSTRAP"
                else None
            ),
        )
        receipt = verify_snapshot_anchor_receipt(
            receipt_document,
            trusted_public_key_path=required_publication_anchor_public_key_path(),
            trusted_public_key_dir=configured_publication_anchor_keyring_directory(),
        )
        observation = verify_snapshot_anchor_head_observation(
            observation_document,
            trusted_public_key_path=required_publication_anchor_public_key_path(),
            trusted_public_key_dir=(
                None if require_current else configured_publication_anchor_keyring_directory()
            ),
        )
    except ValueError as exc:
        raise SnapshotPublicationStateError(
            "external publication evidence authentication failed"
        ) from exc

    if publication_signature_key_id(intent_document) in {
        publication_signature_key_id(receipt_document),
        publication_signature_key_id(observation_document),
    }:
        raise SnapshotPublicationStateError(
            "publisher request and external anchor signer identities overlap"
        )
    authorization = intent_document.get("bootstrap_authorization")
    if isinstance(authorization, Mapping) and publication_signature_key_id(authorization) in {
        publication_signature_key_id(intent_document),
        publication_signature_key_id(receipt_document),
        publication_signature_key_id(observation_document),
    }:
        raise SnapshotPublicationStateError(
            "bootstrap IT and publication authority signer identities overlap"
        )

    intent_sha256 = hashlib.sha256(intent_payload).hexdigest()
    receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
    if require_current and (
        expected_challenge_nonce is None
        or observation.get("challenge_nonce") != expected_challenge_nonce
    ):
        raise SnapshotPublicationStateError(
            "anchor head observation does not match the caller challenge nonce"
        )
    if (
        receipt.get("intent_id") != intent.get("intent_id")
        or receipt.get("intent_sha256") != intent_sha256
        or receipt.get("operation_id") != intent.get("operation_id")
        or receipt.get("transition_type") != intent.get("transition_type")
        or receipt.get("generation_id") != intent.get("generation_id")
        or receipt.get("contract_sha256") != intent.get("contract_sha256")
        or receipt.get("generation_inventory_sha256") != intent.get("generation_inventory_sha256")
        or receipt.get("legacy_pointer_sha256") != intent.get("legacy_pointer_sha256")
        or receipt.get("bootstrap_authorization_id")
        != intent.get("bootstrap_authorization_id")
        or receipt.get("bootstrap_authorization_sha256")
        != intent.get("bootstrap_authorization_sha256")
    ):
        raise SnapshotPublicationStateError(
            "anchor receipt does not close the exact publication intent"
        )
    expected_sequence = int(intent["expected_anchor_sequence"]) + 1
    if receipt.get("sequence") != expected_sequence or receipt.get(
        "previous_receipt_id"
    ) != intent.get("expected_anchor_receipt_id"):
        raise SnapshotPublicationStateError(
            "anchor receipt does not realize the requested compare-and-append"
        )
    expected_pointer = pointer_mapping_from_external_receipt(
        intent,
        receipt,
        receipt_sha256=receipt_sha256,
    )
    if set(pointer) != _EXTERNAL_POINTER_FIELDS or dict(pointer) != expected_pointer:
        raise SnapshotPublicationStateError(
            "LT data pointer is not the exact external receipt projection"
        )
    if (
        observation.get("receipt_id") != receipt.get("receipt_id")
        or observation.get("receipt_sha256") != receipt_sha256
        or observation.get("sequence") != receipt.get("sequence")
        or observation.get("challenge_sha256")
        != external_head_challenge_sha256(
            pointer,
            challenge_nonce=str(observation.get("challenge_nonce", "")),
        )
    ):
        raise SnapshotPublicationStateError(
            "anchor head observation does not select the receipt projection"
        )
    operation_time = _canonical_utc_datetime(
        intent["operation_created_at_utc"],
        label="publication operation",
    )
    committed_time = _canonical_utc_datetime(
        receipt["committed_at_utc"],
        label="anchor commit",
    )
    observed_time = _canonical_utc_datetime(
        observation["observed_at_utc"],
        label="anchor head observation",
    )
    expires_time = _canonical_utc_datetime(
        observation["expires_at_utc"],
        label="anchor head observation expiry",
    )
    if not operation_time <= committed_time <= observed_time < expires_time:
        raise SnapshotPublicationStateError(
            "external publication evidence timestamps are not ordered"
        )
    if require_current:
        reference = _reference_utc(None)
        if observed_time > reference + _MAX_OBSERVATION_CLOCK_SKEW:
            raise SnapshotPublicationStateError("anchor head observation is from the future")
        if expires_time <= reference:
            raise SnapshotPublicationStateError("anchor head observation is expired")
    transition = str(intent.get("transition_type", ""))
    return {
        "status": (
            "VERIFIED_BOOTSTRAPPED_MIGRATED_UNVERIFIED"
            if transition == "BOOTSTRAP"
            else "VERIFIED_EXTERNAL_CAS_PUBLICATION"
        ),
        "transition_type": transition,
        "intent_id": intent["intent_id"],
        "receipt_id": receipt["receipt_id"],
        "receipt_sha256": receipt_sha256,
        "sequence": receipt["sequence"],
        "generation_id": receipt["generation_id"],
        "observation_id": observation["observation_id"],
        "bootstrap_authorization_id": intent.get("bootstrap_authorization_id"),
    }


def verify_publication_authority_separation(
    acquisition_contract: Mapping[str, object],
    *,
    intent: Mapping[str, object],
    receipt: Mapping[str, object],
    observation: Mapping[str, object],
) -> str:
    """Reject one key controlling acquisition, publication request, or anchor state."""

    attestation = acquisition_contract.get("acquisition_attestation")
    if not isinstance(attestation, Mapping):
        raise SnapshotPublicationStateError(
            "acquisition attestation is missing for authority separation"
        )
    acquisition_key_id = str(attestation.get("key_id", "")).strip().lower()
    if not _is_sha256(acquisition_key_id):
        raise SnapshotPublicationStateError(
            "acquisition signer identity is invalid for authority separation"
        )
    request_key_id = publication_signature_key_id(intent)
    receipt_key_id = publication_signature_key_id(receipt)
    observation_key_id = publication_signature_key_id(observation)
    publication_key_ids = {request_key_id, receipt_key_id, observation_key_id}
    authorization = intent.get("bootstrap_authorization")
    bootstrap_key_id = None
    if isinstance(authorization, Mapping):
        bootstrap_key_id = publication_signature_key_id(authorization)
        if bootstrap_key_id in publication_key_ids:
            raise SnapshotPublicationStateError(
                "bootstrap IT and publication authority signer identities overlap"
            )
        publication_key_ids.add(bootstrap_key_id)
    if acquisition_key_id in publication_key_ids:
        raise SnapshotPublicationStateError(
            "acquisition and publication authority signer identities overlap"
        )
    try:
        registry = publication_authority_registry_from_environment()
    except ValueError as exc:
        raise SnapshotPublicationStateError(
            "global publication authority trust registry is invalid"
        ) from exc
    expected_memberships = {
        "acquisition": acquisition_key_id,
        "publication_anchor": receipt_key_id,
        "publication_request": request_key_id,
    }
    for role, key_id in expected_memberships.items():
        if key_id not in registry[role]:
            raise SnapshotPublicationStateError(
                f"{role} signer is outside the global authority trust registry"
            )
    if observation_key_id not in registry["publication_anchor"]:
        raise SnapshotPublicationStateError(
            "publication anchor observation signer is outside the global authority trust registry"
        )
    if bootstrap_key_id is not None and bootstrap_key_id not in registry["publication_bootstrap"]:
        raise SnapshotPublicationStateError(
            "bootstrap signer is outside the global authority trust registry"
        )
    return acquisition_key_id


def _strict_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        document = load_strict_json(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SnapshotPublicationStateError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise SnapshotPublicationStateError(f"{label} must be a mapping")
    return document


def _is_sha256(value: object) -> bool:
    raw = str(value or "")
    return len(raw) == 64 and all(character in "0123456789abcdef" for character in raw)


def _canonical_utc_datetime(value: object, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise SnapshotPublicationStateError(f"{label} timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise SnapshotPublicationStateError(f"{label} timestamp is not UTC")
    return parsed.astimezone(timezone.utc)


def _reference_utc(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise SnapshotPublicationStateError("publication reference time is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise SnapshotPublicationStateError("publication reference time must be UTC")
    return parsed.astimezone(timezone.utc)


def _legacy_publication_anchor_directory_for_tests(
    data_root: str | Path,
    *,
    create: bool,
) -> Path:
    selected = str(os.environ.get(PUBLICATION_ANCHOR_ROOT_ENV, "")).strip()
    if not selected:
        raise SnapshotPublicationStateError(f"{PUBLICATION_ANCHOR_ROOT_ENV} is required")
    root = Path(selected).expanduser()
    if not root.is_absolute():
        raise SnapshotPublicationStateError(f"{PUBLICATION_ANCHOR_ROOT_ENV} must be absolute")
    try:
        root = assert_absolute_path_has_no_links(root)
        governed_root = assert_absolute_path_has_no_links(Path(data_root).absolute())
        if paths_overlap_by_identity(root, governed_root):
            raise SnapshotPublicationStateError(
                "publication anchor root must be external to data_root"
            )
    except (OSError, ValueError) as exc:
        if isinstance(exc, SnapshotPublicationStateError):
            raise
        raise SnapshotPublicationStateError(
            "cannot validate publication anchor root identity"
        ) from exc
    if not root.is_dir():
        raise SnapshotPublicationStateError("publication anchor root must be an existing directory")
    domain = root / publication_domain_sha256()[:32]
    if create:
        domain.mkdir(exist_ok=True)
    try:
        domain = assert_absolute_path_has_no_links(domain)
    except ValueError as exc:
        raise SnapshotPublicationStateError(
            "publication anchor domain contains a link or reparse point"
        ) from exc
    if not domain.is_dir():
        raise SnapshotPublicationStateError("publication anchor domain is unavailable")
    return domain


def _legacy_load_publication_anchor_history_for_tests(
    data_root: str | Path,
) -> list[dict[str, object]]:
    directory = _legacy_publication_anchor_directory_for_tests(data_root, create=False)
    public_key = required_publication_public_key_path()
    entries = sorted(directory.iterdir(), key=lambda path: path.name)
    if any(not path.is_file() or path.suffix.lower() != ".json" for path in entries):
        raise SnapshotPublicationStateError(
            "publication anchor directory contains an unexpected entry"
        )
    history: list[dict[str, object]] = []
    previous_anchor_id: str | None = None
    for expected_sequence, path in enumerate(entries, start=1):
        try:
            payload = read_stable_single_link_file(path, label="publication anchor")
            document = load_strict_json(payload.decode("utf-8"))
            if not isinstance(document, Mapping):
                raise ValueError("anchor is not a mapping")
            anchor = verify_snapshot_publication_anchor(
                document,
                trusted_public_key_path=public_key,
            )
        except (UnicodeDecodeError, ValueError) as exc:
            raise SnapshotPublicationStateError(
                f"publication anchor is invalid: {path.name}"
            ) from exc
        anchor_id = str(anchor.get("anchor_id", ""))
        expected_name = f"{expected_sequence:012d}-{anchor_id}.json"
        if path.name != expected_name:
            raise SnapshotPublicationStateError("publication anchor filename mismatch")
        if anchor.get("sequence") != expected_sequence:
            raise SnapshotPublicationStateError("publication anchor sequence is not contiguous")
        if anchor.get("previous_anchor_id") != previous_anchor_id:
            raise SnapshotPublicationStateError("publication anchor chain is not linked")
        history.append(anchor)
        previous_anchor_id = anchor_id
    previous_event_id: str | None = None
    previous_pointer_sha256: str | None = None
    for anchor in history:
        event = _legacy_read_publication_event_for_tests(
            data_root, str(anchor["event_id"])
        )
        if event.get("revision") != anchor.get("sequence"):
            raise SnapshotPublicationStateError("publication anchor/event sequence mismatch")
        pointer_sha256 = hashlib.sha256(
            canonical_json(_legacy_pointer_mapping_from_event_for_tests(event))
        ).hexdigest()
        if anchor.get("pointer_sha256") != pointer_sha256:
            raise SnapshotPublicationStateError("publication anchor/event pointer hash mismatch")
        if (
            event.get("previous_event_id") != previous_event_id
            or event.get("previous_pointer_sha256") != previous_pointer_sha256
        ):
            raise SnapshotPublicationStateError(
                "publication anchor/event histories are not linked entry by entry"
            )
        previous_event_id = str(anchor["event_id"])
        previous_pointer_sha256 = pointer_sha256
    return history


def _legacy_read_publication_event_for_tests(
    data_root: str | Path, event_id: str
) -> dict[str, object]:
    root = assert_absolute_path_has_no_links(Path(data_root).absolute())
    path = resolve_confined_path(
        root,
        f"views/pfc_lt/events/{event_id}.json",
        label="LT publication event",
        require_exists=True,
        require_file=True,
    )
    try:
        payload = read_stable_single_link_file(path, label="LT publication event")
        document = load_strict_json(payload.decode("utf-8"))
        if not isinstance(document, Mapping):
            raise ValueError("event is not a mapping")
        event = verify_snapshot_publication_event(
            document,
            trusted_public_key_path=required_publication_public_key_path(),
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise SnapshotPublicationStateError("LT publication event is invalid") from exc
    if str(event.get("event_id", "")) != event_id:
        raise SnapshotPublicationStateError("LT publication event path/id mismatch")
    return event


def _legacy_verify_publication_projection_for_tests(
    data_root: str | Path,
    pointer: Mapping[str, object],
    *,
    pointer_sha256: str,
) -> dict[str, object]:
    revision = pointer.get("revision")
    if type(revision) is not int or revision < 1:
        raise SnapshotPublicationStateError("LT publication pointer revision is invalid")
    history = _legacy_load_publication_anchor_history_for_tests(data_root)
    if not history:
        raise SnapshotPublicationStateError("LT publication anchor history is empty")
    head = history[-1]
    event_id = str(pointer.get("publication_event_id", ""))
    if (
        head.get("sequence") != revision
        or head.get("event_id") != event_id
        or head.get("pointer_sha256") != pointer_sha256
    ):
        raise SnapshotPublicationStateError(
            "LT pointer does not match external publication anchor head"
        )
    _verify_pointer_against_anchor(
        data_root,
        pointer,
        pointer_sha256=pointer_sha256,
        anchor=head,
    )
    return {
        "revision": revision,
        "event_id": event_id,
        "anchor_id": head["anchor_id"],
        "publication_domain_sha256": publication_domain_sha256(),
        "status": "VERIFIED_EXTERNAL_MONOTONE_PUBLICATION",
    }


def _legacy_verify_pointer_against_anchor_for_tests(
    data_root: str | Path,
    pointer: Mapping[str, object],
    *,
    pointer_sha256: str,
    anchor: Mapping[str, object],
) -> None:
    """Verify one pointer projection against a selected immutable anchor."""

    _verify_pointer_against_anchor(
        data_root,
        pointer,
        pointer_sha256=pointer_sha256,
        anchor=anchor,
    )


def _verify_pointer_against_anchor(
    data_root: str | Path,
    pointer: Mapping[str, object],
    *,
    pointer_sha256: str,
    anchor: Mapping[str, object],
) -> None:
    revision = pointer.get("revision")
    event_id = str(pointer.get("publication_event_id", ""))
    if (
        anchor.get("sequence") != revision
        or anchor.get("event_id") != event_id
        or anchor.get("pointer_sha256") != pointer_sha256
    ):
        raise SnapshotPublicationStateError("LT pointer does not match selected publication anchor")
    event = _legacy_verify_publication_event_chain_for_tests(data_root, event_id)
    if dict(pointer) != _legacy_pointer_mapping_from_event_for_tests(event):
        raise SnapshotPublicationStateError("LT pointer/event head binding mismatch")


def _legacy_verify_publication_event_chain_for_tests(
    data_root: str | Path,
    event_id: str,
) -> dict[str, object]:
    child = _legacy_read_publication_event_for_tests(data_root, event_id)
    head = child
    revision = int(child["revision"])
    while revision > 1:
        parent_id = str(child.get("previous_event_id", ""))
        parent = _legacy_read_publication_event_for_tests(data_root, parent_id)
        if int(parent.get("revision", -1)) != revision - 1:
            raise SnapshotPublicationStateError("LT publication event revisions are not contiguous")
        expected_previous_pointer = hashlib.sha256(
            canonical_json(_legacy_pointer_mapping_from_event_for_tests(parent))
        ).hexdigest()
        if child.get("previous_pointer_sha256") != expected_previous_pointer:
            raise SnapshotPublicationStateError("LT publication pointer chain is not linked")
        child = parent
        revision -= 1
    if (
        child.get("previous_event_id") is not None
        or child.get("previous_pointer_sha256") is not None
    ):
        raise SnapshotPublicationStateError("LT publication event chain has no genesis")
    return head


def _legacy_pointer_mapping_from_event_for_tests(
    event: Mapping[str, object],
) -> dict[str, object]:
    generation_id = str(event.get("generation_id", ""))
    return {
        "schema_version": "lt_data_pointer.v1",
        "revision": event.get("revision"),
        "publication_event_id": event.get("event_id"),
        "previous_pointer_sha256": event.get("previous_pointer_sha256"),
        "generation_id": generation_id,
        "contract_path": (Path("snapshots") / generation_id / "lt_input_snapshot.json").as_posix(),
        "contract_sha256": event.get("contract_sha256"),
    }


def canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")

"""Authenticated contracts for governed LT snapshot publication."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    process_immutable_directory_entries,
    read_stable_single_link_file,
)

PUBLICATION_SIGNING_PRIVATE_KEY_ENV = "PFC_DATA_PUBLICATION_SIGNING_PRIVATE_KEY_PATH"
PUBLICATION_TRUSTED_PUBLIC_KEY_ENV = "PFC_DATA_PUBLICATION_TRUSTED_PUBLIC_KEY_PATH"
PUBLICATION_ANCHOR_ROOT_ENV = "PFC_DATA_PUBLICATION_ANCHOR_ROOT"
PUBLICATION_DOMAIN_ID_ENV = "PFC_DATA_PUBLICATION_DOMAIN_ID"
PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_ENV = (
    "PFC_DATA_PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_PATH"
)
PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_ENV = "PFC_DATA_PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_PATH"
PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_DIR_ENV = (
    "PFC_DATA_PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_DIR"
)
PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV = "PFC_DATA_PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_PATH"
PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_DIR_ENV = "PFC_DATA_PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_DIR"
PUBLICATION_BOOTSTRAP_TRUSTED_PUBLIC_KEY_ENV = (
    "PFC_DATA_PUBLICATION_BOOTSTRAP_TRUSTED_PUBLIC_KEY_PATH"
)
PUBLICATION_BOOTSTRAP_TRUSTED_PUBLIC_KEY_DIR_ENV = (
    "PFC_DATA_PUBLICATION_BOOTSTRAP_TRUSTED_PUBLIC_KEY_DIR"
)
PUBLICATION_HEAD_OBSERVATION_PATH_ENV = "PFC_DATA_PUBLICATION_HEAD_OBSERVATION_PATH"
PUBLICATION_HEAD_CHALLENGE_NONCE_ENV = "PFC_DATA_PUBLICATION_HEAD_CHALLENGE_NONCE"
GOVERNED_AUTHORITY_TRUST_ENVIRONMENTS = {
    "acquisition": (
        "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_PATH",
        "PFC_DATA_ACQUISITION_TRUSTED_PUBLIC_KEY_DIR",
    ),
    "journal": (
        "PFC_DATA_JOURNAL_TRUSTED_PUBLIC_KEY_PATH",
        "PFC_DATA_JOURNAL_TRUSTED_PUBLIC_KEY_DIR",
    ),
    "model_governance": ("PFC_MODEL_GOVERNANCE_TRUSTED_PUBLIC_KEY_PATH", None),
    "promotion_event": (
        "PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_PATH",
        "PFC_PROMOTION_EVENT_TRUSTED_PUBLIC_KEY_DIR",
    ),
    "promotion_receipt": (
        "PFC_PROMOTION_TRUSTED_PUBLIC_KEY_PATH",
        "PFC_PROMOTION_TRUSTED_PUBLIC_KEY_DIR",
    ),
    "publication_anchor": (
        PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV,
        PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_DIR_ENV,
    ),
    "publication_bootstrap": (
        PUBLICATION_BOOTSTRAP_TRUSTED_PUBLIC_KEY_ENV,
        PUBLICATION_BOOTSTRAP_TRUSTED_PUBLIC_KEY_DIR_ENV,
    ),
    "publication_request": (
        PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_ENV,
        PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_DIR_ENV,
    ),
    "quote_conflict": ("PFC_QUOTE_CONFLICT_POLICY_TRUSTED_PUBLIC_KEY_PATH", None),
    "release_request": (
        "PFC_RELEASE_REQUEST_TRUSTED_PUBLIC_KEY_PATH",
        "PFC_RELEASE_REQUEST_TRUSTED_PUBLIC_KEY_DIR",
    ),
    "rollback_authorization": (
        "PFC_ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_PATH",
        "PFC_ROLLBACK_AUTHORIZATION_TRUSTED_PUBLIC_KEY_DIR",
    ),
    "tier2_base_model_execution": (
        "PFC_TIER2_BASE_MODEL_EXECUTION_TRUSTED_PUBLIC_KEY_PATH",
        None,
    ),
    "tier2_execution": ("PFC_TIER2_EXECUTION_TRUSTED_PUBLIC_KEY_PATH", None),
    "tier2_selection_input_execution": (
        "PFC_TIER2_SELECTION_INPUT_EXECUTION_TRUSTED_PUBLIC_KEY_PATH",
        None,
    ),
    "tier2_timestamp": ("PFC_TIER2_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH", None),
    "tier2_timestamp_journal_witness": (
        "PFC_TIER2_TIMESTAMP_JOURNAL_WITNESS_TRUSTED_PUBLIC_KEY_PATH",
        None,
    ),
    "trusted_time": (
        "PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_PATH",
        "PFC_DATA_TIMESTAMP_TRUSTED_PUBLIC_KEY_DIR",
    ),
}
PUBLICATION_EVENT_SCHEMA = "lt_pointer_publication_event.v3"
PUBLICATION_ANCHOR_SCHEMA = "lt_pointer_publication_anchor.v1"
PUBLICATION_INTENT_SCHEMA = "lt_snapshot_publication_intent.v1"
PUBLICATION_RECEIPT_SCHEMA = "lt_snapshot_anchor_receipt.v1"
PUBLICATION_HEAD_OBSERVATION_SCHEMA = "lt_snapshot_anchor_head_observation.v1"
PUBLICATION_BOOTSTRAP_AUTHORIZATION_SCHEMA = "lt_snapshot_bootstrap_authorization.v1"
BOOTSTRAP_TRANSITION_TYPE = "BOOTSTRAP"
BOOTSTRAP_USE_POLICY = "ONE_SHOT_EXACT_OPERATION"
BOOTSTRAP_REQUIRED_SOURCE_CLASS = "MIGRATED_UNVERIFIED"
SIGNATURE_ALGORITHM = "Ed25519"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "publication_domain_sha256",
        "operation_id",
        "revision",
        "previous_event_id",
        "previous_pointer_sha256",
        "generation_id",
        "contract_sha256",
    }
)
_ANCHOR_FIELDS = frozenset(
    {
        "schema_version",
        "publication_domain_sha256",
        "sequence",
        "event_id",
        "pointer_sha256",
        "previous_anchor_id",
    }
)
_INTENT_FIELDS = frozenset(
    {
        "schema_version",
        "publication_domain_sha256",
        "operation_id",
        "operation_created_at_utc",
        "transition_type",
        "expected_anchor_receipt_id",
        "expected_anchor_sequence",
        "generation_id",
        "contract_path",
        "contract_sha256",
        "generation_inventory_sha256",
        "legacy_pointer_sha256",
        "bootstrap_authorization",
        "bootstrap_authorization_id",
        "bootstrap_authorization_sha256",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "publication_domain_sha256",
        "sequence",
        "previous_receipt_id",
        "operation_id",
        "intent_id",
        "intent_sha256",
        "transition_type",
        "generation_id",
        "contract_sha256",
        "generation_inventory_sha256",
        "legacy_pointer_sha256",
        "bootstrap_authorization_id",
        "bootstrap_authorization_sha256",
        "committed_at_utc",
    }
)
_HEAD_OBSERVATION_FIELDS = frozenset(
    {
        "schema_version",
        "publication_domain_sha256",
        "receipt_id",
        "receipt_sha256",
        "sequence",
        "challenge_nonce",
        "challenge_sha256",
        "observed_at_utc",
        "expires_at_utc",
    }
)
_BOOTSTRAP_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_version",
        "publication_domain_sha256",
        "transition_type",
        "use_policy",
        "operation_id",
        "expected_anchor_sequence",
        "expected_anchor_receipt_id",
        "legacy_pointer_sha256",
        "generation_id",
        "contract_path",
        "contract_sha256",
        "generation_inventory_sha256",
        "required_source_class",
        "required_calibration_eligible",
        "issued_at_utc",
        "expires_at_utc",
    }
)
_MAX_HEAD_OBSERVATION_LIFETIME = timedelta(minutes=5)
_MAX_BOOTSTRAP_AUTHORIZATION_LIFETIME = timedelta(minutes=15)


class SnapshotPublicationAuthenticationError(ValueError):
    """Raised when publication authority evidence is malformed or forged."""


def publication_domain_sha256() -> str:
    raw = str(os.environ.get(PUBLICATION_DOMAIN_ID_ENV, "")).strip().lower()
    if not raw:
        raise SnapshotPublicationAuthenticationError(f"{PUBLICATION_DOMAIN_ID_ENV} is required")
    try:
        domain_id = str(uuid.UUID(raw))
    except ValueError as exc:
        raise SnapshotPublicationAuthenticationError(
            f"{PUBLICATION_DOMAIN_ID_ENV} must be a canonical UUID"
        ) from exc
    if raw != domain_id:
        raise SnapshotPublicationAuthenticationError(
            f"{PUBLICATION_DOMAIN_ID_ENV} must be a canonical UUID"
        )
    return hashlib.sha256(
        f"pfc_lt_data_publication_domain.v1:{domain_id}".encode("ascii")
    ).hexdigest()


def required_publication_private_key_path() -> Path:
    return _required_key_path(PUBLICATION_SIGNING_PRIVATE_KEY_ENV)


def required_publication_public_key_path() -> Path:
    return _required_key_path(PUBLICATION_TRUSTED_PUBLIC_KEY_ENV)


def required_publication_request_private_key_path() -> Path:
    return _required_key_path(PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_ENV)


def required_publication_request_public_key_path() -> Path:
    return _required_key_path(PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_ENV)


def required_publication_anchor_public_key_path() -> Path:
    return _required_key_path(PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_ENV)


def required_publication_bootstrap_public_key_path() -> Path:
    return _required_key_path(PUBLICATION_BOOTSTRAP_TRUSTED_PUBLIC_KEY_ENV)


def configured_publication_request_keyring_directory() -> Path | None:
    return _optional_keyring_directory(PUBLICATION_REQUEST_TRUSTED_PUBLIC_KEY_DIR_ENV)


def configured_publication_anchor_keyring_directory() -> Path | None:
    return _optional_keyring_directory(PUBLICATION_ANCHOR_TRUSTED_PUBLIC_KEY_DIR_ENV)


def configured_publication_bootstrap_keyring_directory() -> Path | None:
    return _optional_keyring_directory(PUBLICATION_BOOTSTRAP_TRUSTED_PUBLIC_KEY_DIR_ENV)


def verify_snapshot_publication_event(
    document: Mapping[str, object],
    *,
    trusted_public_key_path: str | Path,
) -> dict[str, object]:
    return _verify_document(
        document,
        schema=PUBLICATION_EVENT_SCHEMA,
        identity_field="event_id",
        expected_fields=_EVENT_FIELDS,
        trusted_public_key_path=trusted_public_key_path,
    )


def verify_snapshot_publication_anchor(
    document: Mapping[str, object],
    *,
    trusted_public_key_path: str | Path,
) -> dict[str, object]:
    return _verify_document(
        document,
        schema=PUBLICATION_ANCHOR_SCHEMA,
        identity_field="anchor_id",
        expected_fields=_ANCHOR_FIELDS,
        trusted_public_key_path=trusted_public_key_path,
    )


def sign_snapshot_publication_intent(
    payload: Mapping[str, object],
    *,
    private_key_path: str | Path,
    bootstrap_trusted_public_key_path: str | Path | None = None,
    bootstrap_trusted_public_key_dir: str | Path | None = None,
) -> dict[str, object]:
    signed = _sign_document(
        payload,
        schema=PUBLICATION_INTENT_SCHEMA,
        identity_field="intent_id",
        expected_fields=_INTENT_FIELDS,
        private_key_path=private_key_path,
    )
    _verify_intent_bootstrap_authorization(
        signed,
        trusted_public_key_path=bootstrap_trusted_public_key_path,
        trusted_public_key_dir=bootstrap_trusted_public_key_dir,
    )
    return signed


def verify_snapshot_publication_intent(
    document: Mapping[str, object],
    *,
    trusted_public_key_path: str | Path,
    trusted_public_key_dir: str | Path | None = None,
    bootstrap_trusted_public_key_path: str | Path | None = None,
    bootstrap_trusted_public_key_dir: str | Path | None = None,
) -> dict[str, object]:
    verified = _verify_document(
        document,
        schema=PUBLICATION_INTENT_SCHEMA,
        identity_field="intent_id",
        expected_fields=_INTENT_FIELDS,
        trusted_public_key_path=trusted_public_key_path,
        trusted_public_key_dir=trusted_public_key_dir,
    )
    _verify_intent_bootstrap_authorization(
        document,
        trusted_public_key_path=bootstrap_trusted_public_key_path,
        trusted_public_key_dir=bootstrap_trusted_public_key_dir,
    )
    return verified


def verify_snapshot_bootstrap_authorization(
    document: Mapping[str, object],
    *,
    trusted_public_key_path: str | Path,
    trusted_public_key_dir: str | Path | None = None,
) -> dict[str, object]:
    """Authenticate one exact, short-lived IT authorization for legacy bootstrap."""

    return _verify_document(
        document,
        schema=PUBLICATION_BOOTSTRAP_AUTHORIZATION_SCHEMA,
        identity_field="authorization_id",
        expected_fields=_BOOTSTRAP_AUTHORIZATION_FIELDS,
        trusted_public_key_path=trusted_public_key_path,
        trusted_public_key_dir=trusted_public_key_dir,
    )


def verify_snapshot_bootstrap_intent_authorization(
    intent: Mapping[str, object],
    *,
    trusted_public_key_path: str | Path | None = None,
    trusted_public_key_dir: str | Path | None = None,
    admitted_at_utc: datetime | str | None = None,
) -> dict[str, object]:
    """Verify the embedded IT authorization and optional admission-time validity."""

    authorization = _verify_intent_bootstrap_authorization(
        intent,
        trusted_public_key_path=trusted_public_key_path,
        trusted_public_key_dir=trusted_public_key_dir,
    )
    if authorization is None:
        raise SnapshotPublicationAuthenticationError(
            "publication intent is not an authorized bootstrap"
        )
    if admitted_at_utc is not None:
        admitted = (
            admitted_at_utc
            if isinstance(admitted_at_utc, datetime)
            else _require_utc_timestamp(admitted_at_utc, label="bootstrap admitted_at_utc")
        )
        if admitted.tzinfo is None or admitted.utcoffset() != timedelta(0):
            raise SnapshotPublicationAuthenticationError(
                "publication bootstrap admission time must be UTC"
            )
        admitted = admitted.astimezone(timezone.utc)
        operation_created = _require_utc_timestamp(
            intent.get("operation_created_at_utc"),
            label="operation_created_at_utc",
        )
        expires = _require_utc_timestamp(
            authorization.get("expires_at_utc"),
            label="bootstrap expires_at_utc",
        )
        if admitted < operation_created or admitted >= expires:
            raise SnapshotPublicationAuthenticationError(
                "publication bootstrap authorization is not valid at admission"
            )
    return authorization


def verify_snapshot_anchor_receipt(
    document: Mapping[str, object],
    *,
    trusted_public_key_path: str | Path,
    trusted_public_key_dir: str | Path | None = None,
) -> dict[str, object]:
    return _verify_document(
        document,
        schema=PUBLICATION_RECEIPT_SCHEMA,
        identity_field="receipt_id",
        expected_fields=_RECEIPT_FIELDS,
        trusted_public_key_path=trusted_public_key_path,
        trusted_public_key_dir=trusted_public_key_dir,
    )


def verify_snapshot_anchor_head_observation(
    document: Mapping[str, object],
    *,
    trusted_public_key_path: str | Path,
    trusted_public_key_dir: str | Path | None = None,
) -> dict[str, object]:
    return _verify_document(
        document,
        schema=PUBLICATION_HEAD_OBSERVATION_SCHEMA,
        identity_field="observation_id",
        expected_fields=_HEAD_OBSERVATION_FIELDS,
        trusted_public_key_path=trusted_public_key_path,
        trusted_public_key_dir=trusted_public_key_dir,
    )


def publication_private_key_id(path: str | Path) -> str:
    return _public_key_id(_load_private_key(path).public_key())


def publication_public_key_id(path: str | Path) -> str:
    return _public_key_id(_load_public_key(path))


def publication_trusted_key_ids(
    primary_path: str | Path,
    *,
    keyring_directory: str | Path | None = None,
) -> frozenset[str]:
    """Capture every active and historical identity in one trust role."""

    identities = {_public_key_id(_load_public_key(primary_path))}
    if keyring_directory is None:
        return frozenset(identities)
    for path in _validated_keyring_entries(keyring_directory):
        key_id = _public_key_id(_load_public_key(path))
        if key_id in identities:
            raise SnapshotPublicationAuthenticationError(
                "publication trusted key identity is duplicated within one role"
            )
        identities.add(key_id)
    return frozenset(identities)


def publication_keyring_key_ids(
    keyring_directory: str | Path | None,
) -> frozenset[str]:
    """Capture historical identities where the active key is held separately."""

    if keyring_directory is None:
        return frozenset()
    identities: set[str] = set()
    for path in _validated_keyring_entries(keyring_directory):
        key_id = _public_key_id(_load_public_key(path))
        if key_id in identities:
            raise SnapshotPublicationAuthenticationError(
                "publication trusted key identity is duplicated within one role"
            )
        identities.add(key_id)
    return frozenset(identities)


def assert_disjoint_publication_authority_ids(
    authority_ids: Mapping[str, frozenset[str]],
) -> None:
    """Fail closed when any active or historical identity spans trust roles."""

    collisions: dict[str, list[str]] = {}
    owners: dict[str, str] = {}
    for role in sorted(authority_ids):
        for key_id in sorted(authority_ids[role]):
            previous = owners.setdefault(key_id, role)
            if previous != role:
                collisions.setdefault(key_id, [previous]).append(role)
    if collisions:
        rendered = {
            key_id: sorted(set(roles)) for key_id, roles in sorted(collisions.items())
        }
        raise SnapshotPublicationAuthenticationError(
            f"publication authority trust roles overlap: {rendered}"
        )


def publication_authority_registry_from_environment() -> dict[str, frozenset[str]]:
    """Load one complete, globally disjoint active/historical trust registry."""

    missing = sorted(
        primary_env
        for primary_env, _ in GOVERNED_AUTHORITY_TRUST_ENVIRONMENTS.values()
        if not str(os.environ.get(primary_env, "")).strip()
    )
    if missing:
        raise SnapshotPublicationAuthenticationError(
            f"publication authority trust registry is incomplete: {missing}"
        )
    registry: dict[str, frozenset[str]] = {}
    for role, (primary_env, directory_env) in sorted(
        GOVERNED_AUTHORITY_TRUST_ENVIRONMENTS.items()
    ):
        directory = (
            None
            if directory_env is None
            else str(os.environ.get(directory_env, "")).strip() or None
        )
        registry[role] = publication_trusted_key_ids(
            str(os.environ[primary_env]).strip(),
            keyring_directory=directory,
        )
    assert_disjoint_publication_authority_ids(registry)
    return registry


def publication_signature_key_id(document: Mapping[str, object]) -> str:
    signature = document.get("signature")
    if not isinstance(signature, Mapping):
        raise SnapshotPublicationAuthenticationError("publication signature block is invalid")
    key_id = signature.get("key_id")
    _require_sha256(key_id, label="signature.key_id")
    return str(key_id)


def _verify_intent_bootstrap_authorization(
    intent: Mapping[str, object],
    *,
    trusted_public_key_path: str | Path | None,
    trusted_public_key_dir: str | Path | None,
) -> dict[str, object] | None:
    if intent.get("transition_type") != BOOTSTRAP_TRANSITION_TYPE:
        return None
    authorization = intent.get("bootstrap_authorization")
    if not isinstance(authorization, Mapping):
        raise SnapshotPublicationAuthenticationError(
            "publication bootstrap authorization is missing"
        )
    if trusted_public_key_path is None:
        raise SnapshotPublicationAuthenticationError(
            "publication bootstrap trust must be passed explicitly"
        )
    selected_path = Path(trusted_public_key_path)
    selected_directory = (
        None if trusted_public_key_dir is None else Path(trusted_public_key_dir)
    )
    verified = verify_snapshot_bootstrap_authorization(
        authorization,
        trusted_public_key_path=selected_path,
        trusted_public_key_dir=selected_directory,
    )
    if publication_signature_key_id(intent) == publication_signature_key_id(authorization):
        raise SnapshotPublicationAuthenticationError(
            "publisher request and IT bootstrap signer identities overlap"
        )
    _validate_embedded_bootstrap_authorization(intent, authorization=authorization)
    return verified


def _sign_document(
    document: Mapping[str, object],
    *,
    schema: str,
    identity_field: str,
    expected_fields: frozenset[str],
    private_key_path: str | Path,
) -> dict[str, object]:
    unsigned = dict(document)
    unsigned.pop(identity_field, None)
    unsigned.pop("signature", None)
    _validate_payload(unsigned, schema=schema, expected_fields=expected_fields)
    identity = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    payload = {**unsigned, identity_field: identity}
    private_key = _load_private_key(private_key_path)
    payload["signature"] = {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": _public_key_id(private_key.public_key()),
        "value_base64": base64.b64encode(private_key.sign(_canonical_json(payload))).decode(
            "ascii"
        ),
    }
    return payload


def _verify_document(
    document: Mapping[str, object],
    *,
    schema: str,
    identity_field: str,
    expected_fields: frozenset[str],
    trusted_public_key_path: str | Path,
    trusted_public_key_dir: str | Path | None = None,
) -> dict[str, object]:
    payload = dict(document)
    signature = payload.pop("signature", None)
    if not isinstance(signature, Mapping) or set(signature) != {
        "algorithm",
        "key_id",
        "value_base64",
    }:
        raise SnapshotPublicationAuthenticationError("publication signature block is invalid")
    if signature.get("algorithm") != SIGNATURE_ALGORITHM:
        raise SnapshotPublicationAuthenticationError(
            "publication signature algorithm is unsupported"
        )
    identity = str(payload.get(identity_field, ""))
    unsigned = dict(payload)
    unsigned.pop(identity_field, None)
    _validate_payload(unsigned, schema=schema, expected_fields=expected_fields)
    if (
        not _SHA256.fullmatch(identity)
        or hashlib.sha256(_canonical_json(unsigned)).hexdigest() != identity
    ):
        raise SnapshotPublicationAuthenticationError(f"publication {identity_field} mismatch")
    public_key = _trusted_public_key_for_id(
        str(signature.get("key_id", "")),
        primary_path=trusted_public_key_path,
        keyring_directory=trusted_public_key_dir,
    )
    try:
        value = base64.b64decode(str(signature.get("value_base64", "")), validate=True)
        public_key.verify(value, _canonical_json(payload))
    except (TypeError, ValueError, InvalidSignature) as exc:
        raise SnapshotPublicationAuthenticationError("publication signature is invalid") from exc
    return payload


def _validate_payload(
    payload: Mapping[str, object],
    *,
    schema: str,
    expected_fields: frozenset[str],
) -> None:
    if set(payload) != expected_fields or payload.get("schema_version") != schema:
        raise SnapshotPublicationAuthenticationError(
            "publication payload schema or fields are invalid"
        )
    domain = payload.get("publication_domain_sha256")
    if type(domain) is not str or domain != publication_domain_sha256():
        raise SnapshotPublicationAuthenticationError("publication domain mismatch")
    if schema == PUBLICATION_INTENT_SCHEMA:
        _validate_intent_payload(payload)
    elif schema == PUBLICATION_RECEIPT_SCHEMA:
        _require_positive_sequence(payload, field="sequence")
        _validate_receipt_payload(payload)
    elif schema == PUBLICATION_HEAD_OBSERVATION_SCHEMA:
        _require_positive_sequence(payload, field="sequence")
        _validate_head_observation_payload(payload)
    elif schema == PUBLICATION_BOOTSTRAP_AUTHORIZATION_SCHEMA:
        _validate_bootstrap_authorization_payload(payload)
    elif schema == PUBLICATION_EVENT_SCHEMA:
        _require_positive_sequence(payload, field="revision")
        operation_id = str(payload.get("operation_id", "")).strip().lower()
        try:
            canonical_operation_id = str(uuid.UUID(operation_id))
        except ValueError as exc:
            raise SnapshotPublicationAuthenticationError(
                "publication operation_id is invalid"
            ) from exc
        if operation_id != canonical_operation_id:
            raise SnapshotPublicationAuthenticationError("publication operation_id is invalid")
        _require_sha256(payload.get("contract_sha256"), label="contract_sha256")
        _require_portable_id(payload.get("generation_id"), label="generation_id")
        _require_optional_sha256(
            payload.get("previous_event_id"),
            label="previous_event_id",
        )
        _require_optional_sha256(
            payload.get("previous_pointer_sha256"),
            label="previous_pointer_sha256",
        )
    elif schema == PUBLICATION_ANCHOR_SCHEMA:
        _require_positive_sequence(payload, field="sequence")
        _require_sha256(payload.get("event_id"), label="event_id")
        _require_sha256(payload.get("pointer_sha256"), label="pointer_sha256")
        _require_optional_sha256(
            payload.get("previous_anchor_id"),
            label="previous_anchor_id",
        )
    else:
        raise SnapshotPublicationAuthenticationError("publication schema is unsupported")


def _require_positive_sequence(payload: Mapping[str, object], *, field: str) -> None:
    value = payload.get(field)
    if type(value) is not int or value < 1:
        raise SnapshotPublicationAuthenticationError(f"publication {field} is invalid")


def _validate_intent_payload(payload: Mapping[str, object]) -> None:
    _require_operation_id(payload.get("operation_id"))
    _require_utc_timestamp(
        payload.get("operation_created_at_utc"),
        label="operation_created_at_utc",
    )
    transition = payload.get("transition_type")
    if transition == "ADOPT_LEGACY":
        raise SnapshotPublicationAuthenticationError(
            "legacy adoption is forbidden; use an independently authorized BOOTSTRAP"
        )
    if transition not in {"PUBLISH", BOOTSTRAP_TRANSITION_TYPE}:
        raise SnapshotPublicationAuthenticationError("publication transition_type is invalid")
    sequence = payload.get("expected_anchor_sequence")
    if type(sequence) is not int or sequence < 0:
        raise SnapshotPublicationAuthenticationError(
            "publication expected_anchor_sequence is invalid"
        )
    expected_receipt = payload.get("expected_anchor_receipt_id")
    if sequence == 0:
        if expected_receipt is not None:
            raise SnapshotPublicationAuthenticationError(
                "publication genesis cannot declare an expected receipt"
            )
    else:
        _require_sha256(expected_receipt, label="expected_anchor_receipt_id")
    _require_portable_id(payload.get("generation_id"), label="generation_id")
    generation_id = str(payload["generation_id"])
    if (
        payload.get("contract_path")
        != (Path("snapshots") / generation_id / "lt_input_snapshot.json").as_posix()
    ):
        raise SnapshotPublicationAuthenticationError("publication contract_path is not canonical")
    _require_sha256(payload.get("contract_sha256"), label="contract_sha256")
    _require_sha256(
        payload.get("generation_inventory_sha256"),
        label="generation_inventory_sha256",
    )
    legacy_pointer = payload.get("legacy_pointer_sha256")
    authorization = payload.get("bootstrap_authorization")
    authorization_id = payload.get("bootstrap_authorization_id")
    authorization_sha256 = payload.get("bootstrap_authorization_sha256")
    if transition == "PUBLISH":
        if sequence == 0:
            raise SnapshotPublicationAuthenticationError(
                "normal publication cannot create external genesis"
            )
        if any(
            value is not None
            for value in (
                legacy_pointer,
                authorization,
                authorization_id,
                authorization_sha256,
            )
        ):
            raise SnapshotPublicationAuthenticationError(
                "normal publication cannot bind legacy bootstrap evidence"
            )
        return
    if sequence != 0 or expected_receipt is not None:
        raise SnapshotPublicationAuthenticationError(
            "publication bootstrap requires an empty external head"
        )
    _require_sha256(legacy_pointer, label="legacy_pointer_sha256")
    _require_sha256(authorization_id, label="bootstrap_authorization_id")
    _require_sha256(
        authorization_sha256,
        label="bootstrap_authorization_sha256",
    )
    if not isinstance(authorization, Mapping):
        raise SnapshotPublicationAuthenticationError(
            "publication bootstrap authorization is missing"
        )
    _validate_embedded_bootstrap_authorization(
        payload,
        authorization=authorization,
    )


def _validate_embedded_bootstrap_authorization(
    intent: Mapping[str, object],
    *,
    authorization: Mapping[str, object],
) -> None:
    signature = authorization.get("signature")
    unsigned = dict(authorization)
    unsigned.pop("authorization_id", None)
    unsigned.pop("signature", None)
    _validate_payload(
        unsigned,
        schema=PUBLICATION_BOOTSTRAP_AUTHORIZATION_SCHEMA,
        expected_fields=_BOOTSTRAP_AUTHORIZATION_FIELDS,
    )
    if not isinstance(signature, Mapping):
        raise SnapshotPublicationAuthenticationError(
            "publication bootstrap authorization signature is missing"
        )
    authorization_id = str(authorization.get("authorization_id", ""))
    authorization_sha256 = hashlib.sha256(_canonical_json(authorization)).hexdigest()
    if (
        intent.get("bootstrap_authorization_id") != authorization_id
        or intent.get("bootstrap_authorization_sha256") != authorization_sha256
        or authorization.get("operation_id") != intent.get("operation_id")
        or authorization.get("expected_anchor_sequence")
        != intent.get("expected_anchor_sequence")
        or authorization.get("expected_anchor_receipt_id")
        != intent.get("expected_anchor_receipt_id")
        or authorization.get("legacy_pointer_sha256")
        != intent.get("legacy_pointer_sha256")
        or authorization.get("generation_id") != intent.get("generation_id")
        or authorization.get("contract_path") != intent.get("contract_path")
        or authorization.get("contract_sha256") != intent.get("contract_sha256")
        or authorization.get("generation_inventory_sha256")
        != intent.get("generation_inventory_sha256")
    ):
        raise SnapshotPublicationAuthenticationError(
            "publication bootstrap authorization does not bind the exact intent"
        )
    operation_created = _require_utc_timestamp(
        intent.get("operation_created_at_utc"),
        label="operation_created_at_utc",
    )
    issued = _require_utc_timestamp(
        authorization.get("issued_at_utc"),
        label="bootstrap issued_at_utc",
    )
    expires = _require_utc_timestamp(
        authorization.get("expires_at_utc"),
        label="bootstrap expires_at_utc",
    )
    if not issued <= operation_created < expires:
        raise SnapshotPublicationAuthenticationError(
            "publication bootstrap operation is outside its authorization window"
        )


def _validate_receipt_payload(payload: Mapping[str, object]) -> None:
    _require_optional_sha256(
        payload.get("previous_receipt_id"),
        label="previous_receipt_id",
    )
    sequence = int(payload["sequence"])
    if (sequence == 1) != (payload.get("previous_receipt_id") is None):
        raise SnapshotPublicationAuthenticationError(
            "publication receipt predecessor/sequence mismatch"
        )
    _require_operation_id(payload.get("operation_id"))
    _require_sha256(payload.get("intent_id"), label="intent_id")
    _require_sha256(payload.get("intent_sha256"), label="intent_sha256")
    transition = payload.get("transition_type")
    if transition == "ADOPT_LEGACY":
        raise SnapshotPublicationAuthenticationError(
            "legacy adoption is forbidden; use an independently authorized BOOTSTRAP"
        )
    if transition not in {"PUBLISH", BOOTSTRAP_TRANSITION_TYPE}:
        raise SnapshotPublicationAuthenticationError("publication transition_type is invalid")
    _require_portable_id(payload.get("generation_id"), label="generation_id")
    _require_sha256(payload.get("contract_sha256"), label="contract_sha256")
    _require_sha256(
        payload.get("generation_inventory_sha256"),
        label="generation_inventory_sha256",
    )
    legacy_pointer = payload.get("legacy_pointer_sha256")
    authorization_id = payload.get("bootstrap_authorization_id")
    authorization_sha256 = payload.get("bootstrap_authorization_sha256")
    if transition == "PUBLISH":
        if sequence == 1:
            raise SnapshotPublicationAuthenticationError(
                "normal publication cannot create external genesis"
            )
        if any(
            value is not None
            for value in (legacy_pointer, authorization_id, authorization_sha256)
        ):
            raise SnapshotPublicationAuthenticationError(
                "normal publication receipt cannot bind bootstrap evidence"
            )
    else:
        if sequence != 1 or payload.get("previous_receipt_id") is not None:
            raise SnapshotPublicationAuthenticationError(
                "publication bootstrap receipt must be external genesis"
            )
        _require_sha256(legacy_pointer, label="legacy_pointer_sha256")
        _require_sha256(authorization_id, label="bootstrap_authorization_id")
        _require_sha256(
            authorization_sha256,
            label="bootstrap_authorization_sha256",
        )
    _require_utc_timestamp(payload.get("committed_at_utc"), label="committed_at_utc")


def _validate_bootstrap_authorization_payload(payload: Mapping[str, object]) -> None:
    if payload.get("transition_type") != BOOTSTRAP_TRANSITION_TYPE:
        raise SnapshotPublicationAuthenticationError(
            "publication bootstrap authorization transition is invalid"
        )
    if payload.get("use_policy") != BOOTSTRAP_USE_POLICY:
        raise SnapshotPublicationAuthenticationError(
            "publication bootstrap authorization use policy is invalid"
        )
    _require_operation_id(payload.get("operation_id"))
    if (
        payload.get("expected_anchor_sequence") != 0
        or payload.get("expected_anchor_receipt_id") is not None
    ):
        raise SnapshotPublicationAuthenticationError(
            "publication bootstrap authorization requires empty external genesis"
        )
    _require_sha256(payload.get("legacy_pointer_sha256"), label="legacy_pointer_sha256")
    _require_portable_id(payload.get("generation_id"), label="generation_id")
    generation_id = str(payload["generation_id"])
    if (
        payload.get("contract_path")
        != (Path("snapshots") / generation_id / "lt_input_snapshot.json").as_posix()
    ):
        raise SnapshotPublicationAuthenticationError(
            "publication bootstrap authorization contract_path is not canonical"
        )
    _require_sha256(payload.get("contract_sha256"), label="contract_sha256")
    _require_sha256(
        payload.get("generation_inventory_sha256"),
        label="generation_inventory_sha256",
    )
    if payload.get("required_source_class") != BOOTSTRAP_REQUIRED_SOURCE_CLASS:
        raise SnapshotPublicationAuthenticationError(
            "publication bootstrap authorization source class is invalid"
        )
    if payload.get("required_calibration_eligible") is not False:
        raise SnapshotPublicationAuthenticationError(
            "publication bootstrap authorization must preserve calibration_eligible=false"
        )
    issued = _require_utc_timestamp(
        payload.get("issued_at_utc"),
        label="bootstrap issued_at_utc",
    )
    expires = _require_utc_timestamp(
        payload.get("expires_at_utc"),
        label="bootstrap expires_at_utc",
    )
    if expires <= issued or expires - issued > _MAX_BOOTSTRAP_AUTHORIZATION_LIFETIME:
        raise SnapshotPublicationAuthenticationError(
            "publication bootstrap authorization expiry is invalid"
        )


def _validate_head_observation_payload(payload: Mapping[str, object]) -> None:
    _require_sha256(payload.get("receipt_id"), label="receipt_id")
    _require_sha256(payload.get("receipt_sha256"), label="receipt_sha256")
    _require_sha256(payload.get("challenge_nonce"), label="challenge_nonce")
    _require_sha256(payload.get("challenge_sha256"), label="challenge_sha256")
    observed = _require_utc_timestamp(
        payload.get("observed_at_utc"),
        label="observed_at_utc",
    )
    expires = _require_utc_timestamp(
        payload.get("expires_at_utc"),
        label="expires_at_utc",
    )
    if expires <= observed or expires - observed > _MAX_HEAD_OBSERVATION_LIFETIME:
        raise SnapshotPublicationAuthenticationError(
            "publication head observation expiry is invalid"
        )


def _require_operation_id(value: object) -> None:
    raw = str(value or "").strip().lower()
    try:
        canonical = str(uuid.UUID(raw))
    except ValueError as exc:
        raise SnapshotPublicationAuthenticationError("publication operation_id is invalid") from exc
    if raw != canonical:
        raise SnapshotPublicationAuthenticationError("publication operation_id is invalid")


def _require_utc_timestamp(value: object, *, label: str) -> datetime:
    raw = str(value or "").strip()
    if not raw or raw.endswith("Z"):
        raise SnapshotPublicationAuthenticationError(
            f"publication {label} must use canonical +00:00 UTC"
        )
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise SnapshotPublicationAuthenticationError(f"publication {label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise SnapshotPublicationAuthenticationError(f"publication {label} must be UTC")
    canonical = parsed.astimezone(timezone.utc).isoformat()
    if raw != canonical:
        raise SnapshotPublicationAuthenticationError(f"publication {label} is not canonical")
    return parsed


def _trusted_public_key_for_id(
    key_id: str,
    *,
    primary_path: str | Path,
    keyring_directory: str | Path | None,
) -> Ed25519PublicKey:
    if not _SHA256.fullmatch(key_id):
        raise SnapshotPublicationAuthenticationError("publication signature key_id is invalid")
    primary = _load_public_key(primary_path)
    if _public_key_id(primary) == key_id:
        return primary
    if keyring_directory is None:
        raise SnapshotPublicationAuthenticationError("publication signing key is not trusted")
    entries = _validated_keyring_entries(keyring_directory)
    matches = [key for path in entries if _public_key_id(key := _load_public_key(path)) == key_id]
    if len(matches) != 1:
        raise SnapshotPublicationAuthenticationError(
            "publication signing key is not uniquely trusted"
        )
    return matches[0]


def _validated_keyring_entries(keyring_directory: str | Path) -> tuple[Path, ...]:
    selected = Path(keyring_directory).expanduser()
    if not selected.is_absolute():
        raise SnapshotPublicationAuthenticationError(
            "publication trusted keyring directory must be absolute"
        )
    immutable_entries = process_immutable_directory_entries(selected)
    if immutable_entries is not None:
        return immutable_entries
    try:
        directory = assert_absolute_path_has_no_links(selected)
    except (OSError, ValueError) as exc:
        raise SnapshotPublicationAuthenticationError(
            "publication trusted keyring directory is unsafe"
        ) from exc
    if not directory.is_dir():
        raise SnapshotPublicationAuthenticationError(
            "publication trusted keyring directory is unavailable"
        )
    entries = sorted(directory.iterdir(), key=lambda path: path.name)
    if any(path.suffix.lower() != ".pem" or not path.is_file() for path in entries):
        raise SnapshotPublicationAuthenticationError(
            "publication trusted keyring contains an unexpected entry"
        )
    return tuple(entries)


def _required_key_path(environment_name: str) -> Path:
    value = str(os.environ.get(environment_name, "")).strip()
    if not value:
        raise SnapshotPublicationAuthenticationError(f"{environment_name} is required")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SnapshotPublicationAuthenticationError(f"{environment_name} must be absolute")
    return path


def _optional_keyring_directory(environment_name: str) -> Path | None:
    value = str(os.environ.get(environment_name, "")).strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise SnapshotPublicationAuthenticationError(f"{environment_name} must be absolute")
    return path


def _load_private_key(path: str | Path) -> Ed25519PrivateKey:
    try:
        payload = read_stable_single_link_file(
            Path(path).expanduser().absolute(),
            label="publication private key",
        )
        key = serialization.load_pem_private_key(payload, password=None)
    except (OSError, TypeError, ValueError) as exc:
        raise SnapshotPublicationAuthenticationError("publication private key is invalid") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise SnapshotPublicationAuthenticationError("publication private key is not Ed25519")
    return key


def _load_public_key(path: str | Path) -> Ed25519PublicKey:
    try:
        payload = read_stable_single_link_file(
            Path(path).expanduser().absolute(),
            label="trusted publication public key",
        )
        key = serialization.load_pem_public_key(payload)
    except (OSError, TypeError, ValueError) as exc:
        raise SnapshotPublicationAuthenticationError(
            "trusted publication public key is invalid"
        ) from exc
    if not isinstance(key, Ed25519PublicKey):
        raise SnapshotPublicationAuthenticationError(
            "trusted publication public key is not Ed25519"
        )
    return key


def _public_key_id(key: Ed25519PublicKey) -> str:
    raw = key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _require_sha256(value: object, *, label: str) -> None:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise SnapshotPublicationAuthenticationError(f"publication {label} is invalid")


def _require_optional_sha256(value: object, *, label: str) -> None:
    if value is not None:
        _require_sha256(value, label=label)


def _require_portable_id(value: object, *, label: str) -> None:
    text = str(value or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", text):
        raise SnapshotPublicationAuthenticationError(f"publication {label} is invalid")

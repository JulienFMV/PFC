"""Fail-closed local replay of the short-tenor external trust registry.

The adapter verifies a caller-pinned, threshold-signed registry chain and the
exact public keys named by its head.  It carries no market values and grants no
external identity, trusted-time, model, promotion, or production authority.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.strict_structured_data import (
    StrictStructuredDataError,
    load_strict_json,
)
from pfc_shaping.validation.eex_short_tenor_signed_replay import (
    SIGNED_REPLAY_CONTRACT_CONTENT_ID,
    canonical_json_bytes,
    public_key_id,
)

TRUST_REGISTRY_CONTRACT_SCHEMA = (
    "fmv-eex-ch-short-tenor-trust-registry-contract-v1"
)
TRUST_REGISTRY_CONTRACT_STATUS = (
    "PASS_LOCAL_THRESHOLD_REGISTRY_LIFECYCLE_REPLAY_ONLY_NO_EXTERNAL_IDENTITY"
)
TRUST_REGISTRY_CONTRACT_CONTENT_ID = (
    "3ecf96f83be71c5beb24160793d5fcbce2d771edbda74e190bda3d5b851b54fa"
)
TRUST_REGISTRY_CONTRACT_FILE_SHA256 = (
    "e7fbf4f099bcc355073d0f5b591d988100df4168c22b172ce8fc08e4c4a0ed9c"
)
SIGNED_REPLAY_CONTRACT_SHA256 = (
    "f9f447f31458a88aaaf4ce5863f9f23dbd6a77d1683b7759885783457780a957"
)

REGISTRY_SCHEMA = "fmv_external_trust_anchor_registry.v1"
REGISTRY_PAYLOAD_TYPE = "application/vnd.fmv.pfc.trust-registry+json"
SIGNATURE_ALGORITHM = "Ed25519"
KEY_ID_RULE = "sha256_subject_public_key_info_der"
GOVERNANCE_KEY_COUNT = 3
GOVERNANCE_SIGNATURE_THRESHOLD = 2
MAXIMUM_REGISTRY_TTL_DAYS = 31

REQUIRED_ROLES = (
    "eex_acquisition",
    "eex_source_trusted_time",
    "entsoe_acquisition",
    "entsoe_source_trusted_time",
    "selection_governance",
    "training_execution",
    "trusted_time",
)
LIFECYCLE_EVENT_TYPES = (
    "ISSUED",
    "ACTIVATED",
    "RETIRED",
    "REVOKED",
    "COMPROMISED",
)
REVOCATION_REASONS = (
    "AFFILIATION_CHANGED",
    "CESSATION_OF_OPERATION",
    "PRIVILEGE_WITHDRAWN",
    "SUPERSEDED",
    "UNSPECIFIED",
)

_MAX_CHAIN_LENGTH = 128
_MAX_ENVELOPE_BYTES = 512 * 1024
_MAX_PUBLIC_KEY_BYTES = 4 * 1024
_MAX_PAYLOAD_BYTES = 384 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")

_REGISTRY_FIELDS = {
    "schema_version",
    "registry_id",
    "issuer_organization_id",
    "sequence",
    "previous_registry_payload_sha256",
    "issued_at_utc",
    "expires_at_utc",
    "governance_policy_id",
    "policy_content_id",
    "keys",
}
_KEY_FIELDS = {
    "owner_organization_id",
    "role",
    "key_id",
    "public_key_pem_sha256",
    "issued_at_utc",
    "valid_from_utc",
    "valid_until_utc",
    "supersedes_key_id",
    "events",
}
_EVENT_FIELDS = {
    "event_id",
    "event_type",
    "effective_at_utc",
    "recorded_at_utc",
    "reason",
    "incident_id",
}
_AUTHORITATIVE_USE_BLOCKERS = (
    "externally identified owners approve and publish the signed registry "
    "through governed infrastructure",
    "independently bootstrapped governance quorum with governed root-key "
    "rotation and notification channel",
    "independently verified trusted-time authority for registry reference and "
    "historical signature event times",
    "same-snapshot replay of every bound EEX and ENTSO-E PIT artifact",
    "append-only or WORM registry checkpoint event and incident retention",
    "new independently frozen future holdout distinct from sealed T057",
)


class ShortTenorTrustRegistryError(ValueError):
    """Raised when registry trust or lifecycle semantics are ambiguous."""


def validate_trust_registry_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact non-authoritative D229 policy profile."""

    contract = _mapping(document, "trust registry contract")
    _equal(
        contract.get("schema_version"),
        TRUST_REGISTRY_CONTRACT_SCHEMA,
        "trust registry contract schema",
    )
    _equal(
        contract.get("status"),
        TRUST_REGISTRY_CONTRACT_STATUS,
        "trust registry contract status",
    )
    selected = _mapping(contract.get("selected_boundary"), "selected boundary")
    _equal(
        selected.get("signed_replay_contract_content_id"),
        SIGNED_REPLAY_CONTRACT_CONTENT_ID,
        "signed replay contract content binding",
    )
    _equal(
        selected.get("signed_replay_contract_sha256"),
        SIGNED_REPLAY_CONTRACT_SHA256,
        "signed replay contract byte binding",
    )
    profile = _mapping(contract.get("registry_profile"), "registry profile")
    _equal(profile.get("schema_version"), REGISTRY_SCHEMA, "registry schema")
    _equal(
        profile.get("payload_type"),
        REGISTRY_PAYLOAD_TYPE,
        "registry payload type",
    )
    quorum = _mapping(
        contract.get("governance_quorum_profile"),
        "governance quorum profile",
    )
    _equal(
        quorum.get("governance_key_count"),
        GOVERNANCE_KEY_COUNT,
        "governance key count",
    )
    _equal(
        quorum.get("signature_threshold"),
        GOVERNANCE_SIGNATURE_THRESHOLD,
        "governance signature threshold",
    )
    _equal(contract.get("required_roles"), list(REQUIRED_ROLES), "required roles")
    _equal(
        contract.get("required_before_authoritative_use"),
        list(_AUTHORITATIVE_USE_BLOCKERS),
        "authoritative-use blockers",
    )
    content_id = _sha256_json(contract)
    _equal(
        content_id,
        TRUST_REGISTRY_CONTRACT_CONTENT_ID,
        "trust registry contract content id",
    )
    return {
        "schema_version": TRUST_REGISTRY_CONTRACT_SCHEMA,
        "status": TRUST_REGISTRY_CONTRACT_STATUS,
        "contract_content_id": content_id,
        "required_role_count": len(REQUIRED_ROLES),
        "governance_key_count": GOVERNANCE_KEY_COUNT,
        "governance_signature_threshold": GOVERNANCE_SIGNATURE_THRESHOLD,
        "external_owner_identity_verified": False,
        "external_registry_authority_verified": False,
        "independently_trusted_time_verified": False,
        "training_authorized": False,
        "selection_authorized": False,
        "production_authorized": False,
    }


def governance_policy_id(key_ids: Sequence[str]) -> str:
    """Bind the exact sorted 2-of-3 governance public-key identity set."""

    values = tuple(_sha(value, "governance key id") for value in key_ids)
    if len(values) != GOVERNANCE_KEY_COUNT or len(set(values)) != len(values):
        raise ShortTenorTrustRegistryError(
            "governance policy requires three distinct key ids"
        )
    return _sha256_json(
        {
            "key_ids": sorted(values),
            "signature_threshold": GOVERNANCE_SIGNATURE_THRESHOLD,
        }
    )


def lifecycle_event(
    *,
    key_id: str,
    event_type: str,
    effective_at_utc: str,
    recorded_at_utc: str,
    reason: str | None,
    incident_id: str | None,
) -> dict[str, object]:
    """Build one deterministic value-free lifecycle event."""

    selected_key = _sha(key_id, "lifecycle key id")
    selected_type = str(event_type)
    if selected_type not in LIFECYCLE_EVENT_TYPES:
        raise ShortTenorTrustRegistryError("lifecycle event type is invalid")
    effective = _utc_text(effective_at_utc, "lifecycle effective at")
    recorded = _utc_text(recorded_at_utc, "lifecycle recorded at")
    selected_reason = _optional_safe_id(reason, "lifecycle reason")
    selected_incident = _optional_safe_id(incident_id, "lifecycle incident id")
    core = {
        "key_id": selected_key,
        "event_type": selected_type,
        "effective_at_utc": effective,
        "recorded_at_utc": recorded,
        "reason": selected_reason,
        "incident_id": selected_incident,
    }
    return {
        "event_id": _sha256_json(core),
        "event_type": selected_type,
        "effective_at_utc": effective,
        "recorded_at_utc": recorded,
        "reason": selected_reason,
        "incident_id": selected_incident,
    }


def verify_trust_registry_chain_paths(
    *,
    registry_envelope_paths: Sequence[str | Path],
    expected_registry_envelope_sha256s: Sequence[str],
    governance_public_key_paths: Sequence[str | Path],
    expected_governance_public_key_sha256s: Sequence[str],
    registered_public_key_paths: Mapping[str, str | Path],
    expected_registered_public_key_sha256s: Mapping[str, str],
    expected_registry_id: str,
    expected_head_sequence: int,
    expected_head_envelope_sha256: str,
    caller_reference_time_utc: str,
    caller_event_time_utc_by_role: Mapping[str, str],
) -> dict[str, object]:
    """Replay a complete threshold-signed chain and its exact head key set."""

    envelope_values = _bounded_sequence(
        registry_envelope_paths,
        "registry envelope paths",
        maximum=_MAX_CHAIN_LENGTH,
    )
    envelope_hash_values = _bounded_sequence(
        expected_registry_envelope_sha256s,
        "registry envelope hashes",
        maximum=_MAX_CHAIN_LENGTH,
    )
    if len(envelope_values) != len(envelope_hash_values):
        raise ShortTenorTrustRegistryError(
            "registry envelope path and hash counts differ"
        )
    governance_values = _bounded_sequence(
        governance_public_key_paths,
        "governance public key paths",
        exact=GOVERNANCE_KEY_COUNT,
    )
    governance_hash_values = _bounded_sequence(
        expected_governance_public_key_sha256s,
        "governance public key hashes",
        exact=GOVERNANCE_KEY_COUNT,
    )
    registry_id = _safe_id(expected_registry_id, "expected registry id")
    head_sequence = _positive_int(expected_head_sequence, "expected head sequence")
    expected_head_sha = _sha(
        expected_head_envelope_sha256,
        "expected head envelope",
    )
    reference_time = _utc(caller_reference_time_utc, "caller reference time")

    paths: dict[str, Path] = {}
    envelope_raw: list[bytes] = []
    envelope_hashes: list[str] = []
    for index, (raw_path, raw_hash) in enumerate(
        zip(envelope_values, envelope_hash_values, strict=True),
        start=1,
    ):
        path = _absolute(raw_path, f"registry envelope {index}")
        expected_hash = _sha(raw_hash, f"registry envelope {index}")
        paths[f"registry envelope {index}"] = path
        envelope_raw.append(
            _read_exact(
                path,
                expected_sha256=expected_hash,
                label=f"registry envelope {index}",
                max_bytes=_MAX_ENVELOPE_BYTES,
            )
        )
        envelope_hashes.append(expected_hash)
    if envelope_hashes[-1] != expected_head_sha:
        raise ShortTenorTrustRegistryError("head envelope checkpoint differs")

    governance_keys: dict[str, Ed25519PublicKey] = {}
    governance_hashes: dict[str, str] = {}
    for index, (raw_path, raw_hash) in enumerate(
        zip(governance_values, governance_hash_values, strict=True),
        start=1,
    ):
        path = _absolute(raw_path, f"governance public key {index}")
        expected_hash = _sha(raw_hash, f"governance public key {index}")
        paths[f"governance public key {index}"] = path
        payload = _read_exact(
            path,
            expected_sha256=expected_hash,
            label=f"governance public key {index}",
            max_bytes=_MAX_PUBLIC_KEY_BYTES,
        )
        public_key = _load_public_key(payload, f"governance public key {index}")
        key_id = public_key_id(public_key)
        if key_id in governance_keys:
            raise ShortTenorTrustRegistryError(
                "governance public key identities must be distinct"
            )
        governance_keys[key_id] = public_key
        governance_hashes[key_id] = expected_hash
    policy_id = governance_policy_id(tuple(governance_keys))

    assessments: list[dict[str, object]] = []
    payload_hashes: list[str] = []
    payloads: list[bytes] = []
    previous_assessment: dict[str, object] | None = None
    for index, raw in enumerate(envelope_raw, start=1):
        payload, document, signer_ids = _verify_threshold_envelope(
            raw,
            governance_keys=governance_keys,
            label=f"registry envelope {index}",
        )
        assessment = validate_registry_document(document)
        _equal(assessment["registry_id"], registry_id, "registry id")
        _equal(
            assessment["governance_policy_id"],
            policy_id,
            "governance policy id",
        )
        _equal(assessment["sequence"], index, "registry sequence")
        payload_sha = hashlib.sha256(payload).hexdigest()
        if index == 1:
            _equal(
                assessment["previous_registry_payload_sha256"],
                None,
                "first registry predecessor",
            )
        else:
            _equal(
                assessment["previous_registry_payload_sha256"],
                payload_hashes[-1],
                "registry predecessor payload binding",
            )
            if previous_assessment is None:
                raise AssertionError("previous registry assessment is missing")
            _validate_registry_transition(previous_assessment, assessment)
        assessment["verified_governance_signer_key_ids"] = signer_ids
        assessments.append(assessment)
        payload_hashes.append(payload_sha)
        payloads.append(payload)
        previous_assessment = assessment
    head = assessments[-1]
    _equal(head["sequence"], head_sequence, "head registry sequence checkpoint")
    head_issued = _utc(head["issued_at_utc"], "head registry issued at")
    head_expires = _utc(head["expires_at_utc"], "head registry expires at")
    if reference_time < head_issued or reference_time >= head_expires:
        raise ShortTenorTrustRegistryError(
            "head registry is not valid at caller reference time"
        )

    head_entries = {
        str(entry["key_id"]): entry
        for entry in _normalized_keys(head)
    }
    supplied_paths = {
        _sha(key, "registered public key path id"): value
        for key, value in registered_public_key_paths.items()
    }
    supplied_hashes = {
        _sha(key, "registered public key hash id"): _sha(
            value,
            "expected registered public key",
        )
        for key, value in expected_registered_public_key_sha256s.items()
    }
    if set(supplied_paths) != set(head_entries):
        raise ShortTenorTrustRegistryError(
            "registered public key path set differs from head registry key set"
        )
    if set(supplied_hashes) != set(head_entries):
        raise ShortTenorTrustRegistryError(
            "registered public key hash set differs from head registry key set"
        )
    registered_hashes: dict[str, str] = {}
    for key_id in sorted(head_entries):
        path = _absolute(supplied_paths[key_id], f"registered public key {key_id}")
        paths[f"registered public key {key_id}"] = path
        expected_hash = supplied_hashes[key_id]
        _equal(
            head_entries[key_id]["public_key_pem_sha256"],
            expected_hash,
            f"registered public key {key_id} caller hash binding",
        )
        payload = _read_exact(
            path,
            expected_sha256=expected_hash,
            label=f"registered public key {key_id}",
            max_bytes=_MAX_PUBLIC_KEY_BYTES,
        )
        public_key = _load_public_key(payload, f"registered public key {key_id}")
        _equal(public_key_id(public_key), key_id, f"registered public key {key_id}")
        registered_hashes[key_id] = expected_hash
    _require_distinct_paths(paths)
    if set(governance_keys).intersection(head_entries):
        raise ShortTenorTrustRegistryError(
            "governance quorum keys must be external to registered trust roles"
        )

    event_times = {
        _safe_id(role, "caller event role"): _utc_text(value, "caller event time")
        for role, value in caller_event_time_utc_by_role.items()
    }
    if set(event_times) != set(REQUIRED_ROLES):
        raise ShortTenorTrustRegistryError(
            "caller event-time role set differs from required roles"
        )
    resolved: dict[str, str] = {}
    for role in REQUIRED_ROLES:
        event_time = _utc(event_times[role], f"{role} caller event time")
        if event_time > reference_time:
            raise ShortTenorTrustRegistryError(
                f"{role} caller event time exceeds reference time"
            )
        selected = _resolve_role_key(_normalized_keys(head), role, event_time)
        resolved[role] = str(selected["key_id"])

    return {
        "schema_version": "fmv_eex_ch_short_tenor_trust_registry_assessment.v1",
        "status": TRUST_REGISTRY_CONTRACT_STATUS,
        "trust_registry_contract_content_id": TRUST_REGISTRY_CONTRACT_CONTENT_ID,
        "registry_id": registry_id,
        "registry_sequence_count": len(assessments),
        "head_sequence": head_sequence,
        "head_envelope_sha256": expected_head_sha,
        "head_payload_sha256": payload_hashes[-1],
        "head_issued_at_utc": head["issued_at_utc"],
        "head_expires_at_utc": head["expires_at_utc"],
        "caller_reference_time_utc": _utc_text(
            caller_reference_time_utc,
            "caller reference time",
        ),
        "governance_policy_id": policy_id,
        "governance_public_key_sha256": governance_hashes,
        "governance_threshold": GOVERNANCE_SIGNATURE_THRESHOLD,
        "registered_public_key_sha256": registered_hashes,
        "locally_resolved_key_ids_by_role": resolved,
        "chain_payload_sha256s": payload_hashes,
        "chain_envelope_sha256s": envelope_hashes,
        "local_threshold_signature_integrity_verified": True,
        "local_rollback_freeze_and_lifecycle_consistency_verified": True,
        "external_owner_identity_verified": False,
        "external_registry_authority_verified": False,
        "governance_root_bootstrap_verified": False,
        "governance_root_rotation_verified": False,
        "independently_trusted_time_verified": False,
        "point_in_time_availability_proven": False,
        "training_authorized": False,
        "selection_authorized": False,
        "model_input_authorized": False,
        "candidate_assembly_authorized": False,
        "promotion_authorized": False,
        "production_authorized": False,
        "price_values_opened": False,
        "databricks_request_count": 0,
        "warehouse_start_count": 0,
        "network_call_count": 0,
        "h_drive_access_count": 0,
        "remote_write_count": 0,
    }


def validate_registry_document(document: Mapping[str, object]) -> dict[str, object]:
    """Validate one registry payload independently of its signatures."""

    registry = _mapping(document, "trust registry")
    _exact_keys(registry, _REGISTRY_FIELDS, "trust registry")
    _equal(registry.get("schema_version"), REGISTRY_SCHEMA, "registry schema")
    registry_id = _safe_id(registry.get("registry_id"), "registry id")
    issuer = _safe_id(registry.get("issuer_organization_id"), "registry issuer")
    sequence = _positive_int(registry.get("sequence"), "registry sequence")
    previous = registry.get("previous_registry_payload_sha256")
    if sequence == 1:
        _equal(previous, None, "first registry predecessor")
    else:
        previous = _sha(previous, "registry predecessor")
    issued_text = _utc_text(registry.get("issued_at_utc"), "registry issued at")
    expires_text = _utc_text(registry.get("expires_at_utc"), "registry expires at")
    issued = _utc(issued_text, "registry issued at")
    expires = _utc(expires_text, "registry expires at")
    if expires <= issued:
        raise ShortTenorTrustRegistryError("registry expiry must follow issuance")
    if expires - issued > timedelta(days=MAXIMUM_REGISTRY_TTL_DAYS):
        raise ShortTenorTrustRegistryError("registry TTL exceeds 31 days")
    policy_id = _sha(registry.get("governance_policy_id"), "governance policy id")
    _equal(
        registry.get("policy_content_id"),
        TRUST_REGISTRY_CONTRACT_CONTENT_ID,
        "registry policy content id",
    )
    raw_keys = registry.get("keys")
    if not isinstance(raw_keys, list) or not raw_keys:
        raise ShortTenorTrustRegistryError("registry keys must be non-empty")
    normalized = [_validate_key_entry(item, issued) for item in raw_keys]
    order = [
        (str(item["role"]), str(item["valid_from_utc"]), str(item["key_id"]))
        for item in normalized
    ]
    if order != sorted(order):
        raise ShortTenorTrustRegistryError(
            "registry keys must be sorted by role valid_from_utc and key_id"
        )
    key_ids = [str(item["key_id"]) for item in normalized]
    if len(key_ids) != len(set(key_ids)):
        raise ShortTenorTrustRegistryError("registry key ids must be globally unique")
    roles = {str(item["role"]) for item in normalized}
    _equal(roles, set(REQUIRED_ROLES), "registry role inventory")
    by_role = {
        role: [item for item in normalized if item["role"] == role]
        for role in REQUIRED_ROLES
    }
    for role, entries in by_role.items():
        _validate_role_chain(role, entries, issued)
        _resolve_role_key(normalized, role, issued)
    _validate_owner_separation(by_role)
    return {
        "schema_version": REGISTRY_SCHEMA,
        "registry_id": registry_id,
        "issuer_organization_id": issuer,
        "sequence": sequence,
        "previous_registry_payload_sha256": previous,
        "issued_at_utc": issued_text,
        "expires_at_utc": expires_text,
        "governance_policy_id": policy_id,
        "policy_content_id": TRUST_REGISTRY_CONTRACT_CONTENT_ID,
        "registered_key_count": len(normalized),
        "required_role_count": len(REQUIRED_ROLES),
        "active_key_ids_by_role": {
            role: [
                str(item["key_id"])
                for item in entries
                if item["lifecycle_state"] == "ACTIVE"
            ]
            for role, entries in by_role.items()
        },
        "normalized_keys": normalized,
    }


def assert_key_accepted_for_trusted_event_time(
    registry_document: Mapping[str, object],
    *,
    role: str,
    key_id: str,
    event_time_utc: str,
    trusted_event_time_verified: bool,
) -> dict[str, object]:
    """Accept a historical key only after independent event-time verification."""

    if trusted_event_time_verified is not True:
        raise ShortTenorTrustRegistryError(
            "historical key acceptance requires independently trusted event time"
        )
    selected_role = _safe_id(role, "verification role")
    if selected_role not in REQUIRED_ROLES:
        raise ShortTenorTrustRegistryError("verification role is not governed")
    selected_key_id = _sha(key_id, "verification key id")
    event_text = _utc_text(event_time_utc, "verification event time")
    assessment = validate_registry_document(registry_document)
    entry = _resolve_role_key(
        _normalized_keys(assessment),
        selected_role,
        _utc(event_text, "verification event time"),
    )
    _equal(entry["key_id"], selected_key_id, "verification key selection")
    return {
        "role": selected_role,
        "key_id": selected_key_id,
        "event_time_utc": event_text,
        "lifecycle_state_at_registry_issue": entry["lifecycle_state"],
        "historical_signature_time_accepted": True,
        "external_authority_verified": False,
    }


def _validate_key_entry(
    value: object,
    registry_issued_at: datetime,
) -> dict[str, object]:
    entry = _mapping(value, "registry key")
    _exact_keys(entry, _KEY_FIELDS, "registry key")
    owner = _safe_id(entry.get("owner_organization_id"), "key owner")
    role = _safe_id(entry.get("role"), "key role")
    if role not in REQUIRED_ROLES:
        raise ShortTenorTrustRegistryError("registry key role is not governed")
    key_id = _sha(entry.get("key_id"), "registry key id")
    pem_sha = _sha(entry.get("public_key_pem_sha256"), "public key PEM")
    issued_text = _utc_text(entry.get("issued_at_utc"), "key issued at")
    valid_from_text = _utc_text(entry.get("valid_from_utc"), "key valid from")
    valid_until_value = entry.get("valid_until_utc")
    if valid_until_value is None:
        raise ShortTenorTrustRegistryError("every key requires finite valid until")
    valid_until_text = _utc_text(valid_until_value, "key valid until")
    issued = _utc(issued_text, "key issued at")
    valid_from = _utc(valid_from_text, "key valid from")
    valid_until = _utc(valid_until_text, "key valid until")
    if issued > valid_from:
        raise ShortTenorTrustRegistryError("key issuance follows activation")
    if valid_until <= valid_from:
        raise ShortTenorTrustRegistryError("key validity interval is empty")
    supersedes = _optional_sha(entry.get("supersedes_key_id"), "superseded key")
    raw_events = entry.get("events")
    if not isinstance(raw_events, list) or len(raw_events) < 2:
        raise ShortTenorTrustRegistryError(
            "registry key requires issued and activated events"
        )
    events = [
        _validate_lifecycle_event(item, key_id, registry_issued_at)
        for item in raw_events
    ]
    event_ids = [str(item["event_id"]) for item in events]
    if len(event_ids) != len(set(event_ids)):
        raise ShortTenorTrustRegistryError("lifecycle event ids must be unique")
    recorded = [_utc(item["recorded_at_utc"], "event recorded at") for item in events]
    if recorded != sorted(recorded):
        raise ShortTenorTrustRegistryError(
            "lifecycle events must be ordered by recorded time"
        )
    _equal(events[0]["event_type"], "ISSUED", "first lifecycle event")
    _equal(events[0]["effective_at_utc"], issued_text, "issued event time")
    _equal(events[1]["event_type"], "ACTIVATED", "second lifecycle event")
    _equal(
        events[1]["effective_at_utc"],
        valid_from_text,
        "activated event time",
    )
    terminal = [
        item
        for item in events
        if item["event_type"] in {"RETIRED", "REVOKED", "COMPROMISED"}
    ]
    if len(terminal) > 1:
        raise ShortTenorTrustRegistryError("terminal lifecycle event must be unique")
    if terminal and terminal[0] is not events[-1]:
        raise ShortTenorTrustRegistryError("terminal lifecycle event must be last")
    lifecycle_state = "ACTIVE" if not terminal else str(terminal[0]["event_type"])
    if lifecycle_state == "RETIRED":
        _equal(
            terminal[0]["effective_at_utc"],
            valid_until_text,
            "retired event time",
        )
        if terminal[0]["reason"] not in {"ROTATED", "CESSATION_OF_OPERATION"}:
            raise ShortTenorTrustRegistryError("retirement reason is invalid")
        _equal(terminal[0]["incident_id"], None, "retirement incident id")
    elif lifecycle_state in {"REVOKED", "COMPROMISED"}:
        invalid = _utc(terminal[0]["effective_at_utc"], "key invalid since")
        if invalid < valid_from or invalid >= valid_until:
            raise ShortTenorTrustRegistryError(
                "revoked or compromised invalidity is outside validity"
            )
        if lifecycle_state == "COMPROMISED":
            _equal(
                terminal[0]["reason"],
                "KEY_COMPROMISE",
                "compromise reason",
            )
        elif terminal[0]["reason"] not in REVOCATION_REASONS:
            raise ShortTenorTrustRegistryError("revocation reason is invalid")
        if terminal[0]["incident_id"] is None:
            raise ShortTenorTrustRegistryError(
                "revocation or compromise requires incident id"
            )
    return {
        "owner_organization_id": owner,
        "role": role,
        "key_id": key_id,
        "public_key_pem_sha256": pem_sha,
        "issued_at_utc": issued_text,
        "valid_from_utc": valid_from_text,
        "valid_until_utc": valid_until_text,
        "supersedes_key_id": supersedes,
        "events": events,
        "lifecycle_state": lifecycle_state,
    }


def _validate_lifecycle_event(
    value: object,
    key_id: str,
    registry_issued_at: datetime,
) -> dict[str, object]:
    event = _mapping(value, "lifecycle event")
    _exact_keys(event, _EVENT_FIELDS, "lifecycle event")
    event_id = _sha(event.get("event_id"), "lifecycle event id")
    event_type = str(event.get("event_type"))
    if event_type not in LIFECYCLE_EVENT_TYPES:
        raise ShortTenorTrustRegistryError("lifecycle event type is invalid")
    effective_text = _utc_text(event.get("effective_at_utc"), "event effective at")
    recorded_text = _utc_text(event.get("recorded_at_utc"), "event recorded at")
    effective = _utc(effective_text, "event effective at")
    recorded = _utc(recorded_text, "event recorded at")
    if recorded < effective or recorded > registry_issued_at:
        raise ShortTenorTrustRegistryError("lifecycle event chronology is invalid")
    reason = _optional_safe_id(event.get("reason"), "lifecycle reason")
    incident = _optional_safe_id(event.get("incident_id"), "lifecycle incident id")
    if event_type in {"ISSUED", "ACTIVATED"}:
        _equal(reason, None, f"{event_type.lower()} reason")
        _equal(incident, None, f"{event_type.lower()} incident")
    core = {
        "key_id": key_id,
        "event_type": event_type,
        "effective_at_utc": effective_text,
        "recorded_at_utc": recorded_text,
        "reason": reason,
        "incident_id": incident,
    }
    _equal(event_id, _sha256_json(core), "lifecycle event id binding")
    return {"event_id": event_id, **{key: core[key] for key in _EVENT_FIELDS - {"event_id"}}}


def _validate_role_chain(
    role: str,
    entries: list[dict[str, object]],
    registry_issued_at: datetime,
) -> None:
    active_count = sum(item["lifecycle_state"] == "ACTIVE" for item in entries)
    if active_count != 1:
        raise ShortTenorTrustRegistryError(
            f"{role} must have exactly one active key"
        )
    for index, entry in enumerate(entries):
        if index == 0:
            _equal(entry["supersedes_key_id"], None, f"{role} first predecessor")
            continue
        predecessor = entries[index - 1]
        _equal(
            entry["supersedes_key_id"],
            predecessor["key_id"],
            f"{role} rotation predecessor",
        )
        _equal(
            entry["owner_organization_id"],
            predecessor["owner_organization_id"],
            f"{role} rotation owner",
        )
        predecessor_end = _acceptance_end(predecessor)
        if predecessor_end is None:
            raise ShortTenorTrustRegistryError(
                f"{role} predecessor has no closed acceptance interval"
            )
        predecessor_state = str(predecessor["lifecycle_state"])
        if predecessor_state == "ACTIVE":
            raise ShortTenorTrustRegistryError(
                f"{role} replacement predecessor lacks a terminal event"
            )
        if predecessor_state == "RETIRED":
            terminal = predecessor["events"][-1]
            _equal(terminal["reason"], "ROTATED", f"{role} rotation reason")
            _equal(
                entry["valid_from_utc"],
                predecessor_end,
                f"{role} scheduled rotation boundary",
            )
        elif _utc(
            entry["valid_from_utc"],
            f"{role} incident replacement valid from",
        ) < _utc(predecessor_end, f"{role} predecessor acceptance end"):
            raise ShortTenorTrustRegistryError(
                f"{role} incident replacement overlaps predecessor acceptance"
            )
    if entries[-1]["lifecycle_state"] != "ACTIVE":
        raise ShortTenorTrustRegistryError(
            f"{role} active key must be the latest role entry"
        )
    _resolve_role_key(entries, role, registry_issued_at)


def _validate_owner_separation(
    by_role: Mapping[str, list[dict[str, object]]],
) -> None:
    owners = {
        role: {str(entry["owner_organization_id"]) for entry in entries}
        for role, entries in by_role.items()
    }
    if owners["training_execution"].intersection(owners["selection_governance"]):
        raise ShortTenorTrustRegistryError(
            "training and selection governance owners overlap"
        )
    time_roles = {
        "trusted_time",
        "eex_source_trusted_time",
        "entsoe_source_trusted_time",
    }
    time_owners = set().union(*(owners[role] for role in time_roles))
    non_time_owners = set().union(
        *(owners[role] for role in REQUIRED_ROLES if role not in time_roles)
    )
    if time_owners.intersection(non_time_owners):
        raise ShortTenorTrustRegistryError(
            "trusted-time and non-time owners overlap"
        )
    if owners["eex_source_trusted_time"].intersection(owners["eex_acquisition"]):
        raise ShortTenorTrustRegistryError(
            "EEX trusted-time and acquisition owners overlap"
        )
    if owners["entsoe_source_trusted_time"].intersection(
        owners["entsoe_acquisition"]
    ):
        raise ShortTenorTrustRegistryError(
            "ENTSO-E trusted-time and acquisition owners overlap"
        )


def _validate_registry_transition(
    previous: Mapping[str, object],
    current: Mapping[str, object],
) -> None:
    _equal(current["registry_id"], previous["registry_id"], "registry chain id")
    _equal(
        current["issuer_organization_id"],
        previous["issuer_organization_id"],
        "registry chain issuer",
    )
    _equal(
        current["governance_policy_id"],
        previous["governance_policy_id"],
        "registry chain governance policy",
    )
    _equal(
        current["policy_content_id"],
        previous["policy_content_id"],
        "registry chain contract",
    )
    previous_issued = _utc(previous["issued_at_utc"], "previous registry issued at")
    current_issued = _utc(current["issued_at_utc"], "current registry issued at")
    previous_expires = _utc(
        previous["expires_at_utc"],
        "previous registry expires at",
    )
    if current_issued <= previous_issued or current_issued >= previous_expires:
        raise ShortTenorTrustRegistryError(
            "registry issuance is not monotonic before predecessor expiry"
        )
    previous_by_role = _keys_by_role(_normalized_keys(previous))
    current_by_role = _keys_by_role(_normalized_keys(current))
    for role in REQUIRED_ROLES:
        old_entries = previous_by_role[role]
        new_entries = current_by_role[role]
        old_ids = [str(entry["key_id"]) for entry in old_entries]
        new_ids = [str(entry["key_id"]) for entry in new_entries]
        if new_ids[: len(old_ids)] != old_ids:
            raise ShortTenorTrustRegistryError(
                f"{role} key history is not an immutable prefix"
            )
        for old, new in zip(old_entries, new_entries, strict=False):
            for field in (
                "owner_organization_id",
                "role",
                "key_id",
                "public_key_pem_sha256",
                "issued_at_utc",
                "valid_from_utc",
                "supersedes_key_id",
            ):
                _equal(new[field], old[field], f"{role} historical key {field}")
            old_until = old["valid_until_utc"]
            new_until = new["valid_until_utc"]
            if old_until is not None:
                _equal(new_until, old_until, f"{role} historical key valid until")
            elif new_until is not None:
                appended = new["events"][len(old["events"]) :]
                if not appended or appended[-1]["event_type"] != "RETIRED":
                    raise ShortTenorTrustRegistryError(
                        f"{role} valid-until closure lacks appended retirement"
                    )
            old_events = old["events"]
            if new["events"][: len(old_events)] != old_events:
                raise ShortTenorTrustRegistryError(
                    f"{role} lifecycle event history is not an immutable prefix"
                )


def _verify_threshold_envelope(
    envelope_raw: bytes,
    *,
    governance_keys: Mapping[str, Ed25519PublicKey],
    label: str,
) -> tuple[bytes, Mapping[str, object], list[str]]:
    envelope = _load_json_object(envelope_raw, label)
    if canonical_json_bytes(envelope) != envelope_raw:
        raise ShortTenorTrustRegistryError(f"{label} must use canonical JSON bytes")
    _exact_keys(envelope, {"payloadType", "payload", "signatures"}, label)
    _equal(envelope.get("payloadType"), REGISTRY_PAYLOAD_TYPE, f"{label} payload type")
    payload = _base64(envelope.get("payload"), f"{label} payload")
    if not payload or len(payload) > _MAX_PAYLOAD_BYTES:
        raise ShortTenorTrustRegistryError(f"{label} payload size is invalid")
    signatures = envelope.get("signatures")
    if not isinstance(signatures, list) or not (
        GOVERNANCE_SIGNATURE_THRESHOLD
        <= len(signatures)
        <= GOVERNANCE_KEY_COUNT
    ):
        raise ShortTenorTrustRegistryError(
            f"{label} signature count does not meet 2-of-3 profile"
        )
    pae = registry_dsse_pae(payload)
    verified_ids: list[str] = []
    declared_ids: list[str] = []
    for index, value in enumerate(signatures, start=1):
        signature = _mapping(value, f"{label} signature {index}")
        _exact_keys(signature, {"keyid", "sig"}, f"{label} signature {index}")
        declared_id = _sha(signature.get("keyid"), f"{label} signature key id")
        signature_raw = _base64(signature.get("sig"), f"{label} signature")
        if len(signature_raw) != 64:
            raise ShortTenorTrustRegistryError(
                f"{label} Ed25519 signature size is invalid"
            )
        matches: list[str] = []
        for key_id, public_key in governance_keys.items():
            try:
                public_key.verify(signature_raw, pae)
            except InvalidSignature:
                continue
            matches.append(key_id)
        if len(matches) != 1:
            raise ShortTenorTrustRegistryError(
                f"{label} signature does not match exactly one pinned governance key"
            )
        _equal(declared_id, matches[0], f"{label} signature key id")
        declared_ids.append(declared_id)
        verified_ids.append(matches[0])
    if declared_ids != sorted(declared_ids) or len(set(declared_ids)) != len(
        declared_ids
    ):
        raise ShortTenorTrustRegistryError(
            f"{label} signature key ids must be sorted and unique"
        )
    if len(set(verified_ids)) < GOVERNANCE_SIGNATURE_THRESHOLD:
        raise ShortTenorTrustRegistryError(f"{label} governance threshold is not met")
    document = _load_json_object(payload, f"{label} registry payload")
    if canonical_json_bytes(document) != payload:
        raise ShortTenorTrustRegistryError(
            f"{label} registry payload must use canonical JSON bytes"
        )
    return payload, document, verified_ids


def registry_dsse_pae(payload: bytes) -> bytes:
    """Return DSSE v1 PAE for the exact registry payload type."""

    if not isinstance(payload, bytes) or not payload:
        raise ShortTenorTrustRegistryError("registry DSSE payload must be bytes")
    encoded_type = REGISTRY_PAYLOAD_TYPE.encode("utf-8")
    return b" ".join(
        (
            b"DSSEv1",
            str(len(encoded_type)).encode("ascii"),
            encoded_type,
            str(len(payload)).encode("ascii"),
            payload,
        )
    )


def _resolve_role_key(
    entries: list[dict[str, object]],
    role: str,
    event_time: datetime,
) -> dict[str, object]:
    matches = []
    for entry in entries:
        if entry["role"] != role:
            continue
        valid_from = _utc(entry["valid_from_utc"], f"{role} valid from")
        end_text = _acceptance_end(entry)
        end = None if end_text is None else _utc(end_text, f"{role} acceptance end")
        if event_time >= valid_from and (end is None or event_time < end):
            matches.append(entry)
    if len(matches) != 1:
        raise ShortTenorTrustRegistryError(
            f"{role} does not resolve to exactly one key at caller event time"
        )
    return matches[0]


def _acceptance_end(entry: Mapping[str, object]) -> str | None:
    if entry["lifecycle_state"] in {"REVOKED", "COMPROMISED"}:
        return str(entry["events"][-1]["effective_at_utc"])
    value = entry["valid_until_utc"]
    return None if value is None else str(value)


def _load_public_key(raw: bytes, label: str) -> Ed25519PublicKey:
    if b"PRIVATE KEY" in raw:
        raise ShortTenorTrustRegistryError(f"{label} contains private key material")
    try:
        public_key = serialization.load_pem_public_key(raw)
    except (TypeError, ValueError) as exc:
        raise ShortTenorTrustRegistryError(f"{label} is not valid public PEM") from exc
    if not isinstance(public_key, Ed25519PublicKey):
        raise ShortTenorTrustRegistryError(f"{label} must be Ed25519")
    canonical = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    if canonical != raw:
        raise ShortTenorTrustRegistryError(f"{label} PEM bytes are not canonical")
    return public_key


def _read_exact(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    max_bytes: int,
) -> bytes:
    try:
        raw = read_stable_single_link_file(path, label=label, max_bytes=max_bytes)
    except (OSError, TypeError, ValueError) as exc:
        raise ShortTenorTrustRegistryError(f"{label} could not be read safely") from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ShortTenorTrustRegistryError(f"{label} SHA-256 differs")
    return raw


def _absolute(value: str | Path, label: str) -> Path:
    try:
        return assert_absolute_path_has_no_links(value)
    except (OSError, TypeError, ValueError) as exc:
        raise ShortTenorTrustRegistryError(f"{label} path is unsafe") from exc


def _require_distinct_paths(paths: Mapping[str, Path]) -> None:
    items = list(paths.items())
    for index, (left_label, left) in enumerate(items):
        for right_label, right in items[index + 1 :]:
            try:
                same = os.path.samefile(left, right)
            except OSError as exc:
                raise ShortTenorTrustRegistryError(
                    f"cannot establish {left_label}/{right_label} path identity"
                ) from exc
            if same:
                raise ShortTenorTrustRegistryError(
                    "trust registry input files must be pairwise distinct"
                )


def _load_json_object(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = load_strict_json(raw.decode("utf-8"))
    except (UnicodeDecodeError, StrictStructuredDataError) as exc:
        raise ShortTenorTrustRegistryError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ShortTenorTrustRegistryError(f"{label} must be a JSON object")
    return value


def _base64(value: object, label: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ShortTenorTrustRegistryError(f"{label} must be base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ShortTenorTrustRegistryError(f"{label} is not standard base64") from exc
    if base64.b64encode(decoded).decode("ascii") != value:
        raise ShortTenorTrustRegistryError(f"{label} is not canonical base64")
    return decoded


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ShortTenorTrustRegistryError(f"{label} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ShortTenorTrustRegistryError(f"{label} keys differ")


def _safe_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ShortTenorTrustRegistryError(f"{label} is invalid")
    return value


def _optional_safe_id(value: object, label: str) -> str | None:
    return None if value is None else _safe_id(value, label)


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ShortTenorTrustRegistryError(f"{label} must be lowercase SHA-256")
    return value


def _optional_sha(value: object, label: str) -> str | None:
    return None if value is None else _sha(value, label)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ShortTenorTrustRegistryError(f"{label} must be a positive integer")
    return value


def _utc(value: object, label: str) -> datetime:
    text = _utc_text(value, label)
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


def _utc_text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ShortTenorTrustRegistryError(f"{label} must be text")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ShortTenorTrustRegistryError(
            f"{label} must be canonical UTC seconds"
        ) from exc
    return value


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise ShortTenorTrustRegistryError(f"{label} differs")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _bounded_sequence(
    value: Sequence[object],
    label: str,
    *,
    maximum: int | None = None,
    exact: int | None = None,
) -> list[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ShortTenorTrustRegistryError(f"{label} must be a sequence")
    result = list(value)
    if not result:
        raise ShortTenorTrustRegistryError(f"{label} must be non-empty")
    if maximum is not None and len(result) > maximum:
        raise ShortTenorTrustRegistryError(f"{label} exceeds maximum length")
    if exact is not None and len(result) != exact:
        raise ShortTenorTrustRegistryError(f"{label} count differs")
    return result


def _normalized_keys(value: Mapping[str, object]) -> list[dict[str, object]]:
    keys = value.get("normalized_keys")
    if not isinstance(keys, list):
        raise ShortTenorTrustRegistryError("normalized registry keys are missing")
    return keys


def _keys_by_role(
    entries: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    return {
        role: [entry for entry in entries if entry["role"] == role]
        for role in REQUIRED_ROLES
    }


__all__ = [
    "GOVERNANCE_KEY_COUNT",
    "GOVERNANCE_SIGNATURE_THRESHOLD",
    "KEY_ID_RULE",
    "LIFECYCLE_EVENT_TYPES",
    "MAXIMUM_REGISTRY_TTL_DAYS",
    "REGISTRY_PAYLOAD_TYPE",
    "REGISTRY_SCHEMA",
    "REQUIRED_ROLES",
    "SIGNATURE_ALGORITHM",
    "TRUST_REGISTRY_CONTRACT_CONTENT_ID",
    "TRUST_REGISTRY_CONTRACT_FILE_SHA256",
    "TRUST_REGISTRY_CONTRACT_SCHEMA",
    "TRUST_REGISTRY_CONTRACT_STATUS",
    "ShortTenorTrustRegistryError",
    "assert_key_accepted_for_trusted_event_time",
    "governance_policy_id",
    "lifecycle_event",
    "registry_dsse_pae",
    "validate_registry_document",
    "validate_trust_registry_contract",
    "verify_trust_registry_chain_paths",
]

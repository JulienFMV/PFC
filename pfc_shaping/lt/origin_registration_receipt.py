"""Public-key-only verification for CH LT origin receipts and registry HEADs.

The module implements the locally hash-frozen receipt/HEAD wire draft.  It
re-verifies the complete signed request chain before accepting either signed
document.  A cryptographically valid ``countable=true`` receipt remains an
untrusted external claim: this library owns no private key, trust registry,
clock, CAS/WORM service, I/O, or origin-counting authority.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from pfc_shaping.lt.origin_registration_envelope import (
    ORIGIN_REGISTRY_PROTOCOL_ID,
    ORIGIN_REGISTRY_PROTOCOL_SHA256,
    SIGNATURE_ALGORITHM,
    canonical_json_bytes,
    public_key_id,
)
from pfc_shaping.lt.origin_registration_request import (
    REGISTRY_DOMAIN_SHA256,
    VerifiedRegistrationRequest,
    verify_signed_origin_registration_request,
)

RECEIPT_SCHEMA_VERSION = "ch_lt_origin_registration_receipt.v2"
HEAD_OBSERVATION_SCHEMA_VERSION = "ch_lt_origin_registry_head_observation.v1"
WIRE_CONTRACT_ID = "290b108770dc799eeb83ca9f0a046aa8436878858224b68f757a5d9bfacbcc33"
WIRE_CONTRACT_SHA256 = "1f9d1f6495f716b7afb3497195626b24fe427b7dedf3f6f001c00051d6688047"
RECEIPT_STATUS = "CRYPTOGRAPHICALLY_VERIFIED_LOCAL_WIRE_DRAFT_NO_EXTERNAL_AUTHORITY_NO_GO"
HEAD_STATUS = "CRYPTOGRAPHICALLY_VERIFIED_FRESH_LOCAL_WIRE_DRAFT_NO_EXTERNAL_AUTHORITY_NO_GO"

_RECEIPT_ID_DOMAIN = b"FMV_CH_LT_ORIGIN_REGISTRATION_RECEIPT_V2"
_RECEIPT_SIGNATURE_DOMAIN = b"FMV_CH_LT_ORIGIN_REGISTRATION_RECEIPT_SIGNATURE_V1"
_HEAD_ID_DOMAIN = b"FMV_CH_LT_ORIGIN_REGISTRY_HEAD_OBSERVATION_V1"
_HEAD_SIGNATURE_DOMAIN = b"FMV_CH_LT_ORIGIN_REGISTRY_HEAD_OBSERVATION_SIGNATURE_V1"
_MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
_MAX_SEQUENCE = 10_000_000
_MAX_HEAD_TTL = timedelta(minutes=5)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CADENCE_SLOT = re.compile(r"^[0-9]{4}-(0[1-9]|1[0-2])$")
_UTC = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")

_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_sha256",
        "protocol_id",
        "registry_domain_sha256",
        "receipt_id",
        "sequence",
        "previous_receipt_id",
        "registration_operation_id",
        "request_id",
        "request_sha256",
        "cadence_slot_id",
        "exact_schedule_entry_id",
        "exact_schedule_entry_sha256",
        "capture_window_open_utc",
        "capture_window_close_utc",
        "latest_external_registry_commit_utc",
        "origin_id",
        "origin_as_of_utc",
        "committed_at_utc",
        "countable_prospective_origin",
        "registry_signature",
    }
)
_RECEIPT_SIGNING_FIELDS = _RECEIPT_FIELDS - {"registry_signature"}
_RECEIPT_CORE_FIELDS = _RECEIPT_SIGNING_FIELDS - {"receipt_id"}
_HEAD_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_sha256",
        "protocol_id",
        "registry_domain_sha256",
        "sequence",
        "receipt_id",
        "receipt_sha256",
        "challenge_nonce",
        "observed_at_utc",
        "expires_at_utc",
        "observation_id",
        "registry_signature",
    }
)
_HEAD_SIGNING_FIELDS = _HEAD_FIELDS - {"registry_signature"}
_HEAD_CORE_FIELDS = _HEAD_SIGNING_FIELDS - {"observation_id"}
_REQUEST_BINDINGS = {
    "protocol_sha256": "protocol_sha256",
    "protocol_id": "protocol_id",
    "registry_domain_sha256": "registry_domain_sha256",
    "registration_operation_id": "registration_operation_id",
    "sequence": "expected_sequence",
    "previous_receipt_id": "expected_previous_receipt_id",
    "request_id": "request_id",
    "cadence_slot_id": "cadence_slot_id",
    "exact_schedule_entry_id": "exact_schedule_entry_id",
    "exact_schedule_entry_sha256": "exact_schedule_entry_sha256",
    "capture_window_open_utc": "capture_window_open_utc",
    "capture_window_close_utc": "capture_window_close_utc",
    "latest_external_registry_commit_utc": "latest_external_registry_commit_utc",
    "origin_id": "origin_id",
    "origin_as_of_utc": "origin_as_of_utc",
}
_SIGNATURE_FIELDS = frozenset({"algorithm", "key_id", "value_base64"})
_MISSING_RECEIPT_EVIDENCE = (
    "FMV_EXTERNAL_WIRE_CONTRACT_APPROVAL",
    "EXTERNAL_REGISTRY_PUBLIC_KEY_TRUST_ADMISSION",
    "INDEPENDENT_REGISTRY_SERVICE_IDENTITY_ACLS_AND_KEYRING",
    "BUILDER_INACCESSIBLE_REMOTE_COMPARE_AND_APPEND_WORM",
    "TRUSTED_EXTERNAL_COMMIT_TIME_AUTHORITY",
    "FRESH_EXTERNAL_REGISTRY_HEAD_OBSERVATION",
    "INDEPENDENT_SERVICE_CONFORMANCE_AND_SECURITY_EVIDENCE",
)
_MISSING_HEAD_EVIDENCE = tuple(
    item for item in _MISSING_RECEIPT_EVIDENCE if item != "FRESH_EXTERNAL_REGISTRY_HEAD_OBSERVATION"
)


class OriginRegistrationReceiptError(ValueError):
    """Raised when a receipt or HEAD observation fails closed."""


@dataclass(frozen=True, slots=True)
class OriginRegistrationVerificationContext:
    """Exact caller-held inputs required to re-verify one signed request."""

    signed_request_payload: bytes
    request_public_key: Ed25519PublicKey
    information_set_envelope_payload: bytes
    schedule_manifest_payload: bytes
    schedule_public_key: Ed25519PublicKey
    trusted_time_receipt_payload: bytes

    def __post_init__(self) -> None:
        for name in (
            "signed_request_payload",
            "information_set_envelope_payload",
            "schedule_manifest_payload",
            "trusted_time_receipt_payload",
        ):
            value = getattr(self, name)
            if not isinstance(value, bytes) or not value or len(value) > _MAX_DOCUMENT_BYTES:
                raise OriginRegistrationReceiptError(f"{name} exact bytes are invalid")
        if not isinstance(self.request_public_key, Ed25519PublicKey):
            raise TypeError("request_public_key must be an Ed25519PublicKey")
        if not isinstance(self.schedule_public_key, Ed25519PublicKey):
            raise TypeError("schedule_public_key must be an Ed25519PublicKey")


@dataclass(frozen=True, slots=True)
class VerifiedRegistrationReceipt:
    """Verified receipt claim with non-overridable negative local authority."""

    receipt_id: str
    receipt_sha256: str
    request_id: str
    registry_signer_key_id: str
    sequence: int
    previous_receipt_id: str | None
    committed_at_utc: str
    cryptographic_receipt_verified: bool = field(default=True, init=False)
    receipt_claimed_countable_prospective_origin: bool = field(default=True, init=False)
    status: str = field(default=RECEIPT_STATUS, init=False)
    registry_public_key_trust_admitted: bool = field(default=False, init=False)
    external_cas_worm_admitted: bool = field(default=False, init=False)
    trusted_commit_time_admitted: bool = field(default=False, init=False)
    fresh_head_observation_cryptographically_verified: bool = field(default=False, init=False)
    externally_registered: bool = field(default=False, init=False)
    countable_origin: bool = field(default=False, init=False)
    truth_open_authorized: bool = field(default=False, init=False)
    model_training_authorized: bool = field(default=False, init=False)
    model_selection_authorized: bool = field(default=False, init=False)
    scientific_admission: bool = field(default=False, init=False)
    production_authorization: bool = field(default=False, init=False)
    promotion_gate: bool = field(default=False, init=False)
    missing_external_evidence: tuple[str, ...] = field(
        default=_MISSING_RECEIPT_EVIDENCE, init=False
    )

    def __post_init__(self) -> None:
        _require_sha256(self.receipt_id, label="receipt id")
        _require_sha256(self.receipt_sha256, label="receipt SHA-256")
        _require_sha256(self.request_id, label="request id")
        _require_sha256(self.registry_signer_key_id, label="registry signer key id")
        _validate_sequence(self.sequence, self.previous_receipt_id)
        _parse_utc(self.committed_at_utc, label="receipt committed-at")

    def to_manifest(self) -> dict[str, object]:
        return _negative_authority_manifest() | {
            "status": self.status,
            "wire_contract_id": WIRE_CONTRACT_ID,
            "wire_contract_sha256": WIRE_CONTRACT_SHA256,
            "receipt_id": self.receipt_id,
            "receipt_sha256": self.receipt_sha256,
            "request_id": self.request_id,
            "registry_signer_key_id": self.registry_signer_key_id,
            "sequence": self.sequence,
            "previous_receipt_id": self.previous_receipt_id,
            "committed_at_utc": self.committed_at_utc,
            "cryptographic_receipt_verified": self.cryptographic_receipt_verified,
            "receipt_claimed_countable_prospective_origin": (
                self.receipt_claimed_countable_prospective_origin
            ),
            "fresh_head_observation_cryptographically_verified": (
                self.fresh_head_observation_cryptographically_verified
            ),
            "missing_external_evidence": list(self.missing_external_evidence),
        }


@dataclass(frozen=True, slots=True)
class VerifiedRegistryHeadObservation:
    """Verified fresh HEAD claim with non-overridable negative local authority."""

    observation_id: str
    receipt_id: str
    receipt_sha256: str
    request_id: str
    registry_signer_key_id: str
    sequence: int
    challenge_nonce: str
    observed_at_utc: str
    expires_at_utc: str
    verification_time_utc: str
    cryptographic_receipt_verified: bool = field(default=True, init=False)
    cryptographic_head_signature_verified: bool = field(default=True, init=False)
    fresh_head_observation_cryptographically_verified: bool = field(default=True, init=False)
    receipt_claimed_countable_prospective_origin: bool = field(default=True, init=False)
    status: str = field(default=HEAD_STATUS, init=False)
    registry_public_key_trust_admitted: bool = field(default=False, init=False)
    external_cas_worm_admitted: bool = field(default=False, init=False)
    trusted_commit_time_admitted: bool = field(default=False, init=False)
    externally_registered: bool = field(default=False, init=False)
    countable_origin: bool = field(default=False, init=False)
    truth_open_authorized: bool = field(default=False, init=False)
    model_training_authorized: bool = field(default=False, init=False)
    model_selection_authorized: bool = field(default=False, init=False)
    scientific_admission: bool = field(default=False, init=False)
    production_authorization: bool = field(default=False, init=False)
    promotion_gate: bool = field(default=False, init=False)
    missing_external_evidence: tuple[str, ...] = field(default=_MISSING_HEAD_EVIDENCE, init=False)

    def __post_init__(self) -> None:
        for label, value in (
            ("observation id", self.observation_id),
            ("receipt id", self.receipt_id),
            ("receipt SHA-256", self.receipt_sha256),
            ("request id", self.request_id),
            ("registry signer key id", self.registry_signer_key_id),
            ("challenge nonce", self.challenge_nonce),
        ):
            _require_sha256(value, label=label)
        _validate_sequence(self.sequence, None if self.sequence == 1 else "0" * 64)
        for label, value in (
            ("HEAD observed-at", self.observed_at_utc),
            ("HEAD expires-at", self.expires_at_utc),
            ("HEAD verification time", self.verification_time_utc),
        ):
            _parse_utc(value, label=label)

    def to_manifest(self) -> dict[str, object]:
        return _negative_authority_manifest() | {
            "status": self.status,
            "wire_contract_id": WIRE_CONTRACT_ID,
            "wire_contract_sha256": WIRE_CONTRACT_SHA256,
            "observation_id": self.observation_id,
            "receipt_id": self.receipt_id,
            "receipt_sha256": self.receipt_sha256,
            "request_id": self.request_id,
            "registry_signer_key_id": self.registry_signer_key_id,
            "sequence": self.sequence,
            "challenge_nonce": self.challenge_nonce,
            "observed_at_utc": self.observed_at_utc,
            "expires_at_utc": self.expires_at_utc,
            "verification_time_utc": self.verification_time_utc,
            "cryptographic_receipt_verified": self.cryptographic_receipt_verified,
            "cryptographic_head_signature_verified": self.cryptographic_head_signature_verified,
            "fresh_head_observation_cryptographically_verified": (
                self.fresh_head_observation_cryptographically_verified
            ),
            "receipt_claimed_countable_prospective_origin": (
                self.receipt_claimed_countable_prospective_origin
            ),
            "missing_external_evidence": list(self.missing_external_evidence),
        }


def build_origin_registration_receipt_signature_payload(
    *,
    context: OriginRegistrationVerificationContext,
    committed_at_utc: str,
) -> bytes:
    """Build domain-separated receipt bytes for an external registry signer."""

    verified_request, request = _verify_request_context(context)
    committed_at = _parse_utc(committed_at_utc, label="receipt committed-at")
    origin = _parse_utc(request["origin_as_of_utc"], label="receipt origin")
    deadline = _parse_utc(request["latest_external_registry_commit_utc"], label="receipt deadline")
    first_target = _parse_utc(
        request["first_target_delivery_start_utc"], label="first target delivery start"
    )
    if not origin <= committed_at <= deadline < first_target:
        raise OriginRegistrationReceiptError("origin receipt chronology is invalid")
    core: dict[str, object] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "protocol_sha256": request["protocol_sha256"],
        "protocol_id": request["protocol_id"],
        "registry_domain_sha256": request["registry_domain_sha256"],
        "sequence": request["expected_sequence"],
        "previous_receipt_id": request["expected_previous_receipt_id"],
        "registration_operation_id": request["registration_operation_id"],
        "request_id": request["request_id"],
        "request_sha256": verified_request.request_sha256,
        "cadence_slot_id": request["cadence_slot_id"],
        "exact_schedule_entry_id": request["exact_schedule_entry_id"],
        "exact_schedule_entry_sha256": request["exact_schedule_entry_sha256"],
        "capture_window_open_utc": request["capture_window_open_utc"],
        "capture_window_close_utc": request["capture_window_close_utc"],
        "latest_external_registry_commit_utc": request["latest_external_registry_commit_utc"],
        "origin_id": request["origin_id"],
        "origin_as_of_utc": request["origin_as_of_utc"],
        "committed_at_utc": committed_at_utc,
        "countable_prospective_origin": True,
    }
    _exact_fields(core, _RECEIPT_CORE_FIELDS, label="receipt identity core")
    document = {**core, "receipt_id": _domain_hash(_RECEIPT_ID_DOMAIN, core)}
    return _signature_payload(_RECEIPT_SIGNATURE_DOMAIN, document)


def assemble_signed_origin_registration_receipt(
    *, signature_payload: bytes, signer_key_id: str, signature_base64: str
) -> bytes:
    """Attach caller-supplied receipt-signature bytes without a private key."""

    document = _signature_document(
        signature_payload,
        domain=_RECEIPT_SIGNATURE_DOMAIN,
        fields=_RECEIPT_SIGNING_FIELDS,
        label="receipt signature payload",
    )
    _validate_receipt_identity(document)
    signature = _signature_block(signer_key_id, signature_base64)
    return canonical_json_bytes({**document, "registry_signature": signature})


def verify_signed_origin_registration_receipt(
    receipt_payload: bytes,
    *,
    registry_public_key: Ed25519PublicKey,
    context: OriginRegistrationVerificationContext,
) -> VerifiedRegistrationReceipt:
    """Verify receipt cryptography and every request-chain binding."""

    _validate_registry_key_separation(registry_public_key, context)
    verified_request, request = _verify_request_context(context)
    receipt = _strict_canonical_mapping(receipt_payload, label="signed origin receipt")
    _exact_fields(receipt, _RECEIPT_FIELDS, label="signed origin receipt")
    signature = receipt["registry_signature"]
    signing_document = dict(receipt)
    signing_document.pop("registry_signature")
    _validate_receipt_identity(signing_document)
    _verify_signature(
        signature,
        _signature_payload(_RECEIPT_SIGNATURE_DOMAIN, signing_document),
        registry_public_key,
        label="receipt",
    )
    for receipt_field, request_field in _REQUEST_BINDINGS.items():
        if receipt[receipt_field] != request[request_field]:
            raise OriginRegistrationReceiptError(
                f"receipt/request binding mismatch: {receipt_field}"
            )
    if receipt["request_sha256"] != verified_request.request_sha256:
        raise OriginRegistrationReceiptError("receipt request SHA-256 binding mismatch")
    expected = build_origin_registration_receipt_signature_payload(
        context=context,
        committed_at_utc=str(receipt["committed_at_utc"]),
    )
    if signing_document != _signature_document(
        expected,
        domain=_RECEIPT_SIGNATURE_DOMAIN,
        fields=_RECEIPT_SIGNING_FIELDS,
        label="expected receipt signature payload",
    ):
        raise OriginRegistrationReceiptError("signed receipt does not match its request inputs")
    return VerifiedRegistrationReceipt(
        receipt_id=str(receipt["receipt_id"]),
        receipt_sha256=hashlib.sha256(receipt_payload).hexdigest(),
        request_id=verified_request.request_id,
        registry_signer_key_id=public_key_id(registry_public_key),
        sequence=verified_request.expected_sequence,
        previous_receipt_id=verified_request.expected_previous_receipt_id,
        committed_at_utc=str(receipt["committed_at_utc"]),
    )


def build_registry_head_observation_signature_payload(
    *,
    receipt_payload: bytes,
    registry_public_key: Ed25519PublicKey,
    context: OriginRegistrationVerificationContext,
    challenge_nonce: str,
    observed_at_utc: str,
    expires_at_utc: str,
) -> bytes:
    """Build domain-separated fresh-HEAD bytes for the external registry signer."""

    _require_sha256(challenge_nonce, label="HEAD challenge nonce")
    verified_receipt = verify_signed_origin_registration_receipt(
        receipt_payload,
        registry_public_key=registry_public_key,
        context=context,
    )
    observed = _parse_utc(observed_at_utc, label="HEAD observed-at")
    expires = _parse_utc(expires_at_utc, label="HEAD expires-at")
    committed = _parse_utc(verified_receipt.committed_at_utc, label="receipt committed-at")
    if not committed <= observed < expires or expires - observed > _MAX_HEAD_TTL:
        raise OriginRegistrationReceiptError("HEAD observation chronology or TTL is invalid")
    core: dict[str, object] = {
        "schema_version": HEAD_OBSERVATION_SCHEMA_VERSION,
        "protocol_sha256": ORIGIN_REGISTRY_PROTOCOL_SHA256,
        "protocol_id": ORIGIN_REGISTRY_PROTOCOL_ID,
        "registry_domain_sha256": REGISTRY_DOMAIN_SHA256,
        "sequence": verified_receipt.sequence,
        "receipt_id": verified_receipt.receipt_id,
        "receipt_sha256": verified_receipt.receipt_sha256,
        "challenge_nonce": challenge_nonce,
        "observed_at_utc": observed_at_utc,
        "expires_at_utc": expires_at_utc,
    }
    _exact_fields(core, _HEAD_CORE_FIELDS, label="HEAD observation identity core")
    document = {**core, "observation_id": _domain_hash(_HEAD_ID_DOMAIN, core)}
    return _signature_payload(_HEAD_SIGNATURE_DOMAIN, document)


def assemble_signed_registry_head_observation(
    *, signature_payload: bytes, signer_key_id: str, signature_base64: str
) -> bytes:
    """Attach caller-supplied HEAD-signature bytes without a private key."""

    document = _signature_document(
        signature_payload,
        domain=_HEAD_SIGNATURE_DOMAIN,
        fields=_HEAD_SIGNING_FIELDS,
        label="HEAD signature payload",
    )
    _validate_head_identity(document)
    signature = _signature_block(signer_key_id, signature_base64)
    return canonical_json_bytes({**document, "registry_signature": signature})


def verify_signed_registry_head_observation(
    observation_payload: bytes,
    *,
    receipt_payload: bytes,
    registry_public_key: Ed25519PublicKey,
    context: OriginRegistrationVerificationContext,
    expected_challenge_nonce: str,
    verification_time_utc: datetime,
) -> VerifiedRegistryHeadObservation:
    """Verify a nonce-bound fresh HEAD while retaining negative authority."""

    _require_sha256(expected_challenge_nonce, label="expected HEAD challenge nonce")
    verification_time = _canonical_utc_datetime(
        verification_time_utc, label="HEAD verification time"
    )
    verified_receipt = verify_signed_origin_registration_receipt(
        receipt_payload,
        registry_public_key=registry_public_key,
        context=context,
    )
    observation = _strict_canonical_mapping(
        observation_payload, label="signed registry HEAD observation"
    )
    _exact_fields(observation, _HEAD_FIELDS, label="signed registry HEAD observation")
    signature = observation["registry_signature"]
    signing_document = dict(observation)
    signing_document.pop("registry_signature")
    _validate_head_identity(signing_document)
    _verify_signature(
        signature,
        _signature_payload(_HEAD_SIGNATURE_DOMAIN, signing_document),
        registry_public_key,
        label="HEAD",
    )
    expected_bindings = {
        "protocol_sha256": ORIGIN_REGISTRY_PROTOCOL_SHA256,
        "protocol_id": ORIGIN_REGISTRY_PROTOCOL_ID,
        "registry_domain_sha256": REGISTRY_DOMAIN_SHA256,
        "sequence": verified_receipt.sequence,
        "receipt_id": verified_receipt.receipt_id,
        "receipt_sha256": verified_receipt.receipt_sha256,
        "challenge_nonce": expected_challenge_nonce,
    }
    for field_name, expected in expected_bindings.items():
        if observation[field_name] != expected:
            raise OriginRegistrationReceiptError(f"HEAD binding mismatch: {field_name}")
    observed = _parse_utc(observation["observed_at_utc"], label="HEAD observed-at")
    expires = _parse_utc(observation["expires_at_utc"], label="HEAD expires-at")
    committed = _parse_utc(verified_receipt.committed_at_utc, label="receipt committed-at")
    if not committed <= observed <= verification_time <= expires:
        raise OriginRegistrationReceiptError("HEAD observation is stale or not yet valid")
    return VerifiedRegistryHeadObservation(
        observation_id=str(observation["observation_id"]),
        receipt_id=verified_receipt.receipt_id,
        receipt_sha256=verified_receipt.receipt_sha256,
        request_id=verified_receipt.request_id,
        registry_signer_key_id=verified_receipt.registry_signer_key_id,
        sequence=verified_receipt.sequence,
        challenge_nonce=expected_challenge_nonce,
        observed_at_utc=str(observation["observed_at_utc"]),
        expires_at_utc=str(observation["expires_at_utc"]),
        verification_time_utc=_format_utc(verification_time),
    )


def _verify_request_context(
    context: OriginRegistrationVerificationContext,
) -> tuple[VerifiedRegistrationRequest, dict[str, object]]:
    if not isinstance(context, OriginRegistrationVerificationContext):
        raise TypeError("context must be an OriginRegistrationVerificationContext")
    verified = verify_signed_origin_registration_request(
        context.signed_request_payload,
        trusted_request_public_key=context.request_public_key,
        information_set_envelope_payload=context.information_set_envelope_payload,
        schedule_manifest_payload=context.schedule_manifest_payload,
        trusted_schedule_public_key=context.schedule_public_key,
        trusted_time_receipt_payload=context.trusted_time_receipt_payload,
    )
    request = _strict_canonical_mapping(
        context.signed_request_payload, label="signed origin request"
    )
    return verified, request


def _validate_registry_key_separation(
    registry_public_key: Ed25519PublicKey,
    context: OriginRegistrationVerificationContext,
) -> None:
    if not isinstance(context, OriginRegistrationVerificationContext):
        raise TypeError("context must be an OriginRegistrationVerificationContext")
    if not isinstance(registry_public_key, Ed25519PublicKey):
        raise TypeError("registry_public_key must be an Ed25519PublicKey")
    registry_key_id = public_key_id(registry_public_key)
    if registry_key_id in {
        public_key_id(context.request_public_key),
        public_key_id(context.schedule_public_key),
    }:
        raise OriginRegistrationReceiptError(
            "registry, request, and schedule signer roles must be cryptographically disjoint"
        )


def _validate_receipt_identity(document: Mapping[str, object]) -> None:
    _exact_fields(document, _RECEIPT_SIGNING_FIELDS, label="receipt signing document")
    core = dict(document)
    receipt_id = core.pop("receipt_id")
    _exact_fields(core, _RECEIPT_CORE_FIELDS, label="receipt identity core")
    if receipt_id != _domain_hash(_RECEIPT_ID_DOMAIN, core):
        raise OriginRegistrationReceiptError("origin receipt id mismatch")
    if (
        document["schema_version"] != RECEIPT_SCHEMA_VERSION
        or document["protocol_sha256"] != ORIGIN_REGISTRY_PROTOCOL_SHA256
        or document["protocol_id"] != ORIGIN_REGISTRY_PROTOCOL_ID
        or document["registry_domain_sha256"] != REGISTRY_DOMAIN_SHA256
        or document["countable_prospective_origin"] is not True
    ):
        raise OriginRegistrationReceiptError("origin receipt semantics are invalid")
    _canonical_uuid(document["registration_operation_id"])
    _validate_sequence(document["sequence"], document["previous_receipt_id"])
    cadence_slot = document["cadence_slot_id"]
    if not isinstance(cadence_slot, str) or _CADENCE_SLOT.fullmatch(cadence_slot) is None:
        raise OriginRegistrationReceiptError("receipt cadence slot is invalid")
    capture_open = _parse_utc(document["capture_window_open_utc"], label="capture open")
    origin = _parse_utc(document["origin_as_of_utc"], label="receipt origin")
    capture_close = _parse_utc(document["capture_window_close_utc"], label="capture close")
    committed = _parse_utc(document["committed_at_utc"], label="receipt committed-at")
    deadline = _parse_utc(document["latest_external_registry_commit_utc"], label="receipt deadline")
    if (
        not capture_open <= origin <= capture_close <= deadline
        or not origin <= committed <= deadline
    ):
        raise OriginRegistrationReceiptError("origin receipt chronology is invalid")
    for name in (
        "protocol_sha256",
        "registry_domain_sha256",
        "receipt_id",
        "request_id",
        "request_sha256",
        "exact_schedule_entry_id",
        "exact_schedule_entry_sha256",
        "origin_id",
    ):
        _require_sha256(document[name], label=name)


def _validate_head_identity(document: Mapping[str, object]) -> None:
    _exact_fields(document, _HEAD_SIGNING_FIELDS, label="HEAD signing document")
    core = dict(document)
    observation_id = core.pop("observation_id")
    _exact_fields(core, _HEAD_CORE_FIELDS, label="HEAD observation identity core")
    if observation_id != _domain_hash(_HEAD_ID_DOMAIN, core):
        raise OriginRegistrationReceiptError("HEAD observation id mismatch")
    if (
        document["schema_version"] != HEAD_OBSERVATION_SCHEMA_VERSION
        or document["protocol_sha256"] != ORIGIN_REGISTRY_PROTOCOL_SHA256
        or document["protocol_id"] != ORIGIN_REGISTRY_PROTOCOL_ID
        or document["registry_domain_sha256"] != REGISTRY_DOMAIN_SHA256
    ):
        raise OriginRegistrationReceiptError("HEAD observation semantics are invalid")
    _validate_sequence_number(document["sequence"])
    for name in (
        "protocol_sha256",
        "registry_domain_sha256",
        "receipt_id",
        "receipt_sha256",
        "challenge_nonce",
        "observation_id",
    ):
        _require_sha256(document[name], label=name)
    observed = _parse_utc(document["observed_at_utc"], label="HEAD observed-at")
    expires = _parse_utc(document["expires_at_utc"], label="HEAD expires-at")
    if not observed < expires or expires - observed > _MAX_HEAD_TTL:
        raise OriginRegistrationReceiptError("HEAD observation chronology or TTL is invalid")


def _verify_signature(
    signature: object,
    payload: bytes,
    public_key: Ed25519PublicKey,
    *,
    label: str,
) -> None:
    if not isinstance(signature, Mapping):
        raise OriginRegistrationReceiptError(f"{label} signature is invalid")
    _exact_fields(signature, _SIGNATURE_FIELDS, label=f"{label} signature")
    if signature["algorithm"] != SIGNATURE_ALGORITHM or signature["key_id"] != public_key_id(
        public_key
    ):
        raise OriginRegistrationReceiptError(f"{label} signature trust binding is invalid")
    value = _decode_signature(signature["value_base64"], label=label)
    try:
        public_key.verify(value, payload)
    except InvalidSignature as exc:
        raise OriginRegistrationReceiptError(f"{label} signature is invalid") from exc


def _signature_block(signer_key_id: str, signature_base64: str) -> dict[str, str]:
    _require_sha256(signer_key_id, label="registry signer key id")
    _decode_signature(signature_base64, label="registry")
    return {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": signer_key_id,
        "value_base64": signature_base64,
    }


def _signature_payload(domain: bytes, document: Mapping[str, object]) -> bytes:
    return domain + b"\x00" + canonical_json_bytes(document)


def _signature_document(
    payload: bytes,
    *,
    domain: bytes,
    fields: frozenset[str],
    label: str,
) -> dict[str, object]:
    prefix = domain + b"\x00"
    if not isinstance(payload, bytes) or not payload.startswith(prefix):
        raise OriginRegistrationReceiptError(f"{label} domain separation is invalid")
    document = _strict_canonical_mapping(payload[len(prefix) :], label=label)
    _exact_fields(document, fields, label=label)
    return document


def _strict_canonical_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_DOCUMENT_BYTES:
        raise OriginRegistrationReceiptError(f"{label} byte envelope is invalid")
    try:
        parsed = json.loads(payload.decode("ascii"), object_pairs_hook=_reject_duplicate_keys)
    except OriginRegistrationReceiptError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise OriginRegistrationReceiptError(f"{label} is not strict JSON") from exc
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        raise OriginRegistrationReceiptError(f"{label} must be an object")
    try:
        canonical = canonical_json_bytes(parsed)
    except ValueError as exc:
        raise OriginRegistrationReceiptError(f"{label} is not canonical JSON") from exc
    if canonical != payload:
        raise OriginRegistrationReceiptError(f"{label} is not exact canonical JSON")
    return parsed


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise OriginRegistrationReceiptError("strict JSON contains a duplicate key")
        document[key] = value
    return document


def _decode_signature(value: object, *, label: str) -> bytes:
    if not isinstance(value, str):
        raise OriginRegistrationReceiptError(f"{label} signature encoding is invalid")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise OriginRegistrationReceiptError(f"{label} signature encoding is invalid") from exc
    if len(decoded) != 64 or base64.b64encode(decoded).decode("ascii") != value:
        raise OriginRegistrationReceiptError(f"{label} signature encoding is invalid")
    return decoded


def _validate_sequence(sequence: object, previous_receipt_id: object) -> None:
    _validate_sequence_number(sequence)
    if sequence == 1:
        if previous_receipt_id is not None:
            raise OriginRegistrationReceiptError("genesis receipt cannot name a predecessor")
    else:
        _require_sha256(previous_receipt_id, label="previous receipt id")


def _validate_sequence_number(sequence: object) -> None:
    if type(sequence) is not int or not 1 <= sequence <= _MAX_SEQUENCE:
        raise OriginRegistrationReceiptError("registry sequence is invalid")


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise OriginRegistrationReceiptError(f"{label} must be canonical UTC seconds")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise OriginRegistrationReceiptError(f"{label} is invalid") from exc


def _canonical_utc_datetime(value: datetime, *, label: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is not timezone.utc
        or value.microsecond != 0
    ):
        raise OriginRegistrationReceiptError(f"{label} must be UTC with whole seconds")
    return value


def _format_utc(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_uuid(value: object) -> str:
    if not isinstance(value, str):
        raise OriginRegistrationReceiptError("registration operation id is invalid")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise OriginRegistrationReceiptError("registration operation id is invalid") from exc
    if str(parsed) != value:
        raise OriginRegistrationReceiptError("registration operation id must be canonical UUID")
    return value


def _domain_hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + b"\x00" + canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise OriginRegistrationReceiptError(f"{label} must be a lowercase SHA-256")


def _exact_fields(
    value: Mapping[str, object], expected: set[str] | frozenset[str], *, label: str
) -> None:
    if set(value) != set(expected):
        raise OriginRegistrationReceiptError(f"{label} fields are not exact")


def _negative_authority_manifest() -> dict[str, object]:
    return {
        "registry_public_key_trust_admitted": False,
        "external_cas_worm_admitted": False,
        "trusted_commit_time_admitted": False,
        "externally_registered": False,
        "countable_origin": False,
        "truth_open_authorized": False,
        "model_training_authorized": False,
        "model_selection_authorized": False,
        "scientific_admission": False,
        "production_authorization": False,
        "promotion_gate": False,
    }


__all__ = [
    "HEAD_OBSERVATION_SCHEMA_VERSION",
    "HEAD_STATUS",
    "RECEIPT_SCHEMA_VERSION",
    "RECEIPT_STATUS",
    "WIRE_CONTRACT_ID",
    "WIRE_CONTRACT_SHA256",
    "OriginRegistrationReceiptError",
    "OriginRegistrationVerificationContext",
    "VerifiedRegistrationReceipt",
    "VerifiedRegistryHeadObservation",
    "assemble_signed_origin_registration_receipt",
    "assemble_signed_registry_head_observation",
    "build_origin_registration_receipt_signature_payload",
    "build_registry_head_observation_signature_payload",
    "verify_signed_origin_registration_receipt",
    "verify_signed_registry_head_observation",
]

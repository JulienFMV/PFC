"""Authority-negative trust and state conformance for the CH LT origin registry.

The module verifies caller-signed public-key trust bundles and models the
transport-neutral registry state machine in memory.  It owns no private key,
clock, nonce, filesystem, database, network, or external CAS/WORM authority.
Synthetic state transitions and cryptographic verification never count an
origin or admit a trust root.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from pfc_shaping.lt.origin_registration_envelope import (
    SIGNATURE_ALGORITHM,
    canonical_json_bytes,
    public_key_id,
)
from pfc_shaping.lt.origin_registration_receipt import (
    WIRE_CONTRACT_SHA256,
    OriginRegistrationVerificationContext,
    VerifiedRegistrationReceipt,
    verify_signed_origin_registration_receipt,
)
from pfc_shaping.lt.origin_registration_request import REGISTRY_DOMAIN_SHA256

CONTRACT_ID = "84839a92bc62426c964da9eaefcf719be80808c79089b7ac2142af14d033206b"
CONTRACT_SHA256 = "57ce79771d25cf6a858a2c2fef585d202b66d05aff6a107b834f2afb9407df39"
TRUST_BUNDLE_SCHEMA_VERSION = "ch_lt_origin_registry_trust_bundle.v1"
TRUST_BUNDLE_STATUS = "CRYPTOGRAPHICALLY_VERIFIED_LOCAL_TRUST_DRAFT_NO_EXTERNAL_ADMISSION_NO_GO"
SYNTHETIC_APPEND_STATUS = "SYNTHETIC_IN_MEMORY_APPEND_CONFORMANCE_ONLY_NO_AUTHORITY_NO_GO"
SYNTHETIC_REJECTION_SCHEMA_VERSION = "ch_lt_origin_registry_synthetic_rejection.v1"

_BUNDLE_ID_DOMAIN = b"FMV_CH_LT_ORIGIN_REGISTRY_TRUST_BUNDLE_V1"
_BUNDLE_SIGNATURE_DOMAIN = b"FMV_CH_LT_ORIGIN_REGISTRY_TRUST_BUNDLE_SIGNATURE_V1"
_REJECTION_ID_DOMAIN = b"FMV_CH_LT_ORIGIN_REGISTRY_SYNTHETIC_REJECTION_V1"
_MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
_MAX_SEQUENCE = 10_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CADENCE_SLOT = re.compile(r"^[0-9]{4}-(0[1-9]|1[0-2])$")
_UTC = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_SIGNATURE_FIELDS = frozenset({"algorithm", "key_id", "value_base64"})
_KEY_FIELDS = frozenset(
    {
        "key_id",
        "algorithm",
        "public_key_base64",
        "valid_from_utc",
        "valid_until_utc",
        "lifecycle_status",
    }
)
_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "contract_sha256",
        "receipt_head_wire_contract_sha256",
        "registry_domain_sha256",
        "bundle_id",
        "revision",
        "previous_bundle_id",
        "issued_at_utc",
        "registry_keys",
        "trust_root_signature",
    }
)
_BUNDLE_SIGNING_FIELDS = _BUNDLE_FIELDS - {"trust_root_signature"}
_BUNDLE_CORE_FIELDS = _BUNDLE_SIGNING_FIELDS - {"bundle_id"}
_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "ACTIVE_FOR_NEW_SIGNATURES": frozenset(
        {
            "ACTIVE_FOR_NEW_SIGNATURES",
            "HISTORICAL_VERIFY_ONLY",
            "REVOKED_DO_NOT_TRUST",
            "COMPROMISED_DO_NOT_TRUST",
        }
    ),
    "HISTORICAL_VERIFY_ONLY": frozenset(
        {
            "HISTORICAL_VERIFY_ONLY",
            "REVOKED_DO_NOT_TRUST",
            "COMPROMISED_DO_NOT_TRUST",
        }
    ),
    "REVOKED_DO_NOT_TRUST": frozenset({"REVOKED_DO_NOT_TRUST"}),
    "COMPROMISED_DO_NOT_TRUST": frozenset({"COMPROMISED_DO_NOT_TRUST"}),
}
_MISSING_EXTERNAL_EVIDENCE = (
    "FMV_EXTERNAL_TRUST_AND_TRANSPORT_CONTRACT_APPROVAL",
    "TRUST_ROOT_IDENTITY_AND_PUBLIC_KEY_EXTERNAL_ADMISSION",
    "INDEPENDENT_REGISTRY_SERVICE_IDENTITY_ACLS_AND_KEY_CUSTODY",
    "REMOTE_LINEARIZABLE_COMPARE_AND_APPEND_WORM_EVIDENCE",
    "TRUSTED_EXTERNAL_COMMIT_TIME_AUTHORITY",
    "TRANSPORT_AUTHENTICATION_REPLAY_AND_AVAILABILITY_CONTROLS",
    "INDEPENDENT_SERVICE_CONFORMANCE_AND_SECURITY_EVIDENCE",
)


class OriginRegistryConformanceError(ValueError):
    """Raised when local trust or synthetic conformance fails closed."""


class OriginRegistryConformanceConflict(OriginRegistryConformanceError):
    """Bounded state conflict with an optional immutable rejection record."""

    def __init__(
        self,
        error_code: str,
        *,
        rejection: SyntheticRejectedOperation | None = None,
    ) -> None:
        super().__init__(error_code)
        self.error_code = error_code
        self.rejection = rejection


class RegistryKeyLifecycle(str, Enum):
    ACTIVE = "ACTIVE_FOR_NEW_SIGNATURES"
    HISTORICAL = "HISTORICAL_VERIFY_ONLY"
    REVOKED = "REVOKED_DO_NOT_TRUST"
    COMPROMISED = "COMPROMISED_DO_NOT_TRUST"


@dataclass(frozen=True, slots=True)
class RegistryPublicKeySpec:
    """One public registry-key record prepared for a signed trust bundle."""

    public_key: Ed25519PublicKey
    valid_from_utc: str
    valid_until_utc: str
    lifecycle_status: RegistryKeyLifecycle

    def __post_init__(self) -> None:
        if not isinstance(self.public_key, Ed25519PublicKey):
            raise TypeError("public_key must be an Ed25519PublicKey")
        valid_from = _parse_utc(self.valid_from_utc, label="registry key valid-from")
        valid_until = _parse_utc(self.valid_until_utc, label="registry key valid-until")
        if valid_from >= valid_until:
            raise OriginRegistryConformanceError("registry key validity window is invalid")
        if not isinstance(self.lifecycle_status, RegistryKeyLifecycle):
            raise TypeError("lifecycle_status must be a RegistryKeyLifecycle")

    @property
    def key_id(self) -> str:
        return public_key_id(self.public_key)

    def to_wire(self) -> dict[str, object]:
        raw = _public_key_bytes(self.public_key)
        return {
            "key_id": self.key_id,
            "algorithm": SIGNATURE_ALGORITHM,
            "public_key_base64": base64.b64encode(raw).decode("ascii"),
            "valid_from_utc": self.valid_from_utc,
            "valid_until_utc": self.valid_until_utc,
            "lifecycle_status": self.lifecycle_status.value,
        }


@dataclass(frozen=True, slots=True)
class RegistryPublicKeyRecord:
    """Validated immutable public key and its frozen signing lifecycle."""

    key_id: str
    public_key_bytes: bytes = field(repr=False)
    valid_from_utc: str
    valid_until_utc: str
    lifecycle_status: RegistryKeyLifecycle

    def __post_init__(self) -> None:
        _require_sha256(self.key_id, label="registry key id")
        if (
            not isinstance(self.public_key_bytes, bytes)
            or len(self.public_key_bytes) != 32
            or hashlib.sha256(self.public_key_bytes).hexdigest() != self.key_id
        ):
            raise OriginRegistryConformanceError("registry public key identity is invalid")
        if not isinstance(self.lifecycle_status, RegistryKeyLifecycle):
            raise TypeError("lifecycle_status must be a RegistryKeyLifecycle")
        if _parse_utc(self.valid_from_utc, label="registry key valid-from") >= _parse_utc(
            self.valid_until_utc, label="registry key valid-until"
        ):
            raise OriginRegistryConformanceError("registry key validity window is invalid")

    def public_key(self) -> Ed25519PublicKey:
        return Ed25519PublicKey.from_public_bytes(self.public_key_bytes)

    def to_manifest(self) -> dict[str, object]:
        return {
            "key_id": self.key_id,
            "algorithm": SIGNATURE_ALGORITHM,
            "valid_from_utc": self.valid_from_utc,
            "valid_until_utc": self.valid_until_utc,
            "lifecycle_status": self.lifecycle_status.value,
        }


@dataclass(frozen=True, slots=True)
class VerifiedRegistryTrustBundle:
    """Cryptographically verified bundle without external root admission."""

    bundle_id: str
    bundle_sha256: str
    revision: int
    previous_bundle_id: str | None
    issued_at_utc: str
    trust_root_key_id: str
    registry_keys: tuple[RegistryPublicKeyRecord, ...]
    cryptographic_trust_root_signature_verified: bool = field(default=True, init=False)
    registry_key_lifecycle_locally_verified: bool = field(default=True, init=False)
    status: str = field(default=TRUST_BUNDLE_STATUS, init=False)
    trust_root_externally_admitted: bool = field(default=False, init=False)
    external_cas_worm_verified: bool = field(default=False, init=False)
    externally_registered: bool = field(default=False, init=False)
    countable_origin: bool = field(default=False, init=False)
    truth_open_authorized: bool = field(default=False, init=False)
    model_training_authorized: bool = field(default=False, init=False)
    model_selection_authorized: bool = field(default=False, init=False)
    scientific_admission: bool = field(default=False, init=False)
    production_authorization: bool = field(default=False, init=False)
    promotion_gate: bool = field(default=False, init=False)
    missing_external_evidence: tuple[str, ...] = field(
        default=_MISSING_EXTERNAL_EVIDENCE, init=False
    )

    def __post_init__(self) -> None:
        _require_sha256(self.bundle_id, label="trust bundle id")
        _require_sha256(self.bundle_sha256, label="trust bundle SHA-256")
        _validate_revision(self.revision, self.previous_bundle_id)
        _parse_utc(self.issued_at_utc, label="trust bundle issued-at")
        _require_sha256(self.trust_root_key_id, label="trust root key id")
        if not self.registry_keys:
            raise OriginRegistryConformanceError("registry key inventory is empty")
        _validate_record_inventory(self.registry_keys, issued_at_utc=self.issued_at_utc)
        if self.trust_root_key_id in {record.key_id for record in self.registry_keys}:
            raise OriginRegistryConformanceError(
                "trust root and registry signer roles must be cryptographically disjoint"
            )

    @property
    def active_key_id(self) -> str:
        return next(
            key.key_id
            for key in self.registry_keys
            if key.lifecycle_status is RegistryKeyLifecycle.ACTIVE
        )

    def public_key_for_receipt(self, *, key_id: str, committed_at_utc: str) -> Ed25519PublicKey:
        """Resolve a locally eligible key; this does not admit external trust."""

        _require_sha256(key_id, label="receipt registry signer key id")
        committed_at = _parse_utc(committed_at_utc, label="receipt committed-at")
        matches = tuple(key for key in self.registry_keys if key.key_id == key_id)
        if len(matches) != 1:
            raise OriginRegistryConformanceError("receipt registry signer is absent from bundle")
        selected = matches[0]
        if selected.lifecycle_status in {
            RegistryKeyLifecycle.REVOKED,
            RegistryKeyLifecycle.COMPROMISED,
        }:
            raise OriginRegistryConformanceError("receipt registry signer is not trusted")
        valid_from = _parse_utc(selected.valid_from_utc, label="registry key valid-from")
        valid_until = _parse_utc(selected.valid_until_utc, label="registry key valid-until")
        if not valid_from <= committed_at < valid_until:
            raise OriginRegistryConformanceError("receipt commit is outside registry key validity")
        return selected.public_key()

    def to_manifest(self) -> dict[str, object]:
        return _negative_authority_manifest() | {
            "status": self.status,
            "contract_id": CONTRACT_ID,
            "contract_sha256": CONTRACT_SHA256,
            "bundle_id": self.bundle_id,
            "bundle_sha256": self.bundle_sha256,
            "revision": self.revision,
            "previous_bundle_id": self.previous_bundle_id,
            "issued_at_utc": self.issued_at_utc,
            "trust_root_key_id": self.trust_root_key_id,
            "active_key_id": self.active_key_id,
            "registry_keys": [key.to_manifest() for key in self.registry_keys],
            "cryptographic_trust_root_signature_verified": (
                self.cryptographic_trust_root_signature_verified
            ),
            "registry_key_lifecycle_locally_verified": (
                self.registry_key_lifecycle_locally_verified
            ),
            "missing_external_evidence": list(self.missing_external_evidence),
        }


@dataclass(frozen=True, slots=True)
class SyntheticAppendCandidate:
    """Exact synthetic transition candidate; never an external admission."""

    request_payload: bytes = field(repr=False)
    receipt_payload: bytes = field(repr=False)
    registration_operation_id: str
    request_id: str
    cadence_slot_id: str
    origin_id: str
    origin_as_of_utc: str
    expected_sequence: int
    expected_previous_receipt_id: str | None
    receipt_id: str

    def __post_init__(self) -> None:
        _exact_payload_sha256(self.request_payload, label="synthetic request")
        _exact_payload_sha256(self.receipt_payload, label="synthetic receipt")
        _canonical_uuid(self.registration_operation_id)
        for label, value in (
            ("request id", self.request_id),
            ("origin id", self.origin_id),
            ("receipt id", self.receipt_id),
        ):
            _require_sha256(value, label=label)
        if _CADENCE_SLOT.fullmatch(self.cadence_slot_id) is None:
            raise OriginRegistryConformanceError("cadence slot id is invalid")
        _parse_utc(self.origin_as_of_utc, label="origin as-of")
        _validate_sequence(self.expected_sequence, self.expected_previous_receipt_id)

    @property
    def request_sha256(self) -> str:
        return hashlib.sha256(self.request_payload).hexdigest()

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.receipt_payload).hexdigest()


@dataclass(frozen=True, slots=True)
class SyntheticCommittedOperation:
    """One exact in-memory append with permanently negative authority."""

    candidate: SyntheticAppendCandidate = field(repr=False)
    synthetic_sequence: int
    status: str = field(default=SYNTHETIC_APPEND_STATUS, init=False)
    synthetic_state_transition: bool = field(default=True, init=False)
    external_cas_worm_verified: bool = field(default=False, init=False)
    externally_registered: bool = field(default=False, init=False)
    countable_origin: bool = field(default=False, init=False)
    truth_open_authorized: bool = field(default=False, init=False)
    model_training_authorized: bool = field(default=False, init=False)
    model_selection_authorized: bool = field(default=False, init=False)
    scientific_admission: bool = field(default=False, init=False)
    production_authorization: bool = field(default=False, init=False)
    promotion_gate: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, SyntheticAppendCandidate):
            raise TypeError("candidate must be a SyntheticAppendCandidate")
        if self.synthetic_sequence != self.candidate.expected_sequence:
            raise OriginRegistryConformanceError("synthetic append sequence is inconsistent")

    @property
    def request_payload(self) -> bytes:
        return self.candidate.request_payload

    @property
    def receipt_payload(self) -> bytes:
        return self.candidate.receipt_payload

    def to_manifest(self) -> dict[str, object]:
        return _negative_authority_manifest() | {
            "status": self.status,
            "synthetic_state_transition": self.synthetic_state_transition,
            "synthetic_sequence": self.synthetic_sequence,
            "registration_operation_id": self.candidate.registration_operation_id,
            "request_id": self.candidate.request_id,
            "request_sha256": self.candidate.request_sha256,
            "receipt_id": self.candidate.receipt_id,
            "receipt_sha256": self.candidate.receipt_sha256,
            "cadence_slot_id": self.candidate.cadence_slot_id,
            "origin_id": self.candidate.origin_id,
            "origin_as_of_utc": self.candidate.origin_as_of_utc,
        }


@dataclass(frozen=True, slots=True)
class SyntheticAppendOutcome:
    committed: SyntheticCommittedOperation
    idempotent_retry: bool
    countable_origin: bool = field(default=False, init=False)
    production_authorization: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.committed, SyntheticCommittedOperation):
            raise TypeError("committed must be a SyntheticCommittedOperation")
        if type(self.idempotent_retry) is not bool:
            raise TypeError("idempotent_retry must be bool")


@dataclass(frozen=True, slots=True)
class SyntheticRejectedOperation:
    """Sanitized immutable rejection; request and receipt bytes are absent."""

    rejection_id: str
    registration_operation_id: str
    request_id: str
    request_sha256: str
    error_code: str
    expected_sequence: int
    expected_previous_receipt_id: str | None
    observed_head_sequence: int
    observed_head_receipt_id: str | None
    schema_version: str = field(default=SYNTHETIC_REJECTION_SCHEMA_VERSION, init=False)
    countable_origin: bool = field(default=False, init=False)
    production_authorization: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        for label, value in (
            ("rejection id", self.rejection_id),
            ("request id", self.request_id),
            ("request SHA-256", self.request_sha256),
        ):
            _require_sha256(value, label=label)
        _canonical_uuid(self.registration_operation_id)
        _validate_sequence(self.expected_sequence, self.expected_previous_receipt_id)
        if type(self.observed_head_sequence) is not int or self.observed_head_sequence < 0:
            raise OriginRegistryConformanceError("observed HEAD sequence is invalid")
        if self.observed_head_sequence == 0:
            if self.observed_head_receipt_id is not None:
                raise OriginRegistryConformanceError("empty HEAD cannot name a receipt")
        else:
            _require_sha256(self.observed_head_receipt_id, label="observed HEAD receipt id")
        if not isinstance(self.error_code, str) or not re.fullmatch(
            r"[A-Z0-9_]{3,64}", self.error_code
        ):
            raise OriginRegistryConformanceError("rejection error code is invalid")

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "rejection_id": self.rejection_id,
            "registration_operation_id": self.registration_operation_id,
            "request_id": self.request_id,
            "request_sha256": self.request_sha256,
            "error_code": self.error_code,
            "expected_sequence": self.expected_sequence,
            "expected_previous_receipt_id": self.expected_previous_receipt_id,
            "observed_head_sequence": self.observed_head_sequence,
            "observed_head_receipt_id": self.observed_head_receipt_id,
            "countable_origin": self.countable_origin,
            "production_authorization": self.production_authorization,
        }


class SyntheticOriginRegistryConformanceHarness:
    """Thread-safe in-memory model of required registry state semantics."""

    SYNTHETIC_ONLY = True
    COUNTABLE_AUTHORITY = False

    def __init__(self) -> None:
        self._lock = Lock()
        self._commits: list[SyntheticCommittedOperation] = []
        self._committed_by_operation: dict[str, SyntheticCommittedOperation] = {}
        self._committed_by_request: dict[str, SyntheticCommittedOperation] = {}
        self._rejected_by_operation: dict[str, SyntheticRejectedOperation] = {}
        self._unique: dict[str, dict[str, SyntheticCommittedOperation]] = {
            "cadence_slot_id": {},
            "origin_id": {},
            "origin_as_of_utc": {},
        }

    def compare_and_append(self, candidate: SyntheticAppendCandidate) -> SyntheticAppendOutcome:
        if not isinstance(candidate, SyntheticAppendCandidate):
            raise TypeError("candidate must be a SyntheticAppendCandidate")
        with self._lock:
            committed = self._committed_by_operation.get(candidate.registration_operation_id)
            if committed is not None:
                if (
                    committed.request_payload == candidate.request_payload
                    and committed.receipt_payload == candidate.receipt_payload
                ):
                    return SyntheticAppendOutcome(committed=committed, idempotent_retry=True)
                raise OriginRegistryConformanceConflict("DIVERGENT_OPERATION_RETRY")

            rejected = self._rejected_by_operation.get(candidate.registration_operation_id)
            if rejected is not None:
                if rejected.request_sha256 == candidate.request_sha256:
                    raise OriginRegistryConformanceConflict(rejected.error_code, rejection=rejected)
                raise OriginRegistryConformanceConflict("DIVERGENT_REJECTED_OPERATION_RETRY")

            by_request = self._committed_by_request.get(candidate.request_id)
            if by_request is not None:
                self._reject(candidate, "REQUEST_ID_CONFLICT")

            for field_name, values in self._unique.items():
                if getattr(candidate, field_name) in values:
                    self._reject(candidate, f"{field_name.upper()}_CONFLICT")

            head = self._commits[-1] if self._commits else None
            head_sequence = 0 if head is None else head.synthetic_sequence
            head_receipt_id = None if head is None else head.candidate.receipt_id
            if (
                candidate.expected_sequence != head_sequence + 1
                or candidate.expected_previous_receipt_id != head_receipt_id
            ):
                self._reject(candidate, "STALE_HEAD_CONFLICT")

            committed = SyntheticCommittedOperation(
                candidate=candidate,
                synthetic_sequence=head_sequence + 1,
            )
            self._commits.append(committed)
            self._committed_by_operation[candidate.registration_operation_id] = committed
            self._committed_by_request[candidate.request_id] = committed
            for field_name, values in self._unique.items():
                values[getattr(candidate, field_name)] = committed
            return SyntheticAppendOutcome(committed=committed, idempotent_retry=False)

    def get_head(self) -> bytes | None:
        """Return the exact latest receipt bytes, or ``None`` at genesis."""

        with self._lock:
            return None if not self._commits else self._commits[-1].receipt_payload

    def lookup_operation(
        self, registration_operation_id: str, *, expected_request_payload: bytes
    ) -> SyntheticCommittedOperation | SyntheticRejectedOperation:
        operation_id = _canonical_uuid(registration_operation_id)
        request_sha256 = _exact_payload_sha256(
            expected_request_payload, label="operation lookup request"
        )
        with self._lock:
            committed = self._committed_by_operation.get(operation_id)
            if committed is not None:
                if committed.request_payload != expected_request_payload:
                    raise OriginRegistryConformanceConflict("OPERATION_REQUEST_MISMATCH")
                return committed
            rejected = self._rejected_by_operation.get(operation_id)
            if rejected is not None:
                if rejected.request_sha256 != request_sha256:
                    raise OriginRegistryConformanceConflict("OPERATION_REQUEST_MISMATCH")
                return rejected
        raise OriginRegistryConformanceError("UNKNOWN_OPERATION")

    @property
    def rejection_count(self) -> int:
        with self._lock:
            return len(self._rejected_by_operation)

    def _reject(self, candidate: SyntheticAppendCandidate, error_code: str) -> None:
        head = self._commits[-1] if self._commits else None
        observed_head_sequence = 0 if head is None else head.synthetic_sequence
        observed_head_receipt_id = None if head is None else head.candidate.receipt_id
        core: dict[str, object] = {
            "registration_operation_id": candidate.registration_operation_id,
            "request_id": candidate.request_id,
            "request_sha256": candidate.request_sha256,
            "error_code": error_code,
            "expected_sequence": candidate.expected_sequence,
            "expected_previous_receipt_id": candidate.expected_previous_receipt_id,
            "observed_head_sequence": observed_head_sequence,
            "observed_head_receipt_id": observed_head_receipt_id,
        }
        rejection = SyntheticRejectedOperation(
            rejection_id=_domain_hash(_REJECTION_ID_DOMAIN, core),
            registration_operation_id=candidate.registration_operation_id,
            request_id=candidate.request_id,
            request_sha256=candidate.request_sha256,
            error_code=error_code,
            expected_sequence=candidate.expected_sequence,
            expected_previous_receipt_id=candidate.expected_previous_receipt_id,
            observed_head_sequence=observed_head_sequence,
            observed_head_receipt_id=observed_head_receipt_id,
        )
        self._rejected_by_operation[candidate.registration_operation_id] = rejection
        raise OriginRegistryConformanceConflict(error_code, rejection=rejection)


def build_registry_trust_bundle_signature_payload(
    *,
    revision: int,
    previous_bundle_id: str | None,
    issued_at_utc: str,
    registry_keys: Sequence[RegistryPublicKeySpec],
) -> bytes:
    """Build exact domain-separated bytes for an external trust-root signer."""

    _validate_revision(revision, previous_bundle_id)
    _parse_utc(issued_at_utc, label="trust bundle issued-at")
    if isinstance(registry_keys, (str, bytes)) or not isinstance(registry_keys, Sequence):
        raise TypeError("registry_keys must be a sequence")
    if not registry_keys or any(
        not isinstance(key, RegistryPublicKeySpec) for key in registry_keys
    ):
        raise OriginRegistryConformanceError("registry key inventory is invalid")
    wire_keys = sorted(
        (key.to_wire() for key in registry_keys), key=lambda item: str(item["key_id"])
    )
    core: dict[str, object] = {
        "schema_version": TRUST_BUNDLE_SCHEMA_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "receipt_head_wire_contract_sha256": WIRE_CONTRACT_SHA256,
        "registry_domain_sha256": REGISTRY_DOMAIN_SHA256,
        "revision": revision,
        "previous_bundle_id": previous_bundle_id,
        "issued_at_utc": issued_at_utc,
        "registry_keys": wire_keys,
    }
    _validate_bundle_core(core)
    document = {**core, "bundle_id": _domain_hash(_BUNDLE_ID_DOMAIN, core)}
    return _signature_payload(_BUNDLE_SIGNATURE_DOMAIN, document)


def assemble_signed_registry_trust_bundle(
    *, signature_payload: bytes, signer_key_id: str, signature_base64: str
) -> bytes:
    """Attach caller-supplied trust-root signature bytes without a private key."""

    document = _signature_document(signature_payload, label="trust bundle signature payload")
    _validate_bundle_identity(document)
    _require_sha256(signer_key_id, label="trust root signer key id")
    _decode_signature(signature_base64, label="trust root")
    return canonical_json_bytes(
        {
            **document,
            "trust_root_signature": {
                "algorithm": SIGNATURE_ALGORITHM,
                "key_id": signer_key_id,
                "value_base64": signature_base64,
            },
        }
    )


def verify_registry_trust_bundle(
    bundle_payload: bytes, *, trust_root_public_key: Ed25519PublicKey
) -> VerifiedRegistryTrustBundle:
    """Verify one exact bundle under a caller-held, externally unadmitted root."""

    if not isinstance(trust_root_public_key, Ed25519PublicKey):
        raise TypeError("trust_root_public_key must be an Ed25519PublicKey")
    document = _strict_canonical_mapping(bundle_payload, label="signed registry trust bundle")
    _exact_fields(document, _BUNDLE_FIELDS, label="signed registry trust bundle")
    signing_document = dict(document)
    signature = signing_document.pop("trust_root_signature")
    _validate_bundle_identity(signing_document)
    signing_payload = _signature_payload(_BUNDLE_SIGNATURE_DOMAIN, signing_document)
    _verify_signature(
        signature,
        signing_payload,
        trust_root_public_key,
        label="trust root",
    )
    records = tuple(_key_record(value) for value in signing_document["registry_keys"])
    trust_root_id = public_key_id(trust_root_public_key)
    if trust_root_id in {record.key_id for record in records}:
        raise OriginRegistryConformanceError(
            "trust root and registry signer roles must be cryptographically disjoint"
        )
    return VerifiedRegistryTrustBundle(
        bundle_id=str(signing_document["bundle_id"]),
        bundle_sha256=hashlib.sha256(bundle_payload).hexdigest(),
        revision=signing_document["revision"],  # type: ignore[arg-type]
        previous_bundle_id=signing_document["previous_bundle_id"],  # type: ignore[arg-type]
        issued_at_utc=str(signing_document["issued_at_utc"]),
        trust_root_key_id=trust_root_id,
        registry_keys=records,
    )


def verify_registry_trust_bundle_chain(
    bundle_payloads: Sequence[bytes], *, trust_root_public_key: Ed25519PublicKey
) -> VerifiedRegistryTrustBundle:
    """Verify a complete append-only bundle chain and return its current bundle."""

    if isinstance(bundle_payloads, (str, bytes)) or not isinstance(bundle_payloads, Sequence):
        raise TypeError("bundle_payloads must be a sequence")
    if not bundle_payloads:
        raise OriginRegistryConformanceError("trust bundle chain is empty")
    bundles = tuple(
        verify_registry_trust_bundle(payload, trust_root_public_key=trust_root_public_key)
        for payload in bundle_payloads
    )
    previous: VerifiedRegistryTrustBundle | None = None
    for bundle in bundles:
        if previous is None:
            if bundle.revision != 1 or bundle.previous_bundle_id is not None:
                raise OriginRegistryConformanceError("trust bundle chain has no valid genesis")
        else:
            if (
                bundle.revision != previous.revision + 1
                or bundle.previous_bundle_id != previous.bundle_id
                or _parse_utc(bundle.issued_at_utc, label="bundle issued-at")
                <= _parse_utc(previous.issued_at_utc, label="previous bundle issued-at")
            ):
                raise OriginRegistryConformanceError("trust bundle chain linkage is invalid")
            _validate_key_lifecycle_transition(previous, bundle)
        previous = bundle
    return bundles[-1]


def prepare_synthetic_append_candidate(
    *,
    context: OriginRegistrationVerificationContext,
    receipt_payload: bytes,
    trust_bundle: VerifiedRegistryTrustBundle,
) -> SyntheticAppendCandidate:
    """Reverify trust, receipt and request chain before a synthetic transition."""

    if not isinstance(trust_bundle, VerifiedRegistryTrustBundle):
        raise TypeError("trust_bundle must be a VerifiedRegistryTrustBundle")
    receipt = _strict_canonical_mapping(receipt_payload, label="signed origin receipt")
    signature = receipt.get("registry_signature")
    if not isinstance(signature, Mapping):
        raise OriginRegistryConformanceError("receipt registry signature is invalid")
    key_id = signature.get("key_id")
    committed_at = receipt.get("committed_at_utc")
    if not isinstance(key_id, str) or not isinstance(committed_at, str):
        raise OriginRegistryConformanceError("receipt trust lookup fields are invalid")
    registry_public_key = trust_bundle.public_key_for_receipt(
        key_id=key_id, committed_at_utc=committed_at
    )
    verified_receipt = verify_signed_origin_registration_receipt(
        receipt_payload,
        registry_public_key=registry_public_key,
        context=context,
    )
    request = _strict_canonical_mapping(
        context.signed_request_payload, label="signed origin request"
    )
    return _candidate_from_verified(request, receipt_payload, verified_receipt)


def _candidate_from_verified(
    request: Mapping[str, object],
    receipt_payload: bytes,
    verified_receipt: VerifiedRegistrationReceipt,
) -> SyntheticAppendCandidate:
    return SyntheticAppendCandidate(
        request_payload=canonical_json_bytes(request),
        receipt_payload=receipt_payload,
        registration_operation_id=str(request["registration_operation_id"]),
        request_id=verified_receipt.request_id,
        cadence_slot_id=str(request["cadence_slot_id"]),
        origin_id=str(request["origin_id"]),
        origin_as_of_utc=str(request["origin_as_of_utc"]),
        expected_sequence=verified_receipt.sequence,
        expected_previous_receipt_id=verified_receipt.previous_receipt_id,
        receipt_id=verified_receipt.receipt_id,
    )


def _validate_bundle_core(core: Mapping[str, object]) -> None:
    _exact_fields(core, _BUNDLE_CORE_FIELDS, label="trust bundle core")
    if (
        core["schema_version"] != TRUST_BUNDLE_SCHEMA_VERSION
        or core["contract_sha256"] != CONTRACT_SHA256
        or core["receipt_head_wire_contract_sha256"] != WIRE_CONTRACT_SHA256
        or core["registry_domain_sha256"] != REGISTRY_DOMAIN_SHA256
    ):
        raise OriginRegistryConformanceError("trust bundle contract bindings are invalid")
    _validate_revision(core["revision"], core["previous_bundle_id"])
    keys = core["registry_keys"]
    if not isinstance(keys, list) or not keys:
        raise OriginRegistryConformanceError("registry key inventory is invalid")
    records = tuple(_key_record(value) for value in keys)
    _validate_record_inventory(records, issued_at_utc=str(core["issued_at_utc"]))


def _validate_record_inventory(
    records: tuple[RegistryPublicKeyRecord, ...], *, issued_at_utc: str
) -> None:
    key_ids = tuple(record.key_id for record in records)
    if key_ids != tuple(sorted(key_ids)) or len(set(key_ids)) != len(key_ids):
        raise OriginRegistryConformanceError("registry keys must be uniquely ordered")
    active = tuple(
        record for record in records if record.lifecycle_status is RegistryKeyLifecycle.ACTIVE
    )
    if len(active) != 1:
        raise OriginRegistryConformanceError("trust bundle requires exactly one active key")
    if not (
        _parse_utc(active[0].valid_from_utc, label="active key valid-from")
        <= _parse_utc(issued_at_utc, label="trust bundle issued-at")
        < _parse_utc(active[0].valid_until_utc, label="active key valid-until")
    ):
        raise OriginRegistryConformanceError("active registry key is invalid at bundle issuance")


def _validate_bundle_identity(document: Mapping[str, object]) -> None:
    _exact_fields(document, _BUNDLE_SIGNING_FIELDS, label="trust bundle signing document")
    core = dict(document)
    bundle_id = core.pop("bundle_id")
    _validate_bundle_core(core)
    if bundle_id != _domain_hash(_BUNDLE_ID_DOMAIN, core):
        raise OriginRegistryConformanceError("trust bundle id mismatch")


def _key_record(value: object) -> RegistryPublicKeyRecord:
    if not isinstance(value, Mapping):
        raise OriginRegistryConformanceError("registry key entry is invalid")
    _exact_fields(value, _KEY_FIELDS, label="registry key entry")
    if value["algorithm"] != SIGNATURE_ALGORITHM:
        raise OriginRegistryConformanceError("registry key algorithm is invalid")
    raw = _decode_public_key(value["public_key_base64"])
    if value["key_id"] != hashlib.sha256(raw).hexdigest():
        raise OriginRegistryConformanceError("registry public key identity is invalid")
    try:
        lifecycle = RegistryKeyLifecycle(str(value["lifecycle_status"]))
    except ValueError as exc:
        raise OriginRegistryConformanceError("registry key lifecycle is invalid") from exc
    return RegistryPublicKeyRecord(
        key_id=str(value["key_id"]),
        public_key_bytes=raw,
        valid_from_utc=str(value["valid_from_utc"]),
        valid_until_utc=str(value["valid_until_utc"]),
        lifecycle_status=lifecycle,
    )


def _validate_key_lifecycle_transition(
    previous: VerifiedRegistryTrustBundle,
    current: VerifiedRegistryTrustBundle,
) -> None:
    previous_by_id = {key.key_id: key for key in previous.registry_keys}
    current_by_id = {key.key_id: key for key in current.registry_keys}
    if not set(previous_by_id).issubset(current_by_id):
        raise OriginRegistryConformanceError("trust bundle key inventory is not append-only")
    for key_id, prior in previous_by_id.items():
        selected = current_by_id[key_id]
        if (
            prior.public_key_bytes != selected.public_key_bytes
            or prior.valid_from_utc != selected.valid_from_utc
            or prior.valid_until_utc != selected.valid_until_utc
        ):
            raise OriginRegistryConformanceError("registry key immutable fields changed")
        if (
            selected.lifecycle_status.value
            not in _ALLOWED_TRANSITIONS[prior.lifecycle_status.value]
        ):
            raise OriginRegistryConformanceError("registry key lifecycle regressed")


def _validate_revision(revision: object, previous_bundle_id: object) -> None:
    if type(revision) is not int or not 1 <= revision <= _MAX_SEQUENCE:
        raise OriginRegistryConformanceError("trust bundle revision is invalid")
    if revision == 1:
        if previous_bundle_id is not None:
            raise OriginRegistryConformanceError("genesis trust bundle cannot name a predecessor")
    else:
        _require_sha256(previous_bundle_id, label="previous trust bundle id")


def _validate_sequence(sequence: object, previous_receipt_id: object) -> None:
    if type(sequence) is not int or not 1 <= sequence <= _MAX_SEQUENCE:
        raise OriginRegistryConformanceError("registry sequence is invalid")
    if sequence == 1:
        if previous_receipt_id is not None:
            raise OriginRegistryConformanceError("genesis candidate cannot name a predecessor")
    else:
        _require_sha256(previous_receipt_id, label="previous receipt id")


def _signature_payload(domain: bytes, document: Mapping[str, object]) -> bytes:
    return domain + b"\x00" + canonical_json_bytes(document)


def _signature_document(payload: bytes, *, label: str) -> dict[str, object]:
    prefix = _BUNDLE_SIGNATURE_DOMAIN + b"\x00"
    if not isinstance(payload, bytes) or not payload.startswith(prefix):
        raise OriginRegistryConformanceError(f"{label} domain separation is invalid")
    document = _strict_canonical_mapping(payload[len(prefix) :], label=label)
    _exact_fields(document, _BUNDLE_SIGNING_FIELDS, label=label)
    return document


def _verify_signature(
    signature: object,
    payload: bytes,
    public_key: Ed25519PublicKey,
    *,
    label: str,
) -> None:
    if not isinstance(signature, Mapping):
        raise OriginRegistryConformanceError(f"{label} signature is invalid")
    _exact_fields(signature, _SIGNATURE_FIELDS, label=f"{label} signature")
    if signature["algorithm"] != SIGNATURE_ALGORITHM or signature["key_id"] != public_key_id(
        public_key
    ):
        raise OriginRegistryConformanceError(f"{label} signature trust binding is invalid")
    value = _decode_signature(signature["value_base64"], label=label)
    try:
        public_key.verify(value, payload)
    except InvalidSignature as exc:
        raise OriginRegistryConformanceError(f"{label} signature is invalid") from exc


def _strict_canonical_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_DOCUMENT_BYTES:
        raise OriginRegistryConformanceError(f"{label} byte envelope is invalid")
    try:
        parsed = json.loads(payload.decode("ascii"), object_pairs_hook=_reject_duplicate_keys)
    except OriginRegistryConformanceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise OriginRegistryConformanceError(f"{label} is not strict JSON") from exc
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        raise OriginRegistryConformanceError(f"{label} must be an object")
    try:
        canonical = canonical_json_bytes(parsed)
    except ValueError as exc:
        raise OriginRegistryConformanceError(f"{label} is not canonical JSON") from exc
    if canonical != payload:
        raise OriginRegistryConformanceError(f"{label} is not exact canonical JSON")
    return parsed


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise OriginRegistryConformanceError("strict JSON contains a duplicate key")
        document[key] = value
    return document


def _decode_public_key(value: object) -> bytes:
    if not isinstance(value, str):
        raise OriginRegistryConformanceError("registry public key encoding is invalid")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise OriginRegistryConformanceError("registry public key encoding is invalid") from exc
    if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != value:
        raise OriginRegistryConformanceError("registry public key encoding is invalid")
    return raw


def _decode_signature(value: object, *, label: str) -> bytes:
    if not isinstance(value, str):
        raise OriginRegistryConformanceError(f"{label} signature encoding is invalid")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise OriginRegistryConformanceError(f"{label} signature encoding is invalid") from exc
    if len(decoded) != 64 or base64.b64encode(decoded).decode("ascii") != value:
        raise OriginRegistryConformanceError(f"{label} signature encoding is invalid")
    return decoded


def _public_key_bytes(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes_raw()


def _exact_payload_sha256(payload: bytes, *, label: str) -> str:
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_DOCUMENT_BYTES:
        raise OriginRegistryConformanceError(f"{label} exact bytes are invalid")
    return hashlib.sha256(payload).hexdigest()


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise OriginRegistryConformanceError(f"{label} must be canonical UTC seconds")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise OriginRegistryConformanceError(f"{label} is invalid") from exc


def _canonical_uuid(value: object) -> str:
    if not isinstance(value, str):
        raise OriginRegistryConformanceError("registration operation id is invalid")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise OriginRegistryConformanceError("registration operation id is invalid") from exc
    if str(parsed) != value:
        raise OriginRegistryConformanceError("registration operation id must be canonical UUID")
    return value


def _domain_hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + b"\x00" + canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise OriginRegistryConformanceError(f"{label} must be a lowercase SHA-256")


def _exact_fields(
    value: Mapping[str, object], expected: set[str] | frozenset[str], *, label: str
) -> None:
    if set(value) != set(expected):
        raise OriginRegistryConformanceError(f"{label} fields are not exact")


def _negative_authority_manifest() -> dict[str, object]:
    return {
        "trust_root_externally_admitted": False,
        "external_cas_worm_verified": False,
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
    "CONTRACT_ID",
    "CONTRACT_SHA256",
    "SYNTHETIC_APPEND_STATUS",
    "SYNTHETIC_REJECTION_SCHEMA_VERSION",
    "TRUST_BUNDLE_SCHEMA_VERSION",
    "TRUST_BUNDLE_STATUS",
    "OriginRegistryConformanceConflict",
    "OriginRegistryConformanceError",
    "RegistryKeyLifecycle",
    "RegistryPublicKeyRecord",
    "RegistryPublicKeySpec",
    "SyntheticAppendCandidate",
    "SyntheticAppendOutcome",
    "SyntheticCommittedOperation",
    "SyntheticOriginRegistryConformanceHarness",
    "SyntheticRejectedOperation",
    "VerifiedRegistryTrustBundle",
    "assemble_signed_registry_trust_bundle",
    "build_registry_trust_bundle_signature_payload",
    "prepare_synthetic_append_candidate",
    "verify_registry_trust_bundle",
    "verify_registry_trust_bundle_chain",
]

"""Canonical, authority-negative CH LT origin registration requests.

This module translates one verified information-set envelope into the exact
request-v2 signing surface described by the frozen origin-registry protocol.
It accepts opaque trusted-time receipt bytes only to bind their hash; semantic
trusted-time admission remains external.  The module owns no private key,
performs no I/O, and cannot register or count an origin.

Receipt and HEAD wire verification is implemented separately against a local,
hash-frozen construction draft.  External trust-key admission, CAS/WORM
operation and governance approval remain outside this request builder.
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
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from pfc_shaping.lt.origin_registration_envelope import (
    ORIGIN_REGISTRY_PROTOCOL_ID,
    ORIGIN_REGISTRY_PROTOCOL_SHA256,
    SIGNATURE_ALGORITHM,
    canonical_json_bytes,
    public_key_id,
    verify_origin_information_set_envelope,
    verify_signed_schedule_manifest,
)

REQUEST_SCHEMA_VERSION = "ch_lt_origin_registration_request.v2"
REQUEST_STATUS = "CRYPTOGRAPHICALLY_VERIFIED_LOCAL_PREPARATION_ONLY_NO_GO"
RECEIPT_STATUS = "LOCAL_RECEIPT_HEAD_WIRE_VERIFIER_IMPLEMENTED_EXTERNAL_ADMISSION_MISSING_NO_GO"
REGISTRY_LOGICAL_DOMAIN = "FMV_CH_LT_CONFIRMATORY_ORIGINS_V2"
REGISTRY_DOMAIN_SHA256 = hashlib.sha256(REGISTRY_LOGICAL_DOMAIN.encode("ascii")).hexdigest()
CADENCE_CONTRACT_SHA256 = "037619f50cd882a7c65227876c95af3d2433f6536c118ce1e43e86fe359779d6"

_REQUEST_ID_DOMAIN = b"FMV_CH_LT_ORIGIN_REGISTRATION_REQUEST_V2"
_ORIGIN_ID_DOMAIN = b"FMV_CH_LT_ORIGIN_ID_V2"
_MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
_MAX_SEQUENCE = 10_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CADENCE_SLOT = re.compile(r"^[0-9]{4}-(0[1-9]|1[0-2])$")
_UTC = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")

_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "request_id",
        "request_signature",
        "protocol_sha256",
        "protocol_id",
        "registry_domain_sha256",
        "registration_operation_id",
        "expected_sequence",
        "expected_previous_receipt_id",
        "cadence_slot_id",
        "cadence_contract_sha256",
        "exact_schedule_manifest_sha256",
        "exact_schedule_entry_id",
        "exact_schedule_entry_sha256",
        "capture_window_open_utc",
        "capture_window_close_utc",
        "latest_external_registry_commit_utc",
        "origin_as_of_utc",
        "origin_id",
        "origin_target_mask_inventory_sha256",
        "origin_target_mask_inventory_id",
        "origin_available_eex_product_inventory_sha256",
        "eex_vintage_manifest_sha256",
        "monthly_solver_configuration_sha256",
        "candidate_baseline_identity_manifest_sha256",
        "candidate_hypothesis_and_procedure_manifest_sha256",
        "prediction_commitment_sha256",
        "scenario_commitment_sha256",
        "structural_target_universe_commitment_sha256",
        "ex_ante_evaluation_mask_rule_commitment_sha256",
        "calendar_and_strata_manifest_sha256",
        "runtime_receipt_sha256",
        "project_wheel_sha256",
        "project_source_revision",
        "trusted_origin_time_utc",
        "trusted_origin_time_receipt_sha256",
        "first_target_delivery_start_utc",
        "truth_opened",
        "future_holdout_consumed",
        "t057_consumed",
    }
)
_REQUEST_SIGNING_FIELDS = _REQUEST_FIELDS - {"request_signature"}
_REQUEST_CORE_FIELDS = _REQUEST_SIGNING_FIELDS - {"request_id"}
_COMMITMENT_FIELDS = frozenset(
    {
        "origin_target_mask_inventory_sha256",
        "origin_target_mask_inventory_id",
        "origin_available_eex_product_inventory_sha256",
        "eex_vintage_manifest_sha256",
        "monthly_solver_configuration_sha256",
        "candidate_baseline_identity_manifest_sha256",
        "candidate_hypothesis_and_procedure_manifest_sha256",
        "prediction_commitment_sha256",
        "scenario_commitment_sha256",
        "structural_target_universe_commitment_sha256",
        "ex_ante_evaluation_mask_rule_commitment_sha256",
        "calendar_and_strata_manifest_sha256",
        "runtime_receipt_sha256",
        "project_wheel_sha256",
        "project_source_revision",
    }
)
_MISSING_EXTERNAL_EVIDENCE = (
    "TRUSTED_TIME_RECEIPT_SEMANTIC_ADMISSION",
    "REQUEST_SIGNER_EXTERNAL_ROLE_ADMISSION",
    "EXTERNAL_COMPARE_AND_APPEND_RECEIPT",
    "FRESH_EXTERNAL_REGISTRY_HEAD_OBSERVATION",
)
_RECEIPT_CONTRACT_GAPS = (
    "FMV_EXTERNAL_WIRE_CONTRACT_APPROVAL",
    "EXTERNAL_REGISTRY_KEY_IDENTITY_AND_TRUST_REGISTRY_ADMISSION",
    "BUILDER_INACCESSIBLE_REMOTE_COMPARE_AND_APPEND_WORM",
    "INDEPENDENT_SERVICE_CONFORMANCE_AND_SECURITY_EVIDENCE",
)


class OriginRegistrationRequestError(ValueError):
    """Raised when a registration request fails closed."""


@dataclass(frozen=True, slots=True)
class RegistrationHeadExpectation:
    """Caller-held compare-and-append expectation for one request."""

    registration_operation_id: str
    expected_sequence: int
    expected_previous_receipt_id: str | None

    def __post_init__(self) -> None:
        _canonical_uuid(self.registration_operation_id)
        if (
            type(self.expected_sequence) is not int
            or not 1 <= self.expected_sequence <= _MAX_SEQUENCE
        ):
            raise OriginRegistrationRequestError("expected sequence is invalid")
        if self.expected_sequence == 1:
            if self.expected_previous_receipt_id is not None:
                raise OriginRegistrationRequestError(
                    "genesis request cannot name a previous receipt"
                )
        else:
            _require_sha256(
                self.expected_previous_receipt_id,
                label="expected previous receipt id",
            )


@dataclass(frozen=True, slots=True)
class VerifiedRegistrationRequest:
    """Cryptographic verification result with non-overridable local authority."""

    request_id: str
    request_sha256: str
    request_signer_key_id: str
    registration_operation_id: str
    expected_sequence: int
    expected_previous_receipt_id: str | None
    cryptographic_request_signature_verified: bool = field(default=True, init=False)
    status: str = field(default=REQUEST_STATUS, init=False)
    externally_registered: bool = field(default=False, init=False)
    countable_origin: bool = field(default=False, init=False)
    trusted_time_semantically_admitted: bool = field(default=False, init=False)
    receipt_verified: bool = field(default=False, init=False)
    fresh_head_observation_verified: bool = field(default=False, init=False)
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
        _require_sha256(self.request_id, label="request id")
        _require_sha256(self.request_sha256, label="request SHA-256")
        _require_sha256(self.request_signer_key_id, label="request signer key id")
        RegistrationHeadExpectation(
            registration_operation_id=self.registration_operation_id,
            expected_sequence=self.expected_sequence,
            expected_previous_receipt_id=self.expected_previous_receipt_id,
        )

    def to_manifest(self) -> dict[str, object]:
        return {
            "status": self.status,
            "request_id": self.request_id,
            "request_sha256": self.request_sha256,
            "request_signer_key_id": self.request_signer_key_id,
            "registration_operation_id": self.registration_operation_id,
            "expected_sequence": self.expected_sequence,
            "expected_previous_receipt_id": self.expected_previous_receipt_id,
            "cryptographic_request_signature_verified": (
                self.cryptographic_request_signature_verified
            ),
            "externally_registered": self.externally_registered,
            "countable_origin": self.countable_origin,
            "trusted_time_semantically_admitted": (self.trusted_time_semantically_admitted),
            "receipt_verified": self.receipt_verified,
            "fresh_head_observation_verified": (self.fresh_head_observation_verified),
            "truth_open_authorized": self.truth_open_authorized,
            "model_training_authorized": self.model_training_authorized,
            "model_selection_authorized": self.model_selection_authorized,
            "scientific_admission": self.scientific_admission,
            "production_authorization": self.production_authorization,
            "promotion_gate": self.promotion_gate,
            "missing_external_evidence": list(self.missing_external_evidence),
        }


@dataclass(frozen=True, slots=True)
class ReceiptContractReadiness:
    """Local implementation readiness without external registry authority."""

    status: str = field(default=RECEIPT_STATUS, init=False)
    verification_implemented: bool = field(default=True, init=False)
    local_wire_contract_hash_frozen: bool = field(default=True, init=False)
    external_trust_admitted: bool = field(default=False, init=False)
    countable_origin: bool = field(default=False, init=False)
    scientific_admission: bool = field(default=False, init=False)
    production_authorization: bool = field(default=False, init=False)
    promotion_gate: bool = field(default=False, init=False)
    missing_contract_clauses: tuple[str, ...] = field(default=_RECEIPT_CONTRACT_GAPS, init=False)

    def to_manifest(self) -> dict[str, object]:
        return {
            "status": self.status,
            "verification_implemented": self.verification_implemented,
            "local_wire_contract_hash_frozen": self.local_wire_contract_hash_frozen,
            "external_trust_admitted": self.external_trust_admitted,
            "countable_origin": self.countable_origin,
            "scientific_admission": self.scientific_admission,
            "production_authorization": self.production_authorization,
            "promotion_gate": self.promotion_gate,
            "missing_contract_clauses": list(self.missing_contract_clauses),
        }


def build_origin_registration_signature_payload(
    *,
    information_set_envelope_payload: bytes,
    schedule_manifest_payload: bytes,
    trusted_schedule_public_key: Ed25519PublicKey,
    trusted_time_receipt_payload: bytes,
    head: RegistrationHeadExpectation,
) -> bytes:
    """Build exact request-v2 bytes for an independent request signer."""

    if not isinstance(head, RegistrationHeadExpectation):
        raise TypeError("head must be a RegistrationHeadExpectation")
    envelope = verify_origin_information_set_envelope(
        information_set_envelope_payload,
        schedule_manifest_payload=schedule_manifest_payload,
        trusted_schedule_public_key=trusted_schedule_public_key,
    )
    schedule = verify_signed_schedule_manifest(
        schedule_manifest_payload,
        trusted_public_key=trusted_schedule_public_key,
    )
    entry = schedule.entry_for(str(envelope["cadence_slot_id"]))
    trusted_time_sha256 = _exact_payload_sha256(
        trusted_time_receipt_payload,
        label="trusted time receipt",
    )
    commitments = envelope["information_set_commitments"]
    if not isinstance(commitments, Mapping):
        raise OriginRegistrationRequestError("information-set commitments are invalid")
    _exact_fields(commitments, _COMMITMENT_FIELDS, label="information-set commitments")
    origin_as_of = str(envelope["origin_as_of_utc"])
    origin_id = _domain_hash(
        _ORIGIN_ID_DOMAIN,
        {
            "origin_as_of_utc": origin_as_of,
            "origin_target_mask_inventory_id": commitments["origin_target_mask_inventory_id"],
            "origin_target_mask_inventory_sha256": commitments[
                "origin_target_mask_inventory_sha256"
            ],
        },
    )
    core: dict[str, object] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "protocol_sha256": ORIGIN_REGISTRY_PROTOCOL_SHA256,
        "protocol_id": ORIGIN_REGISTRY_PROTOCOL_ID,
        "registry_domain_sha256": REGISTRY_DOMAIN_SHA256,
        "registration_operation_id": head.registration_operation_id,
        "expected_sequence": head.expected_sequence,
        "expected_previous_receipt_id": head.expected_previous_receipt_id,
        "cadence_slot_id": envelope["cadence_slot_id"],
        "cadence_contract_sha256": CADENCE_CONTRACT_SHA256,
        "exact_schedule_manifest_sha256": envelope["schedule_manifest_sha256"],
        "exact_schedule_entry_id": envelope["schedule_entry_id"],
        "exact_schedule_entry_sha256": envelope["schedule_entry_sha256"],
        "capture_window_open_utc": entry["capture_window_open_utc"],
        "capture_window_close_utc": entry["capture_window_close_utc"],
        "latest_external_registry_commit_utc": entry["latest_external_registry_commit_utc"],
        "origin_as_of_utc": origin_as_of,
        "origin_id": origin_id,
        "origin_target_mask_inventory_sha256": commitments["origin_target_mask_inventory_sha256"],
        "origin_target_mask_inventory_id": commitments["origin_target_mask_inventory_id"],
        "origin_available_eex_product_inventory_sha256": commitments[
            "origin_available_eex_product_inventory_sha256"
        ],
        "eex_vintage_manifest_sha256": commitments["eex_vintage_manifest_sha256"],
        "monthly_solver_configuration_sha256": commitments["monthly_solver_configuration_sha256"],
        "candidate_baseline_identity_manifest_sha256": commitments[
            "candidate_baseline_identity_manifest_sha256"
        ],
        "candidate_hypothesis_and_procedure_manifest_sha256": commitments[
            "candidate_hypothesis_and_procedure_manifest_sha256"
        ],
        "prediction_commitment_sha256": commitments["prediction_commitment_sha256"],
        "scenario_commitment_sha256": commitments["scenario_commitment_sha256"],
        "structural_target_universe_commitment_sha256": commitments[
            "structural_target_universe_commitment_sha256"
        ],
        "ex_ante_evaluation_mask_rule_commitment_sha256": commitments[
            "ex_ante_evaluation_mask_rule_commitment_sha256"
        ],
        "calendar_and_strata_manifest_sha256": commitments["calendar_and_strata_manifest_sha256"],
        "runtime_receipt_sha256": commitments["runtime_receipt_sha256"],
        "project_wheel_sha256": commitments["project_wheel_sha256"],
        "project_source_revision": commitments["project_source_revision"],
        "trusted_origin_time_utc": origin_as_of,
        "trusted_origin_time_receipt_sha256": trusted_time_sha256,
        "first_target_delivery_start_utc": envelope["first_target_delivery_start_utc"],
        "truth_opened": False,
        "future_holdout_consumed": False,
        "t057_consumed": False,
    }
    _exact_fields(core, _REQUEST_CORE_FIELDS, label="origin registration request core")
    request_id = _domain_hash(_REQUEST_ID_DOMAIN, core)
    return canonical_json_bytes({**core, "request_id": request_id})


def assemble_signed_origin_registration_request(
    *,
    signature_payload: bytes,
    signer_key_id: str,
    signature_base64: str,
) -> bytes:
    """Attach caller-supplied request-signature bytes without a private key."""

    document = _strict_canonical_mapping(signature_payload, label="request signature payload")
    _exact_fields(document, _REQUEST_SIGNING_FIELDS, label="request signature payload")
    _validate_request_identity(document)
    _require_sha256(signer_key_id, label="request signer key id")
    _decode_signature(signature_base64)
    return canonical_json_bytes(
        {
            **document,
            "request_signature": {
                "algorithm": SIGNATURE_ALGORITHM,
                "key_id": signer_key_id,
                "value_base64": signature_base64,
            },
        }
    )


def verify_signed_origin_registration_request(
    request_payload: bytes,
    *,
    trusted_request_public_key: Ed25519PublicKey,
    information_set_envelope_payload: bytes,
    schedule_manifest_payload: bytes,
    trusted_schedule_public_key: Ed25519PublicKey,
    trusted_time_receipt_payload: bytes,
) -> VerifiedRegistrationRequest:
    """Verify exact request bytes and all locally closed cross-bindings."""

    if not isinstance(trusted_request_public_key, Ed25519PublicKey):
        raise TypeError("trusted_request_public_key must be an Ed25519PublicKey")
    if public_key_id(trusted_request_public_key) == public_key_id(trusted_schedule_public_key):
        raise OriginRegistrationRequestError(
            "request and schedule signer roles must be cryptographically disjoint"
        )
    request = _strict_canonical_mapping(request_payload, label="signed origin request")
    _exact_fields(request, _REQUEST_FIELDS, label="signed origin request")
    signature = request["request_signature"]
    signing_document = dict(request)
    signing_document.pop("request_signature")
    _validate_request_identity(signing_document)
    _verify_signature(
        signature,
        canonical_json_bytes(signing_document),
        trusted_public_key=trusted_request_public_key,
    )
    head = RegistrationHeadExpectation(
        registration_operation_id=str(request["registration_operation_id"]),
        expected_sequence=request["expected_sequence"],  # type: ignore[arg-type]
        expected_previous_receipt_id=request["expected_previous_receipt_id"],  # type: ignore[arg-type]
    )
    expected = build_origin_registration_signature_payload(
        information_set_envelope_payload=information_set_envelope_payload,
        schedule_manifest_payload=schedule_manifest_payload,
        trusted_schedule_public_key=trusted_schedule_public_key,
        trusted_time_receipt_payload=trusted_time_receipt_payload,
        head=head,
    )
    if signing_document != _strict_canonical_mapping(
        expected, label="expected request signature payload"
    ):
        raise OriginRegistrationRequestError(
            "signed request does not match its information-set inputs"
        )
    return VerifiedRegistrationRequest(
        request_id=str(request["request_id"]),
        request_sha256=hashlib.sha256(request_payload).hexdigest(),
        request_signer_key_id=public_key_id(trusted_request_public_key),
        registration_operation_id=head.registration_operation_id,
        expected_sequence=head.expected_sequence,
        expected_previous_receipt_id=head.expected_previous_receipt_id,
    )


def receipt_contract_readiness() -> ReceiptContractReadiness:
    """Return the frozen fail-closed status of external receipt verification."""

    return ReceiptContractReadiness()


def _validate_request_identity(document: Mapping[str, object]) -> None:
    _exact_fields(document, _REQUEST_SIGNING_FIELDS, label="request signing document")
    core = dict(document)
    request_id = core.pop("request_id")
    _exact_fields(core, _REQUEST_CORE_FIELDS, label="request core")
    if request_id != _domain_hash(_REQUEST_ID_DOMAIN, core):
        raise OriginRegistrationRequestError("origin registration request id mismatch")
    if (
        document["schema_version"] != REQUEST_SCHEMA_VERSION
        or document["protocol_sha256"] != ORIGIN_REGISTRY_PROTOCOL_SHA256
        or document["protocol_id"] != ORIGIN_REGISTRY_PROTOCOL_ID
        or document["registry_domain_sha256"] != REGISTRY_DOMAIN_SHA256
        or document["cadence_contract_sha256"] != CADENCE_CONTRACT_SHA256
        or document["trusted_origin_time_utc"] != document["origin_as_of_utc"]
        or document["truth_opened"] is not False
        or document["future_holdout_consumed"] is not False
        or document["t057_consumed"] is not False
    ):
        raise OriginRegistrationRequestError("origin registration request semantics are invalid")
    RegistrationHeadExpectation(
        registration_operation_id=str(document["registration_operation_id"]),
        expected_sequence=document["expected_sequence"],  # type: ignore[arg-type]
        expected_previous_receipt_id=document["expected_previous_receipt_id"],  # type: ignore[arg-type]
    )
    cadence_slot = document["cadence_slot_id"]
    if not isinstance(cadence_slot, str) or _CADENCE_SLOT.fullmatch(cadence_slot) is None:
        raise OriginRegistrationRequestError("request cadence slot is invalid")
    capture_open = _parse_utc(document["capture_window_open_utc"], label="capture open")
    capture_close = _parse_utc(document["capture_window_close_utc"], label="capture close")
    origin = _parse_utc(document["origin_as_of_utc"], label="origin as-of")
    trusted_origin = _parse_utc(document["trusted_origin_time_utc"], label="trusted origin time")
    deadline = _parse_utc(
        document["latest_external_registry_commit_utc"],
        label="external registry deadline",
    )
    first_target = _parse_utc(
        document["first_target_delivery_start_utc"],
        label="first target delivery start",
    )
    if not capture_open <= origin == trusted_origin <= capture_close <= deadline < first_target:
        raise OriginRegistrationRequestError("origin registration chronology is invalid")
    expected_origin_id = _domain_hash(
        _ORIGIN_ID_DOMAIN,
        {
            "origin_as_of_utc": document["origin_as_of_utc"],
            "origin_target_mask_inventory_id": document["origin_target_mask_inventory_id"],
            "origin_target_mask_inventory_sha256": document["origin_target_mask_inventory_sha256"],
        },
    )
    if document["origin_id"] != expected_origin_id:
        raise OriginRegistrationRequestError("origin id binding is invalid")
    for name, value in document.items():
        if name.endswith("_sha256") or name in {
            "request_id",
            "origin_id",
            "origin_target_mask_inventory_id",
            "exact_schedule_entry_id",
            "project_source_revision",
        }:
            _require_sha256(value, label=name)


def _verify_signature(
    signature: object,
    payload: bytes,
    *,
    trusted_public_key: Ed25519PublicKey,
) -> None:
    if not isinstance(signature, Mapping):
        raise OriginRegistrationRequestError("request signature is invalid")
    _exact_fields(
        signature,
        {"algorithm", "key_id", "value_base64"},
        label="request signature",
    )
    if signature["algorithm"] != SIGNATURE_ALGORITHM or signature["key_id"] != public_key_id(
        trusted_public_key
    ):
        raise OriginRegistrationRequestError("request signature trust binding is invalid")
    value = _decode_signature(signature["value_base64"])
    try:
        trusted_public_key.verify(value, payload)
    except InvalidSignature as exc:
        raise OriginRegistrationRequestError("request signature is invalid") from exc


def _strict_canonical_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_DOCUMENT_BYTES:
        raise OriginRegistrationRequestError(f"{label} byte envelope is invalid")
    try:
        parsed = json.loads(payload.decode("ascii"), object_pairs_hook=_reject_duplicate_keys)
    except OriginRegistrationRequestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise OriginRegistrationRequestError(f"{label} is not strict JSON") from exc
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        raise OriginRegistrationRequestError(f"{label} must be an object")
    try:
        canonical = canonical_json_bytes(parsed)
    except ValueError as exc:
        raise OriginRegistrationRequestError(f"{label} is not canonical JSON") from exc
    if canonical != payload:
        raise OriginRegistrationRequestError(f"{label} is not exact canonical JSON")
    return parsed


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise OriginRegistrationRequestError("strict JSON contains a duplicate key")
        document[key] = value
    return document


def _decode_signature(value: object) -> bytes:
    if not isinstance(value, str):
        raise OriginRegistrationRequestError("request signature encoding is invalid")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise OriginRegistrationRequestError("request signature encoding is invalid") from exc
    if len(decoded) != 64 or base64.b64encode(decoded).decode("ascii") != value:
        raise OriginRegistrationRequestError("request signature encoding is invalid")
    return decoded


def _exact_payload_sha256(payload: bytes, *, label: str) -> str:
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_DOCUMENT_BYTES:
        raise OriginRegistrationRequestError(f"{label} exact bytes are invalid")
    return hashlib.sha256(payload).hexdigest()


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise OriginRegistrationRequestError(f"{label} must be canonical UTC seconds")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise OriginRegistrationRequestError(f"{label} is invalid") from exc


def _domain_hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + b"\x00" + canonical_json_bytes(value)).hexdigest()


def _canonical_uuid(value: str) -> str:
    if not isinstance(value, str):
        raise OriginRegistrationRequestError("registration operation id is invalid")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise OriginRegistrationRequestError("registration operation id is invalid") from exc
    if str(parsed) != value:
        raise OriginRegistrationRequestError("registration operation id must be canonical UUID")
    return value


def _exact_fields(
    value: Mapping[str, object], expected: set[str] | frozenset[str], *, label: str
) -> None:
    if set(value) != set(expected):
        raise OriginRegistrationRequestError(f"{label} fields are not exact")


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise OriginRegistrationRequestError(f"{label} must be a lowercase SHA-256")


__all__ = [
    "CADENCE_CONTRACT_SHA256",
    "RECEIPT_STATUS",
    "REGISTRY_DOMAIN_SHA256",
    "REGISTRY_LOGICAL_DOMAIN",
    "REQUEST_SCHEMA_VERSION",
    "REQUEST_STATUS",
    "OriginRegistrationRequestError",
    "ReceiptContractReadiness",
    "RegistrationHeadExpectation",
    "VerifiedRegistrationRequest",
    "assemble_signed_origin_registration_request",
    "build_origin_registration_signature_payload",
    "receipt_contract_readiness",
    "verify_signed_origin_registration_request",
]

"""Outcome-blind preparation boundary for LT origin registration.

The module prepares canonical bytes for independent schedule signing and
verifies a signed pre-registration schedule before binding an origin
information set.  It deliberately has no private-key, registry, truth, model
training, filesystem, or network capability.  A valid schedule signature is
necessary evidence, never sufficient registration authority.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from pfc_shaping.lt.evaluation_protocol import (
    CANONICAL_PROTOCOL_SEMANTIC_SHA256,
    EvaluationProtocol,
    OriginSlot,
    default_evaluation_protocol,
)
from pfc_shaping.validation.ch_lt_origin_registry_protocol import (
    PROTOCOL_ID as ORIGIN_REGISTRY_PROTOCOL_ID,
)
from pfc_shaping.validation.ch_lt_origin_registry_protocol import (
    PROTOCOL_SHA256 as ORIGIN_REGISTRY_PROTOCOL_SHA256,
)

SCHEDULE_MANIFEST_SCHEMA_VERSION = "ch_lt_origin_exact_schedule_manifest.pre_registration.v1"
INFORMATION_SET_SCHEMA_VERSION = "fmv_lt_origin_information_set_envelope.v1"
STATUS = "LOCAL_PREPARATION_ONLY_NOT_EXTERNALLY_REGISTERED_NO_GO"
SIGNATURE_ALGORITHM = "ED25519"

_SCHEDULE_ENTRY_ID_DOMAIN = b"FMV_CH_LT_ORIGIN_SCHEDULE_ENTRY_V2"
_INFORMATION_SET_ID_DOMAIN = b"FMV_LT_ORIGIN_INFORMATION_SET_ENVELOPE_V1"
_MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CADENCE_SLOT = re.compile(r"^[0-9]{4}-(0[1-9]|1[0-2])$")
_UTC = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_LOCAL_TIMEZONE = ZoneInfo("Europe/Zurich")

_SCHEDULE_CORE_FIELDS = frozenset(
    {
        "protocol_sha256",
        "cadence_slot_id",
        "eex_trading_day",
        "capture_window_open_utc",
        "capture_window_close_utc",
        "latest_external_registry_commit_utc",
        "official_eex_calendar_document_sha256",
        "official_settlement_event_definition_sha256",
    }
)
_SCHEDULE_SIGNING_FIELDS = _SCHEDULE_CORE_FIELDS | {"schedule_entry_id"}
_SCHEDULE_ENTRY_FIELDS = _SCHEDULE_SIGNING_FIELDS | {"signature"}
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
_AUTHORITY = {
    "status": STATUS,
    "external_registry_receipt_verified": False,
    "externally_registered": False,
    "countable_origin": False,
    "truth_open_authorized": False,
    "model_training_authorized": False,
    "model_selection_authorized": False,
    "scientific_admission": False,
    "production_authorization": False,
    "promotion_gate": False,
}


class OriginRegistrationEnvelopeError(ValueError):
    """Raised when canonical, signature, chronology, or authority checks fail."""


@dataclass(frozen=True, slots=True)
class ScheduleEntryCore:
    """Externally supplied facts required before one schedule entry is signed."""

    cadence_slot_id: str
    eex_trading_day: str
    capture_window_open_utc: str
    capture_window_close_utc: str
    latest_external_registry_commit_utc: str
    official_eex_calendar_document_sha256: str
    official_settlement_event_definition_sha256: str

    def __post_init__(self) -> None:
        _validate_schedule_core(self.to_manifest())

    def to_manifest(self) -> dict[str, str]:
        return {
            "protocol_sha256": ORIGIN_REGISTRY_PROTOCOL_SHA256,
            "cadence_slot_id": self.cadence_slot_id,
            "eex_trading_day": self.eex_trading_day,
            "capture_window_open_utc": self.capture_window_open_utc,
            "capture_window_close_utc": self.capture_window_close_utc,
            "latest_external_registry_commit_utc": (self.latest_external_registry_commit_utc),
            "official_eex_calendar_document_sha256": (self.official_eex_calendar_document_sha256),
            "official_settlement_event_definition_sha256": (
                self.official_settlement_event_definition_sha256
            ),
        }


@dataclass(frozen=True, slots=True)
class VerifiedSchedule:
    """Immutable verified schedule value; it carries no registry authority."""

    payload_sha256: str
    entry_payloads: tuple[bytes, ...]
    signer_key_id: str

    def entry_for(self, cadence_slot_id: str) -> Mapping[str, object]:
        selected = tuple(
            entry
            for entry in (
                _strict_canonical_mapping(payload, label="verified schedule entry")
                for payload in self.entry_payloads
            )
            if entry["cadence_slot_id"] == cadence_slot_id
        )
        if len(selected) != 1:
            raise OriginRegistrationEnvelopeError(
                "cadence slot does not select exactly one signed schedule entry"
            )
        return selected[0]


def canonical_json_bytes(value: object) -> bytes:
    """Return compact ASCII JSON after rejecting floats and non-string keys."""

    _validate_no_float(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise OriginRegistrationEnvelopeError("canonical JSON value is invalid") from exc


def public_key_id(public_key: Ed25519PublicKey) -> str:
    """Return the origin-registry v2 reference key identity."""

    if not isinstance(public_key, Ed25519PublicKey):
        raise TypeError("public_key must be an Ed25519PublicKey")
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def build_schedule_entry_signature_payload(core: ScheduleEntryCore) -> bytes:
    """Build exact bytes for an independent authority to sign."""

    if not isinstance(core, ScheduleEntryCore):
        raise TypeError("core must be a ScheduleEntryCore")
    manifest = core.to_manifest()
    entry_id = _domain_hash(_SCHEDULE_ENTRY_ID_DOMAIN, manifest)
    return canonical_json_bytes({**manifest, "schedule_entry_id": entry_id})


def assemble_signed_schedule_entry(
    *, signature_payload: bytes, signer_key_id: str, signature_base64: str
) -> bytes:
    """Attach externally produced signature bytes without possessing a private key."""

    document = _strict_canonical_mapping(signature_payload, label="schedule payload")
    _exact_fields(document, _SCHEDULE_SIGNING_FIELDS, label="schedule payload")
    _validate_schedule_signing_document(document)
    _require_sha256(signer_key_id, label="schedule signer key id")
    _decode_signature(signature_base64)
    return canonical_json_bytes(
        {
            **document,
            "signature": {
                "algorithm": SIGNATURE_ALGORITHM,
                "key_id": signer_key_id,
                "value_base64": signature_base64,
            },
        }
    )


def build_schedule_manifest_payload(signed_entries: Sequence[bytes]) -> bytes:
    """Build a deterministic manifest from individually signed entry documents."""

    if isinstance(signed_entries, (str, bytes, bytearray)) or not signed_entries:
        raise OriginRegistrationEnvelopeError("signed schedule entries are required")
    if any(not isinstance(payload, bytes) for payload in signed_entries):
        raise OriginRegistrationEnvelopeError("each signed schedule entry must be exact bytes")
    entries = [
        _strict_canonical_mapping(payload, label="signed schedule entry")
        for payload in signed_entries
    ]
    return canonical_json_bytes(
        {
            "schema_version": SCHEDULE_MANIFEST_SCHEMA_VERSION,
            "evaluation_protocol_sha256": CANONICAL_PROTOCOL_SEMANTIC_SHA256,
            "origin_registry_protocol_sha256": ORIGIN_REGISTRY_PROTOCOL_SHA256,
            "origin_registry_protocol_id": ORIGIN_REGISTRY_PROTOCOL_ID,
            "entries": entries,
        }
    )


def verify_signed_schedule_manifest(
    payload: bytes, *, trusted_public_key: Ed25519PublicKey
) -> VerifiedSchedule:
    """Verify exact canonical bytes, every entry signature, and cohort alignment."""

    if not isinstance(trusted_public_key, Ed25519PublicKey):
        raise TypeError("trusted_public_key must be an Ed25519PublicKey")
    document = _strict_canonical_mapping(payload, label="signed schedule manifest")
    _exact_fields(
        document,
        {
            "schema_version",
            "evaluation_protocol_sha256",
            "origin_registry_protocol_sha256",
            "origin_registry_protocol_id",
            "entries",
        },
        label="signed schedule manifest",
    )
    if (
        document["schema_version"] != SCHEDULE_MANIFEST_SCHEMA_VERSION
        or document["evaluation_protocol_sha256"] != CANONICAL_PROTOCOL_SEMANTIC_SHA256
        or document["origin_registry_protocol_sha256"] != ORIGIN_REGISTRY_PROTOCOL_SHA256
        or document["origin_registry_protocol_id"] != ORIGIN_REGISTRY_PROTOCOL_ID
    ):
        raise OriginRegistrationEnvelopeError("schedule manifest identity is invalid")
    raw_entries = document["entries"]
    if not isinstance(raw_entries, list):
        raise OriginRegistrationEnvelopeError("schedule entries must be an array")
    verified = tuple(
        _verify_signed_schedule_entry(entry, trusted_public_key=trusted_public_key)
        for entry in raw_entries
    )
    _verify_cohort_alignment(verified, default_evaluation_protocol())
    return VerifiedSchedule(
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        entry_payloads=tuple(canonical_json_bytes(entry) for entry in verified),
        signer_key_id=public_key_id(trusted_public_key),
    )


def prepare_origin_information_set_envelope(
    *,
    schedule_manifest_payload: bytes,
    trusted_schedule_public_key: Ed25519PublicKey,
    slot_id: str,
    first_target_delivery_start_utc: str,
    commitments: Mapping[str, str],
) -> bytes:
    """Bind one verified scheduled slot to caller-held information-set hashes."""

    protocol = default_evaluation_protocol()
    slot = _select_origin_slot(protocol, slot_id)
    cadence_slot = slot.slot_id.removeprefix("origin-")
    schedule = verify_signed_schedule_manifest(
        schedule_manifest_payload, trusted_public_key=trusted_schedule_public_key
    )
    entry = schedule.entry_for(cadence_slot)
    _validate_commitments(commitments)
    first_target = _parse_utc(first_target_delivery_start_utc, label="first target delivery start")
    if first_target != _first_delivery_start_utc(slot):
        raise OriginRegistrationEnvelopeError(
            "first target delivery start does not match the frozen local delivery month"
        )
    origin = _require_canonical_slot_time(slot)
    window_open = _parse_utc(str(entry["capture_window_open_utc"]), label="capture open")
    window_close = _parse_utc(str(entry["capture_window_close_utc"]), label="capture close")
    deadline = _parse_utc(
        str(entry["latest_external_registry_commit_utc"]),
        label="external registry deadline",
    )
    if not window_open <= origin <= window_close <= deadline < first_target:
        raise OriginRegistrationEnvelopeError(
            "origin, capture, registry deadline, and first target chronology is invalid"
        )
    entry_bytes = canonical_json_bytes(entry)
    core: dict[str, object] = {
        "schema_version": INFORMATION_SET_SCHEMA_VERSION,
        "evaluation_protocol_sha256": CANONICAL_PROTOCOL_SEMANTIC_SHA256,
        "evaluation_protocol_id": protocol.protocol_id,
        "holdout_id": protocol.holdout.holdout_id,
        "origin_registry_protocol_sha256": ORIGIN_REGISTRY_PROTOCOL_SHA256,
        "origin_registry_protocol_id": ORIGIN_REGISTRY_PROTOCOL_ID,
        "slot_id": slot.slot_id,
        "cadence_slot_id": cadence_slot,
        "origin_as_of_utc": _render_utc(origin),
        "first_delivery_month": slot.first_delivery_month,
        "last_delivery_month": slot.last_delivery_month,
        "first_target_delivery_start_utc": first_target_delivery_start_utc,
        "schedule_manifest_sha256": schedule.payload_sha256,
        "schedule_entry_id": entry["schedule_entry_id"],
        "schedule_entry_sha256": hashlib.sha256(entry_bytes).hexdigest(),
        "schedule_signer_key_id": schedule.signer_key_id,
        "information_set_commitments": dict(commitments),
        "truth_opened": False,
        "future_holdout_consumed": False,
        "realized_maturity_mask_present": False,
        "missing_registration_evidence": [
            "TRUSTED_ORIGIN_TIME_RECEIPT",
            "INDEPENDENT_REQUEST_SIGNATURE",
            "EXTERNAL_COMPARE_AND_APPEND_RECEIPT",
            "FRESH_EXTERNAL_REGISTRY_HEAD_OBSERVATION",
        ],
        "authority": dict(_AUTHORITY),
    }
    envelope_id = _domain_hash(_INFORMATION_SET_ID_DOMAIN, core)
    return canonical_json_bytes({**core, "envelope_id": envelope_id})


def verify_origin_information_set_envelope(
    payload: bytes,
    *,
    schedule_manifest_payload: bytes,
    trusted_schedule_public_key: Ed25519PublicKey,
) -> Mapping[str, object]:
    """Recompute identity, signed-schedule bindings, and negative authority."""

    document = _strict_canonical_mapping(payload, label="origin information-set envelope")
    expected_fields = {
        "schema_version",
        "envelope_id",
        "evaluation_protocol_sha256",
        "evaluation_protocol_id",
        "holdout_id",
        "origin_registry_protocol_sha256",
        "origin_registry_protocol_id",
        "slot_id",
        "cadence_slot_id",
        "origin_as_of_utc",
        "first_delivery_month",
        "last_delivery_month",
        "first_target_delivery_start_utc",
        "schedule_manifest_sha256",
        "schedule_entry_id",
        "schedule_entry_sha256",
        "schedule_signer_key_id",
        "information_set_commitments",
        "truth_opened",
        "future_holdout_consumed",
        "realized_maturity_mask_present",
        "missing_registration_evidence",
        "authority",
    }
    _exact_fields(document, expected_fields, label="origin information-set envelope")
    core = dict(document)
    envelope_id = core.pop("envelope_id")
    if envelope_id != _domain_hash(_INFORMATION_SET_ID_DOMAIN, core):
        raise OriginRegistrationEnvelopeError("information-set envelope id mismatch")
    protocol = default_evaluation_protocol()
    slot = _select_origin_slot(protocol, str(document["slot_id"]))
    schedule = verify_signed_schedule_manifest(
        schedule_manifest_payload, trusted_public_key=trusted_schedule_public_key
    )
    entry = schedule.entry_for(slot.slot_id.removeprefix("origin-"))
    entry_sha256 = hashlib.sha256(canonical_json_bytes(entry)).hexdigest()
    if (
        document["schema_version"] != INFORMATION_SET_SCHEMA_VERSION
        or document["evaluation_protocol_sha256"] != CANONICAL_PROTOCOL_SEMANTIC_SHA256
        or document["evaluation_protocol_id"] != protocol.protocol_id
        or document["holdout_id"] != protocol.holdout.holdout_id
        or document["origin_registry_protocol_sha256"] != ORIGIN_REGISTRY_PROTOCOL_SHA256
        or document["origin_registry_protocol_id"] != ORIGIN_REGISTRY_PROTOCOL_ID
        or document["cadence_slot_id"] != slot.slot_id.removeprefix("origin-")
        or document["origin_as_of_utc"] != _render_utc(_require_canonical_slot_time(slot))
        or document["first_delivery_month"] != slot.first_delivery_month
        or document["last_delivery_month"] != slot.last_delivery_month
        or document["schedule_manifest_sha256"] != schedule.payload_sha256
        or document["schedule_entry_id"] != entry["schedule_entry_id"]
        or document["schedule_entry_sha256"] != entry_sha256
        or document["schedule_signer_key_id"] != schedule.signer_key_id
    ):
        raise OriginRegistrationEnvelopeError("information-set protocol binding is invalid")
    for field in (
        "schedule_manifest_sha256",
        "schedule_entry_id",
        "schedule_entry_sha256",
        "schedule_signer_key_id",
    ):
        _require_sha256(document[field], label=field)
    commitments = document["information_set_commitments"]
    if not isinstance(commitments, Mapping):
        raise OriginRegistrationEnvelopeError("information-set commitments are invalid")
    _validate_commitments(commitments)
    if (
        document["truth_opened"] is not False
        or document["future_holdout_consumed"] is not False
        or document["realized_maturity_mask_present"] is not False
        or document["authority"] != _AUTHORITY
        or document["missing_registration_evidence"]
        != [
            "TRUSTED_ORIGIN_TIME_RECEIPT",
            "INDEPENDENT_REQUEST_SIGNATURE",
            "EXTERNAL_COMPARE_AND_APPEND_RECEIPT",
            "FRESH_EXTERNAL_REGISTRY_HEAD_OBSERVATION",
        ]
    ):
        raise OriginRegistrationEnvelopeError("information-set authority must remain negative")
    first_target = _parse_utc(
        str(document["first_target_delivery_start_utc"]),
        label="first target delivery start",
    )
    if first_target != _first_delivery_start_utc(slot):
        raise OriginRegistrationEnvelopeError(
            "first target delivery start does not match the frozen local delivery month"
        )
    if _require_canonical_slot_time(slot) >= first_target:
        raise OriginRegistrationEnvelopeError("first target delivery must follow the frozen origin")
    origin = _require_canonical_slot_time(slot)
    window_open = _parse_utc(str(entry["capture_window_open_utc"]), label="capture open")
    window_close = _parse_utc(str(entry["capture_window_close_utc"]), label="capture close")
    deadline = _parse_utc(
        str(entry["latest_external_registry_commit_utc"]),
        label="external registry deadline",
    )
    if not window_open <= origin <= window_close <= deadline < first_target:
        raise OriginRegistrationEnvelopeError(
            "signed schedule and information-set chronology is invalid"
        )
    return document


def _verify_signed_schedule_entry(
    raw: object, *, trusted_public_key: Ed25519PublicKey
) -> Mapping[str, object]:
    if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
        raise OriginRegistrationEnvelopeError("signed schedule entry must be an object")
    entry = dict(raw)
    _exact_fields(entry, _SCHEDULE_ENTRY_FIELDS, label="signed schedule entry")
    signature = entry.pop("signature")
    signing_document = dict(entry)
    _validate_schedule_signing_document(signing_document)
    _verify_signature(
        signature,
        canonical_json_bytes(signing_document),
        trusted_public_key=trusted_public_key,
    )
    return dict(raw)


def _validate_schedule_signing_document(document: Mapping[str, object]) -> None:
    _exact_fields(document, _SCHEDULE_SIGNING_FIELDS, label="schedule signing document")
    core = dict(document)
    entry_id = core.pop("schedule_entry_id")
    _validate_schedule_core(core)
    if entry_id != _domain_hash(_SCHEDULE_ENTRY_ID_DOMAIN, core):
        raise OriginRegistrationEnvelopeError("schedule entry id mismatch")


def _validate_schedule_core(core: Mapping[str, object]) -> None:
    _exact_fields(core, _SCHEDULE_CORE_FIELDS, label="schedule entry core")
    if core["protocol_sha256"] != ORIGIN_REGISTRY_PROTOCOL_SHA256:
        raise OriginRegistrationEnvelopeError("schedule protocol binding is invalid")
    slot = core["cadence_slot_id"]
    if not isinstance(slot, str) or _CADENCE_SLOT.fullmatch(slot) is None:
        raise OriginRegistrationEnvelopeError("schedule cadence slot is invalid")
    trading_day = core["eex_trading_day"]
    if not isinstance(trading_day, str) or _DATE.fullmatch(trading_day) is None:
        raise OriginRegistrationEnvelopeError("EEX trading day is invalid")
    try:
        parsed_day = date.fromisoformat(trading_day)
    except ValueError as exc:
        raise OriginRegistrationEnvelopeError("EEX trading day is invalid") from exc
    if parsed_day.isoformat() != trading_day or trading_day[:7] != slot:
        raise OriginRegistrationEnvelopeError("EEX trading day must belong to its cadence slot")
    window_open = _parse_utc(str(core["capture_window_open_utc"]), label="capture open")
    window_close = _parse_utc(str(core["capture_window_close_utc"]), label="capture close")
    deadline = _parse_utc(
        str(core["latest_external_registry_commit_utc"]),
        label="external registry deadline",
    )
    if not window_open <= window_close <= deadline:
        raise OriginRegistrationEnvelopeError("schedule chronology is invalid")
    _require_sha256(
        core["official_eex_calendar_document_sha256"],
        label="official EEX calendar document",
    )
    _require_sha256(
        core["official_settlement_event_definition_sha256"],
        label="official settlement event definition",
    )


def _verify_cohort_alignment(
    entries: tuple[Mapping[str, object], ...], protocol: EvaluationProtocol
) -> None:
    expected_slots = tuple(
        slot.slot_id.removeprefix("origin-") for slot in protocol.holdout.origin_slots
    )
    observed_slots = tuple(str(entry["cadence_slot_id"]) for entry in entries)
    if observed_slots != expected_slots or len(set(observed_slots)) != len(observed_slots):
        raise OriginRegistrationEnvelopeError(
            "signed schedule must match the exact ordered prospective cohort"
        )
    for entry, slot in zip(entries, protocol.holdout.origin_slots, strict=True):
        origin = _require_canonical_slot_time(slot)
        window_open = _parse_utc(str(entry["capture_window_open_utc"]), label="capture open")
        window_close = _parse_utc(str(entry["capture_window_close_utc"]), label="capture close")
        if not window_open <= origin <= window_close:
            raise OriginRegistrationEnvelopeError(
                "local proposed origin is outside its signed capture window"
            )


def _select_origin_slot(protocol: EvaluationProtocol, slot_id: str) -> OriginSlot:
    selected = tuple(slot for slot in protocol.holdout.origin_slots if slot.slot_id == slot_id)
    if len(selected) != 1:
        raise OriginRegistrationEnvelopeError("origin slot is not in the frozen cohort")
    return selected[0]


def _validate_commitments(commitments: Mapping[str, str]) -> None:
    _exact_fields(commitments, _COMMITMENT_FIELDS, label="information-set commitments")
    for field, value in commitments.items():
        _require_sha256(value, label=field)


def _verify_signature(
    signature: object,
    payload: bytes,
    *,
    trusted_public_key: Ed25519PublicKey,
) -> None:
    if not isinstance(signature, Mapping):
        raise OriginRegistrationEnvelopeError("schedule signature is invalid")
    _exact_fields(
        signature,
        {"algorithm", "key_id", "value_base64"},
        label="schedule signature",
    )
    if signature["algorithm"] != SIGNATURE_ALGORITHM or signature["key_id"] != public_key_id(
        trusted_public_key
    ):
        raise OriginRegistrationEnvelopeError("schedule signature trust binding is invalid")
    value = _decode_signature(signature["value_base64"])
    try:
        trusted_public_key.verify(value, payload)
    except InvalidSignature as exc:
        raise OriginRegistrationEnvelopeError("schedule signature is invalid") from exc


def _decode_signature(value: object) -> bytes:
    if not isinstance(value, str):
        raise OriginRegistrationEnvelopeError("schedule signature encoding is invalid")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise OriginRegistrationEnvelopeError("schedule signature encoding is invalid") from exc
    if len(decoded) != 64 or base64.b64encode(decoded).decode("ascii") != value:
        raise OriginRegistrationEnvelopeError("schedule signature encoding is invalid")
    return decoded


def _strict_canonical_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_DOCUMENT_BYTES:
        raise OriginRegistrationEnvelopeError(f"{label} byte envelope is invalid")
    try:
        parsed = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except OriginRegistrationEnvelopeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise OriginRegistrationEnvelopeError(f"{label} is not strict JSON") from exc
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        raise OriginRegistrationEnvelopeError(f"{label} must be an object")
    if canonical_json_bytes(parsed) != payload:
        raise OriginRegistrationEnvelopeError(f"{label} is not exact canonical JSON")
    return parsed


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise OriginRegistrationEnvelopeError("strict JSON contains a duplicate key")
        document[key] = value
    return document


def _validate_no_float(value: object) -> None:
    if isinstance(value, float):
        raise OriginRegistrationEnvelopeError("canonical JSON cannot contain floats")
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise OriginRegistrationEnvelopeError("canonical JSON keys must be strings")
        for child in value.values():
            _validate_no_float(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _validate_no_float(child)


def _domain_hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + b"\x00" + canonical_json_bytes(value)).hexdigest()


def _exact_fields(
    value: Mapping[str, object], expected: set[str] | frozenset[str], *, label: str
) -> None:
    if set(value) != set(expected):
        raise OriginRegistrationEnvelopeError(f"{label} fields are not exact")


def _require_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise OriginRegistrationEnvelopeError(f"{label} must be a lowercase SHA-256")


def _parse_utc(value: str, *, label: str) -> datetime:
    if _UTC.fullmatch(value) is None:
        raise OriginRegistrationEnvelopeError(f"{label} must be canonical UTC seconds")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise OriginRegistrationEnvelopeError(f"{label} is invalid") from exc
    return parsed


def _require_canonical_slot_time(slot: OriginSlot) -> datetime:
    value = slot.origin_as_of_utc
    if value.microsecond or value.utcoffset() != timezone.utc.utcoffset(value):
        raise OriginRegistrationEnvelopeError("origin slot timestamp is not canonical UTC")
    return value


def _first_delivery_start_utc(slot: OriginSlot) -> datetime:
    year = int(slot.first_delivery_month[:4])
    month = int(slot.first_delivery_month[5:])
    return datetime(year, month, 1, tzinfo=_LOCAL_TIMEZONE).astimezone(timezone.utc)


def _render_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "INFORMATION_SET_SCHEMA_VERSION",
    "ORIGIN_REGISTRY_PROTOCOL_ID",
    "ORIGIN_REGISTRY_PROTOCOL_SHA256",
    "OriginRegistrationEnvelopeError",
    "SCHEDULE_MANIFEST_SCHEMA_VERSION",
    "SIGNATURE_ALGORITHM",
    "STATUS",
    "ScheduleEntryCore",
    "VerifiedSchedule",
    "assemble_signed_schedule_entry",
    "build_schedule_entry_signature_payload",
    "build_schedule_manifest_payload",
    "canonical_json_bytes",
    "prepare_origin_information_set_envelope",
    "public_key_id",
    "verify_origin_information_set_envelope",
    "verify_signed_schedule_manifest",
]

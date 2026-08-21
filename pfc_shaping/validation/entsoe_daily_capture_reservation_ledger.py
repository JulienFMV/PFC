"""Validate one-capture-per-Zurich-day reservations entirely offline."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.validation.entsoe_bounded_execution_receipt import (
    validate_bounded_capture_proposal,
)

CONTRACT_SCHEMA = "fmv_entsoe_daily_capture_reservation_ledger_contract.v1"
CONTRACT_STATUS = "STATIC_OFFLINE_DAILY_RESERVATION_LEDGER_NO_EXECUTION_AUTHORITY"
CONTRACT_FILE_SHA256 = "827efab999bd7bb9446f753b4b98be57f2638ff4673481d1f3683458084be251"
CONTRACT_CONTENT_ID = "18bf70f82ae3ab078660b765cf131369757002c056a96873fd66e93ded959544"
D233_CONTRACT_SHA256 = "97edaf93c639953365d20d7539380a706f0f8fdbec9476af94262fb1d4b2204d"
D233_CONTENT_ID = "4f6e644b15593e8d4ae2ca196e1d52a002d4255bdf42e7a137982edb854f0fa0"
D235_PROOF_CONTENT_ID = "93504d69834b299ce361352adce433509cf84bce0fbbb940be00a9bd59616ff1"
D235_MANIFEST_SHA256 = "a10bbfa380652365684213197862139bf958dc33e706d81635cc2a6ae4582f02"
D236_CONTRACT_CONTENT_ID = "834cbba541395072886e1ba80012a76cc55a0a8990b6d354f6d1bfe110a12156"
D236_PROOF_CONTENT_ID = "772da4e0b22540bf22e1715ca146cc0a59adf0c7a9ec508e5a893d8495539247"
D236_MANIFEST_SHA256 = "9d76631395176718b600ca0de8f7303ad8faea931c24298f514a54354da3d6e0"
LEDGER_SCHEMA = "fmv_entsoe_daily_capture_reservation_ledger.v1"
CANDIDATE_SCHEMA = "fmv_entsoe_daily_capture_candidate.v1"
COUNTED_STATES = (
    "AUTHORIZED_RESERVED",
    "EXECUTION_STARTED",
    "SUCCEEDED",
    "FAILED",
)
AUTHORITIES = {
    "real_ledger_admitted": False,
    "real_reservation_authorized": False,
    "capture_execution_authorized": False,
    "point_in_time_availability_proven": False,
    "training_authorized": False,
    "selection_authorized": False,
    "model_input_authorized": False,
    "candidate_assembly_authorized": False,
    "promotion_authorized": False,
    "production_authorized": False,
}
LEDGER_FIELDS = frozenset(
    {"schema_version", "evidence_class", "created_at_utc", "entries", "authorities"}
)
ENTRY_FIELDS = frozenset(
    {
        "capture_day_europe_zurich",
        "reservation_id",
        "capture_batch_id",
        "proposal_content_id",
        "reserved_at_utc",
        "state",
    }
)
CANDIDATE_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_class",
        "capture_day_europe_zurich",
        "reservation_id",
        "proposal",
        "authorities",
    }
)
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class EntsoeDailyReservationError(ValueError):
    """Raised when daily reservation evidence is ambiguous or reusable."""


def validate_daily_reservation_contract(document: Mapping[str, object]) -> dict[str, object]:
    contract = _mapping(document, "daily reservation contract")
    _equal(contract.get("schema_version"), CONTRACT_SCHEMA, "contract schema")
    _equal(contract.get("status"), CONTRACT_STATUS, "contract status")
    _equal(contract.get("decision"), "D-20260805-238", "contract decision")
    bindings = _mapping(contract.get("input_bindings"), "input bindings")
    expected_bindings = {
        "governed_acquisition_package_decision": "D-20260805-233",
        "governed_acquisition_package_contract_content_id": D233_CONTENT_ID,
        "external_time_batch_decision": "D-20260805-235",
        "external_time_batch_proof_content_id": D235_PROOF_CONTENT_ID,
        "external_time_batch_proof_manifest_sha256": D235_MANIFEST_SHA256,
        "bounded_execution_decision": "D-20260805-236",
        "bounded_execution_contract_content_id": D236_CONTRACT_CONTENT_ID,
        "bounded_execution_proof_content_id": D236_PROOF_CONTENT_ID,
        "bounded_execution_proof_manifest_sha256": D236_MANIFEST_SHA256,
    }
    _equal(bindings, expected_bindings, "input bindings")
    daily = _mapping(contract.get("daily_policy"), "daily policy")
    _equal(daily.get("capture_timezone"), "Europe/Zurich", "capture timezone")
    _equal(daily.get("maximum_reservations_per_local_day"), 1, "daily ceiling")
    _equal(daily.get("counted_states"), list(COUNTED_STATES), "counted states")
    for field, value in daily.items():
        if field.endswith("authorized"):
            _equal(value, False, field)
    execution = _mapping(contract.get("execution_policy"), "execution policy")
    for field, value in execution.items():
        if field.endswith("budget"):
            _equal(value, 0, field)
        elif field.endswith("allowed") or field.endswith("authorized"):
            _equal(value, False, field)
    _equal(_mapping(contract.get("authorities"), "contract authorities"), AUTHORITIES, "contract authorities")
    content_id = canonical_sha256(contract)
    _equal(content_id, CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": content_id,
        "current_reservation_budget": 0,
        "capture_execution_authorized": False,
    }


def validate_daily_reservation_ledger(
    document: Mapping[str, object], *, reference_time_utc: datetime
) -> dict[str, object]:
    ledger = _mapping(document, "daily reservation ledger")
    _exact_fields(ledger, LEDGER_FIELDS, "daily reservation ledger")
    _equal(ledger.get("schema_version"), LEDGER_SCHEMA, "ledger schema")
    _equal(ledger.get("evidence_class"), "SYNTHETIC_TEST_ONLY", "ledger evidence")
    created = _parse_utc(ledger.get("created_at_utc"), "ledger created_at_utc")
    reference = _utc_datetime(reference_time_utc, "reference_time_utc")
    if created > reference:
        raise EntsoeDailyReservationError("ledger creation is after reference time")
    _equal(_mapping(ledger.get("authorities"), "ledger authorities"), AUTHORITIES, "ledger authorities")
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise EntsoeDailyReservationError("ledger entries must be a list")
    days: set[str] = set()
    reservations: set[str] = set()
    batches: set[str] = set()
    proposals: set[str] = set()
    previous: datetime | None = None
    normalized: list[dict[str, str]] = []
    for index, raw in enumerate(entries):
        entry = _mapping(raw, f"ledger entry {index}")
        _exact_fields(entry, ENTRY_FIELDS, f"ledger entry {index}")
        day = _day(entry.get("capture_day_europe_zurich"), f"entry {index} day")
        reservation = _uuid(entry.get("reservation_id"), f"entry {index} reservation")
        batch = _uuid(entry.get("capture_batch_id"), f"entry {index} batch")
        proposal = _sha(entry.get("proposal_content_id"), f"entry {index} proposal")
        reserved = _parse_utc(entry.get("reserved_at_utc"), f"entry {index} reserved_at")
        state = entry.get("state")
        if state not in COUNTED_STATES:
            raise EntsoeDailyReservationError(f"entry {index} state is invalid")
        if reserved > created or reserved.astimezone(ZoneInfo("Europe/Zurich")).date().isoformat() != day:
            raise EntsoeDailyReservationError(f"entry {index} day or reservation time differs")
        if previous is not None and reserved <= previous:
            raise EntsoeDailyReservationError("reservation times must be strictly increasing")
        if day in days:
            raise EntsoeDailyReservationError("daily capture ceiling exceeded")
        if reservation in reservations or batch in batches or proposal in proposals:
            raise EntsoeDailyReservationError("reservation, batch and proposal IDs must be unique")
        days.add(day)
        reservations.add(reservation)
        batches.add(batch)
        proposals.add(proposal)
        previous = reserved
        normalized.append({"day": day, "reservation_id": reservation, "capture_batch_id": batch, "proposal_content_id": proposal})
    return {
        "schema_version": LEDGER_SCHEMA,
        "status": "PASS_SYNTHETIC_DAILY_LEDGER_STRUCTURE_ONLY",
        "ledger_content_id": canonical_sha256(ledger),
        "entry_count": len(normalized),
        "days": sorted(days),
        "reservation_ids": sorted(reservations),
        "capture_batch_ids": sorted(batches),
        "proposal_content_ids": sorted(proposals),
        "real_ledger_admitted": False,
        "capture_execution_authorized": False,
    }


def validate_daily_capture_candidate(
    candidate_document: Mapping[str, object],
    ledger_document: Mapping[str, object],
    *,
    reference_time_utc: datetime,
) -> dict[str, object]:
    ledger = validate_daily_reservation_ledger(ledger_document, reference_time_utc=reference_time_utc)
    candidate = _mapping(candidate_document, "daily capture candidate")
    _exact_fields(candidate, CANDIDATE_FIELDS, "daily capture candidate")
    _equal(candidate.get("schema_version"), CANDIDATE_SCHEMA, "candidate schema")
    _equal(candidate.get("evidence_class"), "SYNTHETIC_TEST_ONLY", "candidate evidence")
    day = _day(candidate.get("capture_day_europe_zurich"), "candidate day")
    reservation = _uuid(candidate.get("reservation_id"), "candidate reservation")
    _equal(_mapping(candidate.get("authorities"), "candidate authorities"), AUTHORITIES, "candidate authorities")
    proposal = _mapping(candidate.get("proposal"), "embedded D236 proposal")
    proposal_result = validate_bounded_capture_proposal(proposal, reference_time_utc=reference_time_utc)
    _equal(proposal_result.get("evidence_class"), "SYNTHETIC_TEST_ONLY", "proposal evidence")
    created = _parse_utc(proposal.get("created_at_utc"), "proposal created_at_utc")
    if created.astimezone(ZoneInfo("Europe/Zurich")).date().isoformat() != day:
        raise EntsoeDailyReservationError("candidate day differs from proposal Zurich day")
    if day in ledger["days"]:
        raise EntsoeDailyReservationError("daily capture ceiling already consumed")
    if reservation in ledger["reservation_ids"]:
        raise EntsoeDailyReservationError("reservation ID is already used")
    if proposal_result["capture_batch_id"] in ledger["capture_batch_ids"]:
        raise EntsoeDailyReservationError("capture batch ID is already used")
    if proposal_result["proposal_content_id"] in ledger["proposal_content_ids"]:
        raise EntsoeDailyReservationError("proposal content ID is already used")
    return {
        "schema_version": CANDIDATE_SCHEMA,
        "status": "PASS_STRUCTURALLY_ELIGIBLE_DAY_UNRESERVED_EXECUTION_NOT_AUTHORIZED",
        "candidate_content_id": canonical_sha256(candidate),
        "ledger_content_id": ledger["ledger_content_id"],
        "capture_day_europe_zurich": day,
        "proposal_content_id": proposal_result["proposal_content_id"],
        "cost_fence_validated": proposal_result["cost_fence_validated"],
        "reservation_write_authorized": False,
        "capture_execution_authorized": False,
        "point_in_time_availability_proven": False,
        "model_input_authorized": False,
        "production_authorized": False,
    }


def verify_daily_reservation_paths(
    *, contract_path: str | Path, d233_contract_path: str | Path,
    d235_manifest_path: str | Path, d236_manifest_path: str | Path,
    ledger_path: str | Path, candidate_path: str | Path,
    reference_time_utc: datetime,
) -> dict[str, object]:
    contract = _read_hashed_json(contract_path, "D238 contract", CONTRACT_FILE_SHA256)
    validate_daily_reservation_contract(contract)
    d233 = _read_hashed_json(d233_contract_path, "D233 contract", D233_CONTRACT_SHA256)
    _equal(canonical_sha256(d233), D233_CONTENT_ID, "D233 content ID")
    d235 = _read_hashed_json(d235_manifest_path, "D235 proof manifest", D235_MANIFEST_SHA256)
    _equal(d235.get("content_id"), D235_PROOF_CONTENT_ID, "D235 proof content ID")
    d236 = _read_hashed_json(d236_manifest_path, "D236 proof manifest", D236_MANIFEST_SHA256)
    _equal(d236.get("content_id"), D236_PROOF_CONTENT_ID, "D236 proof content ID")
    ledger = _read_json(ledger_path, "daily ledger")
    candidate = _read_json(candidate_path, "daily candidate")
    return validate_daily_capture_candidate(candidate, ledger, reference_time_utc=reference_time_utc)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_hashed_json(path: str | Path, label: str, expected: str) -> Mapping[str, object]:
    raw = _read(path, label)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise EntsoeDailyReservationError(f"{label} SHA-256 differs")
    return _strict_json(raw, label)


def _read_json(path: str | Path, label: str) -> Mapping[str, object]:
    return _strict_json(_read(path, label), label)


def _read(path: str | Path, label: str) -> bytes:
    selected = Path(path)
    if not selected.is_absolute():
        raise EntsoeDailyReservationError(f"{label} path must be absolute")
    try:
        return read_stable_single_link_file(selected, label=label, max_bytes=2_000_000)
    except (OSError, ValueError) as exc:
        raise EntsoeDailyReservationError(str(exc)) from exc


def _strict_json(raw: bytes, label: str) -> Mapping[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise EntsoeDailyReservationError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result
    try:
        return _mapping(json.loads(raw.decode("utf-8"), object_pairs_hook=pairs), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeDailyReservationError(f"{label} is not strict UTF-8 JSON") from exc


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EntsoeDailyReservationError(f"{label} must be an object")
    return value


def _exact_fields(value: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise EntsoeDailyReservationError(f"{label} fields differ")


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise EntsoeDailyReservationError(f"{label} must use UTC whole-second Z format")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise EntsoeDailyReservationError(f"{label} is invalid") from exc


def _utc_datetime(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise EntsoeDailyReservationError(f"{label} must be aware UTC")
    return value.astimezone(timezone.utc)


def _day(value: object, label: str) -> str:
    if not isinstance(value, str) or _DAY.fullmatch(value) is None:
        raise EntsoeDailyReservationError(f"{label} must be YYYY-MM-DD")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise EntsoeDailyReservationError(f"{label} is invalid") from exc
    return value


def _uuid(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise EntsoeDailyReservationError(f"{label} must be canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise EntsoeDailyReservationError(f"{label} must be canonical UUID") from exc
    if str(parsed) != value or parsed.variant != uuid.RFC_4122:
        raise EntsoeDailyReservationError(f"{label} must be canonical RFC 4122 UUID")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise EntsoeDailyReservationError(f"{label} must be lowercase SHA-256")
    return value


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoeDailyReservationError(f"{label} differs: expected {expected!r}, got {value!r}")

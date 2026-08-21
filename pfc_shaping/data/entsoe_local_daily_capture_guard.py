"""Consume one local Europe/Zurich ENTSO-E capture day atomically."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.validation.entsoe_series_inventory import (
    CONTRACT_CONTENT_ID as D243_CONTRACT_CONTENT_ID,
)
from pfc_shaping.validation.entsoe_series_inventory import (
    CONTRACT_FILE_SHA256 as D243_CONTRACT_FILE_SHA256,
)
from pfc_shaping.validation.entsoe_series_inventory import (
    QUERY_SHA256 as D243_QUERY_SHA256,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_RESERVATION_ROOT = (
    WORKSPACE_ROOT
    / "build"
    / "databricks-control-plane"
    / "entsoe-daily-reservations"
)
RESERVATION_SCHEMA = "fmv_entsoe_local_daily_capture_reservation.v1"
CONTRACT_SCHEMA = "fmv_entsoe_local_daily_capture_guard_contract.v1"
CONTRACT_STATUS = "STATIC_REAL_LOCAL_DAILY_COST_GUARD_NO_CAPTURE_EXECUTION_AUTHORITY"
CONTRACT_FILE_SHA256 = "20deb61e362362c46ab5864020a3a70d9f16bcebc41cb729030707cbea7de5e3"
CONTRACT_CONTENT_ID = "66f846bd91ad23e9111a216acff3ed8058929269175dc5da8d893054a176c312"
STATUS = "PASS_REAL_LOCAL_DAY_CONSUMED_EXECUTION_NOT_STARTED_NO_MODEL_AUTHORITY"
MAX_RESERVATION_AGE_SECONDS = 60
PURPOSE = "ENTSOE_SERIES_DIMENSION_INVENTORY"
AUTHORITIES = {
    "capture_execution_authorized": False,
    "point_in_time_availability_proven": False,
    "model_input_authorized": False,
    "production_authorized": False,
}
RESERVATION_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_class",
        "purpose",
        "query_sha256",
        "d243_contract_content_id",
        "capture_day_europe_zurich",
        "reservation_id",
        "reserved_at_utc",
        "warehouse_id_sha256",
        "clock_source",
        "state",
        "failure_consumes_day",
        "retry_same_day_authorized",
        "authorities",
    }
)
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class EntsoeLocalDailyCaptureGuardError(ValueError):
    """Raised when a local daily reservation is malformed or unsafe."""


class EntsoeCaptureDayConsumedError(EntsoeLocalDailyCaptureGuardError):
    """Raised whenever a marker for the requested local day already exists."""


def validate_local_daily_guard_contract(document: Mapping[str, object]) -> dict[str, object]:
    """Validate the exact D246 contract without reserving or executing."""

    contract = _mapping(document, "local daily guard contract")
    if set(contract) != {
        "schema_version",
        "status",
        "decision",
        "purpose",
        "input_bindings",
        "reservation_policy",
        "secret_policy",
        "execution_boundary",
        "authorities",
    }:
        raise EntsoeLocalDailyCaptureGuardError("local daily guard contract fields differ")
    _equal(contract.get("schema_version"), CONTRACT_SCHEMA, "contract schema")
    _equal(contract.get("status"), CONTRACT_STATUS, "contract status")
    _equal(contract.get("decision"), "D-20260806-246", "contract decision")
    bindings = _mapping(contract.get("input_bindings"), "contract input bindings")
    _equal(bindings.get("series_inventory_decision"), "D-20260806-243", "D243 decision")
    _equal(
        bindings.get("series_inventory_contract_content_id"),
        D243_CONTRACT_CONTENT_ID,
        "D243 contract content ID",
    )
    _equal(
        bindings.get("series_inventory_contract_sha256"),
        D243_CONTRACT_FILE_SHA256,
        "D243 contract SHA-256",
    )
    _equal(bindings.get("series_inventory_query_sha256"), D243_QUERY_SHA256, "D243 query SHA-256")
    _equal(bindings.get("synthetic_predecessor_decision"), "D-20260805-238", "D238 decision")
    _equal(
        bindings.get("synthetic_predecessor_is_real_reservation_authority"),
        False,
        "D238 real reservation authority",
    )
    policy = _mapping(contract.get("reservation_policy"), "reservation policy")
    _equal(
        policy,
        {
            "capture_timezone": "Europe/Zurich",
            "maximum_reservations_per_local_day": 1,
            "maximum_request_age_seconds": MAX_RESERVATION_AGE_SECONDS,
            "reservation_root": "build/databricks-control-plane/entsoe-daily-reservations",
            "marker_name": "YYYY-MM-DD.json",
            "exclusive_create_required": True,
            "single_link_regular_file_required": True,
            "file_fsync_required": True,
            "directory_fsync_available_on_windows": False,
            "power_loss_durability_proven_on_windows": False,
            "existing_marker_always_consumes_day": True,
            "corrupt_or_partial_marker_consumes_day": True,
            "publication_failure_after_create_consumes_day": True,
            "same_day_retry_authorized": False,
            "automatic_marker_deletion_authorized": False,
            "clock_source": "LOCAL_SYSTEM_UTC_UNATTESTED",
        },
        "reservation policy",
    )
    secret_policy = _mapping(contract.get("secret_policy"), "secret policy")
    _equal(
        secret_policy,
        {
            "token_allowed": False,
            "host_allowed": False,
            "http_path_allowed": False,
            "warehouse_identifier_allowed": False,
            "warehouse_identifier_sha256_required": True,
        },
        "secret policy",
    )
    boundary = _mapping(contract.get("execution_boundary"), "execution boundary")
    _equal(
        boundary,
        {
            "warehouse_state_checked_by_guard": False,
            "warehouse_start_count": 0,
            "sql_statement_count": 0,
            "databricks_write_count": 0,
            "capture_execution_authorized": False,
        },
        "execution boundary",
    )
    authorities = _mapping(contract.get("authorities"), "contract authorities")
    _equal(
        authorities,
        {
            "local_daily_cost_guard": True,
            "external_monotone_time": False,
            "cross_host_uniqueness": False,
            "warehouse_execution": False,
            "point_in_time_availability": False,
            "model_input": False,
            "production": False,
        },
        "contract authorities",
    )
    content_id = canonical_sha256(contract)
    _equal(content_id, CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": content_id,
        "local_daily_cost_guard_implemented": True,
        "cross_host_uniqueness_proven": False,
        "capture_execution_authorized": False,
        "model_input_authorized": False,
        "production_authorized": False,
    }


def validate_local_daily_guard_contract_path(path: str | Path) -> dict[str, object]:
    """Read and bind the exact local D245 contract."""

    contract_path = Path(path)
    if not contract_path.is_absolute():
        raise EntsoeLocalDailyCaptureGuardError("local daily guard contract path must be absolute")
    try:
        payload = read_stable_single_link_file(
            contract_path,
            label="D246 local daily guard contract",
            max_bytes=100_000,
        )
    except (OSError, ValueError) as exc:
        raise EntsoeLocalDailyCaptureGuardError("local daily guard contract read failed") from exc
    if hashlib.sha256(payload).hexdigest() != CONTRACT_FILE_SHA256:
        raise EntsoeLocalDailyCaptureGuardError("local daily guard contract SHA-256 differs")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeLocalDailyCaptureGuardError("local daily guard contract JSON is invalid") from exc
    return validate_local_daily_guard_contract(_mapping(document, "local daily guard contract"))


def reserve_entsoe_capture_day(
    reservation_document: Mapping[str, object],
    *,
    reference_time_utc: datetime,
) -> dict[str, object]:
    """Atomically consume one local day before any Warehouse or SQL action."""

    reservation = validate_local_reservation_document(
        reservation_document,
        reference_time_utc=reference_time_utc,
    )
    root = _prepare_canonical_root()
    day = str(reservation["capture_day_europe_zurich"])
    target = assert_absolute_path_has_no_links(root / f"{day}.json")
    if target.exists():
        raise EntsoeCaptureDayConsumedError(
            f"ENTSO-E capture day is already consumed: {day}"
        )
    core = dict(reservation)
    content_id = canonical_sha256(core)
    marker = {"reservation_content_id": content_id, **core}
    payload = canonical_bytes(marker)
    try:
        _write_exclusive_marker(target, payload)
    except FileExistsError as exc:
        raise EntsoeCaptureDayConsumedError(
            f"ENTSO-E capture day is already consumed: {day}"
        ) from exc
    except Exception as exc:
        raise EntsoeLocalDailyCaptureGuardError(
            "daily reservation publication failed; the day remains consumed"
        ) from exc
    try:
        observed = read_stable_single_link_file(
            target,
            label="ENTSO-E local daily reservation",
            max_bytes=100_000,
        )
    except (OSError, ValueError) as exc:
        raise EntsoeLocalDailyCaptureGuardError(
            "daily reservation marker cannot be verified; the day remains consumed"
        ) from exc
    if observed != payload:
        raise EntsoeLocalDailyCaptureGuardError(
            "daily reservation marker differs; the day remains consumed"
        )
    return {
        "schema_version": "fmv_entsoe_local_daily_capture_reservation_receipt.v1",
        "status": STATUS,
        "reservation_content_id": content_id,
        "capture_day_europe_zurich": day,
        "reservation_path": target.as_posix(),
        "reservation_sha256": hashlib.sha256(payload).hexdigest(),
        "day_consumed": True,
        "same_day_retry_authorized": False,
        "warehouse_state_checked": False,
        "warehouse_start_count": 0,
        "sql_statement_count": 0,
        "databricks_write_count": 0,
        "capture_execution_authorized": False,
        "model_input_authorized": False,
        "production_authorized": False,
    }


def validate_local_reservation_document(
    reservation_document: Mapping[str, object],
    *,
    reference_time_utc: datetime,
) -> dict[str, object]:
    """Validate the exact real-local reservation request without writing."""

    reservation = _mapping(reservation_document, "local reservation")
    if set(reservation) != RESERVATION_FIELDS:
        raise EntsoeLocalDailyCaptureGuardError("local reservation fields differ")
    _equal(reservation.get("schema_version"), RESERVATION_SCHEMA, "reservation schema")
    _equal(
        reservation.get("evidence_class"),
        "REAL_LOCAL_WORKSTATION_COST_GUARD",
        "reservation evidence class",
    )
    _equal(reservation.get("purpose"), PURPOSE, "reservation purpose")
    _equal(reservation.get("query_sha256"), D243_QUERY_SHA256, "D243 query SHA-256")
    _equal(
        reservation.get("d243_contract_content_id"),
        D243_CONTRACT_CONTENT_ID,
        "D243 contract content ID",
    )
    day = _day(reservation.get("capture_day_europe_zurich"))
    _uuid(reservation.get("reservation_id"))
    reserved_at = _utc(reservation.get("reserved_at_utc"), "reserved_at_utc")
    reference = _aware_utc(reference_time_utc)
    if reserved_at > reference:
        raise EntsoeLocalDailyCaptureGuardError("reservation time is after reference time")
    age_seconds = int((reference - reserved_at).total_seconds())
    if age_seconds > MAX_RESERVATION_AGE_SECONDS:
        raise EntsoeLocalDailyCaptureGuardError("reservation request is stale")
    actual_day = reserved_at.astimezone(ZoneInfo("Europe/Zurich")).date().isoformat()
    _equal(day, actual_day, "Europe/Zurich reservation day")
    _sha(reservation.get("warehouse_id_sha256"), "warehouse ID SHA-256")
    _equal(
        reservation.get("clock_source"),
        "LOCAL_SYSTEM_UTC_UNATTESTED",
        "clock source",
    )
    _equal(reservation.get("state"), "AUTHORIZED_RESERVED", "reservation state")
    _equal(reservation.get("failure_consumes_day"), True, "failure policy")
    _equal(
        reservation.get("retry_same_day_authorized"),
        False,
        "same-day retry authority",
    )
    _equal(reservation.get("authorities"), AUTHORITIES, "reservation authorities")
    return dict(reservation)


def canonical_bytes(value: object) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _prepare_canonical_root() -> Path:
    try:
        workspace = assert_absolute_path_has_no_links(WORKSPACE_ROOT)
        root = assert_absolute_path_has_no_links(Path(CANONICAL_RESERVATION_ROOT))
    except ValueError as exc:
        raise EntsoeLocalDailyCaptureGuardError(
            "reservation path safety check failed"
        ) from exc
    try:
        relative = root.relative_to(workspace / "build")
    except ValueError as exc:
        raise EntsoeLocalDailyCaptureGuardError(
            "reservation root must remain below workspace build"
        ) from exc
    if not relative.parts:
        raise EntsoeLocalDailyCaptureGuardError("reservation root is too broad")
    try:
        root.mkdir(parents=True, exist_ok=True)
        root = assert_absolute_path_has_no_links(root)
    except (OSError, ValueError) as exc:
        raise EntsoeLocalDailyCaptureGuardError(
            "reservation root creation or validation failed"
        ) from exc
    if not root.is_dir():
        raise EntsoeLocalDailyCaptureGuardError("reservation root is unavailable")
    return root


def _write_exclusive_marker(target: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor: int | None = None
    try:
        descriptor = os.open(target, flags, 0o600)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or int(opened.st_nlink) != 1:
            raise EntsoeLocalDailyCaptureGuardError(
                "daily reservation marker is not a single-link regular file"
            )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeLocalDailyCaptureGuardError(f"{label} must be an object")
    return value


def _day(value: object) -> str:
    if not isinstance(value, str) or _DAY.fullmatch(value) is None:
        raise EntsoeLocalDailyCaptureGuardError("capture day must be YYYY-MM-DD")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise EntsoeLocalDailyCaptureGuardError("capture day is invalid") from exc
    return value


def _uuid(value: object) -> None:
    if not isinstance(value, str):
        raise EntsoeLocalDailyCaptureGuardError("reservation ID must be canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise EntsoeLocalDailyCaptureGuardError("reservation ID must be canonical UUID") from exc
    if str(parsed) != value or parsed.variant != uuid.RFC_4122:
        raise EntsoeLocalDailyCaptureGuardError("reservation ID must be canonical RFC 4122 UUID")


def _sha(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise EntsoeLocalDailyCaptureGuardError(f"{label} must be lowercase SHA-256")


def _utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise EntsoeLocalDailyCaptureGuardError(f"{label} must use UTC whole-second Z")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise EntsoeLocalDailyCaptureGuardError(f"{label} is invalid") from exc


def _aware_utc(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timezone.utc.utcoffset(value)
    ):
        raise EntsoeLocalDailyCaptureGuardError("reference time must be aware UTC")
    return value.astimezone(timezone.utc)


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoeLocalDailyCaptureGuardError(
            f"{label} differs: expected {expected!r}, received {value!r}"
        )


__all__ = [
    "AUTHORITIES",
    "CANONICAL_RESERVATION_ROOT",
    "CONTRACT_CONTENT_ID",
    "CONTRACT_FILE_SHA256",
    "CONTRACT_SCHEMA",
    "CONTRACT_STATUS",
    "MAX_RESERVATION_AGE_SECONDS",
    "PURPOSE",
    "RESERVATION_SCHEMA",
    "STATUS",
    "EntsoeCaptureDayConsumedError",
    "EntsoeLocalDailyCaptureGuardError",
    "canonical_sha256",
    "reserve_entsoe_capture_day",
    "validate_local_daily_guard_contract",
    "validate_local_daily_guard_contract_path",
    "validate_local_reservation_document",
]

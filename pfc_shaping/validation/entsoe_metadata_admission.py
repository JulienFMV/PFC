"""Admit a future ENTSO-E schema result locally without querying Databricks."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from pfc_shaping.path_safety import read_stable_single_link_file

CONTRACT_SCHEMA = "fmv_entsoe_databricks_metadata_admission_contract.v1"
CONTRACT_STATUS = "STATIC_OFFLINE_ADMISSION_CONTRACT_NO_EXECUTION_AUTHORITY"
CONTRACT_FILE_SHA256 = (
    "19033d5e24800abc240d3de4974cc44ec7a3fc7afc6f69db07ee1cf0a6191c43"
)
CONTRACT_CONTENT_ID = (
    "ac0ccb6b5a94e0dee46050b5a765f92efcf800f078b93645917247a3be206564"
)
QUERY_SHA256 = (
    "dec7e207603e3a8b69f5808b42454575f1b4a985a25fa8aca3e5c6a2c95b72fa"
)
RECEIPT_SCHEMA = "fmv_entsoe_databricks_metadata_capture_receipt.v1"
MAX_PAYLOAD_BYTES = 262_144
MAX_ADMISSIBLE_ROWS = 1_023
MAX_CAPTURE_AGE_SECONDS = 3_600
HEADER = (
    "table_catalog",
    "table_schema",
    "table_name",
    "column_name",
    "ordinal_position",
    "data_type",
    "is_nullable",
)
TARGET_TABLES = (
    "dimentsoeseries",
    "factentsoetimeserieslatest",
    "factentsoetimeseriesvintages",
)
RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "query_sha256",
        "statement_id",
        "statement_state",
        "captured_at_utc",
        "row_count",
        "result_truncated",
        "payload_sha256",
        "read_only_metadata_only",
        "remote_write_count",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DATA_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_UTC_SECOND = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class EntsoeMetadataAdmissionError(ValueError):
    """Raised when metadata evidence is incomplete, ambiguous, or unsafe."""


def validate_metadata_admission_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact static D232 admission contract."""

    contract = _mapping(document, "metadata admission contract")
    _equal(contract.get("schema_version"), CONTRACT_SCHEMA, "contract schema")
    _equal(contract.get("status"), CONTRACT_STATUS, "contract status")
    source = _mapping(contract.get("source_query"), "source query")
    _equal(source.get("sha256"), QUERY_SHA256, "source query SHA-256")
    _equal(source.get("statement_count"), 1, "source statement count")
    _equal(source.get("row_limit"), 1_024, "source row limit")
    _equal(source.get("data_table_scan"), False, "source data-table scan")
    _equal(
        source.get("executed_by_this_contract"),
        False,
        "contract execution authority",
    )
    payload = _mapping(contract.get("metadata_payload"), "metadata payload")
    _equal(payload.get("max_bytes"), MAX_PAYLOAD_BYTES, "payload max bytes")
    _equal(
        payload.get("maximum_admissible_rows"),
        MAX_ADMISSIBLE_ROWS,
        "payload maximum rows",
    )
    _equal(payload.get("exact_header"), list(HEADER), "payload header")
    _equal(payload.get("exact_tables"), list(TARGET_TABLES), "payload tables")
    _equal(payload.get("data_values_allowed"), False, "payload value authority")
    receipt = _mapping(contract.get("capture_receipt"), "capture receipt")
    _equal(receipt.get("schema_version"), RECEIPT_SCHEMA, "receipt schema")
    _equal(
        receipt.get("maximum_capture_age_seconds"),
        MAX_CAPTURE_AGE_SECONDS,
        "receipt maximum capture age",
    )
    _equal(
        set(_string_list(receipt.get("required_exact_fields"), "receipt fields")),
        RECEIPT_FIELDS,
        "receipt fields",
    )
    execution = _mapping(contract.get("execution_policy"), "execution policy")
    for field in (
        "current_databricks_statement_budget",
        "current_warehouse_start_budget",
        "current_network_call_budget",
        "current_remote_write_budget",
    ):
        _equal(execution.get(field), 0, f"execution policy {field}")
    _equal(
        execution.get("connector_code_allowed"),
        False,
        "connector code authority",
    )
    authorities = _mapping(contract.get("authorities"), "authorities")
    for field, value in authorities.items():
        _equal(value, False, f"authority {field}")
    content_id = _canonical_sha256(contract)
    _equal(content_id, CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": content_id,
        "current_statement_budget": 0,
        "connector_code_allowed": False,
        "production_authorized": False,
    }


def admit_entsoe_metadata_capture(
    receipt: Mapping[str, object],
    payload: bytes,
    *,
    reference_time_utc: datetime,
) -> dict[str, object]:
    """Validate a content-bound metadata receipt and canonical CSV payload."""

    reference = _utc_datetime(reference_time_utc, "reference_time_utc")
    capture = _mapping(receipt, "metadata capture receipt")
    if set(capture) != RECEIPT_FIELDS:
        raise EntsoeMetadataAdmissionError(
            "metadata capture receipt fields must match the exact contract"
        )
    _equal(capture.get("schema_version"), RECEIPT_SCHEMA, "receipt schema")
    _equal(capture.get("query_sha256"), QUERY_SHA256, "receipt query SHA-256")
    _equal(capture.get("statement_state"), "SUCCEEDED", "statement state")
    _equal(capture.get("result_truncated"), False, "result truncation")
    _equal(
        capture.get("read_only_metadata_only"),
        True,
        "read-only metadata flag",
    )
    _equal(capture.get("remote_write_count"), 0, "remote write count")
    _validate_statement_id(capture.get("statement_id"))
    captured_at = _parse_captured_at(capture.get("captured_at_utc"))
    if captured_at > reference:
        raise EntsoeMetadataAdmissionError("captured_at_utc is after reference time")
    capture_age_seconds = int((reference - captured_at).total_seconds())
    if capture_age_seconds > MAX_CAPTURE_AGE_SECONDS:
        raise EntsoeMetadataAdmissionError("metadata capture is stale")
    expected_payload_hash = _required_sha256(
        capture.get("payload_sha256"), "payload SHA-256"
    )
    payload_hash = hashlib.sha256(payload).hexdigest()
    _equal(payload_hash, expected_payload_hash, "payload SHA-256")
    rows = _parse_metadata_csv(payload)
    row_count = _strict_positive_int(capture.get("row_count"), "receipt row count")
    _equal(len(rows), row_count, "receipt row count")
    profile = _validate_metadata_rows(rows)
    schema_fingerprint = _canonical_sha256(rows)
    return {
        "schema_version": "fmv_entsoe_metadata_schema_admission.v1",
        "status": "PASS_METADATA_SCHEMA_INTEGRITY_MAPPING_STILL_REQUIRED",
        "query_sha256": QUERY_SHA256,
        "statement_id": capture["statement_id"],
        "captured_at_utc": capture["captured_at_utc"],
        "capture_age_seconds": capture_age_seconds,
        "payload_sha256": payload_hash,
        "row_count": len(rows),
        "schema_fingerprint": schema_fingerprint,
        "table_profile": profile,
        "metadata_schema_integrity_verified": True,
        "physical_column_mapping_verified": False,
        "entsoe_values_opened": False,
        "data_export_sql_generated": False,
        "point_in_time_availability_proven": False,
        "training_authorized": False,
        "selection_authorized": False,
        "model_input_authorized": False,
        "candidate_assembly_authorized": False,
        "promotion_authorized": False,
        "production_authorized": False,
        "validator_databricks_request_count": 0,
        "validator_warehouse_start_count": 0,
        "validator_network_call_count": 0,
        "validator_remote_write_count": 0,
    }


def verify_entsoe_metadata_capture_paths(
    *,
    contract_path: str | Path,
    receipt_path: str | Path,
    payload_path: str | Path,
    reference_time_utc: datetime,
) -> dict[str, object]:
    """Read stable local files and perform the offline admission."""

    contract_raw = _read_path(
        contract_path,
        label="D232 metadata admission contract",
        max_bytes=100_000,
    )
    if hashlib.sha256(contract_raw).hexdigest() != CONTRACT_FILE_SHA256:
        raise EntsoeMetadataAdmissionError("metadata admission contract SHA differs")
    contract = _strict_json(contract_raw, "metadata admission contract")
    contract_assessment = validate_metadata_admission_contract(contract)
    receipt_raw = _read_path(
        receipt_path,
        label="metadata capture receipt",
        max_bytes=100_000,
    )
    receipt = _strict_json(receipt_raw, "metadata capture receipt")
    payload = _read_path(
        payload_path,
        label="metadata CSV payload",
        max_bytes=MAX_PAYLOAD_BYTES,
    )
    assessment = admit_entsoe_metadata_capture(
        receipt,
        payload,
        reference_time_utc=reference_time_utc,
    )
    return {
        **assessment,
        "contract_content_id": contract_assessment["contract_content_id"],
    }


def _parse_metadata_csv(payload: bytes) -> list[dict[str, object]]:
    if not payload or len(payload) > MAX_PAYLOAD_BYTES:
        raise EntsoeMetadataAdmissionError("metadata CSV size is outside bounds")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EntsoeMetadataAdmissionError("metadata CSV must be UTF-8") from exc
    if text.startswith("\ufeff"):
        raise EntsoeMetadataAdmissionError("metadata CSV BOM is forbidden")
    if "\r" in text or "\x00" in text or not text.endswith("\n"):
        raise EntsoeMetadataAdmissionError(
            "metadata CSV must use canonical LF with final newline"
        )
    try:
        parsed = list(csv.reader(io.StringIO(text), strict=True))
    except csv.Error as exc:
        raise EntsoeMetadataAdmissionError("metadata CSV syntax is invalid") from exc
    if not parsed or tuple(parsed[0]) != HEADER:
        raise EntsoeMetadataAdmissionError("metadata CSV header differs")
    if len(parsed) == 1 or len(parsed) - 1 > MAX_ADMISSIBLE_ROWS:
        raise EntsoeMetadataAdmissionError("metadata CSV row count is outside bounds")
    rows: list[dict[str, object]] = []
    for row_number, raw in enumerate(parsed[1:], start=2):
        if len(raw) != len(HEADER):
            raise EntsoeMetadataAdmissionError(
                f"metadata CSV field count differs at row {row_number}"
            )
        if any(not value or value.strip() != value for value in raw):
            raise EntsoeMetadataAdmissionError(
                f"metadata CSV has empty or padded field at row {row_number}"
            )
        ordinal = _parse_ordinal(raw[4], row_number)
        rows.append(
            {
                "table_catalog": raw[0],
                "table_schema": raw[1],
                "table_name": raw[2],
                "column_name": raw[3],
                "ordinal_position": ordinal,
                "data_type": raw[5],
                "is_nullable": raw[6],
            }
        )
    return rows


def _validate_metadata_rows(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, int]]:
    for row_number, row in enumerate(rows, start=2):
        _equal(row["table_catalog"], "dev", f"catalog at row {row_number}")
        _equal(row["table_schema"], "gold", f"schema at row {row_number}")
        table = str(row["table_name"])
        if table not in TARGET_TABLES:
            raise EntsoeMetadataAdmissionError(
                f"unexpected target table at row {row_number}: {table}"
            )
    keys = [(str(row["table_name"]), int(row["ordinal_position"])) for row in rows]
    if keys != sorted(keys):
        raise EntsoeMetadataAdmissionError(
            "metadata rows must be ordered by table and ordinal position"
        )
    if len(keys) != len(set(keys)):
        raise EntsoeMetadataAdmissionError("table ordinal positions are duplicated")
    names: set[tuple[str, str]] = set()
    profile: dict[str, dict[str, int]] = {}
    for row_number, row in enumerate(rows, start=2):
        table = str(row["table_name"])
        column = str(row["column_name"])
        if _IDENTIFIER.fullmatch(column) is None:
            raise EntsoeMetadataAdmissionError(
                f"unsafe column identifier at row {row_number}: {column}"
            )
        column_key = (table, column.casefold())
        if column_key in names:
            raise EntsoeMetadataAdmissionError(
                f"case-insensitive column name is duplicated: {table}.{column}"
            )
        names.add(column_key)
        data_type = str(row["data_type"])
        if _DATA_TYPE.fullmatch(data_type) is None:
            raise EntsoeMetadataAdmissionError(
                f"unsupported metadata data type at row {row_number}: {data_type}"
            )
        nullable = str(row["is_nullable"])
        if nullable not in {"YES", "NO"}:
            raise EntsoeMetadataAdmissionError(
                f"is_nullable is invalid at row {row_number}: {nullable}"
            )
        table_profile = profile.setdefault(
            table,
            {"column_count": 0, "nullable_count": 0},
        )
        table_profile["column_count"] += 1
        table_profile["nullable_count"] += int(nullable == "YES")
    if tuple(sorted(profile)) != TARGET_TABLES:
        raise EntsoeMetadataAdmissionError("one or more target tables are missing")
    for table in TARGET_TABLES:
        ordinals = [
            int(row["ordinal_position"])
            for row in rows
            if row["table_name"] == table
        ]
        if ordinals != list(range(1, len(ordinals) + 1)):
            raise EntsoeMetadataAdmissionError(
                f"ordinal positions are not positive and contiguous: {table}"
            )
    return {table: profile[table] for table in TARGET_TABLES}


def _validate_statement_id(value: object) -> None:
    if not isinstance(value, str):
        raise EntsoeMetadataAdmissionError("statement_id must be a lowercase UUID")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise EntsoeMetadataAdmissionError(
            "statement_id must be a lowercase UUID"
        ) from exc
    if str(parsed) != value:
        raise EntsoeMetadataAdmissionError("statement_id must be a lowercase UUID")


def _parse_captured_at(value: object) -> datetime:
    if not isinstance(value, str) or _UTC_SECOND.fullmatch(value) is None:
        raise EntsoeMetadataAdmissionError(
            "captured_at_utc must be UTC at whole-second precision"
        )
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise EntsoeMetadataAdmissionError("captured_at_utc is invalid") from exc


def _utc_datetime(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EntsoeMetadataAdmissionError(f"{label} must be timezone-aware UTC")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise EntsoeMetadataAdmissionError(f"{label} must be timezone-aware UTC")
    return value


def _parse_ordinal(value: str, row_number: int) -> int:
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise EntsoeMetadataAdmissionError(
            f"ordinal_position is invalid at row {row_number}: {value}"
        )
    return int(value)


def _strict_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EntsoeMetadataAdmissionError(f"{label} must be a positive integer")
    return value


def _required_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise EntsoeMetadataAdmissionError(f"{label} is invalid")
    return value


def _read_path(raw_path: str | Path, *, label: str, max_bytes: int) -> bytes:
    path = Path(raw_path)
    if not path.is_absolute():
        raise EntsoeMetadataAdmissionError(f"{label} path must be absolute")
    try:
        return read_stable_single_link_file(path, label=label, max_bytes=max_bytes)
    except (OSError, ValueError) as exc:
        raise EntsoeMetadataAdmissionError(f"{label} path read failed") from exc


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EntsoeMetadataAdmissionError(f"{label} duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeMetadataAdmissionError(f"{label} JSON is invalid") from exc
    return _mapping(value, label)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeMetadataAdmissionError(f"{label} must be a mapping")
    return value


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise EntsoeMetadataAdmissionError(f"{label} must be a string list")
    return value


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoeMetadataAdmissionError(
            f"{label} differs: expected {expected!r}, received {value!r}"
        )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

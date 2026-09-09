"""Assess real Unity Catalog ENTSO-E schemas without opening table rows."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone

CAPTURE_SCHEMA = "fmv_entsoe_unity_catalog_schema_capture.v1"
STATUS = "FAIL_REAL_CONTROL_PLANE_SCHEMA_INCOMPATIBLE_NO_MODEL_AUTHORITY"
TABLES = (
    "dev.gold.dimentsoeseries",
    "dev.gold.factentsoetimeserieslatest",
    "dev.gold.factentsoetimeseriesvintages",
)
AUTHORITIES = {
    "point_in_time_availability_proven": False,
    "model_input_authorized": False,
    "production_authorized": False,
}
REQUIRED = {
    TABLES[0]: {
        "series_id": "SeriesID",
        "field_name": "FieldName",
        "group_name": "GroupName",
        "from_zone": "FromZone",
        "to_zone": "ToZone",
        "unit": "Unit",
        "native_resolution": None,
        "source_endpoint": None,
        "document_id": None,
    },
    TABLES[1]: {
        "series_id": "SeriesID",
        "target_ts_utc": "DateTimeUtc",
        "value": "FieldValue",
        "is_historical": "IsHistorical",
        "quality_flag": None,
        "revision_number": None,
        "meta_load_timestamp_utc": "Meta_Load_Timestamp",
    },
    TABLES[2]: {
        "series_id": "SeriesID",
        "target_ts_utc": "DateTimeUtc",
        "as_of_utc": None,
        "value": "FieldValue",
        "quality_flag": None,
        "revision_number": None,
        "meta_load_timestamp_utc": "Meta_Load_Timestamp",
    },
}
EXPECTED_TYPES = {
    "SeriesID": {"bigint", "long"},
    "DateTimeUtc": {"timestamp"},
    "FieldValue": {"double", "float", "decimal"},
    "IsHistorical": {"boolean"},
    "Meta_Load_Timestamp": {"timestamp"},
}
_SHA = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class EntsoeUnityCatalogSchemaError(ValueError):
    """Raised when control-plane metadata is malformed or ambiguous."""


def assess_unity_catalog_schema(capture: Mapping[str, object]) -> dict[str, object]:
    """Validate a sanitized real capture and report exact contract gaps."""

    document = _mapping(capture, "schema capture")
    _exact(document, {"schema_version", "evidence_class", "source", "captured_at_utc", "workspace_host_sha256", "tables", "execution", "authorities"}, "schema capture")
    _equal(document.get("schema_version"), CAPTURE_SCHEMA, "capture schema")
    _equal(document.get("evidence_class"), "REAL_CONTROL_PLANE_OWNER_ASSERTED", "evidence class")
    _equal(document.get("source"), "UNITY_CATALOG_TABLES_GET", "capture source")
    _utc(document.get("captured_at_utc"), "captured_at_utc")
    _sha(document.get("workspace_host_sha256"), "workspace host hash")
    _equal(document.get("execution"), {"control_plane_get_count": 3, "sql_statement_count": 0, "warehouse_start_count": 0, "databricks_write_count": 0}, "execution counters")
    authorities = _mapping(document.get("authorities"), "authorities")
    _equal(authorities, AUTHORITIES, "authorities")
    raw_tables = document.get("tables")
    if not isinstance(raw_tables, list) or len(raw_tables) != 3:
        raise EntsoeUnityCatalogSchemaError("exactly three tables are required")
    tables: dict[str, Mapping[str, object]] = {}
    columns_by_table: dict[str, dict[str, Mapping[str, object]]] = {}
    for raw in raw_tables:
        table = _mapping(raw, "table metadata")
        _exact(table, {"full_name", "table_type", "data_source_format", "columns"}, "table metadata")
        name = table.get("full_name")
        if name not in TABLES or name in tables:
            raise EntsoeUnityCatalogSchemaError("table names must be exact and unique")
        _equal(table.get("table_type"), "MANAGED", f"{name} table type")
        _equal(table.get("data_source_format"), "DELTA", f"{name} format")
        raw_columns = table.get("columns")
        if not isinstance(raw_columns, list) or not raw_columns:
            raise EntsoeUnityCatalogSchemaError(f"{name} columns are missing")
        columns: dict[str, Mapping[str, object]] = {}
        positions: set[int] = set()
        for raw_column in raw_columns:
            column = _mapping(raw_column, f"{name} column")
            _exact(column, {"name", "type_text", "nullable", "position", "comment"}, f"{name} column")
            column_name = column.get("name")
            position = column.get("position")
            if not isinstance(column_name, str) or column_name in columns:
                raise EntsoeUnityCatalogSchemaError(f"{name} column names differ")
            if isinstance(position, bool) or not isinstance(position, int) or position in positions:
                raise EntsoeUnityCatalogSchemaError(f"{name} column positions differ")
            if not isinstance(column.get("type_text"), str) or not isinstance(column.get("nullable"), bool):
                raise EntsoeUnityCatalogSchemaError(f"{name} column metadata types differ")
            if column.get("comment") is not None and not isinstance(column.get("comment"), str):
                raise EntsoeUnityCatalogSchemaError(f"{name} column comment type differs")
            columns[column_name] = column
            positions.add(position)
        if positions != set(range(len(columns))):
            raise EntsoeUnityCatalogSchemaError(f"{name} positions are not contiguous")
        tables[str(name)] = table
        columns_by_table[str(name)] = columns
    if set(tables) != set(TABLES):
        raise EntsoeUnityCatalogSchemaError("required table set differs")

    mappings: dict[str, dict[str, str]] = {}
    missing: dict[str, list[str]] = {}
    nullable_required: dict[str, list[str]] = {}
    type_failures: list[str] = []
    for table_name, required in REQUIRED.items():
        columns = columns_by_table[table_name]
        mappings[table_name] = {}
        missing[table_name] = []
        nullable_required[table_name] = []
        for logical, physical in required.items():
            if physical is None or physical not in columns:
                missing[table_name].append(logical)
                continue
            mappings[table_name][logical] = physical
            column = columns[physical]
            if column["nullable"]:
                nullable_required[table_name].append(logical)
            allowed = EXPECTED_TYPES.get(physical)
            if allowed is not None and str(column["type_text"]).lower() not in allowed:
                type_failures.append(f"{table_name}.{physical}")
    vintage_columns = columns_by_table[TABLES[2]]
    as_of_candidates = [name for name in ("PublicationTimestampUtc", "PullTimestampUtc") if name in vintage_columns]
    if len(as_of_candidates) != 2:
        raise EntsoeUnityCatalogSchemaError("both vintage PIT timestamp candidates are required")
    findings = [
        {"severity": "CRITICAL", "code": "NORMALIZED_REQUIRED_FIELDS_MISSING", "impact": "D234 cannot compile the real schema without fabrication."},
        {"severity": "CRITICAL", "code": "AS_OF_SEMANTICS_AMBIGUOUS", "impact": "Historical rolling origins can leak revisions."},
        {"severity": "HIGH", "code": "REQUIRED_FIELDS_NULLABLE", "impact": "Completeness must be measured before any model use."},
        {"severity": "HIGH", "code": "SIGN_SEMANTICS_ABSENT", "impact": "Directional flow signs cannot be interpreted safely."},
    ]
    return {
        "schema_version": "fmv_entsoe_unity_catalog_schema_compatibility.v1",
        "status": STATUS,
        "capture_content_id": canonical_sha256(document),
        "table_count": 3,
        "column_count_by_table": {name: len(columns_by_table[name]) for name in TABLES},
        "logical_to_physical": mappings,
        "missing_logical_fields": missing,
        "nullable_required_fields": nullable_required,
        "type_failures": type_failures,
        "vintage_as_of_candidates": as_of_candidates,
        "findings": findings,
        "real_schema_observed": True,
        "table_rows_opened": 0,
        "sql_statement_count": 0,
        "warehouse_start_count": 0,
        "point_in_time_availability_proven": False,
        "model_input_authorized": False,
        "production_authorized": False,
    }


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EntsoeUnityCatalogSchemaError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EntsoeUnityCatalogSchemaError(f"{label} fields differ")


def _sha(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise EntsoeUnityCatalogSchemaError(f"{label} must be lowercase SHA-256")


def _utc(value: object, label: str) -> None:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise EntsoeUnityCatalogSchemaError(f"{label} must use UTC whole-second Z")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise EntsoeUnityCatalogSchemaError(f"{label} is invalid") from exc


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoeUnityCatalogSchemaError(f"{label} differs")

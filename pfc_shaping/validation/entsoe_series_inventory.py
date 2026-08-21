"""Validate one bounded ENTSO-E series-dimension inventory offline."""

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
from pfc_shaping.validation.entsoe_normalized_quality import REQUIRED_GROUPS

INVENTORY_SCHEMA = "fmv_entsoe_series_dimension_inventory_capture.v1"
ASSESSMENT_SCHEMA = "fmv_entsoe_series_dimension_inventory_assessment.v1"
CONTRACT_SCHEMA = "fmv_entsoe_series_inventory_contract.v1"
CONTRACT_STATUS = "STATIC_BOUNDED_DIMENSION_INVENTORY_CONTRACT_NO_EXECUTION_AUTHORITY"
CONTRACT_FILE_SHA256 = "ba8f6945b4a43b54762fa475228edabea4018bfea5871f2581dc53536c0743a1"
CONTRACT_CONTENT_ID = "1e183fc51f2673cfc3fed0035dc4a5e3d84664f51ff8447a355264e5e819ddc6"
QUERY_SHA256 = "16a989d2b1528f79b3ecb7a2d9f8f221a6be67cc1c4d3c622516c8bd0dde95e7"
STATUS = "PASS_BOUNDED_REAL_SERIES_INVENTORY_OWNER_MAPPING_REQUIRED_NO_MODEL_AUTHORITY"
MAX_SIGNATURES = 10_000
QUERY_ROW_LIMIT = MAX_SIGNATURES + 1

SEMANTIC_FIELDS = (
    "group_name",
    "field_name",
    "document_type",
    "business_type",
    "process_type",
    "psr_type",
    "from_zone",
    "to_zone",
    "unit",
)
ROW_FIELDS = frozenset(
    {
        *SEMANTIC_FIELDS,
        "series_count",
        "min_meta_load_timestamp_utc",
        "max_meta_load_timestamp_utc",
        "total_signature_count",
        "total_series_count",
    }
)
DOCUMENT_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_class",
        "query_sha256",
        "statement_id",
        "captured_at_utc",
        "europe_zurich_capture_date",
        "daily_capture_reservation_content_id",
        "catalog",
        "schema",
        "table",
        "columns",
        "rows",
        "execution",
        "authorities",
    }
)
EXECUTION = {
    "statement_count": 1,
    "result_truncated": False,
    "warehouse_state_before": "RUNNING",
    "warehouse_start_count": 0,
    "databricks_write_count": 0,
}
AUTHORITIES = {
    "point_in_time_availability_proven": False,
    "model_input_authorized": False,
    "production_authorized": False,
}
ADDITIONAL_HIGH_VALUE_GROUPS = (
    "generation_unavailability",
    "network_unavailability",
    "installed_generation_capacity",
    "balancing_energy_prices",
    "balancing_energy_volumes",
    "imbalance_price",
    "system_imbalance",
    "reserve_capacity_procurement",
    "reserve_capacity_prices",
    "redispatch",
    "countertrading",
    "net_position",
    "intraday_crosszonal_capacity",
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class EntsoeSeriesInventoryError(ValueError):
    """Raised when a series inventory is incomplete, ambiguous, or unsafe."""


def validate_series_inventory_contract(document: Mapping[str, object]) -> dict[str, object]:
    """Validate the immutable D243 execution-free inventory contract."""

    contract = _mapping(document, "series inventory contract")
    _exact(
        contract,
        frozenset(
            {
                "schema_version",
                "status",
                "decision",
                "purpose",
                "source_query",
                "capture_preconditions",
                "output",
                "taxonomy",
                "authorities",
            }
        ),
        "series inventory contract",
    )
    _equal(contract.get("schema_version"), CONTRACT_SCHEMA, "contract schema")
    _equal(contract.get("status"), CONTRACT_STATUS, "contract status")
    _equal(contract.get("decision"), "D-20260806-243", "contract decision")
    source = _mapping(contract.get("source_query"), "source query")
    _equal(source.get("path"), "docs/data/sql/entsoe_dev_gold_series_inventory.sql", "query path")
    _equal(source.get("sha256"), QUERY_SHA256, "query SHA-256")
    _equal(source.get("statement_count"), 1, "query statement count")
    _equal(source.get("source_table"), "dev.gold.dimentsoeseries", "query source table")
    _equal(source.get("fact_table_scan_count"), 0, "fact-table scan count")
    _equal(source.get("row_limit"), QUERY_ROW_LIMIT, "query row limit")
    _equal(source.get("maximum_admissible_signatures"), MAX_SIGNATURES, "signature cap")
    _equal(source.get("executed_by_this_contract"), False, "contract execution authority")
    preconditions = _mapping(contract.get("capture_preconditions"), "capture preconditions")
    _equal(preconditions.get("warehouse_must_already_be_running"), True, "running Warehouse precondition")
    _equal(preconditions.get("warehouse_start_authorized"), False, "Warehouse start authority")
    _equal(preconditions.get("europe_zurich_daily_reservation_required"), True, "daily reservation requirement")
    _equal(preconditions.get("maximum_capture_count_per_europe_zurich_day"), 1, "daily capture ceiling")
    _equal(preconditions.get("databricks_write_authorized"), False, "Databricks write authority")
    output = _mapping(contract.get("output"), "contract output")
    _equal(output.get("schema_version"), INVENTORY_SCHEMA, "output schema")
    _equal(output.get("exact_columns"), list(ROW_FIELDS_IN_ORDER), "output columns")
    _equal(output.get("market_or_entsoe_fact_values_allowed"), False, "fact-value authority")
    _equal(output.get("truncated_result_admissible"), False, "truncation authority")
    _equal(output.get("empty_result_admissible"), False, "empty-result authority")
    taxonomy = _mapping(contract.get("taxonomy"), "taxonomy")
    _equal(taxonomy.get("required_groups"), list(REQUIRED_GROUPS), "required groups")
    _equal(
        taxonomy.get("additional_high_value_groups"),
        list(ADDITIONAL_HIGH_VALUE_GROUPS),
        "additional high-value groups",
    )
    _equal(taxonomy.get("literal_name_match_is_coverage_authority"), False, "literal-match authority")
    _equal(taxonomy.get("owner_mapping_required_for_family_coverage"), True, "owner mapping requirement")
    authorities = _mapping(contract.get("authorities"), "contract authorities")
    if set(authorities) != {
        "query_execution",
        "logical_family_mapping",
        "family_coverage",
        "point_in_time_availability",
        "model_input",
        "production",
    } or any(value is not False for value in authorities.values()):
        raise EntsoeSeriesInventoryError("contract authorities must be exact and false")
    content_id = canonical_sha256(contract)
    _equal(content_id, CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": content_id,
        "query_execution_authorized": False,
        "warehouse_start_authorized": False,
        "databricks_write_authorized": False,
        "model_input_authorized": False,
        "production_authorized": False,
    }


def validate_series_inventory_contract_path(path: str | Path) -> dict[str, object]:
    """Read and bind the exact local D243 contract."""

    contract_path = Path(path)
    if not contract_path.is_absolute():
        raise EntsoeSeriesInventoryError("series inventory contract path must be absolute")
    try:
        payload = read_stable_single_link_file(
            contract_path,
            label="D243 ENTSO-E series inventory contract",
            max_bytes=100_000,
        )
    except (OSError, ValueError) as exc:
        raise EntsoeSeriesInventoryError("series inventory contract read failed") from exc
    if hashlib.sha256(payload).hexdigest() != CONTRACT_FILE_SHA256:
        raise EntsoeSeriesInventoryError("series inventory contract SHA-256 differs")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeSeriesInventoryError("series inventory contract JSON is invalid") from exc
    return validate_series_inventory_contract(_mapping(document, "series inventory contract"))


def validate_inventory_query(path: str | Path) -> dict[str, object]:
    """Bind the only authorized query text without executing it."""

    query_path = Path(path)
    if not query_path.is_absolute():
        raise EntsoeSeriesInventoryError("inventory query path must be absolute")
    try:
        payload = read_stable_single_link_file(
            query_path,
            label="ENTSO-E series inventory query",
            max_bytes=100_000,
        )
    except (OSError, ValueError) as exc:
        raise EntsoeSeriesInventoryError("inventory query read failed") from exc
    digest = hashlib.sha256(payload).hexdigest()
    if digest != QUERY_SHA256:
        raise EntsoeSeriesInventoryError("inventory query SHA-256 differs")
    try:
        sql = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EntsoeSeriesInventoryError("inventory query must be UTF-8") from exc
    lowered = sql.casefold()
    if not sql.endswith("\n") or "\r" in sql:
        raise EntsoeSeriesInventoryError("inventory query framing differs")
    if "from `dev`.`gold`.`dimentsoeseries`" not in lowered:
        raise EntsoeSeriesInventoryError("inventory query table differs")
    if f"limit {QUERY_ROW_LIMIT}" not in lowered:
        raise EntsoeSeriesInventoryError("inventory query row limit differs")
    for forbidden in (" insert ", " update ", " delete ", " merge ", " create ", " drop ", " alter "):
        if forbidden in f" {lowered} ":
            raise EntsoeSeriesInventoryError("inventory query contains a write token")
    return {
        "query_sha256": digest,
        "statement_count": 1,
        "source_table_count": 1,
        "fact_table_scan_count": 0,
        "row_limit": QUERY_ROW_LIMIT,
        "execution_authorized": False,
    }


def assess_series_inventory(document: Mapping[str, object]) -> dict[str, object]:
    """Validate a captured inventory and report literal taxonomy matches."""

    capture = _mapping(document, "series inventory")
    _exact(capture, DOCUMENT_FIELDS, "series inventory")
    _equal(capture.get("schema_version"), INVENTORY_SCHEMA, "inventory schema")
    _equal(
        capture.get("evidence_class"),
        "REAL_DIMENSION_INVENTORY_OWNER_ASSERTED",
        "evidence class",
    )
    _equal(capture.get("query_sha256"), QUERY_SHA256, "query SHA-256")
    _uuid(capture.get("statement_id"))
    captured_at = _utc(capture.get("captured_at_utc"), "captured_at_utc")
    local_date = captured_at.astimezone(ZoneInfo("Europe/Zurich")).date().isoformat()
    _equal(
        capture.get("europe_zurich_capture_date"),
        local_date,
        "Europe/Zurich capture date",
    )
    _sha(
        capture.get("daily_capture_reservation_content_id"),
        "daily capture reservation content ID",
    )
    _equal(capture.get("catalog"), "dev", "catalog")
    _equal(capture.get("schema"), "gold", "schema")
    _equal(capture.get("table"), "dimentsoeseries", "table")
    _equal(capture.get("columns"), list(ROW_FIELDS_IN_ORDER), "columns")
    _equal(capture.get("execution"), EXECUTION, "execution")
    _equal(capture.get("authorities"), AUTHORITIES, "authorities")

    raw_rows = capture.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise EntsoeSeriesInventoryError("inventory rows must be a non-empty list")
    if len(raw_rows) > MAX_SIGNATURES:
        raise EntsoeSeriesInventoryError("inventory exceeds the admissible signature cap")

    rows: list[Mapping[str, object]] = []
    signatures: set[tuple[object, ...]] = set()
    expected_signature_total: int | None = None
    expected_series_total: int | None = None
    series_sum = 0
    for number, raw_row in enumerate(raw_rows, start=1):
        row = _mapping(raw_row, f"inventory row {number}")
        _exact(row, ROW_FIELDS, f"inventory row {number}")
        for field in SEMANTIC_FIELDS:
            _nullable_clean_string(row.get(field), f"row {number} {field}")
        signature = tuple(row.get(field) for field in SEMANTIC_FIELDS)
        if signature in signatures:
            raise EntsoeSeriesInventoryError("inventory signature is duplicated")
        signatures.add(signature)
        series_count = _positive_int(row.get("series_count"), f"row {number} series_count")
        signature_total = _positive_int(
            row.get("total_signature_count"),
            f"row {number} total_signature_count",
        )
        total_series = _positive_int(
            row.get("total_series_count"),
            f"row {number} total_series_count",
        )
        if expected_signature_total is None:
            expected_signature_total = signature_total
            expected_series_total = total_series
        _equal(signature_total, expected_signature_total, "total signature count")
        _equal(total_series, expected_series_total, "total series count")
        minimum = _nullable_utc(
            row.get("min_meta_load_timestamp_utc"),
            f"row {number} minimum load timestamp",
        )
        maximum = _nullable_utc(
            row.get("max_meta_load_timestamp_utc"),
            f"row {number} maximum load timestamp",
        )
        if (minimum is None) != (maximum is None):
            raise EntsoeSeriesInventoryError("load timestamp bounds must both be null or populated")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise EntsoeSeriesInventoryError("load timestamp bounds are reversed")
        series_sum += series_count
        rows.append(row)
    _equal(expected_signature_total, len(rows), "captured signature count")
    _equal(expected_series_total, series_sum, "captured series count")

    group_counts = _literal_group_counts(rows)
    required_matches = {group: group_counts.get(group, 0) for group in REQUIRED_GROUPS}
    additional_matches = {
        group: group_counts.get(group, 0) for group in ADDITIONAL_HIGH_VALUE_GROUPS
    }
    null_series_counts = {
        field: sum(int(row["series_count"]) for row in rows if row[field] is None)
        for field in SEMANTIC_FIELDS
    }
    units = sorted({str(row["unit"]) for row in rows if row["unit"] is not None})
    return {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": STATUS,
        "inventory_content_id": canonical_sha256(capture),
        "statement_id": capture["statement_id"],
        "captured_at_utc": capture["captured_at_utc"],
        "europe_zurich_capture_date": capture["europe_zurich_capture_date"],
        "signature_count": len(rows),
        "series_count": series_sum,
        "distinct_group_name_count": len(group_counts),
        "literal_required_group_matches": required_matches,
        "literal_additional_group_matches": additional_matches,
        "required_groups_without_literal_match": [
            group for group, count in required_matches.items() if count == 0
        ],
        "additional_groups_without_literal_match": [
            group for group, count in additional_matches.items() if count == 0
        ],
        "null_series_count_by_semantic_field": null_series_counts,
        "observed_units": units,
        "bare_eur_unit_observed": any(unit.casefold() == "eur" for unit in units),
        "eur_per_mwh_unit_observed": any(
            unit.casefold().replace(" ", "") == "eur/mwh" for unit in units
        ),
        "inventory_integrity_verified": True,
        "logical_family_mapping_verified": False,
        "family_coverage_proven": False,
        "data_values_opened": False,
        "point_in_time_availability_proven": False,
        "model_input_authorized": False,
        "production_authorized": False,
        "validator_databricks_request_count": 0,
        "validator_warehouse_start_count": 0,
        "validator_remote_write_count": 0,
    }


ROW_FIELDS_IN_ORDER = (
    *SEMANTIC_FIELDS,
    "series_count",
    "min_meta_load_timestamp_utc",
    "max_meta_load_timestamp_utc",
    "total_signature_count",
    "total_series_count",
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def series_signature_id(row: Mapping[str, object]) -> str:
    """Return the stable identifier for one exact raw semantic signature."""

    signature: dict[str, object] = {}
    for field in SEMANTIC_FIELDS:
        if field not in row:
            raise EntsoeSeriesInventoryError(
                f"series signature field is missing: {field}"
            )
        _nullable_clean_string(row[field], f"series signature {field}")
        signature[field] = row[field]
    return canonical_sha256(signature)


def _literal_group_counts(rows: list[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        raw = row["group_name"]
        if raw is None:
            continue
        normalized = re.sub(r"[^a-z0-9]+", "_", str(raw).casefold()).strip("_")
        counts[normalized] = counts.get(normalized, 0) + int(row["series_count"])
    return counts


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeSeriesInventoryError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise EntsoeSeriesInventoryError(f"{label} fields differ")


def _nullable_clean_string(value: object, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value or value != value.strip() or _CONTROL.search(value):
        raise EntsoeSeriesInventoryError(f"{label} must be null or a clean string")


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EntsoeSeriesInventoryError(f"{label} must be a positive integer")
    return value


def _uuid(value: object) -> None:
    if not isinstance(value, str):
        raise EntsoeSeriesInventoryError("statement_id must be a lowercase UUID")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise EntsoeSeriesInventoryError("statement_id must be a lowercase UUID") from exc
    if str(parsed) != value:
        raise EntsoeSeriesInventoryError("statement_id must be a lowercase UUID")


def _sha(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise EntsoeSeriesInventoryError(f"{label} must be lowercase SHA-256")


def _utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise EntsoeSeriesInventoryError(f"{label} must be UTC with Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise EntsoeSeriesInventoryError(f"{label} is invalid") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EntsoeSeriesInventoryError(f"{label} must be UTC")
    return parsed


def _nullable_utc(value: object, label: str) -> datetime | None:
    if value is None:
        return None
    return _utc(value, label)


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoeSeriesInventoryError(
            f"{label} differs: expected {expected!r}, received {value!r}"
        )


__all__ = [
    "ADDITIONAL_HIGH_VALUE_GROUPS",
    "AUTHORITIES",
    "CONTRACT_CONTENT_ID",
    "CONTRACT_FILE_SHA256",
    "CONTRACT_SCHEMA",
    "CONTRACT_STATUS",
    "EXECUTION",
    "INVENTORY_SCHEMA",
    "QUERY_SHA256",
    "ROW_FIELDS_IN_ORDER",
    "STATUS",
    "EntsoeSeriesInventoryError",
    "assess_series_inventory",
    "canonical_sha256",
    "series_signature_id",
    "validate_inventory_query",
    "validate_series_inventory_contract",
    "validate_series_inventory_contract_path",
]

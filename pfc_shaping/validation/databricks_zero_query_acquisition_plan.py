"""Validate the static D231 acquisition package without contacting Databricks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

from pfc_shaping.path_safety import read_stable_single_link_file

ZERO_QUERY_PLAN_SCHEMA = "fmv-ch-lt-databricks-zero-query-acquisition-plan-v1"
ZERO_QUERY_PLAN_STATUS = "PASS_STATIC_PLAN_ONLY_ZERO_DATABRICKS_EXECUTION"
ZERO_QUERY_PLAN_CONTENT_ID = (
    "3395f45bd1d22663386aa7cd4e93cfe2bc02079fa7cb8b103825b9c5650dc1af"
)
ZERO_QUERY_PLAN_FILE_SHA256 = (
    "f62bd9e0a9ffe6f0daa2b02917b47763b64a340f28ab59cdd664cbdd8ec58999"
)
ENTSOE_SCHEMA_SQL_SHA256 = (
    "dec7e207603e3a8b69f5808b42454575f1b4a985a25fa8aca3e5c6a2c95b72fa"
)
EEX_LOCAL_MANIFEST_SHA256 = (
    "f8ec096be43851d85b16ec2b678d4a695fb0521c2c651e8bcf7c2491a29b50c1"
)
ENTSOE_INTAKE_CONTRACT_SHA256 = (
    "7ede1698099390babfa1d130bfecae61fd1e090888a3d6e6e4f892119db52b87"
)
ENTSOE_METADATA_ROW_LIMIT = 1024
ENTSOE_TARGET_TABLES = (
    "dimentsoeseries",
    "factentsoetimeserieslatest",
    "factentsoetimeseriesvintages",
)
ENTSOE_METADATA_FIELDS = (
    "table_catalog",
    "table_schema",
    "table_name",
    "column_name",
    "ordinal_position",
    "data_type",
    "is_nullable",
)

_FORBIDDEN_SQL_TOKENS = (
    "alter",
    "call",
    "copy",
    "create",
    "delete",
    "drop",
    "execute",
    "grant",
    "insert",
    "merge",
    "optimize",
    "revoke",
    "set",
    "truncate",
    "update",
    "use",
    "vacuum",
)


class DatabricksZeroQueryPlanError(ValueError):
    """Raised when the static zero-query package can spend or drift."""


def validate_zero_query_acquisition_plan(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact D231 no-execution contract."""

    plan = _mapping(document, "zero-query acquisition plan")
    _equal(plan.get("schema_version"), ZERO_QUERY_PLAN_SCHEMA, "plan schema")
    _equal(plan.get("status"), ZERO_QUERY_PLAN_STATUS, "plan status")
    execution = _mapping(plan.get("execution_policy"), "execution policy")
    for field in (
        "current_warehouse_start_budget",
        "current_statement_budget",
        "current_network_call_budget",
        "current_remote_write_budget",
        "future_entsoe_data_statement_budget_before_schema_admission",
    ):
        _equal(execution.get(field), 0, f"execution policy {field}")
    for field in (
        "current_databricks_execution_authorized",
        "automatic_warehouse_start_authorized",
    ):
        _equal(execution.get(field), False, f"execution policy {field}")
    _equal(
        execution.get("future_entsoe_metadata_statement_budget_if_authorized"),
        1,
        "future metadata statement budget",
    )
    _equal(
        execution.get("future_result_row_limit"),
        ENTSOE_METADATA_ROW_LIMIT,
        "future metadata row limit",
    )
    eex = _mapping(plan.get("eex_reuse"), "EEX reuse")
    _equal(eex.get("action"), "REUSE_EXISTING_LOCAL_CAPTURE_NO_QUERY", "EEX action")
    _equal(eex.get("manifest_sha256"), EEX_LOCAL_MANIFEST_SHA256, "EEX manifest")
    _equal(eex.get("new_statement_count"), 0, "EEX new statement count")
    entsoe = _mapping(plan.get("entsoe_schema_stage"), "ENTSO-E schema stage")
    _equal(
        entsoe.get("action"),
        "PREPARE_METADATA_ONLY_DO_NOT_EXECUTE",
        "ENTSO-E action",
    )
    _equal(entsoe.get("target_tables"), list(ENTSOE_TARGET_TABLES), "target tables")
    _equal(entsoe.get("sql_sha256"), ENTSOE_SCHEMA_SQL_SHA256, "schema SQL")
    _equal(
        entsoe.get("selected_metadata_fields"),
        list(ENTSOE_METADATA_FIELDS),
        "metadata fields",
    )
    _equal(entsoe.get("result_row_limit"), ENTSOE_METADATA_ROW_LIMIT, "row limit")
    _equal(entsoe.get("data_values_selected"), False, "data values selected")
    _equal(entsoe.get("data_export_sql_authorized"), False, "data export authority")
    target = _mapping(plan.get("entsoe_target_contract"), "ENTSO-E target contract")
    _equal(target.get("sha256"), ENTSOE_INTAKE_CONTRACT_SHA256, "intake contract")
    authorities = _mapping(plan.get("authorities"), "authorities")
    for field in (
        "databricks_schema_verified",
        "databricks_query_executed",
        "warehouse_started",
        "entsoe_values_opened",
        "entsoe_physical_column_mapping_verified",
        "point_in_time_availability_proven",
        "training_authorized",
        "selection_authorized",
        "model_input_authorized",
        "candidate_assembly_authorized",
        "promotion_authorized",
        "production_authorized",
    ):
        _equal(authorities.get(field), False, f"authority {field}")
    content_id = _sha256_json(plan)
    _equal(content_id, ZERO_QUERY_PLAN_CONTENT_ID, "plan content ID")
    return {
        "schema_version": ZERO_QUERY_PLAN_SCHEMA,
        "status": ZERO_QUERY_PLAN_STATUS,
        "plan_content_id": content_id,
        "current_statement_budget": 0,
        "future_metadata_statement_budget_if_authorized": 1,
        "future_data_statement_budget_before_schema_admission": 0,
        "eex_new_statement_count": 0,
        "databricks_query_executed": False,
        "warehouse_started": False,
        "production_authorized": False,
    }


def verify_zero_query_acquisition_plan_paths(
    *,
    plan_path: str | Path,
    entsoe_schema_sql_path: str | Path,
    eex_local_manifest_path: str | Path,
    entsoe_intake_contract_path: str | Path,
) -> dict[str, object]:
    """Verify exact local files and return only a zero-execution assessment."""

    plan_raw = _read_exact(
        plan_path,
        expected_sha256=ZERO_QUERY_PLAN_FILE_SHA256,
        label="D231 zero-query plan",
        max_bytes=100_000,
    )
    plan = _strict_json(plan_raw, "D231 zero-query plan")
    contract = validate_zero_query_acquisition_plan(plan)
    sql_raw = _read_exact(
        entsoe_schema_sql_path,
        expected_sha256=ENTSOE_SCHEMA_SQL_SHA256,
        label="ENTSO-E schema SQL",
        max_bytes=10_000,
    )
    sql_assessment = validate_entsoe_schema_inventory_sql(sql_raw)
    eex_raw = _read_exact(
        eex_local_manifest_path,
        expected_sha256=EEX_LOCAL_MANIFEST_SHA256,
        label="EEX local manifest",
        max_bytes=100_000,
    )
    eex = _strict_json(eex_raw, "EEX local manifest")
    _validate_eex_manifest(eex)
    entsoe_contract_raw = _read_exact(
        entsoe_intake_contract_path,
        expected_sha256=ENTSOE_INTAKE_CONTRACT_SHA256,
        label="ENTSO-E intake contract",
        max_bytes=100_000,
    )
    entsoe_contract = _strict_json(entsoe_contract_raw, "ENTSO-E intake contract")
    _validate_entsoe_intake_contract(entsoe_contract)
    return {
        "schema_version": "fmv_ch_lt_zero_query_acquisition_assessment.v1",
        "status": ZERO_QUERY_PLAN_STATUS,
        "plan_content_id": contract["plan_content_id"],
        "entsoe_schema_sql_sha256": ENTSOE_SCHEMA_SQL_SHA256,
        "entsoe_metadata_target_tables": list(ENTSOE_TARGET_TABLES),
        "entsoe_metadata_selected_fields": list(ENTSOE_METADATA_FIELDS),
        "entsoe_metadata_row_limit": sql_assessment["row_limit"],
        "entsoe_metadata_statement_prepared": True,
        "entsoe_metadata_statement_executed": False,
        "entsoe_data_statement_generated": False,
        "entsoe_data_values_opened": False,
        "entsoe_physical_column_mapping_verified": False,
        "eex_capture_action": "REUSE_EXISTING_LOCAL_CAPTURE_NO_QUERY",
        "eex_capture_row_count": eex["row_count"],
        "eex_capture_artifact_sha256": eex["artifact"]["sha256"],
        "eex_new_statement_count": 0,
        "current_statement_budget": 0,
        "future_metadata_statement_budget_if_authorized": 1,
        "future_data_statement_budget_before_schema_admission": 0,
        "databricks_request_count": 0,
        "warehouse_start_count": 0,
        "network_call_count": 0,
        "h_drive_access_count": 0,
        "remote_write_count": 0,
        "point_in_time_availability_proven": False,
        "training_authorized": False,
        "selection_authorized": False,
        "model_input_authorized": False,
        "candidate_assembly_authorized": False,
        "promotion_authorized": False,
        "production_authorized": False,
    }


def validate_entsoe_schema_inventory_sql(payload: bytes) -> dict[str, object]:
    """Reject any SQL that can read values, mutate state, or exceed the budget."""

    try:
        sql = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatabricksZeroQueryPlanError("schema SQL must be UTF-8") from exc
    if "\r" in sql or not sql.endswith("\n"):
        raise DatabricksZeroQueryPlanError("schema SQL must use LF and final newline")
    lowered = sql.casefold()
    if ";" in sql or "--" in sql or "/*" in sql or "*/" in sql:
        raise DatabricksZeroQueryPlanError("schema SQL comments or separators forbidden")
    if not re.match(r"^select\s", lowered):
        raise DatabricksZeroQueryPlanError("schema SQL must begin with SELECT")
    if re.search(r"\bselect\s+\*", lowered):
        raise DatabricksZeroQueryPlanError("schema SQL SELECT star forbidden")
    for token in _FORBIDDEN_SQL_TOKENS:
        if re.search(rf"\b{token}\b", lowered):
            raise DatabricksZeroQueryPlanError(
                f"schema SQL forbidden token present: {token}"
            )
    if lowered.count("select") != 1:
        raise DatabricksZeroQueryPlanError("schema SQL must contain one SELECT")
    if re.search(r"\bfrom\s+(?:dev|prd)\.gold\.", lowered):
        raise DatabricksZeroQueryPlanError("schema SQL data-table scan forbidden")
    if lowered.count("from dev.information_schema.columns") != 1:
        raise DatabricksZeroQueryPlanError(
            "schema SQL must read only dev.information_schema.columns"
        )
    selected_prefix = lowered.split("from dev.information_schema.columns", 1)[0]
    for field in ENTSOE_METADATA_FIELDS:
        if not re.search(rf"\b{re.escape(field)}\b", selected_prefix):
            raise DatabricksZeroQueryPlanError(
                f"schema SQL metadata field missing: {field}"
            )
    for table in ENTSOE_TARGET_TABLES:
        if lowered.count(f"'{table}'") != 1:
            raise DatabricksZeroQueryPlanError(
                f"schema SQL target table differs: {table}"
            )
    if "where table_catalog = 'dev'" not in lowered:
        raise DatabricksZeroQueryPlanError("schema SQL dev catalog filter missing")
    if "and table_schema = 'gold'" not in lowered:
        raise DatabricksZeroQueryPlanError("schema SQL gold filter missing")
    if "order by table_name, ordinal_position" not in lowered:
        raise DatabricksZeroQueryPlanError("schema SQL deterministic order missing")
    if not re.search(rf"\blimit\s+{ENTSOE_METADATA_ROW_LIMIT}\s*$", lowered):
        raise DatabricksZeroQueryPlanError("schema SQL bounded LIMIT differs")
    return {
        "statement_count": 1,
        "read_only_metadata_only": True,
        "data_table_scan": False,
        "selected_field_count": len(ENTSOE_METADATA_FIELDS),
        "target_table_count": len(ENTSOE_TARGET_TABLES),
        "row_limit": ENTSOE_METADATA_ROW_LIMIT,
        "executed": False,
    }


def _validate_eex_manifest(manifest: Mapping[str, object]) -> None:
    _equal(
        manifest.get("schema_version"),
        "fmv_databricks_eex_daily_snapshot.v1",
        "EEX manifest schema",
    )
    _equal(
        manifest.get("status"),
        "PASS_LOCAL_SNAPSHOT_ONLY_NOT_MODEL_OR_PROMOTION_AUTHORITY",
        "EEX manifest status",
    )
    _equal(manifest.get("catalog"), "prd", "EEX catalog")
    _equal(manifest.get("schema"), "gold", "EEX schema")
    _equal(manifest.get("row_count"), 82552, "EEX row count")
    _equal(manifest.get("statement_count"), 1, "historical EEX statement count")
    _equal(manifest.get("read_only_sql"), True, "historical EEX read-only SQL")
    artifact = _mapping(manifest.get("artifact"), "EEX artifact")
    _equal(
        artifact.get("sha256"),
        "593e916b6aa18ad83f7bd7941ff68184cd71da8882ef4eb381de46d09ce64812",
        "EEX artifact hash",
    )


def _validate_entsoe_intake_contract(contract: Mapping[str, object]) -> None:
    _equal(
        contract.get("schema_version"),
        "fmv_entsoe_databricks_intake_contract.v1",
        "ENTSO-E intake schema",
    )
    source = _mapping(contract.get("source_tables"), "ENTSO-E source tables")
    _equal(source.get("series_dimension"), ENTSOE_TARGET_TABLES[0], "dimension table")
    _equal(source.get("latest_values"), ENTSOE_TARGET_TABLES[1], "latest table")
    _equal(source.get("vintage_values"), ENTSOE_TARGET_TABLES[2], "vintage table")
    authority = _mapping(contract.get("authority"), "ENTSO-E authority")
    for field in (
        "model_training",
        "rolling_origin_selection",
        "candidate_assembly",
        "production_promotion",
        "legacy_local_substitution_authorized",
        "synthetic_substitution_authorized",
    ):
        _equal(authority.get(field), False, f"ENTSO-E authority {field}")


def _read_exact(
    raw_path: str | Path,
    *,
    expected_sha256: str,
    label: str,
    max_bytes: int,
) -> bytes:
    path = Path(raw_path)
    if not path.is_absolute():
        raise DatabricksZeroQueryPlanError(f"{label} path must be absolute")
    try:
        payload = read_stable_single_link_file(path, label=label, max_bytes=max_bytes)
    except (OSError, ValueError) as exc:
        raise DatabricksZeroQueryPlanError(f"{label} path read failed") from exc
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise DatabricksZeroQueryPlanError(f"{label} SHA-256 differs")
    return payload


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DatabricksZeroQueryPlanError(f"{label} duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8-sig"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatabricksZeroQueryPlanError(f"{label} JSON is invalid") from exc
    return _mapping(value, label)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DatabricksZeroQueryPlanError(f"{label} must be a mapping")
    return value


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise DatabricksZeroQueryPlanError(
            f"{label} differs: expected {expected!r}, received {value!r}"
        )


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

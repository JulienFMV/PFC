from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from pfc_shaping.pipeline import production_phases
from pfc_shaping.validation import databricks_zero_query_acquisition_plan as module
from pfc_shaping.validation.databricks_zero_query_acquisition_plan import (
    ENTSOE_METADATA_ROW_LIMIT,
    ENTSOE_SCHEMA_SQL_SHA256,
    ENTSOE_TARGET_TABLES,
    ZERO_QUERY_PLAN_CONTENT_ID,
    ZERO_QUERY_PLAN_STATUS,
    DatabricksZeroQueryPlanError,
    validate_entsoe_schema_inventory_sql,
    validate_zero_query_acquisition_plan,
    verify_zero_query_acquisition_plan_paths,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "CH-LT-DATABRICKS-ZERO-QUERY-ACQUISITION-PLAN-V1.json"
)
SQL_PATH = ROOT / "docs" / "data" / "sql" / "entsoe_dev_gold_schema_inventory.sql"
EEX_MANIFEST_PATH = (
    ROOT / "build" / "databricks-eex-daily" / "2026-08-05" / "manifest.json"
)
ENTSOE_CONTRACT_PATH = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "ENTSOE-DATABRICKS-INTAKE-CONTRACT-V1.json"
)


def _verify() -> dict[str, object]:
    return verify_zero_query_acquisition_plan_paths(
        plan_path=PLAN_PATH.resolve(),
        entsoe_schema_sql_path=SQL_PATH.resolve(),
        eex_local_manifest_path=EEX_MANIFEST_PATH.resolve(),
        entsoe_intake_contract_path=ENTSOE_CONTRACT_PATH.resolve(),
    )


def test_contract_is_exact_zero_execution_and_non_authoritative() -> None:
    document = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    result = validate_zero_query_acquisition_plan(document)

    assert result["status"] == ZERO_QUERY_PLAN_STATUS
    assert result["plan_content_id"] == ZERO_QUERY_PLAN_CONTENT_ID
    assert result["current_statement_budget"] == 0
    assert result["future_metadata_statement_budget_if_authorized"] == 1
    assert result["future_data_statement_budget_before_schema_admission"] == 0
    assert result["eex_new_statement_count"] == 0
    assert result["databricks_query_executed"] is False
    assert result["warehouse_started"] is False
    assert result["production_authorized"] is False


def test_valid_local_package_reuses_eex_and_prepares_metadata_only() -> None:
    result = _verify()

    assert result["entsoe_metadata_statement_prepared"] is True
    assert result["entsoe_metadata_statement_executed"] is False
    assert result["entsoe_data_statement_generated"] is False
    assert result["entsoe_data_values_opened"] is False
    assert result["entsoe_physical_column_mapping_verified"] is False
    assert result["entsoe_metadata_target_tables"] == list(ENTSOE_TARGET_TABLES)
    assert result["entsoe_metadata_row_limit"] == 1024
    assert result["eex_capture_action"] == "REUSE_EXISTING_LOCAL_CAPTURE_NO_QUERY"
    assert result["eex_capture_row_count"] == 82552
    assert result["eex_new_statement_count"] == 0
    assert result["databricks_request_count"] == 0
    assert result["warehouse_start_count"] == 0
    assert result["network_call_count"] == 0
    assert result["h_drive_access_count"] == 0
    assert result["remote_write_count"] == 0
    assert result["training_authorized"] is False
    assert result["production_authorized"] is False


def test_sql_is_one_bounded_metadata_statement_only() -> None:
    payload = SQL_PATH.read_bytes()
    result = validate_entsoe_schema_inventory_sql(payload)

    assert result == {
        "statement_count": 1,
        "read_only_metadata_only": True,
        "data_table_scan": False,
        "selected_field_count": 7,
        "target_table_count": 3,
        "row_limit": ENTSOE_METADATA_ROW_LIMIT,
        "executed": False,
    }
    assert module.hashlib.sha256(payload).hexdigest() == ENTSOE_SCHEMA_SQL_SHA256


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("select_star", "SELECT star"),
        ("data_scan", "data-table scan"),
        ("second_statement", "comments or separators"),
        ("dml", "forbidden token present: delete"),
        ("wrong_catalog", "dev.information_schema.columns"),
        ("missing_catalog_filter", "dev catalog filter missing"),
        ("missing_field", "metadata field missing: is_nullable"),
        ("missing_table", "target table differs: factentsoetimeseriesvintages"),
        ("wrong_limit", "bounded LIMIT differs"),
        ("comment", "comments or separators"),
    ],
)
def test_sql_mutations_fail_closed(mutation: str, message: str) -> None:
    sql = SQL_PATH.read_text(encoding="utf-8")
    if mutation == "select_star":
        sql = sql.replace("SELECT\n  table_catalog,", "SELECT * ,\n  table_catalog,")
    elif mutation == "data_scan":
        sql = sql.replace(
            "FROM dev.information_schema.columns",
            "FROM dev.gold.factentsoetimeserieslatest",
        )
    elif mutation == "second_statement":
        sql += "; SELECT 1\n"
    elif mutation == "dml":
        sql = sql + "DELETE\n"
    elif mutation == "wrong_catalog":
        sql = sql.replace("dev.information_schema.columns", "prd.information_schema.columns")
    elif mutation == "missing_catalog_filter":
        sql = sql.replace("WHERE table_catalog = 'dev'\n  AND ", "WHERE ")
    elif mutation == "missing_field":
        sql = sql.replace(",\n  is_nullable", "")
    elif mutation == "missing_table":
        sql = sql.replace("    'factentsoetimeseriesvintages'\n", "")
    elif mutation == "wrong_limit":
        sql = sql.replace("LIMIT 1024", "LIMIT 2048")
    elif mutation == "comment":
        sql = "-- metadata\n" + sql

    with pytest.raises(DatabricksZeroQueryPlanError, match=message):
        validate_entsoe_schema_inventory_sql(sql.encode())


@pytest.mark.parametrize(
    "mutation",
    [
        "authorize_now",
        "statement_budget",
        "warehouse_budget",
        "eex_query",
        "entsoe_data_query",
        "model_authority",
        "production_authority",
        "content_only",
    ],
)
def test_plan_rejects_cost_or_authority_mutation(mutation: str) -> None:
    document = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    if mutation == "authorize_now":
        document["execution_policy"]["current_databricks_execution_authorized"] = True
    elif mutation == "statement_budget":
        document["execution_policy"]["current_statement_budget"] = 1
    elif mutation == "warehouse_budget":
        document["execution_policy"]["current_warehouse_start_budget"] = 1
    elif mutation == "eex_query":
        document["eex_reuse"]["new_statement_count"] = 1
    elif mutation == "entsoe_data_query":
        document["entsoe_schema_stage"]["data_export_sql_authorized"] = True
    elif mutation == "model_authority":
        document["authorities"]["model_input_authorized"] = True
    elif mutation == "production_authority":
        document["authorities"]["production_authorized"] = True
    elif mutation == "content_only":
        document["required_after_schema_admission_before_any_model_use"].append(
            "invented shortcut"
        )

    with pytest.raises(DatabricksZeroQueryPlanError):
        validate_zero_query_acquisition_plan(document)


def test_exact_path_hashes_reject_local_manifest_drift(tmp_path: Path) -> None:
    changed = tmp_path / "manifest.json"
    changed.write_bytes(EEX_MANIFEST_PATH.read_bytes() + b" ")

    with pytest.raises(DatabricksZeroQueryPlanError, match="EEX local manifest SHA"):
        verify_zero_query_acquisition_plan_paths(
            plan_path=PLAN_PATH.resolve(),
            entsoe_schema_sql_path=SQL_PATH.resolve(),
            eex_local_manifest_path=changed.resolve(),
            entsoe_intake_contract_path=ENTSOE_CONTRACT_PATH.resolve(),
        )


def test_all_paths_must_be_absolute() -> None:
    with pytest.raises(DatabricksZeroQueryPlanError, match="path must be absolute"):
        verify_zero_query_acquisition_plan_paths(
            plan_path=Path("relative-plan.json"),
            entsoe_schema_sql_path=SQL_PATH.resolve(),
            eex_local_manifest_path=EEX_MANIFEST_PATH.resolve(),
            entsoe_intake_contract_path=ENTSOE_CONTRACT_PATH.resolve(),
        )


def test_validator_has_no_connector_or_execution_surface() -> None:
    source = inspect.getsource(module)
    assert "databricks.sql" not in source
    assert "requests." not in source
    assert "Invoke-RestMethod" not in source
    assert "DATABRICKS_TOKEN" not in source
    assert "query_to_df" not in source
    assert "execute(" not in source


def test_plan_is_absent_from_lt_production() -> None:
    source = inspect.getsource(production_phases)
    assert "databricks_zero_query_acquisition_plan" not in source
    assert "verify_zero_query_acquisition_plan_paths" not in source

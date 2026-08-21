from __future__ import annotations

import copy
from pathlib import Path

import pytest

from pfc_shaping.validation.entsoe_series_inventory import (
    AUTHORITIES,
    CONTRACT_CONTENT_ID,
    CONTRACT_STATUS,
    EXECUTION,
    INVENTORY_SCHEMA,
    QUERY_SHA256,
    ROW_FIELDS_IN_ORDER,
    STATUS,
    EntsoeSeriesInventoryError,
    assess_series_inventory,
    validate_inventory_query,
    validate_series_inventory_contract,
    validate_series_inventory_contract_path,
)

ROOT = Path(__file__).resolve().parents[1]
QUERY = ROOT / "docs" / "data" / "sql" / "entsoe_dev_gold_series_inventory.sql"
CONTRACT = (
    ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "ENTSOE-SERIES-INVENTORY-CONTRACT-V1.json"
)


def _row(
    group_name: str | None,
    field_name: str | None,
    *,
    unit: str | None = "MW",
    series_count: int = 1,
) -> dict[str, object]:
    return {
        "group_name": group_name,
        "field_name": field_name,
        "document_type": "A65",
        "business_type": "A04",
        "process_type": "A16",
        "psr_type": None,
        "from_zone": "CH",
        "to_zone": None,
        "unit": unit,
        "series_count": series_count,
        "min_meta_load_timestamp_utc": "2026-07-28T00:00:00Z",
        "max_meta_load_timestamp_utc": "2026-08-06T11:00:00Z",
        "total_signature_count": 4,
        "total_series_count": 10,
    }


def capture() -> dict[str, object]:
    rows = [
        _row("Actual Load", "load", series_count=2),
        _row("Day Ahead Prices", "price", unit="EUR/MWh", series_count=3),
        _row("Generation Actual", "solar", series_count=4),
        _row("Installed Generation Capacity", "solar", series_count=1),
    ]
    return {
        "schema_version": INVENTORY_SCHEMA,
        "evidence_class": "REAL_DIMENSION_INVENTORY_OWNER_ASSERTED",
        "query_sha256": QUERY_SHA256,
        "statement_id": "123e4567-e89b-12d3-a456-426614174000",
        "captured_at_utc": "2026-08-06T11:30:00Z",
        "europe_zurich_capture_date": "2026-08-06",
        "daily_capture_reservation_content_id": "a" * 64,
        "catalog": "dev",
        "schema": "gold",
        "table": "dimentsoeseries",
        "columns": list(ROW_FIELDS_IN_ORDER),
        "rows": rows,
        "execution": dict(EXECUTION),
        "authorities": dict(AUTHORITIES),
    }


def test_exact_query_is_bound_and_never_executed() -> None:
    result = validate_inventory_query(QUERY)

    assert result == {
        "query_sha256": QUERY_SHA256,
        "statement_count": 1,
        "source_table_count": 1,
        "fact_table_scan_count": 0,
        "row_limit": 10_001,
        "execution_authorized": False,
    }


def test_exact_contract_is_bound_and_has_zero_authority() -> None:
    result = validate_series_inventory_contract_path(CONTRACT)

    assert result["status"] == CONTRACT_STATUS
    assert result["contract_content_id"] == CONTRACT_CONTENT_ID
    assert result["query_execution_authorized"] is False
    assert result["warehouse_start_authorized"] is False
    assert result["databricks_write_authorized"] is False
    assert result["model_input_authorized"] is False


def test_contract_authority_escalation_is_rejected() -> None:
    import json

    document = json.loads(CONTRACT.read_text(encoding="utf-8"))
    document["authorities"]["model_input"] = True

    with pytest.raises(EntsoeSeriesInventoryError, match="authorities must be exact and false"):
        validate_series_inventory_contract(document)


def test_inventory_reports_literal_matches_without_coverage_authority() -> None:
    result = assess_series_inventory(capture())

    assert result["status"] == STATUS
    assert result["signature_count"] == 4
    assert result["series_count"] == 10
    assert result["literal_required_group_matches"]["actual_load"] == 2
    assert result["literal_required_group_matches"]["day_ahead_prices"] == 3
    assert result["literal_required_group_matches"]["generation_actual"] == 4
    assert result["literal_additional_group_matches"]["installed_generation_capacity"] == 1
    assert result["eur_per_mwh_unit_observed"] is True
    assert result["bare_eur_unit_observed"] is False
    assert result["family_coverage_proven"] is False
    assert result["model_input_authorized"] is False
    assert result["validator_databricks_request_count"] == 0


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value.update(execution={**EXECUTION, "statement_count": 2}),
            "execution differs",
        ),
        (
            lambda value: value.update(authorities={**AUTHORITIES, "model_input_authorized": True}),
            "authorities differs",
        ),
        (
            lambda value: value.update(europe_zurich_capture_date="2026-08-05"),
            "Europe/Zurich capture date differs",
        ),
        (
            lambda value: value.update(statement_id="UPPERCASE"),
            "statement_id must be a lowercase UUID",
        ),
        (
            lambda value: value["rows"][0].update(total_series_count=11),
            "total series count differs",
        ),
        (
            lambda value: value["rows"][0].update(group_name=" Actual Load"),
            "must be null or a clean string",
        ),
        (
            lambda value: value["rows"][0].update(
                min_meta_load_timestamp_utc="2026-08-07T00:00:00Z"
            ),
            "load timestamp bounds are reversed",
        ),
    ],
)
def test_capture_fails_closed(mutation, match: str) -> None:
    value = capture()
    mutation(value)

    with pytest.raises(EntsoeSeriesInventoryError, match=match):
        assess_series_inventory(value)


def test_duplicate_signature_is_rejected() -> None:
    value = capture()
    value["rows"][1] = copy.deepcopy(value["rows"][0])

    with pytest.raises(EntsoeSeriesInventoryError, match="signature is duplicated"):
        assess_series_inventory(value)


def test_missing_row_field_is_rejected() -> None:
    value = capture()
    value["rows"][0].pop("psr_type")

    with pytest.raises(EntsoeSeriesInventoryError, match="fields differ"):
        assess_series_inventory(value)


def test_bare_eur_is_exposed_not_silently_accepted() -> None:
    value = capture()
    value["rows"][1]["unit"] = "EUR"

    result = assess_series_inventory(value)

    assert result["bare_eur_unit_observed"] is True
    assert result["eur_per_mwh_unit_observed"] is False


def test_modified_query_is_rejected(tmp_path: Path) -> None:
    altered = tmp_path / "inventory.sql"
    altered.write_bytes(QUERY.read_bytes() + b"\n")

    with pytest.raises(EntsoeSeriesInventoryError, match="SHA-256 differs"):
        validate_inventory_query(altered.resolve())

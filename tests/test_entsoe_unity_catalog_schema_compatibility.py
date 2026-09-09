from __future__ import annotations

import copy

import pytest

from pfc_shaping.validation.entsoe_unity_catalog_schema_compatibility import (
    AUTHORITIES,
    CAPTURE_SCHEMA,
    STATUS,
    TABLES,
    EntsoeUnityCatalogSchemaError,
    assess_unity_catalog_schema,
)


def _column(name: str, type_text: str, nullable: bool, position: int) -> dict[str, object]:
    return {
        "name": name,
        "type_text": type_text,
        "nullable": nullable,
        "position": position,
        "comment": None,
    }


def _table(full_name: str, specs: list[tuple[str, str, bool]]) -> dict[str, object]:
    return {
        "full_name": full_name,
        "table_type": "MANAGED",
        "data_source_format": "DELTA",
        "columns": [
            _column(name, type_text, nullable, position)
            for position, (name, type_text, nullable) in enumerate(specs)
        ],
    }


def real_schema_capture() -> dict[str, object]:
    return {
        "schema_version": CAPTURE_SCHEMA,
        "evidence_class": "REAL_CONTROL_PLANE_OWNER_ASSERTED",
        "source": "UNITY_CATALOG_TABLES_GET",
        "captured_at_utc": "2026-08-06T12:00:00Z",
        "workspace_host_sha256": "a" * 64,
        "tables": [
            _table(
                TABLES[0],
                [
                    ("SeriesID", "bigint", False),
                    ("FieldName", "string", True),
                    ("GroupName", "string", True),
                    ("DocumentType", "string", True),
                    ("BusinessType", "string", True),
                    ("ProcessType", "string", True),
                    ("PsrType", "string", True),
                    ("FromZone", "string", True),
                    ("ToZone", "string", True),
                    ("Unit", "string", True),
                    ("Meta_Load_Timestamp", "timestamp", True),
                ],
            ),
            _table(
                TABLES[1],
                [
                    ("SeriesID", "bigint", False),
                    ("DateTimeUtc", "timestamp", False),
                    ("DateUtc", "date", False),
                    ("FieldValue", "double", True),
                    ("Epoch", "bigint", True),
                    ("PublicationTimestampUtc", "timestamp", True),
                    ("IsHistorical", "boolean", True),
                    ("Meta_Load_Timestamp", "timestamp", True),
                ],
            ),
            _table(
                TABLES[2],
                [
                    ("VintageID", "string", False),
                    ("SeriesID", "bigint", False),
                    ("DateTimeUtc", "timestamp", False),
                    ("DateUtc", "date", False),
                    ("FieldValue", "double", True),
                    ("Epoch", "bigint", True),
                    ("PullTimestampUtc", "timestamp", True),
                    ("PublicationTimestampUtc", "timestamp", True),
                    ("IsHistorical", "boolean", True),
                    ("Meta_Load_Timestamp", "timestamp", True),
                ],
            ),
        ],
        "execution": {
            "control_plane_get_count": 3,
            "sql_statement_count": 0,
            "warehouse_start_count": 0,
            "databricks_write_count": 0,
        },
        "authorities": dict(AUTHORITIES),
    }


def test_real_schema_reports_exact_gaps_without_authority() -> None:
    result = assess_unity_catalog_schema(real_schema_capture())

    assert result["status"] == STATUS
    assert result["column_count_by_table"] == {TABLES[0]: 11, TABLES[1]: 8, TABLES[2]: 10}
    assert result["missing_logical_fields"] == {
        TABLES[0]: ["native_resolution", "source_endpoint", "document_id"],
        TABLES[1]: ["quality_flag", "revision_number"],
        TABLES[2]: ["as_of_utc", "quality_flag", "revision_number"],
    }
    assert result["nullable_required_fields"] == {
        TABLES[0]: ["field_name", "group_name", "from_zone", "to_zone", "unit"],
        TABLES[1]: ["value", "is_historical", "meta_load_timestamp_utc"],
        TABLES[2]: ["value", "meta_load_timestamp_utc"],
    }
    assert result["type_failures"] == []
    assert result["vintage_as_of_candidates"] == ["PublicationTimestampUtc", "PullTimestampUtc"]
    assert result["sql_statement_count"] == 0
    assert result["warehouse_start_count"] == 0
    assert result["model_input_authorized"] is False


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda c: c.update(authorities={**AUTHORITIES, "model_input_authorized": True}), "authorities differs"),
        (lambda c: c.update(execution={**c["execution"], "sql_statement_count": 1}), "execution counters differ"),
        (lambda c: c["tables"].__setitem__(1, copy.deepcopy(c["tables"][0])), "table names must be exact"),
        (lambda c: c["tables"][0].update(table_type="EXTERNAL"), "table type differs"),
        (lambda c: c["tables"][0].update(data_source_format="PARQUET"), "format differs"),
        (lambda c: c["tables"][0]["columns"][0].update(type_text=1), "metadata types differ"),
        (lambda c: c["tables"][0]["columns"][0].update(comment=1), "comment type differs"),
        (lambda c: c["tables"][0]["columns"][0].update(position=1), "positions differ"),
    ],
)
def test_capture_envelope_fails_closed(mutation, match: str) -> None:
    capture = real_schema_capture()
    mutation(capture)

    with pytest.raises(EntsoeUnityCatalogSchemaError, match=match):
        assess_unity_catalog_schema(capture)


def test_physical_type_drift_is_reported_not_hidden() -> None:
    capture = real_schema_capture()
    capture["tables"][1]["columns"][3]["type_text"] = "string"

    result = assess_unity_catalog_schema(capture)

    assert result["type_failures"] == [f"{TABLES[1]}.FieldValue"]


def test_both_vintage_timestamp_candidates_are_required() -> None:
    capture = real_schema_capture()
    capture["tables"][2]["columns"] = [
        column
        for column in capture["tables"][2]["columns"]
        if column["name"] != "PullTimestampUtc"
    ]
    for position, column in enumerate(capture["tables"][2]["columns"]):
        column["position"] = position

    with pytest.raises(EntsoeUnityCatalogSchemaError, match="both vintage PIT timestamp candidates"):
        assess_unity_catalog_schema(capture)

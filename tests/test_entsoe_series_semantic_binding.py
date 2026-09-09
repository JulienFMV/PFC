from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.validation.entsoe_normalized_quality import ROLE_FIELDS
from pfc_shaping.validation.entsoe_series_family_mapping import (
    MAPPING_SCHEMA,
    assess_series_family_mapping,
)
from pfc_shaping.validation.entsoe_series_inventory import (
    AUTHORITIES as INVENTORY_AUTHORITIES,
)
from pfc_shaping.validation.entsoe_series_inventory import (
    EXECUTION,
    INVENTORY_SCHEMA,
    QUERY_SHA256,
    ROW_FIELDS_IN_ORDER,
    assess_series_inventory,
    series_signature_id,
)
from pfc_shaping.validation.entsoe_series_semantic_binding import (
    AUTHORITIES,
    BINDING_SCHEMA,
    EntsoeSeriesSemanticBindingError,
    assess_series_semantic_binding,
    dimension_content_id,
)


def _row(
    group: str,
    field: str,
    *,
    document_type: str,
    series_count: int,
    unit: str = "MW",
    from_zone: str = "CH",
    to_zone: str | None = None,
) -> dict[str, object]:
    return {
        "group_name": group,
        "field_name": field,
        "document_type": document_type,
        "business_type": "A04",
        "process_type": "A16",
        "psr_type": None,
        "from_zone": from_zone,
        "to_zone": to_zone,
        "unit": unit,
        "series_count": series_count,
        "min_meta_load_timestamp_utc": "2026-07-28T00:00:00Z",
        "max_meta_load_timestamp_utc": "2026-08-06T11:00:00Z",
        "total_signature_count": 3,
        "total_series_count": 4,
    }


def inventory() -> dict[str, object]:
    rows = [
        _row("Actual Load", "load", document_type="A65", series_count=2),
        _row("Generation Actual", "solar", document_type="A75", series_count=1),
        _row(
            "Physical Flow",
            "flow",
            document_type="A11",
            series_count=1,
            from_zone="CH",
            to_zone="DE",
        ),
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
        "authorities": dict(INVENTORY_AUTHORITIES),
    }


def _mapped_entry(
    row: dict[str, object],
    family: str,
    *,
    field: str | None = None,
    resolution: str = "PT1H",
    border: str | None = None,
    direction: str | None = None,
    sign: str = "NON_NEGATIVE_PHYSICAL_QUANTITY",
) -> dict[str, object]:
    return {
        "signature_id": series_signature_id(row),
        "disposition": "MAPPED",
        "logical_family": family,
        "logical_field": field,
        "border": border,
        "direction_coverage": direction,
        "canonical_unit": "MW",
        "native_resolution": resolution,
        "sign_semantics": sign,
        "source_semantics_confirmed": True,
        "reason": "Synthetic exact mapping.",
    }


def mapping(source: dict[str, object]) -> dict[str, object]:
    rows = source["rows"]
    assert isinstance(rows, list)
    return {
        "schema_version": MAPPING_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "inventory_content_id": assess_series_inventory(source)[
            "inventory_content_id"
        ],
        "mapping_owner": "Synthetic family owner",
        "mapped_at_utc": "2026-08-06T12:00:00Z",
        "entries": [
            _mapped_entry(rows[0], "actual_load"),
            _mapped_entry(rows[1], "generation_actual", field="solar"),
            _mapped_entry(
                rows[2],
                "crossborder_physical_flows",
                resolution="PT15M",
                border="CH-DE",
                direction="CH_TO_NEIGHBOR_ONLY",
                sign="SIGNED_POSITIVE_CH_TO_NEIGHBOR",
            ),
        ],
        "authorities": {
            "value_quality": False,
            "point_in_time_availability": False,
            "model_input": False,
            "production": False,
        },
    }


def dimension() -> pd.DataFrame:
    records = [
        ("load-a", "actual_load", "load", "CH", "CH", "MW", "PT1H"),
        ("load-b", "actual_load", "load", "CH", "CH", "MW", "PT1H"),
        (
            "solar-a",
            "generation_actual",
            "solar",
            "CH",
            "CH",
            "MW",
            "PT1H",
        ),
        (
            "flow-de",
            "crossborder_physical_flows",
            "flow",
            "CH",
            "DE",
            "MW",
            "PT15M",
        ),
    ]
    rows = [
        {
            "series_id": series_id,
            "group_name": group,
            "field_name": field,
            "from_zone": from_zone,
            "to_zone": to_zone,
            "unit": unit,
            "native_resolution": resolution,
            "source_endpoint": "https://example.test/entsoe",
            "document_id": f"SYNTHETIC-{series_id}",
        }
        for series_id, group, field, from_zone, to_zone, unit, resolution in records
    ]
    return pd.DataFrame(rows, columns=ROLE_FIELDS["series_dimension"])


def binding_document(
    base: pd.DataFrame,
    source: dict[str, object],
    family_mapping: dict[str, object],
) -> dict[str, object]:
    source_rows = source["rows"]
    assert isinstance(source_rows, list)
    raw_by_series = {
        "load-a": source_rows[0],
        "load-b": source_rows[0],
        "solar-a": source_rows[1],
        "flow-de": source_rows[2],
    }
    entries = []
    for series_id in base["series_id"]:
        raw = raw_by_series[str(series_id)]
        semantics = {
            key: raw[key]
            for key in (
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
        }
        entries.append(
            {
                "series_id": series_id,
                "signature_id": series_signature_id(semantics),
                "source_semantics": semantics,
                "reason": "Synthetic physical-series semantic binding.",
            }
        )
    return {
        "schema_version": BINDING_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "base_dimension_content_id": dimension_content_id(base),
        "inventory_content_id": assess_series_inventory(source)[
            "inventory_content_id"
        ],
        "family_mapping_content_id": assess_series_family_mapping(
            source, family_mapping
        )["mapping_content_id"],
        "binding_owner": "Synthetic semantic owner",
        "bound_at_utc": "2026-08-06T12:30:00Z",
        "entries": entries,
        "authorities": dict(AUTHORITIES),
    }


def _inputs():
    source = inventory()
    family_mapping = mapping(source)
    base = dimension()
    bindings = binding_document(base, source, family_mapping)
    return base, source, family_mapping, bindings


def test_exact_series_to_signature_binding_preserves_zero_authority() -> None:
    base, source, family_mapping, bindings = _inputs()

    result = assess_series_semantic_binding(base, source, family_mapping, bindings)

    assert result["physical_series_count"] == 4
    assert result["mapped_signature_count"] == 3
    assert result["physical_series_count_by_logical_family"] == {
        "actual_load": 2,
        "crossborder_physical_flows": 1,
        "generation_actual": 1,
    }
    assert result["all_base_series_bound_once"] is True
    assert result["all_mapped_signature_cardinalities_match"] is True
    assert result["temporal_zone_binding_proven"] is False
    assert result["effective_cadence_package_binding_proven"] is False
    assert result["real_value_rows_opened"] == 0
    assert result["series_semantic_binding_verified"] is False
    assert result["model_input_authorized"] is False
    assert result["validator_databricks_request_count"] == 0


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("base_dimension_content_id", "base dimension content ID differs"),
        ("inventory_content_id", "inventory content ID differs"),
        ("family_mapping_content_id", "family mapping content ID differs"),
    ),
)
def test_binding_is_pinned_to_all_inputs(field: str, message: str) -> None:
    base, source, family_mapping, bindings = _inputs()
    bindings[field] = "f" * 64

    with pytest.raises(EntsoeSeriesSemanticBindingError, match=message):
        assess_series_semantic_binding(base, source, family_mapping, bindings)


def test_missing_or_duplicate_physical_series_binding_is_rejected() -> None:
    base, source, family_mapping, bindings = _inputs()
    bindings["entries"].pop()
    with pytest.raises(EntsoeSeriesSemanticBindingError, match="coverage differs"):
        assess_series_semantic_binding(base, source, family_mapping, bindings)

    base, source, family_mapping, bindings = _inputs()
    bindings["entries"][1]["series_id"] = "load-a"
    with pytest.raises(EntsoeSeriesSemanticBindingError, match="duplicated"):
        assess_series_semantic_binding(base, source, family_mapping, bindings)


def test_raw_semantics_must_hash_to_declared_signature() -> None:
    base, source, family_mapping, bindings = _inputs()
    bindings["entries"][0]["source_semantics"]["document_type"] = "A99"

    with pytest.raises(EntsoeSeriesSemanticBindingError, match="computed signature ID"):
        assess_series_semantic_binding(base, source, family_mapping, bindings)


def test_normalized_family_and_direction_must_match_owner_mapping() -> None:
    base, source, family_mapping, _ = _inputs()
    base.loc[base["series_id"].eq("solar-a"), "group_name"] = "actual_load"
    bindings = binding_document(base, source, family_mapping)
    with pytest.raises(EntsoeSeriesSemanticBindingError, match="logical family differs"):
        assess_series_semantic_binding(base, source, family_mapping, bindings)

    base, source, family_mapping, _ = _inputs()
    base.loc[base["series_id"].eq("flow-de"), ["from_zone", "to_zone"]] = [
        "DE",
        "CH",
    ]
    bindings = binding_document(base, source, family_mapping)
    with pytest.raises(EntsoeSeriesSemanticBindingError, match="border direction differs"):
        assess_series_semantic_binding(base, source, family_mapping, bindings)


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    [
        ("unit", "kW", "normalized unit differs"),
        ("native_resolution", "PT15M", "declared current resolution differs"),
    ],
)
def test_normalized_unit_and_resolution_must_match_owner_mapping(
    column: str, replacement: str, message: str
) -> None:
    base, source, family_mapping, _ = _inputs()
    base.loc[base["series_id"].eq("load-a"), column] = replacement
    bindings = binding_document(base, source, family_mapping)

    with pytest.raises(EntsoeSeriesSemanticBindingError, match=message):
        assess_series_semantic_binding(base, source, family_mapping, bindings)


def test_inventory_series_count_must_equal_bound_physical_series_count() -> None:
    base, source, family_mapping, _ = _inputs()
    source["rows"][0]["series_count"] = 1
    for row in source["rows"]:
        row["total_series_count"] = 3
    family_mapping = mapping(source)
    bindings = binding_document(base, source, family_mapping)

    with pytest.raises(EntsoeSeriesSemanticBindingError, match="cardinality differs"):
        assess_series_semantic_binding(base, source, family_mapping, bindings)


def test_non_mapped_signature_cannot_enter_base_dimension() -> None:
    base, source, family_mapping, _ = _inputs()
    flow = family_mapping["entries"][2]
    flow.update(
        {
            "disposition": "AMBIGUOUS",
            "logical_family": None,
            "logical_field": None,
            "border": None,
            "direction_coverage": None,
            "canonical_unit": None,
            "native_resolution": None,
            "sign_semantics": None,
            "source_semantics_confirmed": False,
        }
    )
    bindings = binding_document(base, source, family_mapping)

    with pytest.raises(EntsoeSeriesSemanticBindingError, match="not owner-mapped"):
        assess_series_semantic_binding(base, source, family_mapping, bindings)


def test_casefold_duplicate_dimension_ids_and_value_columns_are_rejected() -> None:
    base, source, family_mapping, _ = _inputs()
    base.loc[1, "series_id"] = "LOAD-A"
    with pytest.raises(EntsoeSeriesSemanticBindingError, match="after casefold"):
        dimension_content_id(base)

    base = dimension()
    base["value"] = 1.0
    with pytest.raises(EntsoeSeriesSemanticBindingError, match="fields differ"):
        dimension_content_id(base)


def test_owner_assertion_never_grants_real_or_model_authority() -> None:
    base, source, family_mapping, bindings = _inputs()
    family_mapping["evidence_class"] = "REAL_OWNER_REVIEWED"
    family_mapping["mapping_owner"] = "family.owner@example.org"
    bindings = binding_document(base, source, family_mapping)
    bindings["evidence_class"] = "REAL_OWNER_ASSERTED"
    bindings["binding_owner"] = "binding.owner@example.org"

    result = assess_series_semantic_binding(base, source, family_mapping, bindings)

    assert result["evidence_class"] == "REAL_OWNER_ASSERTED"
    assert result["real_series_inventory_coverage_proven"] is False
    assert result["series_semantic_binding_verified"] is False
    assert result["point_in_time_availability_proven"] is False
    assert result["model_input_authorized"] is False


def test_authority_escalation_and_binding_time_are_rejected() -> None:
    base, source, family_mapping, bindings = _inputs()
    bindings["authorities"]["model_input_authorized"] = True
    with pytest.raises(EntsoeSeriesSemanticBindingError, match="authorities differs"):
        assess_series_semantic_binding(base, source, family_mapping, bindings)

    base, source, family_mapping, bindings = _inputs()
    bindings["bound_at_utc"] = "2026-08-06T11:59:59Z"
    with pytest.raises(EntsoeSeriesSemanticBindingError, match="predates"):
        assess_series_semantic_binding(base, source, family_mapping, bindings)


def test_engineer_template_is_value_free_and_deliberately_non_admissible() -> None:
    template_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "data"
        / "templates"
        / "entsoe_series_semantic_binding.v1.json.template"
    )
    document = json.loads(template_path.read_text(encoding="utf-8"))

    assert document["schema_version"] == BINDING_SCHEMA
    assert document["authorities"] == AUTHORITIES
    assert "value" not in template_path.read_text(encoding="utf-8").casefold()
    assert "token" not in template_path.read_text(encoding="utf-8").casefold()
    assert document["entries"][0]["series_id"].startswith("<")

    base, source, family_mapping, _ = _inputs()
    with pytest.raises(EntsoeSeriesSemanticBindingError):
        assess_series_semantic_binding(base, source, family_mapping, document)

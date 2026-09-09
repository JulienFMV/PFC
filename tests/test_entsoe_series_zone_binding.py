from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

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
from pfc_shaping.validation.entsoe_series_zone_binding import (
    AUTHORITIES,
    BINDING_SCHEMA,
    EntsoeSeriesZoneBindingError,
    assess_series_zone_bindings,
)
from pfc_shaping.validation.entsoe_zone_configuration import (
    AUTHORITIES as ZONE_AUTHORITIES,
)
from pfc_shaping.validation.entsoe_zone_configuration import (
    REGISTRY_SCHEMA,
    assess_zone_configuration,
)
from pfc_shaping.validation.entsoe_zone_configuration import (
    canonical_sha256 as zone_sha256,
)

ZONES = ("CH", "DE_LU", "FR", "IT_NORTH", "AT")
START = "2023-01-01T00:00:00Z"
END = "2026-08-06T00:00:00Z"
ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = ROOT / "docs" / "data" / "templates"


def _eic(seed: int) -> str:
    return f"{seed % 100:02d}Y{seed:012d}A"


def _inventory_row(
    group: str,
    field: str,
    zone: str,
    *,
    unit: str,
    document_type: str,
    to_zone: str | None = None,
) -> dict[str, object]:
    return {
        "group_name": group,
        "field_name": field,
        "document_type": document_type,
        "business_type": "A04",
        "process_type": "A16",
        "psr_type": None,
        "from_zone": zone,
        "to_zone": to_zone,
        "unit": unit,
        "series_count": 1,
        "min_meta_load_timestamp_utc": "2026-07-28T00:00:00Z",
        "max_meta_load_timestamp_utc": "2026-08-06T07:00:00Z",
        "total_signature_count": 7,
        "total_series_count": 7,
    }


def inventory() -> dict[str, object]:
    rows = [
        _inventory_row(
            "Day Ahead Prices",
            "price",
            zone,
            unit="EUR",
            document_type=f"PRICE-{position}",
        )
        for position, zone in enumerate(ZONES, start=1)
    ]
    rows.extend(
        [
            _inventory_row("Actual Load", "load", "CH", unit="MW", document_type="LOAD"),
            _inventory_row(
                "Physical Flow",
                "flow",
                "CH",
                unit="MW",
                document_type="FLOW",
                to_zone="DE_LU",
            ),
        ]
    )
    return {
        "schema_version": INVENTORY_SCHEMA,
        "evidence_class": "REAL_DIMENSION_INVENTORY_OWNER_ASSERTED",
        "query_sha256": QUERY_SHA256,
        "statement_id": "123e4567-e89b-12d3-a456-426614174000",
        "captured_at_utc": "2026-08-06T08:00:00Z",
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


def _family_entry(
    row: dict[str, object],
    family: str,
    *,
    border: str | None = None,
    direction: str | None = None,
    sign: str,
    unit: str,
) -> dict[str, object]:
    return {
        "signature_id": series_signature_id(row),
        "disposition": "MAPPED",
        "logical_family": family,
        "logical_field": None,
        "border": border,
        "direction_coverage": direction,
        "canonical_unit": unit,
        "native_resolution": "PT1H",
        "sign_semantics": sign,
        "source_semantics_confirmed": True,
        "reason": "Synthetic exact family mapping.",
    }


def family_mapping(source: dict[str, object]) -> dict[str, object]:
    rows = source["rows"]
    entries = [
        _family_entry(
            row,
            "day_ahead_prices",
            sign="PRICE_CAN_BE_NEGATIVE",
            unit="EUR/MWh",
        )
        for row in rows[:5]
    ]
    entries.append(
        _family_entry(
            rows[5],
            "actual_load",
            sign="NON_NEGATIVE_PHYSICAL_QUANTITY",
            unit="MW",
        )
    )
    entries.append(
        _family_entry(
            rows[6],
            "crossborder_physical_flows",
            border="CH-DE",
            direction="CH_TO_NEIGHBOR_ONLY",
            sign="SIGNED_POSITIVE_CH_TO_NEIGHBOR",
            unit="MW",
        )
    )
    inventory_id = assess_series_inventory(source)["inventory_content_id"]
    return {
        "schema_version": MAPPING_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "inventory_content_id": inventory_id,
        "mapping_owner": "Synthetic family owner",
        "mapped_at_utc": "2026-08-06T08:30:00Z",
        "entries": entries,
        "authorities": {
            "value_quality": False,
            "point_in_time_availability": False,
            "model_input": False,
            "production": False,
        },
    }


def _zone_entry(
    identifier: str,
    zone: str,
    scheme: str,
    *,
    start: str = START,
    end: str = END,
) -> dict[str, object]:
    return {
        "source_identifier_scheme": scheme,
        "source_identifier": identifier,
        "logical_zone": zone,
        "domain_kind": "BIDDING_ZONE",
        "valid_from_utc": start,
        "valid_to_utc": end,
        "source_status": "ACTIVE",
        "source_document_id_kind": "GOVERNED_EIC_REGISTRY_SNAPSHOT_ID",
        "source_document_id": f"SYNTHETIC-{scheme}-{zone}-{start}",
        "source_endpoint": "https://example.invalid/governed-zone-registry",
        "owner_semantics_confirmed": True,
        "reason": "Synthetic exact zone registry entry.",
    }


def zone_registry() -> dict[str, object]:
    entries = [
        _zone_entry(_eic(position), zone, "EIC_Y_CODE")
        for position, zone in enumerate(ZONES, start=1)
    ]
    entries.extend(
        _zone_entry(zone, zone, "OWNER_CANONICAL_LABEL") for zone in ZONES
    )
    return {
        "schema_version": REGISTRY_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "registry_owner": "Synthetic zone owner",
        "registered_at_utc": "2026-08-06T08:45:00Z",
        "coverage_window_start_utc": START,
        "coverage_window_end_utc": END,
        "entries": entries,
        "authorities": dict(ZONE_AUTHORITIES),
    }


def binding_document(
    source: dict[str, object],
    mapping: dict[str, object],
    registry: dict[str, object],
) -> dict[str, object]:
    rows = source["rows"]
    label_entries = {
        entry["logical_zone"]: entry
        for entry in registry["entries"]
        if entry["source_identifier_scheme"] == "OWNER_CANONICAL_LABEL"
    }
    bound_rows = rows[:6]
    bindings = []
    for row in bound_rows:
        zone = str(row["from_zone"])
        bindings.append(
            {
                "signature_id": series_signature_id(row),
                "source_zone_field": "from_zone",
                "source_identifier_scheme": "OWNER_CANONICAL_LABEL",
                "domain_kind": "BIDDING_ZONE",
                "series_valid_from_utc": START,
                "series_valid_to_utc": END,
                "expected_logical_zone": zone,
                "zone_entry_id": zone_sha256(label_entries[zone]),
                "reason": "Synthetic series-to-zone temporal binding.",
            }
        )
    return {
        "schema_version": BINDING_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "inventory_content_id": assess_series_inventory(source)["inventory_content_id"],
        "family_mapping_content_id": assess_series_family_mapping(source, mapping)[
            "mapping_content_id"
        ],
        "zone_registry_content_id": assess_zone_configuration(registry)[
            "registry_content_id"
        ],
        "binding_owner": "Synthetic binding owner",
        "bound_at_utc": "2026-08-06T09:00:00Z",
        "bindings": bindings,
        "authorities": dict(AUTHORITIES),
    }


def _inputs():
    source = inventory()
    mapping = family_mapping(source)
    registry = zone_registry()
    bindings = binding_document(source, mapping, registry)
    return source, mapping, registry, bindings


def test_exact_temporal_bindings_cover_all_five_day_ahead_price_zones() -> None:
    source, mapping, registry, bindings = _inputs()

    result = assess_series_zone_bindings(source, mapping, registry, bindings)

    assert result["binding_entry_count"] == 6
    assert result["zone_scoped_mapped_signature_count"] == 6
    assert result["missing_binding_signature_ids"] == []
    assert result[
        "synthetic_or_owner_asserted_day_ahead_price_window_complete"
    ] is True
    assert result[
        "synthetic_or_owner_asserted_all_coupled_requirements_complete"
    ] is False
    assert result["missing_intervals_by_family_and_logical_zone"]["actual_load"][
        "CH"
    ] == []
    assert result["missing_intervals_by_family_and_logical_zone"]["actual_load"][
        "FR"
    ] == [{"start_utc": START, "end_utc": END}]
    assert result["series_zone_binding_verified"] is False
    assert result["model_input_authorized"] is False
    assert result["validator_databricks_request_count"] == 0


def test_directional_series_cannot_be_smuggled_into_zone_binding() -> None:
    source, mapping, registry, bindings = _inputs()
    flow = source["rows"][6]
    bindings["bindings"].append(
        {
            **copy.deepcopy(bindings["bindings"][0]),
            "signature_id": series_signature_id(flow),
            "zone_entry_id": bindings["bindings"][0]["zone_entry_id"],
            "reason": "Attempted directional shortcut.",
        }
    )

    with pytest.raises(EntsoeSeriesZoneBindingError, match="border mapping"):
        assess_series_zone_bindings(source, mapping, registry, bindings)


def test_selected_raw_zone_field_must_be_populated_and_resolvable() -> None:
    source, mapping, registry, bindings = _inputs()
    bindings["bindings"][0]["source_zone_field"] = "to_zone"

    with pytest.raises(EntsoeSeriesZoneBindingError, match="null or invalid"):
        assess_series_zone_bindings(source, mapping, registry, bindings)


def test_zone_entry_id_and_expected_zone_are_content_bound() -> None:
    source, mapping, registry, bindings = _inputs()
    bindings["bindings"][0]["zone_entry_id"] = "f" * 64
    with pytest.raises(EntsoeSeriesZoneBindingError, match="zone entry ID differs"):
        assess_series_zone_bindings(source, mapping, registry, bindings)

    source, mapping, registry, bindings = _inputs()
    bindings["bindings"][0]["expected_logical_zone"] = "FR"
    with pytest.raises(EntsoeSeriesZoneBindingError, match="logical zone differs"):
        assess_series_zone_bindings(source, mapping, registry, bindings)


@pytest.mark.parametrize(
    ("content_id_field", "expected_message"),
    (
        ("inventory_content_id", "inventory content ID differs"),
        ("family_mapping_content_id", "family mapping content ID differs"),
        ("zone_registry_content_id", "zone registry content ID differs"),
    ),
)
def test_binding_is_pinned_to_all_three_input_artifacts(
    content_id_field: str,
    expected_message: str,
) -> None:
    source, mapping, registry, bindings = _inputs()
    bindings[content_id_field] = "f" * 64

    with pytest.raises(EntsoeSeriesZoneBindingError, match=expected_message):
        assess_series_zone_bindings(source, mapping, registry, bindings)


def test_missing_binding_is_reported_not_inferred_from_raw_label() -> None:
    source, mapping, registry, bindings = _inputs()
    removed = bindings["bindings"].pop()

    result = assess_series_zone_bindings(source, mapping, registry, bindings)

    assert result["missing_binding_signature_ids"] == [removed["signature_id"]]


def test_binding_cannot_cross_registry_code_or_alias_change() -> None:
    source, mapping, registry, bindings = _inputs()
    old = next(
        entry
        for entry in registry["entries"]
        if entry["source_identifier_scheme"] == "OWNER_CANONICAL_LABEL"
        and entry["logical_zone"] == "CH"
    )
    old["valid_to_utc"] = "2025-01-01T00:00:00Z"
    replacement = copy.deepcopy(old)
    replacement["valid_from_utc"] = "2025-01-01T00:00:00Z"
    replacement["valid_to_utc"] = END
    replacement["source_document_id"] = "SYNTHETIC-OWNER-CH-REPLACEMENT"
    registry["entries"].append(replacement)
    bindings = binding_document(source, mapping, registry)

    with pytest.raises(EntsoeSeriesZoneBindingError, match="does not resolve"):
        assess_series_zone_bindings(source, mapping, registry, bindings)


def test_overlapping_intervals_for_one_signature_are_rejected() -> None:
    source, mapping, registry, bindings = _inputs()
    duplicate = copy.deepcopy(bindings["bindings"][0])
    duplicate["series_valid_from_utc"] = "2024-01-01T00:00:00Z"
    duplicate["reason"] = "Overlapping binding interval."
    bindings["bindings"].append(duplicate)

    with pytest.raises(EntsoeSeriesZoneBindingError, match="overlapping"):
        assess_series_zone_bindings(source, mapping, registry, bindings)


def test_real_owner_assertions_never_verify_zone_or_fact_coverage() -> None:
    source, mapping, registry, bindings = _inputs()
    mapping["evidence_class"] = "REAL_OWNER_REVIEWED"
    mapping["mapping_owner"] = "family.owner@example.org"
    registry["evidence_class"] = "REAL_OWNER_ASSERTED"
    registry["registry_owner"] = "zone.owner@example.org"
    bindings = binding_document(source, mapping, registry)
    bindings["evidence_class"] = "REAL_OWNER_ASSERTED"
    bindings["binding_owner"] = "binding.owner@example.org"

    result = assess_series_zone_bindings(source, mapping, registry, bindings)

    assert result["owner_asserted_day_ahead_price_window_complete"] is True
    assert result["official_zone_registry_verified"] is False
    assert result["real_fact_delivery_coverage_proven"] is False
    assert result["series_zone_binding_verified"] is False
    assert result["model_input_authorized"] is False


def test_evidence_classes_and_binding_time_cannot_be_self_declared() -> None:
    source, mapping, registry, bindings = _inputs()
    bindings["evidence_class"] = "REAL_OWNER_ASSERTED"
    bindings["binding_owner"] = "binding.owner@example.org"
    with pytest.raises(EntsoeSeriesZoneBindingError, match="requires owner-asserted"):
        assess_series_zone_bindings(source, mapping, registry, bindings)

    source, mapping, registry, bindings = _inputs()
    bindings["bound_at_utc"] = "2026-08-06T08:40:00Z"
    with pytest.raises(EntsoeSeriesZoneBindingError, match="predates the zone registry"):
        assess_series_zone_bindings(source, mapping, registry, bindings)


def test_authority_escalation_is_rejected() -> None:
    source, mapping, registry, bindings = _inputs()
    bindings["authorities"]["model_input_authorized"] = True

    with pytest.raises(EntsoeSeriesZoneBindingError, match="authorities differs"):
        assess_series_zone_bindings(source, mapping, registry, bindings)


def test_data_engineer_templates_are_value_free_and_non_admissible() -> None:
    registry_raw = (
        TEMPLATE_ROOT / "entsoe_zone_configuration_registry.v1.json.template"
    ).read_text(encoding="utf-8")
    binding_raw = (
        TEMPLATE_ROOT / "entsoe_series_zone_binding.v1.json.template"
    ).read_text(encoding="utf-8")
    registry = json.loads(registry_raw)
    bindings = json.loads(binding_raw)

    assert "DATABRICKS" not in registry_raw + binding_raw
    assert "TOKEN" not in registry_raw + binding_raw
    assert all(value is False for value in registry["authorities"].values())
    assert all(value is False for value in bindings["authorities"].values())
    assert all(
        "<" in entry["source_identifier"]
        for entry in registry["entries"]
    )
    assert not any(
        re.fullmatch(r"[0-9A-Z-]{16}", entry["source_identifier"])
        for entry in registry["entries"]
    )
    with pytest.raises(ValueError):
        assess_zone_configuration(registry)

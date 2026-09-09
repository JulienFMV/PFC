from __future__ import annotations

import copy

import pytest

from pfc_shaping.validation.entsoe_series_family_mapping import (
    MAPPING_SCHEMA,
    STATUS,
    EntsoeSeriesFamilyMappingError,
    assess_series_family_mapping,
)
from pfc_shaping.validation.entsoe_series_inventory import (
    AUTHORITIES,
    EXECUTION,
    INVENTORY_SCHEMA,
    QUERY_SHA256,
    ROW_FIELDS_IN_ORDER,
    canonical_sha256,
    series_signature_id,
)


def _row(
    group_name: str,
    field_name: str,
    *,
    document_type: str,
    unit: str,
    from_zone: str = "CH",
    to_zone: str | None = None,
    series_count: int = 1,
) -> dict[str, object]:
    return {
        "group_name": group_name,
        "field_name": field_name,
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
        "total_signature_count": 4,
        "total_series_count": 10,
    }


def inventory() -> dict[str, object]:
    rows = [
        _row("Actual Load", "load", document_type="A65", unit="MW", series_count=2),
        _row(
            "Day Ahead Prices",
            "price",
            document_type="A44",
            unit="EUR",
            series_count=3,
        ),
        _row(
            "Generation Actual",
            "solar",
            document_type="A75",
            unit="MW",
            series_count=4,
        ),
        _row(
            "Physical Flow",
            "flow",
            document_type="A11",
            unit="MW",
            from_zone="CH",
            to_zone="DE",
            series_count=1,
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
        "authorities": dict(AUTHORITIES),
    }


def _entry(
    row: dict[str, object],
    family: str,
    *,
    logical_field: str | None = None,
    border: str | None = None,
    direction_coverage: str | None = None,
    canonical_unit: str = "MW",
    resolution: str = "PT1H",
    sign: str = "NON_NEGATIVE_PHYSICAL_QUANTITY",
) -> dict[str, object]:
    return {
        "signature_id": series_signature_id(row),
        "disposition": "MAPPED",
        "logical_family": family,
        "logical_field": logical_field,
        "border": border,
        "direction_coverage": direction_coverage,
        "canonical_unit": canonical_unit,
        "native_resolution": resolution,
        "sign_semantics": sign,
        "source_semantics_confirmed": True,
        "reason": "Owner-confirmed exact semantic mapping.",
    }


def mapping(value: dict[str, object]) -> dict[str, object]:
    rows = value["rows"]
    assert isinstance(rows, list)
    entries = [
        _entry(rows[0], "actual_load"),
        _entry(
            rows[1],
            "day_ahead_prices",
            canonical_unit="EUR/MWh",
            sign="PRICE_CAN_BE_NEGATIVE",
        ),
        _entry(rows[2], "generation_actual", logical_field="solar"),
        _entry(
            rows[3],
            "crossborder_physical_flows",
            border="CH-DE",
            direction_coverage="CH_TO_NEIGHBOR_ONLY",
            resolution="PT15M",
            sign="SIGNED_POSITIVE_CH_TO_NEIGHBOR",
        ),
    ]
    return {
        "schema_version": MAPPING_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "inventory_content_id": canonical_sha256(value),
        "mapping_owner": "Synthetic fixture owner",
        "mapped_at_utc": "2026-08-06T12:00:00Z",
        "entries": entries,
        "authorities": {
            "value_quality": False,
            "point_in_time_availability": False,
            "model_input": False,
            "production": False,
        },
    }


def test_exact_owner_mapping_reports_coverage_gaps_without_authority() -> None:
    source = inventory()
    result = assess_series_family_mapping(source, mapping(source))

    assert result["status"] == STATUS
    assert result["signature_count"] == 4
    assert result["mapped_signature_count"] == 4
    assert result["unmapped_signature_count"] == 0
    assert result["mapped_series_count_by_family"]["day_ahead_prices"] == 3
    assert result["mapped_fields_by_family"]["generation_actual"] == ["solar"]
    assert result["missing_generation_actual_fields"] == [
        "hydro_pumped_storage",
        "hydro_reservoir",
        "hydro_run_of_river",
        "nuclear",
        "wind",
    ]
    assert result["missing_borders_by_crossborder_family"][
        "crossborder_physical_flows"
    ] == ["CH-AT", "CH-FR", "CH-IT"]
    assert result["missing_directions_by_crossborder_family_and_border"][
        "crossborder_physical_flows"
    ]["CH-DE"] == ["NEIGHBOR_TO_CH_ONLY"]
    assert result["day_ahead_price_unit_mismatch_signature_ids"] == []
    assert result["all_required_coverage_mapped"] is False
    assert result["family_mapping_verified"] is False
    assert result["external_mapping_owner_identity_verified"] is False
    assert result["model_input_authorized"] is False
    assert result["validator_databricks_request_count"] == 0


def test_raw_bare_eur_can_only_be_normalized_by_explicit_owner_mapping() -> None:
    source = inventory()
    document = mapping(source)
    price_entry = document["entries"][1]
    assert isinstance(price_entry, dict)
    price_entry["canonical_unit"] = "EUR"

    with pytest.raises(EntsoeSeriesFamilyMappingError, match="canonical unit is invalid"):
        assess_series_family_mapping(source, document)


def test_day_ahead_price_mapping_exposes_non_eur_per_mwh_unit() -> None:
    source = inventory()
    document = mapping(source)
    price_entry = document["entries"][1]
    assert isinstance(price_entry, dict)
    price_entry["canonical_unit"] = "MW"

    result = assess_series_family_mapping(source, document)

    assert result["day_ahead_price_unit_mismatch_signature_ids"] == [
        price_entry["signature_id"]
    ]
    assert result["all_required_coverage_mapped"] is False


def test_crossborder_mapping_requires_border_and_documented_sign() -> None:
    source = inventory()
    document = mapping(source)
    flow_entry = document["entries"][3]
    assert isinstance(flow_entry, dict)
    flow_entry["border"] = None

    with pytest.raises(EntsoeSeriesFamilyMappingError, match="canonical CH border"):
        assess_series_family_mapping(source, document)


def test_crossborder_direction_must_match_positive_sign_semantics() -> None:
    source = inventory()
    document = mapping(source)
    flow_entry = document["entries"][3]
    assert isinstance(flow_entry, dict)
    flow_entry["direction_coverage"] = "NEIGHBOR_TO_CH_ONLY"

    with pytest.raises(EntsoeSeriesFamilyMappingError, match="direction and sign"):
        assess_series_family_mapping(source, document)


def test_one_signed_physical_flow_may_explicitly_cover_both_directions() -> None:
    source = inventory()
    document = mapping(source)
    flow_entry = document["entries"][3]
    assert isinstance(flow_entry, dict)
    flow_entry["direction_coverage"] = "BOTH_DIRECTIONS_IN_ONE_SIGNED_SERIES"

    result = assess_series_family_mapping(source, document)

    assert result["missing_directions_by_crossborder_family_and_border"][
        "crossborder_physical_flows"
    ]["CH-DE"] == []


def test_ntc_cannot_claim_both_directions_from_one_series() -> None:
    source = inventory()
    document = mapping(source)
    flow_entry = document["entries"][3]
    assert isinstance(flow_entry, dict)
    flow_entry["logical_family"] = "ntc_day_ahead"
    flow_entry["direction_coverage"] = "BOTH_DIRECTIONS_IN_ONE_SIGNED_SERIES"

    with pytest.raises(EntsoeSeriesFamilyMappingError, match="cannot claim both"):
        assess_series_family_mapping(source, document)


def test_mapping_is_bound_to_exact_inventory_content() -> None:
    source = inventory()
    document = mapping(source)
    document["inventory_content_id"] = "f" * 64

    with pytest.raises(EntsoeSeriesFamilyMappingError, match="inventory content ID differs"):
        assess_series_family_mapping(source, document)


def test_mapping_cannot_predate_the_inventory_capture() -> None:
    source = inventory()
    document = mapping(source)
    document["mapped_at_utc"] = "2026-08-06T11:29:59Z"

    with pytest.raises(EntsoeSeriesFamilyMappingError, match="precedes"):
        assess_series_family_mapping(source, document)


def test_duplicate_and_unknown_signature_ids_are_rejected() -> None:
    source = inventory()
    duplicate = mapping(source)
    duplicate["entries"].append(copy.deepcopy(duplicate["entries"][0]))
    with pytest.raises(EntsoeSeriesFamilyMappingError, match="signature ID is duplicated"):
        assess_series_family_mapping(source, duplicate)

    unknown = mapping(source)
    unknown["entries"][0]["signature_id"] = "f" * 64
    with pytest.raises(EntsoeSeriesFamilyMappingError, match="unknown signature ID"):
        assess_series_family_mapping(source, unknown)


def test_partial_mapping_reports_unmapped_exact_ids() -> None:
    source = inventory()
    document = mapping(source)
    removed = document["entries"].pop()

    result = assess_series_family_mapping(source, document)

    assert result["unmapped_signature_count"] == 1
    assert result["unmapped_signature_ids"] == [removed["signature_id"]]


def test_ambiguous_disposition_cannot_smuggle_normalized_semantics() -> None:
    source = inventory()
    document = mapping(source)
    entry = document["entries"][0]
    entry["disposition"] = "AMBIGUOUS"

    with pytest.raises(EntsoeSeriesFamilyMappingError, match="normalized fields null"):
        assess_series_family_mapping(source, document)


def test_authority_escalation_is_rejected() -> None:
    source = inventory()
    document = mapping(source)
    document["evidence_class"] = "REAL_OWNER_REVIEWED"
    document["mapping_owner"] = "data.engineer@example.org"
    document["authorities"]["model_input"] = True

    with pytest.raises(EntsoeSeriesFamilyMappingError, match="authorities must be exact"):
        assess_series_family_mapping(source, document)


def test_real_owner_review_is_asserted_but_identity_stays_unverified() -> None:
    source = inventory()
    document = mapping(source)
    document["evidence_class"] = "REAL_OWNER_REVIEWED"
    document["mapping_owner"] = "data.engineer@example.org"

    result = assess_series_family_mapping(source, document)

    assert result["owner_asserted_family_mapping_complete"] is True
    assert result["external_mapping_owner_identity_verified"] is False
    assert result["family_mapping_verified"] is False
    assert result["model_input_authorized"] is False


def test_real_mapping_owner_cannot_be_synthetic() -> None:
    source = inventory()
    document = mapping(source)
    document["evidence_class"] = "REAL_OWNER_REVIEWED"
    document["mapping_owner"] = "Synthetic fixture owner"

    with pytest.raises(EntsoeSeriesFamilyMappingError, match="owner is synthetic"):
        assess_series_family_mapping(source, document)


def test_signature_id_uses_only_the_nine_raw_semantic_fields() -> None:
    source = inventory()
    first = source["rows"][0]
    changed = copy.deepcopy(first)
    changed["series_count"] = 99

    assert series_signature_id(first) == series_signature_id(changed)

    changed["process_type"] = "A18"
    assert series_signature_id(first) != series_signature_id(changed)


def test_missing_signature_field_is_rejected() -> None:
    source = inventory()
    row = copy.deepcopy(source["rows"][0])
    row.pop("psr_type")

    with pytest.raises(ValueError, match="signature field is missing"):
        series_signature_id(row)

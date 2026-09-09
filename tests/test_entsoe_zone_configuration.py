from __future__ import annotations

import copy

import pytest

from pfc_shaping.validation.entsoe_zone_configuration import (
    AUTHORITIES,
    REGISTRY_SCHEMA,
    REQUIRED_LOGICAL_ZONES,
    EntsoeZoneConfigurationError,
    assess_zone_configuration,
    resolve_owner_asserted_logical_zone,
    resolve_owner_asserted_zone_interval,
)


def _eic(seed: int) -> str:
    return f"{seed % 100:02d}Y{seed:012d}A"


def _entry(
    seed: int,
    zone: str,
    start: str,
    end: str,
    *,
    status: str = "ACTIVE",
) -> dict[str, object]:
    return {
        "source_identifier_scheme": "EIC_Y_CODE",
        "source_identifier": _eic(seed),
        "logical_zone": zone,
        "domain_kind": "BIDDING_ZONE",
        "valid_from_utc": start,
        "valid_to_utc": end,
        "source_status": status,
        "source_document_id_kind": "OFFICIAL_EIC_REGISTRY_ENTRY_ID",
        "source_document_id": f"SYNTHETIC-EIC-{seed}",
        "source_endpoint": "https://example.invalid/eic-registry",
        "owner_semantics_confirmed": True,
        "reason": "Synthetic temporal zone mapping fixture.",
    }


def registry() -> dict[str, object]:
    start = "2023-01-01T00:00:00Z"
    split = "2025-01-01T00:00:00Z"
    end = "2027-01-01T00:00:00Z"
    entries = [
        _entry(1, "CH", start, split, status="SUPERSEDED"),
        _entry(2, "CH", split, end),
        _entry(3, "DE_LU", start, end),
        _entry(4, "FR", start, end),
        _entry(5, "IT_NORTH", start, end),
        _entry(6, "AT", start, end),
    ]
    return {
        "schema_version": REGISTRY_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "registry_owner": "Synthetic fixture owner",
        "registered_at_utc": "2026-08-06T15:00:00Z",
        "coverage_window_start_utc": start,
        "coverage_window_end_utc": "2026-08-06T00:00:00Z",
        "entries": entries,
        "authorities": dict(AUTHORITIES),
    }


def test_temporal_zone_registry_reports_complete_synthetic_window() -> None:
    result = assess_zone_configuration(registry())

    assert result["entry_count"] == 6
    assert result["required_logical_zones"] == list(REQUIRED_LOGICAL_ZONES)
    assert all(not gaps for gaps in result["missing_intervals_by_logical_zone"].values())
    assert result["bidding_zone_code_change_count_by_logical_zone"]["CH"] == 1
    assert result["bidding_zone_code_change_count_by_logical_zone"]["DE_LU"] == 0
    assert result["synthetic_or_owner_asserted_window_complete"] is True
    assert result["owner_asserted_window_complete"] is False
    assert result["official_eic_check_digit_verified"] is False
    assert result["zone_configuration_verified"] is False
    assert result["model_input_authorized"] is False
    assert result["validator_databricks_request_count"] == 0


def test_exact_identifier_resolves_on_each_side_of_code_change() -> None:
    document = registry()

    assert resolve_owner_asserted_logical_zone(
        document,
        source_identifier_scheme="EIC_Y_CODE",
        source_identifier=_eic(1),
        domain_kind="BIDDING_ZONE",
        at_utc="2024-12-31T23:59:59Z",
    ) == "CH"

    interval = resolve_owner_asserted_zone_interval(
        document,
        source_identifier_scheme="EIC_Y_CODE",
        source_identifier=_eic(2),
        domain_kind="BIDDING_ZONE",
        valid_from_utc="2025-01-01T00:00:00Z",
        valid_to_utc="2026-08-06T00:00:00Z",
    )
    assert interval["logical_zone"] == "CH"
    assert len(interval["zone_entry_id"]) == 64


def test_binding_interval_cannot_cross_a_zone_code_change() -> None:
    document = registry()

    with pytest.raises(EntsoeZoneConfigurationError, match="no unique containing"):
        resolve_owner_asserted_zone_interval(
            document,
            source_identifier_scheme="EIC_Y_CODE",
            source_identifier=_eic(1),
            domain_kind="BIDDING_ZONE",
            valid_from_utc="2024-12-01T00:00:00Z",
            valid_to_utc="2025-02-01T00:00:00Z",
        )
    assert resolve_owner_asserted_logical_zone(
        document,
        source_identifier_scheme="EIC_Y_CODE",
        source_identifier=_eic(2),
        domain_kind="BIDDING_ZONE",
        at_utc="2025-01-01T00:00:00Z",
    ) == "CH"


def test_missing_interval_is_reported_not_filled() -> None:
    document = registry()
    document["entries"][1]["valid_from_utc"] = "2025-02-01T00:00:00Z"

    result = assess_zone_configuration(document)

    assert result["missing_intervals_by_logical_zone"]["CH"] == [
        {
            "start_utc": "2025-01-01T00:00:00Z",
            "end_utc": "2025-02-01T00:00:00Z",
        }
    ]
    assert result["synthetic_or_owner_asserted_window_complete"] is False


def test_registry_may_govern_an_announced_future_configuration_window() -> None:
    document = registry()
    document["coverage_window_start_utc"] = "2026-09-01T00:00:00Z"
    document["coverage_window_end_utc"] = "2026-12-01T00:00:00Z"

    result = assess_zone_configuration(document)

    assert result["synthetic_or_owner_asserted_window_complete"] is True


def test_overlapping_source_identifier_is_rejected() -> None:
    document = registry()
    duplicate = copy.deepcopy(document["entries"][0])
    duplicate["logical_zone"] = "FR"
    duplicate["valid_from_utc"] = "2024-01-01T00:00:00Z"
    duplicate["valid_to_utc"] = "2024-06-01T00:00:00Z"
    document["entries"].append(duplicate)

    with pytest.raises(EntsoeZoneConfigurationError, match="source identifier.*overlap"):
        assess_zone_configuration(document)


def test_two_eic_codes_cannot_overlap_for_one_logical_bidding_zone() -> None:
    document = registry()
    document["entries"][1]["valid_from_utc"] = "2024-12-01T00:00:00Z"

    with pytest.raises(EntsoeZoneConfigurationError, match="logical-zone EIC.*overlap"):
        assess_zone_configuration(document)


@pytest.mark.parametrize(
    "identifier",
    ["short", "10X000000000000A", "10y000000000000A", "10Y00000000000_A"],
)
def test_eic_area_code_requires_exact_16_character_type_y_format(
    identifier: str,
) -> None:
    document = registry()
    document["entries"][0]["source_identifier"] = identifier

    with pytest.raises(EntsoeZoneConfigurationError, match="16-character type-Y"):
        assess_zone_configuration(document)


def test_reversed_interval_and_unsafe_endpoint_are_rejected() -> None:
    reversed_interval = registry()
    reversed_interval["entries"][0]["valid_to_utc"] = "2022-01-01T00:00:00Z"
    with pytest.raises(EntsoeZoneConfigurationError, match="empty or reversed"):
        assess_zone_configuration(reversed_interval)

    unsafe_endpoint = registry()
    unsafe_endpoint["entries"][0]["source_endpoint"] = "http://example.invalid/eic"
    with pytest.raises(EntsoeZoneConfigurationError, match="clean HTTPS"):
        assess_zone_configuration(unsafe_endpoint)


def test_owner_asserted_real_registry_never_becomes_verified_authority() -> None:
    document = registry()
    document["evidence_class"] = "REAL_OWNER_ASSERTED"
    document["registry_owner"] = "data.engineer@example.org"

    result = assess_zone_configuration(document)

    assert result["owner_asserted_window_complete"] is True
    assert result["external_registry_owner_identity_verified"] is False
    assert result["official_eic_registry_membership_verified"] is False
    assert result["zone_configuration_verified"] is False
    assert result["current_exact_dev_zone_coverage_proven"] is False


def test_real_registry_owner_cannot_be_synthetic() -> None:
    document = registry()
    document["evidence_class"] = "REAL_OWNER_ASSERTED"

    with pytest.raises(EntsoeZoneConfigurationError, match="owner is synthetic"):
        assess_zone_configuration(document)


def test_authority_escalation_is_rejected() -> None:
    document = registry()
    document["authorities"]["model_input_authorized"] = True

    with pytest.raises(EntsoeZoneConfigurationError, match="authorities differs"):
        assess_zone_configuration(document)


def test_resolution_outside_validity_fails_closed() -> None:
    document = registry()

    with pytest.raises(EntsoeZoneConfigurationError, match="no unique"):
        resolve_owner_asserted_logical_zone(
            document,
            source_identifier_scheme="EIC_Y_CODE",
            source_identifier=_eic(1),
            domain_kind="BIDDING_ZONE",
            at_utc="2025-01-01T00:00:00Z",
        )


def test_unknown_zone_or_missing_lineage_is_rejected() -> None:
    unknown = registry()
    unknown["entries"][0]["logical_zone"] = "DE_AT"
    with pytest.raises(EntsoeZoneConfigurationError, match="logical zone"):
        assess_zone_configuration(unknown)

    missing_lineage = registry()
    missing_lineage["entries"][0]["source_document_id"] = ""
    with pytest.raises(EntsoeZoneConfigurationError, match="document ID"):
        assess_zone_configuration(missing_lineage)

    document_type_surrogate = registry()
    document_type_surrogate["entries"][0]["source_document_id_kind"] = (
        "ENTSOE_DOCUMENT_TYPE"
    )
    with pytest.raises(EntsoeZoneConfigurationError, match="ID kind"):
        assess_zone_configuration(document_type_surrogate)

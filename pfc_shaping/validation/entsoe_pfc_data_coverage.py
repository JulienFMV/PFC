"""Offline audit of ENTSO-E dev coverage against CH LT PFC data needs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pfc_shaping.path_safety import read_stable_single_link_file

SCHEMA_VERSION = "fmv_entsoe_pfc_data_coverage_audit.v1"
STATUS = (
    "PARTIAL_DEV_MACRO_COVERAGE_EXACT_CURRENT_INVENTORY_UNPROVEN_"
    "NO_MODEL_AUTHORITY"
)
CONTRACT_FILE_SHA256 = (
    "a6697680dda8164f9dc7e96c65c79ee637ab483fe70cf1587acb10b80f6d3cd7"
)
CONTRACT_CONTENT_ID = (
    "a032c630091b491adc2925b7432ed6a916080f10f21ba631975291b7f8b1470c"
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = REPOSITORY_ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = PHASE_ROOT / "ENTSOE-PFC-DATA-COVERAGE-AUDIT-V1.json"

SOURCE_BINDINGS = {
    "d247_family_status_sha256": (
        PHASE_ROOT / "ENTSOE-DEV-FAMILY-COVERAGE-STATUS-20260806.md",
        "c5232db312cd89d469b714b66d63cb0b6e6960ff53c00ec8c2f7ac4a2ca60c10",
    ),
    "d251_mapping_handoff_sha256": (
        PHASE_ROOT / "SESSION-HANDOFF-20260806-ENTSOE-EXACT-SERIES-FAMILY-MAPPING.md",
        "e331e746ec78f837811a46306fdc15e9520486147763895893abeeae252c96e2",
    ),
    "d243_inventory_contract_sha256": (
        PHASE_ROOT / "ENTSOE-SERIES-INVENTORY-CONTRACT-V1.json",
        "ba8f6945b4a43b54762fa475228edabea4018bfea5871f2581dc53536c0743a1",
    ),
    "d250_mapping_contract_sha256": (
        PHASE_ROOT / "ENTSOE-REAL-MAPPING-REMEDIATION-CONTRACT-V1.json",
        "4a0d69a770fe0b45063bc4e5b4be4b9aca1e86c546b40d2bf8816a001cd06b8f",
    ),
}

BASELINE_REQUIRED_GROUPS = (
    "actual_load",
    "day_ahead_prices",
    "generation_actual",
    "generation_forecast",
    "hydro_reservoir_storage",
    "load_forecast",
    "renewables_forecast_day_ahead",
    "renewables_forecast_intraday",
    "crossborder_physical_flows",
    "scheduled_exchanges",
    "ntc_day_ahead",
    "ntc_month_ahead",
    "ntc_year_ahead",
)
REQUIRED_GENERATION_FIELDS = (
    "solar",
    "wind",
    "nuclear",
    "hydro_run_of_river",
    "hydro_reservoir",
    "hydro_pumped_storage",
)
REQUIRED_CH_BORDERS = ("CH-DE", "CH-FR", "CH-IT", "CH-AT")
COUPLED_PRICE_ZONES = ("CH", "DE_LU", "FR", "IT_NORTH", "AT")
NEW_USEFUL_FAMILIES = (
    "intraday_energy_prices",
    "energy_storage_actual_and_capacity",
    "fcr_capacity_and_capacity_prices",
    "transmission_unavailability_impact_on_net_positions",
    "long_term_flow_based_capacity_parameters",
    "short_term_adequacy_forecasts",
    "consumption_unit_unavailability",
    "bidding_zone_and_eic_configuration_history",
    "balancing_elastic_demand_and_product_selection",
)
FALSE_AUTHORITIES = {
    "current_exact_family_inventory_proven": False,
    "all_baseline_groups_proven": False,
    "coupled_zone_coverage_proven": False,
    "additional_family_presence_proven": False,
    "point_in_time_availability_proven": False,
    "model_input_authorized": False,
    "training_authorized": False,
    "selection_authorized": False,
    "candidate_assembly_authorized": False,
    "promotion_authorized": False,
    "production_authorized": False,
}
ZERO_EXECUTION = {
    "databricks_connection_count": 0,
    "databricks_statement_count": 0,
    "warehouse_start_count": 0,
    "network_call_count_excluding_official_documentation_review": 0,
    "databricks_write_count": 0,
    "real_entsoe_row_count_opened": 0,
    "h_drive_access_count": 0,
}


class EntsoePfcDataCoverageError(ValueError):
    """Raised when the D252 coverage evidence is ambiguous or unsafe."""


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_and_assess_entsoe_pfc_data_coverage(
    path: str | Path = CONTRACT_PATH,
) -> dict[str, object]:
    """Read the exact D252 audit and return its value-free coverage verdict."""

    raw = read_stable_single_link_file(
        Path(path).resolve(),
        label="D252 ENTSO-E PFC data coverage audit",
        max_bytes=500_000,
    )
    if hashlib.sha256(raw).hexdigest() != CONTRACT_FILE_SHA256:
        raise EntsoePfcDataCoverageError("D252 audit raw SHA-256 differs")
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoePfcDataCoverageError("D252 audit must be strict UTF-8 JSON") from exc
    if canonical_sha256(document) != CONTRACT_CONTENT_ID:
        raise EntsoePfcDataCoverageError("D252 audit content ID differs")
    return assess_entsoe_pfc_data_coverage(document, verify_sources=True)


def assess_entsoe_pfc_data_coverage(
    document: Mapping[str, object],
    *,
    verify_sources: bool = False,
) -> dict[str, object]:
    """Validate the exact offline coverage classification and summarize gaps."""

    _exact_fields(
        document,
        {
            "schema_version",
            "status",
            "decision",
            "as_of_date",
            "scope",
            "evidence_state",
            "observed_macro_families_2026_08_03",
            "baseline_required_groups",
            "baseline_unproven_dimensions",
            "required_generation_fields",
            "required_ch_borders",
            "coupled_zone_requirements",
            "additional_high_value_families",
            "exploratory_non_blocking_families",
            "newly_identified_useful_families",
            "required_metadata",
            "source_bindings",
            "official_sources",
            "execution",
            "authorities",
        },
        "coverage audit",
    )
    _equal(document["schema_version"], SCHEMA_VERSION, "schema version")
    _equal(document["status"], STATUS, "status")
    _equal(document["decision"], "D-20260806-252", "decision")
    _equal(document["as_of_date"], "2026-08-06", "as-of date")
    _equal(document["baseline_required_groups"], list(BASELINE_REQUIRED_GROUPS), "baseline groups")
    _equal(document["required_generation_fields"], list(REQUIRED_GENERATION_FIELDS), "generation fields")
    _equal(document["required_ch_borders"], list(REQUIRED_CH_BORDERS), "CH borders")
    _equal(document["execution"], ZERO_EXECUTION, "execution counters")
    _equal(document["authorities"], FALSE_AUTHORITIES, "authorities")

    evidence = _mapping(document["evidence_state"], "evidence state")
    if evidence.get("real_table_schemas_observed") is not True:
        raise EntsoePfcDataCoverageError("real table schemas must remain observed")
    for field in (
        "current_exact_dimension_inventory_captured",
        "d243_inventory_executed",
        "current_real_fact_values_opened",
        "coverage_can_be_classified_present_or_absent",
    ):
        if evidence.get(field) is not False:
            raise EntsoePfcDataCoverageError(f"{field} must remain false")

    coupled = _sequence(document["coupled_zone_requirements"], "coupled-zone requirements")
    if len(coupled) != 5:
        raise EntsoePfcDataCoverageError("coupled-zone requirement count differs")
    price_requirement = _mapping(coupled[0], "price-zone requirement")
    _equal(price_requirement.get("families"), ["day_ahead_prices"], "price family")
    _equal(price_requirement.get("logical_zones"), list(COUPLED_PRICE_ZONES), "coupled price zones")
    if price_requirement.get("priority") != "BASELINE_REQUIRED":
        raise EntsoePfcDataCoverageError("coupled day-ahead prices must remain baseline required")

    new_entries = _sequence(document["newly_identified_useful_families"], "new families")
    new_names = tuple(
        str(_mapping(entry, "new family entry").get("family"))
        for entry in new_entries
    )
    _equal(new_names, NEW_USEFUL_FAMILIES, "newly identified families")
    configuration = _mapping(new_entries[7], "zone configuration family")
    if configuration.get("baseline_blocker") is not True:
        raise EntsoePfcDataCoverageError("zone/EIC history must remain a metadata blocker")
    if any(
        _mapping(entry, "new family entry").get("baseline_blocker") is not False
        for position, entry in enumerate(new_entries)
        if position != 7
    ):
        raise EntsoePfcDataCoverageError("optional new families cannot block the first baseline")

    source_bindings = _mapping(document["source_bindings"], "source bindings")
    for name, (_, expected_hash) in SOURCE_BINDINGS.items():
        _equal(source_bindings.get(name), expected_hash, f"source binding {name}")
    _equal(
        source_bindings.get("d241_schema_proof_content_id"),
        "d6c006609d881b51f08be6d60e01f68b59a40be8bdf2898ef0a98491f5771544",
        "D241 proof content ID",
    )
    _equal(
        source_bindings.get("d241_schema_proof_manifest_sha256"),
        "1835f93a517e9c6769079984a376fb9879b22d7c8ce2922aa42b0dd646627ada",
        "D241 proof manifest hash",
    )
    if verify_sources:
        _verify_source_files()

    official_sources = _sequence(document["official_sources"], "official sources")
    if [
        _mapping(source, "official source").get("id") for source in official_sources
    ] != ["entsoe_mop", "entsoe_edi_library", "entsoe_mop_v3r5_consultation"]:
        raise EntsoePfcDataCoverageError("official source inventory differs")

    observed = _sequence(document["observed_macro_families_2026_08_03"], "observed macro families")
    unproven = _sequence(document["baseline_unproven_dimensions"], "unproven dimensions")
    additional = _sequence(document["additional_high_value_families"], "additional families")
    exploratory = _sequence(document["exploratory_non_blocking_families"], "exploratory families")
    return {
        "schema_version": "fmv_entsoe_pfc_data_coverage_assessment.v1",
        "status": STATUS,
        "contract_content_id": canonical_sha256(document),
        "observed_macro_family_count": len(observed),
        "baseline_required_group_count": len(BASELINE_REQUIRED_GROUPS),
        "baseline_unproven_dimension_count": len(unproven),
        "coupled_zone_requirement_count": len(coupled),
        "additional_high_value_family_count": len(additional),
        "exploratory_family_count": len(exploratory),
        "newly_identified_useful_family_count": len(new_entries),
        "current_exact_inventory_proven": False,
        "all_required_data_types_proven_in_dev": False,
        "model_input_authorized": False,
        "execution": dict(ZERO_EXECUTION),
    }


def _verify_source_files() -> None:
    for name, (path, expected_hash) in SOURCE_BINDINGS.items():
        raw = read_stable_single_link_file(path, label=name, max_bytes=5_000_000)
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise EntsoePfcDataCoverageError(f"bound source changed: {name}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoePfcDataCoverageError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise EntsoePfcDataCoverageError(f"{label} must be a list")
    return value


def _exact_fields(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EntsoePfcDataCoverageError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoePfcDataCoverageError(
            f"{label} differs: expected {expected!r}, received {value!r}"
        )


__all__ = [
    "CONTRACT_CONTENT_ID",
    "CONTRACT_FILE_SHA256",
    "CONTRACT_PATH",
    "EntsoePfcDataCoverageError",
    "assess_entsoe_pfc_data_coverage",
    "load_and_assess_entsoe_pfc_data_coverage",
]

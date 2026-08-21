"""Compose value-free ENTSO-E evidence before LT model-input admission.

The composite deliberately preserves predecessor authority boundaries.  A
synthetic structural pass proves that the joins and fail-closed rules compose;
it never authorizes real data, feature use, model selection, or production.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.validation import entsoe_cadence_package_binding as d270
from pfc_shaping.validation import entsoe_incremental_quality as d245
from pfc_shaping.validation import entsoe_lt_feature_availability as d273
from pfc_shaping.validation import entsoe_parquet_streaming_integrity as d244
from pfc_shaping.validation import entsoe_series_family_mapping as d253
from pfc_shaping.validation import entsoe_series_semantic_binding as d272
from pfc_shaping.validation import entsoe_series_zone_binding as d255

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = PHASE / "ENTSOE-LT-MODEL-INPUT-ADMISSION-CONTRACT-V1.json"
D273_CONTRACT_PATH = PHASE / "ENTSOE-LT-FEATURE-AVAILABILITY-CONTRACT-V1.json"

CONTRACT_SCHEMA = "fmv_entsoe_lt_model_input_admission_contract.v1"
CONTRACT_STATUS = "SYNTHETIC_COMPOSITE_ONLY_REAL_MODEL_INPUT_NOT_AUTHORIZED"
CONTRACT_FILE_SHA256 = "59e3f61f0cd6f44e7efe0c990eee9834d94fe5b45616af8058bb3da7e1e6d30c"
CONTRACT_CONTENT_ID = "20144a9715eb6892756f8648c1e25090d4635247ac9683dd3a8ea896166f7d34"
SELECTION_SCHEMA = "fmv_entsoe_lt_physical_feature_selection.v1"
ASSESSMENT_SCHEMA = "fmv_entsoe_lt_model_input_admission_assessment.v1"
PASS_STATUS = "PASS_SYNTHETIC_COMPOSITE_STRUCTURE_REAL_EVIDENCE_REQUIRED"
FAIL_STATUS = "FAIL_SYNTHETIC_COMPOSITE_STRUCTURE_NO_MODEL_AUTHORITY"
EVIDENCE_CLASS = "SYNTHETIC_TEST_ONLY"
MAX_ENTRIES = 100_000
DIRECT_ROLE_ORDER = (
    "REALIZED_ACTUAL",
    "OPERATIONAL_FORECAST",
    "LAGGED_REALIZED",
)
DIRECT_ROLES = frozenset(DIRECT_ROLE_ORDER)
NON_PHYSICAL_ROLES = frozenset(
    {"CALENDAR_KNOWN", "ORIGIN_FROZEN_CLIMATOLOGY", "GOVERNED_SCENARIO_SHAPE"}
)
AUTHORITIES = {
    "real_source_evidence": False,
    "external_owner_identity": False,
    "official_zone_registry": False,
    "point_in_time": False,
    "predictive_value": False,
    "model_input": False,
    "model_selection": False,
    "candidate": False,
    "promotion": False,
    "production": False,
}
EXECUTION = {
    "databricks_connection_count": 0,
    "databricks_statement_count": 0,
    "warehouse_start_count": 0,
    "databricks_write_count": 0,
    "network_call_count": 0,
    "h_drive_access_count": 0,
    "opened_real_value_row_count": 0,
}
SELECTION_AUTHORITIES = {
    "real_feature_lineage": False,
    "point_in_time": False,
    "predictive_value": False,
    "model_input": False,
    "model_selection": False,
    "candidate": False,
    "production": False,
}
_SHA = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z$")


class EntsoeLtModelInputAdmissionError(ValueError):
    """Raised when composite evidence is detached, ambiguous, or unsafe."""


@dataclass(frozen=True)
class EntsoeLtModelInputAdmissionResult:
    """Hashed, value-free composite assessment."""

    assessment: dict[str, object]
    assessment_content_id: str
    selection_content_id: str
    feature_records_content_id: str


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def feature_record_id(record: Mapping[str, object]) -> str:
    return canonical_sha256(_mapping(record, "feature record"))


def validate_model_input_admission_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    contract = _mapping(document, "D280 contract")
    _exact(
        contract,
        {
            "schema_version",
            "decision",
            "status",
            "purpose",
            "predecessors",
            "join_rules",
            "feature_selection",
            "admission_rules",
            "execution",
            "authorities",
        },
        "D280 contract",
    )
    _equal(contract["schema_version"], CONTRACT_SCHEMA, "contract schema")
    _equal(contract["decision"], "D-20260806-280", "contract decision")
    _equal(contract["status"], CONTRACT_STATUS, "contract status")
    _clean(contract["purpose"], "contract purpose")
    predecessors = _mapping(contract["predecessors"], "predecessors")
    _equal(
        predecessors,
        {
            "d270_cadence_package_contract_content_id": d270.CONTRACT_CONTENT_ID,
            "d272_semantic_binding_schema": d272.BINDING_SCHEMA,
            "d253_family_mapping_schema": d253.MAPPING_SCHEMA,
            "d254_zone_registry_schema": "fmv_entsoe_zone_configuration_registry.v1",
            "d255_zone_binding_schema": d255.BINDING_SCHEMA,
            "d273_feature_availability_contract_content_id": d273.CONTRACT_CONTENT_ID,
        },
        "predecessors",
    )
    join_rules = _mapping(contract["join_rules"], "join rules")
    if set(join_rules) != {
        "base_dimension_loaded_from_verified_d270_package",
        "semantic_binding_must_match_exact_dimension_content_id",
        "inventory_family_zone_and_semantic_content_ids_must_match",
        "every_base_series_bound_exactly_once",
        "mapped_signature_cardinality_must_match_inventory",
        "all_required_family_direction_and_unit_coverage_required",
        "all_coupled_zone_windows_required",
        "all_native_cadence_slots_required",
        "selected_feature_targets_must_be_inside_cadence_window",
        "selected_feature_intervals_must_match_effective_native_cadence",
        "selected_feature_origins_must_not_follow_selection_or_reference",
        "feature_selection_must_not_predate_predecessor_evidence",
    } or any(value is not True for value in join_rules.values()):
        raise EntsoeLtModelInputAdmissionError("join rules must be exact true")
    selection = _mapping(contract["feature_selection"], "feature selection")
    _exact(
        selection,
        {
            "schema_version",
            "required_document_fields",
            "required_entry_fields",
            "direct_physical_roles",
            "derived_forecast_error_requires_separate_transform_lineage",
            "calendar_climatology_and_scenario_are_outside_entsoe_physical_selection",
            "all_structurally_eligible_direct_model_records_must_be_selected_once",
            "rejected_or_diagnostic_records_may_be_selected",
            "maximum_entries",
        },
        "feature selection",
    )
    _equal(selection["schema_version"], SELECTION_SCHEMA, "selection schema")
    _equal(
        selection["required_document_fields"],
        [
            "schema_version",
            "evidence_class",
            "base_dimension_content_id",
            "semantic_binding_content_id",
            "feature_records_content_id",
            "selected_at_utc",
            "entries",
            "authorities",
        ],
        "selection document fields",
    )
    _equal(
        selection["required_entry_fields"],
        [
            "feature_record_id",
            "feature_id_sha256",
            "series_id",
            "source_family",
            "information_role",
            "requested_use",
            "reason",
        ],
        "selection entry fields",
    )
    _equal(selection["direct_physical_roles"], list(DIRECT_ROLE_ORDER), "direct roles")
    _equal(
        selection["derived_forecast_error_requires_separate_transform_lineage"],
        True,
        "derived lineage rule",
    )
    _equal(
        selection["calendar_climatology_and_scenario_are_outside_entsoe_physical_selection"],
        True,
        "nonphysical role rule",
    )
    _equal(
        selection["all_structurally_eligible_direct_model_records_must_be_selected_once"],
        True,
        "selection completeness",
    )
    _equal(
        selection["rejected_or_diagnostic_records_may_be_selected"], False, "selection eligibility"
    )
    _equal(selection["maximum_entries"], MAX_ENTRIES, "selection cap")
    rules = _mapping(contract["admission_rules"], "admission rules")
    expected_rules = {
        "d245_incremental_quality_must_pass_through_d270": True,
        "d270_cadence_must_be_complete": True,
        "d272_structure_must_be_complete": True,
        "d253_required_coverage_must_be_complete": True,
        "d255_coupled_zone_coverage_must_be_complete": True,
        "d273_model_records_must_be_structurally_pit_safe": True,
        "direct_feature_selection_must_be_complete": True,
        "nonphysical_roles_outside_physical_selection_block_admission": False,
        "derived_transform_without_lineage_is_admissible": False,
        "synthetic_structural_pass_grants_real_model_input_authority": False,
        "raw_values_or_clear_ids_persisted_in_assessment": False,
        "monthly_effects_may_rewrite_solver_mean": False,
    }
    _equal(rules, expected_rules, "admission rules")
    _equal(contract["execution"], EXECUTION, "execution")
    _equal(contract["authorities"], AUTHORITIES, "authorities")
    _equal(canonical_sha256(contract), CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "real_model_input_admission_implemented": False,
        "databricks_connection_count": 0,
        "production_authorized": False,
    }


def verify_synthetic_entsoe_lt_model_input_admission(
    *,
    contract_path: str | Path,
    package_manifest_path: str | Path,
    quality_context_path: str | Path,
    cadence_manifest_path: str | Path,
    inventory_document: Mapping[str, object],
    family_mapping_document: Mapping[str, object],
    semantic_binding_document: Mapping[str, object],
    zone_registry_document: Mapping[str, object],
    zone_binding_document: Mapping[str, object],
    feature_records: Sequence[Mapping[str, object]],
    feature_selection_document: Mapping[str, object],
    reference_time_utc: datetime,
) -> EntsoeLtModelInputAdmissionResult:
    contract_raw = _read_small(contract_path, "D280 contract", 100_000)
    _equal(hashlib.sha256(contract_raw).hexdigest(), CONTRACT_FILE_SHA256, "D280 contract hash")
    validate_model_input_admission_contract(_strict_json(contract_raw, "D280 contract"))
    d273.validate_feature_availability_contract_path(D273_CONTRACT_PATH)
    reference = _utc_datetime(reference_time_utc, "reference time")
    cadence = d270.verify_synthetic_entsoe_cadence_package(
        contract_path=d270.CONTRACT_PATH,
        package_manifest_path=package_manifest_path,
        quality_context_path=quality_context_path,
        cadence_manifest_path=cadence_manifest_path,
        reference_time_utc=reference,
    )
    dimension, cadence_regimes = _load_verified_physical_metadata(
        package_manifest_path,
        cadence_manifest_path,
        cadence,
    )
    family = d253.assess_series_family_mapping(
        inventory_document,
        family_mapping_document,
    )
    semantic = d272.assess_series_semantic_binding(
        dimension,
        inventory_document,
        family_mapping_document,
        semantic_binding_document,
    )
    zone = d255.assess_series_zone_bindings(
        inventory_document,
        family_mapping_document,
        zone_registry_document,
        zone_binding_document,
    )
    availability = d273.assess_feature_availability(feature_records)
    predecessor_ready_at = max(
        _utc(cadence.summary["metadata_as_of_utc"], "cadence metadata as-of"),
        _utc(family_mapping_document["mapped_at_utc"], "family mapped_at_utc"),
        _utc(semantic_binding_document["bound_at_utc"], "semantic bound_at_utc"),
        _utc(zone_registry_document["registered_at_utc"], "zone registered_at_utc"),
        _utc(zone_binding_document["bound_at_utc"], "zone bound_at_utc"),
    )
    selection = _assess_feature_selection(
        dimension,
        family_mapping_document,
        semantic_binding_document,
        semantic,
        feature_records,
        availability,
        feature_selection_document,
        cadence.summary,
        cadence_regimes,
        predecessor_ready_at,
        reference,
    )
    blockers: list[str] = []
    if (
        cadence.summary["status"] != d270.PASS_STATUS
        or not cadence.summary["all_series_piecewise_grid_complete"]
    ):
        blockers.append("CADENCE_PACKAGE_INCOMPLETE")
    if not family["all_required_coverage_mapped"]:
        blockers.append("REQUIRED_FAMILY_DIRECTION_OR_UNIT_COVERAGE_INCOMPLETE")
    if (
        not semantic["all_base_series_bound_once"]
        or not semantic["all_mapped_signature_cardinalities_match"]
    ):
        blockers.append("PHYSICAL_SERIES_SEMANTIC_BINDING_INCOMPLETE")
    if (
        zone["missing_binding_signature_ids"]
        or not zone["synthetic_or_owner_asserted_all_coupled_requirements_complete"]
    ):
        blockers.append("COUPLED_ZONE_BINDING_INCOMPLETE")
    if not availability["all_model_requests_structurally_pit_safe"]:
        blockers.append("FEATURE_PIT_STRUCTURE_REJECTED")
    if not selection["all_direct_model_records_selected_once"]:
        blockers.append("DIRECT_FEATURE_SELECTION_INCOMPLETE")
    if selection["derived_record_count"]:
        blockers.append("DERIVED_TRANSFORM_LINEAGE_REQUIRED")
    blockers = sorted(set(blockers))
    summary = {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": PASS_STATUS if not blockers else FAIL_STATUS,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "blocker_codes": blockers,
        "all_synthetic_structural_gates_pass": not blockers,
        "bindings": {
            "cadence_summary_content_id": cadence.summary_content_id,
            "cadence_sidecar_manifest_content_id": cadence.sidecar_manifest_content_id,
            "base_dimension_content_id": semantic["base_dimension_content_id"],
            "inventory_content_id": semantic["inventory_content_id"],
            "family_mapping_content_id": family["mapping_content_id"],
            "semantic_binding_content_id": semantic["binding_content_id"],
            "zone_registry_content_id": zone["zone_registry_content_id"],
            "zone_binding_content_id": zone["binding_content_id"],
            "feature_availability_contract_content_id": availability["contract_content_id"],
            "feature_records_content_id": selection["feature_records_content_id"],
            "feature_selection_content_id": selection["selection_content_id"],
        },
        "counts": {
            "physical_series": semantic["physical_series_count"],
            "mapped_signatures": semantic["mapped_signature_count"],
            "cadence_expected_targets": cadence.summary["expected_target_count"],
            "cadence_missing_targets": cadence.summary["missing_native_slot_count"],
            "feature_records": availability["record_count"],
            "selected_direct_features": selection["selected_direct_record_count"],
            "derived_records_requiring_lineage": selection["derived_record_count"],
            "nonphysical_model_records": selection["nonphysical_model_record_count"],
        },
        "selected_feature_profiles": selection["selected_feature_profiles"],
        "d245_incremental_quality_passed": cadence.summary["base_d245_incremental_quality_passed"],
        "required_family_direction_unit_coverage_complete": family["all_required_coverage_mapped"],
        "coupled_zone_coverage_complete": zone[
            "synthetic_or_owner_asserted_all_coupled_requirements_complete"
        ],
        "feature_pit_structure_complete": availability["all_model_requests_structurally_pit_safe"],
        "raw_values_persisted_in_assessment": False,
        "clear_series_or_feature_ids_persisted_in_assessment": False,
        "real_model_input_authorized": False,
        "authorities": dict(AUTHORITIES),
        **EXECUTION,
    }
    return EntsoeLtModelInputAdmissionResult(
        assessment=summary,
        assessment_content_id=canonical_sha256(summary),
        selection_content_id=str(selection["selection_content_id"]),
        feature_records_content_id=str(selection["feature_records_content_id"]),
    )


def _assess_feature_selection(
    dimension: pd.DataFrame,
    family_mapping: Mapping[str, object],
    semantic_binding: Mapping[str, object],
    semantic_assessment: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    availability: Mapping[str, object],
    selection_document: Mapping[str, object],
    cadence_summary: Mapping[str, object],
    cadence_regimes: pd.DataFrame,
    predecessor_ready_at: datetime,
    reference: datetime,
) -> dict[str, object]:
    document = _mapping(selection_document, "feature selection")
    _exact(
        document,
        {
            "schema_version",
            "evidence_class",
            "base_dimension_content_id",
            "semantic_binding_content_id",
            "feature_records_content_id",
            "selected_at_utc",
            "entries",
            "authorities",
        },
        "feature selection",
    )
    _equal(document["schema_version"], SELECTION_SCHEMA, "selection schema")
    _equal(document["evidence_class"], EVIDENCE_CLASS, "selection evidence")
    _equal(
        document["base_dimension_content_id"],
        d272.dimension_content_id(dimension),
        "selection dimension content ID",
    )
    _equal(
        document["semantic_binding_content_id"],
        semantic_assessment["binding_content_id"],
        "selection semantic binding content ID",
    )
    records_id = canonical_sha256(list(records))
    _equal(document["feature_records_content_id"], records_id, "feature records content ID")
    _equal(document["authorities"], SELECTION_AUTHORITIES, "selection authorities")
    selected_at = _utc(document["selected_at_utc"], "selected_at_utc")
    if selected_at > reference:
        raise EntsoeLtModelInputAdmissionError("selection time is after reference")
    if selected_at < predecessor_ready_at:
        raise EntsoeLtModelInputAdmissionError("selection predates predecessor evidence")
    cadence_start = _utc(
        cadence_summary["assessment_start_utc"],
        "cadence assessment start",
    )
    cadence_end = _utc(
        cadence_summary["assessment_end_utc"],
        "cadence assessment end",
    )
    raw_entries = document["entries"]
    if not isinstance(raw_entries, list) or len(raw_entries) > MAX_ENTRIES:
        raise EntsoeLtModelInputAdmissionError("selection entries are invalid")
    outcomes = availability["outcomes"]
    if not isinstance(outcomes, tuple) or len(outcomes) != len(records):
        raise EntsoeLtModelInputAdmissionError("feature outcomes differ")
    record_by_id = {
        feature_record_id(_mapping(record, "feature record")): (
            _mapping(record, "feature record"),
            _mapping(outcome, "feature outcome"),
        )
        for record, outcome in zip(records, outcomes, strict=True)
    }
    family_by_signature = {
        str(entry["signature_id"]): str(entry["logical_family"])
        for entry in family_mapping["entries"]
        if isinstance(entry, Mapping) and entry.get("disposition") == "MAPPED"
    }
    family_by_series = {
        str(entry["series_id"]): family_by_signature[str(entry["signature_id"])]
        for entry in semantic_binding["entries"]
        if isinstance(entry, Mapping)
    }
    selected_ids: set[str] = set()
    selected_profiles: list[dict[str, object]] = []
    for position, raw_entry in enumerate(raw_entries, start=1):
        entry = _mapping(raw_entry, f"selection entry {position}")
        _exact(
            entry,
            {
                "feature_record_id",
                "feature_id_sha256",
                "series_id",
                "source_family",
                "information_role",
                "requested_use",
                "reason",
            },
            f"selection entry {position}",
        )
        record_id = _sha(entry["feature_record_id"], "feature record ID")
        if record_id in selected_ids:
            raise EntsoeLtModelInputAdmissionError("feature record selection is duplicated")
        selected_ids.add(record_id)
        if record_id not in record_by_id:
            raise EntsoeLtModelInputAdmissionError("selection references unknown feature record")
        record, outcome = record_by_id[record_id]
        if outcome["status"] != "STRUCTURALLY_ELIGIBLE_NO_MODEL_AUTHORITY":
            raise EntsoeLtModelInputAdmissionError("rejected or diagnostic feature was selected")
        if record["information_role"] not in DIRECT_ROLES:
            raise EntsoeLtModelInputAdmissionError(
                "nonphysical or derived feature entered physical selection"
            )
        target_start = _utc(record["target_start_utc"], "feature target start")
        target_end = _utc(record["target_end_utc"], "feature target end")
        if target_start < cadence_start or target_end > cadence_end:
            raise EntsoeLtModelInputAdmissionError(
                "selected feature target exceeds cadence assessment window"
            )
        _validate_selected_native_interval(
            cadence_regimes,
            str(entry["series_id"]),
            target_start,
            target_end,
        )
        origin = _utc(record["origin_utc"], "feature origin")
        if origin > selected_at or origin > reference:
            raise EntsoeLtModelInputAdmissionError(
                "selected feature origin follows selection or reference"
            )
        _equal(
            entry["feature_id_sha256"],
            hashlib.sha256(str(record["feature_id"]).encode("utf-8")).hexdigest(),
            "feature ID hash",
        )
        series_id = _clean(entry["series_id"], "selection series ID")
        if series_id not in family_by_series:
            raise EntsoeLtModelInputAdmissionError("selection series is not semantically bound")
        _equal(entry["source_family"], record["source_family"], "selection source family")
        _equal(entry["source_family"], family_by_series[series_id], "series logical family")
        _equal(entry["information_role"], record["information_role"], "selection information role")
        _equal(entry["requested_use"], record["requested_use"], "selection requested use")
        _clean(entry["reason"], f"selection entry {position} reason")
        selected_profiles.append(
            {
                "feature_record_id": record_id,
                "feature_id_sha256": entry["feature_id_sha256"],
                "series_id_sha256": hashlib.sha256(series_id.encode("utf-8")).hexdigest(),
                "source_family": entry["source_family"],
                "information_role": entry["information_role"],
                "requested_use": entry["requested_use"],
                "raw_ids_persisted": False,
            }
        )
    direct_expected = {
        record_id
        for record_id, (record, outcome) in record_by_id.items()
        if record["requested_use"] in d273.MODEL_USES
        and record["information_role"] in DIRECT_ROLES
        and outcome["status"] == "STRUCTURALLY_ELIGIBLE_NO_MODEL_AUTHORITY"
    }
    direct_selected = {
        profile["feature_record_id"]
        for profile in selected_profiles
        if profile["information_role"] in DIRECT_ROLES
    }
    derived = sum(
        1
        for record, outcome in record_by_id.values()
        if record["requested_use"] in d273.MODEL_USES
        and record["information_role"] == "FORECAST_ERROR"
        and outcome["status"] == "STRUCTURALLY_ELIGIBLE_NO_MODEL_AUTHORITY"
    )
    nonphysical = sum(
        1
        for record, outcome in record_by_id.values()
        if record["requested_use"] in d273.MODEL_USES
        and record["information_role"] in NON_PHYSICAL_ROLES
        and outcome["status"] == "STRUCTURALLY_ELIGIBLE_NO_MODEL_AUTHORITY"
    )
    selected_profiles.sort(key=lambda value: str(value["feature_record_id"]))
    return {
        "selection_content_id": canonical_sha256(document),
        "feature_records_content_id": records_id,
        "selected_direct_record_count": len(direct_selected),
        "expected_direct_record_count": len(direct_expected),
        "all_direct_model_records_selected_once": direct_selected == direct_expected,
        "derived_record_count": derived,
        "nonphysical_model_record_count": nonphysical,
        "selected_feature_profiles": selected_profiles,
    }


def _load_verified_physical_metadata(
    package_manifest_path: str | Path,
    cadence_manifest_path: str | Path,
    cadence: d270.EntsoeCadencePackageBindingResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = assert_absolute_path_has_no_links(package_manifest_path)
    manifest = _strict_json(
        _read_small(manifest_path, "base package manifest", 2_000_000),
        "base package manifest",
    )
    cadence_path = assert_absolute_path_has_no_links(cadence_manifest_path)
    cadence_manifest = _strict_json(
        _read_small(cadence_path, "cadence sidecar manifest", 2_000_000),
        "cadence sidecar manifest",
    )
    _equal(
        d270.derive_sidecar_manifest_content_id(cadence_manifest),
        cadence.sidecar_manifest_content_id,
        "cadence manifest reread content ID",
    )
    _equal(
        cadence_manifest["base_package_manifest_content_id"],
        d244.canonical_sha256(manifest),
        "cadence-to-base manifest reread binding",
    )
    _equal(
        cadence_manifest["snapshot_id"],
        manifest["snapshot_id"],
        "cadence-to-base snapshot reread binding",
    )
    declarations = {
        str(item["role"]): _mapping(item, "base artifact declaration")
        for item in manifest["artifacts"]
    }
    batches: list[pa.RecordBatch] = []
    d245._visit_verified_batches(
        manifest_path.parent / d244.ROLE_FILE_NAMES["series_dimension"],
        declarations["series_dimension"],
        "series_dimension",
        batches.append,
    )
    if not batches:
        raise EntsoeLtModelInputAdmissionError("verified dimension is empty")
    table = pa.Table.from_batches(batches, schema=d244.expected_arrow_schema("series_dimension"))
    if table.num_rows > d244.ROLE_MAX_ROWS["series_dimension"]:
        raise EntsoeLtModelInputAdmissionError("dimension row cap exceeded")
    cadence_batches: list[pa.RecordBatch] = []
    d270._stream_sidecar(
        cadence_path.parent / d270.FILE_NAME,
        _mapping(cadence_manifest["artifact"], "cadence artifact declaration"),
        cadence_batches.append,
    )
    if not cadence_batches:
        raise EntsoeLtModelInputAdmissionError("verified cadence sidecar is empty")
    cadence_table = pa.Table.from_batches(
        cadence_batches,
        schema=d270.expected_sidecar_schema(),
    )
    return table.to_pandas(), cadence_table.to_pandas()


def _validate_selected_native_interval(
    regimes: pd.DataFrame,
    series_id: str,
    target_start: datetime,
    target_end: datetime,
) -> None:
    start = pd.Timestamp(target_start)
    end = pd.Timestamp(target_end)
    selected = regimes.loc[regimes["series_id"].eq(series_id)]
    matches = selected.loc[
        selected["valid_from_utc"].le(start)
        & (selected["valid_to_utc"].isna() | selected["valid_to_utc"].ge(end))
    ]
    if len(matches) != 1:
        raise EntsoeLtModelInputAdmissionError(
            "selected feature interval has no unique effective cadence regime"
        )
    resolution = str(matches.iloc[0]["native_resolution"])
    seconds = d270.RESOLUTION_SECONDS.get(resolution)
    if seconds is None:
        raise EntsoeLtModelInputAdmissionError("selected feature effective cadence is unsupported")
    if (target_end - target_start).total_seconds() != seconds:
        raise EntsoeLtModelInputAdmissionError(
            "selected feature interval duration differs from native cadence"
        )
    if int(target_start.timestamp()) % seconds:
        raise EntsoeLtModelInputAdmissionError(
            "selected feature interval is off the absolute UTC native grid"
        )


def _read_small(path: str | Path, label: str, limit: int) -> bytes:
    try:
        return read_stable_single_link_file(path, label=label, max_bytes=limit)
    except (OSError, ValueError) as exc:
        raise EntsoeLtModelInputAdmissionError(f"{label} read failed") from exc


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeLtModelInputAdmissionError(f"{label} JSON is invalid") from exc
    return _mapping(value, label)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EntsoeLtModelInputAdmissionError("JSON contains duplicate keys")
        result[key] = value
    return result


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeLtModelInputAdmissionError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EntsoeLtModelInputAdmissionError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise EntsoeLtModelInputAdmissionError(f"{label} differs")


def _clean(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise EntsoeLtModelInputAdmissionError(f"{label} must be clean text")
    return value


def _sha(value: object, label: str) -> str:
    text = _clean(value, label)
    if _SHA.fullmatch(text) is None:
        raise EntsoeLtModelInputAdmissionError(f"{label} must be lowercase SHA-256")
    return text


def _utc(value: object, label: str) -> datetime:
    text = _clean(value, label)
    if _UTC.fullmatch(text) is None:
        raise EntsoeLtModelInputAdmissionError(f"{label} must be canonical UTC Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise EntsoeLtModelInputAdmissionError(f"{label} is invalid") from exc
    return _utc_datetime(parsed, label)


def _utc_datetime(value: datetime, label: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timezone.utc.utcoffset(value)
    ):
        raise EntsoeLtModelInputAdmissionError(f"{label} must be UTC-aware")
    return value.astimezone(timezone.utc)


__all__ = [
    "ASSESSMENT_SCHEMA",
    "AUTHORITIES",
    "CONTRACT_CONTENT_ID",
    "CONTRACT_FILE_SHA256",
    "CONTRACT_PATH",
    "DIRECT_ROLE_ORDER",
    "DIRECT_ROLES",
    "EntsoeLtModelInputAdmissionError",
    "EntsoeLtModelInputAdmissionResult",
    "PASS_STATUS",
    "SELECTION_AUTHORITIES",
    "SELECTION_SCHEMA",
    "canonical_sha256",
    "feature_record_id",
    "validate_model_input_admission_contract",
    "verify_synthetic_entsoe_lt_model_input_admission",
]

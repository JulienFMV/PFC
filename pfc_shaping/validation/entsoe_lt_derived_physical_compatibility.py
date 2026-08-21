"""Compose D280 physical selections with D281 derived-feature lineage.

The composition is deliberately value-free.  It proves that the exact load
actual and load forecast used by one lagged forecast-error transform resolve
to the same temporal bidding zone, canonical MW unit and effective native
cadence.  It grants no real-data, arithmetic, predictive or model authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.validation import entsoe_cadence_package_binding as d270
from pfc_shaping.validation import entsoe_lt_derived_feature_lineage as d281
from pfc_shaping.validation import entsoe_lt_model_input_admission as d280

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = PHASE / "ENTSOE-LT-DERIVED-PHYSICAL-COMPATIBILITY-CONTRACT-V1.json"

CONTRACT_SCHEMA = "fmv_entsoe_lt_derived_physical_compatibility_contract.v1"
CONTRACT_STATUS = "SYNTHETIC_PHYSICAL_COMPATIBILITY_NO_REAL_OR_MODEL_AUTHORITY"
CONTRACT_FILE_SHA256 = "81158dedb6aaefcfe895590b7a0ac92d94a68c5fa7f4bda5f89f2ae7a163a04d"
CONTRACT_CONTENT_ID = "2976562eb360f93da5135d3f542e57aebd0848a9ee692654bab0477b67d9985e"
ASSESSMENT_SCHEMA = "fmv_entsoe_lt_derived_physical_compatibility_assessment.v1"
PASS_STATUS = "PASS_SYNTHETIC_DERIVED_PHYSICAL_COMPATIBILITY_NO_MODEL_AUTHORITY"
EXPECTED_D280_BLOCKERS = ("DERIVED_TRANSFORM_LINEAGE_REQUIRED",)
FORMULA_ROLES = ("REALIZED_ACTUAL", "OPERATIONAL_FORECAST")
FORMULA_FAMILIES = ("actual_load", "load_forecast")
REQUIRED_UNIT = "MW"

AUTHORITIES = {
    "real_source_evidence": False,
    "external_owner_identity": False,
    "official_zone_registry": False,
    "point_in_time": False,
    "arithmetic_recomputation": False,
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

_SHA = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class EntsoeLtDerivedPhysicalCompatibilityError(ValueError):
    """Raised when exact D280 and D281 evidence cannot be safely composed."""


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_derived_physical_compatibility_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    contract = _mapping(document, "D282 contract")
    _exact(
        contract,
        {
            "schema_version",
            "decision",
            "status",
            "purpose",
            "predecessors",
            "join_grain",
            "physical_compatibility",
            "integrity",
            "execution",
            "authorities",
        },
        "D282 contract",
    )
    _equal(contract["schema_version"], CONTRACT_SCHEMA, "contract schema")
    _equal(contract["decision"], "D-20260806-282", "contract decision")
    _equal(contract["status"], CONTRACT_STATUS, "contract status")
    _clean(contract["purpose"], "contract purpose")
    _equal(
        contract["predecessors"],
        {
            "d280_contract_content_id": d280.CONTRACT_CONTENT_ID,
            "d281_contract_content_id": d281.CONTRACT_CONTENT_ID,
            "accepted_d280_blocker_codes_before_composition": list(EXPECTED_D280_BLOCKERS),
            "d281_status": d281.PASS_STATUS,
        },
        "predecessors",
    )
    _equal(
        contract["join_grain"],
        {
            "one_output_feature_record_per_transform": True,
            "one_selected_physical_series_per_primitive_feature_record": True,
            "primitive_formula_roles_in_order": list(FORMULA_ROLES),
            "primitive_source_families_in_order": list(FORMULA_FAMILIES),
            "duplicate_or_orphan_join_is_admissible": False,
        },
        "join grain",
    )
    _equal(
        contract["physical_compatibility"],
        {
            "required_canonical_unit": REQUIRED_UNIT,
            "logical_zone_must_match": True,
            "target_interval_must_be_covered_by_each_temporal_zone_binding": True,
            "target_interval_must_match_all_lineage_records": True,
            "effective_native_cadence_must_match": True,
            "effective_native_cadence_is_proven_by_d280_native_interval_validation_and_common_target_duration": True,
            "derived_output_inherits_primitive_zone_unit_and_cadence": True,
            "missing_or_ambiguous_temporal_binding_is_admissible": False,
        },
        "physical compatibility",
    )
    _equal(
        contract["integrity"],
        {
            "feature_records_selection_semantics_family_zone_and_lineage_must_remain_content_stable": True,
            "predecessor_content_ids_must_match_exactly": True,
            "clear_series_or_feature_ids_may_be_persisted": False,
            "real_or_feature_values_may_be_opened_or_recomputed": False,
            "synthetic_pass_grants_real_data_or_model_authority": False,
            "monthly_effects_may_rewrite_solver_mean": False,
        },
        "integrity",
    )
    _equal(contract["execution"], EXECUTION, "execution")
    _equal(contract["authorities"], AUTHORITIES, "authorities")
    _equal(canonical_sha256(contract), CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "real_or_feature_values_opened": False,
        "derived_model_input_authorized": False,
        **EXECUTION,
    }


def validate_derived_physical_compatibility_contract_path(
    path: str | Path,
) -> dict[str, object]:
    contract_path = Path(path)
    if not contract_path.is_absolute():
        raise EntsoeLtDerivedPhysicalCompatibilityError("contract path must be absolute")
    try:
        payload = read_stable_single_link_file(
            contract_path,
            label="D282 ENTSO-E derived physical compatibility contract",
            max_bytes=100_000,
        )
    except (OSError, ValueError) as exc:
        raise EntsoeLtDerivedPhysicalCompatibilityError("contract read failed") from exc
    if hashlib.sha256(payload).hexdigest() != CONTRACT_FILE_SHA256:
        raise EntsoeLtDerivedPhysicalCompatibilityError("contract SHA-256 differs")
    return validate_derived_physical_compatibility_contract(_strict_json(payload, "D282 contract"))


def verify_synthetic_entsoe_lt_derived_physical_compatibility(
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
    lineage_document: Mapping[str, object],
    reference_time_utc: datetime,
) -> dict[str, object]:
    """Compose exact predecessor evidence without reading feature values."""

    validate_derived_physical_compatibility_contract_path(contract_path)
    reference = _utc_datetime(reference_time_utc, "reference time")
    tracked_before = _tracked_content_ids(
        feature_records,
        feature_selection_document,
        family_mapping_document,
        semantic_binding_document,
        zone_registry_document,
        zone_binding_document,
        lineage_document,
    )
    d280_result = d280.verify_synthetic_entsoe_lt_model_input_admission(
        contract_path=d280.CONTRACT_PATH,
        package_manifest_path=package_manifest_path,
        quality_context_path=quality_context_path,
        cadence_manifest_path=cadence_manifest_path,
        inventory_document=inventory_document,
        family_mapping_document=family_mapping_document,
        semantic_binding_document=semantic_binding_document,
        zone_registry_document=zone_registry_document,
        zone_binding_document=zone_binding_document,
        feature_records=feature_records,
        feature_selection_document=feature_selection_document,
        reference_time_utc=reference,
    )
    d281_assessment = d281.assess_derived_feature_lineage(
        feature_records,
        lineage_document,
        reference_time_utc=reference,
    )
    tracked_after = _tracked_content_ids(
        feature_records,
        feature_selection_document,
        family_mapping_document,
        semantic_binding_document,
        zone_registry_document,
        zone_binding_document,
        lineage_document,
    )
    if tracked_before != tracked_after:
        raise EntsoeLtDerivedPhysicalCompatibilityError(
            "composite inputs changed during verification"
        )
    return _compose_verified_predecessors(
        d280_result.assessment,
        d280_result.assessment_content_id,
        d281_assessment,
        feature_records,
        feature_selection_document,
        family_mapping_document,
        semantic_binding_document,
        zone_binding_document,
        tracked_before,
        reference,
    )


def _compose_verified_predecessors(
    d280_assessment: Mapping[str, object],
    d280_assessment_content_id: str,
    d281_assessment: Mapping[str, object],
    feature_records: Sequence[Mapping[str, object]],
    feature_selection_document: Mapping[str, object],
    family_mapping_document: Mapping[str, object],
    semantic_binding_document: Mapping[str, object],
    zone_binding_document: Mapping[str, object],
    tracked_content_ids: Mapping[str, str],
    reference_time_utc: datetime,
) -> dict[str, object]:
    reference = _utc_datetime(reference_time_utc, "composition reference time")
    _equal(
        d280_assessment.get("contract_content_id"),
        d280.CONTRACT_CONTENT_ID,
        "D280 contract content ID",
    )
    _equal(
        d281_assessment.get("contract_content_id"),
        d281.CONTRACT_CONTENT_ID,
        "D281 contract content ID",
    )
    _equal(
        canonical_sha256(d280_assessment),
        d280_assessment_content_id,
        "D280 assessment content ID",
    )
    blockers = d280_assessment.get("blocker_codes")
    if blockers != list(EXPECTED_D280_BLOCKERS):
        raise EntsoeLtDerivedPhysicalCompatibilityError(
            "D280 must have exactly the derived-lineage blocker before composition"
        )
    if d280_assessment.get("real_model_input_authorized") is not False:
        raise EntsoeLtDerivedPhysicalCompatibilityError("D280 authority escalated")
    if not _all_false(d280_assessment.get("authorities")):
        raise EntsoeLtDerivedPhysicalCompatibilityError("D280 authorities differ")
    _equal(d281_assessment.get("status"), d281.PASS_STATUS, "D281 status")
    if d281_assessment.get("d280_derived_model_input_admitted") is not False:
        raise EntsoeLtDerivedPhysicalCompatibilityError("D281 authority escalated")
    if not _all_false(d281_assessment.get("authorities")):
        raise EntsoeLtDerivedPhysicalCompatibilityError("D281 authorities differ")
    for field in (
        "raw_values_opened",
        "arithmetic_recomputed",
        "semantic_unit_zone_and_cadence_proven",
    ):
        if d281_assessment.get(field) is not False:
            raise EntsoeLtDerivedPhysicalCompatibilityError(f"D281 {field} boundary differs")
    _equal(
        d280_assessment["bindings"]["feature_records_content_id"],
        d281_assessment["feature_records_content_id"],
        "shared feature records content ID",
    )
    _equal(
        d280_assessment["bindings"]["feature_records_content_id"],
        tracked_content_ids["feature_records"],
        "tracked feature records content ID",
    )
    _equal(
        d280_assessment["bindings"]["feature_selection_content_id"],
        tracked_content_ids["feature_selection"],
        "tracked feature selection content ID",
    )
    _equal(
        d280_assessment["bindings"]["family_mapping_content_id"],
        tracked_content_ids["family_mapping"],
        "tracked family mapping content ID",
    )
    _equal(
        d280_assessment["bindings"]["semantic_binding_content_id"],
        tracked_content_ids["semantic_binding"],
        "tracked semantic binding content ID",
    )
    _equal(
        d280_assessment["bindings"]["zone_binding_content_id"],
        tracked_content_ids["zone_binding"],
        "tracked zone binding content ID",
    )
    _equal(
        d280_assessment["bindings"]["zone_registry_content_id"],
        tracked_content_ids["zone_registry"],
        "tracked zone registry content ID",
    )
    _equal(
        d281_assessment["lineage_content_id"],
        tracked_content_ids["lineage"],
        "tracked lineage content ID",
    )

    records = _unique_by_hash(feature_records, d281.feature_record_id, "feature record")
    selection_by_record = _selection_by_record(feature_selection_document)
    semantic_by_series = _semantic_by_series(semantic_binding_document)
    family_by_signature = _family_by_signature(family_mapping_document)
    raw_profiles = d281_assessment.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise EntsoeLtDerivedPhysicalCompatibilityError("D281 profiles are invalid")

    profiles: list[dict[str, object]] = []
    seen_outputs: set[str] = set()
    for position, raw_profile in enumerate(raw_profiles, start=1):
        profile = _mapping(raw_profile, f"D281 profile {position}")
        output_id = _sha(profile["output_feature_record_id"], "output record ID")
        if output_id in seen_outputs:
            raise EntsoeLtDerivedPhysicalCompatibilityError(
                "derived output is duplicated in D281 profiles"
            )
        seen_outputs.add(output_id)
        if output_id not in records:
            raise EntsoeLtDerivedPhysicalCompatibilityError(
                "D281 output is absent from feature records"
            )
        input_ids = profile.get("input_feature_record_ids_in_formula_order")
        if not isinstance(input_ids, list) or len(input_ids) != 2:
            raise EntsoeLtDerivedPhysicalCompatibilityError("D281 formula inputs are invalid")
        clean_input_ids = tuple(_sha(value, "input record ID") for value in input_ids)
        primitive_profiles = []
        for expected_role, expected_family, record_id in zip(
            FORMULA_ROLES,
            FORMULA_FAMILIES,
            clean_input_ids,
            strict=True,
        ):
            if record_id not in records:
                raise EntsoeLtDerivedPhysicalCompatibilityError(
                    "D281 primitive is absent from feature records"
                )
            if record_id not in selection_by_record:
                raise EntsoeLtDerivedPhysicalCompatibilityError(
                    "D281 primitive is not selected exactly once by D280"
                )
            primitive_profiles.append(
                _physical_primitive_profile(
                    records[record_id],
                    selection_by_record[record_id],
                    semantic_by_series,
                    family_by_signature,
                    zone_binding_document,
                    expected_role,
                    expected_family,
                )
            )
        actual, forecast = primitive_profiles
        output_record = records[output_id]
        for field in ("target_start_utc", "target_end_utc"):
            _equal(actual[field], forecast[field], f"primitive {field}")
            _equal(actual[field], output_record[field], f"derived output {field}")
        if actual["series_id"] == forecast["series_id"]:
            raise EntsoeLtDerivedPhysicalCompatibilityError(
                "actual and forecast cannot use one physical series"
            )
        if actual["canonical_unit"] != REQUIRED_UNIT or forecast["canonical_unit"] != REQUIRED_UNIT:
            raise EntsoeLtDerivedPhysicalCompatibilityError(
                "load forecast-error primitives must use canonical MW"
            )
        if actual["logical_zone"] != forecast["logical_zone"]:
            raise EntsoeLtDerivedPhysicalCompatibilityError(
                "load forecast-error primitive logical zones differ"
            )
        if actual["cadence_seconds"] != forecast["cadence_seconds"]:
            raise EntsoeLtDerivedPhysicalCompatibilityError(
                "load forecast-error effective native cadences differ"
            )
        cadence_seconds = int(actual["cadence_seconds"])
        resolution = _resolution_for_seconds(cadence_seconds)
        profiles.append(
            {
                "transform_id_sha256": _sha(profile["transform_id_sha256"], "transform ID hash"),
                "output_feature_record_id": output_id,
                "input_feature_record_ids_in_formula_order": list(clean_input_ids),
                "actual_series_id_sha256": hashlib.sha256(
                    str(actual["series_id"]).encode("utf-8")
                ).hexdigest(),
                "forecast_series_id_sha256": hashlib.sha256(
                    str(forecast["series_id"]).encode("utf-8")
                ).hexdigest(),
                "logical_zone_sha256": hashlib.sha256(
                    str(actual["logical_zone"]).encode("utf-8")
                ).hexdigest(),
                "canonical_unit": REQUIRED_UNIT,
                "effective_native_resolution": resolution,
                "target_duration_seconds": cadence_seconds,
                "temporal_zone_binding_complete": True,
                "derived_output_inherits_physical_profile": True,
                "raw_ids_persisted": False,
                "raw_or_feature_values_opened": False,
            }
        )

    expected_count = d281_assessment.get("forecast_error_record_count")
    if len(profiles) != expected_count or len(profiles) != d281_assessment.get("transform_count"):
        raise EntsoeLtDerivedPhysicalCompatibilityError(
            "composed transform cardinality differs from D281"
        )
    profiles.sort(key=lambda value: str(value["transform_id_sha256"]))
    assessment = {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": PASS_STATUS,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "reference_time_utc": _utc_text(reference),
        "bindings": {
            "d280_assessment_content_id": _sha(
                d280_assessment_content_id, "D280 assessment content ID"
            ),
            "d281_assessment_content_id": canonical_sha256(d281_assessment),
            "feature_records_content_id": tracked_content_ids["feature_records"],
            "feature_selection_content_id": tracked_content_ids["feature_selection"],
            "family_mapping_content_id": tracked_content_ids["family_mapping"],
            "semantic_binding_content_id": tracked_content_ids["semantic_binding"],
            "zone_registry_content_id": tracked_content_ids["zone_registry"],
            "zone_binding_content_id": tracked_content_ids["zone_binding"],
            "lineage_content_id": tracked_content_ids["lineage"],
        },
        "transform_count": len(profiles),
        "all_primitives_selected_once": True,
        "all_temporal_zone_bindings_complete": True,
        "all_logical_zones_match": True,
        "all_canonical_units_are_mw": True,
        "all_effective_native_cadences_match": True,
        "d280_lineage_blocker_structurally_discharged": True,
        "profiles": profiles,
        "clear_series_or_feature_ids_persisted": False,
        "real_or_feature_values_opened": False,
        "arithmetic_recomputed": False,
        "derived_model_input_authorized": False,
        "authorities": dict(AUTHORITIES),
        **EXECUTION,
    }
    return assessment


def _physical_primitive_profile(
    record: Mapping[str, object],
    selection: Mapping[str, object],
    semantic_by_series: Mapping[str, Mapping[str, object]],
    family_by_signature: Mapping[str, Mapping[str, object]],
    zone_binding_document: Mapping[str, object],
    expected_role: str,
    expected_family: str,
) -> dict[str, object]:
    _equal(record["information_role"], expected_role, "primitive role")
    _equal(record["source_family"], expected_family, "primitive source family")
    _equal(selection["information_role"], expected_role, "selected primitive role")
    _equal(selection["source_family"], expected_family, "selected primitive family")
    series_id = _clean(selection["series_id"], "selected series ID")
    if series_id not in semantic_by_series:
        raise EntsoeLtDerivedPhysicalCompatibilityError(
            "selected series lacks exact semantic binding"
        )
    signature_id = _sha(semantic_by_series[series_id]["signature_id"], "semantic signature ID")
    if signature_id not in family_by_signature:
        raise EntsoeLtDerivedPhysicalCompatibilityError(
            "selected signature lacks exact family mapping"
        )
    family = family_by_signature[signature_id]
    _equal(family["logical_family"], expected_family, "mapped primitive family")
    canonical_unit = _clean(family["canonical_unit"], "canonical unit")
    target_start = _utc(record["target_start_utc"], "primitive target start")
    target_end = _utc(record["target_end_utc"], "primitive target end")
    if target_start >= target_end:
        raise EntsoeLtDerivedPhysicalCompatibilityError(
            "primitive target interval is empty or reversed"
        )
    logical_zone = _zone_for_interval(
        signature_id,
        target_start,
        target_end,
        zone_binding_document,
    )
    cadence_seconds = int((target_end - target_start).total_seconds())
    _resolution_for_seconds(cadence_seconds)
    return {
        "series_id": series_id,
        "canonical_unit": canonical_unit,
        "logical_zone": logical_zone,
        "cadence_seconds": cadence_seconds,
        "target_start_utc": record["target_start_utc"],
        "target_end_utc": record["target_end_utc"],
    }


def _zone_for_interval(
    signature_id: str,
    target_start: datetime,
    target_end: datetime,
    zone_binding_document: Mapping[str, object],
) -> str:
    raw_bindings = zone_binding_document.get("bindings")
    if not isinstance(raw_bindings, list):
        raise EntsoeLtDerivedPhysicalCompatibilityError("zone bindings are invalid")
    matches: list[str] = []
    for raw_binding in raw_bindings:
        binding = _mapping(raw_binding, "zone binding")
        if binding.get("signature_id") != signature_id:
            continue
        start = _utc(binding["series_valid_from_utc"], "zone binding start")
        end = _utc(binding["series_valid_to_utc"], "zone binding end")
        if start <= target_start and end >= target_end:
            matches.append(_clean(binding["expected_logical_zone"], "logical zone"))
    if len(matches) != 1:
        raise EntsoeLtDerivedPhysicalCompatibilityError(
            "primitive target has no unique temporal zone binding"
        )
    return matches[0]


def _selection_by_record(
    document: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw_entries = document.get("entries")
    if not isinstance(raw_entries, list):
        raise EntsoeLtDerivedPhysicalCompatibilityError("selection entries are invalid")
    result: dict[str, Mapping[str, object]] = {}
    for raw_entry in raw_entries:
        entry = _mapping(raw_entry, "selection entry")
        record_id = _sha(entry["feature_record_id"], "selected feature record ID")
        if record_id in result:
            raise EntsoeLtDerivedPhysicalCompatibilityError(
                "selection feature record is duplicated"
            )
        result[record_id] = entry
    return result


def _semantic_by_series(
    document: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw_entries = document.get("entries")
    if not isinstance(raw_entries, list):
        raise EntsoeLtDerivedPhysicalCompatibilityError("semantic binding entries are invalid")
    result: dict[str, Mapping[str, object]] = {}
    for raw_entry in raw_entries:
        entry = _mapping(raw_entry, "semantic binding entry")
        series_id = _clean(entry["series_id"], "semantic series ID")
        if series_id in result:
            raise EntsoeLtDerivedPhysicalCompatibilityError("semantic series is duplicated")
        result[series_id] = entry
    return result


def _family_by_signature(
    document: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw_entries = document.get("entries")
    if not isinstance(raw_entries, list):
        raise EntsoeLtDerivedPhysicalCompatibilityError("family entries are invalid")
    result: dict[str, Mapping[str, object]] = {}
    for raw_entry in raw_entries:
        entry = _mapping(raw_entry, "family mapping entry")
        if entry.get("disposition") != "MAPPED":
            continue
        signature_id = _sha(entry["signature_id"], "family signature ID")
        if signature_id in result:
            raise EntsoeLtDerivedPhysicalCompatibilityError("family signature is duplicated")
        result[signature_id] = entry
    return result


def _unique_by_hash(
    values: Sequence[Mapping[str, object]],
    identifier,
    label: str,
) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for raw_value in values:
        value = _mapping(raw_value, label)
        value_id = identifier(value)
        if value_id in result:
            raise EntsoeLtDerivedPhysicalCompatibilityError(f"{label} is duplicated")
        result[value_id] = value
    return result


def _tracked_content_ids(
    feature_records: Sequence[Mapping[str, object]],
    feature_selection_document: Mapping[str, object],
    family_mapping_document: Mapping[str, object],
    semantic_binding_document: Mapping[str, object],
    zone_registry_document: Mapping[str, object],
    zone_binding_document: Mapping[str, object],
    lineage_document: Mapping[str, object],
) -> dict[str, str]:
    return {
        "feature_records": canonical_sha256(list(feature_records)),
        "feature_selection": canonical_sha256(feature_selection_document),
        "family_mapping": canonical_sha256(family_mapping_document),
        "semantic_binding": canonical_sha256(semantic_binding_document),
        "zone_registry": canonical_sha256(zone_registry_document),
        "zone_binding": canonical_sha256(zone_binding_document),
        "lineage": canonical_sha256(lineage_document),
    }


def _resolution_for_seconds(seconds: int) -> str:
    matches = [
        resolution
        for resolution, candidate in d270.RESOLUTION_SECONDS.items()
        if candidate == seconds
    ]
    if len(matches) != 1:
        raise EntsoeLtDerivedPhysicalCompatibilityError(
            "target duration is not one supported effective native cadence"
        )
    return matches[0]


def _all_false(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(candidate is False for candidate in value.values())
    )


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeLtDerivedPhysicalCompatibilityError(f"{label} JSON is invalid") from exc
    return _mapping(value, label)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EntsoeLtDerivedPhysicalCompatibilityError("JSON contains duplicate keys")
        result[key] = value
    return result


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeLtDerivedPhysicalCompatibilityError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EntsoeLtDerivedPhysicalCompatibilityError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise EntsoeLtDerivedPhysicalCompatibilityError(f"{label} differs")


def _clean(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise EntsoeLtDerivedPhysicalCompatibilityError(f"{label} must be clean text")
    return value


def _sha(value: object, label: str) -> str:
    text = _clean(value, label)
    if _SHA.fullmatch(text) is None:
        raise EntsoeLtDerivedPhysicalCompatibilityError(f"{label} must be lowercase SHA-256")
    return text


def _utc(value: object, label: str) -> datetime:
    text = _clean(value, label)
    if _UTC.fullmatch(text) is None:
        raise EntsoeLtDerivedPhysicalCompatibilityError(f"{label} must be canonical UTC Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise EntsoeLtDerivedPhysicalCompatibilityError(f"{label} is invalid") from exc
    return _utc_datetime(parsed, label)


def _utc_datetime(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EntsoeLtDerivedPhysicalCompatibilityError(f"{label} must be timezone-aware")
    normalized = value.astimezone(timezone.utc)
    if normalized.microsecond:
        raise EntsoeLtDerivedPhysicalCompatibilityError(f"{label} must have whole-second precision")
    return normalized


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = [
    "ASSESSMENT_SCHEMA",
    "AUTHORITIES",
    "CONTRACT_CONTENT_ID",
    "CONTRACT_PATH",
    "EXECUTION",
    "PASS_STATUS",
    "EntsoeLtDerivedPhysicalCompatibilityError",
    "canonical_sha256",
    "validate_derived_physical_compatibility_contract",
    "validate_derived_physical_compatibility_contract_path",
    "verify_synthetic_entsoe_lt_derived_physical_compatibility",
]

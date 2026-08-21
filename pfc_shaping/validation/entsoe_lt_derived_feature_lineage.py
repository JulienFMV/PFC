"""Validate value-free lineage for derived ENTSO-E LT features.

The first supported transform is a lagged load forecast error.  This module
binds exact D273 feature records and chronology, but deliberately does not open
values, recompute arithmetic, or prove physical unit/zone/cadence semantics.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.validation import entsoe_lt_feature_availability as d273

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = PHASE / "ENTSOE-LT-DERIVED-FEATURE-LINEAGE-CONTRACT-V1.json"

CONTRACT_SCHEMA = "fmv_entsoe_lt_derived_feature_lineage_contract.v1"
CONTRACT_STATUS = "VALUE_FREE_DERIVED_FEATURE_LINEAGE_NO_MODEL_AUTHORITY"
CONTRACT_FILE_SHA256 = "d9157ebce9aaf02950231c2bac21bb28a10ee8577b0450bbbbd6eb8b33447919"
CONTRACT_CONTENT_ID = "7f3fdde5c5d5f192cee06b570063173eb4a55344fe2f871a5441bf140cb8c2ac"
LINEAGE_SCHEMA = "fmv_entsoe_lt_derived_feature_lineage.v1"
ASSESSMENT_SCHEMA = "fmv_entsoe_lt_derived_feature_lineage_assessment.v1"
PASS_STATUS = "PASS_VALUE_FREE_DERIVED_LINEAGE_LATER_COMPOSITE_REQUIRED"
EVIDENCE_CLASS = "SYNTHETIC_TEST_ONLY"
MAX_TRANSFORMS = 100_000

TRANSFORM_KIND = "LOAD_FORECAST_ERROR"
FORMULA = "REALIZED_ACTUAL_MINUS_OPERATIONAL_FORECAST"
FORMULA_ROLE_ORDER = ("REALIZED_ACTUAL", "OPERATIONAL_FORECAST")
D273_DEPENDENCY_ROLE_ORDER = ("OPERATIONAL_FORECAST", "REALIZED_ACTUAL")
SOURCE_FAMILY_BY_ROLE = {
    "REALIZED_ACTUAL": "actual_load",
    "OPERATIONAL_FORECAST": "load_forecast",
}
OUTPUT_ROLE = "FORECAST_ERROR"
OUTPUT_SOURCE_FAMILY = "load_forecast_error"
OUTPUT_USES = ("LAGGED_COVARIATE", "DIAGNOSTIC_ONLY")
MISSING_POLICY = "PRESERVE_NULL"
OUTPUT_SIGN = "POSITIVE_WHEN_ACTUAL_ABOVE_FORECAST"
PHYSICAL_BINDING_STATUS = "REQUIRES_LATER_D280_COMPOSITE"

AUTHORITIES = {
    "real_source_evidence": False,
    "point_in_time": False,
    "semantic_unit_zone_cadence": False,
    "arithmetic_recomputation": False,
    "predictive_value": False,
    "model_input": False,
    "model_selection": False,
    "candidate": False,
    "promotion": False,
    "production": False,
}
LINEAGE_AUTHORITIES = {
    "real_feature_lineage": False,
    "semantic_unit_zone_cadence": False,
    "arithmetic_recomputation": False,
    "point_in_time": False,
    "model_input": False,
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
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{2,127}$")


class EntsoeLtDerivedFeatureLineageError(ValueError):
    """Raised when derived feature lineage is ambiguous or unsafe."""


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


def validate_derived_feature_lineage_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    contract = _mapping(document, "D281 contract")
    _exact(
        contract,
        {
            "schema_version",
            "decision",
            "status",
            "purpose",
            "predecessors",
            "supported_transforms",
            "lineage_document",
            "admission_rules",
            "execution",
            "authorities",
        },
        "D281 contract",
    )
    _equal(contract["schema_version"], CONTRACT_SCHEMA, "contract schema")
    _equal(contract["decision"], "D-20260806-281", "contract decision")
    _equal(contract["status"], CONTRACT_STATUS, "contract status")
    _clean(contract["purpose"], "contract purpose")
    _equal(
        contract["predecessors"],
        {
            "d273_feature_availability_contract_content_id": d273.CONTRACT_CONTENT_ID,
            "d280_real_derived_feature_admission_remains_blocked": True,
        },
        "predecessors",
    )
    _equal(contract["supported_transforms"], [_transform_contract()], "transforms")
    _equal(
        contract["lineage_document"],
        {
            "schema_version": LINEAGE_SCHEMA,
            "required_document_fields": [
                "schema_version",
                "evidence_class",
                "feature_availability_contract_content_id",
                "feature_records_content_id",
                "declared_at_utc",
                "transforms",
                "authorities",
            ],
            "required_transform_fields": [
                "transform_id",
                "transform_kind",
                "output_feature_record_id",
                "input_feature_record_ids_in_formula_order",
                "formula",
                "calculation_created_at_utc",
                "missing_value_policy",
                "output_sign_convention",
                "semantic_unit_zone_and_cadence_binding",
                "reason",
            ],
            "maximum_transforms": MAX_TRANSFORMS,
        },
        "lineage document contract",
    )
    _equal(
        contract["admission_rules"],
        {
            "all_referenced_d273_records_must_be_structurally_eligible": True,
            "every_forecast_error_record_must_be_bound_exactly_once": True,
            "input_and_output_feature_record_ids_must_be_content_hashes": True,
            "input_roles_families_target_origin_and_availability_must_match": True,
            "duplicate_or_self_dependency_is_admissible": False,
            "formula_or_operand_order_may_be_inferred": False,
            "semantic_unit_zone_and_cadence_proven_by_this_contract": False,
            "raw_values_opened_or_recomputed": False,
            "synthetic_structural_pass_grants_model_input_authority": False,
            "monthly_effects_may_rewrite_solver_mean": False,
        },
        "admission rules",
    )
    _equal(contract["execution"], EXECUTION, "execution")
    _equal(contract["authorities"], AUTHORITIES, "authorities")
    _equal(canonical_sha256(contract), CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "supported_transform_count": 1,
        "semantic_unit_zone_cadence_proven": False,
        "arithmetic_recomputed": False,
        "model_input_authorized": False,
        "production_authorized": False,
        **EXECUTION,
    }


def validate_derived_feature_lineage_contract_path(
    path: str | Path,
) -> dict[str, object]:
    contract_path = Path(path)
    if not contract_path.is_absolute():
        raise EntsoeLtDerivedFeatureLineageError("contract path must be absolute")
    try:
        payload = read_stable_single_link_file(
            contract_path,
            label="D281 ENTSO-E derived feature lineage contract",
            max_bytes=100_000,
        )
    except (OSError, ValueError) as exc:
        raise EntsoeLtDerivedFeatureLineageError("contract read failed") from exc
    if hashlib.sha256(payload).hexdigest() != CONTRACT_FILE_SHA256:
        raise EntsoeLtDerivedFeatureLineageError("contract SHA-256 differs")
    document = _strict_json(payload, "D281 contract")
    return validate_derived_feature_lineage_contract(document)


def assess_derived_feature_lineage(
    feature_records: Sequence[Mapping[str, object]],
    lineage_document: Mapping[str, object],
    *,
    reference_time_utc: datetime,
) -> dict[str, object]:
    """Bind derived records to exact primitive records without opening values."""

    validate_derived_feature_lineage_contract_path(CONTRACT_PATH)
    reference = _utc_datetime(reference_time_utc, "reference time")
    availability = d273.assess_feature_availability(feature_records)
    records = [_mapping(record, "feature record") for record in feature_records]
    outcomes = availability["outcomes"]
    if not isinstance(outcomes, tuple) or len(outcomes) != len(records):
        raise EntsoeLtDerivedFeatureLineageError("feature outcomes differ")
    record_by_id: dict[str, tuple[Mapping[str, object], Mapping[str, object]]] = {}
    for record, raw_outcome in zip(records, outcomes, strict=True):
        record_id = feature_record_id(record)
        if record_id in record_by_id:
            raise EntsoeLtDerivedFeatureLineageError("feature record hash is duplicated")
        record_by_id[record_id] = (record, _mapping(raw_outcome, "feature outcome"))

    document = _mapping(lineage_document, "lineage document")
    _exact(
        document,
        {
            "schema_version",
            "evidence_class",
            "feature_availability_contract_content_id",
            "feature_records_content_id",
            "declared_at_utc",
            "transforms",
            "authorities",
        },
        "lineage document",
    )
    _equal(document["schema_version"], LINEAGE_SCHEMA, "lineage schema")
    _equal(document["evidence_class"], EVIDENCE_CLASS, "lineage evidence class")
    _equal(
        document["feature_availability_contract_content_id"],
        d273.CONTRACT_CONTENT_ID,
        "feature availability contract content ID",
    )
    records_content_id = canonical_sha256(list(feature_records))
    _equal(
        document["feature_records_content_id"],
        records_content_id,
        "feature records content ID",
    )
    _equal(document["authorities"], LINEAGE_AUTHORITIES, "lineage authorities")
    declared_at = _utc(document["declared_at_utc"], "declared_at_utc")
    if declared_at > reference:
        raise EntsoeLtDerivedFeatureLineageError("lineage declaration follows reference")
    raw_transforms = document["transforms"]
    if (
        not isinstance(raw_transforms, list)
        or not raw_transforms
        or len(raw_transforms) > MAX_TRANSFORMS
    ):
        raise EntsoeLtDerivedFeatureLineageError("lineage transforms are invalid")

    expected_outputs = {
        record_id
        for record_id, (record, _outcome) in record_by_id.items()
        if record["information_role"] == OUTPUT_ROLE
    }
    if not expected_outputs:
        raise EntsoeLtDerivedFeatureLineageError("no forecast-error record exists")
    seen_outputs: set[str] = set()
    seen_transform_ids: set[str] = set()
    seen_input_pairs: set[tuple[str, str]] = set()
    profiles: list[dict[str, object]] = []
    for position, raw_transform in enumerate(raw_transforms, start=1):
        transform = _mapping(raw_transform, f"transform {position}")
        _exact(
            transform,
            {
                "transform_id",
                "transform_kind",
                "output_feature_record_id",
                "input_feature_record_ids_in_formula_order",
                "formula",
                "calculation_created_at_utc",
                "missing_value_policy",
                "output_sign_convention",
                "semantic_unit_zone_and_cadence_binding",
                "reason",
            },
            f"transform {position}",
        )
        transform_id = _identifier(transform["transform_id"], "transform ID")
        if transform_id in seen_transform_ids:
            raise EntsoeLtDerivedFeatureLineageError("transform ID is duplicated")
        seen_transform_ids.add(transform_id)
        _equal(transform["transform_kind"], TRANSFORM_KIND, "transform kind")
        _equal(transform["formula"], FORMULA, "transform formula")
        _equal(transform["missing_value_policy"], MISSING_POLICY, "missing policy")
        _equal(transform["output_sign_convention"], OUTPUT_SIGN, "output sign")
        _equal(
            transform["semantic_unit_zone_and_cadence_binding"],
            PHYSICAL_BINDING_STATUS,
            "physical binding status",
        )
        _clean(transform["reason"], "transform reason")

        output_id = _sha(transform["output_feature_record_id"], "output record ID")
        if output_id not in record_by_id:
            raise EntsoeLtDerivedFeatureLineageError("unknown output feature record")
        if output_id in seen_outputs:
            raise EntsoeLtDerivedFeatureLineageError("forecast-error output is duplicated")
        seen_outputs.add(output_id)
        output, output_outcome = record_by_id[output_id]
        _eligible(output_outcome, "output")
        _equal(output["information_role"], OUTPUT_ROLE, "output role")
        _equal(output["source_family"], OUTPUT_SOURCE_FAMILY, "output source family")
        if output["requested_use"] not in OUTPUT_USES:
            raise EntsoeLtDerivedFeatureLineageError("output requested use is invalid")
        _equal(output["missing_value_policy"], MISSING_POLICY, "output missing policy")

        raw_input_ids = transform["input_feature_record_ids_in_formula_order"]
        if not isinstance(raw_input_ids, list) or len(raw_input_ids) != 2:
            raise EntsoeLtDerivedFeatureLineageError("formula inputs must contain two IDs")
        input_ids = tuple(_sha(value, "input record ID") for value in raw_input_ids)
        if len(set(input_ids)) != 2 or output_id in input_ids:
            raise EntsoeLtDerivedFeatureLineageError("self or duplicate dependency")
        if input_ids in seen_input_pairs:
            raise EntsoeLtDerivedFeatureLineageError("input dependency pair is duplicated")
        seen_input_pairs.add(input_ids)
        inputs: list[Mapping[str, object]] = []
        for expected_role, input_id in zip(FORMULA_ROLE_ORDER, input_ids, strict=True):
            if input_id not in record_by_id:
                raise EntsoeLtDerivedFeatureLineageError("unknown input feature record")
            record, outcome = record_by_id[input_id]
            _eligible(outcome, "input")
            _equal(record["information_role"], expected_role, "formula input role")
            _equal(
                record["source_family"],
                SOURCE_FAMILY_BY_ROLE[expected_role],
                "formula input source family",
            )
            inputs.append(record)

        for field in ("target_start_utc", "target_end_utc", "origin_utc"):
            _equal(inputs[0][field], output[field], f"actual/output {field}")
            _equal(inputs[1][field], output[field], f"forecast/output {field}")
        actual_available = _utc(inputs[0]["available_at_utc"], "actual availability")
        forecast_available = _utc(inputs[1]["available_at_utc"], "forecast availability")
        _equal(
            output["dependency_roles"],
            list(D273_DEPENDENCY_ROLE_ORDER),
            "output dependency roles",
        )
        _equal(
            output["dependency_available_at_utc"],
            [inputs[1]["available_at_utc"], inputs[0]["available_at_utc"]],
            "output dependency availability",
        )
        output_available = _utc(output["available_at_utc"], "output availability")
        if output_available != max(actual_available, forecast_available):
            raise EntsoeLtDerivedFeatureLineageError(
                "output availability differs from exact linked inputs"
            )
        calculation_created = _utc(
            transform["calculation_created_at_utc"],
            "calculation_created_at_utc",
        )
        if calculation_created < output_available:
            raise EntsoeLtDerivedFeatureLineageError("calculation predates input availability")
        if calculation_created > declared_at:
            raise EntsoeLtDerivedFeatureLineageError("calculation follows lineage declaration")
        profiles.append(
            {
                "transform_id_sha256": hashlib.sha256(transform_id.encode("utf-8")).hexdigest(),
                "transform_kind": TRANSFORM_KIND,
                "formula": FORMULA,
                "output_feature_record_id": output_id,
                "input_feature_record_ids_in_formula_order": list(input_ids),
                "output_requested_use": output["requested_use"],
                "semantic_unit_zone_and_cadence_proven": False,
                "arithmetic_recomputed": False,
                "raw_values_opened": False,
            }
        )

    if seen_outputs != expected_outputs:
        raise EntsoeLtDerivedFeatureLineageError(
            "forecast-error records are not bound exactly once"
        )
    profiles.sort(key=lambda profile: str(profile["transform_id_sha256"]))
    return {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": PASS_STATUS,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "lineage_content_id": canonical_sha256(document),
        "feature_records_content_id": records_content_id,
        "feature_availability_assessment_content_id": canonical_sha256(availability),
        "transform_count": len(profiles),
        "forecast_error_record_count": len(expected_outputs),
        "all_forecast_error_records_bound_once": True,
        "all_referenced_records_structurally_eligible": True,
        "exact_input_identity_and_availability_bound": True,
        "profiles": profiles,
        "raw_feature_ids_persisted": False,
        "raw_values_opened": False,
        "arithmetic_recomputed": False,
        "semantic_unit_zone_and_cadence_proven": False,
        "d280_derived_model_input_admitted": False,
        "authorities": dict(AUTHORITIES),
        **EXECUTION,
    }


def assess_derived_feature_lineage_path(
    feature_records: Sequence[Mapping[str, object]],
    lineage_path: str | Path,
    *,
    reference_time_utc: datetime,
) -> dict[str, object]:
    """Stably read one lineage document before applying the value-free gate."""

    path = Path(lineage_path)
    if not path.is_absolute():
        raise EntsoeLtDerivedFeatureLineageError("lineage path must be absolute")
    try:
        payload = read_stable_single_link_file(
            path,
            label="D281 ENTSO-E derived feature lineage document",
            max_bytes=10_000_000,
        )
    except (OSError, ValueError) as exc:
        raise EntsoeLtDerivedFeatureLineageError("lineage document read failed") from exc
    document = _strict_json(payload, "lineage document")
    return assess_derived_feature_lineage(
        feature_records,
        document,
        reference_time_utc=reference_time_utc,
    )


def _transform_contract() -> dict[str, object]:
    return {
        "transform_kind": TRANSFORM_KIND,
        "formula": FORMULA,
        "input_roles_in_formula_order": list(FORMULA_ROLE_ORDER),
        "d273_dependency_roles_order": list(D273_DEPENDENCY_ROLE_ORDER),
        "required_source_families_by_formula_role": dict(SOURCE_FAMILY_BY_ROLE),
        "output_information_role": OUTPUT_ROLE,
        "output_source_family": OUTPUT_SOURCE_FAMILY,
        "permitted_output_uses": list(OUTPUT_USES),
        "target_interval_must_match_all_records": True,
        "origin_must_match_all_records": True,
        "output_availability_is_maximum_input_availability": True,
        "calculation_cannot_precede_input_availability": True,
        "missing_input_or_output_value_policy": MISSING_POLICY,
        "output_sign_convention": OUTPUT_SIGN,
        "semantic_unit_zone_and_cadence_binding": PHYSICAL_BINDING_STATUS,
    }


def _eligible(outcome: Mapping[str, object], label: str) -> None:
    if outcome["status"] != "STRUCTURALLY_ELIGIBLE_NO_MODEL_AUTHORITY":
        raise EntsoeLtDerivedFeatureLineageError(
            f"{label} feature record is not structurally eligible"
        )


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeLtDerivedFeatureLineageError(f"{label} JSON is invalid") from exc
    return _mapping(value, label)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise json.JSONDecodeError("duplicate keys are forbidden", key, 0)
        result[key] = value
    return result


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeLtDerivedFeatureLineageError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EntsoeLtDerivedFeatureLineageError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoeLtDerivedFeatureLineageError(f"{label} differs")


def _clean(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise EntsoeLtDerivedFeatureLineageError(f"{label} must be clean text")
    return value


def _identifier(value: object, label: str) -> str:
    text = _clean(value, label)
    if _IDENTIFIER.fullmatch(text) is None:
        raise EntsoeLtDerivedFeatureLineageError(f"{label} is invalid")
    return text


def _sha(value: object, label: str) -> str:
    text = _clean(value, label)
    if _SHA.fullmatch(text) is None:
        raise EntsoeLtDerivedFeatureLineageError(f"{label} must be lowercase SHA-256")
    return text


def _utc(value: object, label: str) -> datetime:
    text = _clean(value, label)
    if not text.endswith("Z"):
        raise EntsoeLtDerivedFeatureLineageError(f"{label} must be UTC Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise EntsoeLtDerivedFeatureLineageError(f"{label} is invalid") from exc
    return _utc_datetime(parsed, label)


def _utc_datetime(value: datetime, label: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timezone.utc.utcoffset(value)
    ):
        raise EntsoeLtDerivedFeatureLineageError(f"{label} must be UTC-aware")
    return value.astimezone(timezone.utc)


__all__ = [
    "ASSESSMENT_SCHEMA",
    "AUTHORITIES",
    "CONTRACT_CONTENT_ID",
    "CONTRACT_FILE_SHA256",
    "CONTRACT_PATH",
    "CONTRACT_SCHEMA",
    "EntsoeLtDerivedFeatureLineageError",
    "FORMULA",
    "FORMULA_ROLE_ORDER",
    "LINEAGE_AUTHORITIES",
    "LINEAGE_SCHEMA",
    "PASS_STATUS",
    "PHYSICAL_BINDING_STATUS",
    "assess_derived_feature_lineage",
    "assess_derived_feature_lineage_path",
    "canonical_sha256",
    "feature_record_id",
    "validate_derived_feature_lineage_contract",
    "validate_derived_feature_lineage_contract_path",
]

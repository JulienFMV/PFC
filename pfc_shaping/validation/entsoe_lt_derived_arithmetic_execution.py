"""Execute synthetic load forecast-error arithmetic after D282.

The executor accepts canonical decimal text, never binary floats. It binds
every operand to the exact D282 physical profile, preserves nulls and returns
an in-memory output artifact plus a value-free assessment. Real values and
model authority remain outside this gate.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_EVEN, Context, Decimal, InvalidOperation, localcontext
from pathlib import Path

from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.validation import entsoe_lt_derived_feature_lineage as d281
from pfc_shaping.validation import entsoe_lt_derived_physical_compatibility as d282

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = PHASE / "ENTSOE-LT-DERIVED-ARITHMETIC-EXECUTION-CONTRACT-V1.json"

CONTRACT_SCHEMA = "fmv_entsoe_lt_derived_arithmetic_execution_contract.v1"
CONTRACT_STATUS = "SYNTHETIC_DECIMAL_ARITHMETIC_NO_REAL_OR_MODEL_AUTHORITY"
CONTRACT_FILE_SHA256 = "d0c2ce35c10c30d772f57cccc06c24377161bf8353b65eeadb751bec064370b8"
CONTRACT_CONTENT_ID = "92946ffd261fd2724c165e530387a13b12ed5d7b6f24a821f0bcf1c5cc3b5f08"
OPERAND_SCHEMA = "fmv_entsoe_lt_synthetic_decimal_operand_artifact.v1"
OUTPUT_SCHEMA = "fmv_entsoe_lt_synthetic_forecast_error_artifact.v1"
ASSESSMENT_SCHEMA = "fmv_entsoe_lt_derived_arithmetic_execution_assessment.v1"
PASS_STATUS = "PASS_SYNTHETIC_DECIMAL_ARITHMETIC_NO_REAL_OR_MODEL_AUTHORITY"
EVIDENCE_CLASS = "SYNTHETIC_TEST_ONLY"
ACTUAL_ROLE = "REALIZED_ACTUAL"
FORECAST_ROLE = "OPERATIONAL_FORECAST"
MAX_ROWS = 100_000
MAX_SCALE = 6
MAX_SIGNIFICANT_DIGITS = 18
MAX_ABS_MW = Decimal("1000000000")
MAX_ABS_OUTPUT_MW = Decimal("2000000000")
DECIMAL_PRECISION = 34

ALGORITHM_SPEC = {
    "transform_kind": d281.TRANSFORM_KIND,
    "formula": d281.FORMULA,
    "formula_role_order": list(d281.FORMULA_ROLE_ORDER),
    "output_sign_convention": d281.OUTPUT_SIGN,
    "value_encoding": "CANONICAL_BASE10_TEXT_OR_NULL",
    "decimal_precision": DECIMAL_PRECISION,
    "rounding_mode": "ROUND_HALF_EVEN",
    "missing_value_policy": d281.MISSING_POLICY,
    "negative_zero_is_canonicalized_to_zero": True,
}

AUTHORITIES = {
    "real_source_evidence": False,
    "real_arithmetic_correctness": False,
    "point_in_time": False,
    "predictive_value": False,
    "model_input": False,
    "model_selection": False,
    "candidate": False,
    "promotion": False,
    "production": False,
}
ARTIFACT_AUTHORITIES = {
    "real_values": False,
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
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_DECIMAL = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?$")
_CONTEXT = Context(prec=DECIMAL_PRECISION, rounding=ROUND_HALF_EVEN)


class EntsoeLtDerivedArithmeticExecutionError(ValueError):
    """Raised when synthetic derived arithmetic is ambiguous or unsafe."""


@dataclass(frozen=True)
class SyntheticArithmeticExecutionResult:
    """Value-free assessment and separately content-addressed synthetic output."""

    assessment: dict[str, object]
    assessment_content_id: str
    output_artifact: dict[str, object]
    output_artifact_content_id: str


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


ALGORITHM_CONTENT_ID = canonical_sha256(ALGORITHM_SPEC)


def validate_derived_arithmetic_execution_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    contract = _mapping(document, "D283 contract")
    _exact(
        contract,
        {
            "schema_version",
            "decision",
            "status",
            "purpose",
            "predecessor",
            "operand_artifacts",
            "execution_rules",
            "integrity",
            "execution",
            "authorities",
        },
        "D283 contract",
    )
    _equal(contract["schema_version"], CONTRACT_SCHEMA, "contract schema")
    _equal(contract["decision"], "D-20260806-283", "contract decision")
    _equal(contract["status"], CONTRACT_STATUS, "contract status")
    _clean(contract["purpose"], "contract purpose")
    _equal(
        contract["predecessor"],
        {
            "d282_contract_content_id": d282.CONTRACT_CONTENT_ID,
            "required_d282_status": d282.PASS_STATUS,
            "required_transform_kind": d281.TRANSFORM_KIND,
            "formula": d281.FORMULA,
            "formula_role_order": list(d281.FORMULA_ROLE_ORDER),
            "output_sign_convention": d281.OUTPUT_SIGN,
        },
        "predecessor",
    )
    _equal(
        contract["operand_artifacts"],
        {
            "schema_version": OPERAND_SCHEMA,
            "evidence_class": EVIDENCE_CLASS,
            "one_artifact_per_formula_role": True,
            "row_grain": [
                "feature_record_id",
                "target_start_utc",
                "target_end_utc",
            ],
            "value_field": "value_decimal_mw",
            "value_encoding": "CANONICAL_BASE10_TEXT_OR_NULL",
            "maximum_scale": MAX_SCALE,
            "maximum_significant_digits": MAX_SIGNIFICANT_DIGITS,
            "maximum_absolute_mw": str(MAX_ABS_MW),
            "binary_floating_point_is_admissible": False,
            "duplicate_missing_or_orphan_rows_are_admissible": False,
        },
        "operand artifacts",
    )
    _equal(
        contract["execution_rules"],
        {
            "arithmetic_engine": "PYTHON_DECIMAL_IEEE_754_DECIMAL128_CONTEXT",
            "decimal_precision": DECIMAL_PRECISION,
            "rounding_mode": "ROUND_HALF_EVEN",
            "maximum_absolute_output_mw": str(MAX_ABS_OUTPUT_MW),
            "missing_value_policy": d281.MISSING_POLICY,
            "negative_zero_is_canonicalized_to_zero": True,
            "output_rows_are_sorted_by_output_feature_record_id": True,
            "output_artifact_is_content_addressed": True,
            "assessment_may_persist_operand_or_output_values": False,
            "monthly_effect_policy": "ZERO_MEAN_WITHIN_SOLVER_MONTH_IF_LATER_ADMITTED",
        },
        "execution rules",
    )
    _equal(
        contract["integrity"],
        {
            "d282_assessment_must_remain_content_stable": True,
            "operand_artifacts_must_remain_content_stable": True,
            "every_d282_profile_must_execute_exactly_once": True,
            "target_intervals_must_match_between_operands": True,
            "target_duration_must_match_d282": True,
            "unit_zone_resolution_and_record_ids_must_match_d282": True,
            "synthetic_pass_grants_real_arithmetic_or_model_authority": False,
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
        "algorithm_content_id": ALGORITHM_CONTENT_ID,
        "real_arithmetic_authorized": False,
        "model_input_authorized": False,
        **EXECUTION,
    }


def validate_derived_arithmetic_execution_contract_path(
    path: str | Path,
) -> dict[str, object]:
    contract_path = Path(path)
    if not contract_path.is_absolute():
        raise EntsoeLtDerivedArithmeticExecutionError("contract path must be absolute")
    try:
        payload = read_stable_single_link_file(
            contract_path,
            label="D283 ENTSO-E derived arithmetic execution contract",
            max_bytes=100_000,
        )
    except (OSError, ValueError) as exc:
        raise EntsoeLtDerivedArithmeticExecutionError("contract read failed") from exc
    if hashlib.sha256(payload).hexdigest() != CONTRACT_FILE_SHA256:
        raise EntsoeLtDerivedArithmeticExecutionError("contract SHA-256 differs")
    return validate_derived_arithmetic_execution_contract(_strict_json(payload, "D283 contract"))


def execute_synthetic_load_forecast_error(
    *,
    contract_path: str | Path,
    d282_assessment: Mapping[str, object],
    actual_operand_artifact: Mapping[str, object],
    forecast_operand_artifact: Mapping[str, object],
    reference_time_utc: datetime,
) -> SyntheticArithmeticExecutionResult:
    """Execute canonical-decimal synthetic arithmetic and hash the output."""

    validate_derived_arithmetic_execution_contract_path(contract_path)
    reference = _utc_datetime(reference_time_utc, "reference time")
    d282_content_id = canonical_sha256(d282_assessment)
    actual_content_id = canonical_sha256(actual_operand_artifact)
    forecast_content_id = canonical_sha256(forecast_operand_artifact)

    profiles = _validate_d282_assessment(d282_assessment, reference)
    actual_rows = _validate_operand_artifact(
        actual_operand_artifact,
        ACTUAL_ROLE,
        d282_content_id,
        reference,
    )
    forecast_rows = _validate_operand_artifact(
        forecast_operand_artifact,
        FORECAST_ROLE,
        d282_content_id,
        reference,
    )
    expected_actual_ids: set[str] = set()
    expected_forecast_ids: set[str] = set()
    output_rows: list[dict[str, object]] = []
    computed_count = 0
    null_count = 0

    for position, raw_profile in enumerate(profiles, start=1):
        profile = _mapping(raw_profile, f"D282 profile {position}")
        input_ids = profile.get("input_feature_record_ids_in_formula_order")
        if not isinstance(input_ids, list) or len(input_ids) != 2:
            raise EntsoeLtDerivedArithmeticExecutionError(
                "D282 formula input identities are invalid"
            )
        actual_id = _sha(input_ids[0], "actual feature record ID")
        forecast_id = _sha(input_ids[1], "forecast feature record ID")
        output_id = _sha(profile["output_feature_record_id"], "output feature record ID")
        if actual_id in expected_actual_ids or forecast_id in expected_forecast_ids:
            raise EntsoeLtDerivedArithmeticExecutionError(
                "D282 primitive is reused across arithmetic profiles"
            )
        expected_actual_ids.add(actual_id)
        expected_forecast_ids.add(forecast_id)
        if actual_id not in actual_rows or forecast_id not in forecast_rows:
            raise EntsoeLtDerivedArithmeticExecutionError("D282 arithmetic operand row is missing")
        actual = actual_rows[actual_id]
        forecast = forecast_rows[forecast_id]
        _bind_operand_to_profile(actual, profile, "actual")
        _bind_operand_to_profile(forecast, profile, "forecast")
        for field in ("target_start_utc", "target_end_utc"):
            _equal(actual[field], forecast[field], f"operand {field}")
        start = _utc(actual["target_start_utc"], "target start")
        end = _utc(actual["target_end_utc"], "target end")
        duration = int((end - start).total_seconds())
        _equal(duration, profile["target_duration_seconds"], "target duration")

        actual_value = actual["parsed_value"]
        forecast_value = forecast["parsed_value"]
        if actual_value is None or forecast_value is None:
            output_value = None
            null_count += 1
        else:
            with localcontext(_CONTEXT):
                output_decimal = actual_value - forecast_value
            if not output_decimal.is_finite() or abs(output_decimal) > MAX_ABS_OUTPUT_MW:
                raise EntsoeLtDerivedArithmeticExecutionError("derived output magnitude is invalid")
            output_value = _format_decimal(output_decimal)
            computed_count += 1
        output_rows.append(
            {
                "output_feature_record_id": output_id,
                "target_start_utc": actual["target_start_utc"],
                "target_end_utc": actual["target_end_utc"],
                "logical_zone_sha256": profile["logical_zone_sha256"],
                "canonical_unit": "MW",
                "effective_native_resolution": profile["effective_native_resolution"],
                "value_decimal_mw": output_value,
                "input_null_propagated": output_value is None,
            }
        )

    if set(actual_rows) != expected_actual_ids or set(forecast_rows) != expected_forecast_ids:
        raise EntsoeLtDerivedArithmeticExecutionError(
            "operand artifact contains an orphan or substituted row"
        )
    output_rows.sort(key=lambda row: str(row["output_feature_record_id"]))
    output_artifact = {
        "schema_version": OUTPUT_SCHEMA,
        "evidence_class": EVIDENCE_CLASS,
        "algorithm_content_id": ALGORITHM_CONTENT_ID,
        "d282_assessment_content_id": d282_content_id,
        "actual_operand_artifact_content_id": actual_content_id,
        "forecast_operand_artifact_content_id": forecast_content_id,
        "created_at_utc": _utc_text(reference),
        "rows": output_rows,
        "authorities": dict(ARTIFACT_AUTHORITIES),
    }
    output_content_id = canonical_sha256(output_artifact)

    if canonical_sha256(d282_assessment) != d282_content_id:
        raise EntsoeLtDerivedArithmeticExecutionError(
            "D282 assessment changed during arithmetic execution"
        )
    if canonical_sha256(actual_operand_artifact) != actual_content_id:
        raise EntsoeLtDerivedArithmeticExecutionError(
            "actual operand artifact changed during arithmetic execution"
        )
    if canonical_sha256(forecast_operand_artifact) != forecast_content_id:
        raise EntsoeLtDerivedArithmeticExecutionError(
            "forecast operand artifact changed during arithmetic execution"
        )

    assessment = {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": PASS_STATUS,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "reference_time_utc": _utc_text(reference),
        "algorithm_content_id": ALGORITHM_CONTENT_ID,
        "d282_assessment_content_id": d282_content_id,
        "actual_operand_artifact_content_id": actual_content_id,
        "forecast_operand_artifact_content_id": forecast_content_id,
        "output_artifact_content_id": output_content_id,
        "transform_count": len(output_rows),
        "computed_value_count": computed_count,
        "null_propagated_count": null_count,
        "all_operands_bound_to_d282": True,
        "all_targets_and_physical_profiles_match": True,
        "canonical_decimal_arithmetic_executed": True,
        "missing_value_policy": d281.MISSING_POLICY,
        "output_sign_convention": d281.OUTPUT_SIGN,
        "operand_or_output_values_persisted_in_assessment": False,
        "real_values_opened": False,
        "real_arithmetic_authorized": False,
        "model_input_authorized": False,
        "authorities": dict(AUTHORITIES),
        **EXECUTION,
    }
    return SyntheticArithmeticExecutionResult(
        assessment=assessment,
        assessment_content_id=canonical_sha256(assessment),
        output_artifact=output_artifact,
        output_artifact_content_id=output_content_id,
    )


def _validate_d282_assessment(
    assessment: Mapping[str, object],
    reference: datetime,
) -> list[Mapping[str, object]]:
    document = _mapping(assessment, "D282 assessment")
    _exact(
        document,
        {
            "schema_version",
            "status",
            "contract_content_id",
            "reference_time_utc",
            "bindings",
            "transform_count",
            "all_primitives_selected_once",
            "all_temporal_zone_bindings_complete",
            "all_logical_zones_match",
            "all_canonical_units_are_mw",
            "all_effective_native_cadences_match",
            "d280_lineage_blocker_structurally_discharged",
            "profiles",
            "clear_series_or_feature_ids_persisted",
            "real_or_feature_values_opened",
            "arithmetic_recomputed",
            "derived_model_input_authorized",
            "authorities",
            *EXECUTION,
        },
        "D282 assessment",
    )
    _equal(document.get("schema_version"), d282.ASSESSMENT_SCHEMA, "D282 schema")
    _equal(document.get("status"), d282.PASS_STATUS, "D282 status")
    _equal(
        document.get("contract_content_id"),
        d282.CONTRACT_CONTENT_ID,
        "D282 contract content ID",
    )
    _equal(document.get("authorities"), d282.AUTHORITIES, "D282 authorities")
    for field in (
        "all_primitives_selected_once",
        "all_temporal_zone_bindings_complete",
        "all_logical_zones_match",
        "all_canonical_units_are_mw",
        "all_effective_native_cadences_match",
        "d280_lineage_blocker_structurally_discharged",
    ):
        _equal(document.get(field), True, f"D282 {field}")
    for field in (
        "clear_series_or_feature_ids_persisted",
        "real_or_feature_values_opened",
        "arithmetic_recomputed",
        "derived_model_input_authorized",
    ):
        _equal(document.get(field), False, f"D282 {field}")
    for field, expected in EXECUTION.items():
        _equal(document.get(field), expected, f"D282 {field}")
    d282_reference = _utc(document.get("reference_time_utc"), "D282 reference time")
    if d282_reference > reference:
        raise EntsoeLtDerivedArithmeticExecutionError("D283 reference predates D282 assessment")
    raw_profiles = document.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles or len(raw_profiles) > MAX_ROWS:
        raise EntsoeLtDerivedArithmeticExecutionError("D282 profiles are invalid")
    _equal(document.get("transform_count"), len(raw_profiles), "D282 transform count")
    bindings = _mapping(document["bindings"], "D282 bindings")
    _exact(
        bindings,
        {
            "d280_assessment_content_id",
            "d281_assessment_content_id",
            "feature_records_content_id",
            "feature_selection_content_id",
            "family_mapping_content_id",
            "semantic_binding_content_id",
            "zone_registry_content_id",
            "zone_binding_content_id",
            "lineage_content_id",
        },
        "D282 bindings",
    )
    for name, value in bindings.items():
        _sha(value, f"D282 binding {name}")
    profiles: list[Mapping[str, object]] = []
    for position, raw_profile in enumerate(raw_profiles, start=1):
        profile = _mapping(raw_profile, f"D282 profile {position}")
        _exact(
            profile,
            {
                "transform_id_sha256",
                "output_feature_record_id",
                "input_feature_record_ids_in_formula_order",
                "actual_series_id_sha256",
                "forecast_series_id_sha256",
                "logical_zone_sha256",
                "canonical_unit",
                "effective_native_resolution",
                "target_duration_seconds",
                "temporal_zone_binding_complete",
                "derived_output_inherits_physical_profile",
                "raw_ids_persisted",
                "raw_or_feature_values_opened",
            },
            f"D282 profile {position}",
        )
        _sha(profile["transform_id_sha256"], "D282 transform ID hash")
        output_id = _sha(profile["output_feature_record_id"], "D282 output record ID")
        input_ids = profile["input_feature_record_ids_in_formula_order"]
        if not isinstance(input_ids, list) or len(input_ids) != 2:
            raise EntsoeLtDerivedArithmeticExecutionError(
                "D282 formula input identities are invalid"
            )
        clean_inputs = [_sha(value, "D282 input record ID") for value in input_ids]
        if len(set(clean_inputs)) != 2 or output_id in clean_inputs:
            raise EntsoeLtDerivedArithmeticExecutionError(
                "D282 profile record identities are not distinct"
            )
        _sha(profile["actual_series_id_sha256"], "D282 actual series hash")
        _sha(profile["forecast_series_id_sha256"], "D282 forecast series hash")
        _sha(profile["logical_zone_sha256"], "D282 logical zone hash")
        _equal(profile["canonical_unit"], "MW", "D282 canonical unit")
        _clean(profile["effective_native_resolution"], "D282 native resolution")
        duration = profile["target_duration_seconds"]
        if isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0:
            raise EntsoeLtDerivedArithmeticExecutionError("D282 target duration is invalid")
        for field in (
            "temporal_zone_binding_complete",
            "derived_output_inherits_physical_profile",
        ):
            _equal(profile[field], True, f"D282 profile {field}")
        for field in ("raw_ids_persisted", "raw_or_feature_values_opened"):
            _equal(profile[field], False, f"D282 profile {field}")
        profiles.append(profile)
    return profiles


def _validate_operand_artifact(
    artifact: Mapping[str, object],
    role: str,
    d282_content_id: str,
    reference: datetime,
) -> dict[str, dict[str, object]]:
    document = _mapping(artifact, f"{role} operand artifact")
    _exact(
        document,
        {
            "schema_version",
            "evidence_class",
            "role",
            "created_at_utc",
            "d282_assessment_content_id",
            "rows",
            "authorities",
        },
        f"{role} operand artifact",
    )
    _equal(document["schema_version"], OPERAND_SCHEMA, "operand schema")
    _equal(document["evidence_class"], EVIDENCE_CLASS, "operand evidence class")
    _equal(document["role"], role, "operand role")
    _equal(
        document["d282_assessment_content_id"],
        d282_content_id,
        "operand D282 assessment content ID",
    )
    _equal(document["authorities"], ARTIFACT_AUTHORITIES, "operand authorities")
    if _utc(document["created_at_utc"], "operand creation time") > reference:
        raise EntsoeLtDerivedArithmeticExecutionError(
            "operand artifact was created after reference time"
        )
    raw_rows = document["rows"]
    if not isinstance(raw_rows, list) or not raw_rows or len(raw_rows) > MAX_ROWS:
        raise EntsoeLtDerivedArithmeticExecutionError("operand rows are invalid")
    rows: dict[str, dict[str, object]] = {}
    for position, raw_row in enumerate(raw_rows, start=1):
        row = _mapping(raw_row, f"operand row {position}")
        _exact(
            row,
            {
                "feature_record_id",
                "target_start_utc",
                "target_end_utc",
                "series_id_sha256",
                "logical_zone_sha256",
                "canonical_unit",
                "effective_native_resolution",
                "value_decimal_mw",
            },
            f"operand row {position}",
        )
        record_id = _sha(row["feature_record_id"], "operand feature record ID")
        if record_id in rows:
            raise EntsoeLtDerivedArithmeticExecutionError("operand feature record is duplicated")
        start = _utc(row["target_start_utc"], "operand target start")
        end = _utc(row["target_end_utc"], "operand target end")
        if start >= end:
            raise EntsoeLtDerivedArithmeticExecutionError(
                "operand target interval is empty or reversed"
            )
        parsed_value = _decimal_or_null(row["value_decimal_mw"], "operand value")
        rows[record_id] = {**dict(row), "parsed_value": parsed_value}
    return rows


def _bind_operand_to_profile(
    row: Mapping[str, object],
    profile: Mapping[str, object],
    label: str,
) -> None:
    series_field = f"{label}_series_id_sha256"
    _equal(row["series_id_sha256"], profile[series_field], f"{label} series hash")
    _equal(
        row["logical_zone_sha256"],
        profile["logical_zone_sha256"],
        f"{label} logical zone hash",
    )
    _equal(row["canonical_unit"], profile["canonical_unit"], f"{label} unit")
    _equal(
        row["effective_native_resolution"],
        profile["effective_native_resolution"],
        f"{label} effective native resolution",
    )
    _sha(row["series_id_sha256"], f"{label} series hash")
    _sha(row["logical_zone_sha256"], f"{label} logical zone hash")


def _decimal_or_null(value: object, label: str) -> Decimal | None:
    if value is None:
        return None
    if not isinstance(value, str) or _DECIMAL.fullmatch(value) is None:
        raise EntsoeLtDerivedArithmeticExecutionError(
            f"{label} must be canonical base-10 text or null"
        )
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise EntsoeLtDerivedArithmeticExecutionError(f"{label} is invalid") from exc
    if not parsed.is_finite():
        raise EntsoeLtDerivedArithmeticExecutionError(f"{label} magnitude is invalid")
    if max(0, -parsed.as_tuple().exponent) > MAX_SCALE:
        raise EntsoeLtDerivedArithmeticExecutionError(f"{label} scale exceeds limit")
    if len(parsed.as_tuple().digits) > MAX_SIGNIFICANT_DIGITS:
        raise EntsoeLtDerivedArithmeticExecutionError(f"{label} significant digits exceed limit")
    if abs(parsed) > MAX_ABS_MW:
        raise EntsoeLtDerivedArithmeticExecutionError(f"{label} magnitude is invalid")
    if _format_decimal(parsed) != value:
        raise EntsoeLtDerivedArithmeticExecutionError(f"{label} is not canonical")
    return parsed


def _format_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeLtDerivedArithmeticExecutionError(f"{label} JSON is invalid") from exc
    return _mapping(value, label)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EntsoeLtDerivedArithmeticExecutionError("JSON contains duplicate keys")
        result[key] = value
    return result


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeLtDerivedArithmeticExecutionError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EntsoeLtDerivedArithmeticExecutionError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise EntsoeLtDerivedArithmeticExecutionError(f"{label} differs")


def _clean(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise EntsoeLtDerivedArithmeticExecutionError(f"{label} must be clean text")
    return value


def _sha(value: object, label: str) -> str:
    text = _clean(value, label)
    if _SHA.fullmatch(text) is None:
        raise EntsoeLtDerivedArithmeticExecutionError(f"{label} must be lowercase SHA-256")
    return text


def _utc(value: object, label: str) -> datetime:
    text = _clean(value, label)
    if _UTC.fullmatch(text) is None:
        raise EntsoeLtDerivedArithmeticExecutionError(f"{label} must be canonical UTC Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise EntsoeLtDerivedArithmeticExecutionError(f"{label} is invalid") from exc
    return _utc_datetime(parsed, label)


def _utc_datetime(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EntsoeLtDerivedArithmeticExecutionError(f"{label} must be timezone-aware")
    normalized = value.astimezone(timezone.utc)
    if normalized.microsecond:
        raise EntsoeLtDerivedArithmeticExecutionError(f"{label} must have whole-second precision")
    return normalized


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "ALGORITHM_CONTENT_ID",
    "ALGORITHM_SPEC",
    "ARTIFACT_AUTHORITIES",
    "ASSESSMENT_SCHEMA",
    "AUTHORITIES",
    "CONTRACT_CONTENT_ID",
    "CONTRACT_PATH",
    "EXECUTION",
    "OPERAND_SCHEMA",
    "OUTPUT_SCHEMA",
    "PASS_STATUS",
    "EntsoeLtDerivedArithmeticExecutionError",
    "SyntheticArithmeticExecutionResult",
    "canonical_sha256",
    "execute_synthetic_load_forecast_error",
    "validate_derived_arithmetic_execution_contract",
    "validate_derived_arithmetic_execution_contract_path",
]

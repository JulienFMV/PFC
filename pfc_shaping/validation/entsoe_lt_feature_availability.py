"""Value-free PIT admission for ENTSO-E features used by the LT PFC.

The evaluator classifies information timing and missingness semantics only. It
never accepts or opens a feature value. Passing means that a proposed feature
row is structurally compatible with one forecast origin; it grants no data,
predictive, model-selection or production authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pfc_shaping.path_safety import read_stable_single_link_file

CONTRACT_SCHEMA = "fmv_entsoe_lt_feature_availability_contract.v1"
CONTRACT_STATUS = "VALUE_FREE_PIT_FEATURE_ROLE_ALGORITHM_NO_MODEL_AUTHORITY"
CONTRACT_FILE_SHA256 = (
    "3cb013cf11ef787538473fdcceef6287259106a4eb7cf97bfba27ef26693dfb6"
)
CONTRACT_CONTENT_ID = (
    "c7826b4ad2fa5cdb6baff5d077f00ee5fd8d98108cebef9920c147d787df2ab0"
)
ASSESSMENT_SCHEMA = "fmv_entsoe_lt_feature_availability_assessment.v1"
MAX_RECORDS = 100_000

INFORMATION_ROLES = (
    "REALIZED_ACTUAL",
    "OPERATIONAL_FORECAST",
    "FORECAST_ERROR",
    "LAGGED_REALIZED",
    "CALENDAR_KNOWN",
    "ORIGIN_FROZEN_CLIMATOLOGY",
    "GOVERNED_SCENARIO_SHAPE",
)
REQUESTED_USES = frozenset(
    {
        "TRAINING_COVARIATE",
        "PREDICTION_COVARIATE",
        "LAGGED_COVARIATE",
        "DIAGNOSTIC_ONLY",
    }
)
MODEL_USES = REQUESTED_USES - {"DIAGNOSTIC_ONLY"}
SUPPORT_STATES = frozenset({"SUPPORTED", "MISSING", "UNKNOWN"})
MISSING_VALUE_POLICIES = frozenset(
    {"PRESERVE_NULL", "ZERO_FILL", "NEUTRAL_FILL"}
)
MONTHLY_EFFECT_POLICIES = frozenset(
    {"ZERO_MEAN_WITHIN_SOLVER_MONTH", "NOT_APPLICABLE"}
)
RECORD_FIELDS = (
    "feature_id",
    "source_family",
    "information_role",
    "requested_use",
    "target_start_utc",
    "target_end_utc",
    "origin_utc",
    "available_at_utc",
    "source_issue_utc",
    "operational_horizon_end_utc",
    "training_cutoff_utc",
    "dependency_roles",
    "dependency_available_at_utc",
    "support_state",
    "missing_value_policy",
    "monthly_effect_policy",
)

_AUTHORITIES = {
    "real_feature_availability": False,
    "owner_semantics": False,
    "point_in_time": False,
    "predictive_value": False,
    "model_input": False,
    "model_selection": False,
    "candidate": False,
    "promotion": False,
    "production": False,
}
_EXECUTION = {
    "databricks_connection_count": 0,
    "databricks_statement_count": 0,
    "warehouse_start_count": 0,
    "databricks_write_count": 0,
    "network_call_count": 0,
    "h_drive_access_count": 0,
    "opened_real_value_row_count": 0,
}
_SCOPE = {
    "model_scope": "LT_ONLY",
    "target_product": "CH_PFC_N_PLUS_3_YEARS",
    "monthly_level_authority": "SOLVER",
    "hourly_or_quarter_hourly_layers_are_shape_only": True,
    "ct_module_import_authorized": False,
    "target_15_minute_grid_is_source_cadence_authority": False,
}
_ROLE_RULES = (
    {
        "role": "REALIZED_ACTUAL",
        "future_target_use": "FORBIDDEN",
        "historical_training_use": "ONLY_AFTER_TARGET_END_AND_AVAILABILITY",
        "required_dependency_count": 0,
    },
    {
        "role": "OPERATIONAL_FORECAST",
        "future_target_use": "ONLY_WITHIN_ISSUED_HORIZON_AND_AVAILABLE_AT_ORIGIN",
        "historical_training_use": "ONLY_FROM_THE_FOLD_VINTAGE",
        "required_dependency_count": 0,
    },
    {
        "role": "FORECAST_ERROR",
        "future_target_use": "FORBIDDEN",
        "historical_training_use": "LAGGED_ONLY_AFTER_FORECAST_AND_TRUTH_AVAILABILITY",
        "required_dependency_count": 2,
    },
    {
        "role": "LAGGED_REALIZED",
        "future_target_use": "ALLOWED_ONLY_WHEN_LAGGED_TARGET_ENDED_AND_AVAILABLE",
        "historical_training_use": "ALLOWED_WITH_EXACT_ORIGIN_CUTOFF",
        "required_dependency_count": 1,
    },
    {
        "role": "CALENDAR_KNOWN",
        "future_target_use": "ALLOWED_IF_AVAILABLE_AT_ORIGIN",
        "historical_training_use": "ALLOWED",
        "required_dependency_count": 0,
    },
    {
        "role": "ORIGIN_FROZEN_CLIMATOLOGY",
        "future_target_use": "ALLOWED_IF_FIT_CUTOFF_PRECEDES_ORIGIN",
        "historical_training_use": "ALLOWED_WITH_FOLD_SPECIFIC_REFIT",
        "required_dependency_count": 0,
    },
    {
        "role": "GOVERNED_SCENARIO_SHAPE",
        "future_target_use": "CANDIDATE_ONLY_IF_PUBLISHED_BY_ORIGIN",
        "historical_training_use": "NO_EMPIRICAL_AUTHORITY_FROM_THIS_CONTRACT",
        "required_dependency_count": 0,
    },
)
_ELIGIBILITY_RULES = {
    "available_at_not_after_origin": True,
    "training_target_end_not_after_origin": True,
    "lagged_target_end_not_after_origin": True,
    "contemporaneous_prediction_target_start_not_before_origin": True,
    "derived_available_at_equals_max_dependency_availability": True,
    "actual_target_end_not_after_origin": True,
    "realized_availability_not_before_target_end": True,
    "forecast_issue_not_after_origin": True,
    "forecast_issue_not_after_target_start": True,
    "forecast_availability_not_before_issue": True,
    "forecast_target_within_operational_horizon": True,
    "forecast_error_requires_actual_and_forecast_dependencies": True,
    "forecast_error_target_end_not_after_origin": True,
    "lagged_realized_target_end_not_after_origin": True,
    "climatology_training_cutoff_strictly_before_origin": True,
    "scenario_issue_not_after_origin": True,
    "scenario_availability_not_before_issue": True,
    "same_target_actual_or_error_prediction_authorized": False,
    "latest_snapshot_may_reconstruct_historical_origin": False,
    "backfill_gains_retroactive_availability": False,
}
_MISSINGNESS_POLICY = {
    "supported_state_required_for_model_use": True,
    "preserve_null_required": True,
    "zero_fill_authorized": False,
    "neutral_fill_authorized": False,
    "unknown_support_treated_as_supported": False,
    "calendar_effect_may_be_not_applicable": True,
    "non_calendar_lt_effect_must_be_zero_mean_within_solver_month": True,
    "feature_may_rewrite_solver_monthly_mean": False,
}

_CONTEMPORANEOUS_PREDICTION_ROLES = frozenset(
    {
        "OPERATIONAL_FORECAST",
        "CALENDAR_KNOWN",
        "ORIGIN_FROZEN_CLIMATOLOGY",
        "GOVERNED_SCENARIO_SHAPE",
    }
)

_CRITICAL_LEAKAGE_REASONS = frozenset(
    {
        "AVAILABLE_AFTER_ORIGIN",
        "TRAINING_TARGET_NOT_ENDED_AT_ORIGIN",
        "LAGGED_TARGET_NOT_ENDED_AT_ORIGIN",
        "PREDICTION_TARGET_STARTED_BEFORE_ORIGIN",
        "ACTUAL_TARGET_NOT_ENDED_AT_ORIGIN",
        "FORECAST_ISSUED_AFTER_ORIGIN",
        "FORECAST_ISSUED_AFTER_TARGET_START",
        "FORECAST_ERROR_TARGET_NOT_ENDED_AT_ORIGIN",
        "CLIMATOLOGY_CUTOFF_NOT_BEFORE_ORIGIN",
        "SCENARIO_ISSUED_AFTER_ORIGIN",
    }
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,254}$")
_UTC = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z$"
)


class EntsoeLtFeatureAvailabilityError(ValueError):
    """Raised when the contract or value-free feature metadata is malformed."""


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_feature_availability_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact execution-free D273 contract."""

    contract = _mapping(document, "feature availability contract")
    _exact(
        contract,
        {
            "schema_version",
            "decision",
            "status",
            "purpose",
            "scope",
            "information_roles",
            "record_schema",
            "eligibility_rules",
            "missingness_and_shape_policy",
            "execution",
            "authorities",
        },
        "feature availability contract",
    )
    _equal(contract["schema_version"], CONTRACT_SCHEMA, "contract schema")
    _equal(contract["decision"], "D-20260806-273", "contract decision")
    _equal(contract["status"], CONTRACT_STATUS, "contract status")
    _clean_text(contract["purpose"], "contract purpose")
    _equal(contract["scope"], _SCOPE, "contract scope")
    _equal(contract["information_roles"], list(_ROLE_RULES), "information roles")

    record_schema = _mapping(contract["record_schema"], "record schema")
    _exact(
        record_schema,
        {
            "required_fields",
            "timestamp_semantics",
            "maximum_records_per_assessment",
            "raw_values_allowed",
        },
        "record schema",
    )
    _equal(record_schema["required_fields"], list(RECORD_FIELDS), "record fields")
    _equal(
        record_schema["timestamp_semantics"],
        "CANONICAL_UTC_Z_LEFT_CLOSED_RIGHT_OPEN_TARGET",
        "timestamp semantics",
    )
    _equal(record_schema["maximum_records_per_assessment"], MAX_RECORDS, "record cap")
    _equal(record_schema["raw_values_allowed"], False, "raw value authority")
    _equal(contract["eligibility_rules"], _ELIGIBILITY_RULES, "eligibility rules")
    _equal(
        contract["missingness_and_shape_policy"],
        _MISSINGNESS_POLICY,
        "missingness and shape policy",
    )
    _equal(contract["execution"], _EXECUTION, "execution counters")
    _equal(contract["authorities"], _AUTHORITIES, "authorities")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": canonical_sha256(contract),
        "information_role_count": len(_ROLE_RULES),
        "model_input_authorized": False,
        "production_authorized": False,
        **_EXECUTION,
    }


def validate_feature_availability_contract_path(path: str | Path) -> dict[str, object]:
    """Read and hash-bind the exact repo-local D273 contract."""

    contract_path = Path(path)
    if not contract_path.is_absolute():
        raise EntsoeLtFeatureAvailabilityError("contract path must be absolute")
    try:
        payload = read_stable_single_link_file(
            contract_path,
            label="D273 ENTSO-E LT feature availability contract",
            max_bytes=100_000,
        )
    except (OSError, ValueError) as exc:
        raise EntsoeLtFeatureAvailabilityError("contract read failed") from exc
    if hashlib.sha256(payload).hexdigest() != CONTRACT_FILE_SHA256:
        raise EntsoeLtFeatureAvailabilityError("contract SHA-256 differs")
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeLtFeatureAvailabilityError("contract JSON is invalid") from exc
    result = validate_feature_availability_contract(
        _mapping(document, "feature availability contract")
    )
    if result["contract_content_id"] != CONTRACT_CONTENT_ID:
        raise EntsoeLtFeatureAvailabilityError("contract content ID differs")
    return result


def assess_feature_availability(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Classify value-free feature metadata against one or more LT origins."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise EntsoeLtFeatureAvailabilityError("feature records must be a sequence")
    if not records:
        raise EntsoeLtFeatureAvailabilityError("feature records must be non-empty")
    if len(records) > MAX_RECORDS:
        raise EntsoeLtFeatureAvailabilityError("feature record cap exceeded")

    outcomes: list[dict[str, object]] = []
    identities: set[tuple[str, str, str]] = set()
    for position, raw_record in enumerate(records, start=1):
        record = _validate_record(raw_record, position)
        identity = (
            str(record["feature_id"]).casefold(),
            str(record["target_start_utc"]),
            str(record["origin_utc"]),
        )
        if identity in identities:
            raise EntsoeLtFeatureAvailabilityError("feature assessment key is duplicated")
        identities.add(identity)
        outcomes.append(_assess_record(record))

    status_counts = Counter(str(item["status"]) for item in outcomes)
    reason_counts = Counter(
        reason
        for item in outcomes
        for reason in item["reasons"]
        if isinstance(reason, str)
    )
    severity_counts = Counter(str(item["severity"]) for item in outcomes)
    rejected_model = sum(
        1
        for record, outcome in zip(records, outcomes, strict=True)
        if record["requested_use"] in MODEL_USES
        and outcome["status"] != "STRUCTURALLY_ELIGIBLE_NO_MODEL_AUTHORITY"
    )
    return {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": (
            "PASS_VALUE_FREE_PIT_STRUCTURE_NO_MODEL_AUTHORITY"
            if rejected_model == 0
            else "FAIL_VALUE_FREE_FEATURE_LEAKAGE_OR_UNSUPPORTED"
        ),
        "contract_content_id": CONTRACT_CONTENT_ID,
        "record_count": len(outcomes),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "rejected_model_request_count": rejected_model,
        "all_model_requests_structurally_pit_safe": rejected_model == 0,
        "raw_feature_ids_persisted": False,
        "raw_values_opened": False,
        "outcomes": tuple(outcomes),
        "authorities": dict(_AUTHORITIES),
        **_EXECUTION,
    }


def _validate_record(raw_record: object, position: int) -> dict[str, object]:
    record = dict(_mapping(raw_record, f"feature record {position}"))
    _exact(record, set(RECORD_FIELDS), f"feature record {position}")
    for field in ("feature_id", "source_family"):
        value = _clean_text(record[field], f"feature record {position} {field}")
        if _IDENTIFIER.fullmatch(value) is None:
            raise EntsoeLtFeatureAvailabilityError(f"{field} is invalid")
    if record["information_role"] not in INFORMATION_ROLES:
        raise EntsoeLtFeatureAvailabilityError("information role is invalid")
    if record["requested_use"] not in REQUESTED_USES:
        raise EntsoeLtFeatureAvailabilityError("requested use is invalid")
    if record["support_state"] not in SUPPORT_STATES:
        raise EntsoeLtFeatureAvailabilityError("support state is invalid")
    if record["missing_value_policy"] not in MISSING_VALUE_POLICIES:
        raise EntsoeLtFeatureAvailabilityError("missing-value policy is invalid")
    if record["monthly_effect_policy"] not in MONTHLY_EFFECT_POLICIES:
        raise EntsoeLtFeatureAvailabilityError("monthly-effect policy is invalid")

    parsed: dict[str, datetime | None] = {}
    for field in (
        "target_start_utc",
        "target_end_utc",
        "origin_utc",
        "available_at_utc",
        "source_issue_utc",
        "operational_horizon_end_utc",
        "training_cutoff_utc",
    ):
        parsed[field] = _optional_utc(record[field], field)
    if parsed["target_start_utc"] is None or parsed["target_end_utc"] is None:
        raise EntsoeLtFeatureAvailabilityError("target interval timestamps are required")
    if parsed["origin_utc"] is None:
        raise EntsoeLtFeatureAvailabilityError("origin timestamp is required")
    if parsed["target_start_utc"] >= parsed["target_end_utc"]:
        raise EntsoeLtFeatureAvailabilityError("target interval is empty or reversed")

    dependencies = record["dependency_available_at_utc"]
    if not isinstance(dependencies, list):
        raise EntsoeLtFeatureAvailabilityError("dependency availability must be a list")
    dependency_roles = record["dependency_roles"]
    if not isinstance(dependency_roles, list):
        raise EntsoeLtFeatureAvailabilityError("dependency roles must be a list")
    expected_dependency_roles = {
        "FORECAST_ERROR": ["OPERATIONAL_FORECAST", "REALIZED_ACTUAL"],
        "LAGGED_REALIZED": ["REALIZED_ACTUAL"],
    }.get(str(record["information_role"]), [])
    if dependency_roles != expected_dependency_roles:
        raise EntsoeLtFeatureAvailabilityError("dependency roles differ")
    parsed_dependencies = tuple(
        _required_utc(value, "dependency availability") for value in dependencies
    )
    required_count = _ROLE_RULES[INFORMATION_ROLES.index(record["information_role"])][
        "required_dependency_count"
    ]
    if len(parsed_dependencies) != required_count:
        raise EntsoeLtFeatureAvailabilityError("dependency availability count differs")
    return {
        **record,
        "_parsed": parsed,
        "_dependencies": parsed_dependencies,
    }


def _assess_record(record: Mapping[str, object]) -> dict[str, object]:
    parsed = _mapping(record["_parsed"], "parsed timestamps")
    dependencies = record["_dependencies"]
    if not isinstance(dependencies, tuple):
        raise EntsoeLtFeatureAvailabilityError("parsed dependencies differ")
    role = str(record["information_role"])
    requested_use = str(record["requested_use"])
    target_start = _parsed_timestamp(parsed, "target_start_utc")
    target_end = _parsed_timestamp(parsed, "target_end_utc")
    origin = _parsed_timestamp(parsed, "origin_utc")
    available = _parsed_timestamp(parsed, "available_at_utc", required=False)
    issue = _parsed_timestamp(parsed, "source_issue_utc", required=False)
    horizon_end = _parsed_timestamp(
        parsed,
        "operational_horizon_end_utc",
        required=False,
    )
    training_cutoff = _parsed_timestamp(
        parsed,
        "training_cutoff_utc",
        required=False,
    )
    reasons: set[str] = set()

    if available is None:
        reasons.add("AVAILABLE_AT_MISSING")
    elif available > origin:
        reasons.add("AVAILABLE_AFTER_ORIGIN")
    if requested_use == "TRAINING_COVARIATE" and target_end > origin:
        reasons.add("TRAINING_TARGET_NOT_ENDED_AT_ORIGIN")
    if requested_use == "LAGGED_COVARIATE" and target_end > origin:
        reasons.add("LAGGED_TARGET_NOT_ENDED_AT_ORIGIN")
    if (
        requested_use == "PREDICTION_COVARIATE"
        and role in _CONTEMPORANEOUS_PREDICTION_ROLES
        and target_start < origin
    ):
        reasons.add("PREDICTION_TARGET_STARTED_BEFORE_ORIGIN")
    if dependencies and available != max(dependencies):
        reasons.add("DERIVED_AVAILABILITY_DIFFERS_FROM_DEPENDENCIES")
    if record["support_state"] != "SUPPORTED" and requested_use in MODEL_USES:
        reasons.add(f"SUPPORT_{record['support_state']}")
    if record["missing_value_policy"] != "PRESERVE_NULL":
        reasons.add(f"MISSING_POLICY_{record['missing_value_policy']}_FORBIDDEN")
    if role == "CALENDAR_KNOWN":
        if record["monthly_effect_policy"] != "NOT_APPLICABLE":
            reasons.add("CALENDAR_MONTHLY_POLICY_INVALID")
    elif record["monthly_effect_policy"] != "ZERO_MEAN_WITHIN_SOLVER_MONTH":
        reasons.add("NON_CALENDAR_EFFECT_NOT_ZERO_MEAN")

    if role == "REALIZED_ACTUAL":
        if target_end > origin:
            reasons.add("ACTUAL_TARGET_NOT_ENDED_AT_ORIGIN")
        if available is not None and available < target_end:
            reasons.add("ACTUAL_AVAILABLE_BEFORE_TARGET_END")
        if requested_use not in {"TRAINING_COVARIATE", "DIAGNOSTIC_ONLY"}:
            reasons.add("ACTUAL_REQUESTED_AS_PREDICTION_OR_LAGGED_FEATURE")
        _require_null_timing(issue, horizon_end, training_cutoff, reasons)
    elif role == "OPERATIONAL_FORECAST":
        if issue is None:
            reasons.add("FORECAST_ISSUE_MISSING")
        else:
            if issue > origin:
                reasons.add("FORECAST_ISSUED_AFTER_ORIGIN")
            if issue > target_start:
                reasons.add("FORECAST_ISSUED_AFTER_TARGET_START")
            if available is not None and available < issue:
                reasons.add("FORECAST_AVAILABLE_BEFORE_ISSUE")
        if horizon_end is None:
            reasons.add("FORECAST_HORIZON_END_MISSING")
        elif target_end > horizon_end:
            reasons.add("TARGET_OUTSIDE_OPERATIONAL_FORECAST_HORIZON")
        if training_cutoff is not None:
            reasons.add("FORECAST_TRAINING_CUTOFF_MUST_BE_NULL")
    elif role == "FORECAST_ERROR":
        if target_end > origin:
            reasons.add("FORECAST_ERROR_TARGET_NOT_ENDED_AT_ORIGIN")
        if requested_use not in {"LAGGED_COVARIATE", "DIAGNOSTIC_ONLY"}:
            reasons.add("FORECAST_ERROR_NOT_LAGGED_OR_DIAGNOSTIC")
        if len(dependencies) == 2 and dependencies[1] < target_end:
            reasons.add("FORECAST_ERROR_TRUTH_AVAILABLE_BEFORE_TARGET_END")
        _require_null_timing(issue, horizon_end, training_cutoff, reasons)
    elif role == "LAGGED_REALIZED":
        if target_end > origin:
            reasons.add("LAGGED_TARGET_NOT_ENDED_AT_ORIGIN")
        if requested_use not in {"PREDICTION_COVARIATE", "LAGGED_COVARIATE"}:
            reasons.add("LAGGED_REALIZED_USE_INVALID")
        if available is not None and available < target_end:
            reasons.add("LAGGED_REALIZED_AVAILABLE_BEFORE_TARGET_END")
        _require_null_timing(issue, horizon_end, training_cutoff, reasons)
    elif role == "CALENDAR_KNOWN":
        _require_null_timing(issue, horizon_end, training_cutoff, reasons)
    elif role == "ORIGIN_FROZEN_CLIMATOLOGY":
        if training_cutoff is None:
            reasons.add("CLIMATOLOGY_TRAINING_CUTOFF_MISSING")
        elif training_cutoff >= origin:
            reasons.add("CLIMATOLOGY_CUTOFF_NOT_BEFORE_ORIGIN")
        if available is not None and training_cutoff is not None and available < training_cutoff:
            reasons.add("CLIMATOLOGY_AVAILABLE_BEFORE_TRAINING_CUTOFF")
        if issue is not None or horizon_end is not None:
            reasons.add("CLIMATOLOGY_FORECAST_TIMING_MUST_BE_NULL")
    elif role == "GOVERNED_SCENARIO_SHAPE":
        if issue is None:
            reasons.add("SCENARIO_ISSUE_MISSING")
        elif issue > origin:
            reasons.add("SCENARIO_ISSUED_AFTER_ORIGIN")
        if issue is not None and available is not None and available < issue:
            reasons.add("SCENARIO_AVAILABLE_BEFORE_ISSUE")
        if horizon_end is not None or training_cutoff is not None:
            reasons.add("SCENARIO_OPERATIONAL_TIMING_MUST_BE_NULL")
        if requested_use not in {"PREDICTION_COVARIATE", "DIAGNOSTIC_ONLY"}:
            reasons.add("SCENARIO_USE_INVALID")

    if reasons:
        status = "REJECTED_LEAKAGE_OR_UNSUPPORTED"
    elif requested_use == "DIAGNOSTIC_ONLY":
        status = "DIAGNOSTIC_ONLY_NO_MODEL_AUTHORITY"
    else:
        status = "STRUCTURALLY_ELIGIBLE_NO_MODEL_AUTHORITY"
    severity = (
        "CRITICAL"
        if reasons & _CRITICAL_LEAKAGE_REASONS
        else "HIGH"
        if reasons
        else "INFO"
    )
    return {
        "feature_id_sha256": hashlib.sha256(
            str(record["feature_id"]).encode("utf-8")
        ).hexdigest(),
        "information_role": role,
        "requested_use": requested_use,
        "status": status,
        "severity": severity,
        "confidence": "HIGH",
        "reasons": tuple(sorted(reasons)),
        "raw_feature_id_persisted": False,
        "raw_value_opened": False,
        "model_input_authorized": False,
    }


def _require_null_timing(
    issue: datetime | None,
    horizon_end: datetime | None,
    training_cutoff: datetime | None,
    reasons: set[str],
) -> None:
    if issue is not None or horizon_end is not None or training_cutoff is not None:
        reasons.add("ROLE_HAS_INAPPLICABLE_TIMING_FIELDS")


def _parsed_timestamp(
    parsed: Mapping[str, object],
    field: str,
    *,
    required: bool = True,
) -> datetime | None:
    value = parsed[field]
    if value is None and not required:
        return None
    if not isinstance(value, datetime):
        raise EntsoeLtFeatureAvailabilityError(f"parsed {field} differs")
    return value


def _optional_utc(value: object, label: str) -> datetime | None:
    if value is None:
        return None
    return _required_utc(value, label)


def _required_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise EntsoeLtFeatureAvailabilityError(f"{label} must be canonical UTC Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise EntsoeLtFeatureAvailabilityError(f"{label} is invalid") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EntsoeLtFeatureAvailabilityError(f"{label} must be UTC")
    return parsed


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EntsoeLtFeatureAvailabilityError("JSON contains duplicate keys")
        result[key] = value
    return result


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeLtFeatureAvailabilityError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EntsoeLtFeatureAvailabilityError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise EntsoeLtFeatureAvailabilityError(f"{label} differs")


def _clean_text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise EntsoeLtFeatureAvailabilityError(f"{label} is invalid")
    return value


__all__ = [
    "ASSESSMENT_SCHEMA",
    "CONTRACT_CONTENT_ID",
    "CONTRACT_FILE_SHA256",
    "CONTRACT_SCHEMA",
    "CONTRACT_STATUS",
    "INFORMATION_ROLES",
    "MAX_RECORDS",
    "MODEL_USES",
    "RECORD_FIELDS",
    "EntsoeLtFeatureAvailabilityError",
    "assess_feature_availability",
    "canonical_sha256",
    "validate_feature_availability_contract",
    "validate_feature_availability_contract_path",
]

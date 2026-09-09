"""Validate a value-blind Databricks source cost preflight receipt.

This module never connects to Databricks and never authorizes Phase A. It only
checks that a future data-engineer receipt is complete, fresh, hash-bound and
explicit about scan caps and Warehouse state before human cost review.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pfc_shaping.path_safety import read_stable_single_link_file

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = PHASE / "DATABRICKS-PFC-SOURCE-COST-PREFLIGHT-CONTRACT-V1.json"
REQUEST_PATH = PHASE / "DATABRICKS-SPOT-WEATHER-SWISSGRID-DATA-ENGINEER-REQUEST-20260806.md"

CONTRACT_SCHEMA = "fmv_databricks_pfc_source_cost_preflight_contract.v1"
CONTRACT_STATUS = "LOCAL_COST_PREFLIGHT_RECEIPT_VALIDATION_NO_EXECUTION_AUTHORITY"
CONTRACT_FILE_SHA256 = "0beac8ef83e31f039b7e862987785694889ec5f946b082342931cfa4bf1a768c"
CONTRACT_CONTENT_ID = "4d8857e09eb4fd2acadcd51faa9110a3e406cd1a04afea439d7f10c6e9da8771"
REQUEST_SHA256 = "1ab1f396fb37455d6b91f1bd2c374870dc3910214d894294e4e8c81397056b6b"

RECEIPT_SCHEMA = "fmv_databricks_pfc_source_cost_preflight_receipt.v1"
EVIDENCE_CLASS = "DATA_ENGINEER_COST_PREFLIGHT_METADATA_ONLY"
ASSESSMENT_SCHEMA = "fmv_databricks_pfc_source_cost_preflight_assessment.v1"
READY_STATUS = "READY_FOR_HUMAN_COST_REVIEW_NO_EXECUTION_AUTHORITY"
STOPPED_STATUS = "STOP_NO_ACTIVE_WAREHOUSE"
UNCAPPED_STATUS = "STOP_UNCAPPED_SCAN"
RUNNING_STATE = "RUNNING_ALREADY_FOR_SEPARATE_AUTHORIZED_WORKLOAD"
ALLOWED_WAREHOUSE_STATES = (RUNNING_STATE, "STOPPED", "UNAVAILABLE")
ALLOWED_ESTIMATE_METHODS = (
    "QUERY_PLAN_ESTIMATE",
    "DELTA_METADATA_UPPER_BOUND",
    "PLATFORM_EXPORT_QUOTE",
    "UNAVAILABLE",
)
MAXIMUM_AGE_SECONDS = 86_400
MAXIMUM_METADATA_STATEMENTS = 4
MAXIMUM_RUNTIME_SECONDS = 900

EXECUTION = {
    "databricks_connection_count": 0,
    "databricks_statement_count": 0,
    "business_row_count_opened": 0,
    "warehouse_start_count": 0,
    "databricks_write_count": 0,
    "network_call_count": 0,
    "h_drive_access_count": 0,
    "remote_write_count": 0,
}
AUTHORITIES = {
    "local_receipt_integrity": True,
    "cost_estimate_authenticity": False,
    "phase_a_execution": False,
    "phase_b_execution": False,
    "point_in_time": False,
    "predictive_value": False,
    "model_input": False,
    "model_selection": False,
    "candidate": False,
    "ompex_superiority": False,
    "promotion": False,
    "production": False,
}
GO_AUTHORITY = {
    "phase_a_go_authorized": False,
    "phase_b_go_authorized": False,
    "model_input_authorized": False,
    "production_authorized": False,
}
RECEIPT_EXECUTION_KEYS = {
    "control_plane_get_count",
    "metadata_statement_count",
    "business_profile_statement_count",
    "business_row_count_opened",
    "warehouse_start_count",
    "databricks_write_count",
    "retry_count",
}
PROFILE_KEYS = {
    "profile_role",
    "source_table",
    "query_sha256",
    "delta_size_bytes",
    "delta_file_count",
    "partition_columns",
    "estimate_method",
    "estimated_scan_bytes_lower_bound",
    "estimated_scan_bytes_upper_bound",
    "upper_bound_is_hard_cap",
    "pruning_proven",
    "planned_partition_predicate",
}
PROFILE_BINDINGS = (
    {
        "profile_role": "SPOT",
        "source_table": "dev.silver.ge_market_euler_spot",
        "query_relative_path": "docs/data/sql/databricks_dev_spot_profile.sql",
        "query_sha256": "2852484c01335a348475230d5cd9c91c1d6f6ad70edb4dddba93e9c8647656c0",
    },
    {
        "profile_role": "WEATHER",
        "source_table": (
            "prd.silver.weather_meteoswiss_measurement|"
            "prd.silver.weather_meteoswiss_forecast|"
            "prd.silver.weather_openmeteo_forecast"
        ),
        "query_relative_path": "docs/data/sql/databricks_prd_weather_profile.sql",
        "query_sha256": "8183f34dfd9ce322b0ec4011145d62d4984e3250807c4aa107aec644ebae049c",
    },
    {
        "profile_role": "SWISSGRID_BALANCING",
        "source_table": "prd.silver.ge_power_swissgrid_cab",
        "query_relative_path": "docs/data/sql/databricks_prd_swissgrid_balancing_profile.sql",
        "query_sha256": "12bfcd9dbd12ac337c0e701116a50d612a8737507f75e487e9e5626101a32be3",
    },
    {
        "profile_role": "SWISSGRID_TENDER",
        "source_table": "prd.silver.ge_market_swissgrid_sdl_tenders",
        "query_relative_path": "docs/data/sql/databricks_prd_swissgrid_tender_profile.sql",
        "query_sha256": "acfe34cbf60c96ad221be1ec51dde4349164dccdef3d0cf843fa0723fdb814d2",
    },
)
PROFILE_BINDING_BY_ROLE = {item["profile_role"]: item for item in PROFILE_BINDINGS}

_SHA = re.compile(r"^[0-9a-f]{64}$")
_DECIMAL = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]{1,6})?$")


class DatabricksPfcSourceCostPreflightError(ValueError):
    """Raised when a cost preflight receipt or its local bindings drift."""


@dataclass(frozen=True)
class CostPreflightAssessment:
    """Price/value-blind preflight assessment with no execution authority."""

    assessment: dict[str, object]
    assessment_content_id: str


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_snapshot_id(receipt_without_snapshot_id: Mapping[str, object]) -> str:
    return canonical_sha256(receipt_without_snapshot_id)


def validate_cost_preflight_contract(document: Mapping[str, object]) -> dict[str, object]:
    contract = _mapping(document, "D292 contract")
    _exact(
        contract,
        {
            "schema_version",
            "decision",
            "status",
            "purpose",
            "request_binding",
            "profile_bindings",
            "receipt_policy",
            "verdict_policy",
            "execution",
            "authorities",
        },
        "D292 contract",
    )
    _equal(contract["schema_version"], CONTRACT_SCHEMA, "contract schema")
    _equal(contract["decision"], "D-20260806-292", "contract decision")
    _equal(contract["status"], CONTRACT_STATUS, "contract status")
    _clean(contract["purpose"], "contract purpose")
    _equal(
        contract["request_binding"],
        {
            "relative_path": _relative(REQUEST_PATH),
            "sha256": REQUEST_SHA256,
        },
        "request binding",
    )
    _equal(contract["profile_bindings"], list(PROFILE_BINDINGS), "profile bindings")
    _equal(
        contract["receipt_policy"],
        {
            "receipt_schema_version": RECEIPT_SCHEMA,
            "evidence_class": EVIDENCE_CLASS,
            "snapshot_id_derivation": "SHA256_CANONICAL_RECEIPT_WITHOUT_SNAPSHOT_ID",
            "maximum_age_seconds": MAXIMUM_AGE_SECONDS,
            "exact_profile_count": 4,
            "maximum_metadata_statement_count": MAXIMUM_METADATA_STATEMENTS,
            "maximum_runtime_seconds": MAXIMUM_RUNTIME_SECONDS,
            "allowed_warehouse_states": list(ALLOWED_WAREHOUSE_STATES),
            "allowed_estimate_methods": list(ALLOWED_ESTIMATE_METHODS),
            "cost_unit": "DBU",
            "cost_owner_decision": "NOT_REVIEWED",
            "business_profile_statement_count": 0,
            "business_row_count_opened": 0,
            "warehouse_start_count": 0,
            "databricks_write_count": 0,
            "retry_count": 0,
            "phase_a_go_authorized": False,
            "phase_b_go_authorized": False,
        },
        "receipt policy",
    )
    _equal(
        contract["verdict_policy"],
        {
            "ready": READY_STATUS,
            "stopped": STOPPED_STATUS,
            "uncapped": UNCAPPED_STATUS,
            "ready_is_phase_a_go": False,
            "human_cost_review_required": True,
        },
        "verdict policy",
    )
    _equal(contract["execution"], EXECUTION, "contract execution")
    _equal(contract["authorities"], AUTHORITIES, "contract authorities")
    _equal(canonical_sha256(contract), CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "exact_profile_count": len(PROFILE_BINDINGS),
        "phase_a_go_authorized": False,
        **EXECUTION,
    }


def validate_cost_preflight_contract_path(path: str | Path = CONTRACT_PATH) -> dict[str, object]:
    contract_path = _absolute(path, "D292 contract")
    payload = _read(contract_path, "D292 contract", 200_000)
    if hashlib.sha256(payload).hexdigest() != CONTRACT_FILE_SHA256:
        raise DatabricksPfcSourceCostPreflightError("D292 contract SHA-256 differs")
    result = validate_cost_preflight_contract(_strict_json(payload, "D292 contract"))
    _verify_local_binding(REQUEST_PATH, REQUEST_SHA256, "D292 request")
    for binding in PROFILE_BINDINGS:
        _verify_local_binding(
            ROOT / binding["query_relative_path"],
            binding["query_sha256"],
            f"{binding['profile_role']} query",
        )
    if _read(contract_path, "D292 contract", 200_000) != payload:
        raise DatabricksPfcSourceCostPreflightError("D292 contract changed during validation")
    return result


def validate_cost_preflight_receipt(
    document: Mapping[str, object],
    *,
    reference_time_utc: datetime,
    contract_path: str | Path = CONTRACT_PATH,
) -> CostPreflightAssessment:
    validate_cost_preflight_contract_path(contract_path)
    reference = _utc_datetime(reference_time_utc, "reference time")
    receipt = _mapping(document, "cost preflight receipt")
    _exact(
        receipt,
        {
            "schema_version",
            "snapshot_id",
            "evidence_class",
            "created_at_utc",
            "request_sha256",
            "warehouse",
            "execution",
            "profiles",
            "cost_review",
            "go_authority",
        },
        "cost preflight receipt",
    )
    _equal(receipt["schema_version"], RECEIPT_SCHEMA, "receipt schema")
    _equal(receipt["evidence_class"], EVIDENCE_CLASS, "receipt evidence class")
    _equal(receipt["request_sha256"], REQUEST_SHA256, "receipt request binding")
    snapshot_id = _sha(receipt["snapshot_id"], "receipt snapshot ID")
    without_snapshot = {key: value for key, value in receipt.items() if key != "snapshot_id"}
    _equal(snapshot_id, derive_snapshot_id(without_snapshot), "receipt snapshot derivation")

    created = _utc_text_datetime(receipt["created_at_utc"], "receipt creation time")
    if created > reference:
        raise DatabricksPfcSourceCostPreflightError("receipt creation time is in the future")
    age_seconds = int((reference - created).total_seconds())
    if age_seconds > MAXIMUM_AGE_SECONDS:
        raise DatabricksPfcSourceCostPreflightError("receipt is stale")

    warehouse = _validate_warehouse(receipt["warehouse"])
    execution = _validate_receipt_execution(receipt["execution"])
    profiles, all_capped = _validate_profiles(receipt["profiles"])
    cost_review = _validate_cost_review(receipt["cost_review"], all_capped=all_capped)
    _equal(receipt["go_authority"], GO_AUTHORITY, "receipt GO authority")

    warehouse_state = str(warehouse["state_at_preflight"])
    if warehouse_state != RUNNING_STATE:
        status = STOPPED_STATUS
    elif not all_capped:
        status = UNCAPPED_STATUS
    else:
        status = READY_STATUS
    assessment = {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": status,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "receipt_snapshot_id": snapshot_id,
        "created_at_utc": _utc_text(created),
        "reference_time_utc": _utc_text(reference),
        "receipt_age_seconds": age_seconds,
        "profile_count": len(profiles),
        "warehouse_state": warehouse_state,
        "warehouse_started_for_request": False,
        "all_scan_upper_bounds_hard_capped": all_capped,
        "total_delta_size_bytes": sum(int(item["delta_size_bytes"]) for item in profiles),
        "total_estimated_scan_bytes_upper_bound": sum(
            int(item["estimated_scan_bytes_upper_bound"]) for item in profiles
        ),
        "metadata_statement_count": execution["metadata_statement_count"],
        "maximum_runtime_seconds": cost_review["maximum_runtime_seconds"],
        "cost_owner_decision": "NOT_REVIEWED",
        "human_cost_review_required": True,
        "phase_a_go_authorized": False,
        "phase_b_go_authorized": False,
        "model_input_authorized": False,
        "production_authorized": False,
        "authorities": dict(AUTHORITIES),
        **EXECUTION,
    }
    return CostPreflightAssessment(
        assessment=assessment,
        assessment_content_id=canonical_sha256(assessment),
    )


def validate_cost_preflight_receipt_path(
    path: str | Path,
    *,
    reference_time_utc: datetime,
    contract_path: str | Path = CONTRACT_PATH,
) -> CostPreflightAssessment:
    receipt_path = _absolute(path, "cost preflight receipt")
    try:
        receipt_path.resolve(strict=True).relative_to((ROOT / "build").resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise DatabricksPfcSourceCostPreflightError(
            "cost preflight receipt must be below repo-local build"
        ) from exc
    payload = _read(receipt_path, "cost preflight receipt", 200_000)
    result = validate_cost_preflight_receipt(
        _strict_json(payload, "cost preflight receipt"),
        reference_time_utc=reference_time_utc,
        contract_path=contract_path,
    )
    if _read(receipt_path, "cost preflight receipt", 200_000) != payload:
        raise DatabricksPfcSourceCostPreflightError("receipt changed during validation")
    return result


def _validate_warehouse(value: object) -> Mapping[str, object]:
    warehouse = _mapping(value, "warehouse")
    _exact(
        warehouse,
        {
            "warehouse_id_sha256",
            "state_at_preflight",
            "warehouse_size",
            "warehouse_type",
            "started_for_request",
            "auto_start_triggered",
        },
        "warehouse",
    )
    _sha(warehouse["warehouse_id_sha256"], "warehouse ID hash")
    state = _clean(warehouse["state_at_preflight"], "warehouse state")
    if state not in ALLOWED_WAREHOUSE_STATES:
        raise DatabricksPfcSourceCostPreflightError("warehouse state differs")
    _clean(warehouse["warehouse_size"], "warehouse size")
    _clean(warehouse["warehouse_type"], "warehouse type")
    _equal(warehouse["started_for_request"], False, "Warehouse started for request")
    _equal(warehouse["auto_start_triggered"], False, "Warehouse auto-start")
    return warehouse


def _validate_receipt_execution(value: object) -> Mapping[str, int]:
    execution = _mapping(value, "receipt execution")
    _exact(execution, RECEIPT_EXECUTION_KEYS, "receipt execution")
    normalized = {key: _nonnegative_int(execution[key], key) for key in RECEIPT_EXECUTION_KEYS}
    if normalized["metadata_statement_count"] > MAXIMUM_METADATA_STATEMENTS:
        raise DatabricksPfcSourceCostPreflightError("too many metadata statements")
    for key in (
        "business_profile_statement_count",
        "business_row_count_opened",
        "warehouse_start_count",
        "databricks_write_count",
        "retry_count",
    ):
        if normalized[key] != 0:
            raise DatabricksPfcSourceCostPreflightError(f"{key} must be zero")
    return normalized


def _validate_profiles(value: object) -> tuple[list[Mapping[str, object]], bool]:
    if not isinstance(value, list) or len(value) != len(PROFILE_BINDINGS):
        raise DatabricksPfcSourceCostPreflightError("profiles must contain exactly four entries")
    profiles: list[Mapping[str, object]] = []
    seen: set[str] = set()
    all_capped = True
    for index, raw in enumerate(value):
        profile = _mapping(raw, f"profile[{index}]")
        _exact(profile, PROFILE_KEYS, f"profile[{index}]")
        role = _clean(profile["profile_role"], f"profile[{index}] role")
        if role in seen or role not in PROFILE_BINDING_BY_ROLE:
            raise DatabricksPfcSourceCostPreflightError("profile roles differ or repeat")
        seen.add(role)
        binding = PROFILE_BINDING_BY_ROLE[role]
        _equal(profile["source_table"], binding["source_table"], f"{role} source table")
        _equal(profile["query_sha256"], binding["query_sha256"], f"{role} query SHA-256")
        size = _nonnegative_int(profile["delta_size_bytes"], f"{role} Delta size")
        files = _nonnegative_int(profile["delta_file_count"], f"{role} Delta file count")
        if (size == 0) is not (files == 0):
            raise DatabricksPfcSourceCostPreflightError(f"{role} Delta size/file count conflict")
        partitions = profile["partition_columns"]
        if not isinstance(partitions, list):
            raise DatabricksPfcSourceCostPreflightError(f"{role} partitions must be a list")
        cleaned_partitions = [_clean(item, f"{role} partition") for item in partitions]
        if len(cleaned_partitions) != len(set(cleaned_partitions)):
            raise DatabricksPfcSourceCostPreflightError(f"{role} partitions repeat")
        method = _clean(profile["estimate_method"], f"{role} estimate method")
        if method not in ALLOWED_ESTIMATE_METHODS:
            raise DatabricksPfcSourceCostPreflightError(f"{role} estimate method differs")
        lower = _nonnegative_int(
            profile["estimated_scan_bytes_lower_bound"], f"{role} scan lower bound"
        )
        upper = _nonnegative_int(
            profile["estimated_scan_bytes_upper_bound"], f"{role} scan upper bound"
        )
        if lower > upper:
            raise DatabricksPfcSourceCostPreflightError(f"{role} scan bounds invert")
        hard_cap = _bool(profile["upper_bound_is_hard_cap"], f"{role} hard cap")
        pruning = _bool(profile["pruning_proven"], f"{role} pruning")
        predicate = profile["planned_partition_predicate"]
        if pruning:
            if not cleaned_partitions:
                raise DatabricksPfcSourceCostPreflightError(f"{role} pruning lacks partitions")
            _clean(predicate, f"{role} partition predicate")
        elif predicate is not None:
            raise DatabricksPfcSourceCostPreflightError(
                f"{role} unproven pruning must not declare a predicate"
            )
        if method == "UNAVAILABLE":
            if lower != 0 or upper != 0 or hard_cap or pruning:
                raise DatabricksPfcSourceCostPreflightError(
                    f"{role} unavailable estimate fields conflict"
                )
        elif upper == 0 and size > 0:
            raise DatabricksPfcSourceCostPreflightError(
                f"{role} scan cap is zero for non-empty table"
            )
        all_capped = all_capped and hard_cap and method != "UNAVAILABLE"
        profiles.append(profile)
    if seen != set(PROFILE_BINDING_BY_ROLE):
        raise DatabricksPfcSourceCostPreflightError("profile inventory differs")
    return profiles, all_capped


def _validate_cost_review(value: object, *, all_capped: bool) -> Mapping[str, object]:
    review = _mapping(value, "cost review")
    _exact(
        review,
        {
            "cost_unit",
            "estimated_maximum_dbu",
            "proposed_maximum_dbu",
            "maximum_runtime_seconds",
            "cost_owner_decision",
        },
        "cost review",
    )
    _equal(review["cost_unit"], "DBU", "cost unit")
    _equal(review["cost_owner_decision"], "NOT_REVIEWED", "cost owner decision")
    runtime = _nonnegative_int(review["maximum_runtime_seconds"], "maximum runtime")
    if all_capped:
        estimated = _canonical_decimal(review["estimated_maximum_dbu"], "estimated maximum DBU")
        proposed = _canonical_decimal(review["proposed_maximum_dbu"], "proposed maximum DBU")
        if proposed < estimated:
            raise DatabricksPfcSourceCostPreflightError("proposed DBU cap is below estimate")
        if runtime < 1 or runtime > MAXIMUM_RUNTIME_SECONDS:
            raise DatabricksPfcSourceCostPreflightError("maximum runtime differs")
    else:
        _equal(review["estimated_maximum_dbu"], None, "uncapped estimated DBU")
        _equal(review["proposed_maximum_dbu"], None, "uncapped proposed DBU")
        _equal(runtime, 0, "uncapped maximum runtime")
    return review


def _verify_local_binding(path: Path, expected_sha256: str, label: str) -> None:
    payload = _read(path, label, 1_000_000)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise DatabricksPfcSourceCostPreflightError(f"{label} SHA-256 differs")


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatabricksPfcSourceCostPreflightError(f"{label} JSON is invalid") from exc
    return _mapping(value, label)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DatabricksPfcSourceCostPreflightError("JSON contains duplicate keys")
        result[key] = value
    return result


def _read(path: Path, label: str, maximum_bytes: int) -> bytes:
    try:
        return read_stable_single_link_file(path, label=label, max_bytes=maximum_bytes)
    except (OSError, ValueError) as exc:
        raise DatabricksPfcSourceCostPreflightError(f"{label} read failed") from exc


def _absolute(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise DatabricksPfcSourceCostPreflightError(f"{label} path must be absolute")
    return path


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DatabricksPfcSourceCostPreflightError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise DatabricksPfcSourceCostPreflightError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise DatabricksPfcSourceCostPreflightError(f"{label} differs")


def _clean(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise DatabricksPfcSourceCostPreflightError(f"{label} must be clean text")
    return value


def _sha(value: object, label: str) -> str:
    cleaned = _clean(value, label)
    if not _SHA.fullmatch(cleaned):
        raise DatabricksPfcSourceCostPreflightError(f"{label} must be lowercase SHA-256")
    return cleaned


def _bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise DatabricksPfcSourceCostPreflightError(f"{label} must be boolean")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise DatabricksPfcSourceCostPreflightError(f"{label} must be a nonnegative integer")
    return value


def _canonical_decimal(value: object, label: str) -> Decimal:
    if not isinstance(value, str) or not _DECIMAL.fullmatch(value):
        raise DatabricksPfcSourceCostPreflightError(f"{label} must be canonical decimal text")
    try:
        decimal = Decimal(value)
    except InvalidOperation as exc:
        raise DatabricksPfcSourceCostPreflightError(f"{label} is invalid") from exc
    canonical = format(decimal.normalize(), "f") if decimal else "0"
    if canonical != value:
        raise DatabricksPfcSourceCostPreflightError(f"{label} is not canonical")
    return decimal


def _utc_text_datetime(value: object, label: str) -> datetime:
    cleaned = _clean(value, label)
    if not cleaned.endswith("Z"):
        raise DatabricksPfcSourceCostPreflightError(f"{label} must use UTC Z form")
    try:
        parsed = datetime.fromisoformat(cleaned[:-1] + "+00:00")
    except ValueError as exc:
        raise DatabricksPfcSourceCostPreflightError(f"{label} is invalid") from exc
    normalized = _utc_datetime(parsed, label)
    if _utc_text(normalized) != cleaned:
        raise DatabricksPfcSourceCostPreflightError(f"{label} is not canonical")
    return normalized


def _utc_datetime(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise DatabricksPfcSourceCostPreflightError(f"{label} must be timezone-aware")
    normalized = value.astimezone(timezone.utc)
    if normalized.microsecond:
        raise DatabricksPfcSourceCostPreflightError(f"{label} must have whole-second precision")
    return normalized


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


__all__ = [
    "ASSESSMENT_SCHEMA",
    "AUTHORITIES",
    "CONTRACT_CONTENT_ID",
    "CONTRACT_PATH",
    "CostPreflightAssessment",
    "DatabricksPfcSourceCostPreflightError",
    "EVIDENCE_CLASS",
    "EXECUTION",
    "GO_AUTHORITY",
    "PROFILE_BINDINGS",
    "READY_STATUS",
    "RECEIPT_SCHEMA",
    "RUNNING_STATE",
    "STOPPED_STATUS",
    "UNCAPPED_STATUS",
    "canonical_sha256",
    "derive_snapshot_id",
    "validate_cost_preflight_contract",
    "validate_cost_preflight_contract_path",
    "validate_cost_preflight_receipt",
    "validate_cost_preflight_receipt_path",
]

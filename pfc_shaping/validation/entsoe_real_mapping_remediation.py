"""Compile a value-free remediation plan for the observed ENTSO-E schema.

The module deliberately emits no SQL.  It distinguishes safe physical-field
reuse from transformations that require source metadata or an accountable
owner decision.  In particular, the observed ``DateTimeUtc`` is a right edge
and therefore cannot be renamed to an interval-start timestamp.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.validation.entsoe_unity_catalog_schema_compatibility import (
    assess_unity_catalog_schema,
)

CONTRACT_SCHEMA = "fmv_entsoe_real_mapping_remediation_contract.v1"
CONTRACT_STATUS = (
    "STATIC_OFFLINE_REAL_SCHEMA_REMEDIATION_PLAN_NO_SQL_NO_MODEL_AUTHORITY"
)
CONTRACT_DECISION = "D-20260806-250"
CONTRACT_FILE_SHA256 = (
    "4a0d69a770fe0b45063bc4e5b4be4b9aca1e86c546b40d2bf8816a001cd06b8f"
)
CONTRACT_CONTENT_ID = (
    "56826ff4d97074a58f961e08f4c0fe2769431ad9ab3d1afebd5d032e635d8c7b"
)

D241_PROOF_CONTENT_ID = (
    "d6c006609d881b51f08be6d60e01f68b59a40be8bdf2898ef0a98491f5771544"
)
D241_CAPTURE_CONTENT_ID = (
    "d69fdab73ba1d9c55f70f77925f2253d583564d06d922f2b41e035763bca176f"
)
D241_MANIFEST_SHA256 = (
    "1835f93a517e9c6769079984a376fb9879b22d7c8ce2922aa42b0dd646627ada"
)
D241_CAPTURE_SHA256 = (
    "46efff7cf0f568a59c6d0146c25b0a9095f33eed77d765807f600f95662f6312"
)
D241_ASSESSMENT_SHA256 = (
    "eddf1cdb04a8ad2437cc4e90531c4f6bd756dc6575a7c5dc8f6840fed563d294"
)
D241_VALIDATOR_SHA256 = (
    "2e7dfd9e13e04a77371b0bb1b883f6db5a263e7990d11b1c2e6d2da19a7ae1f4"
)
D234_PROOF_CONTENT_ID = (
    "bb7ea1894463cb2c5fc30287d2239f0285cd6a74e901ab51a2f9de7e6794b766"
)
D239_PROOF_CONTENT_ID = (
    "5e5aad7d04529e0efbb9926a1098a485ab3f797941ba2681c0a1609487f4df9b"
)

TABLES = {
    "dimension": "dev.gold.dimentsoeseries",
    "latest": "dev.gold.factentsoetimeserieslatest",
    "vintages": "dev.gold.factentsoetimeseriesvintages",
}
RIGHT_EDGE_COMMENT = (
    "UTC delivery/value timestamp on the right edge of the ENTSO-E interval."
)
ROLE_REQUIRED_PHYSICAL_FIELDS = {
    "dimension": {
        "SeriesID",
        "FieldName",
        "GroupName",
        "FromZone",
        "ToZone",
        "Unit",
    },
    "latest": {
        "SeriesID",
        "DateTimeUtc",
        "FieldValue",
        "IsHistorical",
        "Meta_Load_Timestamp",
    },
    "vintages": {
        "VintageID",
        "SeriesID",
        "DateTimeUtc",
        "FieldValue",
        "PullTimestampUtc",
        "PublicationTimestampUtc",
        "Meta_Load_Timestamp",
    },
}
EXPECTED_BINDINGS = {
    "unity_catalog_schema_decision": "D-20260806-241",
    "unity_catalog_schema_proof_content_id": D241_PROOF_CONTENT_ID,
    "unity_catalog_schema_capture_content_id": D241_CAPTURE_CONTENT_ID,
    "unity_catalog_schema_proof_manifest_sha256": D241_MANIFEST_SHA256,
    "unity_catalog_capture_sha256": D241_CAPTURE_SHA256,
    "unity_catalog_assessment_sha256": D241_ASSESSMENT_SHA256,
    "unity_catalog_validator_sha256": D241_VALIDATOR_SHA256,
    "physical_mapping_compiler_decision": "D-20260805-234",
    "physical_mapping_compiler_proof_content_id": D234_PROOF_CONTENT_ID,
    "normalized_quality_profile_decision": "D-20260805-239",
    "normalized_quality_profile_proof_content_id": D239_PROOF_CONTENT_ID,
}
EXPECTED_TIME_SEMANTICS = {
    "target_ts_utc": "DELIVERY_INTERVAL_START_UTC",
    "physical_datetime_utc": "DELIVERY_INTERVAL_RIGHT_EDGE_UTC",
    "direct_alias_of_right_edge_to_target_start_authorized": False,
    "required_target_transform": "DateTimeUtc_MINUS_NATIVE_RESOLUTION",
    "native_resolution_required_before_target_transform": True,
    "dateutc_is_timestamp_authority": False,
    "epoch_is_timestamp_authority": False,
}
EXPECTED_PIT_POLICY = {
    "latest_is_historical_pit_source": False,
    "source_publication_timestamp": "PublicationTimestampUtc",
    "local_observation_timestamp": "PullTimestampUtc",
    "local_load_timestamp": "Meta_Load_Timestamp",
    "candidate_conservative_as_of": (
        "MAX_OF_PUBLICATION_PULL_AND_LOAD_AFTER_ALL_THREE_ARE_NON_NULL"
    ),
    "candidate_rule_is_owner_approved": False,
    "publication_timestamp_alone_authorized": False,
    "pull_timestamp_alone_authorized": False,
    "load_timestamp_alone_authorized": False,
    "coalesce_missing_availability_components_authorized": False,
    "backfill_gains_retroactive_availability": False,
}
EXPECTED_READINESS = {
    "direct_mapping_fields_verified": True,
    "series_id_cast_design_ready": True,
    "target_interval_start_mapping_ready": False,
    "native_resolution_ready": False,
    "point_in_time_rule_owner_approved": False,
    "source_lineage_ready": False,
    "quality_semantics_ready": False,
    "revision_semantics_ready": False,
    "sign_semantics_ready": False,
    "dimension_value_inventory_ready": False,
    "real_mapping_compilable_by_D234": False,
}
EXPECTED_AUTHORITIES = {
    "real_schema_observed": True,
    "physical_to_normalized_mapping_complete": False,
    "point_in_time_availability_proven": False,
    "real_values_opened": False,
    "training_authorized": False,
    "selection_authorized": False,
    "model_input_authorized": False,
    "candidate_assembly_authorized": False,
    "promotion_authorized": False,
    "production_authorized": False,
}


class EntsoeRealMappingRemediationError(ValueError):
    """Raised when the real-schema remediation evidence is unsafe or drifts."""


def validate_real_mapping_remediation_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact D243 value-free remediation contract."""

    contract = _mapping(document, "mapping remediation contract")
    _exact_fields(
        contract,
        {
            "schema_version",
            "status",
            "decision",
            "purpose",
            "input_bindings",
            "standards_basis",
            "normalized_time_semantics",
            "point_in_time_policy",
            "mapping_plan",
            "forbidden_substitutions",
            "data_engineer_required_decisions",
            "readiness_gates",
            "execution_policy",
            "authorities",
        },
        "mapping remediation contract",
    )
    _equal(contract.get("schema_version"), CONTRACT_SCHEMA, "contract schema")
    _equal(contract.get("status"), CONTRACT_STATUS, "contract status")
    _equal(contract.get("decision"), CONTRACT_DECISION, "contract decision")
    _equal(contract.get("input_bindings"), EXPECTED_BINDINGS, "input bindings")
    _validate_standards(contract.get("standards_basis"))
    _equal(
        contract.get("normalized_time_semantics"),
        EXPECTED_TIME_SEMANTICS,
        "normalized time semantics",
    )
    _equal(
        contract.get("point_in_time_policy"),
        EXPECTED_PIT_POLICY,
        "point-in-time policy",
    )
    _validate_mapping_plan(contract.get("mapping_plan"))
    forbidden = _mapping(
        contract.get("forbidden_substitutions"), "forbidden substitutions"
    )
    if not forbidden or any(value is not False for value in forbidden.values()):
        raise EntsoeRealMappingRemediationError(
            "every forbidden substitution must remain false"
        )
    decisions = contract.get("data_engineer_required_decisions")
    if not isinstance(decisions, list) or len(decisions) != 9:
        raise EntsoeRealMappingRemediationError(
            "data engineer decision inventory differs"
        )
    if not all(isinstance(item, str) and item.strip() == item for item in decisions):
        raise EntsoeRealMappingRemediationError(
            "data engineer decisions must be exact non-padded strings"
        )
    _equal(contract.get("readiness_gates"), EXPECTED_READINESS, "readiness gates")
    execution = _mapping(contract.get("execution_policy"), "execution policy")
    for field, value in execution.items():
        if field.endswith("_budget"):
            _equal(value, 0, f"execution budget {field}")
    _equal(execution.get("connector_code_allowed"), False, "connector authority")
    _equal(
        execution.get("compiled_sql_execution_authorized"),
        False,
        "compiled SQL authority",
    )
    _equal(contract.get("authorities"), EXPECTED_AUTHORITIES, "authorities")
    content_id = canonical_sha256(contract)
    _equal(content_id, CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": content_id,
        "real_schema_observed": True,
        "mapping_complete": False,
        "sql_generated": False,
        "model_input_authorized": False,
        "production_authorized": False,
    }


def assess_real_mapping_remediation(
    capture: Mapping[str, object],
    prior_assessment: Mapping[str, object],
) -> dict[str, object]:
    """Assess the admitted D241 schema without opening any table value."""

    capture_map = _mapping(capture, "D241 schema capture")
    prior = _mapping(prior_assessment, "D241 schema assessment")
    replayed = assess_unity_catalog_schema(capture_map)
    _equal(replayed, prior, "D241 schema assessment replay")
    _equal(
        canonical_sha256(capture_map),
        D241_CAPTURE_CONTENT_ID,
        "D241 capture content ID",
    )
    tables = _capture_tables(capture_map)
    _validate_required_physical_columns(tables)
    for role in ("latest", "vintages"):
        datetime_column = tables[role]["DateTimeUtc"]
        _equal(
            datetime_column.get("comment"),
            RIGHT_EDGE_COMMENT,
            f"{role} DateTimeUtc right-edge declaration",
        )
        _equal(
            datetime_column.get("type_text"),
            "timestamp",
            f"{role} DateTimeUtc type",
        )
    for role in TABLES:
        _equal(
            tables[role]["SeriesID"].get("type_text"),
            "bigint",
            f"{role} SeriesID type",
        )

    findings = [
        {
            "severity": "CRITICAL",
            "code": "RIGHT_EDGE_CANNOT_BE_DIRECT_TARGET_TIMESTAMP",
            "impact": (
                "Direct DateTimeUtc aliasing shifts every delivery interval by one "
                "native step and misaligns prices, load, generation and calendar features."
            ),
        },
        {
            "severity": "CRITICAL",
            "code": "NATIVE_RESOLUTION_REQUIRED_FOR_INTERVAL_START",
            "impact": (
                "The right edge cannot be converted to interval start until each "
                "series has an admitted native resolution."
            ),
        },
        {
            "severity": "CRITICAL",
            "code": "CONSERVATIVE_AS_OF_RULE_NOT_OWNER_APPROVED",
            "impact": (
                "Rolling-origin replay remains forbidden until publication, pull "
                "and load timestamps are jointly governed."
            ),
        },
        {
            "severity": "HIGH",
            "code": "SOURCE_LINEAGE_QUALITY_REVISION_SIGN_MISSING",
            "impact": (
                "Unknown provenance or direction cannot be replaced by defaults "
                "without corrupting quality and cross-border features."
            ),
        },
        {
            "severity": "HIGH",
            "code": "NULLABLE_REQUIRED_FIELDS_REQUIRE_VALUE_PROFILE",
            "impact": (
                "Rows may not be silently dropped or coalesced; coverage must be "
                "measured on the governed export."
            ),
        },
    ]
    return {
        "schema_version": "fmv_entsoe_real_mapping_remediation_assessment.v1",
        "status": "FAIL_REAL_MAPPING_REMEDIATION_REQUIRED_NO_MODEL_AUTHORITY",
        "source_capture_content_id": D241_CAPTURE_CONTENT_ID,
        "source_schema_proof_content_id": D241_PROOF_CONTENT_ID,
        "table_count": 3,
        "physical_column_counts": {
            role: len(columns) for role, columns in tables.items()
        },
        "verified_direct_or_safe_cast_fields": {
            "dimension": [
                "series_id",
                "group_name",
                "field_name",
                "from_zone",
                "to_zone",
                "unit",
            ],
            "latest": ["series_id", "value", "is_historical", "meta_load_timestamp_utc"],
            "vintages": ["series_id", "value", "meta_load_timestamp_utc"],
        },
        "blocked_normalized_fields": {
            "dimension": ["native_resolution", "source_endpoint", "document_id"],
            "latest": ["target_ts_utc", "quality_flag", "revision_number"],
            "vintages": [
                "target_ts_utc",
                "as_of_utc",
                "quality_flag",
                "revision_number",
            ],
        },
        "target_timestamp_policy": {
            "physical_semantics": "RIGHT_EDGE",
            "normalized_semantics": "INTERVAL_START",
            "required_transform": "RIGHT_EDGE_MINUS_NATIVE_RESOLUTION",
            "ready": False,
        },
        "point_in_time_policy": {
            "candidate": EXPECTED_PIT_POLICY["candidate_conservative_as_of"],
            "all_components_non_null_required": True,
            "owner_approved": False,
            "latest_allowed_for_historical_replay": False,
        },
        "findings": findings,
        "readiness_gates": dict(EXPECTED_READINESS),
        "execution": {
            "control_plane_get_count": 0,
            "databricks_connection_count": 0,
            "databricks_statement_count": 0,
            "warehouse_start_count": 0,
            "network_call_count": 0,
            "h_drive_access_count": 0,
            "remote_write_count": 0,
            "table_value_row_count_opened": 0,
        },
        "authorities": dict(EXPECTED_AUTHORITIES),
    }


def verify_real_mapping_remediation_paths(
    *,
    contract_path: str | Path,
    d241_manifest_path: str | Path,
    d241_capture_path: str | Path,
    d241_assessment_path: str | Path,
    d241_validator_path: str | Path,
) -> dict[str, object]:
    """Bind exact local D241 bytes and produce the D243 assessment offline."""

    contract_raw = _read(contract_path, "D243 contract", 200_000)
    _equal(
        hashlib.sha256(contract_raw).hexdigest(),
        CONTRACT_FILE_SHA256,
        "D243 contract SHA-256",
    )
    contract = _strict_json(contract_raw, "D243 contract")
    contract_result = validate_real_mapping_remediation_contract(contract)

    manifest_raw = _read(d241_manifest_path, "D241 proof manifest", 100_000)
    _equal(
        hashlib.sha256(manifest_raw).hexdigest(),
        D241_MANIFEST_SHA256,
        "D241 proof manifest SHA-256",
    )
    manifest = _strict_json(manifest_raw, "D241 proof manifest")
    _equal(manifest.get("content_id"), D241_PROOF_CONTENT_ID, "D241 proof ID")
    _equal(
        manifest.get("capture_content_id"),
        D241_CAPTURE_CONTENT_ID,
        "D241 capture ID",
    )
    execution = _mapping(manifest.get("execution"), "D241 execution")
    _equal(execution.get("sql_statement_count"), 0, "D241 SQL count")
    _equal(execution.get("warehouse_start_count"), 0, "D241 Warehouse starts")
    _equal(execution.get("databricks_write_count"), 0, "D241 writes")
    _equal(execution.get("table_row_count_opened"), 0, "D241 opened rows")

    capture_raw = _read(d241_capture_path, "D241 capture", 100_000)
    assessment_raw = _read(d241_assessment_path, "D241 assessment", 100_000)
    validator_raw = _read(d241_validator_path, "D241 validator", 2_000_000)
    _equal(
        hashlib.sha256(capture_raw).hexdigest(),
        D241_CAPTURE_SHA256,
        "D241 capture SHA-256",
    )
    _equal(
        hashlib.sha256(assessment_raw).hexdigest(),
        D241_ASSESSMENT_SHA256,
        "D241 assessment SHA-256",
    )
    _equal(
        hashlib.sha256(validator_raw).hexdigest(),
        D241_VALIDATOR_SHA256,
        "D241 validator SHA-256",
    )
    artifacts = _mapping(manifest.get("artifacts"), "D241 artifacts")
    _equal(
        _mapping(artifacts.get("capture"), "capture artifact").get("sha256"),
        D241_CAPTURE_SHA256,
        "D241 manifest capture hash",
    )
    _equal(
        _mapping(artifacts.get("assessment"), "assessment artifact").get("sha256"),
        D241_ASSESSMENT_SHA256,
        "D241 manifest assessment hash",
    )
    assessment = assess_real_mapping_remediation(
        _strict_json(capture_raw, "D241 capture"),
        _strict_json(assessment_raw, "D241 assessment"),
    )
    return {
        **assessment,
        "contract_content_id": contract_result["contract_content_id"],
        "source_proof_manifest_sha256": D241_MANIFEST_SHA256,
    }


def data_engineer_request_lines(assessment: Mapping[str, object]) -> list[str]:
    """Return a concise value-free handoff derived from a verified assessment."""

    result = _mapping(assessment, "mapping remediation assessment")
    _equal(
        result.get("status"),
        "FAIL_REAL_MAPPING_REMEDIATION_REQUIRED_NO_MODEL_AUTHORITY",
        "assessment status",
    )
    return [
        "ENTSO-E dev.gold - mapping fields required before PFC use:",
        "1. Publish Period.resolution per SeriesID (do not infer it from the schema).",
        "2. Confirm DateTimeUtc is the interval right edge; export target_ts_utc as right edge minus native resolution.",
        "3. Preserve PublicationTimestampUtc, PullTimestampUtc and Meta_Load_Timestamp; approve as_of_utc=max(all three), with no null component.",
        "4. Publish source document mRID/immutable ID and source endpoint; DocumentType and VintageID are not substitutes.",
        "5. Publish source quality/reason and revisionNumber; never default unknown quality to OK or revision to zero.",
        "6. Publish positive-direction sign semantics per directional series and the exact GroupName/FieldName/unit/border inventory.",
        "7. Report null rejection and coverage loss; do not silently coalesce or drop required fields.",
        "No Databricks write is required or authorized.",
    ]


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_standards(raw: object) -> None:
    standards = _mapping(raw, "standards basis")
    _equal(
        standards.get("entsoe_time_step_formula"),
        "period_start_plus_position_minus_one_times_resolution",
        "ENTSO-E time-step formula",
    )
    guide = standards.get("entsoe_time_series_guide")
    if not isinstance(guide, str) or not guide.startswith("https://www.entsoe.eu/"):
        raise EntsoeRealMappingRemediationError("ENTSO-E guide URL is invalid")
    _equal(
        standards.get("local_table_timestamp_comment"),
        "DateTimeUtc_is_the_UTC_right_edge_of_the_ENTSO_E_interval",
        "local timestamp declaration",
    )


def _validate_mapping_plan(raw: object) -> None:
    plan = _mapping(raw, "mapping plan")
    _equal(set(plan), set(TABLES), "mapping plan roles")
    for role, table in TABLES.items():
        spec = _mapping(plan.get(role), f"mapping role {role}")
        _exact_fields(spec, {"physical_table", "normalized_role", "fields"}, role)
        _equal(spec.get("physical_table"), table, f"physical table {role}")
        fields = _mapping(spec.get("fields"), f"mapping fields {role}")
        if not fields:
            raise EntsoeRealMappingRemediationError(f"mapping fields are empty: {role}")
        for logical, raw_field in fields.items():
            field = _mapping(raw_field, f"mapping field {role}.{logical}")
            mapping_class = field.get("class")
            if not isinstance(mapping_class, str) or not mapping_class:
                raise EntsoeRealMappingRemediationError(
                    f"mapping class is invalid: {role}.{logical}"
                )
    latest_fields = _mapping(
        _mapping(plan["latest"], "latest mapping")["fields"], "latest fields"
    )
    vintage_fields = _mapping(
        _mapping(plan["vintages"], "vintage mapping")["fields"], "vintage fields"
    )
    for role, fields in (("latest", latest_fields), ("vintages", vintage_fields)):
        target = _mapping(fields.get("target_ts_utc"), f"{role} target mapping")
        _equal(target.get("source"), "DateTimeUtc", f"{role} target source")
        _equal(
            target.get("transform"),
            "RIGHT_EDGE_MINUS_NATIVE_RESOLUTION",
            f"{role} target transform",
        )
    as_of = _mapping(vintage_fields.get("as_of_utc"), "vintage as-of mapping")
    _equal(
        as_of.get("sources"),
        ["PublicationTimestampUtc", "PullTimestampUtc", "Meta_Load_Timestamp"],
        "as-of sources",
    )
    _equal(
        as_of.get("candidate_transform"),
        "MAX_AFTER_ALL_NON_NULL",
        "as-of candidate transform",
    )


def _capture_tables(
    capture: Mapping[str, object],
) -> dict[str, dict[str, Mapping[str, object]]]:
    raw_tables = capture.get("tables")
    if not isinstance(raw_tables, list) or len(raw_tables) != len(TABLES):
        raise EntsoeRealMappingRemediationError("D241 table inventory differs")
    by_name: dict[str, Mapping[str, object]] = {}
    for raw in raw_tables:
        table = _mapping(raw, "D241 table")
        name = table.get("full_name")
        if not isinstance(name, str) or name in by_name:
            raise EntsoeRealMappingRemediationError("D241 table name is invalid")
        by_name[name] = table
    _equal(set(by_name), set(TABLES.values()), "D241 table names")
    result: dict[str, dict[str, Mapping[str, object]]] = {}
    for role, name in TABLES.items():
        raw_columns = by_name[name].get("columns")
        if not isinstance(raw_columns, list):
            raise EntsoeRealMappingRemediationError(f"columns are invalid: {role}")
        columns: dict[str, Mapping[str, object]] = {}
        for raw_column in raw_columns:
            column = _mapping(raw_column, f"column {role}")
            column_name = column.get("name")
            if not isinstance(column_name, str) or column_name in columns:
                raise EntsoeRealMappingRemediationError(
                    f"column name is invalid or duplicated: {role}"
                )
            columns[column_name] = column
        result[role] = columns
    return result


def _validate_required_physical_columns(
    tables: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> None:
    for role, required in ROLE_REQUIRED_PHYSICAL_FIELDS.items():
        missing = required.difference(tables[role])
        if missing:
            raise EntsoeRealMappingRemediationError(
                f"required physical columns are missing: {role} {sorted(missing)}"
            )


def _read(path: str | Path, label: str, max_bytes: int) -> bytes:
    try:
        return read_stable_single_link_file(path, label=label, max_bytes=max_bytes)
    except (OSError, ValueError) as exc:
        raise EntsoeRealMappingRemediationError(f"{label} read failed") from exc


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EntsoeRealMappingRemediationError(
                    f"{label} duplicate JSON key: {key}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeRealMappingRemediationError(f"{label} JSON is invalid") from exc
    return _mapping(value, label)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeRealMappingRemediationError(f"{label} must be a mapping")
    return value


def _exact_fields(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EntsoeRealMappingRemediationError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoeRealMappingRemediationError(
            f"{label} differs: expected {expected!r}, received {value!r}"
        )


__all__ = [
    "CONTRACT_CONTENT_ID",
    "CONTRACT_FILE_SHA256",
    "EntsoeRealMappingRemediationError",
    "assess_real_mapping_remediation",
    "canonical_sha256",
    "data_engineer_request_lines",
    "validate_real_mapping_remediation_contract",
    "verify_real_mapping_remediation_paths",
]

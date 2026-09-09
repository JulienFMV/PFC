"""Compile admitted ENTSO-E metadata into non-executable explicit SQL templates."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.validation.entsoe_metadata_admission import (
    CONTRACT_CONTENT_ID as METADATA_ADMISSION_CONTENT_ID,
)
from pfc_shaping.validation.entsoe_metadata_admission import (
    admit_entsoe_metadata_capture,
)

CONTRACT_SCHEMA = "fmv_entsoe_physical_mapping_compiler_contract.v1"
CONTRACT_STATUS = "STATIC_OFFLINE_COMPILER_CONTRACT_NO_EXECUTION_AUTHORITY"
CONTRACT_FILE_SHA256 = (
    "f33bc696d8e7a535f7700b40c13dc90d73e2526073400edcd2588e57432a9b3a"
)
CONTRACT_CONTENT_ID = (
    "5dc43e18dcf107020e24638307a707d8c7afbf894ca1131e62907f8680ad0951"
)
GOVERNED_PACKAGE_CONTRACT_SHA256 = (
    "97edaf93c639953365d20d7539380a706f0f8fdbec9476af94262fb1d4b2204d"
)
GOVERNED_PACKAGE_CONTRACT_CONTENT_ID = (
    "4f6e644b15593e8d4ae2ca196e1d52a002d4255bdf42e7a137982edb854f0fa0"
)
INTAKE_CONTRACT_SHA256 = (
    "7ede1698099390babfa1d130bfecae61fd1e090888a3d6e6e4f892119db52b87"
)
MAPPING_SCHEMA = "fmv_entsoe_physical_column_mapping.v1"
EVIDENCE_CLASSES = frozenset(
    {"REAL_METADATA_OWNER_ASSERTED", "SYNTHETIC_TEST_ONLY"}
)

ROLE_SPECS = {
    "series_dimension": {
        "physical_table": "dimentsoeseries",
        "grain": ("series_id",),
        "required_fields": (
            "series_id",
            "group_name",
            "field_name",
            "from_zone",
            "to_zone",
            "unit",
            "native_resolution",
            "source_endpoint",
            "document_id",
        ),
    },
    "latest_values": {
        "physical_table": "factentsoetimeserieslatest",
        "grain": ("series_id", "target_ts_utc"),
        "required_fields": (
            "series_id",
            "target_ts_utc",
            "value",
            "is_historical",
            "quality_flag",
            "revision_number",
            "meta_load_timestamp_utc",
        ),
    },
    "vintage_values": {
        "physical_table": "factentsoetimeseriesvintages",
        "grain": ("series_id", "target_ts_utc", "as_of_utc"),
        "required_fields": (
            "series_id",
            "target_ts_utc",
            "as_of_utc",
            "value",
            "quality_flag",
            "revision_number",
            "meta_load_timestamp_utc",
        ),
    },
}
TYPE_CLASSES = {
    "identifier": frozenset(
        {"STRING", "VARCHAR", "CHAR", "BIGINT", "INT", "INTEGER", "LONG", "SMALLINT"}
    ),
    "string": frozenset({"STRING", "VARCHAR", "CHAR"}),
    "timestamp_utc_asserted": frozenset({"TIMESTAMP", "TIMESTAMP_NTZ"}),
    "numeric": frozenset(
        {
            "DECIMAL",
            "DOUBLE",
            "FLOAT",
            "REAL",
            "BIGINT",
            "INT",
            "INTEGER",
            "LONG",
            "SMALLINT",
            "TINYINT",
        }
    ),
    "integer": frozenset(
        {"BIGINT", "INT", "INTEGER", "LONG", "SMALLINT", "TINYINT"}
    ),
    "boolean": frozenset({"BOOLEAN"}),
}
FIELD_TYPE_CLASSES = {
    "series_id": "identifier",
    "group_name": "string",
    "field_name": "string",
    "from_zone": "string",
    "to_zone": "string",
    "unit": "string",
    "native_resolution": "string",
    "source_endpoint": "string",
    "document_id": "string",
    "target_ts_utc": "timestamp_utc_asserted",
    "as_of_utc": "timestamp_utc_asserted",
    "value": "numeric",
    "is_historical": "boolean",
    "quality_flag": "string",
    "revision_number": "integer",
    "meta_load_timestamp_utc": "timestamp_utc_asserted",
}
CROSS_TABLE_EQUALITIES = (
    "series_id",
    "target_ts_utc",
    "value",
    "quality_flag",
    "revision_number",
    "meta_load_timestamp_utc",
)
SEMANTIC_ASSERTIONS = {
    "timestamp_timezone": "UTC",
    "source_timestamps_are_utc_owner_asserted": True,
    "native_resolution_is_source_field_not_inferred": True,
    "implicit_upsampling_authorized": False,
    "implicit_forward_fill_authorized": False,
    "implicit_casts_authorized": False,
    "latest_values_point_in_time_authority": False,
    "vintage_availability_field": "as_of_utc",
    "day_ahead_price_unit": "EUR/MWh",
}
MAPPING_AUTHORITIES = {
    "external_mapping_owner_identity_verified": False,
    "execution_authorized": False,
    "point_in_time_availability_proven": False,
    "model_input_authorized": False,
    "production_authorized": False,
}
TOP_LEVEL_MAPPING_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_class",
        "mapping_owner",
        "mapped_at_utc",
        "metadata_schema_fingerprint",
        "metadata_payload_sha256",
        "metadata_statement_id",
        "catalog",
        "schema",
        "table_mappings",
        "semantic_assertions",
        "authorities",
    }
)
ROLE_LIMITS = {
    "series_dimension": 100_001,
    "latest_values": 1_000_001,
    "vintage_values": 1_000_001,
}
ROLE_PARAMETERS = {
    "series_dimension": (),
    "latest_values": ("start_utc", "end_utc"),
    "vintage_values": ("start_utc", "end_utc", "as_of_cutoff_utc"),
}

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OWNER = re.compile(r"^[A-Za-z0-9_.@ -]{3,128}$")
_UTC_SECOND = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_FORBIDDEN_SQL = re.compile(
    r"\b(?:alter|call|copy|create|delete|drop|execute|grant|insert|merge|"
    r"optimize|revoke|set|truncate|update|use|vacuum)\b",
    re.IGNORECASE,
)


class EntsoePhysicalMappingError(ValueError):
    """Raised when a physical mapping or compiled SQL is ambiguous or unsafe."""


def validate_physical_mapping_compiler_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact D234 static compiler contract."""

    contract = _mapping(document, "physical mapping compiler contract")
    _equal(contract.get("schema_version"), CONTRACT_SCHEMA, "contract schema")
    _equal(contract.get("status"), CONTRACT_STATUS, "contract status")
    bindings = _mapping(contract.get("input_bindings"), "input bindings")
    _equal(
        bindings.get("governed_acquisition_package_contract_content_id"),
        GOVERNED_PACKAGE_CONTRACT_CONTENT_ID,
        "governed acquisition package content ID",
    )
    _equal(
        bindings.get("governed_acquisition_package_contract_sha256"),
        GOVERNED_PACKAGE_CONTRACT_SHA256,
        "governed acquisition package SHA-256",
    )
    _equal(
        bindings.get("governed_acquisition_package_decision"),
        "D-20260805-233",
        "governed acquisition package decision",
    )
    _equal(
        bindings.get("metadata_admission_contract_content_id"),
        METADATA_ADMISSION_CONTENT_ID,
        "metadata admission content ID",
    )
    _equal(
        bindings.get("entsoe_intake_contract_sha256"),
        INTAKE_CONTRACT_SHA256,
        "ENTSO-E intake contract SHA-256",
    )
    roles = _mapping(contract.get("normalized_roles"), "normalized roles")
    _equal(set(roles), set(ROLE_SPECS), "normalized role names")
    for role, expected in ROLE_SPECS.items():
        spec = _mapping(roles.get(role), f"role {role}")
        _equal(spec.get("physical_table"), expected["physical_table"], f"{role} table")
        _equal(spec.get("grain"), list(expected["grain"]), f"{role} grain")
        _equal(
            spec.get("required_fields"),
            list(expected["required_fields"]),
            f"{role} fields",
        )
    _equal(contract.get("field_type_classes"), FIELD_TYPE_CLASSES, "field type classes")
    _equal(
        contract.get("cross_table_type_equalities"),
        list(CROSS_TABLE_EQUALITIES),
        "cross-table type equalities",
    )
    _equal(
        contract.get("semantic_assertions_required"),
        SEMANTIC_ASSERTIONS,
        "semantic assertions",
    )
    sql = _mapping(contract.get("sql_templates"), "SQL templates")
    for role in ROLE_SPECS:
        role_sql = _mapping(sql.get(role), f"SQL template {role}")
        _equal(role_sql.get("parameters"), list(ROLE_PARAMETERS[role]), f"{role} params")
        _equal(role_sql.get("row_limit"), ROLE_LIMITS[role], f"{role} row limit")
    execution = _mapping(contract.get("execution_policy"), "execution policy")
    for field in (
        "current_databricks_statement_budget",
        "current_warehouse_start_budget",
        "current_network_call_budget",
        "current_remote_write_budget",
    ):
        _equal(execution.get(field), 0, f"execution policy {field}")
    _equal(
        execution.get("compiled_sql_execution_authorized"),
        False,
        "compiled SQL execution authority",
    )
    authorities = _mapping(contract.get("authorities"), "authorities")
    for field, value in authorities.items():
        _equal(value, False, f"authority {field}")
    content_id = _canonical_sha256(contract)
    _equal(content_id, CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": content_id,
        "current_statement_budget": 0,
        "compiled_sql_execution_authorized": False,
        "production_authorized": False,
    }


def compile_entsoe_physical_mapping(
    receipt: Mapping[str, object],
    metadata_payload: bytes,
    mapping_document: Mapping[str, object],
    *,
    reference_time_utc: datetime,
) -> dict[str, object]:
    """Compile safe SQL templates after D232 metadata admission and mapping checks."""

    admission = admit_entsoe_metadata_capture(
        receipt,
        metadata_payload,
        reference_time_utc=reference_time_utc,
    )
    metadata = _metadata_lookup(metadata_payload)
    mapping = _mapping(mapping_document, "physical mapping")
    if set(mapping) != TOP_LEVEL_MAPPING_FIELDS:
        raise EntsoePhysicalMappingError(
            "physical mapping fields must match the exact contract"
        )
    _equal(mapping.get("schema_version"), MAPPING_SCHEMA, "mapping schema")
    evidence_class = mapping.get("evidence_class")
    if evidence_class not in EVIDENCE_CLASSES:
        raise EntsoePhysicalMappingError("mapping evidence_class is invalid")
    owner = _required_string(mapping.get("mapping_owner"), "mapping owner")
    if _OWNER.fullmatch(owner) is None:
        raise EntsoePhysicalMappingError("mapping owner format is invalid")
    if evidence_class == "SYNTHETIC_TEST_ONLY" and owner != "SYNTHETIC_TEST_FIXTURE":
        raise EntsoePhysicalMappingError("synthetic mapping owner differs")
    if evidence_class == "REAL_METADATA_OWNER_ASSERTED" and owner.startswith(
        "SYNTHETIC"
    ):
        raise EntsoePhysicalMappingError("real mapping owner is synthetic")
    mapped_at = _parse_utc_second(mapping.get("mapped_at_utc"), "mapped_at_utc")
    reference = _utc_datetime(reference_time_utc, "reference_time_utc")
    captured_at = _parse_utc_second(admission["captured_at_utc"], "captured_at_utc")
    if mapped_at < captured_at or mapped_at > reference:
        raise EntsoePhysicalMappingError(
            "mapped_at_utc must be between capture and reference time"
        )
    _equal(
        mapping.get("metadata_schema_fingerprint"),
        admission["schema_fingerprint"],
        "metadata schema fingerprint",
    )
    _equal(
        mapping.get("metadata_payload_sha256"),
        admission["payload_sha256"],
        "metadata payload SHA-256",
    )
    _equal(
        mapping.get("metadata_statement_id"),
        admission["statement_id"],
        "metadata statement ID",
    )
    _equal(mapping.get("catalog"), "dev", "mapping catalog")
    _equal(mapping.get("schema"), "gold", "mapping schema name")
    _equal(
        mapping.get("semantic_assertions"),
        SEMANTIC_ASSERTIONS,
        "mapping semantic assertions",
    )
    _equal(mapping.get("authorities"), MAPPING_AUTHORITIES, "mapping authorities")

    mapped_columns, mapped_types = _validate_table_mappings(
        mapping.get("table_mappings"),
        metadata,
    )
    _validate_cross_table_types(mapped_types)
    templates = {
        role: _compile_role_sql(role, mapped_columns[role])
        for role in ROLE_SPECS
    }
    for role, sql in templates.items():
        _validate_compiled_sql(role, sql, mapped_columns[role])
    template_hashes = {
        role: hashlib.sha256(sql.encode("utf-8")).hexdigest()
        for role, sql in templates.items()
    }
    mapping_content_id = _canonical_sha256(mapping)
    real_metadata_asserted = evidence_class == "REAL_METADATA_OWNER_ASSERTED"
    return {
        "schema_version": "fmv_entsoe_physical_mapping_compilation.v1",
        "status": (
            "PASS_REAL_METADATA_OWNER_ASSERTED_MAPPING_EXECUTION_NOT_AUTHORIZED"
            if real_metadata_asserted
            else "PASS_SYNTHETIC_MAPPING_FIXTURE_EXECUTION_NOT_AUTHORIZED"
        ),
        "mapping_content_id": mapping_content_id,
        "mapping_evidence_class": evidence_class,
        "mapping_owner": owner,
        "metadata_schema_fingerprint": admission["schema_fingerprint"],
        "metadata_payload_sha256": admission["payload_sha256"],
        "metadata_statement_id": admission["statement_id"],
        "mapped_physical_types": mapped_types,
        "sql_templates": templates,
        "sql_template_sha256": template_hashes,
        "sql_parameters": {
            role: list(ROLE_PARAMETERS[role]) for role in ROLE_SPECS
        },
        "sql_row_limits": ROLE_LIMITS,
        "physical_mapping_matches_admitted_metadata": True,
        "real_metadata_evidence_asserted": real_metadata_asserted,
        "external_mapping_owner_identity_verified": False,
        "compiled_sql_execution_authorized": False,
        "entsoe_values_opened": False,
        "point_in_time_availability_proven": False,
        "training_authorized": False,
        "selection_authorized": False,
        "model_input_authorized": False,
        "candidate_assembly_authorized": False,
        "promotion_authorized": False,
        "production_authorized": False,
        "validator_databricks_request_count": 0,
        "validator_warehouse_start_count": 0,
        "validator_network_call_count": 0,
        "validator_remote_write_count": 0,
    }


def verify_entsoe_physical_mapping_paths(
    *,
    compiler_contract_path: str | Path,
    governed_package_contract_path: str | Path,
    intake_contract_path: str | Path,
    receipt_path: str | Path,
    metadata_payload_path: str | Path,
    mapping_path: str | Path,
    reference_time_utc: datetime,
) -> dict[str, object]:
    """Bind exact local files and compile without any connector or execution."""

    contract_raw = _read_path(
        compiler_contract_path,
        label="D234 physical mapping compiler contract",
        max_bytes=100_000,
    )
    if hashlib.sha256(contract_raw).hexdigest() != CONTRACT_FILE_SHA256:
        raise EntsoePhysicalMappingError("compiler contract SHA-256 differs")
    contract = _strict_json(contract_raw, "physical mapping compiler contract")
    contract_assessment = validate_physical_mapping_compiler_contract(contract)
    governed_package_raw = _read_path(
        governed_package_contract_path,
        label="D233 governed acquisition package contract",
        max_bytes=100_000,
    )
    if hashlib.sha256(governed_package_raw).hexdigest() != GOVERNED_PACKAGE_CONTRACT_SHA256:
        raise EntsoePhysicalMappingError(
            "D233 governed acquisition package contract SHA-256 differs"
        )
    governed_package = _strict_json(
        governed_package_raw,
        "D233 governed acquisition package contract",
    )
    _equal(
        _canonical_sha256(governed_package),
        GOVERNED_PACKAGE_CONTRACT_CONTENT_ID,
        "D233 governed acquisition package content ID",
    )
    intake_raw = _read_path(
        intake_contract_path,
        label="ENTSO-E intake contract",
        max_bytes=100_000,
    )
    if hashlib.sha256(intake_raw).hexdigest() != INTAKE_CONTRACT_SHA256:
        raise EntsoePhysicalMappingError("ENTSO-E intake contract SHA-256 differs")
    receipt = _strict_json(
        _read_path(receipt_path, label="metadata receipt", max_bytes=100_000),
        "metadata receipt",
    )
    payload = _read_path(
        metadata_payload_path,
        label="metadata payload",
        max_bytes=262_144,
    )
    mapping = _strict_json(
        _read_path(mapping_path, label="physical mapping", max_bytes=100_000),
        "physical mapping",
    )
    result = compile_entsoe_physical_mapping(
        receipt,
        payload,
        mapping,
        reference_time_utc=reference_time_utc,
    )
    return {
        **result,
        "compiler_contract_content_id": contract_assessment["contract_content_id"],
        "governed_acquisition_package_contract_content_id": (
            GOVERNED_PACKAGE_CONTRACT_CONTENT_ID
        ),
    }


def _metadata_lookup(payload: bytes) -> dict[str, dict[str, str]]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    metadata: dict[str, dict[str, str]] = {}
    for row in reader:
        table = str(row["table_name"])
        column = str(row["column_name"])
        metadata.setdefault(table, {})[column] = str(row["data_type"]).upper()
    return metadata


def _validate_table_mappings(
    raw_mappings: object,
    metadata: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    mappings = _mapping(raw_mappings, "table mappings")
    _equal(set(mappings), set(ROLE_SPECS), "table mapping roles")
    mapped_columns: dict[str, dict[str, str]] = {}
    mapped_types: dict[str, dict[str, str]] = {}
    for role, spec in ROLE_SPECS.items():
        role_mapping = _mapping(mappings.get(role), f"mapping role {role}")
        if set(role_mapping) != {"physical_table", "columns"}:
            raise EntsoePhysicalMappingError(f"mapping role fields differ: {role}")
        table = str(spec["physical_table"])
        _equal(role_mapping.get("physical_table"), table, f"{role} physical table")
        columns = _mapping(role_mapping.get("columns"), f"mapping columns {role}")
        required = tuple(str(field) for field in spec["required_fields"])
        _equal(set(columns), set(required), f"{role} normalized fields")
        physical_columns: dict[str, str] = {}
        physical_types: dict[str, str] = {}
        used: set[str] = set()
        available = metadata.get(table, {})
        for normalized in required:
            physical = _required_string(columns.get(normalized), f"{role}.{normalized}")
            if _IDENTIFIER.fullmatch(physical) is None:
                raise EntsoePhysicalMappingError(
                    f"unsafe physical identifier: {role}.{normalized}={physical}"
                )
            folded = physical.casefold()
            if folded in used:
                raise EntsoePhysicalMappingError(
                    f"physical mapping is not injective: {role}.{physical}"
                )
            used.add(folded)
            if physical not in available:
                raise EntsoePhysicalMappingError(
                    f"physical column is absent from admitted metadata: {table}.{physical}"
                )
            data_type = str(available[physical]).upper()
            type_class = FIELD_TYPE_CLASSES[normalized]
            if data_type not in TYPE_CLASSES[type_class]:
                raise EntsoePhysicalMappingError(
                    f"physical type is incompatible: {role}.{normalized}={data_type}"
                )
            physical_columns[normalized] = physical
            physical_types[normalized] = data_type
        mapped_columns[role] = physical_columns
        mapped_types[role] = physical_types
    return mapped_columns, mapped_types


def _validate_cross_table_types(
    mapped_types: Mapping[str, Mapping[str, str]],
) -> None:
    for field in CROSS_TABLE_EQUALITIES:
        observed = {
            types[field]
            for types in mapped_types.values()
            if field in types
        }
        if len(observed) != 1:
            raise EntsoePhysicalMappingError(
                f"cross-table physical types differ for {field}: {sorted(observed)}"
            )


def _compile_role_sql(role: str, columns: Mapping[str, str]) -> str:
    spec = ROLE_SPECS[role]
    required = tuple(str(field) for field in spec["required_fields"])
    projections = ",\n".join(
        f"  `{columns[field]}` AS `{field}`" for field in required
    )
    table = str(spec["physical_table"])
    lines = ["SELECT", projections, f"FROM `dev`.`gold`.`{table}`"]
    if role in {"latest_values", "vintage_values"}:
        target = columns["target_ts_utc"]
        lines.extend(
            [
                f"WHERE `{target}` >= :start_utc",
                f"  AND `{target}` < :end_utc",
            ]
        )
    if role == "vintage_values":
        lines.append(f"  AND `{columns['as_of_utc']}` <= :as_of_cutoff_utc")
    order = ", ".join(f"`{columns[field]}`" for field in spec["grain"])
    lines.extend([f"ORDER BY {order}", f"LIMIT {ROLE_LIMITS[role]}"])
    return "\n".join(lines) + "\n"


def _validate_compiled_sql(
    role: str,
    sql: str,
    columns: Mapping[str, str],
) -> None:
    lowered = sql.casefold()
    if not sql.startswith("SELECT\n") or not sql.endswith("\n") or "\r" in sql:
        raise EntsoePhysicalMappingError(f"compiled SQL framing differs: {role}")
    if ";" in sql or "--" in sql or "/*" in sql or "*/" in sql:
        raise EntsoePhysicalMappingError(f"compiled SQL comments forbidden: {role}")
    structural_sql = re.sub(r"`[A-Za-z_][A-Za-z0-9_]*`", "``", sql)
    if "select *" in lowered or _FORBIDDEN_SQL.search(structural_sql):
        raise EntsoePhysicalMappingError(f"compiled SQL token forbidden: {role}")
    if len(re.findall(r"\bselect\b", lowered)) != 1:
        raise EntsoePhysicalMappingError(f"compiled SQL SELECT count differs: {role}")
    expected_table = str(ROLE_SPECS[role]["physical_table"])
    if f"FROM `dev`.`gold`.`{expected_table}`" not in sql:
        raise EntsoePhysicalMappingError(f"compiled SQL table differs: {role}")
    expected_parameters = {f":{name}" for name in ROLE_PARAMETERS[role]}
    actual_parameters = set(re.findall(r":[A-Za-z_][A-Za-z0-9_]*", sql))
    if actual_parameters != expected_parameters:
        raise EntsoePhysicalMappingError(f"compiled SQL parameters differ: {role}")
    if f"LIMIT {ROLE_LIMITS[role]}\n" not in sql:
        raise EntsoePhysicalMappingError(f"compiled SQL limit differs: {role}")
    for normalized, physical in columns.items():
        projection = f"`{physical}` AS `{normalized}`"
        if sql.count(projection) != 1:
            raise EntsoePhysicalMappingError(
                f"compiled SQL projection differs: {role}.{normalized}"
            )


def _parse_utc_second(value: object, label: str) -> datetime:
    if not isinstance(value, str) or _UTC_SECOND.fullmatch(value) is None:
        raise EntsoePhysicalMappingError(f"{label} must be UTC at whole seconds")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise EntsoePhysicalMappingError(f"{label} is invalid") from exc


def _utc_datetime(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EntsoePhysicalMappingError(f"{label} must be timezone-aware UTC")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise EntsoePhysicalMappingError(f"{label} must be timezone-aware UTC")
    return value


def _read_path(raw_path: str | Path, *, label: str, max_bytes: int) -> bytes:
    path = Path(raw_path)
    if not path.is_absolute():
        raise EntsoePhysicalMappingError(f"{label} path must be absolute")
    try:
        return read_stable_single_link_file(path, label=label, max_bytes=max_bytes)
    except (OSError, ValueError) as exc:
        raise EntsoePhysicalMappingError(f"{label} path read failed") from exc


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EntsoePhysicalMappingError(f"{label} duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoePhysicalMappingError(f"{label} JSON is invalid") from exc
    return _mapping(value, label)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoePhysicalMappingError(f"{label} must be a mapping")
    return value


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EntsoePhysicalMappingError(f"{label} must be a non-empty string")
    return value


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoePhysicalMappingError(
            f"{label} differs: expected {expected!r}, received {value!r}"
        )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

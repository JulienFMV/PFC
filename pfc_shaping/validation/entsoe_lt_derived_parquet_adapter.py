"""Admit one bounded synthetic Parquet package into the D283 executor.

The adapter is deliberately not a raw ENTSO-E reader. It validates a fixed
normalized Arrow schema, preserves decimal text exactly, splits rows by an
explicit role and delegates every semantic/arithmetic check to D283.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pfc_shaping.parquet_safety import validate_parquet_allocation_budget
from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.validation import entsoe_lt_derived_arithmetic_execution as d283
from pfc_shaping.validation import entsoe_lt_derived_physical_compatibility as d282

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = PHASE / "ENTSOE-LT-DERIVED-PARQUET-ADAPTER-CONTRACT-V1.json"

CONTRACT_SCHEMA = "fmv_entsoe_lt_derived_parquet_adapter_contract.v1"
CONTRACT_STATUS = "SYNTHETIC_BOUNDED_PARQUET_ADAPTER_NO_REAL_OR_MODEL_AUTHORITY"
CONTRACT_FILE_SHA256 = "58133559b8079b35e244fa31bf7e32dc85d35acb7c16698103715f60feb8a3ae"
CONTRACT_CONTENT_ID = "1ebcd75defc4e7641927117a56f19c718fa4278ef9fae2dfc16c735ee8b0635c"
MANIFEST_SCHEMA = "fmv_entsoe_lt_synthetic_parquet_operand_manifest.v1"
ASSESSMENT_SCHEMA = "fmv_entsoe_lt_derived_parquet_adapter_assessment.v1"
PASS_STATUS = "PASS_SYNTHETIC_BOUNDED_PARQUET_ADAPTER_NO_REAL_OR_MODEL_AUTHORITY"
EVIDENCE_CLASS = d283.EVIDENCE_CLASS
ARTIFACT_RELATIVE_PATH = "derived_operand_values.parquet"
MAX_MANIFEST_BYTES = 100_000
MAX_PARQUET_BYTES = 64 * 1024 * 1024
MAX_ROWS = 200_000
EXACT_COLUMNS = 9
MAX_CELLS = 1_800_000
MAX_ROW_GROUPS = 1_024
ALLOWED_PHYSICAL_TYPES = ("BYTE_ARRAY", "INT64")
COMPRESSION = "ZSTD"

COLUMN_SPEC = [
    {"name": "feature_record_id", "arrow_type": "string", "nullable": False},
    {"name": "role", "arrow_type": "string", "nullable": False},
    {
        "name": "target_start_utc",
        "arrow_type": "timestamp[us, tz=UTC]",
        "nullable": False,
    },
    {
        "name": "target_end_utc",
        "arrow_type": "timestamp[us, tz=UTC]",
        "nullable": False,
    },
    {"name": "series_id_sha256", "arrow_type": "string", "nullable": False},
    {"name": "logical_zone_sha256", "arrow_type": "string", "nullable": False},
    {"name": "canonical_unit", "arrow_type": "string", "nullable": False},
    {
        "name": "effective_native_resolution",
        "arrow_type": "string",
        "nullable": False,
    },
    {"name": "value_decimal_mw", "arrow_type": "string", "nullable": True},
]
EXPECTED_ARROW_SCHEMA = pa.schema(
    [
        pa.field("feature_record_id", pa.string(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field(
            "target_start_utc",
            pa.timestamp("us", tz="UTC"),
            nullable=False,
        ),
        pa.field(
            "target_end_utc",
            pa.timestamp("us", tz="UTC"),
            nullable=False,
        ),
        pa.field("series_id_sha256", pa.string(), nullable=False),
        pa.field("logical_zone_sha256", pa.string(), nullable=False),
        pa.field("canonical_unit", pa.string(), nullable=False),
        pa.field("effective_native_resolution", pa.string(), nullable=False),
        pa.field("value_decimal_mw", pa.string(), nullable=True),
    ]
)

AUTHORITIES = dict(d283.AUTHORITIES)
EXECUTION = dict(d283.EXECUTION)
MANIFEST_AUTHORITIES = dict(d283.ARTIFACT_AUTHORITIES)
_SHA = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class EntsoeLtDerivedParquetAdapterError(ValueError):
    """Raised when a synthetic Parquet package is ambiguous or unsafe."""


@dataclass(frozen=True)
class SyntheticParquetAdapterResult:
    """Value-free adapter assessment and the separate in-memory D283 result."""

    assessment: dict[str, object]
    assessment_content_id: str
    arithmetic_result: d283.SyntheticArithmeticExecutionResult


def canonical_sha256(value: object) -> str:
    return d283.canonical_sha256(value)


ARROW_SCHEMA_CONTENT_ID = canonical_sha256(COLUMN_SPEC)


def validate_derived_parquet_adapter_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    contract = _mapping(document, "D284 contract")
    _exact(
        contract,
        {
            "schema_version",
            "decision",
            "status",
            "purpose",
            "predecessor",
            "package",
            "bounds",
            "integrity",
            "execution",
            "authorities",
        },
        "D284 contract",
    )
    _equal(contract["schema_version"], CONTRACT_SCHEMA, "contract schema")
    _equal(contract["decision"], "D-20260806-284", "contract decision")
    _equal(contract["status"], CONTRACT_STATUS, "contract status")
    _clean(contract["purpose"], "contract purpose")
    _equal(
        contract["predecessor"],
        {
            "d282_contract_content_id": d282.CONTRACT_CONTENT_ID,
            "d283_contract_content_id": d283.CONTRACT_CONTENT_ID,
            "d283_algorithm_content_id": d283.ALGORITHM_CONTENT_ID,
        },
        "predecessor",
    )
    _equal(
        contract["package"],
        {
            "manifest_schema_version": MANIFEST_SCHEMA,
            "evidence_class": EVIDENCE_CLASS,
            "artifact_relative_path": ARTIFACT_RELATIVE_PATH,
            "row_grain": [
                "role",
                "feature_record_id",
                "target_start_utc",
                "target_end_utc",
            ],
            "columns_in_order": COLUMN_SPEC,
            "value_encoding": "CANONICAL_BASE10_TEXT_OR_NULL",
            "binary_floating_point_is_admissible": False,
            "compression": COMPRESSION,
        },
        "package",
    )
    _equal(
        contract["bounds"],
        {
            "maximum_manifest_bytes": MAX_MANIFEST_BYTES,
            "maximum_parquet_bytes": MAX_PARQUET_BYTES,
            "maximum_rows": MAX_ROWS,
            "exact_column_count": EXACT_COLUMNS,
            "maximum_cells": MAX_CELLS,
            "maximum_row_groups": MAX_ROW_GROUPS,
            "allowed_parquet_physical_types": list(ALLOWED_PHYSICAL_TYPES),
        },
        "bounds",
    )
    _equal(
        contract["integrity"],
        {
            "manifest_and_parquet_must_be_stable_single_link_files": True,
            "relative_path_is_fixed_and_sibling_only": True,
            "manifest_hash_size_counts_and_schema_must_match": True,
            "d282_assessment_must_remain_content_stable": True,
            "all_rows_must_be_consumed_exactly_once_by_d283": True,
            "arrow_type_coercion_is_admissible": False,
            "assessment_may_persist_operand_or_output_values": False,
            "synthetic_pass_grants_real_or_model_authority": False,
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
        "arrow_schema_content_id": ARROW_SCHEMA_CONTENT_ID,
        "real_data_authorized": False,
        "model_input_authorized": False,
        **EXECUTION,
    }


def validate_derived_parquet_adapter_contract_path(
    path: str | Path,
) -> dict[str, object]:
    contract_path = Path(path)
    if not contract_path.is_absolute():
        raise EntsoeLtDerivedParquetAdapterError("contract path must be absolute")
    payload = _read(contract_path, "D284 contract", MAX_MANIFEST_BYTES)
    if hashlib.sha256(payload).hexdigest() != CONTRACT_FILE_SHA256:
        raise EntsoeLtDerivedParquetAdapterError("contract SHA-256 differs")
    return validate_derived_parquet_adapter_contract(_strict_json(payload, "D284 contract"))


def execute_synthetic_parquet_package(
    *,
    contract_path: str | Path,
    manifest_path: str | Path,
    d282_assessment: Mapping[str, object],
    reference_time_utc: datetime,
) -> SyntheticParquetAdapterResult:
    """Read one normalized synthetic package and execute D283 unchanged."""

    validate_derived_parquet_adapter_contract_path(contract_path)
    reference = _utc_datetime(reference_time_utc, "reference time")
    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        raise EntsoeLtDerivedParquetAdapterError("manifest path must be absolute")
    manifest_payload = _read(manifest_file, "D284 package manifest", MAX_MANIFEST_BYTES)
    manifest = _validate_manifest(
        _strict_json(manifest_payload, "D284 package manifest"),
        d282_assessment=d282_assessment,
        reference=reference,
    )
    manifest_content_id = canonical_sha256(manifest)
    d282_content_id = canonical_sha256(d282_assessment)
    artifact_declaration = _mapping(manifest["artifact"], "manifest artifact")
    relative_path = _clean(artifact_declaration["relative_path"], "artifact relative path")
    _equal(relative_path, ARTIFACT_RELATIVE_PATH, "artifact relative path")
    artifact_path = manifest_file.parent / relative_path
    artifact_payload = _read(artifact_path, "D284 operand Parquet", MAX_PARQUET_BYTES)
    artifact_sha256 = hashlib.sha256(artifact_payload).hexdigest()
    _equal(artifact_sha256, artifact_declaration["sha256"], "Parquet SHA-256")
    _equal(len(artifact_payload), artifact_declaration["size_bytes"], "Parquet size")

    try:
        validate_parquet_allocation_budget(
            artifact_payload,
            label="D284 operand",
            max_rows=MAX_ROWS,
            max_columns=EXACT_COLUMNS,
            max_cells=MAX_CELLS,
            max_row_groups=MAX_ROW_GROUPS,
            allowed_physical_types=ALLOWED_PHYSICAL_TYPES,
        )
        parquet = pq.ParquetFile(BytesIO(artifact_payload))
    except (ValueError, OSError, pa.ArrowException) as exc:
        raise EntsoeLtDerivedParquetAdapterError("Parquet footer admission failed") from exc
    metadata = parquet.metadata
    _equal(int(metadata.num_columns), EXACT_COLUMNS, "Parquet column count")
    _equal(int(metadata.num_rows), artifact_declaration["row_count"], "Parquet row count")
    _equal(
        int(metadata.num_row_groups),
        artifact_declaration["row_group_count"],
        "Parquet row-group count",
    )
    _equal(
        artifact_declaration["arrow_schema_content_id"],
        ARROW_SCHEMA_CONTENT_ID,
        "Arrow schema content ID",
    )
    if not parquet.schema_arrow.equals(EXPECTED_ARROW_SCHEMA, check_metadata=True):
        raise EntsoeLtDerivedParquetAdapterError("Arrow schema differs")
    compressions = {
        str(metadata.row_group(group).column(column).compression)
        for group in range(int(metadata.num_row_groups))
        for column in range(int(metadata.num_columns))
    }
    _equal(compressions, {COMPRESSION}, "Parquet compression")
    try:
        table = parquet.read()
    except (OSError, pa.ArrowException) as exc:
        raise EntsoeLtDerivedParquetAdapterError("Parquet decoding failed") from exc
    if not table.schema.equals(EXPECTED_ARROW_SCHEMA, check_metadata=True):
        raise EntsoeLtDerivedParquetAdapterError("decoded Arrow schema differs")

    actual_rows: list[dict[str, object]] = []
    forecast_rows: list[dict[str, object]] = []
    row_grains: set[tuple[object, ...]] = set()
    for position, row in enumerate(table.to_pylist(), start=1):
        role = row["role"]
        if role not in {d283.ACTUAL_ROLE, d283.FORECAST_ROLE}:
            raise EntsoeLtDerivedParquetAdapterError(f"row {position} role differs")
        start = _timestamp_text(row["target_start_utc"], f"row {position} target start")
        end = _timestamp_text(row["target_end_utc"], f"row {position} target end")
        grain = (role, row["feature_record_id"], start, end)
        if grain in row_grains:
            raise EntsoeLtDerivedParquetAdapterError("Parquet row grain is duplicated")
        row_grains.add(grain)
        operand_row = {
            "feature_record_id": row["feature_record_id"],
            "target_start_utc": start,
            "target_end_utc": end,
            "series_id_sha256": row["series_id_sha256"],
            "logical_zone_sha256": row["logical_zone_sha256"],
            "canonical_unit": row["canonical_unit"],
            "effective_native_resolution": row["effective_native_resolution"],
            "value_decimal_mw": row["value_decimal_mw"],
        }
        (actual_rows if role == d283.ACTUAL_ROLE else forecast_rows).append(operand_row)

    actual_artifact = _operand_artifact(
        role=d283.ACTUAL_ROLE,
        rows=actual_rows,
        created_at_utc=manifest["created_at_utc"],
        d282_content_id=d282_content_id,
    )
    forecast_artifact = _operand_artifact(
        role=d283.FORECAST_ROLE,
        rows=forecast_rows,
        created_at_utc=manifest["created_at_utc"],
        d282_content_id=d282_content_id,
    )
    try:
        arithmetic_result = d283.execute_synthetic_load_forecast_error(
            contract_path=d283.CONTRACT_PATH,
            d282_assessment=d282_assessment,
            actual_operand_artifact=actual_artifact,
            forecast_operand_artifact=forecast_artifact,
            reference_time_utc=reference,
        )
    except d283.EntsoeLtDerivedArithmeticExecutionError as exc:
        raise EntsoeLtDerivedParquetAdapterError("D283 execution rejected package") from exc

    if canonical_sha256(d282_assessment) != d282_content_id:
        raise EntsoeLtDerivedParquetAdapterError("D282 assessment changed during adapter execution")
    if _read(manifest_file, "D284 package manifest", MAX_MANIFEST_BYTES) != manifest_payload:
        raise EntsoeLtDerivedParquetAdapterError("manifest changed during adapter execution")
    if _read(artifact_path, "D284 operand Parquet", MAX_PARQUET_BYTES) != artifact_payload:
        raise EntsoeLtDerivedParquetAdapterError("Parquet changed during adapter execution")

    assessment = {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": PASS_STATUS,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "reference_time_utc": _utc_text(reference),
        "manifest_content_id": manifest_content_id,
        "parquet_sha256": artifact_sha256,
        "arrow_schema_content_id": ARROW_SCHEMA_CONTENT_ID,
        "d282_assessment_content_id": d282_content_id,
        "d283_assessment_content_id": arithmetic_result.assessment_content_id,
        "d283_output_artifact_content_id": arithmetic_result.output_artifact_content_id,
        "row_count": int(metadata.num_rows),
        "actual_row_count": len(actual_rows),
        "forecast_row_count": len(forecast_rows),
        "row_group_count": int(metadata.num_row_groups),
        "all_rows_consumed_exactly_once_by_d283": True,
        "exact_arrow_schema_validated_before_decode": True,
        "canonical_decimal_text_preserved": True,
        "operand_or_output_values_persisted_in_assessment": False,
        "real_values_opened": False,
        "real_data_authorized": False,
        "model_input_authorized": False,
        "authorities": dict(AUTHORITIES),
        **EXECUTION,
    }
    return SyntheticParquetAdapterResult(
        assessment=assessment,
        assessment_content_id=canonical_sha256(assessment),
        arithmetic_result=arithmetic_result,
    )


def _validate_manifest(
    document: Mapping[str, object],
    *,
    d282_assessment: Mapping[str, object],
    reference: datetime,
) -> Mapping[str, object]:
    manifest = _mapping(document, "D284 package manifest")
    _exact(
        manifest,
        {
            "schema_version",
            "evidence_class",
            "created_at_utc",
            "d282_assessment_content_id",
            "artifact",
            "authorities",
        },
        "D284 package manifest",
    )
    _equal(manifest["schema_version"], MANIFEST_SCHEMA, "manifest schema")
    _equal(manifest["evidence_class"], EVIDENCE_CLASS, "manifest evidence class")
    created = _utc(manifest["created_at_utc"], "manifest creation time")
    if created > reference:
        raise EntsoeLtDerivedParquetAdapterError("manifest is future-dated")
    _equal(
        manifest["d282_assessment_content_id"],
        canonical_sha256(d282_assessment),
        "manifest D282 assessment content ID",
    )
    artifact = _mapping(manifest["artifact"], "manifest artifact")
    _exact(
        artifact,
        {
            "relative_path",
            "sha256",
            "size_bytes",
            "row_count",
            "row_group_count",
            "column_count",
            "arrow_schema_content_id",
            "compression",
        },
        "manifest artifact",
    )
    _equal(artifact["relative_path"], ARTIFACT_RELATIVE_PATH, "artifact relative path")
    _sha(artifact["sha256"], "artifact SHA-256")
    _bounded_int(artifact["size_bytes"], 1, MAX_PARQUET_BYTES, "artifact size")
    _bounded_int(artifact["row_count"], 1, MAX_ROWS, "artifact row count")
    _bounded_int(
        artifact["row_group_count"],
        1,
        MAX_ROW_GROUPS,
        "artifact row-group count",
    )
    _equal(artifact["column_count"], EXACT_COLUMNS, "artifact column count")
    _equal(
        artifact["arrow_schema_content_id"],
        ARROW_SCHEMA_CONTENT_ID,
        "artifact Arrow schema content ID",
    )
    _equal(artifact["compression"], COMPRESSION, "artifact compression")
    _equal(manifest["authorities"], MANIFEST_AUTHORITIES, "manifest authorities")
    return manifest


def _operand_artifact(
    *,
    role: str,
    rows: list[dict[str, object]],
    created_at_utc: object,
    d282_content_id: str,
) -> dict[str, object]:
    return {
        "schema_version": d283.OPERAND_SCHEMA,
        "evidence_class": EVIDENCE_CLASS,
        "role": role,
        "created_at_utc": created_at_utc,
        "d282_assessment_content_id": d282_content_id,
        "rows": rows,
        "authorities": dict(d283.ARTIFACT_AUTHORITIES),
    }


def _timestamp_text(value: object, label: str) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EntsoeLtDerivedParquetAdapterError(f"{label} must be timezone-aware")
    normalized = value.astimezone(timezone.utc)
    if normalized.microsecond:
        raise EntsoeLtDerivedParquetAdapterError(f"{label} must have whole-second precision")
    return _utc_text(normalized)


def _read(path: Path, label: str, max_bytes: int) -> bytes:
    try:
        return read_stable_single_link_file(path, label=label, max_bytes=max_bytes)
    except (OSError, ValueError) as exc:
        raise EntsoeLtDerivedParquetAdapterError(f"{label} read failed") from exc


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeLtDerivedParquetAdapterError(f"{label} JSON is invalid") from exc
    return _mapping(value, label)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EntsoeLtDerivedParquetAdapterError("JSON contains duplicate keys")
        result[key] = value
    return result


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeLtDerivedParquetAdapterError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EntsoeLtDerivedParquetAdapterError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise EntsoeLtDerivedParquetAdapterError(f"{label} differs")


def _clean(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise EntsoeLtDerivedParquetAdapterError(f"{label} must be clean text")
    return value


def _sha(value: object, label: str) -> str:
    text = _clean(value, label)
    if _SHA.fullmatch(text) is None:
        raise EntsoeLtDerivedParquetAdapterError(f"{label} must be lowercase SHA-256")
    return text


def _bounded_int(value: object, minimum: int, maximum: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise EntsoeLtDerivedParquetAdapterError(f"{label} is outside bounds")
    return value


def _utc(value: object, label: str) -> datetime:
    text = _clean(value, label)
    if _UTC.fullmatch(text) is None:
        raise EntsoeLtDerivedParquetAdapterError(f"{label} must be canonical UTC Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise EntsoeLtDerivedParquetAdapterError(f"{label} is invalid") from exc
    return _utc_datetime(parsed, label)


def _utc_datetime(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EntsoeLtDerivedParquetAdapterError(f"{label} must be timezone-aware")
    normalized = value.astimezone(timezone.utc)
    if normalized.microsecond:
        raise EntsoeLtDerivedParquetAdapterError(f"{label} must have whole-second precision")
    return normalized


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "ARROW_SCHEMA_CONTENT_ID",
    "ASSESSMENT_SCHEMA",
    "AUTHORITIES",
    "CONTRACT_CONTENT_ID",
    "CONTRACT_PATH",
    "EXECUTION",
    "EXPECTED_ARROW_SCHEMA",
    "MANIFEST_SCHEMA",
    "PASS_STATUS",
    "EntsoeLtDerivedParquetAdapterError",
    "SyntheticParquetAdapterResult",
    "execute_synthetic_parquet_package",
    "validate_derived_parquet_adapter_contract",
    "validate_derived_parquet_adapter_contract_path",
]

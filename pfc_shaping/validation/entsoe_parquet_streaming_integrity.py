"""Bounded, synthetic-only streaming preflight for ENTSO-E Parquet packages."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)

CONTRACT_SCHEMA = "fmv_entsoe_parquet_streaming_integrity_preflight_contract.v1"
CONTRACT_STATUS = (
    "STATIC_OFFLINE_SYNTHETIC_PARQUET_STREAMING_PREFLIGHT_NO_REAL_ADMISSION"
)
CONTRACT_FILE_SHA256 = (
    "f5c7e34d383ec476fb8893348b91ac3aab011d42d36d7321b172d081461e8cca"
)
CONTRACT_CONTENT_ID = (
    "9a79779d7ebe01b0e4dbfc65c40eaa23ead4f1f984685bc5055cc16f7a23969b"
)
D240_CONTRACT_CONTENT_ID = (
    "c4b69dd9e1ba790a5bfe3ec3c2ddc7c6555d907902f7ff429d582b820d43369e"
)
D240_CONTRACT_SHA256 = (
    "26631c82495df6f4616c2841bc73cbb5b9da13f5a265091fac5bcfcb5a21fe49"
)
D240_PROOF_CONTENT_ID = (
    "5595a20c5b997485bbaa0e3aa41f90b131190e3e89d812b1acfdfaaebc88536b"
)
D240_PROOF_MANIFEST_SHA256 = (
    "97fc381771033c31da4f79789750638c70676bf32d1d3e844edc2609b3448ec3"
)
D241_REAL_SCHEMA_PROOF_CONTENT_ID = (
    "d6c006609d881b51f08be6d60e01f68b59a40be8bdf2898ef0a98491f5771544"
)
D241_REAL_SCHEMA_PROOF_MANIFEST_SHA256 = (
    "1835f93a517e9c6769079984a376fb9879b22d7c8ce2922aa42b0dd646627ada"
)
D241_REAL_SCHEMA_CAPTURE_CONTENT_ID = (
    "d69fdab73ba1d9c55f70f77925f2253d583564d06d922f2b41e035763bca176f"
)
D241_REAL_SCHEMA_STATUS = (
    "FAIL_REAL_CONTROL_PLANE_SCHEMA_INCOMPATIBLE_NO_MODEL_AUTHORITY"
)

MANIFEST_SCHEMA = "fmv_entsoe_parquet_package_manifest.v1"
RECEIPT_SCHEMA = "fmv_entsoe_parquet_streaming_integrity_receipt.v1"
ROLE_ORDER = ("series_dimension", "latest_values", "vintage_values")
ROLE_FILE_NAMES = {
    "series_dimension": "series_dimension.parquet",
    "latest_values": "latest_values.parquet",
    "vintage_values": "vintage_values.parquet",
}
ROLE_COLUMNS = {
    "series_dimension": (
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
    "latest_values": (
        "series_id",
        "target_ts_utc",
        "value",
        "is_historical",
        "quality_flag",
        "revision_number",
        "meta_load_timestamp_utc",
    ),
    "vintage_values": (
        "series_id",
        "target_ts_utc",
        "as_of_utc",
        "value",
        "quality_flag",
        "revision_number",
        "meta_load_timestamp_utc",
    ),
}
NORMALIZED_TYPES = {
    "series_dimension": {
        "series_id": "string",
        "group_name": "string",
        "field_name": "string",
        "from_zone": "string",
        "to_zone": "string",
        "unit": "string",
        "native_resolution": "string",
        "source_endpoint": "string",
        "document_id": "string",
    },
    "latest_values": {
        "series_id": "string",
        "target_ts_utc": "timestamp_utc",
        "value": "float64_finite",
        "is_historical": "boolean",
        "quality_flag": "string",
        "revision_number": "int64_non_negative",
        "meta_load_timestamp_utc": "timestamp_utc",
    },
    "vintage_values": {
        "series_id": "string",
        "target_ts_utc": "timestamp_utc",
        "as_of_utc": "timestamp_utc",
        "value": "float64_finite",
        "quality_flag": "string",
        "revision_number": "int64_non_negative",
        "meta_load_timestamp_utc": "timestamp_utc",
    },
}
ARROW_TYPES = {
    "series_dimension": {column: "string" for column in ROLE_COLUMNS["series_dimension"]},
    "latest_values": {
        "series_id": "string",
        "target_ts_utc": "timestamp[ns, tz=UTC]",
        "value": "double",
        "is_historical": "bool",
        "quality_flag": "string",
        "revision_number": "int64",
        "meta_load_timestamp_utc": "timestamp[ns, tz=UTC]",
    },
    "vintage_values": {
        "series_id": "string",
        "target_ts_utc": "timestamp[ns, tz=UTC]",
        "as_of_utc": "timestamp[ns, tz=UTC]",
        "value": "double",
        "quality_flag": "string",
        "revision_number": "int64",
        "meta_load_timestamp_utc": "timestamp[ns, tz=UTC]",
    },
}

MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_class",
        "snapshot_id",
        "capture_batch_id",
        "proposal_content_id",
        "reservation_id",
        "snapshot_reference_time_utc",
        "created_at_utc",
        "artifacts",
        "authorities",
    }
)
ARTIFACT_FIELDS = frozenset(
    {
        "role",
        "format",
        "relative_path",
        "sha256",
        "size_bytes",
        "row_count",
        "row_group_count",
        "columns",
        "column_types",
        "normalized_schema_sha256",
        "arrow_schema_sha256",
        "compression",
        "parquet_format_version",
        "created_by",
    }
)
RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_class",
        "snapshot_id",
        "package_manifest_content_id",
        "verified_at_utc",
        "pyarrow_version",
        "artifacts",
        "authorities",
    }
)
RECEIPT_ARTIFACT_FIELDS = frozenset(
    {
        "role",
        "sha256",
        "size_bytes",
        "row_count",
        "row_group_count",
        "record_batch_count",
        "total_uncompressed_bytes",
        "normalized_schema_sha256",
        "arrow_schema_sha256",
        "compression",
        "parquet_format_version",
        "created_by",
    }
)

HASH_CHUNK_BYTES = 1_048_576
RECORD_BATCH_ROWS = 4_096
MAX_FOOTER_BYTES = 16_777_216
MAX_TOTAL_FILE_BYTES = 4_294_967_296
MAX_TOTAL_UNCOMPRESSED_BYTES = 17_179_869_184
MAX_ROW_GROUPS = 8_192
MAX_ROWS_PER_ROW_GROUP = 1_000_000
MAX_UNCOMPRESSED_BYTES_PER_ROW_GROUP = 268_435_456
MAX_UNCOMPRESSED_TO_FILE_RATIO = 1_000
MAX_RECORD_BATCH_BYTES = 67_108_864
MAX_STRING_VALUE_BYTES = 1_048_576
THRIFT_STRING_SIZE_LIMIT = 1_048_576
THRIFT_CONTAINER_SIZE_LIMIT = 1_000_000
ROLE_MAX_FILE_BYTES = {
    "series_dimension": 67_108_864,
    "latest_values": 1_073_741_824,
    "vintage_values": 3_221_225_472,
}
ROLE_MAX_ROWS = {
    "series_dimension": 100_000,
    "latest_values": 100_000_000,
    "vintage_values": 250_000_000,
}
ALLOWED_COMPRESSION = frozenset({"ZSTD", "SNAPPY"})
ALLOWED_PARQUET_FORMAT_VERSIONS = frozenset({"1.0", "2.4", "2.6"})

MANIFEST_AUTHORITIES = {
    "synthetic_parquet_stream_integrity_verified": False,
    "real_artifact_receipt_admitted": False,
    "real_artifact_hashes_verified": False,
    "real_entsoe_values_opened": False,
    "source_authenticity_verified": False,
    "external_time_verified": False,
    "same_snapshot_point_in_time_availability_proven": False,
    "incremental_quality_profile_verified": False,
    "seasonal_cross_market_diagnostic_authorized": False,
    "training_authorized": False,
    "selection_authorized": False,
    "model_input_authorized": False,
    "candidate_assembly_authorized": False,
    "promotion_authorized": False,
    "production_authorized": False,
}
RECEIPT_AUTHORITIES = {
    **MANIFEST_AUTHORITIES,
    "synthetic_parquet_stream_integrity_verified": True,
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_PARQUET_ROOT = (
    REPOSITORY_ROOT
    / "build"
    / "governed-source-intake"
    / "synthetic-parquet-fixtures"
)
D240_CONTRACT_PATH = (
    REPOSITORY_ROOT
    / ".planning"
    / "phases"
    / "14-lt-audit-remediation"
    / "ENTSOE-ARTIFACT-PACKAGE-INTEGRITY-PREFLIGHT-CONTRACT-V1.json"
)
D240_PROOF_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "build"
    / "databricks-eex-daily"
    / "2026-08-06"
    / "entsoe-artifact-package-integrity-proofs"
    / D240_PROOF_CONTENT_ID
    / "manifest.json"
)
D241_REAL_SCHEMA_PROOF_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "build"
    / "databricks-control-plane"
    / "2026-08-06"
    / "entsoe-unity-catalog-schema-captures"
    / D241_REAL_SCHEMA_PROOF_CONTENT_ID
    / "manifest.json"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class EntsoeParquetStreamingIntegrityError(ValueError):
    """Raised when bounded Parquet integrity or authority is ambiguous."""


@dataclass(frozen=True)
class EntsoeParquetStreamingIntegrityResult:
    """Value-free receipt for one synthetic Parquet package."""

    receipt: Mapping[str, object]
    receipt_content_id: str
    package_manifest_content_id: str
    total_rows: int
    total_record_batches: int
    total_uncompressed_bytes: int


def validate_parquet_streaming_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact D244 static contract."""

    contract = _mapping(document, "D244 contract")
    _equal(contract.get("schema_version"), CONTRACT_SCHEMA, "contract schema")
    _equal(contract.get("status"), CONTRACT_STATUS, "contract status")
    _equal(contract.get("decision"), "D-20260806-244", "contract decision")
    _equal(
        contract.get("input_bindings"),
        {
            "d240_contract_content_id": D240_CONTRACT_CONTENT_ID,
            "d240_contract_sha256": D240_CONTRACT_SHA256,
            "d240_proof_content_id": D240_PROOF_CONTENT_ID,
            "d240_proof_manifest_sha256": D240_PROOF_MANIFEST_SHA256,
            "d241_real_schema_proof_content_id": D241_REAL_SCHEMA_PROOF_CONTENT_ID,
            "d241_real_schema_proof_manifest_sha256": (
                D241_REAL_SCHEMA_PROOF_MANIFEST_SHA256
            ),
            "d241_real_schema_capture_content_id": (
                D241_REAL_SCHEMA_CAPTURE_CONTENT_ID
            ),
            "d241_real_schema_status": D241_REAL_SCHEMA_STATUS,
        },
        "contract input bindings",
    )
    package = _mapping(contract.get("package_manifest"), "package policy")
    _equal(package.get("schema_version"), MANIFEST_SCHEMA, "manifest policy schema")
    _equal(package.get("evidence_class"), "SYNTHETIC_TEST_ONLY", "evidence policy")
    _equal(
        set(package.get("exact_top_level_fields", [])),
        set(MANIFEST_FIELDS),
        "manifest fields",
    )
    artifact = _mapping(contract.get("artifact_declaration"), "artifact policy")
    _equal(set(artifact.get("exact_fields", [])), set(ARTIFACT_FIELDS), "artifact fields")
    _equal(artifact.get("exact_role_order"), list(ROLE_ORDER), "artifact roles")
    _equal(artifact.get("format"), "PARQUET", "artifact format")
    _equal(artifact.get("file_names"), ROLE_FILE_NAMES, "artifact file names")
    _equal(artifact.get("parquet_magic"), "PAR1", "Parquet magic")
    _equal(
        set(artifact.get("allowed_compression", [])),
        set(ALLOWED_COMPRESSION),
        "compression policy",
    )
    _equal(
        set(artifact.get("allowed_parquet_format_versions", [])),
        set(ALLOWED_PARQUET_FORMAT_VERSIONS),
        "Parquet format-version policy",
    )
    _equal(
        contract.get("normalized_logical_types"),
        NORMALIZED_TYPES,
        "normalized logical types",
    )
    _equal(contract.get("arrow_types"), ARROW_TYPES, "Arrow types")
    bounded = _mapping(contract.get("bounded_streaming"), "bounded streaming")
    expected_caps = {
        "hash_chunk_bytes": HASH_CHUNK_BYTES,
        "record_batch_rows": RECORD_BATCH_ROWS,
        "max_footer_bytes": MAX_FOOTER_BYTES,
        "max_total_file_bytes": MAX_TOTAL_FILE_BYTES,
        "max_total_uncompressed_bytes": MAX_TOTAL_UNCOMPRESSED_BYTES,
        "max_row_groups_per_artifact": MAX_ROW_GROUPS,
        "max_rows_per_row_group": MAX_ROWS_PER_ROW_GROUP,
        "max_uncompressed_bytes_per_row_group": MAX_UNCOMPRESSED_BYTES_PER_ROW_GROUP,
        "max_uncompressed_to_file_ratio": MAX_UNCOMPRESSED_TO_FILE_RATIO,
        "max_record_batch_bytes": MAX_RECORD_BATCH_BYTES,
        "max_string_value_bytes": MAX_STRING_VALUE_BYTES,
        "thrift_string_size_limit": THRIFT_STRING_SIZE_LIMIT,
        "thrift_container_size_limit": THRIFT_CONTAINER_SIZE_LIMIT,
        "per_role_max_file_bytes": ROLE_MAX_FILE_BYTES,
        "per_role_max_rows": ROLE_MAX_ROWS,
    }
    for field, expected in expected_caps.items():
        _equal(bounded.get(field), expected, f"bounded streaming {field}")
    for field, expected in {
        "empty_artifact_authorized": False,
        "full_file_bytes_materialized_in_python_authorized": False,
        "pandas_import_or_dataframe_materialization_authorized": False,
        "hash_before_and_after_scan_must_match": True,
        "same_open_descriptor_used_for_hash_and_scan": True,
        "page_checksum_verification_enabled": True,
        "every_record_batch_must_be_visited": True,
        "null_nonfinite_and_negative_revision_values_rejected_during_scan": True,
    }.items():
        _equal(bounded.get(field), expected, f"bounded streaming {field}")
    receipt = _mapping(contract.get("derived_receipt"), "receipt policy")
    _equal(receipt.get("schema_version"), RECEIPT_SCHEMA, "receipt schema")
    _equal(set(receipt.get("exact_top_level_fields", [])), set(RECEIPT_FIELDS), "receipt fields")
    _equal(
        set(receipt.get("artifact_receipt_fields", [])),
        set(RECEIPT_ARTIFACT_FIELDS),
        "receipt artifact fields",
    )
    integration = _mapping(contract.get("integration_policy"), "integration policy")
    _equal(integration.get("d239_full_quality_profile_run_in_this_batch"), False, "D239 authority")
    _equal(integration.get("future_incremental_quality_profiler_required"), True, "incremental profiler requirement")
    _equal(
        integration.get(
            "normalized_parquet_is_target_interface_not_current_unity_catalog_raw_schema"
        ),
        True,
        "normalized target-interface boundary",
    )
    _equal(
        integration.get("current_raw_unity_catalog_export_admission_authorized"),
        False,
        "raw Unity Catalog export authority",
    )
    _equal(
        integration.get("d241_missing_mapping_fields_must_not_be_fabricated"),
        True,
        "D241 mapping blocker",
    )
    execution = _mapping(contract.get("execution_policy"), "execution policy")
    for field in (
        "current_databricks_connection_budget",
        "current_databricks_statement_budget",
        "current_warehouse_start_budget",
        "current_network_call_budget",
        "current_h_drive_access_budget",
        "current_remote_write_budget",
    ):
        _equal(execution.get(field), 0, f"execution {field}")
    _equal(execution.get("connector_code_allowed"), False, "connector authority")
    _equal(execution.get("real_artifact_opening_authorized"), False, "real artifact authority")
    _equal(contract.get("authorities"), RECEIPT_AUTHORITIES, "contract authorities")
    content_id = canonical_sha256(contract)
    _equal(content_id, CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": content_id,
        "synthetic_streaming_implemented": True,
        "full_file_materialization_authorized": False,
        "real_artifact_admission_implemented": False,
        "databricks_connection_budget": 0,
        "warehouse_start_budget": 0,
        "production_authorized": False,
    }


def verify_synthetic_entsoe_parquet_package(
    *,
    contract_path: str | Path,
    package_manifest_path: str | Path,
    reference_time_utc: datetime,
    d240_contract_path: str | Path = D240_CONTRACT_PATH,
    d240_proof_manifest_path: str | Path = D240_PROOF_MANIFEST_PATH,
    d241_real_schema_proof_manifest_path: str | Path = (
        D241_REAL_SCHEMA_PROOF_MANIFEST_PATH
    ),
) -> EntsoeParquetStreamingIntegrityResult:
    """Stream-verify all declared synthetic Parquet artifacts fail-closed."""

    _require_canonical_workspace()
    contract_raw = _read_small(contract_path, "D244 contract", 100_000)
    _equal(hashlib.sha256(contract_raw).hexdigest(), CONTRACT_FILE_SHA256, "D244 contract hash")
    contract = _strict_json_object(contract_raw, "D244 contract")
    validate_parquet_streaming_contract(contract)
    _verify_d240_bindings(d240_contract_path, d240_proof_manifest_path)
    _verify_d241_real_schema_binding(d241_real_schema_proof_manifest_path)

    manifest_path = assert_absolute_path_has_no_links(package_manifest_path)
    manifest_raw = _read_small(manifest_path, "Parquet package manifest", 2_000_000)
    manifest = _strict_json_object(manifest_raw, "Parquet package manifest")
    declarations = _validate_manifest(manifest, reference_time_utc)
    package_dir = _validate_package_directory(manifest_path, str(manifest["snapshot_id"]))
    expected_inventory = {"manifest.json", *ROLE_FILE_NAMES.values()}
    _equal({entry.name for entry in package_dir.iterdir()}, expected_inventory, "package inventory")

    artifacts: list[dict[str, object]] = []
    total_rows = 0
    total_batches = 0
    total_file_bytes = 0
    total_uncompressed = 0
    for declaration in declarations:
        role = str(declaration["role"])
        artifact_path = assert_absolute_path_has_no_links(
            package_dir / str(declaration["relative_path"])
        )
        if artifact_path.parent != package_dir:
            raise EntsoeParquetStreamingIntegrityError(f"artifact escapes package: {role}")
        scanned = _scan_parquet_artifact(
            artifact_path,
            declaration,
            role,
            uncompressed_budget=MAX_TOTAL_UNCOMPRESSED_BYTES - total_uncompressed,
        )
        artifacts.append(scanned)
        total_rows += int(scanned["row_count"])
        total_batches += int(scanned["record_batch_count"])
        total_file_bytes += int(scanned["size_bytes"])
        total_uncompressed += int(scanned["total_uncompressed_bytes"])
    if total_file_bytes > MAX_TOTAL_FILE_BYTES:
        raise EntsoeParquetStreamingIntegrityError("package file-byte cap exceeded")
    if total_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise EntsoeParquetStreamingIntegrityError("package uncompressed-byte cap exceeded")

    reference = _utc_datetime(reference_time_utc, "reference_time_utc")
    package_content_id = canonical_sha256(manifest)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "snapshot_id": manifest["snapshot_id"],
        "package_manifest_content_id": package_content_id,
        "verified_at_utc": _format_utc(reference),
        "pyarrow_version": pa.__version__,
        "artifacts": artifacts,
        "authorities": dict(RECEIPT_AUTHORITIES),
    }
    validate_parquet_streaming_receipt(
        receipt,
        manifest,
        reference_time_utc=reference,
    )
    return EntsoeParquetStreamingIntegrityResult(
        receipt=receipt,
        receipt_content_id=canonical_sha256(receipt),
        package_manifest_content_id=package_content_id,
        total_rows=total_rows,
        total_record_batches=total_batches,
        total_uncompressed_bytes=total_uncompressed,
    )


def validate_parquet_streaming_receipt(
    receipt_document: Mapping[str, object],
    manifest_document: Mapping[str, object],
    *,
    reference_time_utc: datetime,
) -> dict[str, object]:
    """Replay a verifier-derived synthetic-only streaming receipt."""

    receipt = _mapping(receipt_document, "streaming receipt")
    _exact_fields(receipt, RECEIPT_FIELDS, "streaming receipt")
    _equal(receipt.get("schema_version"), RECEIPT_SCHEMA, "receipt schema")
    _equal(receipt.get("evidence_class"), "SYNTHETIC_TEST_ONLY", "receipt evidence")
    manifest = _mapping(manifest_document, "package manifest")
    _equal(receipt.get("snapshot_id"), manifest.get("snapshot_id"), "receipt snapshot")
    _equal(receipt.get("package_manifest_content_id"), canonical_sha256(manifest), "receipt manifest content ID")
    _equal(receipt.get("verified_at_utc"), _format_utc(_utc_datetime(reference_time_utc, "reference_time_utc")), "receipt time")
    _equal(receipt.get("pyarrow_version"), pa.__version__, "receipt PyArrow version")
    artifacts = receipt.get("artifacts")
    declarations = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not isinstance(declarations, list):
        raise EntsoeParquetStreamingIntegrityError("receipt artifacts must be lists")
    if len(artifacts) != len(ROLE_ORDER) or len(declarations) != len(ROLE_ORDER):
        raise EntsoeParquetStreamingIntegrityError("receipt artifact count differs")
    for role, raw_receipt, raw_declaration in zip(ROLE_ORDER, artifacts, declarations, strict=True):
        item = _mapping(raw_receipt, f"receipt artifact {role}")
        declaration = _mapping(raw_declaration, f"declaration {role}")
        _exact_fields(item, RECEIPT_ARTIFACT_FIELDS, f"receipt artifact {role}")
        for field in (
            "role",
            "sha256",
            "size_bytes",
            "row_count",
            "row_group_count",
            "normalized_schema_sha256",
            "arrow_schema_sha256",
            "compression",
            "parquet_format_version",
            "created_by",
        ):
            _equal(item.get(field), declaration.get(field), f"receipt {role}.{field}")
        _positive_int(item.get("record_batch_count"), f"receipt batches {role}")
        _positive_int(item.get("total_uncompressed_bytes"), f"receipt uncompressed {role}")
    _equal(receipt.get("authorities"), RECEIPT_AUTHORITIES, "receipt authorities")
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "PASS_SYNTHETIC_PARQUET_STREAM_INTEGRITY_ONLY",
        "receipt_content_id": canonical_sha256(receipt),
        "synthetic_parquet_stream_integrity_verified": True,
        "incremental_quality_profile_verified": False,
        "real_artifact_receipt_admitted": False,
        "model_input_authorized": False,
        "production_authorized": False,
    }


def derive_normalized_schema_sha256(role: str) -> str:
    if role not in ROLE_ORDER:
        raise EntsoeParquetStreamingIntegrityError("normalized schema role is invalid")
    return canonical_sha256(
        {
            "role": role,
            "columns": list(ROLE_COLUMNS[role]),
            "column_types": NORMALIZED_TYPES[role],
        }
    )


def expected_arrow_schema(role: str) -> pa.Schema:
    if role not in ROLE_ORDER:
        raise EntsoeParquetStreamingIntegrityError("Arrow schema role is invalid")
    type_map = {
        "string": pa.string(),
        "timestamp[ns, tz=UTC]": pa.timestamp("ns", tz="UTC"),
        "double": pa.float64(),
        "bool": pa.bool_(),
        "int64": pa.int64(),
    }
    return pa.schema(
        [
            pa.field(column, type_map[ARROW_TYPES[role][column]], nullable=True)
            for column in ROLE_COLUMNS[role]
        ],
        metadata=None,
    )


def derive_arrow_schema_sha256(role: str) -> str:
    schema = expected_arrow_schema(role)
    return canonical_sha256(
        {
            "role": role,
            "fields": [
                {
                    "name": field.name,
                    "arrow_type": str(field.type),
                    "nullable": field.nullable,
                    "metadata": None,
                }
                for field in schema
            ],
            "schema_metadata": None,
        }
    )


def derive_snapshot_id(manifest_without_snapshot: Mapping[str, object]) -> str:
    value = _mapping(manifest_without_snapshot, "snapshot derivation input")
    required = {
        "capture_batch_id",
        "proposal_content_id",
        "reservation_id",
        "snapshot_reference_time_utc",
        "artifacts",
    }
    if set(value) != required:
        raise EntsoeParquetStreamingIntegrityError("snapshot derivation fields differ")
    return canonical_sha256(value)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scan_parquet_artifact(
    path: Path,
    declaration: Mapping[str, object],
    role: str,
    *,
    uncompressed_budget: int,
) -> dict[str, object]:
    lexical = assert_absolute_path_has_no_links(path)
    try:
        before = lexical.stat(follow_symlinks=False)
    except OSError as exc:
        raise EntsoeParquetStreamingIntegrityError(f"artifact unavailable: {role}") from exc
    if not _is_single_link_regular(before):
        raise EntsoeParquetStreamingIntegrityError(f"artifact must be a single-link regular file: {role}")
    declared_size = int(declaration["size_bytes"])
    _equal(int(before.st_size), declared_size, f"artifact size {role}")
    if declared_size > ROLE_MAX_FILE_BYTES[role]:
        raise EntsoeParquetStreamingIntegrityError(f"artifact file-byte cap exceeded: {role}")
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor: int | None = None
    try:
        descriptor = os.open(lexical, flags)
        opened = os.fstat(descriptor)
        if not _is_single_link_regular(opened) or _stable_identity(opened) != _stable_identity(before):
            raise EntsoeParquetStreamingIntegrityError(f"artifact identity changed before scan: {role}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            digest_before, measured_size = _hash_stream(stream)
            _equal(measured_size, declared_size, f"streamed size {role}")
            _equal(digest_before, declaration["sha256"], f"artifact hash {role}")
            footer_bytes = _validate_parquet_footer(stream, declared_size, role)
            stream.seek(0)
            try:
                parquet = pq.ParquetFile(
                    stream,
                    thrift_string_size_limit=THRIFT_STRING_SIZE_LIMIT,
                    thrift_container_size_limit=THRIFT_CONTAINER_SIZE_LIMIT,
                    page_checksum_verification=True,
                )
                receipt = _inspect_and_scan_parquet(
                    parquet,
                    declaration,
                    role,
                    footer_bytes=footer_bytes,
                    uncompressed_budget=uncompressed_budget,
                )
            except (pa.ArrowException, OSError, ValueError) as exc:
                if isinstance(exc, EntsoeParquetStreamingIntegrityError):
                    raise
                raise EntsoeParquetStreamingIntegrityError(f"Parquet scan failed: {role}") from exc
            stream.seek(0)
            digest_after, size_after = _hash_stream(stream)
            _equal(size_after, measured_size, f"post-scan size {role}")
            _equal(digest_after, digest_before, f"post-scan hash {role}")
            after_descriptor = os.fstat(stream.fileno())
            if _stable_identity(after_descriptor) != _stable_identity(opened):
                raise EntsoeParquetStreamingIntegrityError(f"artifact descriptor changed during scan: {role}")
    except OSError as exc:
        raise EntsoeParquetStreamingIntegrityError(f"artifact cannot be opened safely: {role}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        final = lexical.stat(follow_symlinks=False)
    except OSError as exc:
        raise EntsoeParquetStreamingIntegrityError(f"artifact disappeared after scan: {role}") from exc
    if not _is_single_link_regular(final) or _stable_identity(final) != _stable_identity(before):
        raise EntsoeParquetStreamingIntegrityError(f"artifact path changed during scan: {role}")
    return receipt


def _inspect_and_scan_parquet(
    parquet: pq.ParquetFile,
    declaration: Mapping[str, object],
    role: str,
    *,
    footer_bytes: int,
    uncompressed_budget: int,
) -> dict[str, object]:
    metadata = parquet.metadata
    if metadata is None:
        raise EntsoeParquetStreamingIntegrityError(f"Parquet metadata missing: {role}")
    _equal(metadata.num_rows, declaration["row_count"], f"Parquet row count {role}")
    _equal(metadata.num_row_groups, declaration["row_group_count"], f"row-group count {role}")
    _equal(metadata.num_columns, len(ROLE_COLUMNS[role]), f"column count {role}")
    if metadata.num_row_groups <= 0 or metadata.num_row_groups > MAX_ROW_GROUPS:
        raise EntsoeParquetStreamingIntegrityError(f"row-group cap violated: {role}")
    if int(metadata.serialized_size) > MAX_FOOTER_BYTES:
        raise EntsoeParquetStreamingIntegrityError(f"footer metadata cap exceeded: {role}")
    _equal(int(metadata.serialized_size), footer_bytes, f"footer metadata size {role}")
    format_version = str(metadata.format_version)
    if format_version not in ALLOWED_PARQUET_FORMAT_VERSIONS:
        raise EntsoeParquetStreamingIntegrityError(f"Parquet format version rejected: {role}")
    _equal(
        format_version,
        declaration["parquet_format_version"],
        f"Parquet format version {role}",
    )
    created_by = _nonempty_string(metadata.created_by, f"Parquet created_by {role}")
    _equal(created_by, declaration["created_by"], f"Parquet created_by {role}")
    schema = parquet.schema_arrow
    if schema.metadata is not None or any(field.metadata is not None for field in schema):
        raise EntsoeParquetStreamingIntegrityError(f"Arrow metadata is forbidden: {role}")
    _equal(schema, expected_arrow_schema(role), f"Arrow schema {role}")
    _equal(derive_arrow_schema_sha256(role), declaration["arrow_schema_sha256"], f"Arrow schema hash {role}")

    codecs: set[str] = set()
    total_uncompressed = 0
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        if row_group.num_rows <= 0 or row_group.num_rows > MAX_ROWS_PER_ROW_GROUP:
            raise EntsoeParquetStreamingIntegrityError(f"row-group row cap violated: {role}")
        uncompressed = int(row_group.total_byte_size)
        if uncompressed <= 0 or uncompressed > MAX_UNCOMPRESSED_BYTES_PER_ROW_GROUP:
            raise EntsoeParquetStreamingIntegrityError(f"row-group byte cap violated: {role}")
        total_uncompressed += uncompressed
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            codec = str(column.compression).upper()
            if codec not in ALLOWED_COMPRESSION:
                raise EntsoeParquetStreamingIntegrityError(f"compression codec rejected: {role}")
            if column.file_path not in {None, ""}:
                raise EntsoeParquetStreamingIntegrityError(f"external column file rejected: {role}")
            if int(column.total_compressed_size) <= 0 or int(column.total_uncompressed_size) <= 0:
                raise EntsoeParquetStreamingIntegrityError(f"column chunk size invalid: {role}")
            codecs.add(codec)
    if total_uncompressed > uncompressed_budget:
        raise EntsoeParquetStreamingIntegrityError(
            f"package uncompressed-byte budget exceeded before scan: {role}"
        )
    declared_size = int(declaration["size_bytes"])
    if total_uncompressed > declared_size * MAX_UNCOMPRESSED_TO_FILE_RATIO:
        raise EntsoeParquetStreamingIntegrityError(
            f"Parquet decompression ratio cap exceeded: {role}"
        )
    declared_compression = declaration["compression"]
    _equal(sorted(codecs), declared_compression, f"compression declaration {role}")

    scanned_rows = 0
    batch_count = 0
    for batch in parquet.iter_batches(batch_size=RECORD_BATCH_ROWS, use_threads=False):
        batch_count += 1
        if batch.num_rows <= 0 or batch.num_rows > RECORD_BATCH_ROWS:
            raise EntsoeParquetStreamingIntegrityError(f"record-batch cap violated: {role}")
        if batch.nbytes > MAX_RECORD_BATCH_BYTES:
            raise EntsoeParquetStreamingIntegrityError(
                f"record-batch byte cap violated: {role}"
            )
        _equal(batch.schema, expected_arrow_schema(role), f"record-batch schema {role}")
        _validate_batch_values(batch, role)
        scanned_rows += batch.num_rows
    _equal(scanned_rows, declaration["row_count"], f"visited row count {role}")
    if batch_count <= 0:
        raise EntsoeParquetStreamingIntegrityError(f"no record batch visited: {role}")
    return {
        "role": role,
        "sha256": declaration["sha256"],
        "size_bytes": declaration["size_bytes"],
        "row_count": scanned_rows,
        "row_group_count": metadata.num_row_groups,
        "record_batch_count": batch_count,
        "total_uncompressed_bytes": total_uncompressed,
        "normalized_schema_sha256": declaration["normalized_schema_sha256"],
        "arrow_schema_sha256": declaration["arrow_schema_sha256"],
        "compression": sorted(codecs),
        "parquet_format_version": format_version,
        "created_by": created_by,
    }


def _validate_batch_values(batch: pa.RecordBatch, role: str) -> None:
    for index, column_name in enumerate(ROLE_COLUMNS[role]):
        array = batch.column(index)
        if array.null_count:
            raise EntsoeParquetStreamingIntegrityError(f"null value rejected: {role}.{column_name}")
        if pa.types.is_string(array.type):
            maximum = pc.max(pc.binary_length(array.cast(pa.binary()))).as_py()
            if maximum is None or maximum > MAX_STRING_VALUE_BYTES:
                raise EntsoeParquetStreamingIntegrityError(
                    f"string value byte cap violated: {role}.{column_name}"
                )
    if "value" in ROLE_COLUMNS[role]:
        values = batch.column(batch.schema.get_field_index("value"))
        finite = pc.all(pc.is_finite(values)).as_py()
        if finite is not True:
            raise EntsoeParquetStreamingIntegrityError(f"non-finite value rejected: {role}")
    if "revision_number" in ROLE_COLUMNS[role]:
        revisions = batch.column(batch.schema.get_field_index("revision_number"))
        minimum = pc.min(revisions).as_py()
        if minimum is None or minimum < 0:
            raise EntsoeParquetStreamingIntegrityError(f"negative revision rejected: {role}")


def _validate_manifest(
    manifest: Mapping[str, object],
    reference_time_utc: datetime,
) -> list[Mapping[str, object]]:
    _exact_fields(manifest, MANIFEST_FIELDS, "package manifest")
    _equal(manifest.get("schema_version"), MANIFEST_SCHEMA, "manifest schema")
    _equal(manifest.get("evidence_class"), "SYNTHETIC_TEST_ONLY", "manifest evidence")
    snapshot_id = _sha256(manifest.get("snapshot_id"), "snapshot ID")
    _uuid(manifest.get("capture_batch_id"), "capture batch ID")
    _sha256(manifest.get("proposal_content_id"), "proposal content ID")
    _uuid(manifest.get("reservation_id"), "reservation ID")
    snapshot = _parse_utc(manifest.get("snapshot_reference_time_utc"), "snapshot time")
    created = _parse_utc(manifest.get("created_at_utc"), "created time")
    reference = _utc_datetime(reference_time_utc, "reference_time_utc")
    if snapshot > created or created > reference:
        raise EntsoeParquetStreamingIntegrityError("manifest timestamp order differs")
    _equal(manifest.get("authorities"), MANIFEST_AUTHORITIES, "manifest authorities")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(ROLE_ORDER):
        raise EntsoeParquetStreamingIntegrityError("manifest artifacts differ")
    declarations: list[Mapping[str, object]] = []
    paths: set[str] = set()
    total_bytes = 0
    for role, raw in zip(ROLE_ORDER, artifacts, strict=True):
        declaration = _mapping(raw, f"artifact declaration {role}")
        _exact_fields(declaration, ARTIFACT_FIELDS, f"artifact declaration {role}")
        _equal(declaration.get("role"), role, f"artifact role {role}")
        _equal(declaration.get("format"), "PARQUET", f"artifact format {role}")
        relative = _relative_path(declaration.get("relative_path"), role)
        _equal(relative, ROLE_FILE_NAMES[role], f"artifact file name {role}")
        if relative in paths:
            raise EntsoeParquetStreamingIntegrityError("artifact paths are duplicated")
        paths.add(relative)
        _sha256(declaration.get("sha256"), f"artifact SHA-256 {role}")
        size = _positive_int(declaration.get("size_bytes"), f"artifact size {role}")
        rows = _positive_int(declaration.get("row_count"), f"artifact rows {role}")
        row_groups = _positive_int(declaration.get("row_group_count"), f"artifact row groups {role}")
        if size > ROLE_MAX_FILE_BYTES[role] or rows > ROLE_MAX_ROWS[role] or row_groups > MAX_ROW_GROUPS:
            raise EntsoeParquetStreamingIntegrityError(f"artifact declaration cap exceeded: {role}")
        total_bytes += size
        _equal(declaration.get("columns"), list(ROLE_COLUMNS[role]), f"columns {role}")
        _equal(declaration.get("column_types"), NORMALIZED_TYPES[role], f"logical types {role}")
        _equal(declaration.get("normalized_schema_sha256"), derive_normalized_schema_sha256(role), f"normalized schema hash {role}")
        _equal(declaration.get("arrow_schema_sha256"), derive_arrow_schema_sha256(role), f"Arrow schema hash {role}")
        compression = declaration.get("compression")
        if not isinstance(compression, list) or not compression or compression != sorted(set(compression)) or not set(compression) <= ALLOWED_COMPRESSION:
            raise EntsoeParquetStreamingIntegrityError(f"compression declaration invalid: {role}")
        version = declaration.get("parquet_format_version")
        if version not in ALLOWED_PARQUET_FORMAT_VERSIONS:
            raise EntsoeParquetStreamingIntegrityError(
                f"Parquet format declaration invalid: {role}"
            )
        created_by = _nonempty_string(
            declaration.get("created_by"),
            f"Parquet created_by declaration {role}",
        )
        if len(created_by.encode("utf-8")) > 1_024:
            raise EntsoeParquetStreamingIntegrityError(
                f"Parquet created_by declaration too large: {role}"
            )
        declarations.append(declaration)
    if total_bytes > MAX_TOTAL_FILE_BYTES:
        raise EntsoeParquetStreamingIntegrityError("declared package file-byte cap exceeded")
    derivation = {
        "capture_batch_id": manifest["capture_batch_id"],
        "proposal_content_id": manifest["proposal_content_id"],
        "reservation_id": manifest["reservation_id"],
        "snapshot_reference_time_utc": manifest["snapshot_reference_time_utc"],
        "artifacts": artifacts,
    }
    _equal(snapshot_id, derive_snapshot_id(derivation), "snapshot ID derivation")
    return declarations


def _verify_d240_bindings(
    contract_path: str | Path,
    proof_manifest_path: str | Path,
) -> None:
    contract_raw = _read_small(contract_path, "D240 contract", 100_000)
    _equal(hashlib.sha256(contract_raw).hexdigest(), D240_CONTRACT_SHA256, "D240 contract hash")
    _equal(canonical_sha256(_strict_json_object(contract_raw, "D240 contract")), D240_CONTRACT_CONTENT_ID, "D240 contract content ID")
    proof_raw = _read_small(proof_manifest_path, "D240 proof manifest", 2_000_000)
    _equal(hashlib.sha256(proof_raw).hexdigest(), D240_PROOF_MANIFEST_SHA256, "D240 proof hash")
    proof = _strict_json_object(proof_raw, "D240 proof manifest")
    _equal(proof.get("content_id"), D240_PROOF_CONTENT_ID, "D240 proof content ID")
    execution = _mapping(proof.get("execution"), "D240 execution")
    if any(value != 0 for value in execution.values()):
        raise EntsoeParquetStreamingIntegrityError("D240 execution counters are non-zero")
    artifacts = _mapping(proof.get("artifacts"), "D240 proof artifacts")
    if any(_mapping(item, "D240 proof artifact").get("contains_raw_values") is not False for item in artifacts.values()):
        raise EntsoeParquetStreamingIntegrityError("D240 proof raw-value claim differs")


def _verify_d241_real_schema_binding(proof_manifest_path: str | Path) -> None:
    proof_raw = _read_small(
        proof_manifest_path,
        "D241 real schema proof manifest",
        2_000_000,
    )
    _equal(
        hashlib.sha256(proof_raw).hexdigest(),
        D241_REAL_SCHEMA_PROOF_MANIFEST_SHA256,
        "D241 real schema proof hash",
    )
    proof = _strict_json_object(proof_raw, "D241 real schema proof manifest")
    _equal(
        proof.get("content_id"),
        D241_REAL_SCHEMA_PROOF_CONTENT_ID,
        "D241 real schema proof content ID",
    )
    _equal(
        proof.get("capture_content_id"),
        D241_REAL_SCHEMA_CAPTURE_CONTENT_ID,
        "D241 real schema capture content ID",
    )
    _equal(proof.get("status"), D241_REAL_SCHEMA_STATUS, "D241 real schema status")
    _equal(
        proof.get("execution"),
        {
            "control_plane_get_count": 3,
            "databricks_write_count": 0,
            "sql_statement_count": 0,
            "table_row_count_opened": 0,
            "warehouse_start_count": 0,
        },
        "D241 real schema execution",
    )
    _equal(
        proof.get("authorities"),
        {
            "model_input_authorized": False,
            "point_in_time_availability_proven": False,
            "production_authorized": False,
        },
        "D241 real schema authorities",
    )


def _validate_package_directory(manifest_path: Path, snapshot_id: str) -> Path:
    if manifest_path.name != "manifest.json":
        raise EntsoeParquetStreamingIntegrityError("package manifest file name differs")
    package_dir = assert_absolute_path_has_no_links(manifest_path.parent)
    intake_root = assert_absolute_path_has_no_links(SYNTHETIC_PARQUET_ROOT)
    if package_dir.parent != intake_root or package_dir.name != snapshot_id:
        raise EntsoeParquetStreamingIntegrityError("synthetic Parquet package path differs")
    return package_dir


def _validate_parquet_footer(stream, size_bytes: int, role: str) -> int:
    if size_bytes < 12:
        raise EntsoeParquetStreamingIntegrityError(f"Parquet file is too small: {role}")
    stream.seek(0)
    if stream.read(4) != b"PAR1":
        raise EntsoeParquetStreamingIntegrityError(f"Parquet leading magic differs: {role}")
    stream.seek(-8, os.SEEK_END)
    footer = stream.read(8)
    if len(footer) != 8 or footer[4:] != b"PAR1":
        raise EntsoeParquetStreamingIntegrityError(f"Parquet footer magic differs: {role}")
    footer_bytes = int.from_bytes(footer[:4], "little", signed=False)
    if footer_bytes <= 0 or footer_bytes > MAX_FOOTER_BYTES or footer_bytes + 8 > size_bytes:
        raise EntsoeParquetStreamingIntegrityError(f"Parquet footer size invalid: {role}")
    return footer_bytes


def _hash_stream(stream) -> tuple[str, int]:
    stream.seek(0)
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = stream.read(HASH_CHUNK_BYTES)
        if not chunk:
            break
        if len(chunk) > HASH_CHUNK_BYTES:
            raise EntsoeParquetStreamingIntegrityError("hash chunk cap exceeded")
        digest.update(chunk)
        total += len(chunk)
    return digest.hexdigest(), total


def _read_small(path: str | Path, label: str, max_bytes: int) -> bytes:
    try:
        return read_stable_single_link_file(path, label=label, max_bytes=max_bytes)
    except ValueError as exc:
        raise EntsoeParquetStreamingIntegrityError(str(exc)) from exc


def _strict_json_object(payload: bytes, label: str) -> Mapping[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise EntsoeParquetStreamingIntegrityError(f"{label} duplicate key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise EntsoeParquetStreamingIntegrityError(f"{label} non-finite constant: {value}")

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeParquetStreamingIntegrityError(f"{label} is not strict UTF-8 JSON") from exc
    return _mapping(value, label)


def _require_canonical_workspace() -> None:
    if Path.cwd().resolve() != REPOSITORY_ROOT:
        raise EntsoeParquetStreamingIntegrityError("cwd must be the canonical workspace")


def _relative_path(value: object, role: str) -> str:
    if not isinstance(value, str) or "\\" in value or ":" in value or "//" in value:
        raise EntsoeParquetStreamingIntegrityError(f"artifact path is invalid: {role}")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {"", ".", ".."}:
        raise EntsoeParquetStreamingIntegrityError(f"artifact path is unsafe: {role}")
    return value


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise EntsoeParquetStreamingIntegrityError(f"{label} must use UTC whole-second Z")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise EntsoeParquetStreamingIntegrityError(f"{label} is invalid") from exc


def _utc_datetime(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise EntsoeParquetStreamingIntegrityError(f"{label} must be aware UTC")
    return value.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _uuid(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise EntsoeParquetStreamingIntegrityError(f"{label} must be canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise EntsoeParquetStreamingIntegrityError(f"{label} must be canonical UUID") from exc
    if str(parsed) != value or parsed.variant != uuid.RFC_4122:
        raise EntsoeParquetStreamingIntegrityError(f"{label} must be canonical RFC 4122 UUID")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise EntsoeParquetStreamingIntegrityError(f"{label} must be lowercase SHA-256")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise EntsoeParquetStreamingIntegrityError(f"{label} must be a positive integer")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EntsoeParquetStreamingIntegrityError(f"{label} must be non-empty")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeParquetStreamingIntegrityError(f"{label} must be an object")
    return value


def _exact_fields(value: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    if set(value) != set(expected):
        raise EntsoeParquetStreamingIntegrityError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoeParquetStreamingIntegrityError(f"{label} differs: expected {expected!r}, received {value!r}")


def _stable_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(value.st_nlink),
    )


def _is_single_link_regular(value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    return bool(
        stat.S_ISREG(value.st_mode)
        and int(value.st_nlink) == 1
        and not (os.name == "nt" and attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)))
    )


__all__ = [
    "ARROW_TYPES",
    "CONTRACT_CONTENT_ID",
    "EntsoeParquetStreamingIntegrityError",
    "EntsoeParquetStreamingIntegrityResult",
    "MANIFEST_AUTHORITIES",
    "NORMALIZED_TYPES",
    "RECEIPT_AUTHORITIES",
    "ROLE_COLUMNS",
    "ROLE_FILE_NAMES",
    "ROLE_ORDER",
    "SYNTHETIC_PARQUET_ROOT",
    "canonical_sha256",
    "derive_arrow_schema_sha256",
    "derive_normalized_schema_sha256",
    "derive_snapshot_id",
    "expected_arrow_schema",
    "validate_parquet_streaming_contract",
    "validate_parquet_streaming_receipt",
    "verify_synthetic_entsoe_parquet_package",
]

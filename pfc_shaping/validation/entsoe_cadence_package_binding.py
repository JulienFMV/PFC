"""Bound effective-dated cadence evidence to one streamed ENTSO-E package.

This verifier is deliberately synthetic-only.  It keeps the exact D244 base
package unchanged, replays D245 quality first, and then compares only series
identifiers and delivery timestamps with a separately content-addressed
resolution-regime Parquet sidecar.  Raw numeric values are never opened by the
cadence pass and all SQLite state is ephemeral below ``build/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pfc_shaping.path_safety import assert_absolute_path_has_no_links
from pfc_shaping.validation import entsoe_incremental_quality as d245
from pfc_shaping.validation import entsoe_parquet_streaming_integrity as d244
from pfc_shaping.validation import entsoe_resolution_regimes as d261

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = PHASE / "ENTSOE-CADENCE-PACKAGE-BINDING-CONTRACT-V1.json"
D245_CONTRACT_PATH = PHASE / "ENTSOE-INCREMENTAL-QUALITY-PROFILE-CONTRACT-V1.json"
D261_CONTRACT_PATH = PHASE / "ENTSOE-RESOLUTION-REGIME-CONTRACT-V2.json"
SIDECAR_ROOT = ROOT / "build" / "databricks-eex-daily" / "synthetic-entsoe-cadence-sidecars"
STAGING_ROOT = ROOT / "build" / "databricks-eex-daily" / "d270-cadence-staging"

CONTRACT_SCHEMA = "fmv_entsoe_cadence_package_binding_contract.v1"
CONTRACT_STATUS = "SYNTHETIC_COMPOSITE_ONLY_NO_REAL_DATA_AUTHORITY"
CONTRACT_FILE_SHA256 = "b390cdb3eb19791bd540a07833769fa18f0db3a529c56eed6c05b6567ae508d5"
CONTRACT_CONTENT_ID = "db1ffcd92bb3ae720b44da02a6e6f151a630b9e3887bd84902e91cddc3cb717f"
MANIFEST_SCHEMA = "fmv_entsoe_cadence_sidecar_manifest.v1"
PROFILE_SCHEMA = "fmv_entsoe_cadence_package_profile.v1"
PASS_STATUS = "PASS_SYNTHETIC_CADENCE_BOUND_TO_QUALITY_PACKAGE_NO_GO"
GAP_STATUS = "FAIL_SYNTHETIC_CADENCE_GAPS_NO_GO"
EVIDENCE_CLASS = "SYNTHETIC_TEST_ONLY"
ROLE = "series_resolution_regimes"
FILE_NAME = "series_resolution_regimes.parquet"
RECORD_BATCH_ROWS = 2048
MAX_SIDECAR_ROWS = 1_000_000
MAX_EXPECTED_SLOTS = 5_000_000
MAX_DATABASE_BYTES = 1_073_741_824
MAX_SIDECAR_FILE_BYTES = 268_435_456
MAX_STRING_BYTES = 2048
HASH_CHUNK_BYTES = 1024 * 1024
RESOLUTION_SECONDS = {"PT15M": 900, "PT30M": 1800, "PT1H": 3600}
SIDECAR_COLUMNS = (
    "series_id",
    "valid_from_utc",
    "valid_to_utc",
    "native_resolution",
    "source_document_id",
    "source_locator",
    "owner_confirmed_at_utc",
)
SIDECAR_TYPES = {
    "series_id": "string",
    "valid_from_utc": "timestamp[ns, tz=UTC]",
    "valid_to_utc": "timestamp[ns, tz=UTC]",
    "native_resolution": "string",
    "source_document_id": "string",
    "source_locator": "string",
    "owner_confirmed_at_utc": "timestamp[ns, tz=UTC]",
}
AUTHORITIES = {
    "real_source_evidence": False,
    "external_owner_identity": False,
    "point_in_time_availability": False,
    "value_quality": False,
    "model_input": False,
    "model_selection": False,
    "candidate": False,
    "promotion": False,
    "production": False,
}
_SHA = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}$")


class EntsoeCadencePackageBindingError(ValueError):
    """Raised when a composite cadence package is ambiguous or unsafe."""


@dataclass(frozen=True)
class EntsoeCadencePackageBindingResult:
    """Value-free cadence assessment bound to a D245 quality result."""

    summary: dict[str, object]
    series_profile: tuple[dict[str, object], ...]
    summary_content_id: str
    series_profile_content_id: str
    sidecar_manifest_content_id: str
    d245_summary_content_id: str
    sqlite_peak_bytes: int
    staging_cleaned: bool


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_sidecar_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("series_id", pa.string(), nullable=True),
            pa.field("valid_from_utc", pa.timestamp("ns", tz="UTC"), nullable=True),
            pa.field("valid_to_utc", pa.timestamp("ns", tz="UTC"), nullable=True),
            pa.field("native_resolution", pa.string(), nullable=True),
            pa.field("source_document_id", pa.string(), nullable=True),
            pa.field("source_locator", pa.string(), nullable=True),
            pa.field("owner_confirmed_at_utc", pa.timestamp("ns", tz="UTC"), nullable=True),
        ],
        metadata=None,
    )


def derive_sidecar_arrow_schema_sha256() -> str:
    schema = expected_sidecar_schema()
    return canonical_sha256(
        {
            "role": ROLE,
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


def derive_sidecar_manifest_content_id(manifest: Mapping[str, object]) -> str:
    return canonical_sha256(_mapping(manifest, "sidecar manifest"))


def validate_cadence_package_binding_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    contract = _mapping(document, "D270 contract")
    _exact(
        contract,
        {
            "schema_version",
            "decision",
            "status",
            "purpose",
            "predecessor_bindings",
            "sidecar_package",
            "streaming_quality",
            "cadence_rules",
            "execution",
            "authorities",
        },
        "D270 contract",
    )
    _equal(contract["schema_version"], CONTRACT_SCHEMA, "contract schema")
    _equal(contract["decision"], "D-20260806-270", "contract decision")
    _equal(contract["status"], CONTRACT_STATUS, "contract status")
    _text(contract["purpose"], "contract purpose")

    predecessors = _mapping(contract["predecessor_bindings"], "predecessors")
    _exact(
        predecessors,
        {
            "d244_base_package_remains_exactly_three_roles",
            "d245_incremental_quality_must_pass_first",
            "d261_resolution_contract_schema",
            "d261_resolution_contract_sha256",
            "d261_resolution_contract_content_id",
        },
        "predecessors",
    )
    _equal(predecessors["d244_base_package_remains_exactly_three_roles"], True, "D244 binding")
    _equal(predecessors["d245_incremental_quality_must_pass_first"], True, "D245 binding")
    _equal(predecessors["d261_resolution_contract_schema"], d261.CONTRACT_SCHEMA, "D261 schema")
    _equal(predecessors["d261_resolution_contract_sha256"], d261.CONTRACT_FILE_SHA256, "D261 hash")
    _equal(
        predecessors["d261_resolution_contract_content_id"],
        d261.CONTRACT_CONTENT_ID,
        "D261 content ID",
    )

    sidecar = _mapping(contract["sidecar_package"], "sidecar policy")
    _exact(
        sidecar,
        {
            "manifest_schema",
            "artifact_role",
            "artifact_file_name",
            "content_addressed_directory_required",
            "base_snapshot_id_binding_required",
            "base_manifest_content_id_binding_required",
            "quality_context_content_id_binding_required",
            "explicit_assessment_start_end_required",
            "metadata_as_of_not_before_assessment_end",
            "exact_directory_inventory_required",
        },
        "sidecar policy",
    )
    _equal(sidecar["manifest_schema"], MANIFEST_SCHEMA, "sidecar schema")
    _equal(sidecar["artifact_role"], ROLE, "sidecar role")
    _equal(sidecar["artifact_file_name"], FILE_NAME, "sidecar file")
    if any(value is not True for key, value in sidecar.items() if key.endswith("required")):
        raise EntsoeCadencePackageBindingError("sidecar required controls differ")

    streaming = _mapping(contract["streaming_quality"], "streaming policy")
    _exact(
        streaming,
        {
            "base_package_integrity_replayed",
            "base_incremental_quality_replayed",
            "base_value_columns_opened_by_cadence_pass",
            "sidecar_schema_and_hash_verified",
            "sqlite_spill_is_repo_local_and_ephemeral",
            "expected_and_observed_targets_compared_as_sorted_streams",
            "maximum_record_batch_rows",
            "maximum_sidecar_rows",
            "maximum_expected_slots",
            "maximum_sqlite_bytes",
        },
        "streaming policy",
    )
    for key in (
        "base_package_integrity_replayed",
        "base_incremental_quality_replayed",
        "sidecar_schema_and_hash_verified",
        "sqlite_spill_is_repo_local_and_ephemeral",
        "expected_and_observed_targets_compared_as_sorted_streams",
    ):
        _equal(streaming[key], True, key)
    _equal(streaming["base_value_columns_opened_by_cadence_pass"], False, "cadence value access")
    _equal(streaming["maximum_record_batch_rows"], RECORD_BATCH_ROWS, "batch cap")
    _equal(streaming["maximum_sidecar_rows"], MAX_SIDECAR_ROWS, "sidecar row cap")
    _equal(streaming["maximum_expected_slots"], MAX_EXPECTED_SLOTS, "expected-slot cap")
    _equal(streaming["maximum_sqlite_bytes"], MAX_DATABASE_BYTES, "SQLite cap")

    cadence = _mapping(contract["cadence_rules"], "cadence rules")
    expected_cadence = {
        "exact_series_coverage_required": True,
        "left_closed_right_open_regimes": True,
        "regime_gaps_or_overlaps_allowed": False,
        "transition_must_close_previous_grid": True,
        "dimension_resolution_checked_at_metadata_as_of": True,
        "metadata_as_of_may_be_off_delivery_grid": True,
        "owner_confirmation_not_after_metadata_as_of": True,
        "clean_https_source_locator_required": True,
        "observations_outside_assessment_window_allowed": False,
        "leading_and_trailing_missing_slots_reported": True,
        "implicit_upsampling_authorized": False,
        "implicit_downsampling_authorized": False,
        "implicit_forward_fill_authorized": False,
    }
    _equal(cadence, expected_cadence, "cadence rules")
    execution = _mapping(contract["execution"], "execution")
    if set(execution) != {
        "databricks_connection_budget",
        "databricks_statement_budget",
        "warehouse_start_budget",
        "databricks_write_budget",
        "network_call_budget",
        "h_drive_access_budget",
    } or any(value != 0 for value in execution.values()):
        raise EntsoeCadencePackageBindingError("execution budgets must be exact zero")
    _equal(contract["authorities"], AUTHORITIES, "contract authorities")
    _equal(canonical_sha256(contract), CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "bounded_streaming_implemented": True,
        "base_package_role_count": len(d244.ROLE_ORDER),
        "cadence_value_columns_opened": False,
        "databricks_connection_budget": 0,
        "production_authorized": False,
    }


def verify_synthetic_entsoe_cadence_package(
    *,
    contract_path: str | Path,
    package_manifest_path: str | Path,
    quality_context_path: str | Path,
    cadence_manifest_path: str | Path,
    reference_time_utc: datetime,
) -> EntsoeCadencePackageBindingResult:
    """Verify one synthetic composite with bounded Arrow and SQLite state."""

    _require_workspace_and_temp()
    contract_raw = _read_small(contract_path, "D270 contract", 100_000)
    _equal(hashlib.sha256(contract_raw).hexdigest(), CONTRACT_FILE_SHA256, "D270 contract hash")
    validate_cadence_package_binding_contract(_strict_json(contract_raw, "D270 contract"))
    d261.validate_resolution_regime_contract_path(D261_CONTRACT_PATH)
    reference = _utc_datetime(reference_time_utc, "reference time")

    quality = d245.profile_synthetic_entsoe_incremental_quality(
        contract_path=D245_CONTRACT_PATH,
        package_manifest_path=package_manifest_path,
        quality_context_path=quality_context_path,
        reference_time_utc=reference,
    )
    if quality.summary.get("status") != "PASS_SYNTHETIC_INCREMENTAL_QUALITY_MODEL_NO_GO":
        raise EntsoeCadencePackageBindingError("D245 incremental quality did not pass")

    package_path = assert_absolute_path_has_no_links(package_manifest_path)
    package_manifest = _strict_json(
        _read_small(package_path, "base package manifest", 2_000_000),
        "base package manifest",
    )
    cadence_path = assert_absolute_path_has_no_links(cadence_manifest_path)
    cadence_manifest = _strict_json(
        _read_small(cadence_path, "cadence sidecar manifest", 2_000_000),
        "cadence sidecar manifest",
    )
    manifest_content_id, window = _validate_sidecar_manifest(
        cadence_path,
        cadence_manifest,
        package_manifest,
        quality.context_content_id,
        reference,
    )
    declaration = _mapping(cadence_manifest["artifact"], "cadence artifact declaration")

    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    staging = STAGING_ROOT / str(uuid.uuid4())
    staging.mkdir(exist_ok=False)
    database_path = staging / "cadence-index.sqlite"
    connection: sqlite3.Connection | None = None
    peak_bytes = 0
    counters = {
        "dimension_rows": 0,
        "target_rows": 0,
        "regime_rows": 0,
        "base_record_batches": 0,
        "sidecar_record_batches": 0,
    }
    cleaned = False
    try:
        connection = sqlite3.connect(database_path)
        _configure_database(connection)
        _create_tables(connection)
        declarations = {
            str(item["role"]): _mapping(item, "base artifact declaration")
            for item in package_manifest["artifacts"]
        }
        _stream_base_columns(
            package_path.parent / d244.ROLE_FILE_NAMES["series_dimension"],
            declarations["series_dimension"],
            "series_dimension",
            ("series_id", "group_name", "native_resolution"),
            lambda batch: _consume_dimension(batch, connection, counters),
        )
        _stream_base_columns(
            package_path.parent / d244.ROLE_FILE_NAMES["latest_values"],
            declarations["latest_values"],
            "latest_values",
            ("series_id", "target_ts_utc"),
            lambda batch: _consume_targets(batch, connection, counters),
        )
        counters["base_record_batches"] = connection.total_changes
        _stream_sidecar(
            cadence_path.parent / FILE_NAME,
            declaration,
            lambda batch: _consume_regimes(batch, connection, counters),
        )
        connection.commit()
        peak_bytes = _database_bytes(connection)
        if peak_bytes > MAX_DATABASE_BYTES:
            raise EntsoeCadencePackageBindingError("SQLite byte cap exceeded")
        summary, profiles = _finalize_profile(
            connection,
            counters,
            window,
            quality,
            manifest_content_id,
            peak_bytes,
        )
    finally:
        if connection is not None:
            connection.close()
        if staging.exists():
            for entry in tuple(staging.iterdir()):
                if entry.is_file():
                    entry.unlink()
            staging.rmdir()
        cleaned = not staging.exists()
    if not cleaned:
        raise EntsoeCadencePackageBindingError("SQLite staging cleanup failed")
    summary = {**summary, "sqlite_staging_cleaned": True}
    return EntsoeCadencePackageBindingResult(
        summary=summary,
        series_profile=tuple(profiles),
        summary_content_id=canonical_sha256(summary),
        series_profile_content_id=canonical_sha256(profiles),
        sidecar_manifest_content_id=manifest_content_id,
        d245_summary_content_id=quality.summary_content_id,
        sqlite_peak_bytes=peak_bytes,
        staging_cleaned=True,
    )


def _validate_sidecar_manifest(
    path: Path,
    manifest: Mapping[str, object],
    package: Mapping[str, object],
    quality_context_content_id: str,
    reference: datetime,
) -> tuple[str, tuple[int, int, int]]:
    _exact(
        manifest,
        {
            "schema_version",
            "evidence_class",
            "snapshot_id",
            "base_package_manifest_content_id",
            "quality_context_content_id",
            "assessment_start_utc",
            "assessment_end_utc",
            "metadata_as_of_utc",
            "created_at_utc",
            "artifact",
            "authorities",
        },
        "sidecar manifest",
    )
    _equal(manifest["schema_version"], MANIFEST_SCHEMA, "sidecar manifest schema")
    _equal(manifest["evidence_class"], EVIDENCE_CLASS, "sidecar evidence")
    _equal(manifest["snapshot_id"], package["snapshot_id"], "sidecar snapshot")
    _equal(
        manifest["base_package_manifest_content_id"],
        d244.canonical_sha256(package),
        "base manifest binding",
    )
    _equal(
        manifest["quality_context_content_id"],
        quality_context_content_id,
        "quality context binding",
    )
    _equal(manifest["authorities"], AUTHORITIES, "sidecar authorities")
    start = _datetime_ns(_parse_utc(manifest["assessment_start_utc"], "assessment start"))
    end = _datetime_ns(_parse_utc(manifest["assessment_end_utc"], "assessment end"))
    metadata = _datetime_ns(_parse_utc(manifest["metadata_as_of_utc"], "metadata as-of"))
    created = _parse_utc(manifest["created_at_utc"], "sidecar creation time")
    if start >= end:
        raise EntsoeCadencePackageBindingError("assessment window is empty or reversed")
    if metadata < end:
        raise EntsoeCadencePackageBindingError("metadata as-of precedes assessment end")
    if _ns_to_datetime(metadata) > reference or created > reference:
        raise EntsoeCadencePackageBindingError("sidecar time is after reference")
    artifact = _mapping(manifest["artifact"], "cadence artifact declaration")
    _validate_sidecar_declaration(artifact)
    content_id = derive_sidecar_manifest_content_id(manifest)
    root = assert_absolute_path_has_no_links(SIDECAR_ROOT)
    if path.name != "manifest.json" or path.parent.parent != root or path.parent.name != content_id:
        raise EntsoeCadencePackageBindingError("sidecar content-addressed path differs")
    if {entry.name for entry in path.parent.iterdir()} != {"manifest.json", FILE_NAME}:
        raise EntsoeCadencePackageBindingError("sidecar directory inventory differs")
    return content_id, (start, end, metadata)


def _validate_sidecar_declaration(declaration: Mapping[str, object]) -> None:
    _exact(
        declaration,
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
            "arrow_schema_sha256",
            "compression",
            "parquet_format_version",
            "created_by",
        },
        "sidecar artifact declaration",
    )
    _equal(declaration["role"], ROLE, "sidecar role")
    _equal(declaration["format"], "PARQUET", "sidecar format")
    _equal(declaration["relative_path"], FILE_NAME, "sidecar path")
    _sha(declaration["sha256"], "sidecar hash")
    size = _positive_int(declaration["size_bytes"], "sidecar size")
    rows = _positive_int(declaration["row_count"], "sidecar rows")
    groups = _positive_int(declaration["row_group_count"], "sidecar row groups")
    if size > MAX_SIDECAR_FILE_BYTES or rows > MAX_SIDECAR_ROWS or groups > 4096:
        raise EntsoeCadencePackageBindingError("sidecar declared cap exceeded")
    _equal(declaration["columns"], list(SIDECAR_COLUMNS), "sidecar columns")
    _equal(declaration["column_types"], SIDECAR_TYPES, "sidecar types")
    _equal(
        declaration["arrow_schema_sha256"],
        derive_sidecar_arrow_schema_sha256(),
        "sidecar schema hash",
    )
    _equal(declaration["compression"], ["ZSTD"], "sidecar compression")
    _text(declaration["parquet_format_version"], "sidecar Parquet version")
    _text(declaration["created_by"], "sidecar created_by")


def _stream_base_columns(
    path: Path,
    declaration: Mapping[str, object],
    role: str,
    columns: tuple[str, ...],
    consumer: Callable[[pa.RecordBatch], None],
) -> None:
    def scan(parquet: pq.ParquetFile) -> None:
        _equal(parquet.schema_arrow, d244.expected_arrow_schema(role), f"base schema {role}")
        visited = 0
        for batch in parquet.iter_batches(
            batch_size=RECORD_BATCH_ROWS, columns=list(columns), use_threads=False
        ):
            if batch.num_rows <= 0 or batch.num_rows > RECORD_BATCH_ROWS:
                raise EntsoeCadencePackageBindingError(f"base batch cap exceeded: {role}")
            for column in columns:
                if batch.column(batch.schema.get_field_index(column)).null_count:
                    raise EntsoeCadencePackageBindingError(
                        f"base cadence null rejected: {role}.{column}"
                    )
            consumer(batch)
            visited += batch.num_rows
        _equal(visited, declaration["row_count"], f"base cadence rows {role}")

    _safe_parquet_scan(path, declaration, role, scan)


def _stream_sidecar(
    path: Path,
    declaration: Mapping[str, object],
    consumer: Callable[[pa.RecordBatch], None],
) -> None:
    def scan(parquet: pq.ParquetFile) -> None:
        metadata = parquet.metadata
        if metadata is None:
            raise EntsoeCadencePackageBindingError("sidecar Parquet metadata missing")
        _equal(metadata.num_rows, declaration["row_count"], "sidecar row count")
        _equal(metadata.num_row_groups, declaration["row_group_count"], "sidecar row groups")
        _equal(parquet.schema_arrow, expected_sidecar_schema(), "sidecar Arrow schema")
        codecs = {
            str(metadata.row_group(group).column(column).compression).upper()
            for group in range(metadata.num_row_groups)
            for column in range(metadata.num_columns)
        }
        _equal(sorted(codecs), declaration["compression"], "sidecar codecs")
        _equal(
            str(metadata.format_version),
            declaration["parquet_format_version"],
            "sidecar Parquet version",
        )
        _equal(str(metadata.created_by), declaration["created_by"], "sidecar created_by")
        visited = 0
        for batch in parquet.iter_batches(batch_size=RECORD_BATCH_ROWS, use_threads=False):
            if batch.num_rows <= 0 or batch.num_rows > RECORD_BATCH_ROWS:
                raise EntsoeCadencePackageBindingError("sidecar batch cap exceeded")
            _validate_sidecar_batch(batch)
            consumer(batch)
            visited += batch.num_rows
        _equal(visited, declaration["row_count"], "sidecar visited rows")

    _safe_parquet_scan(path, declaration, ROLE, scan)


def _safe_parquet_scan(
    path: Path,
    declaration: Mapping[str, object],
    role: str,
    scanner: Callable[[pq.ParquetFile], None],
) -> None:
    lexical = assert_absolute_path_has_no_links(path)
    before = lexical.stat(follow_symlinks=False)
    if not d244._is_single_link_regular(before):
        raise EntsoeCadencePackageBindingError(f"artifact is not single-link: {role}")
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
    descriptor: int | None = None
    try:
        descriptor = os.open(lexical, flags)
        opened = os.fstat(descriptor)
        if d244._stable_identity(opened) != d244._stable_identity(before):
            raise EntsoeCadencePackageBindingError(f"artifact identity changed: {role}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            digest_before, size_before = _hash_stream(stream)
            _equal(digest_before, declaration["sha256"], f"artifact hash {role}")
            _equal(size_before, declaration["size_bytes"], f"artifact size {role}")
            stream.seek(0)
            scanner(
                pq.ParquetFile(
                    stream,
                    thrift_string_size_limit=d244.THRIFT_STRING_SIZE_LIMIT,
                    thrift_container_size_limit=d244.THRIFT_CONTAINER_SIZE_LIMIT,
                    page_checksum_verification=True,
                )
            )
            stream.seek(0)
            digest_after, size_after = _hash_stream(stream)
            _equal(digest_after, digest_before, f"post-scan hash {role}")
            _equal(size_after, size_before, f"post-scan size {role}")
            if d244._stable_identity(os.fstat(stream.fileno())) != d244._stable_identity(opened):
                raise EntsoeCadencePackageBindingError(f"artifact descriptor changed: {role}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
    final = lexical.stat(follow_symlinks=False)
    if d244._stable_identity(final) != d244._stable_identity(before):
        raise EntsoeCadencePackageBindingError(f"artifact path changed: {role}")


def _validate_sidecar_batch(batch: pa.RecordBatch) -> None:
    _equal(batch.schema, expected_sidecar_schema(), "sidecar batch schema")
    for column in SIDECAR_COLUMNS:
        array = batch.column(batch.schema.get_field_index(column))
        if column != "valid_to_utc" and array.null_count:
            raise EntsoeCadencePackageBindingError(f"sidecar null rejected: {column}")
        if pa.types.is_string(array.type):
            maximum = pc.max(pc.binary_length(array.cast(pa.binary()))).as_py()
            if maximum is None or maximum > MAX_STRING_BYTES:
                raise EntsoeCadencePackageBindingError(f"sidecar string cap exceeded: {column}")


def _consume_dimension(
    batch: pa.RecordBatch,
    connection: sqlite3.Connection,
    counters: dict[str, int],
) -> None:
    values = {
        name: batch.column(batch.schema.get_field_index(name)).to_pylist()
        for name in batch.schema.names
    }
    for index in range(batch.num_rows):
        series_id = _text(values["series_id"][index], "dimension series ID")
        group = _text(values["group_name"][index], "dimension group")
        resolution = _text(values["native_resolution"][index], "dimension resolution")
        if resolution not in RESOLUTION_SECONDS:
            raise EntsoeCadencePackageBindingError("dimension resolution is unsupported")
        try:
            connection.execute(
                "INSERT INTO dimension(series_id, group_name, declared_resolution) VALUES (?, ?, ?)",
                (series_id, group, resolution),
            )
        except sqlite3.IntegrityError as exc:
            raise EntsoeCadencePackageBindingError("dimension series ID is duplicated") from exc
        counters["dimension_rows"] += 1
    connection.commit()


def _consume_targets(
    batch: pa.RecordBatch,
    connection: sqlite3.Connection,
    counters: dict[str, int],
) -> None:
    ids = batch.column(batch.schema.get_field_index("series_id")).to_pylist()
    timestamps = (
        batch.column(batch.schema.get_field_index("target_ts_utc")).cast(pa.int64()).to_pylist()
    )
    for series_id, target_ns in zip(ids, timestamps, strict=True):
        try:
            connection.execute(
                "INSERT INTO targets(series_id, target_ns) VALUES (?, ?)",
                (_text(series_id, "target series ID"), int(target_ns)),
            )
        except sqlite3.IntegrityError as exc:
            raise EntsoeCadencePackageBindingError("target timestamp key is duplicated") from exc
        counters["target_rows"] += 1
    connection.commit()


def _consume_regimes(
    batch: pa.RecordBatch,
    connection: sqlite3.Connection,
    counters: dict[str, int],
) -> None:
    values = {
        name: batch.column(batch.schema.get_field_index(name)).to_pylist()
        for name in SIDECAR_COLUMNS
    }
    starts = (
        batch.column(batch.schema.get_field_index("valid_from_utc")).cast(pa.int64()).to_pylist()
    )
    ends = batch.column(batch.schema.get_field_index("valid_to_utc")).cast(pa.int64()).to_pylist()
    owners = (
        batch.column(batch.schema.get_field_index("owner_confirmed_at_utc"))
        .cast(pa.int64())
        .to_pylist()
    )
    for index in range(batch.num_rows):
        series_id = _text(values["series_id"][index], "regime series ID")
        resolution = _text(values["native_resolution"][index], "regime resolution")
        document = _text(values["source_document_id"][index], "source document ID")
        locator = _text(values["source_locator"][index], "source locator")
        if resolution not in RESOLUTION_SECONDS:
            raise EntsoeCadencePackageBindingError("regime resolution is unsupported")
        if _IDENTIFIER.fullmatch(document) is None:
            raise EntsoeCadencePackageBindingError("source document ID is invalid")
        parsed = urlsplit(locator)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or any(character.isspace() for character in locator)
        ):
            raise EntsoeCadencePackageBindingError("source locator must be clean HTTPS")
        try:
            connection.execute(
                "INSERT INTO regimes(series_id, start_ns, end_ns, resolution, source_document_id, source_locator, owner_ns) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    series_id,
                    int(starts[index]),
                    None if ends[index] is None else int(ends[index]),
                    resolution,
                    document,
                    locator,
                    int(owners[index]),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise EntsoeCadencePackageBindingError("resolution regime key is duplicated") from exc
        counters["regime_rows"] += 1
    connection.commit()


def _finalize_profile(
    connection: sqlite3.Connection,
    counters: Mapping[str, int],
    window: tuple[int, int, int],
    quality: d245.EntsoeIncrementalQualityResult,
    sidecar_content_id: str,
    sqlite_peak_bytes: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    assessment_start, assessment_end, metadata_as_of = window
    unknown_targets = _scalar(
        connection,
        "SELECT COUNT(*) FROM targets t LEFT JOIN dimension d ON d.series_id=t.series_id WHERE d.series_id IS NULL",
    )
    if unknown_targets:
        raise EntsoeCadencePackageBindingError("target timestamps contain unknown series")
    outside = _scalar(
        connection,
        "SELECT COUNT(*) FROM targets WHERE target_ns < ? OR target_ns >= ?",
        (assessment_start, assessment_end),
    )
    if outside:
        raise EntsoeCadencePackageBindingError("target timestamp lies outside assessment window")
    dimension_ids = {row[0] for row in connection.execute("SELECT series_id FROM dimension")}
    regime_ids = {row[0] for row in connection.execute("SELECT DISTINCT series_id FROM regimes")}
    if dimension_ids != regime_ids:
        raise EntsoeCadencePackageBindingError("resolution regime series coverage differs")

    profiles: list[dict[str, object]] = []
    total_expected = 0
    total_observed = 0
    total_missing = 0
    total_transitions = 0
    for series_id, group_name, declared_resolution in connection.execute(
        "SELECT series_id, group_name, declared_resolution FROM dimension ORDER BY series_id"
    ):
        schedule = list(
            connection.execute(
                "SELECT start_ns, end_ns, resolution, owner_ns FROM regimes WHERE series_id=? ORDER BY start_ns",
                (series_id,),
            )
        )
        _validate_schedule(
            series_id,
            schedule,
            assessment_start,
            assessment_end,
            metadata_as_of,
            declared_resolution,
        )
        expected = _expected_slots(schedule, assessment_start, assessment_end)
        observed = (
            int(row[0])
            for row in connection.execute(
                "SELECT target_ns FROM targets WHERE series_id=? ORDER BY target_ns",
                (series_id,),
            )
        )
        expected_count, observed_count, missing_count = _merge_cadence(
            series_id,
            expected,
            observed,
            MAX_EXPECTED_SLOTS - total_expected,
        )
        resolutions = [
            str(row[2])
            for row in schedule
            if _overlaps(row[0], row[1], assessment_start, assessment_end)
        ]
        profile = {
            "series_id_sha256": hashlib.sha256(str(series_id).encode("utf-8")).hexdigest(),
            "group_name": str(group_name),
            "regime_count": len(resolutions),
            "transition_count": max(0, len(resolutions) - 1),
            "resolution_sequence": resolutions,
            "declared_current_resolution": str(declared_resolution),
            "observed_target_count": observed_count,
            "expected_target_count": expected_count,
            "missing_native_slot_count": missing_count,
            "piecewise_grid_completeness_ratio": float(observed_count / expected_count),
            "raw_series_id_persisted": False,
            "raw_values_opened_or_persisted_by_cadence_pass": False,
        }
        profiles.append(profile)
        total_expected += expected_count
        total_observed += observed_count
        total_missing += missing_count
        total_transitions += int(profile["transition_count"])
    profiles.sort(key=lambda row: str(row["series_id_sha256"]))
    complete = total_missing == 0
    summary = {
        "schema_version": PROFILE_SCHEMA,
        "status": PASS_STATUS if complete else GAP_STATUS,
        "evidence_class": EVIDENCE_CLASS,
        "assessment_start_utc": _ns_to_text(assessment_start),
        "assessment_end_utc": _ns_to_text(assessment_end),
        "metadata_as_of_utc": _ns_to_text(metadata_as_of),
        "series_count": len(dimension_ids),
        "regime_count": int(counters["regime_rows"]),
        "transition_count": total_transitions,
        "expected_target_count": total_expected,
        "observed_target_count": total_observed,
        "missing_native_slot_count": total_missing,
        "all_series_piecewise_grid_complete": complete,
        "base_d245_incremental_quality_passed": True,
        "d245_summary_content_id": quality.summary_content_id,
        "sidecar_manifest_content_id": sidecar_content_id,
        "base_value_columns_opened_by_cadence_pass": False,
        "implicit_upsampling_performed": False,
        "implicit_downsampling_performed": False,
        "implicit_forward_fill_performed": False,
        "raw_series_ids_persisted_in_profile": False,
        "sqlite_peak_bytes": sqlite_peak_bytes,
        "databricks_connection_count": 0,
        "databricks_statement_count": 0,
        "databricks_write_count": 0,
        "warehouse_start_count": 0,
        "network_call_count": 0,
        "h_drive_access_count": 0,
        "authorities": dict(AUTHORITIES),
    }
    return summary, profiles


def _validate_schedule(
    series_id: str,
    schedule: list[tuple[int, int | None, str, int]],
    assessment_start: int,
    assessment_end: int,
    metadata_as_of: int,
    declared_resolution: str,
) -> None:
    if not schedule:
        raise EntsoeCadencePackageBindingError(f"resolution schedule is empty: {series_id}")
    previous_end: int | None = None
    for position, (start, end, resolution, owner) in enumerate(schedule):
        if end is not None and end <= start:
            raise EntsoeCadencePackageBindingError(f"resolution regime is empty: {series_id}")
        if owner > metadata_as_of:
            raise EntsoeCadencePackageBindingError(
                f"owner confirmation is after metadata as-of: {series_id}"
            )
        if position and previous_end != start:
            raise EntsoeCadencePackageBindingError(
                f"resolution regimes have a gap or overlap: {series_id}"
            )
        if position and schedule[position - 1][1] is None:
            raise EntsoeCadencePackageBindingError(f"open-ended regime is not final: {series_id}")
        if end is not None and (end - start) % (RESOLUTION_SECONDS[resolution] * 1_000_000_000):
            raise EntsoeCadencePackageBindingError(
                f"transition does not close previous grid: {series_id}"
            )
        previous_end = end
    if (
        _active_regime(schedule, assessment_start) is None
        or _active_regime(schedule, assessment_end - 1) is None
    ):
        raise EntsoeCadencePackageBindingError(f"assessment window is not covered: {series_id}")
    final = _active_regime(schedule, metadata_as_of)
    if final is None:
        raise EntsoeCadencePackageBindingError(f"metadata as-of is not covered: {series_id}")
    _, _, resolution, _ = final
    if resolution != declared_resolution:
        raise EntsoeCadencePackageBindingError(f"declared current resolution differs: {series_id}")
    end_regime = _active_regime(schedule, assessment_end - 1)
    assert end_regime is not None
    if (assessment_end - end_regime[0]) % (RESOLUTION_SECONDS[end_regime[2]] * 1_000_000_000):
        raise EntsoeCadencePackageBindingError(
            f"assessment end does not close active grid: {series_id}"
        )


def _active_regime(
    schedule: list[tuple[int, int | None, str, int]], target: int
) -> tuple[int, int | None, str, int] | None:
    matches = [row for row in schedule if row[0] <= target and (row[1] is None or target < row[1])]
    if len(matches) > 1:
        raise EntsoeCadencePackageBindingError("multiple active resolution regimes")
    return matches[0] if matches else None


def _expected_slots(
    schedule: list[tuple[int, int | None, str, int]],
    assessment_start: int,
    assessment_end: int,
) -> Iterator[int]:
    for start, end, resolution, _ in schedule:
        overlap_start = max(start, assessment_start)
        overlap_end = assessment_end if end is None else min(end, assessment_end)
        if overlap_end <= overlap_start:
            continue
        step = RESOLUTION_SECONDS[resolution] * 1_000_000_000
        current = start + max(0, (overlap_start - start + step - 1) // step) * step
        while current < overlap_end:
            yield current
            current += step


def _merge_cadence(
    series_id: str,
    expected: Iterator[int],
    observed: Iterator[int],
    remaining_budget: int,
) -> tuple[int, int, int]:
    sentinel = object()
    observed_value: int | object = next(observed, sentinel)
    expected_count = 0
    observed_count = 0
    missing_count = 0
    for expected_value in expected:
        expected_count += 1
        if expected_count > remaining_budget:
            raise EntsoeCadencePackageBindingError("expected target slot cap exceeded")
        while observed_value is not sentinel and int(observed_value) < expected_value:
            raise EntsoeCadencePackageBindingError(
                f"observed target is off active grid: {series_id}"
            )
        if observed_value is not sentinel and int(observed_value) == expected_value:
            observed_count += 1
            observed_value = next(observed, sentinel)
        else:
            missing_count += 1
    if observed_value is not sentinel:
        raise EntsoeCadencePackageBindingError(f"observed target is off active grid: {series_id}")
    return expected_count, observed_count, missing_count


def _overlaps(start: int, end: int | None, window_start: int, window_end: int) -> bool:
    effective_end = window_end if end is None else end
    return start < window_end and effective_end > window_start


def _configure_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA foreign_keys=ON")


def _create_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE dimension(
            series_id TEXT PRIMARY KEY,
            group_name TEXT NOT NULL,
            declared_resolution TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE targets(
            series_id TEXT NOT NULL,
            target_ns INTEGER NOT NULL,
            PRIMARY KEY(series_id, target_ns)
        ) WITHOUT ROWID;
        CREATE TABLE regimes(
            series_id TEXT NOT NULL,
            start_ns INTEGER NOT NULL,
            end_ns INTEGER,
            resolution TEXT NOT NULL,
            source_document_id TEXT NOT NULL,
            source_locator TEXT NOT NULL,
            owner_ns INTEGER NOT NULL,
            PRIMARY KEY(series_id, start_ns)
        ) WITHOUT ROWID;
        """
    )


def _database_bytes(connection: sqlite3.Connection) -> int:
    page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
    page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
    return page_count * page_size


def _scalar(
    connection: sqlite3.Connection, statement: str, parameters: tuple[object, ...] = ()
) -> int:
    return int(connection.execute(statement, parameters).fetchone()[0])


def _require_workspace_and_temp() -> None:
    if Path.cwd().resolve() != ROOT.resolve():
        raise EntsoeCadencePackageBindingError("current directory is not canonical workspace")
    build = (ROOT / "build").resolve()
    for name in ("TEMP", "TMP"):
        value = os.environ.get(name)
        if value is None:
            raise EntsoeCadencePackageBindingError(f"{name} is not configured")
        path = Path(value).resolve()
        if path != build and build not in path.parents:
            raise EntsoeCadencePackageBindingError(f"{name} is outside repo build")


def _read_small(path: str | Path, label: str, limit: int) -> bytes:
    lexical = assert_absolute_path_has_no_links(Path(path).resolve())
    if lexical.stat().st_size > limit:
        raise EntsoeCadencePackageBindingError(f"{label} exceeds byte cap")
    return lexical.read_bytes()


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeCadencePackageBindingError(f"{label} JSON is invalid") from exc
    return _mapping(value, label)


def _hash_stream(stream) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    stream.seek(0)
    while chunk := stream.read(HASH_CHUNK_BYTES):
        digest.update(chunk)
        total += len(chunk)
    return digest.hexdigest(), total


def _parse_utc(value: object, label: str) -> datetime:
    text = _text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EntsoeCadencePackageBindingError(f"{label} is invalid") from exc
    return _utc_datetime(parsed, label)


def _utc_datetime(value: datetime, label: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timezone.utc.utcoffset(value)
    ):
        raise EntsoeCadencePackageBindingError(f"{label} must be UTC-aware")
    return value.astimezone(timezone.utc)


def _datetime_ns(value: datetime) -> int:
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = value - epoch
    return ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1000


def _ns_to_datetime(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc)


def _ns_to_text(value: int) -> str:
    return _ns_to_datetime(value).isoformat().replace("+00:00", "Z")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EntsoeCadencePackageBindingError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EntsoeCadencePackageBindingError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoeCadencePackageBindingError(f"{label} differs")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EntsoeCadencePackageBindingError(f"{label} must be clean non-empty text")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise EntsoeCadencePackageBindingError(f"{label} must be a positive integer")
    return value


def _sha(value: object, label: str) -> str:
    text = _text(value, label)
    if _SHA.fullmatch(text) is None:
        raise EntsoeCadencePackageBindingError(f"{label} must be lowercase SHA-256")
    return text


__all__ = [
    "AUTHORITIES",
    "CONTRACT_CONTENT_ID",
    "CONTRACT_FILE_SHA256",
    "CONTRACT_PATH",
    "EntsoeCadencePackageBindingError",
    "EntsoeCadencePackageBindingResult",
    "FILE_NAME",
    "MANIFEST_SCHEMA",
    "SIDECAR_COLUMNS",
    "SIDECAR_ROOT",
    "SIDECAR_TYPES",
    "canonical_sha256",
    "derive_sidecar_arrow_schema_sha256",
    "derive_sidecar_manifest_content_id",
    "expected_sidecar_schema",
    "validate_cadence_package_binding_contract",
    "verify_synthetic_entsoe_cadence_package",
]

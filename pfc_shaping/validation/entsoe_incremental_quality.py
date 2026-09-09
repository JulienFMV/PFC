"""Bounded incremental quality profiling for synthetic normalized ENTSO-E."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.validation import entsoe_parquet_streaming_integrity as d244

CONTRACT_SCHEMA = "fmv_entsoe_incremental_quality_profile_contract.v1"
CONTRACT_STATUS = (
    "STATIC_OFFLINE_SYNTHETIC_INCREMENTAL_QUALITY_NO_REAL_ADMISSION"
)
CONTRACT_FILE_SHA256 = (
    "15b9c24bb5b0cfefb27fef038497227baa926edb1433434ce0d0f7018f019c7c"
)
CONTRACT_CONTENT_ID = (
    "1baf72727872e0013fd6a70c1e04c2ef0eeabd0faf66659a7f17fb10956c4184"
)
D239_PROOF_CONTENT_ID = (
    "5e5aad7d04529e0efbb9926a1098a485ab3f797941ba2681c0a1609487f4df9b"
)
D239_CONTRACT_CONTENT_ID = (
    "7ff2d5b7808f636f2c181fa92b80cc4ae86dccdcaad84f6da8dc3d13499736ab"
)
D239_PROOF_MANIFEST_SHA256 = (
    "2d733c356b34ad0280802ed9d77b47f4d445e4d82e2ac6c45595f26f81787789"
)
D243_CONTRACT_CONTENT_ID = (
    "1e183fc51f2673cfc3fed0035dc4a5e3d84664f51ff8447a355264e5e819ddc6"
)
D243_CONTRACT_SHA256 = (
    "ba8f6945b4a43b54762fa475228edabea4018bfea5871f2581dc53536c0743a1"
)
D243_QUERY_SHA256 = (
    "16a989d2b1528f79b3ecb7a2d9f8f221a6be67cc1c4d3c622516c8bd0dde95e7"
)
D244_PROOF_CONTENT_ID = (
    "42c1065bf66117a2be2c792424f08d08511056d0df5f3b4b706ac57af1fcf564"
)
D244_PROOF_MANIFEST_SHA256 = (
    "53e4fe281d405fea64e6f6084d0ae129f608c4f6ff88e944d60be3bf904177d4"
)

CONTEXT_SCHEMA = "fmv_entsoe_incremental_quality_context.v1"
CONTEXT_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_class",
        "snapshot_id",
        "snapshot_reference_time_utc",
        "sign_semantics_by_series",
        "backfill_status_by_series",
        "authorities",
    }
)
CONTEXT_AUTHORITIES = {
    "real_artifact_receipt_admitted": False,
    "real_artifact_hashes_verified": False,
    "real_entsoe_values_opened": False,
    "real_series_inventory_admitted": False,
    "source_authenticity_verified": False,
    "same_snapshot_point_in_time_availability_proven": False,
    "training_authorized": False,
    "selection_authorized": False,
    "model_input_authorized": False,
    "candidate_assembly_authorized": False,
    "promotion_authorized": False,
    "production_authorized": False,
}
AUTHORITIES = {
    "synthetic_incremental_quality_profile_verified": True,
    **CONTEXT_AUTHORITIES,
    "seasonal_cross_market_diagnostic_authorized": False,
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = (
    REPOSITORY_ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
)
D239_PROOF_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "build"
    / "databricks-eex-daily"
    / "2026-08-05"
    / "entsoe-normalized-quality-profile-proofs"
    / D239_PROOF_CONTENT_ID
    / "manifest.json"
)
D243_CONTRACT_PATH = PHASE_ROOT / "ENTSOE-SERIES-INVENTORY-CONTRACT-V1.json"
D243_QUERY_PATH = (
    REPOSITORY_ROOT / "docs" / "data" / "sql" / "entsoe_dev_gold_series_inventory.sql"
)
D244_CONTRACT_PATH = (
    PHASE_ROOT / "ENTSOE-PARQUET-STREAMING-INTEGRITY-PREFLIGHT-CONTRACT-V1.json"
)
D244_PROOF_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "build"
    / "databricks-eex-daily"
    / "2026-08-06"
    / "entsoe-parquet-streaming-integrity-proofs"
    / D244_PROOF_CONTENT_ID
    / "manifest.json"
)
QUALITY_CONTEXT_ROOT = (
    REPOSITORY_ROOT
    / "build"
    / "governed-source-intake"
    / "synthetic-quality-contexts"
)
STAGING_ROOT = (
    REPOSITORY_ROOT
    / "build"
    / "governed-source-intake"
    / "incremental-quality-staging"
)

MAX_DATABASE_BYTES = 1_073_741_824
PAGE_SIZE_BYTES = 4_096
MAX_PAGE_COUNT = 262_144
CACHE_KIB = 8_192
MIN_SEASONAL_COMPLETE_DAYS = 730
NANOSECONDS_PER_SECOND = 1_000_000_000
NANOSECONDS_PER_DAY = 86_400 * NANOSECONDS_PER_SECOND
ROLE_GRAINS = {
    "series_dimension": ("series_id",),
    "latest_values": ("series_id", "target_ts_utc"),
    "vintage_values": ("series_id", "target_ts_utc", "as_of_utc"),
}
REQUIRED_GROUPS = (
    "actual_load",
    "day_ahead_prices",
    "generation_actual",
    "generation_forecast",
    "hydro_reservoir_storage",
    "load_forecast",
    "renewables_forecast_day_ahead",
    "renewables_forecast_intraday",
    "crossborder_physical_flows",
    "scheduled_exchanges",
    "ntc_day_ahead",
    "ntc_month_ahead",
    "ntc_year_ahead",
)
CROSSBORDER_GROUPS = frozenset(
    {
        "crossborder_physical_flows",
        "scheduled_exchanges",
        "ntc_day_ahead",
        "ntc_month_ahead",
        "ntc_year_ahead",
    }
)
PRE_TARGET_GROUPS = frozenset(
    {
        "day_ahead_prices",
        "generation_forecast",
        "load_forecast",
        "renewables_forecast_day_ahead",
        "renewables_forecast_intraday",
        "scheduled_exchanges",
        "ntc_day_ahead",
        "ntc_month_ahead",
        "ntc_year_ahead",
    }
)
POST_TARGET_GROUPS = frozenset(
    {
        "actual_load",
        "generation_actual",
        "hydro_reservoir_storage",
        "crossborder_physical_flows",
    }
)
REQUIRED_BORDERS = ("CH-DE", "CH-FR", "CH-IT", "CH-AT")
REQUIRED_GENERATION_FIELDS = frozenset(
    {
        "solar",
        "wind",
        "nuclear",
        "hydro_run_of_river",
        "hydro_reservoir",
        "hydro_pumped_storage",
    }
)
ALLOWED_UNITS = frozenset({"MW", "MWh", "EUR/MWh"})
RESOLUTION_SECONDS = {
    "PT15M": 900,
    "PT30M": 1_800,
    "PT60M": 3_600,
    "PT1H": 3_600,
    "P1D": 86_400,
}
SIGN_SEMANTICS = frozenset(
    {
        "NON_NEGATIVE_PHYSICAL_QUANTITY",
        "SIGNED_POSITIVE_CH_TO_NEIGHBOR",
        "SIGNED_POSITIVE_NEIGHBOR_TO_CH",
        "SIGNED_SOURCE_DEFINED_DOCUMENTED",
        "PRICE_CAN_BE_NEGATIVE",
    }
)
BACKFILL_STATUSES = frozenset(
    {"NATIVE_VINTAGES", "HISTORICAL_BACKFILL", "MIXED_NATIVE_AND_BACKFILL"}
)


class EntsoeIncrementalQualityError(ValueError):
    """Raised when incremental quality evidence is unsafe or ambiguous."""


@dataclass(frozen=True)
class EntsoeIncrementalQualityResult:
    """Value-free summary and per-series profile from a bounded scan."""

    summary: Mapping[str, object]
    series_profile: tuple[Mapping[str, object], ...]
    summary_content_id: str
    series_profile_content_id: str
    context_content_id: str
    d244_runtime_receipt_content_id: str
    sqlite_peak_bytes: int
    staging_cleaned: bool


def validate_incremental_quality_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact D245 offline contract."""

    contract = _mapping(document, "D245 contract")
    _equal(contract.get("schema_version"), CONTRACT_SCHEMA, "contract schema")
    _equal(contract.get("status"), CONTRACT_STATUS, "contract status")
    _equal(contract.get("decision"), "D-20260806-245", "contract decision")
    _equal(
        contract.get("input_bindings"),
        {
            "d239_quality_contract_content_id": D239_CONTRACT_CONTENT_ID,
            "d239_quality_proof_content_id": D239_PROOF_CONTENT_ID,
            "d239_quality_proof_manifest_sha256": D239_PROOF_MANIFEST_SHA256,
            "d243_inventory_contract_content_id": D243_CONTRACT_CONTENT_ID,
            "d243_inventory_contract_sha256": D243_CONTRACT_SHA256,
            "d243_inventory_query_sha256": D243_QUERY_SHA256,
            "d244_streaming_contract_content_id": d244.CONTRACT_CONTENT_ID,
            "d244_streaming_contract_sha256": d244.CONTRACT_FILE_SHA256,
            "d244_streaming_proof_content_id": D244_PROOF_CONTENT_ID,
            "d244_streaming_proof_manifest_sha256": D244_PROOF_MANIFEST_SHA256,
        },
        "contract input bindings",
    )
    context = _mapping(contract.get("quality_context"), "quality context policy")
    _equal(context.get("schema_version"), CONTEXT_SCHEMA, "context schema")
    _equal(context.get("evidence_class"), "SYNTHETIC_TEST_ONLY", "context evidence")
    _equal(set(context.get("exact_fields", [])), set(CONTEXT_FIELDS), "context fields")
    _equal(context.get("exact_series_mapping_required"), True, "mapping policy")
    streaming = _mapping(contract.get("streaming_policy"), "streaming policy")
    for field, expected in {
        "compose_exact_d244_package_verifier_first": True,
        "second_quality_pass_hash_before_and_after_required": True,
        "record_batch_rows": d244.RECORD_BATCH_ROWS,
        "pandas_import_authorized": False,
        "whole_table_or_whole_parquet_materialization_authorized": False,
        "raw_numeric_values_persisted_authorized": False,
        "bounded_to_d244_artifact_row_and_byte_caps": True,
    }.items():
        _equal(streaming.get(field), expected, f"streaming {field}")
    spill = _mapping(contract.get("spill_index_policy"), "spill policy")
    for field, expected in {
        "engine": "SQLITE_STANDARD_LIBRARY",
        "max_database_bytes": MAX_DATABASE_BYTES,
        "page_size_bytes": PAGE_SIZE_BYTES,
        "max_page_count": MAX_PAGE_COUNT,
        "cache_kib": CACHE_KIB,
        "journal_mode": "DELETE",
        "synchronous": "FULL",
        "temp_store": "FILE",
        "locking_mode": "EXCLUSIVE",
        "stored_numeric_market_values_authorized": False,
        "stored_structural_keys_and_value_hashes_authorized": True,
        "cleanup_required_on_success_and_failure": True,
        "residual_database_or_journal_authorized": False,
    }.items():
        _equal(spill.get(field), expected, f"spill {field}")
    checks = _mapping(contract.get("quality_checks"), "quality checks")
    _equal(
        checks.get("exact_grains"),
        {role: list(grain) for role, grain in ROLE_GRAINS.items()},
        "quality grains",
    )
    _equal(checks.get("minimum_seasonal_complete_utc_days"), 730, "seasonal days")
    _equal(checks.get("required_latest_to_vintage_overlap_coverage_ratio"), 1.0, "overlap requirement")
    output = _mapping(contract.get("output_policy"), "output policy")
    _equal(output.get("raw_market_values_persisted_authorized"), False, "raw output")
    _equal(output.get("series_identifiers_persisted_authorized"), False, "series output")
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
    _equal(execution.get("d243_inventory_execution_authorized"), False, "D243 execution authority")
    _equal(contract.get("authorities"), AUTHORITIES, "contract authorities")
    content_id = canonical_sha256(contract)
    _equal(content_id, CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": content_id,
        "bounded_sqlite_spill_implemented": True,
        "pandas_authorized": False,
        "real_artifact_admission_implemented": False,
        "databricks_connection_budget": 0,
        "production_authorized": False,
    }


def profile_synthetic_entsoe_incremental_quality(
    *,
    contract_path: str | Path,
    package_manifest_path: str | Path,
    quality_context_path: str | Path,
    reference_time_utc: datetime,
) -> EntsoeIncrementalQualityResult:
    """Profile one exact synthetic package with bounded Arrow and SQLite state."""

    _require_workspace_and_temp()
    contract_raw = _read_small(contract_path, "D245 contract", 100_000)
    _equal(hashlib.sha256(contract_raw).hexdigest(), CONTRACT_FILE_SHA256, "D245 contract hash")
    validate_incremental_quality_contract(_strict_json(contract_raw, "D245 contract"))
    _verify_bound_sources()
    reference = _utc_datetime(reference_time_utc, "reference_time_utc")
    d244_result = d244.verify_synthetic_entsoe_parquet_package(
        contract_path=D244_CONTRACT_PATH,
        package_manifest_path=package_manifest_path,
        reference_time_utc=reference,
    )
    manifest_path = assert_absolute_path_has_no_links(package_manifest_path)
    manifest = _strict_json(
        _read_small(manifest_path, "Parquet package manifest", 2_000_000),
        "Parquet package manifest",
    )
    context_path = assert_absolute_path_has_no_links(quality_context_path)
    context = _strict_json(
        _read_small(context_path, "incremental quality context", 2_000_000),
        "incremental quality context",
    )
    context_content_id = _validate_context(context_path, context, manifest, reference)
    snapshot_reference = _parse_utc(
        context["snapshot_reference_time_utc"],
        "context snapshot time",
    )

    staging_root = assert_absolute_path_has_no_links(STAGING_ROOT)
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = staging_root / str(uuid.uuid4())
    staging.mkdir(exist_ok=False)
    database_path = staging / "quality-index.sqlite"
    connection: sqlite3.Connection | None = None
    peak_bytes = 0
    cleaned = False
    counters = _empty_counters()
    try:
        connection = sqlite3.connect(database_path)
        _configure_database(connection)
        _create_tables(connection)
        declarations = {
            str(item["role"]): _mapping(item, "artifact declaration")
            for item in manifest["artifacts"]
        }
        dimension_cache: dict[str, dict[str, object]] = {}
        for role in d244.ROLE_ORDER:
            artifact = manifest_path.parent / d244.ROLE_FILE_NAMES[role]
            consumer = _batch_consumer(
                connection,
                role,
                counters,
                dimension_cache,
                context,
                snapshot_reference,
                database_path,
            )
            _visit_verified_batches(
                artifact,
                declarations[role],
                role,
                consumer,
            )
            peak_bytes = max(peak_bytes, _database_bytes(connection))
        summary, series_profile = _finalize_profile(
            connection,
            counters,
            dimension_cache,
            context,
            snapshot_reference,
            d244_result,
            context_content_id,
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
        raise EntsoeIncrementalQualityError("SQLite staging cleanup failed")
    summary = {**summary, "sqlite_staging_cleaned": True}
    return EntsoeIncrementalQualityResult(
        summary=summary,
        series_profile=tuple(series_profile),
        summary_content_id=canonical_sha256(summary),
        series_profile_content_id=canonical_sha256(series_profile),
        context_content_id=context_content_id,
        d244_runtime_receipt_content_id=d244_result.receipt_content_id,
        sqlite_peak_bytes=peak_bytes,
        staging_cleaned=True,
    )


def derive_quality_context_content_id(context: Mapping[str, object]) -> str:
    return canonical_sha256(_mapping(context, "quality context"))


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _batch_consumer(
    connection: sqlite3.Connection,
    role: str,
    counters: dict[str, int],
    dimension_cache: dict[str, dict[str, object]],
    context: Mapping[str, object],
    reference: datetime,
    database_path: Path,
):
    def consume(batch: pa.RecordBatch) -> None:
        if role == "series_dimension":
            _consume_dimension(batch, connection, counters, dimension_cache, context)
        else:
            _consume_fact(
                batch,
                role,
                connection,
                counters,
                dimension_cache,
                context,
                reference,
            )
        connection.commit()
        if _database_bytes(connection) > MAX_DATABASE_BYTES:
            raise EntsoeIncrementalQualityError("SQLite spill byte cap exceeded")
        if database_path.stat().st_size > MAX_DATABASE_BYTES:
            raise EntsoeIncrementalQualityError("SQLite file byte cap exceeded")

    return consume


def _consume_dimension(
    batch: pa.RecordBatch,
    connection: sqlite3.Connection,
    counters: dict[str, int],
    cache: dict[str, dict[str, object]],
    context: Mapping[str, object],
) -> None:
    columns = _batch_columns(batch)
    signs = _mapping(context["sign_semantics_by_series"], "sign semantics")
    backfills = _mapping(context["backfill_status_by_series"], "backfill status")
    for index in range(batch.num_rows):
        counters["dimension_rows_seen"] += 1
        series_id = str(columns["series_id"][index])
        row = {name: columns[name][index] for name in d244.ROLE_COLUMNS["series_dimension"]}
        sign = signs.get(series_id)
        backfill = backfills.get(series_id)
        if sign not in SIGN_SEMANTICS:
            counters["invalid_sign_mapping_count"] += 1
            sign = "INVALID"
        if backfill not in BACKFILL_STATUSES:
            counters["invalid_backfill_mapping_count"] += 1
            backfill = "INVALID"
        resolution = RESOLUTION_SECONDS.get(str(row["native_resolution"]))
        if resolution is None:
            counters["invalid_resolution_count"] += 1
            resolution = 1
        if str(row["unit"]) not in ALLOWED_UNITS:
            counters["invalid_unit_count"] += 1
        values = (
            series_id,
            str(row["group_name"]),
            str(row["field_name"]),
            str(row["from_zone"]),
            str(row["to_zone"]),
            str(row["unit"]),
            int(resolution),
            str(sign),
            str(backfill),
        )
        try:
            connection.execute(
                "INSERT INTO dimension VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
            cache[series_id] = {
                "group_name": values[1],
                "field_name": values[2],
                "from_zone": values[3],
                "to_zone": values[4],
                "unit": values[5],
                "resolution_seconds": values[6],
                "sign_semantics": values[7],
                "backfill_status": values[8],
            }
        except sqlite3.IntegrityError:
            counters["dimension_duplicate_key_count"] += 1


def _consume_fact(
    batch: pa.RecordBatch,
    role: str,
    connection: sqlite3.Connection,
    counters: dict[str, int],
    dimension: Mapping[str, Mapping[str, object]],
    context: Mapping[str, object],
    reference: datetime,
) -> None:
    columns = _batch_columns(batch)
    target_ns = _timestamp_ns(batch, "target_ts_utc")
    load_ns = _timestamp_ns(batch, "meta_load_timestamp_utc")
    as_of_ns = _timestamp_ns(batch, "as_of_utc") if role == "vintage_values" else None
    reference_ns = _datetime_ns(reference)
    table = "latest" if role == "latest_values" else "vintage"
    for index in range(batch.num_rows):
        counters[f"{table}_rows_seen"] += 1
        series_id = str(columns["series_id"][index])
        target = int(target_ns[index])
        loaded = int(load_ns[index])
        revision = int(columns["revision_number"][index])
        value = float(columns["value"][index])
        quality = str(columns["quality_flag"][index])
        fingerprint = _value_fingerprint(value, quality, revision, loaded)
        metadata = dimension.get(series_id)
        if metadata is None:
            counters[f"{table}_orphan_row_count"] += 1
            resolution_seconds = 1
        else:
            resolution_seconds = int(metadata["resolution_seconds"])
            if target % (resolution_seconds * NANOSECONDS_PER_SECOND):
                counters[f"{table}_grid_misaligned_row_count"] += 1
            semantics = str(metadata["sign_semantics"])
            if semantics == "NON_NEGATIVE_PHYSICAL_QUANTITY" and value < 0:
                counters[f"{table}_sign_violation_count"] += 1
        if loaded > reference_ns:
            counters[f"{table}_load_after_snapshot_count"] += 1
        target_day = target // NANOSECONDS_PER_DAY
        if role == "latest_values":
            try:
                connection.execute(
                    "INSERT INTO latest VALUES (?, ?, ?, ?, ?, ?)",
                    (series_id, target, target_day, fingerprint, revision, loaded),
                )
            except sqlite3.IntegrityError:
                counters["latest_duplicate_key_count"] += 1
        else:
            assert as_of_ns is not None
            as_of = int(as_of_ns[index])
            if as_of > loaded:
                counters["vintage_as_of_after_load_count"] += 1
            if as_of > reference_ns:
                counters["vintage_as_of_after_snapshot_count"] += 1
            if metadata is not None:
                group = str(metadata["group_name"])
                if group in PRE_TARGET_GROUPS and as_of > target:
                    counters["vintage_pre_target_availability_violation_count"] += 1
                if group in POST_TARGET_GROUPS and as_of < target:
                    counters["vintage_post_target_availability_violation_count"] += 1
            try:
                connection.execute(
                    "INSERT INTO vintage VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        series_id,
                        target,
                        as_of,
                        target_day,
                        fingerprint,
                        revision,
                        loaded,
                    ),
                )
            except sqlite3.IntegrityError:
                counters["vintage_duplicate_key_count"] += 1


def _finalize_profile(
    connection: sqlite3.Connection,
    counters: Mapping[str, int],
    dimension: Mapping[str, Mapping[str, object]],
    context: Mapping[str, object],
    reference: datetime,
    d244_result: d244.EntsoeParquetStreamingIntegrityResult,
    context_content_id: str,
    peak_bytes: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    dimension_ids = set(dimension)
    context_sign_ids = set(_mapping(context["sign_semantics_by_series"], "signs"))
    context_backfill_ids = set(_mapping(context["backfill_status_by_series"], "backfills"))
    exact_context_mapping = dimension_ids == context_sign_ids == context_backfill_ids
    latest_series = _single_column_set(connection, "SELECT DISTINCT series_id FROM latest")
    vintage_series = _single_column_set(connection, "SELECT DISTINCT series_id FROM vintage")
    latest_count = _scalar_int(connection, "SELECT COUNT(*) FROM latest")
    vintage_count = _scalar_int(connection, "SELECT COUNT(*) FROM vintage")
    overlap_count = _scalar_int(
        connection,
        """
        WITH ranked AS (
          SELECT series_id, target_ns, row_fingerprint,
                 ROW_NUMBER() OVER (
                   PARTITION BY series_id, target_ns ORDER BY as_of_ns DESC
                 ) AS rank
          FROM vintage
        )
        SELECT COUNT(*) FROM latest l
        JOIN ranked v ON v.series_id=l.series_id AND v.target_ns=l.target_ns
        WHERE v.rank=1
        """,
    )
    mismatch_count = _scalar_int(
        connection,
        """
        WITH ranked AS (
          SELECT series_id, target_ns, row_fingerprint,
                 ROW_NUMBER() OVER (
                   PARTITION BY series_id, target_ns ORDER BY as_of_ns DESC
                 ) AS rank
          FROM vintage
        )
        SELECT COUNT(*) FROM latest l
        JOIN ranked v ON v.series_id=l.series_id AND v.target_ns=l.target_ns
        WHERE v.rank=1 AND v.row_fingerprint<>l.row_fingerprint
        """,
    )
    chronology_violations = _scalar_int(
        connection,
        """
        WITH ordered AS (
          SELECT revision_number, load_ns,
                 LAG(revision_number) OVER (
                   PARTITION BY series_id, target_ns ORDER BY as_of_ns
                 ) AS prior_revision,
                 LAG(load_ns) OVER (
                   PARTITION BY series_id, target_ns ORDER BY as_of_ns
                 ) AS prior_load
          FROM vintage
        )
        SELECT COUNT(*) FROM ordered
        WHERE (prior_revision IS NOT NULL AND revision_number<prior_revision)
           OR (prior_load IS NOT NULL AND load_ns<prior_load)
        """,
    )
    required_groups = set(REQUIRED_GROUPS)
    observed_groups = {str(value["group_name"]) for value in dimension.values()}
    groups_present = required_groups <= observed_groups
    generation_fields = {
        str(value["field_name"])
        for value in dimension.values()
        if value["group_name"] in {"generation_actual", "generation_forecast"}
    }
    generation_present = set(REQUIRED_GENERATION_FIELDS) <= generation_fields
    borders_present = _required_borders_present(dimension)
    series_profile = _series_profiles(connection, dimension)
    minimum_complete_days = min(
        (int(item["complete_utc_days"]) for item in series_profile),
        default=0,
    )
    minimum_grid_ratio = min(
        (float(item["grid_completeness_ratio"]) for item in series_profile),
        default=0.0,
    )
    overlap_ratio = overlap_count / latest_count if latest_count else 0.0
    dimension_count = len(dimension_ids)
    latest_join_coverage = (
        (latest_count - counters["latest_orphan_row_count"]) / latest_count
        if latest_count
        else 0.0
    )
    vintage_join_coverage = (
        (vintage_count - counters["vintage_orphan_row_count"]) / vintage_count
        if vintage_count
        else 0.0
    )
    latest_series_coverage = len(latest_series & dimension_ids) / dimension_count if dimension_count else 0.0
    vintage_series_coverage = len(vintage_series & dimension_ids) / dimension_count if dimension_count else 0.0
    hard_failure_counts = {
        key: int(value)
        for key, value in counters.items()
        if key.endswith("_count") and key not in {"backfill_series_count"}
    }
    hard_failure_counts["vintage_chronology_violation_count"] = chronology_violations
    hard_failure_counts["latest_last_vintage_mismatch_count"] = mismatch_count
    hard_failure_counts["context_mapping_mismatch_count"] = 0 if exact_context_mapping else 1
    hard_failure_counts["missing_required_group_count"] = len(required_groups - observed_groups)
    hard_failure_counts["missing_required_generation_field_count"] = len(
        set(REQUIRED_GENERATION_FIELDS) - generation_fields
    )
    hard_failure_counts["missing_crossborder_group_border_pair_count"] = borders_present[1]
    hard_failure_counts["dimension_missing_latest_series_count"] = len(
        dimension_ids - latest_series
    )
    hard_failure_counts["dimension_missing_vintage_series_count"] = len(
        dimension_ids - vintage_series
    )
    hard_pass = all(value == 0 for value in hard_failure_counts.values())
    hard_pass = hard_pass and latest_join_coverage == 1.0 and vintage_join_coverage == 1.0
    hard_pass = hard_pass and latest_series_coverage == 1.0 and vintage_series_coverage == 1.0
    hard_pass = hard_pass and overlap_ratio == 1.0 and groups_present and generation_present and borders_present[0]
    backfill_count = sum(
        value["backfill_status"] != "NATIVE_VINTAGES"
        for value in dimension.values()
    )
    findings = _findings(
        hard_failure_counts,
        minimum_complete_days,
        minimum_grid_ratio,
        overlap_ratio,
        backfill_count,
        context,
    )
    summary = {
        "schema_version": "fmv_entsoe_incremental_quality_profile.v1",
        "status": (
            "PASS_SYNTHETIC_INCREMENTAL_QUALITY_MODEL_NO_GO"
            if hard_pass
            else "FAIL_SYNTHETIC_INCREMENTAL_QUALITY_MODEL_NO_GO"
        ),
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "snapshot_reference_time_utc": _format_utc(reference),
        "source_bindings": {
            "contract_content_id": CONTRACT_CONTENT_ID,
            "quality_context_content_id": context_content_id,
            "d244_runtime_receipt_content_id": d244_result.receipt_content_id,
            "d244_selected_proof_content_id": D244_PROOF_CONTENT_ID,
            "d243_inventory_contract_content_id": D243_CONTRACT_CONTENT_ID,
            "d243_inventory_executed": False,
        },
        "dataset": {
            "dimension_row_count": _scalar_int(connection, "SELECT COUNT(*) FROM dimension"),
            "latest_row_count": latest_count,
            "vintage_row_count": vintage_count,
            "series_count": dimension_count,
            "latest_vintage_overlap_key_count": overlap_count,
            "record_batch_count": d244_result.total_record_batches,
        },
        "checks": {
            "d244_integrity_passed": True,
            "exact_context_series_mapping": exact_context_mapping,
            "primary_keys_unique": all(
                counters[key] == 0
                for key in (
                    "dimension_duplicate_key_count",
                    "latest_duplicate_key_count",
                    "vintage_duplicate_key_count",
                )
            ),
            "dimension_join_coverage": min(latest_join_coverage, vintage_join_coverage),
            "dimension_to_latest_series_coverage": latest_series_coverage,
            "dimension_to_vintage_series_coverage": vintage_series_coverage,
            "required_groups_present": groups_present,
            "required_borders_by_crossborder_group_present": borders_present[0],
            "required_generation_breakdown_present": generation_present,
            "native_grid_alignment_valid": (
                counters["latest_grid_misaligned_row_count"] == 0
                and counters["vintage_grid_misaligned_row_count"] == 0
            ),
            "revision_and_load_chronology_valid": chronology_violations == 0,
            "latest_vintage_overlap_mismatch_count": mismatch_count,
            "seasonal_complete_day_threshold_met": (
                minimum_complete_days >= MIN_SEASONAL_COMPLETE_DAYS
            ),
            "sqlite_stored_raw_numeric_values": False,
            "whole_table_materialized": False,
            "pandas_imported": False,
        },
        "rates": {
            "latest_duplicate_key_rate": _rate(counters["latest_duplicate_key_count"], counters["latest_rows_seen"]),
            "vintage_duplicate_key_rate": _rate(counters["vintage_duplicate_key_count"], counters["vintage_rows_seen"]),
            "latest_dimension_join_coverage_ratio": latest_join_coverage,
            "vintage_dimension_join_coverage_ratio": vintage_join_coverage,
            "dimension_to_latest_series_coverage_ratio": latest_series_coverage,
            "dimension_to_vintage_series_coverage_ratio": vintage_series_coverage,
            "latest_to_vintage_overlap_coverage_ratio": overlap_ratio,
            "minimum_grid_completeness_ratio": minimum_grid_ratio,
        },
        "coverage": {
            "minimum_complete_utc_days": minimum_complete_days,
            "required_complete_utc_days": MIN_SEASONAL_COMPLETE_DAYS,
            "backfill_or_mixed_series_count": backfill_count,
        },
        "failure_counts": hard_failure_counts,
        "findings": findings,
        "sqlite": {
            "engine": sqlite3.sqlite_version,
            "peak_database_bytes": peak_bytes,
            "maximum_database_bytes": MAX_DATABASE_BYTES,
            "raw_numeric_values_stored": False,
            "structural_keys_and_value_hashes_only": True,
        },
        "authorities": dict(AUTHORITIES),
        "execution": {
            "databricks_connection_count": 0,
            "databricks_statement_count": 0,
            "warehouse_start_count": 0,
            "network_call_count": 0,
            "h_drive_access_count": 0,
            "remote_write_count": 0,
        },
    }
    return summary, series_profile


def _series_profiles(
    connection: sqlite3.Connection,
    dimension: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    complete_days = {
        (str(row[0]), int(row[1])): int(row[2])
        for row in connection.execute(
            "SELECT series_id, target_day, COUNT(*) FROM latest GROUP BY series_id, target_day"
        )
    }
    vintage_stats = {
        str(row[0]): (int(row[1]), int(row[2]))
        for row in connection.execute(
            "SELECT series_id, COUNT(*), MIN(as_of_ns) FROM vintage GROUP BY series_id"
        )
    }
    overlap = {
        str(row[0]): int(row[1])
        for row in connection.execute(
            """
            WITH ranked AS (
              SELECT series_id, target_ns,
                     ROW_NUMBER() OVER (
                       PARTITION BY series_id, target_ns ORDER BY as_of_ns DESC
                     ) AS rank
              FROM vintage
            )
            SELECT l.series_id, COUNT(*) FROM latest l
            JOIN ranked v ON v.series_id=l.series_id AND v.target_ns=l.target_ns
            WHERE v.rank=1 GROUP BY l.series_id
            """
        )
    }
    rows = list(
        connection.execute(
            "SELECT series_id, COUNT(*), MIN(target_ns), MAX(target_ns) FROM latest GROUP BY series_id"
        )
    )
    result: list[dict[str, object]] = []
    for series_id, observed, minimum, maximum in rows:
        metadata = dimension.get(str(series_id))
        if metadata is None:
            # Orphan rows are already counted as a hard failure. Excluding them
            # here keeps the value-free profile structurally safe and lets the
            # caller receive the exact diagnostic counts instead of a KeyError.
            continue
        resolution = int(metadata["resolution_seconds"])
        step = resolution * NANOSECONDS_PER_SECOND
        expected = ((int(maximum) - int(minimum)) // step) + 1
        complete = sum(
            count == 86_400 // resolution
            for (candidate, _), count in complete_days.items()
            if candidate == series_id
        )
        vintage_rows, earliest_as_of = vintage_stats.get(str(series_id), (0, 0))
        result.append(
            {
                "schema_version": "fmv_entsoe_incremental_series_profile.v1",
                "series_id_sha256": hashlib.sha256(str(series_id).encode()).hexdigest(),
                "group_name": metadata["group_name"],
                "field_name": metadata["field_name"],
                "unit": metadata["unit"],
                "resolution_seconds": resolution,
                "observed_latest_slots": int(observed),
                "expected_latest_slots": int(expected),
                "missing_latest_slots": max(0, int(expected - int(observed))),
                "grid_completeness_ratio": float(int(observed) / expected),
                "complete_utc_days": int(complete),
                "vintage_row_count": int(vintage_rows),
                "latest_vintage_overlap_key_count": int(overlap.get(str(series_id), 0)),
                "earliest_as_of_utc": _ns_to_utc(earliest_as_of) if earliest_as_of else None,
                "backfill_status": metadata["backfill_status"],
            }
        )
    return sorted(result, key=lambda item: str(item["series_id_sha256"]))


def _findings(
    failures: Mapping[str, int],
    minimum_complete_days: int,
    minimum_grid_ratio: float,
    overlap_ratio: float,
    backfill_count: int,
    context: Mapping[str, object],
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for code, count in sorted(failures.items()):
        if count:
            findings.append(
                {
                    "code": code.upper(),
                    "severity": "CRITICAL",
                    "evidence": {"affected_count": int(count)},
                    "impact": "The normalized dataset cannot pass the incremental quality gate.",
                }
            )
    findings.extend(
        [
            {
                "code": "SYNTHETIC_FIXTURE_NOT_EMPIRICAL_EVIDENCE",
                "severity": "CRITICAL",
                "evidence": {"evidence_class": context["evidence_class"]},
                "impact": "The profile cannot support model choice or validation.",
            },
            {
                "code": "D241_REAL_SCHEMA_MAPPING_REMAINS_INCOMPATIBLE",
                "severity": "CRITICAL",
                "evidence": {"real_normalized_artifact_count": 0},
                "impact": "Current raw Unity Catalog tables cannot enter this normalized gate.",
            },
            {
                "code": "D243_REAL_SERIES_INVENTORY_NOT_EXECUTED",
                "severity": "HIGH",
                "evidence": {"real_series_inventory_capture_count": 0},
                "impact": "Required real family coverage remains unknown.",
            },
            {
                "code": "SEASONAL_COMPLETE_DAY_THRESHOLD_NOT_MET",
                "severity": "HIGH",
                "evidence": {
                    "required_complete_days": MIN_SEASONAL_COMPLETE_DAYS,
                    "minimum_observed_complete_days": minimum_complete_days,
                },
                "impact": "Seasonal diagnostics and selection remain unsupported.",
            },
            {
                "code": "NEW_INDEPENDENT_FUTURE_HOLDOUT_MISSING",
                "severity": "CRITICAL",
                "evidence": {"prospective_holdout_count": 0},
                "impact": "Training, selection and production remain forbidden.",
            },
        ]
    )
    if overlap_ratio < 1.0:
        findings.append(
            {
                "code": "LATEST_TO_VINTAGE_OVERLAP_INCOMPLETE",
                "severity": "HIGH",
                "evidence": {"overlap_coverage_ratio": overlap_ratio},
                "impact": "Historical information sets cannot be reconstructed completely.",
            }
        )
    if minimum_grid_ratio < 1.0:
        findings.append(
            {
                "code": "NATIVE_GRID_COMPLETENESS_GAPS_PRESENT",
                "severity": "HIGH",
                "evidence": {
                    "minimum_grid_completeness_ratio": minimum_grid_ratio
                },
                "impact": "Shape and seasonal statistics may be biased by missing slots.",
            }
        )
    if backfill_count:
        findings.append(
            {
                "code": "BACKFILLED_SERIES_NOT_RETROACTIVE_PIT_EVIDENCE",
                "severity": "HIGH",
                "evidence": {"series_count": backfill_count},
                "impact": "Backfills cannot reconstruct historical information sets.",
            }
        )
    return findings


def _required_borders_present(
    dimension: Mapping[str, Mapping[str, object]],
) -> tuple[bool, int]:
    observed: dict[str, set[str]] = {group: set() for group in CROSSBORDER_GROUPS}
    for metadata in dimension.values():
        group = str(metadata["group_name"])
        if group not in observed:
            continue
        from_zone = str(metadata["from_zone"])
        to_zone = str(metadata["to_zone"])
        zones = {from_zone, to_zone}
        if "CH" in zones and len(zones) == 2:
            neighbor = next(iter(zones - {"CH"}))
            observed[group].add(f"CH-{neighbor}")
    missing = sum(len(set(REQUIRED_BORDERS) - borders) for borders in observed.values())
    return missing == 0, missing


def _visit_verified_batches(
    path: Path,
    declaration: Mapping[str, object],
    role: str,
    consumer,
) -> None:
    lexical = assert_absolute_path_has_no_links(path)
    before = lexical.stat(follow_symlinks=False)
    if not d244._is_single_link_regular(before):
        raise EntsoeIncrementalQualityError(f"quality artifact is not single-link: {role}")
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor: int | None = None
    try:
        descriptor = os.open(lexical, flags)
        opened = os.fstat(descriptor)
        if d244._stable_identity(opened) != d244._stable_identity(before):
            raise EntsoeIncrementalQualityError(f"quality artifact identity changed: {role}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            digest_before, size_before = d244._hash_stream(stream)
            _equal(digest_before, declaration["sha256"], f"quality-pass hash {role}")
            _equal(size_before, declaration["size_bytes"], f"quality-pass size {role}")
            stream.seek(0)
            parquet = pq.ParquetFile(
                stream,
                thrift_string_size_limit=d244.THRIFT_STRING_SIZE_LIMIT,
                thrift_container_size_limit=d244.THRIFT_CONTAINER_SIZE_LIMIT,
                page_checksum_verification=True,
            )
            for batch in parquet.iter_batches(
                batch_size=d244.RECORD_BATCH_ROWS,
                use_threads=False,
            ):
                _equal(batch.schema, d244.expected_arrow_schema(role), f"quality batch schema {role}")
                d244._validate_batch_values(batch, role)
                consumer(batch)
            stream.seek(0)
            digest_after, size_after = d244._hash_stream(stream)
            _equal(digest_after, digest_before, f"quality post-scan hash {role}")
            _equal(size_after, size_before, f"quality post-scan size {role}")
            if d244._stable_identity(os.fstat(stream.fileno())) != d244._stable_identity(opened):
                raise EntsoeIncrementalQualityError(f"quality descriptor changed: {role}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
    final = lexical.stat(follow_symlinks=False)
    if d244._stable_identity(final) != d244._stable_identity(before):
        raise EntsoeIncrementalQualityError(f"quality path changed: {role}")


def _validate_context(
    path: Path,
    context: Mapping[str, object],
    manifest: Mapping[str, object],
    reference: datetime,
) -> str:
    _exact_fields(context, CONTEXT_FIELDS, "quality context")
    _equal(context.get("schema_version"), CONTEXT_SCHEMA, "context schema")
    _equal(context.get("evidence_class"), "SYNTHETIC_TEST_ONLY", "context evidence")
    _equal(context.get("snapshot_id"), manifest.get("snapshot_id"), "context snapshot")
    _equal(
        context.get("snapshot_reference_time_utc"),
        manifest.get("snapshot_reference_time_utc"),
        "context snapshot time",
    )
    snapshot = _parse_utc(context.get("snapshot_reference_time_utc"), "context snapshot time")
    if snapshot > reference:
        raise EntsoeIncrementalQualityError("context snapshot is after reference")
    _equal(context.get("authorities"), CONTEXT_AUTHORITIES, "context authorities")
    content_id = derive_quality_context_content_id(context)
    root = assert_absolute_path_has_no_links(QUALITY_CONTEXT_ROOT)
    if path.name != "context.json" or path.parent.parent != root or path.parent.name != content_id:
        raise EntsoeIncrementalQualityError("quality context content-addressed path differs")
    return content_id


def _verify_bound_sources() -> None:
    sources = (
        (D239_PROOF_MANIFEST_PATH, D239_PROOF_MANIFEST_SHA256, "D239 proof"),
        (D243_CONTRACT_PATH, D243_CONTRACT_SHA256, "D243 contract"),
        (D243_QUERY_PATH, D243_QUERY_SHA256, "D243 query"),
        (D244_PROOF_MANIFEST_PATH, D244_PROOF_MANIFEST_SHA256, "D244 proof"),
    )
    for path, expected, label in sources:
        raw = _read_small(path, label, 2_000_000)
        _equal(hashlib.sha256(raw).hexdigest(), expected, f"{label} hash")
    d239_proof = _strict_json(
        _read_small(D239_PROOF_MANIFEST_PATH, "D239 proof", 2_000_000),
        "D239 proof",
    )
    _equal(d239_proof.get("content_id"), D239_PROOF_CONTENT_ID, "D239 proof content ID")
    d244_proof = _strict_json(
        _read_small(D244_PROOF_MANIFEST_PATH, "D244 proof", 2_000_000),
        "D244 proof",
    )
    _equal(d244_proof.get("content_id"), D244_PROOF_CONTENT_ID, "D244 proof content ID")
    if any(value != 0 for value in _mapping(d244_proof.get("execution"), "D244 execution").values()):
        raise EntsoeIncrementalQualityError("D244 execution counters are non-zero")


def _configure_database(connection: sqlite3.Connection) -> None:
    settings = (
        "PRAGMA page_size=4096",
        "PRAGMA max_page_count=262144",
        "PRAGMA cache_size=-8192",
        "PRAGMA journal_mode=DELETE",
        "PRAGMA synchronous=FULL",
        "PRAGMA temp_store=FILE",
        "PRAGMA locking_mode=EXCLUSIVE",
        "PRAGMA foreign_keys=OFF",
    )
    for statement in settings:
        connection.execute(statement).fetchall()


def _create_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE dimension (
          series_id TEXT PRIMARY KEY,
          group_name TEXT NOT NULL,
          field_name TEXT NOT NULL,
          from_zone TEXT NOT NULL,
          to_zone TEXT NOT NULL,
          unit TEXT NOT NULL,
          resolution_seconds INTEGER NOT NULL,
          sign_semantics TEXT NOT NULL,
          backfill_status TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE latest (
          series_id TEXT NOT NULL,
          target_ns INTEGER NOT NULL,
          target_day INTEGER NOT NULL,
          row_fingerprint BLOB NOT NULL,
          revision_number INTEGER NOT NULL,
          load_ns INTEGER NOT NULL,
          PRIMARY KEY (series_id, target_ns)
        ) WITHOUT ROWID;
        CREATE TABLE vintage (
          series_id TEXT NOT NULL,
          target_ns INTEGER NOT NULL,
          as_of_ns INTEGER NOT NULL,
          target_day INTEGER NOT NULL,
          row_fingerprint BLOB NOT NULL,
          revision_number INTEGER NOT NULL,
          load_ns INTEGER NOT NULL,
          PRIMARY KEY (series_id, target_ns, as_of_ns)
        ) WITHOUT ROWID;
        CREATE INDEX vintage_target_index
          ON vintage(series_id, target_ns, as_of_ns DESC);
        """
    )
    connection.commit()


def _empty_counters() -> dict[str, int]:
    names = (
        "dimension_rows_seen",
        "latest_rows_seen",
        "vintage_rows_seen",
        "dimension_duplicate_key_count",
        "latest_duplicate_key_count",
        "vintage_duplicate_key_count",
        "latest_orphan_row_count",
        "vintage_orphan_row_count",
        "latest_grid_misaligned_row_count",
        "vintage_grid_misaligned_row_count",
        "latest_sign_violation_count",
        "vintage_sign_violation_count",
        "latest_load_after_snapshot_count",
        "vintage_load_after_snapshot_count",
        "vintage_as_of_after_load_count",
        "vintage_as_of_after_snapshot_count",
        "vintage_pre_target_availability_violation_count",
        "vintage_post_target_availability_violation_count",
        "invalid_sign_mapping_count",
        "invalid_backfill_mapping_count",
        "invalid_resolution_count",
        "invalid_unit_count",
    )
    return {name: 0 for name in names}


def _batch_columns(batch: pa.RecordBatch) -> dict[str, list[object]]:
    return {
        field.name: batch.column(index).to_pylist()
        for index, field in enumerate(batch.schema)
    }


def _timestamp_ns(batch: pa.RecordBatch, column: str) -> list[int]:
    index = batch.schema.get_field_index(column)
    return batch.column(index).cast(pa.int64()).to_pylist()


def _value_fingerprint(
    value: float,
    quality_flag: str,
    revision_number: int,
    load_ns: int,
) -> bytes:
    quality = quality_flag.encode("utf-8")
    payload = (
        struct.pack(">d", value)
        + len(quality).to_bytes(4, "big")
        + quality
        + int(revision_number).to_bytes(8, "big", signed=True)
        + int(load_ns).to_bytes(8, "big", signed=True)
    )
    return hashlib.sha256(payload).digest()


def _database_bytes(connection: sqlite3.Connection) -> int:
    page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
    page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
    if page_count > MAX_PAGE_COUNT or page_size != PAGE_SIZE_BYTES:
        raise EntsoeIncrementalQualityError("SQLite page bound differs")
    return page_count * page_size


def _single_column_set(connection: sqlite3.Connection, statement: str) -> set[str]:
    return {str(row[0]) for row in connection.execute(statement)}


def _scalar_int(connection: sqlite3.Connection, statement: str) -> int:
    value = connection.execute(statement).fetchone()
    return int(value[0])


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _datetime_ns(value: datetime) -> int:
    utc = _utc_datetime(value, "datetime")
    return int(utc.timestamp()) * NANOSECONDS_PER_SECOND + utc.microsecond * 1_000


def _ns_to_utc(value: int) -> str:
    seconds, nanoseconds = divmod(int(value), NANOSECONDS_PER_SECOND)
    base = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return base.strftime("%Y-%m-%dT%H:%M:%S") + f".{nanoseconds:09d}Z"


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise EntsoeIncrementalQualityError(f"{label} must be UTC Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise EntsoeIncrementalQualityError(f"{label} is invalid") from exc
    return _utc_datetime(parsed, label)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_datetime(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise EntsoeIncrementalQualityError(f"{label} must be aware UTC")
    return value.astimezone(timezone.utc)


def _require_workspace_and_temp() -> None:
    if Path.cwd().resolve() != REPOSITORY_ROOT:
        raise EntsoeIncrementalQualityError("cwd must be the canonical workspace")
    build = (REPOSITORY_ROOT / "build").resolve()
    for name in ("TEMP", "TMP"):
        raw = os.environ.get(name)
        if raw is None:
            raise EntsoeIncrementalQualityError(f"{name} must be repo-local")
        path = assert_absolute_path_has_no_links(Path(raw).resolve())
        try:
            path.relative_to(build)
        except ValueError as exc:
            raise EntsoeIncrementalQualityError(f"{name} must be below build") from exc


def _read_small(path: str | Path, label: str, max_bytes: int) -> bytes:
    try:
        return read_stable_single_link_file(path, label=label, max_bytes=max_bytes)
    except ValueError as exc:
        raise EntsoeIncrementalQualityError(str(exc)) from exc


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise EntsoeIncrementalQualityError(f"{label} duplicate key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise EntsoeIncrementalQualityError(
            f"{label} contains non-finite constant: {value}"
        )

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeIncrementalQualityError(f"{label} is not strict JSON") from exc
    return _mapping(value, label)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeIncrementalQualityError(f"{label} must be an object")
    return value


def _exact_fields(value: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    if set(value) != set(expected):
        raise EntsoeIncrementalQualityError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoeIncrementalQualityError(
            f"{label} differs: expected {expected!r}, received {value!r}"
        )


__all__ = [
    "AUTHORITIES",
    "CONTEXT_AUTHORITIES",
    "CONTRACT_CONTENT_ID",
    "EntsoeIncrementalQualityError",
    "EntsoeIncrementalQualityResult",
    "QUALITY_CONTEXT_ROOT",
    "STAGING_ROOT",
    "canonical_sha256",
    "derive_quality_context_content_id",
    "profile_synthetic_entsoe_incremental_quality",
    "validate_incremental_quality_contract",
]

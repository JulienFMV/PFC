"""Validate the superseded D293 Gold-only export package.

This module is retained only to replay historical D293 evidence. New LT
exports must follow ``docs/data/DATABRICKS-LT-SNAPSHOT-INTAKE.md`` and the
mixed Gold-serving / ENTSO-E-Silver-PIT acceptance contract. In particular,
this legacy validator must not be used to request or admit the retired Gold
ENTSO-E vintage fact.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = PHASE / "DATABRICKS-GOLD-SNAPSHOT-INTAKE-CONTRACT-V1.json"
REQUEST_PATH = PHASE / "DATABRICKS-SPOT-WEATHER-SWISSGRID-DATA-ENGINEER-REQUEST-20260806.md"
ENTSOE_REQUEST_PATH = PHASE / "ENTSOE-REAL-MAPPING-DATA-ENGINEER-REQUEST-20260806.md"
COST_CONTRACT_PATH = PHASE / "DATABRICKS-PFC-SOURCE-COST-PREFLIGHT-CONTRACT-V1.json"
EXPORT_ROOT = ROOT / "build" / "databricks-exports"

CONTRACT_SCHEMA = "fmv_databricks_gold_snapshot_intake_contract.v1"
CONTRACT_STATUS = "LOCAL_GOLD_EXPORT_INTEGRITY_GATE_READY_NO_REAL_EXPORT"
CONTRACT_FILE_SHA256 = "663b9cbf5dec1de0ff5071ae7876cf6f5849677c7eef8f95ff639466f6298ba9"
CONTRACT_CONTENT_ID = "1ed6b11e19fee7a31be976699230906eaac73d9f344e4cac17191b5623282869"
REQUEST_SHA256 = "1ab1f396fb37455d6b91f1bd2c374870dc3910214d894294e4e8c81397056b6b"
ENTSOE_REQUEST_SHA256 = "65116313a73fca108d1e8e1a406119513a689f5f14c65772764ab58371d89828"
COST_CONTRACT_SHA256 = "0beac8ef83e31f039b7e862987785694889ec5f946b082342931cfa4bf1a768c"
COST_CONTRACT_CONTENT_ID = "4d8857e09eb4fd2acadcd51faa9110a3e406cd1a04afea439d7f10c6e9da8771"

MANIFEST_SCHEMA = "fmv_databricks_gold_snapshot_manifest.v1"
EVIDENCE_CLASS = "GOVERNED_DATABRICKS_GOLD_EXPORT_CANDIDATE"
RESULT_SCHEMA = "fmv_databricks_gold_snapshot_integrity_result.v1"
RESULT_STATUS = "LOCAL_GOLD_EXPORT_INTEGRITY_VERIFIED_NO_DATA_AUTHORITY"
SUPERSEDED_BY = "docs/data/DATABRICKS-LT-SNAPSHOT-INTAKE.md"
MAX_MANIFEST_BYTES = 2_097_152
MAX_COMPANION_BYTES = 1_048_576
MAX_FUTURE_SKEW = timedelta(minutes=5)

ARTIFACT_SPECS = (
    (
        "spot_product_dimension",
        ("prd.gold.dimspotproduct",),
        "spot_product_dimension.parquet",
        False,
    ),
    (
        "spot_price_interval",
        ("prd.gold.factspotpriceinterval",),
        "spot_price_interval.parquet",
        False,
    ),
    (
        "weather_measurement",
        ("prd.gold.factweather",),
        "weather_measurement.parquet",
        False,
    ),
    (
        "weather_forecast_meteoswiss_vintages",
        ("prd.gold.factweatherforecasthistms",),
        "weather_forecast_meteoswiss_vintages.parquet",
        False,
    ),
    (
        "weather_forecast_openmeteo_vintages",
        ("prd.gold.factweatherforecasthistom",),
        "weather_forecast_openmeteo_vintages.parquet",
        False,
    ),
    (
        "swissgrid_balancing",
        ("prd.gold.factswissgridbalancingquarterhourly",),
        "swissgrid_balancing.parquet",
        False,
    ),
    (
        "swissgrid_tender",
        ("prd.gold.factswissgridtenderofferresult",),
        "swissgrid_tender.parquet",
        False,
    ),
    (
        "entsoe_series_dimension",
        ("prd.gold.dimentsoeseries",),
        "entsoe_series_dimension.parquet",
        False,
    ),
    (
        "entsoe_series_resolution_history",
        ("prd.gold.dimentsoeseriesresolutionhistory",),
        "entsoe_series_resolution_history.parquet",
        False,
    ),
    (
        "entsoe_zone_history",
        ("prd.gold.dimentsoezonehistory",),
        "entsoe_zone_history.parquet",
        False,
    ),
    (
        "entsoe_latest",
        ("prd.gold.factentsoetimeserieslatest",),
        "entsoe_latest.parquet",
        False,
    ),
    (
        "entsoe_vintages",
        ("prd.gold.factentsoetimeseriesvintages",),
        "entsoe_vintages.parquet",
        False,
    ),
    (
        "entsoe_series_quality",
        (
            "prd.gold.dimentsoeseries",
            "prd.gold.dimentsoeseriesresolutionhistory",
            "prd.gold.factentsoetimeserieslatest",
            "prd.gold.factentsoetimeseriesvintages",
        ),
        "entsoe_series_quality.parquet",
        False,
    ),
    (
        "entsoe_gap_report",
        (
            "prd.gold.dimentsoeseriesresolutionhistory",
            "prd.gold.factentsoetimeserieslatest",
            "prd.gold.factentsoetimeseriesvintages",
        ),
        "entsoe_gap_report.parquet",
        True,
    ),
    (
        "entsoe_source_reconciliation",
        (
            "prd.gold.dimentsoeseries",
            "prd.gold.factentsoetimeserieslatest",
            "prd.gold.factentsoetimeseriesvintages",
        ),
        "entsoe_source_reconciliation.parquet",
        False,
    ),
    (
        "entsoe_excluded_series",
        (
            "prd.gold.dimentsoeseries",
            "prd.gold.factentsoetimeserieslatest",
            "prd.gold.factentsoetimeseriesvintages",
        ),
        "entsoe_excluded_series.parquet",
        True,
    ),
)

COMPANION_SPECS = (
    ("cost_receipt", "cost_receipt.json", "application/json"),
    ("source_semantics", "source_semantics.md", "text/markdown"),
    ("entsoe_quality_summary", "entsoe_quality_summary.json", "application/json"),
    ("entsoe_family_inventory", "entsoe_family_inventory.json", "application/json"),
)

MANIFEST_FIELDS = {
    "schema_version",
    "evidence_class",
    "snapshot_id",
    "export_batch_id",
    "created_at_utc",
    "source_catalog",
    "source_schema",
    "request_sha256",
    "entsoe_request_sha256",
    "cost_preflight_contract_content_id",
    "total_size_bytes",
    "companions",
    "artifacts",
    "execution",
    "authorities",
}
COMPANION_FIELDS = {"role", "relative_path", "media_type", "sha256", "size_bytes"}
ARTIFACT_FIELDS = {
    "role",
    "source_tables",
    "format",
    "relative_path",
    "sha256",
    "size_bytes",
    "row_count",
    "column_count",
    "row_group_count",
    "schema_sha256",
}
MANIFEST_EXECUTION = {
    "databricks_write_count": 0,
    "source_table_mutation_count": 0,
}
MANIFEST_AUTHORITIES = {
    "source_authenticity_verified": False,
    "cost_approval_verified_by_local_validator": False,
    "semantic_quality_verified": False,
    "content_quality_verified": False,
    "point_in_time_authorized": False,
    "model_input_authorized": False,
    "model_selection_authorized": False,
    "candidate_authorized": False,
    "promotion_authorized": False,
    "production_authorized": False,
}
LOCAL_EXECUTION = {
    "databricks_connection_count": 0,
    "databricks_statement_count": 0,
    "business_value_row_count_opened": 0,
    "warehouse_start_count": 0,
    "databricks_write_count": 0,
    "network_call_count": 0,
    "h_drive_access_count": 0,
    "remote_write_count": 0,
}
RESULT_AUTHORITIES = {
    "local_byte_integrity": True,
    "parquet_footer_integrity": True,
    "real_export_received": True,
    "source_authenticity": False,
    "cost_approval": False,
    "semantic_quality": False,
    "content_quality": False,
    "point_in_time": False,
    "predictive_value": False,
    "model_input": False,
    "model_selection": False,
    "candidate": False,
    "ompex_superiority": False,
    "promotion": False,
    "production": False,
}

_SHA = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class DatabricksGoldSnapshotIntakeError(ValueError):
    """Raised when a Gold export cannot pass local integrity admission."""


@dataclass(frozen=True)
class GoldSnapshotIntegrityResult:
    """Value-blind integrity result for one immutable local export."""

    summary: Mapping[str, object]
    manifest_content_id: str


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_snapshot_id(manifest_without_snapshot_id: Mapping[str, object]) -> str:
    return canonical_sha256(manifest_without_snapshot_id)


def derive_parquet_schema_sha256(schema: pa.Schema) -> str:
    """Fingerprint one flat Arrow schema without file-level metadata."""

    if not isinstance(schema, pa.Schema) or len(schema) < 1:
        raise DatabricksGoldSnapshotIntakeError("Parquet schema must be non-empty")
    names = schema.names
    if len(set(names)) != len(names):
        raise DatabricksGoldSnapshotIntakeError("Parquet column names must be unique")
    if any(pa.types.is_nested(field.type) for field in schema):
        raise DatabricksGoldSnapshotIntakeError("Parquet schema must be flat")
    descriptor = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in schema
    ]
    return canonical_sha256(descriptor)


def validate_gold_snapshot_intake_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    contract = _mapping(document, "D293 contract")
    _exact(
        contract,
        {
            "schema_version",
            "decision",
            "status",
            "purpose",
            "input_bindings",
            "gold_source_contract",
            "manifest_policy",
            "execution",
            "authorities",
        },
        "D293 contract",
    )
    _equal(contract["schema_version"], CONTRACT_SCHEMA, "contract schema")
    _equal(contract["decision"], "D-20260806-293", "contract decision")
    _equal(contract["status"], CONTRACT_STATUS, "contract status")
    _clean(contract["purpose"], "contract purpose")
    bindings = _mapping(contract["input_bindings"], "contract bindings")
    _exact(
        bindings,
        {
            "data_engineer_request_relative_path",
            "data_engineer_request_sha256",
            "entsoe_request_relative_path",
            "entsoe_request_sha256",
            "cost_preflight_contract_relative_path",
            "cost_preflight_contract_sha256",
            "cost_preflight_contract_content_id",
        },
        "contract bindings",
    )
    _equal(
        bindings,
        {
            "data_engineer_request_relative_path": _relative(REQUEST_PATH),
            "data_engineer_request_sha256": REQUEST_SHA256,
            "entsoe_request_relative_path": _relative(ENTSOE_REQUEST_PATH),
            "entsoe_request_sha256": ENTSOE_REQUEST_SHA256,
            "cost_preflight_contract_relative_path": _relative(COST_CONTRACT_PATH),
            "cost_preflight_contract_sha256": COST_CONTRACT_SHA256,
            "cost_preflight_contract_content_id": COST_CONTRACT_CONTENT_ID,
        },
        "contract bindings",
    )
    source = _mapping(contract["gold_source_contract"], "Gold source contract")
    _exact(
        source,
        {
            "catalog",
            "schema",
            "package_root",
            "exact_artifacts",
            "exact_companions",
        },
        "Gold source contract",
    )
    _equal(source["catalog"], "prd", "Gold catalog")
    _equal(source["schema"], "gold", "Gold schema")
    _equal(
        source["package_root"],
        "build/databricks-exports/<snapshot_id>",
        "package root",
    )
    _equal(
        source["exact_artifacts"],
        [
            {
                "role": role,
                "source_tables": list(tables),
                "relative_path": filename,
                "allow_empty": allow_empty,
            }
            for role, tables, filename, allow_empty in ARTIFACT_SPECS
        ],
        "Gold artifact inventory",
    )
    _equal(
        source["exact_companions"],
        [
            {"role": role, "relative_path": filename, "media_type": media_type}
            for role, filename, media_type in COMPANION_SPECS
        ],
        "Gold companion inventory",
    )
    policy = _mapping(contract["manifest_policy"], "manifest policy")
    _exact(
        policy,
        {
            "schema_version",
            "evidence_class",
            "snapshot_id_derivation",
            "total_size_bytes_scope",
            "exact_top_level_fields",
            "exact_companion_fields",
            "exact_artifact_fields",
            "format",
            "minimum_columns_per_artifact",
            "maximum_manifest_bytes",
            "maximum_companion_bytes",
            "parquet_schema_sha256_derivation",
            "flat_non_repeated_parquet_schema_required",
            "unexpected_package_entries_allowed",
            "exact_execution",
            "exact_authorities",
        },
        "manifest policy",
    )
    _equal(policy["schema_version"], MANIFEST_SCHEMA, "manifest schema")
    _equal(policy["evidence_class"], EVIDENCE_CLASS, "evidence class")
    _equal(
        policy["snapshot_id_derivation"],
        "SHA256_CANONICAL_MANIFEST_WITHOUT_SNAPSHOT_ID",
        "snapshot derivation",
    )
    _equal(
        policy["total_size_bytes_scope"],
        "SUM_OF_ARTIFACT_AND_COMPANION_SIZE_BYTES",
        "total size scope",
    )
    _equal(set(policy["exact_top_level_fields"]), MANIFEST_FIELDS, "manifest fields")
    _equal(
        set(policy["exact_companion_fields"]),
        COMPANION_FIELDS,
        "companion fields",
    )
    _equal(
        set(policy["exact_artifact_fields"]),
        ARTIFACT_FIELDS,
        "artifact fields",
    )
    _equal(policy["format"], "PARQUET", "artifact format")
    _equal(policy["minimum_columns_per_artifact"], 1, "minimum columns")
    _equal(policy["maximum_manifest_bytes"], MAX_MANIFEST_BYTES, "manifest budget")
    _equal(
        policy["maximum_companion_bytes"],
        MAX_COMPANION_BYTES,
        "companion budget",
    )
    _equal(
        policy["parquet_schema_sha256_derivation"],
        "SHA256_CANONICAL_JSON_ARRAY_OF_NAME_TYPE_NULLABLE",
        "schema derivation",
    )
    _equal(policy["flat_non_repeated_parquet_schema_required"], True, "flat schema")
    _equal(policy["unexpected_package_entries_allowed"], False, "package inventory")
    _equal(policy["exact_execution"], MANIFEST_EXECUTION, "manifest execution")
    _equal(policy["exact_authorities"], MANIFEST_AUTHORITIES, "manifest authorities")
    _equal(contract["execution"], LOCAL_EXECUTION, "contract execution")
    authorities = _mapping(contract["authorities"], "contract authorities")
    _exact(
        authorities,
        {
            "local_validator_implemented",
            "real_export_received",
            "real_export_byte_integrity",
            "source_authenticity",
            "cost_approval",
            "semantic_quality",
            "content_quality",
            "point_in_time",
            "predictive_value",
            "model_input",
            "model_selection",
            "candidate",
            "ompex_superiority",
            "promotion",
            "production",
        },
        "contract authorities",
    )
    _equal(authorities["local_validator_implemented"], True, "local validator")
    if any(
        value is not False
        for key, value in authorities.items()
        if key != "local_validator_implemented"
    ):
        raise DatabricksGoldSnapshotIntakeError("contract grants unsupported authority")
    _equal(canonical_sha256(contract), CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "exact_artifact_count": len(ARTIFACT_SPECS),
        "exact_companion_count": len(COMPANION_SPECS),
        **LOCAL_EXECUTION,
        "model_input_authorized": False,
        "production_authorized": False,
    }


def validate_gold_snapshot_intake_contract_path(
    path: str | Path = CONTRACT_PATH,
) -> dict[str, object]:
    selected = _absolute(path, "D293 contract")
    payload = _read(selected, "D293 contract", MAX_MANIFEST_BYTES)
    _equal(hashlib.sha256(payload).hexdigest(), CONTRACT_FILE_SHA256, "contract hash")
    result = validate_gold_snapshot_intake_contract(_strict_json(payload, "D293 contract"))
    _verify_local_binding(REQUEST_PATH, REQUEST_SHA256, "D291 request")
    _verify_local_binding(ENTSOE_REQUEST_PATH, ENTSOE_REQUEST_SHA256, "ENTSO-E request")
    _verify_local_binding(COST_CONTRACT_PATH, COST_CONTRACT_SHA256, "D291 cost contract")
    if _read(selected, "D292 contract", MAX_MANIFEST_BYTES) != payload:
        raise DatabricksGoldSnapshotIntakeError("D293 contract changed during validation")
    return result


def validate_gold_snapshot_manifest(
    manifest: Mapping[str, object],
    *,
    reference_time_utc: datetime,
) -> dict[str, object]:
    reference = _utc_datetime(reference_time_utc, "reference time")
    item = _mapping(manifest, "Gold snapshot manifest")
    _exact(item, MANIFEST_FIELDS, "Gold snapshot manifest")
    _equal(item["schema_version"], MANIFEST_SCHEMA, "manifest schema")
    _equal(item["evidence_class"], EVIDENCE_CLASS, "manifest evidence class")
    snapshot_id = _sha(item["snapshot_id"], "snapshot ID")
    _uuid4(item["export_batch_id"], "export batch ID")
    created = _utc_text_datetime(item["created_at_utc"], "manifest creation time")
    if created > reference + MAX_FUTURE_SKEW:
        raise DatabricksGoldSnapshotIntakeError("manifest creation time is in the future")
    _equal(item["source_catalog"], "prd", "source catalog")
    _equal(item["source_schema"], "gold", "source schema")
    _equal(item["request_sha256"], REQUEST_SHA256, "request binding")
    _equal(item["entsoe_request_sha256"], ENTSOE_REQUEST_SHA256, "ENTSO-E request binding")
    _equal(
        item["cost_preflight_contract_content_id"],
        COST_CONTRACT_CONTENT_ID,
        "cost contract binding",
    )
    _nonnegative_int(item["total_size_bytes"], "total size")
    companions = item["companions"]
    if not isinstance(companions, list) or len(companions) != len(COMPANION_SPECS):
        raise DatabricksGoldSnapshotIntakeError("companion inventory differs")
    normalized_companions = []
    for raw, (role, filename, media_type) in zip(companions, COMPANION_SPECS, strict=True):
        normalized_companions.append(_validate_companion(raw, role, filename, media_type))
    artifacts = item["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != len(ARTIFACT_SPECS):
        raise DatabricksGoldSnapshotIntakeError("artifact inventory differs")
    normalized_artifacts = []
    for raw, (role, tables, filename, allow_empty) in zip(artifacts, ARTIFACT_SPECS, strict=True):
        artifact = _mapping(raw, f"{role} artifact")
        _exact(artifact, ARTIFACT_FIELDS, f"{role} artifact")
        _equal(artifact["role"], role, f"{role} role")
        _equal(artifact["source_tables"], list(tables), f"{role} source tables")
        _equal(artifact["format"], "PARQUET", f"{role} format")
        _equal(artifact["relative_path"], filename, f"{role} path")
        _sha(artifact["sha256"], f"{role} hash")
        _positive_int(artifact["size_bytes"], f"{role} size_bytes")
        _positive_int(artifact["column_count"], f"{role} column_count")
        row_count = _nonnegative_int(artifact["row_count"], f"{role} row_count")
        row_group_count = _nonnegative_int(artifact["row_group_count"], f"{role} row_group_count")
        if not allow_empty and (row_count < 1 or row_group_count < 1):
            raise DatabricksGoldSnapshotIntakeError(f"{role} must be non-empty")
        _sha(artifact["schema_sha256"], f"{role} schema hash")
        normalized_artifacts.append(dict(artifact))
    _equal(item["execution"], MANIFEST_EXECUTION, "manifest execution")
    _equal(item["authorities"], MANIFEST_AUTHORITIES, "manifest authorities")
    derivation = dict(item)
    del derivation["snapshot_id"]
    _equal(derive_snapshot_id(derivation), snapshot_id, "snapshot ID derivation")
    return {
        **dict(item),
        "companions": normalized_companions,
        "artifacts": normalized_artifacts,
    }


def validate_gold_snapshot_package(
    manifest_path: str | Path,
    *,
    reference_time_utc: datetime,
    contract_path: str | Path = CONTRACT_PATH,
) -> GoldSnapshotIntegrityResult:
    """Verify exact files and Parquet footers without decoding business values."""

    validate_gold_snapshot_intake_contract_path(contract_path)
    selected_manifest = _absolute(manifest_path, "Gold snapshot manifest")
    if selected_manifest.name != "manifest.json":
        raise DatabricksGoldSnapshotIntakeError("manifest filename must be manifest.json")
    package = assert_absolute_path_has_no_links(selected_manifest.parent)
    export_root = assert_absolute_path_has_no_links(EXPORT_ROOT)
    if _path_key(package.parent) != _path_key(export_root):
        raise DatabricksGoldSnapshotIntakeError("package must be a direct export-root child")
    payload = _read(selected_manifest, "Gold snapshot manifest", MAX_MANIFEST_BYTES)
    manifest = validate_gold_snapshot_manifest(
        _strict_json(payload, "Gold snapshot manifest"),
        reference_time_utc=reference_time_utc,
    )
    snapshot_id = str(manifest["snapshot_id"])
    _equal(package.name, snapshot_id, "package directory snapshot ID")
    expected_names = {
        "manifest.json",
        *(filename for _, filename, _ in COMPANION_SPECS),
        *(filename for _, _, filename, _ in ARTIFACT_SPECS),
    }
    try:
        entries = tuple(package.iterdir())
    except OSError as exc:
        raise DatabricksGoldSnapshotIntakeError("package inventory is unavailable") from exc
    if {entry.name for entry in entries} != expected_names:
        raise DatabricksGoldSnapshotIntakeError("package inventory differs")
    if any(not entry.is_file() for entry in entries):
        raise DatabricksGoldSnapshotIntakeError("package contains a non-file entry")

    companion_results = []
    for declaration, (_, filename, media_type) in zip(
        manifest["companions"], COMPANION_SPECS, strict=True
    ):
        companion_results.append(
            _verify_companion(
                package / filename,
                _mapping(declaration, filename),
                media_type=media_type,
            )
        )

    artifact_results = []
    for declaration, (_, _, _, allow_empty) in zip(
        manifest["artifacts"], ARTIFACT_SPECS, strict=True
    ):
        artifact = _mapping(declaration, "artifact")
        artifact_results.append(
            _verify_parquet_artifact(
                package / str(artifact["relative_path"]),
                artifact,
                allow_empty=allow_empty,
            )
        )
    observed_total = sum(
        int(item["size_bytes"]) for item in (*companion_results, *artifact_results)
    )
    _equal(observed_total, manifest["total_size_bytes"], "package total size")
    if _read(selected_manifest, "Gold snapshot manifest", MAX_MANIFEST_BYTES) != payload:
        raise DatabricksGoldSnapshotIntakeError("manifest changed during validation")
    manifest_content_id = canonical_sha256(manifest)
    summary = {
        "schema_version": RESULT_SCHEMA,
        "status": RESULT_STATUS,
        "snapshot_id": snapshot_id,
        "manifest_content_id": manifest_content_id,
        "artifact_count": len(artifact_results),
        "artifact_row_count": sum(int(item["row_count"]) for item in artifact_results),
        "total_size_bytes": observed_total,
        "artifacts": artifact_results,
        "companions": companion_results,
        "execution": dict(LOCAL_EXECUTION),
        "authorities": dict(RESULT_AUTHORITIES),
    }
    return GoldSnapshotIntegrityResult(summary, manifest_content_id)


def _verify_parquet_artifact(
    path: Path,
    declaration: Mapping[str, object],
    *,
    allow_empty: bool,
) -> dict[str, object]:
    role = str(declaration["role"])
    selected = assert_absolute_path_has_no_links(path)
    try:
        before = selected.stat(follow_symlinks=False)
    except OSError as exc:
        raise DatabricksGoldSnapshotIntakeError(f"{role} artifact is unavailable") from exc
    if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
        raise DatabricksGoldSnapshotIntakeError(
            f"{role} artifact must be a single-link regular file"
        )
    _equal(int(before.st_size), declaration["size_bytes"], f"{role} size")
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor: int | None = None
    parquet = None
    digest = hashlib.sha256()
    streamed = 0
    try:
        descriptor = os.open(selected, flags)
        opened = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(opened):
            raise DatabricksGoldSnapshotIntakeError(
                f"{role} artifact identity changed before streaming"
            )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                streamed += len(chunk)
            stream.seek(0)
            try:
                parquet = pq.ParquetFile(stream)
                metadata = parquet.metadata
                schema = parquet.schema_arrow
            except (OSError, ValueError, pa.ArrowException) as exc:
                raise DatabricksGoldSnapshotIntakeError(
                    f"{role} artifact is not valid Parquet"
                ) from exc
            finally:
                if parquet is not None:
                    parquet.close()
            opened_after = os.fstat(stream.fileno())
            if _file_identity(opened) != _file_identity(opened_after):
                raise DatabricksGoldSnapshotIntakeError(
                    f"{role} artifact changed during footer inspection"
                )
    except OSError as exc:
        raise DatabricksGoldSnapshotIntakeError(f"{role} artifact cannot be read safely") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        after = selected.stat(follow_symlinks=False)
    except OSError as exc:
        raise DatabricksGoldSnapshotIntakeError(
            f"{role} artifact disappeared after validation"
        ) from exc
    if _file_identity(before) != _file_identity(after) or streamed != int(after.st_size):
        raise DatabricksGoldSnapshotIntakeError(f"{role} artifact changed while streaming")
    _equal(digest.hexdigest(), declaration["sha256"], f"{role} artifact hash")
    _equal(int(metadata.num_rows), declaration["row_count"], f"{role} row count")
    _equal(
        int(metadata.num_columns),
        declaration["column_count"],
        f"{role} column count",
    )
    _equal(
        int(metadata.num_row_groups),
        declaration["row_group_count"],
        f"{role} row-group count",
    )
    if not allow_empty and (int(metadata.num_rows) < 1 or int(metadata.num_row_groups) < 1):
        raise DatabricksGoldSnapshotIntakeError(f"{role} Parquet must be non-empty")
    schema_sha256 = derive_parquet_schema_sha256(schema)
    _equal(schema_sha256, declaration["schema_sha256"], f"{role} schema hash")
    return {
        "role": role,
        "sha256": digest.hexdigest(),
        "size_bytes": streamed,
        "row_count": int(metadata.num_rows),
        "column_count": int(metadata.num_columns),
        "row_group_count": int(metadata.num_row_groups),
        "schema_sha256": schema_sha256,
    }


def _verify_companion(
    path: Path,
    declaration: Mapping[str, object],
    *,
    media_type: str,
) -> dict[str, object]:
    filename = path.name
    payload = _read(path, filename, MAX_COMPANION_BYTES)
    _equal(len(payload), declaration["size_bytes"], f"{filename} size")
    digest = hashlib.sha256(payload).hexdigest()
    _equal(digest, declaration["sha256"], f"{filename} hash")
    if media_type == "application/json":
        _strict_json(payload, filename)
    else:
        if payload.startswith(b"\xef\xbb\xbf"):
            raise DatabricksGoldSnapshotIntakeError(f"{filename} has a BOM")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DatabricksGoldSnapshotIntakeError(f"{filename} is not UTF-8") from exc
        if not text.strip():
            raise DatabricksGoldSnapshotIntakeError(f"{filename} is empty")
    return {
        "role": declaration["role"],
        "relative_path": filename,
        "media_type": media_type,
        "sha256": digest,
        "size_bytes": len(payload),
    }


def _validate_companion(
    value: object,
    role: str,
    filename: str,
    media_type: str,
) -> dict[str, object]:
    declaration = _mapping(value, filename)
    _exact(declaration, COMPANION_FIELDS, filename)
    _equal(declaration["role"], role, f"{filename} role")
    _equal(declaration["relative_path"], filename, f"{filename} path")
    _equal(declaration["media_type"], media_type, f"{filename} media type")
    _sha(declaration["sha256"], f"{filename} hash")
    _positive_int(declaration["size_bytes"], f"{filename} size")
    return dict(declaration)


def _verify_local_binding(path: Path, expected_sha256: str, label: str) -> None:
    payload = _read(path, label, 2_000_000)
    _equal(hashlib.sha256(payload).hexdigest(), expected_sha256, f"{label} hash")


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatabricksGoldSnapshotIntakeError(f"{label} is not UTF-8") from exc
    if text.startswith("\ufeff"):
        raise DatabricksGoldSnapshotIntakeError(f"{label} has a BOM")

    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise DatabricksGoldSnapshotIntakeError(f"{label} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid constant {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, DatabricksGoldSnapshotIntakeError):
            raise
        raise DatabricksGoldSnapshotIntakeError(f"{label} is not strict JSON") from exc
    return _mapping(parsed, label)


def _read(path: Path, label: str, max_bytes: int) -> bytes:
    try:
        return read_stable_single_link_file(path, label=label, max_bytes=max_bytes)
    except ValueError as exc:
        raise DatabricksGoldSnapshotIntakeError(str(exc)) from exc


def _absolute(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise DatabricksGoldSnapshotIntakeError(f"{label} path must be absolute")
    try:
        return assert_absolute_path_has_no_links(path)
    except ValueError as exc:
        raise DatabricksGoldSnapshotIntakeError(str(exc)) from exc


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise DatabricksGoldSnapshotIntakeError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise DatabricksGoldSnapshotIntakeError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise DatabricksGoldSnapshotIntakeError(f"{label} differs")


def _clean(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise DatabricksGoldSnapshotIntakeError(f"{label} must be clean text")
    return value


def _sha(value: object, label: str) -> str:
    cleaned = _clean(value, label)
    if not _SHA.fullmatch(cleaned):
        raise DatabricksGoldSnapshotIntakeError(f"{label} must be lowercase SHA-256")
    return cleaned


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise DatabricksGoldSnapshotIntakeError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise DatabricksGoldSnapshotIntakeError(f"{label} must be nonnegative integer")
    return value


def _uuid4(value: object, label: str) -> str:
    cleaned = _clean(value, label)
    try:
        parsed = uuid.UUID(cleaned)
    except ValueError as exc:
        raise DatabricksGoldSnapshotIntakeError(f"{label} must be UUID") from exc
    if parsed.version != 4 or str(parsed) != cleaned:
        raise DatabricksGoldSnapshotIntakeError(f"{label} must be canonical UUID4")
    return cleaned


def _utc_text_datetime(value: object, label: str) -> datetime:
    cleaned = _clean(value, label)
    if not _UTC.fullmatch(cleaned):
        raise DatabricksGoldSnapshotIntakeError(f"{label} must use canonical UTC Z")
    try:
        parsed = datetime.fromisoformat(cleaned[:-1] + "+00:00")
    except ValueError as exc:
        raise DatabricksGoldSnapshotIntakeError(f"{label} is invalid") from exc
    return _utc_datetime(parsed, label)


def _utc_datetime(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise DatabricksGoldSnapshotIntakeError(f"{label} must be timezone-aware")
    normalized = value.astimezone(timezone.utc)
    if normalized.microsecond:
        raise DatabricksGoldSnapshotIntakeError(f"{label} must have whole seconds")
    return normalized


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(value.st_nlink),
    )


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _relative(path: Path) -> str:
    return PurePosixPath(path.relative_to(ROOT)).as_posix()


__all__ = [
    "ARTIFACT_SPECS",
    "CONTRACT_CONTENT_ID",
    "CONTRACT_PATH",
    "DatabricksGoldSnapshotIntakeError",
    "EXPORT_ROOT",
    "GoldSnapshotIntegrityResult",
    "LOCAL_EXECUTION",
    "MANIFEST_AUTHORITIES",
    "MANIFEST_EXECUTION",
    "RESULT_AUTHORITIES",
    "canonical_sha256",
    "derive_parquet_schema_sha256",
    "derive_snapshot_id",
    "validate_gold_snapshot_intake_contract",
    "validate_gold_snapshot_intake_contract_path",
    "validate_gold_snapshot_manifest",
    "validate_gold_snapshot_package",
]

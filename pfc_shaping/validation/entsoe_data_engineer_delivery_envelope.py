"""Verify the complete ENTSO-E data-engineer delivery envelope.

This gate is intentionally value-blind. It validates a fixed local inventory
and hashes every artifact twice through one checked descriptor. Parquet and
JSON artifact contents are left to independent downstream gates.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.validation import entsoe_cadence_package_binding as d270
from pfc_shaping.validation import entsoe_incremental_quality as d245
from pfc_shaping.validation import entsoe_lt_derived_parquet_adapter as d284
from pfc_shaping.validation import entsoe_parquet_streaming_integrity as d244

ROOT = Path(__file__).resolve().parents[2]
BUILD_ROOT = ROOT / "build"
PHASE = ROOT / ".planning" / "phases" / "14-lt-audit-remediation"
CONTRACT_PATH = PHASE / "ENTSOE-DATA-ENGINEER-DELIVERY-ENVELOPE-CONTRACT-V1.json"

CONTRACT_SCHEMA = "fmv_entsoe_data_engineer_delivery_envelope_contract.v1"
CONTRACT_STATUS = "VALUE_BLIND_DELIVERY_ENVELOPE_INTEGRITY_NO_DATA_OR_MODEL_AUTHORITY"
CONTRACT_FILE_SHA256 = "231fcda0a3bf0a8257571275ea5a5debfadfb5651edc94920b6a9ddc7bb9fcb4"
CONTRACT_CONTENT_ID = "fdad6cecd95b3714521949b0dd2663e616b03f968f9cf6e5f033ec5237e00ea6"
MANIFEST_SCHEMA = "fmv_entsoe_data_engineer_delivery_manifest.v1"
EVIDENCE_CLASS = "UNTRUSTED_DATA_ENGINEER_DELIVERY_CANDIDATE"
ASSESSMENT_SCHEMA = "fmv_entsoe_data_engineer_delivery_envelope_assessment.v1"
PASS_STATUS = "PASS_VALUE_BLIND_LOCAL_DELIVERY_ENVELOPE_NO_DATA_OR_MODEL_AUTHORITY"

PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"
JSON_MEDIA_TYPE = "application/json"
MAX_MANIFEST_BYTES = 2_097_152
MAX_TOTAL_ARTIFACT_BYTES = 3_298_534_883_328
HASH_CHUNK_BYTES = 8_388_608
MAX_ARTIFACT_COUNT = 11
MAX_TIMESTAMP_FIELDS = 16

REQUIRED_ARTIFACTS = (
    ("series_dimension", "series_dimension.parquet", PARQUET_MEDIA_TYPE, 8_589_934_592),
    (
        "series_resolution_history",
        "series_resolution_history.parquet",
        PARQUET_MEDIA_TYPE,
        8_589_934_592,
    ),
    ("zone_history", "zone_history.parquet", PARQUET_MEDIA_TYPE, 1_073_741_824),
    ("latest_values", "latest_values.parquet", PARQUET_MEDIA_TYPE, 274_877_906_944),
    (
        "vintage_values",
        "vintage_values.parquet",
        PARQUET_MEDIA_TYPE,
        2_199_023_255_552,
    ),
    ("quality_summary", "quality_summary.json", JSON_MEDIA_TYPE, 67_108_864),
    ("series_quality", "series_quality.parquet", PARQUET_MEDIA_TYPE, 17_179_869_184),
    ("family_inventory", "family_inventory.json", JSON_MEDIA_TYPE, 67_108_864),
    ("gap_report", "gap_report.parquet", PARQUET_MEDIA_TYPE, 68_719_476_736),
    (
        "source_reconciliation",
        "source_reconciliation.parquet",
        PARQUET_MEDIA_TYPE,
        8_589_934_592,
    ),
)
CONDITIONAL_ARTIFACT = (
    "excluded_series",
    "excluded_series.parquet",
    PARQUET_MEDIA_TYPE,
    8_589_934_592,
)
MAXIMUM_SIZE_BY_ROLE = {
    role: maximum_size for role, _, _, maximum_size in (*REQUIRED_ARTIFACTS, CONDITIONAL_ARTIFACT)
}
SOURCE_TABLES = (
    "dimentsoeseries",
    "factentsoetimeserieslatest",
    "factentsoetimeseriesvintages",
)
TIMESTAMP_FIELDS_BY_ROLE = {
    "series_dimension": ("meta_load_timestamp_utc",),
    "series_resolution_history": (
        "valid_from_utc",
        "valid_to_utc",
        "confirmation_timestamp_utc",
    ),
    "zone_history": ("valid_from_utc", "valid_to_utc"),
    "latest_values": (
        "target_start_utc",
        "target_end_utc",
        "publication_timestamp_utc",
        "meta_load_timestamp_utc",
        "first_observed_at_utc",
    ),
    "vintage_values": (
        "target_start_utc",
        "target_end_utc",
        "publication_timestamp_utc",
        "pull_timestamp_utc",
        "meta_load_timestamp_utc",
        "first_observed_at_utc",
        "available_at_utc",
        "forecast_issue_utc",
    ),
    "quality_summary": (),
    "series_quality": (
        "first_target_utc",
        "last_target_utc",
        "minimum_publication_timestamp_utc",
        "maximum_publication_timestamp_utc",
        "minimum_pull_timestamp_utc",
        "maximum_pull_timestamp_utc",
        "minimum_meta_load_timestamp_utc",
        "maximum_meta_load_timestamp_utc",
    ),
    "family_inventory": (),
    "gap_report": ("start_utc", "end_utc", "reattempted_at_utc"),
    "source_reconciliation": (
        "source_read_started_at_utc",
        "source_read_completed_at_utc",
    ),
    "excluded_series": (),
}

MANIFEST_AUTHORITIES = {
    "local_delivery_envelope_integrity": False,
    "source_authenticity": False,
    "schema_correctness": False,
    "data_quality": False,
    "point_in_time": False,
    "predictive_value": False,
    "model_input": False,
    "model_selection": False,
    "candidate": False,
    "promotion": False,
    "production": False,
}
AUTHORITIES = {**MANIFEST_AUTHORITIES, "local_delivery_envelope_integrity": True}
EXECUTION = {
    "databricks_connection_count": 0,
    "databricks_statement_count": 0,
    "warehouse_start_count": 0,
    "databricks_write_count": 0,
    "network_call_count": 0,
    "h_drive_access_count": 0,
    "decoded_parquet_row_count": 0,
    "decoded_json_artifact_count": 0,
    "opened_real_value_row_count": 0,
}

_SHA = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_TIMESTAMP_FIELD = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class EntsoeDataEngineerDeliveryEnvelopeError(ValueError):
    """Raised when a delivery envelope is incomplete, mutable or ambiguous."""


@dataclass(frozen=True)
class DeliveryEnvelopeResult:
    """Value-free result for one locally hashed delivery candidate."""

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


def derive_snapshot_id(manifest_without_snapshot_id: Mapping[str, object]) -> str:
    document = _mapping(manifest_without_snapshot_id, "snapshot derivation input")
    if "snapshot_id" in document:
        raise EntsoeDataEngineerDeliveryEnvelopeError(
            "snapshot derivation input must omit snapshot_id"
        )
    return canonical_sha256(document)


def validate_delivery_envelope_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    contract = _mapping(document, "D285 contract")
    _exact(
        contract,
        {
            "schema_version",
            "decision",
            "status",
            "purpose",
            "downstream_boundaries",
            "manifest",
            "required_artifacts_in_order",
            "conditional_artifact",
            "artifact_declaration",
            "limits",
            "exclusion_policy",
            "execution",
            "authorities",
        },
        "D285 contract",
    )
    _equal(contract["schema_version"], CONTRACT_SCHEMA, "contract schema")
    _equal(contract["decision"], "D-20260806-285", "contract decision")
    _equal(contract["status"], CONTRACT_STATUS, "contract status")
    _clean(contract["purpose"], "contract purpose")
    _equal(
        contract["downstream_boundaries"],
        {
            "d244_parquet_streaming_contract_content_id": d244.CONTRACT_CONTENT_ID,
            "d245_incremental_quality_contract_content_id": d245.CONTRACT_CONTENT_ID,
            "d270_cadence_package_contract_content_id": d270.CONTRACT_CONTENT_ID,
            "d284_derived_parquet_contract_content_id": d284.CONTRACT_CONTENT_ID,
        },
        "downstream boundaries",
    )
    _equal(
        contract["manifest"],
        {
            "schema_version": MANIFEST_SCHEMA,
            "evidence_class": EVIDENCE_CLASS,
            "snapshot_id_derivation": "CANONICAL_SHA256_OF_MANIFEST_WITHOUT_SNAPSHOT_ID",
            "source_catalog": "dev",
            "source_schema": "gold",
            "source_tables": list(SOURCE_TABLES),
            "package_directory_name_equals_snapshot_id": True,
            "package_must_be_under_repo_build": True,
            "fixed_flat_inventory": True,
            "manifest_must_be_stable_single_link": True,
            "artifact_files_must_be_stable_single_link": True,
            "artifact_hashing": "BOUNDED_STREAMING_SHA256_BEFORE_AND_AFTER",
            "artifact_content_is_decoded": False,
            "candidate_manifest_authorities": MANIFEST_AUTHORITIES,
        },
        "manifest policy",
    )
    _equal(
        contract["required_artifacts_in_order"],
        [_artifact_policy(specification) for specification in REQUIRED_ARTIFACTS],
        "required artifacts",
    )
    _equal(
        contract["conditional_artifact"],
        {
            **_artifact_policy(CONDITIONAL_ARTIFACT),
            "required_exactly_when_excluded_series_count_is_positive": True,
        },
        "conditional artifact",
    )
    _equal(
        contract["artifact_declaration"],
        {
            "fields": [
                "role",
                "relative_path",
                "media_type",
                "sha256",
                "size_bytes",
                "logical_record_count",
                "schema_content_id",
                "timestamp_bounds",
            ],
            "timestamp_bound_fields": ["minimum_utc", "maximum_utc"],
            "required_timestamp_fields_by_role": {
                role: list(fields) for role, fields in TIMESTAMP_FIELDS_BY_ROLE.items()
            },
            "timestamp_bound_null_pair_means_no_observed_non_null_timestamp": True,
            "one_sided_null_timestamp_bound_is_admissible": False,
        },
        "artifact declaration policy",
    )
    _equal(
        contract["limits"],
        {
            "maximum_manifest_bytes": MAX_MANIFEST_BYTES,
            "maximum_total_artifact_bytes": MAX_TOTAL_ARTIFACT_BYTES,
            "streaming_hash_chunk_bytes": HASH_CHUNK_BYTES,
            "maximum_artifact_count": MAX_ARTIFACT_COUNT,
            "maximum_timestamp_fields_per_artifact": MAX_TIMESTAMP_FIELDS,
        },
        "limits",
    )
    _equal(
        contract["exclusion_policy"],
        {
            "zero_excluded_series_requires_absent_excluded_series_artifact": True,
            "positive_excluded_series_requires_present_excluded_series_artifact": True,
            "excluded_series_logical_record_count_must_equal_declared_count": True,
            "silent_exclusion_is_admissible": False,
        },
        "exclusion policy",
    )
    _equal(contract["execution"], EXECUTION, "execution")
    _equal(contract["authorities"], AUTHORITIES, "authorities")
    _equal(canonical_sha256(contract), CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "required_artifact_count": len(REQUIRED_ARTIFACTS),
        "maximum_artifact_count": MAX_ARTIFACT_COUNT,
        "content_decoded": False,
        "model_input_authorized": False,
        **EXECUTION,
    }


def validate_delivery_envelope_contract_path(path: str | Path) -> dict[str, object]:
    contract_path = Path(path)
    if not contract_path.is_absolute():
        raise EntsoeDataEngineerDeliveryEnvelopeError("contract path must be absolute")
    payload = _read_small(contract_path, "D285 contract", MAX_MANIFEST_BYTES)
    if hashlib.sha256(payload).hexdigest() != CONTRACT_FILE_SHA256:
        raise EntsoeDataEngineerDeliveryEnvelopeError("contract SHA-256 differs")
    return validate_delivery_envelope_contract(_strict_json(payload, "D285 contract"))


def verify_delivery_envelope(
    *,
    contract_path: str | Path,
    manifest_path: str | Path,
    reference_time_utc: datetime,
) -> DeliveryEnvelopeResult:
    """Verify exact candidate bytes without decoding artifact contents."""

    validate_delivery_envelope_contract_path(contract_path)
    reference = _utc_datetime(reference_time_utc, "reference time")
    manifest_file = _under_build(Path(manifest_path), "delivery manifest")
    if manifest_file.name != "manifest.json":
        raise EntsoeDataEngineerDeliveryEnvelopeError("manifest file name differs")
    manifest_payload = _read_small(
        manifest_file,
        "delivery manifest",
        MAX_MANIFEST_BYTES,
    )
    manifest = _validate_manifest(
        _strict_json(manifest_payload, "delivery manifest"),
        reference,
    )
    snapshot_id = str(manifest["snapshot_id"])
    if manifest_file.parent.name != snapshot_id:
        raise EntsoeDataEngineerDeliveryEnvelopeError(
            "package directory name differs from snapshot ID"
        )
    declarations = manifest["artifacts"]
    if not isinstance(declarations, list):
        raise EntsoeDataEngineerDeliveryEnvelopeError("manifest artifacts differ")
    expected_names = {"manifest.json"} | {
        str(declaration["relative_path"]) for declaration in declarations
    }
    _validate_package_inventory(manifest_file.parent, expected_names)

    declared_total = sum(int(declaration["size_bytes"]) for declaration in declarations)
    if declared_total > MAX_TOTAL_ARTIFACT_BYTES:
        raise EntsoeDataEngineerDeliveryEnvelopeError("package byte budget is exceeded")
    first_pass: dict[str, tuple[str, int]] = {}
    for declaration in declarations:
        role = str(declaration["role"])
        path = _under_build(
            manifest_file.parent / str(declaration["relative_path"]),
            f"artifact {role}",
        )
        digest, size = _stream_hash_once(
            path,
            label=f"artifact {role}",
            maximum_size=MAXIMUM_SIZE_BY_ROLE[role],
        )
        _equal(digest, declaration["sha256"], f"artifact hash {role}")
        _equal(size, declaration["size_bytes"], f"artifact size {role}")
        first_pass[role] = (digest, size)

    receipts: list[dict[str, object]] = []
    observed_total = 0
    for declaration in declarations:
        role = str(declaration["role"])
        path = _under_build(
            manifest_file.parent / str(declaration["relative_path"]),
            f"artifact {role}",
        )
        digest, size = _stream_hash_once(
            path,
            label=f"artifact {role} verification pass",
            maximum_size=MAXIMUM_SIZE_BY_ROLE[role],
        )
        _equal(
            (digest, size),
            first_pass[role],
            f"post-envelope artifact identity {role}",
        )
        observed_total += size
        receipts.append(
            {
                "role": role,
                "sha256": digest,
                "size_bytes": size,
                "logical_record_count": declaration["logical_record_count"],
                "schema_content_id": declaration["schema_content_id"],
                "timestamp_bounds_content_id": canonical_sha256(declaration["timestamp_bounds"]),
            }
        )
    _equal(observed_total, declared_total, "package observed byte total")
    if _read_small(manifest_file, "delivery manifest", MAX_MANIFEST_BYTES) != manifest_payload:
        raise EntsoeDataEngineerDeliveryEnvelopeError(
            "manifest changed during envelope verification"
        )
    _validate_package_inventory(manifest_file.parent, expected_names)

    assessment = {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": PASS_STATUS,
        "contract_content_id": CONTRACT_CONTENT_ID,
        "reference_time_utc": _utc_text(reference),
        "snapshot_id": snapshot_id,
        "package_manifest_content_id": canonical_sha256(manifest),
        "artifact_count": len(declarations),
        "total_artifact_bytes": observed_total,
        "excluded_series_count": manifest["exclusion_summary"]["excluded_series_count"],
        "artifact_receipts": receipts,
        "fixed_inventory_verified": True,
        "artifact_hashes_verified_twice": True,
        "artifact_content_decoded": False,
        "silent_exclusion_possible_under_manifest": False,
        "source_authenticity_authorized": False,
        "schema_or_data_quality_authorized": False,
        "model_input_authorized": False,
        "authorities": dict(AUTHORITIES),
        **EXECUTION,
    }
    return DeliveryEnvelopeResult(
        assessment=assessment,
        assessment_content_id=canonical_sha256(assessment),
    )


def _validate_manifest(
    document: Mapping[str, object],
    reference: datetime,
) -> Mapping[str, object]:
    manifest = _mapping(document, "delivery manifest")
    _exact(
        manifest,
        {
            "schema_version",
            "evidence_class",
            "snapshot_id",
            "created_at_utc",
            "source_catalog",
            "source_schema",
            "source_tables",
            "extraction_window",
            "filters",
            "producer",
            "artifacts",
            "exclusion_summary",
            "authorities",
        },
        "delivery manifest",
    )
    _equal(manifest["schema_version"], MANIFEST_SCHEMA, "manifest schema")
    _equal(manifest["evidence_class"], EVIDENCE_CLASS, "manifest evidence class")
    _sha(manifest["snapshot_id"], "snapshot ID")
    created = _utc(manifest["created_at_utc"], "manifest creation time")
    if created > reference:
        raise EntsoeDataEngineerDeliveryEnvelopeError("manifest is future-dated")
    _equal(manifest["source_catalog"], "dev", "source catalog")
    _equal(manifest["source_schema"], "gold", "source schema")
    _equal(manifest["source_tables"], list(SOURCE_TABLES), "source tables")
    window = _mapping(manifest["extraction_window"], "extraction window")
    _exact(window, {"start_utc", "end_utc"}, "extraction window")
    start = _utc(window["start_utc"], "extraction start")
    end = _utc(window["end_utc"], "extraction end")
    if start >= end or end > created:
        raise EntsoeDataEngineerDeliveryEnvelopeError("extraction window is invalid")
    filters = manifest["filters"]
    if (
        not isinstance(filters, list)
        or len(filters) > 100
        or any(not isinstance(value, str) for value in filters)
    ):
        raise EntsoeDataEngineerDeliveryEnvelopeError("manifest filters are invalid")
    normalized_filters = [_clean(value, "manifest filter") for value in filters]
    if len(set(normalized_filters)) != len(normalized_filters):
        raise EntsoeDataEngineerDeliveryEnvelopeError("manifest filters are duplicated")
    producer = _mapping(manifest["producer"], "producer")
    _exact(producer, {"job_version", "query_version"}, "producer")
    _clean(producer["job_version"], "producer job version")
    _clean(producer["query_version"], "producer query version")
    exclusions = _mapping(manifest["exclusion_summary"], "exclusion summary")
    _exact(
        exclusions,
        {"excluded_series_count", "excluded_series_artifact_present"},
        "exclusion summary",
    )
    excluded_count = _bounded_int(
        exclusions["excluded_series_count"],
        0,
        1_000_000_000,
        "excluded series count",
    )
    artifact_present = exclusions["excluded_series_artifact_present"]
    if not isinstance(artifact_present, bool):
        raise EntsoeDataEngineerDeliveryEnvelopeError("excluded-series artifact flag is invalid")
    _equal(artifact_present, excluded_count > 0, "excluded-series artifact presence")
    declarations = manifest["artifacts"]
    if not isinstance(declarations, list):
        raise EntsoeDataEngineerDeliveryEnvelopeError("manifest artifacts differ")
    expected_specs = list(REQUIRED_ARTIFACTS)
    if excluded_count > 0:
        expected_specs.append(CONDITIONAL_ARTIFACT)
    if len(declarations) != len(expected_specs) or len(declarations) > MAX_ARTIFACT_COUNT:
        raise EntsoeDataEngineerDeliveryEnvelopeError("manifest artifact count differs")
    for position, (raw_declaration, specification) in enumerate(
        zip(declarations, expected_specs, strict=True),
        start=1,
    ):
        declaration = _validate_artifact_declaration(
            raw_declaration,
            specification,
            position,
        )
        if specification[0] == "excluded_series":
            _equal(
                declaration["logical_record_count"],
                excluded_count,
                "excluded-series logical record count",
            )
    _equal(manifest["authorities"], MANIFEST_AUTHORITIES, "manifest authorities")
    without_snapshot = dict(manifest)
    del without_snapshot["snapshot_id"]
    _equal(
        manifest["snapshot_id"],
        derive_snapshot_id(without_snapshot),
        "derived snapshot ID",
    )
    return manifest


def _validate_artifact_declaration(
    value: object,
    specification: tuple[str, str, str, int],
    position: int,
) -> Mapping[str, object]:
    declaration = _mapping(value, f"artifact declaration {position}")
    _exact(
        declaration,
        {
            "role",
            "relative_path",
            "media_type",
            "sha256",
            "size_bytes",
            "logical_record_count",
            "schema_content_id",
            "timestamp_bounds",
        },
        f"artifact declaration {position}",
    )
    role, relative_path, media_type, maximum_size = specification
    _equal(declaration["role"], role, f"artifact role {position}")
    _equal(declaration["relative_path"], relative_path, f"artifact path {role}")
    _equal(declaration["media_type"], media_type, f"artifact media type {role}")
    _sha(declaration["sha256"], f"artifact SHA-256 {role}")
    _bounded_int(declaration["size_bytes"], 1, maximum_size, f"artifact size {role}")
    _bounded_int(
        declaration["logical_record_count"],
        0,
        10_000_000_000_000,
        f"artifact logical record count {role}",
    )
    _sha(declaration["schema_content_id"], f"artifact schema content ID {role}")
    bounds = _mapping(declaration["timestamp_bounds"], f"timestamp bounds {role}")
    if len(bounds) > MAX_TIMESTAMP_FIELDS or tuple(bounds) != TIMESTAMP_FIELDS_BY_ROLE[role]:
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"timestamp bounds fields differ: {role}")
    for field, raw_bound in bounds.items():
        if _TIMESTAMP_FIELD.fullmatch(field) is None:
            raise EntsoeDataEngineerDeliveryEnvelopeError(
                f"timestamp field name is invalid: {role}"
            )
        bound = _mapping(raw_bound, f"timestamp bound {role}.{field}")
        _exact(bound, {"minimum_utc", "maximum_utc"}, f"timestamp bound {role}.{field}")
        minimum_raw = bound["minimum_utc"]
        maximum_raw = bound["maximum_utc"]
        if minimum_raw is None and maximum_raw is None:
            continue
        if minimum_raw is None or maximum_raw is None:
            raise EntsoeDataEngineerDeliveryEnvelopeError(
                f"timestamp bound is one-sided: {role}.{field}"
            )
        minimum = _utc(minimum_raw, f"minimum timestamp {role}.{field}")
        maximum = _utc(maximum_raw, f"maximum timestamp {role}.{field}")
        if minimum > maximum:
            raise EntsoeDataEngineerDeliveryEnvelopeError(
                f"timestamp bound order differs: {role}.{field}"
            )
    return declaration


def _stream_hash_once(
    path: Path,
    *,
    label: str,
    maximum_size: int,
) -> tuple[str, int]:
    try:
        lexical = assert_absolute_path_has_no_links(path)
    except ValueError as exc:
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} path is unsafe") from exc
    try:
        before = lexical.stat(follow_symlinks=False)
    except OSError as exc:
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} is unavailable") from exc
    if not _is_single_link_regular(before):
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} must be a single-link regular file")
    if int(before.st_size) < 1 or int(before.st_size) > maximum_size:
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} size is outside bounds")
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor: int | None = None
    try:
        descriptor = os.open(lexical, flags)
        opened = os.fstat(descriptor)
        if _stable_identity(opened) != _stable_identity(before):
            raise EntsoeDataEngineerDeliveryEnvelopeError(
                f"{label} identity changed before hashing"
            )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            result = _hash_stream(stream, maximum_size)
            if _stable_identity(os.fstat(stream.fileno())) != _stable_identity(opened):
                raise EntsoeDataEngineerDeliveryEnvelopeError(
                    f"{label} descriptor identity changed"
                )
    except OSError as exc:
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} cannot be hashed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        final = lexical.stat(follow_symlinks=False)
    except OSError as exc:
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} disappeared after hashing") from exc
    if _stable_identity(final) != _stable_identity(before):
        raise EntsoeDataEngineerDeliveryEnvelopeError(
            f"{label} path identity changed during hashing"
        )
    return result


def _validate_package_inventory(package: Path, expected_names: set[str]) -> None:
    try:
        entries = tuple(package.iterdir())
    except OSError as exc:
        raise EntsoeDataEngineerDeliveryEnvelopeError("package inventory cannot be listed") from exc
    if {entry.name for entry in entries} != expected_names or any(
        not entry.is_file() for entry in entries
    ):
        raise EntsoeDataEngineerDeliveryEnvelopeError("package inventory differs")


def _hash_stream(stream, maximum_size: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    while chunk := stream.read(HASH_CHUNK_BYTES):
        total += len(chunk)
        if total > maximum_size:
            raise EntsoeDataEngineerDeliveryEnvelopeError("artifact streaming byte cap is exceeded")
        digest.update(chunk)
    return digest.hexdigest(), total


def _is_single_link_regular(value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    return bool(
        stat.S_ISREG(value.st_mode)
        and int(value.st_nlink) == 1
        and not (
            os.name == "nt"
            and attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        )
    )


def _stable_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(value.st_nlink),
    )


def _under_build(path: Path, label: str) -> Path:
    try:
        lexical = assert_absolute_path_has_no_links(path)
    except ValueError as exc:
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} path is unsafe") from exc
    build = BUILD_ROOT.resolve()
    resolved = lexical.resolve()
    if resolved == build or build not in resolved.parents:
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} is outside repo build")
    return lexical


def _read_small(path: Path, label: str, maximum_bytes: int) -> bytes:
    try:
        return read_stable_single_link_file(
            path,
            label=label,
            max_bytes=maximum_bytes,
        )
    except (OSError, ValueError) as exc:
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} read failed") from exc


def _artifact_policy(specification: tuple[str, str, str, int]) -> dict[str, object]:
    role, relative_path, media_type, maximum_size = specification
    return {
        "role": role,
        "relative_path": relative_path,
        "media_type": media_type,
        "maximum_size_bytes": maximum_size,
    }


def _strict_json(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} JSON is invalid") from exc
    return _mapping(value, label)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EntsoeDataEngineerDeliveryEnvelopeError("JSON contains duplicate keys")
        result[key] = value
    return result


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} differs")


def _clean(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} must be clean text")
    return value


def _sha(value: object, label: str) -> str:
    text = _clean(value, label)
    if _SHA.fullmatch(text) is None:
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} must be lowercase SHA-256")
    return text


def _bounded_int(value: object, minimum: int, maximum: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} is outside bounds")
    return value


def _utc(value: object, label: str) -> datetime:
    text = _clean(value, label)
    if _UTC.fullmatch(text) is None:
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} must be canonical UTC Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} is invalid") from exc
    return _utc_datetime(parsed, label)


def _utc_datetime(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} must be timezone-aware")
    normalized = value.astimezone(timezone.utc)
    if normalized.microsecond:
        raise EntsoeDataEngineerDeliveryEnvelopeError(f"{label} must have whole-second precision")
    return normalized


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "ASSESSMENT_SCHEMA",
    "AUTHORITIES",
    "CONDITIONAL_ARTIFACT",
    "CONTRACT_CONTENT_ID",
    "CONTRACT_PATH",
    "EXECUTION",
    "MANIFEST_AUTHORITIES",
    "MANIFEST_SCHEMA",
    "PASS_STATUS",
    "REQUIRED_ARTIFACTS",
    "DeliveryEnvelopeResult",
    "EntsoeDataEngineerDeliveryEnvelopeError",
    "derive_snapshot_id",
    "validate_delivery_envelope_contract",
    "validate_delivery_envelope_contract_path",
    "verify_delivery_envelope",
]

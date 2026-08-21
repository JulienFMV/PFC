"""Verify synthetic ENTSO-E artifact bytes before offline quality profiling."""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pandas as pd

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.validation.entsoe_normalized_quality import (
    EntsoeNormalizedQualityProfile,
    profile_entsoe_normalized_quality,
)

CONTRACT_SCHEMA = "fmv_entsoe_artifact_package_integrity_preflight_contract.v1"
CONTRACT_STATUS = (
    "STATIC_OFFLINE_SYNTHETIC_ARTIFACT_INTEGRITY_PREFLIGHT_NO_REAL_ADMISSION"
)
CONTRACT_FILE_SHA256 = (
    "26631c82495df6f4616c2841bc73cbb5b9da13f5a265091fac5bcfcb5a21fe49"
)
CONTRACT_CONTENT_ID = (
    "c4b69dd9e1ba790a5bfe3ec3c2ddc7c6555d907902f7ff429d582b820d43369e"
)
INTAKE_CONTRACT_SHA256 = (
    "7ede1698099390babfa1d130bfecae61fd1e090888a3d6e6e4f892119db52b87"
)
INTAKE_CONTRACT_CONTENT_ID = (
    "75cb7e9f5d4cedb55e31fbd26b0bdcdf4ff629caf641c984cfec3fc6510b2a60"
)
SOURCE_PROOFS = {
    "d233": {
        "content_id": (
            "314ec85590c787874e2844d7db085236144c601681a5d1722f2735e6b1219d53"
        ),
        "manifest_sha256": (
            "d7f6ad3af60d2efd087718e0515c8b73a6534387ab12c7c7a5f88f42aeadf2b4"
        ),
    },
    "d236": {
        "content_id": (
            "772da4e0b22540bf22e1715ca146cc0a59adf0c7a9ec508e5a893d8495539247"
        ),
        "manifest_sha256": (
            "9d76631395176718b600ca0de8f7303ad8faea931c24298f514a54354da3d6e0"
        ),
    },
    "d238": {
        "content_id": (
            "2b9f6c513e0382e685bee78fb02d6c071a6954a26e715c5784fb28efec878aa8"
        ),
        "manifest_sha256": (
            "922a31ce0bf54b28e00b352bae1e4fa1d66fa357adf687c9a5f456e94d4511f6"
        ),
    },
    "d239": {
        "content_id": (
            "5e5aad7d04529e0efbb9926a1098a485ab3f797941ba2681c0a1609487f4df9b"
        ),
        "manifest_sha256": (
            "2d733c356b34ad0280802ed9d77b47f4d445e4d82e2ac6c45595f26f81787789"
        ),
    },
}

MANIFEST_SCHEMA = "fmv_entsoe_artifact_package_manifest.v1"
RECEIPT_SCHEMA = "fmv_entsoe_artifact_integrity_receipt.v1"
ROLE_ORDER = ("series_dimension", "latest_values", "vintage_values")
ROLE_FILE_NAMES = {
    "series_dimension": "series_dimension.ndjson",
    "latest_values": "latest_values.ndjson",
    "vintage_values": "vintage_values.ndjson",
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
ROLE_TYPES = {
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
ROLE_MAX_BYTES = {
    "series_dimension": 2_097_152,
    "latest_values": 8_388_608,
    "vintage_values": 8_388_608,
}
ROLE_MAX_ROWS = {
    "series_dimension": 10_000,
    "latest_values": 100_000,
    "vintage_values": 100_000,
}
MAX_TOTAL_BYTES = 16_777_216
MAX_LINE_BYTES = 1_048_576

MANIFEST_FIELD_ORDER = (
    "schema_version",
    "evidence_class",
    "snapshot_id",
    "capture_batch_id",
    "proposal_content_id",
    "reservation_id",
    "snapshot_reference_time_utc",
    "created_at_utc",
    "artifacts",
    "sign_semantics_by_series",
    "backfill_status_by_series",
    "authorities",
)
ARTIFACT_FIELD_ORDER = (
    "role",
    "format",
    "relative_path",
    "sha256",
    "size_bytes",
    "row_count",
    "columns",
    "column_types",
    "normalized_schema_sha256",
    "content_encoding",
    "line_ending",
    "compression",
)
RECEIPT_FIELD_ORDER = (
    "schema_version",
    "evidence_class",
    "snapshot_id",
    "capture_batch_id",
    "proposal_content_id",
    "reservation_id",
    "package_manifest_content_id",
    "verified_at_utc",
    "artifacts",
    "quality_profile_summary_sha256",
    "authorities",
)
RECEIPT_ARTIFACT_FIELDS = frozenset(
    {"role", "sha256", "size_bytes", "row_count", "normalized_schema_sha256"}
)
MANIFEST_AUTHORITIES = {
    "synthetic_artifact_integrity_verified": False,
    "real_artifact_receipt_admitted": False,
    "real_artifact_hashes_verified": False,
    "real_entsoe_values_opened": False,
    "source_authenticity_verified": False,
    "external_time_verified": False,
    "same_snapshot_point_in_time_availability_proven": False,
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
    "synthetic_artifact_integrity_verified": True,
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_INTAKE_ROOT = (
    REPOSITORY_ROOT / "build" / "governed-source-intake" / "synthetic-fixtures"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class EntsoeArtifactIntegrityError(ValueError):
    """Raised when artifact integrity or its local authority is ambiguous."""


@dataclass(frozen=True)
class EntsoeArtifactIntegrityResult:
    """Derived receipt and value-free quality result for a synthetic package."""

    receipt: Mapping[str, object]
    receipt_content_id: str
    package_manifest_content_id: str
    quality_profile: EntsoeNormalizedQualityProfile
    synthetic_value_row_count: int


def validate_artifact_integrity_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact D240 static preflight contract."""

    contract = _mapping(document, "artifact integrity contract")
    _equal(contract.get("schema_version"), CONTRACT_SCHEMA, "contract schema")
    _equal(contract.get("status"), CONTRACT_STATUS, "contract status")
    _equal(contract.get("decision"), "D-20260806-240", "contract decision")
    bindings = _mapping(contract.get("input_bindings"), "input bindings")
    for role in SOURCE_PROOFS:
        decision = role.upper()
        expected = SOURCE_PROOFS[role]
        content_field = {
            "d233": "governed_acquisition_package_proof_content_id",
            "d236": "bounded_execution_receipt_proof_content_id",
            "d238": "daily_capture_reservation_proof_content_id",
            "d239": "normalized_quality_profile_proof_content_id",
        }[role]
        manifest_field = content_field.replace("content_id", "manifest_sha256")
        _equal(bindings.get(content_field), expected["content_id"], f"{decision} content")
        _equal(
            bindings.get(manifest_field),
            expected["manifest_sha256"],
            f"{decision} manifest",
        )
    _equal(
        bindings.get("entsoe_intake_contract_content_id"),
        INTAKE_CONTRACT_CONTENT_ID,
        "intake contract content ID",
    )
    _equal(
        bindings.get("entsoe_intake_contract_sha256"),
        INTAKE_CONTRACT_SHA256,
        "intake contract SHA-256",
    )
    package = _mapping(contract.get("package_manifest"), "package manifest policy")
    _equal(
        package.get("exact_top_level_fields"),
        list(MANIFEST_FIELD_ORDER),
        "manifest fields",
    )
    declaration = _mapping(
        contract.get("artifact_declaration"),
        "artifact declaration policy",
    )
    _equal(declaration.get("exact_fields"), list(ARTIFACT_FIELD_ORDER), "artifact fields")
    _equal(declaration.get("exact_role_order"), list(ROLE_ORDER), "artifact roles")
    receipt = _mapping(contract.get("derived_receipt"), "derived receipt policy")
    _equal(receipt.get("exact_top_level_fields"), list(RECEIPT_FIELD_ORDER), "receipt fields")
    _equal(
        set(receipt.get("artifact_receipt_fields", [])),
        set(RECEIPT_ARTIFACT_FIELDS),
        "receipt artifact fields",
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
    _equal(
        execution.get("real_artifact_opening_authorized"),
        False,
        "real artifact opening authority",
    )
    authorities = _mapping(contract.get("authorities"), "contract authorities")
    _equal(authorities, RECEIPT_AUTHORITIES, "contract authorities")
    content_id = canonical_sha256(contract)
    _equal(content_id, CONTRACT_CONTENT_ID, "contract content ID")
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "contract_content_id": content_id,
        "synthetic_artifact_integrity_verification_implemented": True,
        "real_artifact_admission_implemented": False,
        "databricks_connection_budget": 0,
        "warehouse_start_budget": 0,
        "production_authorized": False,
    }


def verify_synthetic_entsoe_artifact_package(
    *,
    contract_path: str | Path,
    intake_contract_path: str | Path,
    source_proof_manifest_paths: Mapping[str, str | Path],
    package_manifest_path: str | Path,
    reference_time_utc: datetime,
) -> EntsoeArtifactIntegrityResult:
    """Verify exact bytes, parse them, and immediately run D239 in memory."""

    _require_canonical_workspace()
    contract_raw = _read_file(contract_path, "D240 contract", 100_000)
    if hashlib.sha256(contract_raw).hexdigest() != CONTRACT_FILE_SHA256:
        raise EntsoeArtifactIntegrityError("D240 contract SHA-256 differs")
    contract = _strict_json_object(contract_raw, "D240 contract")
    validate_artifact_integrity_contract(contract)
    intake_raw = _read_file(intake_contract_path, "ENTSO-E intake contract", 100_000)
    if hashlib.sha256(intake_raw).hexdigest() != INTAKE_CONTRACT_SHA256:
        raise EntsoeArtifactIntegrityError("ENTSO-E intake contract SHA-256 differs")
    intake = _strict_json_object(intake_raw, "ENTSO-E intake contract")
    _equal(canonical_sha256(intake), INTAKE_CONTRACT_CONTENT_ID, "intake content ID")
    _verify_source_proofs(source_proof_manifest_paths)

    manifest_path = assert_absolute_path_has_no_links(package_manifest_path)
    manifest_raw = _read_file(manifest_path, "package manifest", 2_000_000)
    manifest = _strict_json_object(manifest_raw, "package manifest")
    normalized = _validate_package_manifest(manifest, reference_time_utc)
    package_dir = _validate_package_directory(manifest_path, normalized["snapshot_id"])
    declarations = normalized["artifacts"]
    expected_inventory = {"manifest.json", *ROLE_FILE_NAMES.values()}
    actual_inventory = {entry.name for entry in package_dir.iterdir()}
    if actual_inventory != expected_inventory:
        raise EntsoeArtifactIntegrityError("package directory inventory differs")

    frames: dict[str, pd.DataFrame] = {}
    receipt_artifacts: list[dict[str, object]] = []
    total_bytes = 0
    total_rows = 0
    for declaration in declarations:
        role = str(declaration["role"])
        artifact_path = assert_absolute_path_has_no_links(
            package_dir / str(declaration["relative_path"])
        )
        if artifact_path.parent != package_dir:
            raise EntsoeArtifactIntegrityError(f"artifact escapes package: {role}")
        payload = _read_file(
            artifact_path,
            f"artifact {role}",
            ROLE_MAX_BYTES[role],
        )
        digest = hashlib.sha256(payload).hexdigest()
        _equal(digest, declaration["sha256"], f"artifact hash {role}")
        _equal(len(payload), declaration["size_bytes"], f"artifact size {role}")
        total_bytes += len(payload)
        rows = _parse_ndjson_artifact(payload, role)
        _equal(len(rows), declaration["row_count"], f"artifact row count {role}")
        total_rows += len(rows)
        frames[role] = _rows_to_frame(rows, role)
        receipt_artifacts.append(
            {
                "role": role,
                "sha256": digest,
                "size_bytes": len(payload),
                "row_count": len(rows),
                "normalized_schema_sha256": declaration["normalized_schema_sha256"],
            }
        )
    if total_bytes > MAX_TOTAL_BYTES:
        raise EntsoeArtifactIntegrityError("package total byte cap exceeded")

    snapshot_reference = _parse_utc(
        manifest["snapshot_reference_time_utc"],
        "snapshot_reference_time_utc",
    )
    quality = profile_entsoe_normalized_quality(
        frames["series_dimension"],
        frames["latest_values"],
        frames["vintage_values"],
        evidence_class="SYNTHETIC_TEST_ONLY",
        snapshot_reference_time_utc=snapshot_reference,
        sign_semantics_by_series=_mapping(
            manifest["sign_semantics_by_series"],
            "sign semantics",
        ),
        backfill_status_by_series=_mapping(
            manifest["backfill_status_by_series"],
            "backfill status",
        ),
        artifact_hashes_verified=False,
        real_receipt_admitted=False,
    )
    quality_hash = canonical_sha256(quality.summary)
    package_manifest_content_id = canonical_sha256(manifest)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "evidence_class": "SYNTHETIC_TEST_ONLY",
        "snapshot_id": manifest["snapshot_id"],
        "capture_batch_id": manifest["capture_batch_id"],
        "proposal_content_id": manifest["proposal_content_id"],
        "reservation_id": manifest["reservation_id"],
        "package_manifest_content_id": package_manifest_content_id,
        "verified_at_utc": _format_utc(
            _utc_datetime(reference_time_utc, "reference_time_utc")
        ),
        "artifacts": receipt_artifacts,
        "quality_profile_summary_sha256": quality_hash,
        "authorities": dict(RECEIPT_AUTHORITIES),
    }
    validate_derived_integrity_receipt(
        receipt,
        manifest,
        quality_profile_summary_sha256=quality_hash,
        reference_time_utc=reference_time_utc,
    )
    return EntsoeArtifactIntegrityResult(
        receipt=receipt,
        receipt_content_id=canonical_sha256(receipt),
        package_manifest_content_id=package_manifest_content_id,
        quality_profile=quality,
        synthetic_value_row_count=total_rows,
    )


def validate_derived_integrity_receipt(
    receipt_document: Mapping[str, object],
    manifest_document: Mapping[str, object],
    *,
    quality_profile_summary_sha256: str,
    reference_time_utc: datetime,
) -> dict[str, object]:
    """Replay the exact verifier-derived receipt; it remains synthetic only."""

    receipt = _mapping(receipt_document, "derived integrity receipt")
    _exact_fields(receipt, frozenset(RECEIPT_FIELD_ORDER), "derived receipt")
    _equal(receipt.get("schema_version"), RECEIPT_SCHEMA, "receipt schema")
    _equal(receipt.get("evidence_class"), "SYNTHETIC_TEST_ONLY", "receipt evidence")
    manifest = _mapping(manifest_document, "package manifest")
    for field in (
        "snapshot_id",
        "capture_batch_id",
        "proposal_content_id",
        "reservation_id",
    ):
        _equal(receipt.get(field), manifest.get(field), f"receipt {field}")
    _equal(
        receipt.get("package_manifest_content_id"),
        canonical_sha256(manifest),
        "package manifest content ID",
    )
    verified_at = _parse_utc(receipt.get("verified_at_utc"), "receipt verified_at_utc")
    reference = _utc_datetime(reference_time_utc, "reference_time_utc")
    if verified_at != reference:
        raise EntsoeArtifactIntegrityError("receipt verification time differs")
    _sha256(quality_profile_summary_sha256, "quality profile summary SHA-256")
    _equal(
        receipt.get("quality_profile_summary_sha256"),
        quality_profile_summary_sha256,
        "quality profile summary SHA-256",
    )
    artifacts = receipt.get("artifacts")
    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not isinstance(manifest_artifacts, list):
        raise EntsoeArtifactIntegrityError("receipt artifacts must be a list")
    if len(artifacts) != len(ROLE_ORDER):
        raise EntsoeArtifactIntegrityError("receipt artifact count differs")
    for role, artifact, declaration in zip(
        ROLE_ORDER,
        artifacts,
        manifest_artifacts,
        strict=True,
    ):
        item = _mapping(artifact, f"receipt artifact {role}")
        _exact_fields(item, RECEIPT_ARTIFACT_FIELDS, f"receipt artifact {role}")
        for field in RECEIPT_ARTIFACT_FIELDS:
            _equal(item.get(field), declaration.get(field), f"receipt {role}.{field}")
    _equal(receipt.get("authorities"), RECEIPT_AUTHORITIES, "receipt authorities")
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "PASS_SYNTHETIC_ARTIFACT_INTEGRITY_RECEIPT_ONLY",
        "receipt_content_id": canonical_sha256(receipt),
        "synthetic_artifact_integrity_verified": True,
        "real_artifact_receipt_admitted": False,
        "real_entsoe_values_opened": False,
        "model_input_authorized": False,
        "production_authorized": False,
    }


def derive_normalized_schema_sha256(role: str) -> str:
    """Return the exact schema declaration hash for one normalized role."""

    if role not in ROLE_ORDER:
        raise EntsoeArtifactIntegrityError("normalized schema role is invalid")
    return canonical_sha256(
        {
            "role": role,
            "columns": list(ROLE_COLUMNS[role]),
            "column_types": ROLE_TYPES[role],
        }
    )


def derive_snapshot_id(manifest_without_snapshot: Mapping[str, object]) -> str:
    """Derive the content-addressed snapshot ID from exact capture declarations."""

    document = _mapping(manifest_without_snapshot, "snapshot derivation input")
    required = {
        "capture_batch_id",
        "proposal_content_id",
        "reservation_id",
        "snapshot_reference_time_utc",
        "artifacts",
        "sign_semantics_by_series",
        "backfill_status_by_series",
    }
    if set(document) != required:
        raise EntsoeArtifactIntegrityError("snapshot derivation fields differ")
    return canonical_sha256(document)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_package_manifest(
    manifest: Mapping[str, object],
    reference_time_utc: datetime,
) -> dict[str, object]:
    _exact_fields(manifest, frozenset(MANIFEST_FIELD_ORDER), "package manifest")
    _equal(manifest.get("schema_version"), MANIFEST_SCHEMA, "manifest schema")
    _equal(manifest.get("evidence_class"), "SYNTHETIC_TEST_ONLY", "manifest evidence")
    snapshot_id = _sha256(manifest.get("snapshot_id"), "snapshot ID")
    _uuid(manifest.get("capture_batch_id"), "capture batch ID")
    _sha256(manifest.get("proposal_content_id"), "proposal content ID")
    _uuid(manifest.get("reservation_id"), "reservation ID")
    snapshot_reference = _parse_utc(
        manifest.get("snapshot_reference_time_utc"),
        "snapshot reference time",
    )
    created = _parse_utc(manifest.get("created_at_utc"), "manifest created_at_utc")
    reference = _utc_datetime(reference_time_utc, "reference_time_utc")
    if snapshot_reference > created or created > reference:
        raise EntsoeArtifactIntegrityError(
            "manifest timestamps must satisfy snapshot <= created <= reference"
        )
    _equal(manifest.get("authorities"), MANIFEST_AUTHORITIES, "manifest authorities")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(ROLE_ORDER):
        raise EntsoeArtifactIntegrityError("manifest artifacts differ")
    normalized_artifacts: list[Mapping[str, object]] = []
    total_declared_bytes = 0
    paths: set[str] = set()
    for role, raw in zip(ROLE_ORDER, artifacts, strict=True):
        declaration = _mapping(raw, f"artifact declaration {role}")
        _exact_fields(
            declaration,
            frozenset(ARTIFACT_FIELD_ORDER),
            f"artifact declaration {role}",
        )
        _equal(declaration.get("role"), role, f"artifact role {role}")
        _equal(declaration.get("format"), "NDJSON", f"artifact format {role}")
        relative = _artifact_relative_path(declaration.get("relative_path"), role)
        _equal(relative, ROLE_FILE_NAMES[role], f"artifact file name {role}")
        if relative in paths:
            raise EntsoeArtifactIntegrityError("artifact paths are duplicated")
        paths.add(relative)
        _sha256(declaration.get("sha256"), f"artifact hash {role}")
        size = _positive_int(declaration.get("size_bytes"), f"artifact size {role}")
        rows = _positive_int(declaration.get("row_count"), f"artifact rows {role}")
        if size > ROLE_MAX_BYTES[role] or rows > ROLE_MAX_ROWS[role]:
            raise EntsoeArtifactIntegrityError(f"artifact cap exceeded: {role}")
        total_declared_bytes += size
        _equal(declaration.get("columns"), list(ROLE_COLUMNS[role]), f"columns {role}")
        _equal(declaration.get("column_types"), ROLE_TYPES[role], f"types {role}")
        _equal(
            declaration.get("normalized_schema_sha256"),
            derive_normalized_schema_sha256(role),
            f"schema hash {role}",
        )
        _equal(declaration.get("content_encoding"), "UTF-8", f"encoding {role}")
        _equal(declaration.get("line_ending"), "LF_FINAL_LF_REQUIRED", f"LF policy {role}")
        _equal(declaration.get("compression"), "NONE", f"compression {role}")
        normalized_artifacts.append(declaration)
    if total_declared_bytes > MAX_TOTAL_BYTES:
        raise EntsoeArtifactIntegrityError("declared package total byte cap exceeded")
    derivation = {
        "capture_batch_id": manifest["capture_batch_id"],
        "proposal_content_id": manifest["proposal_content_id"],
        "reservation_id": manifest["reservation_id"],
        "snapshot_reference_time_utc": manifest["snapshot_reference_time_utc"],
        "artifacts": artifacts,
        "sign_semantics_by_series": manifest["sign_semantics_by_series"],
        "backfill_status_by_series": manifest["backfill_status_by_series"],
    }
    _equal(snapshot_id, derive_snapshot_id(derivation), "snapshot ID derivation")
    return {"snapshot_id": snapshot_id, "artifacts": normalized_artifacts}


def _validate_package_directory(manifest_path: Path, snapshot_id: str) -> Path:
    if manifest_path.name != "manifest.json":
        raise EntsoeArtifactIntegrityError("package manifest file name differs")
    package_dir = assert_absolute_path_has_no_links(manifest_path.parent)
    intake_root = assert_absolute_path_has_no_links(SYNTHETIC_INTAKE_ROOT)
    if package_dir.parent != intake_root or package_dir.name != snapshot_id:
        raise EntsoeArtifactIntegrityError(
            "synthetic package path is outside its content-addressed intake slot"
        )
    return package_dir


def _parse_ndjson_artifact(payload: bytes, role: str) -> list[dict[str, object]]:
    if not payload or payload.startswith(b"\xef\xbb\xbf"):
        raise EntsoeArtifactIntegrityError(f"artifact is empty or has BOM: {role}")
    if b"\r" in payload or not payload.endswith(b"\n"):
        raise EntsoeArtifactIntegrityError(f"artifact line ending differs: {role}")
    lines = payload[:-1].split(b"\n")
    if not lines or any(not line for line in lines):
        raise EntsoeArtifactIntegrityError(f"artifact contains blank line: {role}")
    if any(len(line) > MAX_LINE_BYTES for line in lines):
        raise EntsoeArtifactIntegrityError(f"artifact line cap exceeded: {role}")
    rows: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        row = _strict_json_object(line, f"{role} row {index}")
        if list(row) != list(ROLE_COLUMNS[role]):
            raise EntsoeArtifactIntegrityError(
                f"artifact column order differs: {role} row {index}"
            )
        normalized: dict[str, object] = {}
        for column, logical_type in ROLE_TYPES[role].items():
            normalized[column] = _validate_logical_value(
                row[column],
                logical_type,
                f"{role}.{column} row {index}",
            )
        rows.append(normalized)
    if len(rows) > ROLE_MAX_ROWS[role]:
        raise EntsoeArtifactIntegrityError(f"artifact row cap exceeded: {role}")
    return rows


def _rows_to_frame(rows: list[dict[str, object]], role: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(ROLE_COLUMNS[role]))
    for column, logical_type in ROLE_TYPES[role].items():
        if logical_type == "timestamp_utc":
            frame[column] = pd.to_datetime(frame[column], utc=True)
        elif logical_type == "float64_finite":
            frame[column] = frame[column].astype("float64")
        elif logical_type == "int64_non_negative":
            frame[column] = frame[column].astype("int64")
        elif logical_type == "boolean":
            frame[column] = frame[column].astype("bool")
    return frame


def _validate_logical_value(value: object, logical_type: str, label: str) -> object:
    if logical_type == "string":
        if not isinstance(value, str) or not value or value != value.strip():
            raise EntsoeArtifactIntegrityError(f"{label} must be a non-empty string")
        return value
    if logical_type == "timestamp_utc":
        _parse_utc(value, label)
        return value
    if logical_type == "float64_finite":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EntsoeArtifactIntegrityError(f"{label} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise EntsoeArtifactIntegrityError(f"{label} must be finite")
        return number
    if logical_type == "boolean":
        if not isinstance(value, bool):
            raise EntsoeArtifactIntegrityError(f"{label} must be boolean")
        return value
    if logical_type == "int64_non_negative":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise EntsoeArtifactIntegrityError(
                f"{label} must be a non-negative integer"
            )
        return value
    raise EntsoeArtifactIntegrityError(f"logical type is unsupported: {logical_type}")


def _verify_source_proofs(paths: Mapping[str, str | Path]) -> None:
    if set(paths) != set(SOURCE_PROOFS):
        raise EntsoeArtifactIntegrityError("source proof roles differ")
    for role, expected in SOURCE_PROOFS.items():
        raw = _read_file(paths[role], f"{role.upper()} proof manifest", 100_000)
        if hashlib.sha256(raw).hexdigest() != expected["manifest_sha256"]:
            raise EntsoeArtifactIntegrityError(f"{role.upper()} proof hash differs")
        manifest = _strict_json_object(raw, f"{role.upper()} proof manifest")
        _equal(manifest.get("content_id"), expected["content_id"], f"{role} content ID")
        execution = _mapping(manifest.get("execution"), f"{role} execution")
        if any(value != 0 for value in execution.values()):
            raise EntsoeArtifactIntegrityError(
                f"{role.upper()} execution counters are not zero"
            )


def _require_canonical_workspace() -> None:
    expected = Path(r"C:\Users\jbattaglia\PFC_LT")
    if REPOSITORY_ROOT != expected or Path.cwd().resolve() != expected:
        raise EntsoeArtifactIntegrityError("canonical workspace is required")


def _read_file(path: str | Path, label: str, max_bytes: int) -> bytes:
    selected = Path(path)
    if not selected.is_absolute():
        raise EntsoeArtifactIntegrityError(f"{label} path must be absolute")
    try:
        return read_stable_single_link_file(
            selected,
            label=label,
            max_bytes=max_bytes,
        )
    except (OSError, ValueError) as exc:
        raise EntsoeArtifactIntegrityError(f"{label} cannot be read safely") from exc


def _strict_json_object(payload: bytes, label: str) -> Mapping[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise EntsoeArtifactIntegrityError(f"{label} duplicate key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise EntsoeArtifactIntegrityError(f"{label} non-finite constant: {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntsoeArtifactIntegrityError(f"{label} is not strict UTF-8 JSON") from exc
    return _mapping(value, label)


def _artifact_relative_path(value: object, role: str) -> str:
    if not isinstance(value, str) or "\\" in value or ":" in value or "//" in value:
        raise EntsoeArtifactIntegrityError(f"artifact path is invalid: {role}")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {"", ".", ".."}:
        raise EntsoeArtifactIntegrityError(f"artifact path is unsafe: {role}")
    return value


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise EntsoeArtifactIntegrityError(f"{label} must use UTC whole-second Z")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise EntsoeArtifactIntegrityError(f"{label} is invalid") from exc


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_datetime(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EntsoeArtifactIntegrityError(f"{label} must be aware UTC")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise EntsoeArtifactIntegrityError(f"{label} must be aware UTC")
    return value.astimezone(timezone.utc)


def _uuid(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise EntsoeArtifactIntegrityError(f"{label} must be canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise EntsoeArtifactIntegrityError(f"{label} must be canonical UUID") from exc
    if str(parsed) != value or parsed.variant != uuid.RFC_4122:
        raise EntsoeArtifactIntegrityError(f"{label} must be canonical RFC 4122 UUID")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise EntsoeArtifactIntegrityError(f"{label} must be lowercase SHA-256")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise EntsoeArtifactIntegrityError(f"{label} must be a positive integer")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EntsoeArtifactIntegrityError(f"{label} must be an object")
    return value


def _exact_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    label: str,
) -> None:
    if set(value) != set(expected):
        raise EntsoeArtifactIntegrityError(f"{label} fields differ")


def _equal(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise EntsoeArtifactIntegrityError(
            f"{label} differs: expected {expected!r}, received {value!r}"
        )


__all__ = [
    "CONTRACT_CONTENT_ID",
    "EntsoeArtifactIntegrityError",
    "EntsoeArtifactIntegrityResult",
    "MANIFEST_AUTHORITIES",
    "RECEIPT_AUTHORITIES",
    "ROLE_COLUMNS",
    "ROLE_FILE_NAMES",
    "ROLE_ORDER",
    "ROLE_TYPES",
    "SOURCE_PROOFS",
    "SYNTHETIC_INTAKE_ROOT",
    "canonical_sha256",
    "derive_normalized_schema_sha256",
    "derive_snapshot_id",
    "validate_artifact_integrity_contract",
    "validate_derived_integrity_receipt",
    "verify_synthetic_entsoe_artifact_package",
]

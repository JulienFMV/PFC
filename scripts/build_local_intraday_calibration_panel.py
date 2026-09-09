"""Build a hash-bound local 15-minute intraday panel from audited quarantines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.governed_lt_acquisition import (
    OBSERVATION_RESOLUTION_PROVENANCE_ATTR,
    approved_provider_parser_payload_for_role,
    dataframe_semantic_sha256,
    energy_price_resolution_provenance,
    transform_provider_raw,
    verify_provider_raw_replay,
)
from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.parquet_safety import validate_parquet_allocation_budget
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    paths_overlap_by_identity,
    read_stable_single_link_file,
)
from scripts.build_lt_provider_verifier_zipapp import audit_verifier_zipapp

ROOT = Path(__file__).resolve().parents[1]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_VERIFIER_ARTIFACT_BYTES = 32 * 1024 * 1024
_CLI_BUILD_PATH_ATTRIBUTES = (
    "base_parquet",
    "audit_json",
    "audit_runtime_receipt_json",
    "verifier_artifact",
    "output_parquet",
    "output_manifest",
)
_PROVIDER_INVENTORY = {
    "acquisition-build-manifest.json",
    "bronze.parquet",
    "provider-parser.py",
    "provider-raw-envelope.json",
    "provider-transform-config.json",
}
_MAX_PROVIDER_BYTES = {
    "acquisition-build-manifest.json": 2 * 1024 * 1024,
    "bronze.parquet": 256 * 1024 * 1024,
    "provider-parser.py": 2 * 1024 * 1024,
    "provider-raw-envelope.json": 96 * 1024 * 1024,
    "provider-transform-config.json": 2 * 1024 * 1024,
}
_FRESH_AUDIT_SCHEMA_VERSION = "local_provider_acquisition_audit.v4"
_FRESH_AUDIT_KEYS = {
    "schema_version",
    "status",
    "production_authorization",
    "promotion_gate",
    "scientific_admission",
    "training_eligible",
    "model_selection_eligible",
    "confirmatory_rolling_origin_eligible",
    "future_holdout_eligible",
    "t057_eligible",
    "candidate_assembly_eligible",
    "local_diagnostic_allowed",
    "acquisition_directory",
    "manifest_sha256",
    "source_role",
    "source_system",
    "received_at_utc",
    "resolution_provenance",
    "audit_runtime",
    "negative_authority",
    "authority_gates",
    "evidence_scope",
    "artifacts",
    "replay",
}
_FALSE_ELIGIBILITY_FIELDS = (
    "production_authorization",
    "promotion_gate",
    "scientific_admission",
    "training_eligible",
    "model_selection_eligible",
    "confirmatory_rolling_origin_eligible",
    "future_holdout_eligible",
    "t057_eligible",
    "candidate_assembly_eligible",
)
_NEGATIVE_AUTHORITY = {
    "signing_status": "UNSIGNED_REQUIRES_INDEPENDENT_AUTHORITIES",
    "publication_status": "NOT_PUBLISHED",
    "namespace_security_status": "BUILDER_MUTABLE_UNTRUSTED_QUARANTINE",
    "authority_handoff_status": ("REQUIRES_BUILDER_INACCESSIBLE_COPY_AND_INDEPENDENT_SIGNATURE"),
}
_SECURITY_BLOCKING_CONTROLS = (
    "TRUSTED_AVAILABLE_AT_AND_REVISION_LINEAGE",
    "OFFICIAL_PRODUCT_AUCTION_SESSION_IDENTITY",
    "CAPTURE_TIME_CA_TLS_IDENTITY_ATTESTATION",
    "INDEPENDENT_SIGNATURE",
    "BUILDER_INACCESSIBLE_IMMUTABLE_FREEZE",
    "EXTERNAL_CAS_WORM_FRESH_MONOTONE_HEAD",
)
_SOURCE_SCIENTIFIC_BLOCKING_EVIDENCE = (
    "TRUSTED_AVAILABLE_AT_AND_REVISION_LINEAGE",
    "PROVIDER_AUTHENTICATED_PRODUCT_AND_SETTLEMENT_SEMANTICS",
    "NATIVE_QUARTER_HOUR_EVIDENCE_FOR_QUARTER_HOUR_CLAIMS",
)
_CANDIDATE_READINESS_BLOCKING_EVIDENCE = (
    "FRESH_GOVERNED_EEX_PIT_HARD_LEVEL_EVIDENCE",
    "MULTI_SEASON_MULTI_REGIME_INDEPENDENT_ROLLING_ORIGINS",
    "DEPENDENCE_AWARE_POWER_AND_PREREGISTERED_UNIQUE_ORIGINS",
    "SEALED_FUTURE_HOLDOUT",
    "CALIBRATED_PROBABILISTIC_SCENARIOS",
)
_FRESH_MANIFEST_KEYS = {
    "schema_version",
    "acquisition_id",
    "source_role",
    "source_system",
    "source_locator",
    "received_at_utc",
    "capture_spec",
    "artifacts",
    "bronze_frame_sha256",
    "resolution_provenance",
    "signing_status",
    "publication_status",
    "namespace_security_status",
    "authority_handoff_status",
    "durability_classification",
    "build_id",
}
_LEGACY_MANIFEST_KEYS = _FRESH_MANIFEST_KEYS - {"resolution_provenance"}
_VERIFIER_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "command",
    "production_authorization",
    "runtime_authority",
    "audit_result",
    "runtime_observation",
}
_VERIFIER_AUDIT_RESULT_KEYS = {
    "path",
    "sha256",
    "schema_version",
    "status",
}
_VERIFIER_RUNTIME_OBSERVATION_KEYS = {
    "source_revision",
    "dependency_versions",
    "receipt",
    "supervisor_attestation",
}
_VERIFIER_RUNTIME_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "dependency_tree_sha256",
    "dependency_import_source",
    "captured_artifact_sha256",
    "captured_artifact_sys_path_count",
    "source_artifact_sys_path_count",
    "captured_dependency_root_sys_path_count",
    "source_dependency_root_sys_path_count",
}
_LEGACY_RECLASSIFICATION_KEYS = {
    "schema_version",
    "status",
    "production_authorization",
    "promotion_gate",
    "scientific_admission",
    "acquisition_directory",
    "legacy_manifest",
    "legacy_audit",
    "provider_artifacts",
    "current_reclassification_parser_sha256",
    "audit_runtime",
    "resolution_provenance",
    "eligibility",
    "supersession_scope",
}
_LEGACY_ELIGIBILITY_KEYS = {
    "native_hourly_price_evidence",
    "native_quarter_hour_price_truth",
    "external_admission",
}
_LEGACY_AUDIT_V1_KEYS = {
    "acquisition_directory",
    "artifacts",
    "manifest_sha256",
    "negative_authority",
    "production_authorization",
    "promotion_gate",
    "received_at_utc",
    "remaining_authority",
    "replay",
    "schema_version",
    "source_role",
    "source_system",
    "status",
}
_LEGACY_REPLAY_KEYS = {
    "bronze_frame_sha256",
    "envelope_sha256",
    "role",
    "schema_version",
    "status",
    "transform_id",
}
_LEGACY_SUPERSESSION_SCOPE = (
    "The v1 audit remains historical local replay evidence but is "
    "superseded for every cadence or native-quarter-hour claim."
)


def _expected_authority_gates() -> dict[str, object]:
    return {
        "security": {
            "status": "MISSING_EXTERNAL_AUTHORITIES",
            "admitted": False,
            "blocking_controls": list(_SECURITY_BLOCKING_CONTROLS),
        },
        "source_scientific_admission": {
            "status": "MISSING_SOURCE_AUTHORITIES",
            "admitted": False,
            "blocking_evidence": list(_SOURCE_SCIENTIFIC_BLOCKING_EVIDENCE),
        },
        "candidate_readiness": {
            "status": "INSUFFICIENT_CONFIRMATORY_EVIDENCE",
            "admitted": False,
            "blocking_evidence": list(_CANDIDATE_READINESS_BLOCKING_EVIDENCE),
        },
    }


def build_panel(
    *,
    base_parquet: Path,
    expected_base_sha256: str,
    audit_json_paths: list[Path],
    expected_audit_sha256: list[str],
    audit_runtime_receipt_json_paths: list[Path],
    expected_audit_runtime_receipt_sha256: list[str],
    verifier_artifact_paths: list[Path],
    expected_verifier_artifact_sha256: list[str],
    start_utc: str,
    end_utc: str,
    output_parquet: Path,
    output_manifest: Path,
) -> dict[str, object]:
    """Build one continuous local panel and publish its manifest last."""

    evidence_count = len(audit_json_paths)
    if (
        evidence_count == 0
        or len(expected_audit_sha256) != evidence_count
        or len(audit_runtime_receipt_json_paths) != evidence_count
        or len(expected_audit_runtime_receipt_sha256) != evidence_count
        or len(verifier_artifact_paths) != evidence_count
        or len(expected_verifier_artifact_sha256) != evidence_count
    ):
        raise ValueError("audit and verifier evidence inputs must be non-empty and aligned")
    base_hash = _canonical_hash(expected_base_sha256, label="base parquet")
    audit_hashes = [_canonical_hash(value, label="audit JSON") for value in expected_audit_sha256]
    runtime_receipt_hashes = [
        _canonical_hash(value, label="verifier runtime receipt")
        for value in expected_audit_runtime_receipt_sha256
    ]
    verifier_artifact_hashes = [
        _canonical_hash(value, label="verifier artifact")
        for value in expected_verifier_artifact_sha256
    ]
    start = _utc_timestamp(start_utc, label="start_utc")
    end = _utc_timestamp(end_utc, label="end_utc")
    if end <= start:
        raise ValueError("panel window is invalid")
    output = _absolute_no_links(output_parquet)
    manifest_path = _absolute_no_links(output_manifest)
    if output.suffix != ".parquet" or manifest_path.suffix != ".json":
        raise ValueError("panel outputs must use .parquet and .json suffixes")
    if not output.parent.is_dir() or not manifest_path.parent.is_dir():
        raise ValueError("panel output parents must be pre-provisioned")
    if paths_overlap_by_identity(output, manifest_path):
        raise ValueError("panel and manifest paths must be distinct")

    base_path = _absolute_no_links(base_parquet)
    base_payload = read_stable_single_link_file(base_path, label="local base intraday parquet")
    if _sha256(base_payload) != base_hash:
        raise ValueError("base parquet differs from the caller-held SHA-256")
    frames = [_read_price_frame(base_payload, label="local base intraday parquet")]
    evidence_files: dict[Path, bytes] = {base_path: base_payload}
    evidence_directories: dict[Path, frozenset[str]] = {}
    sources: list[dict[str, object]] = [
        {
            "kind": "LOCAL_PREEXISTING_BASE_UNGOVERNED",
            "path": str(base_path),
            "sha256": base_hash,
            "size_bytes": len(base_payload),
        }
    ]
    for (
        audit_path_value,
        expected_hash,
        receipt_path_value,
        expected_receipt_hash,
        verifier_artifact_path_value,
        expected_verifier_hash,
    ) in zip(
        audit_json_paths,
        audit_hashes,
        audit_runtime_receipt_json_paths,
        runtime_receipt_hashes,
        verifier_artifact_paths,
        verifier_artifact_hashes,
        strict=True,
    ):
        audit_path = _absolute_no_links(audit_path_value)
        audit_payload = read_stable_single_link_file(
            audit_path,
            label="local provider acquisition audit",
        )
        if _sha256(audit_payload) != expected_hash:
            raise ValueError("audit JSON differs from the caller-held SHA-256")
        audit = _strict_json(audit_payload)
        if not isinstance(audit, dict):
            raise ValueError("provider audit JSON must be a mapping")
        (
            receipt_path,
            receipt_payload,
            verifier_artifact_path,
            verifier_artifact_payload,
            verifier_binding,
        ) = _load_bound_verifier_receipt(
            audit=audit,
            audit_path=audit_path,
            audit_sha256=expected_hash,
            receipt_path_value=receipt_path_value,
            expected_receipt_sha256=expected_receipt_hash,
            verifier_artifact_path_value=verifier_artifact_path_value,
            expected_verifier_artifact_sha256=expected_verifier_hash,
        )
        captured_files, captured_directories, acquisition_payloads = _capture_audit_evidence(
            audit=audit,
            audit_path=audit_path,
            audit_payload=audit_payload,
            receipt_path=receipt_path,
            receipt_payload=receipt_payload,
        )
        captured_files[verifier_artifact_path] = verifier_artifact_payload
        bronze_frame, source = _load_audited_de_bronze(
            audit=audit,
            audit_path=audit_path,
            audit_sha256=expected_hash,
            acquisition_payloads=acquisition_payloads,
            verifier_binding=verifier_binding,
        )
        evidence_files.update(captured_files)
        evidence_directories.update(captured_directories)
        frames.append(bronze_frame)
        sources.append(source)
    protected_paths = (*evidence_files, *evidence_directories)
    for target in (output, manifest_path):
        for protected in protected_paths:
            if paths_overlap_by_identity(target, protected):
                raise ValueError("panel output overlaps protected source evidence")
    combined = pd.concat(frames).sort_index()
    if combined.index.has_duplicates:
        raise ValueError("intraday panel sources overlap")
    panel = combined.loc[(combined.index >= start) & (combined.index < end)].copy()
    expected_index = pd.date_range(start, end - pd.Timedelta(minutes=15), freq="15min")
    if not panel.index.equals(expected_index):
        raise ValueError("intraday panel is not a complete contiguous 15-minute UTC grid")
    if not np.isfinite(panel["price_eur_mwh"].to_numpy(dtype=float)).all():
        raise ValueError("intraday panel contains non-finite prices")
    buffer = BytesIO()
    panel.to_parquet(buffer, index=True)
    panel_payload = buffer.getvalue()
    local = panel.index.tz_convert("Europe/Berlin")
    season = np.select(
        [local.month.isin([12, 1, 2]), local.month.isin([3, 4, 5]), local.month.isin([6, 7, 8])],
        ["Hiver", "Printemps", "Ete"],
        default="Automne",
    )
    season_counts = pd.Series(season).value_counts().sort_index().to_dict()
    manifest = {
        "schema_version": "local_intraday_calibration_panel.v3",
        "status": "COMPLETE_LOCAL_MIXED_AUTHORITY_NO_GO",
        "production_authorization": False,
        "promotion_gate": False,
        "scientific_admission": False,
        "training_eligible": False,
        "model_selection_eligible": False,
        "confirmatory_rolling_origin_eligible": False,
        "future_holdout_eligible": False,
        "t057_eligible": False,
        "candidate_assembly_eligible": False,
        "local_diagnostic_allowed": True,
        "market": "DE-LU_PROXY_FOR_CH_LOCAL_DIAGNOSTIC",
        "start_utc": _iso(start),
        "end_utc": _iso(end),
        "rows": int(len(panel)),
        "frequency_seconds": 900,
        "resolution_provenance": {
            "status": "MIXED_AUTHORITY_OUTPUT_GRID_NOT_NATIVE_TRUTH",
            "native_quarter_hour_truth_eligible": False,
            "reason": (
                "The panel contains an ungoverned base segment; each governed "
                "extension retains its own native-cadence provenance."
            ),
        },
        "season_row_counts": {str(key): int(value) for key, value in season_counts.items()},
        "sources": sources,
        "panel": {
            "path": str(output),
            "sha256": _sha256(panel_payload),
            "size_bytes": len(panel_payload),
        },
        "limitations": [
            "The pre-existing base segment is not governed provider-raw evidence.",
            "Local workstation capture time is not a trusted timestamp authority.",
            "DE-LU shaping transfer to CH is not validated by this panel.",
            "A 15-minute output grid does not imply native 15-minute observations.",
        ],
    }
    manifest_payload = _canonical_json(manifest) + b"\n"
    _assert_evidence_unchanged(
        files=evidence_files,
        directories=evidence_directories,
    )
    _write_exact_retry(
        output,
        panel_payload,
        label="local intraday calibration panel",
    )
    _assert_evidence_unchanged(
        files=evidence_files,
        directories=evidence_directories,
    )
    _write_exact_retry(
        manifest_path,
        manifest_payload,
        label="local intraday calibration panel manifest",
    )
    _assert_evidence_unchanged(
        files=evidence_files,
        directories=evidence_directories,
    )
    return manifest


def _read_price_frame(payload: bytes, *, label: str) -> pd.DataFrame:
    validate_parquet_allocation_budget(
        payload,
        label=label,
        max_rows=2_000_000,
        max_columns=16,
        max_cells=8_000_000,
        max_row_groups=8_192,
        allowed_physical_types={"BOOLEAN", "INT32", "INT64", "FLOAT", "DOUBLE"},
    )
    frame = pd.read_parquet(BytesIO(payload))
    if "price_eur_mwh" not in frame.columns:
        raise ValueError(f"{label} lacks price_eur_mwh")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        raise ValueError(f"{label} index must be timezone-aware")
    frame = frame[["price_eur_mwh"]].copy()
    frame.index = index.tz_convert("UTC")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{label} index must be ordered and unique")
    return frame


def _load_bound_verifier_receipt(
    *,
    audit: dict[str, object],
    audit_path: Path,
    audit_sha256: str,
    receipt_path_value: Path,
    expected_receipt_sha256: str,
    verifier_artifact_path_value: Path,
    expected_verifier_artifact_sha256: str,
) -> tuple[Path, bytes, Path, bytes, dict[str, object]]:
    receipt_path = _absolute_no_links(receipt_path_value)
    if paths_overlap_by_identity(receipt_path, audit_path):
        raise ValueError("audit and verifier receipt paths must be distinct")
    receipt_payload = read_stable_single_link_file(
        receipt_path,
        label="provider verifier runtime receipt",
        max_bytes=4 * 1024 * 1024,
    )
    if _sha256(receipt_payload) != expected_receipt_sha256:
        raise ValueError("verifier runtime receipt differs from caller-held SHA-256")
    receipt = _strict_json(receipt_payload)
    if (
        not isinstance(receipt, dict)
        or set(receipt) != _VERIFIER_RECEIPT_KEYS
        or receipt.get("schema_version") != "fmv_lt_provider_verifier_audit_receipt.v1"
        or receipt.get("status") != "LOCAL_RUNTIME_OBSERVATION_NOT_AUTHORITY"
        or receipt.get("command")
        not in {
            "audit-acquisition",
            "audit-legacy-resolution",
        }
        or receipt.get("production_authorization") is not False
        or receipt.get("runtime_authority") is not False
    ):
        raise ValueError("verifier runtime receipt policy is invalid")
    expected_command = (
        "audit-acquisition"
        if audit.get("schema_version") == _FRESH_AUDIT_SCHEMA_VERSION
        else "audit-legacy-resolution"
    )
    if receipt["command"] != expected_command:
        raise ValueError("verifier runtime receipt command is invalid")
    audit_result = receipt.get("audit_result")
    if (
        not isinstance(audit_result, dict)
        or set(audit_result) != _VERIFIER_AUDIT_RESULT_KEYS
        or audit_result.get("path") != str(audit_path)
        or audit_result.get("sha256") != audit_sha256
        or audit_result.get("schema_version") != audit.get("schema_version")
        or audit_result.get("status") != audit.get("status")
    ):
        raise ValueError("verifier runtime receipt audit binding is invalid")
    observation = receipt.get("runtime_observation")
    if not isinstance(observation, dict) or set(observation) != (
        _VERIFIER_RUNTIME_OBSERVATION_KEYS
    ):
        raise ValueError("verifier runtime observation is invalid")
    dependency_versions = observation.get("dependency_versions")
    if (
        not isinstance(dependency_versions, dict)
        or not dependency_versions
        or any(
            not isinstance(name, str) or not name or not isinstance(version, str) or not version
            for name, version in dependency_versions.items()
        )
        or observation.get("supervisor_attestation") != "UNSIGNED_LOCAL_PROCESS_OBSERVATION"
    ):
        raise ValueError("verifier runtime dependency observation is invalid")
    runtime = observation.get("receipt")
    if (
        not isinstance(runtime, dict)
        or set(runtime) != _VERIFIER_RUNTIME_RECEIPT_KEYS
        or runtime.get("schema_version") != "fmv_lt_provider_verifier_runtime_admission.v1"
        or runtime.get("status") != "PASS"
        or runtime.get("dependency_import_source") != "PROCESS_PRIVATE_EXACT_BYTE_COPY"
        or runtime.get("captured_artifact_sha256") != expected_verifier_artifact_sha256
        or not _strict_int_equals(
            runtime.get("captured_artifact_sys_path_count"),
            1,
        )
        or not _strict_int_equals(
            runtime.get("captured_dependency_root_sys_path_count"),
            1,
        )
        or not _strict_int_equals(
            runtime.get("source_artifact_sys_path_count"),
            0,
        )
        or not _strict_int_equals(
            runtime.get("source_dependency_root_sys_path_count"),
            0,
        )
    ):
        raise ValueError("verifier isolated runtime binding is invalid")
    source_revision = _canonical_hash(
        str(observation.get("source_revision", "")),
        label="verifier source revision",
    )
    dependency_tree_sha256 = _canonical_hash(
        str(runtime.get("dependency_tree_sha256", "")),
        label="verifier dependency tree",
    )
    verifier_artifact_path = _absolute_no_links(verifier_artifact_path_value)
    if paths_overlap_by_identity(verifier_artifact_path, receipt_path) or (
        paths_overlap_by_identity(verifier_artifact_path, audit_path)
    ):
        raise ValueError("verifier artifact overlaps audit or runtime receipt")
    verifier_artifact_payload = read_stable_single_link_file(
        verifier_artifact_path,
        label="provider verifier artifact",
        max_bytes=_MAX_VERIFIER_ARTIFACT_BYTES,
    )
    if _sha256(verifier_artifact_payload) != expected_verifier_artifact_sha256:
        raise ValueError("verifier artifact differs from caller-held SHA-256")
    verifier_artifact_audit = audit_verifier_zipapp(verifier_artifact_path)
    if (
        verifier_artifact_audit.get("status") != "PASS"
        or verifier_artifact_audit.get("production_authorization") is not False
        or verifier_artifact_audit.get("artifact_sha256") != expected_verifier_artifact_sha256
        or verifier_artifact_audit.get("source_revision") != source_revision
        or verifier_artifact_audit.get("dependency_tree_sha256") != dependency_tree_sha256
    ):
        raise ValueError("verifier artifact manifest binding is invalid")
    audit_runtime = audit.get("audit_runtime")
    if (
        not isinstance(audit_runtime, dict)
        or set(audit_runtime)
        != {
            "source_revision",
            "admission_status",
            "dependency_versions",
            "runtime_admission",
            "provider_transform_config_sha256",
        }
        or audit_runtime.get("source_revision") != source_revision
        or audit_runtime.get("admission_status") != "DIRECT_FUNCTION_CALL_NOT_RUNTIME_AUTHORITY"
        or not _json_equal(audit_runtime.get("dependency_versions"), {})
        or not _json_equal(audit_runtime.get("runtime_admission"), {})
    ):
        raise ValueError("audit runtime observation is invalid")
    artifacts = (
        audit.get("artifacts")
        if audit.get("schema_version") == _FRESH_AUDIT_SCHEMA_VERSION
        else audit.get("provider_artifacts")
    )
    config_record = (
        artifacts.get("provider-transform-config.json") if isinstance(artifacts, dict) else None
    )
    if (
        not isinstance(config_record, dict)
        or set(config_record) != {"sha256", "size_bytes"}
        or audit_runtime.get("provider_transform_config_sha256") != config_record.get("sha256")
    ):
        raise ValueError("audit runtime transform-config binding is invalid")
    return (
        receipt_path,
        receipt_payload,
        verifier_artifact_path,
        verifier_artifact_payload,
        {
            "runtime_receipt_path": str(receipt_path),
            "runtime_receipt_sha256": expected_receipt_sha256,
            "verifier_artifact_path": str(verifier_artifact_path),
            "verifier_artifact_sha256": expected_verifier_artifact_sha256,
            "verifier_source_revision": source_revision,
            "dependency_tree_sha256": dependency_tree_sha256,
        },
    )


def _load_audited_de_bronze(
    *,
    audit: dict[str, object],
    audit_path: Path,
    audit_sha256: str,
    acquisition_payloads: dict[str, bytes],
    verifier_binding: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    schema = audit.get("schema_version")
    if schema == _FRESH_AUDIT_SCHEMA_VERSION:
        return _load_fresh_audited_de_bronze(
            audit=audit,
            audit_path=audit_path,
            audit_sha256=audit_sha256,
            acquisition_payloads=acquisition_payloads,
            verifier_binding=verifier_binding,
        )
    if schema == "legacy_provider_resolution_reclassification.v1":
        return _load_reclassified_legacy_de_bronze(
            audit=audit,
            audit_path=audit_path,
            audit_sha256=audit_sha256,
            acquisition_payloads=acquisition_payloads,
            verifier_binding=verifier_binding,
        )
    raise ValueError("provider audit schema is unsupported for the local DE panel")


def _load_fresh_audited_de_bronze(
    *,
    audit: dict[str, object],
    audit_path: Path,
    audit_sha256: str,
    acquisition_payloads: dict[str, bytes],
    verifier_binding: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    if (
        set(audit) != _FRESH_AUDIT_KEYS
        or audit.get("status") != "VERIFIED_LOCAL_QUARANTINE_NOT_PRODUCTION"
        or any(audit.get(field) is not False for field in _FALSE_ELIGIBILITY_FIELDS)
        or audit.get("local_diagnostic_allowed") is not True
        or audit.get("source_role") != "epex_de"
        or not _json_equal(audit.get("negative_authority"), _NEGATIVE_AUTHORITY)
        or not _json_equal(
            audit.get("authority_gates"),
            _expected_authority_gates(),
        )
        or not isinstance(audit.get("artifacts"), dict)
        or not isinstance(audit.get("replay"), dict)
    ):
        raise ValueError("audit JSON is not eligible for the local DE intraday panel")
    directory = _absolute_no_links(str(audit.get("acquisition_directory", "")))
    manifest_payload = acquisition_payloads["acquisition-build-manifest.json"]
    manifest = _strict_json(manifest_payload)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != _FRESH_MANIFEST_KEYS
        or manifest.get("schema_version") != "lt_provider_acquisition_build.v2"
        or manifest.get("source_role") != "epex_de"
        or manifest.get("source_role") != audit.get("source_role")
        or manifest.get("source_system") != audit.get("source_system")
        or manifest.get("received_at_utc") != audit.get("received_at_utc")
        or _sha256(manifest_payload) != audit.get("manifest_sha256")
        or not _json_equal(manifest.get("artifacts"), audit.get("artifacts"))
        or manifest.get("bronze_frame_sha256") != audit.get("replay", {}).get("bronze_frame_sha256")
        or not _json_equal(
            manifest.get("resolution_provenance"),
            audit.get("resolution_provenance"),
        )
        or not _json_equal(
            {
                "signing_status": manifest.get("signing_status"),
                "publication_status": manifest.get("publication_status"),
                "namespace_security_status": manifest.get("namespace_security_status"),
                "authority_handoff_status": manifest.get("authority_handoff_status"),
            },
            _NEGATIVE_AUTHORITY,
        )
    ):
        raise ValueError("fresh acquisition manifest binding is invalid")
    unsigned_manifest = dict(manifest)
    build_id = unsigned_manifest.pop("build_id")
    if build_id != _sha256(_canonical_json(unsigned_manifest)):
        raise ValueError("fresh acquisition manifest build_id is invalid")
    config = _strict_json(acquisition_payloads["provider-transform-config.json"])
    if not isinstance(config, dict):
        raise ValueError("fresh provider transform config is invalid")
    parser_record = audit["artifacts"].get("provider-parser.py")
    if not isinstance(parser_record, dict):
        raise ValueError("fresh provider parser binding is invalid")
    replay = verify_provider_raw_replay(
        role="epex_de",
        source_system=str(audit.get("source_system", "")),
        envelope_payload=acquisition_payloads["provider-raw-envelope.json"],
        bronze_payload=acquisition_payloads["bronze.parquet"],
        parser_payload=acquisition_payloads["provider-parser.py"],
        parser_code_sha256=str(parser_record.get("sha256", "")),
        parser_config=config,
    )
    if not _json_equal(replay, audit["replay"]):
        raise ValueError("fresh provider replay differs from its audit JSON")
    bronze_path = assert_absolute_path_has_no_links(directory / "bronze.parquet")
    bronze_payload = acquisition_payloads["bronze.parquet"]
    actual = {"sha256": _sha256(bronze_payload), "size_bytes": len(bronze_payload)}
    bronze_frame = _read_price_frame(bronze_payload, label="audited provider bronze")
    resolution = energy_price_resolution_provenance(bronze_frame, role="epex_de")
    if not _json_equal(audit.get("resolution_provenance"), resolution) or not _json_equal(
        audit.get("evidence_scope"),
        _expected_evidence_scope(resolution),
    ):
        raise ValueError("audited provider resolution or evidence scope differs from replay")
    return bronze_frame, {
        "kind": "AUDITED_LOCAL_QUARANTINE_NOT_PRODUCTION",
        "audit_path": str(audit_path),
        "audit_sha256": audit_sha256,
        "acquisition_manifest_sha256": audit.get("manifest_sha256"),
        "bronze_path": str(bronze_path),
        "resolution_provenance": resolution,
        "evidence_scope": audit["evidence_scope"],
        "scientific_admission": False,
        "confirmatory_rolling_origin_eligible": False,
        "local_diagnostic_allowed": True,
        "verifier": verifier_binding,
        **actual,
    }


def _load_reclassified_legacy_de_bronze(
    *,
    audit: dict[str, object],
    audit_path: Path,
    audit_sha256: str,
    acquisition_payloads: dict[str, bytes],
    verifier_binding: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    if (
        set(audit) != _LEGACY_RECLASSIFICATION_KEYS
        or audit.get("status") != "LEGACY_RESOLUTION_RECLASSIFIED_LOCAL_NO_GO"
        or audit.get("production_authorization") is not False
        or audit.get("promotion_gate") is not False
        or audit.get("scientific_admission") is not False
        or not isinstance(audit.get("provider_artifacts"), dict)
        or not isinstance(audit.get("legacy_manifest"), dict)
        or not isinstance(audit.get("legacy_audit"), dict)
        or not isinstance(audit.get("resolution_provenance"), dict)
        or not isinstance(audit.get("eligibility"), dict)
        or set(audit["eligibility"]) != _LEGACY_ELIGIBILITY_KEYS
        or audit["eligibility"].get("external_admission") != "MISSING"
        or audit.get("current_reclassification_parser_sha256")
        != _sha256(approved_provider_parser_payload_for_role("epex_de"))
        or audit.get("supersession_scope") != _LEGACY_SUPERSESSION_SCOPE
    ):
        raise ValueError("legacy reclassification is not eligible for local migration")
    resolution = dict(audit["resolution_provenance"])
    if resolution.get("role") != "epex_de":
        raise ValueError("legacy reclassification is not bound to epex_de")
    directory = _absolute_no_links(str(audit.get("acquisition_directory", "")))
    legacy_manifest_record = audit["legacy_manifest"]
    legacy_manifest_path = _absolute_no_links(str(legacy_manifest_record.get("path", "")))
    if (
        set(legacy_manifest_record) != {"path", "sha256", "schema_version"}
        or legacy_manifest_path != directory / "acquisition-build-manifest.json"
    ):
        raise ValueError("legacy reclassification manifest path is not exact")
    manifest_hash = _canonical_hash(
        str(legacy_manifest_record.get("sha256", "")),
        label="legacy manifest",
    )
    manifest_payload = acquisition_payloads["acquisition-build-manifest.json"]
    if _sha256(manifest_payload) != manifest_hash:
        raise ValueError("legacy acquisition manifest differs from reclassification")
    manifest = _strict_json(manifest_payload)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != _LEGACY_MANIFEST_KEYS
        or manifest.get("schema_version") != "lt_provider_acquisition_build.v1"
        or legacy_manifest_record.get("schema_version") != "lt_provider_acquisition_build.v1"
        or manifest.get("source_role") != "epex_de"
        or not _json_equal(manifest.get("artifacts"), audit["provider_artifacts"])
    ):
        raise ValueError("legacy acquisition manifest binding is invalid")
    unsigned_manifest = dict(manifest)
    build_id = unsigned_manifest.pop("build_id")
    if build_id != _sha256(_canonical_json(unsigned_manifest)):
        raise ValueError("legacy acquisition manifest build_id is invalid")
    config = _strict_json(acquisition_payloads["provider-transform-config.json"])
    if not isinstance(config, dict):
        raise ValueError("legacy provider transform config is invalid")
    current = transform_provider_raw(
        envelope_payload=acquisition_payloads["provider-raw-envelope.json"],
        parser_config=config,
    )
    legacy = _read_price_frame(
        acquisition_payloads["bronze.parquet"],
        label="legacy provider bronze replay",
    )
    current_values = current.copy()
    current_values.attrs.clear()
    legacy_values = legacy.copy()
    legacy_values.attrs.clear()
    try:
        pd.testing.assert_frame_equal(
            current_values,
            legacy_values,
            check_dtype=True,
            check_exact=True,
            check_like=False,
            check_freq=False,
        )
    except AssertionError as exc:
        raise ValueError("legacy current provider replay binding is invalid") from exc
    if dataframe_semantic_sha256(current_values) != manifest.get("bronze_frame_sha256"):
        raise ValueError("legacy current provider replay binding is invalid")
    _verify_bound_legacy_audit(
        record=audit["legacy_audit"],
        directory=directory,
        manifest_sha256=manifest_hash,
        manifest=manifest,
    )
    bronze_path = assert_absolute_path_has_no_links(directory / "bronze.parquet")
    bronze_payload = acquisition_payloads["bronze.parquet"]
    actual = {"sha256": _sha256(bronze_payload), "size_bytes": len(bronze_payload)}
    bronze_frame = _read_price_frame(bronze_payload, label="legacy provider bronze")
    if OBSERVATION_RESOLUTION_PROVENANCE_ATTR in bronze_frame.attrs:
        raise ValueError("legacy provider bronze unexpectedly contains resolution metadata")
    bronze_frame.attrs[OBSERVATION_RESOLUTION_PROVENANCE_ATTR] = resolution
    validated_resolution = energy_price_resolution_provenance(
        bronze_frame,
        role="epex_de",
    )
    expected_eligibility = {
        "native_hourly_price_evidence": (
            "LOCAL_UNSIGNED_ONLY"
            if validated_resolution["native_hourly_cadence_eligible"] is True
            else "INELIGIBLE"
        ),
        "native_quarter_hour_price_truth": (
            "LOCAL_UNSIGNED_ONLY"
            if validated_resolution["native_quarter_hour_truth_eligible"] is True
            else "INELIGIBLE"
        ),
        "external_admission": "MISSING",
    }
    if not _json_equal(validated_resolution, resolution) or not _json_equal(
        audit["eligibility"], expected_eligibility
    ):
        raise ValueError("legacy resolution reclassification is inconsistent")
    return bronze_frame, {
        "kind": "LEGACY_RESOLUTION_RECLASSIFICATION_LOCAL_NO_GO",
        "audit_path": str(audit_path),
        "audit_sha256": audit_sha256,
        "acquisition_manifest_sha256": manifest_hash,
        "bronze_path": str(bronze_path),
        "resolution_provenance": resolution,
        "scientific_admission": False,
        "confirmatory_rolling_origin_eligible": False,
        "local_diagnostic_allowed": True,
        "verifier": verifier_binding,
        **actual,
    }


def _verify_bound_legacy_audit(
    *,
    record: dict[str, object],
    directory: Path,
    manifest_sha256: str,
    manifest: dict[str, object],
) -> None:
    if (
        set(record) != {"path", "sha256", "schema_version", "resolution_authority"}
        or record.get("schema_version") != "local_provider_acquisition_audit.v1"
        or record.get("resolution_authority") is not False
    ):
        raise ValueError("legacy audit record is not explicitly non-authoritative")
    path = _absolute_no_links(str(record.get("path", "")))
    expected = _canonical_hash(str(record.get("sha256", "")), label="legacy audit")
    payload = read_stable_single_link_file(
        path,
        label="legacy provider audit",
        max_bytes=4 * 1024 * 1024,
    )
    if _sha256(payload) != expected:
        raise ValueError("legacy provider audit differs from reclassification")
    prior = _strict_json(payload)
    if (
        not isinstance(prior, dict)
        or set(prior) != _LEGACY_AUDIT_V1_KEYS
        or prior.get("schema_version") != "local_provider_acquisition_audit.v1"
        or prior.get("status") != "VERIFIED_LOCAL_QUARANTINE_NOT_PRODUCTION"
        or prior.get("production_authorization") is not False
        or prior.get("promotion_gate") is not False
        or prior.get("source_role") != "epex_de"
        or prior.get("source_role") != manifest.get("source_role")
        or prior.get("source_system") != manifest.get("source_system")
        or prior.get("received_at_utc") != manifest.get("received_at_utc")
        or prior.get("manifest_sha256") != manifest_sha256
        or not _json_equal(prior.get("artifacts"), manifest.get("artifacts"))
        or not _json_equal(prior.get("negative_authority"), _NEGATIVE_AUTHORITY)
        or prior.get("remaining_authority")
        != "BUILDER_INACCESSIBLE_COPY_ACL_FREEZE_TRUSTED_TIME_AND_INDEPENDENT_SIGNATURE"
        or _absolute_no_links(str(prior.get("acquisition_directory", ""))) != directory
    ):
        raise ValueError("legacy provider audit binding is invalid")
    replay = prior.get("replay")
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(replay, dict)
        or set(replay) != _LEGACY_REPLAY_KEYS
        or replay.get("status") != "VERIFIED_EXACT_PROVIDER_REPLAY"
        or replay.get("role") != "epex_de"
        or not isinstance(artifacts, dict)
        or replay.get("bronze_frame_sha256") != manifest.get("bronze_frame_sha256")
        or replay.get("envelope_sha256")
        != artifacts.get("provider-raw-envelope.json", {}).get("sha256")
    ):
        raise ValueError("legacy provider replay binding is invalid")


def _bound_bronze(
    *,
    directory: Path,
    artifacts: dict[str, object],
) -> tuple[Path, bytes, dict[str, object]]:
    bronze_path = assert_absolute_path_has_no_links(directory / "bronze.parquet")
    bronze_payload = read_stable_single_link_file(
        bronze_path,
        label="audited provider bronze parquet",
        max_bytes=256 * 1024 * 1024,
    )
    declared = artifacts.get("bronze.parquet")
    actual = {"sha256": _sha256(bronze_payload), "size_bytes": len(bronze_payload)}
    if declared != actual:
        raise ValueError("audited provider bronze differs from its audit JSON")
    return bronze_path, bronze_payload, actual


def _capture_audit_evidence(
    *,
    audit: dict[str, object],
    audit_path: Path,
    audit_payload: bytes,
    receipt_path: Path,
    receipt_payload: bytes,
) -> tuple[
    dict[Path, bytes],
    dict[Path, frozenset[str]],
    dict[str, bytes],
]:
    directory = _absolute_no_links(str(audit.get("acquisition_directory", "")))
    if paths_overlap_by_identity(audit_path, directory) or paths_overlap_by_identity(
        receipt_path,
        directory,
    ):
        raise ValueError("audit or verifier receipt overlaps acquisition evidence")
    inventory = frozenset(path.name for path in directory.iterdir())
    if inventory != _PROVIDER_INVENTORY:
        raise ValueError("audited acquisition directory inventory is not exact")
    files: dict[Path, bytes] = {
        audit_path: audit_payload,
        receipt_path: receipt_payload,
    }
    acquisition_payloads: dict[str, bytes] = {}
    for name in sorted(_PROVIDER_INVENTORY):
        path = assert_absolute_path_has_no_links(directory / name)
        payload = read_stable_single_link_file(
            path,
            label=f"panel source acquisition {name}",
            max_bytes=_MAX_PROVIDER_BYTES[name],
        )
        files[path] = payload
        acquisition_payloads[name] = payload
    if audit.get("schema_version") == _FRESH_AUDIT_SCHEMA_VERSION:
        manifest_sha256 = audit.get("manifest_sha256")
        artifacts = audit.get("artifacts")
    else:
        manifest_record = audit.get("legacy_manifest")
        prior_record = audit.get("legacy_audit")
        if not isinstance(manifest_record, dict) or not isinstance(prior_record, dict):
            raise ValueError("legacy reclassification evidence bindings are invalid")
        manifest_sha256 = manifest_record.get("sha256")
        artifacts = audit.get("provider_artifacts")
        prior_path = _absolute_no_links(str(prior_record.get("path", "")))
        prior_payload = read_stable_single_link_file(
            prior_path,
            label="panel source legacy audit",
            max_bytes=4 * 1024 * 1024,
        )
        if _sha256(prior_payload) != prior_record.get("sha256"):
            raise ValueError("legacy audit changed during panel admission")
        files[prior_path] = prior_payload
    if _sha256(acquisition_payloads["acquisition-build-manifest.json"]) != (manifest_sha256):
        raise ValueError("acquisition manifest changed during panel admission")
    if not isinstance(artifacts, dict) or set(artifacts) != (
        _PROVIDER_INVENTORY - {"acquisition-build-manifest.json"}
    ):
        raise ValueError("audited acquisition artifact inventory is not exact")
    for name, record in artifacts.items():
        payload = acquisition_payloads[name]
        actual = {"sha256": _sha256(payload), "size_bytes": len(payload)}
        if not _json_equal(record, actual):
            raise ValueError("acquisition artifact changed during panel admission")
    return files, {directory: inventory}, acquisition_payloads


def _assert_evidence_unchanged(
    *,
    files: dict[Path, bytes],
    directories: dict[Path, frozenset[str]],
) -> None:
    for directory, expected in directories.items():
        if frozenset(path.name for path in directory.iterdir()) != expected:
            raise ValueError("panel source evidence directory changed during build")
    for path, payload in files.items():
        try:
            current = read_stable_single_link_file(
                path,
                label="panel source evidence revalidation",
                max_bytes=len(payload),
            )
        except (OSError, ValueError) as exc:
            raise ValueError("panel source evidence changed during build") from exc
        if current != payload:
            raise ValueError("panel source evidence changed during build")


def _write_exact_retry(path: Path, payload: bytes, *, label: str) -> None:
    if path.exists():
        existing = read_stable_single_link_file(
            path,
            label=f"existing {label}",
            max_bytes=len(payload),
        )
        if existing != payload:
            raise FileExistsError(f"refusing to replace divergent {label}: {path}")
        return
    write_durable_exact_bytes(path, payload, label=label)


def _absolute_no_links(path: str | Path) -> Path:
    selected = Path(path).expanduser()
    if not selected.is_absolute():
        raise ValueError("path must be absolute")
    return assert_absolute_path_has_no_links(Path(os.path.abspath(selected)))


def _canonical_hash(value: str, *, label: str) -> str:
    selected = str(value).strip().lower()
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"{label} SHA-256 is invalid")
    return selected


def _utc_timestamp(value: str, *, label: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed.tz_convert("UTC")


def _iso(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _strict_json(payload: bytes) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {value}")
        ),
    )


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_equal(left: object, right: object) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _strict_int_equals(value: object, expected: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value == expected


def _strict_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} is invalid")
    return value


def _expected_evidence_scope(resolution: dict[str, object]) -> dict[str, object]:
    source_count = _strict_positive_int(
        resolution.get("source_observation_count"),
        label="source observation count",
    )
    output_count = _strict_positive_int(
        resolution.get("output_observation_count"),
        label="output observation count",
    )
    if output_count < source_count:
        raise ValueError("output observation count is below source count")
    return {
        "capture_episode_count": 1,
        "source_observation_count": source_count,
        "output_observation_count": output_count,
        "independent_information_unit_upper_bound": source_count,
        "proxy_rows_add_independent_information": False,
        "seasonal_regime_coverage": "NOT_ESTABLISHED_BY_SINGLE_CAPTURE_EPISODE",
    }


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_cli_workspace_and_paths(
    args: argparse.Namespace,
    *,
    root: Path = ROOT,
) -> None:
    """Reject direct CLI use outside the canonical repo-local build boundary."""

    expected_root = Path(os.path.abspath(root))
    if os.path.normcase(os.path.abspath(Path.cwd())) != os.path.normcase(str(expected_root)):
        raise ValueError(f"cwd must be exactly {expected_root}")
    observed_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=expected_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if os.path.normcase(os.path.abspath(observed_root)) != os.path.normcase(str(expected_root)):
        raise ValueError(f"Git top-level must be exactly {expected_root}")
    build_root = expected_root / "build"
    build_boundary = os.path.normcase(str(build_root))
    for attribute in _CLI_BUILD_PATH_ATTRIBUTES:
        raw_values = getattr(args, attribute)
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        if not values:
            raise ValueError(f"--{attribute.replace('_', '-')} requires at least one path")
        for value in values:
            selected = Path(os.path.abspath(Path(value).expanduser()))
            try:
                within_build = (
                    os.path.commonpath((os.path.normcase(str(selected)), build_boundary))
                    == build_boundary
                )
            except ValueError:
                within_build = False
            if not within_build:
                option = "--" + attribute.replace("_", "-")
                raise ValueError(f"{option} must remain below canonical repo build/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-parquet", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--audit-json", action="append", type=Path, required=True)
    parser.add_argument("--expected-audit-sha256", action="append", required=True)
    parser.add_argument(
        "--audit-runtime-receipt-json",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-audit-runtime-receipt-sha256",
        action="append",
        required=True,
    )
    parser.add_argument(
        "--verifier-artifact",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-verifier-artifact-sha256",
        action="append",
        required=True,
    )
    parser.add_argument("--start-utc", required=True)
    parser.add_argument("--end-utc", required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    _require_cli_workspace_and_paths(args)
    result = build_panel(
        base_parquet=args.base_parquet,
        expected_base_sha256=args.expected_base_sha256,
        audit_json_paths=args.audit_json,
        expected_audit_sha256=args.expected_audit_sha256,
        audit_runtime_receipt_json_paths=args.audit_runtime_receipt_json,
        expected_audit_runtime_receipt_sha256=(args.expected_audit_runtime_receipt_sha256),
        verifier_artifact_paths=args.verifier_artifact,
        expected_verifier_artifact_sha256=args.expected_verifier_artifact_sha256,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
        output_parquet=args.output_parquet,
        output_manifest=args.output_manifest,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

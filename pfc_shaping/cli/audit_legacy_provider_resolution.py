"""Reclassify a legacy v1 provider quarantine without rewriting its evidence.

This audit is intentionally narrow. It proves whether current allow-listed
provider parsing reproduces the legacy bronze values and records the cadence
that was present in the provider bytes. It never upgrades the unsigned legacy
quarantine, its old audit, or its authority status.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Mapping

import pandas as pd

from pfc_shaping.build_identity import SOURCE_REVISION
from pfc_shaping.data.governed_lt_acquisition import (
    OBSERVATION_RESOLUTION_PROVENANCE_ATTR,
    approved_provider_parser_payload_for_role,
    energy_price_resolution_provenance,
    transform_provider_raw,
)
from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.exit_codes import EXIT_INTEGRITY_FAILURE
from pfc_shaping.parquet_safety import validate_parquet_allocation_budget
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    paths_overlap_by_identity,
    read_stable_single_link_file,
)

SCHEMA_VERSION = "legacy_provider_resolution_reclassification.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INVENTORY = {
    "acquisition-build-manifest.json",
    "bronze.parquet",
    "provider-parser.py",
    "provider-raw-envelope.json",
    "provider-transform-config.json",
}
_MAX_BYTES_BY_NAME = {
    "acquisition-build-manifest.json": 2 * 1024 * 1024,
    "bronze.parquet": 256 * 1024 * 1024,
    "provider-parser.py": 2 * 1024 * 1024,
    "provider-raw-envelope.json": 96 * 1024 * 1024,
    "provider-transform-config.json": 2 * 1024 * 1024,
}
_LEGACY_MANIFEST_KEYS = {
    "schema_version",
    "acquisition_id",
    "source_role",
    "source_system",
    "source_locator",
    "received_at_utc",
    "capture_spec",
    "artifacts",
    "bronze_frame_sha256",
    "signing_status",
    "publication_status",
    "namespace_security_status",
    "authority_handoff_status",
    "durability_classification",
    "build_id",
}
_PRIOR_AUDIT_KEYS = {
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


def audit_legacy_resolution(
    *,
    acquisition_directory: Path,
    expected_manifest_sha256: str,
    prior_audit_path: Path,
    expected_prior_audit_sha256: str,
    output_json: Path,
) -> dict[str, object]:
    """Emit reclassification without making any runtime-authority claim."""

    expected_manifest = _canonical_hash(
        expected_manifest_sha256,
        label="legacy acquisition manifest",
    )
    expected_prior_audit = _canonical_hash(
        expected_prior_audit_sha256,
        label="legacy acquisition audit",
    )
    directory = _absolute_no_links(acquisition_directory)
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != _INVENTORY:
        raise ValueError("legacy acquisition directory inventory is not exact")
    payloads = {
        name: read_stable_single_link_file(
            directory / name,
            label=f"legacy acquisition {name}",
            max_bytes=_MAX_BYTES_BY_NAME[name],
        )
        for name in sorted(_INVENTORY)
    }
    manifest_payload = payloads["acquisition-build-manifest.json"]
    if _sha256(manifest_payload) != expected_manifest:
        raise ValueError("legacy manifest differs from the caller-held SHA-256")
    manifest = _strict_mapping(manifest_payload, label="legacy acquisition manifest")
    if set(manifest) != _LEGACY_MANIFEST_KEYS:
        raise ValueError("legacy acquisition manifest fields are not exact")
    if manifest.get("schema_version") != "lt_provider_acquisition_build.v1":
        raise ValueError("legacy acquisition manifest schema is unsupported")
    unsigned = dict(manifest)
    build_id = str(unsigned.pop("build_id", ""))
    if build_id != _sha256(_canonical_json(unsigned)):
        raise ValueError("legacy acquisition manifest build_id is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != _INVENTORY - {
        "acquisition-build-manifest.json"
    }:
        raise ValueError("legacy acquisition artifact inventory is not exact")
    records: dict[str, dict[str, object]] = {}
    for name, declared in sorted(artifacts.items()):
        if not isinstance(declared, Mapping) or set(declared) != {
            "sha256",
            "size_bytes",
        }:
            raise ValueError("legacy acquisition artifact record is invalid")
        record = {
            "sha256": _sha256(payloads[name]),
            "size_bytes": len(payloads[name]),
        }
        if dict(declared) != record:
            raise ValueError(f"legacy acquisition artifact differs from manifest: {name}")
        records[name] = record

    prior_path = _absolute_no_links(prior_audit_path)
    prior_payload = read_stable_single_link_file(
        prior_path,
        label="legacy provider acquisition audit",
        max_bytes=4 * 1024 * 1024,
    )
    if _sha256(prior_payload) != expected_prior_audit:
        raise ValueError("legacy audit differs from the caller-held SHA-256")
    prior = _strict_mapping(prior_payload, label="legacy provider acquisition audit")
    if set(prior) != _PRIOR_AUDIT_KEYS:
        raise ValueError("legacy provider acquisition audit fields are not exact")
    if (
        prior.get("schema_version") != "local_provider_acquisition_audit.v1"
        or prior.get("status") != "VERIFIED_LOCAL_QUARANTINE_NOT_PRODUCTION"
        or prior.get("production_authorization") is not False
        or prior.get("promotion_gate") is not False
        or prior.get("manifest_sha256") != expected_manifest
        or _absolute_no_links(str(prior.get("acquisition_directory", ""))) != directory
        or prior.get("source_role") != manifest.get("source_role")
        or prior.get("source_system") != manifest.get("source_system")
        or prior.get("artifacts") != records
    ):
        raise ValueError("legacy provider acquisition audit binding is invalid")

    config = _strict_mapping(
        payloads["provider-transform-config.json"],
        label="legacy provider transform config",
    )
    current = transform_provider_raw(
        envelope_payload=payloads["provider-raw-envelope.json"],
        parser_config=config,
    )
    validate_parquet_allocation_budget(
        payloads["bronze.parquet"],
        label="legacy provider bronze",
        max_rows=4_096,
        max_columns=16,
        max_cells=65_536,
        max_row_groups=2_048,
        allowed_physical_types={"BOOLEAN", "INT32", "INT64", "FLOAT", "DOUBLE"},
    )
    try:
        legacy = pd.read_parquet(BytesIO(payloads["bronze.parquet"]))
    except (OSError, ValueError) as exc:
        raise ValueError("legacy provider bronze is not readable Parquet") from exc
    if OBSERVATION_RESOLUTION_PROVENANCE_ATTR in legacy.attrs:
        raise ValueError("legacy provider bronze already contains resolution provenance")
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
        raise ValueError(
            "current provider replay does not reproduce legacy bronze values"
        ) from exc
    role = str(manifest["source_role"])
    provenance = energy_price_resolution_provenance(current, role=role)
    current_parser = approved_provider_parser_payload_for_role(role)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "LEGACY_RESOLUTION_RECLASSIFIED_LOCAL_NO_GO",
        "production_authorization": False,
        "promotion_gate": False,
        "scientific_admission": False,
        "acquisition_directory": str(directory),
        "legacy_manifest": {
            "path": str(directory / "acquisition-build-manifest.json"),
            "sha256": expected_manifest,
            "schema_version": manifest["schema_version"],
        },
        "legacy_audit": {
            "path": str(prior_path),
            "sha256": expected_prior_audit,
            "schema_version": prior["schema_version"],
            "resolution_authority": False,
        },
        "provider_artifacts": records,
        "current_reclassification_parser_sha256": _sha256(current_parser),
        "audit_runtime": {
            "source_revision": SOURCE_REVISION,
            "admission_status": "DIRECT_FUNCTION_CALL_NOT_RUNTIME_AUTHORITY",
            "dependency_versions": {},
            "runtime_admission": {},
            "provider_transform_config_sha256": records[
                "provider-transform-config.json"
            ]["sha256"],
        },
        "resolution_provenance": provenance,
        "eligibility": {
            "native_hourly_price_evidence": (
                "LOCAL_UNSIGNED_ONLY"
                if provenance["native_hourly_cadence_eligible"] is True
                else "INELIGIBLE"
            ),
            "native_quarter_hour_price_truth": (
                "LOCAL_UNSIGNED_ONLY"
                if provenance["native_quarter_hour_truth_eligible"] is True
                else "INELIGIBLE"
            ),
            "external_admission": "MISSING",
        },
        "supersession_scope": (
            "The v1 audit remains historical local replay evidence but is "
            "superseded for every cadence or native-quarter-hour claim."
        ),
    }
    result_payload = _canonical_json(result) + b"\n"
    destination = _admit_output_path(
        output_json,
        forbidden=(directory, prior_path),
    )
    _assert_evidence_unchanged(
        directory=directory,
        expected=payloads,
        prior_path=prior_path,
        prior_payload=prior_payload,
    )
    _write_exact_retry(
        destination,
        result_payload,
        label="legacy provider resolution reclassification",
    )
    _assert_evidence_unchanged(
        directory=directory,
        expected=payloads,
        prior_path=prior_path,
        prior_payload=prior_payload,
    )
    return result


def _absolute_no_links(path: str | Path) -> Path:
    selected = Path(path).expanduser()
    if not selected.is_absolute():
        raise ValueError("path must be absolute")
    return assert_absolute_path_has_no_links(Path(os.path.abspath(selected)))


def _admit_output_path(
    path: str | Path,
    *,
    forbidden: tuple[Path, ...],
) -> Path:
    destination = _absolute_no_links(path)
    if destination.suffix != ".json":
        raise ValueError("resolution audit output must use a .json suffix")
    if not destination.parent.is_dir():
        raise ValueError("resolution audit output parent must be pre-provisioned")
    for source in forbidden:
        if paths_overlap_by_identity(destination, source):
            raise ValueError("resolution audit output overlaps protected evidence")
    return destination


def _assert_evidence_unchanged(
    *,
    directory: Path,
    expected: dict[str, bytes],
    prior_path: Path,
    prior_payload: bytes,
) -> None:
    if {path.name for path in directory.iterdir()} != _INVENTORY:
        raise ValueError("legacy acquisition directory changed during audit")
    for name, payload in expected.items():
        current = read_stable_single_link_file(
            directory / name,
            label=f"legacy acquisition revalidation {name}",
            max_bytes=_MAX_BYTES_BY_NAME[name],
        )
        if current != payload:
            raise ValueError("legacy acquisition directory changed during audit")
    current_prior = read_stable_single_link_file(
        prior_path,
        label="legacy audit revalidation",
        max_bytes=4 * 1024 * 1024,
    )
    if current_prior != prior_payload:
        raise ValueError("legacy provider acquisition audit changed during audit")


def _write_exact_retry(path: Path, payload: bytes, *, label: str) -> None:
    if path.exists():
        existing = read_stable_single_link_file(
            path,
            label=f"existing {label}",
            max_bytes=4 * 1024 * 1024,
        )
        if existing != payload:
            raise FileExistsError(f"refusing to replace divergent {label}: {path}")
        return
    write_durable_exact_bytes(path, payload, label=label)


def _canonical_hash(value: str, *, label: str) -> str:
    selected = str(value).strip().lower()
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"{label} SHA-256 is invalid")
    return selected


def _strict_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    del argv
    _emit_failure(
        RuntimeError(
            "direct legacy resolution CLI is disabled; use the isolated verifier zipapp"
        ),
        retry_disposition="DO_NOT_RETRY_USE_ISOLATED_VERIFIER_ZIPAPP",
    )
    return EXIT_INTEGRITY_FAILURE


def _emit_failure(exc: Exception, *, retry_disposition: str) -> None:
    print(
        json.dumps(
            {
                "command": "pfc-lt-audit-legacy-resolution",
                "status": "INTEGRITY_FAILURE",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "retry_disposition": retry_disposition,
                "production_authorization": False,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())

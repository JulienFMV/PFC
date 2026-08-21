"""Independently replay and hash-audit one unsigned acquisition quarantine."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

from pfc_shaping.build_identity import SOURCE_REVISION
from pfc_shaping.data.governed_lt_acquisition import verify_provider_raw_replay
from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.exit_codes import EXIT_INTEGRITY_FAILURE
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    paths_overlap_by_identity,
    read_stable_single_link_file,
)

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
_MANIFEST_KEYS = {
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
_AUDIT_SCHEMA_VERSION = "local_provider_acquisition_audit.v4"
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


def audit_quarantine(
    *,
    acquisition_directory: Path,
    expected_manifest_sha256: str,
    output_json: Path,
) -> dict[str, object]:
    """Audit exact bytes without making any runtime-authority claim."""

    if _SHA256.fullmatch(expected_manifest_sha256) is None:
        raise ValueError("expected manifest SHA-256 is invalid")
    directory = _absolute_no_links(acquisition_directory)
    if not directory.is_dir():
        raise ValueError("acquisition directory is unavailable")
    entries = {path.name for path in directory.iterdir()}
    if entries != _INVENTORY:
        raise ValueError("acquisition directory inventory is not exact")
    payloads = {
        name: read_stable_single_link_file(
            directory / name,
            label=f"acquisition quarantine {name}",
            max_bytes=_MAX_BYTES_BY_NAME[name],
        )
        for name in sorted(_INVENTORY)
    }
    actual_manifest_sha256 = _sha256(payloads["acquisition-build-manifest.json"])
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("acquisition manifest differs from the caller-held SHA-256")
    manifest = _strict_json(payloads["acquisition-build-manifest.json"])
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("acquisition manifest fields are not exact")
    if manifest.get("schema_version") != "lt_provider_acquisition_build.v2":
        raise ValueError("acquisition manifest schema is unsupported")
    unsigned = dict(manifest)
    build_id = str(unsigned.pop("build_id", ""))
    if build_id != _sha256(_canonical_json(unsigned)):
        raise ValueError("acquisition manifest build_id is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != _INVENTORY - {
        "acquisition-build-manifest.json"
    }:
        raise ValueError("acquisition artifact inventory is not exact")
    records: dict[str, dict[str, object]] = {}
    for name, declared in sorted(artifacts.items()):
        if not isinstance(declared, dict) or set(declared) != {"sha256", "size_bytes"}:
            raise ValueError("acquisition artifact record is invalid")
        payload = payloads[name]
        record = {"sha256": _sha256(payload), "size_bytes": len(payload)}
        if not _json_equal(record, declared):
            raise ValueError(f"acquisition artifact differs from its manifest: {name}")
        records[name] = record
    if (
        manifest.get("signing_status") != "UNSIGNED_REQUIRES_INDEPENDENT_AUTHORITIES"
        or manifest.get("publication_status") != "NOT_PUBLISHED"
        or manifest.get("namespace_security_status")
        != "BUILDER_MUTABLE_UNTRUSTED_QUARANTINE"
        or manifest.get("authority_handoff_status")
        != "REQUIRES_BUILDER_INACCESSIBLE_COPY_AND_INDEPENDENT_SIGNATURE"
    ):
        raise ValueError("acquisition quarantine overstates its authority")
    config = _strict_json(payloads["provider-transform-config.json"])
    if not isinstance(config, dict):
        raise ValueError("provider transform config is invalid")
    replay = verify_provider_raw_replay(
        role=str(manifest["source_role"]),
        source_system=str(manifest["source_system"]),
        envelope_payload=payloads["provider-raw-envelope.json"],
        bronze_payload=payloads["bronze.parquet"],
        parser_payload=payloads["provider-parser.py"],
        parser_code_sha256=records["provider-parser.py"]["sha256"],
        parser_config=config,
    )
    if replay["bronze_frame_sha256"] != manifest["bronze_frame_sha256"]:
        raise ValueError("provider replay semantic hash differs from the build manifest")
    if not _json_equal(
        replay["resolution_provenance"],
        manifest["resolution_provenance"],
    ):
        raise ValueError("provider replay resolution provenance differs from the build manifest")
    resolution = replay["resolution_provenance"]
    if not isinstance(resolution, dict):
        raise ValueError("provider replay resolution provenance is invalid")
    source_observation_count = _strict_positive_int(
        resolution.get("source_observation_count"),
        label="source observation count",
    )
    output_observation_count = _strict_positive_int(
        resolution.get("output_observation_count"),
        label="output observation count",
    )
    if output_observation_count < source_observation_count:
        raise ValueError("provider replay output count is below source count")
    audit = {
        "schema_version": _AUDIT_SCHEMA_VERSION,
        "status": "VERIFIED_LOCAL_QUARANTINE_NOT_PRODUCTION",
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
        "acquisition_directory": str(directory),
        "manifest_sha256": actual_manifest_sha256,
        "source_role": manifest["source_role"],
        "source_system": manifest["source_system"],
        "received_at_utc": manifest["received_at_utc"],
        "resolution_provenance": replay["resolution_provenance"],
        "audit_runtime": {
            "source_revision": SOURCE_REVISION,
            "admission_status": "DIRECT_FUNCTION_CALL_NOT_RUNTIME_AUTHORITY",
            "dependency_versions": {},
            "runtime_admission": {},
            "provider_transform_config_sha256": records[
                "provider-transform-config.json"
            ]["sha256"],
        },
        "negative_authority": {
            "signing_status": manifest["signing_status"],
            "publication_status": manifest["publication_status"],
            "namespace_security_status": manifest["namespace_security_status"],
            "authority_handoff_status": manifest["authority_handoff_status"],
        },
        "authority_gates": {
            "security": {
                "status": "MISSING_EXTERNAL_AUTHORITIES",
                "admitted": False,
                "blocking_controls": list(_SECURITY_BLOCKING_CONTROLS),
            },
            "source_scientific_admission": {
                "status": "MISSING_SOURCE_AUTHORITIES",
                "admitted": False,
                "blocking_evidence": list(
                    _SOURCE_SCIENTIFIC_BLOCKING_EVIDENCE
                ),
            },
            "candidate_readiness": {
                "status": "INSUFFICIENT_CONFIRMATORY_EVIDENCE",
                "admitted": False,
                "blocking_evidence": list(
                    _CANDIDATE_READINESS_BLOCKING_EVIDENCE
                ),
            },
        },
        "evidence_scope": {
            "capture_episode_count": 1,
            "source_observation_count": source_observation_count,
            "output_observation_count": output_observation_count,
            "independent_information_unit_upper_bound": source_observation_count,
            "proxy_rows_add_independent_information": False,
            "seasonal_regime_coverage": (
                "NOT_ESTABLISHED_BY_SINGLE_CAPTURE_EPISODE"
            ),
        },
        "artifacts": records,
        "replay": replay,
    }
    audit_payload = _canonical_json(audit) + b"\n"
    destination = _admit_output_path(output_json, forbidden=(directory,))
    _assert_quarantine_unchanged(directory, payloads)
    _write_exact_retry(
        destination,
        audit_payload,
        label="local provider acquisition audit",
    )
    _assert_quarantine_unchanged(directory, payloads)
    return audit


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
        raise ValueError("audit output must use a .json suffix")
    if not destination.parent.is_dir():
        raise ValueError("audit output parent must be pre-provisioned")
    for source in forbidden:
        if paths_overlap_by_identity(destination, source):
            raise ValueError("audit output overlaps protected evidence")
    return destination


def _assert_quarantine_unchanged(
    directory: Path,
    expected: dict[str, bytes],
) -> None:
    if {path.name for path in directory.iterdir()} != _INVENTORY:
        raise ValueError("acquisition directory changed during audit")
    for name, payload in expected.items():
        current = read_stable_single_link_file(
            directory / name,
            label=f"acquisition quarantine revalidation {name}",
            max_bytes=_MAX_BYTES_BY_NAME[name],
        )
        if current != payload:
            raise ValueError("acquisition directory changed during audit")


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


def _strict_json(payload: bytes) -> object:
    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=lambda pairs: _reject_duplicates(pairs),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {value}")
        ),
    )


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


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


def _strict_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} is invalid")
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    del argv
    _emit_failure(
        RuntimeError(
            "direct provider audit CLI is disabled; use the isolated verifier zipapp"
        ),
        retry_disposition="DO_NOT_RETRY_USE_ISOLATED_VERIFIER_ZIPAPP",
    )
    return EXIT_INTEGRITY_FAILURE


def _emit_failure(exc: Exception, *, retry_disposition: str) -> None:
    print(
        json.dumps(
            {
                "command": "pfc-lt-audit-acquisition",
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

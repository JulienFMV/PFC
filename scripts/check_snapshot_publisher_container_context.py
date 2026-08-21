#!/usr/bin/env python3
"""Validate and seal the exact Windows publisher container build context."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.publisher_package_contract import PUBLISHER_RUNTIME_CONTRACT_PATH
from pfc_shaping.publisher_runtime_admission import (
    dependency_distribution_inventory,
    dependency_tree_inventory,
)
from scripts.build_snapshot_publisher_zipapp import audit_publisher_zipapp

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_NAME = "fmv-lt-snapshot-publisher.pyz"
DEPENDENCY_ROOT_NAME = "publisher-site-packages"
DEPENDENCY_RECEIPT_NAME = "publisher-dependency-closure.json"
CONTAINER_MANIFEST_NAME = "container-build-manifest.json"
OPERATIONS_CONTRACT_NAME = "operations-contract.json"
WHEELHOUSE_STAGING_MANIFEST_NAME = "publisher-container-wheelhouse-manifest.json"
DOCKERFILE = ROOT / "deploy" / "publisher" / "Dockerfile"

_BASE_IMAGE = re.compile(
    r"^[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[0-9]+)?"
    r"(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)+"
    r"(?::[a-z0-9_][a-z0-9_.-]{0,127})?"
    r"@sha256:(?P<digest>[0-9a-f]{64})$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_KEY_BLOCK = re.compile(
    rb"-----BEGIN (?P<label>PRIVATE KEY|ENCRYPTED PRIVATE KEY|RSA PRIVATE KEY|"
    rb"EC PRIVATE KEY|DSA PRIVATE KEY|OPENSSH PRIVATE KEY)-----\r?\n"
    rb"(?P<body>.{1,1048576}?)\r?\n"
    rb"-----END (?P=label)-----",
    re.DOTALL,
)
_OPERATIONS_CONTRACT_SHA256 = (
    "8d1af62295835df056cedb7faab8040ddbf0fdcfdeaba99f5288f066d33bf433"
)
_EXPECTED_ENTRYPOINT = [
    "python.exe",
    "-I",
    "-S",
    "-B",
    r"C:\pfc\runtime\fmv-lt-snapshot-publisher.pyz",
]


def build_container_context_manifest(
    runtime_context: Path,
    *,
    base_image: str,
    output: Path,
    dockerfile: Path = DOCKERFILE,
) -> dict[str, object]:
    """Verify an exact runtime bundle and bind it to a digest-pinned base."""

    base_digest = _validated_base_image(base_image)
    context = _validated_context(runtime_context)
    manifest_path = output.expanduser()
    if manifest_path != context / CONTAINER_MANIFEST_NAME:
        raise ValueError("container manifest must use the canonical context path")

    expected_entries = {
        ARTIFACT_NAME,
        DEPENDENCY_ROOT_NAME,
        DEPENDENCY_RECEIPT_NAME,
        OPERATIONS_CONTRACT_NAME,
        WHEELHOUSE_STAGING_MANIFEST_NAME,
    }
    present = {entry.name for entry in context.iterdir()}
    if CONTAINER_MANIFEST_NAME in present:
        expected_entries.add(CONTAINER_MANIFEST_NAME)
    if present != expected_entries:
        raise ValueError("publisher container context inventory is not exact")

    artifact = context / ARTIFACT_NAME
    dependency_root = context / DEPENDENCY_ROOT_NAME
    receipt_path = context / DEPENDENCY_RECEIPT_NAME
    operations_path = context / OPERATIONS_CONTRACT_NAME
    wheelhouse_manifest_path = context / WHEELHOUSE_STAGING_MANIFEST_NAME
    if (
        not artifact.is_file()
        or not dependency_root.is_dir()
        or not receipt_path.is_file()
        or not operations_path.is_file()
        or not wheelhouse_manifest_path.is_file()
    ):
        raise ValueError("publisher container context types are invalid")

    artifact_audit = audit_publisher_zipapp(artifact)
    receipt_payload = read_stable_single_link_file(
        receipt_path,
        label="publisher container dependency receipt",
    )
    receipt = _strict_canonical_mapping(receipt_payload, label="dependency receipt")
    receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
    if receipt_sha256 != artifact_audit["dependency_closure_receipt_sha256"]:
        raise ValueError("container dependency receipt differs from the artifact")
    if receipt.get("production_authorization") is not False:
        raise ValueError("container dependency receipt cannot authorize production")
    target = receipt.get("target")
    if not isinstance(target, dict) or target.get("platform_tag") != "win_amd64":
        raise ValueError("publisher container requires a win_amd64 closure")

    tree = dependency_tree_inventory(dependency_root)
    if (
        tree.get("tree_sha256") != receipt.get("tree_sha256")
        or tree.get("file_count") != receipt.get("file_count")
        or tree.get("tree_sha256") != artifact_audit["dependency_tree_sha256"]
    ):
        raise ValueError("container dependency tree differs from bound evidence")

    distributions = dependency_distribution_inventory(dependency_root)
    expected_distributions = _receipt_distributions(receipt)
    if distributions != expected_distributions:
        raise ValueError("container dependency distributions differ from the receipt")
    if dependency_tree_inventory(dependency_root) != tree:
        raise ValueError("container dependency tree changed during admission")
    _reject_zipapp_private_key_blocks(
        artifact,
        expected_sha256=str(artifact_audit["artifact_sha256"]),
    )
    _reject_private_key_blocks(
        [
            receipt_path,
            operations_path,
            wheelhouse_manifest_path,
            *sorted(path for path in dependency_root.rglob("*") if path.is_file()),
        ]
    )

    runtime = _artifact_runtime_contract(
        artifact,
        expected_sha256=str(artifact_audit["artifact_sha256"]),
    )
    python = runtime.get("python")
    if (
        not isinstance(python, dict)
        or python.get("platform") != "win32"
        or python.get("pointer_bits") != 64
        or python.get("abi") != "cp311"
    ):
        raise ValueError("container artifact is not bound to CPython 3.11 Windows x64")
    if runtime.get("production_authorization") is not False:
        raise ValueError("container artifact runtime cannot authorize production")

    operations_payload = read_stable_single_link_file(
        operations_path,
        label="publisher container operations contract",
    )
    operations = _strict_operations_contract(operations_payload)
    wheelhouse_manifest_payload = read_stable_single_link_file(
        wheelhouse_manifest_path,
        label="publisher container wheelhouse staging manifest",
    )
    wheelhouse_manifest = _strict_canonical_mapping(
        wheelhouse_manifest_payload,
        label="wheelhouse staging manifest",
    )
    _validate_wheelhouse_staging_manifest(wheelhouse_manifest, receipt)

    dockerfile_payload = read_stable_single_link_file(
        dockerfile.expanduser(),
        label="publisher container Dockerfile",
    )
    result = {
        "schema_version": "fmv_lt_snapshot_publisher_container_context.v1",
        "status": "PASS",
        "platform": "windows/amd64",
        "base_image": base_image,
        "base_image_digest": base_digest,
        "dockerfile_sha256": hashlib.sha256(dockerfile_payload).hexdigest(),
        "operations_contract_sha256": hashlib.sha256(operations_payload).hexdigest(),
        "wheelhouse_staging_manifest_sha256": hashlib.sha256(
            wheelhouse_manifest_payload
        ).hexdigest(),
        "artifact_sha256": artifact_audit["artifact_sha256"],
        "artifact_runtime_contract_sha256": artifact_audit[
            "runtime_contract_sha256"
        ],
        "runtime_executable_sha256": artifact_audit["runtime_executable_sha256"],
        "dependency_receipt_sha256": receipt_sha256,
        "dependency_tree_sha256": tree["tree_sha256"],
        "dependency_file_count": tree["file_count"],
        "dependency_distributions": [
            {"name": name, "version": version}
            for name, version in sorted(distributions.items())
        ],
        "entrypoint": _EXPECTED_ENTRYPOINT,
        "runtime_user": "ContainerUser",
        "required_runtime_network_policy": operations["runtime"][
            "network_requirement"
        ],
        "required_runtime_root_filesystem": operations["runtime"][
            "root_filesystem_requirement"
        ],
        "publisher_bundle_private_key_block_scan_status": (
            "PASS_NO_SYNTACTIC_PEM_OR_OPENSSH_PRIVATE_KEY_BLOCKS"
        ),
        "base_image_secret_scan_status": "MISSING_REQUIRED_EXTERNAL",
        "image_digest_binding": "REQUIRED_EXTERNAL_RELEASE_ATTESTATION",
        "sbom_status": "MISSING_REQUIRED_EXTERNAL",
        "release_attestation_status": "MISSING_REQUIRED_EXTERNAL",
        "production_authorization": False,
    }
    payload = _canonical_json(result)
    write_durable_exact_bytes(
        manifest_path,
        payload,
        label="publisher container build manifest",
    )
    return result


def _validated_base_image(value: str) -> str:
    selected = value
    match = _BASE_IMAGE.fullmatch(selected)
    if match is None or "latest" in selected.split("@", 1)[0].split(":")[-1]:
        raise ValueError("publisher base image must be an exact lowercase sha256 reference")
    return match.group("digest")


def _reject_private_key_blocks(paths: list[Path]) -> None:
    for path in paths:
        payload = read_stable_single_link_file(
            path,
            label="publisher container private-key block scan input",
        )
        if _contains_syntactic_private_key_block(payload):
            raise ValueError("publisher container bundle contains a private-key block")


def _reject_zipapp_private_key_blocks(
    path: Path,
    *,
    expected_sha256: str,
) -> None:
    payload = read_stable_single_link_file(
        path,
        label="publisher container zipapp private-key block scan input",
    )
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("publisher zipapp changed before private-key block scan")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or len(names) != len(
                {name.casefold() for name in names}
            ):
                raise ValueError("publisher zipapp marker scan inventory is ambiguous")
            for info in infos:
                if info.is_dir():
                    continue
                member = archive.read(info)
                if _contains_syntactic_private_key_block(member):
                    raise ValueError(
                        "publisher container zipapp contains a private-key block"
                    )
    except zipfile.BadZipFile as exc:
        raise ValueError("publisher zipapp private-key scan archive is invalid") from exc


def _contains_syntactic_private_key_block(payload: bytes) -> bool:
    for match in _PRIVATE_KEY_BLOCK.finditer(payload):
        body = match.group("body")
        encoded_lines: list[bytes] = []
        body_started = False
        valid = True
        for line in body.splitlines():
            selected = line.strip()
            if not selected:
                continue
            if not body_started and re.fullmatch(
                rb"[A-Za-z0-9-]+:[ -~]+",
                selected,
            ):
                continue
            body_started = True
            if re.fullmatch(rb"[A-Za-z0-9+/]+={0,2}", selected) is None:
                valid = False
                break
            encoded_lines.append(selected)
        if not valid or not encoded_lines:
            continue
        try:
            decoded = base64.b64decode(b"".join(encoded_lines), validate=True)
        except (ValueError, base64.binascii.Error):
            continue
        if len(decoded) >= 32:
            return True
    return False


def _validated_context(path: Path) -> Path:
    selected = path.expanduser()
    if not selected.is_absolute():
        raise ValueError("publisher container context must be absolute")
    selected = assert_absolute_path_has_no_links(selected)
    if not selected.is_dir():
        raise ValueError("publisher container context is unavailable")
    return selected


def _artifact_runtime_contract(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    payload = read_stable_single_link_file(path, label="publisher container artifact")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("publisher artifact changed after its zipapp audit")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            runtime_payload = archive.read(PUBLISHER_RUNTIME_CONTRACT_PATH)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError("publisher artifact runtime contract is unavailable") from exc
    if not payload:
        raise ValueError("publisher container artifact is empty")
    return _strict_canonical_mapping(runtime_payload, label="artifact runtime contract")


def _receipt_distributions(receipt: dict[str, object]) -> dict[str, str]:
    wheels = receipt.get("wheels")
    if not isinstance(wheels, list) or not wheels:
        raise ValueError("publisher dependency receipt wheel inventory is invalid")
    result: dict[str, str] = {}
    for item in wheels:
        if not isinstance(item, dict):
            raise ValueError("publisher dependency receipt wheel entry is invalid")
        name = str(item.get("name", "")).strip().lower().replace("_", "-").replace(".", "-")
        version = str(item.get("version", "")).strip()
        if not name or not version or name in result:
            raise ValueError("publisher dependency receipt distributions are ambiguous")
        result[name] = version
    return result


def _validate_wheelhouse_staging_manifest(
    staging: dict[str, object],
    receipt: dict[str, object],
) -> None:
    if (
        set(staging)
        != {
            "schema_version",
            "status",
            "source_lock",
            "wheel_count",
            "wheels",
            "production_authorization",
        }
        or staging.get("schema_version")
        != "fmv_lt_snapshot_publisher_wheelhouse_staging.v1"
        or staging.get("status") != "PASS"
        or staging.get("production_authorization") is not False
    ):
        raise ValueError("publisher wheelhouse staging manifest policy is invalid")
    source_lock = staging.get("source_lock")
    receipt_lock = receipt.get("source_lock")
    if (
        not isinstance(source_lock, dict)
        or set(source_lock) != {"path", "sha256"}
        or source_lock.get("path") != "uv.lock"
        or _SHA256.fullmatch(str(source_lock.get("sha256", ""))) is None
        or source_lock != receipt_lock
    ):
        raise ValueError("publisher wheelhouse staging lock differs from the receipt")
    receipt_wheels = receipt.get("wheels")
    expected = []
    if isinstance(receipt_wheels, list):
        for item in receipt_wheels:
            if not isinstance(item, dict):
                break
            projected = {
                "name": item.get("name"),
                "version": item.get("version"),
                "filename": item.get("filename"),
                "sha256": item.get("sha256"),
                "size": item.get("size"),
            }
            if (
                not isinstance(projected["name"], str)
                or not projected["name"]
                or not isinstance(projected["version"], str)
                or not projected["version"]
                or not isinstance(projected["filename"], str)
                or not projected["filename"].endswith(".whl")
                or Path(projected["filename"]).name != projected["filename"]
                or _SHA256.fullmatch(str(projected["sha256"])) is None
                or not isinstance(projected["size"], int)
                or projected["size"] <= 0
            ):
                raise ValueError(
                    "publisher wheelhouse staging receipt inventory is invalid"
                )
            expected.append(projected)
    expected.sort(key=lambda item: str(item["name"]))
    if (
        not expected
        or len({str(item["name"]) for item in expected}) != len(expected)
        or staging.get("wheels") != expected
        or staging.get("wheel_count") != len(expected)
    ):
        raise ValueError("publisher wheelhouse staging inventory differs from the receipt")


def _strict_canonical_mapping(payload: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"publisher {label} is invalid JSON") from exc
    if not isinstance(value, dict) or _canonical_json(value) != payload:
        raise ValueError(f"publisher {label} is not canonical JSON")
    return value


def _strict_operations_contract(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("publisher container operations contract is invalid JSON") from exc
    expected_keys = {
        "schema_version",
        "workload_model",
        "platform",
        "runtime",
        "observability",
        "lifecycle",
        "rollback",
        "external_proofs",
        "production_authorization",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("schema_version")
        != "fmv_lt_snapshot_publisher_container_operations.v1"
        or value.get("workload_model") != "ONE_SHOT_GOVERNED_JOB"
        or value.get("platform") != "windows/amd64"
        or value.get("production_authorization") is not False
    ):
        raise ValueError("publisher container operations contract policy is invalid")
    runtime = value.get("runtime")
    if (
        not isinstance(runtime, dict)
        or set(runtime)
        != {
            "user",
            "isolation_mode",
            "bind_mount_principal",
            "root_filesystem_requirement",
            "network_requirement",
            "scratch_mount",
            "input_mount",
            "data_mount",
            "evidence_mount",
            "secrets",
        }
        or runtime.get("user") != "ContainerUser"
        or runtime.get("isolation_mode")
        != "PROCESS_EXPLICIT_HOST_BASE_OSVERSION_COMPATIBILITY_REQUIRED"
        or runtime.get("bind_mount_principal")
        != "CONTAINERUSER_WITH_EXTERNAL_HOST_ACL_ATTESTATION"
        or runtime.get("root_filesystem_requirement") != "DOCKER_RUN_READ_ONLY"
        or runtime.get("network_requirement")
        != "DENY_BY_DEFAULT_PHASE_ENDPOINT_ALLOWLIST_ONLY"
        or runtime.get("scratch_mount")
        != "EXPLICIT_RW_NAMED_OR_BIND_MOUNT_NO_ANONYMOUS_FALLBACK"
        or runtime.get("input_mount") != "EXPLICIT_RO_DIRECTORY_MOUNT"
        or runtime.get("data_mount")
        != "PREPARE_AND_FINALIZE_EXPLICIT_RW_FMV_DATA_ROOT"
        or runtime.get("evidence_mount")
        != "CAS_AND_FINALIZE_EXPLICIT_RW_RECEIPT_OBSERVATION_ROOT"
        or runtime.get("secrets")
        != "READ_ONLY_DIRECTORY_MOUNTS_PRESENT_AT_START_PATHS_DEFERRED_POST_ADMISSION"
    ):
        raise ValueError("publisher container runtime operations policy is invalid")
    observability = value.get("observability")
    required_metrics = {
        "pfc_publisher_job_total",
        "pfc_publisher_admission_duration_seconds",
        "pfc_publisher_admission_slo_breach_total",
        "pfc_publisher_anchor_conflict_total",
        "pfc_publisher_projection_repair_total",
        "pfc_publisher_indeterminate_authority_total",
    }
    if (
        not isinstance(observability, dict)
        or set(observability)
        != {
            "streams",
            "admission_metric_schema",
            "required_metrics",
            "required_alerts",
            "low_cardinality_labels",
            "log_only_high_cardinality_fields",
        }
        or observability.get("streams")
        != "STRUCTURED_JSON_STDOUT_AND_STDERR_NO_SECRET_VALUES"
        or observability.get("admission_metric_schema")
        != "fmv_lt_snapshot_publisher_admission_metric.v1"
        or _strict_string_set(observability.get("required_metrics")) != required_metrics
        or not _strict_string_set(observability.get("required_alerts"))
        or _strict_string_set(observability.get("low_cardinality_labels"))
        != {"phase", "status", "exit_code", "environment"}
        or not _strict_string_set(observability.get("log_only_high_cardinality_fields"))
    ):
        raise ValueError("publisher container observability policy is invalid")
    lifecycle = value.get("lifecycle")
    expected_exit_policy = {
        "0": "SUCCESS_NO_RETRY",
        "2": "USAGE_OR_CONFIGURATION_ERROR_NO_RETRY_UNTIL_CORRECTED",
        "30": "GOVERNANCE_PENDING_NO_AUTOMATIC_RETRY_NEW_AUTHORITY_EVIDENCE_REQUIRED",
        "40": "CAS_CONFLICT_NO_BLIND_RETRY_RECONCILE_FRESH_HEAD_AND_NEW_INTENT",
        "41": "TRANSITION_BUSY_RETRY_EXACT_OPERATION_WITH_BOUNDED_BACKOFF",
        "50": "INTEGRITY_OR_ADMISSION_FAILURE_QUARANTINE_AND_INCIDENT_NO_AUTOMATIC_RETRY",
        "51": "COMMITTED_PROJECTION_REPAIR_OPERATOR_REPAIR_THEN_FINALIZE_SAME_OPERATION_NO_NEW_CAS",
        "52": "INDETERMINATE_LOOKUP_EXACT_OPERATION_BEFORE_BOUNDED_RETRY",
    }
    if (
        not isinstance(lifecycle, dict)
        or set(lifecycle)
        != {
            "restart_policy",
            "exit_policy",
            "retryable_exit_codes",
            "retry_backoff_seconds",
            "maximum_retry_count",
            "maximum_total_attempts",
            "same_operation_id_required",
        }
        or lifecycle.get("restart_policy") != "NEVER"
        or lifecycle.get("exit_policy") != expected_exit_policy
        or lifecycle.get("retryable_exit_codes") != [41, 52]
        or lifecycle.get("retry_backoff_seconds") != [5, 15, 30, 60, 120]
        or lifecycle.get("maximum_retry_count") != 5
        or lifecycle.get("maximum_total_attempts") != 6
        or lifecycle.get("same_operation_id_required") is not True
    ):
        raise ValueError("publisher container lifecycle policy is invalid")
    rollback = value.get("rollback")
    if (
        not isinstance(rollback, dict)
        or set(rollback)
        != {
            "automatic_rollback",
            "image",
            "data_head",
            "data_correction",
            "compatibility_gate",
            "rollback_evidence",
        }
        or rollback.get("automatic_rollback") is not False
        or rollback.get("data_head") != "NEVER_REWIND_EXTERNAL_CAS_HEAD"
        or rollback.get("data_correction")
        != "PUBLISH_NEW_SIGNED_COMPENSATING_GENERATION"
        or rollback.get("compatibility_gate")
        != "SIGNED_TARGET_DIGEST_PROTOCOL_SCHEMA_TRUST_REGISTRY_DOMAIN_AND_FRESH_CAS_HEAD_BINDING_REQUIRED"
        or "ATTESTED_IMAGE_DIGEST" not in str(rollback.get("image", ""))
    ):
        raise ValueError("publisher container rollback policy is invalid")
    external = value.get("external_proofs")
    if len(_strict_string_set(external)) < 5:
        raise ValueError("publisher container external proof inventory is invalid")
    if hashlib.sha256(payload).hexdigest() != _OPERATIONS_CONTRACT_SHA256:
        raise ValueError("publisher container operations contract bytes differ")
    return value


def _strict_string_set(value: object) -> set[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        return set()
    return set(value)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("publisher JSON object contains a duplicate key")
        result[key] = value
    return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-context", type=Path, required=True)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_container_context_manifest(
        args.runtime_context,
        base_image=args.base_image,
        output=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

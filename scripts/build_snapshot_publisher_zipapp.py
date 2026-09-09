#!/usr/bin/env python3
"""Build and audit the deterministic request-signer-only publisher zipapp."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.publisher_package_contract import (
    PUBLISHER_ARTIFACT_SCHEMA,
    PUBLISHER_ARTIFACT_TIMESTAMP,
    PUBLISHER_EMBEDDED_CONTRACT_FILES,
    PUBLISHER_ENVIRONMENT_CONTRACT_PATH,
    PUBLISHER_FORBIDDEN_FILES,
    PUBLISHER_MANIFEST_PATH,
    PUBLISHER_RUNTIME_CONTRACT_PATH,
    PUBLISHER_RUNTIME_PYTHON_FILES,
    PUBLISHER_SYNTHETIC_FILES,
)
from pfc_shaping.publisher_runtime_admission import (
    PUBLISHER_DEPENDENCY_LOADING_POLICY,
    RUNTIME_DEPENDENCY_ROOT_ENV,
    dependency_distribution_inventory,
    dependency_tree_inventory,
    python_runtime_fingerprint,
)
from scripts.build_snapshot_publisher_runtime_closure import (
    validate_runtime_dependency_closure,
)

ROOT = Path(__file__).resolve().parents[1]
_MAIN = (
    "from pfc_shaping.publisher_runtime_admission import run_admitted_publisher_zipapp\n"
    "raise SystemExit(run_admitted_publisher_zipapp())\n"
).encode("ascii")
_SYNTHETIC = {"__main__.py": _MAIN, "scripts/__init__.py": b""}


def build_publisher_zipapp(
    output: Path,
    dependency_root: Path,
    dependency_receipt: Path,
    wheel_directory: Path,
) -> dict[str, object]:
    target = output.expanduser()
    if not target.is_absolute() or target.suffix != ".pyz":
        raise ValueError("publisher artifact output must be an absolute .pyz path")
    if not target.parent.is_dir():
        raise ValueError("publisher artifact output directory is unavailable")
    source_payloads = {
        name: (ROOT / name).read_bytes() for name in sorted(PUBLISHER_RUNTIME_PYTHON_FILES)
    }
    if PUBLISHER_FORBIDDEN_FILES.intersection(source_payloads):
        raise ValueError("publisher artifact inventory includes a forbidden authority module")
    contract_payloads = {
        artifact_name: (ROOT / source_name).read_bytes()
        for artifact_name, source_name in sorted(PUBLISHER_EMBEDDED_CONTRACT_FILES.items())
    }
    environment_contract, runtime_template = _validate_delivery_contracts(
        contract_payloads,
        require_bound_runtime=False,
    )
    provenance = validate_runtime_dependency_closure(
        wheel_directory,
        dependency_root,
        dependency_receipt,
    )
    runtime_contract = _bind_runtime_contract(
        runtime_template,
        dependency_root,
        provenance,
    )
    contract_payloads[PUBLISHER_RUNTIME_CONTRACT_PATH] = _canonical_json(
        runtime_contract
    )
    environment_contract, runtime_contract = _validate_delivery_contracts(
        contract_payloads,
        require_bound_runtime=True,
    )
    payloads = {**source_payloads, **contract_payloads, **_SYNTHETIC}
    inventory_sha256 = _inventory_sha256(payloads)
    manifest = {
        "schema_version": PUBLISHER_ARTIFACT_SCHEMA,
        "source_inventory_sha256": inventory_sha256,
        "entrypoint": "scripts.publish_governed_lt_input_snapshot:main",
        "authority_class": "SNAPSHOT_PUBLICATION_REQUEST_SIGNER_ONLY",
        "release_attestation": "REQUIRED_EXTERNAL",
        "environment_contract_sha256": hashlib.sha256(
            contract_payloads[PUBLISHER_ENVIRONMENT_CONTRACT_PATH]
        ).hexdigest(),
        "runtime_contract_sha256": hashlib.sha256(
            contract_payloads[PUBLISHER_RUNTIME_CONTRACT_PATH]
        ).hexdigest(),
        "source_lock_sha256": runtime_contract["source_lock"]["sha256"],
        "runtime_delivery_model": runtime_contract["delivery_model"],
        "dependency_tree_sha256": runtime_contract["dependency_closure"][
            "tree_sha256"
        ],
        "dependency_closure_receipt_sha256": runtime_contract[
            "dependency_closure"
        ]["receipt_sha256"],
        "dependency_wheels": runtime_contract["dependency_closure"]["wheels"],
        "runtime_executable_sha256": runtime_contract["python"][
            "executable_sha256"
        ],
        "environment_schema_version": environment_contract["schema_version"],
        "runtime_schema_version": runtime_contract["schema_version"],
        "files": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(payloads.items())
        },
    }
    manifest_payload = _canonical_json(manifest)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".pfc-publisher-build-",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in sorted(payloads.items()):
                _write_member(archive, name, payload)
            _write_member(archive, PUBLISHER_MANIFEST_PATH, manifest_payload)
        write_durable_exact_bytes(
            target,
            temporary.read_bytes(),
            label="snapshot publisher artifact",
        )
    finally:
        temporary.unlink(missing_ok=True)
    return audit_publisher_zipapp(target)


def audit_publisher_zipapp(path: Path) -> dict[str, object]:
    selected = path.expanduser()
    if not selected.is_absolute():
        raise ValueError("publisher artifact is unavailable")
    artifact_payload = _stable_artifact_payload(selected)
    with zipfile.ZipFile(io.BytesIO(artifact_payload)) as archive:
        names = archive.namelist()
        expected = (
            set(PUBLISHER_RUNTIME_PYTHON_FILES)
            | set(PUBLISHER_SYNTHETIC_FILES)
            | set(PUBLISHER_EMBEDDED_CONTRACT_FILES)
            | {PUBLISHER_MANIFEST_PATH}
        )
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("publisher artifact inventory differs from the positive contract")
        if PUBLISHER_FORBIDDEN_FILES.intersection(names):
            raise ValueError("publisher artifact contains a forbidden authority module")
        manifest_payload = archive.read(PUBLISHER_MANIFEST_PATH)
        manifest = json.loads(manifest_payload)
        if _canonical_json(manifest) != manifest_payload:
            raise ValueError("publisher artifact manifest is not canonical JSON")
        if manifest.get("schema_version") != PUBLISHER_ARTIFACT_SCHEMA:
            raise ValueError("publisher artifact manifest schema is invalid")
        expected_hashes = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in sorted(expected - {PUBLISHER_MANIFEST_PATH})
        }
        if manifest.get("files") != expected_hashes:
            raise ValueError("publisher artifact manifest does not bind exact member bytes")
        if manifest.get("source_inventory_sha256") != _inventory_sha256(
            {name: archive.read(name) for name in expected - {PUBLISHER_MANIFEST_PATH}}
        ):
            raise ValueError("publisher artifact source inventory digest is invalid")
        contract_payloads = {
            name: archive.read(name) for name in PUBLISHER_EMBEDDED_CONTRACT_FILES
        }
        environment_contract, runtime_contract = _validate_delivery_contracts(
            contract_payloads,
            validate_source_lock=False,
        )
        if manifest.get("environment_contract_sha256") != hashlib.sha256(
            contract_payloads[PUBLISHER_ENVIRONMENT_CONTRACT_PATH]
        ).hexdigest():
            raise ValueError("publisher environment contract digest is invalid")
        if manifest.get("runtime_contract_sha256") != hashlib.sha256(
            contract_payloads[PUBLISHER_RUNTIME_CONTRACT_PATH]
        ).hexdigest():
            raise ValueError("publisher runtime contract digest is invalid")
        if manifest.get("source_lock_sha256") != runtime_contract["source_lock"]["sha256"]:
            raise ValueError("publisher source lock digest binding is invalid")
        if manifest.get("runtime_delivery_model") != runtime_contract["delivery_model"]:
            raise ValueError("publisher runtime delivery model binding is invalid")
        if manifest.get("dependency_tree_sha256") != runtime_contract[
            "dependency_closure"
        ]["tree_sha256"]:
            raise ValueError("publisher dependency tree binding is invalid")
        if manifest.get("dependency_closure_receipt_sha256") != runtime_contract[
            "dependency_closure"
        ]["receipt_sha256"]:
            raise ValueError("publisher dependency receipt binding is invalid")
        if manifest.get("dependency_wheels") != runtime_contract[
            "dependency_closure"
        ]["wheels"]:
            raise ValueError("publisher dependency wheel binding is invalid")
        if manifest.get("runtime_executable_sha256") != runtime_contract["python"][
            "executable_sha256"
        ]:
            raise ValueError("publisher interpreter binding is invalid")
        if manifest.get("environment_schema_version") != environment_contract["schema_version"]:
            raise ValueError("publisher environment schema binding is invalid")
        if manifest.get("runtime_schema_version") != runtime_contract["schema_version"]:
            raise ValueError("publisher runtime schema binding is invalid")
        signing_symbols = sorted(
            line.strip().split("(", 1)[0].removeprefix("def ")
            for name in PUBLISHER_RUNTIME_PYTHON_FILES
            for line in archive.read(name).decode("utf-8").splitlines()
            if line.startswith("def sign_")
        )
        if signing_symbols != ["sign_snapshot_publication_intent"]:
            raise ValueError("publisher artifact contains a foreign signing capability")
        if any(info.date_time != PUBLISHER_ARTIFACT_TIMESTAMP for info in archive.infolist()):
            raise ValueError("publisher artifact timestamps are not reproducible")
    return {
        "schema_version": "fmv_lt_snapshot_publisher_artifact_audit.v1",
        "status": "PASS",
        "artifact_path": str(selected),
        "artifact_sha256": hashlib.sha256(artifact_payload).hexdigest(),
        "source_inventory_sha256": manifest["source_inventory_sha256"],
        "environment_contract_sha256": manifest["environment_contract_sha256"],
        "runtime_contract_sha256": manifest["runtime_contract_sha256"],
        "source_lock_sha256": manifest["source_lock_sha256"],
        "runtime_delivery_model": manifest["runtime_delivery_model"],
        "dependency_tree_sha256": manifest["dependency_tree_sha256"],
        "dependency_closure_receipt_sha256": manifest[
            "dependency_closure_receipt_sha256"
        ],
        "dependency_wheels": manifest["dependency_wheels"],
        "runtime_executable_sha256": manifest["runtime_executable_sha256"],
        "file_count": len(expected),
        "foreign_authority_signer_present": False,
        "release_attestation_status": "MISSING_REQUIRED_EXTERNAL",
        "production_authorization": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--dependency-root", type=Path, required=True)
    build.add_argument("--dependency-receipt", type=Path, required=True)
    build.add_argument("--wheel-directory", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (
        build_publisher_zipapp(
            args.output,
            args.dependency_root,
            args.dependency_receipt,
            args.wheel_directory,
        )
        if args.command == "build"
        else audit_publisher_zipapp(args.artifact)
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _write_member(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=PUBLISHER_ARTIFACT_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def _inventory_sha256(payloads: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, payload in sorted(payloads.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _stable_artifact_payload(path: Path) -> bytes:
    return read_stable_single_link_file(
        path,
        label="publisher artifact",
    )


def _validate_delivery_contracts(
    payloads: dict[str, bytes],
    *,
    validate_source_lock: bool = True,
    require_bound_runtime: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        environment = json.loads(payloads[PUBLISHER_ENVIRONMENT_CONTRACT_PATH])
        runtime = json.loads(payloads[PUBLISHER_RUNTIME_CONTRACT_PATH])
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("publisher delivery contract is invalid") from exc
    if not isinstance(environment, dict) or environment.get("schema_version") != (
        "fmv_lt_snapshot_publisher_environment.v1"
    ):
        raise ValueError("publisher environment contract is invalid")
    if environment.get("production_authorization") is not False:
        raise ValueError("publisher environment contract cannot authorize production")
    if not isinstance(runtime, dict) or runtime.get("schema_version") != (
        "fmv_lt_snapshot_publisher_runtime.v6"
    ):
        raise ValueError("publisher runtime contract is invalid")
    if runtime.get("artifact_contains_dependencies") is not False:
        raise ValueError("publisher runtime contract misclassifies the zipapp")
    if runtime.get("production_authorization") is not False:
        raise ValueError("publisher runtime contract cannot authorize production")
    if not require_bound_runtime and runtime.get("python") != {
        "implementation": "CPython",
        "version": "3.11.*",
        "abi": "cp311",
        "binding": "INJECT_EXACT_RUNTIME_FINGERPRINT_AT_BUILD",
    }:
        raise ValueError("publisher Python policy template is invalid")
    if runtime.get("launcher_policy") != {
        "required_invocation": "python.exe -I -S -B artifact.pyz",
        "isolated": True,
        "no_site": True,
        "ignore_python_environment": True,
        "safe_path": True,
        "dont_write_bytecode": True,
    }:
        raise ValueError("publisher isolated launcher contract is invalid")
    if (
        runtime.get("dependency_loading_policy")
        != PUBLISHER_DEPENDENCY_LOADING_POLICY
    ):
        raise ValueError("publisher dependency loading policy is invalid")
    if require_bound_runtime:
        _validate_bound_runtime_contract(runtime)
    source_lock = runtime.get("source_lock")
    if not isinstance(source_lock, dict) or source_lock.get("path") != "uv.lock":
        raise ValueError("publisher runtime source lock is invalid")
    if validate_source_lock:
        lock_payload = (ROOT / "uv.lock").read_bytes()
        if source_lock.get("sha256") != hashlib.sha256(lock_payload).hexdigest():
            raise ValueError("publisher runtime contract does not bind the current uv.lock")
    return environment, runtime


def _bind_runtime_contract(
    template: dict[str, object],
    dependency_root: Path,
    provenance: dict[str, object],
) -> dict[str, object]:
    selected = dependency_root.expanduser()
    tree = dependency_tree_inventory(selected)
    if (
        tree["tree_sha256"] != provenance.get("tree_sha256")
        or tree["file_count"] != provenance.get("file_count")
    ):
        raise ValueError(
            "publisher dependency closure changed after receipt validation"
        )
    installed = dependency_distribution_inventory(selected)
    expected = {
        _normalize_distribution_name(str(item["name"])): str(item["version"])
        for item in template["locked_distributions"]
    }
    if installed != expected:
        raise ValueError(
            f"publisher dependency closure distributions diverge: "
            f"installed={installed}, expected={expected}"
        )
    bound = json.loads(json.dumps(template))
    bound["python"] = python_runtime_fingerprint()
    bound["dependency_closure"] = {
        "root_environment": RUNTIME_DEPENDENCY_ROOT_ENV,
        "tree_sha256": tree["tree_sha256"],
        "file_count": tree["file_count"],
        "distribution_count": len(installed),
        "receipt_sha256": provenance["receipt_sha256"],
        "source_lock_sha256": provenance["source_lock"]["sha256"],
        "wheel_provenance": provenance["wheel_provenance"],
        "extraction_policy": provenance["extraction_policy"],
        "wheels": provenance["wheels"],
        "pth_files": "FORBIDDEN",
        "bytecode_files": "FORBIDDEN",
        "filesystem_links": "FORBIDDEN",
        "extra_files_or_distributions": "FORBIDDEN",
    }
    return bound


def _validate_bound_runtime_contract(runtime: dict[str, object]) -> None:
    python = runtime.get("python")
    expected_python_keys = {
        "implementation",
        "version",
        "abi",
        "cache_tag",
        "platform",
        "machine",
        "pointer_bits",
        "executable_sha256",
        "base_executable_sha256",
        "runtime_binaries",
        "runtime_import_tree",
    }
    if not isinstance(python, dict) or set(python) != expected_python_keys:
        raise ValueError("publisher exact Python runtime is not bound")
    if python.get("implementation") != "CPython" or python.get("abi") != "cp311":
        raise ValueError("publisher exact Python runtime is invalid")
    version = python.get("version")
    if (
        not isinstance(version, str)
        or not version.startswith("3.11.")
        or python.get("cache_tag") != "cpython-311"
    ):
        raise ValueError("publisher exact Python runtime violates the 3.11 policy")
    for key in ("executable_sha256", "base_executable_sha256"):
        value = python.get(key)
        if not _is_sha256(value):
            raise ValueError("publisher Python executable digest is invalid")
    binaries = python.get("runtime_binaries")
    if not isinstance(binaries, list) or not binaries:
        raise ValueError("publisher Python runtime binary inventory is invalid")
    for binary in binaries:
        if (
            not isinstance(binary, dict)
            or set(binary) != {"role", "sha256"}
            or not isinstance(binary.get("role"), str)
            or not _is_sha256(binary.get("sha256"))
        ):
            raise ValueError("publisher Python runtime binary entry is invalid")
    import_tree = python.get("runtime_import_tree")
    if (
        not isinstance(import_tree, dict)
        or set(import_tree) != {"tree_sha256", "file_count"}
        or not _is_sha256(import_tree.get("tree_sha256"))
        or not isinstance(import_tree.get("file_count"), int)
        or isinstance(import_tree.get("file_count"), bool)
        or int(import_tree["file_count"]) < 1
    ):
        raise ValueError("publisher Python runtime import tree is invalid")
    closure = runtime.get("dependency_closure")
    if not isinstance(closure, dict) or set(closure) != {
        "root_environment",
        "tree_sha256",
        "file_count",
        "distribution_count",
        "receipt_sha256",
        "source_lock_sha256",
        "wheel_provenance",
        "extraction_policy",
        "wheels",
        "pth_files",
        "bytecode_files",
        "filesystem_links",
        "extra_files_or_distributions",
    }:
        raise ValueError("publisher dependency closure is not bound")
    if closure.get("root_environment") != RUNTIME_DEPENDENCY_ROOT_ENV:
        raise ValueError("publisher dependency closure root contract is invalid")
    if not _is_sha256(closure.get("tree_sha256")):
        raise ValueError("publisher dependency closure digest is invalid")
    if not isinstance(closure.get("file_count"), int) or closure["file_count"] <= 0:
        raise ValueError("publisher dependency closure file count is invalid")
    if (
        not _is_sha256(closure.get("receipt_sha256"))
        or not _is_sha256(closure.get("source_lock_sha256"))
        or closure.get("source_lock_sha256") != runtime["source_lock"]["sha256"]
        or closure.get("wheel_provenance")
        != "VERIFIED_ARCHIVES_AND_RECORD_AGAINST_UV_LOCK"
        or closure.get("extraction_policy")
        != {
            "pth_files": "FORBIDDEN",
            "recorded_bytecode": "VERIFIED_THEN_EXCLUDED",
            "recorded_nested_wheels": "VERIFIED_THEN_EXCLUDED",
        }
    ):
        raise ValueError("publisher dependency closure provenance is invalid")
    wheels = closure.get("wheels")
    if not isinstance(wheels, list) or len(wheels) != closure.get(
        "distribution_count"
    ):
        raise ValueError("publisher dependency closure wheel inventory is invalid")
    distributions = runtime.get("locked_distributions")
    if not isinstance(distributions, list) or closure.get("distribution_count") != len(
        distributions
    ):
        raise ValueError("publisher dependency closure distribution count is invalid")


def _normalize_distribution_name(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(".", "-")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


if __name__ == "__main__":
    raise SystemExit(main())

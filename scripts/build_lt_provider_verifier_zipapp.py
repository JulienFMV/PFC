#!/usr/bin/env python3
"""Build and audit the deterministic provider-evidence verifier zipapp."""

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
from pfc_shaping.package_contract import (
    BUILD_IDENTITY_PLACEHOLDER,
    BUILD_IDENTITY_TEMPLATE,
)
from pfc_shaping.path_safety import read_stable_single_link_file
from pfc_shaping.publisher_runtime_admission import (
    dependency_distribution_inventory,
    dependency_tree_inventory,
    python_runtime_fingerprint,
)
from pfc_shaping.verifier_package_contract import (
    VERIFIER_ARTIFACT_SCHEMA,
    VERIFIER_ARTIFACT_TIMESTAMP,
    VERIFIER_COMMANDS,
    VERIFIER_EMBEDDED_CONTRACT_FILES,
    VERIFIER_FORBIDDEN_FILES,
    VERIFIER_FROZEN_SOURCE_HASHES,
    VERIFIER_MANIFEST_PATH,
    VERIFIER_RUNTIME_CONTRACT_PATH,
    VERIFIER_RUNTIME_PYTHON_FILES,
    VERIFIER_SYNTHETIC_FILES,
)
from pfc_shaping.verifier_runtime_admission import (
    VERIFIER_DEPENDENCY_LOADING_POLICY,
    VERIFIER_DEPENDENCY_ROOT_ENV,
    VERIFIER_INSTALLATION_POLICY,
    VERIFIER_RUNTIME_KEYS,
    VERIFIER_RUNTIME_SCHEMA,
)
from scripts.build_snapshot_publisher_runtime_closure import (
    validate_runtime_dependency_closure,
)

ROOT = Path(__file__).resolve().parents[1]
_MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
_MAX_MEMBER_UNCOMPRESSED_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAIN = (
    "from pfc_shaping.verifier_runtime_admission import "
    "run_admitted_verifier_zipapp\n"
    "raise SystemExit(run_admitted_verifier_zipapp())\n"
).encode("ascii")


def build_verifier_zipapp(
    output: Path,
    dependency_root: Path,
    dependency_receipt: Path,
    wheel_directory: Path,
) -> dict[str, object]:
    target = output.expanduser()
    if not target.is_absolute() or target.suffix != ".pyz":
        raise ValueError("verifier artifact output must be an absolute .pyz path")
    if not target.parent.is_dir():
        raise ValueError("verifier artifact output directory is unavailable")
    source_payloads = {
        name: (ROOT / name).read_bytes()
        for name in sorted(VERIFIER_RUNTIME_PYTHON_FILES)
    }
    for name, expected_sha256 in VERIFIER_FROZEN_SOURCE_HASHES.items():
        if hashlib.sha256(source_payloads[name]).hexdigest() != expected_sha256:
            raise ValueError(f"verifier frozen source differs: {name}")
    source_payloads["pfc_shaping/build_identity.py"] = BUILD_IDENTITY_TEMPLATE.format(
        source_revision=BUILD_IDENTITY_PLACEHOLDER
    ).encode("ascii")
    source_revision = _runtime_source_revision(source_payloads)
    source_payloads["pfc_shaping/build_identity.py"] = BUILD_IDENTITY_TEMPLATE.format(
        source_revision=source_revision
    ).encode("ascii")
    if VERIFIER_FORBIDDEN_FILES.intersection(source_payloads):
        raise ValueError("verifier artifact inventory includes an authority module")

    provenance = validate_runtime_dependency_closure(
        wheel_directory,
        dependency_root,
        dependency_receipt,
        runtime_contract_path=(
            ROOT / VERIFIER_EMBEDDED_CONTRACT_FILES[VERIFIER_RUNTIME_CONTRACT_PATH]
        ),
    )
    template_payload = (
        ROOT / VERIFIER_EMBEDDED_CONTRACT_FILES[VERIFIER_RUNTIME_CONTRACT_PATH]
    ).read_bytes()
    template = _strict_mapping(template_payload)
    runtime_contract = _bind_runtime_contract(
        template,
        dependency_root=dependency_root,
        provenance=provenance,
    )
    runtime_payload = _canonical_json(runtime_contract)
    payloads = {
        **source_payloads,
        VERIFIER_RUNTIME_CONTRACT_PATH: runtime_payload,
        "__main__.py": _MAIN,
        "scripts/__init__.py": b"",
    }
    manifest = {
        "schema_version": VERIFIER_ARTIFACT_SCHEMA,
        "authority_class": "READ_ONLY_PROVIDER_EVIDENCE_VERIFICATION_LOCAL_NO_GO",
        "commands": sorted(VERIFIER_COMMANDS),
        "source_revision": source_revision,
        "source_inventory_sha256": _inventory_sha256(payloads),
        "runtime_contract_sha256": hashlib.sha256(runtime_payload).hexdigest(),
        "source_lock_sha256": runtime_contract["source_lock"]["sha256"],
        "dependency_tree_sha256": runtime_contract["dependency_closure"][
            "tree_sha256"
        ],
        "dependency_closure_receipt_sha256": runtime_contract[
            "dependency_closure"
        ]["receipt_sha256"],
        "runtime_executable_sha256": runtime_contract["python"][
            "executable_sha256"
        ],
        "files": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(payloads.items())
        },
        "production_authorization": False,
    }
    manifest_payload = _canonical_json(manifest)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".pfc-provider-verifier-build-",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in sorted(payloads.items()):
                _write_member(archive, name, payload)
            _write_member(archive, VERIFIER_MANIFEST_PATH, manifest_payload)
        write_durable_exact_bytes(
            target,
            temporary.read_bytes(),
            label="provider verifier artifact",
        )
    finally:
        temporary.unlink(missing_ok=True)
    return audit_verifier_zipapp(target)


def audit_verifier_zipapp(path: Path) -> dict[str, object]:
    selected = path.expanduser()
    if not selected.is_absolute():
        raise ValueError("verifier artifact path must be absolute")
    artifact_payload = read_stable_single_link_file(
        selected,
        label="provider verifier artifact",
        max_bytes=_MAX_ARTIFACT_BYTES,
    )
    with zipfile.ZipFile(io.BytesIO(artifact_payload)) as archive:
        names = archive.namelist()
        members = archive.infolist()
        if any(
            member.file_size > _MAX_MEMBER_UNCOMPRESSED_BYTES
            for member in members
        ) or sum(member.file_size for member in members) > _MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError("verifier artifact exceeds its decompression budget")
        expected = (
            set(VERIFIER_RUNTIME_PYTHON_FILES)
            | set(VERIFIER_SYNTHETIC_FILES)
            | set(VERIFIER_EMBEDDED_CONTRACT_FILES)
            | {VERIFIER_MANIFEST_PATH}
        )
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("verifier artifact inventory differs from its contract")
        if VERIFIER_FORBIDDEN_FILES.intersection(names):
            raise ValueError("verifier artifact contains an authority module")
        manifest_payload = archive.read(VERIFIER_MANIFEST_PATH)
        manifest = _strict_mapping(manifest_payload)
        if _canonical_json(manifest) != manifest_payload:
            raise ValueError("verifier artifact manifest is not canonical JSON")
        if (
            manifest.get("schema_version") != VERIFIER_ARTIFACT_SCHEMA
            or manifest.get("production_authorization") is not False
            or manifest.get("authority_class")
            != "READ_ONLY_PROVIDER_EVIDENCE_VERIFICATION_LOCAL_NO_GO"
            or manifest.get("commands") != sorted(VERIFIER_COMMANDS)
        ):
            raise ValueError("verifier artifact manifest policy is invalid")
        payloads = {
            name: archive.read(name) for name in expected - {VERIFIER_MANIFEST_PATH}
        }
        hashes = {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(payloads.items())
        }
        if manifest.get("files") != hashes:
            raise ValueError("verifier artifact member binding is invalid")
        if manifest.get("source_inventory_sha256") != _inventory_sha256(payloads):
            raise ValueError("verifier artifact source inventory digest is invalid")
        runtime_payload = payloads[VERIFIER_RUNTIME_CONTRACT_PATH]
        runtime = _strict_mapping(runtime_payload)
        _validate_bound_runtime_contract(runtime)
        if manifest.get("runtime_contract_sha256") != hashlib.sha256(
            runtime_payload
        ).hexdigest():
            raise ValueError("verifier runtime contract binding is invalid")
        python_payloads = {
            name: payloads[name] for name in VERIFIER_RUNTIME_PYTHON_FILES
        }
        source_revision = _runtime_source_revision(python_payloads)
        if manifest.get("source_revision") != source_revision:
            raise ValueError("verifier source revision is invalid")
        embedded_identity = BUILD_IDENTITY_TEMPLATE.format(
            source_revision=source_revision
        ).encode("ascii")
        if python_payloads["pfc_shaping/build_identity.py"] != embedded_identity:
            raise ValueError("verifier embedded source revision diverges")
    return {
        "schema_version": "fmv_lt_provider_verifier_artifact_audit.v1",
        "status": "PASS",
        "artifact": str(selected),
        "artifact_sha256": hashlib.sha256(artifact_payload).hexdigest(),
        "artifact_size_bytes": len(artifact_payload),
        "source_revision": source_revision,
        "dependency_tree_sha256": runtime["dependency_closure"]["tree_sha256"],
        "runtime_schema_version": runtime["schema_version"],
        "member_count": len(names),
        "production_authorization": False,
    }


def _bind_runtime_contract(
    template: dict[str, object],
    *,
    dependency_root: Path,
    provenance: dict[str, object],
) -> dict[str, object]:
    _validate_runtime_template(template)
    tree = dependency_tree_inventory(dependency_root)
    if tree != {
        "tree_sha256": provenance.get("tree_sha256"),
        "file_count": provenance.get("file_count"),
    }:
        raise ValueError("verifier dependency closure changed after validation")
    installed = dependency_distribution_inventory(dependency_root)
    expected = {
        _normalize_name(str(item["name"])): str(item["version"])
        for item in template["locked_distributions"]
    }
    if installed != expected:
        raise ValueError("verifier dependency distributions diverge")
    bound = json.loads(json.dumps(template))
    bound["python"] = python_runtime_fingerprint()
    bound["dependency_closure"] = {
        "root_environment": VERIFIER_DEPENDENCY_ROOT_ENV,
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
    _validate_bound_runtime_contract(bound)
    return bound


def _validate_runtime_template(contract: dict[str, object]) -> None:
    if (
        contract.get("schema_version") != VERIFIER_RUNTIME_SCHEMA
        or contract.get("artifact") != VERIFIER_ARTIFACT_SCHEMA
        or contract.get("artifact_contains_dependencies") is not False
        or contract.get("dependency_loading_policy")
        != VERIFIER_DEPENDENCY_LOADING_POLICY
        or contract.get("production_authorization") is not False
        or contract.get("python")
        != {
            "implementation": "CPython",
            "version": "3.11.*",
            "abi": "cp311",
            "binding": "INJECT_EXACT_RUNTIME_FINGERPRINT_AT_BUILD",
        }
    ):
        raise ValueError("verifier runtime template is invalid")
    lock_payload = (ROOT / "uv.lock").read_bytes()
    if contract.get("source_lock") != {
        "path": "uv.lock",
        "sha256": hashlib.sha256(lock_payload).hexdigest(),
    }:
        raise ValueError("verifier runtime template source lock is invalid")


def _validate_bound_runtime_contract(contract: dict[str, object]) -> None:
    if (
        set(contract) != VERIFIER_RUNTIME_KEYS
        or contract.get("schema_version") != VERIFIER_RUNTIME_SCHEMA
        or contract.get("artifact") != VERIFIER_ARTIFACT_SCHEMA
        or contract.get("artifact_contains_dependencies") is not False
        or contract.get("dependency_loading_policy")
        != VERIFIER_DEPENDENCY_LOADING_POLICY
        or contract.get("production_authorization") is not False
        or contract.get("authority_class")
        != "READ_ONLY_PROVIDER_EVIDENCE_VERIFICATION_LOCAL_NO_GO"
        or contract.get("installation_policy") != VERIFIER_INSTALLATION_POLICY
        or contract.get("launcher_policy")
        != {
            "required_invocation": "python.exe -I -S -B artifact.pyz",
            "isolated": True,
            "no_site": True,
            "ignore_python_environment": True,
            "safe_path": True,
            "dont_write_bytecode": True,
        }
    ):
        raise ValueError("verifier bound runtime contract is invalid")
    closure = contract.get("dependency_closure")
    required_closure = {
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
    }
    if not isinstance(closure, dict) or set(closure) != required_closure:
        raise ValueError("verifier dependency closure binding is invalid")
    if closure.get("root_environment") != VERIFIER_DEPENDENCY_ROOT_ENV:
        raise ValueError("verifier dependency root environment is invalid")
    if contract.get("python") != python_runtime_fingerprint():
        raise ValueError("verifier Python runtime binding diverges")


def _runtime_source_revision(payloads: dict[str, bytes]) -> str:
    normalized = dict(payloads)
    normalized["pfc_shaping/build_identity.py"] = BUILD_IDENTITY_TEMPLATE.format(
        source_revision=BUILD_IDENTITY_PLACEHOLDER
    ).encode("ascii")
    digest = hashlib.sha256()
    for name, payload in sorted(normalized.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _write_member(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, VERIFIER_ARTIFACT_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100444 << 16
    archive.writestr(info, payload)


def _inventory_sha256(payloads: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, payload in sorted(payloads.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _strict_mapping(payload: bytes) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError("verifier JSON payload must be a mapping")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _normalize_name(name: str) -> str:
    import re

    return re.sub(r"[-_.]+", "-", name.strip().lower())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--dependency-root", type=Path)
    parser.add_argument("--dependency-receipt", type=Path)
    parser.add_argument("--wheel-directory", type=Path)
    args = parser.parse_args(argv)
    if args.audit:
        result = audit_verifier_zipapp(args.artifact)
    else:
        if any(
            value is None
            for value in (
                args.dependency_root,
                args.dependency_receipt,
                args.wheel_directory,
            )
        ):
            parser.error("build requires dependency root, receipt and wheel directory")
        result = build_verifier_zipapp(
            args.artifact,
            args.dependency_root,
            args.dependency_receipt,
            args.wheel_directory,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a launcherless local LT runtime from already admitted offline bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    paths_overlap_by_identity,
    read_stable_single_link_file,
)
from pfc_shaping.publisher_runtime_admission import (
    dependency_distribution_inventory,
    dependency_tree_inventory,
)
from scripts.build_repo_local_conda_prefix import (
    validate_repo_local_conda_prefix_build_receipt,
)
from scripts.build_snapshot_publisher_runtime_closure import (
    _canonical_json,
    _is_link_or_reparse,
    _normalize_distribution_name,
    _validate_wheel,
    validate_runtime_dependency_closure,
)
from scripts.check_lt_wheel_contract import audit_wheel

ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_LOCK = ROOT / "uv.lock"
_RUNTIME_CONTRACT = ROOT / "deploy" / "publisher" / "runtime-contract.json"
_SCHEMA = "fmv_lt_launcherless_local_runtime.v5"
_PYTHON_MANIFEST_SCHEMA = "fmv_lt_launcherless_python_runtime_manifest.v1"
_CONDA_PREFIX_RECEIPT_SCHEMA = "fmv_lt_launcherless_conda_prefix_build_receipt.v3"
_CONDA_ARCHIVE_LOCK_SCHEMA = "fmv_lt_launcherless_conda_archive_lock.v2"
_CONDA_PREFIX_RECEIPT_STATUS = "PASS_LOCAL_EXPLICIT_ARCHIVE_REPLAY_NOT_PRODUCTION"
_PROJECT_NAME = "fmv-pfc-lt"
_PROJECT_VERSION = "0.14.0"
_PROJECT_WHEEL_NAME = "fmv_pfc_lt-0.14.0-py3-none-any.whl"
_EXPECTED_PYTHON = "3.11.13"
_CLOSURE_NAME = "governed-site-packages"
_CLOSURE_STAGING_NAME = ".governed-site-packages.staging"
_PTH_NAME = "python311._pth"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CLI_BUILD_PATH_ATTRIBUTES = (
    "runtime_prefix",
    "project_wheel",
    "publisher_wheelhouse",
    "publisher_dependency_root",
    "publisher_receipt",
    "additional_wheel_directory",
    "python_runtime_manifest",
    "conda_prefix_build_receipt",
    "receipt_output",
)


def _stable_regular_file(path: Path, *, label: str) -> tuple[Path, bytes]:
    """Read twice through the shared mono-link descriptor primitive."""

    selected = assert_absolute_path_has_no_links(path)
    return selected, read_stable_single_link_file(selected, label=label)


def _require_cli_workspace_and_paths(
    args: argparse.Namespace,
    *,
    root: Path = ROOT,
) -> None:
    """Fail before assembly unless the CLI is wholly repo-local.

    Library-level tests may exercise the builder with temporary directories,
    but the workstation CLI is intentionally stricter: every mutable input or
    output must live below ``build/`` and the lock must be the canonical repo
    lock. This makes a historical AppData wheelhouse unusable even when the
    caller bypasses ``scripts.run_workspace_local``.
    """

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
        selected = Path(os.path.abspath(getattr(args, attribute).expanduser()))
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

    selected_lock = Path(os.path.abspath(args.lock_path.expanduser()))
    canonical_lock = expected_root / "uv.lock"
    if os.path.normcase(str(selected_lock)) != os.path.normcase(str(canonical_lock)):
        raise ValueError("--lock-path must be the canonical repo uv.lock")


# These are the original locked wheels needed by the governed LT wheel but not
# by the already-audited publisher closure.
_ADDITIONAL_WHEELS = (
    ("holidays", "0.83", "holidays", "0.83-py3-none-any"),
    ("openpyxl", "3.1.5", "openpyxl", "3.1.5-py2.py3-none-any"),
    (
        "scikit-learn",
        "1.6.1",
        "scikit-learn",
        "1.6.1-cp311-cp311-win_amd64",
    ),
    ("scipy", "1.13.1", "scipy", "1.13.1-cp311-cp311-win_amd64"),
    ("joblib", "1.5.3", "joblib", "1.5.3-py3-none-any"),
    ("threadpoolctl", "3.6.0", "threadpoolctl", "3.6.0-py3-none-any"),
    ("et-xmlfile", "2.0.0", "et-xmlfile", "2.0.0-py3-none-any"),
)


def build_launcherless_runtime(
    *,
    runtime_prefix: Path,
    project_wheel: Path,
    publisher_wheelhouse: Path,
    publisher_dependency_root: Path,
    publisher_receipt: Path,
    additional_wheel_directory: Path,
    python_runtime_manifest: Path,
    expected_python_runtime_manifest_sha256: str,
    conda_prefix_build_receipt: Path,
    expected_conda_prefix_build_receipt_sha256: str,
    receipt_output: Path,
    lock_path: Path = _DEFAULT_LOCK,
) -> dict[str, object]:
    """Materialize one exact local runtime without invoking an installer."""

    prefix = _existing_unlinked_directory(runtime_prefix, "runtime prefix")
    python = _stable_existing_file(prefix / "python.exe", "runtime Python")
    closure = prefix / _CLOSURE_NAME
    staging = prefix / _CLOSURE_STAGING_NAME
    pth_path = prefix / _PTH_NAME
    receipt = _new_file(receipt_output, "runtime receipt")
    resume_closure = closure.exists()
    if resume_closure and not closure.is_dir():
        raise ValueError("launcherless runtime closure recovery state is invalid")
    if resume_closure and staging.exists():
        raise ValueError("launcherless runtime recovery state is ambiguous")
    if pth_path.exists():
        _, existing_pth = _stable_regular_file(
            pth_path,
            label="recoverable launcherless python311._pth",
        )
        if existing_pth != _runtime_pth_payload():
            raise ValueError("launcherless runtime ._pth recovery state is invalid")

    python_manifest = validate_python_runtime_manifest(
        runtime_prefix=prefix,
        manifest_path=python_runtime_manifest,
        expected_manifest_sha256=expected_python_runtime_manifest_sha256,
        allowed_extra_roots=(staging.name,) if staging.exists() else (),
    )
    conda_receipt = validate_conda_prefix_build_receipt(
        runtime_prefix=prefix,
        receipt_path=conda_prefix_build_receipt,
        expected_receipt_sha256=expected_conda_prefix_build_receipt_sha256,
        python_runtime_manifest=python_manifest,
        expected_python_runtime_manifest_sha256=(expected_python_runtime_manifest_sha256),
    )
    python_probe = _probe_python(python)
    if python_probe["version"] != _EXPECTED_PYTHON:
        raise ValueError("launcherless runtime requires exact CPython 3.11.13")

    publisher_validation = validate_runtime_dependency_closure(
        publisher_wheelhouse,
        publisher_dependency_root,
        publisher_receipt,
        lock_path=lock_path,
    )
    publisher_expected = _publisher_expected_distributions()
    if dependency_distribution_inventory(publisher_dependency_root) != publisher_expected:
        raise ValueError("publisher dependency closure distribution inventory diverges")

    wheel_path, wheel_payload = _stable_regular_file(
        project_wheel, label="governed LT project wheel"
    )
    if wheel_path.name != _PROJECT_WHEEL_NAME:
        raise ValueError("governed LT project wheel filename is not exact")
    wheel_audit = audit_wheel(wheel_path)
    if wheel_audit.get("status") != "PASS":
        raise ValueError("governed LT project wheel failed its positive contract")
    project = _validate_wheel(
        wheel_payload,
        filename=wheel_path.name,
        expected_name=_normalize_distribution_name(_PROJECT_NAME),
        expected_version=_PROJECT_VERSION,
    )

    lock_file, lock_payload = _stable_regular_file(lock_path, label="source lock")
    lock_sha256 = hashlib.sha256(lock_payload).hexdigest()
    additional_wheels = _existing_unlinked_directory(
        additional_wheel_directory, "additional locked wheel directory"
    )
    copied: dict[str, tuple[str, bytes]] = {}
    sources: list[dict[str, object]] = []
    _copy_tree_into_inventory(
        publisher_dependency_root,
        copied,
        source="publisher-closure",
    )
    copied_publisher = _payload_tree_inventory(copied)
    if copied_publisher != {
        "tree_sha256": publisher_validation["tree_sha256"],
        "file_count": publisher_validation["file_count"],
    }:
        raise ValueError("captured publisher closure differs after validation")
    sources.append(
        {
            "kind": "VERIFIED_PUBLISHER_CLOSURE",
            "tree_sha256": publisher_validation["tree_sha256"],
            "receipt_sha256": publisher_validation["receipt_sha256"],
            "distribution_count": publisher_validation["distribution_count"],
        }
    )

    for name, version, _, cache_key in _ADDITIONAL_WHEELS:
        archive_payloads, source = _validate_additional_wheel(
            wheel_directory=additional_wheels,
            lock_payload=lock_payload,
            name=name,
            version=version,
            cache_key=cache_key,
        )
        _copy_payloads_into_inventory(
            archive_payloads,
            copied,
            source=f"uv-cache:{name}",
        )
        sources.append(source)

    _copy_payloads_into_inventory(
        project["installed_payloads"],
        copied,
        source="project-wheel",
    )
    sources.append(
        {
            "kind": "GOVERNED_PROJECT_WHEEL",
            "name": _PROJECT_NAME,
            "version": _PROJECT_VERSION,
            "filename": wheel_path.name,
            "sha256": hashlib.sha256(wheel_payload).hexdigest(),
            "record_sha256": project["record_sha256"],
            "source_revision": wheel_audit["source_revision"],
        }
    )

    expected = {
        **publisher_expected,
        **{
            _normalize_distribution_name(name): version
            for name, version, _, _ in _ADDITIONAL_WHEELS
        },
        _normalize_distribution_name(_PROJECT_NAME): _PROJECT_VERSION,
    }
    expected_tree = _payload_tree_inventory(copied)
    if resume_closure:
        if dependency_tree_inventory(closure) != expected_tree:
            raise ValueError("recoverable launcherless runtime closure bytes diverge")
    else:
        if staging.exists():
            _existing_unlinked_directory(staging, "launcherless staging closure")
        else:
            staging.mkdir()
        _materialize_inventory_tree(staging, copied)
        if dependency_tree_inventory(staging) != expected_tree:
            raise ValueError("staged launcherless runtime closure bytes diverge")
        staging.replace(closure)
    installed = dependency_distribution_inventory(closure)
    if installed != expected:
        raise ValueError(
            "launcherless runtime distribution inventory diverges: "
            f"installed={installed}, expected={expected}"
        )
    tree = dependency_tree_inventory(closure)
    if not pth_path.exists():
        _write_runtime_pth(pth_path)
    runtime_probe = _probe_governed_runtime(python, closure)
    if runtime_probe["distributions"] != expected:
        raise ValueError("launcherless runtime imported distribution set diverges")
    final_tree = dependency_tree_inventory(closure)
    if final_tree != tree:
        raise ValueError("launcherless closure changed during runtime probe")
    validate_python_runtime_manifest(
        runtime_prefix=prefix,
        manifest_path=python_runtime_manifest,
        expected_manifest_sha256=expected_python_runtime_manifest_sha256,
        allowed_extra_roots=(_CLOSURE_NAME,),
        allowed_extra_files=(_PTH_NAME,),
    )
    final_conda_receipt = validate_conda_prefix_build_receipt(
        runtime_prefix=prefix,
        receipt_path=conda_prefix_build_receipt,
        expected_receipt_sha256=expected_conda_prefix_build_receipt_sha256,
        python_runtime_manifest=python_manifest,
        expected_python_runtime_manifest_sha256=(expected_python_runtime_manifest_sha256),
    )
    if final_conda_receipt != conda_receipt:
        raise ValueError("Conda prefix build receipt changed during assembly")
    final_pth_payload = _validated_runtime_pth_payload(pth_path)
    replayed_tree = dependency_tree_inventory(closure)
    if replayed_tree != final_tree:
        raise ValueError("launcherless closure changed during final receipt admission")

    result = {
        "schema_version": _SCHEMA,
        "status": "PASS",
        "runtime_prefix": str(prefix),
        "python": python_probe,
        "python_runtime_manifest": {
            "path": str(python_runtime_manifest),
            "sha256": expected_python_runtime_manifest_sha256,
            "tree_sha256": python_manifest["tree_sha256"],
            "file_count": python_manifest["file_count"],
        },
        "conda_prefix_build_receipt": conda_receipt,
        "python_dot_pth_sha256": hashlib.sha256(final_pth_payload).hexdigest(),
        "project_source_revision": wheel_audit["source_revision"],
        "source_lock": {
            "path": lock_file.name,
            "sha256": lock_sha256,
        },
        "closure": {
            "path": str(closure),
            "tree_sha256": final_tree["tree_sha256"],
            "file_count": final_tree["file_count"],
            "distribution_count": len(installed),
            "distributions": installed,
        },
        "sys_path": runtime_probe["sys_path"],
        "module_origins": runtime_probe["module_origins"],
        "sources": sources,
        "launcher_policy": {
            "required_flags": ["-I", "-B", "-m"],
            "project_console_launchers": "FORBIDDEN",
            "pip_install": "NOT_USED",
            "pth_files": "FORBIDDEN",
            "python_dot_pth": _PTH_NAME,
        },
        "local_quality_authorization": True,
        "production_authorization": False,
    }
    payload = _canonical_json(result)
    write_durable_exact_bytes(receipt, payload, label="launcherless runtime receipt")
    return {
        **result,
        "receipt_path": str(receipt),
        "receipt_sha256": hashlib.sha256(payload).hexdigest(),
    }


def validate_conda_prefix_build_receipt(
    *,
    runtime_prefix: Path,
    receipt_path: Path,
    expected_receipt_sha256: str,
    python_runtime_manifest: dict[str, object],
    expected_python_runtime_manifest_sha256: str,
    root: Path = ROOT,
) -> dict[str, object]:
    """Bind one archive-locked Conda replay receipt to the base prefix."""

    prefix = _existing_unlinked_directory(runtime_prefix, "runtime prefix")
    if _HEX64.fullmatch(expected_receipt_sha256) is None:
        raise ValueError("expected Conda prefix build receipt SHA-256 is invalid")
    selected, payload = _stable_regular_file(
        assert_absolute_path_has_no_links(receipt_path),
        label="caller-held Conda prefix build receipt",
    )
    if paths_overlap_by_identity(selected, prefix):
        raise ValueError("Conda prefix build receipt must be outside the prefix")
    if hashlib.sha256(payload).hexdigest() != expected_receipt_sha256:
        raise ValueError("Conda prefix build receipt differs from caller-held SHA-256")
    try:
        receipt = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Conda prefix build receipt is invalid") from exc
    if not isinstance(receipt, dict) or _canonical_json(receipt) + b"\n" != payload:
        raise ValueError("Conda prefix build receipt is not canonical JSONL")
    expected_keys = {
        "schema_version",
        "status",
        "archive_lock",
        "explicit_spec",
        "runtime_prefix",
        "conda_history",
        "python_runtime_manifest",
        "target_conda_meta",
        "repo_local_build_receipt",
        "package_count",
        "file_count",
        "archive_verified_file_count",
        "generated_nonruntime_file_count",
        "build_policy",
        "prefix_receipt_id",
        "production_authorization",
        "promotion_gate",
    }
    expected_policy = {
        "solver_used": False,
        "network_required": False,
        "copy_only": True,
        "new_prefix_required": True,
        "target_python_executed_before_manifest": False,
        "dependency_metadata_authority": (
            "ARCHIVE_PATHS_BYTES_EXCEPT_DECLARED_GENERATED_NONRUNTIME"
        ),
    }
    if set(receipt) != expected_keys or (
        receipt.get("schema_version") != _CONDA_PREFIX_RECEIPT_SCHEMA
        or receipt.get("status") != _CONDA_PREFIX_RECEIPT_STATUS
        or receipt.get("runtime_prefix") != str(prefix)
        or receipt.get("build_policy") != expected_policy
        or receipt.get("production_authorization") is not False
        or receipt.get("promotion_gate") is not False
    ):
        raise ValueError("Conda prefix build receipt policy is invalid")
    archive_lock = receipt.get("archive_lock")
    if (
        not isinstance(archive_lock, dict)
        or set(archive_lock)
        != {
            "path",
            "sha256",
            "archive_lock_id",
            "archive_set_id",
        }
        or not all(
            _HEX64.fullmatch(str(archive_lock.get(name, "")))
            for name in ("sha256", "archive_lock_id", "archive_set_id")
        )
    ):
        raise ValueError("Conda prefix archive lock identity is invalid")
    archive_lock_path, archive_lock_payload = _stable_regular_file(
        assert_absolute_path_has_no_links(Path(str(archive_lock["path"]))),
        label="caller-held Conda archive lock",
    )
    if hashlib.sha256(archive_lock_payload).hexdigest() != archive_lock["sha256"]:
        raise ValueError("Conda archive lock differs from its receipt hash")
    try:
        archive_lock_document = json.loads(archive_lock_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Conda archive lock is invalid") from exc
    if (
        not isinstance(archive_lock_document, dict)
        or _canonical_json(archive_lock_document) + b"\n" != archive_lock_payload
        or archive_lock_document.get("schema_version") != _CONDA_ARCHIVE_LOCK_SCHEMA
        or archive_lock_document.get("archive_lock_id") != archive_lock["archive_lock_id"]
        or archive_lock_document.get("archive_set_id") != archive_lock["archive_set_id"]
    ):
        raise ValueError("Conda archive lock identity is not derived")
    from scripts.build_launcherless_conda_archive_lock import (
        _validated_explicit_prefix_records,
        verify_conda_archive_lock,
    )

    replayed_lock = verify_conda_archive_lock(
        lock_path=archive_lock_path,
        expected_sha256=str(archive_lock["sha256"]),
    )
    if replayed_lock != archive_lock_document:
        raise ValueError("Conda archive lock replay changed during admission")

    explicit_spec = receipt.get("explicit_spec")
    if not isinstance(explicit_spec, dict) or set(explicit_spec) != {
        "path",
        "sha256",
        "size",
        "line_count",
    }:
        raise ValueError("Conda explicit spec receipt is invalid")
    _, spec_payload = _stable_regular_file(
        assert_absolute_path_has_no_links(Path(str(explicit_spec["path"]))),
        label="caller-held Conda explicit spec",
    )
    expected_spec = (
        "\n".join(str(item) for item in replayed_lock["explicit_spec_lines"]) + "\n"
    ).encode("ascii")
    if (
        spec_payload != expected_spec
        or hashlib.sha256(spec_payload).hexdigest() != explicit_spec.get("sha256")
        or len(spec_payload) != explicit_spec.get("size")
        or len(replayed_lock["explicit_spec_lines"]) != explicit_spec.get("line_count")
    ):
        raise ValueError("Conda explicit spec is not derived from the archive lock")

    repo_local_receipt = receipt.get("repo_local_build_receipt")
    if (
        not isinstance(repo_local_receipt, dict)
        or set(repo_local_receipt)
        != {
            "path",
            "sha256",
            "schema_version",
            "status",
            "command_sha256",
            "external_guard_sha256",
            "external_guard_unchanged",
        }
        or _HEX64.fullmatch(str(repo_local_receipt.get("sha256", ""))) is None
        or _HEX64.fullmatch(
            str(repo_local_receipt.get("external_guard_sha256", ""))
        )
        is None
        or repo_local_receipt.get("external_guard_unchanged") is not True
    ):
        raise ValueError("repo-local Conda build receipt identity is invalid")
    replayed_repo_local_receipt = validate_repo_local_conda_prefix_build_receipt(
        receipt_path=Path(str(repo_local_receipt["path"])),
        expected_sha256=str(repo_local_receipt["sha256"]),
        runtime_prefix=prefix,
        explicit_spec_path=Path(str(explicit_spec["path"])),
        root=root,
    )
    if replayed_repo_local_receipt != repo_local_receipt:
        raise ValueError("repo-local Conda build receipt binding diverges")

    history_record = receipt.get("conda_history")
    if not isinstance(history_record, dict) or set(history_record) != {
        "path",
        "sha256",
        "size",
    }:
        raise ValueError("Conda history receipt is invalid")
    history_path, history_payload = _stable_regular_file(
        assert_absolute_path_has_no_links(Path(str(history_record["path"]))),
        label="replayed Conda history",
    )
    if (
        os.path.normcase(str(history_path))
        != os.path.normcase(str(prefix / "conda-meta" / "history"))
        or hashlib.sha256(history_payload).hexdigest() != history_record.get("sha256")
        or len(history_payload) != history_record.get("size")
    ):
        raise ValueError("Conda replay history binding is invalid")
    manifest_record = receipt.get("python_runtime_manifest")
    if (
        not isinstance(manifest_record, dict)
        or set(manifest_record)
        != {
            "path",
            "sha256",
            "tree_sha256",
            "file_count",
            "python_executable_sha256",
            "python_dll_sha256",
        }
        or (
            manifest_record.get("sha256") != expected_python_runtime_manifest_sha256
            or manifest_record.get("tree_sha256") != python_runtime_manifest.get("tree_sha256")
            or manifest_record.get("file_count") != python_runtime_manifest.get("file_count")
            or manifest_record.get("python_executable_sha256")
            != python_runtime_manifest.get("python_executable_sha256")
            or manifest_record.get("python_dll_sha256")
            != python_runtime_manifest.get("python_dll_sha256")
            or receipt.get("file_count") != python_runtime_manifest.get("file_count")
        )
    ):
        raise ValueError("Conda prefix Python manifest binding is invalid")
    target_meta = receipt.get("target_conda_meta")
    package_count = receipt.get("package_count")
    target_keys = {
        "filename",
        "conda_meta_filename",
        "conda_meta_sha256",
        "installed_file_count",
        "source_record_depends",
        "target_record_depends",
        "source_record_constrains",
        "target_record_constrains",
        "dependency_metadata_matches_source_record",
        "archive_verified_file_count",
        "archive_verified_tree_sha256",
        "generated_nonruntime_file_count",
        "generated_nonruntime_tree_sha256",
    }
    if (
        not isinstance(package_count, int)
        or isinstance(package_count, bool)
        or package_count <= 0
        or not isinstance(target_meta, list)
        or len(target_meta) != package_count
        or any(
            not isinstance(record, dict)
            or set(record) != target_keys
            or _HEX64.fullmatch(str(record.get("archive_verified_tree_sha256", ""))) is None
            or _HEX64.fullmatch(str(record.get("generated_nonruntime_tree_sha256", ""))) is None
            or not isinstance(record.get("installed_file_count"), int)
            or not isinstance(record.get("archive_verified_file_count"), int)
            or not isinstance(record.get("generated_nonruntime_file_count"), int)
            or record["archive_verified_file_count"] + record["generated_nonruntime_file_count"]
            != record["installed_file_count"]
            for record in target_meta
        )
        or len(
            {
                str(record.get("filename", "")).casefold()
                for record in target_meta
                if isinstance(record, dict)
            }
        )
        != package_count
        or _HEX64.fullmatch(str(receipt.get("prefix_receipt_id", ""))) is None
    ):
        raise ValueError("Conda prefix package evidence is invalid")
    archive_verified_count = sum(
        int(record["archive_verified_file_count"]) for record in target_meta
    )
    generated_count = sum(int(record["generated_nonruntime_file_count"]) for record in target_meta)
    if (
        receipt.get("archive_verified_file_count") != archive_verified_count
        or receipt.get("generated_nonruntime_file_count") != generated_count
        or sum(int(record["installed_file_count"]) for record in target_meta) + package_count + 1
        != receipt.get("file_count")
        or sum(int(package["archive_payload_file_count"]) for package in replayed_lock["packages"])
        != archive_verified_count
    ):
        raise ValueError("Conda prefix aggregate file evidence is invalid")
    replayed_target_meta, expected_prefix_files = _validated_explicit_prefix_records(
        prefix=prefix,
        locked_packages=replayed_lock["packages"],
    )
    manifest_files = python_runtime_manifest.get("files")
    if (
        replayed_target_meta != target_meta
        or not isinstance(manifest_files, list)
        or expected_prefix_files != [str(item.get("path", "")) for item in manifest_files]
    ):
        raise ValueError("Conda prefix archive-byte replay evidence diverges")
    core = {
        "archive_lock_sha256": archive_lock["sha256"],
        "archive_lock_id": archive_lock["archive_lock_id"],
        "archive_set_id": archive_lock["archive_set_id"],
        "explicit_spec_sha256": explicit_spec["sha256"],
        "python_runtime_manifest_sha256": (expected_python_runtime_manifest_sha256),
        "python_runtime_tree_sha256": python_runtime_manifest["tree_sha256"],
        "target_conda_meta_sha256": hashlib.sha256(_canonical_json(target_meta)).hexdigest(),
        "package_count": package_count,
        "file_count": receipt["file_count"],
        "archive_verified_file_count": archive_verified_count,
        "generated_nonruntime_file_count": generated_count,
        "repo_local_build_receipt_sha256": repo_local_receipt["sha256"],
        "repo_local_external_guard_sha256": repo_local_receipt[
            "external_guard_sha256"
        ],
    }
    if hashlib.sha256(_canonical_json(core)).hexdigest() != receipt.get("prefix_receipt_id"):
        raise ValueError("Conda prefix receipt ID is not derived")
    return {
        "path": str(selected),
        "sha256": expected_receipt_sha256,
        "prefix_receipt_id": receipt["prefix_receipt_id"],
        "archive_lock_sha256": archive_lock["sha256"],
        "archive_lock_id": archive_lock["archive_lock_id"],
        "archive_set_id": archive_lock["archive_set_id"],
        "package_count": package_count,
        "file_count": receipt["file_count"],
        "archive_verified_file_count": archive_verified_count,
        "generated_nonruntime_file_count": generated_count,
        "repo_local_build_receipt": repo_local_receipt,
    }


def _materialize_inventory_tree(
    root: Path,
    inventory: dict[str, tuple[str, bytes]],
) -> None:
    for relative, payload in sorted(
        inventory.values(),
        key=lambda item: item[0],
    ):
        destination = root.joinpath(*PurePosixPath(relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            _, existing = _stable_regular_file(
                destination,
                label=f"resumable staged runtime file {relative}",
            )
            if existing != payload:
                raise ValueError(f"resumable staged runtime file diverges: {relative}")
        else:
            with destination.open("xb") as handle:
                handle.write(payload)


def _validate_additional_wheel(
    *,
    wheel_directory: Path,
    lock_payload: bytes,
    name: str,
    version: str,
    cache_key: str,
) -> tuple[dict[str, bytes], dict[str, object]]:
    locked = _locked_wheel(lock_payload, name=name, version=version, cache_key=cache_key)
    wheel_path, wheel_payload = _stable_regular_file(
        wheel_directory / locked["filename"],
        label=f"additional locked wheel {name}",
    )
    actual_sha256 = hashlib.sha256(wheel_payload).hexdigest()
    if actual_sha256 != locked["sha256"]:
        raise ValueError(f"additional wheel differs from uv.lock: {name}")
    validated = _validate_wheel(
        wheel_payload,
        filename=wheel_path.name,
        expected_name=_normalize_distribution_name(name),
        expected_version=version,
    )
    return validated["installed_payloads"], {
        "kind": "ORIGINAL_WHEEL_ARCHIVE_VERIFIED_AGAINST_UV_LOCK",
        "name": name,
        "version": version,
        "filename": wheel_path.name,
        "sha256": actual_sha256,
        "size": len(wheel_payload),
        "record_sha256": validated["record_sha256"],
        "wheel_tags": validated["wheel_tags"],
    }


def _locked_wheel(
    lock_payload: bytes,
    *,
    name: str,
    version: str,
    cache_key: str,
) -> dict[str, str]:
    lock = tomllib.loads(lock_payload.decode("utf-8"))
    expected_filename = f"{cache_key.replace('py2.py3', 'py2.py3')}.whl"
    candidates: list[dict[str, str]] = []
    for package in lock.get("package", []):
        if (
            _normalize_distribution_name(str(package.get("name", "")))
            != _normalize_distribution_name(name)
            or str(package.get("version", "")) != version
        ):
            continue
        for wheel in package.get("wheels", []):
            filename = unquote(PurePosixPath(urlsplit(str(wheel["url"])).path).name)
            raw_hash = str(wheel.get("hash", ""))
            if filename.endswith(f"-{cache_key}.whl") or filename == expected_filename:
                digest = raw_hash.removeprefix("sha256:")
                if _HEX64.fullmatch(digest):
                    candidates.append({"filename": filename, "sha256": digest})
    if len(candidates) != 1:
        raise ValueError(f"uv.lock wheel selection is ambiguous: {name}")
    return candidates[0]


def _copy_tree_into_inventory(
    source_root: Path,
    inventory: dict[str, tuple[str, bytes]],
    *,
    source: str,
) -> None:
    payloads: dict[str, bytes] = {}
    for path in sorted(source_root.rglob("*")):
        stat = path.stat(follow_symlinks=False)
        if _is_link_or_reparse(path, stat):
            raise ValueError(f"launcherless source tree contains a link: {source}")
        if path.is_file():
            relative = path.relative_to(source_root).as_posix()
            _, payloads[relative] = _stable_regular_file(
                path, label=f"launcherless source file {relative}"
            )
    _copy_payloads_into_inventory(payloads, inventory, source=source)


def _copy_payloads_into_inventory(
    payloads: dict[str, bytes],
    inventory: dict[str, tuple[str, bytes]],
    *,
    source: str,
) -> None:
    for relative, payload in payloads.items():
        lowered = relative.lower()
        if lowered.endswith(
            (".pth", ".pyc", ".pyo", ".exe", ".cmd", ".bat", ".ps1")
        ) or "__pycache__" in {part.lower() for part in PurePosixPath(relative).parts}:
            raise ValueError(f"launcherless closure contains a forbidden path: {relative}")
        key = relative.casefold()
        if key in inventory:
            previous = inventory[key][0]
            raise ValueError(
                f"launcherless closure path collision: {previous} vs {relative} ({source})"
            )
        inventory[key] = (relative, payload)


def _payload_tree_inventory(
    inventory: dict[str, tuple[str, bytes]],
) -> dict[str, object]:
    digest = hashlib.sha256()
    for relative, payload in sorted(inventory.values(), key=lambda item: item[0]):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).hexdigest().encode("ascii"))
        digest.update(b"\0")
    return {"tree_sha256": digest.hexdigest(), "file_count": len(inventory)}


def build_python_runtime_manifest(
    *,
    runtime_prefix: Path,
    output: Path,
) -> dict[str, object]:
    """Capture one non-executed, mono-linked Python prefix for later admission."""

    prefix = _existing_unlinked_directory(runtime_prefix, "runtime prefix")
    if (prefix / _CLOSURE_NAME).exists() or (prefix / _PTH_NAME).exists():
        raise ValueError("Python runtime manifest requires an unused prefix")
    output_path = _new_file(output, "Python runtime manifest")
    files, tree = _runtime_prefix_inventory(prefix)
    records = [
        {"path": relative, "sha256": sha256, "size": size} for relative, sha256, size in files
    ]
    by_path = {record["path"]: record for record in records}
    for required in ("python.exe", "python311.dll"):
        if required not in by_path:
            raise ValueError(f"Python runtime is missing {required}")
    python_meta = sorted((prefix / "conda-meta").glob("python-3.11.13-*.json"))
    if len(python_meta) != 1:
        raise ValueError("Python runtime Conda identity is unavailable")
    conda_record = json.loads(python_meta[0].read_text(encoding="utf-8"))
    if conda_record.get("name") != "python" or conda_record.get("version") != _EXPECTED_PYTHON:
        raise ValueError("Python runtime Conda version is not exact")
    result = {
        "schema_version": _PYTHON_MANIFEST_SCHEMA,
        "status": "PASS",
        "runtime_prefix": str(prefix),
        "python_version": _EXPECTED_PYTHON,
        "python_executable_sha256": by_path["python.exe"]["sha256"],
        "python_dll_sha256": by_path["python311.dll"]["sha256"],
        "tree_sha256": tree["tree_sha256"],
        "file_count": tree["file_count"],
        "files": records,
        "capture_policy": "CALLER_HELD_BEFORE_FIRST_TARGET_EXECUTION",
        "production_authorization": False,
    }
    payload = _canonical_json(result)
    write_durable_exact_bytes(output_path, payload, label="Python runtime manifest")
    return {
        **result,
        "manifest_path": str(output_path),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }


def validate_python_runtime_manifest(
    *,
    runtime_prefix: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    allowed_extra_roots: tuple[str, ...] = (),
    allowed_extra_files: tuple[str, ...] = (),
) -> dict[str, object]:
    prefix = _existing_unlinked_directory(runtime_prefix, "runtime prefix")
    if _HEX64.fullmatch(expected_manifest_sha256) is None:
        raise ValueError("expected Python runtime manifest SHA-256 is invalid")
    _, payload = _stable_regular_file(manifest_path, label="caller-held Python runtime manifest")
    if hashlib.sha256(payload).hexdigest() != expected_manifest_sha256:
        raise ValueError("Python runtime manifest differs from caller-held SHA-256")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Python runtime manifest is invalid") from exc
    if not isinstance(manifest, dict) or _canonical_json(manifest) != payload:
        raise ValueError("Python runtime manifest is not canonical JSON")
    expected_keys = {
        "schema_version",
        "status",
        "runtime_prefix",
        "python_version",
        "python_executable_sha256",
        "python_dll_sha256",
        "tree_sha256",
        "file_count",
        "files",
        "capture_policy",
        "production_authorization",
    }
    if set(manifest) != expected_keys or (
        manifest.get("schema_version") != _PYTHON_MANIFEST_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("runtime_prefix") != str(prefix)
        or manifest.get("python_version") != _EXPECTED_PYTHON
        or manifest.get("capture_policy") != "CALLER_HELD_BEFORE_FIRST_TARGET_EXECUTION"
        or manifest.get("production_authorization") is not False
    ):
        raise ValueError("Python runtime manifest policy is invalid")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Python runtime manifest inventory is unavailable")
    expected_files: list[tuple[str, str, int]] = []
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256", "size"}
            or not isinstance(record["path"], str)
            or _HEX64.fullmatch(str(record["sha256"])) is None
            or not isinstance(record["size"], int)
            or record["size"] < 0
        ):
            raise ValueError("Python runtime manifest file record is invalid")
        expected_files.append((record["path"], record["sha256"], record["size"]))
    if expected_files != sorted(expected_files) or len(
        {item[0].casefold() for item in expected_files}
    ) != len(expected_files):
        raise ValueError("Python runtime manifest inventory is ambiguous")
    actual_files, actual_tree = _runtime_prefix_inventory(
        prefix,
        allowed_extra_roots=allowed_extra_roots,
        allowed_extra_files=allowed_extra_files,
    )
    if actual_files != expected_files or actual_tree != {
        "tree_sha256": manifest.get("tree_sha256"),
        "file_count": manifest.get("file_count"),
    }:
        raise ValueError("Python runtime prefix differs from its caller-held manifest")
    by_path = {path: sha256 for path, sha256, _ in actual_files}
    if by_path.get("python.exe") != manifest.get("python_executable_sha256") or by_path.get(
        "python311.dll"
    ) != manifest.get("python_dll_sha256"):
        raise ValueError("Python runtime executable or ABI DLL identity diverges")
    return manifest


def _runtime_prefix_inventory(
    prefix: Path,
    *,
    allowed_extra_roots: tuple[str, ...] = (),
    allowed_extra_files: tuple[str, ...] = (),
) -> tuple[list[tuple[str, str, int]], dict[str, object]]:
    allowed_roots = {item.casefold() for item in allowed_extra_roots}
    allowed_files = {item.casefold() for item in allowed_extra_files}
    files: list[tuple[str, str, int]] = []
    for path in sorted(prefix.rglob("*")):
        relative = path.relative_to(prefix).as_posix()
        first = PurePosixPath(relative).parts[0].casefold()
        if first in allowed_roots or relative.casefold() in allowed_files:
            continue
        path_stat = path.stat(follow_symlinks=False)
        if _is_link_or_reparse(path, path_stat):
            raise ValueError(f"Python runtime prefix contains a link: {relative}")
        if path.is_dir():
            continue
        if not stat.S_ISREG(path_stat.st_mode) or int(path_stat.st_nlink) != 1:
            raise ValueError(f"Python runtime prefix file is not mono-linked: {relative}")
        _, payload = _stable_regular_file(path, label=f"Python runtime file {relative}")
        files.append((relative, hashlib.sha256(payload).hexdigest(), len(payload)))
    files.sort()
    digest = hashlib.sha256()
    for relative, sha256, _ in files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\0")
    return files, {"tree_sha256": digest.hexdigest(), "file_count": len(files)}


def _probe_python(python: Path) -> dict[str, object]:
    code = (
        "import hashlib,json,platform,struct,sys;"
        "p=sys.executable;"
        "print(json.dumps({'executable':p,'version':platform.python_version(),"
        "'implementation':platform.python_implementation(),'pointer_bits':struct.calcsize('P')*8,"
        "'python_sha256':hashlib.sha256(open(p,'rb').read()).hexdigest()}))"
    )
    completed = subprocess.run(
        [str(python), "-I", "-B", "-c", code],
        cwd=str(python.parent),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = json.loads(completed.stdout)
    if (
        result.get("implementation") != "CPython"
        or result.get("pointer_bits") != 64
        or Path(str(result.get("executable", ""))) != python
        or _HEX64.fullmatch(str(result.get("python_sha256", ""))) is None
    ):
        raise ValueError("launcherless runtime Python fingerprint is invalid")
    return result


def _probe_governed_runtime(python: Path, closure: Path) -> dict[str, object]:
    names = sorted(dependency_distribution_inventory(closure))
    modules = (
        "pfc_shaping",
        "pfc_shaping.cli.governed_acquisition_builder",
        "pfc_shaping.cli.audit_ch_lt_compute_runtime",
        "pfc_shaping.cli.audit_ch_lt_compute_runtime_manifest",
        "pfc_shaping.cli.audit_ch_lt_estimand_contract",
        "pfc_shaping.cli.audit_ch_market_time_regime",
        "pfc_shaping.cli.build_ch_lt_prospective_capture_ledger",
        "pfc_shaping.cli.governed_release",
    )
    code = (
        "import importlib,json,sys;from importlib import metadata;"
        f"names={names!r};modules={modules!r};"
        "loaded={name:importlib.import_module(name) for name in modules};"
        "print(json.dumps({'sys_path':sys.path,'distributions':{name:metadata.version(name) "
        "for name in names},'module_origins':{name:loaded[name].__file__ for name in modules},"
        "'flags':{'isolated':sys.flags.isolated,'ignore_environment':sys.flags.ignore_environment,"
        "'no_user_site':sys.flags.no_user_site,'safe_path':sys.flags.safe_path,"
        "'dont_write_bytecode':sys.flags.dont_write_bytecode}}))"
    )
    completed = subprocess.run(
        [str(python), "-I", "-B", "-c", code],
        cwd=str(python.parent / "DLLs"),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads(completed.stdout)
    if not all(result["flags"].values()):
        raise ValueError("launcherless runtime flags are not exact")
    expected_root = str(closure).casefold()
    for origin in result["module_origins"].values():
        if not str(origin).casefold().startswith(expected_root + os.sep.casefold()):
            raise ValueError("launcherless module origin escaped the governed closure")
    expected_path = [
        str(python.parent / "Lib"),
        str(python.parent / "DLLs"),
        str(closure),
    ]
    if [str(item).casefold() for item in result["sys_path"]] != [
        item.casefold() for item in expected_path
    ]:
        raise ValueError(
            "launcherless sys.path is not exact: "
            f"actual={result['sys_path']}, expected={expected_path}"
        )
    return result


def _write_runtime_pth(path: Path) -> None:
    with path.open("xb") as handle:
        handle.write(_runtime_pth_payload())


def _runtime_pth_payload() -> bytes:
    return (f"Lib\nDLLs\n{_CLOSURE_NAME}\n").encode("ascii")


def _publisher_expected_distributions() -> dict[str, str]:
    contract = json.loads(_RUNTIME_CONTRACT.read_text(encoding="utf-8"))
    return {
        _normalize_distribution_name(str(item["name"])): str(item["version"])
        for item in contract["locked_distributions"]
    }


def _existing_unlinked_directory(path: Path, label: str) -> Path:
    try:
        selected = assert_absolute_path_has_no_links(path.expanduser())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} path is invalid") from exc
    if not selected.is_dir():
        raise ValueError(f"{label} is unavailable")
    stat = selected.stat(follow_symlinks=False)
    if _is_link_or_reparse(selected, stat):
        raise ValueError(f"{label} is linked")
    return selected


def _stable_existing_file(path: Path, label: str) -> Path:
    selected, _ = _stable_regular_file(path, label=label)
    return selected


def _validated_runtime_pth_payload(path: Path) -> bytes:
    _, payload = _stable_regular_file(
        path,
        label="launcherless runtime python311._pth",
    )
    if payload != _runtime_pth_payload():
        raise ValueError("launcherless runtime python311._pth bytes diverge")
    return payload


def _new_file(path: Path, label: str) -> Path:
    try:
        selected = assert_absolute_path_has_no_links(path.expanduser())
        parent = assert_absolute_path_has_no_links(selected.parent)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} target is invalid") from exc
    if not parent.is_dir() or selected.exists():
        raise ValueError(f"{label} target is invalid")
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-prefix", type=Path, required=True)
    parser.add_argument("--project-wheel", type=Path, required=True)
    parser.add_argument("--publisher-wheelhouse", type=Path, required=True)
    parser.add_argument("--publisher-dependency-root", type=Path, required=True)
    parser.add_argument("--publisher-receipt", type=Path, required=True)
    parser.add_argument("--additional-wheel-directory", type=Path, required=True)
    parser.add_argument("--python-runtime-manifest", type=Path, required=True)
    parser.add_argument("--expected-python-runtime-manifest-sha256", required=True)
    parser.add_argument("--conda-prefix-build-receipt", type=Path, required=True)
    parser.add_argument("--expected-conda-prefix-build-receipt-sha256", required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, default=_DEFAULT_LOCK)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require_cli_workspace_and_paths(args)
    result = build_launcherless_runtime(
        runtime_prefix=args.runtime_prefix,
        project_wheel=args.project_wheel,
        publisher_wheelhouse=args.publisher_wheelhouse,
        publisher_dependency_root=args.publisher_dependency_root,
        publisher_receipt=args.publisher_receipt,
        additional_wheel_directory=args.additional_wheel_directory,
        python_runtime_manifest=args.python_runtime_manifest,
        expected_python_runtime_manifest_sha256=(args.expected_python_runtime_manifest_sha256),
        conda_prefix_build_receipt=args.conda_prefix_build_receipt,
        expected_conda_prefix_build_receipt_sha256=(
            args.expected_conda_prefix_build_receipt_sha256
        ),
        receipt_output=args.receipt_output,
        lock_path=args.lock_path,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"INTEGRITY_FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

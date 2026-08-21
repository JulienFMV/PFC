from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from pfc_shaping.cli import governed_release as release_cli
from pfc_shaping.pipeline import atomic_promotion
from pfc_shaping.pipeline import governed_release as release_pipeline
from pfc_shaping.pipeline import governed_release_cli_contract as contract
from scripts import build_launcherless_local_runtime as builder

_REAL_ADMISSION = contract._assert_launcherless_runtime_admission
_REAL_PRODUCTION_TRANSITION_GUARD = (
    contract.assert_production_transition_runtime_authorized
)


def test_launcherless_admission_requires_caller_held_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(contract.RUNTIME_RECEIPT_PATH_ENV, raising=False)
    monkeypatch.delenv(contract.RUNTIME_RECEIPT_SHA256_ENV, raising=False)

    with pytest.raises(contract.ReleaseCliIdentityError, match="is required"):
        _REAL_ADMISSION()


def test_local_launcherless_runtime_is_never_production_transition_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract, "assert_installed_runtime_sealed", lambda: None)

    with pytest.raises(
        contract.ReleaseCliIdentityError,
        match="independently signed, IT-admitted production runtime attestation",
    ):
        _REAL_PRODUCTION_TRANSITION_GUARD()


def test_atomic_promote_and_rollback_reject_local_runtime_before_path_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract, "assert_installed_runtime_sealed", lambda: None)
    monkeypatch.setattr(
        atomic_promotion,
        "assert_production_transition_runtime_authorized",
        _REAL_PRODUCTION_TRANSITION_GUARD,
    )

    with pytest.raises(
        atomic_promotion.PromotionError,
        match="launcherless local runtime is not production authority",
    ):
        atomic_promotion.promote_candidate(
            "relative-release-root-must-not-be-read",
            run_id="run-001",
            promotion_receipt_path="relative-receipt-must-not-be-read",
            expected_current_event_id=None,
            operation_id="operation-001",
            release_request_id="request-001",
            operation_created_at_utc="2026-07-28T00:00:00+00:00",
        )

    with pytest.raises(
        atomic_promotion.PromotionError,
        match="launcherless local runtime is not production authority",
    ):
        atomic_promotion.rollback_to_candidate(
            "relative-release-root-must-not-be-read",
            rollback_authorization_path="relative-authorization-must-not-be-read",
        )


def test_high_level_promote_and_rollback_reject_before_path_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract, "assert_installed_runtime_sealed", lambda: None)
    monkeypatch.setattr(
        release_pipeline,
        "assert_production_transition_runtime_authorized",
        _REAL_PRODUCTION_TRANSITION_GUARD,
    )

    with pytest.raises(
        release_pipeline.GovernedReleaseError,
        match="launcherless local runtime is not production authority",
    ):
        release_pipeline.promote_release_request(
            release_root=tmp_path / "release",
            workflow_root=tmp_path / "workflow",
            evidence_root=tmp_path / "evidence",
            run_id="run-001",
            request_id="0" * 64,
        )

    with pytest.raises(
        release_pipeline.GovernedReleaseError,
        match="launcherless local runtime is not production authority",
    ):
        release_pipeline.rollback_release(
            release_root=tmp_path / "release",
            rollback_authorization_path=tmp_path / "authorization.json",
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("command", ["promote", "rollback"])
def test_cli_transition_rejects_before_failure_root_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    monkeypatch.setattr(contract, "assert_installed_runtime_sealed", lambda: None)
    monkeypatch.setattr(
        release_cli,
        "assert_production_transition_runtime_authorized",
        _REAL_PRODUCTION_TRANSITION_GUARD,
    )
    release = tmp_path / "release"
    failure = tmp_path / "failure"
    if command == "promote":
        argv = [
            command,
            "--release-root",
            str(release),
            "--failure-root",
            str(failure),
            "--workflow-root",
            str(tmp_path / "workflow"),
            "--evidence-root",
            str(tmp_path / "evidence"),
            "--run-id",
            "run-001",
            "--request-id",
            "0" * 64,
        ]
    else:
        argv = [
            command,
            "--release-root",
            str(release),
            "--failure-root",
            str(failure),
            "--rollback-authorization",
            str(tmp_path / "authorization.json"),
        ]

    with pytest.raises(
        contract.ReleaseCliIdentityError,
        match="launcherless local runtime is not production authority",
    ):
        release_cli._parse_args(argv)

    assert not release.exists()
    assert not failure.exists()


def test_launcherless_admission_rejects_receipt_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(b"{}")
    monkeypatch.setenv(contract.RUNTIME_RECEIPT_PATH_ENV, str(receipt))
    monkeypatch.setenv(contract.RUNTIME_RECEIPT_SHA256_ENV, "0" * 64)

    with pytest.raises(
        contract.ReleaseCliIdentityError,
        match="differs from the caller-held SHA-256",
    ):
        _REAL_ADMISSION()


def test_launcherless_admission_accepts_explicit_receipt_binding_without_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(b"{}")
    monkeypatch.delenv(contract.RUNTIME_RECEIPT_PATH_ENV, raising=False)
    monkeypatch.delenv(contract.RUNTIME_RECEIPT_SHA256_ENV, raising=False)

    with pytest.raises(
        contract.ReleaseCliIdentityError,
        match="differs from the caller-held SHA-256",
    ):
        _REAL_ADMISSION(
            runtime_receipt_path=receipt,
            expected_runtime_receipt_sha256="0" * 64,
        )


def test_launcherless_admission_rejects_partial_explicit_receipt_binding() -> None:
    with pytest.raises(contract.ReleaseCliIdentityError, match="supplied together"):
        _REAL_ADMISSION(runtime_receipt_path=Path("C:/receipt.json"))


def test_runtime_sys_path_comparison_rejects_mutable_prefix_root(
    tmp_path: Path,
) -> None:
    expected = [
        str(tmp_path / "Lib"),
        str(tmp_path / "DLLs"),
        str(tmp_path / "governed-site-packages"),
    ]

    assert contract._same_path_list(expected, expected)
    assert not contract._same_path_list([str(tmp_path), *expected], expected)


def test_runtime_outside_prefix_check_uses_physical_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, Path]] = []

    def _overlap(first: Path, second: Path) -> bool:
        calls.append((first, second))
        return True

    monkeypatch.setattr(contract, "paths_overlap_by_identity", _overlap)

    path = tmp_path / "alias" / "receipt.json"
    prefix = tmp_path / "runtime"
    assert contract._same_or_nested_runtime_path(path, prefix)
    assert calls == [(path, prefix)]


def test_top_level_runtime_receipt_rejects_physical_prefix_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = Path("C:/runtime")
    receipt_path = Path("C:/caller-held-receipt.json")
    receipt = {
        "schema_version": contract._RUNTIME_RECEIPT_SCHEMA,
        "status": "PASS",
        "runtime_prefix": str(prefix),
        "python": {},
        "python_runtime_manifest": {},
        "conda_prefix_build_receipt": {},
        "python_dot_pth_sha256": "0" * 64,
        "project_source_revision": contract.SOURCE_REVISION,
        "source_lock": {},
        "closure": {},
        "sys_path": [],
        "module_origins": {},
        "sources": [],
        "launcher_policy": {},
        "local_quality_authorization": True,
        "production_authorization": False,
    }
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    monkeypatch.setenv(contract.RUNTIME_RECEIPT_PATH_ENV, str(receipt_path))
    monkeypatch.setenv(
        contract.RUNTIME_RECEIPT_SHA256_ENV,
        hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(
        contract,
        "paths_overlap_by_identity",
        lambda first, second: first == receipt_path and second == prefix,
    )
    monkeypatch.setattr(contract, "_stable_runtime_file", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr(contract, "_runtime_prefix_from_receipt", lambda _receipt: prefix)

    with pytest.raises(
        contract.ReleaseCliIdentityError,
        match="runtime receipt cannot be inside the prefix",
    ):
        _REAL_ADMISSION()


def test_launch_time_conda_receipt_binds_archive_set_and_python_manifest(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "runtime"
    conda_meta = prefix / "conda-meta"
    conda_meta.mkdir(parents=True)
    history_path = conda_meta / "history"
    history_payload = b"replay history\n"
    history_path.write_bytes(history_payload)
    python_manifest_path = tmp_path / "python-manifest.json"
    python_manifest = {
        "tree_sha256": "1" * 64,
        "file_count": 3,
        "python_executable_sha256": "2" * 64,
        "python_dll_sha256": "3" * 64,
    }
    python_manifest_record = {
        "path": str(python_manifest_path),
        "sha256": "4" * 64,
        "tree_sha256": python_manifest["tree_sha256"],
        "file_count": python_manifest["file_count"],
    }
    archive_path = tmp_path / "python-3.11.13-test_0.tar.bz2"
    archive_payload = b"locked Conda archive bytes"
    archive_path.write_bytes(archive_payload)
    archive_sha256 = hashlib.sha256(archive_payload).hexdigest()
    archive_md5 = hashlib.md5(archive_payload, usedforsecurity=False).hexdigest()
    metadata_document = {
        "fn": archive_path.name,
        "sha256": archive_sha256,
        "depends": ["vc >=14", "zlib >=1.2"],
        "constrains": [],
    }
    metadata_path = conda_meta / "python-3.11.13-test_0.json"
    metadata_payload = json.dumps(
        metadata_document,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    metadata_path.write_bytes(metadata_payload)
    package = {
        "name": "python",
        "version": "3.11.13",
        "build": "test_0",
        "build_number": 0,
        "subdir": "noarch",
        "channel": "https://repo.example/conda/win-64",
        "source_url": (
            "https://repo.example/conda/win-64/" + archive_path.name
        ),
        "filename": archive_path.name,
        "md5": archive_md5,
        "sha256": archive_sha256,
        "size": len(archive_payload),
        "depends": ["vc >=14"],
        "constrains": [],
        "conda_meta_filename": metadata_path.name,
        "conda_meta_sha256": "a" * 64,
        "cache_root": str(tmp_path),
        "archive_path": str(archive_path),
        "archive_uri": archive_path.as_uri(),
        "archive_payload_tree_sha256": "b" * 64,
        "archive_payload_file_count": 1,
        "prefix_replacement_file_count": 0,
        "noarch_type": "generic",
    }
    portable_package = {
        key: value
        for key, value in package.items()
        if key not in {"cache_root", "archive_path", "archive_uri"}
    }
    archive_set_id = hashlib.sha256(
        json.dumps(
            [portable_package],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    source_history_sha256 = "c" * 64
    lock_identity = {
        "schema_version": contract._CONDA_ARCHIVE_LOCK_SCHEMA,
        "archive_set_id": archive_set_id,
        "conda_history_sha256": source_history_sha256,
        "package_count": 1,
        "total_archive_bytes": len(archive_payload),
    }
    archive_lock_id = hashlib.sha256(
        json.dumps(
            lock_identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    explicit_lines = ["@EXPLICIT", f"{archive_path.as_uri()}#{archive_md5}"]
    lock = {
        "schema_version": contract._CONDA_ARCHIVE_LOCK_SCHEMA,
        "status": "PASS_LOCAL_ARCHIVE_LOCK_NOT_PRODUCTION",
        "source_runtime_prefix": str(tmp_path / "source-runtime"),
        "package_cache_roots": [str(tmp_path)],
        "conda_history": {
            "path": str(tmp_path / "source-history"),
            "sha256": source_history_sha256,
            "size": 1,
        },
        "packages": [package],
        "package_count": 1,
        "total_archive_bytes": len(archive_payload),
        "archive_set_id": archive_set_id,
        "archive_lock_id": archive_lock_id,
        "explicit_spec_lines": explicit_lines,
        "network_required": False,
        "admin_required": False,
        "production_authorization": False,
        "promotion_gate": False,
    }
    archive_lock_path = tmp_path / "archive-lock.json"
    archive_lock_payload = json.dumps(
        lock,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii") + b"\n"
    archive_lock_path.write_bytes(archive_lock_payload)
    archive_lock_sha256 = hashlib.sha256(archive_lock_payload).hexdigest()
    spec_path = tmp_path / "explicit.txt"
    spec_payload = ("\n".join(explicit_lines) + "\n").encode("ascii")
    spec_path.write_bytes(spec_payload)
    guards = {
        str(tmp_path / "user" / ".conda" / "environments.txt"): {
            "exists": False
        },
        str(tmp_path / "user" / ".condarc"): {"exists": False},
    }
    work_root = tmp_path / "work"
    repo_build_document = {
        "admin_required": False,
        "command_sha256": "a" * 64,
        "conda_python": {
            "path": str(tmp_path / "conda-python.exe"),
            "sha256": "b" * 64,
            "size_bytes": 1,
            "link_count": 1,
        },
        "explicit_spec": {
            "path": str(spec_path),
            "sha256": hashlib.sha256(spec_payload).hexdigest(),
            "size_bytes": len(spec_payload),
        },
        "external_guard_after": guards,
        "external_guard_before": guards,
        "external_guard_unchanged": True,
        "mutable_paths": {
            "home": str(work_root / "home"),
            "conda_pkgs": str(work_root / "conda-pkgs"),
            "conda_envs": str(work_root / "conda-envs"),
            "temp": str(work_root / "temp"),
            "condarc": str(work_root / "condarc.yml"),
        },
        "network_required": False,
        "prefix_ready": True,
        "production_authorization": False,
        "promotion_gate": False,
        "runtime_prefix": str(prefix),
        "schema_version": contract._REPO_LOCAL_CONDA_BUILD_SCHEMA,
        "status": contract._REPO_LOCAL_CONDA_BUILD_STATUS,
        "target_exit_code": 0,
        "target_failure_class": None,
        "target_stderr_sha256": "0" * 64,
        "target_stdout_sha256": "0" * 64,
        "work_root": str(work_root),
    }
    repo_build_path = tmp_path / "repo-local-conda-build.json"
    repo_build_payload = json.dumps(
        repo_build_document, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    repo_build_path.write_bytes(repo_build_payload)
    repo_build_identity = {
        "path": str(repo_build_path),
        "sha256": hashlib.sha256(repo_build_payload).hexdigest(),
        "schema_version": contract._REPO_LOCAL_CONDA_BUILD_SCHEMA,
        "status": contract._REPO_LOCAL_CONDA_BUILD_STATUS,
        "command_sha256": repo_build_document["command_sha256"],
        "external_guard_sha256": hashlib.sha256(
            json.dumps(guards, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "external_guard_unchanged": True,
    }
    target_meta = [
        {
            "filename": archive_path.name,
            "conda_meta_filename": metadata_path.name,
            "conda_meta_sha256": hashlib.sha256(metadata_payload).hexdigest(),
            "installed_file_count": 1,
            "source_record_depends": ["vc >=14"],
            "target_record_depends": ["vc >=14", "zlib >=1.2"],
            "source_record_constrains": [],
            "target_record_constrains": [],
            "dependency_metadata_matches_source_record": False,
            "archive_verified_file_count": 1,
            "archive_verified_tree_sha256": "d" * 64,
            "generated_nonruntime_file_count": 0,
            "generated_nonruntime_tree_sha256": "e" * 64,
        }
    ]
    core = {
        "archive_lock_sha256": archive_lock_sha256,
        "archive_lock_id": archive_lock_id,
        "archive_set_id": archive_set_id,
        "explicit_spec_sha256": hashlib.sha256(spec_payload).hexdigest(),
        "python_runtime_manifest_sha256": python_manifest_record["sha256"],
        "python_runtime_tree_sha256": python_manifest["tree_sha256"],
        "target_conda_meta_sha256": hashlib.sha256(
            json.dumps(
                target_meta,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest(),
        "package_count": 1,
        "file_count": python_manifest["file_count"],
        "archive_verified_file_count": 1,
        "generated_nonruntime_file_count": 0,
        "repo_local_build_receipt_sha256": repo_build_identity["sha256"],
        "repo_local_external_guard_sha256": repo_build_identity[
            "external_guard_sha256"
        ],
    }
    prefix_receipt_id = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    receipt = {
        "schema_version": contract._CONDA_PREFIX_RECEIPT_SCHEMA,
        "status": contract._CONDA_PREFIX_RECEIPT_STATUS,
        "archive_lock": {
            "path": str(archive_lock_path),
            "sha256": archive_lock_sha256,
            "archive_lock_id": archive_lock_id,
            "archive_set_id": archive_set_id,
        },
        "explicit_spec": {
            "path": str(spec_path),
            "sha256": core["explicit_spec_sha256"],
            "size": len(spec_payload),
            "line_count": len(explicit_lines),
        },
        "runtime_prefix": str(prefix),
        "conda_history": {
            "path": str(history_path),
            "sha256": hashlib.sha256(history_payload).hexdigest(),
            "size": len(history_payload),
        },
        "python_runtime_manifest": {
            **python_manifest_record,
            "python_executable_sha256": python_manifest[
                "python_executable_sha256"
            ],
            "python_dll_sha256": python_manifest["python_dll_sha256"],
        },
        "target_conda_meta": target_meta,
        "repo_local_build_receipt": repo_build_identity,
        "package_count": 1,
        "file_count": python_manifest["file_count"],
        "archive_verified_file_count": 1,
        "generated_nonruntime_file_count": 0,
        "build_policy": {
            "solver_used": False,
            "network_required": False,
            "copy_only": True,
            "new_prefix_required": True,
            "target_python_executed_before_manifest": False,
            "dependency_metadata_authority": (
                "ARCHIVE_PATHS_BYTES_EXCEPT_DECLARED_GENERATED_NONRUNTIME"
            ),
        },
        "prefix_receipt_id": prefix_receipt_id,
        "production_authorization": False,
        "promotion_gate": False,
    }
    path = tmp_path / "conda-prefix-receipt.json"
    payload = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii") + b"\n"
    path.write_bytes(payload)
    receipt_record = {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "prefix_receipt_id": prefix_receipt_id,
        "archive_lock_sha256": archive_lock_sha256,
        "archive_lock_id": archive_lock_id,
        "archive_set_id": archive_set_id,
        "package_count": 1,
        "file_count": python_manifest["file_count"],
        "archive_verified_file_count": 1,
        "generated_nonruntime_file_count": 0,
        "repo_local_build_receipt": repo_build_identity,
    }

    contract._validate_conda_prefix_build_receipt(
        prefix=prefix,
        receipt_record=receipt_record,
        python_manifest=python_manifest,
        python_manifest_record=python_manifest_record,
    )

    archive_path.write_bytes(b"tampered")
    with pytest.raises(
        contract.ReleaseCliIdentityError,
        match="retained Conda archive bytes diverge",
    ):
        contract._validate_conda_prefix_build_receipt(
            prefix=prefix,
            receipt_record=receipt_record,
            python_manifest=python_manifest,
            python_manifest_record=python_manifest_record,
        )


def test_launch_time_manifest_rejects_unexpected_prefix_file(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "runtime"
    prefix.mkdir()
    (prefix / "python.exe").write_bytes(b"python")
    (prefix / "python311.dll").write_bytes(b"abi")
    conda_meta = prefix / "conda-meta"
    conda_meta.mkdir()
    (conda_meta / "python-3.11.13-test.json").write_text(
        json.dumps({"name": "python", "version": "3.11.13"}),
        encoding="utf-8",
    )
    output = tmp_path / "python-runtime-manifest.json"
    captured = builder.build_python_runtime_manifest(
        runtime_prefix=prefix,
        output=output,
    )
    manifest = json.loads(output.read_bytes())
    receipt_record = {
        "path": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "tree_sha256": captured["tree_sha256"],
        "file_count": captured["file_count"],
    }
    (prefix / "unexpected.dll").write_bytes(b"shadow")

    with pytest.raises(
        contract.ReleaseCliIdentityError,
        match="differs from its caller-held manifest",
    ):
        contract._validate_runtime_prefix_manifest(
            prefix=prefix,
            manifest=manifest,
            receipt_record=receipt_record,
        )


def test_launch_time_closure_rejects_hardlinked_file(tmp_path: Path) -> None:
    closure = tmp_path / "governed-site-packages"
    closure.mkdir()
    original = closure / "module.py"
    original.write_bytes(b"value = 1\n")
    linked = tmp_path / "linked.py"
    try:
        os.link(original, linked)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")

    with pytest.raises(
        contract.ReleaseCliIdentityError,
        match="forbidden file",
    ):
        contract._runtime_tree_inventory(closure)

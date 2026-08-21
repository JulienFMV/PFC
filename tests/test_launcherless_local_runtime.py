from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import build_launcherless_conda_archive_lock as conda_archive_lock
from scripts import build_launcherless_local_runtime as runtime


def _cli_args(root: Path) -> Namespace:
    build = root / "build"
    return Namespace(
        runtime_prefix=build / "conda-runtime-v23",
        project_wheel=build / "wheel-dist" / runtime._PROJECT_WHEEL_NAME,
        publisher_wheelhouse=build / "runtime-inputs" / "publisher-wheelhouse",
        publisher_dependency_root=build / "runtime-inputs" / "publisher-closure",
        publisher_receipt=build / "runtime-inputs" / "publisher-receipt.json",
        additional_wheel_directory=build / "runtime-inputs" / "additional-wheels",
        python_runtime_manifest=build / "runtime-inputs" / "python-manifest.json",
        conda_prefix_build_receipt=build / "runtime-inputs" / "conda-receipt.json",
        receipt_output=build / "runtime-receipts" / "runtime-v23.json",
        lock_path=root / "uv.lock",
    )


def test_cli_boundary_rejects_appdata_even_without_workspace_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        runtime.subprocess, "run", lambda *_args, **_kwargs: Namespace(stdout=str(root))
    )
    args = _cli_args(root)
    args.publisher_wheelhouse = tmp_path / "AppData" / "Local" / "wheelhouse"

    with pytest.raises(ValueError, match="publisher-wheelhouse.*below canonical repo build"):
        runtime._require_cli_workspace_and_paths(args, root=root)


def test_cli_boundary_accepts_only_repo_build_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        runtime.subprocess, "run", lambda *_args, **_kwargs: Namespace(stdout=str(root))
    )

    runtime._require_cli_workspace_and_paths(_cli_args(root), root=root)


def _write_conda_prefix_receipt(
    tmp_path: Path,
    *,
    prefix: Path,
    manifest_sha256: str = "1" * 64,
) -> tuple[Path, str, dict[str, object], dict[str, object]]:
    history_path = prefix / "conda-meta" / "history"
    history_path.parent.mkdir()
    history_payload = b"exact history\n"
    history_path.write_bytes(history_payload)
    spec_path = tmp_path / "explicit.txt"
    spec_payload = b"@EXPLICIT\nfile:///archive.tar.bz2#" + b"a" * 32 + b"\n"
    spec_path.write_bytes(spec_payload)
    lock_document = {
        "schema_version": runtime._CONDA_ARCHIVE_LOCK_SCHEMA,
        "archive_lock_id": "6" * 64,
        "archive_set_id": "7" * 64,
        "explicit_spec_lines": spec_payload.decode("ascii").splitlines(),
        "packages": [{"archive_payload_file_count": 1}],
    }
    lock_path = tmp_path / "archive-lock.json"
    lock_payload = runtime._canonical_json(lock_document) + b"\n"
    lock_path.write_bytes(lock_payload)
    manifest = {
        "tree_sha256": "2" * 64,
        "file_count": 3,
        "python_executable_sha256": "3" * 64,
        "python_dll_sha256": "4" * 64,
        "files": [],
    }
    target_meta = [
        {
            "filename": "python.tar.bz2",
            "conda_meta_filename": "python.json",
            "conda_meta_sha256": "9" * 64,
            "installed_file_count": 1,
            "source_record_depends": [],
            "target_record_depends": [],
            "source_record_constrains": [],
            "target_record_constrains": [],
            "dependency_metadata_matches_source_record": True,
            "archive_verified_file_count": 1,
            "archive_verified_tree_sha256": "a" * 64,
            "generated_nonruntime_file_count": 0,
            "generated_nonruntime_tree_sha256": hashlib.sha256().hexdigest(),
        }
    ]
    repo_local_build_receipt = {
        "path": str(tmp_path / "repo-local-build-receipt.json"),
        "sha256": "b" * 64,
        "schema_version": "fmv_lt_repo_local_conda_prefix_build.v1",
        "status": "PASS_REPO_LOCAL_MUTABLE_PATHS_NOT_PRODUCTION",
        "command_sha256": "c" * 64,
        "external_guard_sha256": "d" * 64,
        "external_guard_unchanged": True,
    }
    core = {
        "archive_lock_sha256": hashlib.sha256(lock_payload).hexdigest(),
        "archive_lock_id": "6" * 64,
        "archive_set_id": "7" * 64,
        "explicit_spec_sha256": hashlib.sha256(spec_payload).hexdigest(),
        "python_runtime_manifest_sha256": manifest_sha256,
        "python_runtime_tree_sha256": manifest["tree_sha256"],
        "target_conda_meta_sha256": hashlib.sha256(
            runtime._canonical_json(target_meta)
        ).hexdigest(),
        "package_count": 1,
        "file_count": manifest["file_count"],
        "archive_verified_file_count": 1,
        "generated_nonruntime_file_count": 0,
        "repo_local_build_receipt_sha256": repo_local_build_receipt["sha256"],
        "repo_local_external_guard_sha256": repo_local_build_receipt[
            "external_guard_sha256"
        ],
    }
    receipt = {
        "schema_version": runtime._CONDA_PREFIX_RECEIPT_SCHEMA,
        "status": runtime._CONDA_PREFIX_RECEIPT_STATUS,
        "archive_lock": {
            "path": str(lock_path),
            "sha256": core["archive_lock_sha256"],
            "archive_lock_id": "6" * 64,
            "archive_set_id": "7" * 64,
        },
        "explicit_spec": {
            "path": str(spec_path),
            "sha256": core["explicit_spec_sha256"],
            "size": len(spec_payload),
            "line_count": 2,
        },
        "runtime_prefix": str(prefix),
        "conda_history": {
            "path": str(history_path),
            "sha256": hashlib.sha256(history_payload).hexdigest(),
            "size": len(history_payload),
        },
        "python_runtime_manifest": {
            "path": str(tmp_path / "python-manifest.json"),
            "sha256": manifest_sha256,
            "tree_sha256": manifest["tree_sha256"],
            "file_count": manifest["file_count"],
            "python_executable_sha256": manifest["python_executable_sha256"],
            "python_dll_sha256": manifest["python_dll_sha256"],
        },
        "target_conda_meta": target_meta,
        "repo_local_build_receipt": repo_local_build_receipt,
        "package_count": 1,
        "file_count": manifest["file_count"],
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
        "prefix_receipt_id": hashlib.sha256(runtime._canonical_json(core)).hexdigest(),
        "production_authorization": False,
        "promotion_gate": False,
    }
    output = tmp_path / "conda-prefix-receipt.json"
    payload = runtime._canonical_json(receipt) + b"\n"
    output.write_bytes(payload)
    return output, hashlib.sha256(payload).hexdigest(), manifest, lock_document


def test_original_wheel_contract_closes_missing_locked_distributions() -> None:
    assert {(name, version) for name, version, _, _ in runtime._ADDITIONAL_WHEELS} == {
        ("holidays", "0.83"),
        ("openpyxl", "3.1.5"),
        ("scikit-learn", "1.6.1"),
        ("scipy", "1.13.1"),
        ("joblib", "1.5.3"),
        ("threadpoolctl", "3.6.0"),
        ("et-xmlfile", "2.0.0"),
    }


def test_conda_prefix_receipt_binds_archive_replay_and_python_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "runtime"
    prefix.mkdir()
    receipt, receipt_sha256, manifest, lock_document = _write_conda_prefix_receipt(
        tmp_path,
        prefix=prefix,
    )
    monkeypatch.setattr(
        conda_archive_lock,
        "verify_conda_archive_lock",
        lambda **_: lock_document,
    )
    expected_repo_receipt = json.loads(receipt.read_bytes())[
        "repo_local_build_receipt"
    ]
    monkeypatch.setattr(
        runtime,
        "validate_repo_local_conda_prefix_build_receipt",
        lambda **_: expected_repo_receipt,
    )
    target_meta = json.loads(receipt.read_bytes())["target_conda_meta"]
    monkeypatch.setattr(
        conda_archive_lock,
        "_validated_explicit_prefix_records",
        lambda **_: (target_meta, []),
    )

    result = runtime.validate_conda_prefix_build_receipt(
        runtime_prefix=prefix,
        receipt_path=receipt,
        expected_receipt_sha256=receipt_sha256,
        python_runtime_manifest=manifest,
        expected_python_runtime_manifest_sha256="1" * 64,
    )

    assert result == {
        "path": str(receipt),
        "sha256": receipt_sha256,
        "prefix_receipt_id": json.loads(receipt.read_bytes())["prefix_receipt_id"],
        "archive_lock_sha256": hashlib.sha256(
            (tmp_path / "archive-lock.json").read_bytes()
        ).hexdigest(),
        "archive_lock_id": "6" * 64,
        "archive_set_id": "7" * 64,
        "package_count": 1,
        "file_count": 3,
        "archive_verified_file_count": 1,
        "generated_nonruntime_file_count": 0,
        "repo_local_build_receipt": expected_repo_receipt,
    }


def test_conda_prefix_receipt_rejects_physical_prefix_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "runtime"
    prefix.mkdir()
    receipt, receipt_sha256, manifest, _ = _write_conda_prefix_receipt(
        tmp_path,
        prefix=prefix,
    )
    monkeypatch.setattr(runtime, "paths_overlap_by_identity", lambda *_: True)

    with pytest.raises(ValueError, match="must be outside the prefix"):
        runtime.validate_conda_prefix_build_receipt(
            runtime_prefix=prefix,
            receipt_path=receipt,
            expected_receipt_sha256=receipt_sha256,
            python_runtime_manifest=manifest,
            expected_python_runtime_manifest_sha256="1" * 64,
        )


def test_conda_prefix_receipt_rejects_manifest_binding_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "runtime"
    prefix.mkdir()
    receipt, receipt_sha256, manifest, lock_document = _write_conda_prefix_receipt(
        tmp_path,
        prefix=prefix,
    )
    monkeypatch.setattr(
        conda_archive_lock,
        "verify_conda_archive_lock",
        lambda **_: lock_document,
    )
    monkeypatch.setattr(
        runtime,
        "validate_repo_local_conda_prefix_build_receipt",
        lambda **_: json.loads(receipt.read_bytes())["repo_local_build_receipt"],
    )
    target_meta = json.loads(receipt.read_bytes())["target_conda_meta"]
    monkeypatch.setattr(
        conda_archive_lock,
        "_validated_explicit_prefix_records",
        lambda **_: (target_meta, []),
    )
    manifest["tree_sha256"] = "9" * 64

    with pytest.raises(ValueError, match="Python manifest binding"):
        runtime.validate_conda_prefix_build_receipt(
            runtime_prefix=prefix,
            receipt_path=receipt,
            expected_receipt_sha256=receipt_sha256,
            python_runtime_manifest=manifest,
            expected_python_runtime_manifest_sha256="1" * 64,
        )


@pytest.mark.parametrize(
    ("name", "version", "cache_key", "filename", "sha256"),
    [
        (
            "holidays",
            "0.83",
            "0.83-py3-none-any",
            "holidays-0.83-py3-none-any.whl",
            "e36a368227b5b62129871463697bfde7e5212f6f77e43640320b727b79a875a8",
        ),
        (
            "scipy",
            "1.13.1",
            "1.13.1-cp311-cp311-win_amd64",
            "scipy-1.13.1-cp311-cp311-win_amd64.whl",
            "5713f62f781eebd8d597eb3f88b8bf9274e79eeabf63afb4a737abc6c84ad37b",
        ),
    ],
)
def test_locked_wheel_selection_is_exact(
    name: str,
    version: str,
    cache_key: str,
    filename: str,
    sha256: str,
) -> None:
    selected = runtime._locked_wheel(
        (runtime.ROOT / "uv.lock").read_bytes(),
        name=name,
        version=version,
        cache_key=cache_key,
    )

    assert selected == {"filename": filename, "sha256": sha256}


@pytest.mark.parametrize(
    "forbidden",
    [
        "pfc-lt.exe",
        "Scripts/pfc-lt.cmd",
        "Scripts/pfc-lt.bat",
        "Scripts/pfc-lt.ps1",
        "runtime-injection.pth",
        "package/__pycache__/module.pyc",
    ],
)
def test_launcherless_inventory_rejects_command_and_path_injection(
    forbidden: str,
) -> None:
    with pytest.raises(ValueError, match="forbidden path"):
        runtime._copy_payloads_into_inventory(
            {forbidden: b"payload"},
            {},
            source="test",
        )


def test_launcherless_inventory_rejects_casefold_collision() -> None:
    inventory: dict[str, tuple[str, bytes]] = {}
    runtime._copy_payloads_into_inventory({"Package/module.py": b"one"}, inventory, source="first")

    with pytest.raises(ValueError, match="path collision"):
        runtime._copy_payloads_into_inventory(
            {"package/MODULE.py": b"two"}, inventory, source="second"
        )


def test_python_dot_pth_is_positive_inventory_only(tmp_path: Path) -> None:
    target = tmp_path / "python311._pth"

    runtime._write_runtime_pth(target)

    assert target.read_text(encoding="ascii") == ("Lib\nDLLs\ngoverned-site-packages\n")
    assert runtime._validated_runtime_pth_payload(target) == (runtime._runtime_pth_payload())
    assert "import site" not in target.read_text(encoding="ascii")


def test_python_dot_pth_final_admission_rejects_divergent_bytes(
    tmp_path: Path,
) -> None:
    target = tmp_path / "python311._pth"
    target.write_text(
        "Lib\nDLLs\n.\ngoverned-site-packages\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="python311._pth bytes diverge"):
        runtime._validated_runtime_pth_payload(target)


def test_source_never_invokes_package_installer_or_project_launcher() -> None:
    source = Path(runtime.__file__).read_text(encoding="utf-8")

    assert "pip install" not in source
    assert "uv pip" not in source
    assert "pfc-lt.exe" not in source
    assert "pfc-lt-build-acquisition.exe" not in source


def test_additional_wheel_rejects_bytes_that_differ_from_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = tmp_path / "example-1.0-py3-none-any.whl"
    wheel.write_bytes(b"tampered")
    monkeypatch.setattr(
        runtime,
        "_locked_wheel",
        lambda *args, **kwargs: {
            "filename": wheel.name,
            "sha256": "0" * 64,
        },
    )

    with pytest.raises(ValueError, match="differs from uv.lock"):
        runtime._validate_additional_wheel(
            wheel_directory=tmp_path,
            lock_payload=b"ignored",
            name="example",
            version="1.0",
            cache_key="1.0-py3-none-any",
        )


def test_staging_materialization_resumes_exact_partial_tree(
    tmp_path: Path,
) -> None:
    staging = tmp_path / runtime._CLOSURE_STAGING_NAME
    (staging / "package").mkdir(parents=True)
    (staging / "package" / "existing.py").write_bytes(b"existing")
    inventory = {
        "package/existing.py": ("package/existing.py", b"existing"),
        "package/missing.py": ("package/missing.py", b"missing"),
    }

    runtime._materialize_inventory_tree(staging, inventory)

    assert (staging / "package" / "existing.py").read_bytes() == b"existing"
    assert (staging / "package" / "missing.py").read_bytes() == b"missing"


def test_staging_materialization_rejects_divergent_partial_tree(
    tmp_path: Path,
) -> None:
    staging = tmp_path / runtime._CLOSURE_STAGING_NAME
    staging.mkdir()
    (staging / "module.py").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="staged runtime file diverges"):
        runtime._materialize_inventory_tree(
            staging,
            {"module.py": ("module.py", b"expected")},
        )

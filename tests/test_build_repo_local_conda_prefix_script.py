from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import build_repo_local_conda_prefix as builder


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "repo"
    build = root / "build"
    root.mkdir()
    build.mkdir()
    (root / ".git").mkdir()
    conda_python = build / "read-only-conda-python.exe"
    conda_python.write_bytes(b"python")
    explicit = build / "explicit.txt"
    explicit.write_text("@EXPLICIT\n", encoding="ascii")
    paths = {
        "conda_python": conda_python,
        "explicit_spec": explicit,
        "runtime_prefix": build / "runtime",
        "work_root": build / "work",
        "receipt_output": build / "receipt.json",
        "real_user_profile": build / "synthetic-user-profile",
    }
    return root, paths


def _patch_workspace(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.chdir(root)

    def _run_git(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout=str(root), stderr="")

    monkeypatch.setattr(builder.subprocess, "run", _run_git)


def test_builder_redirects_every_mutable_conda_path_below_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, paths = _fixture(tmp_path)
    monkeypatch.chdir(root)
    calls: list[dict[str, object]] = []

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = args[0]
        if command == ["git", "rev-parse", "--show-toplevel"]:
            return subprocess.CompletedProcess(command, 0, stdout=str(root), stderr="")
        calls.append(kwargs)
        prefix = paths["runtime_prefix"]
        (prefix / "conda-meta").mkdir(parents=True)
        (prefix / "python.exe").write_bytes(b"python")
        (prefix / "conda-meta" / "history").write_text("history", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(builder.subprocess, "run", _run)
    result = builder.build_repo_local_conda_prefix(root=root, **paths)

    assert result["status"] == "PASS_REPO_LOCAL_MUTABLE_PATHS_NOT_PRODUCTION"
    assert result["external_guard_unchanged"] is True
    environment = calls[0]["env"]
    assert isinstance(environment, dict)
    for key in (
        "HOME",
        "USERPROFILE",
        "CONDARC",
        "CONDA_PKGS_DIRS",
        "CONDA_ENVS_PATH",
        "TEMP",
        "TMP",
    ):
        assert str(environment[key]).startswith(str(root / "build"))
    assert environment["CONDA_REGISTER_ENVS"] == "false"
    receipt = json.loads(paths["receipt_output"].read_text(encoding="utf-8"))
    assert receipt["production_authorization"] is False
    receipt_sha256 = hashlib.sha256(paths["receipt_output"].read_bytes()).hexdigest()
    validated = builder.validate_repo_local_conda_prefix_build_receipt(
        receipt_path=paths["receipt_output"],
        expected_sha256=receipt_sha256,
        runtime_prefix=paths["runtime_prefix"],
        explicit_spec_path=paths["explicit_spec"],
        root=root,
    )
    assert validated["sha256"] == receipt_sha256
    assert validated["external_guard_unchanged"] is True


def test_builder_rejects_mutable_path_outside_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, paths = _fixture(tmp_path)
    _patch_workspace(monkeypatch, root)
    paths["runtime_prefix"] = tmp_path / "outside-runtime"
    with pytest.raises(ValueError, match="below repo build"):
        builder.build_repo_local_conda_prefix(root=root, **paths)


def test_builder_rejects_reparse_parent_before_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, paths = _fixture(tmp_path)
    _patch_workspace(monkeypatch, root)
    original = builder.assert_absolute_path_has_no_links
    target = Path(os.path.abspath(paths["runtime_prefix"]))

    def _reject_reparse(path: Path) -> Path:
        selected = Path(os.path.abspath(path))
        if selected == target:
            raise ValueError("path contains a link or reparse point")
        return original(path)

    monkeypatch.setattr(builder, "assert_absolute_path_has_no_links", _reject_reparse)
    with pytest.raises(ValueError, match="link or reparse point"):
        builder.build_repo_local_conda_prefix(root=root, **paths)


def test_builder_fails_when_external_guard_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, paths = _fixture(tmp_path)
    profile = paths["real_user_profile"]
    registry = profile / ".conda" / "environments.txt"
    registry.parent.mkdir(parents=True)
    registry.write_text("before\n", encoding="utf-8")
    monkeypatch.chdir(root)

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = args[0]
        if command == ["git", "rev-parse", "--show-toplevel"]:
            return subprocess.CompletedProcess(command, 0, stdout=str(root), stderr="")
        registry.write_text("after\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(builder.subprocess, "run", _run)
    with pytest.raises(RuntimeError, match="external user-profile guard changed"):
        builder.build_repo_local_conda_prefix(root=root, **paths)
    receipt = json.loads(paths["receipt_output"].read_text(encoding="utf-8"))
    assert receipt["status"] == "FAIL_REPO_LOCAL_CONDA_PREFIX_BUILD_NO_AUTHORITY"
    assert receipt["external_guard_unchanged"] is False

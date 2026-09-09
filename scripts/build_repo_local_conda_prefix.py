#!/usr/bin/env python3
"""Create one offline Conda prefix while confining every mutable path to build/."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "fmv_lt_repo_local_conda_prefix_build.v1"
_PASS_STATUS = "PASS_REPO_LOCAL_MUTABLE_PATHS_NOT_PRODUCTION"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CONDARC = b"""channels:
  - defaults
offline: true
solver: classic
add_pip_as_python_dependency: false
register_envs: false
"""


def build_repo_local_conda_prefix(
    *,
    conda_python: Path,
    explicit_spec: Path,
    runtime_prefix: Path,
    work_root: Path,
    receipt_output: Path,
    root: Path = ROOT,
    real_user_profile: Path | None = None,
    timeout_seconds: int = 600,
) -> dict[str, object]:
    """Run exact offline creation with a redirected profile and external guards."""

    canonical_root = _canonical_workspace(root)
    build_root = canonical_root / "build"
    selected_python = assert_absolute_path_has_no_links(conda_python)
    if not selected_python.is_file():
        raise ValueError("Conda Python must be an existing read-only interpreter")
    selected_spec = _existing_build_file(explicit_spec, build_root, "explicit spec")
    conda_python_evidence = _file_evidence_allow_hardlinks(selected_python)
    explicit_spec_evidence = _file_evidence(selected_spec)
    prefix = _new_build_path(runtime_prefix, build_root, "runtime prefix")
    work = _new_build_path(work_root, build_root, "Conda work root")
    receipt = _new_build_path(receipt_output, build_root, "build receipt")
    if receipt.suffix != ".json":
        raise ValueError("build receipt must be JSON")
    if _paths_overlap(prefix, work) or _paths_overlap(prefix, receipt):
        raise ValueError("runtime prefix, work root and receipt must not overlap")

    profile = (
        Path(os.path.abspath(Path.home()))
        if real_user_profile is None
        else Path(os.path.abspath(real_user_profile))
    )
    guards = (
        profile / ".conda" / "environments.txt",
        profile / ".condarc",
    )
    before = {str(path): _external_snapshot(path) for path in guards}

    work.mkdir(parents=True, exist_ok=False)
    local_home = work / "home"
    pkgs = work / "conda-pkgs"
    envs = work / "conda-envs"
    temp = work / "temp"
    for directory in (local_home, pkgs, envs, temp):
        directory.mkdir(parents=True, exist_ok=False)
    condarc = work / "condarc.yml"
    write_durable_exact_bytes(condarc, _CONDARC, label="repo-local Conda config")

    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(local_home),
            "USERPROFILE": str(local_home),
            "HOMEDRIVE": local_home.drive,
            "HOMEPATH": str(local_home)[len(local_home.drive) :],
            "CONDARC": str(condarc),
            "CONDA_REGISTER_ENVS": "false",
            "CONDA_PKGS_DIRS": str(pkgs),
            "CONDA_ENVS_PATH": str(envs),
            "TEMP": str(temp),
            "TMP": str(temp),
            "TMPDIR": str(temp),
            "PYTHONDONTWRITEBYTECODE": "1",
            "XDG_CACHE_HOME": str(work / "xdg-cache"),
        }
    )
    command = [
        str(selected_python),
        "-B",
        "-m",
        "conda",
        "create",
        "--offline",
        "--copy",
        "--prefix",
        str(prefix),
        "--file",
        str(selected_spec),
        "--yes",
        "--json",
    ]
    completed: subprocess.CompletedProcess[str] | None = None
    failure: BaseException | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=canonical_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        failure = exc

    after = {str(path): _external_snapshot(path) for path in guards}
    external_unchanged = before == after
    prefix_ready = (
        prefix.is_dir()
        and (prefix / "python.exe").is_file()
        and (prefix / "conda-meta" / "history").is_file()
    )
    returncode = None if completed is None else completed.returncode
    stdout = "" if completed is None else completed.stdout
    stderr = "" if completed is None else completed.stderr
    status = (
        _PASS_STATUS
        if failure is None and returncode == 0 and prefix_ready and external_unchanged
        else "FAIL_REPO_LOCAL_CONDA_PREFIX_BUILD_NO_AUTHORITY"
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "command_sha256": _sha256_json(command),
        "conda_python": conda_python_evidence,
        "explicit_spec": explicit_spec_evidence,
        "runtime_prefix": str(prefix),
        "work_root": str(work),
        "mutable_paths": {
            "home": str(local_home),
            "conda_pkgs": str(pkgs),
            "conda_envs": str(envs),
            "temp": str(temp),
            "condarc": str(condarc),
        },
        "external_guard_before": before,
        "external_guard_after": after,
        "external_guard_unchanged": external_unchanged,
        "target_exit_code": returncode,
        "target_stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "target_stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "target_failure_class": None if failure is None else type(failure).__name__,
        "prefix_ready": prefix_ready,
        "network_required": False,
        "admin_required": False,
        "production_authorization": False,
        "promotion_gate": False,
    }
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    write_durable_exact_bytes(receipt, encoded, label="repo-local Conda build receipt")
    if status != "PASS_REPO_LOCAL_MUTABLE_PATHS_NOT_PRODUCTION":
        if not external_unchanged:
            raise RuntimeError("external user-profile guard changed during Conda build")
        if failure is not None:
            raise RuntimeError("repo-local Conda build did not terminate normally") from failure
        raise RuntimeError(f"repo-local Conda build failed with exit code {returncode}")
    return payload


def _canonical_workspace(root: Path) -> Path:
    expected = Path(os.path.abspath(root))
    if Path(os.path.abspath(Path.cwd())) != expected:
        raise ValueError(f"cwd must be exactly {expected}")
    observed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=expected,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if os.path.normcase(os.path.abspath(observed)) != os.path.normcase(str(expected)):
        raise ValueError(f"Git top-level must be exactly {expected}")
    return expected


def _existing_build_file(path: Path, build_root: Path, label: str) -> Path:
    selected = assert_absolute_path_has_no_links(path)
    if not selected.is_file() or not _within(selected, build_root):
        raise ValueError(f"{label} must be an existing file below repo build/")
    return selected


def _new_build_path(path: Path, build_root: Path, label: str) -> Path:
    try:
        selected = assert_absolute_path_has_no_links(Path(os.path.abspath(path)))
        selected_build_root = assert_absolute_path_has_no_links(build_root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} contains a link or reparse point") from exc
    if not _within(selected, selected_build_root) or selected.exists():
        raise ValueError(f"{label} must be a new path below repo build/")
    return selected


def validate_repo_local_conda_prefix_build_receipt(
    *,
    receipt_path: Path,
    expected_sha256: str,
    runtime_prefix: Path,
    explicit_spec_path: Path,
    root: Path = ROOT,
) -> dict[str, object]:
    """Validate and bind the standard-user confinement receipt to one prefix."""

    canonical_root = Path(os.path.abspath(root))
    build_root = assert_absolute_path_has_no_links(canonical_root / "build")
    expected = expected_sha256.strip().lower()
    if _HEX64.fullmatch(expected) is None:
        raise ValueError("repo-local Conda build receipt SHA-256 is invalid")
    selected = assert_absolute_path_has_no_links(receipt_path)
    if not selected.is_file() or not _within(selected, build_root):
        raise ValueError("repo-local Conda build receipt must be below repo build/")
    payload = read_stable_single_link_file(
        selected, label="repo-local Conda prefix build receipt"
    )
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("repo-local Conda build receipt SHA-256 diverges")
    try:
        receipt = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("repo-local Conda build receipt is invalid") from exc
    canonical = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    if not isinstance(receipt, dict) or canonical != payload:
        raise ValueError("repo-local Conda build receipt is not canonical JSONL")

    prefix = assert_absolute_path_has_no_links(runtime_prefix)
    spec = _existing_build_file(explicit_spec_path, build_root, "explicit spec")
    spec_evidence = receipt.get("explicit_spec")
    before = receipt.get("external_guard_before")
    after = receipt.get("external_guard_after")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != _PASS_STATUS
        or receipt.get("runtime_prefix") != str(prefix)
        or receipt.get("target_exit_code") != 0
        or receipt.get("prefix_ready") is not True
        or receipt.get("external_guard_unchanged") is not True
        or not isinstance(before, dict)
        or before != after
        or receipt.get("network_required") is not False
        or receipt.get("admin_required") is not False
        or receipt.get("production_authorization") is not False
        or receipt.get("promotion_gate") is not False
        or not isinstance(spec_evidence, dict)
        or spec_evidence.get("path") != str(spec)
        or spec_evidence.get("sha256")
        != hashlib.sha256(
            read_stable_single_link_file(spec, label="repo-local Conda explicit spec")
        ).hexdigest()
    ):
        raise ValueError("repo-local Conda build receipt policy is invalid")

    work_root = assert_absolute_path_has_no_links(Path(str(receipt.get("work_root", ""))))
    mutable = receipt.get("mutable_paths")
    if (
        not _within(work_root, build_root)
        or not isinstance(mutable, dict)
        or not mutable
    ):
        raise ValueError("repo-local Conda mutable path inventory is invalid")
    for value in mutable.values():
        path = assert_absolute_path_has_no_links(Path(str(value)))
        if not _within(path, work_root):
            raise ValueError("repo-local Conda mutable path escapes its work root")
    if _paths_overlap(selected, prefix):
        raise ValueError("repo-local Conda build receipt must be outside the prefix")
    return {
        "path": str(selected),
        "sha256": expected,
        "schema_version": SCHEMA_VERSION,
        "status": _PASS_STATUS,
        "command_sha256": receipt.get("command_sha256"),
        "external_guard_sha256": hashlib.sha256(
            json.dumps(before, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "external_guard_unchanged": True,
    }


def _within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((os.path.normcase(str(path)), os.path.normcase(str(root)))) == os.path.normcase(str(root))
    except ValueError:
        return False


def _paths_overlap(left: Path, right: Path) -> bool:
    return _within(left, right) or _within(right, left)


def _external_snapshot(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False}
    selected = assert_absolute_path_has_no_links(path)
    payload = read_stable_single_link_file(selected, label=f"external guard {selected}")
    stat_result = selected.stat()
    return {
        "exists": True,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mtime_ns": stat_result.st_mtime_ns,
    }


def _file_evidence(path: Path) -> dict[str, object]:
    payload = read_stable_single_link_file(path, label=f"build input {path}")
    return {
        "path": str(path),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _file_evidence_allow_hardlinks(path: Path) -> dict[str, object]:
    """Capture a stable read-only interpreter; hardlinks are not mutable outputs."""

    selected = assert_absolute_path_has_no_links(path)
    digest = hashlib.sha256()
    size = 0
    with selected.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Conda Python must be a regular file")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    path_after = selected.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    path_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
    )
    if before_identity != after_identity or after_identity != path_identity:
        raise ValueError("Conda Python changed during stable read")
    if size != after.st_size:
        raise ValueError("Conda Python size changed during stable read")
    return {
        "path": str(selected),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
        "link_count": after.st_nlink,
    }


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda-python", type=Path, required=True)
    parser.add_argument("--explicit-spec", type=Path, required=True)
    parser.add_argument("--runtime-prefix", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_repo_local_conda_prefix(
            conda_python=args.conda_python,
            explicit_spec=args.explicit_spec,
            runtime_prefix=args.runtime_prefix,
            work_root=args.work_root,
            receipt_output=args.receipt_output,
            timeout_seconds=args.timeout_seconds,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "FAIL_REPO_LOCAL_CONDA_PREFIX_BUILD_NO_AUTHORITY",
                    "production_authorization": False,
                    "promotion_gate": False,
                    "error": str(exc),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

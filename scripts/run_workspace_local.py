"""Run one command with mutable state confined to a fresh repo-local namespace.

This is a standard-user workstation harness, not a security sandbox. It makes
the implicit mutable destinations used by Python, pip, Conda, pytest and common
scientific libraries explicit and repo-local. Explicit command arguments must
still respect ``AGENTS.md``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import secrets
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable
from urllib.parse import unquote_to_bytes

# Keep the physical pytest root short enough for nested Windows atomicity tests.
# The human-readable command and its hash remain in the receipt, so long labels
# do not belong in the filesystem namespace.
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,15}")
_EXPECTED_ROOT = Path(r"C:\Users\jbattaglia\PFC_LT")
_ALLOWED_PYTHON_MODULES = {
    "pytest",
    "ruff",
    "scripts.audit_launcherless_conda_prefix",
    "scripts.audit_ch_eex_current_local_capture",
    "scripts.audit_ch_eex_current_repricing_sensitivity",
    "scripts.audit_ch_lt_dependence_power_design",
    "scripts.audit_ch_lt_laptop_model_pair",
    "scripts.audit_ch_lt_local_future_origin_selection",
    "scripts.audit_ch_lt_local_quality_current_selection",
    "scripts.audit_ch_lt_origin_registry_protocol",
    "scripts.audit_ch_lt_origin_target_mask_inventory",
    "scripts.audit_ch_lt_successor_candidate_core",
    "scripts.build_local_intraday_calibration_panel",
    "scripts.build_ch_lt_structural_prediction_commitment",
    "scripts.build_local_test_ch_pfc",
    "scripts.build_launcherless_conda_archive_lock",
    "scripts.build_launcherless_local_runtime",
    "scripts.build_launcherless_python_runtime_manifest",
    "scripts.capture_eex_forward_local",
    "scripts.capture_public_energy_charts_lt",
    "scripts.check_lt_wheel_contract",
    "scripts.verify_ch_lt_preregistration_supersession",
    "pfc_shaping.cli.eex_forward_vintage_builder",
    "pfc_shaping.data.eex_datasource_v2_capture",
    "pfc_shaping.cli.audit_ch_lt_origin_registry_protocol",
    "pfc_shaping.cli.audit_ch_lt_origin_target_mask_inventory",
    "pfc_shaping.cli.verify_ch_lt_preregistration_supersession",
    "pfc_shaping.validation.ch_lt_local_future_origin_selection",
    "pfc_shaping.validation.eex_entsoe_rfc3161_request_der",
}
_SENSITIVE_ENV_MARKERS = {
    "ACCESS_KEY",
    "APIKEY",
    "API_KEY",
    "AUTH_TOKEN",
    "BEARER",
    "CLIENT_SECRET",
    "CREDENTIAL",
    "MTLS",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
    "PUBLIC_KEY",
    "SECRET",
    "SIGNATURE",
    "SIGNING",
    "TRUSTED",
    "TOKEN",
    "JOURNAL",
}
_AUTHORITY_ENV_MARKERS = {
    "EXTERNAL_CAS",
    "PRODUCTION_AUTHORIZATION",
    "PROMOTION_AUTHORIZATION",
    "RELEASE_AUTHORIZATION",
    "RUNTIME_RECEIPT",
}
_SENSITIVE_ENV_PREFIXES = (
    "AWS_",
    "AZURE_",
    "GCP_",
    "GH_",
    "GITHUB_",
    "GOOGLE_",
    "SSH_",
    "VAULT_",
)
_AMBIENT_PATH_ENV_KEYS = {
    "APPDATA",
    "FMV_DATA_ROOT",
    "PFC_CT_DATA_ROOT",
    "PFC_DATA_ROOT",
    "PFC_ENTSOE_DATA_ROOT",
    "PFC_LT_DATA_ROOT",
    "PFC_SHARED_DATA_ROOT",
    "PFC_REQUESTS_CA_BUNDLE",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONWARNINGS",
    "PYTEST_ADDOPTS",
    "PYTEST_PLUGINS",
}
_ALLOWED_PYTHON_FLAGS = {"-B", "-E", "-I", "-S", "-s", "-u"}
_RECEIPT_SCHEMA = "pfc_lt_workspace_local_execution.v6"
_RECONCILIATION_RECEIPT_SCHEMA = "pfc_lt_workspace_local_reconciliation.v1"
_SUPERVISOR_RECEIPT_SCHEMA = "pfc_lt_workspace_local_supervisor.v1"
_INTERNAL_WORKER_FLAG = "--internal-supervised-worker"
_INTERNAL_WORKER_TOKEN_ENV = "PFC_WORKSPACE_LOCAL_INTERNAL_WORKER_TOKEN"
_INTERNAL_WORKER_CAPABILITY_ENV = "PFC_WORKSPACE_LOCAL_INTERNAL_CAPABILITY_PATH"
_INTERNAL_WORKER_PARENT_PID_ENV = "PFC_WORKSPACE_LOCAL_INTERNAL_PARENT_PID"
_INTERNAL_WORKER_PARENT_TOKEN_ENV = "PFC_WORKSPACE_LOCAL_INTERNAL_PARENT_TOKEN"
_EEX_DATASOURCE_MODULE = "pfc_shaping.data.eex_datasource_v2_capture"
_EEX_DATASOURCE_TOKEN_ENV = "PFC_EEX_DATASOURCE_ACCESS_TOKEN"
_EEX_DATASOURCE_TOKEN = re.compile(r"[A-Za-z0-9._~+/=-]{16,4096}")
_SUPERVISOR_SOURCE_LIMIT_BYTES = 4 * 1024 * 1024
_OUTPUT_CAPTURE_LIMIT_BYTES = 64 * 1024 * 1024
_PYTEST_JUNIT_LIMIT_BYTES = 16 * 1024 * 1024
_PYTEST_NATIVE_RESULT_LIMIT_BYTES = 1024 * 1024
_MAX_PYTEST_TEST_COUNT = 10_000_000
_STREAM_CHUNK_BYTES = 64 * 1024
_PIPE_DRAIN_GRACE_SECONDS = 5.0
_PROCESS_TREE_TERMINATION_GRACE_SECONDS = 5.0
_DEFAULT_WALL_TIMEOUT_SECONDS = 30.0 * 60.0
_MAX_WALL_TIMEOUT_SECONDS = 24.0 * 60.0 * 60.0
_WINDOWS_CREATE_SUSPENDED = 0x00000004
_WINDOWS_JOB_OBJECT_TERMINATED_EXIT_CODE = 124
_PYTEST_NATIVE_RESULT_SCHEMA = "pfc_lt_pytest_native_result.v1"
_SOURCE_FILE_LIMIT_BYTES = 16 * 1024 * 1024
_SOURCE_TREE_LIMIT_BYTES = 256 * 1024 * 1024
_IMPORT_CLOSURE_FILE_LIMIT_BYTES = 512 * 1024 * 1024
_IMPORT_CLOSURE_TREE_LIMIT_BYTES = 4 * 1024 * 1024 * 1024
_IMPORT_CLOSURE_FILE_COUNT_LIMIT = 100_000
_PYTHON_PTH_LIMIT_BYTES = 64 * 1024
_LAUNCHERLESS_BUILD_PATH_OPTIONS = {
    "--additional-wheel-directory",
    "--conda-prefix-build-receipt",
    "--project-wheel",
    "--publisher-dependency-root",
    "--publisher-receipt",
    "--publisher-wheelhouse",
    "--python-runtime-manifest",
    "--receipt-output",
    "--runtime-prefix",
}
_EEX_VINTAGE_REQUIRED_PATH_OPTIONS = {
    "--runtime-receipt",
    "--intake-spec",
    "--source-document",
    "--trusted-time-receipt",
    "--output-directory",
    "--trusted-time-public-key",
}
_EEX_VINTAGE_OPTIONAL_PATH_OPTIONS = {
    "--previous-catalog",
    "--previous-history",
    "--previous-catalog-public-key",
}
_LOCAL_PANEL_PATH_OPTIONS = {
    "--base-parquet",
    "--audit-json",
    "--audit-runtime-receipt-json",
    "--verifier-artifact",
    "--output-parquet",
    "--output-manifest",
}
_LOCAL_MODEL_INPUT_PATH_OPTIONS = {
    "--epex-hourly",
    "--forwards",
    "--intraday-epex",
    "--intraday-provenance-manifest",
    "--inventory",
    "--manifest",
}
_LOCAL_MODEL_OUTPUT_PATH_OPTIONS = {
    "--completion-manifest",
    "--expanded-output",
    "--fan-chart-output",
    "--features-output",
    "--governance-report",
    "--output-dir",
    "--summary",
}
_LOCAL_MODEL_VALUE_OPTIONS = {
    "--delivery-local-end-exclusive",
    "--expected-intraday-provenance-sha256",
    "--horizon-days",
    "--intraday-cutoff",
    "--intraday-market",
    "--market",
    "--monthly-solver-constraint-tolerance",
    "--monthly-solver-lambda-shape",
    "--monthly-solver-lambda-smooth-month",
    "--monthly-solver-lambda-smooth-yoy",
    "--monthly-solver-neighbor-shrinkage",
    "--monthly-solver-structural-amplitude",
    "--output-prefix",
    "--require-direct-intraday-seasons",
    "--scenarios",
    "--start-date",
    "--vintage",
    "--weights",
}
_LOCAL_MODEL_FLAG_OPTIONS = {
    "--disable-cascade-trend-for-annual-only",
    "--enable-experimental-sparse-intraday-regularization",
    "--enable-monthly-forward-curve-solver",
    "--monthly-solver-allow-template-structural-fallback",
    "--require-nonflat-intraday-shape",
}
_LOCAL_MODEL_CANONICAL_INPUTS = {
    "--epex-hourly": "data/epex_hourly.parquet",
    "--forwards": "data/eex_forwards_history.parquet",
    "--intraday-epex": (
        "output/phase14/prospective_intraday_panel_20260724/"
        "epex_de_15min_complete.parquet"
    ),
    "--intraday-provenance-manifest": (
        "output/phase14/prospective_intraday_panel_20260724/panel-manifest.json"
    ),
    "--inventory": "data/electrification_scenarios_prod_candidate_neutralized_2030.parquet",
    "--manifest": (
        ".planning/phases/13-lt-electrification-scenario-shape/"
        "SCENARIO-GOVERNANCE-LOCAL-TEST-MANIFEST.yaml"
    ),
}
_LOCAL_MODEL_OUTPUT_FILENAMES = {
    "--completion-manifest": "completion_manifest.json",
    "--expanded-output": "scenario_expanded.parquet",
    "--fan-chart-output": "structural_fan_chart.parquet",
    "--features-output": "hpfc_scenario_features.parquet",
    "--governance-report": "LOCAL-TEST-GOVERNANCE-GATE.md",
    "--summary": "QUALITY.md",
}
_LAPTOP_MODEL_QUALIFICATION_OUTPUT = Path(
    "build/local-model-qualification/ch-lt-laptop-model-pair-v5"
)
_STRUCTURAL_PREDICTION_COMMITMENT_OUTPUT = Path(
    "build/ch-lt-structural-prediction-commitments/"
    "structural-dry-run-20260731T072027Z-v7-quant-contract"
)
_STRUCTURAL_PREDICTION_COMMITMENT_INVENTORY = Path(
    "build/ch-lt-origin-target-mask-inventories/"
    "structural-dry-run-20260731T072027Z-schema-v2.json"
)
_STRUCTURAL_PREDICTION_COMMITMENT_INVENTORY_SHA256 = (
    "a4d5722841df549ba24ca2dd190fa0f5b1d79738fc63680acf93034e148419b7"
)
_LOCAL_FUTURE_ORIGIN_SELECTION_REGISTRY = Path(
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-LOCAL-FUTURE-ORIGIN-SELECTION-V6-20260731.json"
)
_LOCAL_FUTURE_ORIGIN_SELECTION_SHA256 = (
    "11966d1ee85ace46e97006fa74f8aab4789c71f7438de909ed71541aab480df7"
)
_LOCAL_FUTURE_ORIGIN_PACKAGED_MODULE = (
    "pfc_shaping.validation.ch_lt_local_future_origin_selection"
)
_ENERGY_CHARTS_CAPTURE_REQUIRED_OPTIONS = {
    "--role",
    "--start-utc",
    "--end-utc",
    "--raw-cadence-minutes",
    "--acquisition-id",
    "--output-directory",
    "--ca-bundle",
}
_EEX_LOCAL_CAPTURE_REQUIRED_OPTIONS = {
    "--source-document",
    "--expected-source-sha256",
    "--capture-id",
}
_FMV_EEX_WORKBOOK_SOURCE = Path(
    r"\\fmvfs1\data\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX.xlsx"
)
_PORTABLE_CAPTURE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_EEX_CURRENT_SELECTION_REQUIRED_OPTIONS = {
    "--registry",
    "--expected-registry-sha256",
}
_EEX_CURRENT_SELECTION_REGISTRY = Path(
    ".planning/phases/14-lt-audit-remediation/"
    "CH-EEX-CURRENT-LOCAL-CAPTURE-SELECTION-20260730.json"
)
_EEX_CURRENT_SELECTION_SHA256 = (
    "5f0b99aa04fabcb8219cfa34f20ea262a705940cbc3db3ab2e114ba99bb4a778"
)
_PREREGISTRATION_SUPERSESSION_REQUIRED_OPTIONS = {
    "--registry",
    "--expected-registry-sha256",
}
_PREREGISTRATION_SUPERSESSION_REGISTRY = Path(
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-PIT-PREREGISTRATION-SUPERSESSION-20260727.json"
)
_PREREGISTRATION_SUPERSESSION_SHA256 = (
    "76dba0b05948d336268b0a50c16df82ecd1c5138626c3fa87ee219148cb8e5e8"
)
_SUCCESSOR_CANDIDATE_CORE_REQUIRED_OPTIONS = {
    "--core",
    "--expected-core-sha256",
}
_SUCCESSOR_CANDIDATE_CORE = Path(
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-SUCCESSOR-CANDIDATE-CORE-V3-20260730.json"
)
_SUCCESSOR_CANDIDATE_CORE_SHA256 = (
    "e2c2a6f9d3ca677991f3f76f03dec1328492e2f60a4015b7796a2ff6aa6f905a"
)
_DEPENDENCE_POWER_DESIGN_REQUIRED_OPTIONS = {
    "--design",
    "--expected-design-sha256",
}
_DEPENDENCE_POWER_DESIGN = Path(
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-DEPENDENCE-POWER-DESIGN-DRAFT-V1-20260730.json"
)
_DEPENDENCE_POWER_DESIGN_SHA256 = (
    "005b8655b817db10e7f3c227b1c5912d545305b68e262ab82d9aa0b5817a6a91"
)
_ORIGIN_REGISTRY_PROTOCOL_REQUIRED_OPTIONS = {
    "--protocol",
    "--expected-protocol-sha256",
}
_ORIGIN_REGISTRY_PROTOCOL = Path(
    ".planning/phases/14-lt-audit-remediation/"
    "CH-LT-ORIGIN-REGISTRY-PROTOCOL-DRAFT-V2-20260730.json"
)
_ORIGIN_REGISTRY_PROTOCOL_SHA256 = (
    "6ea896ccdb35414b52237f2bcf1065755c3c10444b308ce905b60f472e68c697"
)
_ORIGIN_TARGET_MASK_OUTPUT_DIRECTORY = Path(
    "build/ch-lt-origin-target-mask-inventories"
)


class WorkspaceLocalError(ValueError):
    """Raised before target execution when the local boundary is invalid."""


@dataclass(frozen=True)
class CapturedExecution:
    """Terminal target result plus durable, bounded output evidence."""

    returncode: int
    streams: dict[str, dict[str, object]]
    supervision: dict[str, object]
    resource_usage: dict[str, object]
    timed_out: bool = False
    interrupted: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _physical(path: Path) -> Path:
    return Path(os.path.normcase(os.path.realpath(path)))


def _lexical(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _is_within(path: Path, root: Path) -> bool:
    selected = _physical(path)
    boundary = _physical(root)
    try:
        return os.path.commonpath((str(selected), str(boundary))) == str(boundary)
    except ValueError:
        return False


def _require_within(path: Path, root: Path, *, label: str) -> Path:
    selected = path if path.is_absolute() else root / path
    selected = Path(os.path.abspath(selected))
    if not _is_within(selected, root):
        raise WorkspaceLocalError(f"{label} must remain below the canonical workspace")
    return selected


def canonical_repo_root() -> Path:
    source_root = Path(__file__).resolve().parents[1]
    if _lexical(source_root) != _lexical(_EXPECTED_ROOT):
        raise WorkspaceLocalError("runner source is not in the canonical C: workspace")
    return _EXPECTED_ROOT


def verify_canonical_workspace(root: Path) -> None:
    expected = _lexical(root)
    if _lexical(Path.cwd()) != expected:
        raise WorkspaceLocalError(f"cwd must be exactly {root}")
    observed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if _lexical(Path(observed)) != expected:
        raise WorkspaceLocalError(f"Git top-level must be exactly {root}")


def _reject_reparse_chain(path: Path, root: Path) -> None:
    selected = path
    while True:
        attributes = getattr(os.lstat(selected), "st_file_attributes", 0)
        if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise WorkspaceLocalError(f"reparse point is forbidden: {selected}")
        if _lexical(selected) == _lexical(root):
            return
        if selected.parent == selected or not _is_within(selected.parent, root):
            raise WorkspaceLocalError("path identity escaped the canonical workspace")
        selected = selected.parent


def _probe_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)
    probe = path / ".workspace-local-write-probe"
    renamed = path / ".workspace-local-write-probe-renamed"
    payload = b"pfc-lt-workspace-local-preflight-v1\n"
    with probe.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if probe.read_bytes() != payload:
        raise WorkspaceLocalError(f"preflight read-back mismatch: {path}")
    probe.replace(renamed)
    if renamed.read_bytes() != payload:
        raise WorkspaceLocalError(f"preflight rename read-back mismatch: {path}")
    renamed.unlink()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_new_json(path: Path, payload: dict[str, object]) -> None:
    """Publish one immutable JSON sidecar without replacing an existing file."""

    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _process_start_token(pid: int) -> str | None:
    """Return an OS creation token that distinguishes PID reuse."""

    if pid <= 0:
        return None
    observation = _observe_pid(pid)
    token = observation.get("start_token")
    return str(token) if isinstance(token, str) else None


def _observe_pid(pid: int) -> dict[str, object]:
    if pid <= 0:
        return {"pid": pid, "state": "INVALID"}
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class FileTime(ctypes.Structure):
            _fields_ = (("low", wintypes.DWORD), ("high", wintypes.DWORD))

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            return {
                "pid": pid,
                "state": "ABSENT" if error == 87 else "UNOBSERVABLE",
                "winerror": int(error),
            }
        try:
            exit_code = wintypes.DWORD()
            creation = FileTime()
            exit_time = FileTime()
            kernel = FileTime()
            user = FileTime()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return {
                    "pid": pid,
                    "state": "UNOBSERVABLE",
                    "winerror": int(ctypes.get_last_error()),
                }
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                return {
                    "pid": pid,
                    "state": "UNOBSERVABLE",
                    "winerror": int(ctypes.get_last_error()),
                }
            token = f"{((int(creation.high) << 32) | int(creation.low)):016x}"
            state = "ACTIVE" if int(exit_code.value) == 259 else "EXITED"
            return {
                "pid": pid,
                "state": state,
                "start_token": token,
                "exit_code": None if state == "ACTIVE" else int(exit_code.value),
            }
        finally:
            kernel32.CloseHandle(handle)

    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        raw = stat_path.read_text(encoding="ascii")
        fields_after_name = raw[raw.rfind(")") + 2 :].split()
        token = f"{int(fields_after_name[19]):016x}"
    except FileNotFoundError:
        return {"pid": pid, "state": "ABSENT"}
    except (OSError, UnicodeDecodeError, ValueError, IndexError) as exc:
        return {
            "pid": pid,
            "state": "UNOBSERVABLE",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"pid": pid, "state": "ACTIVE", "start_token": token}


def _new_process_identity(pid: int, *, label: str) -> dict[str, object]:
    observation = _observe_pid(pid)
    if observation.get("state") != "ACTIVE" or not isinstance(
        observation.get("start_token"), str
    ):
        raise WorkspaceLocalError(f"{label} process identity is not observable")
    return {
        "pid": pid,
        "start_token": str(observation["start_token"]),
    }


def _current_parent_pid() -> int:
    """Return the current process' direct parent without spawning a helper."""

    if os.name != "nt":
        parent_pid = os.getppid()
        if parent_pid <= 0:
            raise WorkspaceLocalError("current parent PID is not observable")
        return parent_pid

    import ctypes
    from ctypes import wintypes

    class ProcessBasicInformation(ctypes.Structure):
        _fields_ = [
            ("reserved1", ctypes.c_void_p),
            ("peb_base_address", ctypes.c_void_p),
            ("reserved2_0", ctypes.c_void_p),
            ("reserved2_1", ctypes.c_void_p),
            ("unique_process_id", ctypes.c_size_t),
            ("inherited_from_unique_process_id", ctypes.c_size_t),
        ]

    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll.NtQueryInformationProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.ULONG,
        ctypes.POINTER(wintypes.ULONG),
    ]
    ntdll.NtQueryInformationProcess.restype = wintypes.LONG
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    information = ProcessBasicInformation()
    returned = wintypes.ULONG()
    status = int(
        ntdll.NtQueryInformationProcess(
            kernel32.GetCurrentProcess(),
            0,
            ctypes.byref(information),
            ctypes.sizeof(information),
            ctypes.byref(returned),
        )
    )
    if status != 0:
        raise WorkspaceLocalError(
            f"current parent PID is not observable (NTSTATUS 0x{status & 0xFFFFFFFF:08x})"
        )
    parent_pid = int(information.inherited_from_unique_process_id)
    if parent_pid <= 0:
        raise WorkspaceLocalError("current parent PID is not observable")
    return parent_pid


def _observe_expected_process(identity: object, *, label: str) -> dict[str, object]:
    if not isinstance(identity, dict) or set(identity) != {"pid", "start_token"}:
        raise WorkspaceLocalError(f"{label} process identity is malformed")
    pid = identity.get("pid")
    token = identity.get("start_token")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or not isinstance(token, str)
        or re.fullmatch(r"[0-9a-f]{16}", token) is None
    ):
        raise WorkspaceLocalError(f"{label} process identity is malformed")
    observation = _observe_pid(pid)
    state = observation.get("state")
    if state == "UNOBSERVABLE":
        raise WorkspaceLocalError(f"{label} process state is unobservable")
    current_token = observation.get("start_token")
    exact_identity_active = state == "ACTIVE" and current_token == token
    original_identity_absent = state == "ABSENT" or (
        isinstance(current_token, str) and current_token != token
    )
    exact_identity_exited = state == "EXITED" and current_token == token
    return {
        **observation,
        "expected_start_token": token,
        "exact_identity_active": exact_identity_active,
        "exact_identity_exited": exact_identity_exited,
        "original_identity_absent": original_identity_absent,
    }


def _wall_timeout(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("wall timeout must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0 or parsed > _MAX_WALL_TIMEOUT_SECONDS:
        raise argparse.ArgumentTypeError(
            f"wall timeout must be in (0, {_MAX_WALL_TIMEOUT_SECONDS:g}] seconds"
        )
    return parsed


def _logical_tree_usage(path: Path) -> dict[str, int]:
    """Measure logical mutable bytes without following links or reparse points."""

    regular_file_count = 0
    regular_file_bytes = 0
    directory_count = 0
    skipped_link_or_reparse_count = 0
    other_entry_count = 0
    observed_entry_count = 0
    stack = [path]
    while stack:
        selected = stack.pop()
        directory_count += 1
        with os.scandir(selected) as entries:
            for entry in entries:
                observed_entry_count += 1
                if observed_entry_count > _IMPORT_CLOSURE_FILE_COUNT_LIMIT:
                    raise WorkspaceLocalError(
                        "mutable filesystem telemetry exceeds entry bound"
                    )
                observed = entry.stat(follow_symlinks=False)
                attributes = getattr(observed, "st_file_attributes", 0)
                if entry.is_symlink() or attributes & getattr(
                    stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
                ):
                    skipped_link_or_reparse_count += 1
                elif stat.S_ISDIR(observed.st_mode):
                    stack.append(Path(entry.path))
                elif stat.S_ISREG(observed.st_mode):
                    regular_file_count += 1
                    regular_file_bytes += int(observed.st_size)
                else:
                    other_entry_count += 1
    return {
        "directory_count": directory_count,
        "observed_entry_count": observed_entry_count,
        "regular_file_count": regular_file_count,
        "logical_regular_file_bytes": regular_file_bytes,
        "skipped_link_or_reparse_count": skipped_link_or_reparse_count,
        "other_entry_count": other_entry_count,
    }


def _mutable_filesystem_usage(paths: dict[str, Path]) -> dict[str, object]:
    return {
        "measurement": "LOGICAL_REGULAR_BYTES_NO_LINK_FOLLOW",
        "roots": {
            name: {
                "path": str(path),
                **_logical_tree_usage(path),
            }
            for name, path in sorted(paths.items())
        },
    }


def pytest_addoption(parser: object) -> None:
    """Register the private append-only result path when loaded as a pytest plugin."""

    group = parser.getgroup("pfc-workspace-local")  # type: ignore[attr-defined]
    group.addoption(
        "--pfc-workspace-result-json",
        action="store",
        default=None,
        help="private path used by the workspace-local runner",
    )


def pytest_sessionfinish(session: object, exitstatus: object) -> None:
    """Persist native pytest outcome categories independently from JUnit."""

    config = session.config  # type: ignore[attr-defined]
    path_value = config.getoption("pfc_workspace_result_json")
    if path_value is None:
        return
    terminal = config.pluginmanager.get_plugin("terminalreporter")
    if terminal is None:
        raise WorkspaceLocalError("pytest terminal reporter is required")
    stats = terminal.stats
    payload: dict[str, object] = {
        "schema_version": _PYTEST_NATIVE_RESULT_SCHEMA,
        "exit_status": int(exitstatus),
        "selected": int(session.testscollected),  # type: ignore[attr-defined]
        "deselected": len(stats.get("deselected", [])),
        "passed": len(stats.get("passed", [])),
        "failed": len(stats.get("failed", [])),
        "errors": len(stats.get("error", [])),
        "skipped": len(stats.get("skipped", [])),
        "xfailed": len(stats.get("xfailed", [])),
        "xpassed": len(stats.get("xpassed", [])),
        "production_authorization": False,
        "promotion_authorization": False,
        "scientific_authorization": False,
        "evaluation_authorization": False,
        "runtime_authority": False,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path = Path(path_value)
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def prepare_run(root: Path, run_id: str) -> tuple[Path, dict[str, Path]]:
    if _RUN_ID.fullmatch(run_id) is None:
        raise WorkspaceLocalError("run-id must be a portable identifier of at most 16 characters")
    run_root = _require_within(
        root / "build" / "workspace-local-runs" / run_id,
        root,
        label="run root",
    )
    if run_root.exists():
        raise WorkspaceLocalError("run namespace already exists; rotate to a fresh run-id")
    run_root.parent.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(exist_ok=False)
    paths = {
        name: run_root / name
        for name in (
            "appdata",
            "conda-envs",
            "conda-pkgs",
            "home",
            "joblib-temp",
            "local-appdata",
            "matplotlib-cache",
            "numba-cache",
            "pip-cache",
            "programdata",
            "temp",
            "user-base",
            "uv-cache",
            "xdg-cache",
        )
    }
    # Deep atomic-publication fixtures exceed legacy Windows MAX_PATH when
    # pytest is nested below the descriptive receipt namespace. Keep this
    # fresh, receipt-bound mutable root directly below build/.
    paths["pytest-root"] = _require_within(
        root / "build" / f"wpt-{run_id}",
        root,
        label="pytest root",
    )
    receipt_path = run_root / "execution-receipt.json"
    preflight_receipt: dict[str, object] = {
        "schema_version": _RECEIPT_SCHEMA,
        "status": "PREFLIGHT_PENDING",
        "production_authorization": False,
        "promotion_authorization": False,
        "scientific_authorization": False,
        "evaluation_authorization": False,
        "runtime_authority": False,
        "created_at_utc": _utc_now(),
        "runner_process_id": os.getpid(),
        "runner_process": _new_process_identity(os.getpid(), label="runner"),
        "workspace": str(root),
        "run_root": str(run_root),
        "mutable_paths": {name: str(path) for name, path in sorted(paths.items())},
        "redacted_command": [],
    }
    _write_json(receipt_path, preflight_receipt)
    try:
        for path in paths.values():
            _probe_directory(path)
        _reject_reparse_chain(run_root, root)
        for path in paths.values():
            _reject_reparse_chain(path, root)
    except (OSError, WorkspaceLocalError) as exc:
        preflight_receipt["status"] = "PREFLIGHT_FAILED"
        preflight_receipt["failure_class"] = type(exc).__name__
        preflight_receipt["failure_message"] = str(exc)
        preflight_receipt["finished_at_utc"] = _utc_now()
        _write_json(receipt_path, preflight_receipt)
        raise
    return run_root, paths


def _removed_ambient_environment_keys(inherited: dict[str, str]) -> list[str]:
    return sorted(
        key
        for key in inherited
        if any(marker in key.upper() for marker in _SENSITIVE_ENV_MARKERS)
        or any(marker in key.upper() for marker in _AUTHORITY_ENV_MARKERS)
        or key.upper().startswith(_SENSITIVE_ENV_PREFIXES)
        or key.upper() in _AMBIENT_PATH_ENV_KEYS
    )


def build_environment(
    inherited: dict[str, str], run_root: Path, paths: dict[str, Path]
) -> tuple[dict[str, str], list[str]]:
    condarc = run_root / "condarc.yml"
    condarc.write_text(
        "\n".join(
            (
                "offline: true",
                "always_copy: true",
                "auto_activate_base: false",
                "register_envs: false",
                "pkgs_dirs:",
                f"  - {paths['conda-pkgs'].as_posix()}",
                "envs_dirs:",
                f"  - {paths['conda-envs'].as_posix()}",
                "",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    removed = _removed_ambient_environment_keys(inherited)
    env = {key: value for key, value in inherited.items() if key not in removed}
    env.update(
        {
            "APPDATA": str(paths["appdata"]),
            "CONDARC": str(condarc),
            "CONDA_ENVS_PATH": str(paths["conda-envs"]),
            "CONDA_PKGS_DIRS": str(paths["conda-pkgs"]),
            "CONDA_REGISTER_ENVS": "false",
            "HOME": str(paths["home"]),
            "JOBLIB_TEMP_FOLDER": str(paths["joblib-temp"]),
            "LOCALAPPDATA": str(paths["local-appdata"]),
            "MPLCONFIGDIR": str(paths["matplotlib-cache"]),
            "NUMBA_CACHE_DIR": str(paths["numba-cache"]),
            "PIP_CACHE_DIR": str(paths["pip-cache"]),
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUSERBASE": str(paths["user-base"]),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PROGRAMDATA": str(paths["programdata"]),
            "TEMP": str(paths["temp"]),
            "TMP": str(paths["temp"]),
            "TMPDIR": str(paths["temp"]),
            "USERPROFILE": str(paths["home"]),
            "UV_CACHE_DIR": str(paths["uv-cache"]),
            "UV_OFFLINE": "1",
            "XDG_CACHE_HOME": str(paths["xdg-cache"]),
        }
    )
    return env, removed


def _is_pytest(command: list[str]) -> bool:
    executable = Path(command[0]).name.lower()
    if executable in {"pytest", "pytest.exe"}:
        return True
    return any(
        command[index] == "-m" and command[index + 1] == "pytest"
        for index in range(len(command) - 1)
    )


def _python_module(command: list[str]) -> str:
    index = 1
    while index < len(command) and command[index] in _ALLOWED_PYTHON_FLAGS:
        index += 1
    if index >= len(command) or command[index] != "-m":
        raise WorkspaceLocalError(
            "Python invocation must be allowed flags followed immediately by -m"
        )
    if index + 1 >= len(command) or not command[index + 1]:
        raise WorkspaceLocalError("Python -m is missing its module")
    return command[index + 1]


def _validate_executable(command: list[str], root: Path) -> None:
    executable = Path(command[0])
    basename = executable.name.lower()
    if basename not in {"python", "python.exe"}:
        raise WorkspaceLocalError("only an explicit Python interpreter is permitted")
    if executable.is_absolute():
        if not executable.is_file():
            raise WorkspaceLocalError("selected Python interpreter is unavailable")
        selected = executable.resolve(strict=True)
    elif len(executable.parts) > 1:
        candidate = root / executable
        if not candidate.is_file():
            raise WorkspaceLocalError("selected Python interpreter is unavailable")
        selected = candidate.resolve(strict=True)
    else:
        selected = Path(sys.executable).resolve(strict=True)
    current = Path(sys.executable).resolve(strict=True)
    repo_runtime = False
    if _is_within(selected, root):
        relative = selected.relative_to(root)
        repo_runtime = (
            len(relative.parts) >= 3
            and relative.parts[0].lower() == "build"
            and relative.parts[1].lower().startswith("conda-runtime-")
            and relative.name.lower() == "python.exe"
        )
        _reject_reparse_chain(selected.parent, root)
    if selected != current and not repo_runtime:
        raise WorkspaceLocalError("Python interpreter is not the current or a repo runtime")
    command[0] = str(selected)
    module = _python_module(command)
    if module not in _ALLOWED_PYTHON_MODULES:
        raise WorkspaceLocalError(f"Python module is not allowlisted: {module}")


def _require_eex_datasource_v2_capture_boundary(command: list[str], root: Path) -> None:
    if _python_module(command) != _EEX_DATASOURCE_MODULE:
        return
    module_index = command.index("-m")
    if command[1:module_index] != ["-I", "-B"]:
        raise WorkspaceLocalError(
            "EEX DataSource v2 capture requires exact isolated Python flags -I -B"
        )
    interpreter = Path(command[0])
    if not interpreter.is_absolute() or not _is_within(interpreter, root):
        raise WorkspaceLocalError(
            "EEX DataSource v2 capture requires a repo-local governed runtime"
        )
    relative_interpreter = interpreter.relative_to(root)
    if not (
        len(relative_interpreter.parts) >= 3
        and relative_interpreter.parts[0].lower() == "build"
        and relative_interpreter.parts[1].lower().startswith("conda-runtime-")
        and relative_interpreter.name.lower() == "python.exe"
    ):
        raise WorkspaceLocalError(
            "EEX DataSource v2 capture runtime is outside the admitted namespace"
        )
    arguments = command[module_index + 2 :]
    if len(arguments) != 6 or arguments[0::2] != [
        "--repo-root",
        "--spec",
        "--expected-spec-sha256",
    ]:
        raise WorkspaceLocalError(
            "EEX DataSource v2 capture requires exact governed arguments"
        )
    selected_root = Path(arguments[1])
    if not selected_root.is_absolute() or os.path.normcase(
        os.path.abspath(selected_root)
    ) != os.path.normcase(os.path.abspath(root)):
        raise WorkspaceLocalError("EEX DataSource v2 repo root must be canonical")
    selected_spec = Path(arguments[3])
    if not selected_spec.is_absolute():
        raise WorkspaceLocalError("EEX DataSource v2 spec path must be absolute")
    _require_within(selected_spec, root / "build", label="EEX DataSource v2 spec")
    _reject_reparse_chain(selected_spec, root)
    if selected_spec.suffix.lower() != ".json" or not selected_spec.is_file():
        raise WorkspaceLocalError("EEX DataSource v2 spec is unavailable")
    if re.fullmatch(r"[0-9a-f]{64}", arguments[5]) is None:
        raise WorkspaceLocalError("EEX DataSource v2 expected spec SHA-256 is invalid")


def _selected_eex_datasource_token(
    command: list[str], environment: dict[str, str]
) -> str | None:
    if not _command_targets_eex_datasource(command):
        return None
    token = environment.get(_EEX_DATASOURCE_TOKEN_ENV)
    return token if token is not None and _EEX_DATASOURCE_TOKEN.fullmatch(token) else None


def _command_targets_eex_datasource(command: list[str]) -> bool:
    selected = list(command)
    if selected and selected[0] == "--":
        selected = selected[1:]
    try:
        return bool(selected and _python_module(selected) == _EEX_DATASOURCE_MODULE)
    except WorkspaceLocalError:
        return False


def _forwarded_token_in_bytes(value: bytes, token: str) -> bool:
    token_bytes = token.encode("ascii")
    variants = {
        token_bytes,
        base64.b64encode(token_bytes),
        base64.b64encode(token_bytes).rstrip(b"="),
        base64.urlsafe_b64encode(token_bytes),
        base64.urlsafe_b64encode(token_bytes).rstrip(b"="),
    }
    current = value
    for _ in range(4):
        compact = current.translate(None, b" \t\r\n")
        if any(candidate in current or candidate in compact for candidate in variants):
            return True
        decoded = unquote_to_bytes(current)
        if decoded == current:
            return False
        current = decoded
    compact = current.translate(None, b" \t\r\n")
    if any(candidate in current or candidate in compact for candidate in variants):
        return True
    return unquote_to_bytes(current) != current


def _reject_forwarded_token_in_command(command: list[str], token: str | None) -> None:
    if token is None:
        return
    if any(_forwarded_token_in_bytes(value.encode("utf-8"), token) for value in command):
        raise WorkspaceLocalError(
            "EEX DataSource credential material is forbidden in command arguments"
        )


def _require_launcherless_build_paths(command: list[str], root: Path) -> None:
    """Keep every explicit runtime build input/output below repo ``build/``."""

    if _python_module(command) != "scripts.build_launcherless_local_runtime":
        return
    build_root = root / "build"
    option_values: dict[str, str] = {}
    index = 0
    while index < len(command):
        argument = command[index]
        option, separator, inline_value = argument.partition("=")
        if option in _LAUNCHERLESS_BUILD_PATH_OPTIONS or option == "--lock-path":
            if separator:
                value = inline_value
            else:
                if index + 1 >= len(command):
                    raise WorkspaceLocalError(f"{option} is missing its path")
                value = command[index + 1]
                index += 1
            if not value:
                raise WorkspaceLocalError(f"{option} is missing its path")
            option_values[option] = value
        index += 1
    missing = sorted(_LAUNCHERLESS_BUILD_PATH_OPTIONS - option_values.keys())
    if missing:
        raise WorkspaceLocalError(
            "launcherless runtime command is missing governed path options: " + ", ".join(missing)
        )
    for option in sorted(_LAUNCHERLESS_BUILD_PATH_OPTIONS):
        selected = Path(option_values[option])
        if not selected.is_absolute():
            selected = root / selected
        if not _is_within(selected, build_root):
            raise WorkspaceLocalError(f"{option} must remain below repo build/")
    if "--lock-path" in option_values:
        selected_lock = Path(option_values["--lock-path"])
        if not selected_lock.is_absolute():
            selected_lock = root / selected_lock
        if _lexical(selected_lock) != _lexical(root / "uv.lock"):
            raise WorkspaceLocalError("--lock-path must be the canonical repo uv.lock")


def _require_eex_vintage_build_paths(command: list[str], root: Path) -> None:
    """Confine all EEX workflow evidence and output below repo ``build/``."""

    if _python_module(command) != "pfc_shaping.cli.eex_forward_vintage_builder":
        return
    module_index = command.index("-m")
    launcher_flags = set(command[1:module_index])
    if not {"-I", "-B"}.issubset(launcher_flags):
        raise WorkspaceLocalError("EEX vintage command requires isolated Python -I -B")
    path_options = _EEX_VINTAGE_REQUIRED_PATH_OPTIONS | _EEX_VINTAGE_OPTIONAL_PATH_OPTIONS
    option_values: dict[str, str] = {}
    index = 0
    while index < len(command):
        argument = command[index]
        option, separator, inline_value = argument.partition("=")
        if option in path_options:
            if separator:
                value = inline_value
            else:
                if index + 1 >= len(command):
                    raise WorkspaceLocalError(f"{option} is missing its path")
                value = command[index + 1]
                index += 1
            if not value:
                raise WorkspaceLocalError(f"{option} is missing its path")
            option_values[option] = value
        index += 1
    missing = sorted(_EEX_VINTAGE_REQUIRED_PATH_OPTIONS - option_values.keys())
    if missing:
        raise WorkspaceLocalError(
            "EEX vintage command is missing governed path options: " + ", ".join(missing)
        )
    previous_present = {
        option for option in _EEX_VINTAGE_OPTIONAL_PATH_OPTIONS if option in option_values
    }
    if previous_present and previous_present != _EEX_VINTAGE_OPTIONAL_PATH_OPTIONS:
        raise WorkspaceLocalError(
            "previous EEX catalog, history, and public key must be supplied together"
        )
    build_root = root / "build"
    for option, value in sorted(option_values.items()):
        selected = Path(value)
        if not selected.is_absolute():
            selected = root / selected
        if not _is_within(selected, build_root):
            raise WorkspaceLocalError(f"{option} must remain below repo build/")


def _require_local_panel_paths(command: list[str], root: Path) -> None:
    """Confine the explicitly local, non-authoritative panel workflow."""

    if _python_module(command) != "scripts.build_local_intraday_calibration_panel":
        return
    module_index = command.index("-m")
    if "-B" not in command[1:module_index]:
        raise WorkspaceLocalError("local panel command requires Python -B")
    build_root = root / "build"
    observed: set[str] = set()
    index = module_index + 2
    while index < len(command):
        argument = command[index]
        option, separator, inline_value = argument.partition("=")
        if option in _LOCAL_PANEL_PATH_OPTIONS:
            if separator:
                value = inline_value
            else:
                if index + 1 >= len(command):
                    raise WorkspaceLocalError(f"{option} is missing its path")
                value = command[index + 1]
                index += 1
            if not value:
                raise WorkspaceLocalError(f"{option} is missing its path")
            selected = Path(value)
            if not selected.is_absolute():
                selected = root / selected
            if not _is_within(selected, build_root):
                raise WorkspaceLocalError(f"{option} must remain below repo build/")
            observed.add(option)
        index += 1
    missing = sorted(_LOCAL_PANEL_PATH_OPTIONS - observed)
    if missing:
        raise WorkspaceLocalError(
            "local panel command is missing governed path options: " + ", ".join(missing)
        )


def _require_local_model_boundary(command: list[str], root: Path) -> None:
    """Confine the deterministic CH local-model smoke run and freeze its knobs."""

    if _python_module(command) != "scripts.build_local_test_ch_pfc":
        return
    module_index = command.index("-m")
    if command[1:module_index] != ["-B"]:
        raise WorkspaceLocalError("local CH model command requires exact Python flags: -B")
    accepted_values = (
        _LOCAL_MODEL_INPUT_PATH_OPTIONS
        | _LOCAL_MODEL_OUTPUT_PATH_OPTIONS
        | _LOCAL_MODEL_VALUE_OPTIONS
    )
    option_values: dict[str, str] = {}
    flags: set[str] = set()
    arguments = command[module_index + 2 :]
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        option, separator, inline_value = argument.partition("=")
        if option in _LOCAL_MODEL_FLAG_OPTIONS:
            if separator or option in flags:
                raise WorkspaceLocalError(f"{option} must be supplied exactly once")
            flags.add(option)
            index += 1
            continue
        if option not in accepted_values:
            raise WorkspaceLocalError(f"local CH model argument is not permitted: {argument}")
        if separator:
            value = inline_value
        else:
            if index + 1 >= len(arguments):
                raise WorkspaceLocalError(f"{option} is missing its value")
            value = arguments[index + 1]
            index += 1
        if not value or option in option_values:
            raise WorkspaceLocalError(f"{option} must be supplied exactly once")
        option_values[option] = value
        index += 1
    missing_values = sorted(accepted_values - option_values.keys())
    if missing_values:
        raise WorkspaceLocalError(
            "local CH model command is missing required options: "
            + ", ".join(missing_values)
        )
    missing_flags = sorted(_LOCAL_MODEL_FLAG_OPTIONS - flags)
    if missing_flags:
        raise WorkspaceLocalError(
            "local CH model command is missing required flags: " + ", ".join(missing_flags)
        )

    for option, relative in sorted(_LOCAL_MODEL_CANONICAL_INPUTS.items()):
        selected = Path(option_values[option])
        if not selected.is_absolute():
            selected = root / selected
        if _lexical(selected) != _lexical(root / relative):
            raise WorkspaceLocalError(f"{option} must be the canonical read-only input")

    output_dir = Path(option_values["--output-dir"])
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    governed_parent = root / "build" / "local-model-runs"
    if (
        _lexical(output_dir.parent) != _lexical(governed_parent)
        or _PORTABLE_CAPTURE_ID.fullmatch(output_dir.name) is None
    ):
        raise WorkspaceLocalError(
            "--output-dir must be one portable direct child of build/local-model-runs"
        )
    for option, filename in sorted(_LOCAL_MODEL_OUTPUT_FILENAMES.items()):
        selected = Path(option_values[option])
        if not selected.is_absolute():
            selected = root / selected
        if _lexical(selected) != _lexical(output_dir / filename):
            raise WorkspaceLocalError(f"{option} must be {filename} inside --output-dir")

    exact_values = {
        "--delivery-local-end-exclusive": "2031-01-01T00:00:00",
        "--expected-intraday-provenance-sha256": (
            "af8232e7045eb9aa685d946770a56a108e438a6a71e9cbdab47614abcc5a9e4f"
        ),
        "--intraday-cutoff": "2025-10-01T00:00:00Z",
        "--intraday-market": "DE",
        "--market": "CH",
        "--require-direct-intraday-seasons": "Hiver,Printemps,Ete,Automne",
        "--scenarios": "slow,central,fast",
        "--start-date": "2026-07-31T22:00:00Z",
        "--horizon-days": "1614",
        "--vintage": "2026-06-12",
        "--weights": "0.25,0.50,0.25",
    }
    for option, expected in exact_values.items():
        if option_values[option] != expected:
            raise WorkspaceLocalError(f"{option} must remain frozen at {expected}")
    if _PORTABLE_CAPTURE_ID.fullmatch(option_values["--output-prefix"]) is None:
        raise WorkspaceLocalError("--output-prefix must be a portable identifier")
    try:
        horizon_days = int(option_values["--horizon-days"])
    except ValueError as exc:
        raise WorkspaceLocalError("--horizon-days must be an integer") from exc
    if not 1 <= horizon_days <= 2_000:
        raise WorkspaceLocalError("--horizon-days must be between 1 and 2000")
    try:
        datetime.fromisoformat(option_values["--start-date"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise WorkspaceLocalError("--start-date must be an ISO-8601 timestamp") from exc
    exact_floats = {
        "--monthly-solver-constraint-tolerance": 1e-9,
        "--monthly-solver-lambda-shape": 1.0,
        "--monthly-solver-lambda-smooth-month": 1.0,
        "--monthly-solver-lambda-smooth-yoy": 0.25,
        "--monthly-solver-neighbor-shrinkage": 0.5,
        "--monthly-solver-structural-amplitude": 110.0,
    }
    for option, expected in exact_floats.items():
        try:
            observed = float(option_values[option])
        except ValueError as exc:
            raise WorkspaceLocalError(f"{option} must be numeric") from exc
        if not math.isfinite(observed) or observed != expected:
            raise WorkspaceLocalError(f"{option} must remain frozen at {expected}")


def _require_laptop_model_qualification_boundary(command: list[str], root: Path) -> None:
    """Admit only publication of the exact frozen local A/B qualification."""

    if _python_module(command) != "scripts.audit_ch_lt_laptop_model_pair":
        return
    module_index = command.index("-m")
    if command[1:module_index] != ["-B"]:
        raise WorkspaceLocalError(
            "laptop model qualification requires exact Python flags: -B"
        )
    arguments = command[module_index + 2 :]
    if len(arguments) != 2 or arguments[0] != "--output-directory":
        raise WorkspaceLocalError(
            "laptop model qualification requires only --output-directory"
        )
    selected = Path(arguments[1])
    if not selected.is_absolute():
        selected = root / selected
    if _lexical(selected) != _lexical(root / _LAPTOP_MODEL_QUALIFICATION_OUTPUT):
        raise WorkspaceLocalError(
            "--output-directory must be the canonical laptop qualification path"
        )


def _require_local_quality_selection_boundary(command: list[str]) -> None:
    """Admit only the argument-free canonical current-quality audit."""

    if _python_module(command) != "scripts.audit_ch_lt_local_quality_current_selection":
        return
    module_index = command.index("-m")
    if command[1:module_index] != ["-B"] or command[module_index + 2 :]:
        raise WorkspaceLocalError(
            "current local quality selection audit requires exact Python flags -B and no arguments"
        )


def _require_structural_prediction_commitment_boundary(
    command: list[str], root: Path
) -> None:
    """Admit only the canonical truth-blind, non-countable dry-run output."""

    if _python_module(command) != "scripts.build_ch_lt_structural_prediction_commitment":
        return
    module_index = command.index("-m")
    if command[1:module_index] != ["-B"]:
        raise WorkspaceLocalError(
            "structural prediction commitment requires exact Python flags: -B"
        )
    arguments = command[module_index + 2 :]
    if len(arguments) != 6 or arguments[0::2] != [
        "--inventory",
        "--expected-inventory-sha256",
        "--output-directory",
    ]:
        raise WorkspaceLocalError(
            "structural prediction commitment requires exact inventory, hash and output options"
        )
    inventory = Path(arguments[1])
    if not inventory.is_absolute() or _lexical(inventory) != _lexical(
        root / _STRUCTURAL_PREDICTION_COMMITMENT_INVENTORY
    ):
        raise WorkspaceLocalError("--inventory must be the selected structural inventory")
    if arguments[3] != _STRUCTURAL_PREDICTION_COMMITMENT_INVENTORY_SHA256:
        raise WorkspaceLocalError(
            "--expected-inventory-sha256 must match the selected structural inventory"
        )
    selected = Path(arguments[5])
    if not selected.is_absolute() or _lexical(selected) != _lexical(
        root / _STRUCTURAL_PREDICTION_COMMITMENT_OUTPUT
    ):
        raise WorkspaceLocalError(
            "--output-directory must be the canonical structural commitment path"
        )


def _require_local_future_origin_selection_boundary(
    command: list[str], root: Path
) -> None:
    """Admit only the exact read-only audit of the selected local future origin."""

    module = _python_module(command)
    local_module = "scripts.audit_ch_lt_local_future_origin_selection"
    packaged_module = _LOCAL_FUTURE_ORIGIN_PACKAGED_MODULE
    if module not in {local_module, packaged_module}:
        return
    module_index = command.index("-m")
    expected_flags = ["-B"] if module == local_module else ["-I", "-B"]
    if command[1:module_index] != expected_flags:
        raise WorkspaceLocalError(
            "local future origin audit requires exact Python flags: "
            + " ".join(expected_flags)
        )
    if module == packaged_module:
        interpreter = Path(command[0])
        if not interpreter.is_absolute() or not _is_within(interpreter, root):
            raise WorkspaceLocalError(
                "packaged local future origin audit requires a repo-local runtime"
            )
        relative_interpreter = interpreter.relative_to(root)
        if not (
            len(relative_interpreter.parts) >= 3
            and relative_interpreter.parts[0].lower() == "build"
            and relative_interpreter.parts[1].lower().startswith("conda-runtime-")
            and relative_interpreter.name.lower() == "python.exe"
        ):
            raise WorkspaceLocalError(
                "packaged local future origin runtime is outside the admitted namespace"
            )
    arguments = command[module_index + 2 :]
    if len(arguments) != 4 or arguments[0::2] != [
        "--registry",
        "--expected-registry-sha256",
    ]:
        raise WorkspaceLocalError(
            "local future origin audit requires exact registry and hash options"
        )
    registry = Path(arguments[1])
    if not registry.is_absolute() or _lexical(registry) != _lexical(
        root / _LOCAL_FUTURE_ORIGIN_SELECTION_REGISTRY
    ):
        raise WorkspaceLocalError(
            "--registry must be the selected local future origin registry"
        )
    if arguments[3] != _LOCAL_FUTURE_ORIGIN_SELECTION_SHA256:
        raise WorkspaceLocalError(
            "--expected-registry-sha256 must match the selected local future origin registry"
        )


def _require_energy_charts_capture_boundary(command: list[str], root: Path) -> None:
    """Admit only the native-hourly CH capture into a repo-local namespace."""

    if _python_module(command) != "scripts.capture_public_energy_charts_lt":
        return
    option_values: dict[str, str] = {}
    accepted = _ENERGY_CHARTS_CAPTURE_REQUIRED_OPTIONS
    index = 0
    while index < len(command):
        argument = command[index]
        option, separator, inline_value = argument.partition("=")
        if option in accepted:
            if separator:
                value = inline_value
            else:
                if index + 1 >= len(command):
                    raise WorkspaceLocalError(f"{option} is missing its value")
                value = command[index + 1]
                index += 1
            if not value:
                raise WorkspaceLocalError(f"{option} is missing its value")
            if option in option_values:
                raise WorkspaceLocalError(f"{option} must be supplied exactly once")
            option_values[option] = value
        index += 1
    missing = sorted(_ENERGY_CHARTS_CAPTURE_REQUIRED_OPTIONS - option_values.keys())
    if missing:
        raise WorkspaceLocalError(
            "Energy Charts capture command is missing required options: " + ", ".join(missing)
        )
    if option_values["--role"] != "epex_ch":
        raise WorkspaceLocalError("workspace capture route is restricted to epex_ch")
    if option_values["--raw-cadence-minutes"] != "60":
        raise WorkspaceLocalError(
            "workspace CH day-ahead capture requires native hourly cadence 60"
        )
    output = Path(option_values["--output-directory"])
    if not output.is_absolute():
        output = root / output
    if not _is_within(output, root / "build"):
        raise WorkspaceLocalError("--output-directory must remain below repo build/")
    if "--ca-bundle" in option_values:
        ca_bundle = Path(option_values["--ca-bundle"])
        if not ca_bundle.is_absolute() or not ca_bundle.is_file():
            raise WorkspaceLocalError("--ca-bundle must be an existing absolute file")
        if ca_bundle.drive.lower() != root.drive.lower():
            raise WorkspaceLocalError("--ca-bundle must remain on canonical C drive")


def _require_eex_local_capture_boundary(command: list[str], root: Path) -> None:
    """Admit only the canonical read-only FMV workbook into repo ``build/``."""

    if _python_module(command) != "scripts.capture_eex_forward_local":
        return
    module_index = command.index("-m")
    if "-B" not in command[1:module_index]:
        raise WorkspaceLocalError("local EEX capture requires Python -B")
    option_values: dict[str, str] = {}
    index = module_index + 2
    while index < len(command):
        argument = command[index]
        option, separator, inline_value = argument.partition("=")
        if option not in _EEX_LOCAL_CAPTURE_REQUIRED_OPTIONS:
            raise WorkspaceLocalError(
                f"local EEX capture argument is not permitted: {argument}"
            )
        if separator:
            value = inline_value
        else:
            if index + 1 >= len(command):
                raise WorkspaceLocalError(f"{option} is missing its value")
            value = command[index + 1]
            index += 1
        if not value:
            raise WorkspaceLocalError(f"{option} is missing its value")
        if option in option_values:
            raise WorkspaceLocalError(f"{option} must be supplied exactly once")
        option_values[option] = value
        index += 1
    missing = sorted(_EEX_LOCAL_CAPTURE_REQUIRED_OPTIONS - option_values.keys())
    if missing:
        raise WorkspaceLocalError(
            "local EEX capture command is missing required options: " + ", ".join(missing)
        )
    source = Path(option_values["--source-document"])
    if not source.is_absolute() or _lexical(source) != _lexical(_FMV_EEX_WORKBOOK_SOURCE):
        raise WorkspaceLocalError(
            "--source-document must be the canonical read-only FMV EEX workbook"
        )
    if _is_within(source, root):
        raise WorkspaceLocalError("--source-document must remain external to the workspace")
    if _LOWER_SHA256.fullmatch(option_values["--expected-source-sha256"]) is None:
        raise WorkspaceLocalError("--expected-source-sha256 must be lowercase SHA-256")
    if _PORTABLE_CAPTURE_ID.fullmatch(option_values["--capture-id"]) is None:
        raise WorkspaceLocalError("--capture-id must be a portable identifier")


def _require_eex_current_selection_boundary(command: list[str], root: Path) -> None:
    """Admit only exact read-only audits of the current CH EEX selection."""

    if _python_module(command) not in {
        "scripts.audit_ch_eex_current_local_capture",
        "scripts.audit_ch_eex_current_repricing_sensitivity",
    }:
        return
    module_index = command.index("-m")
    if "-B" not in command[1:module_index]:
        raise WorkspaceLocalError("current EEX selection audit requires Python -B")
    option_values: dict[str, str] = {}
    index = module_index + 2
    while index < len(command):
        argument = command[index]
        option, separator, inline_value = argument.partition("=")
        if option not in _EEX_CURRENT_SELECTION_REQUIRED_OPTIONS:
            raise WorkspaceLocalError(
                f"current EEX selection argument is not permitted: {argument}"
            )
        if separator:
            value = inline_value
        else:
            if index + 1 >= len(command):
                raise WorkspaceLocalError(f"{option} is missing its value")
            value = command[index + 1]
            index += 1
        if not value:
            raise WorkspaceLocalError(f"{option} is missing its value")
        if option in option_values:
            raise WorkspaceLocalError(f"{option} must be supplied exactly once")
        option_values[option] = value
        index += 1
    missing = sorted(_EEX_CURRENT_SELECTION_REQUIRED_OPTIONS - option_values.keys())
    if missing:
        raise WorkspaceLocalError(
            "current EEX selection command is missing required options: "
            + ", ".join(missing)
        )
    registry = Path(option_values["--registry"])
    expected_registry = root / _EEX_CURRENT_SELECTION_REGISTRY
    if not registry.is_absolute():
        registry = root / registry
    if _lexical(registry) != _lexical(expected_registry):
        raise WorkspaceLocalError(
            "--registry must be the canonical current CH EEX selection"
        )
    if option_values["--expected-registry-sha256"] != _EEX_CURRENT_SELECTION_SHA256:
        raise WorkspaceLocalError(
            "--expected-registry-sha256 must match the current CH EEX selection"
        )


def _require_preregistration_supersession_boundary(command: list[str], root: Path) -> None:
    """Admit only the exact read-only D164 supersession verification."""

    module = _python_module(command)
    if module not in {
        "scripts.verify_ch_lt_preregistration_supersession",
        "pfc_shaping.cli.verify_ch_lt_preregistration_supersession",
    }:
        return
    module_index = command.index("-m")
    if "-B" not in command[1:module_index]:
        raise WorkspaceLocalError(
            "preregistration supersession verification requires Python -B"
        )
    required_options = set(_PREREGISTRATION_SUPERSESSION_REQUIRED_OPTIONS)
    if module == "pfc_shaping.cli.verify_ch_lt_preregistration_supersession":
        if "-I" not in command[1:module_index]:
            raise WorkspaceLocalError(
                "packaged preregistration supersession verification requires Python -I"
            )
        required_options.add("--evidence-root")
    option_values: dict[str, str] = {}
    index = module_index + 2
    while index < len(command):
        argument = command[index]
        option, separator, inline_value = argument.partition("=")
        if option not in required_options:
            raise WorkspaceLocalError(
                f"preregistration supersession argument is not permitted: {argument}"
            )
        if separator:
            value = inline_value
        else:
            if index + 1 >= len(command):
                raise WorkspaceLocalError(f"{option} is missing its value")
            value = command[index + 1]
            index += 1
        if not value:
            raise WorkspaceLocalError(f"{option} is missing its value")
        if option in option_values:
            raise WorkspaceLocalError(f"{option} must be supplied exactly once")
        option_values[option] = value
        index += 1
    missing = sorted(
        required_options - option_values.keys()
    )
    if missing:
        raise WorkspaceLocalError(
            "preregistration supersession command is missing required options: "
            + ", ".join(missing)
        )
    registry = Path(option_values["--registry"])
    expected_registry = root / _PREREGISTRATION_SUPERSESSION_REGISTRY
    if not registry.is_absolute():
        registry = root / registry
    if _lexical(registry) != _lexical(expected_registry):
        raise WorkspaceLocalError("--registry must be the canonical D164 registry")
    if "--evidence-root" in option_values:
        evidence_root = Path(option_values["--evidence-root"])
        if not evidence_root.is_absolute() or _lexical(evidence_root) != _lexical(root):
            raise WorkspaceLocalError("--evidence-root must be the canonical workspace")
    if (
        option_values["--expected-registry-sha256"]
        != _PREREGISTRATION_SUPERSESSION_SHA256
    ):
        raise WorkspaceLocalError(
            "--expected-registry-sha256 must match the canonical D164 registry"
        )


def _require_successor_candidate_core_boundary(command: list[str], root: Path) -> None:
    """Admit only the exact read-only, outcome-blind v3 core audit."""

    module = _python_module(command)
    local_module = "scripts.audit_ch_lt_successor_candidate_core"
    if module != local_module:
        return
    module_index = command.index("-m")
    launcher_flags = command[1:module_index]
    expected_flags = ["-B"]
    if launcher_flags != expected_flags:
        raise WorkspaceLocalError(
            "successor candidate core audit requires exact Python flags: "
            + " ".join(expected_flags)
        )
    required_options = set(_SUCCESSOR_CANDIDATE_CORE_REQUIRED_OPTIONS)
    option_values: dict[str, str] = {}
    index = module_index + 2
    while index < len(command):
        argument = command[index]
        option, separator, inline_value = argument.partition("=")
        if option not in required_options:
            raise WorkspaceLocalError(
                f"successor candidate core argument is not permitted: {argument}"
            )
        if separator:
            value = inline_value
        else:
            if index + 1 >= len(command):
                raise WorkspaceLocalError(f"{option} is missing its value")
            value = command[index + 1]
            index += 1
        if not value:
            raise WorkspaceLocalError(f"{option} is missing its value")
        if option in option_values:
            raise WorkspaceLocalError(f"{option} must be supplied exactly once")
        option_values[option] = value
        index += 1
    missing = sorted(required_options - option_values.keys())
    if missing:
        raise WorkspaceLocalError(
            "successor candidate core command is missing required options: "
            + ", ".join(missing)
        )
    core = Path(option_values["--core"])
    if not core.is_absolute() or _lexical(core) != _lexical(root / _SUCCESSOR_CANDIDATE_CORE):
        raise WorkspaceLocalError("--core must be the canonical v3 candidate core")
    if option_values["--expected-core-sha256"] != _SUCCESSOR_CANDIDATE_CORE_SHA256:
        raise WorkspaceLocalError(
            "--expected-core-sha256 must match the canonical v3 candidate core"
        )


def _require_dependence_power_design_boundary(command: list[str], root: Path) -> None:
    """Admit only the exact local read-only dependence/power design audit."""

    module = _python_module(command)
    if module != "scripts.audit_ch_lt_dependence_power_design":
        return
    module_index = command.index("-m")
    if command[1:module_index] != ["-B"]:
        raise WorkspaceLocalError(
            "dependence power design audit requires exact Python flags: -B"
        )
    required_options = set(_DEPENDENCE_POWER_DESIGN_REQUIRED_OPTIONS)
    option_values: dict[str, str] = {}
    index = module_index + 2
    while index < len(command):
        argument = command[index]
        option, separator, inline_value = argument.partition("=")
        if option not in required_options:
            raise WorkspaceLocalError(
                f"dependence power design argument is not permitted: {argument}"
            )
        if separator:
            value = inline_value
        else:
            if index + 1 >= len(command):
                raise WorkspaceLocalError(f"{option} is missing its value")
            value = command[index + 1]
            index += 1
        if not value:
            raise WorkspaceLocalError(f"{option} is missing its value")
        if option in option_values:
            raise WorkspaceLocalError(f"{option} must be supplied exactly once")
        option_values[option] = value
        index += 1
    missing = sorted(required_options - option_values.keys())
    if missing:
        raise WorkspaceLocalError(
            "dependence power design command is missing required options: "
            + ", ".join(missing)
        )
    design = Path(option_values["--design"])
    if not design.is_absolute() or _lexical(design) != _lexical(
        root / _DEPENDENCE_POWER_DESIGN
    ):
        raise WorkspaceLocalError("--design must be the canonical dependence power design")
    if option_values["--expected-design-sha256"] != _DEPENDENCE_POWER_DESIGN_SHA256:
        raise WorkspaceLocalError(
            "--expected-design-sha256 must match the canonical dependence power design"
        )


def _require_origin_target_mask_inventory_boundary(command: list[str], root: Path) -> None:
    """Confine the outcome-blind structural inventory workflow to repo build/."""

    module = _python_module(command)
    local_module = "scripts.audit_ch_lt_origin_target_mask_inventory"
    packaged_module = "pfc_shaping.cli.audit_ch_lt_origin_target_mask_inventory"
    if module not in {local_module, packaged_module}:
        return
    module_index = command.index("-m")
    expected_flags = ["-B"] if module == local_module else ["-I", "-B"]
    if command[1:module_index] != expected_flags:
        raise WorkspaceLocalError(
            "origin target mask inventory requires exact Python flags: "
            + " ".join(expected_flags)
        )
    arguments = command[module_index + 2 :]
    if module == packaged_module:
        if len(arguments) < 2 or arguments[0] != "--evidence-root":
            raise WorkspaceLocalError(
                "packaged origin target mask inventory requires --evidence-root first"
            )
        evidence_root = Path(arguments[1])
        if not evidence_root.is_absolute() or _lexical(evidence_root) != _lexical(root):
            raise WorkspaceLocalError("--evidence-root must be the canonical workspace")
        arguments = arguments[2:]
    if not arguments or arguments[0] not in {"build", "audit"}:
        raise WorkspaceLocalError("origin target mask inventory action must be build or audit")
    action = arguments[0]
    required = (
        {"--origin-as-of-utc", "--output"}
        if action == "build"
        else {"--inventory", "--expected-inventory-sha256"}
    )
    option_values: dict[str, str] = {}
    index = 1
    while index < len(arguments):
        argument = arguments[index]
        option, separator, inline_value = argument.partition("=")
        if option not in required:
            raise WorkspaceLocalError(
                f"origin target mask inventory argument is not permitted: {argument}"
            )
        if separator:
            value = inline_value
        else:
            if index + 1 >= len(arguments):
                raise WorkspaceLocalError(f"{option} is missing its value")
            value = arguments[index + 1]
            index += 1
        if not value or option in option_values:
            raise WorkspaceLocalError(f"{option} must be supplied exactly once")
        option_values[option] = value
        index += 1
    missing = sorted(required - option_values.keys())
    if missing:
        raise WorkspaceLocalError(
            "origin target mask inventory command is missing required options: "
            + ", ".join(missing)
        )
    path_option = "--output" if action == "build" else "--inventory"
    selected = Path(option_values[path_option])
    expected_parent = root / _ORIGIN_TARGET_MASK_OUTPUT_DIRECTORY
    if (
        not selected.is_absolute()
        or _lexical(selected.parent) != _lexical(expected_parent)
        or selected.suffix != ".json"
    ):
        raise WorkspaceLocalError(
            f"{path_option} must be a direct JSON child of the governed build namespace"
        )
    if action == "build":
        try:
            datetime.strptime(option_values["--origin-as-of-utc"], "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as exc:
            raise WorkspaceLocalError(
                "--origin-as-of-utc must use YYYY-MM-DDTHH:MM:SSZ"
            ) from exc
    elif _LOWER_SHA256.fullmatch(
        option_values["--expected-inventory-sha256"]
    ) is None:
        raise WorkspaceLocalError("--expected-inventory-sha256 must be lowercase SHA-256")


def _require_origin_registry_protocol_boundary(command: list[str], root: Path) -> None:
    """Admit only the exact outcome-blind origin-registry protocol audit."""

    module = _python_module(command)
    local_module = "scripts.audit_ch_lt_origin_registry_protocol"
    packaged_module = "pfc_shaping.cli.audit_ch_lt_origin_registry_protocol"
    if module not in {local_module, packaged_module}:
        return
    module_index = command.index("-m")
    expected_flags = ["-B"] if module == local_module else ["-I", "-B"]
    if command[1:module_index] != expected_flags:
        raise WorkspaceLocalError(
            "origin registry protocol audit requires exact Python flags: "
            + " ".join(expected_flags)
        )
    arguments = command[module_index + 2 :]
    if module == packaged_module:
        if len(arguments) < 2 or arguments[0] != "--evidence-root":
            raise WorkspaceLocalError(
                "packaged origin registry protocol audit requires --evidence-root first"
            )
        evidence_root = Path(arguments[1])
        if not evidence_root.is_absolute() or _lexical(evidence_root) != _lexical(root):
            raise WorkspaceLocalError("--evidence-root must be the canonical workspace")
        arguments = arguments[2:]
    required_options = set(_ORIGIN_REGISTRY_PROTOCOL_REQUIRED_OPTIONS)
    option_values: dict[str, str] = {}
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        option, separator, inline_value = argument.partition("=")
        if option not in required_options:
            raise WorkspaceLocalError(
                f"origin registry protocol argument is not permitted: {argument}"
            )
        if separator:
            value = inline_value
        else:
            if index + 1 >= len(arguments):
                raise WorkspaceLocalError(f"{option} is missing its value")
            value = arguments[index + 1]
            index += 1
        if not value or option in option_values:
            raise WorkspaceLocalError(f"{option} must be supplied exactly once")
        option_values[option] = value
        index += 1
    missing = sorted(required_options - option_values.keys())
    if missing:
        raise WorkspaceLocalError(
            "origin registry protocol command is missing required options: "
            + ", ".join(missing)
        )
    protocol = Path(option_values["--protocol"])
    if not protocol.is_absolute() or _lexical(protocol) != _lexical(
        root / _ORIGIN_REGISTRY_PROTOCOL
    ):
        raise WorkspaceLocalError("--protocol must be the canonical origin registry protocol")
    if (
        option_values["--expected-protocol-sha256"]
        != _ORIGIN_REGISTRY_PROTOCOL_SHA256
    ):
        raise WorkspaceLocalError(
            "--expected-protocol-sha256 must match the canonical origin registry protocol"
        )


def _redact_command(command: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    markers = tuple(marker.lower() for marker in _SENSITIVE_ENV_MARKERS)
    eex_datasource = _EEX_DATASOURCE_MODULE in command
    for argument in command:
        lowered = argument.lower()
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if any(marker in lowered for marker in markers):
            if "=" in argument:
                redacted.append(f"{argument.split('=', 1)[0]}=<redacted>")
            else:
                redacted.append(argument)
                redact_next = True
            continue
        if eex_datasource and (
            lowered == "--spec" or lowered.startswith("--spec=")
        ):
            if "=" in argument:
                redacted.append("--spec=<governed-path-redacted>")
            else:
                redacted.append(argument)
                redact_next = True
            continue
        redacted.append(argument)
    return redacted


def _command_sha256(command: list[str]) -> str:
    payload = json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path_identities(run_root: Path, paths: dict[str, Path]) -> dict[str, tuple[int, int]]:
    selected = {"run-root": run_root, **paths}
    return {
        name: (int(os.stat(path).st_dev), int(os.stat(path).st_ino))
        for name, path in selected.items()
    }


def _revalidate_paths(
    root: Path,
    run_root: Path,
    paths: dict[str, Path],
    expected: dict[str, tuple[int, int]],
) -> None:
    selected = {"run-root": run_root, **paths}
    for name, path in selected.items():
        _require_within(path, root, label=name)
        _reject_reparse_chain(path, root)
        observed = (int(os.stat(path).st_dev), int(os.stat(path).st_ino))
        if observed != expected[name]:
            raise WorkspaceLocalError(f"path identity changed after preflight: {name}")


def prepare_command(
    command: list[str],
    root: Path,
    pytest_basetemp: Path,
    pytest_junit_path: Path | None = None,
    pytest_native_result_path: Path | None = None,
) -> list[str]:
    selected = list(command)
    if selected and selected[0] == "--":
        selected = selected[1:]
    if not selected:
        raise WorkspaceLocalError("a command is required unless --preflight-only is used")
    _validate_executable(selected, root)
    _require_launcherless_build_paths(selected, root)
    _require_eex_vintage_build_paths(selected, root)
    _require_eex_datasource_v2_capture_boundary(selected, root)
    _require_local_panel_paths(selected, root)
    _require_local_model_boundary(selected, root)
    _require_laptop_model_qualification_boundary(selected, root)
    _require_local_quality_selection_boundary(selected)
    _require_structural_prediction_commitment_boundary(selected, root)
    _require_local_future_origin_selection_boundary(selected, root)
    _require_energy_charts_capture_boundary(selected, root)
    _require_eex_local_capture_boundary(selected, root)
    _require_eex_current_selection_boundary(selected, root)
    _require_preregistration_supersession_boundary(selected, root)
    _require_successor_candidate_core_boundary(selected, root)
    _require_dependence_power_design_boundary(selected, root)
    _require_origin_target_mask_inventory_boundary(selected, root)
    _require_origin_registry_protocol_boundary(selected, root)
    if not _is_pytest(selected):
        return selected
    if any(argument == "--basetemp" or argument.startswith("--basetemp=") for argument in selected):
        raise WorkspaceLocalError("caller-supplied pytest basetemp is forbidden")
    if any(
        argument in {"--junitxml", "--junit-xml"}
        or argument.startswith("--junitxml=")
        or argument.startswith("--junit-xml=")
        for argument in selected
    ):
        raise WorkspaceLocalError("caller-supplied pytest JUnit output is forbidden")
    if any(
        argument == "--pfc-workspace-result-json"
        or argument.startswith("--pfc-workspace-result-json=")
        for argument in selected
    ):
        raise WorkspaceLocalError("caller-supplied pytest native result output is forbidden")
    forbidden_config_options = {
        "-c",
        "-o",
        "--config-file",
        "--inifile",
        "--override-ini",
        "--rootdir",
        "--confcutdir",
    }
    for argument in selected:
        if argument.startswith("@"):
            raise WorkspaceLocalError("pytest argument-file expansion is forbidden")
        if argument.startswith("-") and not argument.startswith("--"):
            short_option_bundle = argument[1:]
            if not argument.startswith("-p"):
                forbidden_positions = [
                    (short_option_bundle.index(flag), flag)
                    for flag in "cop"
                    if flag in short_option_bundle
                ]
                if forbidden_positions:
                    _, first_forbidden = min(forbidden_positions)
                    if first_forbidden == "p":
                        raise WorkspaceLocalError(
                            "explicit pytest plugin is not allowlisted"
                        )
                    raise WorkspaceLocalError(
                        "caller-supplied pytest configuration override is forbidden"
                    )
        option = argument.split("=", 1)[0]
        if option in forbidden_config_options or (
            argument.startswith("-o") and argument != "-o"
        ) or (
            argument.startswith("-c") and argument != "-c"
        ):
            raise WorkspaceLocalError(
                "caller-supplied pytest configuration override is forbidden"
            )
    index = 0
    while index < len(selected):
        argument = selected[index]
        if argument == "-p":
            if index + 1 >= len(selected):
                raise WorkspaceLocalError("pytest -p is missing its plugin name")
            plugin = selected[index + 1]
            if plugin != "no:cacheprovider":
                raise WorkspaceLocalError("explicit pytest plugin is not allowlisted")
            index += 1
        elif argument.startswith("-p") and argument != "-p":
            if argument[2:] != "no:cacheprovider":
                raise WorkspaceLocalError("explicit pytest plugin is not allowlisted")
        index += 1
    _require_within(pytest_basetemp, root, label="pytest basetemp")
    junit_path = pytest_junit_path or pytest_basetemp.parent / "pytest-results.xml"
    native_result_path = (
        pytest_native_result_path or pytest_basetemp.parent / "pytest-native-result.json"
    )
    _require_within(junit_path, root, label="pytest JUnit output")
    _require_within(native_result_path, root, label="pytest native result output")
    return [
        *selected,
        "--basetemp",
        str(pytest_basetemp),
        "--junitxml",
        str(junit_path),
        "-p",
        "scripts.run_workspace_local",
        "--pfc-workspace-result-json",
        str(native_result_path),
    ]


def _pump_bounded_stream(
    source: BinaryIO,
    destination: Path,
    label: str,
    results: dict[str, dict[str, object]],
    errors: list[BaseException],
) -> None:
    captured_hash = hashlib.sha256()
    full_hash = hashlib.sha256()
    captured_bytes = 0
    total_bytes = 0
    try:
        with destination.open("xb") as handle:
            while True:
                chunk = source.read(_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                total_bytes += len(chunk)
                full_hash.update(chunk)
                remaining = _OUTPUT_CAPTURE_LIMIT_BYTES - captured_bytes
                if remaining > 0:
                    retained = chunk[:remaining]
                    handle.write(retained)
                    captured_hash.update(retained)
                    captured_bytes += len(retained)
            handle.flush()
            os.fsync(handle.fileno())
        results[label] = {
            "path": str(destination),
            "captured_bytes": captured_bytes,
            "total_bytes": total_bytes,
            "sha256": captured_hash.hexdigest(),
            "full_stream_sha256": full_hash.hexdigest(),
            "truncated": total_bytes > captured_bytes,
        }
    except BaseException as exc:  # pragma: no cover - surfaced on the controller thread
        errors.append(exc)
    finally:
        source.close()


def _replay_capture(path: Path, stream: object) -> None:
    binary_stream = getattr(stream, "buffer", None)
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_STREAM_CHUNK_BYTES)
            if not chunk:
                break
            if binary_stream is not None:
                binary_stream.write(chunk)
                binary_stream.flush()
            else:
                stream.write(chunk.decode("utf-8", errors="replace"))  # type: ignore[attr-defined]
                stream.flush()  # type: ignore[attr-defined]


def _close_windows_handle(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    if not kernel32.CloseHandle(wintypes.HANDLE(handle)):
        raise OSError(ctypes.get_last_error(), "CloseHandle failed")


def _create_windows_kill_on_close_job() -> int:
    import ctypes
    from ctypes import wintypes

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("read_operation_count", ctypes.c_ulonglong),
            ("write_operation_count", ctypes.c_ulonglong),
            ("other_operation_count", ctypes.c_ulonglong),
            ("read_transfer_count", ctypes.c_ulonglong),
            ("write_transfer_count", ctypes.c_ulonglong),
            ("other_transfer_count", ctypes.c_ulonglong),
        ]

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("per_process_user_time_limit", ctypes.c_longlong),
            ("per_job_user_time_limit", ctypes.c_longlong),
            ("limit_flags", wintypes.DWORD),
            ("minimum_working_set_size", ctypes.c_size_t),
            ("maximum_working_set_size", ctypes.c_size_t),
            ("active_process_limit", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority_class", wintypes.DWORD),
            ("scheduling_class", wintypes.DWORD),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("basic_limit_information", BasicLimitInformation),
            ("io_info", IoCounters),
            ("process_memory_limit", ctypes.c_size_t),
            ("job_memory_limit", ctypes.c_size_t),
            ("peak_process_memory_used", ctypes.c_size_t),
            ("peak_job_memory_used", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
    information = ExtendedLimitInformation()
    information.basic_limit_information.limit_flags = 0x00002000
    if not kernel32.SetInformationJobObject(
        handle,
        9,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        error = ctypes.get_last_error()
        _close_windows_handle(int(handle))
        raise OSError(error, "SetInformationJobObject failed")
    return int(handle)


def _assign_process_to_windows_job(
    job_handle: int, process: subprocess.Popen[bytes]
) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    process_handle = getattr(process, "_handle", None)
    if process_handle is None or not kernel32.AssignProcessToJobObject(
        wintypes.HANDLE(job_handle), wintypes.HANDLE(int(process_handle))
    ):
        raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")


def _resume_windows_process(process: subprocess.Popen[bytes]) -> None:
    import ctypes
    from ctypes import wintypes

    class ThreadEntry32(ctypes.Structure):
        _fields_ = [
            ("size", wintypes.DWORD),
            ("usage_count", wintypes.DWORD),
            ("thread_id", wintypes.DWORD),
            ("owner_process_id", wintypes.DWORD),
            ("base_priority", wintypes.LONG),
            ("priority_delta", wintypes.LONG),
            ("flags", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    kernel32.Thread32First.restype = wintypes.BOOL
    kernel32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    kernel32.Thread32Next.restype = wintypes.BOOL
    kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenThread.restype = wintypes.HANDLE
    kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel32.ResumeThread.restype = wintypes.DWORD
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000004, 0)
    if snapshot == wintypes.HANDLE(-1).value:
        raise OSError(ctypes.get_last_error(), "target thread snapshot failed")
    thread_ids: list[int] = []
    try:
        entry = ThreadEntry32()
        entry.size = ctypes.sizeof(entry)
        available = kernel32.Thread32First(snapshot, ctypes.byref(entry))
        while available:
            if int(entry.owner_process_id) == process.pid:
                thread_ids.append(int(entry.thread_id))
            available = kernel32.Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        _close_windows_handle(int(snapshot))
    if len(thread_ids) != 1:
        raise OSError("suspended target primary thread is ambiguous")
    thread_handle = kernel32.OpenThread(0x0002, False, thread_ids[0])
    if not thread_handle:
        raise OSError(ctypes.get_last_error(), "target primary thread open failed")
    try:
        previous_suspend_count = kernel32.ResumeThread(thread_handle)
    finally:
        _close_windows_handle(int(thread_handle))
    if previous_suspend_count != 1:
        raise OSError(
            ctypes.get_last_error(),
            "target did not resume from the expected suspended state",
        )


def _windows_job_resource_usage(job_handle: int) -> dict[str, object]:
    import ctypes
    from ctypes import wintypes

    class BasicAccountingInformation(ctypes.Structure):
        _fields_ = [
            ("total_user_time", ctypes.c_longlong),
            ("total_kernel_time", ctypes.c_longlong),
            ("this_period_total_user_time", ctypes.c_longlong),
            ("this_period_total_kernel_time", ctypes.c_longlong),
            ("total_page_fault_count", wintypes.DWORD),
            ("total_processes", wintypes.DWORD),
            ("active_processes", wintypes.DWORD),
            ("total_terminated_processes", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("read_operation_count", ctypes.c_ulonglong),
            ("write_operation_count", ctypes.c_ulonglong),
            ("other_operation_count", ctypes.c_ulonglong),
            ("read_transfer_count", ctypes.c_ulonglong),
            ("write_transfer_count", ctypes.c_ulonglong),
            ("other_transfer_count", ctypes.c_ulonglong),
        ]

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("per_process_user_time_limit", ctypes.c_longlong),
            ("per_job_user_time_limit", ctypes.c_longlong),
            ("limit_flags", wintypes.DWORD),
            ("minimum_working_set_size", ctypes.c_size_t),
            ("maximum_working_set_size", ctypes.c_size_t),
            ("active_process_limit", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority_class", wintypes.DWORD),
            ("scheduling_class", wintypes.DWORD),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("basic_limit_information", BasicLimitInformation),
            ("io_info", IoCounters),
            ("process_memory_limit", ctypes.c_size_t),
            ("job_memory_limit", ctypes.c_size_t),
            ("peak_process_memory_used", ctypes.c_size_t),
            ("peak_job_memory_used", ctypes.c_size_t),
        ]

    class BasicAndIoAccountingInformation(ctypes.Structure):
        _fields_ = [
            ("basic_info", BasicAccountingInformation),
            ("io_info", IoCounters),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    accounting_and_io = BasicAndIoAccountingInformation()
    extended = ExtendedLimitInformation()
    for information_class, target in ((8, accounting_and_io), (9, extended)):
        if not kernel32.QueryInformationJobObject(
            wintypes.HANDLE(job_handle),
            information_class,
            ctypes.byref(target),
            ctypes.sizeof(target),
            None,
        ):
            raise OSError(
                ctypes.get_last_error(),
                f"QueryInformationJobObject({information_class}) failed",
            )
    accounting = accounting_and_io.basic_info
    io_info = accounting_and_io.io_info
    return {
        "platform": "windows_job_object",
        "user_cpu_seconds": int(accounting.total_user_time) / 10_000_000.0,
        "kernel_cpu_seconds": int(accounting.total_kernel_time) / 10_000_000.0,
        "total_page_fault_count": int(accounting.total_page_fault_count),
        "total_processes": int(accounting.total_processes),
        "active_processes": int(accounting.active_processes),
        "total_terminated_processes": int(accounting.total_terminated_processes),
        "peak_process_memory_bytes": int(extended.peak_process_memory_used),
        "peak_job_memory_bytes": int(extended.peak_job_memory_used),
        "read_operation_count": int(io_info.read_operation_count),
        "write_operation_count": int(io_info.write_operation_count),
        "other_operation_count": int(io_info.other_operation_count),
        "read_transfer_bytes": int(io_info.read_transfer_count),
        "write_transfer_bytes": int(io_info.write_transfer_count),
        "other_transfer_bytes": int(io_info.other_transfer_count),
    }


def _terminate_process_tree(
    process: subprocess.Popen[bytes], *, windows_job: int | None
) -> None:
    if os.name == "nt":
        if windows_job is None:
            raise WorkspaceLocalError("Windows target has no Job Object containment")
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        if not kernel32.TerminateJobObject(
            wintypes.HANDLE(windows_job),
            _WINDOWS_JOB_OBJECT_TERMINATED_EXIT_CODE,
        ):
            error = ctypes.get_last_error()
            if process.poll() is None:
                raise OSError(error, "TerminateJobObject failed")
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        try:
            process.wait(timeout=_PROCESS_TREE_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=_PROCESS_TREE_TERMINATION_GRACE_SECONDS)
    if os.name == "nt" and windows_job is not None:
        deadline = time.monotonic() + _PROCESS_TREE_TERMINATION_GRACE_SECONDS
        while True:
            active = int(_windows_job_resource_usage(windows_job)["active_processes"])
            if active == 0:
                break
            if time.monotonic() >= deadline:
                raise WorkspaceLocalError(
                    "Windows Job Object still contains active processes after termination"
                )
            time.sleep(0.01)


def _posix_process_group_active(process_group_id: int) -> bool:
    if os.name == "nt":
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise WorkspaceLocalError("POSIX process-group state is unobservable") from exc
    return True


def _posix_child_usage_snapshot() -> tuple[float, float, int, int] | None:
    if os.name == "nt":
        return None
    import resource

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (
        float(usage.ru_utime),
        float(usage.ru_stime),
        int(usage.ru_maxrss),
        int(usage.ru_majflt + usage.ru_minflt),
    )


def _posix_child_usage_delta(
    before: tuple[float, float, int, int] | None,
) -> dict[str, object]:
    after = _posix_child_usage_snapshot()
    if before is None or after is None:
        raise WorkspaceLocalError("POSIX child resource accounting is unavailable")
    return {
        "platform": "posix_rusage_children",
        "user_cpu_seconds": max(0.0, after[0] - before[0]),
        "kernel_cpu_seconds": max(0.0, after[1] - before[1]),
        "max_rss_platform_units": max(0, after[2]),
        "page_fault_count_delta": max(0, after[3] - before[3]),
    }


def _execute_command_with_capture(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    run_root: Path,
    wall_timeout_seconds: float,
    on_started: Callable[[dict[str, object]], None],
) -> CapturedExecution:
    stdout_path = run_root / "target-stdout.log"
    stderr_path = run_root / "target-stderr.log"
    if not math.isfinite(wall_timeout_seconds) or wall_timeout_seconds <= 0:
        raise WorkspaceLocalError("global wall timeout must be finite and positive")
    started_at = time.monotonic()
    posix_usage_before = _posix_child_usage_snapshot()
    windows_job: int | None = None
    process: subprocess.Popen[bytes] | None = None
    results: dict[str, dict[str, object]] = {}
    errors: list[BaseException] = []
    threads: list[threading.Thread] = []
    timed_out = False
    interrupted = False
    descendant_cleanup_triggered = False
    resource_usage: dict[str, object] = {}
    returncode: int | None = None
    try:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        if os.name == "nt":
            windows_job = _create_windows_kill_on_close_job()
            creationflags |= _WINDOWS_CREATE_SUSPENDED
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
        if process.stdout is None or process.stderr is None:  # pragma: no cover
            _terminate_process_tree(process, windows_job=windows_job)
            raise WorkspaceLocalError("target output pipes were not created")
        threads = [
            threading.Thread(
                target=_pump_bounded_stream,
                args=(process.stdout, stdout_path, "stdout", results, errors),
                daemon=True,
            ),
            threading.Thread(
                target=_pump_bounded_stream,
                args=(process.stderr, stderr_path, "stderr", results, errors),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        if os.name == "nt":
            if windows_job is None:  # pragma: no cover - established above
                raise WorkspaceLocalError("Windows Job Object was not created")
            _assign_process_to_windows_job(windows_job, process)
        target_identity = _new_process_identity(process.pid, label="target")
        containment_mode = (
            "CREATE_SUSPENDED_ASSIGN_KILL_ON_JOB_CLOSE_THEN_RESUME"
            if os.name == "nt"
            else "START_NEW_SESSION_PROCESS_GROUP"
        )
        on_started(
            {
                "target_process": target_identity,
                "process_tree": {
                    "mode": containment_mode,
                    "kill_on_supervisor_exit": True,
                    "global_wall_timeout_seconds": wall_timeout_seconds,
                },
            }
        )
        if os.name == "nt":
            _resume_windows_process(process)
        remaining_timeout = wall_timeout_seconds - (time.monotonic() - started_at)
        try:
            if remaining_timeout <= 0:
                raise subprocess.TimeoutExpired(command, wall_timeout_seconds)
            returncode = process.wait(timeout=remaining_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_tree(process, windows_job=windows_job)
            returncode = process.returncode
        except KeyboardInterrupt:
            interrupted = True
            _terminate_process_tree(process, windows_job=windows_job)
            returncode = process.returncode
        for thread in threads:
            thread.join(timeout=_PIPE_DRAIN_GRACE_SECONDS)
        if any(thread.is_alive() for thread in threads):
            descendant_cleanup_triggered = True
            _terminate_process_tree(process, windows_job=windows_job)
            for thread in threads:
                thread.join(timeout=_PIPE_DRAIN_GRACE_SECONDS)
        if any(thread.is_alive() for thread in threads):
            raise WorkspaceLocalError(
                "target output pipes remained open after process-tree termination"
            )
        if windows_job is not None:
            usage_before_cleanup = _windows_job_resource_usage(windows_job)
            if int(usage_before_cleanup["active_processes"]) > 0:
                descendant_cleanup_triggered = True
                _terminate_process_tree(process, windows_job=windows_job)
            resource_usage = _windows_job_resource_usage(windows_job)
        else:
            if _posix_process_group_active(process.pid):
                descendant_cleanup_triggered = True
                _terminate_process_tree(process, windows_job=None)
            resource_usage = _posix_child_usage_delta(posix_usage_before)
    finally:
        if process is not None and process.poll() is None:
            try:
                _terminate_process_tree(process, windows_job=windows_job)
            except (OSError, subprocess.SubprocessError, WorkspaceLocalError):
                pass
        if windows_job is not None:
            _close_windows_handle(windows_job)
        for thread in threads:
            thread.join(timeout=_PIPE_DRAIN_GRACE_SECONDS)
    if errors:
        raise WorkspaceLocalError(
            f"target output capture failed: {type(errors[0]).__name__}: {errors[0]}"
        )
    if set(results) != {"stdout", "stderr"}:
        raise WorkspaceLocalError("target output capture is incomplete")
    _replay_capture(stdout_path, sys.stdout)
    _replay_capture(stderr_path, sys.stderr)
    if returncode is None:
        raise WorkspaceLocalError("target return code is unavailable")
    resource_usage["wall_duration_seconds"] = time.monotonic() - started_at
    return CapturedExecution(
        returncode=int(returncode),
        streams=results,
        supervision={
            "global_wall_timeout_seconds": wall_timeout_seconds,
            "timed_out": timed_out,
            "interrupted": interrupted,
            "descendant_cleanup_triggered": descendant_cleanup_triggered,
            "tree_termination_confirmed": True,
        },
        resource_usage=resource_usage,
        timed_out=timed_out,
        interrupted=interrupted,
    )


def _stable_bounded_regular_file_bytes(
    path: Path,
    *,
    root: Path,
    limit_bytes: int,
    label: str,
) -> bytes:
    _require_within(path, root, label=label)
    if not path.is_file():
        raise WorkspaceLocalError(f"{label} is missing")
    _reject_reparse_chain(path, root)
    path_before = os.stat(path)
    with path.open("rb") as handle:
        handle_before = os.fstat(handle.fileno())
        path_before_identity = (
            int(path_before.st_dev),
            int(path_before.st_ino),
            int(path_before.st_size),
        )
        handle_before_identity = (
            int(handle_before.st_dev),
            int(handle_before.st_ino),
            int(handle_before.st_size),
        )
        if path_before_identity != handle_before_identity:
            raise WorkspaceLocalError(f"{label} changed before descriptor capture")
        if int(handle_before.st_nlink) != 1:
            raise WorkspaceLocalError(f"{label} must be mono-linked")
        if int(handle_before.st_size) > limit_bytes:
            raise WorkspaceLocalError(f"{label} exceeds its byte-size limit")
        payload = handle.read(limit_bytes + 1)
        handle_after = os.fstat(handle.fileno())
    path_after = os.stat(path)
    handle_after_identity = (
        int(handle_after.st_dev),
        int(handle_after.st_ino),
        int(handle_after.st_size),
    )
    path_after_identity = (
        int(path_after.st_dev),
        int(path_after.st_ino),
        int(path_after.st_size),
    )
    if handle_before_identity != handle_after_identity:
        raise WorkspaceLocalError(f"{label} changed during capture")
    if handle_after_identity != path_after_identity:
        raise WorkspaceLocalError(f"{label} path changed during capture")
    if len(payload) > limit_bytes or len(payload) != int(handle_after.st_size):
        raise WorkspaceLocalError(f"{label} size is inconsistent")
    return payload


def _stable_regular_file_evidence(
    path: Path,
    *,
    limit_bytes: int,
    label: str,
) -> dict[str, object]:
    selected = path.resolve(strict=True)
    attributes = getattr(os.lstat(selected), "st_file_attributes", 0)
    if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
        raise WorkspaceLocalError(f"{label} must not be a reparse point")
    path_before = os.stat(selected)
    digest = hashlib.sha256()
    observed_size = 0
    with selected.open("rb") as handle:
        handle_before = os.fstat(handle.fileno())
        before_identity = (
            int(path_before.st_dev),
            int(path_before.st_ino),
            int(path_before.st_size),
        )
        handle_identity = (
            int(handle_before.st_dev),
            int(handle_before.st_ino),
            int(handle_before.st_size),
        )
        if before_identity != handle_identity:
            raise WorkspaceLocalError(f"{label} changed before descriptor capture")
        if int(handle_before.st_size) > limit_bytes:
            raise WorkspaceLocalError(f"{label} exceeds its byte-size limit")
        while True:
            chunk = handle.read(_STREAM_CHUNK_BYTES)
            if not chunk:
                break
            observed_size += len(chunk)
            if observed_size > limit_bytes:
                raise WorkspaceLocalError(f"{label} exceeds its byte-size limit")
            digest.update(chunk)
        handle_after = os.fstat(handle.fileno())
    path_after = os.stat(selected)
    after_identity = (
        int(handle_after.st_dev),
        int(handle_after.st_ino),
        int(handle_after.st_size),
    )
    path_after_identity = (
        int(path_after.st_dev),
        int(path_after.st_ino),
        int(path_after.st_size),
    )
    if handle_identity != after_identity or after_identity != path_after_identity:
        raise WorkspaceLocalError(f"{label} changed during capture")
    if observed_size != int(handle_after.st_size):
        raise WorkspaceLocalError(f"{label} size is inconsistent")
    return {
        "path": str(selected),
        "size_bytes": observed_size,
        "sha256": digest.hexdigest(),
        "device": int(handle_after.st_dev),
        "inode": int(handle_after.st_ino),
        "link_count": int(handle_after.st_nlink),
    }


def _execution_identity(root: Path, command: list[str]) -> dict[str, object]:
    runner_source = Path(__file__).resolve(strict=True)
    if not _is_within(runner_source, root):
        raise WorkspaceLocalError("runner source escaped the canonical workspace")
    interpreter = Path(command[0]).resolve(strict=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", head) is None or not branch:
        raise WorkspaceLocalError("Git execution identity is invalid")
    source_tree = _source_tree_identity(root)
    target_interpreter = _stable_regular_file_evidence(
        interpreter,
        limit_bytes=128 * 1024 * 1024,
        label="target interpreter",
    )
    return {
        "runner_source": _stable_regular_file_evidence(
            runner_source,
            limit_bytes=4 * 1024 * 1024,
            label="runner source",
        ),
        "target_interpreter": target_interpreter,
        "target_import_closure": _target_import_closure_identity(
            root,
            interpreter,
            command,
            source_tree=source_tree,
        ),
        "source_tree": source_tree,
        "git_head": head,
        "git_branch": branch,
    }


def _require_eex_datasource_execution_identity(
    command: list[str], identity: dict[str, object], root: Path
) -> None:
    if _python_module(command) != _EEX_DATASOURCE_MODULE:
        return
    closure = identity.get("target_import_closure")
    if not isinstance(closure, dict) or closure.get("status") != "BOUND_REPO_LOCAL_PTH":
        raise WorkspaceLocalError(
            "EEX DataSource v2 runtime import closure is not repo-local and bound"
        )
    sys_path = closure.get("sys_path")
    if not isinstance(sys_path, list) or not sys_path:
        raise WorkspaceLocalError("EEX DataSource v2 runtime sys.path is unavailable")
    for value in sys_path:
        if not isinstance(value, str) or not _is_within(Path(value), root):
            raise WorkspaceLocalError(
                "EEX DataSource v2 runtime sys.path escaped the workspace"
            )
    origins = closure.get("module_origins")
    if not isinstance(origins, list):
        raise WorkspaceLocalError("EEX DataSource v2 module origins are unavailable")
    matches = [
        item
        for item in origins
        if isinstance(item, dict) and item.get("module") == _EEX_DATASOURCE_MODULE
    ]
    if len(matches) != 1:
        raise WorkspaceLocalError(
            "EEX DataSource v2 module origin is not uniquely bound"
        )
    origin_path = matches[0].get("path")
    if not isinstance(origin_path, str) or not _is_within(Path(origin_path), root):
        raise WorkspaceLocalError("EEX DataSource v2 module origin escaped the workspace")
    if not origin_path.replace("\\", "/").endswith(
        "/pfc_shaping/data/eex_datasource_v2_capture.py"
    ):
        raise WorkspaceLocalError("EEX DataSource v2 module origin is unexpected")


def _require_local_future_origin_execution_identity(
    command: list[str], identity: dict[str, object], root: Path
) -> None:
    if _python_module(command) != _LOCAL_FUTURE_ORIGIN_PACKAGED_MODULE:
        return
    closure = identity.get("target_import_closure")
    if not isinstance(closure, dict) or closure.get("status") != "BOUND_REPO_LOCAL_PTH":
        raise WorkspaceLocalError(
            "packaged local future origin runtime closure is not repo-local and bound"
        )
    sys_path = closure.get("sys_path")
    if not isinstance(sys_path, list) or not sys_path:
        raise WorkspaceLocalError("packaged local future origin sys.path is unavailable")
    normalized_root = _lexical(root)
    for value in sys_path:
        if not isinstance(value, str) or not _is_within(Path(value), root):
            raise WorkspaceLocalError(
                "packaged local future origin sys.path escaped the workspace"
            )
        if _lexical(Path(value)) == normalized_root:
            raise WorkspaceLocalError(
                "packaged local future origin sys.path contains the source checkout"
            )
    target = identity.get("target_interpreter")
    if not isinstance(target, dict) or not isinstance(target.get("path"), str):
        raise WorkspaceLocalError(
            "packaged local future origin target interpreter is unavailable"
        )
    runtime_root = Path(str(target["path"])).parent.resolve(strict=True)
    origins = closure.get("module_origins")
    if not isinstance(origins, list):
        raise WorkspaceLocalError(
            "packaged local future origin module origins are unavailable"
        )
    matches = [
        item
        for item in origins
        if isinstance(item, dict)
        and item.get("module") == _LOCAL_FUTURE_ORIGIN_PACKAGED_MODULE
    ]
    if len(matches) != 1:
        raise WorkspaceLocalError(
            "packaged local future origin module origin is not uniquely bound"
        )
    origin_value = matches[0].get("path")
    if not isinstance(origin_value, str):
        raise WorkspaceLocalError(
            "packaged local future origin module origin is unavailable"
        )
    origin = Path(origin_value).resolve(strict=True)
    if not _is_within(origin, runtime_root) or not origin.as_posix().endswith(
        "/pfc_shaping/validation/ch_lt_local_future_origin_selection.py"
    ):
        raise WorkspaceLocalError(
            "packaged local future origin module origin escaped the selected runtime"
        )
    package_parent = origin.parents[2]
    if sum(_lexical(Path(value)) == _lexical(package_parent) for value in sys_path) != 1:
        raise WorkspaceLocalError(
            "packaged local future origin package root must occur exactly once in sys.path"
        )


def _directory_tree_identity(
    directory: Path,
    *,
    root: Path,
    label: str,
) -> dict[str, object]:
    selected = directory.resolve(strict=True)
    if not selected.is_dir() or not _is_within(selected, root):
        raise WorkspaceLocalError(f"{label} must be a directory below the workspace")
    _reject_reparse_chain(selected, root)
    entries: list[dict[str, object]] = []
    total_bytes = 0
    normalized_paths: set[str] = set()
    discovered = sorted(
        selected.rglob("*"),
        key=lambda item: item.relative_to(selected).as_posix().lower(),
    )
    for path in discovered:
        _reject_reparse_chain(path, root)
        if not path.is_file():
            continue
        relative = path.relative_to(selected).as_posix()
        normalized = relative.lower()
        if normalized in normalized_paths:
            raise WorkspaceLocalError(f"{label} path identity is ambiguous")
        normalized_paths.add(normalized)
        if len(entries) >= _IMPORT_CLOSURE_FILE_COUNT_LIMIT:
            raise WorkspaceLocalError(f"{label} exceeds its file-count limit")
        evidence = _stable_regular_file_evidence(
            path,
            limit_bytes=_IMPORT_CLOSURE_FILE_LIMIT_BYTES,
            label=f"{label} file {relative}",
        )
        if int(evidence["link_count"]) != 1:
            raise WorkspaceLocalError(f"{label} file must be mono-linked: {relative}")
        total_bytes += int(evidence["size_bytes"])
        if total_bytes > _IMPORT_CLOSURE_TREE_LIMIT_BYTES:
            raise WorkspaceLocalError(f"{label} exceeds its byte-size limit")
        entries.append(
            {
                "path": relative,
                "size_bytes": evidence["size_bytes"],
                "sha256": evidence["sha256"],
            }
        )
    if not entries:
        raise WorkspaceLocalError(f"{label} evidence is empty")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "schema_version": "pfc_lt_directory_content_tree.v1",
        "path": str(selected),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _module_origin_evidence(
    module: str,
    *,
    sys_path_roots: list[Path],
    root: Path,
) -> dict[str, object] | None:
    parts = module.split(".")
    if not parts or any(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) is None for part in parts):
        raise WorkspaceLocalError(f"invalid target module for origin binding: {module}")
    relative = Path(*parts)
    for sys_path_root in sys_path_roots:
        candidates = (
            sys_path_root / relative / "__init__.py",
            (sys_path_root / relative).with_suffix(".py"),
        )
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if not _is_within(candidate, root):
                raise WorkspaceLocalError("module origin escaped the canonical workspace")
            _reject_reparse_chain(candidate, root)
            evidence = _stable_regular_file_evidence(
                candidate,
                limit_bytes=_SOURCE_FILE_LIMIT_BYTES,
                label=f"module origin {module}",
            )
            return {"module": module, **evidence}
    return None


def _target_import_closure_identity(
    root: Path,
    interpreter: Path,
    command: list[str],
    *,
    source_tree: dict[str, object],
) -> dict[str, object]:
    if not _is_within(interpreter, root):
        return {
            "schema_version": "pfc_lt_target_import_closure.v1",
            "status": "UNBOUND_EXTERNAL_INTERPRETER",
        }
    launcher_root = interpreter.parent.resolve(strict=True)
    _reject_reparse_chain(launcher_root, root)
    pth_candidates = sorted(launcher_root.glob("python*._pth"))
    if len(pth_candidates) != 1:
        raise WorkspaceLocalError(
            "repo-local target interpreter requires exactly one adjacent python*._pth"
        )
    pth_path = pth_candidates[0]
    pth_payload = _stable_bounded_regular_file_bytes(
        pth_path,
        root=root,
        limit_bytes=_PYTHON_PTH_LIMIT_BYTES,
        label="target Python _pth",
    )
    pth_evidence = _stable_regular_file_evidence(
        pth_path,
        limit_bytes=_PYTHON_PTH_LIMIT_BYTES,
        label="target Python _pth",
    )
    try:
        pth_text = pth_payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise WorkspaceLocalError("target Python _pth must be UTF-8") from exc
    path_lines: list[str] = []
    for raw_line in pth_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "import site" or line.startswith("import "):
            raise WorkspaceLocalError("target Python _pth may not execute import directives")
        path_lines.append(line)
    if not path_lines:
        raise WorkspaceLocalError("target Python _pth contains no import roots")
    sys_path_roots: list[Path] = []
    normalized_roots: set[str] = set()
    for line in path_lines:
        candidate = Path(line)
        selected = (candidate if candidate.is_absolute() else launcher_root / candidate).resolve(
            strict=True
        )
        if not selected.is_dir() or not _is_within(selected, root):
            raise WorkspaceLocalError("target Python _pth root escaped the workspace")
        _reject_reparse_chain(selected, root)
        normalized = _lexical(selected)
        if normalized in normalized_roots:
            raise WorkspaceLocalError("target Python _pth contains duplicate roots")
        normalized_roots.add(normalized)
        sys_path_roots.append(selected)
    import_roots: list[dict[str, object]] = []
    for position, selected in enumerate(sys_path_roots):
        if _lexical(selected) == _lexical(root):
            identity: dict[str, object] = {
                "schema_version": source_tree["schema_version"],
                "file_count": source_tree["file_count"],
                "total_bytes": source_tree["total_bytes"],
                "tree_sha256": source_tree["tree_sha256"],
            }
        else:
            identity = _directory_tree_identity(
                selected,
                root=root,
                label=f"target import root {position}",
            )
        import_roots.append({"position": position, **identity})
    modules = {"pytest", "pluggy", "requests", "dotenv", "pfc_shaping"}
    if "-m" in command:
        module_index = command.index("-m") + 1
        if module_index >= len(command):
            raise WorkspaceLocalError("target module is missing")
        modules.add(command[module_index])
    module_origins = []
    for module in sorted(modules):
        evidence = _module_origin_evidence(
            module,
            sys_path_roots=sys_path_roots,
            root=root,
        )
        if evidence is not None:
            module_origins.append(evidence)
    return {
        "schema_version": "pfc_lt_target_import_closure.v1",
        "status": "BOUND_REPO_LOCAL_PTH",
        "python_pth": pth_evidence,
        "sys_path": [str(path) for path in sys_path_roots],
        "launcher_tree": _directory_tree_identity(
            launcher_root,
            root=root,
            label="target launcher tree",
        ),
        "import_roots": import_roots,
        "module_origins": module_origins,
    }


def _source_tree_identity(root: Path) -> dict[str, object]:
    selected: set[Path] = set()
    for directory_name in ("pfc_shaping", "scripts", "tests"):
        directory = root / directory_name
        if directory.is_dir():
            selected.update(path for path in directory.rglob("*.py") if path.is_file())
    for filename in (
        "AGENTS.md",
        "PACKAGE.md",
        "pyproject.toml",
        "pytest.ini",
        "setup.cfg",
        "tox.ini",
        "uv.lock",
    ):
        path = root / filename
        if path.is_file():
            selected.add(path)
    entries: list[dict[str, object]] = []
    total_bytes = 0
    normalized_paths: set[str] = set()
    for path in sorted(selected, key=lambda item: item.relative_to(root).as_posix().lower()):
        if not _is_within(path, root):
            raise WorkspaceLocalError("source-tree path escaped the canonical workspace")
        _reject_reparse_chain(path, root)
        relative = path.relative_to(root).as_posix()
        normalized = relative.lower()
        if normalized in normalized_paths:
            raise WorkspaceLocalError("source-tree path identity is ambiguous")
        normalized_paths.add(normalized)
        evidence = _stable_regular_file_evidence(
            path,
            limit_bytes=_SOURCE_FILE_LIMIT_BYTES,
            label=f"source-tree file {relative}",
        )
        total_bytes += int(evidence["size_bytes"])
        if total_bytes > _SOURCE_TREE_LIMIT_BYTES:
            raise WorkspaceLocalError("source-tree evidence exceeds its byte-size limit")
        entries.append(
            {
                "path": relative,
                "size_bytes": evidence["size_bytes"],
                "sha256": evidence["sha256"],
                "device": evidence["device"],
                "inode": evidence["inode"],
                "link_count": evidence["link_count"],
            }
        )
    if not entries:
        raise WorkspaceLocalError("source-tree evidence is empty")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": "pfc_lt_workspace_source_tree.v1",
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _strict_nonnegative_xml_int(value: object, *, label: str) -> int:
    if not isinstance(value, str) or re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise WorkspaceLocalError(f"{label} must be a canonical non-negative integer")
    if len(value) > 8:
        raise WorkspaceLocalError(f"{label} exceeds the supported bound")
    parsed = int(value)
    if parsed > _MAX_PYTEST_TEST_COUNT:
        raise WorkspaceLocalError(f"{label} exceeds the supported bound")
    return parsed


def _pytest_junit_evidence(path: Path, *, root: Path) -> dict[str, object]:
    payload = _stable_bounded_regular_file_bytes(
        path,
        root=root,
        limit_bytes=_PYTEST_JUNIT_LIMIT_BYTES,
        label="pytest JUnit output",
    )
    try:
        xml_root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise WorkspaceLocalError("pytest JUnit output is invalid XML") from exc
    root_tag = xml_root.tag.rsplit("}", 1)[-1]
    if root_tag == "testsuite":
        suites = [xml_root]
    elif root_tag == "testsuites":
        suites = [child for child in xml_root if child.tag.rsplit("}", 1)[-1] == "testsuite"]
    else:
        raise WorkspaceLocalError("pytest JUnit output has an unsupported root")
    if len(suites) != 1:
        raise WorkspaceLocalError("pytest JUnit output must contain exactly one test suite")
    suite = suites[0]
    counts = {
        key: _strict_nonnegative_xml_int(suite.attrib.get(key), label=f"pytest {key}")
        for key in ("tests", "failures", "errors", "skipped")
    }
    passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    if passed < 0:
        raise WorkspaceLocalError("pytest JUnit counts are inconsistent")
    duration_text = suite.attrib.get("time")
    try:
        duration_seconds = float(duration_text) if duration_text is not None else math.nan
    except ValueError as exc:
        raise WorkspaceLocalError("pytest JUnit duration is invalid") from exc
    if not math.isfinite(duration_seconds) or duration_seconds < 0.0:
        raise WorkspaceLocalError("pytest JUnit duration must be finite and non-negative")
    return {
        "applicable": True,
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "tests": counts["tests"],
        "passed": passed,
        "failures": counts["failures"],
        "errors": counts["errors"],
        "skipped": counts["skipped"],
        "duration_seconds": duration_seconds,
    }


def _reject_duplicate_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise WorkspaceLocalError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _strict_nonnegative_json_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkspaceLocalError(f"{label} must be a strict integer")
    if value < 0 or value > _MAX_PYTEST_TEST_COUNT:
        raise WorkspaceLocalError(f"{label} exceeds the supported bound")
    return value


def _pytest_native_evidence(path: Path, *, root: Path) -> dict[str, object]:
    payload = _stable_bounded_regular_file_bytes(
        path,
        root=root,
        limit_bytes=_PYTEST_NATIVE_RESULT_LIMIT_BYTES,
        label="pytest native result",
    )
    try:
        decoded = json.loads(payload, object_pairs_hook=_reject_duplicate_json_pairs)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise WorkspaceLocalError("pytest native result is invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise WorkspaceLocalError("pytest native result must be a JSON object")
    count_keys = {
        "selected",
        "deselected",
        "passed",
        "failed",
        "errors",
        "skipped",
        "xfailed",
        "xpassed",
    }
    authority_keys = {
        "production_authorization",
        "promotion_authorization",
        "scientific_authorization",
        "evaluation_authorization",
        "runtime_authority",
    }
    expected_keys = {"schema_version", "exit_status", *count_keys, *authority_keys}
    if set(decoded) != expected_keys:
        raise WorkspaceLocalError("pytest native result keys are not exact")
    if decoded["schema_version"] != _PYTEST_NATIVE_RESULT_SCHEMA:
        raise WorkspaceLocalError("pytest native result schema is unsupported")
    exit_status = _strict_nonnegative_json_int(
        decoded["exit_status"],
        label="pytest exit status",
    )
    counts = {
        key: _strict_nonnegative_json_int(decoded[key], label=f"pytest {key}")
        for key in count_keys
    }
    for key in authority_keys:
        if decoded[key] is not False:
            raise WorkspaceLocalError(f"pytest native result {key} must be false")
    if exit_status == 0:
        terminal_total = sum(
            counts[key]
            for key in ("passed", "failed", "errors", "skipped", "xfailed", "xpassed")
        )
        if counts["selected"] != terminal_total:
            raise WorkspaceLocalError("pytest native result counts are inconsistent")
        if counts["failed"] != 0 or counts["errors"] != 0:
            raise WorkspaceLocalError("successful pytest native result reports failures")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "exit_status": exit_status,
        **counts,
    }


def _cross_check_pytest_evidence(
    junit: dict[str, object],
    native: dict[str, object],
    *,
    target_exit_code: int,
) -> None:
    if int(native["exit_status"]) != target_exit_code:
        raise WorkspaceLocalError("pytest native exit status differs from target exit")
    if target_exit_code != 0:
        return
    expected = {
        "tests": int(native["selected"]),
        "passed": int(native["passed"]) + int(native["xpassed"]),
        "failures": int(native["failed"]),
        "errors": int(native["errors"]),
        "skipped": int(native["skipped"]) + int(native["xfailed"]),
    }
    for key, value in expected.items():
        if int(junit[key]) != value:
            raise WorkspaceLocalError(f"pytest JUnit/native {key} counts disagree")


def _reconcile_stale_run(root: Path, run_id: str) -> dict[str, object]:
    if _RUN_ID.fullmatch(run_id) is None:
        raise WorkspaceLocalError(
            "reconcile run-id must be a portable identifier of at most 16 characters"
        )
    run_root = _require_within(
        root / "build" / "workspace-local-runs" / run_id,
        root,
        label="stale run root",
    )
    if not run_root.is_dir():
        raise WorkspaceLocalError("stale run namespace is missing")
    _reject_reparse_chain(run_root, root)
    execution_receipt_path = run_root / "execution-receipt.json"
    payload = _stable_bounded_regular_file_bytes(
        execution_receipt_path,
        root=root,
        limit_bytes=4 * 1024 * 1024,
        label="stale execution receipt",
    )
    try:
        execution_receipt = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_json_pairs,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise WorkspaceLocalError("stale execution receipt is invalid JSON") from exc
    if not isinstance(execution_receipt, dict):
        raise WorkspaceLocalError("stale execution receipt must be a JSON object")
    if execution_receipt.get("schema_version") != _RECEIPT_SCHEMA:
        raise WorkspaceLocalError("only execution receipt schema v6 can be reconciled")
    pending_statuses = {"PREFLIGHT_PENDING", "EXECUTION_PENDING", "EXECUTION_STARTED"}
    if execution_receipt.get("status") not in pending_statuses:
        raise WorkspaceLocalError("execution receipt is already terminal")
    if execution_receipt.get("workspace") != str(root):
        raise WorkspaceLocalError("stale execution receipt workspace is inconsistent")
    if execution_receipt.get("run_root") != str(run_root):
        raise WorkspaceLocalError("stale execution receipt run root is inconsistent")
    for key in (
        "production_authorization",
        "promotion_authorization",
        "scientific_authorization",
        "evaluation_authorization",
        "runtime_authority",
    ):
        if execution_receipt.get(key) is not False:
            raise WorkspaceLocalError(f"stale execution receipt {key} must be false")
    runner_observation = _observe_expected_process(
        execution_receipt.get("runner_process"),
        label="stale runner",
    )
    if runner_observation["exact_identity_active"] is True:
        raise WorkspaceLocalError("stale runner identity is still active")
    target_identity = execution_receipt.get("target_process")
    target_observation: dict[str, object] | None = None
    if target_identity is not None:
        target_observation = _observe_expected_process(
            target_identity,
            label="stale target",
        )
        if target_observation["exact_identity_active"] is True:
            raise WorkspaceLocalError("stale target identity is still active")
        process_tree = execution_receipt.get("process_tree")
        if not isinstance(process_tree, dict) or process_tree.get(
            "kill_on_supervisor_exit"
        ) is not True:
            raise WorkspaceLocalError(
                "started stale run lacks kill-on-supervisor-exit containment proof"
            )
    reconciliation_path = run_root / "reconciliation-receipt.json"
    if reconciliation_path.exists():
        raise WorkspaceLocalError("stale run already has a reconciliation receipt")
    _reject_reparse_chain(reconciliation_path.parent, root)
    reconciliation: dict[str, object] = {
        "schema_version": _RECONCILIATION_RECEIPT_SCHEMA,
        "status": "ABANDONED_RUN_CONFIRMED_NO_AUTHORITY",
        "production_authorization": False,
        "promotion_authorization": False,
        "scientific_authorization": False,
        "evaluation_authorization": False,
        "runtime_authority": False,
        "created_at_utc": _utc_now(),
        "workspace": str(root),
        "run_root": str(run_root),
        "execution_receipt": {
            "path": str(execution_receipt_path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "pending_status": execution_receipt["status"],
        },
        "runner_observation": runner_observation,
        "target_observation": target_observation,
        "reconciler_process": _new_process_identity(os.getpid(), label="reconciler"),
        "retry_disposition": "ROTATE_TO_A_FRESH_RUN_ID_ONLY",
    }
    _write_new_json(reconciliation_path, reconciliation)
    return reconciliation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"r-{uuid.uuid4().hex[:12]}")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--wall-timeout-seconds",
        type=_wall_timeout,
        default=_DEFAULT_WALL_TIMEOUT_SECONDS,
    )
    parser.add_argument("--reconcile-stale-run-id")
    parser.add_argument(_INTERNAL_WORKER_FLAG, action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _terminalize_pending_receipt(receipt_path: Path, status: str, exc: BaseException) -> None:
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload["status"] = status
        payload["failure_class"] = type(exc).__name__
        payload["failure_message"] = str(exc)
        payload["finished_at_utc"] = _utc_now()
        _write_json(receipt_path, payload)
    except OSError:
        pass


def _prepare_supervisor_root(root: Path, run_id: str) -> Path:
    if _RUN_ID.fullmatch(run_id) is None:
        raise WorkspaceLocalError("run-id must be a portable identifier of at most 16 characters")
    parent = _require_within(
        root / "build" / "workspace-local-supervisors",
        root,
        label="supervisor parent",
    )
    parent.mkdir(parents=True, exist_ok=True)
    _reject_reparse_chain(parent, root)
    supervisor_root = _require_within(parent / run_id, root, label="supervisor root")
    if supervisor_root.exists():
        raise WorkspaceLocalError("supervisor namespace already exists; rotate run-id")
    _probe_directory(supervisor_root)
    _reject_reparse_chain(supervisor_root, root)
    return supervisor_root


def _capture_supervisor_worker_source(root: Path, supervisor_root: Path) -> dict[str, object]:
    source = root / "scripts" / "run_workspace_local.py"
    payload = _stable_bounded_regular_file_bytes(
        source,
        root=root,
        limit_bytes=_SUPERVISOR_SOURCE_LIMIT_BYTES,
        label="workspace-local supervisor source",
    )
    captured = supervisor_root / "run_workspace_local_worker.py"
    with captured.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "source_path": str(source),
        "captured_path": str(captured),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_internal_worker_capability(
    *,
    root: Path,
    supervisor_root: Path,
    run_id: str,
    worker_source: dict[str, object],
    worker_argv_sha256: str,
    wall_timeout_seconds: float,
) -> tuple[Path, str, dict[str, object]]:
    token = secrets.token_hex(32)
    parent_identity = _new_process_identity(os.getpid(), label="supervisor")
    capability: dict[str, object] = {
        "schema_version": "pfc_lt_workspace_local_internal_capability.v1",
        "workspace": str(root),
        "supervisor_root": str(supervisor_root),
        "run_id": run_id,
        "parent_process": parent_identity,
        "worker_source": worker_source,
        "worker_argv_sha256": worker_argv_sha256,
        "wall_timeout_seconds": wall_timeout_seconds,
        "token": token,
        "production_authorization": False,
        "promotion_authorization": False,
        "scientific_authorization": False,
        "evaluation_authorization": False,
        "runtime_authority": False,
    }
    path = supervisor_root / "worker-capability.json"
    _write_new_json(path, capability)
    return path, token, capability


def _consume_internal_worker_capability(
    *,
    root: Path,
    args: argparse.Namespace,
    worker_argv: list[str],
) -> dict[str, object]:
    values = {
        "token": os.environ.get(_INTERNAL_WORKER_TOKEN_ENV),
        "capability_path": os.environ.get(_INTERNAL_WORKER_CAPABILITY_ENV),
        "parent_pid": os.environ.get(_INTERNAL_WORKER_PARENT_PID_ENV),
        "parent_token": os.environ.get(_INTERNAL_WORKER_PARENT_TOKEN_ENV),
    }
    if any(value is None for value in values.values()):
        raise WorkspaceLocalError("internal worker capability environment is incomplete")
    try:
        parent_pid = int(str(values["parent_pid"]))
    except ValueError as exc:
        raise WorkspaceLocalError("internal worker parent PID is invalid") from exc
    capability_path = _require_within(
        Path(str(values["capability_path"])),
        root,
        label="internal worker capability",
    )
    payload = _stable_bounded_regular_file_bytes(
        capability_path,
        root=root,
        limit_bytes=64 * 1024,
        label="internal worker capability",
    )
    try:
        capability = json.loads(payload, object_pairs_hook=_reject_duplicate_json_pairs)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise WorkspaceLocalError("internal worker capability is invalid JSON") from exc
    if not isinstance(capability, dict):
        raise WorkspaceLocalError("internal worker capability must be an object")
    expected_keys = {
        "schema_version",
        "workspace",
        "supervisor_root",
        "run_id",
        "parent_process",
        "worker_source",
        "worker_argv_sha256",
        "wall_timeout_seconds",
        "token",
        "production_authorization",
        "promotion_authorization",
        "scientific_authorization",
        "evaluation_authorization",
        "runtime_authority",
    }
    if set(capability) != expected_keys:
        raise WorkspaceLocalError("internal worker capability keys are not exact")
    if capability["schema_version"] != "pfc_lt_workspace_local_internal_capability.v1":
        raise WorkspaceLocalError("internal worker capability schema is unsupported")
    if capability["workspace"] != str(root) or capability["run_id"] != args.run_id:
        raise WorkspaceLocalError("internal worker capability scope is inconsistent")
    supervisor_root = _require_within(
        Path(str(capability["supervisor_root"])),
        root,
        label="internal worker supervisor root",
    )
    if _lexical(supervisor_root) != _lexical(capability_path.parent):
        raise WorkspaceLocalError("internal worker supervisor root is inconsistent")
    if not secrets.compare_digest(str(capability["token"]), str(values["token"])):
        raise WorkspaceLocalError("internal worker capability token is invalid")
    parent_identity = capability.get("parent_process")
    if not isinstance(parent_identity, dict):
        raise WorkspaceLocalError("internal worker parent identity is malformed")
    if parent_identity.get("pid") != parent_pid or parent_identity.get(
        "start_token"
    ) != values["parent_token"]:
        raise WorkspaceLocalError("internal worker parent identity binding is inconsistent")
    parent_observation = _observe_expected_process(
        parent_identity,
        label="internal worker parent",
    )
    if parent_observation["exact_identity_active"] is not True:
        raise WorkspaceLocalError("internal worker parent is not active")
    if _current_parent_pid() != parent_pid:
        raise WorkspaceLocalError("internal worker is not a direct child of its supervisor")
    expected_worker_argv_sha256 = capability.get("worker_argv_sha256")
    if (
        not isinstance(expected_worker_argv_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_worker_argv_sha256) is None
        or expected_worker_argv_sha256 != _command_sha256(worker_argv)
    ):
        raise WorkspaceLocalError("internal worker command binding is inconsistent")
    capability_timeout = capability.get("wall_timeout_seconds")
    if (
        not isinstance(capability_timeout, (int, float))
        or isinstance(capability_timeout, bool)
        or not math.isfinite(float(capability_timeout))
        or float(capability_timeout) != float(args.wall_timeout_seconds)
    ):
        raise WorkspaceLocalError("internal worker wall timeout binding is inconsistent")
    worker_source = capability.get("worker_source")
    if not isinstance(worker_source, dict) or set(worker_source) != {
        "source_path",
        "captured_path",
        "size_bytes",
        "sha256",
    }:
        raise WorkspaceLocalError("internal worker source binding is malformed")
    if _lexical(Path(str(worker_source["captured_path"]))) != _lexical(Path(__file__)):
        raise WorkspaceLocalError("internal worker did not execute the captured source")
    captured = _stable_bounded_regular_file_bytes(
        Path(__file__),
        root=root,
        limit_bytes=_SUPERVISOR_SOURCE_LIMIT_BYTES,
        label="captured internal worker source",
    )
    if len(captured) != worker_source["size_bytes"] or hashlib.sha256(
        captured
    ).hexdigest() != worker_source["sha256"]:
        raise WorkspaceLocalError("captured internal worker source changed")
    for key in (
        "production_authorization",
        "promotion_authorization",
        "scientific_authorization",
        "evaluation_authorization",
        "runtime_authority",
    ):
        if capability[key] is not False:
            raise WorkspaceLocalError(f"internal worker capability {key} must be false")
    admission = {
        "schema_version": "pfc_lt_workspace_local_worker_admission.v1",
        "status": "CAPABILITY_CONSUMED_NO_AUTHORITY",
        "created_at_utc": _utc_now(),
        "workspace": str(root),
        "run_id": args.run_id,
        "capability_sha256": hashlib.sha256(payload).hexdigest(),
        "worker_argv_sha256": expected_worker_argv_sha256,
        "worker_process": _new_process_identity(os.getpid(), label="internal worker"),
        "parent_process": parent_identity,
        "production_authorization": False,
        "promotion_authorization": False,
        "scientific_authorization": False,
        "evaluation_authorization": False,
        "runtime_authority": False,
    }
    _write_new_json(supervisor_root / "worker-admission.json", admission)
    capability_path.unlink()
    for key in (
        _INTERNAL_WORKER_TOKEN_ENV,
        _INTERNAL_WORKER_CAPABILITY_ENV,
        _INTERNAL_WORKER_PARENT_PID_ENV,
        _INTERNAL_WORKER_PARENT_TOKEN_ENV,
    ):
        os.environ.pop(key, None)
    return capability


def _worker_admission_evidence(
    *,
    root: Path,
    supervisor_root: Path,
    expected_run_id: str,
    expected_capability_sha256: str,
    expected_worker_argv_sha256: str,
    expected_worker_process: object,
    expected_parent_process: object,
) -> dict[str, object] | None:
    path = supervisor_root / "worker-admission.json"
    if not path.is_file():
        return None
    payload = _stable_bounded_regular_file_bytes(
        path,
        root=root,
        limit_bytes=64 * 1024,
        label="internal worker admission",
    )
    try:
        admission = json.loads(payload, object_pairs_hook=_reject_duplicate_json_pairs)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise WorkspaceLocalError("internal worker admission is invalid JSON") from exc
    if not isinstance(admission, dict):
        raise WorkspaceLocalError("internal worker admission must be an object")
    expected_keys = {
        "schema_version",
        "status",
        "created_at_utc",
        "workspace",
        "run_id",
        "capability_sha256",
        "worker_argv_sha256",
        "worker_process",
        "parent_process",
        "production_authorization",
        "promotion_authorization",
        "scientific_authorization",
        "evaluation_authorization",
        "runtime_authority",
    }
    if set(admission) != expected_keys:
        raise WorkspaceLocalError("internal worker admission keys are not exact")
    if (
        admission["schema_version"]
        != "pfc_lt_workspace_local_worker_admission.v1"
        or admission["status"] != "CAPABILITY_CONSUMED_NO_AUTHORITY"
        or admission["workspace"] != str(root)
        or admission["run_id"] != expected_run_id
        or admission["capability_sha256"] != expected_capability_sha256
        or admission["worker_argv_sha256"] != expected_worker_argv_sha256
        or admission["worker_process"] != expected_worker_process
        or admission["parent_process"] != expected_parent_process
    ):
        raise WorkspaceLocalError("internal worker admission binding is inconsistent")
    for key in (
        "production_authorization",
        "promotion_authorization",
        "scientific_authorization",
        "evaluation_authorization",
        "runtime_authority",
    ):
        if admission[key] is not False:
            raise WorkspaceLocalError(f"internal worker admission {key} must be false")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "status": admission["status"],
    }


def _revoke_worker_capability(
    *,
    root: Path,
    capability_path: Path,
    expected_sha256: str,
) -> bool:
    if not capability_path.is_file():
        return False
    payload = _stable_bounded_regular_file_bytes(
        capability_path,
        root=root,
        limit_bytes=64 * 1024,
        label="unconsumed internal worker capability",
    )
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise WorkspaceLocalError("unconsumed worker capability changed")
    capability_path.unlink()
    return True


def _terminalize_supervisor_timeout(
    *,
    root: Path,
    run_id: str,
    supervisor_root: Path,
) -> dict[str, object] | None:
    receipt_path = root / "build" / "workspace-local-runs" / run_id / "execution-receipt.json"
    if not receipt_path.is_file():
        return None
    payload = _stable_bounded_regular_file_bytes(
        receipt_path,
        root=root,
        limit_bytes=4 * 1024 * 1024,
        label="timed-out worker execution receipt",
    )
    try:
        receipt = json.loads(payload, object_pairs_hook=_reject_duplicate_json_pairs)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise WorkspaceLocalError("timed-out worker receipt is invalid JSON") from exc
    if not isinstance(receipt, dict) or receipt.get("schema_version") != _RECEIPT_SCHEMA:
        raise WorkspaceLocalError("timed-out worker receipt schema is unsupported")
    pending = {"PREFLIGHT_PENDING", "EXECUTION_PENDING", "EXECUTION_STARTED"}
    if receipt.get("status") in pending:
        receipt["status"] = "SUPERVISOR_WALL_TIMEOUT_TREE_TERMINATED_NO_AUTHORITY"
        receipt["runner_exit_code"] = 124
        receipt["finished_at_utc"] = _utc_now()
        receipt["supervisor_termination"] = {
            "supervisor_root": str(supervisor_root),
            "global_wall_timeout": True,
            "tree_termination_confirmed": True,
        }
        _write_json(receipt_path, receipt)
        payload = receipt_path.read_bytes()
    return {
        "path": str(receipt_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "status": receipt["status"],
    }


def _execution_receipt_evidence(root: Path, run_id: str) -> dict[str, object] | None:
    path = root / "build" / "workspace-local-runs" / run_id / "execution-receipt.json"
    if not path.is_file():
        return None
    payload = _stable_bounded_regular_file_bytes(
        path,
        root=root,
        limit_bytes=4 * 1024 * 1024,
        label="worker execution receipt",
    )
    try:
        decoded = json.loads(payload, object_pairs_hook=_reject_duplicate_json_pairs)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise WorkspaceLocalError("worker execution receipt is invalid JSON") from exc
    if not isinstance(decoded, dict) or decoded.get("schema_version") != _RECEIPT_SCHEMA:
        raise WorkspaceLocalError("worker execution receipt schema is unsupported")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "status": decoded.get("status"),
    }


def _build_supervisor_bootstrap_environment(
    supervisor_root: Path,
    inherited: dict[str, str] | None = None,
) -> tuple[dict[str, str], list[str], dict[str, Path]]:
    bootstrap_paths = {
        name: supervisor_root / f"worker-{name}"
        for name in (
            "appdata",
            "conda-envs",
            "conda-pkgs",
            "home",
            "joblib-temp",
            "local-appdata",
            "matplotlib-cache",
            "numba-cache",
            "pip-cache",
            "programdata",
            "pytest-root",
            "temp",
            "user-base",
            "uv-cache",
            "xdg-cache",
        )
    }
    for path in bootstrap_paths.values():
        _probe_directory(path)
    environment, removed_environment_keys = build_environment(
        dict(os.environ) if inherited is None else inherited,
        supervisor_root,
        bootstrap_paths,
    )
    return environment, removed_environment_keys, bootstrap_paths


def _run_supervisor(
    *,
    root: Path,
    args: argparse.Namespace,
    raw_argv: list[str],
) -> int:
    started_at = time.monotonic()
    supervisor_root = _prepare_supervisor_root(root, args.run_id)
    receipt_path = supervisor_root / "supervisor-receipt.json"
    redacted_command = _redact_command(raw_argv)
    receipt: dict[str, object] = {
        "schema_version": _SUPERVISOR_RECEIPT_SCHEMA,
        "status": "SUPERVISOR_PREFLIGHT_PENDING",
        "production_authorization": False,
        "promotion_authorization": False,
        "scientific_authorization": False,
        "evaluation_authorization": False,
        "runtime_authority": False,
        "created_at_utc": _utc_now(),
        "workspace": str(root),
        "run_id": args.run_id,
        "supervisor_root": str(supervisor_root),
        "supervisor_process": _new_process_identity(os.getpid(), label="supervisor"),
        "redacted_command": redacted_command,
        "redacted_command_sha256": _command_sha256(redacted_command),
        "global_wall_timeout_seconds": args.wall_timeout_seconds,
        "execution_receipt": None,
        "resource_usage": {"available": False},
    }
    _write_json(receipt_path, receipt)
    try:
        worker_source = _capture_supervisor_worker_source(root, supervisor_root)
        capability_path, token, capability = _write_internal_worker_capability(
            root=root,
            supervisor_root=supervisor_root,
            run_id=args.run_id,
            worker_source=worker_source,
            worker_argv_sha256=_command_sha256(raw_argv),
            wall_timeout_seconds=args.wall_timeout_seconds,
        )
        capability_payload = _stable_bounded_regular_file_bytes(
            capability_path,
            root=root,
            limit_bytes=64 * 1024,
            label="supervisor worker capability",
        )
        capability_sha256 = hashlib.sha256(capability_payload).hexdigest()
        receipt["worker_source"] = worker_source
        receipt["worker_capability"] = {
            "path": str(capability_path),
            "sha256": capability_sha256,
            "size_bytes": len(capability_payload),
            "one_shot": True,
        }
        receipt["status"] = "SUPERVISOR_ENVIRONMENT_PREFLIGHT_PENDING"
        _write_json(receipt_path, receipt)
    except (OSError, WorkspaceLocalError) as exc:
        receipt["status"] = "SUPERVISOR_SOURCE_PREFLIGHT_FAILED_NO_AUTHORITY"
        receipt["failure_class"] = type(exc).__name__
        receipt["failure_message"] = str(exc)
        receipt["finished_at_utc"] = _utc_now()
        _write_json(receipt_path, receipt)
        raise
    supervisor_inherited = dict(os.environ)
    try:
        environment, removed_environment_keys, bootstrap_paths = (
            _build_supervisor_bootstrap_environment(
                supervisor_root,
                inherited=supervisor_inherited,
            )
        )
    except (OSError, WorkspaceLocalError) as exc:
        revoked = _revoke_worker_capability(
            root=root,
            capability_path=capability_path,
            expected_sha256=capability_sha256,
        )
        receipt["worker_capability"]["revoked_after_worker"] = revoked
        receipt["worker_capability"]["absent_after_worker"] = (
            not capability_path.exists()
        )
        receipt["status"] = "SUPERVISOR_ENVIRONMENT_PREFLIGHT_FAILED_NO_AUTHORITY"
        receipt["failure_class"] = type(exc).__name__
        receipt["failure_message"] = str(exc)
        receipt["finished_at_utc"] = _utc_now()
        _write_json(receipt_path, receipt)
        raise
    receipt["removed_environment_key_count"] = len(removed_environment_keys)
    receipt["eex_datasource_access_token_forwarded"] = False
    receipt["ambient_credential_isolation"] = "ENV_DENYLIST_ONLY_NOT_FILESYSTEM_SANDBOX"
    receipt["mutable_paths"] = {
        name: str(path) for name, path in sorted(bootstrap_paths.items())
    }
    preflight_elapsed = time.monotonic() - started_at
    if preflight_elapsed >= args.wall_timeout_seconds:
        revoked = _revoke_worker_capability(
            root=root,
            capability_path=capability_path,
            expected_sha256=capability_sha256,
        )
        receipt["worker_capability"]["consumed"] = False
        receipt["worker_capability"]["revoked_before_worker_start"] = revoked
        receipt["worker_capability"]["absent_after_worker"] = (
            not capability_path.exists()
        )
        receipt["status"] = "SUPERVISOR_PREFLIGHT_WALL_TIMEOUT_NO_AUTHORITY"
        receipt["runner_exit_code"] = 124
        receipt["resource_usage"] = {
            "available": False,
            "wall_duration_seconds": preflight_elapsed,
        }
        receipt["finished_at_utc"] = _utc_now()
        _write_json(receipt_path, receipt)
        return 124
    _write_json(receipt_path, receipt)
    environment[_INTERNAL_WORKER_TOKEN_ENV] = token
    environment[_INTERNAL_WORKER_CAPABILITY_ENV] = str(capability_path)
    environment[_INTERNAL_WORKER_PARENT_PID_ENV] = str(os.getpid())
    environment[_INTERNAL_WORKER_PARENT_TOKEN_ENV] = str(
        capability["parent_process"]["start_token"]
    )
    worker_command = [
        sys.executable,
        "-I",
        "-B",
        str(worker_source["captured_path"]),
        _INTERNAL_WORKER_FLAG,
        *raw_argv,
    ]
    windows_job: int | None = None
    worker: subprocess.Popen[bytes] | None = None
    worker_exit_code: int | None = None
    timed_out = False
    descendant_cleanup_triggered = False
    posix_usage_before = _posix_child_usage_snapshot()
    try:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        if os.name == "nt":
            windows_job = _create_windows_kill_on_close_job()
            creationflags |= _WINDOWS_CREATE_SUSPENDED
        worker = subprocess.Popen(
            worker_command,
            cwd=root,
            env=environment,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
        if windows_job is not None:
            _assign_process_to_windows_job(windows_job, worker)
        receipt["worker_process"] = _new_process_identity(worker.pid, label="worker")
        receipt["process_tree"] = {
            "mode": (
                "CREATE_SUSPENDED_ASSIGN_KILL_ON_JOB_CLOSE_THEN_RESUME"
                if os.name == "nt"
                else "START_NEW_SESSION_PROCESS_GROUP"
            ),
            "kill_on_supervisor_exit": True,
        }
        receipt["status"] = "WORKER_STARTED"
        receipt["worker_started_at_utc"] = _utc_now()
        _write_json(receipt_path, receipt)
        if os.name == "nt":
            _resume_windows_process(worker)
        remaining = args.wall_timeout_seconds - (time.monotonic() - started_at)
        try:
            if remaining <= 0:
                raise subprocess.TimeoutExpired(worker_command, args.wall_timeout_seconds)
            worker_exit_code = worker.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_tree(worker, windows_job=windows_job)
            worker_exit_code = worker.returncode
        if windows_job is not None:
            usage = _windows_job_resource_usage(windows_job)
            if int(usage["active_processes"]) > 0:
                descendant_cleanup_triggered = True
                _terminate_process_tree(worker, windows_job=windows_job)
                usage = _windows_job_resource_usage(windows_job)
        else:
            if _posix_process_group_active(worker.pid):
                descendant_cleanup_triggered = True
                _terminate_process_tree(worker, windows_job=None)
            usage = _posix_child_usage_delta(posix_usage_before)
        worker_admission = _worker_admission_evidence(
            root=root,
            supervisor_root=supervisor_root,
            expected_run_id=args.run_id,
            expected_capability_sha256=capability_sha256,
            expected_worker_argv_sha256=_command_sha256(raw_argv),
            expected_worker_process=receipt["worker_process"],
            expected_parent_process=receipt["supervisor_process"],
        )
        receipt["worker_admission"] = worker_admission
        capability_was_unconsumed = _revoke_worker_capability(
            root=root,
            capability_path=capability_path,
            expected_sha256=capability_sha256,
        )
        receipt["worker_capability"]["consumed"] = worker_admission is not None
        receipt["worker_capability"]["revoked_after_worker"] = capability_was_unconsumed
        receipt["worker_capability"]["absent_after_worker"] = not capability_path.exists()
        if timed_out:
            receipt["execution_receipt"] = _terminalize_supervisor_timeout(
                root=root,
                run_id=args.run_id,
                supervisor_root=supervisor_root,
            )
            receipt["status"] = "SUPERVISOR_WALL_TIMEOUT_TREE_TERMINATED_NO_AUTHORITY"
            receipt["runner_exit_code"] = 124
            return_code = 124
        elif worker_exit_code == 0 and worker_admission is None:
            receipt["execution_receipt"] = _execution_receipt_evidence(root, args.run_id)
            receipt["status"] = "WORKER_CAPABILITY_ADMISSION_MISSING_NO_AUTHORITY"
            receipt["runner_exit_code"] = 2
            return_code = 2
        elif descendant_cleanup_triggered:
            receipt["execution_receipt"] = _execution_receipt_evidence(root, args.run_id)
            receipt["status"] = "WORKER_DESCENDANT_LEAK_TERMINATED_NO_AUTHORITY"
            receipt["runner_exit_code"] = 2
            return_code = 2
        else:
            receipt["execution_receipt"] = _execution_receipt_evidence(root, args.run_id)
            receipt["status"] = (
                "WORKER_EXIT_ZERO_NOT_AUTHORITY"
                if worker_exit_code == 0
                else "WORKER_EXIT_NONZERO_NO_AUTHORITY"
            )
            receipt["runner_exit_code"] = int(worker_exit_code)
            return_code = int(worker_exit_code)
        receipt["worker_exit_code"] = int(worker_exit_code)
        elapsed_before_terminal_write = time.monotonic() - started_at
        deadline_exceeded = elapsed_before_terminal_write >= args.wall_timeout_seconds
        if deadline_exceeded and not timed_out:
            receipt["status"] = "SUPERVISOR_POSTFLIGHT_WALL_TIMEOUT_NO_AUTHORITY"
            receipt["runner_exit_code"] = 124
            return_code = 124
        usage["wall_duration_seconds"] = elapsed_before_terminal_write
        receipt["resource_usage"] = {"available": True, **usage}
        receipt["deadline"] = {
            "budget_seconds": args.wall_timeout_seconds,
            "elapsed_before_terminal_write_seconds": elapsed_before_terminal_write,
            "overshoot_before_terminal_write_seconds": max(
                0.0,
                elapsed_before_terminal_write - args.wall_timeout_seconds,
            ),
            "deadline_exceeded": deadline_exceeded,
            "strict_return_bound": False,
            "cleanup_and_terminal_fsync_may_overshoot": True,
        }
        receipt["terminal_receipt_write_attempts"] = 1
        receipt["finished_at_utc"] = _utc_now()
        _write_json(receipt_path, receipt)
        elapsed_after_terminal_write = time.monotonic() - started_at
        if elapsed_after_terminal_write >= args.wall_timeout_seconds and not deadline_exceeded:
            receipt["status"] = "SUPERVISOR_TERMINAL_FSYNC_WALL_TIMEOUT_NO_AUTHORITY"
            receipt["runner_exit_code"] = 124
            receipt["resource_usage"]["wall_duration_seconds"] = (
                elapsed_after_terminal_write
            )
            receipt["deadline"]["elapsed_after_first_terminal_write_seconds"] = (
                elapsed_after_terminal_write
            )
            receipt["deadline"]["overshoot_after_first_terminal_write_seconds"] = max(
                0.0,
                elapsed_after_terminal_write - args.wall_timeout_seconds,
            )
            receipt["deadline"]["deadline_exceeded"] = True
            receipt["terminal_receipt_write_attempts"] = 2
            receipt["finished_at_utc"] = _utc_now()
            _write_json(receipt_path, receipt)
            return_code = 124
        return return_code
    except BaseException as exc:
        if worker is not None and worker.poll() is None:
            _terminate_process_tree(worker, windows_job=windows_job)
        revoked = _revoke_worker_capability(
            root=root,
            capability_path=capability_path,
            expected_sha256=capability_sha256,
        )
        receipt["worker_capability"]["revoked_after_worker"] = revoked
        receipt["worker_capability"]["absent_after_worker"] = (
            not capability_path.exists()
        )
        receipt["status"] = "SUPERVISOR_FAILED_NO_AUTHORITY"
        receipt["failure_class"] = type(exc).__name__
        receipt["failure_message"] = str(exc)
        receipt["finished_at_utc"] = _utc_now()
        _write_json(receipt_path, receipt)
        raise
    finally:
        if worker is not None and worker.poll() is None:
            _terminate_process_tree(worker, windows_job=windows_job)
        if windows_job is not None:
            _close_windows_handle(windows_job)


def _worker_main(
    argv: list[str] | None = None,
    *,
    root_override: Path | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if args.internal_supervised_worker:
        raise WorkspaceLocalError("internal worker flag must be consumed by supervisor")
    root = root_override if root_override is not None else canonical_repo_root()
    verify_canonical_workspace(root)
    if args.reconcile_stale_run_id is not None:
        if args.preflight_only or args.command:
            raise WorkspaceLocalError(
                "stale reconciliation cannot be combined with execution options"
            )
        reconciled = _reconcile_stale_run(root, args.reconcile_stale_run_id)
        print(json.dumps(reconciled, sort_keys=True))
        return 0
    run_root, paths = prepare_run(root, args.run_id)
    receipt_path = run_root / "execution-receipt.json"
    worker_inherited = dict(os.environ)
    try:
        env, removed_environment_keys = build_environment(worker_inherited, run_root, paths)
        identities = _path_identities(run_root, paths)
        pytest_basetemp = paths["pytest-root"] / "basetemp"
        pytest_junit_path = run_root / "pytest-results.xml"
        pytest_native_result_path = run_root / "pytest-native-result.json"
        mutable_usage_before = _mutable_filesystem_usage(paths)
    except (OSError, WorkspaceLocalError) as exc:
        _terminalize_pending_receipt(receipt_path, "ENVIRONMENT_PREFLIGHT_FAILED", exc)
        raise
    command: list[str] = []
    receipt: dict[str, object] = {
        "schema_version": _RECEIPT_SCHEMA,
        "status": "PREFLIGHT_PASS_NOT_EXECUTED" if args.preflight_only else "EXECUTION_PENDING",
        "production_authorization": False,
        "promotion_authorization": False,
        "scientific_authorization": False,
        "evaluation_authorization": False,
        "runtime_authority": False,
        "ambient_credential_isolation": "ENV_DENYLIST_ONLY_NOT_FILESYSTEM_SANDBOX",
        "created_at_utc": _utc_now(),
        "runner_process_id": os.getpid(),
        "runner_process": _new_process_identity(os.getpid(), label="runner"),
        "workspace": str(root),
        "run_root": str(run_root),
        "mutable_paths": {name: str(path) for name, path in sorted(paths.items())},
        "pytest_basetemp": str(pytest_basetemp),
        "pytest_junit_path": str(pytest_junit_path),
        "pytest_native_result_path": str(pytest_native_result_path),
        "redacted_command": [],
        "removed_environment_key_count": len(removed_environment_keys),
        "execution_identity": None,
        "target_output": {
            "capture_limit_bytes_per_stream": _OUTPUT_CAPTURE_LIMIT_BYTES,
            "complete": False,
            "streams": {},
        },
        "pytest_result": {"applicable": False},
        "process_tree": {
            "mode": "NOT_STARTED",
            "kill_on_supervisor_exit": False,
            "global_wall_timeout_seconds": args.wall_timeout_seconds,
        },
        "resource_usage": {
            "available": False,
            "mutable_filesystem_before": mutable_usage_before,
        },
    }
    if not args.preflight_only:
        attempted_command = list(args.command)
        if attempted_command and attempted_command[0] == "--":
            attempted_command = attempted_command[1:]
        receipt["redacted_command"] = _redact_command(attempted_command)
        receipt["redacted_command_sha256"] = _command_sha256(list(receipt["redacted_command"]))
        try:
            command = prepare_command(
                args.command,
                root,
                pytest_basetemp,
                pytest_junit_path,
                pytest_native_result_path,
            )
            execution_identity = _execution_identity(root, command)
            _require_eex_datasource_execution_identity(command, execution_identity, root)
            _require_local_future_origin_execution_identity(
                command, execution_identity, root
            )
            receipt["execution_identity"] = execution_identity
            receipt["eex_datasource_access_token_forwarded"] = False
        except (OSError, WorkspaceLocalError, subprocess.SubprocessError) as exc:
            receipt["status"] = "REJECTED_BEFORE_EXECUTION"
            receipt["failure_class"] = type(exc).__name__
            receipt["failure_message"] = str(exc)
            receipt["finished_at_utc"] = _utc_now()
            _write_json(receipt_path, receipt)
            raise
        receipt["redacted_command"] = _redact_command(command)
        receipt["redacted_command_sha256"] = _command_sha256(list(receipt["redacted_command"]))
        try:
            _revalidate_paths(root, run_root, paths, identities)
        except (OSError, WorkspaceLocalError) as exc:
            receipt["status"] = "PATH_REVALIDATION_FAILED"
            receipt["failure_class"] = type(exc).__name__
            receipt["failure_message"] = str(exc)
            receipt["finished_at_utc"] = _utc_now()
            _write_json(receipt_path, receipt)
            raise
    _write_json(receipt_path, receipt)
    if args.preflight_only:
        receipt["finished_at_utc"] = _utc_now()
        _write_json(receipt_path, receipt)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    try:
        def record_started(started: dict[str, object]) -> None:
            target_process = started.get("target_process")
            process_tree = started.get("process_tree")
            if not isinstance(target_process, dict) or not isinstance(process_tree, dict):
                raise WorkspaceLocalError("target start evidence is malformed")
            receipt["status"] = "EXECUTION_STARTED"
            receipt["target_process"] = target_process
            receipt["process_tree"] = process_tree
            receipt["target_started_at_utc"] = _utc_now()
            _write_json(receipt_path, receipt)

        completed = _execute_command_with_capture(
            command,
            cwd=root,
            env=env,
            run_root=run_root,
            wall_timeout_seconds=args.wall_timeout_seconds,
            on_started=record_started,
        )
    except KeyboardInterrupt as exc:
        receipt["status"] = "INTERRUPTED"
        receipt["failure_class"] = type(exc).__name__
        receipt["failure_message"] = str(exc)
        receipt["finished_at_utc"] = _utc_now()
        _write_json(receipt_path, receipt)
        raise
    except (OSError, WorkspaceLocalError) as exc:
        receipt["status"] = (
            "OUTPUT_CAPTURE_FAILED"
            if isinstance(exc, WorkspaceLocalError)
            else "LAUNCH_FAILED"
        )
        receipt["failure_class"] = type(exc).__name__
        receipt["failure_message"] = str(exc)
        receipt["finished_at_utc"] = _utc_now()
        _write_json(receipt_path, receipt)
        raise
    try:
        if _execution_identity(root, command) != receipt["execution_identity"]:
            raise WorkspaceLocalError("execution identity changed during target run")
        verified_streams: dict[str, dict[str, object]] = {}
        for label, filename in (
            ("stdout", "target-stdout.log"),
            ("stderr", "target-stderr.log"),
        ):
            expected_path = run_root / filename
            observed = completed.streams[label]
            if observed.get("path") != str(expected_path):
                raise WorkspaceLocalError(f"{label} capture path is inconsistent")
            captured = _stable_bounded_regular_file_bytes(
                expected_path,
                root=root,
                limit_bytes=_OUTPUT_CAPTURE_LIMIT_BYTES,
                label=f"target {label} capture",
            )
            if observed.get("captured_bytes") != len(captured):
                raise WorkspaceLocalError(f"{label} capture size is inconsistent")
            if observed.get("sha256") != hashlib.sha256(captured).hexdigest():
                raise WorkspaceLocalError(f"{label} capture hash is inconsistent")
            verified_streams[label] = dict(observed)
        output_complete = not any(
            bool(item["truncated"]) for item in verified_streams.values()
        )
        receipt["target_output"] = {
            "capture_limit_bytes_per_stream": _OUTPUT_CAPTURE_LIMIT_BYTES,
            "complete": output_complete,
            "streams": verified_streams,
        }
        receipt["supervision"] = completed.supervision
        receipt["resource_usage"] = {
            "available": True,
            **completed.resource_usage,
            "mutable_filesystem_before": mutable_usage_before,
            "mutable_filesystem_after": _mutable_filesystem_usage(paths),
        }
        supervision_terminal = completed.timed_out or completed.interrupted or bool(
            completed.supervision.get("descendant_cleanup_triggered")
        )
        if _is_pytest(command) and not supervision_terminal:
            pytest_result = _pytest_junit_evidence(
                pytest_junit_path,
                root=root,
            )
            pytest_native_result = _pytest_native_evidence(
                pytest_native_result_path,
                root=root,
            )
            _cross_check_pytest_evidence(
                pytest_result,
                pytest_native_result,
                target_exit_code=int(completed.returncode),
            )
            if completed.returncode == 0 and (
                int(pytest_result["failures"]) != 0
                or int(pytest_result["errors"]) != 0
            ):
                raise WorkspaceLocalError(
                    "successful pytest exit conflicts with JUnit failure counts"
                )
            pytest_result["native_result"] = pytest_native_result
            pytest_result["deselected"] = int(pytest_native_result["deselected"])
            pytest_result["xfailed"] = int(pytest_native_result["xfailed"])
            pytest_result["xpassed"] = int(pytest_native_result["xpassed"])
            receipt["pytest_result"] = pytest_result
        elif _is_pytest(command):
            receipt["pytest_result"] = {
                "applicable": True,
                "complete": False,
                "reason": (
                    "GLOBAL_WALL_TIMEOUT"
                    if completed.timed_out
                    else (
                        "SUPERVISOR_INTERRUPT"
                        if completed.interrupted
                        else "DESCENDANT_PROCESS_LEAK"
                    )
                ),
            }
        if not output_complete:
            receipt["status"] = "TARGET_OUTPUT_CAPTURE_LIMIT_EXCEEDED"
            receipt["target_exit_code"] = int(completed.returncode)
            receipt["runner_exit_code"] = 2
            receipt["finished_at_utc"] = _utc_now()
            _write_json(receipt_path, receipt)
            return 2
    except (OSError, WorkspaceLocalError) as exc:
        receipt["status"] = "TARGET_EVIDENCE_INVALID"
        receipt["failure_class"] = type(exc).__name__
        receipt["failure_message"] = str(exc)
        receipt["target_exit_code"] = int(completed.returncode)
        receipt["runner_exit_code"] = 2
        receipt["finished_at_utc"] = _utc_now()
        _write_json(receipt_path, receipt)
        return 2
    try:
        _revalidate_paths(root, run_root, paths, identities)
    except (OSError, WorkspaceLocalError) as exc:
        receipt["status"] = "POST_EXECUTION_PATH_REVALIDATION_FAILED"
        receipt["failure_class"] = type(exc).__name__
        receipt["failure_message"] = str(exc)
        receipt["target_exit_code"] = int(completed.returncode)
        receipt["finished_at_utc"] = _utc_now()
        _write_json(receipt_path, receipt)
        raise
    if completed.timed_out:
        receipt["status"] = "TARGET_WALL_TIMEOUT_TREE_TERMINATED_NO_AUTHORITY"
        receipt["target_exit_code"] = int(completed.returncode)
        receipt["runner_exit_code"] = 124
        receipt["finished_at_utc"] = _utc_now()
        _write_json(receipt_path, receipt)
        return 124
    if completed.interrupted:
        receipt["status"] = "INTERRUPTED_TREE_TERMINATED_NO_AUTHORITY"
        receipt["target_exit_code"] = int(completed.returncode)
        receipt["runner_exit_code"] = 130
        receipt["finished_at_utc"] = _utc_now()
        _write_json(receipt_path, receipt)
        return 130
    if completed.supervision.get("descendant_cleanup_triggered") is True:
        receipt["status"] = "TARGET_DESCENDANT_LEAK_TERMINATED_NO_AUTHORITY"
        receipt["target_exit_code"] = int(completed.returncode)
        receipt["runner_exit_code"] = 2
        receipt["finished_at_utc"] = _utc_now()
        _write_json(receipt_path, receipt)
        return 2
    receipt["status"] = (
        "TARGET_EXIT_ZERO_NOT_AUTHORITY" if completed.returncode == 0 else "TARGET_EXIT_NONZERO"
    )
    receipt["target_exit_code"] = int(completed.returncode)
    receipt["runner_exit_code"] = int(completed.returncode)
    receipt["finished_at_utc"] = _utc_now()
    _write_json(receipt_path, receipt)
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(raw_argv)
    inherited = dict(os.environ)
    eex_access_token = _selected_eex_datasource_token(args.command, inherited)
    eex_token_was_present = _EEX_DATASOURCE_TOKEN_ENV in inherited
    os.environ.pop(_EEX_DATASOURCE_TOKEN_ENV, None)
    _reject_forwarded_token_in_command(raw_argv, eex_access_token)
    if eex_token_was_present and _command_targets_eex_datasource(args.command):
        raise WorkspaceLocalError(
            "EEX DataSource token forwarding is disabled until an independently "
            "admitted immutable runtime and secret handoff exist"
        )
    if args.internal_supervised_worker:
        if raw_argv.count(_INTERNAL_WORKER_FLAG) != 1 or raw_argv[0] != _INTERNAL_WORKER_FLAG:
            raise WorkspaceLocalError("internal worker flag position is invalid")
        root = _EXPECTED_ROOT
        verify_canonical_workspace(root)
        worker_argv = raw_argv[1:]
        _consume_internal_worker_capability(
            root=root,
            args=args,
            worker_argv=worker_argv,
        )
        return _worker_main(worker_argv, root_override=root)
    reserved = {
        key for key in (
            _INTERNAL_WORKER_TOKEN_ENV,
            _INTERNAL_WORKER_CAPABILITY_ENV,
            _INTERNAL_WORKER_PARENT_PID_ENV,
            _INTERNAL_WORKER_PARENT_TOKEN_ENV,
        )
        if key in os.environ
    }
    if reserved:
        raise WorkspaceLocalError(
            "reserved internal worker environment is forbidden in supervisor mode"
        )
    root = canonical_repo_root()
    verify_canonical_workspace(root)
    if args.reconcile_stale_run_id is not None:
        if args.preflight_only or args.command:
            raise WorkspaceLocalError(
                "stale reconciliation cannot be combined with execution options"
            )
        reconciled = _reconcile_stale_run(root, args.reconcile_stale_run_id)
        print(json.dumps(reconciled, sort_keys=True))
        return 0
    return _run_supervisor(root=root, args=args, raw_argv=raw_argv)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, WorkspaceLocalError, subprocess.SubprocessError) as exc:
        print(f"WORKSPACE_LOCAL_PRECHECK_FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

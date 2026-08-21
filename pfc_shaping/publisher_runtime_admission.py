"""Fail-closed admission of the separately installed publisher runtime."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import platform
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import sysconfig
import time
import zipfile
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from pfc_shaping.exit_codes import EXIT_INTEGRITY_FAILURE
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)

_RUNTIME_CONTRACT_PATH = "FMV-SNAPSHOT-PUBLISHER-RUNTIME-CONTRACT.json"
_ENVIRONMENT_CONTRACT_PATH = "FMV-SNAPSHOT-PUBLISHER-ENVIRONMENT-CONTRACT.json"
_ARTIFACT_MANIFEST_PATH = "FMV-SNAPSHOT-PUBLISHER-MANIFEST.json"
_ARTIFACT_SCHEMA = "fmv_lt_snapshot_publisher_artifact.v2"
_RUNTIME_SCHEMA = "fmv_lt_snapshot_publisher_runtime.v6"
RUNTIME_DEPENDENCY_ROOT_ENV = "PFC_SNAPSHOT_PUBLISHER_DEPENDENCY_ROOT"
SUPERVISOR_SCRATCH_ROOT_ENV = "PFC_SNAPSHOT_PUBLISHER_SCRATCH_ROOT"
SUPERVISOR_WORKER_TIMEOUT_ENV = "PFC_SNAPSHOT_PUBLISHER_WORKER_TIMEOUT_SECONDS"
SUPERVISOR_ADMISSION_SLO_ENV = "PFC_SNAPSHOT_PUBLISHER_ADMISSION_SLO_SECONDS"
_SUPERVISED_WORKER_ENV = "_PFC_SNAPSHOT_PUBLISHER_SUPERVISED_WORKER"
_SUPERVISED_CAPTURE_ROOT_ENV = "_PFC_SNAPSHOT_PUBLISHER_CAPTURE_ROOT"
_SUPERVISED_WORKER_TOKEN_ENV = "_PFC_SNAPSHOT_PUBLISHER_WORKER_TOKEN"
_SUPERVISOR_PREFIX = "fmv-pfc-publisher-supervisor-"
_SUPERVISOR_CAPABILITY = ".worker-capability.json"
_SUPERVISOR_DEFERRED_CAPABILITY = ".post-admission-environment.json"
_SUPERVISOR_ADMISSION_METRIC = ".runtime-admission-metric.json"
_WINDOWS_CREATE_SUSPENDED = 0x00000004
_MIN_SUPERVISOR_SCRATCH_FREE_BYTES = 512 * 1024 * 1024
_MAX_WINDOWS_CAPTURE_PATH_CHARS = 240
_MAX_ARTIFACT_MEMBER_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_ARTIFACT_TOTAL_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_HOST_ENVIRONMENT_NAMES = frozenset(
    {
        "APPDATA",
        "COMSPEC",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "NUMBER_OF_PROCESSORS",
        "OS",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_IDENTIFIER",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "USERDOMAIN",
        "USERNAME",
        "USERPROFILE",
        "WINDIR",
    }
)
_SENSITIVE_ENVIRONMENT_MARKERS = (
    "API_KEY",
    "PRIVATE_KEY",
    "MTLS_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
)
_HARD_FORBIDDEN_ENVIRONMENT = frozenset(
    {
        _SUPERVISED_WORKER_ENV,
        _SUPERVISED_CAPTURE_ROOT_ENV,
        _SUPERVISED_WORKER_TOKEN_ENV,
        "PFC_DATA_PUBLICATION_BOOTSTRAP_SIGNING_PRIVATE_KEY_PATH",
        "PFC_DATA_PUBLICATION_ANCHOR_SIGNING_PRIVATE_KEY_PATH",
        "PFC_DATA_ACQUISITION_SIGNING_PRIVATE_KEY_PATH",
        "PFC_DATA_TIMESTAMP_SIGNING_PRIVATE_KEY_PATH",
        "PFC_DATA_JOURNAL_SIGNING_PRIVATE_KEY_PATH",
        "PFC_PROMOTION_SIGNING_PRIVATE_KEY_PATH",
        "PFC_PROMOTION_EVENT_SIGNING_PRIVATE_KEY_PATH",
        "PFC_ROLLBACK_AUTHORIZATION_SIGNING_PRIVATE_KEY_PATH",
        "PFC_MODEL_GOVERNANCE_SIGNING_PRIVATE_KEY_PATH",
        "PFC_RELEASE_REQUEST_SIGNING_PRIVATE_KEY_PATH",
        "PFC_QUOTE_CONFLICT_POLICY_SIGNING_PRIVATE_KEY_PATH",
    }
)
_PHASE_DEFERRED_ENVIRONMENT = {
    "prepare": frozenset(
        {
            "PFC_DATA_PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_PATH",
            "PFC_DATA_PUBLICATION_ANCHOR_MTLS_KEY_PATH",
        }
    ),
    "cas": frozenset({"PFC_DATA_PUBLICATION_ANCHOR_MTLS_KEY_PATH"}),
    "finalize": frozenset({"PFC_DATA_PUBLICATION_ANCHOR_MTLS_KEY_PATH"}),
}
PUBLISHER_DEPENDENCY_LOADING_POLICY = {
    "verified_source_root_imported": False,
    "capture_mode": "PROCESS_PRIVATE_EXACT_BYTE_COPY",
    "source_revalidation": "POST_CAPTURE_FULL_TREE",
    "captured_tree_reverification": True,
    "captured_distribution_reverification": True,
    "capture_cleanup": "SUPERVISING_PARENT_AFTER_WORKER_EXIT",
    "supervisor_capability": (
        "ONE_SHOT_TOKEN_PARENT_PID_ARTIFACT_PATH_AND_SCRATCH_BOUND"
    ),
    "supervisor_scratch": "EXPLICIT_GOVERNED_NO_REPARSE_ROOT_WITH_CAPACITY_FLOOR",
    "supervisor_deadline": "REQUIRED_TIMEOUT_AND_SLO_WITH_STRUCTURED_METRIC",
    "windows_process_tree": (
        "CREATE_SUSPENDED_ASSIGN_KILL_ON_JOB_CLOSE_THEN_RESUME"
    ),
    "host_read_only_attestation": "REQUIRED_EXTERNAL_SIGNED_ATTESTATION",
    "host_capture_isolation": (
        "DEDICATED_IDENTITY_NO_UNTRUSTED_COPROCESS_SAME_TOKEN"
    ),
}


def _allocate_inherited_acl_runtime_directory(
    *, prefix: str, parent: str | Path
) -> Path:
    selected_parent = Path(parent)
    for _ in range(32):
        selected = selected_parent / f"{prefix}{secrets.token_hex(16)}"
        try:
            selected.mkdir()
        except FileExistsError:
            continue
        return selected
    raise PublisherRuntimeAdmissionError(
        "runtime private scratch directory cannot be allocated"
    )


class ProcessRuntimeDirectory:
    """Random inherited-ACL directory for Windows sandbox/service scratch roots."""

    def __init__(self, *, prefix: str, parent: str | Path) -> None:
        self.name = str(
            _allocate_inherited_acl_runtime_directory(prefix=prefix, parent=parent)
        )
        self._active = True

    def cleanup(self) -> None:
        if getattr(self, "_active", False):
            shutil.rmtree(self.name, onerror=_remove_readonly_runtime_entry)
            self._active = False

    def __del__(self) -> None:
        try:
            self.cleanup()
        except BaseException:
            pass


_CAPTURED_DEPENDENCY_RUNTIME: ProcessRuntimeDirectory | None = None


def cleanup_captured_dependency_runtime() -> None:
    """Remove an admitted process-private dependency capture exactly once."""

    global _CAPTURED_DEPENDENCY_RUNTIME
    if _CAPTURED_DEPENDENCY_RUNTIME is not None:
        _CAPTURED_DEPENDENCY_RUNTIME.cleanup()
        _CAPTURED_DEPENDENCY_RUNTIME = None


def _remove_readonly_runtime_entry(
    operation: Callable[[str], object],
    path: str,
    _error: object,
) -> None:
    os.chmod(path, stat.S_IRWXU)
    operation(path)


class PublisherRuntimeAdmissionError(RuntimeError):
    """Raised before publisher business modules load under a divergent runtime."""


class PublisherPostAdmissionEnvironmentDeliveryError(PublisherRuntimeAdmissionError):
    """Retain a sealed metric when deferred-authority delivery fails."""

    def __init__(
        self,
        message: str,
        *,
        admission_metric: dict[str, object],
    ) -> None:
        super().__init__(message)
        self.admission_metric = admission_metric


def verify_publisher_runtime(artifact_path: str | Path) -> dict[str, object]:
    return verify_content_addressed_runtime(
        artifact_path,
        artifact_manifest_path=_ARTIFACT_MANIFEST_PATH,
        artifact_schema=_ARTIFACT_SCHEMA,
        runtime_contract_path=_RUNTIME_CONTRACT_PATH,
        runtime_schema=_RUNTIME_SCHEMA,
        dependency_root_environment=RUNTIME_DEPENDENCY_ROOT_ENV,
        expected_dependency_loading_policy=PUBLISHER_DEPENDENCY_LOADING_POLICY,
        admission_schema="fmv_lt_snapshot_publisher_runtime_admission.v3",
        capture_parent=None,
        capture_prefix="fmv-pfc-publisher-runtime-",
        required_runtime_keys=None,
        required_runtime_policy=None,
    )


def verify_content_addressed_runtime(
    artifact_path: str | Path,
    *,
    artifact_manifest_path: str,
    artifact_schema: str,
    runtime_contract_path: str,
    runtime_schema: str,
    dependency_root_environment: str,
    expected_dependency_loading_policy: dict[str, object],
    admission_schema: str,
    capture_parent: str | Path | None,
    capture_prefix: str,
    required_runtime_keys: frozenset[str] | None,
    required_runtime_policy: Mapping[str, object] | None,
) -> dict[str, object]:
    """Admit one zipapp before any third-party or business-module import."""

    artifact = Path(artifact_path).expanduser()
    if not artifact.is_absolute() or not artifact.is_file() or not zipfile.is_zipfile(artifact):
        raise PublisherRuntimeAdmissionError("publisher zipapp artifact is unavailable")
    payload = _verified_artifact_member_payload(
        artifact,
        member_name=runtime_contract_path,
        manifest_hash_key="runtime_contract_sha256",
        artifact_manifest_path=artifact_manifest_path,
        artifact_schema=artifact_schema,
    )
    contract = _strict_json_mapping(payload)
    if required_runtime_keys is not None and set(contract) != required_runtime_keys:
        raise PublisherRuntimeAdmissionError(
            "publisher runtime contract field inventory is invalid"
        )
    if required_runtime_policy is not None and any(
        contract.get(key) != value for key, value in required_runtime_policy.items()
    ):
        raise PublisherRuntimeAdmissionError(
            "publisher runtime contract policy is invalid"
        )
    if (
        contract.get("schema_version") != runtime_schema
        or contract.get("artifact") != artifact_schema
    ):
        raise PublisherRuntimeAdmissionError("publisher runtime contract schema is invalid")
    if contract.get("dependency_loading_policy") != expected_dependency_loading_policy:
        raise PublisherRuntimeAdmissionError(
            "publisher dependency loading policy is invalid"
        )
    _verify_isolated_launcher(contract)
    if contract.get("python") != python_runtime_fingerprint():
        raise PublisherRuntimeAdmissionError("publisher Python runtime fingerprint diverges")
    dependency_root = _verify_dependency_closure(
        contract,
        root_environment=dependency_root_environment,
    )
    expected_distributions = _verify_distribution_inventory(contract, dependency_root)
    _assert_dependency_source_not_on_sys_path(dependency_root)
    captured_root = _capture_verified_dependency_closure(
        dependency_root,
        expected_tree={
            "tree_sha256": contract["dependency_closure"]["tree_sha256"],
            "file_count": contract["dependency_closure"]["file_count"],
        },
        expected_distributions=expected_distributions,
        capture_parent=capture_parent,
        capture_prefix=capture_prefix,
    )
    _assert_dependency_source_not_on_sys_path(dependency_root)
    _activate_captured_dependencies(
        artifact=artifact,
        captured_root=captured_root,
        expected_distributions=expected_distributions,
    )
    distributions = contract["locked_distributions"]
    source_dependency_lexical = _lexical_path(dependency_root)
    captured_dependency_lexical = _lexical_path(captured_root)
    captured_root_count = sum(
        _lexical_path(Path(item).expanduser()) == captured_dependency_lexical
        for item in sys.path
    )
    source_root_count = sum(
        _lexical_path(Path(item).expanduser()) == source_dependency_lexical
        for item in sys.path
    )
    return {
        "schema_version": admission_schema,
        "status": "PASS",
        "python": platform.python_version(),
        "distribution_count": len(distributions),
        "dependency_versions": dict(sorted(expected_distributions.items())),
        "dependency_tree_sha256": contract["dependency_closure"]["tree_sha256"],
        "dependency_import_source": "PROCESS_PRIVATE_EXACT_BYTE_COPY",
        "captured_dependency_root_sys_path_count": captured_root_count,
        "source_dependency_root_sys_path_count": source_root_count,
        "isolated_launcher": True,
        "production_authorization": False,
    }


def python_runtime_fingerprint() -> dict[str, object]:
    """Return the exact interpreter identity bound into a release artifact."""

    _assert_supported_python_runtime()
    executable = Path(sys.executable)
    base_executable = Path(getattr(sys, "_base_executable", sys.executable))
    if not executable.is_file() or not base_executable.is_file():
        raise PublisherRuntimeAdmissionError("publisher Python executable is unavailable")
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "cache_tag": sys.implementation.cache_tag,
        "platform": sys.platform,
        "machine": platform.machine(),
        "pointer_bits": 64 if sys.maxsize > 2**32 else 32,
        "executable_sha256": _file_sha256(executable),
        "base_executable_sha256": _file_sha256(base_executable),
        "runtime_binaries": _runtime_binary_fingerprints(),
        "runtime_import_tree": _runtime_import_tree_fingerprint(),
    }


def _activate_captured_dependencies(
    *,
    artifact: Path,
    captured_root: Path,
    expected_distributions: dict[str, str],
) -> None:
    """Put exact captured bytes before stdlib roots and bind loaded origins."""

    artifact_lexical = _lexical_path(artifact)
    positions = [
        index
        for index, item in enumerate(sys.path)
        if _lexical_path(Path(item).expanduser()) == artifact_lexical
    ]
    if len(positions) != 1:
        raise PublisherRuntimeAdmissionError(
            "publisher artifact sys.path identity is ambiguous"
        )
    top_levels = _captured_distribution_top_levels(
        captured_root,
        expected_distributions=expected_distributions,
    )
    sys.path.insert(positions[0] + 1, str(captured_root))
    admitted_sys_path = tuple(sys.path)
    importlib.invalidate_caches()
    for top_level in top_levels:
        try:
            module = importlib.import_module(top_level)
        except Exception as exc:
            raise PublisherRuntimeAdmissionError(
                f"publisher captured dependency import failed: {top_level}"
            ) from exc
        _assert_module_loaded_from_capture(
            module,
            captured_root=captured_root,
            top_level=top_level,
        )
    _assert_sys_path_unchanged_after_dependency_imports(
        admitted_sys_path,
        captured_root=captured_root,
    )


def _assert_sys_path_unchanged_after_dependency_imports(
    admitted_sys_path: tuple[str, ...],
    *,
    captured_root: Path,
) -> None:
    if tuple(sys.path) != admitted_sys_path:
        raise PublisherRuntimeAdmissionError(
            "publisher dependency imports mutated the admitted sys.path"
        )
    captured_lexical = _lexical_path(captured_root)
    captured_entries = [
        item for item in sys.path if _lexical_path(Path(item).expanduser()) == captured_lexical
    ]
    if captured_entries != [str(captured_root)]:
        raise PublisherRuntimeAdmissionError(
            "publisher private dependency capture sys.path identity is ambiguous"
        )


def _captured_distribution_top_levels(
    root: Path,
    *,
    expected_distributions: dict[str, str],
) -> tuple[str, ...]:
    discovered: dict[str, set[str]] = {}
    for distribution in metadata.distributions(path=[str(root)]):
        name = distribution.metadata.get("Name")
        if not name:
            raise PublisherRuntimeAdmissionError(
                "publisher captured distribution has no canonical name"
            )
        normalized = _normalize_distribution_name(name)
        declared = distribution.read_text("top_level.txt") or ""
        top_levels: set[str] = set()
        files = distribution.files or ()
        if files:
            for relative in files:
                parts = tuple(relative.parts)
                if not parts or ".dist-info" in parts[0] or ".data" in parts[0]:
                    continue
                if len(parts) == 1 and parts[0].endswith((".py", ".pyd", ".so")):
                    top_levels.add(Path(parts[0]).stem.split(".", 1)[0])
                elif len(parts) > 1 and parts[1] == "__init__.py":
                    top_levels.add(parts[0])
        else:
            top_levels = {
                line.strip()
                for line in declared.splitlines()
                if line.strip()
            }
        if not top_levels or any(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None
            for value in top_levels
        ):
            raise PublisherRuntimeAdmissionError(
                f"publisher captured distribution top-level inventory is invalid: {normalized}"
            )
        discovered[normalized] = top_levels
    if set(discovered) != set(expected_distributions):
        raise PublisherRuntimeAdmissionError(
            "publisher captured distribution top-level inventory diverges"
        )
    return tuple(sorted(set().union(*discovered.values())))


def _assert_module_loaded_from_capture(
    module: object,
    *,
    captured_root: Path,
    top_level: str,
) -> None:
    locations: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        locations.append(Path(str(module_file)))
    spec = getattr(module, "__spec__", None)
    search_locations = getattr(spec, "submodule_search_locations", None)
    if search_locations is not None:
        locations.extend(Path(str(value)) for value in search_locations)
    if not locations:
        raise PublisherRuntimeAdmissionError(
            f"publisher captured dependency origin is unavailable: {top_level}"
        )
    root_lexical = _lexical_path(captured_root)
    for location in locations:
        location_lexical = _lexical_path(location)
        try:
            common = os.path.commonpath((root_lexical, location_lexical))
        except ValueError as exc:
            raise PublisherRuntimeAdmissionError(
                f"publisher captured dependency origin diverges: {top_level}"
            ) from exc
        if common != root_lexical:
            raise PublisherRuntimeAdmissionError(
                f"publisher captured dependency origin diverges: {top_level}"
            )


def _assert_supported_python_runtime() -> None:
    if (
        platform.python_implementation() != "CPython"
        or (sys.version_info.major, sys.version_info.minor) != (3, 11)
        or sys.implementation.cache_tag != "cpython-311"
    ):
        raise PublisherRuntimeAdmissionError(
            "publisher build and runtime require CPython 3.11 with cache tag cpython-311"
        )


def dependency_tree_inventory(root: Path) -> dict[str, object]:
    """Hash every admitted dependency byte while rejecting executable path hooks."""

    selected, files = _dependency_file_entries(root)
    digest = hashlib.sha256()
    for relative, path in files:
        payload_sha256 = _file_sha256(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload_sha256.encode("ascii"))
        digest.update(b"\0")
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": len(files),
    }


def _dependency_file_entries(root: Path) -> tuple[Path, list[tuple[str, Path]]]:
    selected = root.expanduser()
    if not selected.is_absolute() or not selected.is_dir():
        raise PublisherRuntimeAdmissionError("publisher dependency root is unavailable")
    root_stat = selected.stat(follow_symlinks=False)
    if _is_link_or_reparse(selected, root_stat):
        raise PublisherRuntimeAdmissionError("publisher dependency root is linked")
    files: list[tuple[str, Path]] = []
    pending = [selected]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                entry_path = Path(entry.path)
                entry_stat = entry.stat(follow_symlinks=False)
                if _is_link_or_reparse(entry_path, entry_stat):
                    raise PublisherRuntimeAdmissionError(
                        "publisher dependency closure contains a filesystem link"
                    )
                if stat.S_ISDIR(entry_stat.st_mode):
                    pending.append(entry_path)
                    continue
                if not stat.S_ISREG(entry_stat.st_mode):
                    raise PublisherRuntimeAdmissionError(
                        "publisher dependency closure contains a non-regular file"
                    )
                if int(entry_stat.st_nlink) > 1:
                    raise PublisherRuntimeAdmissionError(
                        "publisher dependency closure contains a hardlinked file: "
                        f"{entry_path} (links={entry_stat.st_nlink})"
                    )
                relative = entry_path.relative_to(selected).as_posix()
                lowered = relative.lower()
                if lowered.endswith(".pth"):
                    raise PublisherRuntimeAdmissionError(
                        "publisher dependency closure contains a forbidden .pth"
                    )
                if (
                    lowered.endswith((".pyc", ".pyo"))
                    or "/__pycache__/" in f"/{lowered}"
                ):
                    raise PublisherRuntimeAdmissionError(
                        "publisher dependency closure contains forbidden bytecode"
                    )
                files.append((relative, entry_path))
    return selected, sorted(files)


def dependency_distribution_inventory(root: Path) -> dict[str, str]:
    """Return the exact normalized distribution inventory in one closure root."""

    installed: dict[str, str] = {}
    for distribution in metadata.distributions(path=[str(root)]):
        name = distribution.metadata.get("Name")
        if not name:
            raise PublisherRuntimeAdmissionError(
                "publisher installed distribution has no canonical name"
            )
        normalized = _normalize_distribution_name(name)
        if normalized in installed:
            raise PublisherRuntimeAdmissionError(
                "publisher installed distribution inventory is ambiguous"
            )
        installed[normalized] = distribution.version
    return installed


def _verify_isolated_launcher(contract: dict[str, Any]) -> None:
    policy = contract.get("launcher_policy")
    if not isinstance(policy, dict) or policy != {
        "required_invocation": "python.exe -I -S -B artifact.pyz",
        "isolated": True,
        "no_site": True,
        "ignore_python_environment": True,
        "safe_path": True,
        "dont_write_bytecode": True,
    }:
        raise PublisherRuntimeAdmissionError("publisher launcher policy is invalid")
    _assert_isolated_launcher_flags()
    artifact = Path(sys.argv[0]).absolute()
    base = Path(sys.base_prefix).resolve()
    _isolate_runtime_sys_path(artifact=artifact, base=base)


def _assert_isolated_launcher_flags() -> None:
    flags = sys.flags
    if not (
        flags.isolated
        and flags.no_site
        and flags.ignore_environment
        and flags.no_user_site
        and flags.safe_path
        and flags.dont_write_bytecode
    ):
        raise PublisherRuntimeAdmissionError(
            "publisher must be invoked with the admitted isolated -I -S -B launcher"
        )


def _isolate_runtime_sys_path(*, artifact: Path, base: Path) -> None:
    allowed = {
        _lexical_path(artifact),
        _lexical_path(base / "python311.zip"),
        _lexical_path(base / "DLLs"),
        _lexical_path(base / "Lib"),
    }
    base_lexical = _lexical_path(base)
    isolated: list[str] = []
    seen: set[str] = set()
    for item in sys.path:
        if not item:
            raise PublisherRuntimeAdmissionError(
                "publisher sys.path contains an empty entry"
            )
        candidate_lexical = _lexical_path(Path(item).expanduser())
        if candidate_lexical == base_lexical:
            continue
        if candidate_lexical not in allowed or candidate_lexical in seen:
            raise PublisherRuntimeAdmissionError("publisher sys.path is not isolated")
        isolated.append(item)
        seen.add(candidate_lexical)
    if _lexical_path(artifact) not in seen:
        raise PublisherRuntimeAdmissionError(
            "publisher artifact is absent from isolated sys.path"
        )
    sys.path[:] = isolated


def _lexical_path(path: Path) -> str:
    try:
        return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))
    except OSError as exc:
        raise PublisherRuntimeAdmissionError(
            "publisher runtime path cannot be normalized"
        ) from exc


def _assert_dependency_source_not_on_sys_path(dependency_root: Path) -> None:
    try:
        source_lexical = os.path.normcase(
            os.path.abspath(os.path.normpath(str(dependency_root)))
        )
        source_exists = dependency_root.exists()
    except OSError as exc:
        raise PublisherRuntimeAdmissionError(
            "publisher dependency source path cannot be compared"
        ) from exc
    if not source_exists:
        raise PublisherRuntimeAdmissionError(
            "publisher dependency source path disappeared"
        )
    for item in sys.path:
        if not item:
            raise PublisherRuntimeAdmissionError(
                "publisher sys.path contains an empty entry"
            )
        candidate = Path(item).expanduser()
        try:
            candidate_lexical = os.path.normcase(
                os.path.abspath(os.path.normpath(str(candidate)))
            )
            same_object = candidate.exists() and os.path.samefile(
                candidate, dependency_root
            )
        except OSError as exc:
            raise PublisherRuntimeAdmissionError(
                "publisher sys.path dependency source identity cannot be compared"
            ) from exc
        if candidate_lexical == source_lexical or same_object:
            raise PublisherRuntimeAdmissionError(
                "publisher verified source dependency root is already importable"
            )


def _verify_dependency_closure(
    contract: dict[str, Any],
    *,
    root_environment: str = RUNTIME_DEPENDENCY_ROOT_ENV,
) -> Path:
    closure = contract.get("dependency_closure")
    required_keys = {
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
    if not isinstance(closure, dict) or set(closure) != required_keys:
        raise PublisherRuntimeAdmissionError("publisher dependency closure is not bound")
    if closure.get("root_environment") != root_environment:
        raise PublisherRuntimeAdmissionError("publisher dependency root contract is invalid")
    if any(
        closure.get(key) != "FORBIDDEN"
        for key in (
            "pth_files",
            "bytecode_files",
            "filesystem_links",
            "extra_files_or_distributions",
        )
    ):
        raise PublisherRuntimeAdmissionError("publisher dependency policy is not fail closed")
    expected_sha256 = closure.get("tree_sha256")
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise PublisherRuntimeAdmissionError("publisher dependency tree digest is invalid")
    if (
        _SHA256.fullmatch(str(closure.get("receipt_sha256", ""))) is None
        or _SHA256.fullmatch(str(closure.get("source_lock_sha256", ""))) is None
        or closure.get("wheel_provenance")
        != "VERIFIED_ARCHIVES_AND_RECORD_AGAINST_UV_LOCK"
        or closure.get("extraction_policy")
        != {
            "pth_files": "FORBIDDEN",
            "recorded_bytecode": "VERIFIED_THEN_EXCLUDED",
            "recorded_nested_wheels": "VERIFIED_THEN_EXCLUDED",
        }
    ):
        raise PublisherRuntimeAdmissionError(
            "publisher dependency wheel provenance is invalid"
        )
    _verify_wheel_provenance(closure.get("wheels"))
    raw_root = os.environ.get(root_environment, "").strip()
    root = Path(raw_root).expanduser()
    inventory = dependency_tree_inventory(root)
    if inventory != {
        "tree_sha256": expected_sha256,
        "file_count": closure.get("file_count"),
    }:
        raise PublisherRuntimeAdmissionError("publisher dependency tree diverges")
    return root


def _verify_wheel_provenance(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise PublisherRuntimeAdmissionError(
            "publisher dependency wheel inventory is unavailable"
        )
    required = {
        "name",
        "version",
        "filename",
        "sha256",
        "size",
        "record_sha256",
        "wheel_tags",
        "installed_file_count",
        "excluded_recorded_member_count",
    }
    names: set[str] = set()
    filenames: set[str] = set()
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != required
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("version"), str)
            or not isinstance(item.get("filename"), str)
            or _SHA256.fullmatch(str(item.get("sha256", ""))) is None
            or _SHA256.fullmatch(str(item.get("record_sha256", ""))) is None
            or not isinstance(item.get("size"), int)
            or item["size"] <= 0
            or not isinstance(item.get("installed_file_count"), int)
            or item["installed_file_count"] <= 0
            or not isinstance(item.get("excluded_recorded_member_count"), int)
            or item["excluded_recorded_member_count"] < 0
            or not isinstance(item.get("wheel_tags"), list)
            or not item["wheel_tags"]
            or any(not isinstance(tag, str) or not tag for tag in item["wheel_tags"])
        ):
            raise PublisherRuntimeAdmissionError(
                "publisher dependency wheel entry is invalid"
            )
        normalized = _normalize_distribution_name(item["name"])
        filename = item["filename"].casefold()
        if normalized in names or filename in filenames:
            raise PublisherRuntimeAdmissionError(
                "publisher dependency wheel inventory is ambiguous"
            )
        names.add(normalized)
        filenames.add(filename)


def _verify_distribution_inventory(
    contract: dict[str, Any],
    root: Path,
) -> dict[str, str]:
    distributions = contract.get("locked_distributions")
    if not isinstance(distributions, list) or not distributions:
        raise PublisherRuntimeAdmissionError(
            "publisher distribution contract is unavailable"
        )
    expected: dict[str, str] = {}
    for item in distributions:
        if not isinstance(item, dict) or set(item) != {"name", "version"}:
            raise PublisherRuntimeAdmissionError(
                "publisher distribution contract entry is invalid"
            )
        name = str(item["name"]).strip()
        version = str(item["version"]).strip()
        normalized = _normalize_distribution_name(name)
        if not name or not version or normalized in expected:
            raise PublisherRuntimeAdmissionError(
                "publisher distribution contract is ambiguous"
            )
        expected[normalized] = version
    installed = dependency_distribution_inventory(root)
    if installed != expected:
        raise PublisherRuntimeAdmissionError(
            f"publisher runtime distributions diverge: installed={installed}, expected={expected}"
        )
    closure = contract["dependency_closure"]
    if closure.get("distribution_count") != len(installed):
        raise PublisherRuntimeAdmissionError(
            "publisher dependency distribution count diverges"
        )
    return expected


def _capture_verified_dependency_closure(
    source_root: Path,
    *,
    expected_tree: dict[str, object],
    expected_distributions: dict[str, str],
    capture_parent: str | Path | None = None,
    capture_prefix: str = "fmv-pfc-publisher-runtime-",
) -> Path:
    global _CAPTURED_DEPENDENCY_RUNTIME

    if _CAPTURED_DEPENDENCY_RUNTIME is not None:
        raise PublisherRuntimeAdmissionError(
            "publisher dependency runtime was already captured"
        )
    if (
        set(expected_tree) != {"tree_sha256", "file_count"}
        or _SHA256.fullmatch(str(expected_tree.get("tree_sha256", ""))) is None
        or not isinstance(expected_tree.get("file_count"), int)
        or expected_tree["file_count"] <= 0
    ):
        raise PublisherRuntimeAdmissionError(
            "publisher expected dependency inventory is invalid"
        )
    selected, entries = _dependency_file_entries(source_root)
    selected_capture_parent = (
        _supervised_capture_root() if capture_parent is None else str(capture_parent)
    )
    if selected_capture_parent is None:
        selected_capture_parent = os.environ.get("TEMP") or os.environ.get("TMP")
    if selected_capture_parent is None:
        raise PublisherRuntimeAdmissionError(
            "publisher dependency capture requires an explicit scratch root"
        )
    selected_capture_parent_path = assert_absolute_path_has_no_links(
        Path(selected_capture_parent)
    )
    if not selected_capture_parent_path.is_dir():
        raise PublisherRuntimeAdmissionError(
            "publisher dependency capture scratch root is unavailable"
        )
    runtime = ProcessRuntimeDirectory(
        prefix=capture_prefix,
        parent=selected_capture_parent_path,
    )
    captured = Path(runtime.name) / "site-packages"
    try:
        _assert_capture_path_headroom(captured, entries)
        captured.mkdir()
        for relative, source in entries:
            destination = captured.joinpath(*relative.split("/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_stable_runtime_file(source, destination)
        source_inventory = dependency_tree_inventory(selected)
        captured_inventory = dependency_tree_inventory(captured)
        if source_inventory != expected_tree:
            raise PublisherRuntimeAdmissionError(
                "publisher dependency source changed during private capture"
            )
        if captured_inventory != expected_tree:
            raise PublisherRuntimeAdmissionError(
                "publisher private dependency capture diverges"
            )
        captured_distributions = dependency_distribution_inventory(captured)
        if captured_distributions != expected_distributions:
            raise PublisherRuntimeAdmissionError(
                "publisher private dependency distributions diverge"
            )
        _make_capture_read_only(captured)
        if dependency_tree_inventory(captured) != expected_tree:
            raise PublisherRuntimeAdmissionError(
                "publisher private dependency capture changed while sealing"
            )
    except BaseException:
        runtime.cleanup()
        raise
    _CAPTURED_DEPENDENCY_RUNTIME = runtime
    return captured


def _assert_capture_path_headroom(
    captured_root: Path,
    entries: list[tuple[str, Path]],
) -> None:
    if os.name != "nt":
        return
    longest = max(
        len(str(captured_root.joinpath(*relative.split("/"))))
        for relative, _ in entries
    )
    if longest > _MAX_WINDOWS_CAPTURE_PATH_CHARS:
        raise PublisherRuntimeAdmissionError(
            "publisher governed scratch path has insufficient Windows path headroom"
        )


def _copy_stable_runtime_file(source: Path, destination: Path) -> None:
    try:
        before = source.stat(follow_symlinks=False)
    except OSError as exc:
        raise PublisherRuntimeAdmissionError(
            "publisher dependency file cannot be inspected for capture"
        ) from exc
    if _is_link_or_reparse(source, before) or not stat.S_ISREG(before.st_mode):
        raise PublisherRuntimeAdmissionError(
            "publisher dependency capture source is linked or non-regular"
        )
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            opened_before = os.fstat(reader.fileno())
            if _file_identity(opened_before) != _file_identity(before):
                raise PublisherRuntimeAdmissionError(
                    "publisher dependency file changed before capture"
                )
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(chunk)
            opened_after = os.fstat(reader.fileno())
        after = source.stat(follow_symlinks=False)
    except PublisherRuntimeAdmissionError:
        raise
    except OSError as exc:
        raise PublisherRuntimeAdmissionError(
            "publisher dependency file capture failed"
        ) from exc
    if not (
        _file_identity(before)
        == _file_identity(opened_before)
        == _file_identity(opened_after)
        == _file_identity(after)
    ):
        raise PublisherRuntimeAdmissionError(
            "publisher dependency file changed during capture"
        )


def _make_capture_read_only(root: Path) -> None:
    _, entries = _dependency_file_entries(root)
    for _, path in entries:
        path.chmod(stat.S_IREAD)
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        directory.chmod(stat.S_IREAD | stat.S_IEXEC)
    root.chmod(stat.S_IREAD | stat.S_IEXEC)


def run_admitted_publisher_zipapp() -> int:
    if os.environ.get(_SUPERVISED_WORKER_ENV) == "1":
        admission_started = time.monotonic()
        try:
            supervisor_root, deferred_environment_expected = (
                _validated_supervisor_worker_context()
            )
        except (PublisherRuntimeAdmissionError, OSError) as exc:
            return _runtime_admission_failure(exc)
        return _run_admitted_publisher_worker(
            supervisor_root=supervisor_root,
            admission_started=admission_started,
            deferred_environment_expected=deferred_environment_expected,
        )
    if any(
        name in os.environ
        for name in (
            _SUPERVISED_WORKER_ENV,
            _SUPERVISED_CAPTURE_ROOT_ENV,
            _SUPERVISED_WORKER_TOKEN_ENV,
        )
    ):
        return _runtime_admission_failure(
            PublisherRuntimeAdmissionError(
                "publisher internal capture environment is reserved"
            )
        )
    return _run_supervised_publisher_worker()


def _run_supervised_publisher_worker() -> int:
    try:
        _assert_isolated_launcher_flags()
    except PublisherRuntimeAdmissionError as exc:
        return _runtime_admission_failure(exc)
    artifact: Path | None = None
    supervisor_root: Path | None = None
    worker: subprocess.Popen[bytes] | None = None
    worker_exit_code: int | None = None
    windows_job: int | None = None
    cleanup_error: OSError | None = None
    runtime_error: BaseException | None = None
    worker_timeout_seconds: float | None = None
    admission_slo_seconds: float | None = None
    admission_metric: dict[str, object] | None = None
    worker_token: str | None = None
    try:
        scratch_parent, worker_timeout_seconds, admission_slo_seconds = (
            _validated_runtime_supervision_config()
        )
        artifact = _publisher_artifact_path()
        phase = _publisher_command_phase(sys.argv[1:])
        supervisor_root = _allocate_inherited_acl_runtime_directory(
            prefix=_SUPERVISOR_PREFIX,
            parent=scratch_parent,
        )
        environment_contract = _verified_environment_contract(artifact)
        worker_token = secrets.token_hex(32)
        worker_environment, deferred_environment = _supervised_worker_environment(
            environment_contract,
            phase=phase,
            supervisor_root=supervisor_root,
            worker_token=worker_token,
        )
        capability = {
            "artifact": str(artifact),
            "deferred_environment_expected": bool(deferred_environment),
            "parent_pid": os.getpid(),
            "supervisor_root": str(supervisor_root),
            "token": worker_token,
        }
        capability_path = supervisor_root / _SUPERVISOR_CAPABILITY
        capability_descriptor = os.open(
            capability_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        with os.fdopen(capability_descriptor, "wb") as handle:
            handle.write(
                json.dumps(capability, sort_keys=True, separators=(",", ":")).encode(
                    "ascii"
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        command = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(artifact),
            *sys.argv[1:],
        ]
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        if os.name == "nt":
            creationflags |= _WINDOWS_CREATE_SUSPENDED
        worker = subprocess.Popen(
            command,
            env=worker_environment,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
        if os.name == "nt":
            windows_job = _create_windows_kill_on_close_job()
            _assign_process_to_windows_job(windows_job, worker)
            _resume_windows_process(worker)
        worker_started = time.monotonic()
        if deferred_environment:
            admission_metric = _wait_for_admission_then_deliver_environment(
                worker=worker,
                supervisor_root=supervisor_root,
                deferred_environment=deferred_environment,
                expected_token=worker_token,
                expected_deadline_seconds=worker_timeout_seconds,
                expected_slo_seconds=admission_slo_seconds,
                timeout_seconds=worker_timeout_seconds,
            )
            elapsed = time.monotonic() - worker_started
            remaining_timeout = max(0.001, worker_timeout_seconds - elapsed)
        else:
            remaining_timeout = worker_timeout_seconds
        worker_exit_code = worker.wait(timeout=remaining_timeout)
        if worker_exit_code == 0 and admission_metric is None:
            admission_metric = _load_supervised_admission_metric(
                supervisor_root,
                expected_token=worker_token,
                expected_deadline_seconds=worker_timeout_seconds,
                expected_slo_seconds=admission_slo_seconds,
                expected_worker_pid=worker.pid,
            )
    except KeyboardInterrupt:
        if worker is not None:
            _terminate_supervised_worker(worker)
        raise
    except PublisherPostAdmissionEnvironmentDeliveryError as exc:
        admission_metric = exc.admission_metric
        runtime_error = exc
        if worker is not None and worker.poll() is None:
            try:
                _terminate_supervised_worker(worker)
            except (OSError, subprocess.SubprocessError) as termination_exc:
                runtime_error = termination_exc
    except (OSError, PublisherRuntimeAdmissionError) as exc:
        runtime_error = exc
        if worker is not None and worker.poll() is None:
            try:
                _terminate_supervised_worker(worker)
            except (OSError, subprocess.SubprocessError) as termination_exc:
                runtime_error = termination_exc
    except subprocess.TimeoutExpired:
        runtime_error = PublisherRuntimeAdmissionError(
            "publisher worker exceeded the governed runtime deadline"
        )
        if worker is not None and worker.poll() is None:
            try:
                _terminate_supervised_worker(worker)
                worker_exit_code = worker.poll()
            except (OSError, subprocess.SubprocessError) as termination_exc:
                runtime_error = termination_exc
    finally:
        if windows_job is not None:
            try:
                _close_windows_handle(windows_job)
            except OSError as exc:
                cleanup_error = exc
        if supervisor_root is not None:
            capture_cleanup_error = _cleanup_supervisor_root(supervisor_root)
            if capture_cleanup_error is not None:
                cleanup_error = capture_cleanup_error
    if runtime_error is not None:
        if admission_metric is not None:
            _emit_admission_metric(admission_metric, stream=sys.stdout)
        if cleanup_error is not None:
            return _runtime_admission_failure(
                PublisherRuntimeAdmissionError(
                    "publisher supervisor failed and capture cleanup failed"
                )
            )
        return _runtime_admission_failure(runtime_error)
    if cleanup_error is not None:
        failure = PublisherRuntimeAdmissionError(
            "publisher supervised capture cleanup failed after worker exit"
        )
        if worker_exit_code is None or worker_exit_code == 0:
            if admission_metric is not None:
                _emit_admission_metric(admission_metric, stream=sys.stdout)
            return _runtime_admission_failure(failure)
    if worker_exit_code is None:
        if admission_metric is not None:
            _emit_admission_metric(admission_metric, stream=sys.stdout)
        return _runtime_admission_failure(
            PublisherRuntimeAdmissionError("publisher worker did not start")
        )
    if worker_exit_code != 0:
        # Preserve the worker's sole machine-readable failure document on
        # stderr while retaining the independently sealed admission metric on
        # the structured stdout stream.
        # A pre-admission worker failure has no sealed metric by definition;
        # preserve its sole failure document and fail-closed exit. Every
        # admitted business phase has deferred authority and therefore returns
        # its metric before business execution begins.
        if admission_metric is not None:
            _emit_admission_metric(admission_metric, stream=sys.stdout)
        return worker_exit_code
    if admission_metric is None:
        return _runtime_admission_failure(
            PublisherRuntimeAdmissionError(
                "publisher runtime admission metric was not retained"
            )
        )
    _emit_admission_metric(admission_metric, stream=sys.stderr)
    return worker_exit_code


def _emit_admission_metric(
    admission_metric: dict[str, object],
    *,
    stream: object,
) -> None:
    print(json.dumps(admission_metric, sort_keys=True), file=stream)


def _cleanup_supervisor_root(supervisor_root: Path) -> OSError | None:
    cleanup_error: OSError | None = None
    for attempt in range(3):
        try:
            shutil.rmtree(supervisor_root)
            return None
        except FileNotFoundError:
            return None
        except OSError as exc:
            cleanup_error = exc
            if attempt < 2:
                time.sleep(0.1 * (attempt + 1))
    return cleanup_error


def _publisher_command_phase(argv: list[str]) -> str | None:
    if argv and argv[0] in _PHASE_DEFERRED_ENVIRONMENT:
        return argv[0]
    return None


def _verified_environment_contract(artifact: Path) -> dict[str, Any]:
    payload = _verified_artifact_member_payload(
        artifact,
        member_name=_ENVIRONMENT_CONTRACT_PATH,
        manifest_hash_key="environment_contract_sha256",
    )
    contract = _strict_json_mapping(payload)
    return _validated_environment_contract_mapping(contract)


def _validated_environment_contract_mapping(
    contract: dict[str, Any],
) -> dict[str, Any]:
    required_keys = {
        "schema_version",
        "artifact",
        "host_passthrough",
        "common_required",
        "common_optional_keyrings",
        "phases",
        "path_authority",
        "runtime_supervision",
        "worker_runtime_overrides",
        "forbidden_in_all_phases",
        "forbidden_outside_prepare",
        "production_authorization",
    }
    if (
        set(contract) != required_keys
        or contract.get("schema_version") != "fmv_lt_snapshot_publisher_environment.v1"
        or contract.get("artifact") != _ARTIFACT_SCHEMA
        or contract.get("production_authorization") is not False
        or set(_strict_environment_name_list(contract.get("host_passthrough")))
        != _SAFE_HOST_ENVIRONMENT_NAMES
        or contract.get("worker_runtime_overrides")
        != {
            "TEMP": "SUPERVISOR_ROOT",
            "TMP": "SUPERVISOR_ROOT",
        }
    ):
        raise PublisherRuntimeAdmissionError(
            "publisher environment contract is invalid"
        )
    common_required = _strict_environment_name_list(contract.get("common_required"))
    common_optional = _strict_environment_name_list(
        contract.get("common_optional_keyrings")
    )
    forbidden_all = set(
        _strict_environment_name_list(contract.get("forbidden_in_all_phases"))
    )
    forbidden_outside_prepare = set(
        _strict_environment_name_list(contract.get("forbidden_outside_prepare"))
    )
    if (
        not _HARD_FORBIDDEN_ENVIRONMENT.issubset(forbidden_all)
        or forbidden_outside_prepare
        != {"PFC_DATA_PUBLICATION_REQUEST_SIGNING_PRIVATE_KEY_PATH"}
    ):
        raise PublisherRuntimeAdmissionError(
            "publisher environment exclusion contract is invalid"
        )
    phases = contract.get("phases")
    if not isinstance(phases, dict) or set(phases) != set(
        _PHASE_DEFERRED_ENVIRONMENT
    ):
        raise PublisherRuntimeAdmissionError(
            "publisher environment phase contract is invalid"
        )
    for phase, value in phases.items():
        if not isinstance(value, dict) or set(value) not in (
            {"required"},
            {"required", "conditional"},
        ):
            raise PublisherRuntimeAdmissionError(
                "publisher environment phase contract is invalid"
            )
        _strict_environment_name_list(value.get("required"))
        conditional = value.get("conditional", {})
        if not isinstance(conditional, dict):
            raise PublisherRuntimeAdmissionError(
                "publisher environment conditional contract is invalid"
            )
        for names in conditional.values():
            _strict_environment_name_list(names)
    if set(common_required) & set(common_optional):
        raise PublisherRuntimeAdmissionError(
            "publisher common environment contract is ambiguous"
        )
    return contract


def _strict_environment_name_list(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or any(
            not isinstance(name, str)
            or re.fullmatch(r"[A-Z_][A-Z0-9_]*", name) is None
            for name in value
        )
        or len(value) != len(set(value))
    ):
        raise PublisherRuntimeAdmissionError(
            "publisher environment name inventory is invalid"
        )
    return tuple(value)


def _supervised_worker_environment(
    contract: dict[str, Any],
    *,
    phase: str | None,
    supervisor_root: Path,
    worker_token: str,
) -> tuple[dict[str, str], dict[str, str]]:
    common = {
        *_strict_environment_name_list(contract["common_required"]),
        *_strict_environment_name_list(contract["common_optional_keyrings"]),
    }
    phase_names: set[str] = set()
    if phase is not None:
        phase_contract = contract["phases"][phase]
        phase_names.update(_strict_environment_name_list(phase_contract["required"]))
        for names in phase_contract.get("conditional", {}).values():
            phase_names.update(_strict_environment_name_list(names))
    permitted_application = common | phase_names
    forbidden_all = _HARD_FORBIDDEN_ENVIRONMENT | set(
        _strict_environment_name_list(contract["forbidden_in_all_phases"])
    )
    present_environment_names = {name.upper() for name in os.environ}
    present_forbidden = sorted(forbidden_all & present_environment_names)
    if present_forbidden:
        raise PublisherRuntimeAdmissionError(
            f"publisher environment contains forbidden authority: {present_forbidden}"
        )
    if phase != "prepare":
        forbidden_outside = set(
            _strict_environment_name_list(contract["forbidden_outside_prepare"])
        )
        present_outside = sorted(forbidden_outside & present_environment_names)
        if present_outside:
            raise PublisherRuntimeAdmissionError(
                "publisher environment contains a phase-forbidden private authority"
            )

    permitted_deferred = _PHASE_DEFERRED_ENVIRONMENT.get(phase or "", frozenset())
    worker_environment: dict[str, str] = {}
    deferred_environment: dict[str, str] = {}
    for name, value in os.environ.items():
        normalized_name = name.upper()
        sensitive = any(
            marker in normalized_name for marker in _SENSITIVE_ENVIRONMENT_MARKERS
        )
        if (
            sensitive
            and (
                normalized_name.startswith("PFC_")
                or normalized_name.startswith("FMV_")
            )
            and normalized_name not in permitted_deferred
        ):
            raise PublisherRuntimeAdmissionError(
                "publisher environment contains an uncontracted sensitive authority: "
                f"{normalized_name}"
            )
        if normalized_name in permitted_deferred:
            if value:
                deferred_environment[normalized_name] = value
        elif (
            normalized_name in permitted_application
            or normalized_name in _SAFE_HOST_ENVIRONMENT_NAMES
        ):
            worker_environment[normalized_name] = value
    worker_environment[_SUPERVISED_WORKER_ENV] = "1"
    worker_environment[_SUPERVISED_CAPTURE_ROOT_ENV] = str(supervisor_root)
    worker_environment[_SUPERVISED_WORKER_TOKEN_ENV] = worker_token
    worker_environment["TEMP"] = str(supervisor_root)
    worker_environment["TMP"] = str(supervisor_root)
    return worker_environment, deferred_environment


def _validated_runtime_supervision_config() -> tuple[Path, float, float]:
    raw_scratch = os.environ.get(SUPERVISOR_SCRATCH_ROOT_ENV, "").strip()
    if not raw_scratch:
        raise PublisherRuntimeAdmissionError(
            "publisher supervisor scratch root is required"
        )
    try:
        scratch = assert_absolute_path_has_no_links(Path(raw_scratch).expanduser())
    except ValueError as exc:
        raise PublisherRuntimeAdmissionError(
            "publisher supervisor scratch root is invalid"
        ) from exc
    if not scratch.is_dir() or not os.access(scratch, os.W_OK | os.X_OK):
        raise PublisherRuntimeAdmissionError(
            "publisher supervisor scratch root is unavailable"
        )
    try:
        free_bytes = shutil.disk_usage(scratch).free
    except OSError as exc:
        raise PublisherRuntimeAdmissionError(
            "publisher supervisor scratch capacity is unavailable"
        ) from exc
    if free_bytes < _MIN_SUPERVISOR_SCRATCH_FREE_BYTES:
        raise PublisherRuntimeAdmissionError(
            "publisher supervisor scratch capacity is below the fail-closed minimum"
        )

    timeout = _positive_runtime_seconds(
        SUPERVISOR_WORKER_TIMEOUT_ENV,
        maximum=3600.0,
    )
    slo = _positive_runtime_seconds(
        SUPERVISOR_ADMISSION_SLO_ENV,
        maximum=timeout,
    )
    return scratch, timeout, slo


def _positive_runtime_seconds(name: str, *, maximum: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise PublisherRuntimeAdmissionError(f"{name} is invalid") from exc
    if not raw or not math.isfinite(value) or value <= 0.0 or value > maximum:
        raise PublisherRuntimeAdmissionError(f"{name} is invalid")
    return value


def _publisher_artifact_path() -> Path:
    return Path(sys.argv[0]).absolute()


def _supervised_capture_root() -> str | None:
    value = os.environ.get(_SUPERVISED_CAPTURE_ROOT_ENV)
    if value is None:
        return None
    selected = Path(value).expanduser()
    try:
        selected_stat = selected.stat(follow_symlinks=False)
    except OSError as exc:
        raise PublisherRuntimeAdmissionError(
            "publisher supervised capture root is unavailable"
        ) from exc
    if (
        not selected.is_absolute()
        or not selected.is_dir()
        or not selected.name.startswith(_SUPERVISOR_PREFIX)
        or _is_link_or_reparse(selected, selected_stat)
    ):
        raise PublisherRuntimeAdmissionError(
            "publisher supervised capture root is invalid"
        )
    scratch_parent, _, _ = _validated_runtime_supervision_config()
    try:
        governed_parent = os.path.samefile(selected.parent, scratch_parent)
    except OSError as exc:
        raise PublisherRuntimeAdmissionError(
            "publisher supervised capture parent cannot be verified"
        ) from exc
    if not governed_parent:
        raise PublisherRuntimeAdmissionError(
            "publisher supervised capture root is outside governed scratch"
        )
    return str(selected)


def _validated_supervisor_worker_context() -> tuple[Path, bool]:
    selected_value = os.environ.get(_SUPERVISED_CAPTURE_ROOT_ENV)
    token = os.environ.get(_SUPERVISED_WORKER_TOKEN_ENV, "")
    if selected_value is None or re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise PublisherRuntimeAdmissionError(
            "publisher supervised worker capability is missing"
        )
    selected = Path(_supervised_capture_root() or "")
    capability_path = selected / _SUPERVISOR_CAPABILITY
    try:
        before = capability_path.stat(follow_symlinks=False)
        if (
            _is_link_or_reparse(capability_path, before)
            or not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
        ):
            raise PublisherRuntimeAdmissionError(
                "publisher supervisor capability is not a mono-linked file"
            )
        with capability_path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            payload = handle.read()
            opened_after = os.fstat(handle.fileno())
        after = capability_path.stat(follow_symlinks=False)
    except PublisherRuntimeAdmissionError:
        raise
    except OSError as exc:
        raise PublisherRuntimeAdmissionError(
            "publisher supervisor capability is unavailable"
        ) from exc
    if not (
        _file_identity(before)
        == _file_identity(opened_before)
        == _file_identity(opened_after)
        == _file_identity(after)
    ):
        raise PublisherRuntimeAdmissionError(
            "publisher supervisor capability changed during capture"
        )
    capability = _strict_json_mapping(payload)
    expected = {
        "artifact": str(Path(sys.argv[0]).absolute()),
        "parent_pid": os.getppid(),
        "supervisor_root": str(selected),
        "token": token,
    }
    if (
        set(capability) != {*expected, "deferred_environment_expected"}
        or not isinstance(capability.get("deferred_environment_expected"), bool)
        or any(
        capability.get(key) != value for key, value in expected.items()
        )
    ):
        raise PublisherRuntimeAdmissionError(
            "publisher supervised worker capability is invalid"
        )
    deferred_environment_expected = bool(capability["deferred_environment_expected"])
    try:
        capability_path.unlink()
    except OSError as exc:
        raise PublisherRuntimeAdmissionError(
            "publisher supervisor capability cannot be consumed"
        ) from exc
    return selected, deferred_environment_expected


def _validated_deferred_worker_environment(
    value: object,
    *,
    phase: str | None,
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise PublisherRuntimeAdmissionError(
            "publisher deferred environment capability is invalid"
        )
    permitted = _PHASE_DEFERRED_ENVIRONMENT.get(phase or "", frozenset())
    if any(
        not isinstance(name, str)
        or name not in permitted
        or not isinstance(selected, str)
        or not selected
        for name, selected in value.items()
    ):
        raise PublisherRuntimeAdmissionError(
            "publisher deferred environment capability is invalid"
        )
    return dict(value)


def _wait_for_admission_then_deliver_environment(
    *,
    worker: subprocess.Popen[bytes],
    supervisor_root: Path,
    deferred_environment: dict[str, str],
    expected_token: str,
    expected_deadline_seconds: float,
    expected_slo_seconds: float,
    timeout_seconds: float,
) -> dict[str, object] | None:
    deadline = time.monotonic() + timeout_seconds
    metric_path = supervisor_root / _SUPERVISOR_ADMISSION_METRIC
    while time.monotonic() < deadline:
        if metric_path.exists():
            metric = _load_supervised_admission_metric(
                supervisor_root,
                expected_token=expected_token,
                expected_deadline_seconds=expected_deadline_seconds,
                expected_slo_seconds=expected_slo_seconds,
                expected_worker_pid=worker.pid,
            )
            try:
                _write_post_admission_environment_capability(
                    supervisor_root,
                    deferred_environment=deferred_environment,
                    token=expected_token,
                    worker_pid=worker.pid,
                )
            except (OSError, PublisherRuntimeAdmissionError) as exc:
                raise PublisherPostAdmissionEnvironmentDeliveryError(
                    "publisher post-admission authority delivery failed",
                    admission_metric=metric,
                ) from exc
            return metric
        if worker.poll() is not None:
            return None
        time.sleep(0.01)
    raise subprocess.TimeoutExpired("publisher-worker-admission", timeout_seconds)


def _write_post_admission_environment_capability(
    supervisor_root: Path,
    *,
    deferred_environment: dict[str, str],
    token: str,
    worker_pid: int,
) -> None:
    payload = {
        "schema_version": "fmv_lt_snapshot_publisher_post_admission_environment.v1",
        "deferred_environment": deferred_environment,
        "token": token,
        "worker_pid": worker_pid,
    }
    _write_atomic_exclusive_json(
        supervisor_root / _SUPERVISOR_DEFERRED_CAPABILITY,
        payload,
    )


def _write_atomic_exclusive_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_after_atomic_link_publication(
    path: Path,
    *,
    label: str,
    max_bytes: int,
    retry_seconds: float,
) -> bytes:
    deadline = time.monotonic() + max(0.0, retry_seconds)
    while True:
        try:
            return read_stable_single_link_file(
                path,
                label=label,
                max_bytes=max_bytes,
            )
        except ValueError:
            try:
                link_count = int(path.stat(follow_symlinks=False).st_nlink)
            except OSError:
                link_count = 0
            if link_count == 1:
                # The first descriptor may have observed the intentional
                # two-link publication window while the following stat
                # already observes the temporary name removed. Re-read once
                # in that stable state; any genuine validation failure still
                # propagates immediately.
                return read_stable_single_link_file(
                    path,
                    label=label,
                    max_bytes=max_bytes,
                )
            if link_count == 0 or time.monotonic() >= deadline:
                raise
            time.sleep(0.001)


def _consume_post_admission_environment(supervisor_root: Path) -> dict[str, str]:
    _, timeout_seconds, _ = _validated_runtime_supervision_config()
    path = supervisor_root / _SUPERVISOR_DEFERRED_CAPABILITY
    deadline = time.monotonic() + timeout_seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise PublisherRuntimeAdmissionError(
                "publisher post-admission environment capability timed out"
            )
        time.sleep(0.01)
    try:
        payload = _read_after_atomic_link_publication(
            path,
            label="publisher post-admission environment capability",
            max_bytes=16384,
            retry_seconds=max(0.0, deadline - time.monotonic()),
        )
        capability = _strict_json_mapping(payload)
        expected_keys = {
            "schema_version",
            "deferred_environment",
            "token",
            "worker_pid",
        }
        token = os.environ.get(_SUPERVISED_WORKER_TOKEN_ENV, "")
        if (
            set(capability) != expected_keys
            or capability.get("schema_version")
            != "fmv_lt_snapshot_publisher_post_admission_environment.v1"
            or capability.get("worker_pid") != os.getpid()
            or not secrets.compare_digest(str(capability.get("token", "")), token)
        ):
            raise PublisherRuntimeAdmissionError(
                "publisher post-admission environment capability is invalid"
            )
        environment = _validated_deferred_worker_environment(
            capability.get("deferred_environment"),
            phase=_publisher_command_phase(sys.argv[1:]),
        )
        path.unlink()
        return environment
    except PublisherRuntimeAdmissionError:
        raise
    except (OSError, ValueError) as exc:
        raise PublisherRuntimeAdmissionError(
            "publisher post-admission environment capability is unavailable"
        ) from exc


def _terminate_supervised_worker(worker: subprocess.Popen[bytes]) -> None:
    if worker.poll() is not None:
        return
    if os.name == "nt":
        worker.terminate()
    else:
        os.killpg(worker.pid, signal.SIGTERM)
    try:
        worker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            worker.kill()
        else:
            os.killpg(worker.pid, signal.SIGKILL)
        worker.wait(timeout=5)


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
    job_handle: int, worker: subprocess.Popen[bytes]
) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    process_handle = getattr(worker, "_handle", None)
    if process_handle is None or not kernel32.AssignProcessToJobObject(
        wintypes.HANDLE(job_handle), wintypes.HANDLE(int(process_handle))
    ):
        error = ctypes.get_last_error()
        _terminate_supervised_worker(worker)
        raise OSError(error, "AssignProcessToJobObject failed")


def _resume_windows_process(worker: subprocess.Popen[bytes]) -> None:
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
        _terminate_supervised_worker(worker)
        raise OSError(ctypes.get_last_error(), "publisher thread snapshot failed")
    thread_ids: list[int] = []
    try:
        entry = ThreadEntry32()
        entry.size = ctypes.sizeof(entry)
        available = kernel32.Thread32First(snapshot, ctypes.byref(entry))
        while available:
            if int(entry.owner_process_id) == worker.pid:
                thread_ids.append(int(entry.thread_id))
            available = kernel32.Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        _close_windows_handle(int(snapshot))
    if len(thread_ids) != 1:
        _terminate_supervised_worker(worker)
        raise OSError("publisher suspended worker primary thread is ambiguous")
    thread_handle = kernel32.OpenThread(0x0002, False, thread_ids[0])
    if not thread_handle:
        _terminate_supervised_worker(worker)
        raise OSError(ctypes.get_last_error(), "publisher primary thread open failed")
    try:
        previous_suspend_count = kernel32.ResumeThread(thread_handle)
    finally:
        _close_windows_handle(int(thread_handle))
    if previous_suspend_count != 1:
        error = ctypes.get_last_error()
        _terminate_supervised_worker(worker)
        raise OSError(
            error,
            "publisher worker did not resume from the expected suspended state",
        )


def _close_windows_handle(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    if not kernel32.CloseHandle(wintypes.HANDLE(handle)):
        raise OSError(ctypes.get_last_error(), "CloseHandle failed")


def _run_admitted_publisher_worker(
    *,
    supervisor_root: Path,
    admission_started: float,
    deferred_environment_expected: bool = False,
) -> int:
    try:
        verify_publisher_runtime(Path(sys.argv[0]).absolute())
        _write_supervised_admission_metric(
            supervisor_root,
            admission_duration_seconds=time.monotonic() - admission_started,
        )
    except (PublisherRuntimeAdmissionError, OSError) as exc:
        return _runtime_admission_failure(exc)
    try:
        deferred_environment = (
            _consume_post_admission_environment(supervisor_root)
            if deferred_environment_expected
            else {}
        )
        _activate_deferred_worker_environment(deferred_environment)
    except PublisherRuntimeAdmissionError as exc:
        return _runtime_admission_failure(exc)
    from scripts.publish_governed_lt_input_snapshot import main

    return main()


def _activate_deferred_worker_environment(deferred_environment: dict[str, str]) -> None:
    collisions = sorted(
        set(deferred_environment) & {name.upper() for name in os.environ}
    )
    if collisions:
        raise PublisherRuntimeAdmissionError(
            "publisher deferred environment collides with the admitted worker environment"
        )
    os.environ.update(deferred_environment)


def _write_supervised_admission_metric(
    supervisor_root: Path,
    *,
    admission_duration_seconds: float,
) -> None:
    token = os.environ.get(_SUPERVISED_WORKER_TOKEN_ENV, "")
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise PublisherRuntimeAdmissionError(
            "publisher admission metric capability is missing"
        )
    _, deadline_seconds, slo_seconds = _validated_runtime_supervision_config()
    if not math.isfinite(admission_duration_seconds) or admission_duration_seconds < 0.0:
        raise PublisherRuntimeAdmissionError(
            "publisher admission duration is invalid"
        )
    serialized_duration = round(admission_duration_seconds, 6)
    slo_met = serialized_duration <= slo_seconds
    payload = {
        "schema_version": "fmv_lt_snapshot_publisher_admission_metric.v1",
        "metric": "admission_duration_seconds",
        "value": serialized_duration,
        "slo_seconds": slo_seconds,
        "deadline_seconds": deadline_seconds,
        "slo_met": slo_met,
        "status": "PASS" if slo_met else "SLO_BREACH",
        "worker_pid": os.getpid(),
        "production_authorization": False,
        "token": token,
    }
    try:
        _write_atomic_exclusive_json(
            supervisor_root / _SUPERVISOR_ADMISSION_METRIC,
            payload,
        )
    except OSError as exc:
        raise PublisherRuntimeAdmissionError(
            "publisher admission metric cannot be sealed"
        ) from exc


def _load_supervised_admission_metric(
    supervisor_root: Path,
    *,
    expected_token: str,
    expected_deadline_seconds: float,
    expected_slo_seconds: float,
    expected_worker_pid: int,
) -> dict[str, object]:
    try:
        payload = _read_after_atomic_link_publication(
            supervisor_root / _SUPERVISOR_ADMISSION_METRIC,
            label="publisher runtime admission metric",
            max_bytes=4096,
            retry_seconds=min(1.0, expected_deadline_seconds),
        )
    except ValueError as exc:
        raise PublisherRuntimeAdmissionError(
            "publisher runtime admission metric is unavailable"
        ) from exc
    metric = _strict_json_mapping(payload)
    expected_keys = {
        "schema_version",
        "metric",
        "value",
        "slo_seconds",
        "deadline_seconds",
        "slo_met",
        "status",
        "worker_pid",
        "production_authorization",
        "token",
    }
    value = metric.get("value")
    if (
        set(metric) != expected_keys
        or metric.get("schema_version")
        != "fmv_lt_snapshot_publisher_admission_metric.v1"
        or metric.get("metric") != "admission_duration_seconds"
        or not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
        or metric.get("slo_seconds") != expected_slo_seconds
        or metric.get("deadline_seconds") != expected_deadline_seconds
        or not isinstance(metric.get("slo_met"), bool)
        or metric.get("slo_met") != (float(value) <= expected_slo_seconds)
        or metric.get("status")
        != ("PASS" if float(value) <= expected_slo_seconds else "SLO_BREACH")
        or not isinstance(metric.get("worker_pid"), int)
        or int(metric["worker_pid"]) != expected_worker_pid
        or metric.get("production_authorization") is not False
        or not secrets.compare_digest(str(metric.get("token", "")), expected_token)
    ):
        raise PublisherRuntimeAdmissionError(
            "publisher runtime admission metric is invalid"
        )
    return {key: value for key, value in metric.items() if key != "token"}


def _runtime_admission_failure(exc: BaseException) -> int:
    print(
        json.dumps(
            {
                "schema_version": "lt_snapshot_publication_cli_result.v1",
                "status": "RUNTIME_ADMISSION_FAILED",
                "command": "runtime-admission",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "production_authorization": False,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return EXIT_INTEGRITY_FAILURE


def _verified_runtime_contract_payload(artifact: Path) -> bytes:
    return _verified_artifact_member_payload(
        artifact,
        member_name=_RUNTIME_CONTRACT_PATH,
        manifest_hash_key="runtime_contract_sha256",
    )


def _verified_artifact_member_payload(
    artifact: Path,
    *,
    member_name: str,
    manifest_hash_key: str,
    artifact_manifest_path: str = _ARTIFACT_MANIFEST_PATH,
    artifact_schema: str = _ARTIFACT_SCHEMA,
) -> bytes:
    try:
        with zipfile.ZipFile(artifact) as archive:
            names = archive.namelist()
            members = archive.infolist()
            if any(
                member.file_size > _MAX_ARTIFACT_MEMBER_UNCOMPRESSED_BYTES
                for member in members
            ) or sum(
                member.file_size for member in members
            ) > _MAX_ARTIFACT_TOTAL_UNCOMPRESSED_BYTES:
                raise PublisherRuntimeAdmissionError(
                    "governed runtime artifact exceeds its decompression budget"
                )
            if len(names) != len(set(names)) or artifact_manifest_path not in names:
                raise PublisherRuntimeAdmissionError(
                    "publisher artifact member inventory is ambiguous"
                )
            manifest_payload = archive.read(artifact_manifest_path)
            manifest = _strict_json_mapping(manifest_payload)
            if (
                manifest.get("schema_version") != artifact_schema
                or json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(
                    "ascii"
                )
                != manifest_payload
            ):
                raise PublisherRuntimeAdmissionError(
                    "publisher artifact manifest is invalid"
                )
            files = manifest.get("files")
            if not isinstance(files, dict) or set(names) != {
                *files,
                artifact_manifest_path,
            }:
                raise PublisherRuntimeAdmissionError(
                    "publisher artifact positive inventory diverges"
                )
            payloads: dict[str, bytes] = {}
            for name, expected_sha256 in files.items():
                if not isinstance(name, str) or not isinstance(expected_sha256, str):
                    raise PublisherRuntimeAdmissionError(
                        "publisher artifact file binding is invalid"
                    )
                member_payload = archive.read(name)
                if hashlib.sha256(member_payload).hexdigest() != expected_sha256:
                    raise PublisherRuntimeAdmissionError(
                        "publisher artifact member bytes diverge"
                    )
                payloads[name] = member_payload
            if manifest.get("source_inventory_sha256") != _inventory_sha256(payloads):
                raise PublisherRuntimeAdmissionError(
                    "publisher artifact inventory digest diverges"
                )
            selected_payload = payloads.get(member_name)
            if selected_payload is None or manifest.get(
                manifest_hash_key
            ) != hashlib.sha256(selected_payload).hexdigest():
                raise PublisherRuntimeAdmissionError(
                    "publisher embedded contract binding diverges"
                )
            return selected_payload
    except PublisherRuntimeAdmissionError:
        raise
    except (KeyError, OSError, UnicodeDecodeError, ValueError, zipfile.BadZipFile) as exc:
        raise PublisherRuntimeAdmissionError(
            "publisher artifact cannot be verified"
        ) from exc


def _strict_json_mapping(payload: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise PublisherRuntimeAdmissionError(
            "publisher runtime contract is not strict JSON"
        ) from exc
    if not isinstance(value, dict):
        raise PublisherRuntimeAdmissionError("publisher runtime contract must be a mapping")
    return value


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _file_sha256(path: Path) -> str:
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise PublisherRuntimeAdmissionError(
            "publisher runtime file cannot be inspected"
        ) from exc
    if _is_link_or_reparse(path, before) or not stat.S_ISREG(before.st_mode):
        raise PublisherRuntimeAdmissionError(
            "publisher runtime file is linked or non-regular"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        if _file_identity(opened_before) != _file_identity(before):
            raise PublisherRuntimeAdmissionError(
                "publisher runtime file changed before hashing"
            )
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        opened_after = os.fstat(handle.fileno())
    try:
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise PublisherRuntimeAdmissionError(
            "publisher runtime file disappeared after hashing"
        ) from exc
    if not (
        _file_identity(before)
        == _file_identity(opened_before)
        == _file_identity(opened_after)
        == _file_identity(after)
    ):
        raise PublisherRuntimeAdmissionError(
            "publisher runtime file changed while hashing"
        )
    return digest.hexdigest()


def _inventory_sha256(payloads: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, payload in sorted(payloads.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_binary_fingerprints() -> list[dict[str, str]]:
    candidates: dict[str, Path] = {
        "executable": Path(sys.executable),
        "base_executable": Path(getattr(sys, "_base_executable", sys.executable)),
    }
    base = Path(sys.base_prefix)
    if os.name == "nt":
        candidates["python_abi_library"] = base / (
            f"python{sys.version_info.major}{sys.version_info.minor}.dll"
        )
        stable_abi = base / "python3.dll"
        if stable_abi.is_file():
            candidates["python_stable_abi_library"] = stable_abi
    else:
        library = sysconfig.get_config_var("LDLIBRARY")
        library_dir = sysconfig.get_config_var("LIBDIR")
        if library and library_dir:
            candidates["python_abi_library"] = Path(library_dir) / str(library)
    missing = sorted(role for role, path in candidates.items() if not path.is_file())
    if missing:
        raise PublisherRuntimeAdmissionError(
            f"publisher Python runtime binaries are unavailable: {missing}"
        )
    return [
        {
            "role": role,
            "sha256": _file_sha256(path),
        }
        for role, path in sorted(candidates.items())
    ]


def _runtime_import_tree_fingerprint() -> dict[str, object]:
    """Hash importable stdlib/DLL bytes admitted ahead of business execution."""

    base = Path(sys.base_prefix)
    roots = (base / "Lib", base / "DLLs")
    files: list[tuple[str, Path]] = []
    for root in roots:
        try:
            root_stat = root.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise PublisherRuntimeAdmissionError(
                "publisher runtime import root is unreadable"
            ) from exc
        if _is_link_or_reparse(root, root_stat) or not stat.S_ISDIR(root_stat.st_mode):
            raise PublisherRuntimeAdmissionError(
                "publisher runtime import root is linked or non-directory"
            )
        pending = [root]
        while pending:
            directory = pending.pop()
            try:
                entries = list(os.scandir(directory))
            except OSError as exc:
                raise PublisherRuntimeAdmissionError(
                    "publisher runtime import tree is unreadable"
                ) from exc
            for entry in entries:
                path = Path(entry.path)
                path_stat = entry.stat(follow_symlinks=False)
                if _is_link_or_reparse(path, path_stat):
                    raise PublisherRuntimeAdmissionError(
                        "publisher runtime import tree contains a filesystem link"
                    )
                if stat.S_ISDIR(path_stat.st_mode):
                    if (
                        directory == base / "Lib"
                        and entry.name.casefold() == "site-packages"
                    ):
                        continue
                    pending.append(path)
                elif stat.S_ISREG(path_stat.st_mode):
                    files.append((path.relative_to(base).as_posix(), path))
                else:
                    raise PublisherRuntimeAdmissionError(
                        "publisher runtime import tree contains a non-regular entry"
                    )
    python_zip = base / "python311.zip"
    if python_zip.is_file():
        zip_stat = python_zip.stat(follow_symlinks=False)
        if _is_link_or_reparse(python_zip, zip_stat) or int(zip_stat.st_nlink) > 1:
            raise PublisherRuntimeAdmissionError(
                "publisher runtime python311.zip is linked"
            )
        files.append((python_zip.relative_to(base).as_posix(), python_zip))
    digest = hashlib.sha256()
    for relative, path in sorted(files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return {"tree_sha256": digest.hexdigest(), "file_count": len(files)}


def _is_link_or_reparse(path: Path, path_stat: os.stat_result) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(path_stat, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse)


def _file_identity(path_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(path_stat.st_dev),
        int(path_stat.st_ino),
        int(path_stat.st_size),
        int(path_stat.st_mtime_ns),
        int(path_stat.st_nlink),
    )

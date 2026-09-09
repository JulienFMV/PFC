"""Pre-import admission and dispatch for the provider-evidence verifier zipapp."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

import pfc_shaping.publisher_runtime_admission as runtime_core
from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.exit_codes import EXIT_INTEGRITY_FAILURE
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    paths_overlap_by_identity,
    read_stable_single_link_file,
)
from pfc_shaping.publisher_runtime_admission import (
    ProcessRuntimeDirectory,
    PublisherRuntimeAdmissionError,
    verify_content_addressed_runtime,
)
from pfc_shaping.verifier_package_contract import (
    VERIFIER_ARTIFACT_SCHEMA,
    VERIFIER_COMMANDS,
    VERIFIER_MANIFEST_PATH,
    VERIFIER_RUNTIME_CONTRACT_PATH,
)

VERIFIER_RUNTIME_SCHEMA = "fmv_lt_provider_verifier_runtime.v1"
VERIFIER_ADMISSION_SCHEMA = "fmv_lt_provider_verifier_runtime_admission.v1"
VERIFIER_DEPENDENCY_ROOT_ENV = "PFC_LT_VERIFIER_DEPENDENCY_ROOT"
VERIFIER_SCRATCH_ROOT_ENV = "PFC_LT_VERIFIER_SCRATCH_ROOT"
VERIFIER_DEPENDENCY_LOADING_POLICY: dict[str, object] = {
    "verified_source_root_imported": False,
    "capture_mode": "PROCESS_PRIVATE_EXACT_BYTE_COPY",
    "source_revalidation": "POST_CAPTURE_FULL_TREE",
    "captured_tree_reverification": True,
    "captured_distribution_reverification": True,
    "capture_cleanup": "SUPERVISING_PARENT_AFTER_WORKER_EXIT",
    "artifact_import_mode": "PROCESS_PRIVATE_EXACT_BYTE_COPY",
    "supervisor_capability": (
        "ONE_SHOT_TOKEN_PARENT_PID_SOURCE_BYTES_AND_GOVERNED_ROOT_BOUND"
    ),
    "windows_process_tree": (
        "CREATE_SUSPENDED_ASSIGN_KILL_ON_JOB_CLOSE_THEN_RESUME"
    ),
    "worker_environment": "MINIMAL_ALLOWLIST_NO_CALLER_SECRETS",
    "worker_timeout_seconds": 900,
    "host_read_only_attestation": "REQUIRED_EXTERNAL_SIGNED_ATTESTATION",
    "host_capture_isolation": (
        "DEDICATED_IDENTITY_NO_UNTRUSTED_COPROCESS_SAME_TOKEN"
    ),
}
VERIFIER_RUNTIME_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "delivery_model",
        "artifact_contains_dependencies",
        "python",
        "launcher_policy",
        "dependency_loading_policy",
        "dependency_closure",
        "locked_distributions",
        "source_lock",
        "installation_policy",
        "authority_class",
        "production_authorization",
    }
)
VERIFIER_INSTALLATION_POLICY = {
    "network_access": "DENIED",
    "dependency_source": "FMV_APPROVED_HASH_VERIFIED_WHEELHOUSE",
    "exact_versions_required": True,
    "isolated_launcher_required": True,
    "content_addressed_dependency_closure_required": True,
    "target_platform_wheel_attestation": "REQUIRED_EXTERNAL_SIGNED_ATTESTATION",
    "sbom": "REQUIRED_EXTERNAL_SIGNED_ATTESTATION",
    "release_attestation": "REQUIRED_EXTERNAL_SIGNED_ATTESTATION",
}

_MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
_MIN_SCRATCH_FREE_BYTES = 512 * 1024 * 1024
_WORKER_TIMEOUT_SECONDS = 900
_SUPERVISED_WORKER_ENV = "_PFC_LT_VERIFIER_SUPERVISED_WORKER"
_SUPERVISOR_ROOT_ENV = "_PFC_LT_VERIFIER_SUPERVISOR_ROOT"
_SUPERVISOR_TOKEN_ENV = "_PFC_LT_VERIFIER_SUPERVISOR_TOKEN"
_SUPERVISOR_CAPABILITY = ".worker-capability.json"
_INTERNAL_ENVIRONMENT = frozenset(
    {_SUPERVISED_WORKER_ENV, _SUPERVISOR_ROOT_ENV, _SUPERVISOR_TOKEN_ENV}
)
_FORBIDDEN_CAPABILITY_MARKERS = (
    "SIGNING_PRIVATE_KEY",
    "ANCHOR_MTLS_KEY",
)
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
        "SYSTEMROOT",
        "USERDOMAIN",
        "USERNAME",
        "USERPROFILE",
        "WINDIR",
    }
)


def run_admitted_verifier_zipapp() -> int:
    """Supervise a captured worker that imports only admitted dependency bytes."""

    try:
        _assert_no_authority_capability_environment()
        if os.environ.get(_SUPERVISED_WORKER_ENV) == "1":
            return _run_supervised_worker()
        if any(name in os.environ for name in _INTERNAL_ENVIRONMENT):
            raise PublisherRuntimeAdmissionError(
                "provider verifier internal supervisor environment is reserved"
            )
        return _run_supervisor()
    except (OSError, RuntimeError, ValueError) as exc:
        return _emit_failure(
            exc,
            retry_disposition="DO_NOT_RETRY_UNADMITTED_RUNTIME",
        )


def _run_supervisor() -> int:
    _assert_isolated_launcher_flags()
    _command_name(sys.argv[1:])
    source = Path(sys.argv[0]).absolute()
    source_payload = read_stable_single_link_file(
        source,
        label="provider verifier source artifact",
        max_bytes=_MAX_ARTIFACT_BYTES,
    )
    scratch = _validated_scratch_root()
    runtime = ProcessRuntimeDirectory(
        prefix="vv-",
        parent=scratch,
    )
    captured = Path(runtime.name) / "provider-verifier.pyz"
    worker: subprocess.Popen[bytes] | None = None
    worker_exit_code: int | None = None
    windows_job: int | None = None
    runtime_error: BaseException | None = None
    cleanup_error: OSError | None = None
    try:
        _write_captured_artifact(captured, source_payload)
        token = secrets.token_hex(32)
        _write_worker_capability(
            Path(runtime.name),
            source_artifact=source,
            captured_artifact=captured,
            source_artifact_sha256=hashlib.sha256(source_payload).hexdigest(),
            scratch_root=scratch,
            token=token,
        )
        environment = _supervised_worker_environment(
            supervisor_root=Path(runtime.name),
            token=token,
        )
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        if os.name == "nt":
            creationflags |= runtime_core._WINDOWS_CREATE_SUSPENDED
        worker = subprocess.Popen(
            [sys.executable, "-I", "-S", "-B", str(captured), *sys.argv[1:]],
            env=environment,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
        if os.name == "nt":
            windows_job = runtime_core._create_windows_kill_on_close_job()
            runtime_core._assign_process_to_windows_job(windows_job, worker)
            runtime_core._resume_windows_process(worker)
        worker_exit_code = worker.wait(timeout=_WORKER_TIMEOUT_SECONDS)
    except KeyboardInterrupt:
        if worker is not None:
            runtime_core._terminate_supervised_worker(worker)
        raise
    except subprocess.TimeoutExpired:
        runtime_error = PublisherRuntimeAdmissionError(
            "provider verifier worker exceeded its fixed deadline"
        )
        if worker is not None and worker.poll() is None:
            try:
                runtime_core._terminate_supervised_worker(worker)
                worker_exit_code = worker.poll()
            except (OSError, subprocess.SubprocessError) as exc:
                runtime_error = exc
    except (OSError, RuntimeError, ValueError) as exc:
        runtime_error = exc
        if worker is not None and worker.poll() is None:
            try:
                runtime_core._terminate_supervised_worker(worker)
            except (OSError, subprocess.SubprocessError) as termination_exc:
                runtime_error = termination_exc
    finally:
        if worker is not None and worker.poll() is None:
            try:
                runtime_core._terminate_supervised_worker(worker)
            except (OSError, subprocess.SubprocessError) as exc:
                runtime_error = runtime_error or exc
        if windows_job is not None:
            try:
                runtime_core._close_windows_handle(windows_job)
            except OSError as exc:
                cleanup_error = exc
        runtime_cleanup_error = _cleanup_runtime_directory(runtime)
        if runtime_cleanup_error is not None:
            cleanup_error = runtime_cleanup_error
    if runtime_error is not None:
        disposition = (
            "DO_NOT_RETRY_RUNTIME_RESIDUE"
            if cleanup_error is not None
            else "DO_NOT_RETRY_UNADMITTED_RUNTIME"
        )
        return _emit_failure(runtime_error, retry_disposition=disposition)
    if cleanup_error is not None:
        return _emit_failure(
            cleanup_error,
            retry_disposition="DO_NOT_RETRY_RUNTIME_RESIDUE",
        )
    if worker_exit_code is None:
        return _emit_failure(
            PublisherRuntimeAdmissionError("provider verifier worker did not start"),
            retry_disposition="DO_NOT_RETRY_UNADMITTED_RUNTIME",
        )
    return worker_exit_code


def _run_supervised_worker() -> int:
    _assert_isolated_launcher_flags()
    command = _command_name(sys.argv[1:])
    source_artifact, captured_artifact, supervisor_root = (
        _consume_worker_capability()
    )
    try:
        admission = verify_content_addressed_runtime(
            captured_artifact,
            artifact_manifest_path=VERIFIER_MANIFEST_PATH,
            artifact_schema=VERIFIER_ARTIFACT_SCHEMA,
            runtime_contract_path=VERIFIER_RUNTIME_CONTRACT_PATH,
            runtime_schema=VERIFIER_RUNTIME_SCHEMA,
            dependency_root_environment=VERIFIER_DEPENDENCY_ROOT_ENV,
            expected_dependency_loading_policy=VERIFIER_DEPENDENCY_LOADING_POLICY,
            admission_schema=VERIFIER_ADMISSION_SCHEMA,
            capture_parent=supervisor_root,
            capture_prefix="vd-",
            required_runtime_keys=VERIFIER_RUNTIME_KEYS,
            required_runtime_policy={
                "delivery_model": (
                    "ISOLATED_ZIPAPP_WITH_CONTENT_ADDRESSED_DEPENDENCY_CLOSURE"
                ),
                "artifact_contains_dependencies": False,
                "installation_policy": VERIFIER_INSTALLATION_POLICY,
                "authority_class": (
                    "READ_ONLY_PROVIDER_EVIDENCE_VERIFICATION_LOCAL_NO_GO"
                ),
                "production_authorization": False,
            },
        )
        admission.update(_artifact_sys_path_receipt(source_artifact, captured_artifact))
        admission["runtime_executable"] = runtime_core.python_runtime_fingerprint()
        admission.update(
            {
                "runtime_authority": False,
                "supervisor_attestation": (
                    "UNSIGNED_LOCAL_PROCESS_OBSERVATION_NOT_AUTHORITY"
                ),
            }
        )
        _assert_exact_runtime_path_receipt(admission)
        return _dispatch_admitted(
            command,
            sys.argv[2:],
            admission,
            supervisor_root=supervisor_root,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return _emit_failure(
            exc,
            retry_disposition="DO_NOT_RETRY_UNADMITTED_RUNTIME",
        )
    except SystemExit as exc:
        return int(exc.code or 0)


def _write_captured_artifact(captured: Path, source_payload: bytes) -> None:
    with captured.open("xb") as handle:
        handle.write(source_payload)
        handle.flush()
        os.fsync(handle.fileno())
    if read_stable_single_link_file(
        captured,
        label="captured provider verifier artifact",
        max_bytes=_MAX_ARTIFACT_BYTES,
    ) != source_payload:
        raise PublisherRuntimeAdmissionError(
            "captured provider verifier artifact differs"
        )
    captured.chmod(stat.S_IREAD)
    if read_stable_single_link_file(
        captured,
        label="sealed provider verifier artifact",
        max_bytes=_MAX_ARTIFACT_BYTES,
    ) != source_payload:
        raise PublisherRuntimeAdmissionError(
            "captured provider verifier artifact changed while sealing"
        )


def _write_worker_capability(
    supervisor_root: Path,
    *,
    source_artifact: Path,
    captured_artifact: Path,
    source_artifact_sha256: str,
    scratch_root: Path,
    token: str,
) -> None:
    payload = json.dumps(
        {
            "captured_artifact": str(captured_artifact),
            "parent_pid": os.getpid(),
            "source_artifact": str(source_artifact),
            "source_artifact_sha256": source_artifact_sha256,
            "scratch_root": str(scratch_root),
            "supervisor_root": str(supervisor_root),
            "token": token,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    path = supervisor_root / _SUPERVISOR_CAPABILITY
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _consume_worker_capability() -> tuple[Path, Path, Path]:
    token = os.environ.get(_SUPERVISOR_TOKEN_ENV, "")
    root_value = os.environ.get(_SUPERVISOR_ROOT_ENV, "")
    if len(token) != 64 or any(character not in "0123456789abcdef" for character in token):
        raise PublisherRuntimeAdmissionError(
            "provider verifier worker token is invalid"
        )
    root = assert_absolute_path_has_no_links(Path(root_value))
    if not root.is_dir() or not root.name.startswith("vv-"):
        raise PublisherRuntimeAdmissionError(
            "provider verifier supervisor root is invalid"
        )
    scratch = _validated_scratch_root()
    try:
        governed_parent = os.path.samefile(root.parent, scratch)
    except OSError as exc:
        raise PublisherRuntimeAdmissionError(
            "provider verifier supervisor parent cannot be verified"
        ) from exc
    if not governed_parent:
        raise PublisherRuntimeAdmissionError(
            "provider verifier supervisor root is outside governed scratch"
        )
    payload = read_stable_single_link_file(
        root / _SUPERVISOR_CAPABILITY,
        label="provider verifier worker capability",
        max_bytes=16 * 1024,
    )
    capability = _strict_json_mapping(payload)
    captured = Path(str(capability.get("captured_artifact", "")))
    source = Path(str(capability.get("source_artifact", "")))
    source_sha256 = str(capability.get("source_artifact_sha256", ""))
    expected = {
        "captured_artifact": str(Path(sys.argv[0]).absolute()),
        "parent_pid": os.getppid(),
        "source_artifact": str(source),
        "source_artifact_sha256": source_sha256,
        "scratch_root": str(scratch),
        "supervisor_root": str(root),
        "token": token,
    }
    if (
        capability != expected
        or captured != Path(sys.argv[0]).absolute()
        or not secrets.compare_digest(str(capability.get("token", "")), token)
    ):
        raise PublisherRuntimeAdmissionError(
            "provider verifier worker capability is invalid"
        )
    if (
        not source.is_absolute()
        or not captured.is_absolute()
        or _path_key(captured.parent) != _path_key(root)
        or _path_key(source) == _path_key(captured)
    ):
        raise PublisherRuntimeAdmissionError(
            "provider verifier worker path capability is invalid"
        )
    source_payload = read_stable_single_link_file(
        source,
        label="provider verifier supervisor source artifact",
        max_bytes=_MAX_ARTIFACT_BYTES,
    )
    captured_payload = read_stable_single_link_file(
        captured,
        label="provider verifier supervisor captured artifact",
        max_bytes=_MAX_ARTIFACT_BYTES,
    )
    if (
        len(source_sha256) != 64
        or hashlib.sha256(source_payload).hexdigest() != source_sha256
        or source_payload != captured_payload
    ):
        raise PublisherRuntimeAdmissionError(
            "provider verifier supervisor source-to-capture binding is invalid"
        )
    (root / _SUPERVISOR_CAPABILITY).unlink()
    return source, captured, root


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
            "provider verifier requires python.exe -I -S -B"
        )


def _validated_scratch_root() -> Path:
    raw = os.environ.get(VERIFIER_SCRATCH_ROOT_ENV, "").strip()
    selected = Path(raw).expanduser()
    if not raw or not selected.is_absolute():
        raise PublisherRuntimeAdmissionError(
            f"{VERIFIER_SCRATCH_ROOT_ENV} must be an absolute existing directory"
        )
    scratch = assert_absolute_path_has_no_links(Path(os.path.abspath(selected)))
    if not scratch.is_dir():
        raise PublisherRuntimeAdmissionError(
            f"{VERIFIER_SCRATCH_ROOT_ENV} is unavailable"
        )
    if shutil.disk_usage(scratch).free < _MIN_SCRATCH_FREE_BYTES:
        raise PublisherRuntimeAdmissionError(
            "provider verifier scratch capacity is below the required floor"
        )
    return scratch


def _artifact_sys_path_receipt(source: Path, captured: Path) -> dict[str, object]:
    source_key = _path_key(source)
    captured_key = _path_key(captured)
    captured_payload = read_stable_single_link_file(
        captured,
        label="admitted provider verifier artifact",
        max_bytes=_MAX_ARTIFACT_BYTES,
    )
    return {
        "captured_artifact_sha256": hashlib.sha256(captured_payload).hexdigest(),
        "captured_artifact_sys_path_count": sum(
            bool(item) and _path_key(Path(item).expanduser()) == captured_key
            for item in sys.path
        ),
        "source_artifact_sys_path_count": sum(
            bool(item) and _path_key(Path(item).expanduser()) == source_key
            for item in sys.path
        ),
    }


def _assert_exact_runtime_path_receipt(admission: Mapping[str, object]) -> None:
    expected = {
        "captured_artifact_sys_path_count": 1,
        "source_artifact_sys_path_count": 0,
        "captured_dependency_root_sys_path_count": 1,
        "source_dependency_root_sys_path_count": 0,
    }
    if any(admission.get(key) != value for key, value in expected.items()):
        raise PublisherRuntimeAdmissionError(
            "provider verifier runtime sys.path receipt is not exact"
        )


def _dispatch_admitted(
    command: str,
    argv: list[str],
    admission: Mapping[str, object],
    *,
    supervisor_root: Path,
) -> int:
    if command == "runtime-check":
        if argv:
            raise PublisherRuntimeAdmissionError(
                "provider verifier runtime-check accepts no arguments"
            )
        print(json.dumps(dict(admission), sort_keys=True))
        return 0
    parser = argparse.ArgumentParser(prog=f"provider-verifier.pyz {command}")
    if command == "audit-current-eex":
        parser.add_argument("--repository-root", type=Path, required=True)
        parser.add_argument("--registry", type=Path, required=True)
        parser.add_argument("--expected-registry-sha256", required=True)
        parser.add_argument("--output-root", type=Path, required=True)
    else:
        parser.add_argument("--acquisition-directory", type=Path, required=True)
        parser.add_argument("--expected-manifest-sha256", required=True)
        if command == "audit-legacy-resolution":
            parser.add_argument("--prior-audit", type=Path, required=True)
            parser.add_argument("--expected-prior-audit-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--runtime-receipt-json", type=Path, required=True)
    args = parser.parse_args(argv)
    protected_paths = [
        supervisor_root,
        Path(os.environ[VERIFIER_SCRATCH_ROOT_ENV]),
        Path(os.environ[VERIFIER_DEPENDENCY_ROOT_ENV]),
    ]
    if command == "audit-current-eex":
        protected_paths.extend(
            (
                args.registry,
                args.repository_root / ".planning",
                args.repository_root / "build" / "eex-forward-local-captures",
                args.repository_root / "build" / "workspace-local-runs",
                args.repository_root
                / "pfc_shaping"
                / "data"
                / "eex_forward_vintage_intake.py",
                args.repository_root / "pfc_shaping" / "data" / "ingest_forwards.py",
            )
        )
    else:
        protected_paths.append(args.acquisition_directory)
        if command == "audit-legacy-resolution":
            protected_paths.append(args.prior_audit)
    if command == "audit-current-eex":
        audit_output, receipt_output = _admit_current_eex_output_pair(
            repository_root=args.repository_root,
            output_root=args.output_root,
            audit_output=args.output_json,
            receipt_output=args.runtime_receipt_json,
            protected_paths=tuple(protected_paths),
        )
    else:
        audit_output, receipt_output = _admit_business_output_pair(
            audit_output=args.output_json,
            receipt_output=args.runtime_receipt_json,
            protected_paths=tuple(protected_paths),
        )
    dependency_versions = admission.get("dependency_versions")
    if not isinstance(dependency_versions, dict) or not dependency_versions:
        raise PublisherRuntimeAdmissionError(
            "provider verifier dependency receipt is unavailable"
        )
    from pfc_shaping.build_identity import SOURCE_REVISION

    if not isinstance(SOURCE_REVISION, str):
        raise PublisherRuntimeAdmissionError(
            "provider verifier source revision is unavailable"
        )
    runtime_receipt = {
        key: admission[key]
        for key in (
            "schema_version",
            "status",
            "dependency_tree_sha256",
            "dependency_import_source",
            "captured_artifact_sha256",
            "captured_artifact_sys_path_count",
            "source_artifact_sys_path_count",
            "captured_dependency_root_sys_path_count",
            "source_dependency_root_sys_path_count",
            "runtime_executable",
        )
    }
    try:
        if command == "audit-current-eex":
            from scripts.audit_ch_eex_current_local_capture import (
                audit_current_eex_capture,
            )

            result = audit_current_eex_capture(
                registry_path=args.registry,
                expected_registry_sha256=args.expected_registry_sha256,
                repository_root=args.repository_root,
            )
            write_durable_exact_bytes(
                audit_output,
                json.dumps(
                    result,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
                + b"\n",
                label="provider verifier current EEX audit result",
            )
        elif command == "audit-acquisition":
            from pfc_shaping.cli.audit_provider_acquisition_quarantine import (
                audit_quarantine,
            )

            result = audit_quarantine(
                acquisition_directory=args.acquisition_directory,
                expected_manifest_sha256=args.expected_manifest_sha256,
                output_json=audit_output,
            )
        else:
            from pfc_shaping.cli.audit_legacy_provider_resolution import (
                audit_legacy_resolution,
            )

            result = audit_legacy_resolution(
                acquisition_directory=args.acquisition_directory,
                expected_manifest_sha256=args.expected_manifest_sha256,
                prior_audit_path=args.prior_audit,
                expected_prior_audit_sha256=args.expected_prior_audit_sha256,
                output_json=audit_output,
            )
        receipt = _write_runtime_observation_receipt(
            command=command,
            audit_result=result,
            audit_output=audit_output,
            receipt_output=receipt_output,
            source_revision=SOURCE_REVISION,
            dependency_versions=dependency_versions,
            runtime_observation=runtime_receipt,
            supervisor_root=supervisor_root,
            protected_paths=tuple(protected_paths),
        )
    except (OSError, ValueError) as exc:
        return _emit_failure(
            exc,
            retry_disposition="EXACT_RETRY_OR_QUARANTINE_AFTER_REPAIR",
        )
    print(json.dumps(receipt, sort_keys=True))
    return 0


def _supervised_worker_environment(*, supervisor_root: Path, token: str) -> dict[str, str]:
    dependency_root = os.environ.get(VERIFIER_DEPENDENCY_ROOT_ENV, "").strip()
    scratch_root = os.environ.get(VERIFIER_SCRATCH_ROOT_ENV, "").strip()
    if not dependency_root or not scratch_root:
        raise PublisherRuntimeAdmissionError(
            "provider verifier governed runtime roots are unavailable"
        )
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.upper() in _SAFE_HOST_ENVIRONMENT_NAMES
    }
    environment.update(
        {
            VERIFIER_DEPENDENCY_ROOT_ENV: dependency_root,
            VERIFIER_SCRATCH_ROOT_ENV: scratch_root,
            _SUPERVISED_WORKER_ENV: "1",
            _SUPERVISOR_ROOT_ENV: str(supervisor_root),
            _SUPERVISOR_TOKEN_ENV: token,
            "TEMP": str(supervisor_root),
            "TMP": str(supervisor_root),
        }
    )
    return environment


def _cleanup_runtime_directory(runtime: ProcessRuntimeDirectory) -> OSError | None:
    cleanup_error: OSError | None = None
    for attempt in range(3):
        try:
            runtime.cleanup()
            return None
        except OSError as exc:
            cleanup_error = exc
            if attempt < 2:
                time.sleep(0.1 * (attempt + 1))
    return cleanup_error


def _admit_business_output_pair(
    *,
    audit_output: Path,
    receipt_output: Path,
    protected_paths: tuple[Path, ...],
) -> tuple[Path, Path]:
    audit_path = assert_absolute_path_has_no_links(audit_output.expanduser())
    receipt_path = assert_absolute_path_has_no_links(receipt_output.expanduser())
    admitted_protected = tuple(
        assert_absolute_path_has_no_links(path.expanduser())
        for path in protected_paths
    )
    if (
        audit_path.suffix != ".json"
        or receipt_path.suffix != ".json"
        or not audit_path.parent.is_dir()
        or not receipt_path.parent.is_dir()
        or paths_overlap_by_identity(audit_path, receipt_path)
    ):
        raise PublisherRuntimeAdmissionError(
            "provider verifier business output pair is invalid"
        )
    for destination in (audit_path, receipt_path):
        if any(
            paths_overlap_by_identity(destination, protected)
            for protected in admitted_protected
        ):
            raise PublisherRuntimeAdmissionError(
                "provider verifier business output overlaps protected runtime or evidence"
            )
    return audit_path, receipt_path


def _admit_current_eex_output_pair(
    *,
    repository_root: Path,
    output_root: Path,
    audit_output: Path,
    receipt_output: Path,
    protected_paths: tuple[Path, ...],
) -> tuple[Path, Path]:
    repository = assert_absolute_path_has_no_links(repository_root.expanduser())
    selected_root = assert_absolute_path_has_no_links(output_root.expanduser())
    governed_parent = assert_absolute_path_has_no_links(
        repository / "build" / "provider-verifier-eex-results"
    )
    if (
        not repository.is_dir()
        or not governed_parent.is_dir()
        or not selected_root.is_dir()
        or _path_key(selected_root.parent) != _path_key(governed_parent)
        or selected_root.name in {"", ".", ".."}
    ):
        raise PublisherRuntimeAdmissionError(
            "provider verifier current EEX output root is not governed"
        )
    expected_audit = selected_root / "current-eex-audit.json"
    expected_receipt = selected_root / "runtime-receipt.json"
    if (
        _path_key(audit_output.expanduser()) != _path_key(expected_audit)
        or _path_key(receipt_output.expanduser()) != _path_key(expected_receipt)
    ):
        raise PublisherRuntimeAdmissionError(
            "provider verifier current EEX output names are not exact"
        )
    return _admit_business_output_pair(
        audit_output=expected_audit,
        receipt_output=expected_receipt,
        protected_paths=protected_paths,
    )


def _write_runtime_observation_receipt(
    *,
    command: str,
    audit_result: Mapping[str, object],
    audit_output: Path,
    receipt_output: Path,
    source_revision: str,
    dependency_versions: Mapping[str, str],
    runtime_observation: Mapping[str, object],
    supervisor_root: Path,
    protected_paths: tuple[Path, ...],
) -> dict[str, object]:
    audit_path = assert_absolute_path_has_no_links(audit_output.expanduser())
    receipt_path = assert_absolute_path_has_no_links(receipt_output.expanduser())
    if (
        not audit_path.is_absolute()
        or not receipt_path.is_absolute()
        or receipt_path.suffix != ".json"
        or not receipt_path.parent.is_dir()
        or paths_overlap_by_identity(receipt_path, audit_path)
        or paths_overlap_by_identity(receipt_path, supervisor_root)
    ):
        raise PublisherRuntimeAdmissionError(
            "provider verifier runtime receipt destination is invalid"
        )
    for destination in (audit_path, receipt_path):
        if any(
            paths_overlap_by_identity(destination, protected)
            for protected in protected_paths
        ):
            raise PublisherRuntimeAdmissionError(
                "provider verifier output boundary changed after admission"
            )
    audit_payload = read_stable_single_link_file(
        audit_path,
        label="provider verifier business audit result",
        max_bytes=4 * 1024 * 1024,
    )
    expected_payload = json.dumps(
        dict(audit_result),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    if audit_payload != expected_payload:
        raise PublisherRuntimeAdmissionError(
            "provider verifier business audit output differs from its result"
        )
    _revalidate_business_audit_evidence(audit_result, command=command)
    receipt = {
        "schema_version": "fmv_lt_provider_verifier_audit_receipt.v1",
        "status": "LOCAL_RUNTIME_OBSERVATION_NOT_AUTHORITY",
        "runtime_authority": False,
        "production_authorization": False,
        "command": command,
        "audit_result": {
            "path": str(audit_path),
            "sha256": hashlib.sha256(audit_payload).hexdigest(),
            "schema_version": audit_result.get("schema_version"),
            "status": audit_result.get("status"),
        },
        "runtime_observation": {
            "source_revision": source_revision,
            "dependency_versions": dict(dependency_versions),
            "receipt": dict(runtime_observation),
            "supervisor_attestation": "UNSIGNED_LOCAL_PROCESS_OBSERVATION",
        },
    }
    payload = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    if receipt_path.exists():
        existing = read_stable_single_link_file(
            receipt_path,
            label="existing provider verifier runtime receipt",
            max_bytes=4 * 1024 * 1024,
        )
        if existing != payload:
            raise FileExistsError(
                f"refusing to replace divergent runtime receipt: {receipt_path}"
            )
    else:
        write_durable_exact_bytes(
            receipt_path,
            payload,
            label="provider verifier runtime receipt",
        )
    return receipt


def _revalidate_business_audit_evidence(
    audit_result: Mapping[str, object],
    *,
    command: str,
) -> None:
    if command == "audit-current-eex":
        from scripts.audit_ch_eex_current_local_capture import (
            audit_current_eex_capture,
        )

        replay = audit_current_eex_capture(
            registry_path=Path(str(audit_result.get("registry_path", ""))),
            expected_registry_sha256=str(audit_result.get("registry_sha256", "")),
            repository_root=Path(str(audit_result.get("repository_root", ""))),
        )
        if dict(replay) != dict(audit_result):
            raise PublisherRuntimeAdmissionError(
                "provider verifier current EEX evidence changed after business audit"
            )
        return
    directory = assert_absolute_path_has_no_links(
        Path(str(audit_result.get("acquisition_directory", "")))
    )
    records_value = audit_result.get("artifacts")
    if records_value is None:
        records_value = audit_result.get("provider_artifacts")
    if not isinstance(records_value, Mapping):
        raise PublisherRuntimeAdmissionError(
            "provider verifier audit evidence inventory is unavailable"
        )
    records = dict(records_value)
    if set(path.name for path in directory.iterdir()) != {
        *records,
        "acquisition-build-manifest.json",
    }:
        raise PublisherRuntimeAdmissionError(
            "provider verifier acquisition changed after business audit"
        )
    for name, record_value in records.items():
        if not isinstance(name, str) or not isinstance(record_value, Mapping):
            raise PublisherRuntimeAdmissionError(
                "provider verifier audit evidence record is invalid"
            )
        record = dict(record_value)
        expected_size = record.get("size_bytes")
        expected_sha256 = record.get("sha256")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
            or not isinstance(expected_sha256, str)
        ):
            raise PublisherRuntimeAdmissionError(
                "provider verifier audit evidence binding is invalid"
            )
        payload = read_stable_single_link_file(
            directory / name,
            label=f"provider verifier post-audit evidence {name}",
            max_bytes=expected_size,
        )
        if (
            len(payload) != expected_size
            or hashlib.sha256(payload).hexdigest() != expected_sha256
        ):
            raise PublisherRuntimeAdmissionError(
                "provider verifier audit evidence changed after business audit"
            )
    legacy_manifest = audit_result.get("legacy_manifest")
    if legacy_manifest is None:
        manifest_path = directory / "acquisition-build-manifest.json"
        manifest_sha256 = audit_result.get("manifest_sha256")
    elif isinstance(legacy_manifest, Mapping):
        manifest_path = Path(str(legacy_manifest.get("path", "")))
        manifest_sha256 = legacy_manifest.get("sha256")
    else:
        raise PublisherRuntimeAdmissionError(
            "provider verifier manifest binding is invalid"
        )
    manifest_payload = read_stable_single_link_file(
        manifest_path,
        label="provider verifier post-audit acquisition manifest",
        max_bytes=2 * 1024 * 1024,
    )
    if hashlib.sha256(manifest_payload).hexdigest() != manifest_sha256:
        raise PublisherRuntimeAdmissionError(
            "provider verifier manifest changed after business audit"
        )
    legacy_audit = audit_result.get("legacy_audit")
    if legacy_audit is not None:
        if not isinstance(legacy_audit, Mapping):
            raise PublisherRuntimeAdmissionError(
                "provider verifier legacy audit binding is invalid"
            )
        prior_payload = read_stable_single_link_file(
            Path(str(legacy_audit.get("path", ""))),
            label="provider verifier post-audit legacy audit",
            max_bytes=4 * 1024 * 1024,
        )
        if hashlib.sha256(prior_payload).hexdigest() != legacy_audit.get("sha256"):
            raise PublisherRuntimeAdmissionError(
                "provider verifier legacy audit changed after business audit"
            )


def _command_name(argv: list[str]) -> str:
    if not argv or argv[0] not in VERIFIER_COMMANDS:
        raise PublisherRuntimeAdmissionError(
            "provider verifier command is missing or unsupported"
        )
    return argv[0]


def _assert_no_authority_capability_environment() -> None:
    forbidden = sorted(
        name
        for name in os.environ
        if name.upper().startswith(("PFC_", "FMV_"))
        and any(marker in name.upper() for marker in _FORBIDDEN_CAPABILITY_MARKERS)
    )
    if forbidden:
        raise PublisherRuntimeAdmissionError(
            "provider verifier refuses signing or mTLS private-key capability"
        )


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def _strict_json_mapping(payload: bytes) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError("provider verifier capability must be a JSON mapping")
    return value


def _emit_failure(exc: BaseException, *, retry_disposition: str) -> int:
    print(
        json.dumps(
            {
                "schema_version": "fmv_lt_provider_verifier_result.v1",
                "status": "INTEGRITY_FAILURE",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "retry_disposition": retry_disposition,
                "production_authorization": False,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return EXIT_INTEGRITY_FAILURE

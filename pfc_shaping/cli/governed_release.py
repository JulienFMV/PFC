#!/usr/bin/env python3
"""Run exactly one governed LT build, finalize, register, audit, promote, rollback or status phase."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pfc_shaping.build_identity import SOURCE_REVISION
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    paths_overlap_by_identity,
)
from pfc_shaping.pipeline.atomic_promotion import (
    PromotionBusyError,
    PromotionConflictError,
    PromotionError,
    PromotionProjectionRepairRequired,
    prepare_immutable_directory,
    unlink_with_retry,
    validate_existing_immutable_file,
    write_immutable_bytes,
)
from pfc_shaping.pipeline.governed_release import (
    GovernedReleaseError,
    audit_release_request,
    promote_release_request,
    register_assembled_release_request,
    release_status,
    rollback_release,
)
from pfc_shaping.pipeline.governed_release_cli_contract import (
    EXIT_CAS_CONFLICT,
    EXIT_GOVERNANCE_PENDING,
    EXIT_INTEGRITY_FAILURE,
    EXIT_PROJECTION_REPAIR_REQUIRED,
    EXIT_SUCCESS,
    EXIT_TRANSITION_BUSY,
    ReleaseCliIdentityError,
    absolute_path,
    add_build_arguments,
    add_finalize_arguments,
    assert_installed_runtime_sealed,
    assert_phase_private_key_scope,
    assert_production_transition_runtime_authorized,
    canonical_sha256,
    resolve_build_data_root,
)
from pfc_shaping.version import __version__

BUILD_POLICY_REQUIRED_EXIT = EXIT_GOVERNANCE_PENDING
AUDIT_REJECTED_EXIT = EXIT_GOVERNANCE_PENDING
DIST_NAME = "fmv-pfc-lt"
FAILURE_SCHEMA = "governed_lt_operational_failure.v1"
MAX_ERROR_MESSAGE_CHARS = 2048
STRICT_JSON_MAX_DEPTH = 64


def package_version() -> str:
    try:
        selected = version(DIST_NAME)
    except PackageNotFoundError:
        selected = __version__
    revision = SOURCE_REVISION or "SOURCE_CHECKOUT_UNSEALED"
    return f"{selected} source_revision={revision}"


def _assert_module_runtime() -> None:
    """Admit the installed runtime before argparse can expose any command."""

    assert_installed_runtime_sealed()


def main(argv: list[str] | None = None) -> int:
    operation_id = uuid.uuid4().hex
    process_id = os.getpid()
    failure: Exception | None = None
    try:
        _assert_module_runtime()
        args = _parse_args(argv)
    except ReleaseCliIdentityError as exc:
        print(
            _strict_json(
                {
                    "schema_version": "governed_lt_transition_error.v1",
                    "status": "INTEGRITY_FAILURE",
                    "error": _safe_error_message(exc),
                    "operation_id": operation_id,
                    "process_id": process_id,
                }
            )
        )
        return EXIT_INTEGRITY_FAILURE
    try:
        if args.command != "status":
            _probe_failure_root(args)
        if args.command == "build":
            result = _run_build_phase(args)
            exit_code = BUILD_POLICY_REQUIRED_EXIT
        elif args.command == "finalize":
            result = _run_finalize_phase(args)
            exit_code = EXIT_SUCCESS
        elif args.command == "rollback":
            result = rollback_release(
                release_root=args.release_root,
                rollback_authorization_path=args.rollback_authorization,
            )
            exit_code = EXIT_SUCCESS
        else:
            common = {
                "release_root": args.release_root,
                "workflow_root": args.workflow_root,
                "evidence_root": args.evidence_root,
                "run_id": args.run_id,
            }
            result, exit_code = _run_release_transition(args, common)
    except PromotionConflictError as exc:
        failure = exc
        result = {
            "schema_version": "governed_lt_transition_error.v1",
            "status": "CAS_CONFLICT",
            "error": _safe_error_message(exc),
        }
        _attach_lock_cleanup_status(result, exc)
        exit_code = EXIT_CAS_CONFLICT
    except PromotionBusyError as exc:
        failure = exc
        result = {
            "schema_version": "governed_lt_transition_error.v1",
            "status": "BUSY",
            "error": _safe_error_message(exc),
        }
        _attach_lock_cleanup_status(result, exc)
        exit_code = EXIT_TRANSITION_BUSY
    except PromotionProjectionRepairRequired as exc:
        failure = exc
        result = {
            "schema_version": "governed_lt_transition_error.v1",
            "status": "POINTER_REPAIR_REQUIRED",
            "event_id": exc.event_id,
            "commit_status": exc.commit_status,
            "lock_cleanup_status": getattr(exc, "lock_cleanup_status", "OK"),
            "error": _safe_error_message(exc),
        }
        _attach_lock_cleanup_status(result, exc)
        exit_code = EXIT_PROJECTION_REPAIR_REQUIRED
    except (PromotionError, GovernedReleaseError) as exc:
        failure = exc
        result = {
            "schema_version": "governed_lt_transition_error.v1",
            "status": "INTEGRITY_FAILURE",
            "error": _safe_error_message(exc),
        }
        _attach_lock_cleanup_status(result, exc)
        exit_code = EXIT_INTEGRITY_FAILURE
    except Exception as exc:
        failure = exc
        result = {
            "schema_version": "governed_lt_transition_error.v1",
            "status": "INTEGRITY_FAILURE",
            "error_type": type(exc).__name__,
            "error": _safe_error_message(exc),
        }
        _attach_lock_cleanup_status(result, exc)
        exit_code = EXIT_INTEGRITY_FAILURE
    if type(result) is not dict:
        failure = TypeError("phase result must be an exact dict")
        result = {
            "schema_version": "governed_lt_transition_error.v1",
            "status": "INTEGRITY_FAILURE",
            "error_type": "TypeError",
            "error": "phase result must be an exact dict",
        }
        exit_code = EXIT_INTEGRITY_FAILURE
    result["operation_id"] = operation_id
    result["process_id"] = process_id
    if failure is not None:
        _attach_failure_manifest(
            args,
            result=result,
            exit_code=exit_code,
            failure=failure,
            operation_id=operation_id,
            process_id=process_id,
        )
    try:
        serialized = _strict_json(result)
    except (TypeError, ValueError) as exc:
        fallback: dict[str, object] = {
            "schema_version": "governed_lt_transition_error.v1",
            "status": "INTEGRITY_FAILURE",
            "error_type": type(exc).__name__,
            "error": "result is not strict-JSON serializable",
            "operation_id": operation_id,
            "process_id": process_id,
        }
        _attach_failure_manifest(
            args,
            result=fallback,
            exit_code=EXIT_INTEGRITY_FAILURE,
            failure=exc,
            operation_id=operation_id,
            process_id=process_id,
        )
        print(_strict_json(fallback))
        return EXIT_INTEGRITY_FAILURE
    print(serialized)
    return exit_code


def _attach_lock_cleanup_status(
    result: dict[str, object],
    failure: BaseException,
) -> None:
    if getattr(failure, "lock_cleanup_status", None) == "FAIL":
        result["lock_cleanup_status"] = "FAIL"
        result["lock_cleanup_error_type"] = str(
            getattr(failure, "lock_cleanup_error_type", "UnknownError")
        )


def _run_build_phase(args: argparse.Namespace) -> dict[str, object]:
    from pfc_shaping.pipeline.candidate_build import build_candidate

    return build_candidate(args)


def _run_finalize_phase(args: argparse.Namespace) -> dict[str, object]:
    from pfc_shaping.pipeline.candidate_finalize import finalize_candidate

    return finalize_candidate(args)


def _run_release_transition(
    args: argparse.Namespace,
    common: dict[str, object],
) -> tuple[dict[str, object], int]:
    if args.command == "register":
        result = register_assembled_release_request(
            **common,
            expected_current_event_id=args.expected_current_event_id,
            request_nonce=args.request_nonce,
        )
    elif args.command == "audit":
        result = audit_release_request(
            **common,
            request_id=args.request_id,
            data_root=args.data_root,
            signing_private_key=args.signing_private_key,
            require_assembled_registration=True,
        )
    elif args.command == "promote":
        result = promote_release_request(
            **common,
            request_id=args.request_id,
            require_assembled_registration=True,
        )
    else:
        result = release_status(**common, request_id=args.request_id)
    if args.command == "audit" and result.get("approved") is not True:
        return result, AUDIT_REJECTED_EXIT
    return result, EXIT_SUCCESS


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {package_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build",
        help="build one staged candidate and stop for independent policy",
    )
    add_build_arguments(build)
    finalize = subparsers.add_parser(
        "finalize",
        help="apply signed source policy and finalize one candidate",
    )
    add_finalize_arguments(finalize)
    transition_help = {
        "register": "register artifacts derived from the finalized canonical seal",
        "audit": "run the independent capstone and sign its receipt",
        "promote": "verify the approved receipt and atomically promote",
        "rollback": "apply a fresh signed rollback authorization to one exact promotion",
        "status": "inspect release state without writing",
    }
    parsers = {
        command: subparsers.add_parser(command, help=help_text)
        for command, help_text in transition_help.items()
    }
    for command, child in parsers.items():
        child.add_argument("--release-root", type=absolute_path, required=True)
        if command != "status":
            child.add_argument("--failure-root", type=absolute_path, required=True)
        if command == "rollback":
            continue
        child.add_argument("--workflow-root", type=absolute_path, required=True)
        child.add_argument("--evidence-root", type=absolute_path, required=True)
        child.add_argument("--run-id", required=True)
    audit = parsers["audit"]
    register = parsers["register"]
    expected = register.add_mutually_exclusive_group(required=True)
    expected.add_argument(
        "--expect-no-current",
        action="store_const",
        const=None,
        dest="expected_current_event_id",
    )
    expected.add_argument(
        "--expect-current-event-id",
        dest="expected_current_event_id",
        metavar="EVENT_ID",
        type=canonical_sha256,
    )
    register.add_argument(
        "--request-nonce",
        default=None,
        help="Operator-supplied stable nonce for a fresh request/audit after receipt expiry",
    )
    parsers["audit"].add_argument("--request-id", required=True, type=canonical_sha256)
    parsers["promote"].add_argument("--request-id", required=True, type=canonical_sha256)
    parsers["status"].add_argument("--request-id", default=None, type=canonical_sha256)
    parsers["rollback"].add_argument(
        "--rollback-authorization",
        type=absolute_path,
        required=True,
    )
    audit.add_argument(
        "--data-root",
        type=absolute_path,
        required=True,
    )
    audit.add_argument(
        "--signing-private-key",
        type=absolute_path,
        default=os.environ.get("PFC_PROMOTION_SIGNING_PRIVATE_KEY_PATH"),
        help="Required only when the bound audit receipt does not already exist",
    )
    args = parser.parse_args(argv)
    assert_phase_private_key_scope(args.command)
    if args.command in {"promote", "rollback"}:
        assert_production_transition_runtime_authorized()
    elif args.command != "status":
        assert_installed_runtime_sealed()
    if args.command == "build":
        args = resolve_build_data_root(args, parser)
    if args.command != "status":
        _assert_failure_root_contract(args.failure_root, args.release_root)
        _assert_failure_root_is_dedicated(args)
    return args


def _assert_failure_root_contract(failure_root: Path, release_root: Path) -> None:
    try:
        failure = assert_absolute_path_has_no_links(failure_root)
        release = assert_absolute_path_has_no_links(release_root)
        overlaps = paths_overlap_by_identity(failure, release)
    except (OSError, ValueError) as exc:
        raise ReleaseCliIdentityError(
            "release_root and failure_root must be absolute paths without links"
        ) from exc
    if overlaps:
        raise ReleaseCliIdentityError("failure_root must be external to release_root")


def _assert_failure_root_is_dedicated(args: argparse.Namespace) -> None:
    failure = Path(args.failure_root)
    _assert_failure_root_external_to_paths(failure, _protected_paths(args))


def _protected_paths(args: argparse.Namespace) -> list[tuple[str, Path]]:
    protected: list[tuple[str, Path]] = []
    for name, value in vars(args).items():
        if name in {"failure_root", "release_root"} or not isinstance(value, Path):
            continue
        protected.append((name, value))
    for name, value in os.environ.items():
        if not value.strip():
            continue
        sensitive_path = (
            name.startswith("PFC_")
            and any(token in name for token in ("_KEY_", "_KEYRING", "_ROOT", "_PATH", "_DIR"))
        ) or name in {"FMV_DATA_ROOT"}
        if sensitive_path:
            protected.append((f"environment:{name}", Path(value)))
    return protected


def _assert_failure_root_external_to_paths(
    failure: Path,
    protected: list[tuple[str, Path]],
) -> None:
    for name, path in protected:
        try:
            overlaps = paths_overlap_by_identity(failure, path)
        except (OSError, ValueError) as exc:
            raise ReleaseCliIdentityError(
                f"cannot establish failure_root separation from {name}"
            ) from exc
        if overlaps:
            raise ReleaseCliIdentityError(
                f"failure_root must be dedicated and external to {name}"
            )


def _probe_failure_root(args: argparse.Namespace) -> None:
    try:
        root = assert_absolute_path_has_no_links(args.failure_root)
    except (OSError, ValueError) as exc:
        raise ReleaseCliIdentityError("cannot validate failure_root readiness") from exc
    if not root.is_dir():
        raise ReleaseCliIdentityError("failure_root must be pre-provisioned")
    _assert_failure_root_contract(root, args.release_root)
    _assert_failure_root_is_dedicated(args)
    directory = _failure_manifest_directory(
        root,
        run_id=str(getattr(args, "run_id", "unknown")),
        phase=str(args.command),
    )
    try:
        directory.mkdir(parents=True, exist_ok=True)
        assert_absolute_path_has_no_links(directory)
    except (OSError, ValueError) as exc:
        raise ReleaseCliIdentityError(
            "failure_root cannot provision the run/phase directory"
        ) from exc
    _assert_failure_root_contract(root, args.release_root)
    _assert_failure_root_is_dedicated(args)
    stale_probes = list(directory.glob(".pfc-failure-probe-*.json"))
    for stale in stale_probes:
        validate_existing_immutable_file(stale, label="failure-root probe")
    if stale_probes:
        raise ReleaseCliIdentityError(
            "failure_root contains a probe requiring explicit reconciliation"
        )
    if list(directory.glob(".pfc-*.tmp")):
        raise ReleaseCliIdentityError("failure_root contains unresolved temporary residue")
    probe = directory / f".pfc-failure-probe-{os.getpid()}-{uuid.uuid4().hex}.json"
    try:
        write_immutable_bytes(probe, b'{"schema_version":"failure_root_probe.v1"}\n')
        validate_existing_immutable_file(probe, label="failure-root probe")
    except Exception as exc:
        raise ReleaseCliIdentityError("failure_root write probe failed") from exc
    finally:
        if probe.exists():
            unlink_with_retry(probe)


def _attach_failure_manifest(
    args: argparse.Namespace,
    *,
    result: dict[str, object],
    exit_code: int,
    failure: Exception,
    operation_id: str,
    process_id: int,
) -> None:
    failure_root = getattr(args, "failure_root", None)
    if failure_root is None:
        return
    try:
        path = _write_failure_manifest(
            failure_root=Path(failure_root),
            release_root=Path(args.release_root),
            run_id=str(getattr(args, "run_id", "unknown")),
            phase=str(args.command),
            status=str(result.get("status", "INTEGRITY_FAILURE")),
            exit_code=exit_code,
            failure=failure,
            operation_id=operation_id,
            process_id=process_id,
            lock_cleanup_status=result.get("lock_cleanup_status"),
            lock_cleanup_error_type=result.get("lock_cleanup_error_type"),
            protected_paths=_protected_paths(args),
        )
    except Exception as record_error:
        result["failure_manifest_status"] = "UNAVAILABLE"
        result["failure_recording_error_type"] = type(record_error).__name__
        return
    result["failure_manifest_status"] = "RECORDED"
    result["failure_manifest"] = str(path)


def _write_failure_manifest(
    *,
    failure_root: Path,
    release_root: Path,
    run_id: str,
    phase: str,
    status: str,
    exit_code: int,
    failure: Exception,
    operation_id: str,
    process_id: int,
    lock_cleanup_status: object = None,
    lock_cleanup_error_type: object = None,
    protected_paths: list[tuple[str, Path]] | None = None,
) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", operation_id):
        raise ValueError("operation_id must be a 128-bit lowercase hexadecimal value")
    if type(process_id) is not int or process_id <= 0:
        raise ValueError("process_id must be a positive integer")
    root = assert_absolute_path_has_no_links(failure_root)
    if not root.is_dir():
        raise OSError("failure_root must be pre-provisioned")
    protected = list(protected_paths or [])
    _assert_failure_root_boundaries(root, release_root, protected)
    directory = _failure_manifest_directory(root, run_id=run_id, phase=phase)
    directory.mkdir(parents=True, exist_ok=True)
    assert_absolute_path_has_no_links(directory)
    _assert_failure_root_boundaries(root, release_root, protected)
    prepare_immutable_directory(directory, label="failure manifest")
    _assert_failure_root_boundaries(root, release_root, protected)
    now = datetime.now(timezone.utc)
    name = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex}.json"
    destination = directory / name
    payload = {
        "schema_version": FAILURE_SCHEMA,
        "status": status,
        "phase": phase,
        "run_id": run_id,
        "failed_at_utc": now.isoformat(),
        "source_revision": SOURCE_REVISION,
        "operation_id": operation_id,
        "process_id": process_id,
        "exit_code": int(exit_code),
        "error_type": type(failure).__name__,
        "error": _safe_error_message(failure),
        "promotion_eligible": False,
    }
    if lock_cleanup_status == "FAIL":
        payload["lock_cleanup_status"] = "FAIL"
        payload["lock_cleanup_error_type"] = (
            lock_cleanup_error_type
            if type(lock_cleanup_error_type) is str
            else "UnknownError"
        )
    raw = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _assert_failure_root_boundaries(root, release_root, protected)
    write_immutable_bytes(destination, raw)
    _assert_failure_root_boundaries(root, release_root, protected)
    prepare_immutable_directory(directory, label="failure manifest")
    _assert_failure_root_boundaries(root, release_root, protected)
    validate_existing_immutable_file(destination, label="failure manifest")
    return destination


def _assert_failure_root_boundaries(
    failure_root: Path,
    release_root: Path,
    protected_paths: list[tuple[str, Path]],
) -> None:
    _assert_failure_root_contract(failure_root, release_root)
    _assert_failure_root_external_to_paths(failure_root, protected_paths)


def _failure_manifest_directory(root: Path, *, run_id: str, phase: str) -> Path:
    safe_run_id = (
        run_id
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id)
        else "invalid-run-id"
    )
    safe_phase = phase if re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", phase) else "invalid-phase"
    return root / safe_run_id / safe_phase


def _safe_error_message(error: BaseException) -> str:
    message = str(error).replace("\x00", "").replace("\r", " ").replace("\n", " ")
    message = re.sub(
        r"(?i)(password|secret|token|private[_ -]?key)\s*[:=]\s*\S+",
        r"\1=<redacted>",
        message,
    )
    return message[:MAX_ERROR_MESSAGE_CHARS]


def _strict_json(payload: object) -> str:
    _validate_strict_json_value(payload, depth=0, ancestors=set())
    return json.dumps(
        payload,
        sort_keys=True,
        allow_nan=False,
    )


def _validate_strict_json_value(
    value: object,
    *,
    depth: int,
    ancestors: set[int],
) -> None:
    if depth > STRICT_JSON_MAX_DEPTH:
        raise ValueError("strict JSON maximum nesting depth exceeded")
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("strict JSON rejects non-finite numbers")
        return
    if type(value) not in {list, dict}:
        raise TypeError(f"strict JSON rejects type {type(value).__name__}")
    identity = id(value)
    if identity in ancestors:
        raise ValueError("strict JSON rejects cyclic containers")
    ancestors.add(identity)
    try:
        if type(value) is list:
            for item in value:
                _validate_strict_json_value(
                    item,
                    depth=depth + 1,
                    ancestors=ancestors,
                )
            return
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("strict JSON requires string object keys")
            _validate_strict_json_value(
                item,
                depth=depth + 1,
                ancestors=ancestors,
            )
    finally:
        ancestors.remove(identity)


if __name__ == "__main__":
    raise SystemExit(main())

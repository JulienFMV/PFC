"""Non-promotional storage semantics drill for governed LT releases."""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Callable, Iterable
import uuid

from pfc_shaping.pipeline.atomic_promotion import (
    ImmutableArtifactExistsError,
    PROMOTION_LOCK_WAIT_SECONDS_ENV,
    PromotionBusyError,
    PromotionError,
    _assert_path_has_no_links,
    _atomic_write_json,
    _fsync_directory,
    _path_is_link,
    _promotion_lock,
    _read_bytes_with_retry,
    _replace_directory,
    _fsync_tree,
    _validate_existing_immutable_file,
    _write_bytes_exclusive_fsync,
    _write_bytes_exclusive_fsync_with_post_link,
)


DRILL_NAMESPACE = ".pfc-lt-storage-drill"
REPORT_SCHEMA_VERSION = "pfc_lt_storage_drill_report.v1"
REPORT_FILE = "storage_drill_report.v1.json"
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
WORKER_CRASH_EXIT = 73
WORKER_CONTENTION_LOST_EXIT = 17
DEFAULT_TIMEOUT_SECONDS = 30.0
JOB_OWNER_ENV = "PFC_STORAGE_DRILL_WINDOWS_JOB_OWNER"
_WINDOWS_JOB_HANDLE: object | None = None

EXTERNAL_ATTESTATIONS = (
    "WORM or immutable-retention enforcement",
    "ACL and service-identity least privilege",
    "HSM or managed signing-key custody",
    "SMB appliance durability across node or power loss",
    "backup, restore, retention and disaster-recovery drill",
)
FORBIDDEN_RELEASE_ROOT_MARKERS = (
    "release_domain.json",
    "current.json",
    "promotion_events",
    "promotion_receipts",
    "candidates",
)
FORBIDDEN_CANDIDATE_MARKERS = (
    Path("candidate_bundle_manifest.json"),
    Path("manifests") / "candidate_run_manifest.json",
)


class StorageDrillError(RuntimeError):
    """Raised when the drill cannot be executed within its safety contract."""


def run_storage_drill(
    *,
    drill_root: str | Path,
    run_id: str | None = None,
    alias_roots: Iterable[str | Path] = (),
    required_alias_classes: Iterable[str] = (),
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """Run storage checks in one new isolated directory and archive the report."""

    root = _validate_drill_root(drill_root)
    _assert_not_release_root(root)
    selected_run_id = run_id or _default_run_id()
    _validate_run_id(selected_run_id)
    if not math.isfinite(timeout_seconds) or not 5 <= timeout_seconds <= 300:
        raise StorageDrillError("timeout_seconds must be finite and between 5 and 300")
    selected_alias_roots = tuple(alias_roots)
    selected_required_alias_classes = tuple(required_alias_classes)
    _assert_no_production_private_keys()

    namespace = root / DRILL_NAMESPACE
    if namespace.exists():
        _validate_directory(namespace, label="drill namespace")
    else:
        namespace.mkdir()
        _fsync_directory(root)
    run_dir = namespace / selected_run_id
    try:
        run_dir.mkdir()
    except FileExistsError as exc:
        raise StorageDrillError(f"drill run already exists: {run_dir}") from exc
    _fsync_directory(namespace)
    _assert_drill_workspace_is_empty(run_dir)

    started_at = _utc_now()
    service_identity = _service_identity()
    service_sid = _service_sid()
    checks = [
        _service_identity_check(service_identity=service_identity, service_sid=service_sid)
    ]
    for probe_name in (
        "crash_after_hardlink_recovery",
        "multi_process_exclusive_hardlink_contention",
        "transition_lock_contention",
        "atomic_replace_visibility",
        "directory_finalize_replace_visibility",
    ):
        check = _capture_supervised_probe(
            probe_name,
            run_dir=run_dir,
            timeout_seconds=timeout_seconds,
            required=True,
        )
        _raise_on_process_cleanup_failure(check)
        checks.append(check)
    alias_check = _capture_supervised_probe(
        "alias_samefile_and_visibility",
        run_dir=run_dir,
        timeout_seconds=timeout_seconds,
        required=bool(selected_alias_roots or selected_required_alias_classes),
        extra_arguments=_alias_probe_arguments(
            alias_roots=selected_alias_roots,
            required_alias_classes=selected_required_alias_classes,
        ),
    )
    _raise_on_process_cleanup_failure(alias_check)
    checks.extend(
        [
            alias_check,
            {
            "name": "distributed_multi_host_smb_semantics",
            "required": False,
            "status": "UNSUPPORTED",
            "details": {
                "reason": "this single-host harness cannot attest SMB multi-client leases"
            },
        },
            {
            "name": "external_control_attestations",
            "required": False,
            "status": "UNSUPPORTED",
            "details": {"required_external_attestations": list(EXTERNAL_ATTESTATIONS)},
            },
        ]
    )
    inventory_check = _capture_supervised_probe(
        "artifact_inventory_integrity",
        run_dir=run_dir,
        timeout_seconds=timeout_seconds,
        required=True,
    )
    _raise_on_process_cleanup_failure(inventory_check)
    checks.append(inventory_check)
    try:
        artifact_inventory = _artifact_inventory(run_dir)
    except Exception as exc:
        inventory_check.update(
            {
                "status": "FAIL",
                "details": {"error": f"final inventory failed: {type(exc).__name__}: {exc}"},
            }
        )
        artifact_inventory = []
    else:
        inventory_check["details"] = {"artifacts": artifact_inventory}
    required_checks = [check for check in checks if check["required"] is True]
    overall_status = (
        "PASS"
        if all(check["status"] == "PASS" for check in required_checks)
        else "FAIL"
    )
    report_path = run_dir / REPORT_FILE
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": selected_run_id,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "status": overall_status,
        "status_scope": "APPLICATION_FILESYSTEM_SEMANTICS",
        "evidence_scope": "STORAGE_DRILL_ONLY",
        "promotion_ready": False,
        "production_authorization": False,
        "production_qualification_status": "UNSUPPORTED",
        "non_promotional": True,
        "drill_root": str(root),
        "run_directory": str(run_dir),
        "report_path": str(report_path),
        "report_write_mode": "exclusive_create_fsync",
        "platform": sys.platform,
        "os_version": platform.platform(),
        "hostname": socket.gethostname(),
        "service_identity": service_identity,
        "service_sid": service_sid,
        "supervisor_pid": os.getpid(),
        "python_version": sys.version.split()[0],
        "source_revision": os.environ.get("PFC_SOURCE_REVISION"),
        "volume_device_id": int(run_dir.stat().st_dev),
        "tested_code_sha256": {
            "atomic_promotion.py": _file_sha256(
                Path(__file__).with_name("atomic_promotion.py")
            ),
            "storage_drill.py": _file_sha256(Path(__file__)),
            "path_safety.py": _file_sha256(Path(__file__).resolve().parents[1] / "path_safety.py"),
            "run_lt_release_storage_drill.py": _file_sha256(
                Path(__file__).resolve().parents[2]
                / "scripts"
                / "run_lt_release_storage_drill.py"
            ),
        },
        "checks": checks,
        "artifact_inventory": artifact_inventory,
        "artifact_inventory_exclusions": [REPORT_FILE],
        "required_external_attestations": list(EXTERNAL_ATTESTATIONS),
        "limitations": (
            "A PASS validates only observed application-level filesystem semantics "
            "for this run; it is not production promotion evidence."
        ),
    }
    payload = _canonical_json_bytes(report)
    try:
        _write_report_exclusive(report_path, payload)
        _validate_existing_immutable_file(report_path, label="storage drill report")
    except (OSError, PromotionError) as exc:
        raise StorageDrillError(f"cannot archive storage drill report: {exc}") from exc
    return {
        **report,
        "report_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _validate_drill_root(value: str | Path) -> Path:
    selected = Path(value).expanduser()
    if not selected.is_absolute():
        raise StorageDrillError("drill_root must be an absolute existing directory")
    lexical = Path(os.path.abspath(selected))
    _validate_directory(lexical, label="drill root")
    return lexical


def _validate_directory(path: Path, *, label: str) -> None:
    try:
        _assert_path_has_no_links(path, label=label)
    except PromotionError as exc:
        raise StorageDrillError(str(exc)) from exc
    if not path.is_dir():
        raise StorageDrillError(f"{label} is not an existing directory: {path}")
    for component in (path, *path.parents):
        if component.exists() and _path_is_link(component):
            raise StorageDrillError(f"{label} cannot contain a Windows reparse point")


def _assert_not_release_root(root: Path) -> None:
    for ancestor in (root, *root.parents):
        release_markers = [
            name for name in FORBIDDEN_RELEASE_ROOT_MARKERS if (ancestor / name).exists()
        ]
        candidate_markers = [
            marker.as_posix()
            for marker in FORBIDDEN_CANDIDATE_MARKERS
            if (ancestor / marker).exists()
        ]
        present = release_markers + candidate_markers
        if present:
            raise StorageDrillError(
                "drill_root is inside a governed release or candidate; "
                f"ancestor={ancestor}, markers={present}"
            )


def _validate_worker_run_dir(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise StorageDrillError("worker run directory must be absolute")
    lexical = Path(os.path.abspath(path))
    _validate_directory(lexical, label="worker run directory")
    if lexical.parent.name != DRILL_NAMESPACE:
        raise StorageDrillError("worker run directory is outside the drill namespace")
    _validate_run_id(lexical.name)
    return lexical


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_RE.fullmatch(str(run_id)):
        raise StorageDrillError(f"invalid drill run_id: {run_id!r}")
    if run_id.endswith(".") or run_id.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
        raise StorageDrillError(f"invalid Windows drill run_id: {run_id!r}")


def _assert_drill_workspace_is_empty(run_dir: Path) -> None:
    if any(run_dir.iterdir()):
        raise StorageDrillError("new drill workspace is unexpectedly non-empty")


def _assert_no_production_private_keys() -> None:
    exposed = sorted(
        name
        for name, value in os.environ.items()
        if name.startswith("PFC_") and "PRIVATE_KEY" in name and str(value).strip()
    )
    if exposed:
        raise StorageDrillError(
            f"production private-key variables must not be exposed to the drill: {exposed}"
        )


def _capture_check(
    name: str,
    *,
    required: bool,
    operation: Callable[[], dict[str, object]],
) -> dict[str, object]:
    started = time.monotonic()
    try:
        details = operation()
    except Exception as exc:  # The report must retain every failed drill stage.
        return {
            "name": name,
            "required": required,
            "status": "FAIL",
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
            "details": {"error": f"{type(exc).__name__}: {exc}"},
        }
    return {
        "name": name,
        "required": required,
        "status": "PASS",
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "details": details,
    }


def _capture_supervised_probe(
    name: str,
    *,
    run_dir: Path,
    timeout_seconds: float,
    required: bool,
    extra_arguments: tuple[str, ...] = (),
) -> dict[str, object]:
    started = time.monotonic()
    result_path = run_dir / f"probe-{name}.json"
    worker = _start_worker(
        "run-probe",
        run_dir,
        "--probe-name",
        name,
        "--result",
        str(result_path),
        "--timeout-seconds",
        str(timeout_seconds),
        *extra_arguments,
        supervised_tree=True,
    )
    timed_out = False
    stderr = b""
    communication_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        try:
            _, stderr = worker.communicate(timeout=timeout_seconds + 2)
        except subprocess.TimeoutExpired:
            timed_out = True
        except BaseException as exc:
            communication_error = exc
    finally:
        try:
            _terminate_worker_tree(worker)
        except BaseException as exc:
            cleanup_error = exc
    cancellation = next(
        (
            error
            for error in (communication_error, cleanup_error)
            if isinstance(error, (KeyboardInterrupt, SystemExit))
        ),
        None,
    )
    if cancellation is not None:
        secondary = cleanup_error if cancellation is communication_error else communication_error
        if secondary is not None:
            cancellation.add_note(f"secondary supervision failure: {secondary}")
        raise cancellation
    if cleanup_error is not None:
        failed = _failed_check(
            name,
            required=required,
            started=started,
            error=f"probe process cleanup failed: {cleanup_error}",
        )
        details = failed["details"]
        assert isinstance(details, dict)
        details["process_cleanup_status"] = "FAIL"
        return failed
    if communication_error is not None:
        return _failed_check(
            name,
            required=required,
            started=started,
            error=f"probe communication failed: {communication_error}",
        )
    if timed_out:
        return _failed_check(
            name,
            required=required,
            started=started,
            error="probe process exceeded its supervised deadline",
        )
    if worker.returncode != 0:
        return _failed_check(
            name,
            required=required,
            started=started,
            error=(
                f"probe worker exit={worker.returncode}: "
                f"{stderr.decode(errors='replace').strip()}"
            ),
        )
    try:
        check = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _failed_check(
            name,
            required=required,
            started=started,
            error=f"cannot read probe result: {exc}",
        )
    if not isinstance(check, dict) or check.get("name") != name:
        return _failed_check(
            name,
            required=required,
            started=started,
            error="probe result identity is invalid",
        )
    check["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
    return check


def _failed_check(
    name: str,
    *,
    required: bool,
    started: float,
    error: str,
) -> dict[str, object]:
    return {
        "name": name,
        "required": required,
        "status": "FAIL",
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "details": {"error": error},
    }


def _raise_on_process_cleanup_failure(check: dict[str, object]) -> None:
    details = check.get("details")
    if isinstance(details, dict) and details.get("process_cleanup_status") == "FAIL":
        raise StorageDrillError(
            f"fatal process cleanup failure in {check.get('name')}: {details.get('error')}"
        )


def _alias_probe_arguments(
    *,
    alias_roots: tuple[str | Path, ...],
    required_alias_classes: tuple[str, ...],
) -> tuple[str, ...]:
    arguments: list[str] = []
    for alias in alias_roots:
        arguments.extend(("--alias-root", str(alias)))
    for alias_class in required_alias_classes:
        arguments.extend(("--required-alias-class", alias_class))
    return tuple(arguments)


def _service_identity_check(
    *,
    service_identity: str,
    service_sid: str | None,
) -> dict[str, object]:
    if os.name == "nt" and not service_sid:
        return {
            "name": "service_identity_capture",
            "required": True,
            "status": "FAIL",
            "details": {"service_identity": service_identity, "service_sid": None},
        }
    return {
        "name": "service_identity_capture",
        "required": os.name == "nt",
        "status": "PASS" if os.name == "nt" else "UNSUPPORTED",
        "details": {"service_identity": service_identity, "service_sid": service_sid},
    }


def _check_crash_recovery(run_dir: Path, timeout_seconds: float) -> dict[str, object]:
    directory = run_dir / "crash-recovery"
    directory.mkdir()
    target = directory / "immutable.bin"
    payload = b"pfc-storage-drill-crash-recovery-v1\n"
    completed = subprocess.run(
        _worker_command("crash-hardlink", run_dir, "--target", str(target)),
        capture_output=True,
        env=_worker_environment(),
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != WORKER_CRASH_EXIT:
        raise StorageDrillError(
            f"crash worker exit={completed.returncode}: "
            f"{completed.stderr.decode(errors='replace').strip()}"
        )
    if target.read_bytes() != payload:
        raise StorageDrillError("crash target payload is incomplete")
    decoy = directory / f".pfc-{os.getpid()}-000000000000.tmp"
    _write_control_exclusive(decoy, b"decoy\n")
    before_links = int(target.stat().st_nlink)
    if before_links != 2:
        raise StorageDrillError(f"expected two links after crash, observed {before_links}")
    _validate_existing_immutable_file(target, label="crash-recovery artifact")
    after_links = int(target.stat().st_nlink)
    if after_links != 1:
        raise StorageDrillError(f"recovery left {after_links} hard links")
    if decoy.read_bytes() != b"decoy\n":
        raise StorageDrillError("recovery modified an unrelated matching temporary")
    target_identity = (int(target.stat().st_dev), int(target.stat().st_ino))
    decoy_identity = (int(decoy.stat().st_dev), int(decoy.stat().st_ino))
    if 0 in target_identity or target_identity == decoy_identity:
        raise StorageDrillError("filesystem identities are null or non-discriminating")
    return {
        "worker_exit": completed.returncode,
        "link_count_before_recovery": before_links,
        "link_count_after_recovery": after_links,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "target_file_identity": list(target_identity),
        "decoy_preserved": True,
    }


def _check_exclusive_contention(
    run_dir: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    directory = run_dir / "exclusive-contention"
    directory.mkdir()
    target = directory / "winner.bin"
    barrier = directory / "start.barrier"
    payloads = tuple(f"exclusive-writer-{index}\n".encode() for index in range(4))
    workers: list[subprocess.Popen[bytes]] = []
    ready_files = [directory / f"writer-{index}.ready" for index in range(len(payloads))]
    try:
        for index, payload in enumerate(payloads):
            workers.append(
                _start_worker(
                    "exclusive-write",
                    run_dir,
                    "--target",
                    str(target),
                    "--barrier",
                    str(barrier),
                    "--payload-hex",
                    payload.hex(),
                    "--ready",
                    str(ready_files[index]),
                )
            )
        deadline = time.monotonic() + timeout_seconds
        for ready, worker in zip(ready_files, workers, strict=True):
            _wait_for_file(
                ready,
                worker=worker,
                timeout_seconds=_remaining_seconds(deadline),
            )
        _write_control_exclusive(barrier, b"start\n")
        _fsync_directory(directory)
        outputs = [
            worker.communicate(timeout=_remaining_seconds(deadline))
            for worker in workers
        ]
    finally:
        _terminate_workers(workers)
    exits = [int(worker.returncode or 0) for worker in workers]
    if exits.count(0) != 1 or exits.count(WORKER_CONTENTION_LOST_EXIT) != 3:
        stderr = [item[1].decode(errors="replace").strip() for item in outputs]
        raise StorageDrillError(f"exclusive contention exits={exits}, stderr={stderr}")
    winner = target.read_bytes()
    if winner not in payloads:
        raise StorageDrillError("exclusive winner payload is not one complete candidate")
    link_count = int(target.stat().st_nlink)
    if link_count != 1:
        raise StorageDrillError(f"exclusive winner has {link_count} hard links")
    leftovers = sorted(path.name for path in directory.glob(".pfc-*.tmp"))
    if leftovers:
        raise StorageDrillError(f"exclusive writers left temporaries: {leftovers}")
    return {
        "worker_exits": exits,
        "winner_sha256": hashlib.sha256(winner).hexdigest(),
        "link_count": link_count,
    }


def _check_lock_contention(run_dir: Path, timeout_seconds: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    directory = run_dir / "lock-contention"
    directory.mkdir()
    ready = directory / "holder.ready"
    release = directory / "holder.release"
    worker = _start_worker(
        "hold-lock",
        run_dir,
        "--target",
        str(directory),
        "--ready",
        str(ready),
        "--release",
        str(release),
    )
    try:
        _wait_for_file(
            ready,
            worker=worker,
            timeout_seconds=_remaining_seconds(deadline),
        )
        _expect_lock_busy_with_zero_wait(directory)
        busy_observed = True
        _write_control_exclusive(release, b"release\n")
        _, stderr = worker.communicate(timeout=_remaining_seconds(deadline))
        if worker.returncode != 0:
            raise StorageDrillError(
                f"lock holder exit={worker.returncode}: "
                f"{stderr.decode(errors='replace').strip()}"
            )
    finally:
        _terminate_workers([worker])
    if (directory / ".promotion.lock").exists():
        raise StorageDrillError("lock artifact remained after normal release")
    with _promotion_lock(directory):
        reacquired_after_release = True

    stale_ready = directory / "stale.ready"
    stale_release = directory / "stale.release"
    stale_worker = _start_worker(
        "hold-lock",
        run_dir,
        "--target",
        str(directory),
        "--ready",
        str(stale_ready),
        "--release",
        str(stale_release),
    )
    lock_path = directory / ".promotion.lock"
    try:
        _wait_for_file(
            stale_ready,
            worker=stale_worker,
            timeout_seconds=_remaining_seconds(deadline),
        )
        stale_owner = lock_path.read_bytes()
        stale_worker.terminate()
        stale_worker.wait(timeout=_remaining_seconds(deadline))
        _expect_lock_busy_with_zero_wait(directory)
        stale_busy_observed = True
        if lock_path.read_bytes() != stale_owner:
            raise StorageDrillError("abandoned lock ownership changed")
        lock_path.unlink()
        _fsync_directory(directory)
    finally:
        _terminate_workers([stale_worker])
    return {
        "busy_observed": busy_observed,
        "holder_exit": worker.returncode,
        "reacquired_after_release": reacquired_after_release,
        "abandoned_lock_remained_busy": stale_busy_observed,
    }


def _check_atomic_visibility(run_dir: Path, timeout_seconds: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    directory = run_dir / "atomic-visibility"
    directory.mkdir()
    pointer = directory / "visibility-pointer.json"
    stop = directory / "reader.stop"
    documents = (
        {"generation": "A", "padding": "0" * 17},
        {"generation": "B", "padding": "1" * 113},
    )
    variants = tuple(_canonical_json_bytes(item) for item in documents)
    _atomic_write_json(pointer, documents[0], deadline=deadline)
    readers: list[tuple[subprocess.Popen[bytes], Path, Path, Path]] = []
    for index in range(2):
        ready = directory / f"reader-{index}.ready"
        result = directory / f"reader-{index}-result.json"
        observation_dir = directory / f"reader-{index}-observations"
        observation_dir.mkdir()
        readers.append(
            (
                _start_worker(
                    "read-atomic",
                    run_dir,
                    "--target",
                    str(pointer),
                    "--ready",
                    str(ready),
                    "--release",
                    str(stop),
                    "--result",
                    str(result),
                    "--observation-dir",
                    str(observation_dir),
                    "--allowed-hex",
                    variants[0].hex(),
                    "--allowed-hex",
                    variants[1].hex(),
                ),
                ready,
                result,
                observation_dir,
            )
        )
    replace_count = 1_000
    try:
        for worker, ready, _, _ in readers:
            _wait_for_file(
                ready,
                worker=worker,
                timeout_seconds=_remaining_seconds(deadline),
            )
        _atomic_write_json(pointer, documents[1], deadline=deadline)
        variant_b_hash = hashlib.sha256(variants[1]).hexdigest()
        for worker, _, _, observation_dir in readers:
            _wait_for_file(
                observation_dir / f"{variant_b_hash}.seen",
                worker=worker,
                timeout_seconds=_remaining_seconds(deadline),
            )
        for index in range(1, replace_count):
            _remaining_seconds(deadline)
            _atomic_write_json(
                pointer,
                documents[(index + 1) % 2],
                deadline=deadline,
            )
        time.sleep(0.05)
        _write_control_exclusive(stop, b"stop\n")
        for worker, _, _, _ in readers:
            _, stderr = worker.communicate(timeout=_remaining_seconds(deadline))
            if worker.returncode != 0:
                raise StorageDrillError(
                    f"atomic reader exit={worker.returncode}: "
                    f"{stderr.decode(errors='replace').strip()}"
                )
    finally:
        _terminate_workers([item[0] for item in readers])
    reader_observations = [
        json.loads(result.read_text(encoding="utf-8"))
        for _, _, result, _ in readers
    ]
    for observations in reader_observations:
        if observations.get("invalid_reads") != 0 or observations.get("read_errors") != 0:
            raise StorageDrillError(f"non-atomic reader observations: {observations}")
        variant_reads = observations.get("variant_reads")
        if not isinstance(variant_reads, dict) or any(
            int(variant_reads.get(hashlib.sha256(value).hexdigest(), 0)) < 1
            for value in variants
        ):
            raise StorageDrillError(
                f"reader did not observe every atomic generation: {variant_reads}"
            )
    return {"replace_count": replace_count, "readers": reader_observations}


def _check_directory_replace(run_dir: Path, timeout_seconds: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    directory = run_dir / "directory-replace"
    source = directory / "source-tree"
    destination = directory / "published-tree"
    nested = source / "nested"
    nested.mkdir(parents=True)
    expected = {
        "root.txt": b"directory-replace-root-v1\n",
        "nested/payload.bin": b"\x00\x01directory-replace-payload\xff",
    }
    for relative, payload in expected.items():
        path = source / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_control_exclusive(path, payload)
    _fsync_tree(source)

    ready = directory / "reader.ready"
    stop = directory / "reader.stop"
    result = directory / "reader-result.json"
    observations = directory / "observations"
    observations.mkdir()
    arguments: list[str] = []
    for relative, payload in expected.items():
        arguments.extend(("--tree-file", f"{relative}={payload.hex()}"))
    worker = _start_worker(
        "read-tree",
        run_dir,
        "--target",
        str(destination),
        "--ready",
        str(ready),
        "--release",
        str(stop),
        "--result",
        str(result),
        "--observation-dir",
        str(observations),
        *arguments,
    )
    try:
        _wait_for_file(
            ready,
            worker=worker,
            timeout_seconds=_remaining_seconds(deadline),
        )
        _replace_directory(source, destination)
        _wait_for_file(
            observations / "complete.seen",
            worker=worker,
            timeout_seconds=_remaining_seconds(deadline),
        )
        _write_control_exclusive(stop, b"stop\n")
        _, stderr = worker.communicate(timeout=_remaining_seconds(deadline))
        if worker.returncode != 0:
            raise StorageDrillError(
                f"directory reader exit={worker.returncode}: "
                f"{stderr.decode(errors='replace').strip()}"
            )
    finally:
        _terminate_workers([worker])
    reader_result = json.loads(result.read_text(encoding="utf-8"))
    if int(reader_result.get("absent_observations", 0)) < 1:
        raise StorageDrillError("directory reader did not observe pre-publication absence")
    if reader_result.get("partial_observations") != 0:
        raise StorageDrillError(f"partial directory publication: {reader_result}")
    if int(reader_result.get("complete_observations", 0)) < 1:
        raise StorageDrillError("published directory was never observed complete")
    if source.exists() or not destination.is_dir():
        raise StorageDrillError("directory replacement left an invalid source/destination")
    return {"expected_files": sorted(expected), **reader_result}


def _capture_alias_check(
    *,
    root: Path,
    run_dir: Path,
    run_id: str,
    alias_roots: tuple[str | Path, ...],
    required_alias_classes: tuple[str, ...],
    timeout_seconds: float,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    allowed_classes = {"drive", "unc_short", "unc_fqdn"}
    invalid_classes = sorted(set(required_alias_classes).difference(allowed_classes))
    if invalid_classes:
        return {
            "name": "alias_samefile_and_visibility",
            "required": True,
            "status": "FAIL",
            "details": {"error": f"invalid required alias classes: {invalid_classes}"},
        }
    if not alias_roots:
        missing = sorted(set(required_alias_classes))
        return {
            "name": "alias_samefile_and_visibility",
            "required": bool(missing),
            "status": "FAIL" if missing else "UNSUPPORTED",
            "details": {
                "reason": "no --alias-root was configured",
                "missing_required_alias_classes": missing,
            },
        }
    details: list[dict[str, object]] = []
    primary_lexical = os.path.normcase(os.path.abspath(str(root)))
    observed_classes = {_alias_class(str(root))}
    try:
        sentinel = run_dir / "alias-visibility.sentinel"
        payload = uuid.uuid4().hex.encode("ascii") + b"\n"
        _write_bytes_exclusive_fsync(sentinel, payload)
        for value in alias_roots:
            _remaining_seconds(deadline)
            alias_lexical = os.path.normcase(os.path.abspath(str(value)))
            if alias_lexical == primary_lexical:
                raise StorageDrillError("alias must be lexically distinct from drill_root")
            alias = _validate_drill_root(value)
            alias_class = _alias_class(str(value))
            if alias_class not in allowed_classes:
                raise StorageDrillError(f"invalid deployment alias spelling: {value}")
            observed_classes.add(alias_class)
            same_file = os.path.samefile(root, alias)
            alias_sentinel = alias / DRILL_NAMESPACE / run_id / sentinel.name
            visible = alias_sentinel.is_file() and alias_sentinel.read_bytes() == payload
            details.append(
                {
                    "alias_root": str(alias),
                    "alias_class": alias_class,
                    "samefile": same_file,
                    "sentinel_visible": visible,
                }
            )
            if not same_file or not visible:
                raise StorageDrillError(f"alias does not identify the drill root: {alias}")
            primary_alias_dir = run_dir / "alias-cross-path"
            if not primary_alias_dir.exists():
                primary_alias_dir.mkdir()
            secondary_alias_dir = alias / DRILL_NAMESPACE / run_id / "alias-cross-path"
            with _promotion_lock(primary_alias_dir):
                _expect_lock_busy_with_zero_wait(secondary_alias_dir)
            alias_pointer = secondary_alias_dir / "alias-pointer.json"
            _remaining_seconds(deadline)
            _atomic_write_json(
                alias_pointer,
                {"writer": "alias"},
                deadline=deadline,
            )
            primary_pointer = primary_alias_dir / "alias-pointer.json"
            if json.loads(primary_pointer.read_text(encoding="utf-8")) != {
                "writer": "alias"
            }:
                raise StorageDrillError("cross-alias atomic write was not visible")
        missing = sorted(set(required_alias_classes).difference(observed_classes))
        if missing:
            raise StorageDrillError(f"required alias classes are missing: {missing}")
    except Exception as exc:
        return {
            "name": "alias_samefile_and_visibility",
            "required": True,
            "status": "FAIL",
            "details": {"aliases": details, "error": f"{type(exc).__name__}: {exc}"},
        }
    return {
        "name": "alias_samefile_and_visibility",
        "required": True,
        "status": "PASS",
        "details": {
            "aliases": details,
            "observed_alias_classes": sorted(observed_classes),
            "required_alias_classes": sorted(set(required_alias_classes)),
        },
    }


def _worker_command(operation: str, run_dir: Path, *arguments: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pfc_shaping.pipeline.storage_drill",
        "_worker",
        operation,
        "--run-dir",
        str(run_dir),
        *arguments,
    ]


def _start_worker(
    operation: str,
    run_dir: Path,
    *arguments: str,
    supervised_tree: bool = False,
) -> subprocess.Popen[bytes]:
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    if os.name == "nt" and supervised_tree:
        creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(
        _worker_command(operation, run_dir, *arguments),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_worker_environment(job_owner=supervised_tree),
        creationflags=creationflags,
        start_new_session=supervised_tree and os.name != "nt",
    )


def _worker_environment(*, job_owner: bool = False) -> dict[str, str]:
    environment = os.environ.copy()
    repository_root = str(Path(__file__).resolve().parents[2])
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        repository_root if not current else os.pathsep.join((repository_root, current))
    )
    if job_owner and os.name == "nt":
        environment[JOB_OWNER_ENV] = "1"
    return environment


def _terminate_workers(workers: Iterable[subprocess.Popen[bytes]]) -> None:
    for worker in workers:
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)


def _terminate_worker_tree(worker: subprocess.Popen[bytes]) -> None:
    if os.name == "nt" and worker.poll() is not None:
        # The supervised child owns a KILL_ON_JOB_CLOSE Job Object.
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(worker.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except (OSError, subprocess.SubprocessError):
            worker.kill()
    else:
        _terminate_posix_process_group(worker)
        return
    try:
        worker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            worker.kill()
        worker.wait(timeout=5)


def _terminate_posix_process_group(worker: subprocess.Popen[bytes]) -> None:
    process_group = worker.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        if worker.poll() is None:
            worker.wait(timeout=5)
        return
    grace_deadline = time.monotonic() + 1
    while time.monotonic() < grace_deadline and _process_group_exists(process_group):
        time.sleep(0.02)
    if _process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if worker.poll() is None:
        worker.wait(timeout=5)
    drain_deadline = time.monotonic() + 5
    while time.monotonic() < drain_deadline and _process_group_exists(process_group):
        time.sleep(0.02)
    if _process_group_exists(process_group):
        raise StorageDrillError(
            f"supervised POSIX process group did not terminate: {process_group}"
        )


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_file(
    path: Path,
    *,
    worker: subprocess.Popen[bytes],
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        exit_code = worker.poll()
        if exit_code is not None:
            _, stderr = worker.communicate()
            raise StorageDrillError(
                f"worker exited before readiness ({exit_code}): "
                f"{stderr.decode(errors='replace').strip()}"
            )
        if time.monotonic() >= deadline:
            raise StorageDrillError(f"worker readiness timed out: {path}")
        time.sleep(0.02)


def _worker_main(args: argparse.Namespace) -> int:
    run_dir = _validate_worker_run_dir(args.run_dir)
    if args.operation == "run-probe":
        return _run_probe_worker(run_dir, args)
    if args.operation in {"test-hang-tree", "test-hang-tree-ignore-term"}:
        if not args.result:
            raise StorageDrillError("test heartbeat path is required")
        heartbeat = _worker_path(run_dir, args.result)
        ignore_term = args.operation == "test-hang-tree-ignore-term"
        script = (
            "from pathlib import Path; import signal, sys, time; "
            "p=Path(sys.argv[1]); "
            + (
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                if ignore_term
                else ""
            )
            +
            "\nwhile True:\n"
            " p.open('ab').write(b'x'); time.sleep(0.02)\n"
        )
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        subprocess.Popen(
            [sys.executable, "-c", script, str(heartbeat)],
            creationflags=creationflags,
        )
        time.sleep(60)
        return 0
    target = _worker_path(run_dir, args.target) if args.target else None
    if args.operation == "crash-hardlink":
        assert target is not None
        payload = b"pfc-storage-drill-crash-recovery-v1\n"

        def crash_after_link(_temporary: Path, _target: Path) -> None:
            os._exit(WORKER_CRASH_EXIT)

        _write_bytes_exclusive_fsync_with_post_link(
            target,
            payload,
            post_link=crash_after_link,
        )
        raise StorageDrillError("crash hook returned unexpectedly")
    if args.operation == "exclusive-write":
        assert target is not None and args.barrier and args.payload_hex and args.ready
        barrier = _worker_path(run_dir, args.barrier)
        ready = _worker_path(run_dir, args.ready)
        _write_control_exclusive(ready, b"ready\n")
        _worker_wait_for_path(barrier)
        try:
            _write_bytes_exclusive_fsync(target, bytes.fromhex(args.payload_hex))
        except ImmutableArtifactExistsError:
            return WORKER_CONTENTION_LOST_EXIT
        except PromotionError as exc:
            print(f"unexpected exclusive-write failure: {exc}", file=sys.stderr)
            return 18
        return 0
    if args.operation == "hold-lock":
        assert target is not None and args.ready and args.release
        ready = _worker_path(run_dir, args.ready)
        release = _worker_path(run_dir, args.release)
        with _promotion_lock(target):
            _write_control_exclusive(ready, b"ready\n")
            _worker_wait_for_path(release)
        return 0
    if args.operation == "read-atomic":
        assert (
            target is not None
            and args.ready
            and args.release
            and args.result
            and args.observation_dir
        )
        ready = _worker_path(run_dir, args.ready)
        stop = _worker_path(run_dir, args.release)
        result = _worker_path(run_dir, args.result)
        observation_dir = _worker_path(run_dir, args.observation_dir)
        allowed = {
            hashlib.sha256(bytes.fromhex(value)).hexdigest(): bytes.fromhex(value)
            for value in args.allowed_hex
        }
        observations: dict[str, object] = {
            "valid_reads": 0,
            "invalid_reads": 0,
            "read_errors": 0,
            "variant_reads": {key: 0 for key in allowed},
        }
        ready_published = False
        seen: set[str] = set()
        while not stop.exists():
            try:
                payload = _read_bytes_with_retry(target)
            except (OSError, PromotionError):
                observations["read_errors"] = int(observations["read_errors"]) + 1
            else:
                matching_hash = next(
                    (key for key, value in allowed.items() if payload == value),
                    None,
                )
                if matching_hash is None:
                    observations["invalid_reads"] = int(observations["invalid_reads"]) + 1
                else:
                    observations["valid_reads"] = int(observations["valid_reads"]) + 1
                    variant_reads = observations["variant_reads"]
                    assert isinstance(variant_reads, dict)
                    variant_reads[matching_hash] = int(variant_reads[matching_hash]) + 1
                    if matching_hash not in seen:
                        _write_control_exclusive(
                            observation_dir / f"{matching_hash}.seen",
                            b"seen\n",
                        )
                        seen.add(matching_hash)
                    if not ready_published:
                        _write_control_exclusive(ready, b"ready\n")
                        ready_published = True
            time.sleep(0.001)
        _write_control_exclusive(
            result,
            (json.dumps(observations, sort_keys=True) + "\n").encode("utf-8"),
        )
        return 0
    if args.operation == "read-tree":
        assert (
            target is not None
            and args.ready
            and args.release
            and args.result
            and args.observation_dir
        )
        ready = _worker_path(run_dir, args.ready)
        stop = _worker_path(run_dir, args.release)
        result = _worker_path(run_dir, args.result)
        observation_dir = _worker_path(run_dir, args.observation_dir)
        expected: dict[str, bytes] = {}
        for item in args.tree_file:
            relative, separator, encoded = item.partition("=")
            relative_path = Path(relative)
            if not separator or relative_path.is_absolute() or ".." in relative_path.parts:
                raise StorageDrillError(f"invalid expected tree file: {item!r}")
            expected[relative_path.as_posix()] = bytes.fromhex(encoded)
        observations = {
            "absent_observations": 0,
            "complete_observations": 0,
            "partial_observations": 0,
        }
        complete_published = False
        absent_published = False
        while not stop.exists():
            if not target.exists():
                observations["absent_observations"] += 1
                if not absent_published:
                    _write_control_exclusive(
                        observation_dir / "absent.seen",
                        b"seen\n",
                    )
                    _write_control_exclusive(ready, b"ready\n")
                    absent_published = True
            elif not target.is_dir():
                observations["partial_observations"] += 1
            else:
                observed_paths = {
                    path.relative_to(target).as_posix()
                    for path in target.rglob("*")
                    if path.is_file()
                }
                complete = observed_paths == set(expected) and all(
                    (target / Path(relative)).read_bytes() == payload
                    for relative, payload in expected.items()
                )
                key = "complete_observations" if complete else "partial_observations"
                observations[key] += 1
                if complete and not complete_published:
                    _write_control_exclusive(
                        observation_dir / "complete.seen",
                        b"seen\n",
                    )
                    complete_published = True
            time.sleep(0)
        _write_control_exclusive(
            result,
            (json.dumps(observations, sort_keys=True) + "\n").encode("utf-8"),
        )
        return 0
    raise StorageDrillError(f"unknown worker operation: {args.operation}")


def _run_probe_worker(run_dir: Path, args: argparse.Namespace) -> int:
    if not args.probe_name or not args.result or args.timeout_seconds is None:
        raise StorageDrillError("supervised probe arguments are incomplete")
    timeout_seconds = float(args.timeout_seconds)
    probes: dict[str, Callable[[], dict[str, object]]] = {
        "crash_after_hardlink_recovery": lambda: _check_crash_recovery(
            run_dir,
            timeout_seconds,
        ),
        "multi_process_exclusive_hardlink_contention": lambda: (
            _check_exclusive_contention(run_dir, timeout_seconds)
        ),
        "transition_lock_contention": lambda: _check_lock_contention(
            run_dir,
            timeout_seconds,
        ),
        "atomic_replace_visibility": lambda: _check_atomic_visibility(
            run_dir,
            timeout_seconds,
        ),
        "directory_finalize_replace_visibility": lambda: _check_directory_replace(
            run_dir,
            timeout_seconds,
        ),
        "artifact_inventory_integrity": lambda: {
            "artifacts": _artifact_inventory(run_dir)
        },
    }
    if args.probe_name == "alias_samefile_and_visibility":
        check = _capture_alias_check(
            root=run_dir.parent.parent,
            run_dir=run_dir,
            run_id=run_dir.name,
            alias_roots=tuple(args.alias_root),
            required_alias_classes=tuple(args.required_alias_class),
            timeout_seconds=timeout_seconds,
        )
    else:
        operation = probes.get(args.probe_name)
        if operation is None:
            raise StorageDrillError(f"unknown supervised probe: {args.probe_name}")
        check = _capture_check(args.probe_name, required=True, operation=operation)
    result = _worker_path(run_dir, args.result)
    _write_control_exclusive(result, _canonical_json_bytes(check))
    return 0


def _worker_path(run_dir: Path, value: str | Path) -> Path:
    selected = Path(value)
    if not selected.is_absolute():
        raise StorageDrillError("worker artifact paths must be absolute")
    lexical = Path(os.path.abspath(selected))
    try:
        lexical.relative_to(run_dir)
    except ValueError as exc:
        raise StorageDrillError(f"worker path escapes the run directory: {lexical}") from exc
    if _path_is_link(lexical):
        raise StorageDrillError(f"worker path cannot be a link: {lexical}")
    if lexical.is_file() and int(lexical.stat().st_nlink) != 1:
        raise StorageDrillError(f"worker file cannot have hardlinks: {lexical}")
    try:
        _assert_path_has_no_links(lexical.parent, label="worker artifact parent")
    except PromotionError as exc:
        raise StorageDrillError(str(exc)) from exc
    return lexical


def _worker_wait_for_path(path: Path) -> None:
    deadline = time.monotonic() + DEFAULT_TIMEOUT_SECONDS
    while not path.exists():
        if time.monotonic() >= deadline:
            raise StorageDrillError(f"worker timed out waiting for {path}")
        time.sleep(0.02)


def _expect_lock_busy_with_zero_wait(root: Path) -> None:
    previous = os.environ.get(PROMOTION_LOCK_WAIT_SECONDS_ENV)
    os.environ[PROMOTION_LOCK_WAIT_SECONDS_ENV] = "0"
    try:
        try:
            with _promotion_lock(root):
                raise StorageDrillError("contender acquired a held lock")
        except PromotionBusyError:
            return
    finally:
        if previous is None:
            os.environ.pop(PROMOTION_LOCK_WAIT_SECONDS_ENV, None)
        else:
            os.environ[PROMOTION_LOCK_WAIT_SECONDS_ENV] = previous


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise StorageDrillError("storage drill check exceeded its deadline")
    return remaining


def _alias_class(value: str) -> str:
    if re.match(r"^[A-Za-z]:[\\/]", value):
        return "drive"
    normalized = value.replace("/", "\\")
    if normalized.startswith(("\\\\?\\", "\\\\.\\")):
        return "invalid"
    if normalized.startswith("\\\\"):
        parts = normalized[2:].split("\\")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            return "invalid"
        host, share = parts[0], parts[1]
        if not re.fullmatch(r"[^\x00-\x1f\\/:*?\"<>|]+", share):
            return "invalid"
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            return "invalid"
        labels = host.split(".")
        if any(
            not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
            for label in labels
        ):
            return "invalid"
        return "unc_fqdn" if len(labels) >= 2 else "unc_short"
    return "other"


def _default_run_id() -> str:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{timestamp}-{uuid.uuid4().hex[:12]}"


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _service_identity() -> str:
    domain = os.environ.get("USERDOMAIN", "").strip()
    user = getpass.getuser()
    return f"{domain}\\{user}" if domain else user


def _service_sid() -> str | None:
    if os.name != "nt":
        return None
    creationflags = subprocess.CREATE_NO_WINDOW
    try:
        completed = subprocess.run(
            ["whoami", "/user", "/fo", "csv", "/nh"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
            creationflags=creationflags,
        )
        row = next(csv.reader([completed.stdout.strip()]))
    except (OSError, subprocess.SubprocessError, StopIteration, csv.Error):
        return None
    return row[1].strip() if len(row) >= 2 else None


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_report_exclusive(path: Path, payload: bytes) -> None:
    """Archive a report even when the target volume has failed hardlink checks."""

    _write_control_exclusive(path, payload)


def _write_control_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _artifact_inventory(root: Path) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            path = Path(entry.path)
            if _path_is_link(path):
                raise StorageDrillError(f"drill artifact cannot be a link: {path}")
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
                continue
            if not entry.is_file(follow_symlinks=False):
                raise StorageDrillError(f"drill artifact is not a regular file: {path}")
            if int(path.stat().st_nlink) != 1:
                raise StorageDrillError(f"drill artifact has forbidden hardlinks: {path}")
            payload = path.read_bytes()
            inventory.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return sorted(inventory, key=lambda item: str(item["path"]))


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _parse_worker_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("operation")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--target")
    parser.add_argument("--barrier")
    parser.add_argument("--payload-hex")
    parser.add_argument("--ready")
    parser.add_argument("--release")
    parser.add_argument("--result")
    parser.add_argument("--observation-dir")
    parser.add_argument("--allowed-hex", action="append", default=[])
    parser.add_argument("--tree-file", action="append", default=[])
    parser.add_argument("--probe-name")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--alias-root", action="append", default=[])
    parser.add_argument("--required-alias-class", action="append", default=[])
    return parser.parse_args(argv)


def _module_main(argv: list[str] | None = None) -> int:
    if os.environ.get(JOB_OWNER_ENV) == "1":
        _enter_windows_kill_on_close_job()
    selected = list(sys.argv[1:] if argv is None else argv)
    if not selected or selected[0] != "_worker":
        raise StorageDrillError("storage_drill module only exposes its internal worker")
    return _worker_main(_parse_worker_args(selected[1:]))


def _enter_windows_kill_on_close_job() -> None:
    """Own this process tree with a Windows Job Object before spawning children."""

    global _WINDOWS_JOB_HANDLE
    if os.name != "nt" or _WINDOWS_JOB_HANDLE is not None:
        return
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
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise StorageDrillError(
            f"CreateJobObjectW failed with WinError {ctypes.get_last_error()}"
        )
    information = ExtendedLimitInformation()
    information.basic_limit_information.limit_flags = 0x00002000
    configured = kernel32.SetInformationJobObject(
        job,
        9,
        ctypes.byref(information),
        ctypes.sizeof(information),
    )
    assigned = configured and kernel32.AssignProcessToJobObject(
        job,
        kernel32.GetCurrentProcess(),
    )
    if not assigned:
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise StorageDrillError(f"Windows Job Object setup failed with WinError {error}")
    _WINDOWS_JOB_HANDLE = job


if __name__ == "__main__":
    try:
        raise SystemExit(_module_main())
    except StorageDrillError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

"""Crash-recoverable publication of exact local evidence bytes."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)


class DurableArtifactExistsError(ValueError):
    """Raised when an immutable destination already contains different bytes."""


class DurableArtifactRepairRequired(ValueError):
    """Raised when an unowned residue requires an explicit operator decision."""

    def __init__(self, result: Mapping[str, object]) -> None:
        self.result = dict(result)
        super().__init__("durable artifact directory requires explicit repair")


def durable_artifact_durability_classification() -> str:
    """Describe the strongest durability property implemented on this platform."""

    if os.name == "nt":
        return "PROCESS_CRASH_RECOVERABLE_POWER_LOSS_UNPROVEN_ON_WINDOWS"
    return "PROCESS_CRASH_AND_DIRECTORY_FSYNC_DURABLE"


def validated_durable_target(path: Path, *, label: str) -> Path:
    target = path.expanduser()
    if not target.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    target = assert_absolute_path_has_no_links(target)
    parent = assert_absolute_path_has_no_links(target.parent)
    if not parent.is_dir():
        raise ValueError(f"{label} directory is unavailable")
    return target


def write_durable_exact_bytes(
    path: Path,
    payload: bytes,
    *,
    label: str,
    replace_existing_if: Callable[[bytes], bool] | None = None,
) -> Path:
    target = validated_durable_target(path, label=label)
    _recover_link_residue(target, label=label)
    temporary_prefix = _temporary_prefix(target)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=temporary_prefix,
        suffix=".tmp",
        dir=target.parent,
    )
    temporary: Path | None = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
            _fsync_directory(target.parent)
        except FileExistsError:
            if _resolve_existing_target(
                target,
                temporary,
                payload,
                label=label,
                replace_existing_if=replace_existing_if,
            ):
                temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
            _fsync_directory(target.parent)
    return target


def recover_durable_exact_bytes_residue(
    path: Path,
    payload: bytes,
    *,
    label: str,
    exclusive_writer_lease_proven: bool,
) -> Path:
    """Recover only a provably exact residue left by this durable writer."""

    if exclusive_writer_lease_proven is not True:
        raise ValueError(f"{label} recovery requires a proven exclusive writer lease")
    target = validated_durable_target(path, label=label)
    _recover_exact_writer_residue(target, payload=payload, label=label)
    return target


def validate_durable_file(path: Path, *, label: str) -> None:
    read_stable_single_link_file(path, label=label)


def prepare_durable_json_directory(directory: Path, *, label: str) -> None:
    selected = assert_absolute_path_has_no_links(directory)
    if not selected.is_dir():
        raise ValueError(f"{label} is not a regular directory")
    for target in sorted(selected.glob("*.json"), key=lambda path: path.name):
        _recover_link_residue(target, label=label)
        validate_durable_file(target, label=label)
    for temporary in sorted(selected.glob(".*.tmp"), key=lambda path: path.name):
        if not temporary.is_file() or int(temporary.stat().st_nlink) != 1:
            raise ValueError(f"{label} contains ambiguous temporary residue")
        raise DurableArtifactRepairRequired(
            _repair_result(
                reason="UNOWNED_TEMPORARY_RESIDUE",
                path=temporary,
                operator_action="VERIFY_WRITER_STOPPED_THEN_QUARANTINE_RESIDUE",
            )
        )
    repair_locks = sorted(selected.glob(".pfc-repair-*.lock"), key=lambda path: path.name)
    if repair_locks:
        raise DurableArtifactRepairRequired(
            _repair_result(
                reason="UNOWNED_REPAIR_LOCK",
                path=repair_locks[0],
                operator_action="VERIFY_WRITER_STOPPED_THEN_REMOVE_REPAIR_LOCK",
            )
        )
    repair_candidates = selected / ".pfc-repair-candidates"
    if repair_candidates.exists():
        repair_candidates = assert_absolute_path_has_no_links(repair_candidates)
        candidates = sorted(repair_candidates.glob("*.repair-candidate"))
        if (
            not repair_candidates.is_dir()
            or not candidates
            or any(
                not candidate.is_file()
                or int(candidate.stat(follow_symlinks=False).st_nlink) != 1
                for candidate in candidates
            )
            or any(
                entry not in candidates for entry in repair_candidates.iterdir()
            )
        ):
            raise ValueError(f"{label} contains an invalid repair candidate inventory")
        raise DurableArtifactRepairRequired(
            _repair_result(
                reason="PENDING_REPAIR_CANDIDATE",
                path=candidates[0],
                operator_action=(
                    "VERIFY_AUTHORITY_AND_CURRENT_TARGET_THEN_PROMOTE_CANDIDATE_WITH_EXTERNAL_CAS"
                ),
                repair_candidate_path=str(candidates[0]),
                repair_candidate_sha256=hashlib.sha256(
                    read_stable_single_link_file(
                        candidates[0], label=f"{label} repair candidate"
                    )
                ).hexdigest(),
                automatic_replacement_performed=False,
            )
        )
    entries = list(selected.iterdir())
    if any(path.suffix != ".json" for path in entries):
        raise ValueError(f"{label} contains an unexpected entry")
    for artifact in entries:
        validate_durable_file(artifact, label=label)


def _recover_link_residue(target: Path, *, label: str) -> None:
    if not target.exists():
        return
    try:
        target_stat = target.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"cannot inspect {label}") from exc
    if int(target_stat.st_nlink) == 1:
        return
    matching = []
    candidates = {
        *target.parent.glob(f"{_temporary_prefix(target)}*.tmp"),
        *target.parent.glob(f".{target.name}.*.tmp"),
    }
    for candidate in candidates:
        try:
            candidate_stat = candidate.stat(follow_symlinks=False)
        except OSError:
            continue
        if (
            candidate_stat.st_dev == target_stat.st_dev
            and candidate_stat.st_ino == target_stat.st_ino
        ):
            matching.append(candidate)
    if int(target_stat.st_nlink) != 2 or len(matching) != 1:
        raise ValueError(f"{label} has ambiguous hardlink residue")
    matching[0].unlink()
    _fsync_directory(target.parent)
    if int(target.stat(follow_symlinks=False).st_nlink) != 1:
        raise ValueError(f"{label} hardlink recovery failed")


def _recover_exact_writer_residue(
    target: Path,
    *,
    payload: bytes,
    label: str,
) -> None:
    candidates = {
        *target.parent.glob(f"{_temporary_prefix(target)}*.tmp"),
        *target.parent.glob(f".{target.name}.*.tmp"),
    }
    if target.exists():
        target_metadata = target.stat(follow_symlinks=False)
        link_count = int(target_metadata.st_nlink)
        if link_count == 2:
            matching = {
                candidate
                for candidate in candidates
                if _same_file_identity(candidate, target_metadata)
            }
            if len(matching) != 1 or matching != candidates:
                raise ValueError(f"{label} has ambiguous hardlink residue")
            candidate = matching.pop()
            if _read_stable_regular_file_with_link_count(
                target,
                expected_link_count=2,
                max_bytes=len(payload),
                label=label,
            ) != payload:
                raise ValueError(f"{label} has divergent hardlink residue")
            candidate.unlink()
            _fsync_directory(target.parent)
            if int(target.stat(follow_symlinks=False).st_nlink) != 1:
                raise ValueError(f"{label} hardlink recovery failed")
            return
        if link_count != 1:
            raise ValueError(f"{label} has ambiguous hardlink residue")
        if not candidates:
            return
        if read_stable_single_link_file(
            target,
            label=label,
            max_bytes=len(payload),
        ) != payload:
            raise ValueError(f"{label} target diverges from temporary residue")
    elif not candidates:
        return
    if len(candidates) != 1:
        raise ValueError(f"{label} has ambiguous temporary residue")
    candidate = candidates.pop()
    try:
        candidate_payload = read_stable_single_link_file(
            candidate,
            label=f"{label} temporary residue",
            max_bytes=len(payload),
        )
    except ValueError as exc:
        raise ValueError(f"{label} has divergent temporary residue") from exc
    if candidate_payload != payload:
        raise ValueError(f"{label} has divergent temporary residue")
    candidate.unlink()
    _fsync_directory(target.parent)


def _same_file_identity(candidate: Path, target_metadata: os.stat_result) -> bool:
    try:
        candidate_metadata = candidate.stat(follow_symlinks=False)
    except OSError:
        return False
    return (
        candidate_metadata.st_dev == target_metadata.st_dev
        and candidate_metadata.st_ino == target_metadata.st_ino
    )


def _read_stable_regular_file_with_link_count(
    path: Path,
    *,
    expected_link_count: int,
    max_bytes: int,
    label: str,
) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != expected_link_count
            or int(before.st_size) > max_bytes
        ):
            raise ValueError(f"{label} hardlink residue is invalid")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > max_bytes:
            raise ValueError(f"{label} hardlink residue exceeds the maximum size")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ValueError(f"{label} hardlink residue changed during read")
    return payload


def _resolve_existing_target(
    target: Path,
    temporary: Path,
    payload: bytes,
    *,
    label: str,
    replace_existing_if: Callable[[bytes], bool] | None,
) -> bool:
    if not target.exists():
        os.link(temporary, target)
        _fsync_directory(target.parent)
        return False
    existing = read_stable_single_link_file(target, label=label)
    if existing == payload:
        return False
    if replace_existing_if is None or not replace_existing_if(existing):
        raise DurableArtifactExistsError(f"{label} already exists with other bytes")
    candidate = _stage_repair_candidate(target, payload, label=label)
    raise DurableArtifactRepairRequired(
        _repair_result(
            reason="AUTHORITATIVE_BYTES_DIFFER_FROM_EXISTING_TARGET",
            path=target,
            operator_action=(
                "VERIFY_AUTHORITY_AND_CURRENT_TARGET_THEN_PROMOTE_CANDIDATE_WITH_EXTERNAL_CAS"
            ),
            target_path=str(target),
            observed_target_sha256=hashlib.sha256(existing).hexdigest(),
            repair_candidate_path=str(candidate),
            repair_candidate_sha256=hashlib.sha256(payload).hexdigest(),
            authoritative_sha256=hashlib.sha256(payload).hexdigest(),
            automatic_replacement_performed=False,
        )
    )


def _temporary_prefix(target: Path) -> str:
    return f".pfc-{_target_name_digest(target)}-"


def _target_name_digest(target: Path) -> str:
    return hashlib.sha256(target.name.encode("utf-8")).hexdigest()[:12]


def _repair_result(
    *,
    reason: str,
    path: Path,
    operator_action: str,
    **evidence: object,
) -> dict[str, object]:
    return {
        "schema_version": "durable_artifact_repair.v2",
        "status": "LOCAL_ARTIFACT_REPAIR_REQUIRED",
        "reason": reason,
        "path": str(path),
        "operator_action": operator_action,
        "automatic_deletion_performed": False,
        "durability_classification": durable_artifact_durability_classification(),
        **evidence,
    }


def _stage_repair_candidate(target: Path, payload: bytes, *, label: str) -> Path:
    candidate_directory = target.parent / ".pfc-repair-candidates"
    candidate_directory.mkdir(mode=0o700, exist_ok=True)
    _fsync_directory(target.parent)
    candidate_directory = assert_absolute_path_has_no_links(candidate_directory)
    digest = hashlib.sha256(payload).hexdigest()
    candidate = candidate_directory / (
        f"{_target_name_digest(target)}-{digest}.repair-candidate"
    )
    return write_durable_exact_bytes(
        candidate,
        payload,
        label=f"authoritative repair candidate for {label}",
    )


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        # Python exposes no portable Windows directory flush. Production
        # qualification therefore requires an exact-volume power-loss drill.
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

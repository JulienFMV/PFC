"""Cross-platform filesystem path safety primitives."""

from __future__ import annotations

import os
import stat
import threading
from pathlib import Path
from typing import Iterator

_PROCESS_IMMUTABLE_LOCK = threading.RLock()
_PROCESS_IMMUTABLE_FILES: tuple[tuple[str, bytes], ...] = ()
_PROCESS_IMMUTABLE_DIRECTORIES: tuple[tuple[str, tuple[str, ...]], ...] = ()


def path_is_link(path: Path) -> bool:
    """Detect symlinks and Windows reparse points, including on Python 3.11."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    return os.name == "nt" and bool(attributes & 0x400)


def assert_absolute_path_has_no_links(path: str | Path) -> Path:
    """Return one lexical absolute path after rejecting links/reparse points."""

    raw = os.fspath(path)
    windows_path = raw.replace("/", "\\").lower()
    if os.name == "nt" and windows_path.startswith(("\\\\.\\", "\\\\?\\")):
        raise ValueError("Windows device namespaces are forbidden")
    lexical = Path(path)
    if not lexical.is_absolute():
        raise ValueError("path must be absolute")
    lexical = Path(os.path.abspath(lexical))
    for component in (lexical, *lexical.parents):
        try:
            if path_is_link(component):
                raise ValueError(f"path contains a link or reparse point: {component}")
        except OSError as exc:
            raise ValueError(f"cannot validate path component: {component}") from exc
    return lexical


def read_stable_single_link_file(
    path: str | Path,
    *,
    label: str,
    max_bytes: int | None = None,
) -> bytes:
    """Read exact bytes through a checked descriptor without following links."""

    if max_bytes is not None and (
        not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0
    ):
        raise ValueError(f"{label} maximum size is invalid")
    frozen = process_immutable_file_bytes(path)
    if frozen is not None:
        if max_bytes is not None and len(frozen) > max_bytes:
            raise ValueError(f"{label} exceeds the maximum size")
        return frozen
    return _read_unfrozen_stable_single_link_file(
        path,
        label=label,
        max_bytes=max_bytes,
    )


def register_process_immutable_file_bytes(
    path: str | Path,
    payload: bytes,
    *,
    label: str,
) -> Path:
    """Bind one verified path to immutable bytes for the process lifetime."""

    global _PROCESS_IMMUTABLE_FILES
    lexical = assert_absolute_path_has_no_links(path)
    immutable = bytes(payload)
    if _read_unfrozen_stable_single_link_file(lexical, label=label) != immutable:
        raise ValueError(f"{label} bytes differ before immutable registration")
    key = _process_path_key(lexical)
    with _PROCESS_IMMUTABLE_LOCK:
        existing = dict(_PROCESS_IMMUTABLE_FILES).get(key)
        if existing is not None and existing != immutable:
            raise ValueError(f"{label} immutable process binding already differs")
        if existing is None:
            _PROCESS_IMMUTABLE_FILES = (*_PROCESS_IMMUTABLE_FILES, (key, immutable))
    return lexical


def freeze_process_immutable_file(path: str | Path, *, label: str) -> Path:
    """Capture a stable mono-linked file once and register its immutable bytes."""

    lexical = assert_absolute_path_has_no_links(path)
    payload = _read_unfrozen_stable_single_link_file(lexical, label=label)
    return register_process_immutable_file_bytes(lexical, payload, label=label)


def freeze_process_immutable_directory(
    path: str | Path,
    *,
    label: str,
    required_suffix: str = ".pem",
) -> tuple[Path, ...]:
    """Capture and seal one flat immutable file inventory for process lifetime."""

    global _PROCESS_IMMUTABLE_DIRECTORIES
    directory = assert_absolute_path_has_no_links(path)
    if not directory.is_dir():
        raise ValueError(f"{label} is unavailable")
    entries = tuple(sorted(directory.iterdir(), key=lambda item: item.name))
    if any(
        entry.suffix.lower() != required_suffix.lower() or not entry.is_file()
        for entry in entries
    ):
        raise ValueError(f"{label} contains an unexpected entry")
    for entry in entries:
        freeze_process_immutable_file(entry, label=f"{label} entry")
    directory_key = _process_path_key(directory)
    entry_keys = tuple(_process_path_key(entry) for entry in entries)
    with _PROCESS_IMMUTABLE_LOCK:
        existing = dict(_PROCESS_IMMUTABLE_DIRECTORIES).get(directory_key)
        if existing is not None and existing != entry_keys:
            raise ValueError(f"{label} immutable process inventory already differs")
        if existing is None:
            _PROCESS_IMMUTABLE_DIRECTORIES = (
                *_PROCESS_IMMUTABLE_DIRECTORIES,
                (directory_key, entry_keys),
            )
    return entries


def process_immutable_file_bytes(path: str | Path) -> bytes | None:
    """Return startup-captured bytes, rejecting additions under sealed roots."""

    key = _process_path_key(path)
    with _PROCESS_IMMUTABLE_LOCK:
        frozen = dict(_PROCESS_IMMUTABLE_FILES).get(key)
        if frozen is not None:
            return frozen
        for directory, _ in _PROCESS_IMMUTABLE_DIRECTORIES:
            if key == directory or key.startswith(directory + os.sep):
                raise ValueError(
                    "path is outside the immutable process directory inventory"
                )
    return None


def process_immutable_directory_entries(path: str | Path) -> tuple[Path, ...] | None:
    """Return the startup-captured flat inventory for one sealed directory."""

    key = _process_path_key(path)
    with _PROCESS_IMMUTABLE_LOCK:
        entries = dict(_PROCESS_IMMUTABLE_DIRECTORIES).get(key)
    if entries is None:
        return None
    return tuple(Path(entry) for entry in entries)


def _read_unfrozen_stable_single_link_file(
    path: str | Path,
    *,
    label: str,
    max_bytes: int | None = None,
) -> bytes:
    """Read current filesystem bytes without consulting process captures."""

    lexical = assert_absolute_path_has_no_links(path)
    try:
        before = lexical.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
        raise ValueError(f"{label} must be a single-link regular file")
    if max_bytes is not None and int(before.st_size) > max_bytes:
        raise ValueError(f"{label} exceeds the maximum size")
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor: int | None = None
    try:
        descriptor = os.open(lexical, flags)
        opened = os.fstat(descriptor)
        if not _is_single_link_regular(opened) or (
            _stable_file_identity(before) != _stable_file_identity(opened)
        ):
            raise ValueError(f"{label} identity changed before read")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            payload = stream.read() if max_bytes is None else stream.read(max_bytes + 1)
        if max_bytes is not None and len(payload) > max_bytes:
            raise ValueError(f"{label} exceeds the maximum size")
    except OSError as exc:
        raise ValueError(f"{label} cannot be read safely") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        after = lexical.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} disappeared after read") from exc
    if not _is_single_link_regular(after) or (
        _stable_file_identity(before) != _stable_file_identity(after)
        or len(payload) != int(after.st_size)
    ):
        raise ValueError(f"{label} changed while being read")

    # A single descriptor pass can still accept a same-size in-place rewrite on
    # filesystems where a writer can restore mtime (and, on Python 3.11/Windows,
    # where st_ctime is creation time rather than the NTFS change time). Reopen
    # the lexical path and compare exact bytes as well as descriptor/path
    # identity. Host ACLs remain the outer trust boundary, but an ordinary
    # concurrent mutation can no longer produce an audit for stale bytes.
    verification_descriptor: int | None = None
    try:
        verification_descriptor = os.open(lexical, flags)
        verification_opened = os.fstat(verification_descriptor)
        if not _is_single_link_regular(verification_opened) or (
            _stable_file_identity(after)
            != _stable_file_identity(verification_opened)
        ):
            raise ValueError(f"{label} identity changed before verification read")
        with os.fdopen(verification_descriptor, "rb") as stream:
            verification_descriptor = None
            verification = (
                stream.read()
                if max_bytes is None
                else stream.read(max_bytes + 1)
            )
    except OSError as exc:
        raise ValueError(f"{label} cannot be verification-read safely") from exc
    finally:
        if verification_descriptor is not None:
            os.close(verification_descriptor)
    try:
        final = lexical.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} disappeared after verification read") from exc
    if (
        not _is_single_link_regular(final)
        or _stable_file_identity(after) != _stable_file_identity(final)
        or len(verification) != int(final.st_size)
        or verification != payload
    ):
        raise ValueError(f"{label} changed during stable verification")
    return payload


def _process_path_key(path: str | Path) -> str:
    lexical = Path(path).expanduser()
    if not lexical.is_absolute():
        raise ValueError("process immutable file path must be absolute")
    return os.path.normcase(os.path.abspath(lexical))


def _stable_file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(value.st_nlink),
    )


def _is_single_link_regular(value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    return bool(
        stat.S_ISREG(value.st_mode)
        and int(value.st_nlink) == 1
        and not (
            os.name == "nt"
            and attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        )
    )


def paths_overlap_by_identity(first: str | Path, second: str | Path) -> bool:
    """Detect equality/ancestry across lexical, drive and UNC aliases.

    Existing ancestors are compared with ``samefile``. Relative suffixes below
    a physically identical ancestor are then compared, so the check also works
    when one or both final paths have not been created yet.
    """

    left = assert_absolute_path_has_no_links(first)
    right = assert_absolute_path_has_no_links(second)
    if _paths_overlap_lexically(left, right):
        return True
    for left_anchor, left_tail in _existing_ancestor_chain(left):
        for right_anchor, right_tail in _existing_ancestor_chain(right):
            try:
                same = os.path.samefile(left_anchor, right_anchor)
            except OSError as exc:
                raise ValueError("cannot establish path identity") from exc
            if same and (
                _parts_are_prefix(left_tail, right_tail)
                or _parts_are_prefix(right_tail, left_tail)
            ):
                return True
    return False


def _existing_ancestor_chain(path: Path) -> Iterator[tuple[Path, tuple[str, ...]]]:
    current = path
    tail: tuple[str, ...] = ()
    while True:
        try:
            current.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ValueError("cannot establish existing path ancestry") from exc
        else:
            yield current, tail
        parent = current.parent
        if parent == current:
            break
        tail = (current.name, *tail)
        current = parent


def _parts_are_prefix(prefix: tuple[str, ...], value: tuple[str, ...]) -> bool:
    if len(prefix) > len(value):
        return False
    normalized_prefix = tuple(os.path.normcase(item) for item in prefix)
    normalized_value = tuple(os.path.normcase(item) for item in value[: len(prefix)])
    return normalized_prefix == normalized_value


def _paths_overlap_lexically(first: Path, second: Path) -> bool:
    left = os.path.normcase(os.path.normpath(str(first)))
    right = os.path.normcase(os.path.normpath(str(second)))
    if left == right:
        return True
    try:
        common = os.path.normcase(os.path.commonpath([left, right]))
    except ValueError:
        return False
    return common in {left, right}

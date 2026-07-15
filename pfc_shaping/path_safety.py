"""Cross-platform filesystem path safety primitives."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Iterator


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

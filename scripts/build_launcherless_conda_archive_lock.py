#!/usr/bin/env python3
"""Lock one launcherless CPython prefix to retained exact Conda archives."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    paths_overlap_by_identity,
    read_stable_single_link_file,
)
from scripts.build_launcherless_local_runtime import validate_python_runtime_manifest
from scripts.build_repo_local_conda_prefix import (
    validate_repo_local_conda_prefix_build_receipt,
)
from scripts.build_snapshot_publisher_runtime_closure import (
    _canonical_json,
    _is_link_or_reparse,
)

_SCHEMA = "fmv_lt_launcherless_conda_archive_lock.v2"
_STATUS = "PASS_LOCAL_ARCHIVE_LOCK_NOT_PRODUCTION"
_PREFIX_RECEIPT_SCHEMA = "fmv_lt_launcherless_conda_prefix_build_receipt.v3"
_PREFIX_RECEIPT_STATUS = "PASS_LOCAL_EXPLICIT_ARCHIVE_REPLAY_NOT_PRODUCTION"
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.+!-]+$")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "source_runtime_prefix",
    "package_cache_roots",
    "conda_history",
    "packages",
    "package_count",
    "total_archive_bytes",
    "archive_set_id",
    "archive_lock_id",
    "explicit_spec_lines",
    "network_required",
    "admin_required",
    "production_authorization",
    "promotion_gate",
}


def _stable_regular_file(path: Path, *, label: str) -> tuple[Path, bytes]:
    """Read twice through the shared mono-link descriptor primitive."""

    selected = assert_absolute_path_has_no_links(path)
    return selected, read_stable_single_link_file(selected, label=label)
_PACKAGE_KEYS = {
    "name",
    "version",
    "build",
    "build_number",
    "subdir",
    "channel",
    "source_url",
    "filename",
    "md5",
    "sha256",
    "size",
    "depends",
    "constrains",
    "conda_meta_filename",
    "conda_meta_sha256",
    "cache_root",
    "archive_path",
    "archive_uri",
    "archive_payload_tree_sha256",
    "archive_payload_file_count",
    "prefix_replacement_file_count",
    "noarch_type",
}
_MAX_CONDA_MEMBER_BYTES = 512 * 1024 * 1024
_MAX_CONDA_PACKAGE_FILES = 100_000


def build_conda_archive_lock(
    *,
    runtime_prefix: Path,
    package_cache_roots: tuple[Path, ...],
    output: Path,
) -> dict[str, object]:
    """Write one deterministic new-name lock after replaying every archive."""

    destination = _absolute_new_file(output, "Conda archive lock output")
    payload = conda_archive_lock_payload(
        runtime_prefix=runtime_prefix,
        package_cache_roots=package_cache_roots,
    )
    encoded = _canonical_json(payload) + b"\n"
    with destination.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def conda_archive_lock_payload(
    *,
    runtime_prefix: Path,
    package_cache_roots: tuple[Path, ...],
) -> dict[str, object]:
    """Capture the exact Conda records and retained archive bytes."""

    prefix = _absolute_unlinked_directory(runtime_prefix, "runtime prefix")
    cache_roots = _validated_cache_roots(package_cache_roots)
    metadata_root = _absolute_unlinked_directory(
        prefix / "conda-meta", "runtime conda-meta"
    )
    history_path, history = _stable_regular_file(
        metadata_root / "history", label="Conda history"
    )
    if not history or b"conda version:" not in history or b"# cmd:" not in history:
        raise ValueError("Conda history does not identify its build command")

    metadata_paths = sorted(metadata_root.glob("*.json"), key=lambda item: item.name)
    if not metadata_paths:
        raise ValueError("Conda package metadata inventory is empty")

    packages: list[dict[str, object]] = []
    names: set[str] = set()
    filenames: set[str] = set()
    for metadata_path in metadata_paths:
        selected_metadata, metadata_bytes = _stable_regular_file(
            assert_absolute_path_has_no_links(metadata_path),
            label=f"Conda metadata {metadata_path.name}",
        )
        record = _strict_conda_json(metadata_bytes)
        package = _locked_package(
            record=record,
            metadata_path=selected_metadata,
            metadata_bytes=metadata_bytes,
            cache_roots=cache_roots,
        )
        name = str(package["name"])
        filename = str(package["filename"])
        if name in names or filename in filenames:
            raise ValueError("Conda package inventory is ambiguous")
        names.add(name)
        filenames.add(filename)
        packages.append(package)

    packages.sort(
        key=lambda item: (
            str(item["name"]).casefold(),
            str(item["version"]),
            str(item["build"]),
            str(item["filename"]),
        )
    )
    portable_packages = [_portable_package(item) for item in packages]
    archive_set_id = hashlib.sha256(_canonical_json(portable_packages)).hexdigest()
    total_archive_bytes = sum(int(item["size"]) for item in packages)
    history_record = {
        "path": str(history_path),
        "sha256": hashlib.sha256(history).hexdigest(),
        "size": len(history),
    }
    lock_identity = {
        "schema_version": _SCHEMA,
        "archive_set_id": archive_set_id,
        "conda_history_sha256": history_record["sha256"],
        "package_count": len(packages),
        "total_archive_bytes": total_archive_bytes,
    }
    explicit_spec_lines = ["@EXPLICIT"] + [
        f"{item['archive_uri']}#{item['md5']}" for item in packages
    ]
    return {
        "schema_version": _SCHEMA,
        "status": _STATUS,
        "source_runtime_prefix": str(prefix),
        "package_cache_roots": [str(item) for item in cache_roots],
        "conda_history": history_record,
        "packages": packages,
        "package_count": len(packages),
        "total_archive_bytes": total_archive_bytes,
        "archive_set_id": archive_set_id,
        "archive_lock_id": hashlib.sha256(
            _canonical_json(lock_identity)
        ).hexdigest(),
        "explicit_spec_lines": explicit_spec_lines,
        "network_required": False,
        "admin_required": False,
        "production_authorization": False,
        "promotion_gate": False,
    }


def verify_conda_archive_lock(
    *, lock_path: Path, expected_sha256: str
) -> dict[str, object]:
    """Replay an existing lock, including every retained archive byte."""

    expected = expected_sha256.strip().lower()
    if _HEX64.fullmatch(expected) is None:
        raise ValueError("expected Conda archive lock SHA-256 is invalid")
    _, encoded = _stable_regular_file(lock_path, label="Conda archive lock")
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise ValueError("Conda archive lock SHA-256 diverges")
    lock = _strict_conda_json(encoded)
    if _canonical_json(lock) + b"\n" != encoded:
        raise ValueError("Conda archive lock is not canonical JSONL")
    _validate_lock_shape(lock)
    replay = conda_archive_lock_payload(
        runtime_prefix=Path(str(lock["source_runtime_prefix"])),
        package_cache_roots=tuple(
            Path(str(item)) for item in lock["package_cache_roots"]
        ),
    )
    if replay != lock:
        raise ValueError("Conda archive lock replay diverges")
    return lock


def materialize_conda_explicit_spec(
    *, lock_path: Path, expected_sha256: str, output: Path
) -> tuple[dict[str, object], dict[str, object]]:
    """Write the one exact local-file explicit spec bound by a verified lock."""

    destination = _absolute_new_file(output, "Conda explicit spec output")
    lock = verify_conda_archive_lock(
        lock_path=lock_path, expected_sha256=expected_sha256
    )
    lines = lock["explicit_spec_lines"]
    if (
        not isinstance(lines, list)
        or len(lines) != int(lock["package_count"]) + 1
        or lines[0] != "@EXPLICIT"
        or any(not isinstance(item, str) or not item for item in lines)
    ):
        raise ValueError("Conda explicit spec lines are invalid")
    payload = ("\n".join(lines) + "\n").encode("ascii")
    with destination.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return lock, {
        "path": str(destination),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "line_count": len(lines),
    }


def audit_conda_prefix_build(
    *,
    lock_path: Path,
    expected_lock_sha256: str,
    explicit_spec_path: Path,
    expected_explicit_spec_sha256: str,
    runtime_prefix: Path,
    python_runtime_manifest: Path,
    expected_python_runtime_manifest_sha256: str,
    repo_local_build_receipt: Path,
    expected_repo_local_build_receipt_sha256: str,
    output: Path,
    root: Path = Path(__file__).resolve().parents[1],
) -> dict[str, object]:
    """Bind an unused exact prefix to the archive lock that built it."""

    destination = _absolute_new_file(output, "Conda prefix build receipt output")
    lock = verify_conda_archive_lock(
        lock_path=lock_path, expected_sha256=expected_lock_sha256
    )
    prefix = _absolute_unlinked_directory(runtime_prefix, "replayed runtime prefix")
    source_prefix = _absolute_unlinked_directory(
        Path(str(lock["source_runtime_prefix"])),
        "source runtime prefix",
    )
    if paths_overlap_by_identity(prefix, source_prefix):
        raise ValueError("replayed runtime prefix must be a new namespace")
    spec_path, spec_bytes = _stable_regular_file(
        explicit_spec_path, label="Conda explicit spec"
    )
    repo_local_receipt = validate_repo_local_conda_prefix_build_receipt(
        receipt_path=repo_local_build_receipt,
        expected_sha256=expected_repo_local_build_receipt_sha256,
        runtime_prefix=prefix,
        explicit_spec_path=spec_path,
        root=root,
    )
    explicit_sha256 = expected_explicit_spec_sha256.strip().lower()
    expected_spec = ("\n".join(lock["explicit_spec_lines"]) + "\n").encode(
        "ascii"
    )
    if (
        _HEX64.fullmatch(explicit_sha256) is None
        or hashlib.sha256(spec_bytes).hexdigest() != explicit_sha256
        or spec_bytes != expected_spec
    ):
        raise ValueError("Conda explicit spec differs from the archive lock")

    target_records, expected_prefix_files = _validated_explicit_prefix_records(
        prefix=prefix,
        locked_packages=lock["packages"],
    )
    archive_verified_file_count = sum(
        int(item["archive_verified_file_count"]) for item in target_records
    )
    generated_nonruntime_file_count = sum(
        int(item["generated_nonruntime_file_count"]) for item in target_records
    )
    history_path, history_bytes = _stable_regular_file(
        prefix / "conda-meta" / "history", label="replayed Conda history"
    )
    _validate_explicit_history(
        history_bytes,
        prefix=prefix,
        explicit_spec_path=spec_path,
        package_count=int(lock["package_count"]),
    )
    python_manifest = validate_python_runtime_manifest(
        runtime_prefix=prefix,
        manifest_path=python_runtime_manifest,
        expected_manifest_sha256=expected_python_runtime_manifest_sha256,
    )
    manifest_files = python_manifest.get("files")
    if not isinstance(manifest_files, list):
        raise ValueError("Python runtime manifest file inventory is unavailable")
    actual_prefix_files = [str(item.get("path", "")) for item in manifest_files]
    if sorted(expected_prefix_files) != actual_prefix_files:
        raise ValueError("replayed prefix differs from Conda package file inventories")

    lock_path_selected, lock_bytes = _stable_regular_file(
        lock_path, label="Conda archive lock"
    )
    manifest_path_selected, manifest_bytes = _stable_regular_file(
        python_runtime_manifest, label="Python runtime manifest"
    )
    core = {
        "archive_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "archive_lock_id": lock["archive_lock_id"],
        "archive_set_id": lock["archive_set_id"],
        "explicit_spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "python_runtime_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "python_runtime_tree_sha256": python_manifest["tree_sha256"],
        "target_conda_meta_sha256": hashlib.sha256(
            _canonical_json(target_records)
        ).hexdigest(),
        "package_count": lock["package_count"],
        "file_count": python_manifest["file_count"],
        "archive_verified_file_count": archive_verified_file_count,
        "generated_nonruntime_file_count": generated_nonruntime_file_count,
        "repo_local_build_receipt_sha256": repo_local_receipt["sha256"],
        "repo_local_external_guard_sha256": repo_local_receipt[
            "external_guard_sha256"
        ],
    }
    result = {
        "schema_version": _PREFIX_RECEIPT_SCHEMA,
        "status": _PREFIX_RECEIPT_STATUS,
        "archive_lock": {
            "path": str(lock_path_selected),
            "sha256": core["archive_lock_sha256"],
            "archive_lock_id": lock["archive_lock_id"],
            "archive_set_id": lock["archive_set_id"],
        },
        "explicit_spec": {
            "path": str(spec_path),
            "sha256": core["explicit_spec_sha256"],
            "size": len(spec_bytes),
            "line_count": len(lock["explicit_spec_lines"]),
        },
        "runtime_prefix": str(prefix),
        "conda_history": {
            "path": str(history_path),
            "sha256": hashlib.sha256(history_bytes).hexdigest(),
            "size": len(history_bytes),
        },
        "python_runtime_manifest": {
            "path": str(manifest_path_selected),
            "sha256": core["python_runtime_manifest_sha256"],
            "tree_sha256": python_manifest["tree_sha256"],
            "file_count": python_manifest["file_count"],
            "python_executable_sha256": python_manifest[
                "python_executable_sha256"
            ],
            "python_dll_sha256": python_manifest["python_dll_sha256"],
        },
        "target_conda_meta": target_records,
        "repo_local_build_receipt": repo_local_receipt,
        "package_count": lock["package_count"],
        "file_count": python_manifest["file_count"],
        "archive_verified_file_count": archive_verified_file_count,
        "generated_nonruntime_file_count": generated_nonruntime_file_count,
        "build_policy": {
            "solver_used": False,
            "network_required": False,
            "copy_only": True,
            "new_prefix_required": True,
            "target_python_executed_before_manifest": False,
            "dependency_metadata_authority": (
                "ARCHIVE_PATHS_BYTES_EXCEPT_DECLARED_GENERATED_NONRUNTIME"
            ),
        },
        "prefix_receipt_id": hashlib.sha256(_canonical_json(core)).hexdigest(),
        "production_authorization": False,
        "promotion_gate": False,
    }
    encoded = _canonical_json(result) + b"\n"
    with destination.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return result


def _locked_package(
    *,
    record: dict[str, object],
    metadata_path: Path,
    metadata_bytes: bytes,
    cache_roots: tuple[Path, ...],
) -> dict[str, object]:
    required = {
        "name",
        "version",
        "build",
        "build_number",
        "subdir",
        "channel",
        "fn",
        "url",
        "md5",
        "sha256",
        "size",
        "depends",
        "constrains",
    }
    if not required.issubset(record):
        raise ValueError("Conda package metadata is incomplete")

    name = _safe_token(record["name"], "package name")
    version = _safe_token(record["version"], "package version")
    build = _safe_token(record["build"], "package build")
    filename = _safe_archive_filename(record["fn"])
    build_number = record["build_number"]
    size = record["size"]
    subdir = record["subdir"]
    channel = record["channel"]
    source_url = record["url"]
    md5 = str(record["md5"])
    sha256 = str(record["sha256"])
    if (
        not isinstance(build_number, int)
        or isinstance(build_number, bool)
        or build_number < 0
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or subdir not in {"win-64", "noarch"}
        or not isinstance(channel, str)
        or not isinstance(source_url, str)
        or _HEX32.fullmatch(md5) is None
        or _HEX64.fullmatch(sha256) is None
    ):
        raise ValueError("Conda package identity is invalid")
    parsed_url = urlsplit(source_url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.query
        or parsed_url.fragment
        or not parsed_url.path.endswith(f"/{filename}")
        or source_url.rsplit("/", 1)[0] != channel
    ):
        raise ValueError("Conda package source URL is invalid")

    depends = _string_list(record["depends"], "depends")
    constrains = _string_list(record["constrains"], "constrains")
    matches = [root / filename for root in cache_roots if (root / filename).is_file()]
    if len(matches) != 1:
        raise ValueError("Conda package archive resolution is not unique")
    archive_path, archive = _stable_regular_file(
        matches[0], label=f"Conda archive {filename}"
    )
    if (
        len(archive) != size
        or hashlib.md5(archive, usedforsecurity=False).hexdigest() != md5
        or hashlib.sha256(archive).hexdigest() != sha256
    ):
        raise ValueError("Conda package archive bytes diverge from metadata")
    if metadata_path.name != f"{name}-{version}-{build}.json":
        raise ValueError("Conda metadata filename diverges from package identity")
    archive_contract = _conda_archive_payload_contract(
        archive,
        filename=filename,
        expected_identity={
            "name": name,
            "version": version,
            "build": build,
            "build_number": build_number,
            "subdir": subdir,
        },
    )

    cache_root = archive_path.parent
    return {
        "name": name,
        "version": version,
        "build": build,
        "build_number": build_number,
        "subdir": subdir,
        "channel": channel,
        "source_url": source_url,
        "filename": filename,
        "md5": md5,
        "sha256": sha256,
        "size": size,
        "depends": depends,
        "constrains": constrains,
        "conda_meta_filename": metadata_path.name,
        "conda_meta_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "cache_root": str(cache_root),
        "archive_path": str(archive_path),
        "archive_uri": archive_path.as_uri(),
        **archive_contract,
    }


def _conda_archive_payload_contract(
    archive: bytes,
    *,
    filename: str,
    expected_identity: dict[str, object],
) -> dict[str, object]:
    inventory = _conda_archive_payload_inventory(
        archive,
        filename=filename,
        expected_identity=expected_identity,
    )
    return {
        "archive_payload_tree_sha256": inventory["tree_sha256"],
        "archive_payload_file_count": len(inventory["paths"]),
        "prefix_replacement_file_count": inventory[
            "prefix_replacement_file_count"
        ],
        "noarch_type": inventory["noarch_type"],
    }


def _conda_archive_payload_inventory(
    archive: bytes,
    *,
    filename: str,
    expected_identity: dict[str, object],
) -> dict[str, object]:
    members = _conda_archive_members(archive, filename=filename)
    try:
        index = _strict_conda_json(members["info/index.json"])
        paths_document = _strict_conda_json(members["info/paths.json"])
    except KeyError as exc:
        raise ValueError("Conda archive lacks index or paths metadata") from exc
    comparable = {
        "name": index.get("name"),
        "version": index.get("version"),
        "build": index.get("build"),
        "build_number": index.get("build_number"),
        "subdir": index.get("subdir"),
    }
    if comparable != expected_identity:
        raise ValueError("Conda archive index identity diverges")
    raw_paths = paths_document.get("paths")
    if (
        paths_document.get("paths_version") != 1
        or not isinstance(raw_paths, list)
        or len(raw_paths) > _MAX_CONDA_PACKAGE_FILES
    ):
        raise ValueError(f"Conda archive paths metadata is invalid: {filename}")
    link_payload = members.get("info/link.json")
    noarch_type = "none"
    entry_points: tuple[str, ...] = ()
    if link_payload is not None:
        link = _strict_conda_json(link_payload)
        noarch = link.get("noarch")
        if noarch is not None:
            if not isinstance(noarch, dict) or noarch.get("type") not in {
                "generic",
                "python",
            }:
                raise ValueError("Conda archive noarch policy is invalid")
            noarch_type = str(noarch["type"])
            raw_entry_points = noarch.get("entry_points", [])
            if not isinstance(raw_entry_points, list) or any(
                not isinstance(item, str) or "=" not in item
                for item in raw_entry_points
            ):
                raise ValueError("Conda archive entry-point policy is invalid")
            entry_points = tuple(raw_entry_points)
    elif expected_identity["subdir"] == "noarch":
        noarch_type = "generic"

    payload_paths = {
        path: payload
        for path, payload in members.items()
        if not path.startswith("info/")
    }
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    digest = hashlib.sha256()
    replacement_count = 0
    for raw in raw_paths:
        if not isinstance(raw, dict):
            raise ValueError("Conda archive path record is invalid")
        relative = _portable_prefix_path(raw.get("_path"))
        folded = relative.casefold()
        if folded in seen:
            raise ValueError("Conda archive payload path inventory collides")
        seen.add(folded)
        if raw.get("path_type") != "hardlink":
            raise ValueError("Conda archive non-regular payload is forbidden")
        payload = payload_paths.get(relative)
        sha256 = raw.get("sha256")
        size = raw.get("size_in_bytes")
        if (
            payload is None
            or _HEX64.fullmatch(str(sha256)) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or len(payload) != size
            or hashlib.sha256(payload).hexdigest() != sha256
        ):
            raise ValueError("Conda archive payload differs from info/paths.json")
        if _is_conda_lifecycle_script_path(relative):
            raise ValueError("Conda lifecycle execution payload is forbidden")
        placeholder = raw.get("prefix_placeholder")
        file_mode = raw.get("file_mode")
        if placeholder is not None:
            if (
                not isinstance(placeholder, str)
                or not placeholder
                or file_mode not in {"text", "binary"}
            ):
                raise ValueError("Conda prefix replacement metadata is invalid")
            replacement_count += 1
        elif file_mode is not None:
            raise ValueError("Conda file mode lacks a prefix placeholder")
        records.append(
            {
                "source_path": relative,
                "sha256": sha256,
                "size": size,
                "prefix_placeholder": placeholder,
                "file_mode": file_mode,
                "payload": payload,
            }
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(sha256).encode("ascii"))
        digest.update(b"\0")
    if {path.casefold() for path in payload_paths} != seen:
        raise ValueError("Conda archive payload inventory differs from paths metadata")
    return {
        "tree_sha256": digest.hexdigest(),
        "paths": records,
        "prefix_replacement_file_count": replacement_count,
        "noarch_type": noarch_type,
        "entry_points": entry_points,
    }


def _conda_archive_members(archive: bytes, *, filename: str) -> dict[str, bytes]:
    if filename.endswith(".tar.bz2"):
        return _tar_members(archive, mode="r:bz2")
    if not filename.endswith(".conda"):
        raise ValueError("unsupported Conda archive format")
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as outer:
            names = outer.namelist()
            info_names = [name for name in names if name.startswith("info-") and name.endswith(".tar.zst")]
            package_names = [name for name in names if name.startswith("pkg-") and name.endswith(".tar.zst")]
            allowed = {"metadata.json", *info_names, *package_names}
            if (
                len(info_names) != 1
                or len(package_names) != 1
                or set(names) != allowed
                or len(names) != len(set(names))
            ):
                raise ValueError("Conda v2 archive member inventory is invalid")
            members: dict[str, bytes] = {}
            for nested_name in (*info_names, *package_names):
                member = outer.getinfo(nested_name)
                if (
                    member.file_size <= 0
                    or member.file_size > _MAX_CONDA_MEMBER_BYTES
                    or member.compress_size <= 0
                ):
                    raise ValueError("Conda v2 archive member size is invalid")
                compressed = outer.read(member)
                nested = _zstd_decompress_limited(compressed)
                for path, payload in _tar_members(nested, mode="r:").items():
                    if path.casefold() in {item.casefold() for item in members}:
                        raise ValueError("Conda v2 archive path inventory collides")
                    members[path] = payload
            return members
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("Conda v2 archive is invalid") from exc


def _zstd_decompress_limited(payload: bytes) -> bytes:
    try:
        import zstandard  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError(
            "zstandard is required to audit .conda archive payloads"
        ) from exc
    reader = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(payload))
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = reader.read(min(1024 * 1024, _MAX_CONDA_MEMBER_BYTES + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > _MAX_CONDA_MEMBER_BYTES:
            raise ValueError("Conda nested tar exceeds the decompression budget")
    return b"".join(chunks)


def _tar_members(payload: bytes, *, mode: str) -> dict[str, bytes]:
    try:
        archive = tarfile.open(fileobj=io.BytesIO(payload), mode=mode)
    except (tarfile.TarError, OSError) as exc:
        raise ValueError("Conda package tar archive is invalid") from exc
    result: dict[str, bytes] = {}
    folded: set[str] = set()
    with archive:
        members = archive.getmembers()
        if len(members) > _MAX_CONDA_PACKAGE_FILES + 10_000:
            raise ValueError("Conda package tar inventory exceeds the budget")
        for member in members:
            relative = _portable_prefix_path(member.name)
            key = relative.casefold()
            if key in folded:
                raise ValueError("Conda package tar path inventory collides")
            folded.add(key)
            if member.isdir():
                continue
            if not member.isfile() or member.size < 0 or member.size > _MAX_CONDA_MEMBER_BYTES:
                raise ValueError("Conda package tar contains a forbidden member")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("Conda package tar member is unreadable")
            data = handle.read(_MAX_CONDA_MEMBER_BYTES + 1)
            if len(data) != member.size or len(data) > _MAX_CONDA_MEMBER_BYTES:
                raise ValueError("Conda package tar member size diverges")
            result[relative] = data
    return result


def _is_conda_lifecycle_script_path(relative: str) -> bool:
    name = PurePosixPath(relative).name.casefold()
    return any(
        marker in name
        for marker in (
            "pre-link",
            "pre_link",
            "post-link",
            "post_link",
            "pre-unlink",
            "pre_unlink",
            "post-unlink",
            "post_unlink",
        )
    )


def _portable_package(package: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in package.items()
        if key not in {"cache_root", "archive_path", "archive_uri"}
    }


def _validated_explicit_prefix_records(
    *, prefix: Path, locked_packages: object
) -> tuple[list[dict[str, object]], list[str]]:
    if not isinstance(locked_packages, list) or not locked_packages:
        raise ValueError("locked Conda package inventory is unavailable")
    locked = {str(item["filename"]): item for item in locked_packages}
    metadata_root = _absolute_unlinked_directory(
        prefix / "conda-meta", "replayed runtime conda-meta"
    )
    metadata_paths = sorted(metadata_root.glob("*.json"), key=lambda item: item.name)
    if len(metadata_paths) != len(locked):
        raise ValueError("replayed Conda metadata count diverges")
    expected_files = {"conda-meta/history"}
    casefolded_files = {"conda-meta/history"}
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for metadata_path in metadata_paths:
        selected, metadata_bytes = _stable_regular_file(
            assert_absolute_path_has_no_links(metadata_path),
            label=f"replayed Conda metadata {metadata_path.name}",
        )
        record = _strict_conda_json(metadata_bytes)
        filename = _safe_archive_filename(record.get("fn"))
        package = locked.get(filename)
        if package is None or filename in seen:
            raise ValueError("replayed Conda package identity is not locked")
        seen.add(filename)
        actual_depends = _string_list(record.get("depends"), "depends")
        expected_depends = list(package["depends"])
        actual_constrains = _string_list(record.get("constrains"), "constrains")
        expected_constrains = list(package["constrains"])
        comparable = {
            "name": record.get("name"),
            "version": record.get("version"),
            "build": record.get("build"),
            "build_number": record.get("build_number"),
            "subdir": record.get("subdir"),
            "filename": filename,
            "md5": record.get("md5"),
            "sha256": record.get("sha256"),
            "size": record.get("size"),
        }
        expected_comparable = {
            key: package[key] for key in comparable
        }
        if comparable != expected_comparable:
            differing = sorted(
                key
                for key in comparable
                if comparable[key] != expected_comparable[key]
            )
            raise ValueError(
                "replayed Conda package metadata diverges from lock: "
                f"{filename}: {','.join(differing)}"
            )
        if record.get("channel") != "<unknown>" or record.get("url") != package.get(
            "archive_uri"
        ):
            raise ValueError("replayed Conda package is not bound to the local archive")
        expected_metadata_name = (
            f"{package['name']}-{package['version']}-{package['build']}.json"
        )
        if selected.name != expected_metadata_name:
            raise ValueError("replayed Conda metadata filename diverges")
        metadata_relative = f"conda-meta/{selected.name}"
        expected_files.add(metadata_relative)
        casefolded_files.add(metadata_relative.casefold())
        package_files = record.get("files")
        if not isinstance(package_files, list):
            raise ValueError("replayed Conda package file inventory is unavailable")
        for item in package_files:
            relative = _portable_prefix_path(item)
            folded = relative.casefold()
            if folded in casefolded_files:
                raise ValueError("replayed Conda package file inventory collides")
            expected_files.add(relative)
            casefolded_files.add(folded)
        archive_evidence = _validate_archive_payloads_against_prefix(
            package=package,
            prefix=prefix,
            installed_files=tuple(
                _portable_prefix_path(item) for item in package_files
            ),
        )
        records.append(
            {
                "filename": filename,
                "conda_meta_filename": selected.name,
                "conda_meta_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                "installed_file_count": len(package_files),
                "source_record_depends": expected_depends,
                "target_record_depends": actual_depends,
                "source_record_constrains": expected_constrains,
                "target_record_constrains": actual_constrains,
                "dependency_metadata_matches_source_record": (
                    actual_depends == expected_depends
                    and actual_constrains == expected_constrains
                ),
                **archive_evidence,
            }
        )
    if seen != set(locked):
        raise ValueError("replayed Conda package inventory is incomplete")
    records.sort(key=lambda item: str(item["filename"]))
    return records, sorted(expected_files)


def _validate_archive_payloads_against_prefix(
    *,
    package: dict[str, object],
    prefix: Path,
    installed_files: tuple[str, ...],
) -> dict[str, object]:
    archive_path = assert_absolute_path_has_no_links(
        Path(str(package["archive_path"]))
    )
    selected, archive = _stable_regular_file(
        archive_path,
        label=f"Conda archive replay {package['filename']}",
    )
    if (
        selected.name != package["filename"]
        or len(archive) != package["size"]
        or hashlib.sha256(archive).hexdigest() != package["sha256"]
    ):
        raise ValueError("Conda archive changed during prefix-byte replay")
    inventory = _conda_archive_payload_inventory(
        archive,
        filename=str(package["filename"]),
        expected_identity={
            "name": package["name"],
            "version": package["version"],
            "build": package["build"],
            "build_number": package["build_number"],
            "subdir": package["subdir"],
        },
    )
    if {
        "archive_payload_tree_sha256": inventory["tree_sha256"],
        "archive_payload_file_count": len(inventory["paths"]),
        "prefix_replacement_file_count": inventory[
            "prefix_replacement_file_count"
        ],
        "noarch_type": inventory["noarch_type"],
    } != {
        key: package[key]
        for key in (
            "archive_payload_tree_sha256",
            "archive_payload_file_count",
            "prefix_replacement_file_count",
            "noarch_type",
        )
    }:
        raise ValueError("Conda archive payload contract differs from the lock")

    mapped_sources: dict[str, str] = {}
    verified_files: list[tuple[str, str, int]] = []
    for record in inventory["paths"]:
        source = str(record["source_path"])
        target = _installed_archive_path(
            source,
            noarch_type=str(inventory["noarch_type"]),
        )
        folded = target.casefold()
        if folded in mapped_sources:
            raise ValueError("Conda installed archive path inventory collides")
        mapped_sources[folded] = target
        target_path = assert_absolute_path_has_no_links(
            prefix.joinpath(*PurePosixPath(target).parts)
        )
        _, actual = _stable_regular_file(
            target_path,
            label=f"archive-derived installed file {target}",
        )
        candidates = _prefix_replacement_candidates(
            bytes(record["payload"]),
            placeholder=record["prefix_placeholder"],
            file_mode=record["file_mode"],
            prefix=prefix,
        )
        if actual not in candidates:
            raise ValueError(
                "installed Conda file bytes differ from the locked archive: "
                f"{target}"
            )
        verified_files.append(
            (target, hashlib.sha256(actual).hexdigest(), len(actual))
        )

    installed_by_fold = {item.casefold(): item for item in installed_files}
    if not set(mapped_sources).issubset(installed_by_fold):
        raise ValueError("Conda target record omits an archive-derived file")
    generated = sorted(
        installed_by_fold[key]
        for key in set(installed_by_fold).difference(mapped_sources)
    )
    generated_files = _validate_generated_noarch_files(
        prefix=prefix,
        generated=generated,
        mapped_sources=mapped_sources,
        noarch_type=str(inventory["noarch_type"]),
        entry_points=tuple(inventory["entry_points"]),
    )
    return {
        "archive_verified_file_count": len(verified_files),
        "archive_verified_tree_sha256": _file_tuple_tree_sha256(verified_files),
        "generated_nonruntime_file_count": len(generated_files),
        "generated_nonruntime_tree_sha256": _file_tuple_tree_sha256(
            generated_files
        ),
    }


def _installed_archive_path(source: str, *, noarch_type: str) -> str:
    if noarch_type != "python":
        return source
    if source.startswith("site-packages/"):
        return "Lib/site-packages/" + source.removeprefix("site-packages/")
    if source.startswith("python-scripts/"):
        return "Scripts/" + source.removeprefix("python-scripts/")
    return source


def _prefix_replacement_candidates(
    payload: bytes,
    *,
    placeholder: object,
    file_mode: object,
    prefix: Path,
) -> tuple[bytes, ...]:
    if placeholder is None:
        return (payload,)
    marker = str(placeholder).encode("utf-8")
    if marker not in payload:
        raise ValueError("Conda prefix placeholder is absent from archive payload")
    if file_mode == "binary" and os.name == "nt":
        if b"PK\x05\x06" in payload:
            raise ValueError(
                "Conda Windows pyzzer binary prefix replacement is unsupported"
            )
        return (payload,)
    replacements = (
        {prefix.as_posix().encode("utf-8")}
        if os.name == "nt" and file_mode == "text"
        else {
            str(prefix).encode("utf-8"),
            prefix.as_posix().encode("utf-8"),
        }
    )
    candidates: set[bytes] = set()
    for replacement in replacements:
        if file_mode == "text":
            candidates.add(payload.replace(marker, replacement))
        elif file_mode == "binary":
            if len(replacement) > len(marker):
                raise ValueError("Conda prefix exceeds binary placeholder width")
            padded = replacement + b"\0" * (len(marker) - len(replacement))
            candidates.add(payload.replace(marker, padded))
        else:
            raise ValueError("Conda prefix replacement mode is invalid")
    return tuple(candidates)


def _validate_generated_noarch_files(
    *,
    prefix: Path,
    generated: list[str],
    mapped_sources: dict[str, str],
    noarch_type: str,
    entry_points: tuple[str, ...],
) -> list[tuple[str, str, int]]:
    if generated and noarch_type != "python":
        raise ValueError("Conda target contains non-archive generated files")
    entry_names = {item.split("=", 1)[0].strip() for item in entry_points}
    expected_entry_paths = {
        f"Scripts/{name}-script.py" for name in entry_names
    } | {f"Scripts/{name}.exe" for name in entry_names}
    records: list[tuple[str, str, int]] = []
    for relative in generated:
        allowed = relative in expected_entry_paths
        if relative.startswith("Lib/site-packages/") and relative.endswith(
            ".pyc"
        ):
            source = _source_path_for_pyc(relative)
            allowed = source.casefold() in mapped_sources
        if relative.startswith("Lib/site-packages/") and relative.endswith(
            ".dist-info/REQUESTED"
        ):
            allowed = True
        if not allowed:
            raise ValueError(
                "Conda target contains an unrecognized generated file: "
                f"{relative}"
            )
        path = assert_absolute_path_has_no_links(
            prefix.joinpath(*PurePosixPath(relative).parts)
        )
        _, payload = _stable_regular_file(
            path,
            label=f"generated non-runtime Conda file {relative}",
        )
        records.append(
            (relative, hashlib.sha256(payload).hexdigest(), len(payload))
        )
    return records


def _source_path_for_pyc(relative: str) -> str:
    path = PurePosixPath(relative)
    if len(path.parts) < 3 or path.parent.name != "__pycache__":
        return ""
    stem = path.name.split(".cpython-311", 1)[0]
    return (path.parent.parent / f"{stem}.py").as_posix()


def _file_tuple_tree_sha256(files: list[tuple[str, str, int]]) -> str:
    digest = hashlib.sha256()
    for relative, sha256, _ in sorted(files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _portable_prefix_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Conda installed file path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Conda installed file path is not portable")
    return path.as_posix()


def _validate_explicit_history(
    payload: bytes,
    *,
    prefix: Path,
    explicit_spec_path: Path,
    package_count: int,
) -> None:
    try:
        history = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("replayed Conda history is invalid") from exc
    command_lines = [line for line in history.splitlines() if line.startswith("# cmd: ")]
    if (
        len(command_lines) != 1
        or " create " not in command_lines[0]
        or " --offline " not in command_lines[0]
        or " --copy " not in command_lines[0]
        or f" --prefix {prefix} " not in command_lines[0]
        or f" --file {explicit_spec_path} " not in command_lines[0]
        or "https://" in history
        or history.count("[url=file:///") != package_count
    ):
        raise ValueError("replayed Conda history does not prove explicit offline copy")


def _validate_lock_shape(lock: dict[str, object]) -> None:
    packages = lock.get("packages")
    roots = lock.get("package_cache_roots")
    history = lock.get("conda_history")
    if (
        set(lock) != _TOP_LEVEL_KEYS
        or lock.get("schema_version") != _SCHEMA
        or lock.get("status") != _STATUS
        or lock.get("network_required") is not False
        or lock.get("admin_required") is not False
        or lock.get("production_authorization") is not False
        or lock.get("promotion_gate") is not False
        or not isinstance(packages, list)
        or not packages
        or not isinstance(roots, list)
        or not roots
        or not isinstance(history, dict)
        or set(history) != {"path", "sha256", "size"}
        or _HEX64.fullmatch(str(history.get("sha256", ""))) is None
        or _HEX64.fullmatch(str(lock.get("archive_set_id", ""))) is None
        or _HEX64.fullmatch(str(lock.get("archive_lock_id", ""))) is None
        or lock.get("package_count") != len(packages)
    ):
        raise ValueError("Conda archive lock schema is invalid")
    for package in packages:
        if not isinstance(package, dict) or set(package) != _PACKAGE_KEYS:
            raise ValueError("Conda archive lock package entry is invalid")


def _validated_cache_roots(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    if not paths:
        raise ValueError("at least one Conda package cache root is required")
    roots = tuple(
        _absolute_unlinked_directory(path, "package cache root") for path in paths
    )
    for index, root in enumerate(roots):
        for other in roots[index + 1 :]:
            if paths_overlap_by_identity(root, other):
                raise ValueError("Conda package cache roots overlap")
    return roots


def _absolute_unlinked_directory(path: Path, label: str) -> Path:
    try:
        selected = assert_absolute_path_has_no_links(path.expanduser())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} path is invalid") from exc
    if not selected.is_dir():
        raise ValueError(f"{label} is unavailable")
    path_stat = selected.stat(follow_symlinks=False)
    if _is_link_or_reparse(selected, path_stat) or not stat.S_ISDIR(path_stat.st_mode):
        raise ValueError(f"{label} is linked or not a directory")
    return selected


def _absolute_new_file(path: Path, label: str) -> Path:
    try:
        selected = assert_absolute_path_has_no_links(path.expanduser())
        parent = assert_absolute_path_has_no_links(selected.parent)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} path is invalid") from exc
    if not parent.is_dir():
        raise ValueError(f"{label} parent is unavailable")
    if selected.exists() or selected.is_symlink():
        raise ValueError(f"{label} must not already exist")
    return selected


def _safe_token(value: object, label: str) -> str:
    if not isinstance(value, str) or _SAFE_TOKEN.fullmatch(value) is None:
        raise ValueError(f"Conda {label} is invalid")
    return value


def _safe_archive_filename(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.endswith((".conda", ".tar.bz2"))
        or "/" in value
        or "\\" in value
        or value in {".conda", ".tar.bz2"}
    ):
        raise ValueError("Conda archive filename is invalid")
    return value


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"Conda package {label} is invalid")
    normalized = sorted(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Conda package {label} contains duplicates")
    return normalized


def _strict_conda_json(payload: bytes) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            payload.decode("utf-8"), object_pairs_hook=reject_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Conda JSON document is invalid") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Conda JSON document must be a mapping")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build a new exact archive lock")
    build.add_argument("--runtime-prefix", type=Path, required=True)
    build.add_argument(
        "--package-cache-root", type=Path, action="append", required=True
    )
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="replay an existing archive lock")
    verify.add_argument("--lock", type=Path, required=True)
    verify.add_argument("--expected-sha256", required=True)
    explicit = subparsers.add_parser(
        "explicit-spec", help="materialize the verified local-file explicit spec"
    )
    explicit.add_argument("--lock", type=Path, required=True)
    explicit.add_argument("--expected-sha256", required=True)
    explicit.add_argument("--output", type=Path, required=True)
    audit = subparsers.add_parser(
        "audit-prefix", help="bind a new unused prefix to the verified archive lock"
    )
    audit.add_argument("--lock", type=Path, required=True)
    audit.add_argument("--expected-lock-sha256", required=True)
    audit.add_argument("--explicit-spec", type=Path, required=True)
    audit.add_argument("--expected-explicit-spec-sha256", required=True)
    audit.add_argument("--runtime-prefix", type=Path, required=True)
    audit.add_argument("--python-runtime-manifest", type=Path, required=True)
    audit.add_argument("--expected-python-runtime-manifest-sha256", required=True)
    audit.add_argument("--repo-local-build-receipt", type=Path, required=True)
    audit.add_argument("--expected-repo-local-build-receipt-sha256", required=True)
    audit.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        result = build_conda_archive_lock(
            runtime_prefix=args.runtime_prefix,
            package_cache_roots=tuple(args.package_cache_root),
            output=args.output,
        )
        explicit_spec = None
    elif args.command == "verify":
        result = verify_conda_archive_lock(
            lock_path=args.lock, expected_sha256=args.expected_sha256
        )
        explicit_spec = None
    elif args.command == "explicit-spec":
        result, explicit_spec = materialize_conda_explicit_spec(
            lock_path=args.lock,
            expected_sha256=args.expected_sha256,
            output=args.output,
        )
    else:
        receipt = audit_conda_prefix_build(
            lock_path=args.lock,
            expected_lock_sha256=args.expected_lock_sha256,
            explicit_spec_path=args.explicit_spec,
            expected_explicit_spec_sha256=args.expected_explicit_spec_sha256,
            runtime_prefix=args.runtime_prefix,
            python_runtime_manifest=args.python_runtime_manifest,
            expected_python_runtime_manifest_sha256=(
                args.expected_python_runtime_manifest_sha256
            ),
            repo_local_build_receipt=args.repo_local_build_receipt,
            expected_repo_local_build_receipt_sha256=(
                args.expected_repo_local_build_receipt_sha256
            ),
            output=args.output,
        )
        summary = {
            "schema_version": receipt["schema_version"],
            "status": receipt["status"],
            "prefix_receipt_id": receipt["prefix_receipt_id"],
            "archive_set_id": receipt["archive_lock"]["archive_set_id"],
            "package_count": receipt["package_count"],
            "file_count": receipt["file_count"],
            "production_authorization": False,
        }
        sys.stdout.buffer.write(_canonical_json(summary) + b"\n")
        return 0
    summary = {
        "schema_version": result["schema_version"],
        "status": result["status"],
        "archive_lock_id": result["archive_lock_id"],
        "archive_set_id": result["archive_set_id"],
        "package_count": result["package_count"],
        "total_archive_bytes": result["total_archive_bytes"],
        "production_authorization": False,
    }
    if explicit_spec is not None:
        summary["explicit_spec"] = explicit_spec
    sys.stdout.buffer.write(_canonical_json(summary) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

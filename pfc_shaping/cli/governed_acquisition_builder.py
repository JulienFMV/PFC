"""Build unsigned provider-raw LT acquisition artifacts from exact body files.

The capture body files and metadata must already exist. This process has no
network, signing, journal, publication, or shared-cache authority.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import stat
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterator, Mapping

if os.name == "nt":
    from ctypes import wintypes
else:
    import fcntl

from pfc_shaping.data.governed_lt_acquisition import (
    MAX_PROVIDER_CAPTURE_SPEC_BYTES,
    MAX_PROVIDER_DOCUMENT_BYTES,
    MAX_PROVIDER_DOCUMENT_TOTAL_BYTES,
    CapturedProviderDocument,
    approved_provider_parser_payload_for_role,
    build_provider_transform_config,
    build_raw_envelope,
    dataframe_semantic_sha256,
    energy_price_resolution_provenance,
    provider_document_max_bytes_for_role,
    transform_provider_raw,
    validate_provider_document_inventory,
)
from pfc_shaping.durable_artifact import (
    durable_artifact_durability_classification,
    recover_durable_exact_bytes_residue,
    write_durable_exact_bytes,
)
from pfc_shaping.exit_codes import EXIT_INTEGRITY_FAILURE
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from pfc_shaping.pipeline.governed_release_cli_contract import (
    ReleaseCliIdentityError,
    assert_installed_runtime_sealed,
)
from pfc_shaping.pipeline.strict_structured_data import load_strict_json

CAPTURE_SPEC_SCHEMA = "lt_provider_capture_spec.v1"
BUILD_MANIFEST_SCHEMA = "lt_provider_acquisition_build.v2"
_SPEC_KEYS = {
    "schema_version",
    "acquisition_id",
    "source_role",
    "source_system",
    "source_locator",
    "provider_id",
    "received_at_utc",
    "start_utc",
    "end_utc",
    "documents",
}
_DOCUMENT_KEYS = {
    "document_id",
    "body_path",
    "request_url",
    "request_parameters",
    "credential_reference",
    "response_status_code",
    "response_media_type",
    "response_content_encoding",
    "response_headers",
}

_DirectoryIdentity = tuple[str, int, bytes]


@dataclass(frozen=True)
class _PinnedDirectory:
    path: Path
    identity: _DirectoryIdentity
    native_handle: int
    exclusive_writer_lease: bool


def build_acquisition(
    *,
    capture_spec_path: Path,
    output_directory: Path,
) -> Path:
    """Materialize an unsigned exact-byte envelope and its replayed bronze frame."""

    raw_spec_path = Path(capture_spec_path)
    if not raw_spec_path.is_absolute():
        raise ValueError("LT provider capture specification path must be absolute")
    _assert_windows_local_canonical_path(
        raw_spec_path,
        label="LT provider capture specification",
    )
    spec_path = assert_absolute_path_has_no_links(raw_spec_path)
    output, output_parent_identity = _admit_output_directory(output_directory)
    try:
        spec_payload = read_stable_single_link_file(
            spec_path,
            label="LT provider capture specification",
            max_bytes=MAX_PROVIDER_CAPTURE_SPEC_BYTES,
        )
        spec = load_strict_json(spec_payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("LT provider capture specification is invalid") from exc
    if not isinstance(spec, Mapping) or set(spec) != _SPEC_KEYS:
        raise ValueError("LT provider capture specification fields are not exact")
    if spec.get("schema_version") != CAPTURE_SPEC_SCHEMA:
        raise ValueError("LT provider capture specification schema is unsupported")
    raw_documents = spec.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError("LT provider capture documents are missing")
    role = str(spec.get("source_role", ""))
    role_document_budget = min(
        MAX_PROVIDER_DOCUMENT_BYTES,
        provider_document_max_bytes_for_role(role),
    )
    validate_provider_document_inventory(
        role,
        [
            str(document.get("document_id", ""))
            for document in raw_documents
            if isinstance(document, Mapping)
        ],
    )

    captured: list[CapturedProviderDocument] = []
    body_paths: list[tuple[Mapping[str, object], Path, int]] = []
    for raw_document in raw_documents:
        if not isinstance(raw_document, Mapping) or set(raw_document) != _DOCUMENT_KEYS:
            raise ValueError("LT provider capture document fields are not exact")
        body_path = Path(str(raw_document.get("body_path", "")))
        if not body_path.is_absolute():
            raise ValueError("LT provider capture body_path must be absolute")
        _assert_windows_local_canonical_path(
            body_path,
            label="LT provider capture body_path",
        )
        body_path = assert_absolute_path_has_no_links(body_path)
        try:
            body_size = body_path.stat().st_size
        except OSError as exc:
            raise ValueError("LT provider capture body is unavailable") from exc
        if body_size < 1 or body_size > role_document_budget:
            raise ValueError("LT provider capture body exceeds the maximum size")
        body_paths.append((raw_document, body_path, body_size))
    if sum(size for _, _, size in body_paths) > MAX_PROVIDER_DOCUMENT_TOTAL_BYTES:
        raise ValueError("LT provider capture cumulative body budget is exceeded")

    total_body_bytes = 0
    for raw_document, body_path, _ in body_paths:
        body = read_stable_single_link_file(
            body_path,
            label=f"provider body {raw_document.get('document_id', '')}",
            max_bytes=role_document_budget,
        )
        total_body_bytes += len(body)
        if total_body_bytes > MAX_PROVIDER_DOCUMENT_TOTAL_BYTES:
            raise ValueError("LT provider capture cumulative body budget is exceeded")
        parameters = raw_document.get("request_parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("LT provider request_parameters must be a mapping")
        response_headers = raw_document.get("response_headers")
        if not isinstance(response_headers, Mapping):
            raise ValueError("LT provider response_headers must be a mapping")
        captured.append(
            CapturedProviderDocument(
                document_id=str(raw_document.get("document_id", "")),
                request_url=str(raw_document.get("request_url", "")),
                request_parameters={str(key): str(value) for key, value in parameters.items()},
                response_media_type=str(raw_document.get("response_media_type", "")),
                body=body,
                response_headers={
                    str(key): str(value)
                    for key, value in response_headers.items()
                },
                credential_reference=(
                    None
                    if raw_document.get("credential_reference") is None
                    else str(raw_document.get("credential_reference"))
                ),
                response_status_code=raw_document.get("response_status_code"),
                response_content_encoding=str(
                    raw_document.get("response_content_encoding", "")
                ),
            )
        )

    envelope = build_raw_envelope(
        acquisition_id=str(spec.get("acquisition_id", "")),
        source_role=str(spec.get("source_role", "")),
        source_system=str(spec.get("source_system", "")),
        source_locator=str(spec.get("source_locator", "")),
        provider_id=str(spec.get("provider_id", "")),
        received_at_utc=str(spec.get("received_at_utc", "")),
        documents=captured,
    )
    transform_config = build_provider_transform_config(
        role=str(spec.get("source_role", "")),
        start_utc=str(spec.get("start_utc", "")),
        end_utc=str(spec.get("end_utc", "")),
    )
    if transform_config["source_system"] != spec.get("source_system"):
        raise ValueError("LT provider capture source_system is not eligible for the role")
    bronze = transform_provider_raw(
        envelope_payload=envelope,
        parser_config=transform_config,
    )
    resolution_provenance = (
        energy_price_resolution_provenance(bronze, role=role)
        if role.startswith("epex_")
        else None
    )
    bronze_buffer = BytesIO()
    bronze.to_parquet(bronze_buffer, index=True)
    bronze_payload = bronze_buffer.getvalue()
    config_payload = _canonical_json(transform_config)
    parser_payload = approved_provider_parser_payload_for_role(
        str(spec.get("source_role", ""))
    )

    output = assert_absolute_path_has_no_links(output)
    _assert_directory_identity(
        output.parent,
        expected=output_parent_identity,
        label="LT provider acquisition output parent",
    )
    artifacts = {
        "provider-raw-envelope.json": envelope,
        "provider-transform-config.json": config_payload,
        "bronze.parquet": bronze_payload,
        "provider-parser.py": parser_payload,
    }
    manifest = {
        "schema_version": BUILD_MANIFEST_SCHEMA,
        "acquisition_id": str(spec["acquisition_id"]),
        "source_role": str(spec["source_role"]),
        "source_system": str(spec["source_system"]),
        "source_locator": str(spec["source_locator"]),
        "received_at_utc": str(spec["received_at_utc"]),
        "capture_spec": {
            "sha256": _sha256(spec_payload),
            "size_bytes": len(spec_payload),
        },
        "artifacts": {
            name: {"sha256": _sha256(payload), "size_bytes": len(payload)}
            for name, payload in artifacts.items()
        },
        "bronze_frame_sha256": dataframe_semantic_sha256(bronze),
        "resolution_provenance": resolution_provenance,
        "signing_status": "UNSIGNED_REQUIRES_INDEPENDENT_AUTHORITIES",
        "publication_status": "NOT_PUBLISHED",
        "namespace_security_status": "BUILDER_MUTABLE_UNTRUSTED_QUARANTINE",
        "authority_handoff_status": (
            "REQUIRES_BUILDER_INACCESSIBLE_COPY_AND_INDEPENDENT_SIGNATURE"
        ),
        "durability_classification": durable_artifact_durability_classification(),
    }
    manifest_payload = _canonical_json(manifest)
    build_id = _sha256(manifest_payload)
    manifest["build_id"] = build_id
    manifest_payload = _canonical_json(manifest)
    expected = {**artifacts, "acquisition-build-manifest.json": manifest_payload}
    return _materialize_output(
        output=output,
        output_parent_identity=output_parent_identity,
        build_id=build_id,
        artifacts=artifacts,
        manifest_payload=manifest_payload,
        expected=expected,
    )


def _materialize_output(
    *,
    output: Path,
    output_parent_identity: _DirectoryIdentity,
    build_id: str,
    artifacts: Mapping[str, bytes],
    manifest_payload: bytes,
    expected: Mapping[str, bytes],
    manifest_name: str = "acquisition-build-manifest.json",
) -> Path:
    if not manifest_name or Path(manifest_name).name != manifest_name:
        raise ValueError("LT artifact manifest name is invalid")
    if expected.get(manifest_name) != manifest_payload:
        raise ValueError("LT artifact manifest payload is not bound to expected output")
    with _pin_output_parent(
        output.parent,
        expected=output_parent_identity,
    ) as parent_pin:
        output = assert_absolute_path_has_no_links(output)
        if output.exists():
            output_identity = _directory_identity(output)
            with _pin_directory(
                output,
                expected=output_identity,
                delete_access=False,
            ):
                _validate_complete_output(output, expected)
                _assert_directory_identity(
                    output,
                    expected=output_identity,
                    label="LT provider acquisition output",
                )
            return output / manifest_name

        staging = output.parent / f".{output.name}.staging-{build_id[:16]}"
        if staging.exists():
            staging = assert_absolute_path_has_no_links(staging)
            if not staging.is_dir():
                raise ValueError("LT provider acquisition staging residue is invalid")
        else:
            _mkdir_child(parent_pin, staging)
            staging = assert_absolute_path_has_no_links(staging)

        staging_identity = _directory_identity(staging)
        with _pin_directory(
            staging,
            expected=staging_identity,
            delete_access=True,
        ) as staging_pin:
            for name, payload in expected.items():
                recover_durable_exact_bytes_residue(
                    staging / name,
                    payload,
                    label=f"LT provider acquisition staged {name}",
                    exclusive_writer_lease_proven=staging_pin.exclusive_writer_lease,
                )
            _validate_partial_output(staging, expected)
            for name, payload in artifacts.items():
                _write_new(staging / name, payload)
            _write_new(staging / manifest_name, manifest_payload)
            _validate_complete_output(staging, expected)
            _assert_directory_identity(
                staging,
                expected=staging_identity,
                label="LT provider acquisition staging",
            )
            try:
                _rename_pinned_child(
                    parent_pin=parent_pin,
                    child_pin=staging_pin,
                    destination=output,
                )
            except OSError:
                if not output.exists():
                    raise
                output_identity = _directory_identity(output)
                with _pin_directory(
                    output,
                    expected=output_identity,
                    delete_access=False,
                ):
                    _validate_complete_output(output, expected)
                    _assert_directory_identity(
                        output,
                        expected=output_identity,
                        label="LT provider acquisition raced output",
                    )
                return output / manifest_name

            _assert_directory_identity(
                output,
                expected=staging_identity,
                label="LT provider acquisition promoted output",
            )
            _validate_complete_output(output, expected)
            _assert_directory_identity(
                output,
                expected=staging_identity,
                label="LT provider acquisition promoted output",
            )
        _fsync_directory(output.parent)
        return output / manifest_name


def _admit_output_directory(output_directory: Path) -> tuple[Path, _DirectoryIdentity]:
    raw_output = Path(output_directory)
    if not raw_output.is_absolute():
        raise ValueError("LT provider acquisition output directory must be absolute")
    _assert_windows_local_canonical_path(
        raw_output,
        label="LT provider acquisition output directory",
    )
    output = assert_absolute_path_has_no_links(raw_output)
    parent = assert_absolute_path_has_no_links(output.parent)
    try:
        identity = _directory_identity(parent)
    except (OSError, ValueError) as exc:
        raise ValueError(
            "LT provider acquisition output parent must be a pre-provisioned directory"
        ) from exc
    return output, identity


def _assert_windows_local_canonical_path(path: Path, *, label: str) -> None:
    if os.name != "nt":
        return
    raw = os.fspath(path)
    normalized = raw.replace("/", "\\")
    if normalized.startswith("\\\\"):
        raise ValueError(f"{label} must use a local fixed drive, not UNC")
    drive, tail = os.path.splitdrive(normalized)
    if re.fullmatch(r"[A-Za-z]:", drive) is None or not tail.startswith("\\"):
        raise ValueError(f"{label} must use a local fixed drive")
    if ":" in tail:
        raise ValueError(f"{label} must not use an NTFS alternate data stream")
    components = [component for component in tail.split("\\") if component]
    if any(component.endswith((".", " ")) for component in components):
        raise ValueError(f"{label} contains a trailing-dot/space Windows alias")
    if any(component in {".", ".."} for component in components):
        raise ValueError(f"{label} contains a relative Windows path component")
    if any(any(ord(character) < 32 for character in component) for component in components):
        raise ValueError(f"{label} contains a Windows control character")
    if any("~" in component for component in components):
        raise ValueError(f"{label} contains a Windows 8.3 alias marker")
    if any(any(character in '<>"|?*' for character in component) for component in components):
        raise ValueError(f"{label} contains a forbidden Windows filename character")
    if any(_is_windows_reserved_component(component) for component in components):
        raise ValueError(f"{label} contains a reserved Windows device name")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_drive_type = kernel32.GetDriveTypeW
    get_drive_type.argtypes = [wintypes.LPCWSTR]
    get_drive_type.restype = wintypes.UINT
    if int(get_drive_type(f"{drive}\\")) != 3:
        raise ValueError(f"{label} must use a local fixed drive")
    device_target = _windows_drive_device_target(drive)
    if not _is_windows_physical_volume_target(device_target):
        raise ValueError(
            f"{label} drive must map directly to a physical local volume"
        )


def _windows_drive_device_target(drive: str) -> str:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    query_dos_device = kernel32.QueryDosDeviceW
    query_dos_device.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    query_dos_device.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(32768)
    length = int(query_dos_device(drive.upper(), buffer, len(buffer)))
    if length <= 0:
        raise ctypes.WinError(ctypes.get_last_error())
    targets = [target for target in buffer[:length].split("\x00") if target]
    if len(targets) != 1:
        raise ValueError("Windows drive has an ambiguous DOS device mapping")
    return targets[0]


def _is_windows_physical_volume_target(target: str) -> bool:
    separator = chr(92)
    prefix = f"{separator}Device{separator}HarddiskVolume"
    suffix = target[len(prefix) :] if target.casefold().startswith(prefix.casefold()) else ""
    return bool(suffix) and suffix.isascii() and suffix.isdecimal()


def _is_windows_reserved_component(component: str) -> bool:
    stem = component.split(".", maxsplit=1)[0].upper()
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update({f"COM{digit}" for digit in "123456789"})
    reserved.update({f"LPT{digit}" for digit in "123456789"})
    reserved.update({f"COM{digit}" for digit in "¹²³"})
    reserved.update({f"LPT{digit}" for digit in "¹²³"})
    return stem in reserved


def _assert_directory_identity(
    directory: Path,
    *,
    expected: _DirectoryIdentity,
    label: str,
) -> None:
    selected = assert_absolute_path_has_no_links(directory)
    try:
        current = _directory_identity(selected)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} identity changed") from exc
    if current != expected:
        raise ValueError(f"{label} identity changed")


def _directory_identity(directory: Path) -> _DirectoryIdentity:
    selected = assert_absolute_path_has_no_links(directory)
    pin = _open_native_directory(
        selected,
        delete_access=False,
        share_delete=True,
    )
    try:
        return pin.identity
    finally:
        _close_native_directory(pin)


@contextmanager
def _pin_directory(
    directory: Path,
    *,
    expected: _DirectoryIdentity,
    delete_access: bool,
) -> Iterator[_PinnedDirectory]:
    selected = assert_absolute_path_has_no_links(directory)
    pin = _open_native_directory(
        selected,
        delete_access=delete_access,
        share_delete=False,
    )
    try:
        if pin.identity != expected:
            raise ValueError("LT provider acquisition directory identity changed")
        assert_absolute_path_has_no_links(selected)
        yield pin
    finally:
        _close_native_directory(pin)


@contextmanager
def _pin_output_parent(
    parent: Path,
    *,
    expected: _DirectoryIdentity,
) -> Iterator[_PinnedDirectory]:
    selected = assert_absolute_path_has_no_links(parent)
    pins: list[_PinnedDirectory] = []
    try:
        if os.name == "nt":
            ancestry = tuple(reversed((selected, *selected.parents)))
            for component in ancestry:
                pins.append(
                    _open_native_directory(
                        assert_absolute_path_has_no_links(component),
                        delete_access=False,
                        share_delete=False,
                    )
                )
        else:
            pins.append(
                _open_native_directory(
                    selected,
                    delete_access=False,
                    share_delete=False,
                )
            )
        parent_pin = pins[-1]
        if parent_pin.identity != expected:
            raise ValueError("LT provider acquisition output parent identity changed")
        assert_absolute_path_has_no_links(selected)
        yield parent_pin
    finally:
        for pin in reversed(pins):
            _close_native_directory(pin)


def _open_native_directory(
    directory: Path,
    *,
    delete_access: bool,
    share_delete: bool,
) -> _PinnedDirectory:
    if os.name == "nt":
        return _open_windows_directory(
            directory,
            delete_access=delete_access,
            share_delete=share_delete,
        )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(directory, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or int(metadata.st_ino) <= 0:
            raise ValueError("directory identity is unavailable")
        identity: _DirectoryIdentity = (
            "posix",
            int(metadata.st_dev),
            int(metadata.st_ino).to_bytes(16, byteorder="big", signed=False),
        )
        return _PinnedDirectory(
            path=directory,
            identity=identity,
            native_handle=descriptor,
            exclusive_writer_lease=_acquire_posix_writer_lease(
                descriptor,
                required=delete_access and not share_delete,
            ),
        )
    except Exception:
        os.close(descriptor)
        raise


def _close_native_directory(pin: _PinnedDirectory) -> None:
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        if not close_handle(wintypes.HANDLE(pin.native_handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        return
    os.close(pin.native_handle)


def _open_windows_directory(
    directory: Path,
    *,
    delete_access: bool,
    share_delete: bool,
) -> _PinnedDirectory:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    desired_access = 0x0001 | 0x0080
    if delete_access:
        desired_access |= 0x00010000
    share_mode = 0x00000001 | 0x00000002
    if share_delete:
        share_mode |= 0x00000004
    handle = create_file(
        str(directory),
        desired_access,
        share_mode,
        None,
        3,
        0x02000000 | 0x00200000,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    handle_value = int(handle) if handle is not None else 0
    if handle_value in {0, invalid_handle}:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        identity = _windows_directory_identity(handle_value)
        return _PinnedDirectory(
            path=directory,
            identity=identity,
            native_handle=handle_value,
            exclusive_writer_lease=delete_access and not share_delete,
        )
    except Exception:
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        close_handle(wintypes.HANDLE(handle_value))
        raise


def _windows_directory_identity(handle: int) -> _DirectoryIdentity:
    class _FileId128(ctypes.Structure):
        _fields_ = [("identifier", ctypes.c_ubyte * 16)]

    class _FileIdInfo(ctypes.Structure):
        _fields_ = [
            ("volume_serial_number", ctypes.c_ulonglong),
            ("file_id", _FileId128),
        ]

    class _FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("reparse_tag", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_info = kernel32.GetFileInformationByHandleEx
    get_info.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    get_info.restype = wintypes.BOOL
    attributes = _FileAttributeTagInfo()
    if not get_info(
        wintypes.HANDLE(handle),
        9,
        ctypes.byref(attributes),
        ctypes.sizeof(attributes),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if not int(attributes.file_attributes) & 0x00000010:
        raise ValueError("pinned path is not a directory")
    if int(attributes.file_attributes) & 0x00000400:
        raise ValueError("pinned directory is a reparse point")
    info = _FileIdInfo()
    if not get_info(
        wintypes.HANDLE(handle),
        18,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    file_id = bytes(info.file_id.identifier)
    volume = int(info.volume_serial_number)
    if volume <= 0 or not any(file_id):
        raise ValueError("Windows directory FILE_ID_INFO is unavailable")
    return ("windows-file-id-info-v1", volume, file_id)


def _acquire_posix_writer_lease(descriptor: int, *, required: bool) -> bool:
    if os.name == "nt" or not required:
        return False
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return True


def _mkdir_child(parent_pin: _PinnedDirectory, child: Path) -> None:
    if child.parent != parent_pin.path or not child.name or child.name in {".", ".."}:
        raise ValueError("LT provider acquisition staging path is not a direct child")
    if os.name == "nt":
        child.mkdir()
        return
    os.mkdir(child.name, dir_fd=parent_pin.native_handle)


def _rename_pinned_child(
    *,
    parent_pin: _PinnedDirectory,
    child_pin: _PinnedDirectory,
    destination: Path,
) -> None:
    if (
        child_pin.path.parent != parent_pin.path
        or destination.parent != parent_pin.path
        or not destination.name
        or destination.name in {".", ".."}
    ):
        raise ValueError("LT provider acquisition rename paths are not direct siblings")
    if os.name == "nt":
        _windows_rename_directory_handle(
            parent_handle=parent_pin.native_handle,
            child_handle=child_pin.native_handle,
            destination_name=destination.name,
        )
        return
    os.rename(
        child_pin.path.name,
        destination.name,
        src_dir_fd=parent_pin.native_handle,
        dst_dir_fd=parent_pin.native_handle,
    )


def _windows_rename_directory_handle(
    *,
    parent_handle: int,
    child_handle: int,
    destination_name: str,
) -> None:
    if (
        not destination_name
        or destination_name in {".", ".."}
        or any(separator in destination_name for separator in ("/", "\\", ":"))
        or destination_name.endswith((".", " "))
    ):
        raise ValueError("LT provider acquisition destination name is unsafe")

    class _FileRenameInfo(ctypes.Structure):
        _fields_ = [
            ("flags_or_replace", wintypes.DWORD),
            ("root_directory", wintypes.HANDLE),
            ("file_name_length", wintypes.DWORD),
            ("file_name", wintypes.WCHAR * 1),
        ]

    encoded_name = destination_name.encode("utf-16-le")
    file_name_offset = _FileRenameInfo.file_name.offset
    buffer = ctypes.create_string_buffer(
        ctypes.sizeof(_FileRenameInfo) + len(encoded_name)
    )
    info = ctypes.cast(buffer, ctypes.POINTER(_FileRenameInfo)).contents
    info.flags_or_replace = 0
    info.root_directory = parent_handle
    info.file_name_length = len(encoded_name)
    ctypes.memmove(
        ctypes.addressof(buffer) + file_name_offset,
        encoded_name,
        len(encoded_name),
    )
    class _IoStatusBlock(ctypes.Structure):
        _fields_ = [
            ("status_or_pointer", ctypes.c_void_p),
            ("information", ctypes.c_size_t),
        ]

    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    set_info = ntdll.NtSetInformationFile
    set_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_IoStatusBlock),
        wintypes.LPVOID,
        wintypes.ULONG,
        ctypes.c_int,
    ]
    set_info.restype = ctypes.c_long
    io_status = _IoStatusBlock()
    status = int(set_info(
        wintypes.HANDLE(child_handle),
        ctypes.byref(io_status),
        buffer,
        len(buffer),
        10,
    ))
    if status < 0:
        rtl_status_to_dos_error = ntdll.RtlNtStatusToDosError
        rtl_status_to_dos_error.argtypes = [wintypes.ULONG]
        rtl_status_to_dos_error.restype = wintypes.ULONG
        win_error = int(rtl_status_to_dos_error(status & 0xFFFFFFFF))
        raise ctypes.WinError(win_error)


def _write_new(path: Path, payload: bytes) -> None:
    write_durable_exact_bytes(path, payload, label=f"LT provider acquisition {path.name}")


def _validate_partial_output(directory: Path, expected: Mapping[str, bytes]) -> None:
    entries = {entry.name: entry for entry in directory.iterdir()}
    unexpected = set(entries).difference(expected)
    if unexpected:
        raise ValueError(
            f"LT provider acquisition staging has unexpected entries: {sorted(unexpected)}"
        )
    for name, path in entries.items():
        payload = expected[name]
        if read_stable_single_link_file(
            path,
            label=f"LT provider acquisition staged {name}",
            max_bytes=len(payload),
        ) != payload:
            raise ValueError("LT provider acquisition staging requires explicit repair")


def _validate_complete_output(directory: Path, expected: Mapping[str, bytes]) -> None:
    selected = assert_absolute_path_has_no_links(directory)
    if not selected.is_dir():
        raise ValueError("LT provider acquisition output is not a directory")
    entries = {entry.name: entry for entry in selected.iterdir()}
    if set(entries) != set(expected):
        raise ValueError("LT provider acquisition output requires explicit repair")
    _validate_partial_output(selected, expected)


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-spec", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        assert_installed_runtime_sealed()
    except ReleaseCliIdentityError as exc:
        _emit_failure(exc, retry_disposition="DO_NOT_RETRY_UNSEALED_RUNTIME")
        return EXIT_INTEGRITY_FAILURE
    args = _parse_args(argv)
    try:
        manifest = build_acquisition(
            capture_spec_path=args.capture_spec,
            output_directory=args.output_directory,
        )
    except (OSError, ValueError) as exc:
        _emit_failure(exc, retry_disposition="QUARANTINE_OR_EXACT_RETRY_AFTER_REPAIR")
        return EXIT_INTEGRITY_FAILURE
    print(manifest)
    return 0


def _emit_failure(exc: Exception, *, retry_disposition: str) -> None:
    print(
        json.dumps(
            {
                "command": "pfc-lt-build-acquisition",
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


if __name__ == "__main__":
    raise SystemExit(main())

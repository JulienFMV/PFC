#!/usr/bin/env python3
"""Build the isolated publisher dependency closure from approved wheels."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import sys
import tomllib
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.publisher_runtime_admission import (
    PublisherRuntimeAdmissionError,
    dependency_distribution_inventory,
    dependency_tree_inventory,
    python_runtime_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_CONTRACT = ROOT / "deploy" / "publisher" / "runtime-contract.json"
_DEFAULT_LOCK = ROOT / "uv.lock"
_RECEIPT_SCHEMA = "fmv_lt_snapshot_publisher_dependency_closure.v2"
_TARGET = {
    "python_tag": "cp311",
    "abi_policy": ["cp311", "abi3", "none"],
    "platform_tag": "win_amd64",
}
_EXTRACTION_POLICY = {
    "pth_files": "FORBIDDEN",
    "recorded_bytecode": "VERIFIED_THEN_EXCLUDED",
    "recorded_nested_wheels": "VERIFIED_THEN_EXCLUDED",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_MEMBER_BYTES = 256 * 1024 * 1024
_MAX_WHEEL_EXPANDED_BYTES = 768 * 1024 * 1024


def build_runtime_dependency_closure(
    wheel_directory: Path,
    output: Path,
    receipt_output: Path,
    *,
    lock_path: Path = _DEFAULT_LOCK,
    runtime_contract_path: Path = _RUNTIME_CONTRACT,
) -> dict[str, object]:
    """Verify locked wheels and extract their exact runtime payloads."""

    _assert_build_runtime()
    wheels_root = _absolute_existing_directory(wheel_directory, "wheel directory")
    target = _absolute_new_directory(output, "dependency closure output")
    receipt_path = _absolute_new_file(receipt_output, "dependency closure receipt")
    if _same_or_nested(receipt_path, target):
        raise ValueError("publisher dependency receipt must be outside the closure")
    lock_file, lock_payload = _stable_regular_file(
        lock_path, label="publisher source lock"
    )
    lock_sha256 = hashlib.sha256(lock_payload).hexdigest()
    _, contract_payload = _stable_regular_file(
        runtime_contract_path,
        label="runtime dependency contract",
    )
    contract = _strict_json_mapping(contract_payload)
    expected = _expected_distributions(contract)
    selected = _select_locked_wheels(
        lock_payload=lock_payload,
        lock_path=lock_file,
        wheel_directory=wheels_root,
        expected=expected,
    )

    extracted: dict[str, tuple[bytes, str]] = {}
    wheel_receipts: list[dict[str, object]] = []
    target_created = False
    try:
        target.mkdir()
        target_created = True
        for wheel in selected:
            archive_path, archive_payload = _stable_regular_file(
                wheels_root / str(wheel["filename"]),
                label="publisher wheel",
            )
            actual_sha256 = hashlib.sha256(archive_payload).hexdigest()
            if actual_sha256 != wheel["sha256"] or len(archive_payload) != wheel["size"]:
                raise ValueError(
                    f"publisher wheel differs from uv.lock: {archive_path.name}"
                )
            validated = _validate_wheel(
                archive_payload,
                filename=archive_path.name,
                expected_name=str(wheel["name"]),
                expected_version=str(wheel["version"]),
            )
            for installed_path, payload in validated["installed_payloads"].items():
                collision_key = installed_path.casefold()
                if collision_key in extracted:
                    previous_path = extracted[collision_key][1]
                    raise ValueError(
                        "publisher wheel installation path collision: "
                        f"{previous_path} vs {installed_path}"
                    )
                extracted[collision_key] = (payload, installed_path)
            wheel_receipts.append(
                {
                    "name": wheel["name"],
                    "version": wheel["version"],
                    "filename": archive_path.name,
                    "sha256": actual_sha256,
                    "size": len(archive_payload),
                    "record_sha256": validated["record_sha256"],
                    "wheel_tags": validated["wheel_tags"],
                    "installed_file_count": len(validated["installed_payloads"]),
                    "excluded_recorded_member_count": validated[
                        "excluded_recorded_member_count"
                    ],
                }
            )

        for payload, installed_path in sorted(
            extracted.values(), key=lambda item: item[1]
        ):
            destination = target.joinpath(*PurePosixPath(installed_path).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(payload)

        inventory = dependency_tree_inventory(target)
        installed = dependency_distribution_inventory(target)
        if installed != expected:
            raise ValueError(
                "publisher extracted distribution inventory diverges: "
                f"installed={installed}, expected={expected}"
            )
        receipt = {
            "schema_version": _RECEIPT_SCHEMA,
            "status": "PASS",
            "source_lock": {
                "path": lock_file.name,
                "sha256": lock_sha256,
            },
            "target": _TARGET,
            "extraction_policy": _EXTRACTION_POLICY,
            "tree_sha256": inventory["tree_sha256"],
            "file_count": inventory["file_count"],
            "distribution_count": len(installed),
            "wheel_provenance": "VERIFIED_ARCHIVES_AND_RECORD_AGAINST_UV_LOCK",
            "wheels": sorted(wheel_receipts, key=lambda item: str(item["name"])),
            "production_authorization": False,
        }
        receipt_payload = _canonical_json(receipt)
        write_durable_exact_bytes(
            receipt_path,
            receipt_payload,
            label="publisher dependency closure receipt",
        )
    except Exception:
        if target_created:
            shutil.rmtree(target, ignore_errors=True)
        raise
    return {
        **receipt,
        "dependency_root": str(target),
        "receipt_path": str(receipt_path),
        "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
    }


def validate_runtime_dependency_closure(
    wheel_directory: Path,
    dependency_root: Path,
    receipt_path: Path,
    *,
    lock_path: Path = _DEFAULT_LOCK,
    runtime_contract_path: Path = _RUNTIME_CONTRACT,
) -> dict[str, object]:
    """Replay a closure receipt against exact wheel archives and extracted bytes."""

    _assert_build_runtime()
    wheels_root = _absolute_existing_directory(wheel_directory, "wheel directory")
    closure_root = _absolute_existing_directory(
        dependency_root, "dependency closure"
    )
    _, receipt_payload = _stable_regular_file(
        receipt_path, label="publisher dependency closure receipt"
    )
    receipt = _strict_json_mapping(receipt_payload)
    if _canonical_json(receipt) != receipt_payload:
        raise ValueError("publisher dependency closure receipt is not canonical JSON")
    _validate_receipt_shape(receipt)

    lock_file, lock_payload = _stable_regular_file(
        lock_path, label="publisher source lock"
    )
    if receipt["source_lock"] != {
        "path": lock_file.name,
        "sha256": hashlib.sha256(lock_payload).hexdigest(),
    }:
        raise ValueError("publisher dependency receipt source lock diverges")
    _, contract_payload = _stable_regular_file(
        runtime_contract_path,
        label="runtime dependency contract",
    )
    contract = _strict_json_mapping(contract_payload)
    expected = _expected_distributions(contract)
    selected = _select_locked_wheels(
        lock_payload=lock_payload,
        lock_path=lock_file,
        wheel_directory=wheels_root,
        expected=expected,
    )
    expected_payloads: dict[str, tuple[bytes, str]] = {}
    expected_wheels: list[dict[str, object]] = []
    for wheel in selected:
        _, archive_payload = _stable_regular_file(
            wheels_root / str(wheel["filename"]), label="publisher wheel"
        )
        if (
            hashlib.sha256(archive_payload).hexdigest() != wheel["sha256"]
            or len(archive_payload) != wheel["size"]
        ):
            raise ValueError(
                f"publisher wheel differs from uv.lock: {wheel['filename']}"
            )
        validated = _validate_wheel(
            archive_payload,
            filename=str(wheel["filename"]),
            expected_name=str(wheel["name"]),
            expected_version=str(wheel["version"]),
        )
        for installed_path, payload in validated["installed_payloads"].items():
            collision_key = installed_path.casefold()
            if collision_key in expected_payloads:
                raise ValueError(
                    f"publisher wheel installation path collision: {installed_path}"
                )
            expected_payloads[collision_key] = (payload, installed_path)
        expected_wheels.append(
            {
                "name": wheel["name"],
                "version": wheel["version"],
                "filename": wheel["filename"],
                "sha256": wheel["sha256"],
                "size": wheel["size"],
                "record_sha256": validated["record_sha256"],
                "wheel_tags": validated["wheel_tags"],
                "installed_file_count": len(validated["installed_payloads"]),
                "excluded_recorded_member_count": validated[
                    "excluded_recorded_member_count"
                ],
            }
        )
    if receipt["wheels"] != sorted(expected_wheels, key=lambda item: str(item["name"])):
        raise ValueError("publisher dependency receipt wheel evidence diverges")

    closure_inventory = dependency_tree_inventory(closure_root)
    if closure_inventory != {
        "tree_sha256": receipt["tree_sha256"],
        "file_count": receipt["file_count"],
    }:
        raise ValueError("publisher dependency closure tree diverges from its receipt")
    actual_paths: dict[str, Path] = {}
    for path in closure_root.rglob("*"):
        if path.is_file():
            relative = path.relative_to(closure_root).as_posix()
            key = relative.casefold()
            if key in actual_paths:
                raise ValueError("publisher dependency closure has a case collision")
            actual_paths[key] = path
    if set(actual_paths) != set(expected_payloads):
        raise ValueError("publisher dependency closure inventory differs from wheels")
    for key, path in actual_paths.items():
        _, actual_payload = _stable_regular_file(
            path, label="publisher extracted dependency"
        )
        if actual_payload != expected_payloads[key][0]:
            raise ValueError(
                "publisher dependency closure bytes differ from wheel RECORD: "
                f"{expected_payloads[key][1]}"
            )
    installed = dependency_distribution_inventory(closure_root)
    if installed != expected or receipt["distribution_count"] != len(installed):
        raise ValueError("publisher dependency closure distributions diverge")
    return {
        **receipt,
        "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument(
        "--runtime-contract",
        type=Path,
        default=_RUNTIME_CONTRACT,
    )
    args = parser.parse_args(argv)
    result = build_runtime_dependency_closure(
        args.wheel_directory,
        args.output,
        args.receipt_output,
        runtime_contract_path=args.runtime_contract,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _select_locked_wheels(
    *,
    lock_payload: bytes,
    lock_path: Path,
    wheel_directory: Path,
    expected: dict[str, str],
) -> list[dict[str, object]]:
    try:
        lock = tomllib.loads(lock_payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"publisher source lock is invalid: {lock_path}") from exc
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise ValueError("publisher source lock package inventory is invalid")
    by_name: dict[str, dict[str, object]] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("publisher source lock package entry is invalid")
        name = _normalize_distribution_name(str(package.get("name", "")))
        if name in expected:
            if name in by_name:
                raise ValueError(f"publisher source lock package is ambiguous: {name}")
            by_name[name] = package
    if set(by_name) != set(expected):
        raise ValueError("publisher source lock omits a required distribution")

    selected: list[dict[str, object]] = []
    for normalized_name, version in sorted(expected.items()):
        package = by_name[normalized_name]
        if str(package.get("version", "")) != version:
            raise ValueError(
                f"publisher source lock version diverges for {normalized_name}"
            )
        candidates: list[tuple[int, str, dict[str, object]]] = []
        wheels = package.get("wheels")
        if not isinstance(wheels, list):
            raise ValueError(f"publisher source lock has no wheels: {normalized_name}")
        for item in wheels:
            if (
                not isinstance(item, dict)
                or not {"url", "hash", "size"}.issubset(item)
                or not set(item).issubset({"url", "hash", "size", "upload-time"})
            ):
                raise ValueError("publisher source lock wheel entry is invalid")
            filename = unquote(Path(urlsplit(str(item["url"])).path).name)
            priority = _wheel_filename_priority(filename)
            if priority is None or not (wheel_directory / filename).is_file():
                continue
            raw_hash = str(item["hash"])
            size = item["size"]
            if not raw_hash.startswith("sha256:") or not isinstance(size, int) or size <= 0:
                raise ValueError("publisher source lock wheel hash or size is invalid")
            digest = raw_hash.removeprefix("sha256:")
            if _SHA256.fullmatch(digest) is None:
                raise ValueError("publisher source lock wheel digest is invalid")
            candidates.append(
                (
                    priority,
                    filename,
                    {
                        "name": normalized_name,
                        "version": version,
                        "filename": filename,
                        "sha256": digest,
                        "size": size,
                    },
                )
            )
        if not candidates:
            raise ValueError(
                f"publisher compatible locked wheel is unavailable: {normalized_name}"
            )
        candidates.sort(key=lambda item: (item[0], item[1]))
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            raise ValueError(
                f"publisher compatible locked wheel selection is ambiguous: {normalized_name}"
            )
        selected.append(candidates[0][2])
    return selected


def _validate_wheel(
    payload: bytes,
    *,
    filename: str,
    expected_name: str,
    expected_version: str,
) -> dict[str, object]:
    if _wheel_filename_priority(filename) is None:
        raise ValueError(f"publisher wheel target tag is incompatible: {filename}")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"publisher wheel is not a valid ZIP: {filename}") from exc
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError(f"publisher wheel has duplicate members: {filename}")
        folded = [name.casefold() for name in names]
        if len(folded) != len(set(folded)):
            raise ValueError(f"publisher wheel has case-colliding members: {filename}")
        file_infos: dict[str, zipfile.ZipInfo] = {}
        expanded_size = 0
        for info in infos:
            relative = _safe_wheel_member(info.filename)
            _validate_zip_member_type(info, filename=filename)
            if info.is_dir():
                continue
            if info.flag_bits & 0x1:
                raise ValueError(f"publisher wheel has an encrypted member: {filename}")
            if info.file_size < 0 or info.file_size > _MAX_MEMBER_BYTES:
                raise ValueError(f"publisher wheel member size is unsafe: {filename}")
            expanded_size += info.file_size
            if expanded_size > _MAX_WHEEL_EXPANDED_BYTES:
                raise ValueError(f"publisher wheel expanded size is unsafe: {filename}")
            _reject_forbidden_installed_path(relative, filename=filename)
            file_infos[relative.as_posix()] = info

        metadata_paths = sorted(
            name for name in file_infos if name.endswith(".dist-info/METADATA")
        )
        if len(metadata_paths) != 1:
            raise ValueError(f"publisher wheel METADATA inventory is invalid: {filename}")
        dist_info = metadata_paths[0].removesuffix("/METADATA")
        wheel_path = f"{dist_info}/WHEEL"
        record_path = f"{dist_info}/RECORD"
        if wheel_path not in file_infos or record_path not in file_infos:
            raise ValueError(f"publisher wheel metadata closure is incomplete: {filename}")
        metadata_payload = archive.read(metadata_paths[0])
        metadata = BytesParser(policy=policy.compat32).parsebytes(metadata_payload)
        names_header = metadata.get_all("Name", [])
        versions_header = metadata.get_all("Version", [])
        if (
            len(names_header) != 1
            or len(versions_header) != 1
            or _normalize_distribution_name(str(names_header[0])) != expected_name
            or str(versions_header[0]).strip() != expected_version
        ):
            raise ValueError(f"publisher wheel name/version metadata diverges: {filename}")

        wheel_metadata = BytesParser(policy=policy.compat32).parsebytes(
            archive.read(wheel_path)
        )
        tags = sorted(str(tag).strip() for tag in wheel_metadata.get_all("Tag", []))
        filename_tags = _expanded_filename_tags(filename)
        if (
            not tags
            or set(tags) != filename_tags
            or not any(_wheel_tag_priority(tag) is not None for tag in tags)
        ):
            raise ValueError(f"publisher wheel filename and WHEEL tags diverge: {filename}")

        member_payloads = {name: archive.read(name) for name in file_infos}
        _validate_record(
            member_payloads,
            record_path=record_path,
            filename=filename,
        )
        installed_payloads: dict[str, bytes] = {}
        excluded_recorded_member_count = 0
        for source_path, member_payload in member_payloads.items():
            if _excluded_recorded_member(source_path):
                excluded_recorded_member_count += 1
                continue
            installed_path = _installed_wheel_path(source_path, filename=filename)
            collision_key = installed_path.casefold()
            if any(key.casefold() == collision_key for key in installed_payloads):
                raise ValueError(
                    f"publisher wheel installed path collision: {filename}"
                )
            installed_payloads[installed_path] = member_payload
        return {
            "installed_payloads": installed_payloads,
            "record_sha256": hashlib.sha256(member_payloads[record_path]).hexdigest(),
            "wheel_tags": tags,
            "excluded_recorded_member_count": excluded_recorded_member_count,
        }


def _validate_record(
    member_payloads: dict[str, bytes],
    *,
    record_path: str,
    filename: str,
) -> None:
    try:
        rows = list(csv.reader(io.StringIO(member_payloads[record_path].decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"publisher wheel RECORD is invalid: {filename}") from exc
    records: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3:
            raise ValueError(f"publisher wheel RECORD row is invalid: {filename}")
        path, digest, size = row
        safe_path = _safe_wheel_member(path).as_posix()
        if safe_path in records:
            raise ValueError(f"publisher wheel RECORD has duplicate paths: {filename}")
        records[safe_path] = (digest, size)
    if set(records) != set(member_payloads):
        raise ValueError(f"publisher wheel RECORD inventory diverges: {filename}")
    for path, payload in member_payloads.items():
        digest, raw_size = records[path]
        if path == record_path:
            if digest or raw_size:
                raise ValueError(
                    f"publisher wheel RECORD self-entry must be unhashed: {filename}"
                )
            continue
        if not digest.startswith("sha256=") or not raw_size.isdecimal():
            raise ValueError(f"publisher wheel RECORD proof is incomplete: {filename}")
        encoded = digest.removeprefix("sha256=")
        try:
            decoded = base64.b64decode(
                encoded + "=" * (-len(encoded) % 4),
                altchars=b"-_",
                validate=True,
            )
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError(f"publisher wheel RECORD digest is invalid: {filename}") from exc
        if decoded != hashlib.sha256(payload).digest() or int(raw_size) != len(payload):
            raise ValueError(f"publisher wheel RECORD hash or size diverges: {filename}")


def _installed_wheel_path(source_path: str, *, filename: str) -> str:
    relative = PurePosixPath(source_path)
    if len(relative.parts) >= 3 and relative.parts[0].endswith(".data"):
        scheme = relative.parts[1]
        if scheme not in {"purelib", "platlib"}:
            raise ValueError(
                f"publisher wheel contains unsupported .data scheme {scheme}: {filename}"
            )
        relative = PurePosixPath(*relative.parts[2:])
        if not relative.parts:
            raise ValueError(f"publisher wheel .data path is empty: {filename}")
    return relative.as_posix()


def _safe_wheel_member(name: str) -> PurePosixPath:
    if not name or "\\" in name or "\x00" in name:
        raise ValueError("publisher wheel contains an unsafe member path")
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or not relative.parts
        or ":" in relative.parts[0]
        or relative.as_posix() != name.rstrip("/")
    ):
        raise ValueError("publisher wheel contains an unsafe member path")
    return relative


def _reject_forbidden_installed_path(relative: PurePosixPath, *, filename: str) -> None:
    lowered = relative.as_posix().lower()
    if lowered.endswith(".pth"):
        raise ValueError(f"publisher wheel contains a forbidden runtime path: {filename}")


def _excluded_recorded_member(source_path: str) -> bool:
    relative = PurePosixPath(source_path)
    lowered = source_path.lower()
    return (
        lowered.endswith((".pyc", ".pyo", ".whl"))
        or "__pycache__" in {part.lower() for part in relative.parts}
    )


def _validate_zip_member_type(info: zipfile.ZipInfo, *, filename: str) -> None:
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if stat.S_ISLNK(mode) or file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ValueError(f"publisher wheel contains a linked/special member: {filename}")


def _wheel_filename_priority(filename: str) -> int | None:
    if not filename.lower().endswith(".whl"):
        return None
    try:
        _, python_tag, abi_tag, platform_tag = filename[:-4].rsplit("-", 3)
    except ValueError:
        return None
    return _tag_components_priority(python_tag, abi_tag, platform_tag)


def _expanded_filename_tags(filename: str) -> set[str]:
    try:
        _, python_tag, abi_tag, platform_tag = filename[:-4].rsplit("-", 3)
    except ValueError:
        return set()
    return {
        f"{python}-{abi}-{platform_tag}"
        for python in python_tag.split(".")
        for abi in abi_tag.split(".")
    }


def _wheel_tag_priority(tag: str) -> int | None:
    try:
        python_tag, abi_tag, platform_tag = tag.split("-", 2)
    except ValueError:
        return None
    return _tag_components_priority(python_tag, abi_tag, platform_tag)


def _tag_components_priority(
    python_tag: str, abi_tag: str, platform_tag: str
) -> int | None:
    if (python_tag, abi_tag, platform_tag) == ("cp311", "cp311", "win_amd64"):
        return 0
    if (python_tag, abi_tag, platform_tag) == ("cp311", "abi3", "win_amd64"):
        return 1
    python_tags = set(python_tag.split("."))
    if abi_tag == "none" and platform_tag == "any" and python_tags in (
        {"py3"},
        {"py2", "py3"},
    ):
        return 2
    return None


def _validate_receipt_shape(receipt: dict[str, object]) -> None:
    if set(receipt) != {
        "schema_version",
        "status",
        "source_lock",
        "target",
        "extraction_policy",
        "tree_sha256",
        "file_count",
        "distribution_count",
        "wheel_provenance",
        "wheels",
        "production_authorization",
    }:
        raise ValueError("publisher dependency closure receipt shape is invalid")
    if (
        receipt.get("schema_version") != _RECEIPT_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("target") != _TARGET
        or receipt.get("extraction_policy") != _EXTRACTION_POLICY
        or receipt.get("wheel_provenance")
        != "VERIFIED_ARCHIVES_AND_RECORD_AGAINST_UV_LOCK"
        or receipt.get("production_authorization") is not False
    ):
        raise ValueError("publisher dependency closure receipt policy is invalid")
    if _SHA256.fullmatch(str(receipt.get("tree_sha256", ""))) is None:
        raise ValueError("publisher dependency closure receipt digest is invalid")
    if not isinstance(receipt.get("file_count"), int) or receipt["file_count"] <= 0:
        raise ValueError("publisher dependency closure receipt file count is invalid")
    if not isinstance(receipt.get("distribution_count"), int) or receipt[
        "distribution_count"
    ] <= 0:
        raise ValueError("publisher dependency closure receipt distribution count is invalid")
    source_lock = receipt.get("source_lock")
    if (
        not isinstance(source_lock, dict)
        or set(source_lock) != {"path", "sha256"}
        or source_lock.get("path") != "uv.lock"
        or _SHA256.fullmatch(str(source_lock.get("sha256", ""))) is None
    ):
        raise ValueError("publisher dependency closure receipt source lock is invalid")
    wheels = receipt.get("wheels")
    if not isinstance(wheels, list) or len(wheels) != receipt["distribution_count"]:
        raise ValueError("publisher dependency closure receipt wheels are invalid")
    wheel_keys = {
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
    for wheel in wheels:
        if (
            not isinstance(wheel, dict)
            or set(wheel) != wheel_keys
            or _SHA256.fullmatch(str(wheel.get("sha256", ""))) is None
            or _SHA256.fullmatch(str(wheel.get("record_sha256", ""))) is None
            or not isinstance(wheel.get("size"), int)
            or wheel["size"] <= 0
            or not isinstance(wheel.get("installed_file_count"), int)
            or wheel["installed_file_count"] <= 0
            or not isinstance(wheel.get("excluded_recorded_member_count"), int)
            or wheel["excluded_recorded_member_count"] < 0
            or not isinstance(wheel.get("wheel_tags"), list)
            or not wheel["wheel_tags"]
        ):
            raise ValueError("publisher dependency closure receipt wheel entry is invalid")


def _expected_distributions(contract: dict[str, object]) -> dict[str, str]:
    items = contract.get("locked_distributions")
    if not isinstance(items, list) or not items:
        raise ValueError("publisher runtime distribution contract is unavailable")
    expected: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict) or set(item) != {"name", "version"}:
            raise ValueError("publisher runtime distribution entry is invalid")
        normalized = _normalize_distribution_name(str(item["name"]))
        version = str(item["version"])
        if not normalized or not version or normalized in expected:
            raise ValueError("publisher runtime distribution contract is ambiguous")
        expected[normalized] = version
    return expected


def _assert_build_runtime() -> None:
    try:
        fingerprint = python_runtime_fingerprint()
    except PublisherRuntimeAdmissionError as exc:
        raise ValueError(str(exc)) from exc
    if (
        sys.platform != "win32"
        or fingerprint.get("abi") != "cp311"
        or fingerprint.get("pointer_bits") != 64
    ):
        raise ValueError("publisher closure build target must be CPython 3.11 win_amd64")


def _absolute_existing_directory(path: Path, label: str) -> Path:
    selected = path.expanduser()
    if not selected.is_absolute() or not selected.is_dir():
        raise ValueError(f"publisher {label} is unavailable")
    path_stat = selected.stat(follow_symlinks=False)
    if _is_link_or_reparse(selected, path_stat):
        raise ValueError(f"publisher {label} is linked")
    return selected


def _absolute_new_directory(path: Path, label: str) -> Path:
    selected = path.expanduser()
    if not selected.is_absolute() or not selected.parent.is_dir():
        raise ValueError(f"publisher {label} parent is unavailable")
    if selected.exists():
        raise ValueError(f"publisher {label} must not already exist")
    return selected


def _absolute_new_file(path: Path, label: str) -> Path:
    selected = path.expanduser()
    if not selected.is_absolute() or not selected.parent.is_dir():
        raise ValueError(f"publisher {label} parent is unavailable")
    if selected.exists():
        raise ValueError(f"publisher {label} must not already exist")
    return selected


def _stable_regular_file(path: Path, *, label: str) -> tuple[Path, bytes]:
    selected = path.expanduser()
    if not selected.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    try:
        before = selected.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if (
        _is_link_or_reparse(selected, before)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
    ):
        raise ValueError(f"{label} must be a mono-linked regular file")
    with selected.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        if _file_identity(opened_before) != _file_identity(before):
            raise ValueError(f"{label} changed before capture")
        payload = handle.read()
        opened_after = os.fstat(handle.fileno())
    try:
        after = selected.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} disappeared after capture") from exc
    if not (
        _file_identity(before)
        == _file_identity(opened_before)
        == _file_identity(opened_after)
        == _file_identity(after)
    ):
        raise ValueError(f"{label} changed during capture")
    return selected, payload


def _strict_json_mapping(payload: bytes) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        parsed = json.loads(
            payload.decode("utf-8"), object_pairs_hook=reject_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("publisher JSON document is invalid") from exc
    if not isinstance(parsed, dict):
        raise ValueError("publisher JSON document must be a mapping")
    return parsed


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _same_or_nested(path: Path, parent: Path) -> bool:
    try:
        return path == parent or path.is_relative_to(parent)
    except ValueError:
        return False


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


if __name__ == "__main__":
    raise SystemExit(main())

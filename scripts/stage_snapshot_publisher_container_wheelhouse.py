#!/usr/bin/env python3
"""Stage the exact lock-bound publisher wheels for a Windows container build."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)
from scripts.build_snapshot_publisher_runtime_closure import (
    _DEFAULT_LOCK,
    _RUNTIME_CONTRACT,
    _assert_build_runtime,
    _expected_distributions,
    _select_locked_wheels,
    _strict_json_mapping,
)

STAGING_MANIFEST_NAME = "publisher-container-wheelhouse-manifest.json"


def stage_container_wheelhouse(
    source: Path,
    output: Path,
    manifest_output: Path,
    *,
    lock_path: Path = _DEFAULT_LOCK,
    runtime_contract_path: Path = _RUNTIME_CONTRACT,
) -> dict[str, object]:
    """Copy exactly the admitted wheel bytes and seal their inventory."""

    _assert_build_runtime()
    source_root = _existing_unlinked_directory(source, label="source wheelhouse")
    target = _new_directory(output, label="container wheel staging")
    manifest_path = _new_file(manifest_output, label="wheel staging manifest")
    if manifest_path != target.parent / STAGING_MANIFEST_NAME:
        raise ValueError("publisher wheel staging manifest path is not canonical")

    lock_file = assert_absolute_path_has_no_links(lock_path.expanduser())
    lock_payload = read_stable_single_link_file(lock_file, label="publisher source lock")
    contract_file = assert_absolute_path_has_no_links(runtime_contract_path.expanduser())
    contract_payload = read_stable_single_link_file(
        contract_file,
        label="publisher runtime contract",
    )
    contract = _strict_json_mapping(contract_payload)
    selected = _select_locked_wheels(
        lock_payload=lock_payload,
        lock_path=lock_file,
        wheel_directory=source_root,
        expected=_expected_distributions(contract),
    )
    expected_names = {str(item["filename"]) for item in selected}
    present_entries = list(source_root.iterdir())
    if (
        len(present_entries) != len(expected_names)
        or {entry.name for entry in present_entries} != expected_names
    ):
        raise ValueError("publisher source wheelhouse inventory is not exact")

    payloads: dict[str, bytes] = {}
    for item in selected:
        filename = str(item["filename"])
        payload = read_stable_single_link_file(
            source_root / filename,
            label="publisher source wheel",
        )
        if (
            len(payload) != int(item["size"])
            or hashlib.sha256(payload).hexdigest() != item["sha256"]
        ):
            raise ValueError(f"publisher source wheel differs from uv.lock: {filename}")
        payloads[filename] = payload

    target.mkdir()
    for filename, payload in sorted(payloads.items()):
        destination = target / filename
        write_durable_exact_bytes(
            destination,
            payload,
            label="publisher staged wheel",
        )
        replay = read_stable_single_link_file(destination, label="publisher staged wheel")
        if replay != payload:
            raise ValueError("publisher staged wheel replay differs")

    result = {
        "schema_version": "fmv_lt_snapshot_publisher_wheelhouse_staging.v1",
        "status": "PASS",
        "source_lock": {
            "path": "uv.lock",
            "sha256": hashlib.sha256(lock_payload).hexdigest(),
        },
        "wheel_count": len(selected),
        "wheels": sorted(selected, key=lambda item: str(item["name"])),
        "production_authorization": False,
    }
    write_durable_exact_bytes(
        manifest_path,
        _canonical_json(result),
        label="publisher wheel staging manifest",
    )
    return result


def audit_staged_container_wheelhouse(
    source: Path,
    manifest_output: Path,
    *,
    lock_path: Path = _DEFAULT_LOCK,
    runtime_contract_path: Path = _RUNTIME_CONTRACT,
) -> dict[str, object]:
    """Re-admit exact staged bytes immediately before Docker context capture."""

    _assert_build_runtime()
    source_root = _existing_unlinked_directory(source, label="staged wheelhouse")
    manifest_path = assert_absolute_path_has_no_links(manifest_output.expanduser())
    if manifest_path != source_root.parent / STAGING_MANIFEST_NAME:
        raise ValueError("publisher wheel staging manifest path is not canonical")
    manifest_payload = read_stable_single_link_file(
        manifest_path,
        label="publisher wheel staging manifest",
    )

    lock_file = assert_absolute_path_has_no_links(lock_path.expanduser())
    lock_payload = read_stable_single_link_file(lock_file, label="publisher source lock")
    contract_file = assert_absolute_path_has_no_links(runtime_contract_path.expanduser())
    contract_payload = read_stable_single_link_file(
        contract_file,
        label="publisher runtime contract",
    )
    contract = _strict_json_mapping(contract_payload)
    selected = _select_locked_wheels(
        lock_payload=lock_payload,
        lock_path=lock_file,
        wheel_directory=source_root,
        expected=_expected_distributions(contract),
    )
    expected_names = {str(item["filename"]) for item in selected}
    present_entries = list(source_root.iterdir())
    if (
        len(present_entries) != len(expected_names)
        or {entry.name for entry in present_entries} != expected_names
    ):
        raise ValueError("publisher staged wheelhouse inventory is not exact")
    for item in selected:
        filename = str(item["filename"])
        payload = read_stable_single_link_file(
            source_root / filename,
            label="publisher staged wheel",
        )
        if (
            len(payload) != int(item["size"])
            or hashlib.sha256(payload).hexdigest() != item["sha256"]
        ):
            raise ValueError(f"publisher staged wheel differs from uv.lock: {filename}")

    result = {
        "schema_version": "fmv_lt_snapshot_publisher_wheelhouse_staging.v1",
        "status": "PASS",
        "source_lock": {
            "path": "uv.lock",
            "sha256": hashlib.sha256(lock_payload).hexdigest(),
        },
        "wheel_count": len(selected),
        "wheels": sorted(selected, key=lambda item: str(item["name"])),
        "production_authorization": False,
    }
    if manifest_payload != _canonical_json(result):
        raise ValueError("publisher wheel staging manifest differs from staged bytes")
    return result


def _existing_unlinked_directory(path: Path, *, label: str) -> Path:
    selected = path.expanduser()
    if not selected.is_absolute():
        raise ValueError(f"publisher {label} path must be absolute")
    selected = assert_absolute_path_has_no_links(selected)
    if not selected.is_dir():
        raise ValueError(f"publisher {label} is unavailable")
    return selected


def _new_directory(path: Path, *, label: str) -> Path:
    selected = path.expanduser()
    if not selected.is_absolute():
        raise ValueError(f"publisher {label} path must be absolute")
    parent = assert_absolute_path_has_no_links(selected.parent)
    if not parent.is_dir() or selected.exists():
        raise ValueError(f"publisher {label} must be a new directory")
    return selected


def _new_file(path: Path, *, label: str) -> Path:
    selected = path.expanduser()
    if not selected.is_absolute():
        raise ValueError(f"publisher {label} path must be absolute")
    parent = assert_absolute_path_has_no_links(selected.parent)
    if not parent.is_dir() or selected.exists():
        raise ValueError(f"publisher {label} must be a new file")
    return selected


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args(argv)
    if args.audit:
        if args.output is not None:
            parser.error("--output is forbidden with --audit")
        result = audit_staged_container_wheelhouse(
            args.source,
            args.manifest_output,
        )
    else:
        if args.output is None:
            parser.error("--output is required unless --audit is selected")
        result = stage_container_wheelhouse(
            args.source,
            args.output,
            args.manifest_output,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

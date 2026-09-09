#!/usr/bin/env python3
"""Archive reusable ENTSO-related parquets without making them LT-eligible."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import sys
from collections.abc import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.data.lt_input_sources import dataframe_sha256  # noqa: E402
from pfc_shaping.data.shared_data_root import (  # noqa: E402
    FMV_DATA_ROOT_ENV,
    PFC_ENTSOE_DATA_ROOT_ENV,
    PFC_SHARED_DATA_ROOT_ENV,
    resolve_data_domain_root,
    resolve_fmv_data_root,
)


ENTSO_FILES = (
    "entso_15min.parquet",
    "entso_border_15min.parquet",
    "entso_fundamentals_15min.parquet",
    "de_renewable_forecast.parquet",
    "fr_nuclear_forecast_15min.parquet",
    "multi_country_forecast_15min.parquet",
    "outages_15min.parquet",
)
_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def archive_entso_dataset(
    *,
    source_dir: Path,
    data_root: Path | None = None,
    entsoe_root: Path | None = None,
    import_id: str,
    source_label: str,
    excluded_files: Iterable[str] = (),
) -> Path:
    source_dir = Path(source_dir).expanduser()
    if not source_dir.is_absolute():
        raise ValueError("ENTSO source_dir must be absolute")
    entsoe_store = _resolve_entsoe_root(data_root=data_root, entsoe_root=entsoe_root)
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"ENTSO source_dir does not exist: {source_dir}")
    source_receipts = _source_parquet_receipts(source_dir)
    source_parquets = sorted(source_receipts)
    explicit_exclusions = sorted({_validate_relative_source_name(value) for value in excluded_files})
    unknown = sorted(set(source_parquets).difference(ENTSO_FILES).difference(explicit_exclusions))
    missing_exclusions = sorted(set(explicit_exclusions).difference(source_parquets))
    invalid_exclusions = sorted(set(explicit_exclusions).intersection(ENTSO_FILES))
    if missing_exclusions:
        raise ValueError(f"excluded source parquet is absent: {missing_exclusions}")
    if invalid_exclusions:
        raise ValueError(f"governed ENTSO parquet cannot be excluded: {invalid_exclusions}")
    if unknown:
        raise ValueError(
            "source parquet must be archived or excluded explicitly: "
            f"{unknown}"
        )
    _validate_portable_id(import_id, label="import_id")
    _validate_source_label(source_label)
    imports = (entsoe_store / "imports").resolve()
    if not imports.is_relative_to(entsoe_store):
        raise ValueError("ENTSO imports directory resolves outside entsoe_root")
    imports.mkdir(parents=True, exist_ok=True)
    if imports.resolve() != imports:
        raise ValueError("ENTSO imports directory changed during initialization")
    final = imports / import_id
    staging = imports / f".{import_id}.staging-{os.getpid()}"
    if final.exists() or staging.exists():
        raise FileExistsError(f"ENTSO import already exists: {final}")
    staging.mkdir()
    files: dict[str, object] = {}
    try:
        for name in ENTSO_FILES:
            source = source_dir / name
            if not source.is_file():
                continue
            payload = source.read_bytes()
            if hashlib.sha256(payload).hexdigest() != source_receipts[name]["sha256"]:
                raise ValueError(f"ENTSO source changed during archival: {name}")
            destination = staging / name
            _write_bytes(destination, payload)
            files[name] = _parquet_metadata(destination, payload)
        if not files:
            raise FileNotFoundError(f"no ENTSO-related parquet found in {source_dir}")
        if _source_parquet_receipts(source_dir) != source_receipts:
            raise ValueError("ENTSO source inventory changed during archival")
        manifest = {
            "schema_version": "entso_reusable_import.v2",
            "import_id": import_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_label": source_label,
            "calibration_eligible": False,
            "model_activation": "FORBIDDEN_UNTIL_GOVERNED_IMPORT",
            "source_coherence": "UNVERIFIED_MULTI_FILE_IMPORT",
            "source_parquet_inventory": source_parquets,
            "source_parquet_receipts": source_receipts,
            "excluded_source_parquets": explicit_exclusions,
            "files": files,
        }
        manifest_path = staging / "manifest.json"
        _write_json(manifest_path, manifest)
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        archive_content_hash = hashlib.sha256(
            json.dumps(
                {name: item["sha256"] for name, item in sorted(files.items())},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        _write_json(
            staging / "archive_receipt.json",
            {
                "schema_version": "entso_reusable_import_receipt.v1",
                "manifest_sha256": manifest_hash,
                "archive_content_sha256": archive_content_hash,
            },
        )
        _verify_entso_archive(staging, expected_import_id=import_id)
        os.replace(staging, final)
        verify_entso_archive(final)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return final


def _resolve_entsoe_root(
    *,
    data_root: Path | None,
    entsoe_root: Path | None,
) -> Path:
    """Resolve the direct ENTSO-E domain root before any filesystem mutation."""

    try:
        return resolve_data_domain_root(
            "entsoe",
            data_root=data_root,
            domain_root=entsoe_root,
            compatibility_envs=(PFC_SHARED_DATA_ROOT_ENV,),
            domain_env=PFC_ENTSOE_DATA_ROOT_ENV,
        )
    except ValueError as exc:
        message = str(exc).replace("entsoe domain_root", "ENTSO entsoe_root")
        message = message.replace("FMV data_root", "ENTSO data_root")
        raise ValueError(message) from exc


def _parquet_metadata(path: Path, payload: bytes) -> dict[str, object]:
    frame = pd.read_parquet(path)
    return _frame_metadata(frame, payload, path.name)


def _frame_metadata(
    frame: pd.DataFrame,
    payload: bytes,
    logical_path: str,
) -> dict[str, object]:
    index = frame.index
    return {
        "path": logical_path,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(frame),
        "frame_sha256": dataframe_sha256(frame),
        "columns": [str(value) for value in frame.columns],
        "index_min": str(index.min()),
        "index_max": str(index.max()),
    }


def _source_parquet_receipts(source_dir: Path) -> dict[str, dict[str, object]]:
    receipts: dict[str, dict[str, object]] = {}
    source_root = source_dir.resolve()
    for path in sorted(source_dir.rglob("*.parquet")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(source_root) or _is_link_or_junction(path):
            raise ValueError(f"ENTSO source parquet escapes or links outside source root: {path}")
        logical_path = path.relative_to(source_dir).as_posix()
        payload = path.read_bytes()
        receipts[logical_path] = {
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return receipts


def verify_entso_archive(path: Path) -> dict[str, object]:
    return _verify_entso_archive(path, expected_import_id=None)


def _verify_entso_archive(
    path: Path,
    *,
    expected_import_id: str | None,
) -> dict[str, object]:
    archive = path.resolve()
    manifest_path = archive / "manifest.json"
    receipt_path = archive / "archive_receipt.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version")
    if schema_version not in {"entso_reusable_import.v1", "entso_reusable_import.v2"}:
        raise ValueError("invalid ENTSO import manifest schema")
    import_id = str(manifest.get("import_id", ""))
    _validate_portable_id(import_id, label="manifest import_id")
    if archive.name != import_id and expected_import_id != import_id:
        raise ValueError("ENTSO import directory name does not match manifest import_id")
    if manifest.get("calibration_eligible") is not False:
        raise ValueError("reusable ENTSO import cannot be calibration eligible")
    if manifest.get("model_activation") != "FORBIDDEN_UNTIL_GOVERNED_IMPORT":
        raise ValueError("reusable ENTSO import model activation must remain forbidden")
    if receipt.get("schema_version") != "entso_reusable_import_receipt.v1":
        raise ValueError("invalid ENTSO import receipt schema")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_hash != receipt.get("manifest_sha256"):
        raise ValueError("ENTSO import manifest hash mismatch")
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("ENTSO import file manifest is empty")
    if schema_version == "entso_reusable_import.v2":
        inventory = manifest.get("source_parquet_inventory")
        exclusions = manifest.get("excluded_source_parquets")
        source_receipts = manifest.get("source_parquet_receipts")
        if (
            not isinstance(inventory, list)
            or not isinstance(exclusions, list)
            or not isinstance(source_receipts, dict)
        ):
            raise ValueError("ENTSO v2 import source inventory is missing")
        if len(inventory) != len(set(inventory)) or len(exclusions) != len(set(exclusions)):
            raise ValueError("ENTSO v2 import source inventory contains duplicates")
        inventory_set = {_validate_relative_source_name(value) for value in inventory}
        exclusion_set = {_validate_relative_source_name(value) for value in exclusions}
        if set(declared).intersection(exclusion_set):
            raise ValueError("ENTSO v2 archived and excluded source sets overlap")
        if inventory_set != set(declared).union(exclusion_set):
            raise ValueError("ENTSO v2 source inventory partition mismatch")
        if set(source_receipts) != inventory_set:
            raise ValueError("ENTSO v2 source receipt inventory mismatch")
        unsupported = sorted(set(declared).difference(ENTSO_FILES))
        if unsupported:
            raise ValueError(f"ENTSO v2 manifest contains unsupported archive files: {unsupported}")
        for name, entry in declared.items():
            if not isinstance(entry, dict) or entry.get("path") != name:
                raise ValueError(f"ENTSO v2 manifest path mismatch: {name}")
        for name, source_receipt in source_receipts.items():
            if not isinstance(source_receipt, dict):
                raise ValueError(f"ENTSO v2 source receipt is invalid: {name}")
            if int(source_receipt.get("size_bytes", -1)) < 0 or not _is_sha256(
                source_receipt.get("sha256")
            ):
                raise ValueError(f"ENTSO v2 source receipt is incomplete: {name}")
    actual_names = {
        item.relative_to(archive).as_posix()
        for item in archive.rglob("*")
        if item.is_file()
        and item.relative_to(archive).as_posix()
        not in {"manifest.json", "archive_receipt.json"}
    }
    if actual_names != set(declared):
        raise ValueError("ENTSO import file set mismatch")
    for name, expected in declared.items():
        source = (archive / name).resolve()
        if not source.is_relative_to(archive):
            raise ValueError(f"ENTSO import file escapes archive: {name}")
        payload = source.read_bytes()
        if len(payload) != int(expected.get("size_bytes", -1)):
            raise ValueError(f"ENTSO import file size mismatch: {name}")
        if hashlib.sha256(payload).hexdigest() != expected.get("sha256"):
            raise ValueError(f"ENTSO import file hash mismatch: {name}")
        frame = pd.read_parquet(BytesIO(payload))
        actual_metadata = _frame_metadata(frame, payload, name)
        for key in ("rows", "frame_sha256", "columns", "index_min", "index_max"):
            if actual_metadata[key] != expected.get(key):
                raise ValueError(f"ENTSO import {key} mismatch: {name}")
        if schema_version == "entso_reusable_import.v2":
            source_receipt = manifest["source_parquet_receipts"][name]
            if (
                int(source_receipt.get("size_bytes", -1)) != len(payload)
                or source_receipt.get("sha256") != hashlib.sha256(payload).hexdigest()
            ):
                raise ValueError(f"ENTSO v2 archived source receipt mismatch: {name}")
    content_hash = hashlib.sha256(
        json.dumps(
            {name: item["sha256"] for name, item in sorted(declared.items())},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if content_hash != receipt.get("archive_content_sha256"):
        raise ValueError("ENTSO import archive content hash mismatch")
    return manifest


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: object) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _write_bytes(path, raw)


def _validate_portable_id(value: str, *, label: str) -> None:
    if (
        not _PORTABLE_ID.fullmatch(value)
        or value in {".", ".."}
        or value.endswith((".", " "))
        or value.split(".", 1)[0].upper() in _WINDOWS_RESERVED
    ):
        raise ValueError(f"{label} is not a portable path-safe identifier")


def _validate_source_label(value: str) -> None:
    if not value or len(value) > 128 or any(char in value for char in "\\/:"):
        raise ValueError("source_label must be a portable descriptive identifier")


def _validate_relative_source_name(value: object) -> str:
    name = str(value).strip().replace("\\", "/")
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".parquet":
        raise ValueError(f"invalid relative source parquet name: {value}")
    return path.as_posix()


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction and is_junction())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    default_entsoe_root = os.environ.get(PFC_ENTSOE_DATA_ROOT_ENV)
    default_data_root = None
    if FMV_DATA_ROOT_ENV in os.environ or PFC_SHARED_DATA_ROOT_ENV in os.environ:
        default_data_root = resolve_fmv_data_root(
            compatibility_envs=(PFC_SHARED_DATA_ROOT_ENV,),
        )
    roots = parser.add_mutually_exclusive_group(
        required=default_entsoe_root is None and default_data_root is None
    )
    roots.add_argument(
        "--entsoe-root",
        type=Path,
        default=default_entsoe_root,
        help=f"Direct ENTSO-E root containing imports (or {PFC_ENTSOE_DATA_ROOT_ENV})",
    )
    roots.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help=(
            "Shared parent whose entsoe child contains imports "
            f"(or {FMV_DATA_ROOT_ENV}; {PFC_SHARED_DATA_ROOT_ENV} is deprecated)"
        ),
    )
    parser.add_argument("--import-id", required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument(
        "--exclude-parquet",
        action="append",
        default=[],
        help="Source-relative non-ENTSO parquet to exclude explicitly (repeatable)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    path = archive_entso_dataset(
        source_dir=args.source_dir,
        data_root=args.data_root,
        entsoe_root=args.entsoe_root,
        import_id=args.import_id,
        source_label=args.source_label,
        excluded_files=args.exclude_parquet,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Publish one immutable, hash-bound LT input snapshot and atomic pointer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.data.shared_data_root import (  # noqa: E402
    configured_lt_data_root,
    resolve_confined_path,
    resolve_fmv_data_root,
)

SOURCE_LAYOUT = {
    "epex_ch": "market/epex/epex_15min.parquet",
    "epex_de": "market/epex/epex_de_15min.parquet",
    "entso": "entsoe/curated/entso_15min.parquet",
    "hydro": "hydro/curated/hydro_reservoir.parquet",
    "outages": "entsoe/curated/outages_15min.parquet",
    "commodities": "market/commodities/commodities_cache.parquet",
    "eex_forwards_history": "market/eex/eex_forwards_history.parquet",
    "epex_at": "market/epex/epex_at_15min.parquet",
    "epex_fr": "market/epex/epex_fr_15min.parquet",
    "epex_it": "market/epex/epex_it_15min.parquet",
}
REQUIRED_ROLES = {"epex_ch", "epex_de", "entso", "hydro"}
EXPECTED_CADENCE_SECONDS = {
    "epex_ch": 900,
    "epex_de": 900,
    "entso": 900,
    "hydro": 604800,
    "outages": 900,
    "epex_at": 900,
    "epex_fr": 900,
    "epex_it": 900,
}
SOURCE_SYSTEMS = {
    "epex_ch": "EPEX_SPOT",
    "epex_de": "EPEX_SPOT",
    "epex_at": "EPEX_SPOT",
    "epex_fr": "EPEX_SPOT",
    "epex_it": "EPEX_SPOT",
    "entso": "ENTSOE_TRANSPARENCY_PLATFORM",
    "outages": "ENTSOE_TRANSPARENCY_PLATFORM",
    "hydro": "FMV_HYDRO_CURATED",
    "commodities": "FMV_COMMODITIES_CURATED",
    "eex_forwards_history": "EEX_MARKET_DATA",
}
_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def create_snapshot(
    *,
    source_root: Path,
    data_root: Path,
    generation_id: str,
    available_at: str,
    provenance: str,
) -> Path:
    source_root = source_root.resolve()
    data_root = resolve_fmv_data_root(data_root, require_exists=True)
    _validate_portable_id(generation_id, label="generation_id")
    view_root = data_root / "views" / "pfc_lt"
    resolve_confined_path(data_root, view_root, label="pfc_lt view directory")
    snapshots = resolve_confined_path(
        data_root,
        "snapshots",
        label="shared snapshots directory",
    )
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshots = resolve_confined_path(
        data_root,
        snapshots,
        label="shared snapshots directory",
        require_exists=True,
    )
    final = snapshots / generation_id
    staging = snapshots / f".{generation_id}.staging-{os.getpid()}"
    if final.exists() or staging.exists():
        raise FileExistsError(f"LT input generation already exists: {final}")
    staging.mkdir()
    resolve_confined_path(
        data_root,
        staging,
        label="LT snapshot staging directory",
        require_exists=True,
    )
    files: dict[str, object] = {}
    try:
        for role, relative in SOURCE_LAYOUT.items():
            source = source_root / Path(relative)
            if not source.is_file():
                if role in REQUIRED_ROLES:
                    raise FileNotFoundError(f"required LT input source is missing: {source}")
                continue
            destination = staging / Path(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = source.read_bytes()
            _write_bytes(destination, payload)
            files[role] = {
                "path": Path(relative).as_posix(),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "acquisition_id": generation_id,
                "available_at_utc": available_at,
                "calibration_eligible": False,
                "expected_cadence_seconds": EXPECTED_CADENCE_SECONDS.get(role),
                "source_system": SOURCE_SYSTEMS[role],
            }
        contract = {
            "schema_version": "lt_input_snapshot.v1",
            "layout": "external_v2",
            "generation_id": generation_id,
            "acquisition_id": generation_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "available_at_utc": available_at,
            "calibration_eligible": False,
            "source_class": "MIGRATED_UNVERIFIED",
            "provenance": provenance,
            "files": files,
        }
        contract_path = staging / "lt_input_snapshot.json"
        _write_json(contract_path, contract)
        contract_hash = _sha256(contract_path)
        _replace_with_retry(staging, final)
        pointer = {
            "schema_version": "lt_data_pointer.v1",
            "generation_id": generation_id,
            "contract_path": (Path("snapshots") / generation_id / "lt_input_snapshot.json").as_posix(),
            "contract_sha256": contract_hash,
        }
        resolve_confined_path(data_root, view_root, label="pfc_lt view directory")
        view_root.mkdir(parents=True, exist_ok=True)
        resolve_confined_path(
            data_root,
            view_root,
            label="pfc_lt view directory",
            require_exists=True,
        )
        _write_json_atomic(view_root / "current.json", pointer)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return final


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: object) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _write_bytes(path, raw)


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    if temporary.exists():
        temporary.unlink()
    _write_json(temporary, payload)
    _replace_with_retry(temporary, path)


def _replace_with_retry(source: Path, destination: Path) -> None:
    for attempt in range(5):
        try:
            os.replace(source, destination)
            _fsync_directory(destination.parent)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_portable_id(value: str, *, label: str) -> None:
    if (
        not _PORTABLE_ID.fullmatch(value)
        or value in {".", ".."}
        or value.endswith((".", " "))
        or value.split(".", 1)[0].upper() in _WINDOWS_RESERVED
    ):
        raise ValueError(f"{label} is not a portable path-safe identifier")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = configured_lt_data_root()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_root,
        required=default_root is None,
        help="Shared root (or FMV_DATA_ROOT; legacy aliases are deprecated)",
    )
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--available-at", required=True)
    parser.add_argument("--provenance", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    path = create_snapshot(
        source_root=args.source_root,
        data_root=args.data_root,
        generation_id=args.generation_id,
        available_at=args.available_at,
        provenance=args.provenance,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

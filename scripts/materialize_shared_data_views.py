#!/usr/bin/env python3
"""Materialize consumer views in the shared FMV data root without moving data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.data.shared_data_root import (  # noqa: E402
    configured_lt_data_root,
    is_link_or_junction,
    resolve_confined_path,
    resolve_fmv_data_root,
)
from pfc_shaping.data.lt_input_sources import (  # noqa: E402
    validate_lt_input_contract_semantics,
)


def materialize_lt_view(data_root: str | Path) -> Path:
    """Copy the legacy LT pointer into its consumer view, refusing divergence."""

    root = resolve_fmv_data_root(data_root, require_exists=True)
    legacy = root / "current.json"
    if not legacy.is_file():
        raise FileNotFoundError(f"legacy LT pointer is missing: {legacy}")
    legacy = resolve_confined_path(
        root,
        legacy,
        label="legacy LT pointer",
        require_exists=True,
        require_file=True,
    )
    payload = legacy.read_bytes()
    pointer = json.loads(payload.decode("utf-8"))
    if pointer.get("schema_version") != "lt_data_pointer.v1":
        raise ValueError("legacy LT pointer schema_version must be lt_data_pointer.v1")
    contract_path = _resolve_pointer_contract(root, pointer)
    contract_payload = contract_path.read_bytes()
    if hashlib.sha256(contract_payload).hexdigest() != pointer.get("contract_sha256"):
        raise ValueError("legacy LT pointer contract_sha256 mismatch")
    contract = json.loads(contract_payload.decode("utf-8"))
    validate_lt_input_contract_semantics(contract)
    if str(contract.get("generation_id", "")) != str(pointer.get("generation_id", "")):
        raise ValueError("LT pointer/contract generation_id mismatch")

    destination = root / "views" / "pfc_lt" / "current.json"
    _verify_view_path(root, destination)
    if destination.exists():
        if (
            not destination.is_file()
            or is_link_or_junction(destination)
            or destination.read_bytes() != payload
        ):
            raise ValueError("existing pfc_lt view diverges from legacy LT pointer")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    _verify_view_path(root, destination)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    if destination.read_bytes() != payload:
        raise ValueError("materialized pfc_lt view bytes do not match legacy pointer")
    return destination


def _verify_view_path(root: Path, destination: Path) -> None:
    resolve_confined_path(root, destination, label="pfc_lt view")


def _resolve_pointer_contract(root: Path, pointer: dict[str, object]) -> Path:
    logical = str(pointer.get("contract_path", "")).strip()
    generation_id = str(pointer.get("generation_id", "")).strip()
    canonical = (
        Path("snapshots") / generation_id / "lt_input_snapshot.json"
    ).as_posix()
    if logical != canonical:
        raise ValueError("legacy LT pointer contract_path is not canonical")
    relative = Path(logical)
    if not logical or relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError("legacy LT pointer contract_path must be relative and confined")
    return resolve_confined_path(
        root,
        relative,
        label="legacy LT pointer contract",
        require_exists=True,
        require_file=True,
    )


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = configured_lt_data_root()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_root,
        required=default_root is None,
        help="Shared root (or FMV_DATA_ROOT; PFC_LT_DATA_ROOT is deprecated)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    print(materialize_lt_view(args.data_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

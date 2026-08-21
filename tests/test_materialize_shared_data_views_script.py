from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import pytest

from scripts.materialize_shared_data_views import materialize_lt_view


def _legacy_pointer(root: Path) -> bytes:
    root.mkdir()
    contract_path = root / "snapshots" / "generation-001" / "lt_input_snapshot.json"
    contract_path.parent.mkdir(parents=True)
    contract_payload = (
        json.dumps(
            {
                "schema_version": "lt_input_snapshot.v1",
                "layout": "external_v2",
                "generation_id": "generation-001",
                "acquisition_id": "generation-001",
                "available_at_utc": "2026-07-13T10:00:00Z",
                "calibration_eligible": False,
                "source_class": "MIGRATED_UNVERIFIED",
                "files": {
                    role: {
                        "path": f"{role}.parquet",
                        "acquisition_id": "generation-001",
                        "available_at_utc": "2026-07-13T10:00:00Z",
                        "calibration_eligible": False,
                    }
                    for role in ("epex_ch", "epex_de", "entso", "hydro")
                },
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    contract_path.write_bytes(contract_payload)
    payload = (
        json.dumps(
            {
                "schema_version": "lt_data_pointer.v1",
                "generation_id": "generation-001",
                "contract_path": "snapshots/generation-001/lt_input_snapshot.json",
                "contract_sha256": hashlib.sha256(contract_payload).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    (root / "current.json").write_bytes(payload)
    return payload


def test_materialize_lt_view_preserves_pointer_bytes(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    payload = _legacy_pointer(root)

    destination = materialize_lt_view(root)

    assert destination == root / "views" / "pfc_lt" / "current.json"
    assert destination.read_bytes() == payload
    assert (root / "current.json").read_bytes() == payload


def test_materialize_lt_view_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    _legacy_pointer(root)

    first = materialize_lt_view(root)
    second = materialize_lt_view(root)

    assert second == first


def test_materialize_lt_view_refuses_existing_divergent_pointer(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    _legacy_pointer(root)
    destination = root / "views" / "pfc_lt" / "current.json"
    destination.parent.mkdir(parents=True)
    destination.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="view diverges"):
        materialize_lt_view(root)


def test_materialize_lt_view_rejects_non_lt_pointer(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    root.mkdir()
    (root / "current.json").write_text(
        json.dumps({"schema_version": "other.v1"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version"):
        materialize_lt_view(root)


def test_materialize_lt_view_rejects_linked_views_directory(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    _legacy_pointer(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, root / "views", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="link or junction"):
        materialize_lt_view(root)

    assert list(outside.iterdir()) == []


def test_materialize_lt_view_requires_canonical_contract_path(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    _legacy_pointer(root)
    pointer_path = root / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    canonical = root / pointer["contract_path"]
    alternate = canonical.with_name("alternate.json")
    alternate.write_bytes(canonical.read_bytes())
    pointer["contract_path"] = "snapshots/generation-001/alternate.json"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(ValueError, match="contract_path is not canonical"):
        materialize_lt_view(root)

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from pfc_shaping.data import eex_forward_local_capture as capture_module
from pfc_shaping.data.eex_forward_local_capture import (
    STATUS,
    EexForwardLocalCaptureError,
    capture_eex_forward_workbook,
)


def _write_workbook(
    path: Path,
    *,
    product_code: str = "Y01_2027_BASE",
    price: float = 80.0,
) -> None:
    frame = pd.DataFrame(
        [
            [None, product_code, "Y01_2027_PEAK"],
            [None, "ISIN_BASE", "ISIN_PEAK"],
            ["Date", None, None],
            ["29.07.2026", price, 92.0],
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="CH", index=False, header=False)


def _fixed_clock() -> datetime:
    return datetime(2026, 7, 29, 20, 0, tzinfo=timezone.utc)


def test_local_capture_preserves_exact_bytes_but_grants_no_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    source = external / "Price_Report_EEX.xlsx"
    _write_workbook(source)
    monkeypatch.setattr(capture_module, "_EXPECTED_SOURCE_PATH", source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    manifest_path = capture_eex_forward_workbook(
        source_document=source,
        expected_source_sha256=digest,
        capture_id="capture-001",
        repo_root=root,
        clock=_fixed_clock,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    captured = manifest_path.parent / "source.xlsx"
    assert manifest["status"] == STATUS
    assert captured.read_bytes() == source.read_bytes()
    assert manifest["source_document_sha256"] == digest
    assert manifest["market_inventory"]["CH"]["latest_snapshot_date"] == "2026-07-29"
    assert manifest["market_inventory"]["CH"]["quote_count"] == 2
    assert manifest["trusted_time_receipt_present"] is False
    assert manifest["scientific_admission"] is False
    assert manifest["monthly_solver_hard_level_eligible"] is False
    assert manifest["training_eligible"] is False
    assert manifest["model_selection_eligible"] is False
    assert manifest["holdout_eligible"] is False
    assert manifest["candidate_assembly_eligible"] is False
    assert manifest["production_authorization"] is False
    assert manifest["promotion_gate"] is False


def test_hash_mismatch_leaves_attempt_only_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    source = external / "Price_Report_EEX.xlsx"
    _write_workbook(source)
    monkeypatch.setattr(capture_module, "_EXPECTED_SOURCE_PATH", source)

    with pytest.raises(EexForwardLocalCaptureError, match="SHA-256 mismatch"):
        capture_eex_forward_workbook(
            source_document=source,
            expected_source_sha256="0" * 64,
            capture_id="capture-002",
            repo_root=root,
            clock=_fixed_clock,
        )

    output = root / "build" / "eex-forward-local-captures" / "capture-002"
    assert sorted(path.name for path in output.iterdir()) == ["capture-attempt.json"]


def test_unknown_quoted_product_is_not_captured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    source = external / "Price_Report_EEX.xlsx"
    _write_workbook(source, product_code="SEASON_2027_BASE")
    monkeypatch.setattr(capture_module, "_EXPECTED_SOURCE_PATH", source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(EexForwardLocalCaptureError, match="unknown quoted product"):
        capture_eex_forward_workbook(
            source_document=source,
            expected_source_sha256=digest,
            capture_id="capture-003",
            repo_root=root,
            clock=_fixed_clock,
        )

    output = root / "build" / "eex-forward-local-captures" / "capture-003"
    assert sorted(path.name for path in output.iterdir()) == ["capture-attempt.json"]


def test_workspace_source_is_rejected_before_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    source = root / "Price_Report_EEX.xlsx"
    _write_workbook(source)
    monkeypatch.setattr(capture_module, "_EXPECTED_SOURCE_PATH", source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(EexForwardLocalCaptureError, match="external to the workspace"):
        capture_eex_forward_workbook(
            source_document=source,
            expected_source_sha256=digest,
            capture_id="capture-004",
            repo_root=root,
            clock=_fixed_clock,
        )

    assert not (root / "build" / "eex-forward-local-captures").exists()


def test_capture_id_is_strictly_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    source = external / "Price_Report_EEX.xlsx"
    _write_workbook(source)
    monkeypatch.setattr(capture_module, "_EXPECTED_SOURCE_PATH", source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    arguments = {
        "source_document": source,
        "expected_source_sha256": digest,
        "capture_id": "capture-005",
        "repo_root": root,
        "clock": _fixed_clock,
    }

    capture_eex_forward_workbook(**arguments)
    with pytest.raises(FileExistsError):
        capture_eex_forward_workbook(**arguments)


def test_noncanonical_external_source_is_rejected_before_capture(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "build").mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    source = external / "Price_Report_EEX.xlsx"
    _write_workbook(source)

    with pytest.raises(EexForwardLocalCaptureError, match="canonical read-only FMV"):
        capture_eex_forward_workbook(
            source_document=source,
            expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            capture_id="capture-006",
            repo_root=root,
            clock=_fixed_clock,
        )

    assert not (root / "build" / "eex-forward-local-captures").exists()

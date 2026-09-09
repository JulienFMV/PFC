from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.audit_ch_lt_current_evidence_selection as current_selection_module
from scripts.audit_ch_lt_current_evidence_selection import (
    AUDIT_STATUS,
    REGISTRY_ID,
    REGISTRY_RELATIVE_PATH,
    REGISTRY_SHA256,
    CurrentEvidenceSelectionError,
    audit_current_selection,
    compute_registry_id,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT.joinpath(*REGISTRY_RELATIVE_PATH.split("/"))


def test_current_registry_binds_v10_supersedes_v9_and_keeps_embedded_v5_historical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT)

    result = audit_current_selection(
        registry_path=REGISTRY,
        expected_registry_sha256=REGISTRY_SHA256,
    )

    assert result["status"] == AUDIT_STATUS
    assert result["registry_id"] == REGISTRY_ID
    assert result["current_prospective_ledger_version"] == "v10"
    assert result["superseded_prospective_ledger_version"] == "v9"
    assert result["market_time_embedded_ledger_version"] == "v5"
    assert result["market_time_embedded_ledger_is_historical_snapshot"] is True
    assert result["independent_rolling_origin_count"] == 0
    assert result["t057_consumed"] is False
    assert result["production_authorization"] is False


def test_registry_id_binds_exact_content() -> None:
    document = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert compute_registry_id(document) == document["registry_id"] == REGISTRY_ID


def test_rejects_caller_hash_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)

    with pytest.raises(CurrentEvidenceSelectionError, match="frozen identity"):
        audit_current_selection(
            registry_path=REGISTRY,
            expected_registry_sha256="0" * 64,
        )


def test_rejects_shadow_registry_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    shadow = tmp_path / REGISTRY.name
    shadow.write_bytes(REGISTRY.read_bytes())

    with pytest.raises(CurrentEvidenceSelectionError, match="not canonical"):
        audit_current_selection(
            registry_path=shadow,
            expected_registry_sha256=REGISTRY_SHA256,
        )


@pytest.mark.parametrize(
    "relative",
    [
        "../outside.json",
        "build/../../outside.json",
        "C:relative.json",
        "C:.planning/evidence.json",
        "build/evidence.json:stream",
        "build/evidence.json::$DATA",
        "build/CON.json",
    ],
)
def test_rejects_repository_traversal_before_read(
    tmp_path: Path, relative: str
) -> None:
    root = tmp_path / "root"
    root.mkdir()

    with pytest.raises(CurrentEvidenceSelectionError, match="path is invalid"):
        current_selection_module._read_linked_json(
            root,
            {"path": relative, "sha256": "0" * 64},
            label="traversal probe",
        )


def test_current_audit_fails_closed_when_recursive_evidence_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT)

    def _missing_evidence(**_: object) -> dict[str, object]:
        raise current_selection_module.ChLtProspectiveCaptureLedgerError(
            "referenced evidence is missing"
        )

    monkeypatch.setattr(
        current_selection_module,
        "build_prospective_capture_ledger",
        _missing_evidence,
    )
    with pytest.raises(CurrentEvidenceSelectionError, match="recursive evidence"):
        audit_current_selection(
            registry_path=REGISTRY,
            expected_registry_sha256=REGISTRY_SHA256,
        )


def test_current_audit_rejects_recursive_reconstruction_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT)
    real_builder = current_selection_module.build_prospective_capture_ledger

    def _changed_evidence(**kwargs: object) -> dict[str, object]:
        rebuilt = real_builder(**kwargs)
        rebuilt["status"] = "POST_LEDGER_EVIDENCE_CHANGED"
        return rebuilt

    monkeypatch.setattr(
        current_selection_module,
        "build_prospective_capture_ledger",
        _changed_evidence,
    )
    with pytest.raises(CurrentEvidenceSelectionError, match="exact reconstruction"):
        audit_current_selection(
            registry_path=REGISTRY,
            expected_registry_sha256=REGISTRY_SHA256,
        )


def test_cli_emits_only_local_negative_authority(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)

    assert (
        main(
            [
                "--registry",
                str(REGISTRY),
                "--expected-registry-sha256",
                REGISTRY_SHA256,
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == AUDIT_STATUS
    assert result["scientific_admission"] is False
    assert result["publication_authorized"] is False
    assert result["promotion_gate"] is False
    assert result["production_authorization"] is False

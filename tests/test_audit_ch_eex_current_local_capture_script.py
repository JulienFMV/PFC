from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_ch_eex_current_local_capture import (
    AUDIT_STATUS,
    REGISTRY_ID,
    REGISTRY_RELATIVE_PATH,
    REGISTRY_SHA256,
    CurrentEexCaptureError,
    _validate_target_output,
    audit_current_eex_capture,
    compute_quote_commitments,
    compute_selection_id,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT.joinpath(*REGISTRY_RELATIVE_PATH.split("/"))
CURRENT_SOURCE = (
    ROOT
    / "build"
    / "eex-forward-local-captures"
    / "eex-ch-20260730-v1"
    / "source.xlsx"
)
LOCAL_EVIDENCE_PRESENT = REGISTRY.is_file() and CURRENT_SOURCE.is_file()


def _workbook_bytes(path: Path, *, base_price: float) -> bytes:
    frame = pd.DataFrame(
        [
            [None, "Y01_2027_BASE", "Y01_2027_PEAK"],
            [None, "ISIN_BASE", "ISIN_PEAK"],
            ["Date", None, None],
            ["29.07.2026", base_price, 92.0],
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="CH", index=False, header=False)
    return path.read_bytes()


def test_quote_commitment_is_deterministic_and_price_sensitive(tmp_path: Path) -> None:
    first = compute_quote_commitments(
        _workbook_bytes(tmp_path / "first.xlsx", base_price=80.0)
    )
    replay = compute_quote_commitments(
        (tmp_path / "first.xlsx").read_bytes()
    )
    changed = compute_quote_commitments(
        _workbook_bytes(tmp_path / "changed.xlsx", base_price=80.5)
    )

    assert first == replay
    assert first["snapshot_date"] == "2026-07-29"
    assert first["quote_count"] == 2
    assert first["quote_values_disclosed"] is False
    assert (
        first["quote_identity_commitment_sha256"]
        == changed["quote_identity_commitment_sha256"]
    )
    assert (
        first["quote_value_commitment_sha256"]
        != changed["quote_value_commitment_sha256"]
    )
    assert set(first) == {
        "snapshot_date",
        "quote_count",
        "quote_identity_commitment_sha256",
        "quote_value_commitment_sha256",
        "quote_commitment_algorithm",
        "quote_values_disclosed",
    }


def test_selection_id_binds_exact_registry_content() -> None:
    document = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert compute_selection_id(document) == document["selection_id"] == REGISTRY_ID


@pytest.mark.skipif(
    not LOCAL_EVIDENCE_PRESENT,
    reason="repo-local EEX capture evidence is not present in this checkout",
)
def test_current_local_capture_selection_passes_without_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT)

    result = audit_current_eex_capture(
        registry_path=REGISTRY,
        expected_registry_sha256=REGISTRY_SHA256,
    )

    assert result["status"] == AUDIT_STATUS
    assert result["selection_id"] == REGISTRY_ID
    assert result["current_capture_id"] == "eex-ch-20260730-v1"
    assert result["superseded_capture_id"] == "eex-ch-20260729-v2"
    assert result["current_snapshot_date"] == "2026-07-29"
    assert result["quote_count"] == 40
    assert result["quote_values_disclosed"] is False
    assert result["scientific_admission"] is False
    assert result["monthly_solver_hard_level_eligible"] is False
    assert result["t057_consumed"] is False
    assert result["production_authorization"] is False


def test_rejects_caller_hash_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)

    with pytest.raises(CurrentEexCaptureError, match="frozen identity"):
        audit_current_eex_capture(
            registry_path=REGISTRY,
            expected_registry_sha256="0" * 64,
        )


def test_rejects_shadow_registry_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT)
    shadow = tmp_path / REGISTRY.name
    shadow.write_bytes(REGISTRY.read_bytes())

    with pytest.raises(CurrentEexCaptureError, match="not canonical"):
        audit_current_eex_capture(
            registry_path=shadow,
            expected_registry_sha256=REGISTRY_SHA256,
        )


def _target_output(run_root: Path) -> dict[str, object]:
    stdout = b"ok\n"
    stderr = b""
    (run_root / "target-stdout.log").write_bytes(stdout)
    (run_root / "target-stderr.log").write_bytes(stderr)

    def stream(name: str, payload: bytes) -> dict[str, object]:
        import hashlib

        digest = hashlib.sha256(payload).hexdigest()
        return {
            "captured_bytes": len(payload),
            "full_stream_sha256": digest,
            "path": str(run_root / f"target-{name}.log"),
            "sha256": digest,
            "total_bytes": len(payload),
            "truncated": False,
        }

    return {
        "capture_limit_bytes_per_stream": 1024,
        "complete": True,
        "streams": {
            "stdout": stream("stdout", stdout),
            "stderr": stream("stderr", stderr),
        },
    }


def test_runner_output_validation_rejects_missing_log(tmp_path: Path) -> None:
    output = _target_output(tmp_path)
    (tmp_path / "target-stdout.log").unlink()

    with pytest.raises(ValueError, match="unavailable"):
        _validate_target_output(output, run_root=tmp_path, label="capture receipt")


def test_runner_output_validation_rejects_tampered_log(tmp_path: Path) -> None:
    output = _target_output(tmp_path)
    (tmp_path / "target-stdout.log").write_bytes(b"changed\n")

    with pytest.raises(CurrentEexCaptureError, match="SHA-256 mismatch"):
        _validate_target_output(output, run_root=tmp_path, label="capture receipt")


def test_runner_output_validation_rejects_truncation(tmp_path: Path) -> None:
    output = _target_output(tmp_path)
    output["streams"]["stdout"]["truncated"] = True

    with pytest.raises(CurrentEexCaptureError, match="truncation"):
        _validate_target_output(output, run_root=tmp_path, label="capture receipt")


@pytest.mark.skipif(
    not LOCAL_EVIDENCE_PRESENT,
    reason="repo-local EEX capture evidence is not present in this checkout",
)
def test_cli_emits_commitments_but_no_quote_values(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
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
    assert result["quote_values_disclosed"] is False
    assert "quotes" not in result
    assert "prices" not in result
    assert result["scientific_admission"] is False
    assert result["monthly_solver_hard_level_eligible"] is False
    assert result["promotion_gate"] is False
    assert result["production_authorization"] is False

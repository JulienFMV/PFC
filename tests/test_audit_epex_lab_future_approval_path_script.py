from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_epex_lab_future_approval_path import audit_future_approval_path


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _readiness_payload(*, approved: bool = False, strict: bool = True, production: bool = False) -> dict:
    checks = [
        {"name": "product_all_gates_pass", "status": "PASS", "value": True},
        {"name": "adjusted_production_manifest_approved", "status": "PASS" if production else "FAIL", "value": production},
        {"name": "adjusted_export_manifest_production_ready", "status": "PASS" if production else "FAIL", "value": production},
        {"name": "adjusted_selected_artifact_production_ready", "status": "PASS" if production else "FAIL", "value": production},
        {"name": "adjusted_capstone_approved", "status": "PASS" if production else "FAIL", "value": production},
    ]
    return {
        "schema_version": "epex_lab_promotion_readiness.v1",
        "approved": approved,
        "status": "PROMOTION_READY" if approved else "STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING",
        "strict_diagnostics_pass": strict,
        "production_chain_pass": production,
        "selected_adjusted_csv": "adjusted.csv",
        "missing_production_evidence": [] if production else ["adjusted_production_manifest"],
        "checks": checks,
    }


def _spot_payload(**overrides) -> dict:
    payload = {
        "status": "DIAGNOSTIC_PASS",
        "promotion_gate": False,
        "production_approved": False,
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "benchmark_policy": "rolling_origin_epex_spot_no_ompex_lab_only",
    }
    payload.update(overrides)
    return payload


def test_future_approval_path_reports_no_go_blockers(tmp_path: Path) -> None:
    readiness = _write_json(tmp_path / "readiness.json", _readiness_payload())
    spot = _write_json(tmp_path / "spot.json", _spot_payload())

    summary = audit_future_approval_path(
        readiness_json=readiness,
        spot_backtest_summary=spot,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "NO_GO_PRODUCTION_CHAIN_INCOMPLETE"
    assert summary["approved"] is False
    assert summary["strict_diagnostics_pass"] is True
    assert "adjusted_production_manifest" in summary["remaining_blockers"]
    assert "adjusted_capstone_approved" in summary["remaining_blockers"]
    assert summary["spot_backtest_policy"]["pass"] is True
    assert summary["next_actions"]


def test_future_approval_path_reports_promotion_ready_candidate(tmp_path: Path) -> None:
    readiness = _write_json(
        tmp_path / "readiness.json",
        _readiness_payload(approved=True, production=True),
    )

    summary = audit_future_approval_path(
        readiness_json=readiness,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "PROMOTION_READY_CANDIDATE"
    assert summary["approved"] is True
    assert summary["remaining_blockers"] == []


def test_future_approval_path_blocks_bad_spot_policy(tmp_path: Path) -> None:
    readiness = _write_json(tmp_path / "readiness.json", _readiness_payload())
    spot = _write_json(tmp_path / "spot.json", _spot_payload(ompex_used_in_backtest=True))

    summary = audit_future_approval_path(
        readiness_json=readiness,
        spot_backtest_summary=spot,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "NO_GO_SPOT_BACKTEST_POLICY_FAIL"
    assert summary["spot_backtest_policy"]["pass"] is False
    assert summary["spot_backtest_policy"]["checks"]["ompex_not_backtest"] is False

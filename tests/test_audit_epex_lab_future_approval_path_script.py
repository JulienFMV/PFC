from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_epex_lab_future_approval_path import PRODUCTION_CHECKS, audit_future_approval_path


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _readiness_payload(*, approved: bool = False, strict: bool = True, production: bool = False) -> dict:
    checks = [
        {"name": "product_all_gates_pass", "status": "PASS", "value": True},
    ]
    for name in PRODUCTION_CHECKS:
        value = production
        if name.endswith("_bound"):
            value = {
                "locked_holdout_summary_sha256": "holdout-sha" if production else None,
            }
        checks.append({"name": name, "status": "PASS" if production else "FAIL", "value": value})
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


def _locked_holdout_run_payload(**overrides) -> dict:
    payload = {
        "schema_version": "epex_lab_locked_holdout_run.v1",
        "status": "LOCKED_HOLDOUT_PASS",
        "promotion_gate": False,
        "production_approved": False,
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "coverage_ready": True,
        "backtest_ran": True,
        "audit_ran": True,
        "holdout_pass": True,
    }
    payload.update(overrides)
    return payload


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def test_future_approval_path_blocks_promotion_ready_without_locked_holdout(tmp_path: Path) -> None:
    readiness = _write_json(
        tmp_path / "readiness.json",
        _readiness_payload(approved=True, production=True),
    )

    summary = audit_future_approval_path(
        readiness_json=readiness,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "MISSING_LOCKED_HOLDOUT"
    assert summary["approved"] is False
    assert "locked_holdout_pass" in summary["remaining_blockers"]


def test_future_approval_path_blocks_when_locked_holdout_coverage_pending(tmp_path: Path) -> None:
    readiness = _write_json(
        tmp_path / "readiness.json",
        _readiness_payload(approved=True, production=True),
    )
    holdout = _write_json(
        tmp_path / "holdout.json",
        _locked_holdout_run_payload(
            status="WAITING_FOR_FULL_SPOT_COVERAGE",
            coverage_ready=False,
            backtest_ran=False,
            audit_ran=False,
            holdout_pass=False,
        ),
    )

    summary = audit_future_approval_path(
        readiness_json=readiness,
        locked_holdout_summary=holdout,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING"
    assert summary["approved"] is False
    assert "locked_holdout_pass" in summary["remaining_blockers"]
    assert summary["locked_holdout_policy"]["pass"] is False
    assert summary["locked_holdout_policy"]["checks"]["coverage_ready"] is False


def test_future_approval_path_allows_promotion_ready_with_passing_locked_holdout(tmp_path: Path) -> None:
    holdout = _write_json(tmp_path / "holdout.json", _locked_holdout_run_payload())
    payload = _readiness_payload(approved=True, production=True)
    for check in payload["checks"]:
        if isinstance(check.get("value"), dict) and "locked_holdout_summary_sha256" in check["value"]:
            check["value"]["locked_holdout_summary_sha256"] = _sha256(holdout)
    readiness = _write_json(tmp_path / "readiness.json", payload)

    summary = audit_future_approval_path(
        readiness_json=readiness,
        locked_holdout_summary=holdout,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "PROMOTION_READY_CANDIDATE"
    assert summary["approved"] is True
    assert summary["locked_holdout_policy"]["pass"] is True


def test_future_approval_path_blocks_synthetic_ready_payload_missing_production_checks(tmp_path: Path) -> None:
    readiness = _write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": "epex_lab_promotion_readiness.v1",
            "approved": True,
            "status": "PROMOTION_READY",
            "strict_diagnostics_pass": True,
            "production_chain_pass": True,
            "selected_adjusted_csv": "adjusted.csv",
            "missing_production_evidence": [],
            "checks": [],
        },
    )
    holdout = _write_json(tmp_path / "holdout.json", _locked_holdout_run_payload())

    summary = audit_future_approval_path(
        readiness_json=readiness,
        locked_holdout_summary=holdout,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_HASH_MISMATCH"
    assert summary["approved"] is False
    assert "adjusted_production_manifest_locked_holdout_bound" in summary["missing_production_checks"]
    assert "locked_holdout_sha_bound" in summary["remaining_blockers"]


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

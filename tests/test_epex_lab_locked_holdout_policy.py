from __future__ import annotations

import json
from pathlib import Path

from scripts.epex_lab_locked_holdout_policy import build_locked_plan_identity, locked_holdout_policy


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_plan(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "locked_plan.json",
        {
            "schema_version": "epex_lab_locked_holdout_plan.v1",
            "plan_id": "t057_locked_t056_future_holdout",
            "benchmark_policy": "locked_future_no_ompex_holdout",
            "frozen_at_utc": "2026-07-09T00:00:00Z",
            "holdout_start_utc": "2026-07-10T00:00:00Z",
            "holdout_end_utc": "2026-07-24T00:00:00Z",
            "baseline_csv_sha256": "b" * 64,
            "adjusted_csv_sha256": "a" * 64,
            "lab_manifest_sha256": "l" * 64,
            "selection_summary_sha256": "s" * 64,
        },
    )


def _passing_run_summary(tmp_path: Path) -> dict:
    plan = _write_plan(tmp_path)
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    return {
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
        "locked_plan_identity": build_locked_plan_identity(plan_payload, plan_json=plan),
    }


def test_locked_holdout_policy_accepts_plan_bound_run_summary(tmp_path: Path) -> None:
    policy = locked_holdout_policy(_passing_run_summary(tmp_path))

    assert policy["pass"] is True
    assert policy["checks"]["plan_json_file_sha_bound"] is True
    assert policy["checks"]["plan_identity_matches_plan_json"] is True


def test_locked_holdout_policy_rejects_passing_summary_without_plan_identity(tmp_path: Path) -> None:
    summary = _passing_run_summary(tmp_path)
    summary.pop("locked_plan_identity")

    policy = locked_holdout_policy(summary)

    assert policy["pass"] is False
    assert policy["checks"]["locked_plan_identity_present"] is False
    assert policy["status"] == "NO_GO_LOCKED_HOLDOUT_FAIL"


def test_locked_holdout_policy_rejects_tampered_plan_after_summary(tmp_path: Path) -> None:
    summary = _passing_run_summary(tmp_path)
    plan = Path(summary["locked_plan_identity"]["plan_json"])
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["plan_id"] = "different_plan"
    plan.write_text(json.dumps(payload), encoding="utf-8")

    policy = locked_holdout_policy(summary)

    assert policy["pass"] is False
    assert policy["checks"]["plan_json_file_sha_bound"] is False

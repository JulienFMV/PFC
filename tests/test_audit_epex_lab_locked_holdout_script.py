from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.audit_epex_lab_locked_holdout import audit_holdout


def test_audit_epex_lab_locked_holdout_passes_no_ompex_window(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    summary = _write_summary(tmp_path, baseline_sha="base", adjusted_sha="adjusted")
    post = tmp_path / "post.csv"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-07-10", periods=4, freq="h", tz="UTC").astype(str),
            "baseline_abs_error_eur_mwh": [4.0, 4.0, 4.0, 4.0],
            "adjusted_abs_error_eur_mwh": [3.0, 3.5, 3.0, 3.5],
        }
    ).to_csv(post, index=False)

    audit = audit_holdout(plan_json=plan, spot_backtest_summary=summary, output=tmp_path / "audit.json")

    assert audit["status"] == "LOCKED_HOLDOUT_PASS"
    assert audit["holdout_pass"] is True
    assert audit["approved"] is False
    assert audit["checks"]["summary_no_ompex"] is True
    assert audit["holdout_metrics"]["hours"] == 4
    assert audit["holdout_metrics"]["residual_mae_improvement_eur_mwh"] > 0
    assert (tmp_path / "audit.json").exists()


def test_audit_epex_lab_locked_holdout_fails_degraded_window(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    summary = _write_summary(tmp_path, baseline_sha="base", adjusted_sha="adjusted")
    post = tmp_path / "post.csv"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-07-10", periods=4, freq="h", tz="UTC").astype(str),
            "baseline_abs_error_eur_mwh": [3.0, 3.0, 3.0, 3.0],
            "adjusted_abs_error_eur_mwh": [4.0, 4.0, 4.0, 4.0],
        }
    ).to_csv(post, index=False)

    audit = audit_holdout(plan_json=plan, spot_backtest_summary=summary)

    assert audit["status"] == "NO_GO_LOCKED_HOLDOUT_FAIL"
    assert audit["holdout_pass"] is False
    assert audit["checks"]["holdout_non_degraded"] is False


def _write_plan(tmp_path: Path) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "epex_lab_locked_holdout_plan.v1",
                "benchmark_policy": "locked_future_no_ompex_holdout",
                "ompex_used_in_model": False,
                "ompex_used_in_selection": False,
                "ompex_used_in_backtest": False,
                "holdout_start_utc": "2026-07-10T00:00:00Z",
                "holdout_end_utc": "2026-07-11T00:00:00Z",
                "backtest": {"valuation_timestamp_utc": "2026-07-09T00:00:00Z"},
                "pass_criteria": {
                    "baseline_csv_sha256": "base",
                    "adjusted_csv_sha256": "adjusted",
                    "strict_lab_gate_pass": True,
                    "min_holdout_hours": 4,
                    "min_residual_mae_improvement_eur_mwh": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_summary(tmp_path: Path, *, baseline_sha: str, adjusted_sha: str) -> Path:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_policy": "rolling_origin_epex_spot_no_ompex_lab_only",
                "promotion_gate": False,
                "production_approved": False,
                "ompex_used_in_model": False,
                "ompex_used_in_selection": False,
                "ompex_used_in_backtest": False,
                "strict_lab_gate_pass": True,
                "valuation_timestamp_utc": "2026-07-09T00:00:00+00:00",
                "source_hashes": {
                    "baseline_csv": baseline_sha,
                    "adjusted_csv": adjusted_sha,
                },
                "outputs": {
                    "post_valuation_timestamp_residuals_csv": str(tmp_path / "post.csv"),
                },
            }
        ),
        encoding="utf-8",
    )
    return path

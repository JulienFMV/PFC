from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.check_epex_lab_locked_holdout_coverage import check_coverage


def test_check_epex_lab_locked_holdout_coverage_ready(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, min_hours=4)
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="h"),
    ).to_parquet(spot)

    summary = check_coverage(plan_json=plan, spot_parquet=spot, output=tmp_path / "coverage.json")

    assert summary["status"] == "READY_TO_RUN_HOLDOUT_BACKTEST"
    assert summary["ready_to_run_backtest"] is True
    assert summary["expected_holdout_hours"] == 4
    assert summary["observed_holdout_hours"] == 4
    assert summary["missing_holdout_hours"] == 0
    assert summary["checks"]["full_window_covered"] is True
    assert (tmp_path / "coverage.json").exists()


def test_check_epex_lab_locked_holdout_coverage_waits_for_missing_hours(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, min_hours=4)
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(
            [
                "2026-07-10T00:00:00Z",
                "2026-07-10T01:00:00Z",
                "2026-07-10T03:00:00Z",
            ]
        ),
    ).to_parquet(spot)

    summary = check_coverage(plan_json=plan, spot_parquet=spot)

    assert summary["status"] == "WAITING_FOR_FULL_SPOT_COVERAGE"
    assert summary["ready_to_run_backtest"] is False
    assert summary["observed_holdout_hours"] == 3
    assert summary["missing_holdout_hours"] == 1
    assert summary["first_missing_holdout_utc"] == "2026-07-10T02:00:00Z"
    assert summary["checks"]["full_window_covered"] is False


def _write_plan(tmp_path: Path, *, min_hours: int) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "epex_lab_locked_holdout_plan.v1",
                "plan_id": "test_holdout",
                "ompex_used_in_model": False,
                "ompex_used_in_selection": False,
                "ompex_used_in_backtest": False,
                "holdout_start_utc": "2026-07-10T00:00:00Z",
                "holdout_end_utc": "2026-07-10T04:00:00Z",
                "pass_criteria": {"min_holdout_hours": min_hours},
            }
        ),
        encoding="utf-8",
    )
    return path

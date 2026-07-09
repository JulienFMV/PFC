from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.discover_epex_spot_parquet_candidates import discover_candidates, main


def test_discover_epex_spot_parquet_candidates_ranks_latest_spot(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    root = tmp_path / "output" / "phase14"
    older = root / "older" / "spot.parquet"
    newer = root / "newer" / "spot.parquet"
    invalid = root / "invalid" / "not_spot.parquet"
    _write_spot(older, "2026-07-10T00:00:00Z", periods=2)
    _write_spot(newer, "2026-07-10T00:00:00Z", periods=4)
    invalid.parent.mkdir(parents=True)
    pd.DataFrame({"other": [1.0]}, index=pd.date_range("2026-07-10T00:00:00Z", periods=1, freq="h")).to_parquet(
        invalid
    )

    summary = discover_candidates(
        plan_json=plan,
        search_roots=[root],
        output_json=tmp_path / "discovery.json",
    )

    assert summary["schema_version"] == "epex_spot_parquet_discovery.v1"
    assert summary["candidate_count"] == 2
    assert summary["best_candidate"]["path"] == str(newer.resolve())
    assert summary["best_candidate"]["spot_max_utc"] == "2026-07-10T03:00:00Z"
    assert summary["best_candidate"]["spot_hours_until_latest_required_holdout"] == 0.0
    assert summary["best_candidate"]["expected_holdout_hours"] == 4
    assert summary["best_candidate"]["observed_holdout_hours"] == 4
    assert summary["best_candidate"]["missing_holdout_hours"] == 0
    assert summary["best_candidate"]["first_missing_holdout_utc"] is None
    assert summary["best_candidate"]["last_missing_holdout_utc"] is None
    assert summary["best_candidate"]["full_window_covered"] is True
    assert summary["plan_json_sha256"] == _sha256(plan)
    assert "check_epex_lab_locked_holdout_coverage.py" in summary["recommended_commands"]["coverage_check"]
    assert "run_epex_lab_locked_holdout.py" in summary["recommended_commands"]["run_locked_holdout"]
    assert (tmp_path / "discovery.json").exists()


def test_discover_epex_spot_parquet_candidates_prefers_hourly_grid_by_default(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    root = tmp_path / "output"
    hourly = root / "hourly.parquet"
    intraday = root / "intraday.parquet"
    _write_spot(hourly, "2026-07-10T00:00:00Z", periods=3)
    intraday.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="15min"),
    ).to_parquet(intraday)

    summary = discover_candidates(
        plan_json=plan,
        search_roots=[root],
        output_json=tmp_path / "discovery.json",
    )

    assert summary["require_hourly_grid"] is True
    assert summary["candidate_count"] == 1
    assert summary["best_candidate"]["path"] == str(hourly.resolve())
    assert summary["best_candidate"]["hourly_grid_compatible"] is True

    inclusive = discover_candidates(
        plan_json=plan,
        search_roots=[root],
        output_json=tmp_path / "discovery_inclusive.json",
        require_hourly_grid=False,
    )

    assert inclusive["require_hourly_grid"] is False
    assert inclusive["candidate_count"] == 2


def test_discover_epex_spot_parquet_candidates_reports_lag(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    root = tmp_path / "output"
    spot = root / "spot.parquet"
    _write_spot(spot, "2026-07-09T21:00:00Z", periods=2)

    summary = discover_candidates(
        plan_json=plan,
        search_roots=[root],
        output_json=tmp_path / "discovery.json",
    )

    assert summary["best_candidate"]["spot_max_utc"] == "2026-07-09T22:00:00Z"
    assert summary["latest_required_holdout_utc"] == "2026-07-10T03:00:00Z"
    assert summary["best_candidate"]["spot_hours_until_latest_required_holdout"] == 5.0
    assert summary["best_candidate"]["expected_holdout_hours"] == 4
    assert summary["best_candidate"]["observed_holdout_hours"] == 0
    assert summary["best_candidate"]["missing_holdout_hours"] == 4
    assert summary["best_candidate"]["first_missing_holdout_utc"] == "2026-07-10T00:00:00Z"
    assert summary["best_candidate"]["last_missing_holdout_utc"] == "2026-07-10T03:00:00Z"
    assert summary["best_candidate"]["full_window_covered"] is False


def test_discover_epex_spot_parquet_candidates_cli_returns_nonzero_without_candidates(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    root = tmp_path / "empty"
    root.mkdir()

    code = main(
        [
            "--plan-json",
            str(plan),
            "--search-root",
            str(root),
            "--output-json",
            str(tmp_path / "discovery.json"),
        ]
    )

    assert code == 1
    summary = json.loads((tmp_path / "discovery.json").read_text(encoding="utf-8"))
    assert summary["candidate_count"] == 0
    assert summary["recommended_commands"]["coverage_check"] is None


def _write_plan(tmp_path: Path) -> Path:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "plan_id": "test_holdout",
                "holdout_start_utc": "2026-07-10T00:00:00Z",
                "holdout_end_utc": "2026-07-10T04:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return plan


def _write_spot(path: Path, start: str, *, periods: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"price_eur_mwh": [float(i) for i in range(periods)]},
        index=pd.date_range(start, periods=periods, freq="h"),
    ).to_parquet(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()

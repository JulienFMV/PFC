from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd

from scripts.check_epex_lab_locked_holdout_coverage import CANDIDATE_PRICE_COLUMNS, check_coverage, main


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
    assert summary["non_finite_holdout_price_rows"] == 0
    assert summary["checks"]["plan_benchmark_policy_locked"] is True
    assert summary["checks"]["baseline_csv_sha256_bound"] is True
    assert summary["checks"]["adjusted_csv_sha256_bound"] is True
    assert summary["checks"]["spot_price_column_present"] is True
    assert summary["checks"]["holdout_prices_finite"] is True
    assert summary["checks"]["full_window_covered"] is True
    assert summary["locked_plan_identity"]["plan_id"] == "test_holdout"
    assert summary["locked_plan_identity"]["plan_json_sha256"] == _sha256(plan)
    assert "run_epex_lab_locked_holdout.py" in summary["next_action"]
    assert (tmp_path / "coverage.json").exists()


def test_check_epex_lab_locked_holdout_coverage_cli_exits_zero_when_ready(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, min_hours=4)
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="h"),
    ).to_parquet(spot)

    code = main(
        [
            "--plan-json",
            str(plan),
            "--spot-parquet",
            str(spot),
            "--output",
            str(tmp_path / "coverage.json"),
        ]
    )

    assert code == 0


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


def test_check_epex_lab_locked_holdout_coverage_waits_without_price_column(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, min_hours=4)
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"other_price": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="h"),
    ).to_parquet(spot)

    summary = check_coverage(plan_json=plan, spot_parquet=spot)

    assert summary["status"] == "WAITING_FOR_FULL_SPOT_COVERAGE"
    assert summary["ready_to_run_backtest"] is False
    assert summary["missing_holdout_hours"] == 0
    assert summary["checks"]["full_window_covered"] is True
    assert summary["checks"]["spot_price_column_present"] is False
    assert summary["checks"]["holdout_prices_finite"] is False
    assert summary["non_finite_holdout_price_rows"] is None


def test_check_epex_lab_locked_holdout_coverage_waits_for_non_finite_prices(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, min_hours=4)
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": ["1.0", "2.0", "bad", "inf"]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="h"),
    ).to_parquet(spot)

    summary = check_coverage(plan_json=plan, spot_parquet=spot)

    assert summary["status"] == "WAITING_FOR_FULL_SPOT_COVERAGE"
    assert summary["ready_to_run_backtest"] is False
    assert summary["missing_holdout_hours"] == 0
    assert summary["checks"]["full_window_covered"] is True
    assert summary["checks"]["spot_price_column_present"] is True
    assert summary["checks"]["holdout_prices_finite"] is False
    assert summary["non_finite_holdout_price_rows"] == 2


def test_check_epex_lab_locked_holdout_coverage_waits_for_wrong_benchmark_policy(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, min_hours=4, benchmark_policy="ad_hoc_holdout")
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="h"),
    ).to_parquet(spot)

    summary = check_coverage(plan_json=plan, spot_parquet=spot)

    assert summary["status"] == "WAITING_FOR_FULL_SPOT_COVERAGE"
    assert summary["ready_to_run_backtest"] is False
    assert summary["missing_holdout_hours"] == 0
    assert summary["checks"]["plan_benchmark_policy_locked"] is False
    assert summary["checks"]["full_window_covered"] is True


def test_check_epex_lab_locked_holdout_coverage_blocks_missing_baseline_csv(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, min_hours=4)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    Path(payload["baseline_csv"]).unlink()
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="h"),
    ).to_parquet(spot)

    summary = check_coverage(plan_json=plan, spot_parquet=spot)

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_SOURCE_MISSING_OR_HASH_MISMATCH"
    assert summary["ready_to_run_backtest"] is False
    assert summary["checks"]["full_window_covered"] is True
    assert summary["checks"]["baseline_csv_file_exists"] is False
    assert summary["checks"]["baseline_csv_sha256_bound"] is False


def test_check_epex_lab_locked_holdout_coverage_blocks_adjusted_csv_hash_mismatch(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, min_hours=4)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    Path(payload["adjusted_csv"]).write_text("tampered", encoding="utf-8")
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="h"),
    ).to_parquet(spot)

    summary = check_coverage(plan_json=plan, spot_parquet=spot)

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_SOURCE_MISSING_OR_HASH_MISMATCH"
    assert summary["ready_to_run_backtest"] is False
    assert summary["checks"]["full_window_covered"] is True
    assert summary["checks"]["adjusted_csv_file_exists"] is True
    assert summary["checks"]["adjusted_csv_sha256_bound"] is False


def test_check_epex_lab_locked_holdout_coverage_blocks_missing_candidate_column(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, min_hours=4)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    candidate = pd.read_csv(payload["adjusted_csv"]).drop(columns=["structural_width_eur_mwh"])
    candidate.to_csv(payload["adjusted_csv"], index=False)
    _rewrite_plan_candidate_hash(plan, file_key="adjusted_csv")
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="h"),
    ).to_parquet(spot)

    summary = check_coverage(plan_json=plan, spot_parquet=spot)

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_SOURCE_MISSING_OR_HASH_MISMATCH"
    assert summary["ready_to_run_backtest"] is False
    assert summary["checks"]["adjusted_csv_sha256_bound"] is True
    assert summary["checks"]["adjusted_candidate_required_columns_present"] is False


def test_check_epex_lab_locked_holdout_coverage_blocks_candidate_holdout_gap(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path, min_hours=4)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    candidate = pd.read_csv(payload["baseline_csv"]).iloc[:3]
    candidate.to_csv(payload["baseline_csv"], index=False)
    _rewrite_plan_candidate_hash(plan, file_key="baseline_csv")
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="h"),
    ).to_parquet(spot)

    summary = check_coverage(plan_json=plan, spot_parquet=spot)

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_SOURCE_MISSING_OR_HASH_MISMATCH"
    assert summary["ready_to_run_backtest"] is False
    assert summary["checks"]["baseline_csv_sha256_bound"] is True
    assert summary["checks"]["baseline_candidate_holdout_window_covered"] is False
    assert summary["baseline_candidate_missing_holdout_hours"] == 1


def test_check_epex_lab_locked_holdout_coverage_cli_exits_nonzero_when_waiting(tmp_path: Path) -> None:
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

    code = main(
        [
            "--plan-json",
            str(plan),
            "--spot-parquet",
            str(spot),
            "--output",
            str(tmp_path / "coverage.json"),
        ]
    )

    assert code == 1


def _write_plan(
    tmp_path: Path,
    *,
    min_hours: int,
    benchmark_policy: str = "locked_future_no_ompex_holdout",
) -> Path:
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    _write_candidate_csv(baseline)
    _write_candidate_csv(adjusted)
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "epex_lab_locked_holdout_plan.v1",
                "plan_id": "test_holdout",
                "benchmark_policy": benchmark_policy,
                "ompex_used_in_model": False,
                "ompex_used_in_selection": False,
                "ompex_used_in_backtest": False,
                "holdout_start_utc": "2026-07-10T00:00:00Z",
                "holdout_end_utc": "2026-07-10T04:00:00Z",
                "baseline_csv": str(baseline),
                "baseline_csv_sha256": _sha256(baseline),
                "adjusted_csv": str(adjusted),
                "adjusted_csv_sha256": _sha256(adjusted),
                "pass_criteria": {
                    "min_holdout_hours": min_hours,
                    "baseline_csv_sha256": _sha256(baseline),
                    "adjusted_csv_sha256": _sha256(adjusted),
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_candidate_csv(path: Path, *, periods: int = 4) -> None:
    utc = pd.date_range("2026-07-10T00:00:00Z", periods=periods, freq="h")
    local = utc.tz_convert("Europe/Zurich")
    frame = pd.DataFrame(
        {
            "timestamp_ch": local.strftime("%d.%m.%Y %H:%M"),
            "utc_offset_ch": "UTC" + local.strftime("%z").str.slice(0, 3) + ":" + local.strftime("%z").str.slice(3, 5),
        }
    )
    for offset, column in enumerate(CANDIDATE_PRICE_COLUMNS):
        frame[column] = 50.0 + float(offset)
    frame.to_csv(path, index=False)


def _rewrite_plan_candidate_hash(plan: Path, *, file_key: str) -> None:
    payload = json.loads(plan.read_text(encoding="utf-8"))
    sha_key = f"{file_key}_sha256"
    payload[sha_key] = _sha256(Path(payload[file_key]))
    payload.setdefault("pass_criteria", {})[sha_key] = payload[sha_key]
    plan.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd

from scripts.run_epex_lab_locked_holdout import main, run_locked_holdout


def test_run_epex_lab_locked_holdout_waits_when_coverage_missing(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=2, freq="h"),
    ).to_parquet(spot)

    summary = run_locked_holdout(plan_json=plan, spot_parquet=spot, output_dir=tmp_path / "out")

    assert summary["status"] == "WAITING_FOR_FULL_SPOT_COVERAGE"
    assert summary["coverage_ready"] is False
    assert summary["backtest_ran"] is False
    assert summary["audit_ran"] is False
    assert (tmp_path / "out" / "coverage_status.json").exists()
    assert (tmp_path / "out" / "locked_holdout_run_summary.json").exists()
    assert not (tmp_path / "out" / "spot_backtest_summary.json").exists()


def test_run_epex_lab_locked_holdout_cli_exits_nonzero_when_coverage_missing(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=2, freq="h"),
    ).to_parquet(spot)

    code = main(
        [
            "--plan-json",
            str(plan),
            "--spot-parquet",
            str(spot),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 1


def test_run_epex_lab_locked_holdout_runs_backtest_and_audit_when_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _write_plan(tmp_path)
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="h"),
    ).to_parquet(spot)

    def fake_backtest_against_spot(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        (output_dir / "post.csv").write_text(
            "timestamp_utc,baseline_abs_error_eur_mwh,adjusted_abs_error_eur_mwh\n"
            "2026-07-10T00:00:00Z,4,3\n"
            "2026-07-10T01:00:00Z,4,3\n"
            "2026-07-10T02:00:00Z,4,3\n"
            "2026-07-10T03:00:00Z,4,3\n",
            encoding="utf-8",
        )
        payload = {
            "schema_version": "epex_shape_lab_spot_backtest.v1",
            "read_only": True,
            "benchmark_policy": "rolling_origin_epex_spot_no_ompex_lab_only",
            "promotion_gate": False,
            "production_approved": False,
            "independent_production_evidence": False,
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
            "ompex_used_in_backtest": False,
            "strict_lab_gate_pass": True,
            "valuation_timestamp_utc": "2026-07-09T00:00:00+00:00",
            "status": "DIAGNOSTIC_PASS",
            "source_hashes": {
                "baseline_csv": "base",
                "adjusted_csv": "adjusted",
            },
            "post_valuation_metrics": {"residual_mae_improvement_eur_mwh": 1.0},
            "outputs": {"post_valuation_timestamp_residuals_csv": str(output_dir / "post.csv")},
            "output_hashes": {"post_valuation_timestamp_residuals_csv": _sha256(output_dir / "post.csv")},
        }
        (output_dir / "spot_backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("scripts.run_epex_lab_locked_holdout.backtest_against_spot", fake_backtest_against_spot)

    summary = run_locked_holdout(plan_json=plan, spot_parquet=spot, output_dir=tmp_path / "out")

    assert summary["status"] == "LOCKED_HOLDOUT_PASS"
    assert summary["coverage_ready"] is True
    assert summary["backtest_ran"] is True
    assert summary["audit_ran"] is True
    assert summary["holdout_pass"] is True
    assert (tmp_path / "out" / "locked_holdout_audit.json").exists()


def test_run_epex_lab_locked_holdout_cli_exits_zero_when_holdout_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _write_plan(tmp_path)
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="h"),
    ).to_parquet(spot)

    def fake_backtest_against_spot(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        (output_dir / "post.csv").write_text(
            "timestamp_utc,baseline_abs_error_eur_mwh,adjusted_abs_error_eur_mwh\n"
            "2026-07-10T00:00:00Z,4,3\n"
            "2026-07-10T01:00:00Z,4,3\n"
            "2026-07-10T02:00:00Z,4,3\n"
            "2026-07-10T03:00:00Z,4,3\n",
            encoding="utf-8",
        )
        payload = {
            "schema_version": "epex_shape_lab_spot_backtest.v1",
            "read_only": True,
            "benchmark_policy": "rolling_origin_epex_spot_no_ompex_lab_only",
            "promotion_gate": False,
            "production_approved": False,
            "independent_production_evidence": False,
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
            "ompex_used_in_backtest": False,
            "strict_lab_gate_pass": True,
            "valuation_timestamp_utc": "2026-07-09T00:00:00+00:00",
            "status": "DIAGNOSTIC_PASS",
            "source_hashes": {
                "baseline_csv": "base",
                "adjusted_csv": "adjusted",
            },
            "post_valuation_metrics": {"residual_mae_improvement_eur_mwh": 1.0},
            "outputs": {"post_valuation_timestamp_residuals_csv": str(output_dir / "post.csv")},
            "output_hashes": {"post_valuation_timestamp_residuals_csv": _sha256(output_dir / "post.csv")},
        }
        (output_dir / "spot_backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("scripts.run_epex_lab_locked_holdout.backtest_against_spot", fake_backtest_against_spot)

    code = main(
        [
            "--plan-json",
            str(plan),
            "--spot-parquet",
            str(spot),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 0


def _write_plan(tmp_path: Path) -> Path:
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    baseline.write_text("baseline", encoding="utf-8")
    adjusted.write_text("adjusted", encoding="utf-8")
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "epex_lab_locked_holdout_plan.v1",
                "plan_id": "test_holdout",
                "benchmark_policy": "locked_future_no_ompex_holdout",
                "ompex_used_in_model": False,
                "ompex_used_in_selection": False,
                "ompex_used_in_backtest": False,
                "holdout_start_utc": "2026-07-10T00:00:00Z",
                "holdout_end_utc": "2026-07-10T04:00:00Z",
                "baseline_csv": str(baseline),
                "adjusted_csv": str(adjusted),
                "backtest": {
                    "valuation_timestamp_utc": "2026-07-09T00:00:00Z",
                    "lookback_years": 2,
                    "eval_days": 1,
                    "embargo_days": 1,
                    "max_auto_folds": 12,
                    "min_eval_hours": 24,
                },
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

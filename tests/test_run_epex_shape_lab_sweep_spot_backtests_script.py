from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_epex_shape_lab_sweep_spot_backtests import run_sweep_spot_backtests


def test_run_sweep_spot_backtests_runs_only_eligible_trials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = tmp_path / "candidate.csv"
    spot = tmp_path / "spot.parquet"
    _write_bytes(candidate, b"candidate")
    _write_bytes(spot, b"spot")
    adjusted_true = tmp_path / "sweep" / "trial_true" / "candidate_epex_shape_lab_adjusted.csv"
    adjusted_false = tmp_path / "sweep" / "trial_false" / "candidate_epex_shape_lab_adjusted.csv"
    _write_bytes(adjusted_true, b"adjusted_true")
    _write_bytes(adjusted_false, b"adjusted_false")
    plan_json = _write_plan(tmp_path, candidate, spot, ["trial_true", "trial_false"])
    sweep_summary = _write_sweep_summary(tmp_path, ["trial_true", "trial_false"], [True, False])
    calls: list[Path] = []

    def fake_backtest(**kwargs):
        calls.append(kwargs["adjusted_csv"])
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = _backtest_payload(candidate, kwargs["adjusted_csv"], spot)
        (output_dir / "spot_backtest_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    monkeypatch.setattr("scripts.run_epex_shape_lab_sweep_spot_backtests.backtest_against_spot", fake_backtest)

    summary = run_sweep_spot_backtests(
        plan_json=plan_json,
        sweep_summary=sweep_summary,
        output_root=tmp_path / "spot_backtests",
        output_summary=tmp_path / "run_summary.json",
        resume=False,
    )

    assert calls == [adjusted_true]
    assert summary["trial_count_eligible"] == 1
    assert summary["trial_count_backtested"] == 1
    assert summary["ompex_used_in_backtest"] is False


def test_run_sweep_spot_backtests_reuses_current_backtest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = tmp_path / "candidate.csv"
    spot = tmp_path / "spot.parquet"
    adjusted = tmp_path / "sweep" / "trial_true" / "candidate_epex_shape_lab_adjusted.csv"
    _write_bytes(candidate, b"candidate")
    _write_bytes(spot, b"spot")
    _write_bytes(adjusted, b"adjusted")
    plan_json = _write_plan(tmp_path, candidate, spot, ["trial_true"])
    sweep_summary = _write_sweep_summary(tmp_path, ["trial_true"], [True])
    existing = tmp_path / "spot_backtests" / "trial_true"
    existing.mkdir(parents=True)
    (existing / "spot_backtest_summary.json").write_text(
        json.dumps(_backtest_payload(candidate, adjusted, spot)),
        encoding="utf-8",
    )

    def fail_if_called(**kwargs):
        raise AssertionError("backtest should have been reused")

    monkeypatch.setattr("scripts.run_epex_shape_lab_sweep_spot_backtests.backtest_against_spot", fail_if_called)

    summary = run_sweep_spot_backtests(
        plan_json=plan_json,
        sweep_summary=sweep_summary,
        output_root=tmp_path / "spot_backtests",
        output_summary=tmp_path / "run_summary.json",
    )

    assert summary["reused_existing_count"] == 1


def test_run_sweep_spot_backtests_rejects_ompex_sweep(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.csv"
    spot = tmp_path / "spot.parquet"
    adjusted = tmp_path / "sweep" / "trial_true" / "candidate_epex_shape_lab_adjusted.csv"
    _write_bytes(candidate, b"candidate")
    _write_bytes(spot, b"spot")
    _write_bytes(adjusted, b"adjusted")
    plan_json = _write_plan(tmp_path, candidate, spot, ["trial_true"])
    sweep_summary = _write_sweep_summary(tmp_path, ["trial_true"], [True])
    payload = json.loads(sweep_summary.read_text(encoding="utf-8"))
    payload["ompex_used_in_selection"] = True
    sweep_summary.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must not use OMPEX"):
        run_sweep_spot_backtests(
            plan_json=plan_json,
            sweep_summary=sweep_summary,
            output_root=tmp_path / "spot_backtests",
            output_summary=tmp_path / "run_summary.json",
        )


def _write_plan(tmp_path: Path, candidate: Path, spot: Path, trial_ids: list[str]) -> Path:
    trials = [
        {
            "trial_id": trial_id,
            "output_dir": str(tmp_path / "sweep" / trial_id),
            "parameters": {},
        }
        for trial_id in trial_ids
    ]
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "activation_status": "lab_only",
                "benchmark_policy": "pre_registered_independent_no_ompex",
                "production_approved": False,
                "ompex_used_in_model": False,
                "ompex_used_in_selection": False,
                "candidate_csv": str(candidate),
                "candidate_csv_sha256": _sha256(candidate),
                "spot_parquet": str(spot),
                "spot_parquet_sha256": _sha256(spot),
                "valuation_timestamp": "2026-07-07T00:00:00Z",
                "trials": trials,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_sweep_summary(tmp_path: Path, trial_ids: list[str], eligible: list[bool]) -> Path:
    ranking_csv = tmp_path / "ranking.csv"
    pd.DataFrame({"trial_id": trial_ids, "eligible_for_selection": eligible}).to_csv(ranking_csv, index=False)
    path = tmp_path / "sweep_summary.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_policy": "executed_independent_no_ompex",
                "production_approved": False,
                "ompex_used_in_model": False,
                "ompex_used_in_selection": False,
                "ranking_csv": str(ranking_csv),
            }
        ),
        encoding="utf-8",
    )
    return path


def _backtest_payload(baseline: Path, adjusted: Path, spot: Path) -> dict:
    return {
        "benchmark_policy": "rolling_origin_epex_spot_no_ompex_lab_only",
        "production_approved": False,
        "promotion_gate": False,
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "status": "DIAGNOSTIC_PASS",
        "strict_lab_gate_pass": True,
        "source_hashes": {
            "baseline_csv": _sha256(baseline),
            "adjusted_csv": _sha256(adjusted),
            "spot_parquet": _sha256(spot),
        },
    }


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.summarize_epex_shape_lab_spot_backtests import summarize_spot_backtests


def test_summarize_spot_backtests_selects_weak_bucket_but_does_not_replace_incumbent(tmp_path: Path) -> None:
    sweep_summary = _write_sweep(tmp_path, ["trial_good", "trial_bad"])
    backtest_root = tmp_path / "backtests"
    _write_backtest(
        backtest_root / "trial_good" / "spot_backtest_summary.json",
        overall=0.30,
        night=0.12,
        night_pos=10,
        ramp=0.034,
        ramp_pos=10,
        evening=0.32,
        solar=0.29,
        weekend=0.20,
        post=0.23,
        sha="good",
    )
    _write_backtest(
        backtest_root / "trial_bad" / "spot_backtest_summary.json",
        overall=0.35,
        night=0.13,
        night_pos=10,
        ramp=0.02,
        ramp_pos=6,
        evening=0.31,
        solar=0.30,
        weekend=0.21,
        post=0.20,
        sha="bad",
    )
    incumbent = tmp_path / "incumbent.json"
    _write_backtest(
        incumbent,
        overall=0.40,
        night=0.03,
        night_pos=5,
        ramp=0.035,
        ramp_pos=8,
        evening=0.45,
        solar=0.44,
        weekend=0.29,
        post=0.30,
        sha="incumbent",
    )

    summary = summarize_spot_backtests(
        sweep_summary=sweep_summary,
        backtest_root=backtest_root,
        output_dir=tmp_path / "out",
        incumbent_backtest=incumbent,
    )

    assert summary["benchmark_policy"] == "summarized_eligible_trials_no_ompex"
    assert summary["promotion_gate"] is False
    assert summary["best_weak_bucket_trial"]["trial_id"] == "trial_good"
    assert summary["replacement_verdict"]["replace_incumbent"] is False
    assert summary["replacement_verdict"]["status"] == "WEAK_BUCKET_GAIN_BUT_INCUMBENT_STILL_DOMINATES_CORE_METRICS"
    assert (tmp_path / "out" / "spot_backtest_trial_ranking.csv").exists()


def test_summarize_spot_backtests_rejects_ompex_backtest(tmp_path: Path) -> None:
    sweep_summary = _write_sweep(tmp_path, ["trial"])
    backtest_root = tmp_path / "backtests"
    path = backtest_root / "trial" / "spot_backtest_summary.json"
    _write_backtest(path, overall=1.0, night=1.0, ramp=1.0)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["ompex_used_in_backtest"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="uses OMPEX"):
        summarize_spot_backtests(
            sweep_summary=sweep_summary,
            backtest_root=backtest_root,
            output_dir=tmp_path / "out",
        )


def test_summarize_spot_backtests_does_not_treat_false_string_as_eligible(tmp_path: Path) -> None:
    ranking_csv = tmp_path / "sweep_ranking.csv"
    ranking_csv.write_text("trial_id,eligible_for_selection\ntrial_false,False\ntrial_true,True\n", encoding="utf-8")
    sweep_summary = tmp_path / "sweep_summary.json"
    sweep_summary.write_text(
        json.dumps(
            {
                "benchmark_policy": "executed_independent_no_ompex",
                "ompex_used_in_model": False,
                "ompex_used_in_selection": False,
                "production_approved": False,
                "ranking_csv": str(ranking_csv),
            }
        ),
        encoding="utf-8",
    )
    backtest_root = tmp_path / "backtests"
    _write_backtest(backtest_root / "trial_true" / "spot_backtest_summary.json", overall=1.0, night=1.0, ramp=1.0)

    summary = summarize_spot_backtests(
        sweep_summary=sweep_summary,
        backtest_root=backtest_root,
        output_dir=tmp_path / "out",
    )

    assert summary["trial_count_from_sweep"] == 1
    assert summary["best_weak_bucket_trial"]["trial_id"] == "trial_true"


def _write_sweep(tmp_path: Path, trial_ids: list[str]) -> Path:
    ranking_csv = tmp_path / "sweep_ranking.csv"
    pd.DataFrame(
        {
            "trial_id": trial_ids,
            "eligible_for_selection": [True] * len(trial_ids),
        }
    ).to_csv(ranking_csv, index=False)
    path = tmp_path / "sweep_summary.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_policy": "executed_independent_no_ompex",
                "ompex_used_in_model": False,
                "ompex_used_in_selection": False,
                "production_approved": False,
                "ranking_csv": str(ranking_csv),
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_backtest(
    path: Path,
    *,
    overall: float,
    night: float,
    ramp: float,
    night_pos: int = 10,
    ramp_pos: int = 10,
    evening: float = 0.3,
    solar: float = 0.3,
    weekend: float = 0.2,
    post: float = 0.2,
    sha: str = "abc",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_policy": "rolling_origin_epex_spot_no_ompex_lab_only",
        "production_approved": False,
        "promotion_gate": False,
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "status": "DIAGNOSTIC_PASS",
        "strict_lab_gate_pass": True,
        "source_hashes": {"adjusted_csv": sha},
        "rolling_metrics": {"mean_profile_mae_improvement_eur_mwh": overall},
        "post_valuation_metrics": {"residual_mae_improvement_eur_mwh": post},
        "rolling_bucket_metrics": {
            "residual_level:all": {
                "mean_mae_improvement_eur_mwh": overall,
                "positive_mae_improvement_fold_count": 10,
            },
            "residual_level:night_00_05": {
                "mean_mae_improvement_eur_mwh": night,
                "positive_mae_improvement_fold_count": night_pos,
            },
            "hourly_ramp:all": {
                "mean_mae_improvement_eur_mwh": ramp,
                "positive_mae_improvement_fold_count": ramp_pos,
            },
            "residual_level:evening_ramp_17_21": {
                "mean_mae_improvement_eur_mwh": evening,
                "positive_mae_improvement_fold_count": 10,
            },
            "residual_level:solar_tail_mar_oct_10_16": {
                "mean_mae_improvement_eur_mwh": solar,
                "positive_mae_improvement_fold_count": 10,
            },
            "residual_level:weekend": {
                "mean_mae_improvement_eur_mwh": weekend,
                "positive_mae_improvement_fold_count": 10,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

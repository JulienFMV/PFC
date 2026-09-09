from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analyze_epex_shape_lab_sweep_sensitivity import analyze_sweep_sensitivity


def test_analyze_sweep_sensitivity_writes_parameter_response(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    selection = _write_selection(tmp_path)

    summary = analyze_sweep_sensitivity(
        plan_json=plan,
        selection_summary=selection,
        output_dir=tmp_path / "sensitivity",
    )

    assert summary["schema_version"] == "epex_shape_lab_sweep_sensitivity.v1"
    assert summary["benchmark_policy"] == "parameter_sensitivity_no_ompex_lab_only"
    assert summary["ompex_used_in_model"] is False
    assert summary["replacement_candidate_count"] == 0
    assert (
        summary["next_hypothesis_hint"]
        == "protect_post_valuation_before_expanding_weak_bucket_gains"
    )
    assert summary["best_by_metric"]["overall_mae_improvement_eur_mwh"]["trial_id"] == "trial_low"
    assert summary["best_by_metric"]["post_valuation_mae_improvement_eur_mwh"]["trial_id"] == "trial_high"
    sensitivity = pd.read_csv(tmp_path / "sensitivity" / "parameter_sensitivity.csv")
    low_tail = sensitivity[sensitivity["parameter"].eq("low_tail_intensity")]
    assert set(low_tail["value"].round(2)) == {0.1, 0.25}
    merged = pd.read_csv(tmp_path / "sensitivity" / "trial_parameter_metrics.csv")
    assert {"trial_id", "low_tail_intensity", "overall_mae_improvement_eur_mwh"}.issubset(merged.columns)


def test_analyze_sweep_sensitivity_rejects_ompex_selection(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    selection = _write_selection(tmp_path)
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["ompex_used_in_backtest"] = True
    selection.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must not use OMPEX"):
        analyze_sweep_sensitivity(
            plan_json=plan,
            selection_summary=selection,
            output_dir=tmp_path / "sensitivity",
        )


def _write_plan(tmp_path: Path) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "activation_status": "lab_only",
                "benchmark_policy": "pre_registered_independent_no_ompex",
                "production_approved": False,
                "ompex_used_in_model": False,
                "ompex_used_in_selection": False,
                "trials": [
                    {
                        "trial_id": "trial_low",
                        "parameters": {
                            "low_tail_intensity": 0.1,
                            "night_intensity": 0.55,
                        },
                    },
                    {
                        "trial_id": "trial_high",
                        "parameters": {
                            "low_tail_intensity": 0.25,
                            "night_intensity": 0.55,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_selection(tmp_path: Path) -> Path:
    ranking = tmp_path / "ranking.csv"
    pd.DataFrame(
        [
            {
                "trial_id": "trial_low",
                "strict_lab_gate_pass": True,
                "overall_mae_improvement_eur_mwh": 0.46,
                "post_valuation_mae_improvement_eur_mwh": 0.29,
                "weak_bucket_candidate": True,
                "replacement_candidate": False,
            },
            {
                "trial_id": "trial_high",
                "strict_lab_gate_pass": True,
                "overall_mae_improvement_eur_mwh": 0.45,
                "post_valuation_mae_improvement_eur_mwh": 0.31,
                "weak_bucket_candidate": False,
                "replacement_candidate": False,
            },
        ]
    ).to_csv(ranking, index=False)
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_policy": "summarized_eligible_trials_no_ompex",
                "production_approved": False,
                "promotion_gate": False,
                "ompex_used_in_model": False,
                "ompex_used_in_selection": False,
                "ompex_used_in_backtest": False,
                "ranking_csv": str(ranking),
                "replacement_verdict": {
                    "replace_incumbent": False,
                    "degraded_vs_incumbent": ["post_valuation_mae_improvement_eur_mwh"],
                },
            }
        ),
        encoding="utf-8",
    )
    return path

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plan_epex_lab_locked_holdout import build_plan


def test_plan_epex_lab_locked_holdout_binds_selected_no_ompex_candidate(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    baseline.write_text("baseline", encoding="utf-8")
    adjusted.write_text("adjusted", encoding="utf-8")
    adjusted_sha = _sha256(adjusted)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "replacement_verdict": {"replace_incumbent": True},
                "selected_adjusted_csv_sha256": adjusted_sha,
                "selected_trial": {"adjusted_csv_sha256": adjusted_sha},
                "ompex_used_in_model": False,
                "ompex_used_in_selection": False,
                "ompex_used_in_backtest": False,
            }
        ),
        encoding="utf-8",
    )
    lab_manifest = tmp_path / "lab.json"
    lab_manifest.write_text(json.dumps({"config": {"weekend_intensity": 0.75}}), encoding="utf-8")

    plan = build_plan(
        baseline_csv=baseline,
        adjusted_csv=adjusted,
        selection_summary=selection,
        lab_manifest=lab_manifest,
        output=tmp_path / "plan.json",
        frozen_at_utc="2026-07-09T00:00:00Z",
        holdout_start_utc="2026-07-10T00:00:00Z",
        holdout_end_utc="2026-07-24T00:00:00Z",
        min_holdout_hours=300,
    )

    assert plan["schema_version"] == "epex_lab_locked_holdout_plan.v1"
    assert plan["benchmark_policy"] == "locked_future_no_ompex_holdout"
    assert plan["production_approved"] is False
    assert plan["adjusted_csv_sha256"] == adjusted_sha
    assert plan["selection_policy"]["pass"] is True
    assert plan["locked_lab_config"] == {"weekend_intensity": 0.75}
    assert plan["pass_criteria"]["min_holdout_hours"] == 300
    assert "OMPEX" in plan["forbidden_inputs"]
    assert "compare_hpfc_ompex" not in plan["commands"]["run_future_backtest_template"]
    assert (tmp_path / "plan.json").exists()


def test_plan_epex_lab_locked_holdout_rejects_unbound_selection(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    baseline.write_text("baseline", encoding="utf-8")
    adjusted.write_text("adjusted", encoding="utf-8")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "replacement_verdict": {"replace_incumbent": True},
                "selected_adjusted_csv_sha256": "wrong",
                "ompex_used_in_model": False,
                "ompex_used_in_selection": False,
                "ompex_used_in_backtest": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="selection summary is not bound"):
        build_plan(
            baseline_csv=baseline,
            adjusted_csv=adjusted,
            selection_summary=selection,
            frozen_at_utc="2026-07-09T00:00:00Z",
            holdout_start_utc="2026-07-10T00:00:00Z",
            holdout_end_utc="2026-07-24T00:00:00Z",
        )


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.plan_epex_shape_lab_sweep import _parse_grid, build_plan


def test_plan_epex_shape_lab_sweep_is_pre_registered_and_no_ompex(tmp_path) -> None:
    candidate = tmp_path / "candidate.csv"
    spot = tmp_path / "spot.parquet"
    candidate.write_text("timestamp_ch,price_weighted_mean_eur_mwh\n", encoding="utf-8")
    pd.DataFrame({"price_eur_mwh": [1.0]}, index=pd.date_range("2025-01-01", periods=1, tz="UTC")).to_parquet(spot)

    plan = build_plan(
        candidate_csv=candidate,
        spot_parquet=spot,
        output_root=tmp_path / "sweep",
        valuation_timestamp="2026-07-07T00:00:00Z",
        max_abs_delta_eur_mwh=6.0,
        grid={
            "weekend_intensity": [0.25, 0.5],
            "low_tail_intensity": [0.25],
            "peak_subshape_intensity": [0.25, 0.5],
        },
    )

    assert plan["activation_status"] == "lab_only"
    assert plan["production_approved"] is False
    assert plan["ompex_used_in_selection"] is False
    assert "OMPEX" in plan["forbidden_selection_inputs"]
    assert plan["trial_count"] == 4
    assert all("compare_hpfc_ompex_benchmark" not in trial["commands"]["run_ab"] for trial in plan["trials"])
    assert all("audit_governance_no_ompex" in trial["commands"] for trial in plan["trials"])
    assert all("explain_adjustment_no_ompex" in trial["commands"] for trial in plan["trials"])
    assert all("scripts/explain_epex_shape_lab_adjustment.py" in trial["commands"]["explain_adjustment_no_ompex"] for trial in plan["trials"])
    assert all("compare_hpfc_ompex_benchmark" not in trial["commands"]["explain_adjustment_no_ompex"] for trial in plan["trials"])
    assert "shape explainability with projection/cap/floor decomposition" in plan["selection_basis"]
    assert plan["selection_thresholds"]["max_epex_spot_age_days"] == 14.0
    assert plan["scoring_policy"]["midday_weight"] == 1.0
    assert plan["scoring_policy"]["ramp_penalty_weight"] == 1.0
    assert plan["grid"]["evening_recovery_intensity"] == [0.0]
    assert plan["grid"]["night_intensity"] == [0.0]
    assert plan["grid"]["ramp_intensity"] == [0.0]
    json.dumps(plan)


def test_plan_epex_shape_lab_sweep_can_pre_register_delta_cap_grid(tmp_path) -> None:
    candidate = tmp_path / "candidate.csv"
    spot = tmp_path / "spot.parquet"
    candidate.write_text("timestamp_ch,price_weighted_mean_eur_mwh\n", encoding="utf-8")
    pd.DataFrame({"price_eur_mwh": [1.0]}, index=pd.date_range("2025-01-01", periods=1, tz="UTC")).to_parquet(spot)

    plan = build_plan(
        candidate_csv=candidate,
        spot_parquet=spot,
        output_root=tmp_path / "sweep",
        valuation_timestamp="2026-07-07T00:00:00Z",
        max_abs_delta_eur_mwh=6.0,
        max_abs_delta_grid=[2.0, 4.0],
        grid={
            "weekend_intensity": [0.25],
            "low_tail_intensity": [0.25],
            "peak_subshape_intensity": [0.25, 0.5],
        },
    )

    assert plan["max_abs_delta_grid"] == [2.0, 4.0]
    assert plan["trial_count"] == 4
    assert {trial["parameters"]["max_abs_delta_eur_mwh"] for trial in plan["trials"]} == {2.0, 4.0}
    assert all("_d" in trial["trial_id"] for trial in plan["trials"])
    assert all("--lab-manifest" in trial["commands"]["explain_adjustment_no_ompex"] for trial in plan["trials"])


def test_plan_epex_shape_lab_sweep_can_pre_register_t047_night_ramp_dimensions(tmp_path) -> None:
    candidate = tmp_path / "candidate.csv"
    spot = tmp_path / "spot.parquet"
    candidate.write_text("timestamp_ch,price_weighted_mean_eur_mwh\n", encoding="utf-8")
    pd.DataFrame({"price_eur_mwh": [1.0]}, index=pd.date_range("2025-01-01", periods=1, tz="UTC")).to_parquet(spot)

    plan = build_plan(
        candidate_csv=candidate,
        spot_parquet=spot,
        output_root=tmp_path / "sweep",
        valuation_timestamp="2026-07-07T00:00:00Z",
        max_abs_delta_eur_mwh=3.0,
        plan_id="epex_shape_lab_sweep_t047_v3",
        grid={
            "weekend_intensity": [0.5],
            "low_tail_intensity": [0.25],
            "peak_subshape_intensity": [0.75],
            "evening_recovery_intensity": [0.0, 0.5],
            "night_intensity": [0.0, 0.25],
            "ramp_intensity": [0.0, 0.25],
        },
    )

    assert plan["plan_id"] == "epex_shape_lab_sweep_t047_v3"
    assert plan["trial_count"] == 8
    assert plan["scoring_policy"]["night_weight"] == 0.5
    assert {trial["parameters"]["evening_recovery_intensity"] for trial in plan["trials"]} == {0.0, 0.5}
    assert {trial["parameters"]["night_intensity"] for trial in plan["trials"]} == {0.0, 0.25}
    assert {trial["parameters"]["ramp_intensity"] for trial in plan["trials"]} == {0.0, 0.25}
    assert all("--night-intensity" in trial["commands"]["run_ab"] for trial in plan["trials"])
    assert all("--ramp-intensity" in trial["commands"]["run_ab"] for trial in plan["trials"])
    assert all("--evening-recovery-intensity" in trial["commands"]["run_ab"] for trial in plan["trials"])
    assert all("compare_hpfc_ompex_benchmark" not in trial["commands"]["run_ab"] for trial in plan["trials"])


def test_plan_epex_shape_lab_sweep_rejects_unknown_selection_threshold(tmp_path) -> None:
    candidate = tmp_path / "candidate.csv"
    spot = tmp_path / "spot.parquet"
    candidate.write_text("timestamp_ch,price_weighted_mean_eur_mwh\n", encoding="utf-8")
    pd.DataFrame({"price_eur_mwh": [1.0]}, index=pd.date_range("2025-01-01", periods=1, tz="UTC")).to_parquet(spot)

    with pytest.raises(ValueError, match="unknown selection thresholds keys"):
        build_plan(
            candidate_csv=candidate,
            spot_parquet=spot,
            output_root=tmp_path / "sweep",
            valuation_timestamp="2026-07-07T00:00:00Z",
            max_abs_delta_eur_mwh=6.0,
            grid={
                "weekend_intensity": [0.25],
                "low_tail_intensity": [0.25],
                "peak_subshape_intensity": [0.25],
            },
            selection_thresholds={"min_solar_tail_mae_improvement_eur_mwh": 0.25},
        )


def test_plan_epex_shape_lab_sweep_accepts_midday_scoring_weight(tmp_path) -> None:
    candidate = tmp_path / "candidate.csv"
    spot = tmp_path / "spot.parquet"
    candidate.write_text("timestamp_ch,price_weighted_mean_eur_mwh\n", encoding="utf-8")
    pd.DataFrame({"price_eur_mwh": [1.0]}, index=pd.date_range("2025-01-01", periods=1, tz="UTC")).to_parquet(spot)

    plan = build_plan(
        candidate_csv=candidate,
        spot_parquet=spot,
        output_root=tmp_path / "sweep",
        valuation_timestamp="2026-07-07T00:00:00Z",
        max_abs_delta_eur_mwh=6.0,
        grid={
            "weekend_intensity": [0.25],
            "low_tail_intensity": [0.25],
            "peak_subshape_intensity": [0.25],
        },
        scoring_policy={"midday_weight": 1.25},
    )

    assert plan["scoring_policy"]["midday_weight"] == 1.25


def test_plan_epex_shape_lab_sweep_accepts_json_file_arguments(tmp_path) -> None:
    grid_json = tmp_path / "grid.json"
    grid_json.write_text('{"weekend_intensity":[0.5],"low_tail_intensity":[0.25],"peak_subshape_intensity":[0.75]}')

    parsed = _parse_grid(f"@{grid_json}")

    assert parsed == {
        "weekend_intensity": [0.5],
        "low_tail_intensity": [0.25],
        "peak_subshape_intensity": [0.75],
    }

from __future__ import annotations

import json

import pandas as pd

from scripts.plan_epex_shape_lab_sweep import build_plan


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
    assert plan["selection_thresholds"]["max_epex_spot_age_days"] == 14.0
    assert plan["scoring_policy"]["ramp_penalty_weight"] == 1.0
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

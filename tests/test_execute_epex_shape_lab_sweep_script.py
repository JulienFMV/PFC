from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.execute_epex_shape_lab_sweep import execute_sweep
from scripts.plan_epex_shape_lab_sweep import build_plan


def _candidate_csv(path, timestamps: pd.DatetimeIndex) -> None:
    values = [80.0 + (ts.hour / 24.0) for ts in timestamps]
    frame = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%d.%m.%Y %H:%M"),
            "utc_offset_ch": timestamps.strftime("UTC%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%d.%m.%Y %H:%M"),
            "price_slow_eur_mwh": [v - 2.0 for v in values],
            "price_central_eur_mwh": values,
            "price_fast_eur_mwh": [v + 2.0 for v in values],
            "price_weighted_mean_eur_mwh": values,
            "structural_p10_eur_mwh": [v - 8.0 for v in values],
            "structural_p50_eur_mwh": values,
            "structural_p90_eur_mwh": [v + 12.0 for v in values],
            "structural_width_eur_mwh": [20.0] * len(values),
        }
    )
    frame.to_csv(path, index=False)


def _spot_parquet(path) -> None:
    timestamps = pd.date_range("2024-01-01", "2025-12-31 23:00", freq="h", tz="Europe/Zurich")
    price = 70.0 + (timestamps.hour.to_numpy(dtype=float) / 24.0)
    pd.DataFrame({"price_eur_mwh": price}, index=timestamps.tz_convert("UTC")).to_parquet(path)


def test_execute_epex_shape_lab_sweep_runs_pre_registered_trials_without_ompex(tmp_path) -> None:
    candidate = tmp_path / "candidate.csv"
    spot = tmp_path / "spot.parquet"
    timestamps = pd.date_range("2030-01-01", "2030-01-07 23:00", freq="h", tz="Europe/Zurich")
    _candidate_csv(candidate, timestamps)
    _spot_parquet(spot)
    plan = build_plan(
        candidate_csv=candidate,
        spot_parquet=spot,
        output_root=tmp_path / "sweep",
        valuation_timestamp="2026-01-01T00:00:00Z",
        max_abs_delta_eur_mwh=2.0,
        grid={
            "weekend_intensity": [0.25],
            "low_tail_intensity": [0.25],
            "peak_subshape_intensity": [0.25],
        },
    )
    plan_json = tmp_path / "plan.json"
    plan_json.write_text(json.dumps(plan), encoding="utf-8")

    summary = execute_sweep(plan_json=plan_json, output_summary=tmp_path / "summary.json")

    assert summary["benchmark_policy"] == "executed_independent_no_ompex"
    assert summary["ompex_used_in_selection"] is False
    assert summary["trial_count_executed"] == 1
    assert summary["eligible_count"] == 1
    assert "OMPEX" in summary["forbidden_selection_inputs"]
    assert (tmp_path / "summary.csv").exists()


def test_execute_epex_shape_lab_sweep_rejects_negative_max_trials(tmp_path) -> None:
    plan_json = _single_trial_plan_json(tmp_path)

    with pytest.raises(ValueError, match="max-trials"):
        execute_sweep(plan_json=plan_json, output_summary=tmp_path / "summary.json", max_trials=-1)


def test_execute_epex_shape_lab_sweep_rejects_trial_output_outside_sweep_root(tmp_path) -> None:
    plan_json = _single_trial_plan_json(tmp_path)
    plan = json.loads(plan_json.read_text(encoding="utf-8"))
    plan["trials"][0]["output_dir"] = str(tmp_path.parent / "outside_sweep")
    plan_json.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="outside sweep root"):
        execute_sweep(plan_json=plan_json, output_summary=tmp_path / "summary.json")


def test_execute_epex_shape_lab_sweep_recomputes_stale_resume_manifest(tmp_path) -> None:
    plan_json = _single_trial_plan_json(tmp_path)
    execute_sweep(plan_json=plan_json, output_summary=tmp_path / "summary.json")
    plan = json.loads(plan_json.read_text(encoding="utf-8"))
    manifest_json = tmp_path / "sweep" / plan["trials"][0]["trial_id"] / "ab_lab_manifest.json"
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    manifest["source_hashes"]["candidate_csv"] = "stale"
    manifest_json.write_text(json.dumps(manifest), encoding="utf-8")

    execute_sweep(plan_json=plan_json, output_summary=tmp_path / "summary_rerun.json")

    refreshed = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert refreshed["source_hashes"]["candidate_csv"] == plan["candidate_csv_sha256"]


def test_execute_epex_shape_lab_sweep_best_trial_is_none_when_no_trial_eligible(tmp_path) -> None:
    plan_json = _single_trial_plan_json(tmp_path)
    execute_sweep(plan_json=plan_json, output_summary=tmp_path / "summary.json")
    plan = json.loads(plan_json.read_text(encoding="utf-8"))
    governance_json = (
        tmp_path
        / "sweep"
        / plan["trials"][0]["trial_id"]
        / "governance_audit"
        / "epex_shape_lab_governance_audit.json"
    )
    governance = json.loads(governance_json.read_text(encoding="utf-8"))
    governance["status"] = "FAIL"
    governance["failed_count"] = 1
    governance_json.write_text(json.dumps(governance), encoding="utf-8")

    summary = execute_sweep(plan_json=plan_json, output_summary=tmp_path / "summary_failed.json")

    assert summary["eligible_count"] == 0
    assert summary["best_trial"] is None


def _single_trial_plan_json(tmp_path):
    candidate = tmp_path / "candidate.csv"
    spot = tmp_path / "spot.parquet"
    timestamps = pd.date_range("2030-01-01", "2030-01-07 23:00", freq="h", tz="Europe/Zurich")
    _candidate_csv(candidate, timestamps)
    _spot_parquet(spot)
    plan = build_plan(
        candidate_csv=candidate,
        spot_parquet=spot,
        output_root=tmp_path / "sweep",
        valuation_timestamp="2026-01-01T00:00:00Z",
        max_abs_delta_eur_mwh=2.0,
        grid={
            "weekend_intensity": [0.25],
            "low_tail_intensity": [0.25],
            "peak_subshape_intensity": [0.25],
        },
    )
    plan_json = tmp_path / "plan.json"
    plan_json.write_text(json.dumps(plan), encoding="utf-8")
    return plan_json

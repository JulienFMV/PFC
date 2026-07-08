from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.summarize_epex_shape_lab_stability import summarize_stability


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _case(
    tmp_path: Path,
    label: str,
    *,
    ramp_increase: float = 0.5,
    peak_offpeak_spread_delta: float = 1.0,
    p10_negative_hours: int = 12,
    include_extended_metrics: bool = True,
) -> tuple[str, Path, Path]:
    summary = tmp_path / f"{label}_summary.json"
    governance = tmp_path / f"{label}_governance.json"
    payload = {
        "benchmark_policy": "independent_no_ompex",
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "n_hours": 24,
        "finite_adjusted_ok": True,
        "quantile_order_adjusted_ok": True,
        "weighted_negative_hours_adjusted": 0,
        "min_adjusted_price_eur_mwh": 1.0,
        "max_abs_delta_eur_mwh": 3.0,
        "max_abs_monthly_mean_delta_eur_mwh": 1e-8,
        "max_abs_width_delta_eur_mwh": 0.0,
        "ramp_abs_p99_baseline_eur_mwh": 10.0,
        "ramp_abs_p99_adjusted_eur_mwh": 10.0 + ramp_increase,
    }
    if include_extended_metrics:
        payload.update(
            {
                "max_abs_implied_width_delta_eur_mwh": 0.0,
                "max_abs_reported_minus_implied_width_adjusted_eur_mwh": 0.0,
                "max_abs_peak_offpeak_spread_delta_eur_mwh": peak_offpeak_spread_delta,
                "max_abs_month_hour_mean_delta_eur_mwh": 2.5,
                "max_abs_boundary_delta_jump_eur_mwh": 0.25,
                "p10_negative_hours_adjusted": p10_negative_hours,
                "p10_negative_cluster_max_hours": 6,
            }
        )
    _write_json(summary, payload)
    _write_json(governance, {"status": "PASS", "failed_count": 0})
    return label, summary, governance


def test_summarize_epex_shape_lab_stability_passes_multiple_no_ompex_cases(tmp_path: Path) -> None:
    output = tmp_path / "out"
    decision = summarize_stability(
        cases=[_case(tmp_path, "asof1"), _case(tmp_path, "asof2", ramp_increase=0.8)],
        output_dir=output,
    )

    assert decision["status"] == "PASS"
    assert decision["case_count"] == 2
    assert decision["passed_case_count"] == 2
    assert decision["promotion_gate"] is False
    cases = pd.read_csv(output / "stability_cases.csv")
    assert set(cases["label"]) == {"asof1", "asof2"}
    written = json.loads((output / "stability_summary.json").read_text(encoding="utf-8"))
    assert written["benchmark_policy"] == "multi_date_independent_no_ompex"


def test_summarize_epex_shape_lab_stability_fails_ramp_threshold(tmp_path: Path) -> None:
    decision = summarize_stability(
        cases=[_case(tmp_path, "asof1", ramp_increase=1.5)],
        output_dir=tmp_path / "out",
        max_ramp_p99_increase=1.0,
    )

    assert decision["status"] == "FAIL"
    assert decision["failed_case_count"] == 1


def test_summarize_epex_shape_lab_stability_fails_peak_offpeak_threshold(tmp_path: Path) -> None:
    decision = summarize_stability(
        cases=[_case(tmp_path, "asof1", peak_offpeak_spread_delta=6.0)],
        output_dir=tmp_path / "out",
        max_peak_offpeak_spread_delta=5.0,
    )

    assert decision["status"] == "FAIL"
    cases = pd.read_csv(tmp_path / "out" / "stability_cases.csv")
    assert cases.loc[0, "max_abs_peak_offpeak_spread_delta_eur_mwh"] == 6.0


def test_summarize_epex_shape_lab_stability_fails_when_extended_metrics_are_missing(tmp_path: Path) -> None:
    decision = summarize_stability(
        cases=[_case(tmp_path, "asof1", include_extended_metrics=False)],
        output_dir=tmp_path / "out",
    )

    assert decision["status"] == "FAIL"
    assert decision["failed_case_count"] == 1

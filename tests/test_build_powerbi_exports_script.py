from __future__ import annotations

import pandas as pd

from scripts.build_powerbi_exports import _quality_gate_issues, load_hourly


def test_powerbi_quality_gate_blocks_failed_shape_and_seasonality() -> None:
    seasonal = pd.DataFrame(
        [
            {
                "severity": "critical",
                "reason": "annual-only synthetic monthly amplitude collapses versus quoted/partial previous year",
            }
        ]
    )
    cross_year = pd.DataFrame(
        [
            {
                "severity": "warning",
                "reason": "same-month values are near-cloned despite a non-zero parent bucket spread",
            },
            {
                "severity": "warning",
                "reason": "same-month values are near-cloned despite a non-zero parent bucket spread",
            },
        ]
    )

    issues = _quality_gate_issues(
        shape_metrics={
            "score_10": 3.25,
            "finite_ok": 1.0,
            "quantile_order": 1.0,
            "max_eex_base_error_eur_mwh": 0.0,
            "max_eex_peak_error_eur_mwh": 17.47,
            "eex_peak_residual_count": 1.0,
            "negative_gate_status": "PASS",
        },
        seasonal_checks=seasonal,
        monthly_split_checks=pd.DataFrame(),
        monthly_path_checks=pd.DataFrame(),
        cross_year_checks=cross_year,
        calendar_checks=pd.DataFrame(),
    )

    assert "shape_score_10=3.25 < 8.50" in issues
    assert "max_eex_peak_error_eur_mwh=17.470000 > 0.010000" in issues
    assert "seasonal_critical_flags=1" in issues
    assert "cross_year_near_clone_warnings=2" in issues


def test_powerbi_quality_gate_passes_clean_metrics() -> None:
    issues = _quality_gate_issues(
        shape_metrics={
            "score_10": 9.1,
            "finite_ok": 1.0,
            "quantile_order": 1.0,
            "max_eex_base_error_eur_mwh": 0.0,
            "max_eex_peak_error_eur_mwh": 0.0,
            "eex_peak_residual_count": 1.0,
            "negative_gate_status": "PASS",
        },
        seasonal_checks=pd.DataFrame({"severity": ["ok"]}),
        monthly_split_checks=pd.DataFrame({"severity": ["ok"]}),
        monthly_path_checks=pd.DataFrame({"severity": ["ok"]}),
        cross_year_checks=pd.DataFrame({"severity": ["ok"], "reason": ["coherent"]}),
        calendar_checks=pd.DataFrame({"severity": ["ok"]}),
    )

    assert issues == []


def test_powerbi_quality_gate_blocks_structural_invariant_failures() -> None:
    issues = _quality_gate_issues(
        shape_metrics={
            "score_10": 8.5,
            "finite_ok": 0.0,
            "quantile_order": 0.0,
            "max_eex_base_error_eur_mwh": 0.0,
            "max_eex_peak_error_eur_mwh": 0.0,
            "eex_peak_residual_count": 1.0,
            "negative_gate_status": "PASS",
        },
        seasonal_checks=pd.DataFrame({"severity": ["ok"]}),
        monthly_split_checks=pd.DataFrame({"severity": ["ok"]}),
        monthly_path_checks=pd.DataFrame({"severity": ["ok"]}),
        cross_year_checks=pd.DataFrame({"severity": ["ok"], "reason": ["coherent"]}),
        calendar_checks=pd.DataFrame({"severity": ["ok"]}),
    )

    assert "finite_ok=FAILED" in issues
    assert "quantile_order=FAILED" in issues


def test_powerbi_quality_gate_requires_peak_repricing_evidence() -> None:
    issues = _quality_gate_issues(
        shape_metrics={
            "score_10": 9.1,
            "finite_ok": 1.0,
            "quantile_order": 1.0,
            "max_eex_base_error_eur_mwh": 0.0,
            "max_eex_peak_error_eur_mwh": 0.0,
            "eex_peak_residual_count": 0.0,
            "negative_gate_status": "PASS",
        },
        seasonal_checks=pd.DataFrame({"severity": ["ok"]}),
        monthly_split_checks=pd.DataFrame({"severity": ["ok"]}),
        monthly_path_checks=pd.DataFrame({"severity": ["ok"]}),
        cross_year_checks=pd.DataFrame({"severity": ["ok"], "reason": ["coherent"]}),
        calendar_checks=pd.DataFrame({"severity": ["ok"]}),
    )

    assert "eex_peak_residual_count=0" in issues


def test_powerbi_load_hourly_orders_structural_fallback_from_crossed_scenarios(tmp_path) -> None:
    path = tmp_path / "hourly.csv"
    pd.DataFrame(
        {
            "timestamp_ch": ["01.07.2026 12:00", "01.07.2026 13:00"],
            "utc_offset_ch": ["UTC+02:00", "UTC+02:00"],
            "timestamp_utc": ["01.07.2026 10:00", "01.07.2026 11:00"],
            "price_slow_eur_mwh": [120.0, 80.0],
            "price_central_eur_mwh": [100.0, 100.0],
            "price_fast_eur_mwh": [90.0, 130.0],
            "price_weighted_mean_eur_mwh": [102.5, 102.5],
        }
    ).to_csv(path, index=False)

    loaded = load_hourly(path)

    assert loaded["structural_p10_eur_mwh"].tolist() == [90.0, 80.0]
    assert loaded["structural_p50_eur_mwh"].tolist() == [100.0, 100.0]
    assert loaded["structural_p90_eur_mwh"].tolist() == [120.0, 130.0]
    assert (loaded["structural_width_eur_mwh"] >= 0.0).all()

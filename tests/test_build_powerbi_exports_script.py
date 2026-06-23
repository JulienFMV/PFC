from __future__ import annotations

import pandas as pd
import pytest

import scripts.build_powerbi_exports as powerbi_script
from scripts.build_powerbi_exports import _quality_gate_issues, build_exports, load_hourly, main


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
            "max_eex_base_error_eur_mwh": 0.0,
            "max_eex_peak_error_eur_mwh": 17.47,
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
            "max_eex_base_error_eur_mwh": 0.0,
            "max_eex_peak_error_eur_mwh": 0.0,
            "negative_gate_status": "PASS",
        },
        seasonal_checks=pd.DataFrame({"severity": ["ok"]}),
        monthly_split_checks=pd.DataFrame({"severity": ["ok"]}),
        monthly_path_checks=pd.DataFrame({"severity": ["ok"]}),
        cross_year_checks=pd.DataFrame({"severity": ["ok"], "reason": ["coherent"]}),
        calendar_checks=pd.DataFrame({"severity": ["ok"]}),
    )

    assert issues == []


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


def test_powerbi_load_hourly_preserves_valid_partial_structural_columns(tmp_path) -> None:
    path = tmp_path / "hourly.csv"
    pd.DataFrame(
        {
            "timestamp_ch": ["01.07.2026 12:00"],
            "utc_offset_ch": ["UTC+02:00"],
            "timestamp_utc": ["01.07.2026 10:00"],
            "price_slow_eur_mwh": [120.0],
            "price_central_eur_mwh": [100.0],
            "price_fast_eur_mwh": [90.0],
            "price_weighted_mean_eur_mwh": [102.5],
            "structural_p10_eur_mwh": [70.0],
            "structural_p50_eur_mwh": [95.0],
            "structural_p90_eur_mwh": [140.0],
        }
    ).to_csv(path, index=False)

    loaded = load_hourly(path)

    assert loaded.loc[0, "structural_p10_eur_mwh"] == 70.0
    assert loaded.loc[0, "structural_p50_eur_mwh"] == 95.0
    assert loaded.loc[0, "structural_p90_eur_mwh"] == 140.0
    assert loaded.loc[0, "structural_width_eur_mwh"] == 70.0


def test_powerbi_build_exports_blocks_failed_gates_after_normalizing_structural_columns(
    tmp_path,
    monkeypatch,
) -> None:
    csv = _write_minimal_hourly_csv(tmp_path / "hourly.csv")
    forwards = tmp_path / "forwards.parquet"
    spot = tmp_path / "spot.parquet"
    forwards.write_text("stub\n", encoding="utf-8")
    spot.write_text("stub\n", encoding="utf-8")
    _stub_audits(monkeypatch, score_10=3.25)

    with pytest.raises(RuntimeError, match="shape_score_10=3.25 < 8.50"):
        build_exports(csv, forwards, spot, tmp_path / "powerbi")

    assert not (tmp_path / "powerbi" / "hfc_hourly_powerbi.csv").exists()


def test_powerbi_build_exports_allow_failed_gates_writes_diagnostic_summary(
    tmp_path,
    monkeypatch,
) -> None:
    csv = _write_minimal_hourly_csv(tmp_path / "hourly.csv")
    forwards = tmp_path / "forwards.parquet"
    spot = tmp_path / "spot.parquet"
    forwards.write_text("stub\n", encoding="utf-8")
    spot.write_text("stub\n", encoding="utf-8")
    _stub_audits(monkeypatch, score_10=3.25)

    build_exports(csv, forwards, spot, tmp_path / "powerbi", allow_failed_gates=True)

    summary = pd.read_csv(tmp_path / "powerbi" / "summary_metrics.csv")
    metrics = dict(zip(summary["metric"], summary["value"]))
    assert metrics["powerbi_quality_gate_status"] == "FAILED_DIAGNOSTIC"
    assert "shape_score_10=3.25 < 8.50" in metrics["powerbi_quality_gate_issues"]


def test_powerbi_cli_allow_failed_gates_passes_through(tmp_path, monkeypatch) -> None:
    csv = _write_minimal_hourly_csv(tmp_path / "hourly.csv")
    forwards = tmp_path / "forwards.parquet"
    spot = tmp_path / "spot.parquet"
    forwards.write_text("stub\n", encoding="utf-8")
    spot.write_text("stub\n", encoding="utf-8")
    output_dir = tmp_path / "powerbi"
    _stub_audits(monkeypatch, score_10=3.25)

    code = main(
        [
            "--csv",
            str(csv),
            "--forwards",
            str(forwards),
            "--spot",
            str(spot),
            "--output-dir",
            str(output_dir),
            "--allow-failed-gates",
        ]
    )

    assert code == 0
    assert (output_dir / "summary_metrics.csv").exists()


def _write_minimal_hourly_csv(path) -> object:
    timestamps = pd.date_range("2026-07-01 00:00", periods=24, freq="h", tz="Europe/Zurich")
    pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%d.%m.%Y %H:%M"),
            "utc_offset_ch": ["UTC+02:00"] * len(timestamps),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%d.%m.%Y %H:%M"),
            "price_slow_eur_mwh": [90.0] * len(timestamps),
            "price_central_eur_mwh": [100.0] * len(timestamps),
            "price_fast_eur_mwh": [110.0] * len(timestamps),
            "price_weighted_mean_eur_mwh": [100.0] * len(timestamps),
        }
    ).to_csv(path, index=False)
    return path


def _stub_audits(monkeypatch, *, score_10: float) -> None:
    def audit_shape(csv_path, forwards_path):
        frame = pd.read_csv(csv_path)
        assert {"structural_p10_eur_mwh", "structural_p50_eur_mwh", "structural_p90_eur_mwh", "structural_width_eur_mwh"}.issubset(frame.columns)
        annual = pd.DataFrame(
            [
                {
                    "year": 2026,
                    "mean_eur_mwh": 100.0,
                    "evening_minus_midday_eur_mwh": 0.0,
                    "weekend_minus_weekday_eur_mwh": 0.0,
                    "structural_width_mean_eur_mwh": 20.0,
                    "structural_width_p95_eur_mwh": 20.0,
                }
            ]
        )
        residuals = pd.DataFrame({"load_type": ["BASE"], "product": ["2026-07"], "abs_error_eur_mwh": [0.0]})
        metrics = {
            "score_10": score_10,
            "max_eex_base_error_eur_mwh": 0.0,
            "max_eex_peak_error_eur_mwh": 0.0,
            "weighted_negative_hours": 0,
            "negative_gate_status": "PASS",
            "p10_negative_hours": 0,
            "min_price_eur_mwh": 90.0,
        }
        return annual, residuals, metrics

    def audit_seasonal(csv_path, forwards_path):
        return {
            "seasonal_checks": pd.DataFrame({"severity": ["ok"], "reason": ["ok"]}),
            "monthly_path_checks": pd.DataFrame({"severity": ["ok"], "reason": ["ok"]}),
            "cross_year_month_shape_checks": pd.DataFrame({"severity": ["ok"], "reason": ["ok"]}),
            "monthly_split_checks": pd.DataFrame({"severity": ["ok"], "reason": ["ok"]}),
            "calendar_checks": pd.DataFrame({"severity": ["ok"], "reason": ["ok"]}),
        }

    def audit_vs_spot(csv_path, spot_path):
        frame = pd.read_csv(csv_path)
        assert "structural_p10_eur_mwh" in frame.columns
        summary = {
            "score_10": 9.0,
            "latest_hfc_peak_offpeak_spread_eur_mwh": 1.0,
            "latest_hfc_evening_midday_spread_eur_mwh": 1.0,
            "latest_hfc_winter_summer_spread_eur_mwh": 1.0,
            "latest_hfc_shape_corr_vs_spot": 1.0,
        }
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), summary

    monkeypatch.setattr(powerbi_script, "audit_shape", audit_shape)
    monkeypatch.setattr(powerbi_script, "audit_seasonal", audit_seasonal)
    monkeypatch.setattr(powerbi_script, "audit_vs_spot", audit_vs_spot)

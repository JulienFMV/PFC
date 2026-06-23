from __future__ import annotations

import json
from pathlib import Path

import pytest
import pandas as pd

import scripts.export_local_test_ch_hourly_csv as export_script
from pfc_shaping.calibration.monthly_curve_lambda_calibration import config_hash
from scripts.export_local_test_ch_hourly_csv import (
    _ch_market_holiday_pressure,
    _ch_market_nonworking_pressure,
    _eex_peak_mask,
    _latest_eex_prices_by_load_type,
    _is_ch_nonworking_day,
    _negative_capture_weight,
    _peak_shape_down_weight,
    _peak_shape_up_weight,
    apply_final_quant_annual_smoothness_calibration,
    apply_neighbor_annual_residual_shape_anchor,
    apply_neighbor_monthly_spread_anchor,
    apply_synthetic_monthly_path_smoothing,
    apply_post_calibration_negative_rebalancer,
    apply_post_calibration_peak_shape_rebalancer,
    apply_local_test_structural_shape_upgrade,
    calibrate_hourly_to_eex,
    calibrate_hourly_to_eex_base_peak,
    previous_business_day,
    require_forward_date,
    to_hourly_csv_frame,
)
from pfc_shaping.lt.model.holiday_pressure import ch_market_nonworking_pressure


def _solver_manifest(
    *,
    fan: pd.DataFrame,
    fan_path: Path,
    forwards_path: Path,
    forward_snapshot_date: str = "2026-06-15",
    market: str = "CH",
) -> dict[str, object]:
    solver_config = {
        "monthly_level_authority": "solver",
        "skip_legacy_level_cascade": True,
        "skip_legacy_base_smoothing": True,
    }
    return {
        "monthly_curve_schema_version": 1,
        "market": market,
        "forward_snapshot_date": forward_snapshot_date,
        "solver_config": solver_config,
        "solver_config_hash": export_script._sha256_json(solver_config),
        "active_config_hash": config_hash(solver_config),
        "source_hashes": {"forwards_path": export_script._file_sha256(forwards_path)},
        "active_constraints_hash": "constraints-hash",
        "monthly_solution_hash": "solver-solution-hash",
        "fan_chart_weighted_monthly_hash": export_script._fan_weighted_monthly_solution_hash(
            fan,
            timezone=export_script.LOCAL_TZ,
        ),
        "fan_chart_sha256": export_script._file_sha256(fan_path),
        "solver_kkt": {
            "max_abs_constraint_residual": 0.0,
            "stationarity_residual": 0.0,
            "condition_number": 1.0,
            "active_constraint_rank": 1,
            "nullspace_dimension": 0,
            "ridge_used": False,
            "solved_by_lstsq": False,
            "dropped_yoy_rows": 0,
        },
        "monthly_level_authority": "solver",
        "skip_legacy_level_cascade": True,
        "skip_legacy_base_smoothing": True,
    }


def test_to_hourly_csv_frame_filters_local_window_and_averages():
    index = pd.date_range("2026-06-12 22:00", periods=96, freq="15min", tz="UTC")
    fan = pd.DataFrame(
        {
            "curve_slow": range(96),
            "curve_central": range(10, 106),
            "curve_fast": range(20, 116),
            "weighted_mean": range(30, 126),
            "structural_p10": range(40, 136),
            "structural_p50": range(50, 146),
            "structural_p90": range(60, 156),
            "structural_width": [1.0] * 96,
        },
        index=index,
    )

    out = to_hourly_csv_frame(fan, local_start_date="2026-06-13", local_end_date="2026-06-13")

    assert len(out) == 24
    assert out.columns.tolist() == [
        "timestamp_ch",
        "utc_offset_ch",
        "timestamp_utc",
        "price_slow_eur_mwh",
        "price_central_eur_mwh",
        "price_fast_eur_mwh",
        "price_weighted_mean_eur_mwh",
        "structural_scenario_low_eur_mwh",
        "structural_scenario_central_eur_mwh",
        "structural_scenario_high_eur_mwh",
        "structural_scenario_spread_eur_mwh",
        "structural_p10_eur_mwh",
        "structural_p50_eur_mwh",
        "structural_p90_eur_mwh",
        "structural_width_eur_mwh",
    ]
    assert out.loc[0, "timestamp_ch"] == "13.06.2026 00:00"
    assert out.loc[0, "utc_offset_ch"] == "UTC+02:00"
    assert out.loc[0, "timestamp_utc"] == "2026-06-12T22:00:00+0000"
    assert out.loc[0, "price_weighted_mean_eur_mwh"] == 31.5


def test_to_hourly_csv_frame_prefers_ordered_structural_bracket_columns():
    index = pd.date_range("2026-06-12 22:00", periods=96, freq="15min", tz="UTC")
    fan = pd.DataFrame(
        {
            "curve_slow": [120.0] * len(index),
            "curve_central": [100.0] * len(index),
            "curve_fast": [90.0] * len(index),
            "weighted_mean": [102.5] * len(index),
            "structural_scenario_low": [90.0] * len(index),
            "structural_scenario_central": [100.0] * len(index),
            "structural_scenario_high": [120.0] * len(index),
            "structural_scenario_spread": [30.0] * len(index),
            "structural_p10": [120.0] * len(index),
            "structural_p50": [100.0] * len(index),
            "structural_p90": [90.0] * len(index),
            "structural_width": [-30.0] * len(index),
        },
        index=index,
    )

    out = to_hourly_csv_frame(fan, local_start_date="2026-06-13", local_end_date="2026-06-13")

    assert out["structural_scenario_low_eur_mwh"].iloc[0] == pytest.approx(90.0)
    assert out["structural_scenario_central_eur_mwh"].iloc[0] == pytest.approx(100.0)
    assert out["structural_scenario_high_eur_mwh"].iloc[0] == pytest.approx(120.0)
    assert out["structural_scenario_spread_eur_mwh"].iloc[0] == pytest.approx(30.0)
    assert out["structural_p10_eur_mwh"].iloc[0] == pytest.approx(90.0)
    assert out["structural_p50_eur_mwh"].iloc[0] == pytest.approx(100.0)
    assert out["structural_p90_eur_mwh"].iloc[0] == pytest.approx(120.0)
    assert out["structural_width_eur_mwh"].iloc[0] == pytest.approx(30.0)
    assert out["structural_p10_eur_mwh"].iloc[0] <= out["structural_p50_eur_mwh"].iloc[0]
    assert out["structural_p50_eur_mwh"].iloc[0] <= out["structural_p90_eur_mwh"].iloc[0]


def test_calibrate_hourly_to_eex_scales_product_mean():
    hourly = pd.DataFrame(
        {
            "timestamp_ch": ["2026-07-01 00:00:00+0200", "2026-07-01 01:00:00+0200"],
            "timestamp_utc": ["2026-06-30 22:00:00+0000", "2026-06-30 23:00:00+0000"],
            "price_slow_eur_mwh": [40.0, 60.0],
            "price_central_eur_mwh": [40.0, 60.0],
            "price_fast_eur_mwh": [40.0, 60.0],
            "price_weighted_mean_eur_mwh": [40.0, 60.0],
            "structural_p10_eur_mwh": [40.0, 60.0],
            "structural_p50_eur_mwh": [40.0, 60.0],
            "structural_p90_eur_mwh": [40.0, 60.0],
            "structural_width_eur_mwh": [1.0, 1.0],
        }
    )

    calibrated, audit = calibrate_hourly_to_eex(hourly, forward_prices={"2026-07": 100.0})

    assert calibrated["price_weighted_mean_eur_mwh"].mean() == 100.0
    assert audit.loc[0, "product"] == "2026-07"
    assert audit.loc[0, "scale_factor"] == 2.0


def test_calibrate_hourly_to_eex_respects_quoted_partial_quarter_and_cal():
    timestamps = pd.date_range("2028-01-01", "2028-12-31 23:00", freq="h", tz="Europe/Zurich")
    q1_mask = timestamps.quarter == 1
    base = pd.Series(70.0, index=timestamps)
    base.loc[q1_mask] = 100.0
    hourly = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": base.to_numpy(),
            "price_central_eur_mwh": base.to_numpy(),
            "price_fast_eur_mwh": base.to_numpy(),
            "price_weighted_mean_eur_mwh": base.to_numpy(),
            "structural_p10_eur_mwh": base.to_numpy(),
            "structural_p50_eur_mwh": base.to_numpy(),
            "structural_p90_eur_mwh": base.to_numpy(),
            "structural_width_eur_mwh": [5.0] * len(timestamps),
        }
    )

    calibrated, audit = calibrate_hourly_to_eex(
        hourly,
        forward_prices={"2028": 79.98, "2028-Q1": 110.76},
    )
    parsed = pd.to_datetime(calibrated["timestamp_ch"], utc=True).dt.tz_convert("Europe/Zurich")
    out_q1 = calibrated.loc[parsed.dt.quarter == 1, "price_weighted_mean_eur_mwh"]

    assert out_q1.mean() == pytest.approx(110.76, abs=1e-6)
    assert calibrated["price_weighted_mean_eur_mwh"].mean() == pytest.approx(79.98, abs=1e-6)
    assert set(audit["product"]) == {"2028-Q1", "2028-RESIDUAL"}


def test_calibrate_hourly_to_eex_respects_quoted_month_and_quarter():
    timestamps = pd.date_range("2028-01-01", "2028-03-31 23:00", freq="h", tz="Europe/Zurich")
    hourly = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": [80.0] * len(timestamps),
            "price_central_eur_mwh": [80.0] * len(timestamps),
            "price_fast_eur_mwh": [80.0] * len(timestamps),
            "price_weighted_mean_eur_mwh": [80.0] * len(timestamps),
            "structural_p10_eur_mwh": [75.0] * len(timestamps),
            "structural_p50_eur_mwh": [80.0] * len(timestamps),
            "structural_p90_eur_mwh": [85.0] * len(timestamps),
            "structural_width_eur_mwh": [10.0] * len(timestamps),
        }
    )

    calibrated, audit = calibrate_hourly_to_eex(
        hourly,
        forward_prices={"2028-Q1": 100.0, "2028-01": 120.0},
    )
    parsed = pd.to_datetime(calibrated["timestamp_ch"], utc=True).dt.tz_convert("Europe/Zurich")

    assert calibrated.loc[parsed.dt.month == 1, "price_weighted_mean_eur_mwh"].mean() == pytest.approx(120.0)
    assert calibrated["price_weighted_mean_eur_mwh"].mean() == pytest.approx(100.0)
    assert set(audit["product"]) == {"2028-01", "2028-Q1-RESIDUAL"}


def test_calibrate_hourly_to_eex_rejects_non_positive_scale_factor():
    timestamps = pd.date_range("2028-01-01", periods=48, freq="h", tz="Europe/Zurich")
    hourly = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": [80.0] * len(timestamps),
            "price_central_eur_mwh": [80.0] * len(timestamps),
            "price_fast_eur_mwh": [80.0] * len(timestamps),
            "price_weighted_mean_eur_mwh": [80.0] * len(timestamps),
            "structural_p10_eur_mwh": [75.0] * len(timestamps),
            "structural_p50_eur_mwh": [80.0] * len(timestamps),
            "structural_p90_eur_mwh": [85.0] * len(timestamps),
            "structural_width_eur_mwh": [10.0] * len(timestamps),
        }
    )

    with pytest.raises(ValueError, match="non-positive/non-finite scale factor"):
        calibrate_hourly_to_eex(hourly, forward_prices={"2028": -10.0})


def test_latest_eex_prices_anchor_snapshot_on_latest_base(tmp_path):
    path = tmp_path / "forwards.parquet"
    pd.DataFrame(
        {
            "market": ["CH", "CH", "CH"],
            "load_type": ["BASE", "PEAK", "PEAK"],
            "date": [
                pd.Timestamp("2026-06-15"),
                pd.Timestamp("2026-06-15"),
                pd.Timestamp("2026-06-16"),
            ],
            "product": ["2030", "2030", "2030"],
            "price": [68.80, 70.38, 71.00],
        }
    ).to_parquet(path, index=False)

    latest, prices = _latest_eex_prices_by_load_type(path, market="CH")

    assert latest == pd.Timestamp("2026-06-15")
    assert prices["BASE"]["2030"] == 68.80
    assert prices["PEAK"]["2030"] == 70.38


def test_quote_aware_monthly_smoothing_flag_off_does_not_call_smoother(tmp_path, monkeypatch):
    fan_path = tmp_path / "fan.parquet"
    out_path = tmp_path / "hourly.csv"
    report_path = tmp_path / "report.md"
    forwards_path = tmp_path / "forwards.parquet"
    index = pd.date_range("2026-06-30 22:00", periods=96, freq="15min", tz="UTC")
    pd.DataFrame(
        {
            "curve_slow": [90.0] * len(index),
            "curve_central": [100.0] * len(index),
            "curve_fast": [110.0] * len(index),
            "weighted_mean": [100.0] * len(index),
            "structural_p10": [90.0] * len(index),
            "structural_p50": [100.0] * len(index),
            "structural_p90": [110.0] * len(index),
            "structural_width": [20.0] * len(index),
        },
        index=index,
    ).to_parquet(fan_path)
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-15"), pd.Timestamp("2026-06-15")],
            "product": ["2026-07", "2026-07"],
            "load_type": ["BASE", "PEAK"],
            "product_type": ["Month", "Month"],
            "price": [93.91, 88.56],
            "market": ["CH", "CH"],
            "source": ["TEST", "TEST"],
        }
    ).to_parquet(forwards_path, index=False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("quote-aware monthly smoothing must be flag-gated")

    monkeypatch.setattr(export_script, "apply_quote_aware_monthly_smoothing", fail_if_called)

    rc = export_script.main(
        [
            "--skip-build",
            "--allow-stale-forwards",
            "--local-start-date",
            "2026-07-01",
            "--local-end-date",
            "2026-07-01",
            "--fan-chart-output",
            str(fan_path),
            "--forwards",
            str(forwards_path),
            "--output",
            str(out_path),
            "--report",
            str(report_path),
            "--fan-chart-monthly-level-authority",
            "legacy",
            "--skip-powerbi-refresh",
        ]
    )

    assert rc == 0
    assert out_path.exists()
    assert "quote-aware monthly smoothing: `OFF`" in report_path.read_text(encoding="utf-8")


def test_final_eex_peak_calibration_runs_after_final_mutators(tmp_path, monkeypatch):
    fan_path = tmp_path / "fan.parquet"
    out_path = tmp_path / "hourly.csv"
    report_path = tmp_path / "report.md"
    forwards_path = tmp_path / "forwards.parquet"
    index = pd.date_range("2026-06-30 22:00", periods=96, freq="15min", tz="UTC")
    pd.DataFrame(
        {
            "curve_slow": [80.0] * len(index),
            "curve_central": [80.0] * len(index),
            "curve_fast": [80.0] * len(index),
            "weighted_mean": [80.0] * len(index),
            "structural_p10": [80.0] * len(index),
            "structural_p50": [80.0] * len(index),
            "structural_p90": [80.0] * len(index),
            "structural_width": [0.0] * len(index),
        },
        index=index,
    ).to_parquet(fan_path)
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-15"), pd.Timestamp("2026-06-15")],
            "product": ["2026-07", "2026-07"],
            "load_type": ["BASE", "PEAK"],
            "product_type": ["Month", "Month"],
            "price": [80.0, 95.0],
            "market": ["CH", "CH"],
            "source": ["TEST", "TEST"],
        }
    ).to_parquet(forwards_path, index=False)

    def damage_peak_after_initial_calibration(hourly, *, ts_ch, **kwargs):
        out = hourly.copy()
        peak = _eex_peak_mask(ts_ch, country="CH")
        price_cols = [column for column in export_script.PRICE_COLUMNS if column in out.columns]
        out.loc[peak, price_cols] = out.loc[peak, price_cols].astype(float) + 25.0
        return out, pd.DataFrame([{"step": "damaged_peak"}])

    monkeypatch.setattr(export_script, "apply_seam_nullspace_smoothing", damage_peak_after_initial_calibration)

    rc = export_script.main(
        [
            "--skip-build",
            "--allow-stale-forwards",
            "--local-start-date",
            "2026-07-01",
            "--local-end-date",
            "2026-07-01",
            "--fan-chart-output",
            str(fan_path),
            "--forwards",
            str(forwards_path),
            "--output",
            str(out_path),
            "--report",
            str(report_path),
            "--enable-eex-peak-calibration",
            "--enable-final-seam-nullspace-smoothing",
            "--fan-chart-monthly-level-authority",
            "legacy",
            "--skip-powerbi-refresh",
        ]
    )

    assert rc == 0
    out = pd.read_csv(out_path)
    ts_ch = export_script._parse_timestamp_ch(out["timestamp_ch"], out.get("utc_offset_ch"))
    peak = _eex_peak_mask(ts_ch, country="CH")
    assert out["price_weighted_mean_eur_mwh"].mean() == pytest.approx(80.0, abs=1e-6)
    assert out.loc[peak, "price_weighted_mean_eur_mwh"].mean() == pytest.approx(95.0, abs=1e-6)


def test_skip_build_monthly_solver_fan_blocks_mutating_postprocessor(tmp_path):
    fan_path = tmp_path / "fan.parquet"
    out_path = tmp_path / "hourly.csv"
    report_path = tmp_path / "report.md"
    forwards_path = tmp_path / "forwards.parquet"
    index = pd.date_range("2026-06-30 22:00", periods=31 * 96, freq="15min", tz="UTC")
    pd.DataFrame(
        {
            "curve_slow": [80.0] * len(index),
            "curve_central": [80.0] * len(index),
            "curve_fast": [80.0] * len(index),
            "weighted_mean": [80.0] * len(index),
        },
        index=index,
    ).to_parquet(fan_path)
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-15")],
            "product": ["2026-07"],
            "load_type": ["BASE"],
            "product_type": ["Month"],
            "price": [80.0],
            "market": ["CH"],
            "source": ["TEST"],
        }
    ).to_parquet(forwards_path, index=False)

    with pytest.raises(ValueError, match="monthly_level_authority=solver"):
        export_script.main(
            [
                "--skip-build",
                "--allow-stale-forwards",
                "--local-start-date",
                "2026-07-01",
                "--local-end-date",
                "2026-07-31",
                "--fan-chart-output",
                str(fan_path),
                "--forwards",
                str(forwards_path),
                "--output",
                str(out_path),
                "--report",
                str(report_path),
                "--enable-final-seam-nullspace-smoothing",
                "--fan-chart-monthly-level-authority",
                "solver",
                "--skip-powerbi-refresh",
            ]
        )


def test_skip_build_rejects_solver_manifest_when_fan_violates_eex_constraints(tmp_path):
    fan_path = tmp_path / "fan.parquet"
    out_path = tmp_path / "hourly.csv"
    report_path = tmp_path / "report.md"
    forwards_path = tmp_path / "forwards.parquet"
    index = pd.date_range("2026-06-30 22:00", periods=31 * 96, freq="15min", tz="UTC")
    fan = pd.DataFrame(
        {
            "curve_slow": [80.0] * len(index),
            "curve_central": [80.0] * len(index),
            "curve_fast": [80.0] * len(index),
            "weighted_mean": [80.0] * len(index),
        },
        index=index,
    )
    fan.to_parquet(fan_path)
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-15")],
            "product": ["2026-07"],
            "load_type": ["BASE"],
            "product_type": ["Month"],
            "price": [100.0],
            "market": ["CH"],
            "source": ["TEST"],
        }
    ).to_parquet(forwards_path, index=False)
    fan_path.with_suffix(".monthly_curve_manifest.json").write_text(
        json.dumps(_solver_manifest(fan=fan, fan_path=fan_path, forwards_path=forwards_path)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="violates EEX BASE constraint"):
        export_script.main(
            [
                "--skip-build",
                "--allow-stale-forwards",
                "--local-start-date",
                "2026-07-01",
                "--local-end-date",
                "2026-07-31",
                "--fan-chart-output",
                str(fan_path),
                "--forwards",
                str(forwards_path),
                "--output",
                str(out_path),
                "--report",
                str(report_path),
                "--skip-powerbi-refresh",
            ]
        )


def test_skip_build_reads_solver_authority_from_bound_fan_manifest(tmp_path):
    fan_path = tmp_path / "fan.parquet"
    out_path = tmp_path / "hourly.csv"
    report_path = tmp_path / "report.md"
    forwards_path = tmp_path / "forwards.parquet"
    index = pd.date_range("2026-06-30 22:00", periods=31 * 96, freq="15min", tz="UTC")
    fan = pd.DataFrame(
        {
            "curve_slow": [80.0] * len(index),
            "curve_central": [100.0] * len(index),
            "curve_fast": [120.0] * len(index),
            "weighted_mean": [100.0] * len(index),
        },
        index=index,
    )
    fan.to_parquet(fan_path)
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-15")],
            "product": ["2026-07"],
            "load_type": ["BASE"],
            "product_type": ["Month"],
            "price": [100.0],
            "market": ["CH"],
            "source": ["TEST"],
        }
    ).to_parquet(forwards_path, index=False)
    fan_path.with_suffix(".monthly_curve_manifest.json").write_text(
        json.dumps(_solver_manifest(fan=fan, fan_path=fan_path, forwards_path=forwards_path)),
        encoding="utf-8",
    )

    rc = export_script.main(
        [
            "--skip-build",
            "--allow-stale-forwards",
            "--local-start-date",
            "2026-07-01",
            "--local-end-date",
            "2026-07-31",
            "--fan-chart-output",
            str(fan_path),
            "--forwards",
            str(forwards_path),
            "--output",
            str(out_path),
            "--report",
            str(report_path),
            "--weights",
            "0.0,0.0,1.0",
            "--skip-powerbi-refresh",
        ]
    )

    assert rc == 0
    out = pd.read_csv(out_path)
    assert out["price_weighted_mean_eur_mwh"].mean() == pytest.approx(100.0)
    assert out["structural_p90_eur_mwh"].mean() == pytest.approx(120.0)
    report = report_path.read_text(encoding="utf-8")
    assert "SKIPPED_SOLVER_AUTHORITY" in report
    assert "monthly level authority: `solver`" in report


def test_solver_fan_base_constraints_check_overlapping_complete_parent_quotes() -> None:
    index = pd.date_range("2025-12-31 23:00", periods=365 * 96, freq="15min", tz="UTC")
    fan = pd.DataFrame({"weighted_mean": [80.0] * len(index)}, index=index)
    prices = {f"2026-{month:02d}": 80.0 for month in range(1, 13)}
    prices["2026"] = 100.0

    with pytest.raises(ValueError, match="product=2026"):
        export_script._validate_solver_fan_base_constraints(
            fan,
            base_forward_prices=prices,
            timezone=export_script.LOCAL_TZ,
        )


def test_solver_fan_base_constraints_reject_partial_windows_without_full_quote() -> None:
    index = pd.date_range("2026-06-30 22:00", periods=96, freq="15min", tz="UTC")
    fan = pd.DataFrame({"weighted_mean": [100.0] * len(index)}, index=index)

    with pytest.raises(ValueError, match="no fully covered EEX BASE constraints"):
        export_script._validate_solver_fan_base_constraints(
            fan,
            base_forward_prices={"2026-07": 100.0},
            timezone=export_script.LOCAL_TZ,
        )


def test_skip_build_rejects_minimal_solver_authority_manifest(tmp_path):
    fan_path = tmp_path / "fan.parquet"
    out_path = tmp_path / "hourly.csv"
    report_path = tmp_path / "report.md"
    forwards_path = tmp_path / "forwards.parquet"
    index = pd.date_range("2026-06-30 22:00", periods=96, freq="15min", tz="UTC")
    pd.DataFrame(
        {
            "curve_slow": [80.0] * len(index),
            "curve_central": [80.0] * len(index),
            "curve_fast": [80.0] * len(index),
            "weighted_mean": [80.0] * len(index),
        },
        index=index,
    ).to_parquet(fan_path)
    fan_path.with_suffix(".monthly_curve_manifest.json").write_text(
        json.dumps({"monthly_level_authority": "solver"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-15")],
            "product": ["2026-07"],
            "load_type": ["BASE"],
            "product_type": ["Month"],
            "price": [100.0],
            "market": ["CH"],
            "source": ["TEST"],
        }
    ).to_parquet(forwards_path, index=False)

    with pytest.raises(ValueError, match="missing required fields"):
        export_script.main(
            [
                "--skip-build",
                "--allow-stale-forwards",
                "--local-start-date",
                "2026-07-01",
                "--local-end-date",
                "2026-07-01",
                "--fan-chart-output",
                str(fan_path),
                "--forwards",
                str(forwards_path),
                "--output",
                str(out_path),
                "--report",
                str(report_path),
                "--skip-powerbi-refresh",
            ]
        )


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda manifest: manifest.update({"solver_config": {}}), "solver_config"),
        (lambda manifest: manifest.update({"source_hashes": {}}), "source_hashes"),
        (lambda manifest: manifest.update({"solver_kkt": {}}), "solver_kkt"),
        (lambda manifest: manifest.update({"active_config_hash": "not-the-config-hash"}), "active_config_hash"),
        (lambda manifest: manifest.update({"solver_config_hash": ""}), "solver_config_hash"),
        (lambda manifest: manifest.update({"fan_chart_sha256": "not-the-fan-hash"}), "fan_chart_sha256"),
        (lambda manifest: manifest["source_hashes"].update({"forwards_path": "not-the-forwards-hash"}), "forwards_path"),
        (
            lambda manifest: manifest.update({"fan_chart_weighted_monthly_hash": "not-the-solution-hash"}),
            "fan_chart_weighted_monthly_hash",
        ),
        (
            lambda manifest: manifest["solver_kkt"].update({"max_abs_constraint_residual": float("nan")}),
            "non-finite",
        ),
        (
            lambda manifest: manifest["solver_kkt"].update({"stationarity_residual": 1.0}),
            "stationarity_residual",
        ),
        (
            lambda manifest: manifest["solver_kkt"].update({"condition_number": 0.0}),
            "condition_number",
        ),
        (
            lambda manifest: manifest["solver_kkt"].update({"condition_number": 1e13}),
            "condition_number",
        ),
        (
            lambda manifest: manifest["solver_kkt"].update({"active_constraint_rank": -1}),
            "non-negative integer",
        ),
        (lambda manifest: manifest["solver_kkt"].update({"ridge_used": ""}), "boolean"),
        (lambda manifest: manifest["solver_kkt"].update({"solved_by_lstsq": ""}), "boolean"),
        (lambda manifest: manifest["solver_kkt"].update({"ridge_used": True}), "must be false"),
        (lambda manifest: manifest["solver_kkt"].update({"solved_by_lstsq": True}), "must be false"),
    ],
)
def test_solver_authority_manifest_rejects_empty_or_nonfinite_proof(tmp_path, mutate, match):
    fan_path = tmp_path / "fan.parquet"
    forwards_path = tmp_path / "forwards.parquet"
    index = pd.date_range("2026-06-30 22:00", periods=31 * 96, freq="15min", tz="UTC")
    fan = pd.DataFrame(
        {
            "curve_slow": [80.0] * len(index),
            "curve_central": [100.0] * len(index),
            "curve_fast": [120.0] * len(index),
            "weighted_mean": [100.0] * len(index),
        },
        index=index,
    )
    fan.to_parquet(fan_path)
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-15")],
            "product": ["2026-07"],
            "load_type": ["BASE"],
            "product_type": ["Month"],
            "price": [100.0],
            "market": ["CH"],
            "source": ["TEST"],
        }
    ).to_parquet(forwards_path, index=False)
    manifest = _solver_manifest(fan=fan, fan_path=fan_path, forwards_path=forwards_path)
    mutate(manifest)

    with pytest.raises(ValueError, match=match):
        export_script._validate_solver_authority_manifest(
            manifest,
            manifest_path=Path("fan.monthly_curve_manifest.json"),
            fan_output=fan_path,
            fan=fan,
            forwards_path=forwards_path,
            base_forward_prices={"2026-07": 100.0},
            latest_forward_date=pd.Timestamp("2026-06-15"),
            market="CH",
        )


def test_skip_build_without_manifest_requires_explicit_legacy_authority(tmp_path):
    fan_path = tmp_path / "fan.parquet"
    out_path = tmp_path / "hourly.csv"
    report_path = tmp_path / "report.md"
    forwards_path = tmp_path / "forwards.parquet"
    index = pd.date_range("2026-06-30 22:00", periods=96, freq="15min", tz="UTC")
    pd.DataFrame(
        {
            "curve_slow": [80.0] * len(index),
            "curve_central": [80.0] * len(index),
            "curve_fast": [80.0] * len(index),
            "weighted_mean": [80.0] * len(index),
        },
        index=index,
    ).to_parquet(fan_path)

    with pytest.raises(ValueError, match="requires a fan chart monthly authority manifest"):
        export_script.main(
            [
                "--skip-build",
                "--allow-stale-forwards",
                "--local-start-date",
                "2026-07-01",
                "--local-end-date",
                "2026-07-01",
                "--fan-chart-output",
                str(fan_path),
                "--forwards",
                str(forwards_path),
                "--output",
                str(out_path),
                "--report",
                str(report_path),
                "--skip-powerbi-refresh",
            ]
        )


@pytest.mark.parametrize(
    "flag",
    [
        "--enable-structural-shape-upgrade",
        "--enable-post-calibration-negative-rebalancer",
        "--enable-post-calibration-peak-shape-rebalancer",
        "--enable-eex-peak-calibration",
    ],
)
def test_fresh_monthly_solver_blocks_level_mutating_postprocessors(tmp_path, flag):
    with pytest.raises(ValueError, match="mutating local-test post-processors"):
        export_script.main(
            [
                "--allow-stale-forwards",
                "--local-start-date",
                "2026-07-01",
                "--local-end-date",
                "2026-07-01",
                "--fan-chart-output",
                str(tmp_path / "fan.parquet"),
                "--forwards",
                str(tmp_path / "forwards.parquet"),
                "--output",
                str(tmp_path / "hourly.csv"),
                "--report",
                str(tmp_path / "report.md"),
                "--enable-monthly-forward-curve-solver",
                flag,
                "--skip-powerbi-refresh",
            ]
        )


def test_fresh_monthly_solver_build_skips_export_recalibration(tmp_path, monkeypatch):
    fan_path = tmp_path / "fan.parquet"
    out_path = tmp_path / "hourly.csv"
    report_path = tmp_path / "report.md"
    forwards_path = tmp_path / "forwards.parquet"
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-15")],
            "product": ["2026-07"],
            "load_type": ["BASE"],
            "product_type": ["Month"],
            "price": [100.0],
            "market": ["CH"],
            "source": ["TEST"],
        }
    ).to_parquet(forwards_path, index=False)

    def fake_build(argv):
        assert argv[argv.index("--weights") + 1] == "0.0,0.0,1.0"
        fan_output = Path(argv[argv.index("--fan-chart-output") + 1])
        index = pd.date_range("2026-06-30 22:00", periods=31 * 96, freq="15min", tz="UTC")
        fan = pd.DataFrame(
            {
                "curve_slow": [80.0] * len(index),
                "curve_central": [100.0] * len(index),
                "curve_fast": [120.0] * len(index),
                "weighted_mean": [100.0] * len(index),
            },
            index=index,
        )
        fan.to_parquet(fan_output)
        fan_output.with_suffix(".monthly_curve_manifest.json").write_text(
            json.dumps(_solver_manifest(fan=fan, fan_path=fan_output, forwards_path=forwards_path)),
            encoding="utf-8",
        )

    monkeypatch.setattr(export_script, "build_local_test_ch_pfc", fake_build)

    rc = export_script.main(
        [
            "--allow-stale-forwards",
            "--local-start-date",
            "2026-07-01",
            "--local-end-date",
            "2026-07-31",
            "--fan-chart-output",
            str(fan_path),
            "--forwards",
            str(forwards_path),
            "--output",
            str(out_path),
            "--report",
            str(report_path),
            "--enable-monthly-forward-curve-solver",
            "--weights",
            "0.0,0.0,1.0",
            "--skip-powerbi-refresh",
        ]
    )

    assert rc == 0
    out = pd.read_csv(out_path)
    assert out["price_weighted_mean_eur_mwh"].mean() == pytest.approx(100.0)
    assert "SKIPPED_SOLVER_AUTHORITY" in report_path.read_text(encoding="utf-8")


def test_skip_build_rejects_authority_declaration_conflicting_with_manifest(tmp_path):
    fan_path = tmp_path / "fan.parquet"
    out_path = tmp_path / "hourly.csv"
    report_path = tmp_path / "report.md"
    forwards_path = tmp_path / "forwards.parquet"
    index = pd.date_range("2026-06-30 22:00", periods=96, freq="15min", tz="UTC")
    fan = pd.DataFrame(
        {
            "curve_slow": [80.0] * len(index),
            "curve_central": [80.0] * len(index),
            "curve_fast": [80.0] * len(index),
            "weighted_mean": [80.0] * len(index),
        },
        index=index,
    )
    fan.to_parquet(fan_path)
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-06-15")],
            "product": ["2026-07"],
            "load_type": ["BASE"],
            "product_type": ["Month"],
            "price": [80.0],
            "market": ["CH"],
            "source": ["TEST"],
        }
    ).to_parquet(forwards_path, index=False)
    fan_path.with_suffix(".monthly_curve_manifest.json").write_text(
        json.dumps(_solver_manifest(fan=fan, fan_path=fan_path, forwards_path=forwards_path)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicts with manifest"):
        export_script.main(
            [
                "--skip-build",
                "--allow-stale-forwards",
                "--local-start-date",
                "2026-07-01",
                "--local-end-date",
                "2026-07-01",
                "--fan-chart-output",
                str(fan_path),
                "--forwards",
                str(forwards_path),
                "--output",
                str(out_path),
                "--report",
                str(report_path),
                "--fan-chart-monthly-level-authority",
                "legacy",
                "--skip-powerbi-refresh",
            ]
        )


def test_eex_peak_mask_excludes_ch_national_holidays():
    timestamps = pd.Series(
        pd.DatetimeIndex(
            [
                "2030-08-01 10:00",  # Swiss National Day, Thursday
                "2030-08-02 10:00",  # regular Friday
                "2030-08-03 10:00",  # Saturday
                "2030-08-02 20:00",  # outside EEX peak window
            ],
            tz="Europe/Zurich",
        )
    )

    mask = _eex_peak_mask(timestamps)

    assert mask.tolist() == [False, True, False, False]


def test_eex_peak_mask_supports_neighbor_holiday_calendars():
    timestamps = pd.Series(
        pd.DatetimeIndex(
            [
                "2028-07-14 10:00",  # FR national holiday, regular Friday elsewhere
                "2028-07-15 10:00",
            ],
            tz="Europe/Zurich",
        )
    )

    fr_mask = _eex_peak_mask(timestamps, country="FR")
    at_mask = _eex_peak_mask(timestamps, country="AT")

    assert fr_mask.tolist() == [False, False]
    assert at_mask.tolist() == [True, False]


def test_eex_peak_mask_rejects_unsupported_country():
    timestamps = pd.Series(pd.date_range("2028-07-14 10:00", periods=1, freq="h", tz="Europe/Zurich"))

    with pytest.raises(ValueError, match="unsupported EEX peak holiday country"):
        _eex_peak_mask(timestamps, country="XX")


def test_synthetic_monthly_path_smoothing_reduces_jump_and_preserves_bucket_mean():
    timestamps = []
    prices = []
    for month, price in [(1, 100.0), (2, 102.0), (3, 70.0)]:
        for day in [1, 2]:
            for hour in range(24):
                timestamps.append(pd.Timestamp(year=2030, month=month, day=day, hour=hour, tz="Europe/Zurich"))
                prices.append(price)
    ts = pd.Series(pd.DatetimeIndex(timestamps))
    hourly = pd.DataFrame(
        {
            "timestamp_ch": ts.dt.strftime("%d.%m.%Y %H:%M"),
            "utc_offset_ch": "UTC+01:00",
            "timestamp_utc": ts.dt.tz_convert("UTC").dt.strftime("%d.%m.%Y %H:%M"),
            "price_slow_eur_mwh": prices,
            "price_central_eur_mwh": prices,
            "price_fast_eur_mwh": prices,
            "price_weighted_mean_eur_mwh": prices,
            "structural_p10_eur_mwh": prices,
            "structural_p50_eur_mwh": prices,
            "structural_p90_eur_mwh": prices,
            "structural_width_eur_mwh": [0.0] * len(prices),
        }
    )
    before_mean = float(hourly["price_weighted_mean_eur_mwh"].mean())

    smoothed, audit = apply_synthetic_monthly_path_smoothing(
        hourly,
        ts_ch=ts,
        base_forward_prices={"2030": before_mean},
        peak_forward_prices={},
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        intensity=1.0,
        smoothness_lambda=20.0,
        max_month_delta_eur_mwh=20.0,
        negative_price_floor=-30.0,
    )

    after_monthly = smoothed.assign(month=ts.dt.month).groupby("month")["price_weighted_mean_eur_mwh"].mean()
    assert audit["product"].unique().tolist() == ["2030"]
    assert smoothed["price_weighted_mean_eur_mwh"].mean() == pytest.approx(before_mean, abs=1e-6)
    assert abs(after_monthly.loc[3] - after_monthly.loc[2]) < 32.0


def test_calibrate_hourly_to_eex_base_peak_matches_base_and_peak_quotes():
    timestamps = pd.date_range("2030-07-01", periods=24 * 14, freq="h", tz="Europe/Zurich")
    hourly = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": [80.0] * len(timestamps),
            "price_central_eur_mwh": [80.0] * len(timestamps),
            "price_fast_eur_mwh": [80.0] * len(timestamps),
            "price_weighted_mean_eur_mwh": [80.0] * len(timestamps),
            "structural_p10_eur_mwh": [80.0] * len(timestamps),
            "structural_p50_eur_mwh": [80.0] * len(timestamps),
            "structural_p90_eur_mwh": [80.0] * len(timestamps),
            "structural_width_eur_mwh": [0.0] * len(timestamps),
        }
    )
    peak = _eex_peak_mask(pd.Series(timestamps))

    calibrated, audit = calibrate_hourly_to_eex_base_peak(
        hourly,
        base_forward_prices={"2030": 80.0},
        peak_forward_prices={"2030": 90.0},
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        negative_price_floor=-30.0,
    )

    assert calibrated["price_weighted_mean_eur_mwh"].mean() == pytest.approx(80.0, abs=1e-6)
    assert calibrated.loc[peak, "price_weighted_mean_eur_mwh"].mean() == pytest.approx(90.0, abs=1e-6)
    assert calibrated.loc[~peak, "price_weighted_mean_eur_mwh"].mean() < 80.0
    assert calibrated["structural_width_eur_mwh"].max() == pytest.approx(0.0, abs=1e-9)
    assert audit.loc[0, "base_residual_eur_mwh"] == pytest.approx(0.0, abs=1e-9)
    assert audit.loc[0, "peak_residual_eur_mwh"] == pytest.approx(0.0, abs=1e-9)


def test_eex_peak_calibration_after_peak_rebalancer_forces_contract_quote():
    timestamps = pd.date_range("2030-07-01", periods=24 * 14, freq="h", tz="Europe/Zurich")
    hourly = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": [80.0] * len(timestamps),
            "price_central_eur_mwh": [80.0] * len(timestamps),
            "price_fast_eur_mwh": [80.0] * len(timestamps),
            "price_weighted_mean_eur_mwh": [80.0] * len(timestamps),
            "structural_p10_eur_mwh": [80.0] * len(timestamps),
            "structural_p50_eur_mwh": [80.0] * len(timestamps),
            "structural_p90_eur_mwh": [80.0] * len(timestamps),
            "structural_width_eur_mwh": [0.0] * len(timestamps),
        }
    )
    weights = {"slow": 0.25, "central": 0.5, "fast": 0.25}
    rebalanced, _ = apply_post_calibration_peak_shape_rebalancer(
        hourly,
        forward_prices={"2030": 80.0},
        weights=weights,
        intensity=1.0,
        negative_price_floor=-30.0,
        max_weighted_negative_hours=0,
    )
    peak = _eex_peak_mask(pd.Series(timestamps))

    calibrated, _ = calibrate_hourly_to_eex_base_peak(
        rebalanced,
        base_forward_prices={"2030": 80.0},
        peak_forward_prices={"2030": 88.0},
        weights=weights,
        negative_price_floor=-30.0,
    )

    assert calibrated["price_weighted_mean_eur_mwh"].mean() == pytest.approx(80.0, abs=1e-6)
    assert calibrated.loc[peak, "price_weighted_mean_eur_mwh"].mean() == pytest.approx(88.0, abs=1e-6)


def test_previous_business_day_skips_weekends():
    assert previous_business_day("2026-06-12") == pd.Timestamp("2026-06-11")
    assert previous_business_day("2026-06-15") == pd.Timestamp("2026-06-12")


def test_require_forward_date_accepts_previous_business_day():
    required = require_forward_date(
        pd.Timestamp("2026-06-11"),
        required_forward_date=None,
        valuation_date="2026-06-12",
    )

    assert required == pd.Timestamp("2026-06-11")


def test_require_forward_date_rejects_stale_snapshot():
    with pytest.raises(ValueError, match="stale EEX CH BASE forwards"):
        require_forward_date(
            pd.Timestamp("2026-06-05"),
            required_forward_date=None,
            valuation_date="2026-06-12",
        )


def test_structural_shape_upgrade_preserves_product_means_and_widens_fan():
    timestamps = pd.date_range("2026-07-01", periods=48, freq="h", tz="Europe/Zurich")
    hourly = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": [80.0] * len(timestamps),
            "price_central_eur_mwh": [80.0] * len(timestamps),
            "price_fast_eur_mwh": [80.0] * len(timestamps),
            "price_weighted_mean_eur_mwh": [80.0] * len(timestamps),
            "structural_p10_eur_mwh": [80.0] * len(timestamps),
            "structural_p50_eur_mwh": [80.0] * len(timestamps),
            "structural_p90_eur_mwh": [80.0] * len(timestamps),
            "structural_width_eur_mwh": [0.0] * len(timestamps),
        }
    )

    upgraded, audit = apply_local_test_structural_shape_upgrade(
        hourly,
        forward_prices={"2026-07": 80.0},
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        intensity=1.0,
    )

    for column in ["price_slow_eur_mwh", "price_central_eur_mwh", "price_fast_eur_mwh"]:
        assert upgraded[column].mean() == pytest.approx(80.0)
    assert upgraded["structural_width_eur_mwh"].max() > 0.0
    assert set(audit["scenario"]) == {"slow", "central", "fast"}


def test_negative_price_capture_is_explicit_and_mean_preserving():
    timestamps = pd.date_range("2030-07-01", periods=24 * 14, freq="h", tz="Europe/Zurich")
    hourly = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": [8.0] * len(timestamps),
            "price_central_eur_mwh": [8.0] * len(timestamps),
            "price_fast_eur_mwh": [8.0] * len(timestamps),
            "price_weighted_mean_eur_mwh": [8.0] * len(timestamps),
            "structural_p10_eur_mwh": [8.0] * len(timestamps),
            "structural_p50_eur_mwh": [8.0] * len(timestamps),
            "structural_p90_eur_mwh": [8.0] * len(timestamps),
            "structural_width_eur_mwh": [0.0] * len(timestamps),
        }
    )

    upgraded, audit = apply_local_test_structural_shape_upgrade(
        hourly,
        forward_prices={"2030": 8.0},
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        intensity=1.0,
        spread_intensity=2.0,
        enable_negative_price_capture=True,
        negative_capture_intensity=2.0,
        negative_price_floor=-30.0,
    )

    assert upgraded["price_fast_eur_mwh"].min() < 0.0
    assert upgraded["structural_p10_eur_mwh"].min() < 0.0
    for column in ["price_slow_eur_mwh", "price_central_eur_mwh", "price_fast_eur_mwh"]:
        assert upgraded[column].mean() == pytest.approx(8.0)
    assert int(audit["negative_hours"].sum()) > 0


def test_post_calibration_negative_rebalancer_preserves_eex_bucket_means():
    timestamps = pd.date_range("2030-07-01", periods=24 * 14, freq="h", tz="Europe/Zurich")
    hourly = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": [16.0] * len(timestamps),
            "price_central_eur_mwh": [8.0] * len(timestamps),
            "price_fast_eur_mwh": [5.0] * len(timestamps),
            "price_weighted_mean_eur_mwh": [9.25] * len(timestamps),
            "structural_p10_eur_mwh": [5.0] * len(timestamps),
            "structural_p50_eur_mwh": [8.0] * len(timestamps),
            "structural_p90_eur_mwh": [16.0] * len(timestamps),
            "structural_width_eur_mwh": [11.0] * len(timestamps),
        }
    )

    rebalanced, audit = apply_post_calibration_negative_rebalancer(
        hourly,
        forward_prices={"2030": 9.25},
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        intensity=1.0,
        negative_price_floor=-30.0,
        max_weighted_negative_hours=0,
    )

    assert rebalanced["price_slow_eur_mwh"].mean() == pytest.approx(16.0)
    assert rebalanced["price_central_eur_mwh"].mean() == pytest.approx(8.0)
    assert rebalanced["price_fast_eur_mwh"].mean() == pytest.approx(5.0)
    assert rebalanced["price_weighted_mean_eur_mwh"].mean() == pytest.approx(9.25)
    assert rebalanced["price_fast_eur_mwh"].min() < 0.0
    assert rebalanced["structural_p10_eur_mwh"].min() < 0.0
    assert int((rebalanced["price_weighted_mean_eur_mwh"] < 0.0).sum()) == 0
    assert int(audit["negative_hours"].sum()) > 0


def test_weighted_negative_capture_is_opt_in_and_mean_preserving():
    timestamps = pd.date_range("2030-07-01", periods=24 * 14, freq="h", tz="Europe/Zurich")
    hourly = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": [10.0] * len(timestamps),
            "price_central_eur_mwh": [2.0] * len(timestamps),
            "price_fast_eur_mwh": [1.0] * len(timestamps),
            "price_weighted_mean_eur_mwh": [3.75] * len(timestamps),
            "structural_p10_eur_mwh": [1.0] * len(timestamps),
            "structural_p50_eur_mwh": [2.0] * len(timestamps),
            "structural_p90_eur_mwh": [10.0] * len(timestamps),
            "structural_width_eur_mwh": [9.0] * len(timestamps),
        }
    )

    rebalanced, audit = apply_post_calibration_negative_rebalancer(
        hourly,
        forward_prices={"2030": 3.75},
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        intensity=1.0,
        weighted_negative_capture_intensity=1.0,
        negative_price_floor=-30.0,
        max_weighted_negative_hours=999,
    )

    assert rebalanced["price_slow_eur_mwh"].mean() == pytest.approx(10.0)
    assert rebalanced["price_central_eur_mwh"].mean() == pytest.approx(2.0)
    assert rebalanced["price_fast_eur_mwh"].mean() == pytest.approx(1.0)
    assert rebalanced["price_weighted_mean_eur_mwh"].mean() == pytest.approx(3.75)
    assert rebalanced["price_central_eur_mwh"].min() < 0.0
    assert rebalanced["price_weighted_mean_eur_mwh"].min() < 0.0
    assert int((rebalanced["price_weighted_mean_eur_mwh"] < 0.0).sum()) > 0
    assert {"central", "fast"}.issubset(set(audit["scenario"]))


def test_neighbor_monthly_spread_anchor_follows_guide_and_preserves_buckets():
    timestamps = pd.date_range("2027-01-01", "2027-03-31 23:00", freq="h", tz="Europe/Zurich")
    hourly = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": [120.0] * len(timestamps),
            "price_central_eur_mwh": [120.0] * len(timestamps),
            "price_fast_eur_mwh": [120.0] * len(timestamps),
            "price_weighted_mean_eur_mwh": [120.0] * len(timestamps),
            "structural_p10_eur_mwh": [120.0] * len(timestamps),
            "structural_p50_eur_mwh": [120.0] * len(timestamps),
            "structural_p90_eur_mwh": [120.0] * len(timestamps),
            "structural_width_eur_mwh": [0.0] * len(timestamps),
        }
    )
    ts_ch = pd.Series(timestamps, index=hourly.index)
    peak_mask = _eex_peak_mask(ts_ch, country="CH")
    peak_idx = peak_mask[peak_mask].index
    for col in [
        "price_slow_eur_mwh",
        "price_central_eur_mwh",
        "price_fast_eur_mwh",
        "price_weighted_mean_eur_mwh",
        "structural_p10_eur_mwh",
        "structural_p50_eur_mwh",
        "structural_p90_eur_mwh",
    ]:
        hourly.loc[peak_idx, col] = 125.0

    anchored, audit = apply_neighbor_monthly_spread_anchor(
        hourly,
        ts_ch=ts_ch,
        base_forward_prices={"2027-Q1": 120.0},
        peak_forward_prices={"2027-Q1": 125.0},
        neighbor_base_forward_prices={
            "2027-Q1": 100.0,
            "2027-01": 112.0,
            "2027-02": 100.0,
            "2027-03": 88.0,
        },
        neighbor_peak_forward_prices={
            "2027-Q1": 105.0,
            "2027-01": 128.0,
            "2027-02": 108.0,
            "2027-03": 79.0,
        },
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        max_weighted_negative_hours=0,
    )

    anchored_ts = pd.Series(timestamps, index=anchored.index)
    anchored_peak = _eex_peak_mask(anchored_ts, country="CH")
    month = anchored_ts.dt.month
    monthly_base = anchored.groupby(month)["price_weighted_mean_eur_mwh"].mean()
    monthly_peak = anchored.loc[anchored_peak].groupby(month.loc[anchored_peak])[
        "price_weighted_mean_eur_mwh"
    ].mean()

    assert monthly_base.loc[1] > monthly_base.loc[2] > monthly_base.loc[3]
    assert monthly_peak.loc[1] > monthly_peak.loc[2] > monthly_peak.loc[3]
    assert anchored["price_weighted_mean_eur_mwh"].mean() == pytest.approx(120.0, abs=1e-6)
    assert anchored.loc[anchored_peak, "price_weighted_mean_eur_mwh"].mean() == pytest.approx(125.0, abs=1e-6)
    assert set(audit["load_type"]) == {"BASE", "PEAK"}
    assert set(audit["source"]) == {"neighbor_current"}


def test_neighbor_annual_residual_shape_anchor_recenters_neighbor_shape_without_level_leakage():
    timestamps = pd.date_range("2030-01-01", "2030-12-31 23:00", freq="h", tz="Europe/Zurich")
    month = pd.Series(timestamps.month)
    residual_mask = month >= 4
    q1_target = 100.0
    residual_target = 70.0
    q1_hours = int((month <= 3).sum())
    residual_hours = int(residual_mask.sum())
    annual_target = (q1_target * q1_hours + residual_target * residual_hours) / len(timestamps)

    initial = pd.Series(90.0, index=range(len(timestamps)))
    initial.loc[month <= 3] = q1_target
    initial.loc[(month >= 4) & (month <= 6)] = 50.0
    initial.loc[month >= 7] = 85.0
    hourly = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": initial.to_numpy(),
            "price_central_eur_mwh": initial.to_numpy(),
            "price_fast_eur_mwh": initial.to_numpy(),
            "price_weighted_mean_eur_mwh": initial.to_numpy(),
            "structural_p10_eur_mwh": initial.to_numpy(),
            "structural_p50_eur_mwh": initial.to_numpy(),
            "structural_p90_eur_mwh": initial.to_numpy(),
            "structural_width_eur_mwh": [0.0] * len(timestamps),
        }
    )
    neighbor_months = {
        f"2030-{m:02d}": price
        for m, price in {
            4: 60.0,
            5: 64.0,
            6: 67.0,
            7: 69.0,
            8: 72.0,
            9: 74.0,
            10: 78.0,
            11: 83.0,
            12: 86.0,
        }.items()
    }

    anchored, audit = apply_neighbor_annual_residual_shape_anchor(
        hourly,
        ts_ch=pd.Series(timestamps, index=hourly.index),
        base_forward_prices={"2030": annual_target, "2030-Q1": q1_target},
        neighbor_base_forward_prices=neighbor_months,
        historical_base_deviations={},
        neighbor_current_weight=1.0,
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        intensity=1.0,
        max_month_delta_eur_mwh=100.0,
        negative_price_floor=-30.0,
        max_weighted_negative_hours=0,
    )

    anchored_month = pd.Series(timestamps.month, index=anchored.index)
    monthly = anchored.groupby(anchored_month)["price_weighted_mean_eur_mwh"].mean()
    residual_mean = anchored.loc[anchored_month >= 4, "price_weighted_mean_eur_mwh"].mean()
    raw_neighbor = pd.Series({int(key[-2:]): value for key, value in neighbor_months.items()})
    month_counts = anchored_month[anchored_month >= 4].value_counts().sort_index()
    raw_neighbor_mean = (raw_neighbor * month_counts).sum() / month_counts.sum()
    expected_targets = raw_neighbor - raw_neighbor_mean + residual_target

    assert set(audit["product"]) == {"2030-RESIDUAL"}
    assert set(audit["source"]) == {"neighbor_month"}
    assert monthly.loc[1] == pytest.approx(q1_target, abs=1e-9)
    assert monthly.loc[2] == pytest.approx(q1_target, abs=1e-9)
    assert monthly.loc[3] == pytest.approx(q1_target, abs=1e-9)
    assert residual_mean == pytest.approx(residual_target, abs=1e-6)
    assert monthly.loc[4] == pytest.approx(expected_targets.loc[4], abs=1e-6)
    assert monthly.loc[12] == pytest.approx(expected_targets.loc[12], abs=1e-6)
    assert (monthly.loc[4] - raw_neighbor.loc[4]) == pytest.approx(
        monthly.loc[12] - raw_neighbor.loc[12],
        abs=1e-6,
    )


def test_final_quant_annual_smoothness_reduces_annual_only_month_cliff():
    timestamps = pd.date_range("2030-01-01", "2030-12-31 23:00", freq="h", tz="Europe/Zurich")
    month = pd.Series(timestamps.month, index=range(len(timestamps)))
    initial = pd.Series(90.0, index=range(len(timestamps)))
    initial.loc[month == 2] = 40.0
    target = float(initial.mean())
    hourly = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": initial.to_numpy(),
            "price_central_eur_mwh": initial.to_numpy(),
            "price_fast_eur_mwh": initial.to_numpy(),
            "price_weighted_mean_eur_mwh": initial.to_numpy(),
            "structural_p10_eur_mwh": initial.to_numpy(),
            "structural_p50_eur_mwh": initial.to_numpy(),
            "structural_p90_eur_mwh": initial.to_numpy(),
            "structural_width_eur_mwh": [0.0] * len(timestamps),
        }
    )

    smoothed, audit = apply_final_quant_annual_smoothness_calibration(
        hourly,
        ts_ch=pd.Series(timestamps, index=hourly.index),
        base_forward_prices={"2030": target},
        peak_forward_prices={},
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        lambda_smooth_h=0.0,
        lambda_smooth_m=10000.0,
        lambda_seam=10000.0,
        negative_price_floor=-30.0,
        max_weighted_negative_hours=0,
    )

    smoothed_month = pd.Series(timestamps.month, index=smoothed.index)
    before_monthly = hourly.groupby(month)["price_weighted_mean_eur_mwh"].mean()
    after_monthly = smoothed.groupby(smoothed_month)["price_weighted_mean_eur_mwh"].mean()

    assert set(audit["product"]) == {"2030"}
    assert float(smoothed["price_weighted_mean_eur_mwh"].mean()) == pytest.approx(target, abs=1e-6)
    assert after_monthly.loc[2] > before_monthly.loc[2]
    assert (after_monthly.max() - after_monthly.min()) < (before_monthly.max() - before_monthly.min())


def test_post_calibration_negative_rebalancer_respects_floor():
    timestamps = pd.date_range("2030-07-01", periods=24 * 7, freq="h", tz="Europe/Zurich")
    hourly = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": [10.0] * len(timestamps),
            "price_central_eur_mwh": [4.0] * len(timestamps),
            "price_fast_eur_mwh": [2.0] * len(timestamps),
            "price_weighted_mean_eur_mwh": [5.0] * len(timestamps),
            "structural_p10_eur_mwh": [2.0] * len(timestamps),
            "structural_p50_eur_mwh": [4.0] * len(timestamps),
            "structural_p90_eur_mwh": [10.0] * len(timestamps),
            "structural_width_eur_mwh": [8.0] * len(timestamps),
        }
    )

    rebalanced, _ = apply_post_calibration_negative_rebalancer(
        hourly,
        forward_prices={"2030": 5.0},
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        intensity=5.0,
        negative_price_floor=-6.0,
        max_weighted_negative_hours=999,
    )

    assert rebalanced["price_fast_eur_mwh"].min() >= -6.000001


def test_post_calibration_peak_shape_rebalancer_preserves_mean_and_lifts_peak():
    timestamps = pd.date_range("2030-07-01", periods=24 * 14, freq="h", tz="Europe/Zurich")
    hourly = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": [80.0] * len(timestamps),
            "price_central_eur_mwh": [80.0] * len(timestamps),
            "price_fast_eur_mwh": [80.0] * len(timestamps),
            "price_weighted_mean_eur_mwh": [80.0] * len(timestamps),
            "structural_p10_eur_mwh": [80.0] * len(timestamps),
            "structural_p50_eur_mwh": [80.0] * len(timestamps),
            "structural_p90_eur_mwh": [80.0] * len(timestamps),
            "structural_width_eur_mwh": [0.0] * len(timestamps),
        }
    )
    peak = (timestamps.weekday < 5) & (timestamps.hour >= 8) & (timestamps.hour <= 19)

    rebalanced, audit = apply_post_calibration_peak_shape_rebalancer(
        hourly,
        forward_prices={"2030": 80.0},
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        intensity=1.0,
        negative_price_floor=-30.0,
        max_weighted_negative_hours=0,
    )

    assert rebalanced["price_weighted_mean_eur_mwh"].mean() == pytest.approx(80.0)
    assert rebalanced.loc[peak, "price_weighted_mean_eur_mwh"].mean() > 80.0
    assert rebalanced.loc[~peak, "price_weighted_mean_eur_mwh"].mean() < 80.0
    assert not audit.empty


def test_ch_holidays_are_nonworking_for_local_test_shape_overlays():
    national_day = pd.Timestamp("2030-08-01 12:00", tz="Europe/Zurich")
    labour_day = pd.Timestamp("2030-05-01 12:00", tz="Europe/Zurich")
    cantonal_day = pd.Timestamp("2030-03-19 12:00", tz="Europe/Zurich")
    nearby_workday = pd.Timestamp("2030-05-02 12:00", tz="Europe/Zurich")
    cantonal_peak = pd.Timestamp("2030-03-19 18:00", tz="Europe/Zurich")
    nearby_peak = pd.Timestamp("2030-05-02 18:00", tz="Europe/Zurich")

    assert _is_ch_nonworking_day(national_day)
    assert _is_ch_nonworking_day(labour_day)
    assert not _is_ch_nonworking_day(nearby_workday)
    assert _ch_market_holiday_pressure(labour_day) >= 1.0
    assert _ch_market_holiday_pressure(national_day) >= 1.0
    assert 0.0 < _ch_market_holiday_pressure(cantonal_day) < _ch_market_holiday_pressure(labour_day)
    assert _ch_market_nonworking_pressure(nearby_workday) == 0.0
    assert _negative_capture_weight(labour_day) > _negative_capture_weight(nearby_workday)
    assert _negative_capture_weight(cantonal_day) == 0.0
    assert _peak_shape_up_weight(national_day) == 0.0
    assert _peak_shape_up_weight(labour_day) == 0.0
    assert _peak_shape_up_weight(cantonal_peak) < _peak_shape_up_weight(nearby_peak)
    assert _peak_shape_down_weight(labour_day) > 0.0
    assert ch_market_nonworking_pressure(labour_day) == _ch_market_nonworking_pressure(labour_day)

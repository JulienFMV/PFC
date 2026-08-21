from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_fan_to_hourly_parity import diagnose_parity
from scripts.export_local_test_ch_hourly_csv import to_hourly_csv_frame


def _write_fan(path: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    values = np.arange(len(index), dtype=float)
    fan = pd.DataFrame(
        {
            "curve_slow": values + 70.0,
            "curve_central": values + 75.0,
            "curve_fast": values + 80.0,
            "weighted_mean": values + 76.0,
            "structural_scenario_low": values + 68.0,
            "structural_scenario_central": values + 76.0,
            "structural_scenario_high": values + 86.0,
            "structural_scenario_spread": np.full(len(index), 18.0),
        },
        index=index,
    )
    fan.to_parquet(path)
    return fan


def test_diagnose_fan_to_hourly_parity_writes_explainable_outputs(tmp_path: Path) -> None:
    index = pd.date_range("2027-01-03 23:00", periods=96, freq="15min", tz="UTC")
    fan_path = tmp_path / "fan.parquet"
    fan = _write_fan(fan_path, index)
    reference = to_hourly_csv_frame(fan, local_start_date="2027-01-04", local_end_date="2027-01-04")
    reference["price_weighted_mean_eur_mwh"] = reference["price_weighted_mean_eur_mwh"] + 2.0
    reference_path = tmp_path / "reference.csv"
    reference.to_csv(reference_path, index=False)

    output = tmp_path / "diagnostic"
    summary = diagnose_parity(
        fan_parquet=fan_path,
        reference_csv=reference_path,
        local_start_date="2027-01-04",
        local_end_date="2027-01-04",
        output_dir=output,
    )

    assert summary["schema_version"] == "fan_to_hourly_parity_diagnostic.v1"
    assert summary["read_only"] is True
    assert summary["promotion_gate"] is False
    assert summary["aligned_rows"] == 24
    assert summary["coverage_status"] == "PASS"
    assert summary["missing_timestamps_in_fan"] == 0
    assert summary["missing_timestamps_in_reference"] == 0
    assert summary["fan_coverage_ratio"] == pytest.approx(1.0)
    assert summary["reference_coverage_ratio"] == pytest.approx(1.0)
    assert summary["max_abs_weighted_delta_eur_mwh"] == pytest.approx(2.0)
    assert summary["mean_abs_weighted_delta_eur_mwh"] == pytest.approx(2.0)
    assert (output / "fan_hourly_converted.csv").exists()
    assert (output / "aligned_fan_reference.csv").exists()
    assert (output / "column_delta_summary.csv").exists()
    assert (output / "monthly_delta_summary.csv").exists()
    assert (output / "load_type_delta_summary.csv").exists()
    written = json.loads((output / "fan_to_hourly_parity_summary.json").read_text(encoding="utf-8"))
    assert written["outputs"]["fan_hourly_converted_csv"].endswith("fan_hourly_converted.csv")


def test_diagnose_fan_to_hourly_parity_reports_partial_timestamp_overlap(tmp_path: Path) -> None:
    index = pd.date_range("2027-01-03 23:00", periods=96, freq="15min", tz="UTC")
    fan_path = tmp_path / "fan.parquet"
    fan = _write_fan(fan_path, index)
    reference = to_hourly_csv_frame(fan, local_start_date="2027-01-04", local_end_date="2027-01-04")
    reference = reference.iloc[:-1].copy()
    reference_path = tmp_path / "reference.csv"
    reference.to_csv(reference_path, index=False)

    summary = diagnose_parity(
        fan_parquet=fan_path,
        reference_csv=reference_path,
        local_start_date="2027-01-04",
        local_end_date="2027-01-04",
        output_dir=tmp_path / "diagnostic",
    )

    assert summary["coverage_status"] == "PARTIAL_OVERLAP"
    assert summary["aligned_rows"] == 23
    assert summary["missing_timestamps_in_fan"] == 0
    assert summary["missing_timestamps_in_reference"] == 1
    assert summary["fan_coverage_ratio"] == pytest.approx(23 / 24)
    assert summary["reference_coverage_ratio"] == pytest.approx(1.0)

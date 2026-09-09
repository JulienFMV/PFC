from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.run_epex_shape_lab_ab import run_ab


def _candidate_csv(path, timestamps: pd.DatetimeIndex) -> None:
    values = np.where(timestamps.month == 1, 80.0, 90.0).astype(float)
    frame = pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%d.%m.%Y %H:%M"),
            "utc_offset_ch": timestamps.strftime("UTC%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%d.%m.%Y %H:%M"),
            "price_slow_eur_mwh": values - 2.0,
            "price_central_eur_mwh": values,
            "price_fast_eur_mwh": values + 2.0,
            "price_weighted_mean_eur_mwh": values,
            "structural_p10_eur_mwh": values - 8.0,
            "structural_p50_eur_mwh": values,
            "structural_p90_eur_mwh": values + 12.0,
            "structural_width_eur_mwh": np.full(len(values), 20.0),
        }
    )
    frame.to_csv(path, index=False)


def _spot_parquet(path) -> None:
    timestamps = pd.date_range("2024-01-01", "2025-12-31 23:00", freq="h", tz="Europe/Zurich")
    hour = timestamps.hour.to_numpy(dtype=float)
    weekend_discount = np.where(timestamps.weekday >= 5, -5.0, 0.0)
    solar_tail = np.where((timestamps.month >= 3) & (timestamps.month <= 10) & (hour >= 10) & (hour <= 16), -4.0, 0.0)
    price = 70.0 + 4.0 * np.sin(hour / 24.0 * 2.0 * np.pi) + weekend_discount + solar_tail
    pd.DataFrame(
        {"price_eur_mwh": price},
        index=timestamps.tz_convert("UTC"),
    ).to_parquet(path)


def test_run_epex_shape_lab_ab_writes_lab_only_manifest_and_preserves_monthly_constraints(tmp_path) -> None:
    candidate = tmp_path / "candidate.csv"
    spot = tmp_path / "spot.parquet"
    output = tmp_path / "ab"
    timestamps = pd.date_range("2030-01-01", "2030-02-28 23:00", freq="h", tz="Europe/Zurich")
    _candidate_csv(candidate, timestamps)
    _spot_parquet(spot)

    manifest = run_ab(
        candidate_csv=candidate,
        spot_parquet=spot,
        output_dir=output,
        valuation_timestamp="2026-01-01T00:00:00Z",
        weekend_intensity=1.0,
        low_tail_intensity=0.5,
        peak_subshape_intensity=0.5,
        evening_recovery_intensity=0.25,
        max_abs_delta_eur_mwh=6.0,
        preserve_peak=True,
    )

    assert manifest["activation_status"] == "lab_only"
    assert manifest["production_approved"] is False
    assert manifest["ompex_used_in_selection"] is False
    assert manifest["constraint_summary"]["base_monthly_constraints"] == 2
    assert manifest["constraint_summary"]["peak_monthly_constraints"] == 2
    assert manifest["constraint_summary"]["max_after_abs_error_eur_mwh"] == pytest.approx(0.0, abs=1e-6)
    assert manifest["candidate_delta_summary"]["max_abs_delta_eur_mwh"] > 0.0
    assert manifest["config"]["evening_recovery_intensity"] == pytest.approx(0.25)
    assert (output / "candidate_epex_shape_lab_adjusted.csv").exists()
    assert (output / "constraint_residuals_before_after.csv").exists()
    assert (output / "pre_registered_ab_plan.json").exists()
    json.dumps(manifest)

    plan = json.loads((output / "pre_registered_ab_plan.json").read_text(encoding="utf-8"))
    assert "OMPEX as target" in plan["forbidden_checks"]
    assert plan["config"]["evening_recovery_intensity"] == pytest.approx(0.25)

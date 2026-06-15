from __future__ import annotations

import pytest
import pandas as pd

from scripts.export_local_test_ch_hourly_csv import (
    apply_local_test_structural_shape_upgrade,
    calibrate_hourly_to_eex,
    previous_business_day,
    require_forward_date,
    to_hourly_csv_frame,
)


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
        "structural_p10_eur_mwh",
        "structural_p50_eur_mwh",
        "structural_p90_eur_mwh",
        "structural_width_eur_mwh",
    ]
    assert out.loc[0, "timestamp_ch"] == "13.06.2026 00:00"
    assert out.loc[0, "utc_offset_ch"] == "UTC+02:00"
    assert out.loc[0, "price_weighted_mean_eur_mwh"] == 31.5


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

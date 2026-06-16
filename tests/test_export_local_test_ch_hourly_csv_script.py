from __future__ import annotations

import pytest
import pandas as pd

from scripts.export_local_test_ch_hourly_csv import (
    _ch_market_holiday_pressure,
    _ch_market_nonworking_pressure,
    _eex_peak_mask,
    _latest_eex_prices_by_load_type,
    _is_ch_nonworking_day,
    _negative_capture_weight,
    _peak_shape_down_weight,
    _peak_shape_up_weight,
    apply_post_calibration_negative_rebalancer,
    apply_post_calibration_peak_shape_rebalancer,
    apply_local_test_structural_shape_upgrade,
    calibrate_hourly_to_eex,
    calibrate_hourly_to_eex_base_peak,
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

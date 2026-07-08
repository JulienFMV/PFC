from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pfc_shaping.lt.model.epex_shape_lab import (
    ABShapeLabConfig,
    apply_epex_ab_shape_lab,
    build_ab_lab_manifest,
    fit_epex_shape_templates,
)
from pfc_shaping.lt.model.shape_constraints import build_base_peak_offpeak_constraint_system, eex_peak_mask


def _hourly_frame(timestamps: pd.DatetimeIndex, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_ch": timestamps.strftime("%Y-%m-%d %H:%M:%S%z"),
            "timestamp_utc": timestamps.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z"),
            "price_slow_eur_mwh": values,
            "price_central_eur_mwh": values,
            "price_fast_eur_mwh": values,
            "price_weighted_mean_eur_mwh": values,
            "structural_p10_eur_mwh": values,
            "structural_p50_eur_mwh": values,
            "structural_p90_eur_mwh": values,
            "structural_width_eur_mwh": np.zeros(len(values)),
        }
    )


def _spot_history(start: str = "2024-01-01", end: str = "2025-12-31 23:00") -> pd.DataFrame:
    timestamps = pd.date_range(start, end, freq="h", tz="Europe/Zurich")
    month_level = 80.0 + 10.0 * np.sin((timestamps.month.to_numpy(dtype=float) - 1.0) / 12.0 * 2.0 * np.pi)
    hour = timestamps.hour.to_numpy(dtype=float)
    peak_hump = np.where((hour >= 8) & (hour <= 19), 5.0 * np.sin((hour - 8.0) / 11.0 * np.pi), 0.0)
    weekend_discount = np.where(timestamps.weekday >= 5, -7.0, 0.0)
    solar_tail = np.where((timestamps.month >= 3) & (timestamps.month <= 10) & (hour >= 10) & (hour <= 16), -6.0, 0.0)
    price = month_level + peak_hump + weekend_discount + solar_tail
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps.tz_convert("UTC"),
            "price_eur_mwh": price,
        }
    )


def test_epex_shape_lab_preserves_base_and_peak_constraints() -> None:
    spot = _spot_history()
    config = ABShapeLabConfig(
        valuation_timestamp="2026-01-01T00:00:00Z",
        weekend_intensity=1.0,
        low_tail_intensity=0.5,
        peak_subshape_intensity=0.8,
        max_abs_delta_eur_mwh=12.0,
    )
    templates, _ = fit_epex_shape_templates(spot, config)

    timestamps = pd.date_range("2030-01-01", "2030-02-28 23:00", freq="h", tz="Europe/Zurich")
    values = np.where(timestamps.month == 1, 92.0, 74.0).astype(float)
    hourly = _hourly_frame(timestamps, values)
    peak_mask = eex_peak_mask(timestamps.tz_convert("UTC"), country="CH")
    base_forward_prices = {
        "2030-01": float(hourly.loc[timestamps.month == 1, "price_weighted_mean_eur_mwh"].mean()),
        "2030-02": float(hourly.loc[timestamps.month == 2, "price_weighted_mean_eur_mwh"].mean()),
    }
    peak_forward_prices = {
        "2030-01": float(hourly.loc[(timestamps.month == 1) & peak_mask, "price_weighted_mean_eur_mwh"].mean()),
        "2030-02": float(hourly.loc[(timestamps.month == 2) & peak_mask, "price_weighted_mean_eur_mwh"].mean()),
    }

    adjusted, audit = apply_epex_ab_shape_lab(
        hourly,
        templates,
        ts_ch=pd.Series(timestamps, index=hourly.index),
        base_forward_prices=base_forward_prices,
        peak_forward_prices=peak_forward_prices,
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        config=config,
    )

    system = build_base_peak_offpeak_constraint_system(
        timestamps.tz_convert("UTC"),
        base_forward_prices,
        peak_forward_prices,
        country="CH",
    )
    residuals = system.residuals(adjusted["price_weighted_mean_eur_mwh"].to_numpy(dtype=float))

    assert not audit.empty
    assert float(audit["max_abs_projected_delta_eur_mwh"].iloc[0]) > 0.0
    assert float(audit["max_delta_constraint_residual_eur_mwh"].iloc[0]) == pytest.approx(0.0, abs=1e-8)
    assert float(audit["max_after_constraint_abs_error_eur_mwh"].iloc[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(residuals["abs_error"].max()) == pytest.approx(0.0, abs=1e-6)
    assert (adjusted["structural_p10_eur_mwh"] <= adjusted["structural_p50_eur_mwh"]).all()
    assert (adjusted["structural_p50_eur_mwh"] <= adjusted["structural_p90_eur_mwh"]).all()


def test_epex_shape_lab_shifts_existing_fan_without_rebuilding_width() -> None:
    spot = _spot_history()
    config = ABShapeLabConfig(
        valuation_timestamp="2026-01-01T00:00:00Z",
        weekend_intensity=1.0,
        max_abs_delta_eur_mwh=10.0,
    )
    templates, _ = fit_epex_shape_templates(spot, config)
    timestamps = pd.date_range("2030-01-01", "2030-01-31 23:00", freq="h", tz="Europe/Zurich")
    hourly = _hourly_frame(timestamps, np.full(len(timestamps), 80.0))
    hourly["structural_p10_eur_mwh"] = 65.0
    hourly["structural_p50_eur_mwh"] = 80.0
    hourly["structural_p90_eur_mwh"] = 110.0
    hourly["structural_width_eur_mwh"] = 45.0

    adjusted, audit = apply_epex_ab_shape_lab(
        hourly,
        templates,
        ts_ch=pd.Series(timestamps, index=hourly.index),
        base_forward_prices={"2030-01": 80.0},
        peak_forward_prices={},
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        config=config,
    )

    assert not audit.empty
    delta = adjusted["price_weighted_mean_eur_mwh"] - hourly["price_weighted_mean_eur_mwh"]
    assert adjusted["structural_p10_eur_mwh"].to_numpy(dtype=float) == pytest.approx((65.0 + delta).to_numpy(dtype=float))
    assert adjusted["structural_p50_eur_mwh"].to_numpy(dtype=float) == pytest.approx((80.0 + delta).to_numpy(dtype=float))
    assert adjusted["structural_p90_eur_mwh"].to_numpy(dtype=float) == pytest.approx((110.0 + delta).to_numpy(dtype=float))
    assert adjusted["structural_width_eur_mwh"].to_numpy(dtype=float) == pytest.approx(np.full(len(timestamps), 45.0))


def test_epex_shape_lab_rejects_non_monthly_base_constraints_by_default() -> None:
    spot = _spot_history()
    config = ABShapeLabConfig(
        valuation_timestamp="2026-01-01T00:00:00Z",
        weekend_intensity=1.0,
    )
    templates, _ = fit_epex_shape_templates(spot, config)
    timestamps = pd.date_range("2030-01-01", "2030-02-28 23:00", freq="h", tz="Europe/Zurich")
    hourly = _hourly_frame(timestamps, np.full(len(timestamps), 80.0))

    with pytest.raises(ValueError, match="monthly BASE constraints"):
        apply_epex_ab_shape_lab(
            hourly,
            templates,
            ts_ch=pd.Series(timestamps, index=hourly.index),
            base_forward_prices={"2030": 80.0},
            peak_forward_prices={},
            weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
            config=config,
        )


def test_epex_shape_lab_accepts_dst_offset_timestamp_strings() -> None:
    spot = _spot_history()
    config = ABShapeLabConfig(
        valuation_timestamp="2026-01-01T00:00:00Z",
        weekend_intensity=1.0,
        max_abs_delta_eur_mwh=10.0,
    )
    templates, _ = fit_epex_shape_templates(spot, config)
    timestamps = pd.date_range("2030-03-01", "2030-04-30 23:00", freq="h", tz="Europe/Zurich")
    hourly = _hourly_frame(timestamps, np.full(len(timestamps), 80.0))

    adjusted, audit = apply_epex_ab_shape_lab(
        hourly,
        templates,
        ts_ch=hourly["timestamp_ch"],
        base_forward_prices={"2030-03": 80.0, "2030-04": 80.0},
        peak_forward_prices={},
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        config=config,
    )

    assert not audit.empty
    assert len(adjusted) == len(hourly)
    assert float(adjusted.loc[timestamps.month == 3, "price_weighted_mean_eur_mwh"].mean()) == pytest.approx(
        80.0,
        abs=1e-6,
    )


def test_epex_shape_fit_is_point_in_time_and_ignores_future_rows() -> None:
    base = _spot_history(start="2024-01-01", end="2024-03-31 23:00")
    future = _spot_history(start="2026-01-01", end="2026-01-31 23:00")
    future["price_eur_mwh"] = 10_000.0
    config = ABShapeLabConfig(valuation_timestamp="2025-01-01T00:00:00Z", lookback_years=2)

    templates_clean, info_clean = fit_epex_shape_templates(base, config)
    templates_with_future, info_with_future = fit_epex_shape_templates(pd.concat([base, future], ignore_index=True), config)

    pd.testing.assert_frame_equal(templates_clean, templates_with_future)
    assert pd.Timestamp(info_with_future["max_timestamp_utc"]) < pd.Timestamp("2025-01-01T00:00:00Z")
    assert info_clean["benchmark_policy"] == "advisory_only_ompex_forbidden_as_input_target_loss_or_gate"


def test_epex_shape_fit_requires_explicit_valuation_timestamp() -> None:
    with pytest.raises(ValueError, match="explicit valuation_timestamp"):
        fit_epex_shape_templates(_spot_history(), ABShapeLabConfig())


def test_epex_shape_fit_rejects_naive_spot_timestamp_column() -> None:
    spot = _spot_history()
    spot["timestamp"] = pd.to_datetime(spot["timestamp_utc"]).dt.tz_localize(None)
    spot = spot.drop(columns=["timestamp_utc"])

    with pytest.raises(ValueError, match="timezone-aware"):
        fit_epex_shape_templates(
            spot,
            ABShapeLabConfig(valuation_timestamp="2026-01-01T00:00:00Z"),
        )


def test_epex_shape_lab_noop_when_all_intensities_are_zero() -> None:
    spot = _spot_history()
    config = ABShapeLabConfig(valuation_timestamp="2026-01-01T00:00:00Z")
    templates, _ = fit_epex_shape_templates(spot, config)
    timestamps = pd.date_range("2030-01-01", "2030-01-31 23:00", freq="h", tz="Europe/Zurich")
    hourly = _hourly_frame(timestamps, np.full(len(timestamps), 80.0))

    adjusted, audit = apply_epex_ab_shape_lab(
        hourly,
        templates,
        ts_ch=pd.Series(timestamps, index=hourly.index),
        base_forward_prices={"2030-01": 80.0},
        peak_forward_prices={},
        weights={"slow": 0.25, "central": 0.5, "fast": 0.25},
        config=config,
    )

    pd.testing.assert_frame_equal(adjusted, hourly)
    assert audit.empty


def test_epex_shape_lab_manifest_forbids_ompex_as_model_input() -> None:
    config = ABShapeLabConfig(valuation_timestamp=pd.Timestamp("2026-01-01T00:00:00Z"), weekend_intensity=1.0)
    manifest = build_ab_lab_manifest(
        config,
        fit_info={"source": "EPEX_CH_spot_history", "max_timestamp_utc": pd.Timestamp("2025-12-31T23:00:00Z")},
        source_hashes={"epex_spot": "abc123"},
        audit=pd.DataFrame([{"max_after_constraint_abs_error_eur_mwh": np.float64(0.0)}]),
    )

    assert manifest["ab_shape_lab"] == "epex_only_off_by_default"
    assert manifest["activation_status"] == "lab_only"
    assert manifest["production_approved"] is False
    assert manifest["benchmark_policy"] == "advisory_only_ompex_forbidden_as_input_target_loss_or_gate"
    assert manifest["ompex_used_in_selection"] is False
    assert manifest["source_hashes_required_for_promotion"] is True
    assert "OMPEX" in manifest["forbidden_inputs"]
    json.dumps(manifest)

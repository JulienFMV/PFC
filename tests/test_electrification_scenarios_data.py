from __future__ import annotations

import pandas as pd
import pytest

from pfc_shaping.data.electrification_scenarios import (
    asof_electrification_scenarios,
    assert_scenario_coverage,
    derive_hpfc_scenario_features,
    interpolate_electrification_scenario_years,
    validate_electrification_scenarios,
)


def _frame():
    return pd.DataFrame(
        [
            {
                "publication_date": "2025-01-01",
                "source": "unit",
                "scenario": "central",
                "country": "CH",
                "delivery_year": 2030,
                "delivery_month": None,
                "scenario_weight": 0.5,
                "demand_twh": 60.0,
                "pv_gw": 20.0,
                "battery_power_gw": 3.0,
                "battery_energy_gwh": 12.0,
                "ntc_ch_de_gw": 4.0,
            },
            {
                "publication_date": "2026-01-01",
                "source": "unit",
                "scenario": "central",
                "country": "CH",
                "delivery_year": 2031,
                "delivery_month": 7,
                "scenario_weight": 0.5,
                "demand_twh": 62.0,
                "pv_gw": 25.0,
            },
        ]
    )


def test_validate_normalizes_publication_date_to_utc_and_numeric_columns():
    out = validate_electrification_scenarios(_frame())

    assert str(out["publication_date"].dt.tz) == "UTC"
    assert out["delivery_year"].dtype.kind in {"i", "u"}
    assert float(out.loc[0, "pv_gw"]) == 20.0


def test_validate_rejects_missing_required_columns():
    with pytest.raises(KeyError, match="required columns"):
        validate_electrification_scenarios(pd.DataFrame([{"scenario": "central"}]))


def test_validate_rejects_negative_capacity():
    frame = _frame()
    frame.loc[0, "battery_power_gw"] = -1.0
    with pytest.raises(ValueError, match="battery_power_gw"):
        validate_electrification_scenarios(frame)


def test_validate_rejects_infinite_capacity():
    frame = _frame()
    frame.loc[0, "battery_power_gw"] = float("inf")
    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_electrification_scenarios(frame)


def test_validate_rejects_managed_charging_share_outside_unit_interval():
    frame = _frame()
    frame.loc[0, "managed_charging_share"] = 1.5
    with pytest.raises(ValueError, match="managed_charging_share"):
        validate_electrification_scenarios(frame)


def test_validate_rejects_bad_delivery_month():
    frame = _frame()
    frame.loc[0, "delivery_month"] = 13
    with pytest.raises(ValueError, match="delivery_month"):
        validate_electrification_scenarios(frame)


def test_validate_rejects_unknown_track():
    frame = _frame()
    frame["track"] = "forecast"

    with pytest.raises(ValueError, match="track"):
        validate_electrification_scenarios(frame)


def test_validate_actual_rows_require_measurement_date():
    frame = _frame()
    frame["track"] = "actual"
    frame["measurement_date"] = pd.NaT

    with pytest.raises(ValueError, match="measurement_date"):
        validate_electrification_scenarios(frame)


def test_derive_hpfc_scenario_features_normalizes_physical_drivers():
    frame = pd.DataFrame(
        [
            {
                "publication_date": "2025-01-01",
                "source": "unit",
                "scenario": "central",
                "country": "CH",
                "delivery_year": 2030,
                "demand_twh": 87.6,
                "peak_load_gw": 12.0,
                "winter_demand_twh": 40.0,
                "pv_twh": 21.9,
                "pv_gw": 99.0,
                "wind_gw": 5.0,
                "battery_power_gw": 2.0,
                "battery_energy_gwh": 20.0,
                "ev_twh": 8.76,
                "heatpump_twh": 4.38,
                "managed_charging_share": 0.4,
                "hydro_twh": 35.0,
                "hydro_capacity_gw": 10.0,
                "hydro_reservoir_twh": 8.0,
                "nuclear_gw": 3.0,
                "dispatchable_gw": 1.0,
                "net_import_twh": -2.0,
                "ntc_ch_de_gw": 4.0,
                "ntc_ch_fr_gw": 3.0,
            }
        ]
    )

    out = derive_hpfc_scenario_features(frame).iloc[0]

    assert float(out["avg_load_gw"]) == pytest.approx(10.0)
    assert float(out["pv_penetration"]) == pytest.approx(0.25)
    assert float(out["wind_penetration"]) == pytest.approx(0.5)
    assert float(out["battery_power_penetration"]) == pytest.approx(0.2)
    assert float(out["battery_energy_cover_h"]) == pytest.approx(2.0)
    assert float(out["battery_penetration"]) == pytest.approx(0.35)
    assert float(out["ev_penetration"]) == pytest.approx(0.10)
    assert float(out["heatpump_penetration"]) == pytest.approx(0.05)
    assert float(out["hydro_energy_share"]) == pytest.approx(35.0 / 87.6)
    assert float(out["hydro_capacity_penetration"]) == pytest.approx(10.0 / 12.0)
    assert float(out["hydro_reservoir_cover"]) == pytest.approx(8.0 / 40.0)
    assert float(out["hydro_reservoir_available"]) == 1.0
    assert float(out["import_dependency"]) == pytest.approx(-2.0 / 87.6)
    assert float(out["ntc_penetration"]) == pytest.approx(7.0 / 12.0)
    assert float(out["firm_capacity_penetration"]) == pytest.approx(4.0 / 12.0)
    assert float(out["managed_charging_share"]) == pytest.approx(0.4)


def test_derive_hpfc_scenario_features_renormalizes_missing_hydro_reservoir():
    frame = pd.DataFrame(
        [
            {
                "publication_date": "2025-01-01",
                "source": "unit",
                "scenario": "central",
                "country": "DE",
                "delivery_year": 2030,
                "demand_twh": 100.0,
                "peak_load_gw": 10.0,
                "winter_demand_twh": 50.0,
                "hydro_twh": 20.0,
                "hydro_capacity_gw": 5.0,
                "hydro_reservoir_twh": pd.NA,
            }
        ]
    )

    out = derive_hpfc_scenario_features(frame).iloc[0]

    avg_load_gw = 100.0 * 1000.0 / 8760.0
    expected_capacity = 5.0 / avg_load_gw
    expected = (0.20 * 0.20 + 0.45 * expected_capacity) / (0.20 + 0.45)
    assert float(out["hydro_reservoir_cover"]) == 0.0
    assert float(out["hydro_reservoir_available"]) == 0.0
    assert float(out["hydro_flexibility"]) == pytest.approx(expected)


def test_asof_filters_future_publications():
    out = asof_electrification_scenarios(_frame(), pd.Timestamp("2025-06-01", tz="UTC"))

    assert len(out) == 1
    assert int(out.iloc[0]["delivery_year"]) == 2030


def test_asof_filters_actual_rows_with_future_measurement_date():
    frame = _frame()
    frame["track"] = "scenario"
    frame["measurement_date"] = pd.NaT
    actual = frame.iloc[[0]].copy()
    actual["track"] = "actual"
    actual["measurement_date"] = pd.Timestamp("2025-07-01", tz="UTC")
    actual["delivery_year"] = 2025
    mixed = pd.concat([frame.iloc[[0]], actual], ignore_index=True)

    out = asof_electrification_scenarios(mixed, pd.Timestamp("2025-06-01", tz="UTC"))

    assert len(out) == 1
    assert out.iloc[0]["track"] == "scenario"


def test_assert_scenario_coverage_accepts_year_level_rows():
    frame = asof_electrification_scenarios(_frame(), pd.Timestamp("2025-06-01", tz="UTC"))
    assert_scenario_coverage(
        frame,
        country="CH",
        scenarios=["central"],
        periods=[pd.Period("2030-07", "M")],
    )


def test_assert_scenario_coverage_reports_missing_periods():
    frame = asof_electrification_scenarios(_frame(), pd.Timestamp("2025-06-01", tz="UTC"))
    with pytest.raises(KeyError, match="missing electrification scenario coverage"):
        assert_scenario_coverage(
            frame,
            country="CH",
            scenarios=["fast"],
            periods=[pd.Period("2030-07", "M")],
        )


def test_interpolate_electrification_scenario_years_preserves_provenance():
    frame = pd.DataFrame(
        [
            {
                "publication_date": "2025-01-01",
                "source": "unit",
                "scenario": "central",
                "country": "CH",
                "delivery_year": 2030,
                "quality_flag": "official",
                "demand_twh": 70.0,
                "pv_twh": 10.0,
            },
            {
                "publication_date": "2025-01-01",
                "source": "unit",
                "scenario": "central",
                "country": "CH",
                "delivery_year": 2035,
                "quality_flag": "official",
                "demand_twh": 80.0,
                "pv_twh": 20.0,
            },
        ]
    )

    out = interpolate_electrification_scenario_years(frame, [2030, 2031, 2035])
    row_2031 = out[out["delivery_year"] == 2031].iloc[0]

    assert len(out) == 3
    assert row_2031["scenario"] == "central"
    assert row_2031["quality_flag"] == "official_interpolated"
    assert float(row_2031["demand_twh"]) == pytest.approx(72.0)
    assert float(row_2031["pv_twh"]) == pytest.approx(12.0)


def test_interpolate_electrification_scenario_years_can_forbid_clamping():
    frame = pd.DataFrame(
        [
            {
                "publication_date": "2025-01-01",
                "source": "unit",
                "scenario": "central",
                "country": "CH",
                "delivery_year": 2030,
                "demand_twh": 70.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="requires bracketing published years"):
        interpolate_electrification_scenario_years(frame, [2030, 2031], allow_clamp=False)

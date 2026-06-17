from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_ep2050_multi_scenario_pfc import (
    build_weighted_fan_chart,
    derive_slow_central_fast_inventory,
)


def _enriched_inventory() -> pd.DataFrame:
    rows = []
    for scenario, demand_twh, pv_twh, ev_twh in [
        ("WWB", 70.0, 6.0, 0.10),
        ("ZERO_Basis", 74.0, 14.0, 1.00),
    ]:
        rows.append(
            {
                "publication_date": pd.Timestamp("2026-06-05", tz="UTC"),
                "source": "OFEN_EP2050_hourly_11142+ch_first_pfc_proxy_v0",
                "scenario": scenario,
                "country": "CH",
                "delivery_year": 2030,
                "delivery_month": np.nan,
                "scenario_weight": 0.5,
                "quality_flag": "official_proxy_enriched",
                "demand_twh": demand_twh,
                "peak_load_gw": 15.0,
                "winter_demand_twh": 20.0,
                "pv_gw": pv_twh / 1.5,
                "pv_twh": pv_twh,
                "wind_gw": 0.2,
                "wind_twh": 0.4,
                "battery_power_gw": 2.0,
                "battery_energy_gwh": 5.0,
                "ev_twh": ev_twh,
                "heatpump_twh": 4.0,
                "managed_charging_share": 0.35,
                "hydro_twh": 40.0,
                "hydro_capacity_gw": 15.0,
                "hydro_reservoir_twh": 4.0,
                "nuclear_gw": 1.2,
                "dispatchable_gw": 0.0,
                "gas_gw": 0.0,
                "coal_gw": 0.0,
                "import_twh": 16.0,
                "export_twh": 8.0,
                "net_import_twh": 8.0,
                "ntc_ch_de_gw": 4.0,
                "ntc_ch_fr_gw": 4.0,
                "ntc_ch_it_gw": 3.0,
            }
        )
    return pd.DataFrame(rows)


def test_derive_slow_central_fast_inventory_is_bounded_and_governed():
    out = derive_slow_central_fast_inventory(
        _enriched_inventory(),
        weights={"slow": 0.25, "central": 0.50, "fast": 0.25},
        mapping_publication_date="2026-06-05",
    )

    assert list(out["scenario"]) == ["slow", "central", "fast"]
    assert dict(zip(out["scenario"], out["scenario_weight"])) == {
        "slow": 0.25,
        "central": 0.50,
        "fast": 0.25,
    }
    central = out[out["scenario"] == "central"].iloc[0]
    assert float(central["demand_twh"]) == pytest.approx(72.0)
    assert float(central["pv_twh"]) == pytest.approx(10.0)
    assert float(central["ev_twh"]) == pytest.approx(0.55)
    assert central["quality_flag"] == "internal_midpoint_proxy_enriched"
    assert str(central["original_scenario"]) == "midpoint(WWB,ZERO_Basis)"
    assert out["publication_date"].max() == pd.Timestamp("2026-06-05", tz="UTC")


def test_derive_slow_central_fast_inventory_fails_if_source_bound_missing():
    frame = _enriched_inventory()
    frame = frame[frame["scenario"] != "WWB"]

    with pytest.raises(KeyError, match="missing source scenarios"):
        derive_slow_central_fast_inventory(frame)


def test_build_weighted_fan_chart_writes_ordered_structural_columns():
    idx = pd.date_range("2030-01-01", periods=3, freq="1h", tz="UTC")
    frames = {
        "slow": pd.DataFrame({"price_shape": [60.0, 62.0, 64.0]}, index=idx),
        "central": pd.DataFrame({"price_shape": [65.0, 67.0, 69.0]}, index=idx),
        "fast": pd.DataFrame({"price_shape": [70.0, 72.0, 74.0]}, index=idx),
    }

    fan = build_weighted_fan_chart(
        frames,
        weights={"slow": 0.25, "central": 0.50, "fast": 0.25},
    )

    assert list(fan.columns) == [
        "curve_slow",
        "curve_central",
        "curve_fast",
        "weighted_mean",
        "structural_scenario_low",
        "structural_scenario_central",
        "structural_scenario_high",
        "structural_scenario_spread",
    ]
    assert float(fan["weighted_mean"].iloc[0]) == pytest.approx(65.0)
    assert np.all(fan["structural_scenario_low"] <= fan["structural_scenario_central"])
    assert np.all(fan["structural_scenario_central"] <= fan["structural_scenario_high"])
    assert float(fan["structural_scenario_spread"].mean()) > 0.0

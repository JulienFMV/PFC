from __future__ import annotations

import json

import pandas as pd

from scripts.build_local_test_ch_pfc import _build_fan, _expand_inventory, _fan_weighted_monthly_hash, _file_sha256


def _inventory():
    rows = []
    for scenario, weight, pv in [("slow", 0.25, 7.0), ("central", 0.5, 9.0), ("fast", 0.25, 12.0)]:
        rows.append(
            {
                "publication_date": pd.Timestamp("2026-01-01", tz="UTC"),
                "country": "CH",
                "scenario": scenario,
                "delivery_year": 2030,
                "delivery_month": None,
                "scenario_weight": weight,
                "quality_flag": "official_partial_proxy_neutralized_explicit",
                "source": "unit",
                "demand_twh": 70.0,
                "peak_load_gw": 14.0,
                "winter_demand_twh": 25.0,
                "pv_gw": pv,
                "pv_twh": pv,
                "wind_gw": 0.2,
                "wind_twh": 0.3,
                "battery_power_gw": 2.0,
                "battery_energy_gwh": 6.0,
                "ev_twh": 0.2,
                "heatpump_twh": 4.0,
                "managed_charging_share": 0.3,
                "hydro_twh": 40.0,
                "hydro_capacity_gw": 17.0,
                "hydro_reservoir_twh": 4.0,
                "nuclear_gw": 3.0,
                "dispatchable_gw": 3.0,
                "gas_gw": 0.1,
                "coal_gw": 0.0,
                "net_import_twh": 8.0,
                "ntc_ch_de_gw": 1.0,
                "ntc_ch_fr_gw": 1.0,
                "ntc_ch_it_gw": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_expand_inventory_clamps_single_year_for_local_test(tmp_path):
    output = tmp_path / "expanded.parquet"

    expanded = _expand_inventory(
        _inventory(),
        years=[2030, 2031],
        scenarios=["slow", "central", "fast"],
        output=output,
    )

    assert output.exists()
    assert set(expanded["delivery_year"]) == {2030, 2031}
    assert len(expanded) == 6
    assert expanded.loc[expanded["delivery_year"] == 2031, "quality_flag"].str.contains("interpolated").all()


def test_build_fan_has_ordered_scenario_bracket():
    index = pd.date_range("2030-01-01", periods=2, freq="15min", tz="UTC")
    curves = {
        "slow": pd.DataFrame({"price_shape": [40.0, 42.0]}, index=index),
        "central": pd.DataFrame({"price_shape": [45.0, 47.0]}, index=index),
        "fast": pd.DataFrame({"price_shape": [50.0, 52.0]}, index=index),
    }

    fan = _build_fan(curves, {"slow": 0.25, "central": 0.5, "fast": 0.25})

    assert (fan["structural_scenario_low"] <= fan["structural_scenario_central"]).all()
    assert (fan["structural_scenario_central"] <= fan["structural_scenario_high"]).all()
    assert (fan["structural_scenario_spread"] > 0).all()


def test_solver_manifest_fan_hash_fields_are_reproducible(tmp_path):
    index = pd.date_range("2030-01-01", periods=4, freq="15min", tz="UTC")
    fan = pd.DataFrame(
        {
            "curve_slow": [90.0, 91.0, 92.0, 93.0],
            "curve_central": [100.0, 101.0, 102.0, 103.0],
            "curve_fast": [110.0, 111.0, 112.0, 113.0],
            "weighted_mean": [100.0, 101.0, 102.0, 103.0],
        },
        index=index,
    )
    fan_path = tmp_path / "fan.parquet"
    fan.to_parquet(fan_path)
    manifest = {
        "fan_chart_sha256": _file_sha256(fan_path),
        "fan_chart_weighted_monthly_hash": _fan_weighted_monthly_hash(fan, timezone="Europe/Zurich"),
        "scenario_weights": {"slow": 0.25, "central": 0.5, "fast": 0.25},
    }
    manifest_path = fan_path.with_suffix(".monthly_curve_manifest.json")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert loaded["fan_chart_sha256"] == _file_sha256(fan_path)
    assert loaded["fan_chart_weighted_monthly_hash"] == _fan_weighted_monthly_hash(fan, timezone="Europe/Zurich")
    assert loaded["scenario_weights"] == {"central": 0.5, "fast": 0.25, "slow": 0.25}

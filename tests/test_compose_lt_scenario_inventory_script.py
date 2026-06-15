from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.compose_lt_scenario_inventory import compose_partial_inventory


def _supply(path):
    rows = []
    for country in ["CH", "DE"]:
        for scenario, weight in [("slow", 0.25), ("central", 0.50), ("fast", 0.25)]:
            rows.append(
                {
                    "publication_date": pd.Timestamp("2024-05-31", tz="UTC"),
                    "measurement_date": pd.NaT,
                    "track": "scenario",
                    "source": "tyndp_supply",
                    "scenario_edition": "TYNDP2024",
                    "scenario": scenario,
                    "country": country,
                    "delivery_year": 2030,
                    "delivery_month": np.nan,
                    "scenario_weight": weight,
                    "quality_flag": "official_tyndp_supply_partial",
                    "ingested_at_utc": pd.Timestamp("2026-06-11", tz="UTC"),
                    "demand_twh": np.nan,
                    "pv_gw": 10.0,
                    "wind_gw": 1.0,
                    "battery_energy_gwh": 2.0,
                    "nuclear_gw": 1.0,
                    "gas_eur_mwh": 30.0,
                    "coal_eur_mwh": 12.0,
                    "co2_eur_t": 80.0,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _ch(path):
    rows = []
    for scenario, weight in [("slow", 0.25), ("central", 0.50), ("fast", 0.25)]:
        rows.append(
            {
                "publication_date": pd.Timestamp("2026-06-05", tz="UTC"),
                "source": "ep2050",
                "scenario": scenario,
                "country": "CH",
                "delivery_year": 2030,
                "delivery_month": np.nan,
                "scenario_weight": weight,
                "quality_flag": "official_proxy_enriched_scenario_mapped",
                "demand_twh": 70.0,
                "pv_gw": 5.0,
                "battery_power_gw": 2.0,
                "battery_energy_gwh": 5.0,
                "ev_twh": 0.2,
                "heatpump_twh": 4.0,
                "hydro_twh": 40.0,
                "hydro_capacity_gw": 12.0,
                "hydro_reservoir_twh": 4.0,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _neighbor_demand(path):
    rows = []
    for country in ["DE"]:
        for scenario, weight in [("slow", 0.25), ("central", 0.50), ("fast", 0.25)]:
            rows.append(
                {
                    "publication_date": pd.Timestamp("2026-06-11", tz="UTC"),
                    "measurement_date": pd.NaT,
                    "track": "scenario",
                    "source": "demand_bridge",
                    "scenario_edition": "bridge",
                    "scenario": scenario,
                    "country": country,
                    "delivery_year": 2030,
                    "delivery_month": np.nan,
                    "scenario_weight": weight,
                    "quality_flag": "internal_tyndp_demand_bridge_partial_proxy",
                    "ingested_at_utc": pd.Timestamp("2026-06-11", tz="UTC"),
                    "demand_twh": 500.0,
                    "peak_load_gw": 80.0,
                    "winter_demand_twh": 260.0,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_compose_partial_inventory_overlays_ch_without_zero_filling(tmp_path):
    supply = tmp_path / "supply.parquet"
    ch = tmp_path / "ch.parquet"
    _supply(supply)
    _ch(ch)

    frame, features, provenance = compose_partial_inventory(
        tyndp_supply_path=supply,
        ch_ep2050_path=ch,
        years=[2030],
        vintage="2026-06-11",
        countries=["CH", "DE"],
        scenarios=["slow", "central", "fast"],
    )

    ch_row = frame[(frame["country"] == "CH") & (frame["scenario"] == "central")].iloc[0]
    de_row = frame[(frame["country"] == "DE") & (frame["scenario"] == "central")].iloc[0]
    assert len(frame) == 6
    assert len(features) == 6
    assert ch_row["demand_twh"] == 70.0
    assert ch_row["pv_gw"] == 10.0
    assert ch_row["battery_power_gw"] == 2.0
    assert de_row["demand_twh"] is pd.NA or pd.isna(de_row["demand_twh"])
    assert de_row["battery_power_gw"] is pd.NA or pd.isna(de_row["battery_power_gw"])
    assert "partial" in ch_row["quality_flag"]
    assert set(provenance["column"]).issuperset({"demand_twh", "hydro_twh", "ev_twh"})


def test_compose_partial_inventory_can_overlay_neighbor_demand_bridge(tmp_path):
    supply = tmp_path / "supply.parquet"
    ch = tmp_path / "ch.parquet"
    demand = tmp_path / "neighbor_demand.parquet"
    _supply(supply)
    _ch(ch)
    _neighbor_demand(demand)

    frame, _, provenance = compose_partial_inventory(
        tyndp_supply_path=supply,
        ch_ep2050_path=ch,
        neighbor_demand_path=demand,
        years=[2030],
        vintage="2026-06-11",
        countries=["CH", "DE"],
        scenarios=["slow", "central", "fast"],
    )

    de_row = frame[(frame["country"] == "DE") & (frame["scenario"] == "central")].iloc[0]
    assert de_row["demand_twh"] == 500.0
    assert de_row["peak_load_gw"] == 80.0
    assert de_row["winter_demand_twh"] == 260.0
    assert pd.isna(de_row["battery_power_gw"])
    assert "partial" in de_row["quality_flag"]
    assert "proxy" in de_row["quality_flag"]
    assert "TYNDP2024 neighbour demand bridge" in set(provenance["source"])

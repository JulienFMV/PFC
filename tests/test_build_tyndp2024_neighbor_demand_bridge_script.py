from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.build_tyndp2024_neighbor_demand_bridge import build_neighbor_demand_bridge


def _demand_output():
    df = pd.DataFrame(np.nan, index=range(6), columns=range(15), dtype=object)
    df.iloc[0, 10:15] = ["REF", "DE", "DE", "GA", "GA"]
    df.iloc[1, 0:15] = [
        "STUDY",
        "COUNTRY",
        "TYPE",
        "DASHBOARD_ID",
        "SECTOR",
        "SUBSECTOR",
        "ENERGY_CARRIER",
        "PARAMETER",
        "ENERGY_TYPE",
        "UNIT",
        2019,
        2040,
        2050,
        2040,
        2050,
    ]
    rows = [
        ("DE", "households_final_demand_electricity_energetic", 100.0, 140.0, 160.0),
        ("DE", "industry_final_demand_electricity_energetic", 200.0, 260.0, 220.0),
        ("FR", "households_final_demand_electricity_energetic", 50.0, 60.0, 80.0),
        ("DE", "other_gas", 999.0, 999.0, 999.0),
    ]
    for i, (country, dashboard, ref, de_2040, ga_2040) in enumerate(rows, start=2):
        df.iloc[i, 0:10] = [
            "TYNDP2024",
            country,
            "Output",
            dashboard,
            "Sector",
            "Total",
            "Electricity" if dashboard != "other_gas" else "Gas",
            "Energy demand",
            "Energetic",
            "TWh",
        ]
        df.iloc[i, 10] = ref
        df.iloc[i, 11] = de_2040
        df.iloc[i, 13] = ga_2040
    return df


def _entso():
    idx = pd.date_range("2025-01-01", "2026-01-01", freq="15min", inclusive="left", tz="UTC")
    hour = idx.hour.to_numpy()
    winter = ((idx.month <= 3) | (idx.month >= 10)).astype(float)
    return pd.DataFrame(
        {
            "load_de_mw": 50_000.0 + 5_000.0 * (hour >= 18) + 3_000.0 * winter,
            "load_fr_mw": 40_000.0 + 4_000.0 * (hour >= 18) + 2_000.0 * winter,
        },
        index=idx,
    )


def test_neighbor_demand_bridge_interpolates_and_keeps_proxy_flag():
    frame, diagnostics = build_neighbor_demand_bridge(
        demand_output=_demand_output(),
        entso=_entso(),
        countries=["DE", "FR"],
        vintage="2026-06-11",
        publication_date="2026-06-11",
        ingested_at_utc="2026-06-11",
    )

    de_slow = frame[(frame["country"] == "DE") & (frame["scenario"] == "slow")].iloc[0]
    de_fast = frame[(frame["country"] == "DE") & (frame["scenario"] == "fast")].iloc[0]
    assert len(frame) == 6
    assert set(frame["quality_flag"]) == {"internal_tyndp_demand_bridge_partial_proxy"}
    assert de_slow["demand_twh"] < de_fast["demand_twh"]
    assert de_slow["peak_load_gw"] > 0.0
    assert de_slow["winter_demand_twh"] > 0.0
    assert set(diagnostics["historical_year"]) == {2025}

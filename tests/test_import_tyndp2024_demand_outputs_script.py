from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.import_tyndp2024_demand_outputs import build_tyndp2024_demand_inventory


def _demand_output():
    df = pd.DataFrame(np.nan, index=range(8), columns=range(15), dtype=object)
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
        ("DE", "Households", 100.0, 110.0, 90.0, 95.0),
        ("DE", "Industry", 200.0, 220.0, 180.0, 190.0),
        ("FR", "Households", 50.0, 60.0, 40.0, 45.0),
        ("CH", "Households", 1.0, 1.0, 1.0, 1.0),
        ("DE", "Households", 999.0, 999.0, 999.0, 999.0),
    ]
    for idx, (country, sector, de_2040, de_2050, ga_2040, ga_2050) in enumerate(rows, start=2):
        df.iloc[idx, 0:10] = [
            "TYNDP2024",
            country,
            "Output",
            f"{sector.lower()}_final_demand_electricity_energetic",
            sector,
            "Total",
            "Electricity" if idx != 6 else "Gas",
            "Energy demand",
            "Energetic",
            "TWh",
        ]
        df.iloc[idx, 11:15] = [de_2040, de_2050, ga_2040, ga_2050]
    return df


def test_build_tyndp2024_demand_inventory_sums_official_demand_outputs():
    frame = build_tyndp2024_demand_inventory(
        _demand_output(),
        countries=["DE", "FR", "CH"],
        publication_date="2024-05-31",
        ingested_at_utc="2026-06-11",
    )

    de_2040 = frame[
        (frame["country"] == "DE")
        & (frame["scenario"] == "tyndp_distributed_energy")
        & (frame["delivery_year"] == 2040)
    ].iloc[0]
    fr_ga_2050 = frame[
        (frame["country"] == "FR")
        & (frame["scenario"] == "tyndp_global_ambition")
        & (frame["delivery_year"] == 2050)
    ].iloc[0]

    assert set(frame["country"]) == {"DE", "FR", "CH"}
    assert de_2040["demand_twh"] == 300.0
    assert fr_ga_2050["demand_twh"] == 45.0
    assert set(frame["quality_flag"]) == {"official_tyndp_demand_partial"}
    assert frame["pv_gw"].isna().all()


def test_build_tyndp2024_demand_inventory_preserves_all_nan_as_missing():
    df = _demand_output()
    de_mask = df.iloc[:, 1].astype(str).eq("DE")
    df.loc[de_mask, 11] = np.nan

    frame = build_tyndp2024_demand_inventory(
        df,
        countries=["DE"],
        publication_date="2024-05-31",
        ingested_at_utc="2026-06-11",
    )

    row = frame[
        (frame["country"] == "DE")
        & (frame["scenario"] == "tyndp_distributed_energy")
        & (frame["delivery_year"] == 2040)
    ].iloc[0]
    assert pd.isna(row["demand_twh"])

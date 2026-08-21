from __future__ import annotations

import pandas as pd

from pfc_shaping.data.electrification_scenarios import RECOMMENDED_COLUMNS
from scripts.bridge_lt_p0_structural_fields import build_p0_bridge


def test_p0_bridge_fills_traceable_fields_and_keeps_proxy_quality(tmp_path):
    idx = pd.date_range("2021-01-01", periods=8, freq="15min", tz="UTC")
    entso = pd.DataFrame({"load_mw": [1000, 1100, 1200, 1300, 1400, 1200, 1100, 1000]}, index=idx)
    entso_path = tmp_path / "entso.parquet"
    entso.to_parquet(entso_path)

    row = {column: pd.NA for column in RECOMMENDED_COLUMNS}
    row.update(
        {
            "publication_date": pd.Timestamp("2026-01-01", tz="UTC"),
            "measurement_date": pd.NaT,
            "track": "scenario",
            "source": "unit",
            "component_ids": "unit",
            "scenario_edition": "unit",
            "scenario": "central",
            "country": "CH",
            "delivery_year": 2030,
            "delivery_month": None,
            "scenario_weight": 1.0,
            "quality_flag": "official_partial_proxy",
            "ingested_at_utc": pd.Timestamp("2026-01-02", tz="UTC"),
            "demand_twh": 87.6,
            "pv_gw": 10.0,
            "wind_gw": 2.0,
            "battery_energy_gwh": 8.0,
            "nuclear_gw": 1.0,
            "peak_load_gw": pd.NA,
            "winter_demand_twh": pd.NA,
            "pv_twh": pd.NA,
            "wind_twh": pd.NA,
            "battery_power_gw": pd.NA,
            "dispatchable_gw": pd.NA,
            "ntc_ch_de_gw": pd.NA,
        }
    )
    frame = pd.DataFrame(
        [
            row
        ]
    )

    out, changes = build_p0_bridge(
        frame,
        entso_path=entso_path,
        demand_workbook=tmp_path / "missing.xlsb",
        countries=["CH"],
    )
    row = out.iloc[0]

    assert float(row["peak_load_gw"]) > 0
    assert float(row["winter_demand_twh"]) > 0
    assert float(row["pv_twh"]) > 0
    assert float(row["wind_twh"]) > 0
    assert float(row["battery_power_gw"]) == 2.0
    assert float(row["dispatchable_gw"]) == 1.0
    assert pd.isna(row["ntc_ch_de_gw"])
    assert "p0_structural_bridge_proxy" in row["quality_flag"]
    assert "internal_p0_structural_bridge" in row["component_ids"]
    assert set(changes["column"]) >= {
        "peak_load_gw",
        "winter_demand_twh",
        "pv_twh",
        "wind_twh",
        "battery_power_gw",
        "dispatchable_gw",
    }

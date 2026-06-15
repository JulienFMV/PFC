from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.import_tyndp2024_supply_inputs import build_tyndp2024_supply_inventory
from scripts.validate_electrification_scenarios import main as validate_main


def _sheet(code_col: int, value_cols: dict[int, dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(np.nan, index=range(14), columns=range(16), dtype=object)
    for row, code in enumerate(["CH00", "DE00", "FR00", "AT00", "ITN1", "ITSI"], start=4):
        df.iat[row, code_col] = code
        for col, by_code in value_cols.items():
            df.iat[row, col] = by_code.get(code, 0.0)
    return df


def _write_workbook(path):
    mw_values = {
        2: {"CH00": 1000.0, "ITN1": 2000.0, "ITSI": 3000.0},
        5: {"CH00": 1500.0, "ITN1": 2500.0, "ITSI": 3500.0},
        7: {"CH00": 2000.0, "ITN1": 3000.0, "ITSI": 4000.0},
    }
    battery_values = {
        3: {"CH00": 100.0, "ITN1": 200.0, "ITSI": 300.0},
        6: {"CH00": 150.0, "ITN1": 250.0, "ITSI": 350.0},
        8: {"CH00": 200.0, "ITN1": 300.0, "ITSI": 400.0},
    }
    fuel = pd.DataFrame(np.nan, index=range(12), columns=range(9), dtype=object)
    fuel.iat[2, 1] = "Fuel"
    fuel.iat[2, 3] = 2030
    fuel.iat[3, 1] = "Hard coal"
    fuel.iat[3, 3] = 2.0
    fuel.iat[4, 1] = "Natural Gas"
    fuel.iat[4, 3] = 10.0
    fuel.iat[5, 1] = "CO2 price"
    fuel.iat[5, 3] = 100.0
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _sheet(1, mw_values).to_excel(writer, sheet_name="1.1.", header=False, index=False)
        _sheet(1, mw_values).to_excel(writer, sheet_name="1.2.", header=False, index=False)
        _sheet(1, mw_values).to_excel(writer, sheet_name="1.3.", header=False, index=False)
        _sheet(1, mw_values).to_excel(writer, sheet_name="1.4.", header=False, index=False)
        _sheet(2, battery_values).to_excel(writer, sheet_name="1.5.", header=False, index=False)
        _sheet(2, battery_values).to_excel(writer, sheet_name="1.6.", header=False, index=False)
        fuel.to_excel(writer, sheet_name="3.1.", header=False, index=False)


def test_build_tyndp2024_supply_inventory_extracts_country_scenario_rows(tmp_path):
    workbook = tmp_path / "supply.xlsx"
    _write_workbook(workbook)

    frame = build_tyndp2024_supply_inventory(
        workbook,
        countries=["CH", "IT"],
        scenarios=["slow", "central", "fast"],
        years=[2030],
        publication_date="2024-05-31",
        ingested_at_utc="2026-06-11",
    )

    assert len(frame) == 6
    ch_slow = frame[(frame["country"] == "CH") & (frame["scenario"] == "slow")].iloc[0]
    it_fast = frame[(frame["country"] == "IT") & (frame["scenario"] == "fast")].iloc[0]
    assert ch_slow["pv_gw"] == 1.0
    assert ch_slow["wind_gw"] == 2.0
    assert ch_slow["battery_energy_gwh"] == 0.2
    assert ch_slow["gas_eur_mwh"] == 36.0
    assert it_fast["pv_gw"] == 7.0
    assert it_fast["battery_energy_gwh"] == 1.4
    assert set(frame["quality_flag"]) == {"official_tyndp_supply_partial"}


def test_tyndp2024_supply_inventory_does_not_pass_production_gate(tmp_path):
    workbook = tmp_path / "supply.xlsx"
    path = tmp_path / "tyndp_supply.parquet"
    report = tmp_path / "gate.md"
    _write_workbook(workbook)
    frame = build_tyndp2024_supply_inventory(
        workbook,
        countries=["CH", "DE"],
        scenarios=["slow", "central", "fast"],
        years=[2030],
        publication_date="2024-05-31",
        ingested_at_utc="2026-06-11",
    )
    frame.to_parquet(path, index=False)

    rc = validate_main([
        "--path",
        str(path),
        "--vintage",
        "2026-06-11",
        "--years",
        "2030",
        "--required-countries",
        "CH,DE",
        "--require-production",
        "--report",
        str(report),
    ])

    text = report.read_text(encoding="utf-8")
    assert rc == 2
    assert "official_tyndp_supply_partial" in text
    assert "missing_critical_production_values" in text

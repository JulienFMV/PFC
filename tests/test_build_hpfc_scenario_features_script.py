from __future__ import annotations

import pandas as pd

from scripts.build_hpfc_scenario_features import main


def test_build_hpfc_scenario_features_script_writes_gold_table(tmp_path):
    path = tmp_path / "scenarios.parquet"
    output = tmp_path / "features.parquet"
    pd.DataFrame(
        [
            {
                "publication_date": pd.Timestamp("2025-12-01", tz="UTC"),
                "source": "unit",
                "scenario": "central",
                "country": "CH",
                "delivery_year": 2030,
                "demand_twh": 87.6,
                "pv_gw": 10.0,
                "wind_gw": 5.0,
                "battery_power_gw": 2.0,
                "battery_energy_gwh": 20.0,
                "ev_twh": 5.0,
                "heatpump_twh": 4.0,
                "hydro_twh": 35.0,
                "hydro_capacity_gw": 10.0,
                "hydro_reservoir_twh": 8.0,
            },
            {
                "publication_date": pd.Timestamp("2026-02-01", tz="UTC"),
                "source": "future",
                "scenario": "central",
                "country": "CH",
                "delivery_year": 2030,
                "demand_twh": 99.0,
                "pv_gw": 99.0,
            },
        ]
    ).to_parquet(path)

    rc = main([
        "--path", str(path),
        "--output", str(output),
        "--vintage", "2026-01-15",
    ])

    assert rc == 0
    out = pd.read_parquet(output)
    assert len(out) == 1
    assert list(out["source"]) == ["unit"]
    assert "pv_penetration" in out.columns
    assert "hydro_flexibility" in out.columns

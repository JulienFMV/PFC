from __future__ import annotations

import pandas as pd
import pytest

from pfc_shaping.data.electrification_scenarios import RECOMMENDED_COLUMNS, validate_electrification_scenarios
from scripts.enrich_electrification_scenarios import enrich_scenarios, main


def _frame():
    return pd.DataFrame(
        [
            {
                "publication_date": pd.Timestamp("2022-10-01", tz="UTC"),
                "source": "OFEN_EP2050_hourly_11142",
                "scenario": "ZERO_Basis",
                "country": "CH",
                "delivery_year": 2030,
                "delivery_month": None,
                "quality_flag": "official",
                "demand_twh": 70.0,
                "peak_load_gw": 10.0,
                "pv_twh": 9.0,
                "pv_gw": 8.0,
                "wind_twh": 0.6,
                "wind_gw": 0.4,
                "hydro_twh": 42.0,
                "hydro_capacity_gw": 9.5,
                "nuclear_gw": 1.0,
                "ev_twh": 2.0,
                "heatpump_twh": 1.5,
                "import_twh": 20.0,
                "export_twh": 12.0,
                "net_import_twh": 8.0,
            }
        ]
    )


def test_enrich_scenarios_fills_recommended_columns_without_overwriting_official_fields():
    out = enrich_scenarios(
        _frame(),
        assumption_publication_date="2026-06-05",
    )
    row = out.iloc[0]

    assert validate_electrification_scenarios(out, require_recommended=True).equals(out)
    assert RECOMMENDED_COLUMNS <= set(out.columns)
    assert float(row["demand_twh"]) == pytest.approx(70.0)
    assert float(row["pv_twh"]) == pytest.approx(9.0)
    assert float(row["battery_power_gw"]) == pytest.approx(2.0)
    assert float(row["battery_energy_gwh"]) == pytest.approx(5.0)
    assert float(row["managed_charging_share"]) == pytest.approx(0.35)
    assert float(row["scenario_weight"]) == pytest.approx(1.0)
    assert "proxy_enriched" in row["quality_flag"]
    assert "ch_first_pfc_proxy_v0" in row["source"]
    assert str(out["publication_date"].dt.tz) == "UTC"
    assert row["publication_date"] == pd.Timestamp("2026-06-05", tz="UTC")


def test_enrich_scenarios_uses_hourly_winter_demand_when_available(tmp_path):
    hourly = pd.DataFrame(
        [
            {
                "scenario": "ZERO_Basis",
                "year": 2030,
                "datetime": pd.Timestamp("2030-01-01 00:00", tz="Europe/Zurich"),
                "technology": "total_consumption",
                "gwh": 1.0,
            },
            {
                "scenario": "ZERO_Basis",
                "year": 2030,
                "datetime": pd.Timestamp("2030-07-01 00:00", tz="Europe/Zurich"),
                "technology": "total_consumption",
                "gwh": 100.0,
            },
        ]
    )
    hourly_path = tmp_path / "hourly.parquet"
    hourly.to_parquet(hourly_path, index=False)

    out = enrich_scenarios(
        _frame(),
        assumption_publication_date="2026-06-05",
        hourly_path=hourly_path,
    )

    assert float(out.iloc[0]["winter_demand_twh"]) == pytest.approx(0.001)


def test_enrichment_script_writes_require_recommended_table(tmp_path):
    src = tmp_path / "src.parquet"
    dst = tmp_path / "dst.parquet"
    _frame().to_parquet(src, index=False)

    rc = main([
        "--input", str(src),
        "--output", str(dst),
        "--assumption-publication-date", "2026-06-05",
    ])

    assert rc == 0
    out = pd.read_parquet(dst)
    validate_electrification_scenarios(out, require_recommended=True)

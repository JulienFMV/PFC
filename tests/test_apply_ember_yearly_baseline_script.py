from __future__ import annotations

import pandas as pd

from pfc_shaping.data.electrification_scenarios import RECOMMENDED_COLUMNS
from scripts.apply_ember_yearly_baseline import (
    COMPONENT_ID,
    apply_ember_baseline,
    load_ember_yearly_baseline,
)


def _scenario_frame() -> pd.DataFrame:
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
            "country": "DE",
            "delivery_year": 2030,
            "delivery_month": None,
            "scenario_weight": 1.0,
            "quality_flag": "official_partial_proxy",
            "ingested_at_utc": pd.Timestamp("2026-01-02", tz="UTC"),
            "demand_twh": 500.0,
            "hydro_twh": pd.NA,
            "hydro_capacity_gw": pd.NA,
            "net_import_twh": pd.NA,
            "import_twh": pd.NA,
            "export_twh": pd.NA,
            "gas_gw": pd.NA,
            "coal_gw": pd.NA,
            "dispatchable_gw": pd.NA,
        }
    )
    return pd.DataFrame([row])


def test_load_ember_yearly_baseline_uses_latest_reported_values_and_skips_zero(tmp_path):
    csv_path = tmp_path / "ember.csv"
    pd.DataFrame(
        [
            ["Germany", "DEU", 2024, "Capacity", "Fuel", "Hydro", "GW", 5.0],
            ["Germany", "DEU", 2025, "Capacity", "Fuel", "Hydro", "GW", 6.0],
            ["Germany", "DEU", 2025, "Electricity generation", "Fuel", "Hydro", "TWh", 20.0],
            ["Germany", "DEU", 2025, "Electricity imports", "Electricity imports", "Net Imports", "TWh", 3.0],
            ["Germany", "DEU", 2025, "Capacity", "Fuel", "Gas", "GW", 30.0],
            ["Germany", "DEU", 2025, "Capacity", "Fuel", "Coal", "GW", 0.0],
            ["Germany", "DEU", 2025, "Capacity", "Fuel", "Nuclear", "GW", 0.0],
        ],
        columns=["Area", "ISO 3 code", "Year", "Category", "Subcategory", "Variable", "Unit", "Value"],
    ).to_csv(csv_path, index=False)

    baseline = load_ember_yearly_baseline(csv_path, countries=["DE"])
    values = baseline.set_index("field")["value"].to_dict()

    assert values["hydro_capacity_gw"] == 6.0
    assert values["hydro_twh"] == 20.0
    assert values["net_import_twh"] == 3.0
    assert values["gas_gw"] == 30.0
    assert "coal_gw" not in values
    assert values["dispatchable_gw"] == 30.0


def test_apply_ember_baseline_does_not_invent_gross_cross_border_fields():
    baseline = pd.DataFrame(
        [
            {
                "country": "DE",
                "field": "net_import_twh",
                "value": 3.0,
                "source_year": 2025,
                "method": "unit",
            },
            {
                "country": "DE",
                "field": "hydro_twh",
                "value": 20.0,
                "source_year": 2025,
                "method": "unit",
            },
        ]
    )

    out, changes = apply_ember_baseline(_scenario_frame(), baseline)
    row = out.iloc[0]

    assert float(row["net_import_twh"]) == 3.0
    assert float(row["hydro_twh"]) == 20.0
    assert pd.isna(row["import_twh"])
    assert pd.isna(row["export_twh"])
    assert COMPONENT_ID in row["component_ids"]
    assert "ember_yearly_baseline_proxy" in row["quality_flag"]
    assert sorted(changes["column"].tolist()) == ["hydro_twh", "net_import_twh"]

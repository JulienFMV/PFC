from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_lt_data_contract import main, materialize_scenario_contract


def _scenario_source(path):
    rows = []
    for year in [2025, 2030, 2035]:
        for scenario, pv_twh in [("slow", 6.0), ("central", 8.0), ("fast", 10.0)]:
            rows.append(
                {
                    "publication_date": pd.Timestamp("2026-06-05", tz="UTC"),
                    "source": "unit",
                    "scenario": scenario,
                    "country": "CH",
                    "delivery_year": year,
                    "delivery_month": np.nan,
                    "scenario_weight": {"slow": 0.25, "central": 0.50, "fast": 0.25}[scenario],
                    "quality_flag": "official_proxy_enriched",
                    "demand_twh": 70.0 + 0.5 * (year - 2025),
                    "peak_load_gw": 15.0,
                    "winter_demand_twh": 20.0,
                    "pv_gw": pv_twh / 1.5,
                    "pv_twh": pv_twh,
                    "wind_gw": 0.2,
                    "wind_twh": 0.4,
                    "battery_power_gw": 2.0,
                    "battery_energy_gwh": 5.0,
                    "ev_twh": 0.2,
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
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_materialize_lt_data_contract_writes_canonical_files(tmp_path):
    source = tmp_path / "source.parquet"
    scenario_output = tmp_path / "electrification_scenarios.parquet"
    features_output = tmp_path / "hpfc_scenario_features.parquet"
    local_root = tmp_path / "local"
    report = tmp_path / "report.md"
    _scenario_source(source)

    rc = main(
        [
            "--local-root",
            str(local_root),
            "--scenario-source",
            str(source),
            "--scenario-output",
            str(scenario_output),
            "--features-output",
            str(features_output),
            "--years",
            "2027-2031",
            "--report",
            str(report),
        ]
    )

    assert rc == 0
    scenarios = pd.read_parquet(scenario_output)
    features = pd.read_parquet(features_output)
    assert len(scenarios) == 15
    assert len(features) == 15
    assert set(scenarios["scenario"]) == {"slow", "central", "fast"}
    assert (local_root / "entsoe" / "raw_xml").exists()
    text = report.read_text(encoding="utf-8")
    assert "production complete: `NO - local proxy only`" in text
    assert "TYNDP 2024 multi-country scenarios" in text


def test_materialize_scenario_contract_requires_requested_coverage(tmp_path):
    source = tmp_path / "source.parquet"
    _scenario_source(source)

    with pytest.raises(ValueError, match="bracketing"):
        materialize_scenario_contract(
            source_path=source,
            scenario_output=tmp_path / "out.parquet",
            features_output=tmp_path / "features.parquet",
            years=[2036],
            vintage="2026-06-05",
            country="CH",
            scenarios=["slow", "central", "fast"],
        )

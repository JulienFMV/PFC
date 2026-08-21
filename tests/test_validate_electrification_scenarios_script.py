from __future__ import annotations

import pandas as pd

from scripts.validate_electrification_scenarios import main


def _inventory(path):
    rows = []
    for scenario in ["slow", "central", "fast"]:
        rows.append(
            {
                "publication_date": pd.Timestamp("2025-12-01", tz="UTC"),
                "source": "unit",
                "scenario": scenario,
                "country": "CH",
                "delivery_year": 2030,
                "delivery_month": None,
                "scenario_weight": 1.0 / 3.0,
                "quality_flag": "official",
                "demand_twh": 60.0,
                "pv_gw": 20.0,
                "battery_power_gw": 3.0,
                "battery_energy_gwh": 10.0,
            }
        )
    pd.DataFrame(rows).to_parquet(path)


def _production_inventory(path):
    rows = []
    for country in ["CH", "DE"]:
        for scenario, weight in [("slow", 0.25), ("central", 0.50), ("fast", 0.25)]:
            rows.append(
                {
                    "publication_date": pd.Timestamp("2025-12-01", tz="UTC"),
                    "measurement_date": pd.NaT,
                    "track": "scenario",
                    "source": "tyndp_unit",
                    "scenario_edition": "unit-2026",
                    "scenario": scenario,
                    "country": country,
                    "delivery_year": 2030,
                    "delivery_month": None,
                    "scenario_weight": weight,
                    "quality_flag": "official",
                    "ingested_at_utc": pd.Timestamp("2026-01-01", tz="UTC"),
                    "demand_twh": 60.0,
                    "peak_load_gw": 10.0,
                    "winter_demand_twh": 18.0,
                    "pv_gw": 20.0,
                    "pv_twh": 10.0,
                    "wind_gw": 5.0,
                    "wind_twh": 8.0,
                    "battery_power_gw": 3.0,
                    "battery_energy_gwh": 10.0,
                    "ev_twh": 2.0,
                    "heatpump_twh": 3.0,
                    "managed_charging_share": 0.4,
                    "hydro_twh": 35.0,
                    "hydro_capacity_gw": 12.0,
                    "hydro_reservoir_twh": 4.0,
                    "nuclear_gw": 1.0,
                    "dispatchable_gw": 2.0,
                    "gas_gw": 1.0,
                    "coal_gw": 0.5,
                    "import_twh": 8.0,
                    "export_twh": 4.0,
                    "net_import_twh": 4.0,
                    "ntc_ch_de_gw": 4.0,
                    "ntc_ch_fr_gw": 4.0,
                    "ntc_ch_it_gw": 3.0,
                    "ntc_ch_at_gw": 1.0,
                    "gas_eur_mwh": 30.0,
                    "coal_eur_mwh": 12.0,
                    "co2_eur_t": 80.0,
                    "electrolysis_twh": 1.0,
                    "p2x_gw": 0.5,
                    "dsm_gw": 0.3,
                }
            )
    pd.DataFrame(rows).to_parquet(path)


def test_validate_scenario_inventory_script_ok(tmp_path):
    path = tmp_path / "scenarios.parquet"
    report = tmp_path / "report.md"
    _inventory(path)

    rc = main([
        "--path", str(path),
        "--vintage", "2026-01-15",
        "--years", "2030",
        "--report", str(report),
    ])

    assert rc == 0
    assert "coverage: `OK`" in report.read_text(encoding="utf-8")


def test_validate_scenario_inventory_script_fails_on_missing_coverage(tmp_path):
    path = tmp_path / "scenarios.parquet"
    report = tmp_path / "report.md"
    _inventory(path)

    rc = main([
        "--path", str(path),
        "--vintage", "2026-01-15",
        "--years", "2031",
        "--report", str(report),
    ])

    assert rc == 2
    assert "coverage: `FAILED`" in report.read_text(encoding="utf-8")


def test_validate_scenario_inventory_script_production_gate_ok(tmp_path):
    path = tmp_path / "scenarios.parquet"
    report = tmp_path / "report.md"
    _production_inventory(path)

    rc = main([
        "--path", str(path),
        "--vintage", "2026-01-15",
        "--years", "2030",
        "--required-countries", "CH,DE",
        "--require-production",
        "--report", str(report),
    ])

    text = report.read_text(encoding="utf-8")
    assert rc == 0
    assert "production gate: `OK`" in text


def test_validate_scenario_inventory_script_production_gate_requires_country_scenario_years(tmp_path):
    path = tmp_path / "scenarios.parquet"
    report = tmp_path / "report.md"
    _production_inventory(path)
    df = pd.read_parquet(path)
    df = df[~((df["country"] == "DE") & (df["scenario"] == "fast"))]
    df.to_parquet(path, index=False)

    rc = main([
        "--path", str(path),
        "--vintage", "2026-01-15",
        "--years", "2030",
        "--required-countries", "CH,DE",
        "--require-production",
        "--report", str(report),
    ])

    text = report.read_text(encoding="utf-8")
    assert rc == 2
    assert "production gate: `FAILED`" in text
    assert "missing_required_country_scenario_years" in text
    assert "DE" in text
    assert "fast" in text


def test_validate_scenario_inventory_script_production_gate_rejects_missing_critical_values(tmp_path):
    path = tmp_path / "scenarios.parquet"
    report = tmp_path / "report.md"
    _production_inventory(path)
    df = pd.read_parquet(path)
    df.loc[0, "demand_twh"] = pd.NA
    df.to_parquet(path, index=False)

    rc = main([
        "--path", str(path),
        "--vintage", "2026-01-15",
        "--years", "2030",
        "--required-countries", "CH,DE",
        "--require-production",
        "--report", str(report),
    ])

    text = report.read_text(encoding="utf-8")
    assert rc == 2
    assert "missing_critical_production_values" in text
    assert "demand_twh" in text


def test_validate_scenario_inventory_script_production_gate_uses_latest_asof_rows(tmp_path):
    path = tmp_path / "scenarios.parquet"
    report = tmp_path / "report.md"
    _production_inventory(path)
    df = pd.read_parquet(path)
    older = df.copy()
    older["publication_date"] = pd.Timestamp("2025-01-01", tz="UTC")
    older["demand_twh"] = float("nan")
    combined = pd.concat([older, df], ignore_index=True)
    combined.to_parquet(path, index=False)

    rc = main([
        "--path", str(path),
        "--vintage", "2026-01-15",
        "--years", "2030",
        "--required-countries", "CH,DE",
        "--require-production",
        "--report", str(report),
    ])

    text = report.read_text(encoding="utf-8")
    assert rc == 0
    assert "production gate: `OK`" in text
    assert "missing_critical_production_values" not in text


def test_validate_scenario_inventory_script_production_gate_rejects_proxy(tmp_path):
    path = tmp_path / "scenarios.parquet"
    report = tmp_path / "report.md"
    _inventory(path)
    df = pd.read_parquet(path)
    df["quality_flag"] = "official_proxy_enriched"
    df.to_parquet(path, index=False)

    rc = main([
        "--path", str(path),
        "--vintage", "2026-01-15",
        "--years", "2030",
        "--require-production",
        "--report", str(report),
    ])

    text = report.read_text(encoding="utf-8")
    assert rc == 2
    assert "production gate: `FAILED`" in text
    assert "missing_production_columns" in text
    assert "unacceptable_quality_flags" in text

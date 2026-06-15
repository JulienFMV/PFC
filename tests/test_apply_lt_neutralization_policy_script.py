from __future__ import annotations

import pandas as pd

from scripts.apply_lt_neutralization_policy import SOURCE_ID, apply_neutralization_policy, main


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "publication_date": pd.Timestamp("2026-01-01", tz="UTC"),
                "measurement_date": pd.NaT,
                "track": "scenario",
                "source": "unit",
                "component_ids": "unit",
                "scenario_edition": "unit-2030",
                "scenario": "central",
                "country": "CH",
                "delivery_year": 2030,
                "delivery_month": None,
                "scenario_weight": 1.0,
                "quality_flag": "official_partial",
                "ingested_at_utc": pd.Timestamp("2026-01-02", tz="UTC"),
                "demand_twh": 60.0,
                "peak_load_gw": 10.0,
                "winter_demand_twh": 20.0,
                "pv_gw": 20.0,
                "pv_twh": 18.0,
                "wind_gw": 8.0,
                "wind_twh": 12.0,
                "battery_power_gw": 3.0,
                "battery_energy_gwh": 10.0,
                "ev_twh": 2.0,
                "heatpump_twh": 4.0,
                "managed_charging_share": pd.NA,
                "hydro_twh": 35.0,
                "hydro_capacity_gw": 12.0,
                "hydro_reservoir_twh": 4.0,
                "nuclear_gw": 1.0,
                "dispatchable_gw": 5.0,
                "gas_gw": 2.0,
                "coal_gw": pd.NA,
                "import_twh": pd.NA,
                "export_twh": pd.NA,
                "net_import_twh": 4.0,
                "ntc_ch_de_gw": 4.0,
                "ntc_ch_fr_gw": 4.0,
                "ntc_ch_it_gw": 3.0,
                "ntc_ch_at_gw": 1.0,
                "gas_eur_mwh": 30.0,
                "coal_eur_mwh": 12.0,
                "co2_eur_t": 80.0,
                "electrolysis_twh": pd.NA,
                "p2x_gw": pd.NA,
                "dsm_gw": pd.NA,
            },
            {
                "publication_date": pd.Timestamp("2026-01-01", tz="UTC"),
                "measurement_date": pd.NaT,
                "track": "scenario",
                "source": "unit",
                "component_ids": "unit",
                "scenario_edition": "unit-2030",
                "scenario": "central",
                "country": "DE",
                "delivery_year": 2030,
                "delivery_month": None,
                "scenario_weight": 1.0,
                "quality_flag": "official_partial",
                "ingested_at_utc": pd.Timestamp("2026-01-02", tz="UTC"),
                "demand_twh": 60.0,
                "peak_load_gw": 10.0,
                "winter_demand_twh": 20.0,
                "pv_gw": 20.0,
                "pv_twh": 18.0,
                "wind_gw": 8.0,
                "wind_twh": 12.0,
                "battery_power_gw": 3.0,
                "battery_energy_gwh": 10.0,
                "ev_twh": 2.0,
                "heatpump_twh": 4.0,
                "managed_charging_share": pd.NA,
                "hydro_twh": 35.0,
                "hydro_capacity_gw": 12.0,
                "hydro_reservoir_twh": pd.NA,
                "nuclear_gw": 1.0,
                "dispatchable_gw": 5.0,
                "gas_gw": 2.0,
                "coal_gw": 9.0,
                "import_twh": pd.NA,
                "export_twh": pd.NA,
                "net_import_twh": 4.0,
                "ntc_ch_de_gw": 4.0,
                "ntc_ch_fr_gw": 4.0,
                "ntc_ch_it_gw": 3.0,
                "ntc_ch_at_gw": 1.0,
                "gas_eur_mwh": 30.0,
                "coal_eur_mwh": 12.0,
                "co2_eur_t": 80.0,
                "electrolysis_twh": pd.NA,
                "p2x_gw": pd.NA,
                "dsm_gw": pd.NA,
            },
        ]
    )


def test_apply_neutralization_policy_fills_only_missing_fields_with_justification():
    out, audit = apply_neutralization_policy(_frame())

    ch = out[out["country"] == "CH"].iloc[0]
    de = out[out["country"] == "DE"].iloc[0]
    assert float(ch["coal_gw"]) == 0.0
    assert float(de["coal_gw"]) == 9.0
    assert pd.isna(de.get("coal_gw_zero_justification"))
    assert str(ch["coal_gw_zero_justification"]).strip()
    assert float(ch["managed_charging_share"]) == 0.0
    assert str(ch["managed_charging_share_zero_justification"]).strip()
    assert SOURCE_ID in str(ch["component_ids"])
    assert set(audit["field"]) == {"coal_gw", "dsm_gw", "electrolysis_twh", "managed_charging_share", "p2x_gw"}


def test_apply_lt_neutralization_policy_script_writes_outputs(tmp_path):
    input_path = tmp_path / "inventory.parquet"
    output_path = tmp_path / "neutralized.parquet"
    features_path = tmp_path / "features.parquet"
    audit_path = tmp_path / "audit.csv"
    report_path = tmp_path / "report.md"
    _frame().to_parquet(input_path, index=False)

    rc = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--feature-output",
            str(features_path),
            "--audit-output",
            str(audit_path),
            "--report",
            str(report_path),
        ]
    )

    assert rc == 0
    assert output_path.exists()
    assert features_path.exists()
    assert audit_path.exists()
    assert "LT Neutralization Policy" in report_path.read_text(encoding="utf-8")

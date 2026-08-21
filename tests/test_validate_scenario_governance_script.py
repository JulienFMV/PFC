from __future__ import annotations

import yaml
import pandas as pd

from scripts.validate_scenario_governance import main


def _inventory(path, *, quality_flag="official_governed"):
    rows = []
    for country in ["CH", "DE"]:
        for scenario, weight in [("slow", 0.25), ("central", 0.50), ("fast", 0.25)]:
            rows.append(
                {
                    "publication_date": pd.Timestamp("2026-01-01", tz="UTC"),
                    "measurement_date": pd.NaT,
                    "track": "scenario",
                    "source": "unit",
                    "component_ids": "unit",
                    "scenario_edition": "unit-2030",
                    "scenario": scenario,
                    "country": country,
                    "delivery_year": 2030,
                    "delivery_month": None,
                    "scenario_weight": weight,
                    "quality_flag": quality_flag,
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
                    "managed_charging_share": 0.4,
                    "hydro_twh": 35.0,
                    "hydro_capacity_gw": 12.0,
                    "hydro_reservoir_twh": 4.0,
                    "nuclear_gw": 1.0,
                    "dispatchable_gw": 5.0,
                    "gas_gw": 2.0,
                    "coal_gw": 1.0,
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
    pd.DataFrame(rows).to_parquet(path, index=False)


def _manifest(path, *, approved=True):
    payload = {
        "governance_version": 1,
        "decision_id": "unit",
        "status": "approved" if approved else "draft",
        "approved_for_production": approved,
        "approval_date": "2026-01-03" if approved else None,
        "approved_by": ["risk-committee"] if approved else [],
        "scope": {
            "countries": ["CH", "DE"],
            "scenarios": ["slow", "central", "fast"],
            "years": [2030],
        },
        "scenario_weights": {"slow": 0.25, "central": 0.50, "fast": 0.25},
        "source_components": [
            {
                "source_id": "unit",
                "local_path": str(path.parent / "inventory.parquet"),
                "publication_date": "2026-01-01",
            }
        ],
        "rules": {
            "allow_proxy_or_partial_quality_flags": False,
            "require_complete_critical_values": True,
            "require_source_component_links": True,
            "require_zero_justification": True,
            "zero_allowed_fields": ["coal_gw"],
            "physical_bounds": {
                "managed_charging_share": {"min": 0.0, "max": 1.0},
                "ntc_ch_de_gw": {"min": 0.0, "max": 30.0},
            },
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _local_test_manifest(path):
    payload = {
        "governance_version": 1,
        "decision_id": "unit-local-test",
        "status": "agent_approved_local_test",
        "approved_for_production": False,
        "approved_for_local_test": True,
        "approval_date": "2026-01-03",
        "approved_by": [],
        "local_test_approved_by_agents": [
            {"role": "quant_scenario_reviewer", "verdict": "no_go_production_ok_local_test"},
            {"role": "data_engineering_reviewer", "verdict": "no_go_production_ok_local_test"},
            {"role": "model_risk_reviewer", "verdict": "no_go_production_ok_local_test"},
        ],
        "scope": {
            "countries": ["CH", "DE"],
            "scenarios": ["slow", "central", "fast"],
            "years": [2030],
        },
        "scenario_weights": {"slow": 0.25, "central": 0.50, "fast": 0.25},
        "source_components": [
            {
                "source_id": "unit",
                "local_path": str(path.parent / "inventory.parquet"),
                "publication_date": "2026-01-01",
            }
        ],
        "rules": {
            "allow_proxy_or_partial_quality_flags": True,
            "require_complete_critical_values": True,
            "require_source_component_links": True,
            "require_zero_justification": True,
            "zero_allowed_fields": ["coal_gw"],
            "physical_bounds": {
                "managed_charging_share": {"min": 0.0, "max": 1.0},
                "ntc_ch_de_gw": {"min": 0.0, "max": 30.0},
            },
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_validate_scenario_governance_ok_for_approved_complete_inventory(tmp_path):
    inventory = tmp_path / "inventory.parquet"
    manifest = tmp_path / "manifest.yaml"
    report = tmp_path / "report.md"
    _inventory(inventory)
    _manifest(manifest, approved=True)

    rc = main([
        "--inventory", str(inventory),
        "--manifest", str(manifest),
        "--vintage", "2026-01-15",
        "--countries", "CH,DE",
        "--scenarios", "slow,central,fast",
        "--years", "2030",
        "--report", str(report),
    ])

    text = report.read_text(encoding="utf-8")
    assert rc == 0
    assert "governance gate: `OK`" in text


def test_validate_scenario_governance_rejects_draft_proxy_inventory(tmp_path):
    inventory = tmp_path / "inventory.parquet"
    manifest = tmp_path / "manifest.yaml"
    report = tmp_path / "report.md"
    _inventory(inventory, quality_flag="official_partial_proxy")
    _manifest(manifest, approved=False)

    rc = main([
        "--inventory", str(inventory),
        "--manifest", str(manifest),
        "--vintage", "2026-01-15",
        "--countries", "CH,DE",
        "--scenarios", "slow,central,fast",
        "--years", "2030",
        "--report", str(report),
    ])

    text = report.read_text(encoding="utf-8")
    assert rc == 2
    assert "manifest_not_approved" in text
    assert "unacceptable_quality_flags" in text


def test_validate_scenario_governance_accepts_agent_approved_local_test_proxy(tmp_path):
    inventory = tmp_path / "inventory.parquet"
    manifest = tmp_path / "manifest.yaml"
    report = tmp_path / "report.md"
    _inventory(inventory, quality_flag="official_partial_proxy")
    _local_test_manifest(manifest)

    rc = main([
        "--inventory", str(inventory),
        "--manifest", str(manifest),
        "--vintage", "2026-01-15",
        "--countries", "CH,DE",
        "--scenarios", "slow,central,fast",
        "--years", "2030",
        "--mode", "local-test",
        "--report", str(report),
    ])

    text = report.read_text(encoding="utf-8")
    assert rc == 0
    assert "mode: `local-test`" in text
    assert "governance gate: `OK`" in text


def test_validate_scenario_governance_rejects_missing_production_columns(tmp_path):
    inventory = tmp_path / "inventory.parquet"
    manifest = tmp_path / "manifest.yaml"
    report = tmp_path / "report.md"
    _inventory(inventory)
    frame = pd.read_parquet(inventory).drop(columns=["ntc_ch_at_gw"])
    frame.to_parquet(inventory, index=False)
    _manifest(manifest, approved=True)

    rc = main([
        "--inventory", str(inventory),
        "--manifest", str(manifest),
        "--vintage", "2026-01-15",
        "--countries", "CH,DE",
        "--scenarios", "slow,central,fast",
        "--years", "2030",
        "--report", str(report),
    ])

    text = report.read_text(encoding="utf-8")
    assert rc == 2
    assert "missing_production_columns" in text
    assert "ntc_ch_at_gw" in text


def test_validate_scenario_governance_accepts_net_import_without_gross_flows(tmp_path):
    inventory = tmp_path / "inventory.parquet"
    manifest = tmp_path / "manifest.yaml"
    report = tmp_path / "report.md"
    _inventory(inventory)
    frame = pd.read_parquet(inventory)
    frame["import_twh"] = pd.NA
    frame["export_twh"] = pd.NA
    frame.to_parquet(inventory, index=False)
    _manifest(manifest, approved=True)

    rc = main([
        "--inventory", str(inventory),
        "--manifest", str(manifest),
        "--vintage", "2026-01-15",
        "--countries", "CH,DE",
        "--scenarios", "slow,central,fast",
        "--years", "2030",
        "--report", str(report),
    ])

    text = report.read_text(encoding="utf-8")
    assert rc == 0
    assert "governance gate: `OK`" in text


def test_validate_scenario_governance_rejects_missing_cross_border_balance(tmp_path):
    inventory = tmp_path / "inventory.parquet"
    manifest = tmp_path / "manifest.yaml"
    report = tmp_path / "report.md"
    _inventory(inventory)
    frame = pd.read_parquet(inventory)
    frame["import_twh"] = pd.NA
    frame["export_twh"] = pd.NA
    frame["net_import_twh"] = pd.NA
    frame.to_parquet(inventory, index=False)
    _manifest(manifest, approved=True)

    rc = main([
        "--inventory", str(inventory),
        "--manifest", str(manifest),
        "--vintage", "2026-01-15",
        "--countries", "CH,DE",
        "--scenarios", "slow,central,fast",
        "--years", "2030",
        "--report", str(report),
    ])

    text = report.read_text(encoding="utf-8")
    assert rc == 2
    assert "missing_conditional_critical_groups" in text
    assert "cross_border_balance" in text


def test_validate_scenario_governance_scopes_hydro_reservoir_to_ch(tmp_path):
    inventory = tmp_path / "inventory.parquet"
    manifest = tmp_path / "manifest.yaml"
    report = tmp_path / "report.md"
    _inventory(inventory)
    frame = pd.read_parquet(inventory)
    frame.loc[frame["country"] == "DE", "hydro_reservoir_twh"] = pd.NA
    frame.to_parquet(inventory, index=False)
    _manifest(manifest, approved=True)

    rc = main([
        "--inventory", str(inventory),
        "--manifest", str(manifest),
        "--vintage", "2026-01-15",
        "--countries", "CH,DE",
        "--scenarios", "slow,central,fast",
        "--years", "2030",
        "--report", str(report),
    ])

    assert rc == 0


def test_validate_scenario_governance_rejects_missing_ch_hydro_reservoir(tmp_path):
    inventory = tmp_path / "inventory.parquet"
    manifest = tmp_path / "manifest.yaml"
    report = tmp_path / "report.md"
    _inventory(inventory)
    frame = pd.read_parquet(inventory)
    frame.loc[frame["country"] == "CH", "hydro_reservoir_twh"] = pd.NA
    frame.to_parquet(inventory, index=False)
    _manifest(manifest, approved=True)

    rc = main([
        "--inventory", str(inventory),
        "--manifest", str(manifest),
        "--vintage", "2026-01-15",
        "--countries", "CH,DE",
        "--scenarios", "slow,central,fast",
        "--years", "2030",
        "--report", str(report),
    ])

    text = report.read_text(encoding="utf-8")
    assert rc == 2
    assert "missing_critical_values" in text
    assert "hydro_reservoir_twh" in text


def test_validate_scenario_governance_rejects_unknown_source_component_link(tmp_path):
    inventory = tmp_path / "inventory.parquet"
    manifest = tmp_path / "manifest.yaml"
    report = tmp_path / "report.md"
    _inventory(inventory)
    frame = pd.read_parquet(inventory)
    frame["component_ids"] = "unknown_component"
    frame.to_parquet(inventory, index=False)
    _manifest(manifest, approved=True)

    rc = main([
        "--inventory", str(inventory),
        "--manifest", str(manifest),
        "--vintage", "2026-01-15",
        "--countries", "CH,DE",
        "--scenarios", "slow,central,fast",
        "--years", "2030",
        "--report", str(report),
    ])

    text = report.read_text(encoding="utf-8")
    assert rc == 2
    assert "source_component_link_issues" in text
    assert "unknown_component" in text


def test_validate_scenario_governance_rejects_unjustified_critical_zero(tmp_path):
    inventory = tmp_path / "inventory.parquet"
    manifest = tmp_path / "manifest.yaml"
    report = tmp_path / "report.md"
    _inventory(inventory)
    frame = pd.read_parquet(inventory)
    frame["ntc_ch_de_gw"] = 0.0
    frame.to_parquet(inventory, index=False)
    _manifest(manifest, approved=True)

    rc = main([
        "--inventory", str(inventory),
        "--manifest", str(manifest),
        "--vintage", "2026-01-15",
        "--countries", "CH,DE",
        "--scenarios", "slow,central,fast",
        "--years", "2030",
        "--report", str(report),
    ])

    text = report.read_text(encoding="utf-8")
    assert rc == 2
    assert "unjustified_zero_values" in text
    assert "ntc_ch_de_gw" in text


def test_validate_scenario_governance_rejects_physical_bound_violation(tmp_path):
    inventory = tmp_path / "inventory.parquet"
    manifest = tmp_path / "manifest.yaml"
    report = tmp_path / "report.md"
    _inventory(inventory)
    frame = pd.read_parquet(inventory)
    frame["ntc_ch_de_gw"] = 31.0
    frame.to_parquet(inventory, index=False)
    _manifest(manifest, approved=True)

    rc = main([
        "--inventory", str(inventory),
        "--manifest", str(manifest),
        "--vintage", "2026-01-15",
        "--countries", "CH,DE",
        "--scenarios", "slow,central,fast",
        "--years", "2030",
        "--report", str(report),
    ])

    text = report.read_text(encoding="utf-8")
    assert rc == 2
    assert "physical_bounds_violations" in text
    assert "ntc_ch_de_gw" in text

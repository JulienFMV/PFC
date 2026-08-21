from __future__ import annotations

import yaml
import pandas as pd

from pfc_shaping.data.electrification_scenarios import PRODUCTION_COLUMNS
from scripts.build_lt_data_gap_register import main


def _inventory(path):
    weights = {"slow": 0.25, "central": 0.50, "fast": 0.25}
    rows = []
    for country in ["CH", "DE"]:
        for scenario, weight in weights.items():
            row = {column: 1.0 for column in PRODUCTION_COLUMNS}
            row.update(
                {
                    "publication_date": pd.Timestamp("2026-01-01", tz="UTC"),
                    "measurement_date": pd.NaT,
                    "track": "scenario",
                    "source": "unit",
                    "scenario_edition": "unit-2030",
                    "scenario": scenario,
                    "country": country,
                    "delivery_year": 2030,
                    "delivery_month": None,
                    "scenario_weight": weight,
                    "quality_flag": "official_partial_proxy",
                    "ingested_at_utc": pd.Timestamp("2026-01-02", tz="UTC"),
                    "managed_charging_share": 0.4,
                    "net_import_twh": 0.0,
                }
            )
            if country == "DE":
                row["ntc_ch_de_gw"] = pd.NA
            rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _manifest(path, inventory):
    payload = {
        "governance_version": 1,
        "decision_id": "unit",
        "status": "draft",
        "approved_for_production": False,
        "approval_date": None,
        "approved_by": [],
        "scope": {"countries": ["CH", "DE"], "scenarios": ["slow", "central", "fast"], "years": [2030]},
        "scenario_weights": {"slow": 0.25, "central": 0.50, "fast": 0.25},
        "source_components": [
            {
                "source_id": "unit",
                "local_path": str(inventory),
                "publication_date": "2026-01-01",
            }
        ],
        "rules": {
            "allow_proxy_or_partial_quality_flags": False,
            "require_complete_critical_values": True,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_build_lt_data_gap_register_classifies_governance_quality_and_field_gaps(tmp_path):
    inventory = tmp_path / "inventory.parquet"
    manifest = tmp_path / "manifest.yaml"
    csv_output = tmp_path / "gaps.csv"
    report = tmp_path / "gaps.md"
    _inventory(inventory)
    _manifest(manifest, inventory)

    rc = main(
        [
            "--inventory",
            str(inventory),
            "--manifest",
            str(manifest),
            "--vintage",
            "2026-01-15",
            "--countries",
            "CH,DE",
            "--scenarios",
            "slow,central,fast",
            "--years",
            "2030",
            "--csv-output",
            str(csv_output),
            "--report",
            str(report),
        ]
    )

    gaps = pd.read_csv(csv_output)
    text = report.read_text(encoding="utf-8")
    assert rc == 2
    assert {"manifest_not_approved", "unacceptable_quality_flags", "missing_critical_values"} <= set(gaps["issue"])
    ntc = gaps[gaps["field"] == "ntc_ch_de_gw"].iloc[0]
    assert ntc["family"] == "interconnection"
    assert ntc["priority"] == "P0"
    assert "LT Scenario Data Gap Register" in text

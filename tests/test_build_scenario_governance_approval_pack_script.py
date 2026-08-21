from __future__ import annotations

import pandas as pd
import yaml

from scripts.build_scenario_governance_approval_pack import main


def test_approval_pack_returns_no_go_when_manifest_or_gaps_block(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    gap_register = tmp_path / "gaps.csv"
    governance_report = tmp_path / "governance.md"
    expert_reviews = tmp_path / "reviews.md"
    output = tmp_path / "pack.md"

    manifest.write_text(
        yaml.safe_dump(
            {
                "decision_id": "unit",
                "status": "draft",
                "approved_for_production": False,
                "approval_date": None,
                "approved_by": [],
                "scope": {"countries": ["CH"], "scenarios": ["central"], "years": [2030]},
                "scenario_weights": {"central": 1.0},
                "source_components": [{"source_id": "unit", "local_path": "inventory.parquet"}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "priority": "P0",
                "issue": "missing_critical_values",
                "family": "interconnection",
                "field": "ntc_ch_de_gw",
                "missing_rows": 1,
                "effective_rows": 1,
                "owner": "cross_border_scenarios",
                "source_candidate": "Swissgrid",
                "required_action": "Add governed NTC.",
                "prod_status": "BLOCKING",
            }
        ]
    ).to_csv(gap_register, index=False)
    governance_report.write_text("# Governance\n\nFAILED", encoding="utf-8")
    expert_reviews.write_text("# Expert Reviews\n\nNO-GO", encoding="utf-8")

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--governance-report",
            str(governance_report),
            "--gap-register",
            str(gap_register),
            "--expert-reviews",
            str(expert_reviews),
            "--output",
            str(output),
        ]
    )

    text = output.read_text(encoding="utf-8")
    assert rc == 2
    assert "pack_recommendation: `NO-GO`" in text
    assert "manifest status is not approved" in text
    assert "Add governed NTC." in text
    assert "NO-GO" in text

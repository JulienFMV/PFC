from __future__ import annotations

import json

import pandas as pd

from scripts.build_epex_lab_promotion_bundle import build_bundle
from scripts.check_epex_lab_promotion_readiness import check_readiness


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_epex_lab_promotion_bundle_is_non_production_and_checker_bound(tmp_path) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    adjusted_csv = tmp_path / "adjusted.csv"
    baseline_csv.write_text("timestamp_ch,price_weighted_mean_eur_mwh\n", encoding="utf-8")
    adjusted_csv.write_text("timestamp_ch,price_weighted_mean_eur_mwh\n", encoding="utf-8")

    lab = tmp_path / "lab.json"
    monthly = tmp_path / "monthly.json"
    product = tmp_path / "product.json"
    powerbi = tmp_path / "powerbi.csv"
    policy = tmp_path / "policy.json"
    independent = tmp_path / "independent.json"
    governance = tmp_path / "governance.json"
    ompex = tmp_path / "ompex.json"

    _write_json(
        lab,
        {
            "activation_status": "lab_only",
            "production_approved": False,
            "candidate_csv": str(baseline_csv),
            "ompex_used_in_selection": False,
            "outputs": {"adjusted_csv": str(adjusted_csv)},
        },
    )
    _write_json(
        monthly,
        {
            "monthly_level_authority": "solver",
            "monthly_solution_hash": "m" * 64,
            "active_constraints_hash": "c" * 64,
            "active_config_hash": "a" * 64,
        },
    )
    _write_json(
        product,
        {
            "all_gates_pass": True,
            "critical_count": 0,
            "unsupported_count": 0,
            "blocking_quote_conflict_count": 0,
        },
    )
    pd.DataFrame(
        [
            {"metric": "powerbi_quality_gate_status", "value": "PASS"},
            {"metric": "weighted_negative_hours", "value": "0"},
            {"metric": "monthly_path_critical_flags", "value": "0"},
        ]
    ).to_csv(powerbi, index=False)
    _write_json(policy, {"production_approved": True})
    _write_json(
        independent,
        {
            "benchmark_policy": "independent_no_ompex",
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
        },
    )
    _write_json(governance, {"status": "PASS"})
    _write_json(
        ompex,
        {
            "benchmark_policy": "advisory_delta_after_no_ompex_selection",
            "read_only": True,
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
        },
    )

    paths = build_bundle(
        lab_manifest=lab,
        baseline_monthly_manifest=monthly,
        product_summary=product,
        powerbi_summary=powerbi,
        source_hierarchy_policy=policy,
        independent_summary=independent,
        governance_audit=governance,
        ompex_advisory_delta=ompex,
        output_dir=tmp_path / "bundle",
    )

    export_manifest = json.loads(paths["adjusted_export_manifest"].read_text(encoding="utf-8"))
    selected = json.loads(paths["adjusted_selected_artifact"].read_text(encoding="utf-8"))
    capstone = json.loads(paths["adjusted_local_capstone"].read_text(encoding="utf-8"))
    assert export_manifest["production_approved"] is False
    assert selected["production_promotion_approved"] is False
    assert capstone["approved"] is False

    readiness = check_readiness(
        lab_manifest=lab,
        governance_audit=governance,
        independent_summary=independent,
        product_summary=product,
        powerbi_summary=powerbi,
        ompex_advisory_delta=ompex,
        adjusted_export_manifest=paths["adjusted_export_manifest"],
        adjusted_selected_config=paths["adjusted_selected_artifact"],
        adjusted_capstone=paths["adjusted_local_capstone"],
        output=tmp_path / "decision.json",
    )

    assert readiness["strict_diagnostics_pass"] is True
    assert readiness["approved"] is False
    assert readiness["missing_production_evidence"] == ["adjusted_production_manifest"]

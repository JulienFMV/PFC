from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.build_epex_lab_adjusted_production_manifest import build_manifest
from scripts.check_epex_lab_promotion_readiness import check_readiness


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_inputs(tmp_path):
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
    return lab, monthly, product, powerbi, policy, independent, governance, ompex


def test_adjusted_production_manifest_builder_is_no_go_by_default(tmp_path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, _ompex = _write_inputs(tmp_path)

    manifest = build_manifest(
        lab_manifest=lab,
        baseline_monthly_manifest=monthly,
        product_summary=product,
        powerbi_summary=powerbi,
        source_hierarchy_policy=policy,
        independent_summary=independent,
        governance_audit=governance,
        production_run_id="prod-run-1",
        production_entrypoint="pfc_shaping.pipeline.production_phases",
        git_commit="a" * 40,
        output=tmp_path / "adjusted_production_manifest.json",
    )

    assert manifest["schema_version"] == "epex_lab_adjusted_production_manifest.v1"
    assert manifest["contract_pass"] is True
    assert manifest["production_approved"] is False
    assert manifest["production_promotion_approved"] is False
    assert manifest["adjusted_csv_sha256"]
    assert manifest["lab_manifest_sha256"]
    assert {check["status"] for check in manifest["checks"]} == {"PASS"}


def test_default_adjusted_production_manifest_does_not_unlock_readiness(tmp_path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, ompex = _write_inputs(tmp_path)
    production_manifest = tmp_path / "adjusted_production_manifest.json"
    export_manifest = tmp_path / "export.json"
    selected_config = tmp_path / "selected.json"
    capstone = tmp_path / "capstone.json"
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]

    build_manifest(
        lab_manifest=lab,
        baseline_monthly_manifest=monthly,
        product_summary=product,
        powerbi_summary=powerbi,
        source_hierarchy_policy=policy,
        independent_summary=independent,
        governance_audit=governance,
        output=production_manifest,
    )
    _write_json(
        export_manifest,
        {
            "production_approved": True,
            "production_promotion_approved": True,
            "adjusted_csv": adjusted_csv,
        },
    )
    _write_json(
        selected_config,
        {
            "selection_status": "PRODUCTION_APPROVED",
            "production_approved": True,
            "production_promotion_approved": True,
            "selected_adjusted_csv": adjusted_csv,
        },
    )
    _write_json(capstone, {"approved": True})

    readiness = check_readiness(
        lab_manifest=lab,
        governance_audit=governance,
        independent_summary=independent,
        product_summary=product,
        powerbi_summary=powerbi,
        ompex_advisory_delta=ompex,
        adjusted_production_manifest=production_manifest,
        adjusted_export_manifest=export_manifest,
        adjusted_selected_config=selected_config,
        adjusted_capstone=capstone,
        output=tmp_path / "decision.json",
    )

    assert readiness["strict_diagnostics_pass"] is True
    assert readiness["production_chain_pass"] is False
    assert readiness["approved"] is False
    assert {
        check["name"]: check["status"] for check in readiness["checks"]
    }["adjusted_production_manifest_approved"] == "FAIL"


def test_adjusted_production_manifest_approval_requires_run_identity(tmp_path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, _ompex = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="production approval requires complete run identity"):
        build_manifest(
            lab_manifest=lab,
            baseline_monthly_manifest=monthly,
            product_summary=product,
            powerbi_summary=powerbi,
            source_hierarchy_policy=policy,
            independent_summary=independent,
            governance_audit=governance,
            production_approved=True,
            production_promotion_approved=True,
            output=tmp_path / "adjusted_production_manifest.json",
        )

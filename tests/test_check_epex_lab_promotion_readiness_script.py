from __future__ import annotations

import json

import pandas as pd

from scripts.check_epex_lab_promotion_readiness import check_readiness


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_inputs(tmp_path, *, powerbi_status: str = "PASS"):
    lab = tmp_path / "lab.json"
    governance = tmp_path / "governance.json"
    independent = tmp_path / "independent.json"
    product = tmp_path / "product.json"
    powerbi = tmp_path / "powerbi.csv"
    ompex = tmp_path / "ompex.json"

    _write_json(
        lab,
        {
            "activation_status": "lab_only",
            "production_approved": False,
            "ompex_used_in_selection": False,
            "outputs": {"adjusted_csv": "adjusted.csv"},
        },
    )
    _write_json(governance, {"status": "PASS"})
    _write_json(
        independent,
        {
            "benchmark_policy": "independent_no_ompex",
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
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
            {"metric": "powerbi_quality_gate_status", "value": powerbi_status},
            {"metric": "weighted_negative_hours", "value": "0"},
            {"metric": "monthly_path_critical_flags", "value": "0"},
            {"metric": "cross_year_month_shape_critical_flags", "value": "0"},
        ]
    ).to_csv(powerbi, index=False)
    _write_json(
        ompex,
        {
            "benchmark_policy": "advisory_delta_after_no_ompex_selection",
            "read_only": True,
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
        },
    )
    return lab, governance, independent, product, powerbi, ompex


def test_epex_lab_readiness_reports_no_go_when_production_chain_missing(tmp_path) -> None:
    lab, governance, independent, product, powerbi, ompex = _write_inputs(tmp_path)

    summary = check_readiness(
        lab_manifest=lab,
        governance_audit=governance,
        independent_summary=independent,
        product_summary=product,
        powerbi_summary=powerbi,
        ompex_advisory_delta=ompex,
        output=tmp_path / "decision.json",
    )

    assert summary["approved"] is False
    assert summary["strict_diagnostics_pass"] is True
    assert summary["production_chain_pass"] is False
    assert summary["status"] == "STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING"
    assert "adjusted_capstone" in summary["missing_production_evidence"]


def test_epex_lab_readiness_fails_when_powerbi_gate_fails(tmp_path) -> None:
    lab, governance, independent, product, powerbi, ompex = _write_inputs(tmp_path, powerbi_status="FAIL")

    summary = check_readiness(
        lab_manifest=lab,
        governance_audit=governance,
        independent_summary=independent,
        product_summary=product,
        powerbi_summary=powerbi,
        ompex_advisory_delta=ompex,
        output=tmp_path / "decision.json",
    )

    assert summary["approved"] is False
    assert summary["strict_diagnostics_pass"] is False
    assert summary["status"] == "STRICT_DIAGNOSTICS_FAIL"

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_epex_lab_adjusted_production_chain import build_chain
from scripts.build_epex_lab_adjusted_production_manifest import build_manifest
from scripts.check_epex_lab_promotion_readiness import check_readiness


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_inputs(tmp_path: Path):
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
            {"metric": "cross_year_month_shape_critical_flags", "value": "0"},
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


def _write_source_provenance(tmp_path: Path, *, lab: Path, adjusted_csv: str) -> Path:
    source_csv = tmp_path / "baseline.csv"
    source_export = tmp_path / "source_export_manifest.json"
    _write_json(
        source_export,
        {
            "schema_version": "test_source_export_manifest.v1",
            "candidate_csv": str(source_csv),
            "candidate_csv_sha256": _sha256(source_csv),
        },
    )
    source_provenance = tmp_path / "source_provenance.json"
    _write_json(
        source_provenance,
        {
            "schema_version": "epex_lab_adjusted_lt_candidate_stage.v1",
            "schema_role": "source_provenance",
            "activation_status": "staged_lab_only",
            "production_approved": False,
            "production_promotion_approved": False,
            "source_kind": "candidate_csv",
            "source_promotion_eligible": True,
            "source_path": str(source_csv),
            "source_sha256": _sha256(source_csv),
            "source_export_manifest": str(source_export),
            "source_export_manifest_sha256": _sha256(source_export),
            "staged_candidate_csv": str(source_csv),
            "staged_candidate_csv_sha256": _sha256(source_csv),
            "lab_manifest": str(lab),
            "lab_manifest_sha256": _sha256(lab),
            "adjusted_csv": adjusted_csv,
            "adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
            "production_contract_blockers": [],
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
        },
    )
    return source_provenance


def test_adjusted_production_chain_rejects_no_go_manifest(tmp_path: Path) -> None:
    adjusted_csv = tmp_path / "adjusted.csv"
    adjusted_csv.write_text("timestamp_ch,price_weighted_mean_eur_mwh\n", encoding="utf-8")
    production_manifest = tmp_path / "prod.json"
    _write_json(
        production_manifest,
        {
            "schema_version": "epex_lab_adjusted_production_manifest.v1",
            "production_approved": False,
            "production_promotion_approved": False,
            "contract_pass": True,
            "source_provenance_pass": True,
            "adjusted_csv": str(adjusted_csv),
            "adjusted_csv_sha256": _sha256(adjusted_csv),
            "production_run_id": "prod-run-1",
            "production_entrypoint": "pfc_shaping.pipeline.production_phases",
            "git_commit": "a" * 40,
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
        },
    )

    with pytest.raises(ValueError, match="production_approved"):
        build_chain(adjusted_production_manifest=production_manifest, output_dir=tmp_path / "chain")


def test_adjusted_production_chain_rejects_self_attested_source_provenance(tmp_path: Path) -> None:
    adjusted_csv = tmp_path / "adjusted.csv"
    adjusted_csv.write_text("timestamp_ch,price_weighted_mean_eur_mwh\n", encoding="utf-8")
    production_manifest = tmp_path / "prod.json"
    _write_json(
        production_manifest,
        {
            "schema_version": "epex_lab_adjusted_production_manifest.v1",
            "production_approved": True,
            "production_promotion_approved": True,
            "contract_pass": True,
            "source_provenance_pass": True,
            "source_kind": "candidate_csv",
            "source_promotion_eligible": True,
            "adjusted_csv": str(adjusted_csv),
            "adjusted_csv_sha256": _sha256(adjusted_csv),
            "production_run_id": "prod-run-1",
            "production_entrypoint": "pfc_shaping.pipeline.production_phases",
            "git_commit": "a" * 40,
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
        },
    )

    with pytest.raises(ValueError, match="source_provenance_manifest"):
        build_chain(adjusted_production_manifest=production_manifest, output_dir=tmp_path / "chain")


def test_adjusted_production_chain_rejects_tampered_source_provenance(tmp_path: Path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source_provenance = _write_source_provenance(tmp_path, lab=lab, adjusted_csv=adjusted_csv)
    source = json.loads(source_provenance.read_text(encoding="utf-8"))
    source["source_sha256"] = "0" * 64
    source_provenance.write_text(json.dumps(source), encoding="utf-8")
    production_manifest = tmp_path / "adjusted_production_manifest.json"
    build_manifest(
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
        source_provenance_manifest=source_provenance,
        production_approved=True,
        production_promotion_approved=True,
        output=production_manifest,
    )

    with pytest.raises(ValueError, match="source_provenance_source_sha256"):
        build_chain(adjusted_production_manifest=production_manifest, output_dir=tmp_path / "chain")


def test_adjusted_production_chain_builds_artifacts_that_unlock_readiness(tmp_path: Path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source_provenance = _write_source_provenance(tmp_path, lab=lab, adjusted_csv=adjusted_csv)
    production_manifest = tmp_path / "adjusted_production_manifest.json"
    build_manifest(
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
        source_provenance_manifest=source_provenance,
        production_approved=True,
        production_promotion_approved=True,
        output=production_manifest,
    )

    paths = build_chain(
        adjusted_production_manifest=production_manifest,
        output_dir=tmp_path / "chain",
    )
    export_manifest = json.loads(paths["adjusted_export_manifest"].read_text(encoding="utf-8"))
    selected = json.loads(paths["adjusted_selected_artifact"].read_text(encoding="utf-8"))
    capstone = json.loads(paths["adjusted_capstone"].read_text(encoding="utf-8"))

    assert export_manifest["production_approved"] is True
    assert selected["selection_status"] == "PRODUCTION_APPROVED"
    assert capstone["approved"] is True
    assert export_manifest["adjusted_production_manifest_sha256"] == _sha256(production_manifest)
    assert capstone["adjusted_selected_artifact_sha256"] == _sha256(paths["adjusted_selected_artifact"])

    readiness = check_readiness(
        lab_manifest=lab,
        governance_audit=governance,
        independent_summary=independent,
        product_summary=product,
        powerbi_summary=powerbi,
        ompex_advisory_delta=ompex,
        adjusted_production_manifest=production_manifest,
        adjusted_export_manifest=paths["adjusted_export_manifest"],
        adjusted_selected_config=paths["adjusted_selected_artifact"],
        adjusted_capstone=paths["adjusted_capstone"],
        output=tmp_path / "decision.json",
    )

    assert readiness["approved"] is True
    assert readiness["production_chain_pass"] is True
    assert readiness["status"] == "PROMOTION_READY"

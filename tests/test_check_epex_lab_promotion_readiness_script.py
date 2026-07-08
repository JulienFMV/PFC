from __future__ import annotations

import json
import hashlib

import pandas as pd

from scripts.check_epex_lab_promotion_readiness import check_readiness


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_inputs(tmp_path, *, powerbi_status: str = "PASS"):
    lab = tmp_path / "lab.json"
    governance = tmp_path / "governance.json"
    independent = tmp_path / "independent.json"
    product = tmp_path / "product.json"
    powerbi = tmp_path / "powerbi.csv"
    ompex = tmp_path / "ompex.json"
    adjusted_csv = tmp_path / "adjusted.csv"
    adjusted_csv.write_text("timestamp_ch,price_weighted_mean_eur_mwh\n", encoding="utf-8")

    _write_json(
        lab,
        {
            "activation_status": "lab_only",
            "production_approved": False,
            "ompex_used_in_selection": False,
            "outputs": {"adjusted_csv": str(adjusted_csv)},
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


def _write_source_provenance(tmp_path, *, lab, adjusted_csv: str, source_kind: str = "candidate_csv"):
    source_csv = tmp_path / "source.csv"
    source_csv.write_text("timestamp_ch,price_weighted_mean_eur_mwh\n", encoding="utf-8")
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
    blockers = [] if source_kind == "candidate_csv" else ["source_kind_fan_parquet_requires_audited_hourly_export"]
    _write_json(
        source_provenance,
        {
            "schema_version": "epex_lab_adjusted_lt_candidate_stage.v1",
            "schema_role": "source_provenance",
            "activation_status": "staged_lab_only",
            "production_approved": False,
            "production_promotion_approved": False,
            "promotion_scope": "LT_EPEX_LAB_STAGING_NO_GO",
            "source_kind": source_kind,
            "source_promotion_eligible": source_kind == "candidate_csv",
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
            "production_contract_blockers": blockers,
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
        },
    )
    return source_provenance


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


def test_epex_lab_readiness_rejects_unapproved_production_manifest(tmp_path) -> None:
    lab, governance, independent, product, powerbi, ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    production_manifest = tmp_path / "prod.json"
    export_manifest = tmp_path / "export.json"
    selected_config = tmp_path / "selected.json"
    capstone = tmp_path / "capstone.json"
    _write_json(
        production_manifest,
        {
            "schema_version": "epex_lab_adjusted_production_manifest.v1",
            "production_approved": False,
            "production_promotion_approved": False,
            "adjusted_csv": adjusted_csv,
        },
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

    summary = check_readiness(
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

    assert summary["approved"] is False
    assert summary["strict_diagnostics_pass"] is True
    assert summary["production_chain_pass"] is False
    assert {
        check["name"]: check["status"] for check in summary["checks"]
    }["adjusted_production_manifest_approved"] == "FAIL"


def test_epex_lab_readiness_can_pass_with_separate_approved_production_chain(tmp_path) -> None:
    lab, governance, independent, product, powerbi, ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source_provenance = _write_source_provenance(tmp_path, lab=lab, adjusted_csv=adjusted_csv)
    production_manifest = tmp_path / "prod.json"
    export_manifest = tmp_path / "export.json"
    selected_config = tmp_path / "selected.json"
    capstone = tmp_path / "capstone.json"
    run_identity = {
        "production_run_id": "prod-run-1",
        "production_entrypoint": "pfc_shaping.pipeline.production_phases",
        "git_commit": "a" * 40,
    }
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
            "source_provenance_manifest": str(source_provenance),
            "source_provenance_manifest_sha256": _sha256(source_provenance),
            "adjusted_csv": adjusted_csv,
            "adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
            **run_identity,
        },
    )
    production_manifest_sha = _sha256(production_manifest)
    _write_json(
        export_manifest,
        {
            "schema_version": "epex_lab_adjusted_export_manifest.v1",
            "production_approved": True,
            "production_promotion_approved": True,
            "adjusted_csv": adjusted_csv,
            "adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
            "adjusted_production_manifest": str(production_manifest),
            "adjusted_production_manifest_sha256": production_manifest_sha,
            **run_identity,
        },
    )
    _write_json(
        selected_config,
        {
            "schema_version": "epex_lab_selected_artifact.v1",
            "selection_status": "PRODUCTION_APPROVED",
            "production_approved": True,
            "production_promotion_approved": True,
            "selected_adjusted_csv": adjusted_csv,
            "selected_adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
            "adjusted_production_manifest": str(production_manifest),
            "adjusted_production_manifest_sha256": production_manifest_sha,
            **run_identity,
        },
    )
    export_manifest_sha = _sha256(export_manifest)
    selected_config_sha = _sha256(selected_config)
    _write_json(
        capstone,
        {
            "schema_version": "epex_lab_production_capstone.v1",
            "approved": True,
            "production_chain_pass": True,
            "adjusted_production_manifest": str(production_manifest),
            "adjusted_production_manifest_sha256": production_manifest_sha,
            "adjusted_export_manifest": str(export_manifest),
            "adjusted_export_manifest_sha256": export_manifest_sha,
            "adjusted_selected_artifact": str(selected_config),
            "adjusted_selected_artifact_sha256": selected_config_sha,
            **run_identity,
        },
    )

    summary = check_readiness(
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

    assert summary["approved"] is True
    assert summary["strict_diagnostics_pass"] is True
    assert summary["production_chain_pass"] is True
    assert summary["status"] == "PROMOTION_READY"


def test_epex_lab_readiness_rejects_unbound_approved_export_manifest(tmp_path) -> None:
    lab, governance, independent, product, powerbi, ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source_provenance = _write_source_provenance(tmp_path, lab=lab, adjusted_csv=adjusted_csv)
    production_manifest = tmp_path / "prod.json"
    export_manifest = tmp_path / "export.json"
    selected_config = tmp_path / "selected.json"
    capstone = tmp_path / "capstone.json"
    run_identity = {
        "production_run_id": "prod-run-1",
        "production_entrypoint": "pfc_shaping.pipeline.production_phases",
        "git_commit": "a" * 40,
    }
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
            "source_provenance_manifest": str(source_provenance),
            "source_provenance_manifest_sha256": _sha256(source_provenance),
            "adjusted_csv": adjusted_csv,
            "adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
            **run_identity,
        },
    )
    production_manifest_sha = _sha256(production_manifest)
    _write_json(
        export_manifest,
        {
            "schema_version": "epex_lab_adjusted_export_manifest.v1",
            "production_approved": True,
            "production_promotion_approved": True,
            "adjusted_csv": adjusted_csv,
            "adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
            "adjusted_production_manifest_sha256": "b" * 64,
            **run_identity,
        },
    )
    _write_json(
        selected_config,
        {
            "schema_version": "epex_lab_selected_artifact.v1",
            "selection_status": "PRODUCTION_APPROVED",
            "production_approved": True,
            "production_promotion_approved": True,
            "selected_adjusted_csv": adjusted_csv,
            "selected_adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
            "adjusted_production_manifest": str(production_manifest),
            "adjusted_production_manifest_sha256": production_manifest_sha,
            **run_identity,
        },
    )
    _write_json(
        capstone,
        {
            "schema_version": "epex_lab_production_capstone.v1",
            "approved": True,
            "production_chain_pass": True,
            "adjusted_production_manifest": str(production_manifest),
            "adjusted_production_manifest_sha256": production_manifest_sha,
            "adjusted_export_manifest": str(export_manifest),
            "adjusted_export_manifest_sha256": _sha256(export_manifest),
            "adjusted_selected_artifact": str(selected_config),
            "adjusted_selected_artifact_sha256": _sha256(selected_config),
            **run_identity,
        },
    )

    summary = check_readiness(
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

    checks = {check["name"]: check["status"] for check in summary["checks"]}
    assert summary["approved"] is False
    assert summary["production_chain_pass"] is False
    assert checks["adjusted_export_manifest_production_chain_bound"] == "FAIL"


def test_epex_lab_readiness_rejects_self_attested_source_provenance(tmp_path) -> None:
    lab, governance, independent, product, powerbi, ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    production_manifest = tmp_path / "prod.json"
    export_manifest = tmp_path / "export.json"
    selected_config = tmp_path / "selected.json"
    capstone = tmp_path / "capstone.json"
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
            "adjusted_csv": adjusted_csv,
            "adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
        },
    )
    _write_json(
        export_manifest,
        {
            "schema_version": "epex_lab_adjusted_export_manifest.v1",
            "production_approved": True,
            "production_promotion_approved": True,
            "adjusted_csv": adjusted_csv,
            "adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
        },
    )
    _write_json(
        selected_config,
        {
            "schema_version": "epex_lab_selected_artifact.v1",
            "selection_status": "PRODUCTION_APPROVED",
            "production_approved": True,
            "production_promotion_approved": True,
            "selected_adjusted_csv": adjusted_csv,
            "selected_adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
        },
    )
    _write_json(
        capstone,
        {
            "schema_version": "epex_lab_production_capstone.v1",
            "approved": True,
            "production_chain_pass": True,
        },
    )

    summary = check_readiness(
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

    checks = {check["name"]: check["status"] for check in summary["checks"]}
    assert summary["approved"] is False
    assert summary["production_chain_pass"] is False
    assert checks["source_provenance_manifest_present"] == "FAIL"


def test_epex_lab_readiness_rejects_fan_source_provenance(tmp_path) -> None:
    lab, governance, independent, product, powerbi, ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source_provenance = _write_source_provenance(
        tmp_path,
        lab=lab,
        adjusted_csv=adjusted_csv,
        source_kind="fan_parquet",
    )
    production_manifest = tmp_path / "prod.json"
    export_manifest = tmp_path / "export.json"
    selected_config = tmp_path / "selected.json"
    capstone = tmp_path / "capstone.json"
    _write_json(
        production_manifest,
        {
            "schema_version": "epex_lab_adjusted_production_manifest.v1",
            "production_approved": True,
            "production_promotion_approved": True,
            "contract_pass": True,
            "source_provenance_pass": True,
            "source_kind": "fan_parquet",
            "source_promotion_eligible": False,
            "source_provenance_manifest": str(source_provenance),
            "source_provenance_manifest_sha256": _sha256(source_provenance),
            "adjusted_csv": adjusted_csv,
            "adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
        },
    )
    _write_json(
        export_manifest,
        {
            "schema_version": "epex_lab_adjusted_export_manifest.v1",
            "production_approved": True,
            "production_promotion_approved": True,
            "adjusted_csv": adjusted_csv,
            "adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
        },
    )
    _write_json(
        selected_config,
        {
            "schema_version": "epex_lab_selected_artifact.v1",
            "selection_status": "PRODUCTION_APPROVED",
            "production_approved": True,
            "production_promotion_approved": True,
            "selected_adjusted_csv": adjusted_csv,
            "selected_adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
        },
    )
    _write_json(
        capstone,
        {
            "schema_version": "epex_lab_production_capstone.v1",
            "approved": True,
            "production_chain_pass": True,
        },
    )

    summary = check_readiness(
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

    checks = {check["name"]: check["status"] for check in summary["checks"]}
    assert summary["approved"] is False
    assert summary["production_chain_pass"] is False
    assert checks["source_provenance_candidate_csv"] == "FAIL"
    assert checks["source_provenance_promotion_eligible"] == "FAIL"

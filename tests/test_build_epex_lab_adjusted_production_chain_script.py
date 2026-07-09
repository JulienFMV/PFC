from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_epex_lab_adjusted_production_chain import build_chain
from scripts.build_epex_lab_adjusted_production_manifest import build_manifest
from scripts.check_epex_lab_promotion_readiness import check_readiness
from scripts.epex_lab_locked_holdout_policy import build_locked_plan_identity


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
    _write_json(policy, {"production_approved": True})
    _write_json(
        product,
        {
            "all_gates_pass": True,
            "critical_count": 0,
            "unsupported_count": 0,
            "blocking_quote_conflict_count": 0,
            "source_hierarchy_policy": {
                "path": str(policy),
                "sha256": _sha256(policy),
                "status": "ACCEPTED_PRODUCTION_APPROVED",
                "production_approved": True,
                "blocking_quote_conflict_count": 0,
            },
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


def _write_selection_summary(tmp_path: Path) -> Path:
    selection = tmp_path / "selection_summary.json"
    _write_json(
        selection,
        {
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
            "ompex_used_in_backtest": False,
            "replacement_verdict": {"replace_incumbent": True},
            "selected_trial": {
                "adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
            },
            "selected_adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
        },
    )
    return selection


def _write_locked_plan(tmp_path: Path) -> Path:
    plan = tmp_path / "locked_plan.json"
    payload = {
        "schema_version": "epex_lab_locked_holdout_plan.v1",
        "plan_id": "test_locked_holdout",
        "benchmark_policy": "locked_future_no_ompex_holdout",
        "frozen_at_utc": "2026-07-09T00:00:00Z",
        "holdout_start_utc": "2026-07-10T00:00:00Z",
        "holdout_end_utc": "2026-07-24T00:00:00Z",
        "baseline_csv_sha256": "b" * 64,
        "adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
        "lab_manifest_sha256": "l" * 64,
        "selection_summary_sha256": "s" * 64,
    }
    _write_json(plan, payload)
    return plan


def _write_locked_holdout(tmp_path: Path, *, passed: bool = True) -> Path:
    locked_holdout = tmp_path / "locked_holdout.json"
    plan = _write_locked_plan(tmp_path)
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    identity = build_locked_plan_identity(plan_payload, plan_json=plan)
    coverage_payload = _ready_coverage(passed=passed, identity=identity)
    coverage = tmp_path / "coverage_status.json"
    _write_json(coverage, coverage_payload)
    backtest = tmp_path / "spot_backtest_summary.json"
    _write_json(backtest, _passing_backtest(passed=passed))
    audit = tmp_path / "locked_holdout_audit.json"
    _write_json(audit, _passing_audit(identity=identity, backtest=backtest, passed=passed))
    _write_json(
        locked_holdout,
        {
            "schema_version": "epex_lab_locked_holdout_run.v1",
            "status": "LOCKED_HOLDOUT_PASS" if passed else "WAITING_FOR_FULL_SPOT_COVERAGE",
            "benchmark_policy": "locked_future_no_ompex_holdout",
            "expected_plan_json_sha256": identity["plan_json_sha256"],
            "actual_plan_json_sha256": identity["plan_json_sha256"],
            "promotion_gate": False,
            "production_approved": False,
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
            "ompex_used_in_backtest": False,
            "coverage_status": str(coverage),
            "coverage_status_sha256": _sha256(coverage),
            "coverage": coverage_payload,
            "coverage_ready": passed,
            "backtest_ran": passed,
            "audit_ran": passed,
            "holdout_pass": passed,
            "locked_plan_identity": identity,
            "spot_backtest_summary": str(backtest),
            "spot_backtest_summary_sha256": _sha256(backtest),
            "locked_holdout_audit": str(audit),
            "locked_holdout_audit_sha256": _sha256(audit),
        },
    )
    return locked_holdout


def _ready_coverage(*, passed: bool = True, identity: dict) -> dict:
    timestamp_set_sha256 = "c" * 64 if passed else None
    return {
        "schema_version": "epex_lab_locked_holdout_coverage.v1",
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "locked_plan_identity": identity,
        "baseline_csv_sha256": identity["baseline_csv_sha256"],
        "adjusted_csv_sha256": identity["adjusted_csv_sha256"],
        "baseline_candidate_timestamp_count": 4 if passed else 0,
        "baseline_candidate_timestamp_min_utc": "2026-07-10T00:00:00Z" if passed else None,
        "baseline_candidate_timestamp_max_utc": "2026-07-10T03:00:00Z" if passed else None,
        "baseline_candidate_timestamp_set_sha256": timestamp_set_sha256,
        "adjusted_candidate_timestamp_count": 4 if passed else 0,
        "adjusted_candidate_timestamp_min_utc": "2026-07-10T00:00:00Z" if passed else None,
        "adjusted_candidate_timestamp_max_utc": "2026-07-10T03:00:00Z" if passed else None,
        "adjusted_candidate_timestamp_set_sha256": timestamp_set_sha256,
        "status": "READY_TO_RUN_HOLDOUT_BACKTEST" if passed else "WAITING_FOR_FULL_SPOT_COVERAGE",
        "ready_to_run_backtest": passed,
        "blocking_checks": [] if passed else ["full_window_covered", "min_holdout_hours_met"],
        "checks": {
            "baseline_csv_sha256_bound": passed,
            "adjusted_csv_sha256_bound": passed,
            "baseline_candidate_required_columns_present": passed,
            "baseline_candidate_utc_offset_present": passed,
            "baseline_candidate_timestamps_parseable": passed,
            "baseline_candidate_no_duplicate_timestamps": passed,
            "baseline_candidate_price_columns_finite": passed,
            "baseline_candidate_holdout_window_covered": passed,
            "adjusted_candidate_required_columns_present": passed,
            "adjusted_candidate_utc_offset_present": passed,
            "adjusted_candidate_timestamps_parseable": passed,
            "adjusted_candidate_no_duplicate_timestamps": passed,
            "adjusted_candidate_price_columns_finite": passed,
            "adjusted_candidate_holdout_window_covered": passed,
            "candidate_timestamp_sets_identical": passed,
            "candidate_timestamp_set_matches_plan": passed,
            "candidate_timestamp_count_matches_plan": passed,
            "full_window_covered": passed,
            "min_holdout_hours_met": passed,
            "no_duplicate_holdout_rows": passed,
            "spot_price_column_present": passed,
            "holdout_prices_finite": passed,
        },
    }


def _passing_backtest(*, passed: bool = True) -> dict:
    return {
        "schema_version": "epex_shape_lab_spot_backtest.v1",
        "status": "DIAGNOSTIC_PASS" if passed else "DIAGNOSTIC_FAIL",
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "independent_production_evidence": False,
        "benchmark_policy": "rolling_origin_epex_spot_no_ompex_lab_only",
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "strict_lab_gate_pass": passed,
    }


def _passing_audit(*, identity: dict, backtest: Path, passed: bool = True) -> dict:
    return {
        "schema_version": "epex_lab_locked_holdout_audit.v1",
        "status": "LOCKED_HOLDOUT_PASS" if passed else "NO_GO_LOCKED_HOLDOUT_FAIL",
        "holdout_pass": passed,
        "promotion_gate": False,
        "production_approved": False,
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "locked_plan_identity": identity,
        "spot_backtest_summary": str(backtest),
        "spot_backtest_summary_sha256": _sha256(backtest),
    }


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
    selection = _write_selection_summary(tmp_path)
    locked_holdout = _write_locked_holdout(tmp_path)
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
        selection_summary=selection,
        locked_holdout_summary=locked_holdout,
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


def test_adjusted_production_chain_rejects_self_attested_selection_policy(tmp_path: Path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, _ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source_provenance = _write_source_provenance(tmp_path, lab=lab, adjusted_csv=adjusted_csv)
    selection = _write_selection_summary(tmp_path)
    locked_holdout = _write_locked_holdout(tmp_path)
    production_manifest = tmp_path / "adjusted_production_manifest.json"
    build_manifest(
        lab_manifest=lab,
        baseline_monthly_manifest=monthly,
        product_summary=product,
        powerbi_summary=powerbi,
        source_hierarchy_policy=policy,
        independent_summary=independent,
        governance_audit=governance,
        selection_summary=selection,
        locked_holdout_summary=locked_holdout,
        production_run_id="prod-run-1",
        production_entrypoint="pfc_shaping.pipeline.production_phases",
        git_commit="a" * 40,
        source_provenance_manifest=source_provenance,
        production_approved=True,
        production_promotion_approved=True,
        output=production_manifest,
    )
    manifest = json.loads(production_manifest.read_text(encoding="utf-8"))
    manifest["selection_policy_pass"] = True
    manifest.pop("selection_summary")
    manifest.pop("selection_summary_sha256")
    production_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="selection_summary"):
        build_chain(adjusted_production_manifest=production_manifest, output_dir=tmp_path / "chain")


def test_adjusted_production_chain_rejects_tampered_selection_summary(tmp_path: Path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, _ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source_provenance = _write_source_provenance(tmp_path, lab=lab, adjusted_csv=adjusted_csv)
    selection = _write_selection_summary(tmp_path)
    locked_holdout = _write_locked_holdout(tmp_path)
    production_manifest = tmp_path / "adjusted_production_manifest.json"
    build_manifest(
        lab_manifest=lab,
        baseline_monthly_manifest=monthly,
        product_summary=product,
        powerbi_summary=powerbi,
        source_hierarchy_policy=policy,
        independent_summary=independent,
        governance_audit=governance,
        selection_summary=selection,
        locked_holdout_summary=locked_holdout,
        production_run_id="prod-run-1",
        production_entrypoint="pfc_shaping.pipeline.production_phases",
        git_commit="a" * 40,
        source_provenance_manifest=source_provenance,
        production_approved=True,
        production_promotion_approved=True,
        output=production_manifest,
    )
    selection_payload = json.loads(selection.read_text(encoding="utf-8"))
    selection_payload["replacement_verdict"] = {"replace_incumbent": False}
    selection.write_text(json.dumps(selection_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="selection_summary_sha256"):
        build_chain(adjusted_production_manifest=production_manifest, output_dir=tmp_path / "chain")


def test_adjusted_production_chain_revalidates_strict_product_evidence(tmp_path: Path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, _ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source_provenance = _write_source_provenance(tmp_path, lab=lab, adjusted_csv=adjusted_csv)
    selection = _write_selection_summary(tmp_path)
    locked_holdout = _write_locked_holdout(tmp_path)
    production_manifest = tmp_path / "adjusted_production_manifest.json"
    build_manifest(
        lab_manifest=lab,
        baseline_monthly_manifest=monthly,
        product_summary=product,
        powerbi_summary=powerbi,
        source_hierarchy_policy=policy,
        independent_summary=independent,
        governance_audit=governance,
        selection_summary=selection,
        locked_holdout_summary=locked_holdout,
        production_run_id="prod-run-1",
        production_entrypoint="pfc_shaping.pipeline.production_phases",
        git_commit="a" * 40,
        source_provenance_manifest=source_provenance,
        production_approved=True,
        production_promotion_approved=True,
        output=production_manifest,
    )
    product_payload = json.loads(product.read_text(encoding="utf-8"))
    product_payload["all_gates_pass"] = False
    product.write_text(json.dumps(product_payload), encoding="utf-8")
    manifest = json.loads(production_manifest.read_text(encoding="utf-8"))
    manifest["product_summary_sha256"] = _sha256(product)
    production_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="product_summary_all_gates_pass"):
        build_chain(adjusted_production_manifest=production_manifest, output_dir=tmp_path / "chain")


def test_adjusted_production_chain_rejects_approved_manifest_without_locked_holdout(tmp_path: Path) -> None:
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
            "selection_policy_pass": True,
        },
    )

    with pytest.raises(ValueError, match="locked_holdout"):
        build_chain(adjusted_production_manifest=production_manifest, output_dir=tmp_path / "chain")


def test_adjusted_production_chain_rejects_tampered_locked_holdout_summary(tmp_path: Path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, _ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source_provenance = _write_source_provenance(tmp_path, lab=lab, adjusted_csv=adjusted_csv)
    selection = _write_selection_summary(tmp_path)
    locked_holdout = _write_locked_holdout(tmp_path)
    production_manifest = tmp_path / "adjusted_production_manifest.json"
    build_manifest(
        lab_manifest=lab,
        baseline_monthly_manifest=monthly,
        product_summary=product,
        powerbi_summary=powerbi,
        source_hierarchy_policy=policy,
        independent_summary=independent,
        governance_audit=governance,
        selection_summary=selection,
        locked_holdout_summary=locked_holdout,
        production_run_id="prod-run-1",
        production_entrypoint="pfc_shaping.pipeline.production_phases",
        git_commit="a" * 40,
        source_provenance_manifest=source_provenance,
        production_approved=True,
        production_promotion_approved=True,
        output=production_manifest,
    )
    holdout_payload = json.loads(locked_holdout.read_text(encoding="utf-8"))
    holdout_payload["holdout_pass"] = False
    locked_holdout.write_text(json.dumps(holdout_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="locked_holdout_summary_sha256"):
        build_chain(adjusted_production_manifest=production_manifest, output_dir=tmp_path / "chain")


def test_adjusted_production_chain_builds_artifacts_that_unlock_readiness(tmp_path: Path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source_provenance = _write_source_provenance(tmp_path, lab=lab, adjusted_csv=adjusted_csv)
    selection = _write_selection_summary(tmp_path)
    locked_holdout = _write_locked_holdout(tmp_path)
    production_manifest = tmp_path / "adjusted_production_manifest.json"
    build_manifest(
        lab_manifest=lab,
        baseline_monthly_manifest=monthly,
        product_summary=product,
        powerbi_summary=powerbi,
        source_hierarchy_policy=policy,
        independent_summary=independent,
        governance_audit=governance,
        selection_summary=selection,
        locked_holdout_summary=locked_holdout,
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
    assert export_manifest["locked_holdout_summary_sha256"] == _sha256(locked_holdout)
    assert selected["locked_holdout_summary_sha256"] == _sha256(locked_holdout)
    assert capstone["locked_holdout_summary_sha256"] == _sha256(locked_holdout)
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
        locked_holdout_summary=locked_holdout,
        output=tmp_path / "decision.json",
    )

    assert readiness["approved"] is True
    assert readiness["production_chain_pass"] is True
    assert readiness["status"] == "PROMOTION_READY"

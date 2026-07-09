from __future__ import annotations

import json
import hashlib

import pandas as pd

from scripts.check_epex_lab_promotion_readiness import REQUIRED_PRODUCTION_CHECKS, check_readiness
from scripts.epex_lab_locked_holdout_policy import build_locked_plan_identity


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


def _write_selection_summary(tmp_path, *, adjusted_csv: str, replace_incumbent: bool = True):
    selection = tmp_path / "selection_summary.json"
    adjusted_sha = _sha256(tmp_path / "adjusted.csv")
    _write_json(
        selection,
        {
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
            "ompex_used_in_backtest": False,
            "replacement_verdict": {"replace_incumbent": replace_incumbent},
            "selected_adjusted_csv_sha256": adjusted_sha,
            "selected_trial": {"adjusted_csv_sha256": adjusted_sha},
        },
    )
    return selection


def _write_locked_plan(tmp_path):
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


def _write_locked_holdout(tmp_path, *, passed: bool = True):
    locked_holdout = tmp_path / "locked_holdout.json"
    plan = _write_locked_plan(tmp_path)
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    identity = build_locked_plan_identity(plan_payload, plan_json=plan)
    coverage_payload = _ready_coverage(passed=passed)
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
            "status": "LOCKED_HOLDOUT_PASS" if passed else "NO_GO_LOCKED_HOLDOUT_FAIL",
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


def _ready_coverage(*, passed: bool = True):
    return {
        "status": "READY_TO_RUN_HOLDOUT_BACKTEST" if passed else "WAITING_FOR_FULL_SPOT_COVERAGE",
        "ready_to_run_backtest": passed,
        "checks": {
            "baseline_csv_sha256_bound": passed,
            "adjusted_csv_sha256_bound": passed,
            "baseline_candidate_required_columns_present": passed,
            "baseline_candidate_timestamps_parseable": passed,
            "baseline_candidate_no_duplicate_timestamps": passed,
            "baseline_candidate_price_columns_finite": passed,
            "baseline_candidate_holdout_window_covered": passed,
            "adjusted_candidate_required_columns_present": passed,
            "adjusted_candidate_timestamps_parseable": passed,
            "adjusted_candidate_no_duplicate_timestamps": passed,
            "adjusted_candidate_price_columns_finite": passed,
            "adjusted_candidate_holdout_window_covered": passed,
            "candidate_timestamp_sets_identical": passed,
            "full_window_covered": passed,
            "min_holdout_hours_met": passed,
            "no_duplicate_holdout_rows": passed,
            "spot_price_column_present": passed,
            "holdout_prices_finite": passed,
        },
    }


def _passing_backtest(*, passed: bool = True):
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


def _passing_audit(*, identity: dict, backtest: Path, passed: bool = True):
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


def _write_approved_chain(
    tmp_path,
    *,
    divergent_selected_holdout_sha: bool = False,
    divergent_capstone_holdout_sha: bool = False,
    production_holdout_policy_pass: bool = True,
):
    lab, governance, independent, product, powerbi, ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source_provenance = _write_source_provenance(tmp_path, lab=lab, adjusted_csv=adjusted_csv)
    selection = _write_selection_summary(tmp_path, adjusted_csv=adjusted_csv)
    locked_holdout = _write_locked_holdout(tmp_path)
    production_manifest = tmp_path / "prod.json"
    export_manifest = tmp_path / "export.json"
    selected_config = tmp_path / "selected.json"
    capstone = tmp_path / "capstone.json"
    run_identity = {
        "production_run_id": "prod-run-1",
        "production_entrypoint": "pfc_shaping.pipeline.production_phases",
        "git_commit": "a" * 40,
    }
    locked_holdout_fields = {
        "locked_holdout_summary": str(locked_holdout),
        "locked_holdout_summary_sha256": _sha256(locked_holdout),
        "locked_holdout_policy_pass": True,
    }
    production_locked_holdout_fields = {
        **locked_holdout_fields,
        "locked_holdout_policy_pass": production_holdout_policy_pass,
    }
    _write_json(
        production_manifest,
        {
            "schema_version": "epex_lab_adjusted_production_manifest.v1",
            "production_approved": True,
            "production_promotion_approved": True,
            "contract_pass": True,
            "source_provenance_pass": True,
            "selection_policy_pass": True,
            "selection_summary": str(selection),
            "selection_summary_sha256": _sha256(selection),
            "source_kind": "candidate_csv",
            "source_promotion_eligible": True,
            "source_provenance_manifest": str(source_provenance),
            "source_provenance_manifest_sha256": _sha256(source_provenance),
            "adjusted_csv": adjusted_csv,
            "adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
            **production_locked_holdout_fields,
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
            **locked_holdout_fields,
            **run_identity,
        },
    )
    selected_holdout_fields = dict(locked_holdout_fields)
    if divergent_selected_holdout_sha:
        selected_holdout_fields["locked_holdout_summary_sha256"] = "1" * 64
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
            **selected_holdout_fields,
            **run_identity,
        },
    )
    export_manifest_sha = _sha256(export_manifest)
    selected_config_sha = _sha256(selected_config)
    capstone_holdout_fields = dict(locked_holdout_fields)
    if divergent_capstone_holdout_sha:
        capstone_holdout_fields["locked_holdout_summary_sha256"] = "2" * 64
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
            **capstone_holdout_fields,
            **run_identity,
        },
    )
    return {
        "lab": lab,
        "governance": governance,
        "independent": independent,
        "product": product,
        "powerbi": powerbi,
        "ompex": ompex,
        "locked_holdout": locked_holdout,
        "production_manifest": production_manifest,
        "export_manifest": export_manifest,
        "selected_config": selected_config,
        "capstone": capstone,
    }


def _check_readiness_for_chain(paths, tmp_path, output_name="decision.json"):
    return check_readiness(
        lab_manifest=paths["lab"],
        governance_audit=paths["governance"],
        independent_summary=paths["independent"],
        product_summary=paths["product"],
        powerbi_summary=paths["powerbi"],
        ompex_advisory_delta=paths["ompex"],
        adjusted_production_manifest=paths["production_manifest"],
        adjusted_export_manifest=paths["export_manifest"],
        adjusted_selected_config=paths["selected_config"],
        adjusted_capstone=paths["capstone"],
        locked_holdout_summary=paths["locked_holdout"],
        output=tmp_path / output_name,
    )


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
    selection = _write_selection_summary(tmp_path, adjusted_csv=adjusted_csv)
    production_manifest = tmp_path / "prod.json"
    export_manifest = tmp_path / "export.json"
    selected_config = tmp_path / "selected.json"
    capstone = tmp_path / "capstone.json"
    locked_holdout = _write_locked_holdout(tmp_path)
    locked_holdout_fields = {
        "locked_holdout_summary": str(locked_holdout),
        "locked_holdout_summary_sha256": _sha256(locked_holdout),
        "locked_holdout_policy_pass": True,
    }
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
            "selection_policy_pass": True,
            "selection_summary": str(selection),
            "selection_summary_sha256": _sha256(selection),
            "source_kind": "candidate_csv",
            "source_promotion_eligible": True,
            "source_provenance_manifest": str(source_provenance),
            "source_provenance_manifest_sha256": _sha256(source_provenance),
            "adjusted_csv": adjusted_csv,
            "adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
            **locked_holdout_fields,
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
            **locked_holdout_fields,
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
            **locked_holdout_fields,
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
            **locked_holdout_fields,
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
        locked_holdout_summary=locked_holdout,
        output=tmp_path / "decision.json",
    )

    assert summary["approved"] is True
    assert summary["strict_diagnostics_pass"] is True
    assert summary["production_chain_pass"] is True
    assert summary["status"] == "PROMOTION_READY"
    assert summary["required_production_checks"] == REQUIRED_PRODUCTION_CHECKS

    holdout_payload = json.loads(locked_holdout.read_text(encoding="utf-8"))
    holdout_payload["same_path_hash_change"] = True
    locked_holdout.write_text(json.dumps(holdout_payload), encoding="utf-8")
    changed_summary = check_readiness(
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
        locked_holdout_summary=locked_holdout,
        output=tmp_path / "decision_changed_holdout.json",
    )
    changed_checks = {check["name"]: check["status"] for check in changed_summary["checks"]}
    assert changed_summary["approved"] is False
    assert changed_summary["production_chain_pass"] is False
    assert changed_checks["adjusted_production_manifest_locked_holdout_bound"] == "FAIL"


def test_epex_lab_readiness_rejects_divergent_export_locked_holdout_hash(tmp_path) -> None:
    lab, governance, independent, product, powerbi, ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source_provenance = _write_source_provenance(tmp_path, lab=lab, adjusted_csv=adjusted_csv)
    selection = _write_selection_summary(tmp_path, adjusted_csv=adjusted_csv)
    locked_holdout = _write_locked_holdout(tmp_path)
    production_manifest = tmp_path / "prod.json"
    export_manifest = tmp_path / "export.json"
    selected_config = tmp_path / "selected.json"
    capstone = tmp_path / "capstone.json"
    run_identity = {
        "production_run_id": "prod-run-1",
        "production_entrypoint": "pfc_shaping.pipeline.production_phases",
        "git_commit": "a" * 40,
    }
    locked_holdout_fields = {
        "locked_holdout_summary": str(locked_holdout),
        "locked_holdout_summary_sha256": _sha256(locked_holdout),
        "locked_holdout_policy_pass": True,
    }
    _write_json(
        production_manifest,
        {
            "schema_version": "epex_lab_adjusted_production_manifest.v1",
            "production_approved": True,
            "production_promotion_approved": True,
            "contract_pass": True,
            "source_provenance_pass": True,
            "selection_policy_pass": True,
            "selection_summary": str(selection),
            "selection_summary_sha256": _sha256(selection),
            "source_kind": "candidate_csv",
            "source_promotion_eligible": True,
            "source_provenance_manifest": str(source_provenance),
            "source_provenance_manifest_sha256": _sha256(source_provenance),
            "adjusted_csv": adjusted_csv,
            "adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
            **locked_holdout_fields,
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
            **locked_holdout_fields,
            "locked_holdout_summary_sha256": "0" * 64,
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
            **locked_holdout_fields,
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
            **locked_holdout_fields,
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
        locked_holdout_summary=locked_holdout,
        output=tmp_path / "decision.json",
    )

    checks = {check["name"]: check["status"] for check in summary["checks"]}
    assert summary["approved"] is False
    assert summary["production_chain_pass"] is False
    assert summary["required_production_checks"] == REQUIRED_PRODUCTION_CHECKS
    assert checks["adjusted_export_manifest_production_chain_bound"] == "FAIL"


def test_epex_lab_readiness_rejects_divergent_selected_locked_holdout_hash(tmp_path) -> None:
    paths = _write_approved_chain(tmp_path, divergent_selected_holdout_sha=True)

    summary = _check_readiness_for_chain(paths, tmp_path)

    checks = {check["name"]: check["status"] for check in summary["checks"]}
    assert summary["approved"] is False
    assert summary["production_chain_pass"] is False
    assert checks["adjusted_selected_artifact_production_chain_bound"] == "FAIL"


def test_epex_lab_readiness_rejects_divergent_capstone_locked_holdout_hash(tmp_path) -> None:
    paths = _write_approved_chain(tmp_path, divergent_capstone_holdout_sha=True)

    summary = _check_readiness_for_chain(paths, tmp_path)

    checks = {check["name"]: check["status"] for check in summary["checks"]}
    assert summary["approved"] is False
    assert summary["production_chain_pass"] is False
    assert checks["adjusted_capstone_production_chain_bound"] == "FAIL"


def test_epex_lab_readiness_rejects_locked_holdout_policy_false_on_bound_manifest(tmp_path) -> None:
    paths = _write_approved_chain(tmp_path, production_holdout_policy_pass=False)

    summary = _check_readiness_for_chain(paths, tmp_path)

    checks = {check["name"]: check["status"] for check in summary["checks"]}
    assert summary["approved"] is False
    assert summary["production_chain_pass"] is False
    assert checks["adjusted_production_manifest_locked_holdout_bound"] == "FAIL"


def test_epex_lab_readiness_requires_locked_holdout_for_complete_production_chain(tmp_path) -> None:
    lab, governance, independent, product, powerbi, ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source_provenance = _write_source_provenance(tmp_path, lab=lab, adjusted_csv=adjusted_csv)
    selection = _write_selection_summary(tmp_path, adjusted_csv=adjusted_csv)
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
            "selection_policy_pass": True,
            "selection_summary": str(selection),
            "selection_summary_sha256": _sha256(selection),
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

    checks = {check["name"]: check for check in summary["checks"]}
    assert summary["approved"] is False
    assert summary["production_chain_pass"] is False
    assert checks["locked_holdout_pass"]["status"] == "FAIL"
    assert checks["locked_holdout_pass"]["value"]["status"] == "MISSING_LOCKED_HOLDOUT"


def test_epex_lab_readiness_rejects_self_attested_selection_policy(tmp_path) -> None:
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
            "selection_policy_pass": True,
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
    assert checks["adjusted_production_manifest_selection_policy_pass"] == "FAIL"


def test_epex_lab_readiness_rejects_unbound_approved_export_manifest(tmp_path) -> None:
    lab, governance, independent, product, powerbi, ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source_provenance = _write_source_provenance(tmp_path, lab=lab, adjusted_csv=adjusted_csv)
    selection = _write_selection_summary(tmp_path, adjusted_csv=adjusted_csv)
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
            "selection_policy_pass": True,
            "selection_summary": str(selection),
            "selection_summary_sha256": _sha256(selection),
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
    selection = _write_selection_summary(tmp_path, adjusted_csv=adjusted_csv)
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
            "selection_policy_pass": True,
            "selection_summary": str(selection),
            "selection_summary_sha256": _sha256(selection),
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
    selection = _write_selection_summary(tmp_path, adjusted_csv=adjusted_csv)
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
            "selection_policy_pass": True,
            "selection_summary": str(selection),
            "selection_summary_sha256": _sha256(selection),
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

from __future__ import annotations

import json
import hashlib

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


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_source_provenance(tmp_path, lab_manifest, adjusted_csv: str, *, source_kind: str = "candidate_csv"):
    source = tmp_path / "source_provenance.json"
    source_csv = tmp_path / "baseline.csv"
    source_export_manifest = tmp_path / "source_export_manifest.json"
    blockers = [] if source_kind == "candidate_csv" else ["source_kind_fan_parquet_requires_audited_hourly_export"]
    _write_json(
        source_export_manifest,
        {
            "schema_version": "test_source_export_manifest.v1",
            "candidate_csv": str(source_csv),
            "candidate_csv_sha256": _sha256(source_csv),
        },
    )
    _write_json(
        source,
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
            "source_export_manifest": str(source_export_manifest),
            "source_export_manifest_sha256": _sha256(source_export_manifest),
            "staged_candidate_csv": str(source_csv),
            "staged_candidate_csv_sha256": _sha256(source_csv),
            "lab_manifest": str(lab_manifest),
            "lab_manifest_sha256": _sha256(lab_manifest),
            "production_contract_blockers": blockers,
            "adjusted_csv": adjusted_csv,
            "adjusted_csv_sha256": _sha256(tmp_path / "adjusted.csv"),
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
        },
    )
    return source


def _write_selection_summary(
    tmp_path,
    adjusted_csv: str,
    *,
    replace_incumbent: bool = True,
    include_ompex_flags: bool = True,
    ompex_used_in_model: bool = False,
    ompex_used_in_selection: bool = False,
    ompex_used_in_backtest: bool = False,
    selected_sha: str | None = None,
):
    selection = tmp_path / "selection_summary.json"
    sha = selected_sha or _sha256(tmp_path / "adjusted.csv")
    payload = {
        "replacement_verdict": {"replace_incumbent": replace_incumbent},
        "selected_trial": {
            "adjusted_csv_sha256": sha,
        },
        "selected_adjusted_csv_sha256": sha,
    }
    if include_ompex_flags:
        payload.update(
            {
                "ompex_used_in_model": ompex_used_in_model,
                "ompex_used_in_selection": ompex_used_in_selection,
                "ompex_used_in_backtest": ompex_used_in_backtest,
            }
        )
    _write_json(
        selection,
        payload,
    )
    return selection


def _write_locked_holdout(tmp_path, *, passed: bool = True):
    locked_holdout = tmp_path / "locked_holdout.json"
    _write_json(
        locked_holdout,
        {
            "schema_version": "epex_lab_locked_holdout_run.v1",
            "status": "LOCKED_HOLDOUT_PASS" if passed else "WAITING_FOR_FULL_SPOT_COVERAGE",
            "promotion_gate": False,
            "production_approved": False,
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
            "ompex_used_in_backtest": False,
            "coverage_ready": passed,
            "backtest_ran": passed,
            "audit_ran": passed,
            "holdout_pass": passed,
        },
    )
    return locked_holdout


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
    assert manifest["contract_pass"] is False
    assert manifest["production_approved"] is False
    assert manifest["production_promotion_approved"] is False
    assert manifest["source_provenance_pass"] is False
    assert manifest["adjusted_csv_sha256"]
    assert manifest["lab_manifest_sha256"]
    checks = {check["name"]: check["status"] for check in manifest["checks"]}
    assert checks["source_provenance_manifest_present"] == "FAIL"
    assert "locked_holdout_pass" not in checks


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


def test_adjusted_production_manifest_approval_requires_run_identity_and_source_provenance(tmp_path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, _ompex = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match="source_provenance_manifest"):
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


def test_adjusted_production_manifest_approval_rejects_invalid_git_commit(tmp_path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, _ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source = _write_source_provenance(tmp_path, lab, adjusted_csv, source_kind="candidate_csv")
    selection = _write_selection_summary(tmp_path, adjusted_csv)
    locked_holdout = _write_locked_holdout(tmp_path)

    with pytest.raises(ValueError, match="git_commit_must_be_40_hex"):
        build_manifest(
            lab_manifest=lab,
            baseline_monthly_manifest=monthly,
            product_summary=product,
            powerbi_summary=powerbi,
            source_hierarchy_policy=policy,
            independent_summary=independent,
            governance_audit=governance,
            selection_summary=selection,
            production_run_id="prod-run-1",
            production_entrypoint="pfc_shaping.pipeline.production_phases",
            git_commit="not-a-commit",
            source_provenance_manifest=source,
            locked_holdout_summary=locked_holdout,
            production_approved=True,
            production_promotion_approved=True,
            output=tmp_path / "adjusted_production_manifest.json",
        )


def test_adjusted_production_manifest_approval_requires_locked_holdout(tmp_path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, _ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source = _write_source_provenance(tmp_path, lab, adjusted_csv, source_kind="candidate_csv")
    selection = _write_selection_summary(tmp_path, adjusted_csv)

    with pytest.raises(ValueError, match="locked holdout pass"):
        build_manifest(
            lab_manifest=lab,
            baseline_monthly_manifest=monthly,
            product_summary=product,
            powerbi_summary=powerbi,
            source_hierarchy_policy=policy,
            independent_summary=independent,
            governance_audit=governance,
            selection_summary=selection,
            production_run_id="prod-run-1",
            production_entrypoint="pfc_shaping.pipeline.production_phases",
            git_commit="a" * 40,
            source_provenance_manifest=source,
            production_approved=True,
            production_promotion_approved=True,
            output=tmp_path / "adjusted_production_manifest.json",
        )


def test_adjusted_production_manifest_approval_rejects_failed_locked_holdout(tmp_path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, _ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source = _write_source_provenance(tmp_path, lab, adjusted_csv, source_kind="candidate_csv")
    selection = _write_selection_summary(tmp_path, adjusted_csv)
    locked_holdout = _write_locked_holdout(tmp_path, passed=False)

    with pytest.raises(ValueError, match="NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING"):
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
            source_provenance_manifest=source,
            production_approved=True,
            production_promotion_approved=True,
            output=tmp_path / "adjusted_production_manifest.json",
        )


def test_adjusted_production_manifest_can_be_approved_with_candidate_csv_source_provenance(tmp_path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, _ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source = _write_source_provenance(tmp_path, lab, adjusted_csv, source_kind="candidate_csv")
    selection = _write_selection_summary(tmp_path, adjusted_csv)
    locked_holdout = _write_locked_holdout(tmp_path)

    manifest = build_manifest(
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
        source_provenance_manifest=source,
        production_approved=True,
        production_promotion_approved=True,
        output=tmp_path / "adjusted_production_manifest.json",
    )

    assert manifest["contract_pass"] is True
    assert manifest["source_kind"] == "candidate_csv"
    assert manifest["source_provenance_pass"] is True
    assert manifest["selection_policy_pass"] is True
    assert manifest["locked_holdout_policy_pass"] is True
    assert manifest["locked_holdout_summary_sha256"] == _sha256(locked_holdout)
    assert manifest["production_approved"] is True
    assert manifest["production_promotion_approved"] is True


def test_adjusted_production_manifest_approval_requires_selection_policy(tmp_path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, _ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source = _write_source_provenance(tmp_path, lab, adjusted_csv, source_kind="candidate_csv")
    selection = _write_selection_summary(tmp_path, adjusted_csv, replace_incumbent=False)
    locked_holdout = _write_locked_holdout(tmp_path)

    manifest = build_manifest(
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
        source_provenance_manifest=source,
        production_approved=True,
        production_promotion_approved=True,
        output=tmp_path / "adjusted_production_manifest.json",
    )

    checks = {check["name"]: check["status"] for check in manifest["checks"]}
    assert checks["selection_policy_pass"] == "FAIL"
    assert manifest["contract_pass"] is False
    assert manifest["selection_policy_pass"] is False
    assert manifest["production_approved"] is False
    assert manifest["production_promotion_approved"] is False


def test_adjusted_production_manifest_selection_policy_requires_explicit_no_ompex(tmp_path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, _ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source = _write_source_provenance(tmp_path, lab, adjusted_csv, source_kind="candidate_csv")
    selection = _write_selection_summary(tmp_path, adjusted_csv, include_ompex_flags=False)
    locked_holdout = _write_locked_holdout(tmp_path)

    manifest = build_manifest(
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
        source_provenance_manifest=source,
        production_approved=True,
        production_promotion_approved=True,
        output=tmp_path / "adjusted_production_manifest.json",
    )

    checks = {check["name"]: check["status"] for check in manifest["checks"]}
    assert checks["selection_policy_pass"] == "FAIL"
    assert manifest["selection_policy_pass"] is False
    assert manifest["production_approved"] is False


def test_adjusted_production_manifest_selection_policy_rejects_ompex_flags(tmp_path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, _ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source = _write_source_provenance(tmp_path, lab, adjusted_csv, source_kind="candidate_csv")
    selection = _write_selection_summary(tmp_path, adjusted_csv, ompex_used_in_backtest=True)
    locked_holdout = _write_locked_holdout(tmp_path)

    manifest = build_manifest(
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
        source_provenance_manifest=source,
        production_approved=True,
        production_promotion_approved=True,
        output=tmp_path / "adjusted_production_manifest.json",
    )

    checks = {check["name"]: check["status"] for check in manifest["checks"]}
    assert checks["selection_policy_pass"] == "FAIL"
    assert manifest["selection_policy_pass"] is False
    assert manifest["production_approved"] is False


def test_adjusted_production_manifest_selection_policy_requires_selected_sha_match(tmp_path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, _ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source = _write_source_provenance(tmp_path, lab, adjusted_csv, source_kind="candidate_csv")
    selection = _write_selection_summary(tmp_path, adjusted_csv, selected_sha="0" * 64)
    locked_holdout = _write_locked_holdout(tmp_path)

    manifest = build_manifest(
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
        source_provenance_manifest=source,
        production_approved=True,
        production_promotion_approved=True,
        output=tmp_path / "adjusted_production_manifest.json",
    )

    checks = {check["name"]: check["status"] for check in manifest["checks"]}
    assert checks["selection_policy_pass"] == "FAIL"
    assert manifest["selection_policy_pass"] is False
    assert manifest["production_approved"] is False


def test_adjusted_production_manifest_rejects_fan_source_provenance(tmp_path) -> None:
    lab, monthly, product, powerbi, policy, independent, governance, _ompex = _write_inputs(tmp_path)
    adjusted_csv = json.loads(lab.read_text(encoding="utf-8"))["outputs"]["adjusted_csv"]
    source = _write_source_provenance(tmp_path, lab, adjusted_csv, source_kind="fan_parquet")
    selection = _write_selection_summary(tmp_path, adjusted_csv)
    locked_holdout = _write_locked_holdout(tmp_path)

    manifest = build_manifest(
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
        source_provenance_manifest=source,
        production_approved=True,
        production_promotion_approved=True,
        output=tmp_path / "adjusted_production_manifest.json",
    )

    checks = {check["name"]: check["status"] for check in manifest["checks"]}
    assert checks["source_provenance_candidate_csv"] == "FAIL"
    assert checks["source_provenance_promotion_eligible"] == "FAIL"
    assert manifest["contract_pass"] is False
    assert manifest["source_provenance_pass"] is False
    assert manifest["production_approved"] is False

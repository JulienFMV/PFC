"""Check promotion readiness for a selected EPEX shape-lab artifact.

This checker intentionally separates strict diagnostic evidence from production
promotion evidence.  A lab artifact can pass product and Power BI diagnostics
while still being NO-GO production until it has its own production/export/
selected/capstone chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from scripts.epex_lab_selection_policy import selection_policy_manifest_value
    from scripts.epex_lab_locked_holdout_policy import ENERGY_CHARTS_RUN_SCHEMA_VERSION
    from scripts.epex_lab_locked_holdout_policy import locked_holdout_policy as evaluate_locked_holdout_policy
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from epex_lab_selection_policy import selection_policy_manifest_value
    from epex_lab_locked_holdout_policy import ENERGY_CHARTS_RUN_SCHEMA_VERSION
    from epex_lab_locked_holdout_policy import locked_holdout_policy as evaluate_locked_holdout_policy


REQUIRED_PRODUCTION_CHECKS = [
    "adjusted_production_manifest_approved",
    "adjusted_production_manifest_run_identity_valid",
    "adjusted_production_manifest_locked_holdout_bound",
    "adjusted_export_manifest_production_ready",
    "adjusted_export_manifest_production_chain_bound",
    "adjusted_selected_artifact_production_ready",
    "adjusted_selected_artifact_production_chain_bound",
    "adjusted_capstone_approved",
    "adjusted_capstone_production_chain_bound",
    "locked_holdout_pass",
    "locked_holdout_queue_pass",
]


def check_readiness(
    *,
    lab_manifest: Path,
    governance_audit: Path,
    independent_summary: Path,
    product_summary: Path,
    powerbi_summary: Path,
    ompex_advisory_delta: Path | None = None,
    adjusted_production_manifest: Path | None = None,
    adjusted_export_manifest: Path | None = None,
    adjusted_selected_config: Path | None = None,
    adjusted_capstone: Path | None = None,
    locked_holdout_summary: Path | None = None,
    locked_holdout_queue_summary: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    lab = _load_json(lab_manifest)
    governance = _load_json(governance_audit)
    independent = _load_json(independent_summary)
    product = _load_json(product_summary)
    powerbi = _load_powerbi_summary(powerbi_summary)
    ompex = _load_json(ompex_advisory_delta) if ompex_advisory_delta is not None else None

    diagnostics_checks = [
        _check("lab_activation_lab_only", lab.get("activation_status") == "lab_only", lab.get("activation_status")),
        _check("lab_not_production_approved", lab.get("production_approved") is False, lab.get("production_approved")),
        _check("lab_ompex_not_selection", lab.get("ompex_used_in_selection") is False, lab.get("ompex_used_in_selection")),
        _check("governance_pass", governance.get("status") == "PASS", governance.get("status")),
        _check(
            "independent_no_ompex",
            independent.get("benchmark_policy") == "independent_no_ompex"
            and independent.get("ompex_used_in_model") is False
            and independent.get("ompex_used_in_selection") is False,
            independent.get("benchmark_policy"),
        ),
        _check("product_all_gates_pass", product.get("all_gates_pass") is True, product.get("all_gates_pass")),
        _check("product_no_critical", int(product.get("critical_count", -1)) == 0, product.get("critical_count")),
        _check("product_no_unsupported", int(product.get("unsupported_count", -1)) == 0, product.get("unsupported_count")),
        _check(
            "product_no_blocking_quote_conflicts",
            int(product.get("blocking_quote_conflict_count", -1)) == 0,
            product.get("blocking_quote_conflict_count"),
        ),
        _check(
            "powerbi_quality_gate_pass",
            powerbi.get("powerbi_quality_gate_status") == "PASS",
            powerbi.get("powerbi_quality_gate_status"),
        ),
        _check(
            "powerbi_no_weighted_negative_hours",
            _float_value(powerbi.get("weighted_negative_hours")) == 0.0,
            powerbi.get("weighted_negative_hours"),
        ),
        _check(
            "powerbi_no_critical_flags",
            _powerbi_critical_count(powerbi) == 0,
            _powerbi_critical_count(powerbi),
        ),
    ]
    if ompex is not None:
        diagnostics_checks.append(
            _check(
                "ompex_advisory_not_selection",
                ompex.get("read_only") is True
                and ompex.get("ompex_used_in_model") is False
                and ompex.get("ompex_used_in_selection") is False,
                ompex.get("benchmark_policy"),
            )
        )

    checks = list(diagnostics_checks)
    production_paths = {
        "adjusted_production_manifest": adjusted_production_manifest,
        "adjusted_export_manifest": adjusted_export_manifest,
        "adjusted_selected_config": adjusted_selected_config,
        "adjusted_capstone": adjusted_capstone,
    }
    missing_production_evidence = [
        name for name, path in production_paths.items() if path is None or not path.exists()
    ]
    production_bundle_complete = not missing_production_evidence
    locked_holdout_required = bool(production_bundle_complete or locked_holdout_summary is not None)
    locked_holdout_policy = None
    locked_holdout_valid = not locked_holdout_required
    locked_holdout_sha256 = None
    if locked_holdout_summary is not None and locked_holdout_summary.exists():
        locked_holdout_policy = evaluate_locked_holdout_policy(_load_json(locked_holdout_summary))
        locked_holdout_valid = locked_holdout_policy.get("pass") is True
        locked_holdout_sha256 = _sha256(locked_holdout_summary)
    elif locked_holdout_required:
        locked_holdout_policy = {"provided": False, "pass": False, "status": "MISSING_LOCKED_HOLDOUT"}
        locked_holdout_valid = False
    if locked_holdout_required:
        checks.append(
            _check(
                "locked_holdout_pass",
                locked_holdout_valid,
                locked_holdout_policy,
            )
        )
    locked_holdout_queue_policy = None
    locked_holdout_queue_valid = True
    locked_holdout_queue_required = bool(production_bundle_complete or locked_holdout_queue_summary is not None)
    if locked_holdout_queue_summary is not None and locked_holdout_queue_summary.exists():
        locked_holdout_queue_policy = _locked_holdout_queue_policy(
            _load_json(locked_holdout_queue_summary),
            locked_holdout_policy=locked_holdout_policy,
        )
        locked_holdout_queue_valid = locked_holdout_queue_policy["pass"] is True
        checks.append(
            _check(
                "locked_holdout_queue_pass",
                locked_holdout_queue_valid,
                locked_holdout_queue_policy,
            )
        )
    elif locked_holdout_queue_required:
        locked_holdout_queue_policy = {
            "provided": False,
            "pass": False,
            "status": "MISSING_LOCKED_HOLDOUT_QUEUE_SUMMARY",
            "path": str(locked_holdout_queue_summary) if locked_holdout_queue_summary is not None else None,
        }
        locked_holdout_queue_valid = False
        checks.append(
            _check(
                "locked_holdout_queue_pass",
                locked_holdout_queue_valid,
                locked_holdout_queue_policy,
            )
        )
    if adjusted_capstone is not None and adjusted_capstone.exists():
        capstone = _load_json(adjusted_capstone)
        capstone_schema_ok = bool(capstone.get("schema_version"))
        checks.extend(
            [
                _check("adjusted_capstone_schema", capstone_schema_ok, capstone.get("schema_version")),
            ]
        )
    else:
        capstone = None
        capstone_schema_ok = False
        capstone_approved = False
    if adjusted_production_manifest is not None and adjusted_production_manifest.exists():
        production_manifest = _load_json(adjusted_production_manifest)
        production_manifest_sha256 = _sha256(adjusted_production_manifest)
        production_manifest_bound = _manifest_bound_to_adjusted_csv(
            production_manifest,
            path_key="adjusted_csv",
            sha_key="adjusted_csv_sha256",
            adjusted_csv=(lab.get("outputs") or {}).get("adjusted_csv"),
        )
        source_provenance_checks = _production_manifest_source_provenance_checks(
            production_manifest,
            adjusted_csv=Path(str((lab.get("outputs") or {}).get("adjusted_csv", ""))),
        )
        source_provenance_valid = all(check["status"] == "PASS" for check in source_provenance_checks)
        selection_policy_value = selection_policy_manifest_value(
            production_manifest,
            adjusted_csv=Path(str((lab.get("outputs") or {}).get("adjusted_csv", ""))),
        )
        selection_policy_valid = selection_policy_value.get("validated") is True
        production_manifest_identity_valid = _production_run_identity_valid(production_manifest)
        production_manifest_locked_holdout_bound = _artifact_bound_to_locked_holdout(
            production_manifest,
            locked_holdout_path=locked_holdout_summary,
            locked_holdout_sha256=locked_holdout_sha256,
        )
        production_manifest_approved = (
            production_manifest.get("schema_version") == "epex_lab_adjusted_production_manifest.v1"
            and production_manifest.get("production_approved") is True
            and production_manifest.get("production_promotion_approved") is True
            and production_manifest.get("contract_pass") is True
            and production_manifest.get("source_provenance_pass") is True
            and selection_policy_valid
            and production_manifest_bound
            and production_manifest_locked_holdout_bound
            and source_provenance_valid
            and production_manifest_identity_valid
        )
        checks.extend(
            [
                _check(
                    "adjusted_production_manifest_schema",
                    production_manifest.get("schema_version") == "epex_lab_adjusted_production_manifest.v1",
                    production_manifest.get("schema_version"),
                ),
                _check(
                    "adjusted_production_manifest_bound",
                    production_manifest_bound,
                    production_manifest.get("adjusted_csv"),
                ),
                _check(
                    "adjusted_production_manifest_approved",
                    production_manifest.get("production_approved") is True
                    and production_manifest.get("production_promotion_approved") is True,
                    {
                        "production_approved": production_manifest.get("production_approved"),
                        "production_promotion_approved": production_manifest.get("production_promotion_approved"),
                    },
                ),
                _check(
                    "adjusted_production_manifest_contract_pass",
                    production_manifest.get("contract_pass") is True,
                    production_manifest.get("contract_pass"),
                ),
                _check(
                    "adjusted_production_manifest_source_provenance_pass",
                    production_manifest.get("source_provenance_pass") is True and source_provenance_valid,
                    {
                        "self_attested": production_manifest.get("source_provenance_pass"),
                        "validated": source_provenance_valid,
                    },
                ),
                _check(
                    "adjusted_production_manifest_selection_policy_pass",
                    selection_policy_valid,
                    selection_policy_value,
                ),
                _check(
                    "adjusted_production_manifest_run_identity_valid",
                    production_manifest_identity_valid,
                    _production_run_identity_value(production_manifest),
                ),
                _check(
                    "adjusted_production_manifest_locked_holdout_bound",
                    production_manifest_locked_holdout_bound,
                    _locked_holdout_binding_value(production_manifest),
                ),
            ]
        )
        checks.extend(source_provenance_checks)
    else:
        production_manifest = None
        production_manifest_sha256 = None
        production_manifest_approved = False
    if adjusted_export_manifest is not None and adjusted_export_manifest.exists():
        export_manifest = _load_json(adjusted_export_manifest)
        export_manifest_bound = _manifest_bound_to_adjusted_csv(
            export_manifest,
            path_key="adjusted_csv",
            sha_key="adjusted_csv_sha256",
            adjusted_csv=(lab.get("outputs") or {}).get("adjusted_csv"),
        )
        export_manifest_schema_ok = export_manifest.get("schema_version") == "epex_lab_adjusted_export_manifest.v1"
        export_manifest_chain_bound = _artifact_bound_to_production_manifest(
            export_manifest,
            production_manifest_path=adjusted_production_manifest,
            production_manifest_sha256=production_manifest_sha256,
            production_manifest=production_manifest,
        )
        export_manifest_production_ready = (
            export_manifest_schema_ok
            and export_manifest.get("production_approved") is True
            and export_manifest.get("production_promotion_approved") is True
            and export_manifest_bound
            and export_manifest_chain_bound
        )
        checks.extend(
            [
                _check(
                    "adjusted_export_manifest_schema",
                    export_manifest_schema_ok,
                    export_manifest.get("schema_version"),
                ),
                _check(
                    "adjusted_export_manifest_bound",
                    export_manifest_bound,
                    export_manifest.get("adjusted_csv"),
                ),
                _check(
                    "adjusted_export_manifest_production_ready",
                    export_manifest_production_ready,
                    {
                        "production_approved": export_manifest.get("production_approved"),
                        "production_promotion_approved": export_manifest.get("production_promotion_approved"),
                    },
                ),
                _check(
                    "adjusted_export_manifest_production_chain_bound",
                    export_manifest_chain_bound,
                    _production_chain_binding_value(export_manifest),
                ),
            ]
        )
    else:
        export_manifest = None
        export_manifest_production_ready = False
    if adjusted_selected_config is not None and adjusted_selected_config.exists():
        selected_artifact = _load_json(adjusted_selected_config)
        selected_artifact_bound = _manifest_bound_to_adjusted_csv(
            selected_artifact,
            path_key="selected_adjusted_csv",
            sha_key="selected_adjusted_csv_sha256",
            adjusted_csv=(lab.get("outputs") or {}).get("adjusted_csv"),
        )
        selected_artifact_schema_ok = selected_artifact.get("schema_version") == "epex_lab_selected_artifact.v1"
        selected_artifact_chain_bound = _artifact_bound_to_production_manifest(
            selected_artifact,
            production_manifest_path=adjusted_production_manifest,
            production_manifest_sha256=production_manifest_sha256,
            production_manifest=production_manifest,
        )
        selected_artifact_production_ready = (
            selected_artifact_schema_ok
            and
            selected_artifact.get("production_approved") is True
            and selected_artifact.get("production_promotion_approved") is True
            and selected_artifact.get("selection_status") == "PRODUCTION_APPROVED"
            and selected_artifact_bound
            and selected_artifact_chain_bound
        )
        checks.extend(
            [
                _check(
                    "adjusted_selected_artifact_schema",
                    selected_artifact_schema_ok,
                    selected_artifact.get("schema_version"),
                ),
                _check(
                    "adjusted_selected_artifact_bound",
                    selected_artifact_bound,
                    selected_artifact.get("selected_adjusted_csv"),
                ),
                _check(
                    "adjusted_selected_artifact_production_ready",
                    selected_artifact_production_ready,
                    {
                        "selection_status": selected_artifact.get("selection_status"),
                        "production_approved": selected_artifact.get("production_approved"),
                        "production_promotion_approved": selected_artifact.get("production_promotion_approved"),
                    },
                ),
                _check(
                    "adjusted_selected_artifact_production_chain_bound",
                    selected_artifact_chain_bound,
                    _production_chain_binding_value(selected_artifact),
                ),
            ]
        )
    else:
        selected_artifact = None
        selected_artifact_production_ready = False
    if capstone is not None:
        capstone_chain_bound = _capstone_bound_to_production_chain(
            capstone,
            production_manifest_path=adjusted_production_manifest,
            production_manifest_sha256=production_manifest_sha256,
            production_manifest=production_manifest,
            export_manifest_path=adjusted_export_manifest,
            export_manifest=export_manifest,
            selected_artifact_path=adjusted_selected_config,
            selected_artifact=selected_artifact,
        )
        capstone_approved = (
            capstone_schema_ok
            and capstone.get("approved") is True
            and capstone.get("production_chain_pass") is True
            and capstone_chain_bound
        )
        checks.extend(
            [
                _check(
                    "adjusted_capstone_approved",
                    capstone_approved,
                    {
                        "approved": capstone.get("approved"),
                        "production_chain_pass": capstone.get("production_chain_pass"),
                    },
                ),
                _check(
                    "adjusted_capstone_production_chain_bound",
                    capstone_chain_bound,
                    _production_chain_binding_value(capstone),
                ),
            ]
        )
    strict_diagnostics_pass = all(check["status"] == "PASS" for check in diagnostics_checks)
    production_chain_pass = (
        not missing_production_evidence
        and locked_holdout_valid
        and locked_holdout_queue_valid
        and production_manifest_approved
        and export_manifest_production_ready
        and selected_artifact_production_ready
        and capstone_approved
    )
    approved = bool(strict_diagnostics_pass and production_chain_pass)
    check_by_name = {str(check.get("name")): check for check in checks}
    failed_production_checks = [
        name for name in REQUIRED_PRODUCTION_CHECKS if (check_by_name.get(name) or {}).get("status") == "FAIL"
    ]
    missing_production_checks = [name for name in REQUIRED_PRODUCTION_CHECKS if name not in check_by_name]
    production_blocking_stage, next_required_step = _production_blocking_route(
        approved=approved,
        strict_diagnostics_pass=bool(strict_diagnostics_pass),
        missing_production_evidence=missing_production_evidence,
        missing_production_checks=missing_production_checks,
        failed_production_checks=failed_production_checks,
        locked_holdout_policy=locked_holdout_policy,
        locked_holdout_queue_policy=locked_holdout_queue_policy,
    )
    recommended_commands = _recommended_commands(
        production_blocking_stage=production_blocking_stage,
        locked_holdout_policy=locked_holdout_policy,
    )
    status = (
        "PROMOTION_READY"
        if approved
        else "STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING"
        if strict_diagnostics_pass
        else "STRICT_DIAGNOSTICS_FAIL"
    )
    summary = {
        "schema_version": "epex_lab_promotion_readiness.v1",
        "approved": approved,
        "status": status,
        "strict_diagnostics_pass": bool(strict_diagnostics_pass),
        "production_chain_pass": bool(production_chain_pass),
        "missing_production_evidence": missing_production_evidence,
        "missing_production_checks": missing_production_checks,
        "failed_production_checks": failed_production_checks,
        "production_blocking_stage": production_blocking_stage,
        "next_required_step": next_required_step,
        "recommended_commands": recommended_commands,
        "required_production_checks": REQUIRED_PRODUCTION_CHECKS,
        "checks": checks,
        "selected_adjusted_csv": (lab.get("outputs") or {}).get("adjusted_csv"),
        "lab_manifest": str(lab_manifest),
        "governance_audit": str(governance_audit),
        "independent_summary": str(independent_summary),
        "product_summary": str(product_summary),
        "powerbi_summary": str(powerbi_summary),
        "ompex_advisory_delta": str(ompex_advisory_delta) if ompex_advisory_delta is not None else None,
        "locked_holdout_summary": str(locked_holdout_summary) if locked_holdout_summary is not None else None,
        "locked_holdout_policy": locked_holdout_policy,
        "locked_holdout_queue_summary": (
            str(locked_holdout_queue_summary) if locked_holdout_queue_summary is not None else None
        ),
        "locked_holdout_queue_policy": locked_holdout_queue_policy,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return summary


def _locked_holdout_queue_policy(
    queue: dict[str, Any],
    *,
    locked_holdout_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    checks = {
        "schema_version": queue.get("schema_version") == "epex_lab_locked_holdout_queue_audit.v1",
        "read_only": queue.get("read_only") is True,
        "promotion_gate_false": queue.get("promotion_gate") is False,
        "production_approved_false": queue.get("production_approved") is False,
        "invalid_plan_count_zero": int(queue.get("invalid_plan_count", -1)) == 0,
        "policy_invalid_plan_count_zero": int(queue.get("policy_invalid_plan_count", -1)) == 0,
        "artifact_invalid_plan_count_zero": int(queue.get("artifact_invalid_plan_count", -1)) == 0,
        "duplicate_plan_id_count_zero": int(queue.get("duplicate_plan_id_count", -1)) == 0,
        "overlapping_window_count_zero": int(queue.get("overlapping_window_count", -1)) == 0,
        "queue_issues_empty": list(queue.get("queue_issues") or []) == [],
    }
    expected_plan_sha = None
    if locked_holdout_policy is not None:
        expected_plan_sha = locked_holdout_policy.get("plan_json_sha256") or locked_holdout_policy.get(
            "expected_plan_json_sha256"
        )
    plan_shas = {
        str(row.get("plan_json_sha256"))
        for row in queue.get("plans", [])
        if isinstance(row, dict) and row.get("plan_json_sha256")
    }
    if expected_plan_sha:
        checks["locked_holdout_plan_in_queue"] = str(expected_plan_sha) in plan_shas
    queue_status = queue.get("status")
    passed = all(checks.values())
    return {
        "provided": True,
        "pass": passed,
        "status": "LOCKED_HOLDOUT_QUEUE_PASS" if passed else _locked_holdout_queue_no_go_status(queue_status),
        "queue_status": queue_status,
        "plan_count": queue.get("plan_count"),
        "expected_locked_holdout_plan_sha256": expected_plan_sha,
        "checks": checks,
    }


def _locked_holdout_queue_no_go_status(queue_status: Any) -> str:
    return "NO_GO_LOCKED_HOLDOUT_QUEUE_INVALID"


def _production_blocking_route(
    *,
    approved: bool,
    strict_diagnostics_pass: bool,
    missing_production_evidence: list[str],
    missing_production_checks: list[str],
    failed_production_checks: list[str],
    locked_holdout_policy: dict[str, Any] | None,
    locked_holdout_queue_policy: dict[str, Any] | None,
) -> tuple[str, str]:
    if approved:
        return "promotion_ready", "run_independent_capstone_review"
    if not strict_diagnostics_pass:
        return "strict_diagnostics", "resolve_strict_diagnostic_failures"
    if locked_holdout_policy is not None and locked_holdout_policy.get("pass") is not True:
        status = str(locked_holdout_policy.get("status") or "")
        if status == "MISSING_LOCKED_HOLDOUT":
            return "locked_holdout_missing", "provide_passing_locked_holdout_summary"
        if status == "NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING":
            return "locked_holdout_coverage", "wait_for_full_spot_coverage_then_run_locked_holdout"
        if status == "NO_GO_LOCKED_HOLDOUT_INPUT_INVALID":
            return "locked_holdout_input_invalid", "fix_locked_holdout_plan_or_spot_inputs_then_rerun_preflight"
        return "locked_holdout_failure", "resolve_locked_holdout_failure_without_retuning_locked_window"
    if locked_holdout_queue_policy is not None and locked_holdout_queue_policy.get("pass") is not True:
        return "locked_holdout_queue", "fix_locked_holdout_queue_before_promotion_review"
    if missing_production_evidence:
        return "production_evidence", "generate_adjusted_production_export_selected_capstone_evidence"
    if missing_production_checks:
        return "readiness_contract", "regenerate_readiness_with_required_production_checks"
    if failed_production_checks:
        return "production_checks", "replace_local_diagnostic_flags_with_real_production_artifacts"
    return "production_chain", "investigate_production_chain_failure"


def _recommended_commands(
    *,
    production_blocking_stage: str,
    locked_holdout_policy: dict[str, Any] | None,
) -> dict[str, str]:
    if production_blocking_stage != "locked_holdout_coverage":
        return {}
    plan_json = (
        str(locked_holdout_policy.get("plan_json"))
        if locked_holdout_policy is not None and locked_holdout_policy.get("plan_json")
        else "<T057_PLAN_JSON>"
    )
    plan_sha = (
        str(locked_holdout_policy.get("expected_plan_json_sha256") or locked_holdout_policy.get("plan_json_sha256"))
        if locked_holdout_policy is not None
        and (locked_holdout_policy.get("expected_plan_json_sha256") or locked_holdout_policy.get("plan_json_sha256"))
        else "<T057_PLAN_JSON_SHA256>"
    )
    energy_output_dir = (
        str(locked_holdout_policy.get("output_dir"))
        if locked_holdout_policy is not None
        and locked_holdout_policy.get("schema_version") == ENERGY_CHARTS_RUN_SCHEMA_VERSION
        and locked_holdout_policy.get("output_dir")
        else "<ENERGY_CHARTS_LOCKED_HOLDOUT_OUTPUT_DIR>"
    )
    bzn = (
        str(locked_holdout_policy.get("bzn"))
        if locked_holdout_policy is not None and locked_holdout_policy.get("bzn")
        else "CH"
    )
    return {
        "run_energy_charts_locked_holdout": (
            "python scripts/run_energy_charts_epex_locked_holdout.py "
            f"--plan-json {_quote_cli_arg(plan_json)} "
            f"--expected-plan-sha256 {_quote_cli_arg(plan_sha)} "
            f"--output-dir {_quote_cli_arg(energy_output_dir)} "
            f"--bzn {_quote_cli_arg(bzn)}"
        ),
        "run_locked_holdout": (
            "python scripts/run_epex_lab_locked_holdout.py "
            f"--plan-json {_quote_cli_arg(plan_json)} "
            f"--expected-plan-sha256 {_quote_cli_arg(plan_sha)} "
            "--spot-parquet <FRESH_FUTURE_SPOT_PARQUET> "
            "--output-dir <T057_HOLDOUT_OUTPUT_DIR>"
        ),
    }


def _quote_cli_arg(value: str) -> str:
    if value.startswith("<") and value.endswith(">"):
        return value
    if not value or any(char.isspace() for char in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def _check(name: str, passed: bool, value: Any) -> dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "value": value}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_powerbi_summary(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    if not {"metric", "value"}.issubset(frame.columns):
        raise ValueError(f"Power BI summary must contain metric,value columns: {path}")
    return {str(row.metric): str(row.value) for row in frame.itertuples(index=False)}


def _float_value(value: Any) -> float:
    return float(value)


def _powerbi_critical_count(summary: dict[str, str]) -> int:
    total = 0
    for key, value in summary.items():
        if key.endswith("_critical_flags"):
            total += int(float(value))
    return total


def _same_path(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    left_text = os.path.normcase(os.path.normpath(str(left)))
    right_text = os.path.normcase(os.path.normpath(str(right)))
    if left_text == right_text:
        return True
    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except (OSError, TypeError, ValueError):
        return False


def _manifest_bound_to_adjusted_csv(
    manifest: dict[str, Any],
    *,
    path_key: str,
    sha_key: str,
    adjusted_csv: Any,
) -> bool:
    if _same_path(manifest.get(path_key), adjusted_csv):
        return True
    expected_sha = manifest.get(sha_key)
    if expected_sha is None or adjusted_csv is None:
        return False
    path = Path(str(adjusted_csv))
    if not path.exists():
        return False
    return expected_sha == _sha256(path)


def _production_run_identity_valid(manifest: dict[str, Any] | None) -> bool:
    if manifest is None:
        return False
    return (
        bool(str(manifest.get("production_run_id") or "").strip())
        and bool(str(manifest.get("production_entrypoint") or "").strip())
        and re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("git_commit") or "").strip()) is not None
    )


def _production_run_identity_value(manifest: dict[str, Any] | None) -> dict[str, Any]:
    manifest = manifest or {}
    return {
        "production_run_id": manifest.get("production_run_id"),
        "production_entrypoint": manifest.get("production_entrypoint"),
        "git_commit": manifest.get("git_commit"),
    }


def _artifact_bound_to_production_manifest(
    artifact: dict[str, Any] | None,
    *,
    production_manifest_path: Path | None,
    production_manifest_sha256: str | None,
    production_manifest: dict[str, Any] | None,
) -> bool:
    if artifact is None or production_manifest is None or production_manifest_sha256 is None:
        return False
    path_bound = _same_path(artifact.get("adjusted_production_manifest"), production_manifest_path)
    sha_bound = artifact.get("adjusted_production_manifest_sha256") == production_manifest_sha256
    identity_bound = _production_run_identity_matches(artifact, production_manifest)
    holdout_bound = _artifact_locked_holdout_matches_production(artifact, production_manifest)
    return bool((path_bound or sha_bound) and identity_bound and holdout_bound)


def _artifact_bound_to_locked_holdout(
    artifact: dict[str, Any] | None,
    *,
    locked_holdout_path: Path | None,
    locked_holdout_sha256: str | None,
) -> bool:
    if artifact is None or locked_holdout_sha256 is None:
        return False
    path_bound = _same_path(artifact.get("locked_holdout_summary"), locked_holdout_path)
    sha_bound = artifact.get("locked_holdout_summary_sha256") == locked_holdout_sha256
    policy_bound = artifact.get("locked_holdout_policy_pass") is True
    return bool(sha_bound and policy_bound and (path_bound or artifact.get("locked_holdout_summary")))


def _artifact_locked_holdout_matches_production(
    artifact: dict[str, Any],
    production_manifest: dict[str, Any],
) -> bool:
    expected_sha = production_manifest.get("locked_holdout_summary_sha256")
    if not expected_sha or production_manifest.get("locked_holdout_policy_pass") is not True:
        return False
    return bool(
        artifact.get("locked_holdout_policy_pass") is True
        and artifact.get("locked_holdout_summary_sha256") == expected_sha
    )


def _production_run_identity_matches(artifact: dict[str, Any], production_manifest: dict[str, Any]) -> bool:
    for key in ["production_run_id", "production_entrypoint", "git_commit"]:
        expected = production_manifest.get(key)
        value = artifact.get(key)
        if expected and value != expected:
            return False
        if expected and not value:
            return False
    return True


def _capstone_bound_to_production_chain(
    capstone: dict[str, Any],
    *,
    production_manifest_path: Path | None,
    production_manifest_sha256: str | None,
    production_manifest: dict[str, Any] | None,
    export_manifest_path: Path | None,
    export_manifest: dict[str, Any] | None,
    selected_artifact_path: Path | None,
    selected_artifact: dict[str, Any] | None,
) -> bool:
    if not _artifact_bound_to_production_manifest(
        capstone,
        production_manifest_path=production_manifest_path,
        production_manifest_sha256=production_manifest_sha256,
        production_manifest=production_manifest,
    ):
        return False
    export_bound = _same_path(capstone.get("adjusted_export_manifest"), export_manifest_path)
    selected_bound = _same_path(capstone.get("adjusted_selected_artifact"), selected_artifact_path)
    if export_manifest_path is not None and export_manifest_path.exists():
        export_bound = export_bound or capstone.get("adjusted_export_manifest_sha256") == _sha256(export_manifest_path)
    if selected_artifact_path is not None and selected_artifact_path.exists():
        selected_bound = selected_bound or capstone.get("adjusted_selected_artifact_sha256") == _sha256(selected_artifact_path)
    return bool(export_bound and selected_bound and export_manifest is not None and selected_artifact is not None)


def _production_chain_binding_value(artifact: dict[str, Any] | None) -> dict[str, Any]:
    artifact = artifact or {}
    return {
        "adjusted_production_manifest": artifact.get("adjusted_production_manifest"),
        "adjusted_production_manifest_sha256": artifact.get("adjusted_production_manifest_sha256"),
        "adjusted_export_manifest": artifact.get("adjusted_export_manifest"),
        "adjusted_export_manifest_sha256": artifact.get("adjusted_export_manifest_sha256"),
        "adjusted_selected_artifact": artifact.get("adjusted_selected_artifact"),
        "adjusted_selected_artifact_sha256": artifact.get("adjusted_selected_artifact_sha256"),
        "production_run_id": artifact.get("production_run_id"),
        "production_entrypoint": artifact.get("production_entrypoint"),
        "git_commit": artifact.get("git_commit"),
        "locked_holdout_summary": artifact.get("locked_holdout_summary"),
        "locked_holdout_summary_sha256": artifact.get("locked_holdout_summary_sha256"),
        "locked_holdout_policy_pass": artifact.get("locked_holdout_policy_pass"),
    }


def _locked_holdout_binding_value(artifact: dict[str, Any] | None) -> dict[str, Any]:
    artifact = artifact or {}
    return {
        "locked_holdout_summary": artifact.get("locked_holdout_summary"),
        "locked_holdout_summary_sha256": artifact.get("locked_holdout_summary_sha256"),
        "locked_holdout_policy_pass": artifact.get("locked_holdout_policy_pass"),
    }


def _production_manifest_source_provenance_checks(
    production_manifest: dict[str, Any],
    *,
    adjusted_csv: Path,
) -> list[dict[str, Any]]:
    provenance_path_text = production_manifest.get("source_provenance_manifest")
    provenance_sha = production_manifest.get("source_provenance_manifest_sha256")
    checks = [
        _check("source_provenance_manifest_present", bool(provenance_path_text), provenance_path_text),
    ]
    if not provenance_path_text:
        return checks
    provenance_path = Path(str(provenance_path_text))
    sha_bound = provenance_path.exists() and provenance_sha == _sha256(provenance_path)
    checks.append(
        _check(
            "source_provenance_manifest_sha_bound",
            sha_bound,
            {"source_provenance_manifest": str(provenance_path), "source_provenance_manifest_sha256": provenance_sha},
        )
    )
    if not sha_bound:
        return checks
    source_provenance = _load_json(provenance_path)
    checks.extend(_source_provenance_checks(source_provenance, adjusted_csv))
    checks.extend(
        [
            _check(
                "production_manifest_source_kind_matches_provenance",
                production_manifest.get("source_kind") == source_provenance.get("source_kind") == "candidate_csv",
                {
                    "production_manifest": production_manifest.get("source_kind"),
                    "source_provenance": source_provenance.get("source_kind"),
                },
            ),
            _check(
                "production_manifest_source_eligibility_matches_provenance",
                production_manifest.get("source_promotion_eligible") is True
                and source_provenance.get("source_promotion_eligible") is True,
                {
                    "production_manifest": production_manifest.get("source_promotion_eligible"),
                    "source_provenance": source_provenance.get("source_promotion_eligible"),
                },
            ),
        ]
    )
    return checks


def _source_provenance_checks(source_provenance: dict[str, Any], adjusted_csv: Path) -> list[dict[str, Any]]:
    adjusted_path = str(source_provenance.get("adjusted_csv"))
    adjusted_sha = source_provenance.get("adjusted_csv_sha256")
    expected_sha = _sha256(adjusted_csv) if adjusted_csv.exists() else None
    source_path = Path(str(source_provenance.get("source_path", "")))
    staged_candidate = Path(str(source_provenance.get("staged_candidate_csv", "")))
    lab_manifest = Path(str(source_provenance.get("lab_manifest", "")))
    source_export_manifest = Path(str(source_provenance.get("source_export_manifest", "")))
    return [
        _check(
            "source_provenance_schema",
            source_provenance.get("schema_version") == "epex_lab_adjusted_lt_candidate_stage.v1",
            source_provenance.get("schema_version"),
        ),
        _check(
            "source_provenance_schema_role",
            source_provenance.get("schema_role") == "source_provenance",
            source_provenance.get("schema_role"),
        ),
        _check(
            "source_provenance_lab_only",
            source_provenance.get("activation_status") == "staged_lab_only"
            and source_provenance.get("production_approved") is False
            and source_provenance.get("production_promotion_approved") is False,
            {
                "activation_status": source_provenance.get("activation_status"),
                "production_approved": source_provenance.get("production_approved"),
                "production_promotion_approved": source_provenance.get("production_promotion_approved"),
            },
        ),
        _check(
            "source_provenance_candidate_csv",
            source_provenance.get("source_kind") == "candidate_csv",
            source_provenance.get("source_kind"),
        ),
        _check(
            "source_provenance_promotion_eligible",
            source_provenance.get("source_promotion_eligible") is True,
            source_provenance.get("source_promotion_eligible"),
        ),
        _check(
            "source_provenance_no_contract_blockers",
            list(source_provenance.get("production_contract_blockers") or []) == [],
            source_provenance.get("production_contract_blockers"),
        ),
        _check(
            "source_provenance_adjusted_csv_bound",
            expected_sha is not None and (_same_path(adjusted_path, adjusted_csv) or adjusted_sha == expected_sha),
            {"adjusted_csv": adjusted_path, "adjusted_csv_sha256": adjusted_sha},
        ),
        _check(
            "source_provenance_source_sha_bound",
            source_path.exists() and source_provenance.get("source_sha256") == _sha256(source_path),
            {"source_path": str(source_path), "source_sha256": source_provenance.get("source_sha256")},
        ),
        _check(
            "source_provenance_staged_candidate_sha_bound",
            staged_candidate.exists()
            and source_provenance.get("staged_candidate_csv_sha256") == _sha256(staged_candidate),
            {
                "staged_candidate_csv": str(staged_candidate),
                "staged_candidate_csv_sha256": source_provenance.get("staged_candidate_csv_sha256"),
            },
        ),
        _check(
            "source_provenance_lab_manifest_sha_bound",
            lab_manifest.exists() and source_provenance.get("lab_manifest_sha256") == _sha256(lab_manifest),
            {"lab_manifest": str(lab_manifest), "lab_manifest_sha256": source_provenance.get("lab_manifest_sha256")},
        ),
        _check(
            "source_provenance_source_export_manifest_sha_bound",
            source_export_manifest.exists()
            and source_provenance.get("source_export_manifest_sha256") == _sha256(source_export_manifest),
            {
                "source_export_manifest": str(source_export_manifest),
                "source_export_manifest_sha256": source_provenance.get("source_export_manifest_sha256"),
            },
        ),
        _check(
            "source_provenance_source_export_manifest_bound",
            source_export_manifest.exists()
            and _source_export_manifest_bound_to_csv(source_export_manifest, source_path),
            {"source_export_manifest": str(source_export_manifest), "source_path": str(source_path)},
        ),
        _check(
            "source_provenance_no_ompex",
            source_provenance.get("ompex_used_in_model") is False
            and source_provenance.get("ompex_used_in_selection") is False,
            {
                "ompex_used_in_model": source_provenance.get("ompex_used_in_model"),
                "ompex_used_in_selection": source_provenance.get("ompex_used_in_selection"),
            },
        ),
    ]


def _source_export_manifest_bound_to_csv(manifest_path: Path, source_csv: Path) -> bool:
    try:
        manifest = _load_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return False
    if not manifest.get("schema_version"):
        return False
    candidate_path_keys = [
        "output_csv",
        "candidate_csv",
        "source_csv",
        "adjusted_csv",
        "selected_adjusted_csv",
    ]
    for key in candidate_path_keys:
        if _same_path(manifest.get(key), source_csv):
            return True
    candidate_sha_keys = [
        "output_csv_sha256",
        "candidate_csv_sha256",
        "source_csv_sha256",
        "adjusted_csv_sha256",
        "selected_adjusted_csv_sha256",
    ]
    expected_sha = _sha256(source_csv) if source_csv.exists() else None
    return expected_sha is not None and any(manifest.get(key) == expected_sha for key in candidate_sha_keys)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-manifest", type=Path, required=True)
    parser.add_argument("--governance-audit", type=Path, required=True)
    parser.add_argument("--independent-summary", type=Path, required=True)
    parser.add_argument("--product-summary", type=Path, required=True)
    parser.add_argument("--powerbi-summary", type=Path, required=True)
    parser.add_argument("--ompex-advisory-delta", type=Path, default=None)
    parser.add_argument("--adjusted-production-manifest", type=Path, default=None)
    parser.add_argument("--adjusted-export-manifest", type=Path, default=None)
    parser.add_argument("--adjusted-selected-config", type=Path, default=None)
    parser.add_argument("--adjusted-capstone", type=Path, default=None)
    parser.add_argument("--locked-holdout-summary", type=Path, default=None)
    parser.add_argument("--locked-holdout-queue-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    summary = check_readiness(
        lab_manifest=args.lab_manifest,
        governance_audit=args.governance_audit,
        independent_summary=args.independent_summary,
        product_summary=args.product_summary,
        powerbi_summary=args.powerbi_summary,
        ompex_advisory_delta=args.ompex_advisory_delta,
        adjusted_production_manifest=args.adjusted_production_manifest,
        adjusted_export_manifest=args.adjusted_export_manifest,
        adjusted_selected_config=args.adjusted_selected_config,
        adjusted_capstone=args.adjusted_capstone,
        locked_holdout_summary=args.locked_holdout_summary,
        locked_holdout_queue_summary=args.locked_holdout_queue_summary,
        output=args.output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0 if summary["approved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

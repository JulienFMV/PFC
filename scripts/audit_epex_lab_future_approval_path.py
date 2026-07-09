"""Audit the future production approval path for an EPEX shape-lab candidate.

This read-only helper summarizes what is already proven by readiness evidence
and what still blocks production promotion.  It does not promote artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.epex_lab_locked_holdout_policy import locked_holdout_policy as evaluate_locked_holdout_policy
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from epex_lab_locked_holdout_policy import locked_holdout_policy as evaluate_locked_holdout_policy


PRODUCTION_CHECKS = [
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
]

REQUIRED_PRODUCTION_EVIDENCE = [
    "adjusted_production_manifest",
    "adjusted_export_manifest",
    "adjusted_selected_config",
    "adjusted_capstone",
]


def audit_future_approval_path(
    *,
    readiness_json: Path,
    output: Path,
    spot_backtest_summary: Path | None = None,
    locked_holdout_summary: Path | None = None,
) -> dict[str, Any]:
    readiness = _read_json(readiness_json)
    spot = _read_json(spot_backtest_summary) if spot_backtest_summary is not None else None
    locked_holdout = _read_json(locked_holdout_summary) if locked_holdout_summary is not None else None
    checks = {str(check.get("name")): check for check in readiness.get("checks", [])}
    required_checks = _required_production_checks(readiness)
    failed_checks = [name for name, check in checks.items() if check.get("status") != "PASS"]
    missing_production_checks = [name for name in required_checks if name not in checks]
    failed_production_checks = [name for name in required_checks if name in failed_checks]
    missing = list(readiness.get("missing_production_evidence") or [])
    missing_or_failed = sorted(set(missing + missing_production_checks + failed_production_checks))
    spot_policy = _spot_policy(spot) if spot is not None else None
    holdout_policy = evaluate_locked_holdout_policy(locked_holdout) if locked_holdout is not None else None
    holdout_required = bool(readiness.get("approved") is True or readiness.get("production_chain_pass") is True)
    if holdout_required and holdout_policy is None:
        holdout_policy = {"provided": False, "pass": False, "status": "MISSING_LOCKED_HOLDOUT"}
        missing_or_failed = sorted(set(missing_or_failed + ["locked_holdout_pass"]))
    if holdout_required and holdout_policy is not None and holdout_policy.get("pass") is True:
        expected_sha = _readiness_locked_holdout_sha(checks)
        actual_sha = _sha256(locked_holdout_summary) if locked_holdout_summary is not None else None
        if not expected_sha or actual_sha != expected_sha:
            holdout_policy = {
                **holdout_policy,
                "pass": False,
                "status": "NO_GO_LOCKED_HOLDOUT_HASH_MISMATCH",
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
            }
            missing_or_failed = sorted(set(missing_or_failed + ["locked_holdout_sha_bound"]))
    if holdout_policy is not None and holdout_policy["pass"] is not True:
        missing_or_failed = sorted(set(missing_or_failed + ["locked_holdout_pass"]))

    if spot_policy is not None and not spot_policy["pass"]:
        status = "NO_GO_SPOT_BACKTEST_POLICY_FAIL"
    elif holdout_policy is not None and not holdout_policy["pass"]:
        status = str(holdout_policy["status"])
    elif missing_production_checks:
        status = "NO_GO_PRODUCTION_CHAIN_INCOMPLETE"
    elif readiness.get("approved") is True and readiness.get("status") == "PROMOTION_READY":
        status = "PROMOTION_READY_CANDIDATE"
    elif readiness.get("strict_diagnostics_pass") is not True:
        status = "NO_GO_STRICT_DIAGNOSTICS_FAIL"
    elif readiness.get("production_chain_pass") is not True:
        status = "NO_GO_PRODUCTION_CHAIN_INCOMPLETE"
    else:
        status = "NO_GO_UNCLASSIFIED"
    blocking_stage, next_required_step = _blocking_stage(
        status=status,
        missing=missing,
        missing_production_checks=missing_production_checks,
        failed_production_checks=failed_production_checks,
        spot_policy=spot_policy,
        holdout_policy=holdout_policy,
    )
    recommended_commands = _recommended_commands(
        blocking_stage=blocking_stage,
        holdout_policy=holdout_policy,
    )

    summary = {
        "schema_version": "epex_lab_future_approval_path_audit.v1",
        "read_only": True,
        "promotion_gate": False,
        "approved": bool(status == "PROMOTION_READY_CANDIDATE"),
        "status": status,
        "blocking_stage": blocking_stage,
        "next_required_step": next_required_step,
        "recommended_commands": recommended_commands,
        "readiness_json": str(readiness_json),
        "readiness_status": readiness.get("status"),
        "readiness_approved": bool(readiness.get("approved")),
        "strict_diagnostics_pass": bool(readiness.get("strict_diagnostics_pass")),
        "production_chain_pass": bool(readiness.get("production_chain_pass")),
        "selected_adjusted_csv": readiness.get("selected_adjusted_csv"),
        "missing_production_evidence": missing,
        "failed_production_checks": failed_production_checks,
        "missing_production_checks": missing_production_checks,
        "failed_check_count": int(len(failed_checks)),
        "failed_checks": failed_checks,
        "remaining_blockers": missing_or_failed,
        "required_production_evidence": REQUIRED_PRODUCTION_EVIDENCE,
        "required_production_checks": required_checks,
        "spot_backtest_policy": spot_policy,
        "locked_holdout_summary": str(locked_holdout_summary) if locked_holdout_summary is not None else None,
        "locked_holdout_policy": holdout_policy,
        "next_actions": _next_actions(
            status=status,
            missing=missing,
            missing_production_checks=missing_production_checks,
            failed_production_checks=failed_production_checks,
            spot_policy=spot_policy,
            holdout_policy=holdout_policy,
        ),
        "note": (
            "This audit summarizes readiness evidence for review. It does not "
            "turn local bundles, spot diagnostics, or source provenance into "
            "production approval."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _spot_policy(spot: dict[str, Any] | None) -> dict[str, Any]:
    if spot is None:
        return {"provided": False, "pass": True}
    checks = {
        "promotion_gate_false": spot.get("promotion_gate") is False,
        "production_approved_false": spot.get("production_approved") is False,
        "ompex_not_model": spot.get("ompex_used_in_model") is False,
        "ompex_not_selection": spot.get("ompex_used_in_selection") is False,
        "ompex_not_backtest": spot.get("ompex_used_in_backtest") is False,
        "diagnostic_pass": spot.get("status") == "DIAGNOSTIC_PASS",
    }
    return {
        "provided": True,
        "summary": spot.get("status"),
        "benchmark_policy": spot.get("benchmark_policy"),
        "pass": all(checks.values()),
        "checks": checks,
    }


def _readiness_locked_holdout_sha(checks: dict[str, dict[str, Any]]) -> str | None:
    candidate_checks = [
        "adjusted_production_manifest_locked_holdout_bound",
        "adjusted_export_manifest_production_chain_bound",
        "adjusted_selected_artifact_production_chain_bound",
        "adjusted_capstone_production_chain_bound",
    ]
    for name in candidate_checks:
        value = (checks.get(name) or {}).get("value")
        if isinstance(value, dict):
            candidate = value.get("locked_holdout_summary_sha256")
            if isinstance(candidate, str) and candidate:
                return candidate
    return None


def _required_production_checks(readiness: dict[str, Any]) -> list[str]:
    declared = readiness.get("required_production_checks")
    if not isinstance(declared, list):
        return list(PRODUCTION_CHECKS)
    combined = list(PRODUCTION_CHECKS)
    for item in declared:
        if isinstance(item, str) and item not in combined:
            combined.append(item)
    return combined


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _next_actions(
    *,
    status: str,
    missing: list[str],
    missing_production_checks: list[str],
    failed_production_checks: list[str],
    spot_policy: dict[str, Any] | None,
    holdout_policy: dict[str, Any] | None,
) -> list[str]:
    actions: list[str] = []
    if spot_policy is not None and not spot_policy["pass"]:
        actions.append("Fix spot backtest policy flags; spot diagnostics must remain lab-only and no-OMPEX.")
    if holdout_policy is not None and not holdout_policy["pass"]:
        if status == "NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING":
            actions.append("Wait for full locked holdout spot coverage, then rerun the locked holdout runner.")
        else:
            actions.append("Resolve locked holdout failure without retuning against the locked window.")
    if missing:
        actions.append("Generate real adjusted production/export/selected/capstone evidence for missing items.")
    if missing_production_checks:
        actions.append("Regenerate readiness with the full required production-check set before promotion review.")
    if failed_production_checks:
        actions.append("Replace local diagnostic approval flags with real production-approved adjusted artifacts.")
    if status == "NO_GO_STRICT_DIAGNOSTICS_FAIL":
        actions.append("Resolve strict diagnostic failures before any production-path work.")
    if not actions and status == "PROMOTION_READY_CANDIDATE":
        actions.append("Run independent capstone review against the exact adjusted CSV and source policy hashes.")
    if not actions:
        actions.append("Investigate unclassified readiness state before promotion.")
    return actions


def _blocking_stage(
    *,
    status: str,
    missing: list[str],
    missing_production_checks: list[str],
    failed_production_checks: list[str],
    spot_policy: dict[str, Any] | None,
    holdout_policy: dict[str, Any] | None,
) -> tuple[str, str]:
    if spot_policy is not None and not spot_policy["pass"]:
        return "spot_policy", "fix_spot_backtest_policy_flags"
    if holdout_policy is not None and not holdout_policy["pass"]:
        if status == "NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING":
            return "locked_holdout_coverage", "wait_for_full_spot_coverage_then_run_locked_holdout"
        return "locked_holdout_failure", "resolve_locked_holdout_failure_without_retuning_locked_window"
    if status == "NO_GO_STRICT_DIAGNOSTICS_FAIL":
        return "strict_diagnostics", "resolve_strict_diagnostic_failures"
    if missing:
        return "production_evidence", "generate_adjusted_production_export_selected_capstone_evidence"
    if missing_production_checks:
        return "readiness_contract", "regenerate_readiness_with_required_production_checks"
    if failed_production_checks:
        return "production_checks", "replace_local_diagnostic_flags_with_real_production_artifacts"
    if status == "PROMOTION_READY_CANDIDATE":
        return "promotion_ready_candidate", "run_independent_capstone_review"
    return "unclassified", "investigate_readiness_state"


def _recommended_commands(
    *,
    blocking_stage: str,
    holdout_policy: dict[str, Any] | None,
) -> dict[str, str]:
    if blocking_stage != "locked_holdout_coverage":
        return {}
    plan_json = (
        str(holdout_policy.get("plan_json"))
        if holdout_policy is not None and holdout_policy.get("plan_json")
        else "<T057_PLAN_JSON>"
    )
    plan_sha = (
        str(holdout_policy.get("expected_plan_json_sha256") or holdout_policy.get("plan_json_sha256"))
        if holdout_policy is not None
        and (holdout_policy.get("expected_plan_json_sha256") or holdout_policy.get("plan_json_sha256"))
        else "<T057_PLAN_JSON_SHA256>"
    )
    return {
        "run_locked_holdout": (
            "python scripts/run_epex_lab_locked_holdout.py "
            f"--plan-json {plan_json} "
            f"--expected-plan-sha256 {plan_sha} "
            "--spot-parquet <FRESH_FUTURE_SPOT_PARQUET> "
            "--output-dir <T057_HOLDOUT_OUTPUT_DIR>"
        )
    }


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-json", type=Path, required=True)
    parser.add_argument("--spot-backtest-summary", type=Path, default=None)
    parser.add_argument("--locked-holdout-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = audit_future_approval_path(
        readiness_json=args.readiness_json,
        spot_backtest_summary=args.spot_backtest_summary,
        locked_holdout_summary=args.locked_holdout_summary,
        output=args.output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["approved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

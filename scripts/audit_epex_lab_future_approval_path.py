"""Audit the future production approval path for an EPEX shape-lab candidate.

This read-only helper summarizes what is already proven by readiness evidence
and what still blocks production promotion.  It does not promote artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PRODUCTION_CHECKS = [
    "adjusted_production_manifest_approved",
    "adjusted_production_manifest_run_identity_valid",
    "adjusted_export_manifest_production_ready",
    "adjusted_export_manifest_production_chain_bound",
    "adjusted_selected_artifact_production_ready",
    "adjusted_selected_artifact_production_chain_bound",
    "adjusted_capstone_approved",
    "adjusted_capstone_production_chain_bound",
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
) -> dict[str, Any]:
    readiness = _read_json(readiness_json)
    spot = _read_json(spot_backtest_summary) if spot_backtest_summary is not None else None
    checks = {str(check.get("name")): check for check in readiness.get("checks", [])}
    failed_checks = [name for name, check in checks.items() if check.get("status") != "PASS"]
    failed_production_checks = [name for name in PRODUCTION_CHECKS if name in failed_checks]
    missing = list(readiness.get("missing_production_evidence") or [])
    missing_or_failed = sorted(set(missing + failed_production_checks))
    spot_policy = _spot_policy(spot) if spot is not None else None

    if readiness.get("approved") is True and readiness.get("status") == "PROMOTION_READY":
        status = "PROMOTION_READY_CANDIDATE"
    elif spot_policy is not None and not spot_policy["pass"]:
        status = "NO_GO_SPOT_BACKTEST_POLICY_FAIL"
    elif readiness.get("strict_diagnostics_pass") is not True:
        status = "NO_GO_STRICT_DIAGNOSTICS_FAIL"
    elif readiness.get("production_chain_pass") is not True:
        status = "NO_GO_PRODUCTION_CHAIN_INCOMPLETE"
    else:
        status = "NO_GO_UNCLASSIFIED"

    summary = {
        "schema_version": "epex_lab_future_approval_path_audit.v1",
        "read_only": True,
        "promotion_gate": False,
        "approved": bool(status == "PROMOTION_READY_CANDIDATE"),
        "status": status,
        "readiness_json": str(readiness_json),
        "readiness_status": readiness.get("status"),
        "readiness_approved": bool(readiness.get("approved")),
        "strict_diagnostics_pass": bool(readiness.get("strict_diagnostics_pass")),
        "production_chain_pass": bool(readiness.get("production_chain_pass")),
        "selected_adjusted_csv": readiness.get("selected_adjusted_csv"),
        "missing_production_evidence": missing,
        "failed_production_checks": failed_production_checks,
        "failed_check_count": int(len(failed_checks)),
        "failed_checks": failed_checks,
        "remaining_blockers": missing_or_failed,
        "required_production_evidence": REQUIRED_PRODUCTION_EVIDENCE,
        "spot_backtest_policy": spot_policy,
        "next_actions": _next_actions(
            status=status,
            missing=missing,
            failed_production_checks=failed_production_checks,
            spot_policy=spot_policy,
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


def _next_actions(
    *,
    status: str,
    missing: list[str],
    failed_production_checks: list[str],
    spot_policy: dict[str, Any] | None,
) -> list[str]:
    actions: list[str] = []
    if spot_policy is not None and not spot_policy["pass"]:
        actions.append("Fix spot backtest policy flags; spot diagnostics must remain lab-only and no-OMPEX.")
    if missing:
        actions.append("Generate real adjusted production/export/selected/capstone evidence for missing items.")
    if failed_production_checks:
        actions.append("Replace local diagnostic approval flags with real production-approved adjusted artifacts.")
    if status == "NO_GO_STRICT_DIAGNOSTICS_FAIL":
        actions.append("Resolve strict diagnostic failures before any production-path work.")
    if not actions and status == "PROMOTION_READY_CANDIDATE":
        actions.append("Run independent capstone review against the exact adjusted CSV and source policy hashes.")
    if not actions:
        actions.append("Investigate unclassified readiness state before promotion.")
    return actions


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-json", type=Path, required=True)
    parser.add_argument("--spot-backtest-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = audit_future_approval_path(
        readiness_json=args.readiness_json,
        spot_backtest_summary=args.spot_backtest_summary,
        output=args.output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

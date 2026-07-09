"""Shared policy checks for locked EPEX lab holdout evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PLAN_SCHEMA_VERSION = "epex_lab_locked_holdout_plan.v1"
RUN_SCHEMA_VERSION = "epex_lab_locked_holdout_run.v1"
AUDIT_SCHEMA_VERSION = "epex_lab_locked_holdout_audit.v1"


def build_locked_plan_identity(plan: dict[str, Any], *, plan_json: Path | None = None) -> dict[str, Any]:
    criteria = plan.get("pass_criteria") or {}
    identity = {
        "plan_id": plan.get("plan_id"),
        "plan_schema_version": plan.get("schema_version"),
        "benchmark_policy": plan.get("benchmark_policy"),
        "frozen_at_utc": plan.get("frozen_at_utc"),
        "holdout_start_utc": plan.get("holdout_start_utc"),
        "holdout_end_utc": plan.get("holdout_end_utc"),
        "baseline_csv": plan.get("baseline_csv"),
        "baseline_csv_sha256": plan.get("baseline_csv_sha256") or criteria.get("baseline_csv_sha256"),
        "adjusted_csv": plan.get("adjusted_csv"),
        "adjusted_csv_sha256": plan.get("adjusted_csv_sha256") or criteria.get("adjusted_csv_sha256"),
        "lab_manifest": plan.get("lab_manifest"),
        "lab_manifest_sha256": plan.get("lab_manifest_sha256"),
        "selection_summary": plan.get("selection_summary"),
        "selection_summary_sha256": plan.get("selection_summary_sha256"),
    }
    if plan_json is not None:
        identity["plan_json"] = str(plan_json)
        identity["plan_json_sha256"] = _sha256(plan_json)
    return identity


def locked_holdout_policy(summary: dict[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {"provided": False, "pass": False, "status": "MISSING_LOCKED_HOLDOUT"}
    schema = summary.get("schema_version")
    checks = {
        "promotion_gate_false": summary.get("promotion_gate") is False,
        "production_approved_false": summary.get("production_approved") is False,
        "ompex_not_model": summary.get("ompex_used_in_model") is False,
        "ompex_not_selection": summary.get("ompex_used_in_selection") is False,
        "ompex_not_backtest": summary.get("ompex_used_in_backtest") is False,
    }
    if schema == RUN_SCHEMA_VERSION:
        checks.update(
            {
                "coverage_ready": summary.get("coverage_ready") is True,
                "backtest_ran": summary.get("backtest_ran") is True,
                "audit_ran": summary.get("audit_ran") is True,
                "holdout_pass": summary.get("holdout_pass") is True,
                "status_pass": summary.get("status") == "LOCKED_HOLDOUT_PASS",
            }
        )
        status = (
            "NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING"
            if summary.get("status") == "WAITING_FOR_FULL_SPOT_COVERAGE"
            else "NO_GO_LOCKED_HOLDOUT_FAIL"
        )
    elif schema == AUDIT_SCHEMA_VERSION:
        checks.update(
            {
                "holdout_pass": summary.get("holdout_pass") is True,
                "status_pass": summary.get("status") == "LOCKED_HOLDOUT_PASS",
            }
        )
        status = "NO_GO_LOCKED_HOLDOUT_FAIL"
    else:
        checks["known_schema"] = False
        status = "NO_GO_LOCKED_HOLDOUT_POLICY_INVALID"
    checks.update(_identity_checks(summary))
    passed = all(checks.values())
    return {
        "provided": True,
        "schema_version": schema,
        "summary": summary.get("status"),
        "pass": passed,
        "status": "LOCKED_HOLDOUT_PASS" if passed else status,
        "plan_json": _identity(summary).get("plan_json"),
        "plan_json_sha256": _identity(summary).get("plan_json_sha256"),
        "plan_id": _identity(summary).get("plan_id"),
        "holdout_start_utc": _identity(summary).get("holdout_start_utc"),
        "holdout_end_utc": _identity(summary).get("holdout_end_utc"),
        "baseline_csv_sha256": _identity(summary).get("baseline_csv_sha256"),
        "adjusted_csv_sha256": _identity(summary).get("adjusted_csv_sha256"),
        "spot_parquet": summary.get("spot_parquet"),
        "output_dir": summary.get("output_dir"),
        "checks": checks,
    }


def _identity_checks(summary: dict[str, Any]) -> dict[str, bool]:
    identity = _identity(summary)
    checks = {
        "locked_plan_identity_present": bool(identity),
        "plan_id_present": bool(str(identity.get("plan_id") or "").strip()),
        "plan_json_present": bool(str(identity.get("plan_json") or "").strip()),
        "plan_json_sha256_present": bool(str(identity.get("plan_json_sha256") or "").strip()),
        "holdout_start_utc_present": bool(str(identity.get("holdout_start_utc") or "").strip()),
        "holdout_end_utc_present": bool(str(identity.get("holdout_end_utc") or "").strip()),
        "baseline_csv_sha256_present": bool(str(identity.get("baseline_csv_sha256") or "").strip()),
        "adjusted_csv_sha256_present": bool(str(identity.get("adjusted_csv_sha256") or "").strip()),
    }
    plan_json_text = identity.get("plan_json")
    if not plan_json_text:
        checks["plan_json_file_sha_bound"] = False
        checks["plan_identity_matches_plan_json"] = False
        return checks
    plan_json = Path(str(plan_json_text))
    checks["plan_json_file_exists"] = plan_json.exists()
    checks["plan_json_file_sha_bound"] = (
        plan_json.exists() and identity.get("plan_json_sha256") == _sha256(plan_json)
    )
    if not checks["plan_json_file_sha_bound"]:
        checks["plan_identity_matches_plan_json"] = False
        return checks
    try:
        plan = json.loads(plan_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        checks["plan_identity_matches_plan_json"] = False
        return checks
    expected = build_locked_plan_identity(plan, plan_json=plan_json)
    keys = [
        "plan_id",
        "plan_schema_version",
        "benchmark_policy",
        "frozen_at_utc",
        "holdout_start_utc",
        "holdout_end_utc",
        "baseline_csv_sha256",
        "adjusted_csv_sha256",
        "lab_manifest_sha256",
        "selection_summary_sha256",
        "plan_json_sha256",
    ]
    checks["plan_identity_matches_plan_json"] = all(identity.get(key) == expected.get(key) for key in keys)
    return checks


def _identity(summary: dict[str, Any]) -> dict[str, Any]:
    value = summary.get("locked_plan_identity")
    if isinstance(value, dict):
        return value
    return {
        key: summary.get(key)
        for key in [
            "plan_id",
            "plan_schema_version",
            "benchmark_policy",
            "frozen_at_utc",
            "holdout_start_utc",
            "holdout_end_utc",
            "baseline_csv_sha256",
            "adjusted_csv_sha256",
            "lab_manifest_sha256",
            "selection_summary_sha256",
            "plan_json",
            "plan_json_sha256",
        ]
        if summary.get(key) is not None
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

"""Shared policy checks for locked EPEX lab holdout evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PLAN_SCHEMA_VERSION = "epex_lab_locked_holdout_plan.v1"
RUN_SCHEMA_VERSION = "epex_lab_locked_holdout_run.v1"
AUDIT_SCHEMA_VERSION = "epex_lab_locked_holdout_audit.v1"
COVERAGE_SCHEMA_VERSION = "epex_lab_locked_holdout_coverage.v1"
LOCKED_HOLDOUT_POLICY = "locked_future_no_ompex_holdout"
SPOT_BACKTEST_SCHEMA_VERSION = "epex_shape_lab_spot_backtest.v1"
SPOT_BACKTEST_POLICY = "rolling_origin_epex_spot_no_ompex_lab_only"


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
        resolved_plan_json = _resolved_path(plan_json)
        identity["plan_json"] = str(resolved_plan_json)
        identity["plan_json_sha256"] = _sha256(resolved_plan_json)
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
        coverage = summary.get("coverage") if isinstance(summary.get("coverage"), dict) else {}
        coverage_checks = coverage.get("checks") if isinstance(coverage.get("checks"), dict) else {}
        checks.update(
            {
                "benchmark_policy_locked": summary.get("benchmark_policy") == LOCKED_HOLDOUT_POLICY,
                "expected_plan_json_sha256_present": bool(
                    str(summary.get("expected_plan_json_sha256") or "").strip()
                ),
                "actual_plan_json_sha256_present": bool(str(summary.get("actual_plan_json_sha256") or "").strip()),
                "expected_plan_json_sha256_bound": _expected_plan_sha_bound(summary),
                "coverage_ready": summary.get("coverage_ready") is True,
                "coverage_schema": coverage.get("schema_version") == COVERAGE_SCHEMA_VERSION,
                "coverage_read_only": coverage.get("read_only") is True,
                "coverage_promotion_gate_false": coverage.get("promotion_gate") is False,
                "coverage_production_approved_false": coverage.get("production_approved") is False,
                "coverage_identity_matches_run": _same_identity(_identity(coverage), _identity(summary)),
                "coverage_baseline_csv_sha256_matches_identity": coverage.get("baseline_csv_sha256")
                == _identity(summary).get("baseline_csv_sha256"),
                "coverage_adjusted_csv_sha256_matches_identity": coverage.get("adjusted_csv_sha256")
                == _identity(summary).get("adjusted_csv_sha256"),
                "coverage_candidate_timestamp_set_sha256_present": _candidate_timestamp_set_sha256_present(coverage),
                "coverage_candidate_timestamp_set_sha256_equal": _candidate_timestamp_set_sha256_equal(coverage),
                "coverage_candidate_timestamp_counts_valid": _candidate_timestamp_counts_valid(coverage),
                "coverage_candidate_timestamp_bounds_equal": _candidate_timestamp_bounds_equal(coverage),
                "coverage_status_ready": coverage.get("status") == "READY_TO_RUN_HOLDOUT_BACKTEST",
                "coverage_embedded_ready": coverage.get("ready_to_run_backtest") is True,
                "coverage_full_window_covered": coverage_checks.get("full_window_covered") is True,
                "coverage_min_holdout_hours_met": coverage_checks.get("min_holdout_hours_met") is True,
                "coverage_no_duplicate_holdout_rows": coverage_checks.get("no_duplicate_holdout_rows") is True,
                "coverage_spot_price_column_present": coverage_checks.get("spot_price_column_present") is True,
                "coverage_holdout_prices_finite": coverage_checks.get("holdout_prices_finite") is True,
                "coverage_baseline_csv_sha256_bound": coverage_checks.get("baseline_csv_sha256_bound") is True,
                "coverage_adjusted_csv_sha256_bound": coverage_checks.get("adjusted_csv_sha256_bound") is True,
                "coverage_baseline_candidate_schema_ready": _candidate_coverage_checks_pass(
                    coverage_checks,
                    prefix="baseline_candidate",
                ),
                "coverage_adjusted_candidate_schema_ready": _candidate_coverage_checks_pass(
                    coverage_checks,
                    prefix="adjusted_candidate",
                ),
                "coverage_candidate_timestamp_sets_identical": coverage_checks.get("candidate_timestamp_sets_identical")
                is True,
                "coverage_candidate_timestamp_set_matches_plan": coverage_checks.get("candidate_timestamp_set_matches_plan")
                is True,
                "coverage_candidate_timestamp_count_matches_plan": coverage_checks.get(
                    "candidate_timestamp_count_matches_plan"
                )
                is True,
                "coverage_status_sha256_bound": _file_sha_bound(
                    summary,
                    path_key="coverage_status",
                    sha_key="coverage_status_sha256",
                ),
                "coverage_status_matches_embedded": _coverage_status_matches_embedded(summary),
                "backtest_ran": summary.get("backtest_ran") is True,
                "audit_ran": summary.get("audit_ran") is True,
                "holdout_pass": summary.get("holdout_pass") is True,
                "status_pass": summary.get("status") == "LOCKED_HOLDOUT_PASS",
                "spot_backtest_summary_sha256_bound": _file_sha_bound(
                    summary,
                    path_key="spot_backtest_summary",
                    sha_key="spot_backtest_summary_sha256",
                ),
                "locked_holdout_audit_sha256_bound": _file_sha_bound(
                    summary,
                    path_key="locked_holdout_audit",
                    sha_key="locked_holdout_audit_sha256",
                ),
            }
        )
        checks.update(_linked_backtest_checks(summary))
        checks.update(_linked_audit_checks(summary))
        summary_status = summary.get("status")
        status = (
            "NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING"
            if summary_status == "WAITING_FOR_FULL_SPOT_COVERAGE"
            else str(summary_status)
            if summary_status
            in {
                "NO_GO_LOCKED_HOLDOUT_PLAN_HASH_MISMATCH",
                "NO_GO_LOCKED_HOLDOUT_SOURCE_MISSING_OR_HASH_MISMATCH",
            }
            else "NO_GO_LOCKED_HOLDOUT_FAIL"
        )
    elif schema == AUDIT_SCHEMA_VERSION:
        checks["run_summary_schema_required"] = False
        status = "NO_GO_LOCKED_HOLDOUT_RUN_SUMMARY_REQUIRED"
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
        "expected_plan_json_sha256": summary.get("expected_plan_json_sha256"),
        "actual_plan_json_sha256": summary.get("actual_plan_json_sha256"),
        "plan_id": _identity(summary).get("plan_id"),
        "holdout_start_utc": _identity(summary).get("holdout_start_utc"),
        "holdout_end_utc": _identity(summary).get("holdout_end_utc"),
        "baseline_csv_sha256": _identity(summary).get("baseline_csv_sha256"),
        "adjusted_csv_sha256": _identity(summary).get("adjusted_csv_sha256"),
        "spot_parquet": summary.get("spot_parquet"),
        "output_dir": summary.get("output_dir"),
        "checks": checks,
    }


def _coverage_status_matches_embedded(summary: dict[str, Any]) -> bool:
    coverage = summary.get("coverage")
    if not isinstance(coverage, dict):
        return False
    linked = _read_bound_json(summary, path_key="coverage_status", sha_key="coverage_status_sha256")
    return linked == coverage


def _expected_plan_sha_bound(summary: dict[str, Any]) -> bool:
    expected = summary.get("expected_plan_json_sha256")
    actual = summary.get("actual_plan_json_sha256")
    identity_sha = _identity(summary).get("plan_json_sha256")
    return bool(expected and actual and identity_sha and expected == actual == identity_sha)


def _candidate_coverage_checks_pass(checks: dict[str, Any], *, prefix: str) -> bool:
    return all(
        checks.get(name) is True
        for name in [
            f"{prefix}_required_columns_present",
            f"{prefix}_utc_offset_present",
            f"{prefix}_timestamps_parseable",
            f"{prefix}_no_duplicate_timestamps",
            f"{prefix}_price_columns_finite",
            f"{prefix}_holdout_window_covered",
        ]
    )


def _candidate_timestamp_set_sha256_present(coverage: dict[str, Any]) -> bool:
    return bool(
        str(coverage.get("baseline_candidate_timestamp_set_sha256") or "").strip()
        and str(coverage.get("adjusted_candidate_timestamp_set_sha256") or "").strip()
    )


def _candidate_timestamp_set_sha256_equal(coverage: dict[str, Any]) -> bool:
    baseline = coverage.get("baseline_candidate_timestamp_set_sha256")
    adjusted = coverage.get("adjusted_candidate_timestamp_set_sha256")
    return bool(baseline and adjusted and baseline == adjusted)


def _candidate_timestamp_counts_valid(coverage: dict[str, Any]) -> bool:
    baseline = coverage.get("baseline_candidate_timestamp_count")
    adjusted = coverage.get("adjusted_candidate_timestamp_count")
    return isinstance(baseline, int) and isinstance(adjusted, int) and baseline > 0 and baseline == adjusted


def _candidate_timestamp_bounds_equal(coverage: dict[str, Any]) -> bool:
    keys = [
        "baseline_candidate_timestamp_min_utc",
        "baseline_candidate_timestamp_max_utc",
        "adjusted_candidate_timestamp_min_utc",
        "adjusted_candidate_timestamp_max_utc",
    ]
    if not all(str(coverage.get(key) or "").strip() for key in keys):
        return False
    return (
        coverage.get("baseline_candidate_timestamp_min_utc") == coverage.get("adjusted_candidate_timestamp_min_utc")
        and coverage.get("baseline_candidate_timestamp_max_utc") == coverage.get("adjusted_candidate_timestamp_max_utc")
    )


def _linked_backtest_checks(summary: dict[str, Any]) -> dict[str, bool]:
    linked = _read_bound_json(
        summary,
        path_key="spot_backtest_summary",
        sha_key="spot_backtest_summary_sha256",
    )
    if linked is None:
        return {
            "linked_backtest_json_readable": False,
            "linked_backtest_schema": False,
            "linked_backtest_status_pass": False,
            "linked_backtest_strict_lab_gate_pass": False,
            "linked_backtest_lab_only": False,
            "linked_backtest_no_ompex": False,
            "linked_backtest_policy": False,
        }
    return {
        "linked_backtest_json_readable": True,
        "linked_backtest_schema": linked.get("schema_version") == SPOT_BACKTEST_SCHEMA_VERSION,
        "linked_backtest_status_pass": linked.get("status") == "DIAGNOSTIC_PASS",
        "linked_backtest_strict_lab_gate_pass": linked.get("strict_lab_gate_pass") is True,
        "linked_backtest_lab_only": linked.get("promotion_gate") is False
        and linked.get("production_approved") is False
        and linked.get("independent_production_evidence") is False,
        "linked_backtest_no_ompex": linked.get("ompex_used_in_model") is False
        and linked.get("ompex_used_in_selection") is False
        and linked.get("ompex_used_in_backtest") is False,
        "linked_backtest_policy": linked.get("benchmark_policy") == SPOT_BACKTEST_POLICY,
    }


def _linked_audit_checks(summary: dict[str, Any]) -> dict[str, bool]:
    linked = _read_bound_json(
        summary,
        path_key="locked_holdout_audit",
        sha_key="locked_holdout_audit_sha256",
    )
    if linked is None:
        return {
            "linked_audit_json_readable": False,
            "linked_audit_schema": False,
            "linked_audit_status_pass": False,
            "linked_audit_holdout_pass": False,
            "linked_audit_lab_only": False,
            "linked_audit_no_ompex": False,
            "linked_audit_identity_matches_run": False,
            "linked_audit_backtest_path_matches_run": False,
            "linked_audit_backtest_sha_matches_run": False,
        }
    return {
        "linked_audit_json_readable": True,
        "linked_audit_schema": linked.get("schema_version") == AUDIT_SCHEMA_VERSION,
        "linked_audit_status_pass": linked.get("status") == "LOCKED_HOLDOUT_PASS",
        "linked_audit_holdout_pass": linked.get("holdout_pass") is True,
        "linked_audit_lab_only": linked.get("promotion_gate") is False and linked.get("production_approved") is False,
        "linked_audit_no_ompex": linked.get("ompex_used_in_model") is False
        and linked.get("ompex_used_in_selection") is False
        and linked.get("ompex_used_in_backtest") is False,
        "linked_audit_identity_matches_run": _same_identity(_identity(linked), _identity(summary)),
        "linked_audit_backtest_path_matches_run": linked.get("spot_backtest_summary")
        == summary.get("spot_backtest_summary"),
        "linked_audit_backtest_sha_matches_run": linked.get("spot_backtest_summary_sha256")
        == summary.get("spot_backtest_summary_sha256"),
    }


def _identity_checks(summary: dict[str, Any]) -> dict[str, bool]:
    identity = _identity(summary)
    checks = {
        "locked_plan_identity_present": isinstance(summary.get("locked_plan_identity"), dict)
        and bool(summary.get("locked_plan_identity")),
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


def _same_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
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
    return all(left.get(key) == right.get(key) for key in keys)


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


def _file_sha_bound(summary: dict[str, Any], *, path_key: str, sha_key: str) -> bool:
    path_text = summary.get(path_key)
    expected_sha = summary.get(sha_key)
    if not path_text or not expected_sha:
        return False
    path = Path(str(path_text))
    return path.exists() and expected_sha == _sha256(path)


def _read_bound_json(summary: dict[str, Any], *, path_key: str, sha_key: str) -> dict[str, Any] | None:
    if not _file_sha_bound(summary, path_key=path_key, sha_key=sha_key):
        return None
    try:
        payload = json.loads(Path(str(summary[path_key])).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)

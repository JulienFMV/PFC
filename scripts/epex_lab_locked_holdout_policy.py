"""Shared policy checks for locked EPEX lab holdout evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)

PLAN_SCHEMA_VERSION = "epex_lab_locked_holdout_plan.v1"
RUN_SCHEMA_VERSION = "epex_lab_locked_holdout_run.v2"
AUDIT_SCHEMA_VERSION = "epex_lab_locked_holdout_audit.v2"
COVERAGE_SCHEMA_VERSION = "epex_lab_locked_holdout_coverage.v1"
ENERGY_CHARTS_RUN_SCHEMA_VERSION = "energy_charts_epex_locked_holdout_run.v2"
LOCKED_HOLDOUT_POLICY = "locked_future_no_ompex_holdout"
SPOT_BACKTEST_SCHEMA_VERSION = "epex_shape_lab_spot_backtest.v3"
SPOT_BACKTEST_POLICY = "rolling_origin_epex_spot_no_ompex_lab_only"
T057_PLAN_ID = "t057_locked_t056_future_holdout"
T057_PLAN_SHA256 = "f2b5ce94d7eb892ec4f0b2e46b209d09b078db8d15765009fba4ba0cb21ec1cd"
T057_CAPTURE_SCHEMA_VERSION = "energy_charts_epex_locked_holdout_capture.v1"
T057_PLAN_RELATIVE_PATH = Path(
    ".planning/phases/14-lt-audit-remediation/locked_holdout_plan_t057_t056_asof20260709.json"
)
T057_OUTPUT_RELATIVE_PATH = Path(
    "output/phase14/t057_locked_t056_future_holdout/energy_charts_locked_runner_20260724"
)


def canonical_t057_plan_path() -> Path:
    return _resolved_path(Path(__file__).resolve().parent.parent / T057_PLAN_RELATIVE_PATH)


def canonical_t057_output_path() -> Path:
    return _resolved_path(Path(__file__).resolve().parent.parent / T057_OUTPUT_RELATIVE_PATH)


def build_locked_plan_identity(
    plan: dict[str, Any], *, plan_json: Path | None = None
) -> dict[str, Any]:
    criteria = plan.get("pass_criteria") or {}
    identity = {
        "plan_id": plan.get("plan_id"),
        "plan_schema_version": plan.get("schema_version"),
        "benchmark_policy": plan.get("benchmark_policy"),
        "frozen_at_utc": plan.get("frozen_at_utc"),
        "holdout_start_utc": plan.get("holdout_start_utc"),
        "holdout_end_utc": plan.get("holdout_end_utc"),
        "baseline_csv": plan.get("baseline_csv"),
        "baseline_csv_sha256": plan.get("baseline_csv_sha256")
        or criteria.get("baseline_csv_sha256"),
        "adjusted_csv": plan.get("adjusted_csv"),
        "adjusted_csv_sha256": plan.get("adjusted_csv_sha256")
        or criteria.get("adjusted_csv_sha256"),
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
    if schema == ENERGY_CHARTS_RUN_SCHEMA_VERSION:
        return _energy_charts_locked_holdout_policy(summary, checks)
    if schema == RUN_SCHEMA_VERSION:
        coverage = summary.get("coverage") if isinstance(summary.get("coverage"), dict) else {}
        coverage_checks = coverage.get("checks") if isinstance(coverage.get("checks"), dict) else {}
        checks.update(
            {
                "benchmark_policy_locked": summary.get("benchmark_policy") == LOCKED_HOLDOUT_POLICY,
                "expected_plan_json_sha256_present": bool(
                    str(summary.get("expected_plan_json_sha256") or "").strip()
                ),
                "actual_plan_json_sha256_present": bool(
                    str(summary.get("actual_plan_json_sha256") or "").strip()
                ),
                "expected_plan_json_sha256_bound": _expected_plan_sha_bound(summary),
                "coverage_ready": summary.get("coverage_ready") is True,
                "coverage_schema": coverage.get("schema_version") == COVERAGE_SCHEMA_VERSION,
                "coverage_read_only": coverage.get("read_only") is True,
                "coverage_promotion_gate_false": coverage.get("promotion_gate") is False,
                "coverage_production_approved_false": coverage.get("production_approved") is False,
                "coverage_identity_matches_run": _same_identity(
                    _identity(coverage), _identity(summary)
                ),
                "coverage_baseline_csv_sha256_matches_identity": coverage.get("baseline_csv_sha256")
                == _identity(summary).get("baseline_csv_sha256"),
                "coverage_adjusted_csv_sha256_matches_identity": coverage.get("adjusted_csv_sha256")
                == _identity(summary).get("adjusted_csv_sha256"),
                "coverage_candidate_timestamp_set_sha256_present": _candidate_timestamp_set_sha256_present(
                    coverage
                ),
                "coverage_candidate_timestamp_set_sha256_equal": _candidate_timestamp_set_sha256_equal(
                    coverage
                ),
                "coverage_candidate_timestamp_counts_valid": _candidate_timestamp_counts_valid(
                    coverage
                ),
                "coverage_candidate_timestamp_bounds_equal": _candidate_timestamp_bounds_equal(
                    coverage
                ),
                "coverage_status_ready": coverage.get("status") == "READY_TO_RUN_HOLDOUT_BACKTEST",
                "coverage_embedded_ready": coverage.get("ready_to_run_backtest") is True,
                "coverage_blocking_checks_clear": coverage.get("blocking_checks") == [],
                "coverage_full_window_covered": coverage_checks.get("full_window_covered") is True,
                "coverage_min_holdout_hours_met": coverage_checks.get("min_holdout_hours_met")
                is True,
                "coverage_no_duplicate_holdout_rows": coverage_checks.get(
                    "no_duplicate_holdout_rows"
                )
                is True,
                "coverage_spot_price_column_present": coverage_checks.get(
                    "spot_price_column_present"
                )
                is True,
                "coverage_holdout_prices_finite": coverage_checks.get("holdout_prices_finite")
                is True,
                "coverage_baseline_csv_sha256_bound": coverage_checks.get(
                    "baseline_csv_sha256_bound"
                )
                is True,
                "coverage_adjusted_csv_sha256_bound": coverage_checks.get(
                    "adjusted_csv_sha256_bound"
                )
                is True,
                "coverage_baseline_candidate_schema_ready": _candidate_coverage_checks_pass(
                    coverage_checks,
                    prefix="baseline_candidate",
                ),
                "coverage_adjusted_candidate_schema_ready": _candidate_coverage_checks_pass(
                    coverage_checks,
                    prefix="adjusted_candidate",
                ),
                "coverage_candidate_timestamp_sets_identical": coverage_checks.get(
                    "candidate_timestamp_sets_identical"
                )
                is True,
                "coverage_candidate_timestamp_set_matches_plan": coverage_checks.get(
                    "candidate_timestamp_set_matches_plan"
                )
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
                "t057_capture_route_bound": _t057_capture_route_bound(summary),
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
                "NO_GO_LOCKED_HOLDOUT_INPUT_INVALID",
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
        "bzn": summary.get("bzn"),
        "checks": checks,
    }


def _energy_charts_locked_holdout_policy(
    summary: dict[str, Any],
    checks: dict[str, bool],
) -> dict[str, Any]:
    checks.update(
        {
            "benchmark_policy_locked": summary.get("benchmark_policy") == LOCKED_HOLDOUT_POLICY,
            "expected_plan_json_sha256_present": bool(
                str(summary.get("expected_plan_json_sha256") or "").strip()
            ),
            "actual_plan_json_sha256_present": bool(
                str(summary.get("actual_plan_json_sha256") or "").strip()
            ),
            "expected_plan_json_sha256_bound": _expected_plan_sha_bound(summary),
            "spot_fetch_ran": summary.get("spot_fetch_ran") is True,
            "spot_fetch_summary_sha256_bound": _file_sha_bound(
                summary,
                path_key="spot_fetch_summary",
                sha_key="spot_fetch_summary_sha256",
            ),
            "spot_fetch_summary_matches_embedded": _spot_fetch_summary_matches_embedded(summary),
        }
    )
    if summary.get("status") in {
        "LOCKED_HOLDOUT_SPOT_WAITING",
        "LOCKED_HOLDOUT_WINDOW_NOT_COMPLETE",
    }:
        checks.update(
            {
                "waiting_status": True,
                "locked_holdout_not_run": summary.get("locked_holdout_ran") is False,
                "holdout_pass_false": summary.get("holdout_pass") is False,
            }
        )
        checks.update(_identity_checks(summary))
        return _locked_holdout_policy_result(
            summary,
            checks,
            passed=False,
            status="NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING",
            extra={"operator_wrapper_status": summary.get("status")},
        )

    linked = _read_bound_json(
        summary,
        path_key="locked_holdout_run_summary",
        sha_key="locked_holdout_run_summary_sha256",
    )
    linked_policy = locked_holdout_policy(linked) if linked is not None else None
    checks.update(
        {
            "locked_holdout_ran": summary.get("locked_holdout_ran") is True,
            "linked_locked_holdout_run_summary_sha256_bound": linked is not None,
            "linked_locked_holdout_embedded_matches": _linked_locked_holdout_matches_embedded(
                summary, linked
            ),
            "linked_locked_holdout_policy_pass": linked_policy is not None
            and linked_policy.get("pass") is True,
            "status_matches_linked": linked is not None
            and summary.get("status") == linked.get("status"),
            "holdout_pass_matches_linked": linked is not None
            and summary.get("holdout_pass") is True
            and linked.get("holdout_pass") is True,
            "capture_seal_sha256_bound": _file_sha_bound(
                summary,
                path_key="capture_seal",
                sha_key="capture_seal_sha256",
            ),
            "capture_policy_first_provider_no_overwrite": summary.get("capture_policy")
            == "FIRST_PROVIDER_CAPTURE_FAIL_CLOSED_NO_OVERWRITE",
        }
    )
    checks.update(_identity_checks(summary))
    passed = all(checks.values())
    status = (
        "LOCKED_HOLDOUT_PASS"
        if passed
        else (
            linked_policy.get("status")
            if linked_policy is not None
            else "NO_GO_LOCKED_HOLDOUT_FAIL"
        )
    )
    return _locked_holdout_policy_result(
        summary,
        checks,
        passed=passed,
        status=status,
        extra={
            "operator_wrapper_status": summary.get("status"),
            "linked_locked_holdout_policy": linked_policy,
        },
    )


def _locked_holdout_policy_result(
    summary: dict[str, Any],
    checks: dict[str, bool],
    *,
    passed: bool,
    status: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "provided": True,
        "schema_version": summary.get("schema_version"),
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
    if extra:
        result.update(extra)
    return result


def _spot_fetch_summary_matches_embedded(summary: dict[str, Any]) -> bool:
    linked = _read_bound_json(
        summary,
        path_key="spot_fetch_summary",
        sha_key="spot_fetch_summary_sha256",
    )
    return linked is not None and linked == summary.get("spot_fetch")


def _linked_locked_holdout_matches_embedded(
    summary: dict[str, Any],
    linked: dict[str, Any] | None,
) -> bool:
    if linked is None:
        return False
    embedded = summary.get("locked_holdout")
    return not isinstance(embedded, dict) or embedded == linked


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


def _t057_capture_route_bound(summary: dict[str, Any]) -> bool:
    if _identity(summary).get("plan_id") != T057_PLAN_ID:
        return True
    provenance = summary.get("spot_provenance")
    capture = provenance.get("capture_seal") if isinstance(provenance, dict) else None
    plan_json = _resolved_path(Path(str(_identity(summary).get("plan_json", ""))))
    output_dir = _resolved_path(Path(str(summary.get("output_dir", ""))))
    spot_parquet = _resolved_path(Path(str(summary.get("spot_parquet", ""))))
    capture_path = (
        _resolved_path(Path(str(capture.get("capture_seal", ""))))
        if isinstance(capture, dict)
        else Path()
    )
    return bool(
        _identity(summary).get("plan_json_sha256") == T057_PLAN_SHA256
        and summary.get("expected_plan_json_sha256") == T057_PLAN_SHA256
        and summary.get("actual_plan_json_sha256") == T057_PLAN_SHA256
        and plan_json == canonical_t057_plan_path()
        and output_dir == canonical_t057_output_path() / "locked_holdout_runner"
        and spot_parquet.parent == canonical_t057_output_path()
        and isinstance(capture, dict)
        and capture.get("required") is True
        and capture.get("pass") is True
        and capture.get("status") == "T057_CAPTURE_SEAL_BOUND"
        and _file_sha_bound(
            capture,
            path_key="capture_seal",
            sha_key="capture_seal_sha256",
        )
        and capture_path.parent == canonical_t057_output_path()
    )


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
    return (
        isinstance(baseline, int)
        and isinstance(adjusted, int)
        and baseline > 0
        and baseline == adjusted
    )


def _candidate_timestamp_bounds_equal(coverage: dict[str, Any]) -> bool:
    keys = [
        "baseline_candidate_timestamp_min_utc",
        "baseline_candidate_timestamp_max_utc",
        "adjusted_candidate_timestamp_min_utc",
        "adjusted_candidate_timestamp_max_utc",
    ]
    if not all(str(coverage.get(key) or "").strip() for key in keys):
        return False
    return coverage.get("baseline_candidate_timestamp_min_utc") == coverage.get(
        "adjusted_candidate_timestamp_min_utc"
    ) and coverage.get("baseline_candidate_timestamp_max_utc") == coverage.get(
        "adjusted_candidate_timestamp_max_utc"
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
            "linked_backtest_unique_ordered_cutoffs": False,
            "linked_backtest_non_overlapping_evaluations": False,
            "linked_backtest_lab_only": False,
            "linked_backtest_no_ompex": False,
            "linked_backtest_policy": False,
        }
    return {
        "linked_backtest_json_readable": True,
        "linked_backtest_schema": linked.get("schema_version") == SPOT_BACKTEST_SCHEMA_VERSION,
        "linked_backtest_status_pass": linked.get("status") == "DIAGNOSTIC_PASS",
        "linked_backtest_strict_lab_gate_pass": linked.get("strict_lab_gate_pass") is True,
        "linked_backtest_unique_ordered_cutoffs": (
            (linked.get("strict_lab_checks") or {}).get("rolling_folds_unique_ordered_cutoffs")
            is True
        ),
        "linked_backtest_non_overlapping_evaluations": (
            (linked.get("strict_lab_checks") or {}).get("rolling_folds_non_overlapping_evaluations")
            is True
        ),
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
            "linked_audit_unique_ordered_cutoffs_replayed": False,
            "linked_audit_non_overlapping_evaluations_replayed": False,
            "linked_audit_rolling_metrics_recomputed": False,
            "linked_audit_rolling_bucket_metrics_recomputed": False,
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
        "linked_audit_unique_ordered_cutoffs_replayed": (
            (linked.get("checks") or {}).get(
                "rolling_folds_unique_ordered_cutoffs_independently_replayed"
            )
            is True
        ),
        "linked_audit_non_overlapping_evaluations_replayed": (
            (linked.get("checks") or {}).get(
                "rolling_folds_non_overlapping_evaluations_independently_replayed"
            )
            is True
        ),
        "linked_audit_rolling_metrics_recomputed": (
            (linked.get("checks") or {}).get("rolling_metrics_independently_recomputed") is True
        ),
        "linked_audit_rolling_bucket_metrics_recomputed": (
            (linked.get("checks") or {}).get("rolling_bucket_metrics_independently_recomputed")
            is True
        ),
        "linked_audit_lab_only": linked.get("promotion_gate") is False
        and linked.get("production_approved") is False,
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
    try:
        plan_json = _admitted_path(Path(str(plan_json_text)))
    except (OSError, RuntimeError, ValueError):
        checks["plan_json_file_exists"] = False
        checks["plan_json_file_sha_bound"] = False
        checks["plan_identity_matches_plan_json"] = False
        return checks
    payload = _stable_payload(plan_json, label="locked holdout plan", max_bytes=4 * 1024 * 1024)
    checks["plan_json_file_exists"] = payload is not None
    actual_sha = hashlib.sha256(payload).hexdigest() if payload is not None else None
    checks["plan_json_file_sha_bound"] = (
        payload is not None and identity.get("plan_json_sha256") == actual_sha
    )
    if not checks["plan_json_file_sha_bound"]:
        checks["plan_identity_matches_plan_json"] = False
        return checks
    try:
        plan = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_mapping,
            parse_constant=_reject_json_constant,
        )
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        checks["plan_identity_matches_plan_json"] = False
        return checks
    expected = build_locked_plan_identity(plan)
    expected["plan_json"] = str(plan_json)
    expected["plan_json_sha256"] = actual_sha
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
    checks["plan_identity_matches_plan_json"] = all(
        identity.get(key) == expected.get(key) for key in keys
    )
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
    payload = _stable_payload(Path(str(path_text)), label=path_key, max_bytes=256 * 1024 * 1024)
    return payload is not None and expected_sha == hashlib.sha256(payload).hexdigest()


def _read_bound_json(
    summary: dict[str, Any], *, path_key: str, sha_key: str
) -> dict[str, Any] | None:
    path_text = summary.get(path_key)
    expected_sha = summary.get(sha_key)
    if not path_text or not expected_sha:
        return None
    payload_bytes = _stable_payload(Path(str(path_text)), label=path_key, max_bytes=4 * 1024 * 1024)
    if payload_bytes is None or hashlib.sha256(payload_bytes).hexdigest() != expected_sha:
        return None
    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_unique_json_mapping,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _unique_json_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _stable_payload(path: Path, *, label: str, max_bytes: int) -> bytes | None:
    try:
        return read_stable_single_link_file(
            _admitted_path(path),
            label=label,
            max_bytes=max_bytes,
        )
    except (OSError, RuntimeError, ValueError):
        return None


def _admitted_path(path: Path) -> Path:
    selected = path.expanduser()
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    return assert_absolute_path_has_no_links(selected)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)

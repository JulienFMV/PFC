from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_epex_lab_future_approval_path import (
    PRODUCTION_CHECKS,
    audit_future_approval_path,
    main,
)
from scripts.epex_lab_locked_holdout_policy import build_locked_plan_identity


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _readiness_payload(*, approved: bool = False, strict: bool = True, production: bool = False) -> dict:
    checks = [
        {"name": "product_all_gates_pass", "status": "PASS", "value": True},
    ]
    for name in PRODUCTION_CHECKS:
        value = production
        if name.endswith("_bound"):
            value = {
                "locked_holdout_summary_sha256": "holdout-sha" if production else None,
            }
        if name == "locked_holdout_queue_pass":
            value = _queue_policy_value(passed=production)
        checks.append({"name": name, "status": "PASS" if production else "FAIL", "value": value})
    return {
        "schema_version": "epex_lab_promotion_readiness.v1",
        "approved": approved,
        "status": "PROMOTION_READY" if approved else "STRICT_DIAGNOSTICS_PASS_PRODUCTION_CHAIN_MISSING",
        "strict_diagnostics_pass": strict,
        "production_chain_pass": production,
        "selected_adjusted_csv": "adjusted.csv",
        "missing_production_evidence": [] if production else ["adjusted_production_manifest"],
        "checks": checks,
    }


def _queue_policy_value(*, passed: bool = True) -> dict:
    return {
        "provided": True,
        "pass": passed,
        "status": "LOCKED_HOLDOUT_QUEUE_PASS" if passed else "NO_GO_LOCKED_HOLDOUT_QUEUE_INVALID",
        "queue_status": "WAITING_FOR_FUTURE_HOLDOUT_WINDOWS",
        "expected_locked_holdout_plan_sha256": "plan-sha",
        "checks": {
            "schema_version": True,
            "read_only": True,
            "production_approved_false": True,
            "promotion_gate_false": True,
            "invalid_plan_count_zero": True,
            "policy_invalid_plan_count_zero": True,
            "artifact_invalid_plan_count_zero": True,
            "duplicate_plan_id_count_zero": True,
            "overlapping_window_count_zero": True,
            "queue_issues_empty": True,
            "locked_holdout_plan_in_queue": passed,
        },
    }


def _spot_payload(**overrides) -> dict:
    payload = {
        "status": "DIAGNOSTIC_PASS",
        "promotion_gate": False,
        "production_approved": False,
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "benchmark_policy": "rolling_origin_epex_spot_no_ompex_lab_only",
    }
    payload.update(overrides)
    return payload


def _write_locked_plan(tmp_path: Path) -> Path:
    plan = tmp_path / "locked_plan.json"
    _write_json(
        plan,
        {
            "schema_version": "epex_lab_locked_holdout_plan.v1",
            "plan_id": "test_locked_holdout",
            "benchmark_policy": "locked_future_no_ompex_holdout",
            "frozen_at_utc": "2026-07-09T00:00:00Z",
            "holdout_start_utc": "2026-07-10T00:00:00Z",
            "holdout_end_utc": "2026-07-24T00:00:00Z",
            "baseline_csv_sha256": "b" * 64,
            "adjusted_csv_sha256": "a" * 64,
            "lab_manifest_sha256": "l" * 64,
            "selection_summary_sha256": "s" * 64,
        },
    )
    return plan


def _locked_holdout_run_payload(tmp_path: Path, **overrides) -> dict:
    plan = _write_locked_plan(tmp_path)
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    identity = build_locked_plan_identity(plan_payload, plan_json=plan)
    coverage_payload = _ready_coverage(identity=identity)
    coverage = _write_json(tmp_path / "coverage_status.json", coverage_payload)
    backtest = _write_json(tmp_path / "spot_backtest_summary.json", _passing_backtest())
    audit = _write_json(tmp_path / "locked_holdout_audit.json", _passing_audit(identity=identity, backtest=backtest))
    payload = {
        "schema_version": "epex_lab_locked_holdout_run.v2",
        "status": "LOCKED_HOLDOUT_PASS",
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
        "coverage_ready": True,
        "backtest_ran": True,
        "audit_ran": True,
        "holdout_pass": True,
        "locked_plan_identity": identity,
        "plan_json": str(plan),
        "spot_parquet": "spot.parquet",
        "output_dir": "out",
        "spot_backtest_summary": str(backtest),
        "spot_backtest_summary_sha256": _sha256(backtest),
        "locked_holdout_audit": str(audit),
        "locked_holdout_audit_sha256": _sha256(audit),
    }
    payload.update(overrides)
    return payload


def _ready_coverage(*, identity: dict) -> dict:
    timestamp_set_sha256 = "c" * 64
    return {
        "schema_version": "epex_lab_locked_holdout_coverage.v1",
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "locked_plan_identity": identity,
        "baseline_csv_sha256": identity["baseline_csv_sha256"],
        "adjusted_csv_sha256": identity["adjusted_csv_sha256"],
        "baseline_candidate_timestamp_count": 4,
        "baseline_candidate_timestamp_min_utc": "2026-07-10T00:00:00Z",
        "baseline_candidate_timestamp_max_utc": "2026-07-10T03:00:00Z",
        "baseline_candidate_timestamp_set_sha256": timestamp_set_sha256,
        "adjusted_candidate_timestamp_count": 4,
        "adjusted_candidate_timestamp_min_utc": "2026-07-10T00:00:00Z",
        "adjusted_candidate_timestamp_max_utc": "2026-07-10T03:00:00Z",
        "adjusted_candidate_timestamp_set_sha256": timestamp_set_sha256,
        "status": "READY_TO_RUN_HOLDOUT_BACKTEST",
        "ready_to_run_backtest": True,
        "blocking_checks": [],
        "checks": {
            "baseline_csv_sha256_bound": True,
            "adjusted_csv_sha256_bound": True,
            "baseline_candidate_required_columns_present": True,
            "baseline_candidate_utc_offset_present": True,
            "baseline_candidate_timestamps_parseable": True,
            "baseline_candidate_no_duplicate_timestamps": True,
            "baseline_candidate_price_columns_finite": True,
            "baseline_candidate_holdout_window_covered": True,
            "adjusted_candidate_required_columns_present": True,
            "adjusted_candidate_utc_offset_present": True,
            "adjusted_candidate_timestamps_parseable": True,
            "adjusted_candidate_no_duplicate_timestamps": True,
            "adjusted_candidate_price_columns_finite": True,
            "adjusted_candidate_holdout_window_covered": True,
            "candidate_timestamp_sets_identical": True,
            "candidate_timestamp_set_matches_plan": True,
            "candidate_timestamp_count_matches_plan": True,
            "full_window_covered": True,
            "min_holdout_hours_met": True,
            "no_duplicate_holdout_rows": True,
            "spot_price_column_present": True,
            "holdout_prices_finite": True,
        },
    }


def _passing_backtest() -> dict:
    return {
        "schema_version": "epex_shape_lab_spot_backtest.v3",
        "status": "DIAGNOSTIC_PASS",
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "independent_production_evidence": False,
        "benchmark_policy": "rolling_origin_epex_spot_no_ompex_lab_only",
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "strict_lab_gate_pass": True,
        "strict_lab_checks": {
            "rolling_folds_unique_ordered_cutoffs": True,
            "rolling_folds_non_overlapping_evaluations": True,
        },
    }


def _passing_audit(*, identity: dict, backtest: Path) -> dict:
    return {
        "schema_version": "epex_lab_locked_holdout_audit.v2",
        "status": "LOCKED_HOLDOUT_PASS",
        "holdout_pass": True,
        "checks": {
            "rolling_folds_unique_ordered_cutoffs_independently_replayed": True,
            "rolling_folds_non_overlapping_evaluations_independently_replayed": True,
            "rolling_metrics_independently_recomputed": True,
            "rolling_bucket_metrics_independently_recomputed": True,
        },
        "promotion_gate": False,
        "production_approved": False,
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "locked_plan_identity": identity,
        "spot_backtest_summary": str(backtest),
        "spot_backtest_summary_sha256": _sha256(backtest),
    }


def _write_locked_holdout_run(tmp_path: Path, **overrides) -> Path:
    return _write_json(tmp_path / "holdout.json", _locked_holdout_run_payload(tmp_path, **overrides))


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_future_approval_path_reports_no_go_blockers(tmp_path: Path) -> None:
    readiness = _write_json(tmp_path / "readiness.json", _readiness_payload())
    spot = _write_json(tmp_path / "spot.json", _spot_payload())

    summary = audit_future_approval_path(
        readiness_json=readiness,
        spot_backtest_summary=spot,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "NO_GO_PRODUCTION_CHAIN_INCOMPLETE"
    assert summary["approved"] is False
    assert summary["strict_diagnostics_pass"] is True
    assert "adjusted_production_manifest" in summary["remaining_blockers"]
    assert "adjusted_capstone_approved" in summary["remaining_blockers"]
    assert summary["spot_backtest_policy"]["pass"] is True
    assert summary["required_production_checks"] == PRODUCTION_CHECKS
    assert summary["blocking_stage"] == "production_evidence"
    assert summary["next_required_step"] == "generate_adjusted_production_export_selected_capstone_evidence"
    assert summary["next_actions"]


def test_future_approval_path_blocks_promotion_ready_without_locked_holdout(tmp_path: Path) -> None:
    readiness = _write_json(
        tmp_path / "readiness.json",
        _readiness_payload(approved=True, production=True),
    )

    summary = audit_future_approval_path(
        readiness_json=readiness,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "MISSING_LOCKED_HOLDOUT"
    assert summary["approved"] is False
    assert "locked_holdout_pass" in summary["remaining_blockers"]


def test_future_approval_path_blocks_production_chain_pass_without_locked_holdout_even_if_not_approved(
    tmp_path: Path,
) -> None:
    readiness = _write_json(
        tmp_path / "readiness.json",
        _readiness_payload(approved=False, production=True),
    )

    summary = audit_future_approval_path(
        readiness_json=readiness,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "MISSING_LOCKED_HOLDOUT"
    assert summary["approved"] is False
    assert "locked_holdout_pass" in summary["remaining_blockers"]


def test_future_approval_path_blocks_when_locked_holdout_coverage_pending(tmp_path: Path) -> None:
    readiness = _write_json(
        tmp_path / "readiness.json",
        _readiness_payload(approved=True, production=True),
    )
    holdout = _write_locked_holdout_run(
        tmp_path,
        status="WAITING_FOR_FULL_SPOT_COVERAGE",
        coverage_ready=False,
        backtest_ran=False,
        audit_ran=False,
        holdout_pass=False,
    )

    summary = audit_future_approval_path(
        readiness_json=readiness,
        locked_holdout_summary=holdout,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_COVERAGE_PENDING"
    assert summary["approved"] is False
    assert "locked_holdout_pass" in summary["remaining_blockers"]
    assert summary["locked_holdout_policy"]["pass"] is False
    assert summary["locked_holdout_policy"]["checks"]["coverage_ready"] is False
    assert summary["blocking_stage"] == "locked_holdout_coverage"
    assert summary["next_required_step"] == "wait_for_full_spot_coverage_then_run_locked_holdout"
    plan_json = str((tmp_path / "locked_plan.json").resolve())
    if any(char.isspace() for char in plan_json):
        plan_json = f'"{plan_json}"'
    assert summary["recommended_commands"]["run_energy_charts_locked_holdout"] == (
        "python scripts/run_energy_charts_epex_locked_holdout.py "
        f"--plan-json {plan_json} "
        f"--expected-plan-sha256 {_sha256(tmp_path / 'locked_plan.json')} "
        "--output-dir <ENERGY_CHARTS_LOCKED_HOLDOUT_OUTPUT_DIR> "
        "--bzn CH"
    )
    assert summary["recommended_commands"]["run_locked_holdout"] == (
        "python scripts/run_epex_lab_locked_holdout.py "
        f"--plan-json {plan_json} "
        f"--expected-plan-sha256 {_sha256(tmp_path / 'locked_plan.json')} "
        "--spot-parquet <FRESH_FUTURE_SPOT_PARQUET> "
        "--output-dir <T057_HOLDOUT_OUTPUT_DIR>"
    )


def test_future_approval_path_routes_locked_holdout_input_invalid(tmp_path: Path) -> None:
    readiness = _write_json(
        tmp_path / "readiness.json",
        _readiness_payload(approved=True, production=True),
    )
    holdout = _write_locked_holdout_run(
        tmp_path,
        status="NO_GO_LOCKED_HOLDOUT_INPUT_INVALID",
        coverage_ready=False,
        backtest_ran=False,
        audit_ran=False,
        holdout_pass=False,
    )

    summary = audit_future_approval_path(
        readiness_json=readiness,
        locked_holdout_summary=holdout,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_INPUT_INVALID"
    assert summary["approved"] is False
    assert summary["blocking_stage"] == "locked_holdout_input_invalid"
    assert summary["next_required_step"] == "fix_locked_holdout_plan_or_spot_inputs_then_rerun_preflight"
    assert summary["recommended_commands"] == {}
    assert "Fix the locked holdout plan policy or supplied spot parquet" in summary["next_actions"][0]


def test_future_approval_path_allows_promotion_ready_with_passing_locked_holdout(tmp_path: Path) -> None:
    holdout = _write_locked_holdout_run(tmp_path)
    payload = _readiness_payload(approved=True, production=True)
    for check in payload["checks"]:
        if isinstance(check.get("value"), dict) and "locked_holdout_summary_sha256" in check["value"]:
            check["value"]["locked_holdout_summary_sha256"] = _sha256(holdout)
    readiness = _write_json(tmp_path / "readiness.json", payload)

    summary = audit_future_approval_path(
        readiness_json=readiness,
        locked_holdout_summary=holdout,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "PROMOTION_READY_CANDIDATE"
    assert summary["approved"] is True
    assert summary["locked_holdout_policy"]["pass"] is True
    assert summary["blocking_stage"] == "promotion_ready_candidate"
    assert summary["next_required_step"] == "run_independent_capstone_review"
    assert summary["recommended_commands"] == {}


def test_future_approval_path_blocks_approved_payload_with_failed_production_check(tmp_path: Path) -> None:
    holdout = _write_locked_holdout_run(tmp_path)
    payload = _readiness_payload(approved=True, production=True)
    for check in payload["checks"]:
        if isinstance(check.get("value"), dict) and "locked_holdout_summary_sha256" in check["value"]:
            check["value"]["locked_holdout_summary_sha256"] = _sha256(holdout)
        if check.get("name") == "adjusted_capstone_approved":
            check["status"] = "FAIL"
            check["value"] = False
    readiness = _write_json(tmp_path / "readiness.json", payload)

    summary = audit_future_approval_path(
        readiness_json=readiness,
        locked_holdout_summary=holdout,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "NO_GO_PRODUCTION_CHAIN_INCOMPLETE"
    assert summary["approved"] is False
    assert summary["failed_production_checks"] == ["adjusted_capstone_approved"]
    assert "adjusted_capstone_approved" in summary["remaining_blockers"]
    assert summary["blocking_stage"] == "production_checks"


def test_future_approval_path_routes_locked_holdout_queue_pending(tmp_path: Path) -> None:
    holdout = _write_locked_holdout_run(tmp_path)
    payload = _readiness_payload(approved=True, production=True)
    for check in payload["checks"]:
        if isinstance(check.get("value"), dict) and "locked_holdout_summary_sha256" in check["value"]:
            check["value"]["locked_holdout_summary_sha256"] = _sha256(holdout)
        if check.get("name") == "locked_holdout_queue_pass":
            check["status"] = "FAIL"
            check["value"] = {
                "provided": True,
                "pass": False,
                "status": "NO_GO_LOCKED_HOLDOUT_QUEUE_PENDING",
                "queue_status": "WAITING_FOR_FUTURE_HOLDOUT_WINDOWS",
            }
    payload["approved"] = False
    payload["production_chain_pass"] = False
    readiness = _write_json(tmp_path / "readiness.json", payload)

    summary = audit_future_approval_path(
        readiness_json=readiness,
        locked_holdout_summary=holdout,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_QUEUE_PENDING"
    assert summary["approved"] is False
    assert summary["blocking_stage"] == "locked_holdout_queue"
    assert summary["next_required_step"] == "fix_locked_holdout_queue_before_promotion_review"
    assert summary["locked_holdout_policy"]["pass"] is True
    assert summary["locked_holdout_queue_policy"]["queue_status"] == "WAITING_FOR_FUTURE_HOLDOUT_WINDOWS"
    assert "Wait for locked holdout queue completion" in summary["next_actions"][0]


def test_future_approval_path_blocks_unbound_locked_holdout_queue_pass(tmp_path: Path) -> None:
    holdout = _write_locked_holdout_run(tmp_path)
    payload = _readiness_payload(approved=True, production=True)
    for check in payload["checks"]:
        if isinstance(check.get("value"), dict) and "locked_holdout_summary_sha256" in check["value"]:
            check["value"]["locked_holdout_summary_sha256"] = _sha256(holdout)
        if check.get("name") == "locked_holdout_queue_pass":
            check["status"] = "PASS"
            check["value"] = True
    readiness = _write_json(tmp_path / "readiness.json", payload)

    summary = audit_future_approval_path(
        readiness_json=readiness,
        locked_holdout_summary=holdout,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_QUEUE_UNBOUND"
    assert summary["approved"] is False
    assert summary["blocking_stage"] == "locked_holdout_queue"
    assert summary["locked_holdout_policy"]["pass"] is True
    assert summary["locked_holdout_queue_policy"]["pass"] is False
    assert "locked_holdout_queue_pass" in summary["remaining_blockers"]


def test_future_approval_path_blocks_synthetic_ready_payload_missing_production_checks(tmp_path: Path) -> None:
    readiness = _write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": "epex_lab_promotion_readiness.v1",
            "approved": True,
            "status": "PROMOTION_READY",
            "strict_diagnostics_pass": True,
            "production_chain_pass": True,
            "selected_adjusted_csv": "adjusted.csv",
            "missing_production_evidence": [],
            "checks": [],
        },
    )
    holdout = _write_locked_holdout_run(tmp_path)

    summary = audit_future_approval_path(
        readiness_json=readiness,
        locked_holdout_summary=holdout,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_HASH_MISMATCH"
    assert summary["approved"] is False
    assert "adjusted_production_manifest_locked_holdout_bound" in summary["missing_production_checks"]
    assert "locked_holdout_queue_pass" in summary["missing_production_checks"]
    assert "locked_holdout_sha_bound" in summary["remaining_blockers"]
    assert "Regenerate readiness with the full required production-check set before promotion review." in summary["next_actions"]


def test_future_approval_path_keeps_internal_required_checks_when_readiness_declares_subset(
    tmp_path: Path,
) -> None:
    readiness = _write_json(
        tmp_path / "readiness.json",
        {
            "schema_version": "epex_lab_promotion_readiness.v1",
            "approved": True,
            "status": "PROMOTION_READY",
            "strict_diagnostics_pass": True,
            "production_chain_pass": True,
            "selected_adjusted_csv": "adjusted.csv",
            "missing_production_evidence": [],
            "required_production_checks": ["locked_holdout_pass"],
            "checks": [
                {"name": "locked_holdout_pass", "status": "PASS", "value": True},
            ],
        },
    )
    holdout = _write_locked_holdout_run(tmp_path)

    summary = audit_future_approval_path(
        readiness_json=readiness,
        locked_holdout_summary=holdout,
        output=tmp_path / "out.json",
    )

    assert summary["approved"] is False
    assert "adjusted_production_manifest_approved" in summary["required_production_checks"]
    assert "adjusted_production_manifest_approved" in summary["missing_production_checks"]
    assert "locked_holdout_queue_pass" in summary["required_production_checks"]
    assert "locked_holdout_queue_pass" in summary["missing_production_checks"]


def test_future_approval_path_blocks_bad_spot_policy(tmp_path: Path) -> None:
    readiness = _write_json(tmp_path / "readiness.json", _readiness_payload())
    spot = _write_json(tmp_path / "spot.json", _spot_payload(ompex_used_in_backtest=True))

    summary = audit_future_approval_path(
        readiness_json=readiness,
        spot_backtest_summary=spot,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "NO_GO_SPOT_BACKTEST_POLICY_FAIL"
    assert summary["spot_backtest_policy"]["pass"] is False
    assert summary["spot_backtest_policy"]["checks"]["ompex_not_backtest"] is False
    assert summary["blocking_stage"] == "spot_policy"


def test_future_approval_path_rejects_unexpected_spot_benchmark_policy(tmp_path: Path) -> None:
    readiness = _write_json(tmp_path / "readiness.json", _readiness_payload())
    spot = _write_json(tmp_path / "spot.json", _spot_payload(benchmark_policy="advisory"))

    summary = audit_future_approval_path(
        readiness_json=readiness,
        spot_backtest_summary=spot,
        output=tmp_path / "out.json",
    )

    assert summary["status"] == "NO_GO_SPOT_BACKTEST_POLICY_FAIL"
    assert summary["spot_backtest_policy"]["pass"] is False
    assert summary["spot_backtest_policy"]["checks"]["benchmark_policy_no_ompex_lab_only"] is False
    assert summary["blocking_stage"] == "spot_policy"


def test_future_approval_path_cli_exits_nonzero_for_no_go(tmp_path: Path) -> None:
    readiness = _write_json(tmp_path / "readiness.json", _readiness_payload())

    code = main(
        [
            "--readiness-json",
            str(readiness),
            "--output",
            str(tmp_path / "out.json"),
        ]
    )

    assert code == 1


def test_future_approval_path_cli_exits_zero_for_promotion_ready_candidate(tmp_path: Path) -> None:
    holdout = _write_locked_holdout_run(tmp_path)
    payload = _readiness_payload(approved=True, production=True)
    for check in payload["checks"]:
        if isinstance(check.get("value"), dict) and "locked_holdout_summary_sha256" in check["value"]:
            check["value"]["locked_holdout_summary_sha256"] = _sha256(holdout)
    readiness = _write_json(tmp_path / "readiness.json", payload)

    code = main(
        [
            "--readiness-json",
            str(readiness),
            "--locked-holdout-summary",
            str(holdout),
            "--output",
            str(tmp_path / "out.json"),
        ]
    )

    assert code == 0

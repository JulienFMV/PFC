from __future__ import annotations

import json
from pathlib import Path

from scripts.epex_lab_locked_holdout_policy import build_locked_plan_identity, locked_holdout_policy


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_plan(tmp_path: Path) -> Path:
    return _write_json(
        tmp_path / "locked_plan.json",
        {
            "schema_version": "epex_lab_locked_holdout_plan.v1",
            "plan_id": "t057_locked_t056_future_holdout",
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


def _passing_run_summary(tmp_path: Path) -> dict:
    plan = _write_plan(tmp_path)
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    identity = build_locked_plan_identity(plan_payload, plan_json=plan)
    coverage_payload = _ready_coverage(identity=identity)
    coverage = _write_json(tmp_path / "coverage_status.json", coverage_payload)
    backtest = _write_json(tmp_path / "spot_backtest_summary.json", _passing_backtest())
    audit = _write_json(
        tmp_path / "locked_holdout_audit.json",
        _passing_audit(identity=identity, backtest=backtest),
    )
    return {
        "schema_version": "epex_lab_locked_holdout_run.v1",
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
        "spot_backtest_summary": str(backtest),
        "spot_backtest_summary_sha256": _sha256(backtest),
        "locked_holdout_audit": str(audit),
        "locked_holdout_audit_sha256": _sha256(audit),
    }


def test_locked_holdout_policy_accepts_plan_bound_run_summary(tmp_path: Path) -> None:
    policy = locked_holdout_policy(_passing_run_summary(tmp_path))

    assert policy["pass"] is True
    assert policy["checks"]["plan_json_file_sha_bound"] is True
    assert policy["checks"]["plan_identity_matches_plan_json"] is True


def test_build_locked_plan_identity_resolves_plan_json_path(tmp_path: Path, monkeypatch) -> None:
    plan = _write_plan(tmp_path)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    monkeypatch.chdir(tmp_path)

    identity = build_locked_plan_identity(payload, plan_json=Path("locked_plan.json"))

    assert identity["plan_json"] == str(plan.resolve())
    assert identity["plan_json_sha256"] == _sha256(plan)


def test_locked_holdout_policy_rejects_passing_summary_without_plan_identity(tmp_path: Path) -> None:
    summary = _passing_run_summary(tmp_path)
    summary.pop("locked_plan_identity")

    policy = locked_holdout_policy(summary)

    assert policy["pass"] is False
    assert policy["checks"]["locked_plan_identity_present"] is False
    assert policy["status"] == "NO_GO_LOCKED_HOLDOUT_FAIL"


def test_locked_holdout_policy_rejects_tampered_plan_after_summary(tmp_path: Path) -> None:
    summary = _passing_run_summary(tmp_path)
    plan = Path(summary["locked_plan_identity"]["plan_json"])
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["plan_id"] = "different_plan"
    plan.write_text(json.dumps(payload), encoding="utf-8")

    policy = locked_holdout_policy(summary)

    assert policy["pass"] is False
    assert policy["checks"]["plan_json_file_sha_bound"] is False


def test_locked_holdout_policy_rejects_tampered_backtest_summary(tmp_path: Path) -> None:
    summary = _passing_run_summary(tmp_path)
    backtest = Path(summary["spot_backtest_summary"])
    backtest.write_text(json.dumps({"status": "DIAGNOSTIC_FAIL"}), encoding="utf-8")

    policy = locked_holdout_policy(summary)

    assert policy["pass"] is False
    assert policy["checks"]["spot_backtest_summary_sha256_bound"] is False


def test_locked_holdout_policy_rejects_audit_only_summary(tmp_path: Path) -> None:
    summary = _passing_run_summary(tmp_path)
    audit = json.loads(Path(summary["locked_holdout_audit"]).read_text(encoding="utf-8"))

    policy = locked_holdout_policy(audit)

    assert policy["pass"] is False
    assert policy["status"] == "NO_GO_LOCKED_HOLDOUT_RUN_SUMMARY_REQUIRED"
    assert policy["checks"]["run_summary_schema_required"] is False


def test_locked_holdout_policy_rejects_coverage_file_mismatch(tmp_path: Path) -> None:
    summary = _passing_run_summary(tmp_path)
    coverage = Path(summary["coverage_status"])
    coverage.write_text(json.dumps({"status": "READY_TO_RUN_HOLDOUT_BACKTEST"}), encoding="utf-8")
    summary["coverage_status_sha256"] = _sha256(coverage)

    policy = locked_holdout_policy(summary)

    assert policy["pass"] is False
    assert policy["checks"]["coverage_status_matches_embedded"] is False


def test_locked_holdout_policy_rejects_coverage_without_candidate_checks(tmp_path: Path) -> None:
    summary = _passing_run_summary(tmp_path)
    coverage = summary["coverage"]
    for key in list(coverage["checks"]):
        if key.startswith("baseline_candidate_") or key.startswith("adjusted_candidate_"):
            coverage["checks"].pop(key)
    coverage["checks"].pop("candidate_timestamp_sets_identical")
    coverage_path = Path(summary["coverage_status"])
    coverage_path.write_text(json.dumps(coverage), encoding="utf-8")
    summary["coverage_status_sha256"] = _sha256(coverage_path)

    policy = locked_holdout_policy(summary)

    assert policy["pass"] is False
    assert policy["checks"]["coverage_baseline_candidate_schema_ready"] is False
    assert policy["checks"]["coverage_adjusted_candidate_schema_ready"] is False
    assert policy["checks"]["coverage_candidate_timestamp_sets_identical"] is False


def test_locked_holdout_policy_rejects_coverage_without_raw_candidate_timestamp_evidence(tmp_path: Path) -> None:
    summary = _passing_run_summary(tmp_path)
    coverage = summary["coverage"]
    for key in [
        "baseline_candidate_timestamp_count",
        "baseline_candidate_timestamp_min_utc",
        "baseline_candidate_timestamp_max_utc",
        "baseline_candidate_timestamp_set_sha256",
        "adjusted_candidate_timestamp_count",
        "adjusted_candidate_timestamp_min_utc",
        "adjusted_candidate_timestamp_max_utc",
        "adjusted_candidate_timestamp_set_sha256",
    ]:
        coverage.pop(key)
    coverage_path = Path(summary["coverage_status"])
    coverage_path.write_text(json.dumps(coverage), encoding="utf-8")
    summary["coverage_status_sha256"] = _sha256(coverage_path)

    policy = locked_holdout_policy(summary)

    assert policy["pass"] is False
    assert policy["checks"]["coverage_candidate_timestamp_set_sha256_present"] is False
    assert policy["checks"]["coverage_candidate_timestamp_counts_valid"] is False
    assert policy["checks"]["coverage_candidate_timestamp_bounds_equal"] is False


def test_locked_holdout_policy_rejects_coverage_identity_mismatch(tmp_path: Path) -> None:
    summary = _passing_run_summary(tmp_path)
    coverage = summary["coverage"]
    coverage["locked_plan_identity"] = {**coverage["locked_plan_identity"], "plan_id": "different"}
    coverage_path = Path(summary["coverage_status"])
    coverage_path.write_text(json.dumps(coverage), encoding="utf-8")
    summary["coverage_status_sha256"] = _sha256(coverage_path)

    policy = locked_holdout_policy(summary)

    assert policy["pass"] is False
    assert policy["checks"]["coverage_identity_matches_run"] is False


def test_locked_holdout_policy_rejects_coverage_with_blocking_checks(tmp_path: Path) -> None:
    summary = _passing_run_summary(tmp_path)
    coverage = summary["coverage"]
    coverage["blocking_checks"] = ["full_window_covered"]
    coverage_path = Path(summary["coverage_status"])
    coverage_path.write_text(json.dumps(coverage), encoding="utf-8")
    summary["coverage_status_sha256"] = _sha256(coverage_path)

    policy = locked_holdout_policy(summary)

    assert policy["pass"] is False
    assert policy["checks"]["coverage_blocking_checks_clear"] is False


def test_locked_holdout_policy_rejects_expected_plan_sha_mismatch(tmp_path: Path) -> None:
    summary = _passing_run_summary(tmp_path)
    summary["expected_plan_json_sha256"] = "0" * 64

    policy = locked_holdout_policy(summary)

    assert policy["pass"] is False
    assert policy["checks"]["expected_plan_json_sha256_bound"] is False


def test_locked_holdout_policy_preserves_input_invalid_status(tmp_path: Path) -> None:
    summary = _passing_run_summary(tmp_path)
    summary["status"] = "NO_GO_LOCKED_HOLDOUT_INPUT_INVALID"
    summary["coverage_ready"] = False
    summary["backtest_ran"] = False
    summary["audit_ran"] = False
    summary["holdout_pass"] = False

    policy = locked_holdout_policy(summary)

    assert policy["pass"] is False
    assert policy["status"] == "NO_GO_LOCKED_HOLDOUT_INPUT_INVALID"


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
        "schema_version": "epex_shape_lab_spot_backtest.v1",
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
    }


def _passing_audit(*, identity: dict, backtest: Path) -> dict:
    return {
        "schema_version": "epex_lab_locked_holdout_audit.v1",
        "status": "LOCKED_HOLDOUT_PASS",
        "holdout_pass": True,
        "promotion_gate": False,
        "production_approved": False,
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "locked_plan_identity": identity,
        "spot_backtest_summary": str(backtest),
        "spot_backtest_summary_sha256": _sha256(backtest),
    }


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

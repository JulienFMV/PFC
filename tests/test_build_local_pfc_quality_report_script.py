from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.build_local_pfc_quality_report as quality_module
import scripts.epex_lab_locked_holdout_policy as holdout_policy
from scripts.build_local_pfc_quality_report import build_local_quality_report


def test_local_quality_report_is_hash_bound_and_reproducible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_summary, repo = _fixture(tmp_path, monkeypatch)
    output_json = tmp_path / "quality.json"
    output_markdown = tmp_path / "quality.md"

    first = build_local_quality_report(
        run_summary=run_summary,
        expected_run_summary_sha256=_sha256(run_summary),
        repo_root=repo,
        output_json=output_json,
        output_markdown=output_markdown,
    )
    first_json = output_json.read_bytes()
    first_markdown = output_markdown.read_bytes()
    second = build_local_quality_report(
        run_summary=run_summary,
        expected_run_summary_sha256=_sha256(run_summary),
        repo_root=repo,
        output_json=output_json,
        output_markdown=output_markdown,
    )

    assert first == second
    assert output_json.read_bytes() == first_json
    assert output_markdown.read_bytes() == first_markdown
    assert first["schema_version"] == "pfc_local_quality_report.v2"
    assert first["status"] == "LOCAL_DIAGNOSTIC_PASS_NOT_PRODUCTION"
    assert first["production_authorization"] is False
    assert first["holdout"]["relative_mae_improvement_percent"] == "10.000000"
    assert first["gates"]["locked_t057_policy_pass"] is True
    assert first["gates"]["rolling_aggregates_recomputed_from_captured_csv"] is True
    assert "Production : **NON AUTORIS" in output_markdown.read_text(encoding="utf-8")


def test_local_quality_report_rejects_registry_superseded_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_summary, repo = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        quality_module,
        "supersession_record_for_sha256",
        lambda _sha256: {
            "authoritative_correction": {
                "path": "output/correction-v2.json",
                "sha256": "a" * 64,
            }
        },
    )

    with pytest.raises(ValueError, match="superseded fail-closed"):
        build_local_quality_report(
            run_summary=run_summary,
            expected_run_summary_sha256=_sha256(run_summary),
            repo_root=repo,
            output_json=tmp_path / "quality-superseded.json",
            output_markdown=tmp_path / "quality-superseded.md",
        )
def test_local_quality_report_rejects_tampered_independent_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_summary, repo = _fixture(tmp_path, monkeypatch)
    audit = run_summary.parent / "locked_holdout_runner" / "locked_holdout_audit.json"
    audit.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="locked holdout audit SHA-256 mismatch"):
        _build(run_summary, repo, tmp_path)


def test_local_quality_report_requires_caller_held_run_summary_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_summary, repo = _fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="caller-held anchor"):
        build_local_quality_report(
            run_summary=run_summary,
            expected_run_summary_sha256="0" * 64,
            repo_root=repo,
            output_json=tmp_path / "quality.json",
            output_markdown=tmp_path / "quality.md",
        )


def test_local_quality_report_rejects_output_inside_locked_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_summary, repo = _fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="outside locked T057 evidence"):
        build_local_quality_report(
            run_summary=run_summary,
            expected_run_summary_sha256=_sha256(run_summary),
            repo_root=repo,
            output_json=run_summary.parent / "quality.json",
            output_markdown=tmp_path / "quality.md",
        )


def test_local_quality_report_rejects_inconsistent_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_summary, repo = _fixture(tmp_path, monkeypatch)
    top = _read_json(run_summary)
    locked = top["locked_holdout"]
    locked["holdout_metrics"]["adjusted_residual_mae_eur_mwh"] = 99.0
    _persist_locked(run_summary, top)

    with pytest.raises(ValueError, match="do not match independent audit"):
        _build(run_summary, repo, tmp_path)


def test_local_quality_report_rejects_malformed_audit_criteria(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_summary, repo = _fixture(tmp_path, monkeypatch)
    top = _read_json(run_summary)
    audit = Path(top["locked_holdout"]["locked_holdout_audit"])
    payload = _read_json(audit)
    payload["criteria"] = []
    _write_json(audit, payload)
    top["locked_holdout"]["locked_holdout_audit_sha256"] = _sha256(audit)
    _persist_locked(run_summary, top)

    with pytest.raises(ValueError, match="audit criteria must be a mapping"):
        _build(run_summary, repo, tmp_path)


def test_local_quality_report_rejects_consistent_but_negative_uplift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_summary, repo = _fixture(tmp_path, monkeypatch)
    top = _read_json(run_summary)
    metrics = {
        "baseline_residual_mae_eur_mwh": 10.0,
        "adjusted_residual_mae_eur_mwh": 11.0,
        "residual_mae_improvement_eur_mwh": -1.0,
    }
    audit = Path(top["locked_holdout"]["locked_holdout_audit"])
    audit_payload = _read_json(audit)
    audit_payload["holdout_metrics"] = metrics
    _write_json(audit, audit_payload)
    top["locked_holdout"]["holdout_metrics"] = metrics
    top["locked_holdout"]["locked_holdout_audit_sha256"] = _sha256(audit)
    _persist_locked(run_summary, top)

    with pytest.raises(ValueError, match="future holdout improvement must be strictly positive"):
        _build(run_summary, repo, tmp_path)


def test_local_quality_report_rejects_forged_rolling_aggregates_even_when_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_summary, repo = _fixture(tmp_path, monkeypatch)
    top = _read_json(run_summary)
    locked = top["locked_holdout"]
    spot = Path(locked["spot_backtest_summary"])
    spot_payload = _read_json(spot)
    spot_payload["rolling_metrics"]["mean_profile_mae_improvement_eur_mwh"] = 9.0
    _write_json(spot, spot_payload)
    locked["spot_backtest_summary_sha256"] = _sha256(spot)
    audit = Path(locked["locked_holdout_audit"])
    audit_payload = _read_json(audit)
    audit_payload["spot_backtest_summary_sha256"] = _sha256(spot)
    _write_json(audit, audit_payload)
    locked["locked_holdout_audit_sha256"] = _sha256(audit)
    _persist_locked(run_summary, top)

    with pytest.raises(ValueError, match="rolling metrics .*captured CSV recomputation"):
        _build(run_summary, repo, tmp_path)


def test_local_quality_report_rejects_injected_summary_flags_without_audit_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_summary, repo = _fixture(tmp_path, monkeypatch)
    top = _read_json(run_summary)
    locked = top["locked_holdout"]
    audit = Path(locked["locked_holdout_audit"])
    audit_payload = _read_json(audit)
    audit_payload["checks"].pop("rolling_folds_unique_ordered_cutoffs_independently_replayed")
    audit_payload["checks"].pop("rolling_folds_non_overlapping_evaluations_independently_replayed")
    _write_json(audit, audit_payload)
    locked["locked_holdout_audit_sha256"] = _sha256(audit)
    _persist_locked(run_summary, top)

    with pytest.raises(ValueError, match="locked T057 holdout policy did not pass"):
        _build(run_summary, repo, tmp_path)


def _build(run_summary: Path, repo: Path, tmp_path: Path) -> dict[str, object]:
    return build_local_quality_report(
        run_summary=run_summary,
        expected_run_summary_sha256=_sha256(run_summary),
        repo_root=repo,
        output_json=tmp_path / "quality.json",
        output_markdown=tmp_path / "quality.md",
    )


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_root = repo / "output" / "t057"
    runner = run_root / "locked_holdout_runner"
    runner.mkdir(parents=True)

    for relative, payload in {
        "baseline.csv": b"baseline",
        "adjusted.csv": b"adjusted",
        "lab.json": b"lab",
        "selection.json": b"selection",
    }.items():
        (repo / relative).write_bytes(payload)
    plan_payload = {
        "schema_version": holdout_policy.PLAN_SCHEMA_VERSION,
        "plan_id": holdout_policy.T057_PLAN_ID,
        "benchmark_policy": holdout_policy.LOCKED_HOLDOUT_POLICY,
        "frozen_at_utc": "2026-07-09T00:00:00Z",
        "holdout_start_utc": "2026-07-10T00:00:00Z",
        "holdout_end_utc": "2026-07-24T00:00:00Z",
        "baseline_csv": "baseline.csv",
        "baseline_csv_sha256": _sha256(repo / "baseline.csv"),
        "adjusted_csv": "adjusted.csv",
        "adjusted_csv_sha256": _sha256(repo / "adjusted.csv"),
        "lab_manifest": "lab.json",
        "lab_manifest_sha256": _sha256(repo / "lab.json"),
        "selection_summary": "selection.json",
        "selection_summary_sha256": _sha256(repo / "selection.json"),
    }
    plan = repo / "plan.json"
    _write_json(plan, plan_payload)
    identity = holdout_policy.build_locked_plan_identity(plan_payload, plan_json=plan)
    monkeypatch.setattr(holdout_policy, "T057_PLAN_SHA256", identity["plan_json_sha256"])
    monkeypatch.setattr(holdout_policy, "canonical_t057_plan_path", lambda: plan.resolve())
    monkeypatch.setattr(holdout_policy, "canonical_t057_output_path", lambda: run_root.resolve())

    attempt = run_root / "attempt.json"
    capture = run_root / "energy_charts_locked_holdout_capture_seal.json"
    provider = run_root / "provider.json"
    parquet = run_root / "spot.parquet"
    for path, payload in (
        (attempt, b"attempt"),
        (capture, b'{"schema_version":"energy_charts_epex_locked_holdout_capture.v1"}'),
        (provider, b"provider"),
        (parquet, b"parquet"),
    ):
        path.write_bytes(payload)

    folds = runner / "rolling_spot_profile_folds.csv"
    folds.write_text(
        "cutoff_utc,eval_start_utc,eval_end_exclusive_utc,fold_eligible,"
        "baseline_profile_mae_eur_mwh,adjusted_profile_mae_eur_mwh,"
        "profile_mae_improvement_eur_mwh,baseline_profile_correlation,"
        "adjusted_profile_correlation,train_eval_profile_mae_eur_mwh,"
        "historical_fold_not_independent_of_current_candidate_fit,"
        "eval_after_current_valuation\n"
        "2026-06-01T00:00:00Z,2026-06-02T00:00:00Z,2026-06-16T00:00:00Z,True,"
        "8.0,7.0,1.0,0.8,0.9,6.0,True,False\n"
        "2026-06-16T00:00:00Z,2026-06-17T00:00:00Z,2026-07-01T00:00:00Z,True,"
        "8.0,7.0,1.0,0.8,0.9,6.0,True,False\n",
        encoding="utf-8",
    )
    bucket_csv = runner / "rolling_spot_bucket_metrics.csv"
    bucket_csv.write_text(
        "cutoff_utc,bucket,metric_type,hours,baseline_mae_eur_mwh,"
        "adjusted_mae_eur_mwh,mae_improvement_eur_mwh,baseline_rmse_eur_mwh,"
        "adjusted_rmse_eur_mwh,rmse_improvement_eur_mwh,adjusted_error_p95_eur_mwh\n"
        "2026-06-01T00:00:00Z,all,residual_level,336,8.0,7.0,1.0,9.0,8.0,1.0,12.0\n"
        "2026-06-16T00:00:00Z,all,residual_level,336,8.0,7.0,1.0,9.0,8.0,1.0,12.0\n",
        encoding="utf-8",
    )
    rolling_metrics = {
        "mean_baseline_profile_mae_eur_mwh": 8.0,
        "mean_adjusted_profile_mae_eur_mwh": 7.0,
        "mean_profile_mae_improvement_eur_mwh": 1.0,
        "median_profile_mae_improvement_eur_mwh": 1.0,
        "positive_mae_improvement_fold_count": 2,
        "mean_baseline_profile_correlation": 0.8,
        "mean_adjusted_profile_correlation": 0.9,
        "mean_train_eval_profile_mae_eur_mwh": 6.0,
        "historical_fold_not_independent_count": 2,
        "post_valuation_fold_count": 0,
    }
    rolling_buckets = {
        "residual_level:all": {
            "fold_count": 2,
            "total_hours": 672,
            "mean_baseline_mae_eur_mwh": 8.0,
            "mean_adjusted_mae_eur_mwh": 7.0,
            "mean_mae_improvement_eur_mwh": 1.0,
            "positive_mae_improvement_fold_count": 2,
            "mean_baseline_rmse_eur_mwh": 9.0,
            "mean_adjusted_rmse_eur_mwh": 8.0,
            "mean_rmse_improvement_eur_mwh": 1.0,
            "mean_adjusted_error_p95_eur_mwh": 12.0,
        }
    }
    spot_payload = {
        "schema_version": holdout_policy.SPOT_BACKTEST_SCHEMA_VERSION,
        "status": "DIAGNOSTIC_PASS",
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "independent_production_evidence": False,
        "benchmark_policy": holdout_policy.SPOT_BACKTEST_POLICY,
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "rolling_fold_count": 2,
        "eligible_rolling_fold_count": 2,
        "rolling_metrics": rolling_metrics,
        "rolling_bucket_metrics": rolling_buckets,
        "outputs": {
            "rolling_spot_profile_folds_csv": str(folds.resolve()),
            "rolling_spot_bucket_metrics_csv": str(bucket_csv.resolve()),
        },
        "output_hashes": {
            "rolling_spot_profile_folds_csv": _sha256(folds),
            "rolling_spot_bucket_metrics_csv": _sha256(bucket_csv),
        },
        "strict_lab_checks": {
            "rolling_folds_no_temporal_leak": True,
            "rolling_folds_unique_ordered_cutoffs": True,
            "rolling_folds_non_overlapping_evaluations": True,
        },
        "strict_lab_gate_pass": True,
    }
    spot = runner / "spot_backtest_summary.json"
    _write_json(spot, spot_payload)

    metrics = {
        "baseline_residual_mae_eur_mwh": 10.0,
        "adjusted_residual_mae_eur_mwh": 9.0,
        "residual_mae_improvement_eur_mwh": 1.0,
    }
    audit_payload = {
        "schema_version": holdout_policy.AUDIT_SCHEMA_VERSION,
        "status": "LOCKED_HOLDOUT_PASS",
        "holdout_pass": True,
        "promotion_gate": False,
        "production_approved": False,
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "locked_plan_identity": identity,
        "spot_backtest_summary": str(spot.resolve()),
        "spot_backtest_summary_sha256": _sha256(spot),
        "holdout_metrics": metrics,
        "criteria": {"strict_lab_gate_pass": True},
        "checks": {
            "rolling_folds_temporal_integrity_independently_replayed": True,
            "rolling_folds_unique_ordered_cutoffs_independently_replayed": True,
            "rolling_folds_non_overlapping_evaluations_independently_replayed": True,
            "rolling_metrics_independently_recomputed": True,
            "rolling_bucket_metrics_independently_recomputed": True,
        },
    }
    audit = runner / "locked_holdout_audit.json"
    _write_json(audit, audit_payload)

    coverage_payload = _ready_coverage(identity)
    coverage = runner / "coverage_status.json"
    _write_json(coverage, coverage_payload)
    locked = {
        "schema_version": holdout_policy.RUN_SCHEMA_VERSION,
        "status": "LOCKED_HOLDOUT_PASS",
        "benchmark_policy": holdout_policy.LOCKED_HOLDOUT_POLICY,
        "expected_plan_json_sha256": identity["plan_json_sha256"],
        "actual_plan_json_sha256": identity["plan_json_sha256"],
        "promotion_gate": False,
        "production_approved": False,
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "coverage_status": str(coverage.resolve()),
        "coverage_status_sha256": _sha256(coverage),
        "coverage": coverage_payload,
        "coverage_ready": True,
        "backtest_ran": True,
        "audit_ran": True,
        "holdout_pass": True,
        "strict_lab_gate_pass": True,
        "holdout_metrics": metrics,
        "locked_plan_identity": identity,
        "spot_backtest_summary": str(spot.resolve()),
        "spot_backtest_summary_sha256": _sha256(spot),
        "locked_holdout_audit": str(audit.resolve()),
        "locked_holdout_audit_sha256": _sha256(audit),
        "spot_parquet": str(parquet.resolve()),
        "output_dir": str(runner.resolve()),
        "spot_provenance": {
            "pass": True,
            "capture_seal": {
                "required": True,
                "pass": True,
                "status": "T057_CAPTURE_SEAL_BOUND",
                "capture_seal": str(capture.resolve()),
                "capture_seal_sha256": _sha256(capture),
            },
        },
    }
    locked_summary = runner / "locked_holdout_run_summary.json"
    _write_json(locked_summary, locked)

    fetch_payload = {
        "schema_version": "energy_charts_epex_spot_hourly_fetch.v1",
        "status": "OK",
        "full_window_covered": True,
        "expected_hour_count": 336,
        "observed_hour_count": 1000,
        "output_parquet": str(parquet.resolve()),
        "output_parquet_sha256": _sha256(parquet),
    }
    fetch = run_root / "fetch.json"
    _write_json(fetch, fetch_payload)
    top = {
        "schema_version": holdout_policy.ENERGY_CHARTS_RUN_SCHEMA_VERSION,
        "status": "LOCKED_HOLDOUT_PASS",
        "benchmark_policy": holdout_policy.LOCKED_HOLDOUT_POLICY,
        "expected_plan_json_sha256": identity["plan_json_sha256"],
        "actual_plan_json_sha256": identity["plan_json_sha256"],
        "production_approved": False,
        "promotion_gate": False,
        "ompex_used_in_backtest": False,
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "holdout_pass": True,
        "holdout_start_utc": "2026-07-10T00:00:00Z",
        "holdout_end_utc": "2026-07-24T00:00:00Z",
        "output_dir": str(run_root.resolve()),
        "spot_parquet": str(parquet.resolve()),
        "attempt_seal": str(attempt.resolve()),
        "attempt_seal_sha256": _sha256(attempt),
        "capture_seal": str(capture.resolve()),
        "capture_seal_sha256": _sha256(capture),
        "capture_policy": "FIRST_PROVIDER_CAPTURE_FAIL_CLOSED_NO_OVERWRITE",
        "provider_raw_json": str(provider.resolve()),
        "provider_raw_json_sha256": _sha256(provider),
        "spot_fetch_ran": True,
        "spot_fetch_summary": str(fetch.resolve()),
        "spot_fetch_summary_sha256": _sha256(fetch),
        "spot_fetch": fetch_payload,
        "locked_holdout_ran": True,
        "locked_holdout_run_summary": str(locked_summary.resolve()),
        "locked_holdout_run_summary_sha256": _sha256(locked_summary),
        "locked_holdout": locked,
        "locked_plan_identity": identity,
    }
    run_summary = run_root / "energy_charts_locked_holdout_run_summary.json"
    _write_json(run_summary, top)
    return run_summary, repo


def _ready_coverage(identity: dict[str, object]) -> dict[str, object]:
    timestamp_set_sha256 = "c" * 64
    checks = {
        "baseline_csv_sha256_bound": True,
        "adjusted_csv_sha256_bound": True,
        "candidate_timestamp_sets_identical": True,
        "candidate_timestamp_set_matches_plan": True,
        "candidate_timestamp_count_matches_plan": True,
        "full_window_covered": True,
        "min_holdout_hours_met": True,
        "no_duplicate_holdout_rows": True,
        "spot_price_column_present": True,
        "holdout_prices_finite": True,
    }
    for prefix in ("baseline_candidate", "adjusted_candidate"):
        for suffix in (
            "required_columns_present",
            "utc_offset_present",
            "timestamps_parseable",
            "no_duplicate_timestamps",
            "price_columns_finite",
            "holdout_window_covered",
        ):
            checks[f"{prefix}_{suffix}"] = True
    return {
        "schema_version": holdout_policy.COVERAGE_SCHEMA_VERSION,
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "locked_plan_identity": identity,
        "baseline_csv_sha256": identity["baseline_csv_sha256"],
        "adjusted_csv_sha256": identity["adjusted_csv_sha256"],
        "baseline_candidate_timestamp_count": 336,
        "baseline_candidate_timestamp_min_utc": "2026-07-10T00:00:00Z",
        "baseline_candidate_timestamp_max_utc": "2026-07-23T23:00:00Z",
        "baseline_candidate_timestamp_set_sha256": timestamp_set_sha256,
        "adjusted_candidate_timestamp_count": 336,
        "adjusted_candidate_timestamp_min_utc": "2026-07-10T00:00:00Z",
        "adjusted_candidate_timestamp_max_utc": "2026-07-23T23:00:00Z",
        "adjusted_candidate_timestamp_set_sha256": timestamp_set_sha256,
        "expected_holdout_hours": 336,
        "missing_holdout_hours": 0,
        "status": "READY_TO_RUN_HOLDOUT_BACKTEST",
        "ready_to_run_backtest": True,
        "blocking_checks": [],
        "checks": checks,
    }


def _persist_locked(run_summary: Path, top: dict[str, object]) -> None:
    locked_summary = Path(top["locked_holdout_run_summary"])
    _write_json(locked_summary, top["locked_holdout"])
    top["locked_holdout_run_summary_sha256"] = _sha256(locked_summary)
    _write_json(run_summary, top)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

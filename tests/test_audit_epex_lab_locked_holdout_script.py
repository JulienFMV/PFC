from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.audit_epex_lab_locked_holdout as audit_module
from scripts.audit_epex_lab_locked_holdout import audit_holdout, main
from scripts.backtest_epex_shape_lab_against_spot import (
    PRICE_COLUMNS,
    _aggregate_bucket_metrics,
    _aggregate_fold_metrics,
    _candidate_profiles,
    _evaluate_fold,
    _fold_bucket_rows,
    _join_candidates,
    _jsonable,
    _load_candidate,
    _load_spot,
    _rolling_cutoffs,
)


def test_audit_epex_lab_locked_holdout_passes_no_ompex_window(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    post = tmp_path / "post.csv"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-07-10", periods=4, freq="h", tz="UTC").astype(str),
            "baseline_abs_error_eur_mwh": [4.0, 4.0, 4.0, 4.0],
            "adjusted_abs_error_eur_mwh": [3.0, 3.5, 3.0, 3.5],
        }
    ).to_csv(post, index=False)
    summary = _write_summary(tmp_path)

    audit = audit_holdout(
        plan_json=plan, spot_backtest_summary=summary, output=tmp_path / "audit.json"
    )

    assert audit["status"] == "LOCKED_HOLDOUT_PASS"
    assert audit["holdout_pass"] is True
    assert audit["approved"] is False
    assert audit["promotion_gate"] is False
    assert audit["production_approved"] is False
    assert audit["ompex_used_in_model"] is False
    assert audit["ompex_used_in_selection"] is False
    assert audit["ompex_used_in_backtest"] is False
    assert audit["checks"]["summary_no_ompex"] is True
    assert audit["holdout_metrics"]["hours"] == 4
    assert audit["holdout_metrics"]["residual_mae_improvement_eur_mwh"] > 0
    assert (tmp_path / "audit.json").exists()


def test_audit_epex_lab_locked_holdout_cli_exits_zero_when_passed(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    post = tmp_path / "post.csv"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-07-10", periods=4, freq="h", tz="UTC").astype(str),
            "baseline_abs_error_eur_mwh": [4.0, 4.0, 4.0, 4.0],
            "adjusted_abs_error_eur_mwh": [3.0, 3.5, 3.0, 3.5],
        }
    ).to_csv(post, index=False)
    summary = _write_summary(tmp_path)

    code = main(
        [
            "--plan-json",
            str(plan),
            "--spot-backtest-summary",
            str(summary),
            "--output",
            str(tmp_path / "audit.json"),
        ]
    )

    assert code == 0


def test_audit_epex_lab_locked_holdout_fails_degraded_window(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    post = tmp_path / "post.csv"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-07-10", periods=4, freq="h", tz="UTC").astype(str),
            "baseline_abs_error_eur_mwh": [3.0, 3.0, 3.0, 3.0],
            "adjusted_abs_error_eur_mwh": [4.0, 4.0, 4.0, 4.0],
        }
    ).to_csv(post, index=False)
    summary = _write_summary(tmp_path)

    audit = audit_holdout(plan_json=plan, spot_backtest_summary=summary)

    assert audit["status"] == "NO_GO_LOCKED_HOLDOUT_FAIL"
    assert audit["holdout_pass"] is False
    assert audit["checks"]["holdout_non_degraded"] is False


def test_audit_epex_lab_locked_holdout_rejects_non_lab_backtest_schema(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    post = tmp_path / "post.csv"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-07-10", periods=4, freq="h", tz="UTC").astype(str),
            "baseline_abs_error_eur_mwh": [4.0, 4.0, 4.0, 4.0],
            "adjusted_abs_error_eur_mwh": [3.0, 3.0, 3.0, 3.0],
        }
    ).to_csv(post, index=False)
    summary = _write_summary(tmp_path, schema_version="other.v1")

    audit = audit_holdout(plan_json=plan, spot_backtest_summary=summary)

    assert audit["status"] == "NO_GO_LOCKED_HOLDOUT_FAIL"
    assert audit["holdout_pass"] is False
    assert audit["checks"]["summary_schema"] is False


def test_audit_epex_lab_locked_holdout_rejects_diagnostic_fail_summary(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    post = tmp_path / "post.csv"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-07-10", periods=4, freq="h", tz="UTC").astype(str),
            "baseline_abs_error_eur_mwh": [4.0, 4.0, 4.0, 4.0],
            "adjusted_abs_error_eur_mwh": [3.0, 3.0, 3.0, 3.0],
        }
    ).to_csv(post, index=False)
    summary = _write_summary(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["status"] = "DIAGNOSTIC_FAIL"
    summary.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_holdout(plan_json=plan, spot_backtest_summary=summary)

    assert audit["status"] == "NO_GO_LOCKED_HOLDOUT_FAIL"
    assert audit["holdout_pass"] is False
    assert audit["checks"]["summary_status_pass"] is False


def test_audit_epex_lab_locked_holdout_rejects_tampered_post_csv(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    post = tmp_path / "post.csv"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-07-10", periods=4, freq="h", tz="UTC").astype(str),
            "baseline_abs_error_eur_mwh": [4.0, 4.0, 4.0, 4.0],
            "adjusted_abs_error_eur_mwh": [3.0, 3.0, 3.0, 3.0],
        }
    ).to_csv(post, index=False)
    summary = _write_summary(tmp_path)
    post.write_text(
        "timestamp_utc,baseline_abs_error_eur_mwh,adjusted_abs_error_eur_mwh\n"
        "2026-07-10T00:00:00Z,4,3\n",
        encoding="utf-8",
    )

    audit = audit_holdout(plan_json=plan, spot_backtest_summary=summary)

    assert audit["status"] == "NO_GO_LOCKED_HOLDOUT_FAIL"
    assert audit["holdout_pass"] is False
    assert audit["checks"]["post_valuation_csv_sha256_bound"] is False


def test_audit_replays_provider_raw_instead_of_trusting_declared_hashes(
    tmp_path: Path,
) -> None:
    plan = _write_plan(tmp_path)
    post = tmp_path / "post.csv"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-07-10", periods=4, freq="h", tz="UTC").astype(str),
            "baseline_abs_error_eur_mwh": [4.0] * 4,
            "adjusted_abs_error_eur_mwh": [3.0] * 4,
        }
    ).to_csv(post, index=False)
    summary = _write_summary(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    provenance = payload["spot_provenance"]
    raw = Path(provenance["provider_raw_json"])
    raw_payload = json.loads(raw.read_text(encoding="utf-8"))
    raw_payload["price"] = [price + 100.0 for price in raw_payload["price"]]
    raw.write_text(json.dumps(raw_payload), encoding="utf-8")
    acquisition = Path(provenance["acquisition_summary"])
    acquisition_payload = json.loads(acquisition.read_text(encoding="utf-8"))
    acquisition_payload["provider_raw_json_sha256"] = _sha256(raw)
    acquisition.write_text(json.dumps(acquisition_payload), encoding="utf-8")
    provenance["provider_raw_json_sha256"] = _sha256(raw)
    provenance["acquisition_summary_sha256"] = _sha256(acquisition)
    summary.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_holdout(plan_json=plan, spot_backtest_summary=summary)

    assert audit["holdout_pass"] is False
    assert audit["checks"]["spot_provider_raw_semantic_replay"] is False


def test_audit_replays_fold_temporal_integrity_instead_of_trusting_flags(
    tmp_path: Path,
) -> None:
    plan = _write_plan(tmp_path)
    post = tmp_path / "post.csv"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-07-10", periods=4, freq="h", tz="UTC").astype(str),
            "baseline_abs_error_eur_mwh": [4.0] * 4,
            "adjusted_abs_error_eur_mwh": [3.0] * 4,
        }
    ).to_csv(post, index=False)
    summary = _write_summary(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    folds = Path(payload["outputs"]["rolling_spot_profile_folds_csv"])
    fold_payload = pd.read_csv(folds)
    fold_payload.loc[0, "train_start_utc"] = "2026-07-08T00:00:00Z"
    fold_payload.loc[0, "no_temporal_leak"] = True
    fold_payload.to_csv(folds, index=False)
    payload["output_hashes"]["rolling_spot_profile_folds_csv"] = _sha256(folds)
    summary.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_holdout(plan_json=plan, spot_backtest_summary=summary)

    assert audit["holdout_pass"] is False
    assert audit["checks"]["rolling_folds_temporal_integrity_independently_replayed"] is False


def test_audit_rejects_cherry_picked_fold_with_self_consistent_counts(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    post = tmp_path / "post.csv"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-07-10", periods=4, freq="h", tz="UTC").astype(str),
            "baseline_abs_error_eur_mwh": [4.0] * 4,
            "adjusted_abs_error_eur_mwh": [3.0] * 4,
        }
    ).to_csv(post, index=False)
    summary = _write_summary(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    folds = Path(payload["outputs"]["rolling_spot_profile_folds_csv"])
    cherry_picked = pd.read_csv(folds).iloc[[0]]
    cherry_picked.to_csv(folds, index=False)
    payload["rolling_fold_count"] = 1
    payload["eligible_rolling_fold_count"] = int(cherry_picked["fold_eligible"].sum())
    payload["output_hashes"]["rolling_spot_profile_folds_csv"] = _sha256(folds)
    summary.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_holdout(plan_json=plan, spot_backtest_summary=summary)

    assert audit["holdout_pass"] is False
    assert audit["checks"]["rolling_folds_csv_independently_hashed"] is True
    assert audit["checks"]["rolling_folds_count_independently_replayed"] is False


def test_audit_rejects_forged_rolling_aggregates_with_valid_csv_hashes(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    post = tmp_path / "post.csv"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-07-10", periods=4, freq="h", tz="UTC").astype(str),
            "baseline_abs_error_eur_mwh": [4.0] * 4,
            "adjusted_abs_error_eur_mwh": [3.0] * 4,
        }
    ).to_csv(post, index=False)
    summary = _write_summary(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["rolling_metrics"]["mean_profile_mae_improvement_eur_mwh"] += 10.0
    summary.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_holdout(plan_json=plan, spot_backtest_summary=summary)

    assert audit["holdout_pass"] is False
    assert audit["checks"]["rolling_folds_csv_independently_hashed"] is True
    assert audit["checks"]["rolling_bucket_csv_independently_hashed"] is True
    assert audit["checks"]["rolling_metrics_independently_recomputed"] is False
    assert audit["checks"]["rolling_bucket_metrics_independently_recomputed"] is True


def test_audit_rejects_forged_fold_rows_with_recascaded_hash_and_aggregate(
    tmp_path: Path,
) -> None:
    plan = _write_plan(tmp_path)
    post = tmp_path / "post.csv"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-07-10", periods=4, freq="h", tz="UTC").astype(
                str
            ),
            "baseline_abs_error_eur_mwh": [4.0] * 4,
            "adjusted_abs_error_eur_mwh": [3.0] * 4,
        }
    ).to_csv(post, index=False)
    summary = _write_summary(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    folds = Path(payload["outputs"]["rolling_spot_profile_folds_csv"])
    forged = pd.read_csv(folds)
    forged.loc[0, "profile_mae_improvement_eur_mwh"] = 999.0
    forged.loc[0, "adjusted_profile_correlation"] = -0.999
    forged.loc[0, "historical_fold_not_independent_of_current_candidate_fit"] = False
    forged.to_csv(folds, index=False)
    eligible = forged[forged["fold_eligible"].astype(bool)]
    payload["rolling_metrics"] = _aggregate_fold_metrics(eligible)
    payload["output_hashes"]["rolling_spot_profile_folds_csv"] = _sha256(folds)
    summary.write_text(json.dumps(_jsonable(payload)), encoding="utf-8")

    audit = audit_holdout(plan_json=plan, spot_backtest_summary=summary)

    assert audit["holdout_pass"] is False
    assert audit["checks"]["rolling_folds_csv_independently_hashed"] is True
    assert audit["checks"]["rolling_folds_temporal_integrity_independently_replayed"] is False
    assert audit["checks"]["rolling_metrics_independently_recomputed"] is False


def test_audit_rejects_forged_bucket_rows_with_recascaded_hash_and_aggregate(
    tmp_path: Path,
) -> None:
    plan = _write_plan(tmp_path)
    post = tmp_path / "post.csv"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-07-10", periods=4, freq="h", tz="UTC").astype(
                str
            ),
            "baseline_abs_error_eur_mwh": [4.0] * 4,
            "adjusted_abs_error_eur_mwh": [3.0] * 4,
        }
    ).to_csv(post, index=False)
    summary = _write_summary(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    buckets = Path(payload["outputs"]["rolling_spot_bucket_metrics_csv"])
    forged = pd.read_csv(buckets)
    forged.loc[0, "mae_improvement_eur_mwh"] = 999.0
    forged.loc[0, "adjusted_error_p95_eur_mwh"] = 0.0
    forged.to_csv(buckets, index=False)
    payload["rolling_bucket_metrics"] = _aggregate_bucket_metrics(forged)
    payload["output_hashes"]["rolling_spot_bucket_metrics_csv"] = _sha256(buckets)
    summary.write_text(json.dumps(_jsonable(payload)), encoding="utf-8")

    audit = audit_holdout(plan_json=plan, spot_backtest_summary=summary)

    assert audit["holdout_pass"] is False
    assert audit["checks"]["rolling_bucket_csv_independently_hashed"] is True
    assert audit["checks"]["rolling_bucket_metrics_independently_recomputed"] is False


def test_replay_row_comparison_rejects_fractional_integer_evidence(tmp_path: Path) -> None:
    _write_plan(tmp_path)
    post = tmp_path / "post.csv"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-07-10", periods=4, freq="h", tz="UTC").astype(
                str
            ),
            "baseline_abs_error_eur_mwh": [4.0] * 4,
            "adjusted_abs_error_eur_mwh": [3.0] * 4,
        }
    ).to_csv(post, index=False)
    summary = _write_summary(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    folds = pd.read_csv(Path(payload["outputs"]["rolling_spot_profile_folds_csv"]))
    buckets = pd.read_csv(Path(payload["outputs"]["rolling_spot_bucket_metrics_csv"]))
    forged_folds = folds.copy()
    forged_folds["train_hours"] = forged_folds["train_hours"].astype(float)
    forged_folds.loc[0, "train_hours"] = float(folds.loc[0, "train_hours"]) + 0.9
    forged_buckets = buckets.copy()
    forged_buckets["hours"] = forged_buckets["hours"].astype(float)
    forged_buckets.loc[0, "hours"] = float(buckets.loc[0, "hours"]) + 0.9

    assert audit_module._fold_determinants_match(folds, folds.copy()) is True
    assert audit_module._fold_determinants_match(forged_folds, folds) is False
    assert audit_module._bucket_rows_match(buckets, buckets.copy()) is True
    assert audit_module._bucket_rows_match(forged_buckets, buckets) is False


def test_audit_capture_reuses_the_same_bytes_for_parse_and_hash(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    original = b'{"status":"ORIGINAL"}'
    evidence.write_bytes(original)
    token = audit_module._AUDIT_CAPTURE.set({})
    try:
        assert audit_module._read_json(evidence)["status"] == "ORIGINAL"
        evidence.write_bytes(b'{"status":"MUTATED"}')
        assert audit_module._sha256(evidence) == hashlib.sha256(original).hexdigest()
    finally:
        audit_module._AUDIT_CAPTURE.reset(token)


def test_audit_output_is_append_only(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    post = tmp_path / "post.csv"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-07-10", periods=4, freq="h", tz="UTC").astype(str),
            "baseline_abs_error_eur_mwh": [4.0] * 4,
            "adjusted_abs_error_eur_mwh": [3.0] * 4,
        }
    ).to_csv(post, index=False)
    summary = _write_summary(tmp_path)
    output = tmp_path / "audit.json"
    audit_holdout(plan_json=plan, spot_backtest_summary=summary, output=output)
    before = output.read_bytes()

    with pytest.raises(FileExistsError, match="append-only"):
        audit_holdout(plan_json=plan, spot_backtest_summary=summary, output=output)

    assert output.read_bytes() == before


def test_audit_rejects_forged_residuals_even_when_report_hash_is_updated(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    post = tmp_path / "post.csv"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-07-10", periods=4, freq="h", tz="UTC").astype(str),
            "baseline_abs_error_eur_mwh": [4.0] * 4,
            "adjusted_abs_error_eur_mwh": [3.0] * 4,
        }
    ).to_csv(post, index=False)
    summary = _write_summary(tmp_path)
    forged = pd.read_csv(post)
    forged["adjusted_abs_error_eur_mwh"] = 0.0
    forged.to_csv(post, index=False)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["output_hashes"]["post_valuation_timestamp_residuals_csv"] = _sha256(post)
    summary.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_holdout(plan_json=plan, spot_backtest_summary=summary)

    assert audit["holdout_pass"] is False
    assert audit["checks"]["post_valuation_csv_sha256_bound"] is True
    assert audit["checks"]["holdout_residuals_independently_replayed"] is False


def test_audit_rejects_noncanonical_t057_route_before_reading_summary(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["plan_id"] = "t057_locked_t056_future_holdout"
    plan.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_holdout(
        plan_json=plan,
        spot_backtest_summary=tmp_path / "missing-summary.json",
        output=tmp_path / "audit.json",
    )

    assert audit["status"] == "NO_GO_T057_CANONICAL_ROUTE_MISMATCH"
    assert audit["holdout_pass"] is False
    assert not (tmp_path / "audit.json").exists()


def test_audit_epex_lab_locked_holdout_cli_exits_nonzero_when_failed(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    post = tmp_path / "post.csv"
    pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-07-10", periods=4, freq="h", tz="UTC").astype(str),
            "baseline_abs_error_eur_mwh": [3.0, 3.0, 3.0, 3.0],
            "adjusted_abs_error_eur_mwh": [4.0, 4.0, 4.0, 4.0],
        }
    ).to_csv(post, index=False)
    summary = _write_summary(tmp_path)

    code = main(
        [
            "--plan-json",
            str(plan),
            "--spot-backtest-summary",
            str(summary),
            "--output",
            str(tmp_path / "audit.json"),
        ]
    )

    assert code == 1


def _write_plan(tmp_path: Path) -> Path:
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    baseline.write_text("placeholder\n", encoding="utf-8")
    adjusted.write_text("placeholder\n", encoding="utf-8")
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            _jsonable(
                {
                    "schema_version": "epex_lab_locked_holdout_plan.v1",
                    "activation_status": "locked_future_holdout",
                    "benchmark_policy": "locked_future_no_ompex_holdout",
                    "ompex_used_in_model": False,
                    "ompex_used_in_selection": False,
                    "ompex_used_in_backtest": False,
                    "holdout_start_utc": "2026-07-10T00:00:00Z",
                    "holdout_end_utc": "2026-07-10T04:00:00Z",
                    "baseline_csv": str(baseline),
                    "adjusted_csv": str(adjusted),
                    "backtest": {
                        "valuation_timestamp_utc": "2026-07-09T00:00:00Z",
                        "lookback_years": 2,
                        "eval_days": 1,
                        "embargo_days": 1,
                        "min_eval_hours": 4,
                    },
                    "pass_criteria": {
                        "baseline_csv_sha256": _sha256(baseline),
                        "adjusted_csv_sha256": _sha256(adjusted),
                        "strict_lab_gate_pass": True,
                        "min_holdout_hours": 4,
                        "min_residual_mae_improvement_eur_mwh": 0.0,
                    },
                }
            )
        ),
        encoding="utf-8",
    )
    return path


def _write_summary(
    tmp_path: Path,
    *,
    schema_version: str = "epex_shape_lab_spot_backtest.v3",
) -> Path:
    path = tmp_path / "summary.json"
    post_csv = tmp_path / "post.csv"
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    spot = tmp_path / "spot.parquet"
    raw = tmp_path / "provider-raw.json"
    acquisition = tmp_path / "acquisition-summary.json"
    folds = tmp_path / "folds.csv"
    buckets = tmp_path / "buckets.csv"
    timestamps = pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="h")
    spot_timestamps = pd.date_range("2024-07-01T00:00:00Z", timestamps[-1], freq="h")
    spot_prices = [10.0 + float(timestamp.hour) for timestamp in spot_timestamps]
    holdout_prices = spot_prices[-4:]
    pd.DataFrame({"price_eur_mwh": spot_prices}, index=spot_timestamps).to_parquet(spot)
    reported = pd.read_csv(post_csv)
    _write_candidate_for_reported_errors(
        baseline,
        timestamps=timestamps,
        spot_prices=holdout_prices,
        absolute_errors=reported["baseline_abs_error_eur_mwh"].tolist(),
    )
    _write_candidate_for_reported_errors(
        adjusted,
        timestamps=timestamps,
        spot_prices=holdout_prices,
        absolute_errors=reported["adjusted_abs_error_eur_mwh"].tolist(),
    )
    plan_path = tmp_path / "plan.json"
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_payload["pass_criteria"]["baseline_csv_sha256"] = _sha256(baseline)
    plan_payload["pass_criteria"]["adjusted_csv_sha256"] = _sha256(adjusted)
    plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")
    raw.write_text(
        json.dumps(
            {
                "unix_seconds": [int(timestamp.timestamp()) for timestamp in spot_timestamps],
                "price": spot_prices,
            }
        ),
        encoding="utf-8",
    )
    acquisition.write_text(
        json.dumps(
            {
                "schema_version": "energy_charts_epex_spot_hourly_fetch.v1",
                "status": "OK",
                "source": "energy-charts.info",
                "bzn": "CH",
                "acquired_at_utc": "2026-07-10T05:00:00Z",
                "start_utc": str(spot_timestamps[0]),
                "end_utc": "2026-07-10T04:00:00Z",
                "full_window_covered": True,
                "expected_hour_count": len(spot_timestamps),
                "observed_hour_count": len(spot_timestamps),
                "missing_hour_count": 0,
                "raw_observation_count": len(spot_timestamps),
                "output_parquet_sha256": _sha256(spot),
                "provider_raw_json": str(raw),
                "provider_raw_json_sha256": _sha256(raw),
            }
        ),
        encoding="utf-8",
    )
    loaded_spot = _load_spot(spot)
    profiles = _candidate_profiles(
        _join_candidates(
            _load_candidate(baseline, label="baseline-test"),
            _load_candidate(adjusted, label="adjusted-test"),
        )
    )
    valuation = pd.Timestamp("2026-07-09T00:00:00Z")
    cutoffs = _rolling_cutoffs(
        loaded_spot,
        valuation_utc=valuation,
        rolling_cutoffs=None,
        lookback_years=2,
        eval_days=1,
        embargo_days=1,
        max_auto_folds=12,
    )
    fold_frame = pd.DataFrame(
        [
            _evaluate_fold(
                loaded_spot,
                profiles,
                cutoff=cutoff,
                lookback_years=2,
                eval_days=1,
                embargo_days=1,
                valuation_utc=valuation,
                min_eval_hours=4,
            )
            for cutoff in cutoffs
        ]
    )
    fold_frame.to_csv(folds, index=False)
    bucket_frame = pd.DataFrame(
        [
            row
            for cutoff in cutoffs
            for row in _fold_bucket_rows(
                loaded_spot,
                profiles,
                cutoff=cutoff,
                eval_days=1,
                embargo_days=1,
            )
        ]
    )
    bucket_frame.to_csv(buckets, index=False)
    eligible_folds = fold_frame[fold_frame["fold_eligible"].astype(bool)]
    path.write_text(
        json.dumps(
            _jsonable(
                {
                    "schema_version": schema_version,
                    "status": "DIAGNOSTIC_PASS",
                    "read_only": True,
                    "benchmark_policy": "rolling_origin_epex_spot_no_ompex_lab_only",
                    "promotion_gate": False,
                    "production_approved": False,
                    "independent_production_evidence": False,
                    "ompex_used_in_model": False,
                    "ompex_used_in_selection": False,
                    "ompex_used_in_backtest": False,
                    "strict_lab_gate_pass": True,
                    "rolling_fold_count": len(fold_frame),
                    "eligible_rolling_fold_count": int(fold_frame["fold_eligible"].sum()),
                    "rolling_metrics": _aggregate_fold_metrics(eligible_folds),
                    "rolling_bucket_metrics": _aggregate_bucket_metrics(bucket_frame),
                    "strict_lab_checks": {
                        "rolling_folds_present": True,
                        "rolling_folds_no_temporal_leak": True,
                        "rolling_folds_unique_ordered_cutoffs": True,
                        "rolling_folds_non_overlapping_evaluations": True,
                    },
                    "evaluation_window": {
                        "start_utc": "2026-07-10T00:00:00Z",
                        "end_utc": "2026-07-10T04:00:00Z",
                        "end_exclusive": True,
                        "spot_filtered_before_residual_centering": True,
                        "all_residuals_centered_on_exact_evaluation_overlap": True,
                    },
                    "valuation_timestamp_utc": "2026-07-09T00:00:00+00:00",
                    "source_hashes": {
                        "baseline_csv": _sha256(baseline),
                        "adjusted_csv": _sha256(adjusted),
                        "spot_parquet": _sha256(spot),
                    },
                    "spot_parquet": str(spot),
                    "spot_provenance": {
                        "pass": True,
                        "status": "SPOT_PROVIDER_RAW_BOUND",
                        "acquisition_summary": str(acquisition),
                        "acquisition_summary_sha256": _sha256(acquisition),
                        "spot_parquet": str(spot),
                        "spot_parquet_sha256": _sha256(spot),
                        "provider_raw_json": str(raw),
                        "provider_raw_json_sha256": _sha256(raw),
                    },
                    "outputs": {
                        "post_valuation_timestamp_residuals_csv": str(post_csv),
                        "rolling_spot_profile_folds_csv": str(folds),
                        "rolling_spot_bucket_metrics_csv": str(buckets),
                    },
                    "output_hashes": {
                        "post_valuation_timestamp_residuals_csv": _sha256(post_csv)
                        if post_csv.exists()
                        else None,
                        "rolling_spot_profile_folds_csv": _sha256(folds),
                        "rolling_spot_bucket_metrics_csv": _sha256(buckets),
                    },
                }
            )
        ),
        encoding="utf-8",
    )
    return path


def _write_candidate_for_reported_errors(
    path: Path,
    *,
    timestamps: pd.DatetimeIndex,
    spot_prices: list[float],
    absolute_errors: list[float],
) -> None:
    signs = next(
        candidate
        for candidate in itertools.product((-1.0, 1.0), repeat=len(absolute_errors))
        if abs(sum(sign * error for sign, error in zip(candidate, absolute_errors, strict=True)))
        < 1e-12
    )
    spot = pd.Series(spot_prices, dtype=float)
    spot_residual = spot - spot.mean()
    candidate = spot_residual + pd.Series(
        [sign * error for sign, error in zip(signs, absolute_errors, strict=True)]
    )
    local = timestamps.tz_convert("Europe/Zurich")
    offset = local.strftime("%z")
    frame = pd.DataFrame(
        {
            "timestamp_ch": local.strftime("%d.%m.%Y %H:%M"),
            "utc_offset_ch": "UTC" + offset.str.slice(0, 3) + ":" + offset.str.slice(3, 5),
            "price_weighted_mean_eur_mwh": candidate,
        }
    )
    for column in PRICE_COLUMNS:
        if column not in frame:
            frame[column] = candidate
    frame.to_csv(path, index=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

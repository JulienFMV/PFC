"""Audit a locked future no-OMPEX holdout backtest for an EPEX lab candidate.

The audit validates a completed spot backtest against a pre-registered locked
holdout plan. It reports whether the holdout evidence passes the plan criteria,
but it never approves production promotion.
"""

from __future__ import annotations

import argparse
import contextvars
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from pfc_shaping.path_safety import (
    assert_absolute_path_has_no_links,
    read_stable_single_link_file,
)

try:
    from scripts.backtest_epex_shape_lab_against_spot import (
        PRICE_COLUMNS as _BT_PRICE_COLUMNS,
    )
    from scripts.backtest_epex_shape_lab_against_spot import (
        _candidate_profiles as _bt_candidate_profiles,
    )
    from scripts.backtest_epex_shape_lab_against_spot import (
        _evaluate_fold as _bt_evaluate_fold,
    )
    from scripts.backtest_epex_shape_lab_against_spot import (
        _fold_bucket_rows as _bt_fold_bucket_rows,
    )
    from scripts.backtest_epex_shape_lab_against_spot import (
        _join_candidates as _bt_join_candidates,
    )
    from scripts.backtest_epex_shape_lab_against_spot import (
        _rolling_cutoffs as _bt_rolling_cutoffs,
    )
    from scripts.epex_lab_locked_holdout_policy import (
        AUDIT_SCHEMA_VERSION,
        SPOT_BACKTEST_SCHEMA_VERSION,
        T057_CAPTURE_SCHEMA_VERSION,
        T057_PLAN_ID,
        T057_PLAN_SHA256,
        build_locked_plan_identity,
        canonical_t057_output_path,
        canonical_t057_plan_path,
    )
    from scripts.export_local_test_ch_hourly_csv import _parse_timestamp_ch
    from scripts.fetch_energy_charts_epex_spot_hourly import replay_provider_raw_hourly_bytes
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from backtest_epex_shape_lab_against_spot import (
        PRICE_COLUMNS as _BT_PRICE_COLUMNS,
    )
    from backtest_epex_shape_lab_against_spot import (
        _candidate_profiles as _bt_candidate_profiles,
    )
    from backtest_epex_shape_lab_against_spot import (
        _evaluate_fold as _bt_evaluate_fold,
    )
    from backtest_epex_shape_lab_against_spot import (
        _fold_bucket_rows as _bt_fold_bucket_rows,
    )
    from backtest_epex_shape_lab_against_spot import (
        _join_candidates as _bt_join_candidates,
    )
    from backtest_epex_shape_lab_against_spot import (
        _rolling_cutoffs as _bt_rolling_cutoffs,
    )
    from epex_lab_locked_holdout_policy import (
        AUDIT_SCHEMA_VERSION,
        SPOT_BACKTEST_SCHEMA_VERSION,
        T057_CAPTURE_SCHEMA_VERSION,
        T057_PLAN_ID,
        T057_PLAN_SHA256,
        build_locked_plan_identity,
        canonical_t057_output_path,
        canonical_t057_plan_path,
    )
    from export_local_test_ch_hourly_csv import _parse_timestamp_ch
    from fetch_energy_charts_epex_spot_hourly import (
        replay_provider_raw_hourly_bytes,
    )


PLAN_SCHEMA_VERSION = "epex_lab_locked_holdout_plan.v1"
SUMMARY_POLICY = "rolling_origin_epex_spot_no_ompex_lab_only"
SUMMARY_SCHEMA_VERSION = SPOT_BACKTEST_SCHEMA_VERSION
AUDIT_PRICE = "price_weighted_mean_eur_mwh"
_AUDIT_CAPTURE: contextvars.ContextVar[dict[str, bytes] | None] = contextvars.ContextVar(
    "epex_locked_holdout_audit_capture",
    default=None,
)
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_EVIDENCE_BYTES = 256 * 1024 * 1024


def audit_holdout(
    *,
    plan_json: Path,
    spot_backtest_summary: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    token = _AUDIT_CAPTURE.set({})
    try:
        return _audit_holdout_captured(
            plan_json=plan_json,
            spot_backtest_summary=spot_backtest_summary,
            output=output,
        )
    finally:
        _AUDIT_CAPTURE.reset(token)


def _audit_holdout_captured(
    *,
    plan_json: Path,
    spot_backtest_summary: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    plan_json = _resolved_path(plan_json)
    spot_backtest_summary = _resolved_path(spot_backtest_summary)
    output = _resolved_path(output) if output is not None else None
    plan = _read_json(plan_json)
    if plan.get("plan_id") == T057_PLAN_ID and not _is_canonical_t057_audit_route(
        plan_json=plan_json,
        spot_backtest_summary=spot_backtest_summary,
        output=output,
    ):
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "read_only": True,
            "promotion_gate": False,
            "production_approved": False,
            "approved": False,
            "status": "NO_GO_T057_CANONICAL_ROUTE_MISMATCH",
            "holdout_pass": False,
            "plan_json": str(plan_json),
            "checks": {"t057_canonical_route": False},
        }
    summary = _read_json(spot_backtest_summary)
    criteria = plan.get("pass_criteria") or {}
    post_csv = Path(
        str((summary.get("outputs") or {}).get("post_valuation_timestamp_residuals_csv", ""))
    )
    if not post_csv.is_absolute():
        candidate = spot_backtest_summary.parent / post_csv
        if candidate.exists():
            post_csv = candidate
    post = _load_post_valuation(post_csv)
    baseline_path = _evidence_path(plan.get("baseline_csv"), base=_repo_root())
    adjusted_path = _evidence_path(plan.get("adjusted_csv"), base=_repo_root())
    spot_path = _evidence_path(
        (summary.get("spot_provenance") or {}).get("spot_parquet") or summary.get("spot_parquet"),
        base=spot_backtest_summary.parent,
    )
    replayed_window, residual_replay_matches = _independent_holdout_residual_replay(
        baseline_path=baseline_path,
        adjusted_path=adjusted_path,
        spot_path=spot_path,
        reported_post=post,
        start_utc=str(plan.get("holdout_start_utc")),
        end_utc=str(plan.get("holdout_end_utc")),
    )
    window = replayed_window
    metrics = _window_metrics(replayed_window)
    expected_timestamps = _expected_holdout_timestamps(
        start_utc=str(plan.get("holdout_start_utc")),
        end_utc=str(plan.get("holdout_end_utc")),
    )
    observed_timestamps = (
        pd.DatetimeIndex(window["timestamp_utc"])
        if not window.empty
        else pd.DatetimeIndex([], tz="UTC")
    )
    strict_checks = summary.get("strict_lab_checks") or {}
    evaluation_window = summary.get("evaluation_window") or {}
    spot_provenance = summary.get("spot_provenance") or {}
    spot_evidence_checks = _independent_spot_evidence_checks(
        plan=plan,
        plan_json=plan_json,
        summary=summary,
        spot_provenance=spot_provenance,
        summary_path=spot_backtest_summary,
    )
    rolling_evidence_checks = _independent_rolling_evidence_checks(
        plan=plan,
        summary=summary,
        summary_path=spot_backtest_summary,
        spot_path=spot_path,
        baseline_path=baseline_path,
        adjusted_path=adjusted_path,
    )
    checks = {
        "plan_schema": plan.get("schema_version") == PLAN_SCHEMA_VERSION,
        "plan_activation_locked": plan.get("activation_status") == "locked_future_holdout",
        "plan_no_ompex": plan.get("ompex_used_in_model") is False
        and plan.get("ompex_used_in_selection") is False
        and plan.get("ompex_used_in_backtest") is False,
        "summary_schema": summary.get("schema_version") == SUMMARY_SCHEMA_VERSION,
        "summary_read_only": summary.get("read_only") is True,
        "summary_not_production_evidence": summary.get("independent_production_evidence") is False,
        "summary_policy": summary.get("benchmark_policy") == SUMMARY_POLICY,
        "summary_no_ompex": summary.get("ompex_used_in_model") is False
        and summary.get("ompex_used_in_selection") is False
        and summary.get("ompex_used_in_backtest") is False,
        "summary_lab_only": summary.get("promotion_gate") is False
        and summary.get("production_approved") is False,
        "spot_provider_raw_provenance": bool(
            spot_provenance.get("pass") is True
            and spot_provenance.get("status") == "SPOT_PROVIDER_RAW_BOUND"
        ),
        "summary_status_pass": summary.get("status") == "DIAGNOSTIC_PASS",
        "strict_lab_gate_pass": summary.get("strict_lab_gate_pass") is True
        and criteria.get("strict_lab_gate_pass") is True,
        "rolling_folds_present": int(summary.get("eligible_rolling_fold_count", 0)) > 0
        and strict_checks.get("rolling_folds_present") is True,
        "rolling_folds_no_temporal_leak": strict_checks.get("rolling_folds_no_temporal_leak")
        is True,
        "rolling_folds_unique_ordered_cutoffs": strict_checks.get(
            "rolling_folds_unique_ordered_cutoffs"
        )
        is True,
        "rolling_folds_non_overlapping_evaluations": strict_checks.get(
            "rolling_folds_non_overlapping_evaluations"
        )
        is True,
        "evaluation_window_start_bound": _utc_equal(
            evaluation_window.get("start_utc"), plan.get("holdout_start_utc")
        ),
        "evaluation_window_end_bound": _utc_equal(
            evaluation_window.get("end_utc"), plan.get("holdout_end_utc")
        ),
        "evaluation_window_filtered_before_centering": evaluation_window.get(
            "spot_filtered_before_residual_centering"
        )
        is True,
        "evaluation_all_residuals_same_support": evaluation_window.get(
            "all_residuals_centered_on_exact_evaluation_overlap"
        )
        is True,
        "baseline_csv_sha256_bound": (summary.get("source_hashes") or {}).get("baseline_csv")
        == criteria.get("baseline_csv_sha256"),
        "adjusted_csv_sha256_bound": (summary.get("source_hashes") or {}).get("adjusted_csv")
        == criteria.get("adjusted_csv_sha256"),
        "baseline_csv_sha256_independently_replayed": bool(
            baseline_path is not None
            and baseline_path.is_file()
            and _sha256(baseline_path) == criteria.get("baseline_csv_sha256")
        ),
        "adjusted_csv_sha256_independently_replayed": bool(
            adjusted_path is not None
            and adjusted_path.is_file()
            and _sha256(adjusted_path) == criteria.get("adjusted_csv_sha256")
        ),
        "valuation_timestamp_bound": _utc_text(summary.get("valuation_timestamp_utc"))
        == _utc_text((plan.get("backtest") or {}).get("valuation_timestamp_utc")),
        "post_valuation_csv_present": post_csv.exists(),
        "post_valuation_csv_sha256_bound": post_csv.exists()
        and (summary.get("output_hashes") or {}).get("post_valuation_timestamp_residuals_csv")
        == _sha256(post_csv),
        "post_valuation_contains_only_locked_window": int(len(post)) == int(len(window)),
        "holdout_residuals_independently_replayed": residual_replay_matches,
        "holdout_timestamp_set_complete": bool(
            len(observed_timestamps) == len(expected_timestamps)
            and not observed_timestamps.duplicated().any()
            and observed_timestamps.equals(expected_timestamps)
        ),
        "holdout_window_hours": int(metrics["hours"]) >= int(criteria.get("min_holdout_hours", 1)),
        "holdout_non_degraded": float(
            metrics.get("residual_mae_improvement_eur_mwh")
            if metrics.get("residual_mae_improvement_eur_mwh") is not None
            else float("-inf")
        )
        >= float(criteria.get("min_residual_mae_improvement_eur_mwh", 0.0)),
        **spot_evidence_checks,
        **rolling_evidence_checks,
    }
    status = "LOCKED_HOLDOUT_PASS" if all(checks.values()) else "NO_GO_LOCKED_HOLDOUT_FAIL"
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "approved": False,
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "status": status,
        "holdout_pass": bool(status == "LOCKED_HOLDOUT_PASS"),
        "plan_json": str(plan_json),
        "locked_plan_identity": _build_locked_plan_identity_from_capture(plan, plan_json),
        "spot_backtest_summary": str(spot_backtest_summary),
        "spot_backtest_summary_sha256": _sha256(spot_backtest_summary),
        "post_valuation_timestamp_residuals_csv": str(post_csv),
        "post_valuation_timestamp_residuals_csv_sha256": _sha256(post_csv)
        if post_csv.exists()
        else None,
        "checks": checks,
        "holdout_window": {
            "start_utc": plan.get("holdout_start_utc"),
            "end_utc": plan.get("holdout_end_utc"),
        },
        "holdout_metrics": metrics,
        "criteria": criteria,
        "note": (
            "A passing locked holdout is independent scientific evidence only. "
            "Production still requires the adjusted production/export/selected/"
            "capstone chain."
        ),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"locked holdout audit output is append-only: {output}")
        write_durable_exact_bytes(
            output,
            (json.dumps(_jsonable(audit), indent=2, sort_keys=True) + "\n").encode("utf-8"),
            label="locked holdout audit",
        )
    return audit


def _independent_spot_evidence_checks(
    *,
    plan: dict[str, Any],
    plan_json: Path,
    summary: dict[str, Any],
    spot_provenance: dict[str, Any],
    summary_path: Path,
) -> dict[str, bool]:
    t057_required = plan.get("plan_id") == T057_PLAN_ID
    failed = {
        "spot_parquet_independently_hashed": False,
        "spot_acquisition_summary_independently_hashed": False,
        "spot_acquisition_summary_replayed": False,
        "spot_provider_raw_independently_hashed": False,
        "spot_provider_raw_path_bound": False,
        "spot_provider_raw_semantic_replay": False,
        "spot_acquisition_scope_replayed": False,
        "t057_capture_seal_independently_replayed": not t057_required,
    }
    try:
        spot = _evidence_path(
            spot_provenance.get("spot_parquet") or summary.get("spot_parquet"),
            base=summary_path.parent,
        )
        acquisition = _evidence_path(
            spot_provenance.get("acquisition_summary"),
            base=summary_path.parent,
        )
        provider_raw = _evidence_path(
            spot_provenance.get("provider_raw_json"),
            base=summary_path.parent,
        )
        if spot is None or acquisition is None or provider_raw is None:
            return failed
        if not spot.is_file() or not acquisition.is_file() or not provider_raw.is_file():
            return failed
        spot_sha = _sha256(spot)
        acquisition_sha = _sha256(acquisition)
        provider_raw_sha = _sha256(provider_raw)
        acquisition_payload = _read_json(acquisition)
        acquisition_raw = _evidence_path(
            acquisition_payload.get("provider_raw_json"),
            base=acquisition.parent,
        )
        source_hashes = summary.get("source_hashes") or {}
        replayed, replayed_coverage = replay_provider_raw_hourly_bytes(
            _capture_bytes(provider_raw, label="spot provider raw JSON"),
            start=str(acquisition_payload.get("start_utc")),
            end=str(acquisition_payload.get("end_utc")),
        )
        derived = _load_spot_for_audit(spot)[["price_eur_mwh"]].sort_index()
        replayed = replayed[["price_eur_mwh"]].sort_index()
        pd.testing.assert_frame_equal(
            derived,
            replayed,
            check_exact=True,
            check_names=False,
        )
        raw_path_bound = bool(acquisition_raw is not None and acquisition_raw == provider_raw)
        capture_seal_replayed = not t057_required
        if t057_required:
            capture_evidence = spot_provenance.get("capture_seal")
            if not isinstance(capture_evidence, dict):
                return failed
            capture_seal = _evidence_path(
                capture_evidence.get("capture_seal"),
                base=summary_path.parent,
            )
            if capture_seal is None or not capture_seal.is_file():
                return failed
            seal = _read_json(capture_seal)
            expected_seal_keys = {
                "schema_version",
                "plan_id",
                "plan_json_sha256",
                "bzn",
                "spot_parquet_sha256",
                "spot_fetch_summary_sha256",
                "provider_raw_json_sha256",
                "attempt_seal_sha256",
                "acquired_at_utc",
                "production_authorization",
            }
            attempt_seal = canonical_t057_output_path() / (
                "energy_charts_locked_holdout_attempt_seal.json"
            )
            attempt = _read_json(attempt_seal)
            expected_attempt_keys = {
                "schema_version",
                "plan_id",
                "plan_json_sha256",
                "bzn",
                "output_dir",
                "attempted_at_utc",
                "production_authorization",
            }
            capture_seal_replayed = bool(
                capture_evidence.get("required") is True
                and capture_evidence.get("pass") is True
                and capture_evidence.get("status") == "T057_CAPTURE_SEAL_BOUND"
                and capture_evidence.get("capture_seal_sha256") == _sha256(capture_seal)
                and set(seal) == expected_seal_keys
                and seal.get("schema_version") == T057_CAPTURE_SCHEMA_VERSION
                and seal.get("plan_id") == T057_PLAN_ID
                and seal.get("plan_json_sha256") == _sha256(plan_json)
                and seal.get("bzn") == "CH"
                and seal.get("spot_parquet_sha256") == spot_sha
                and seal.get("spot_fetch_summary_sha256") == acquisition_sha
                and seal.get("provider_raw_json_sha256") == provider_raw_sha
                and attempt_seal.is_file()
                and seal.get("attempt_seal_sha256") == _sha256(attempt_seal)
                and set(attempt) == expected_attempt_keys
                and attempt.get("schema_version") == "energy_charts_epex_locked_holdout_attempt.v1"
                and attempt.get("plan_id") == T057_PLAN_ID
                and attempt.get("plan_json_sha256") == T057_PLAN_SHA256
                and attempt.get("bzn") == "CH"
                and _resolved_path(Path(str(attempt.get("output_dir", ""))))
                == canonical_t057_output_path()
                and isinstance(attempt.get("attempted_at_utc"), str)
                and attempt.get("production_authorization") is False
                and seal.get("acquired_at_utc") == acquisition_payload.get("acquired_at_utc")
                and seal.get("production_authorization") is False
                and capture_seal.parent == acquisition.parent == spot.parent
            )
        return {
            "spot_parquet_independently_hashed": bool(
                source_hashes.get("spot_parquet") == spot_sha
                and spot_provenance.get("spot_parquet_sha256") == spot_sha
                and acquisition_payload.get("output_parquet_sha256") == spot_sha
            ),
            "spot_acquisition_summary_independently_hashed": bool(
                spot_provenance.get("acquisition_summary_sha256") == acquisition_sha
            ),
            "spot_acquisition_summary_replayed": bool(
                acquisition_payload.get("schema_version")
                == "energy_charts_epex_spot_hourly_fetch.v1"
                and acquisition_payload.get("status") == "OK"
                and acquisition_payload.get("full_window_covered") is True
            ),
            "spot_provider_raw_independently_hashed": bool(
                spot_provenance.get("provider_raw_json_sha256") == provider_raw_sha
                and acquisition_payload.get("provider_raw_json_sha256") == provider_raw_sha
            ),
            "spot_provider_raw_path_bound": raw_path_bound,
            "spot_provider_raw_semantic_replay": True,
            "spot_acquisition_scope_replayed": bool(
                acquisition_payload.get("source") == "energy-charts.info"
                and acquisition_payload.get("bzn") == "CH"
                and isinstance(acquisition_payload.get("acquired_at_utc"), str)
                and _utc_text(acquisition_payload.get("acquired_at_utc")) is not None
                and replayed_coverage.get("full_window_covered") is True
                and acquisition_payload.get("expected_hour_count")
                == replayed_coverage.get("expected_hour_count")
                and acquisition_payload.get("observed_hour_count")
                == replayed_coverage.get("observed_hour_count")
                and acquisition_payload.get("missing_hour_count") == 0
                and acquisition_payload.get("raw_observation_count")
                == replayed_coverage.get("raw_observation_count")
            ),
            "t057_capture_seal_independently_replayed": capture_seal_replayed,
        }
    except (
        AssertionError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
    ):
        return failed


def _independent_rolling_evidence_checks(
    *,
    plan: dict[str, Any],
    summary: dict[str, Any],
    summary_path: Path,
    spot_path: Path | None,
    baseline_path: Path | None,
    adjusted_path: Path | None,
) -> dict[str, bool]:
    failed = {
        "rolling_folds_csv_independently_hashed": False,
        "rolling_bucket_csv_independently_hashed": False,
        "rolling_folds_count_independently_replayed": False,
        "rolling_folds_temporal_integrity_independently_replayed": False,
        "rolling_folds_unique_ordered_cutoffs_independently_replayed": False,
        "rolling_folds_non_overlapping_evaluations_independently_replayed": False,
        "rolling_metrics_independently_recomputed": False,
        "rolling_bucket_metrics_independently_recomputed": False,
    }
    try:
        outputs = summary.get("outputs") or {}
        hashes = summary.get("output_hashes") or {}
        folds_path = _evidence_path(
            outputs.get("rolling_spot_profile_folds_csv"),
            base=summary_path.parent,
        )
        buckets_path = _evidence_path(
            outputs.get("rolling_spot_bucket_metrics_csv"),
            base=summary_path.parent,
        )
        if (
            folds_path is None
            or not folds_path.is_file()
            or buckets_path is None
            or not buckets_path.is_file()
            or spot_path is None
            or not spot_path.is_file()
            or baseline_path is None
            or not baseline_path.is_file()
            or adjusted_path is None
            or not adjusted_path.is_file()
        ):
            return failed
        folds = pd.read_csv(io.BytesIO(_capture_bytes(folds_path, label="rolling folds CSV")))
        buckets = pd.read_csv(io.BytesIO(_capture_bytes(buckets_path, label="rolling bucket CSV")))
        required = {
            "cutoff_utc",
            "train_start_utc",
            "train_end_exclusive_utc",
            "eval_start_utc",
            "eval_end_exclusive_utc",
            "valuation_timestamp_utc",
            "train_hours",
            "eval_hours",
            "common_profile_cells",
            "train_eval_common_profile_cells",
            "baseline_profile_mae_eur_mwh",
            "adjusted_profile_mae_eur_mwh",
            "profile_mae_improvement_eur_mwh",
            "baseline_profile_rmse_eur_mwh",
            "adjusted_profile_rmse_eur_mwh",
            "profile_rmse_improvement_eur_mwh",
            "baseline_profile_correlation",
            "adjusted_profile_correlation",
            "train_eval_profile_mae_eur_mwh",
            "train_eval_profile_correlation",
            "fold_eligible",
            "no_temporal_leak",
            "eval_after_current_valuation",
            "historical_fold_not_independent_of_current_candidate_fit",
        }
        if not required.issubset(folds.columns):
            return failed
        cfg = plan.get("backtest") or {}
        lookback_years = int(cfg.get("lookback_years", 2))
        eval_days = int(cfg.get("eval_days", 30))
        embargo_days = int(cfg.get("embargo_days", 1))
        min_eval_hours = int(cfg.get("min_eval_hours", 24))
        max_auto_folds = int(cfg.get("max_auto_folds", 12))
        valuation = _utc_timestamp(cfg.get("valuation_timestamp_utc"))
        spot = _load_spot_for_audit(spot_path)
        candidate_profiles = _bt_candidate_profiles(
            _bt_join_candidates(
                _load_candidate_for_rolling_replay(baseline_path, label="baseline-audit"),
                _load_candidate_for_rolling_replay(adjusted_path, label="adjusted-audit"),
            )
        )
        expected_cutoffs = _bt_rolling_cutoffs(
            spot,
            valuation_utc=valuation,
            rolling_cutoffs=None,
            lookback_years=lookback_years,
            eval_days=eval_days,
            embargo_days=embargo_days,
            max_auto_folds=max_auto_folds,
        )
        expected_rows = pd.DataFrame(
            [
                _bt_evaluate_fold(
                    spot,
                    candidate_profiles,
                    cutoff=cutoff,
                    lookback_years=lookback_years,
                    eval_days=eval_days,
                    embargo_days=embargo_days,
                    valuation_utc=valuation,
                    min_eval_hours=min_eval_hours,
                )
                for cutoff in expected_cutoffs
            ]
        )
        expected_buckets = pd.DataFrame(
            [
                row
                for cutoff in expected_cutoffs
                for row in _bt_fold_bucket_rows(
                    spot,
                    candidate_profiles,
                    cutoff=cutoff,
                    eval_days=eval_days,
                    embargo_days=embargo_days,
                )
            ]
        )
        eligible = _bool_series(folds["fold_eligible"])
        no_leak = _bool_series(folds["no_temporal_leak"])
        actual_cutoffs = pd.DatetimeIndex([_utc_timestamp(value) for value in folds["cutoff_utc"]])
        actual_eval_starts = pd.DatetimeIndex(
            [_utc_timestamp(value) for value in folds["eval_start_utc"]]
        )
        actual_eval_ends = pd.DatetimeIndex(
            [_utc_timestamp(value) for value in folds["eval_end_exclusive_utc"]]
        )
        expected_cutoff_index = pd.DatetimeIndex(expected_cutoffs)
        unique_ordered_cutoffs = bool(
            len(actual_cutoffs) > 0
            and not actual_cutoffs.duplicated().any()
            and actual_cutoffs.is_monotonic_increasing
            and not expected_cutoff_index.duplicated().any()
            and expected_cutoff_index.is_monotonic_increasing
        )
        non_overlapping_evaluations = bool(
            len(actual_eval_starts) > 0
            and len(actual_eval_starts) == len(actual_eval_ends)
            and actual_eval_starts.is_monotonic_increasing
            and all(
                actual_eval_starts[index] >= actual_eval_ends[index - 1]
                for index in range(1, len(actual_eval_starts))
            )
        )
        replay_matches = _fold_determinants_match(folds, expected_rows)
        bucket_replay_matches = _bucket_rows_match(buckets, expected_buckets)
        expected_eligible = (
            _bool_series(expected_rows["fold_eligible"])
            if not expected_rows.empty
            else pd.Series(dtype=bool)
        )
        recomputed_rolling = _audit_aggregate_folds(expected_rows[expected_eligible])
        recomputed_buckets = _audit_aggregate_buckets(expected_buckets)
        return {
            "rolling_folds_csv_independently_hashed": bool(
                hashes.get("rolling_spot_profile_folds_csv") == _sha256(folds_path)
            ),
            "rolling_bucket_csv_independently_hashed": bool(
                hashes.get("rolling_spot_bucket_metrics_csv") == _sha256(buckets_path)
            ),
            "rolling_folds_count_independently_replayed": bool(
                len(folds) == int(summary.get("rolling_fold_count", -1))
                and len(folds) == len(expected_rows)
                and int(eligible.sum()) == int(summary.get("eligible_rolling_fold_count", -1))
                and int(eligible.sum()) == int(expected_eligible.sum())
                and int(eligible.sum()) > 0
            ),
            "rolling_folds_temporal_integrity_independently_replayed": bool(
                len(folds) > 0 and no_leak.all() and replay_matches
            ),
            "rolling_folds_unique_ordered_cutoffs_independently_replayed": bool(
                unique_ordered_cutoffs and replay_matches
            ),
            "rolling_folds_non_overlapping_evaluations_independently_replayed": bool(
                non_overlapping_evaluations and replay_matches
            ),
            "rolling_metrics_independently_recomputed": _aggregate_mapping_matches(
                summary.get("rolling_metrics"),
                recomputed_rolling,
            )
            and replay_matches,
            "rolling_bucket_metrics_independently_recomputed": bool(
                bucket_replay_matches
                and _aggregate_mapping_matches(
                    summary.get("rolling_bucket_metrics"),
                    recomputed_buckets,
                )
            ),
        }
    except (OSError, ValueError, TypeError, KeyError):
        return failed


def _audit_aggregate_folds(folds: pd.DataFrame) -> dict[str, Any]:
    if folds.empty:
        return {}
    numeric = folds.copy()
    improvement = pd.to_numeric(numeric["profile_mae_improvement_eur_mwh"], errors="raise")
    return {
        "mean_baseline_profile_mae_eur_mwh": _finite_mean(numeric["baseline_profile_mae_eur_mwh"]),
        "mean_adjusted_profile_mae_eur_mwh": _finite_mean(numeric["adjusted_profile_mae_eur_mwh"]),
        "mean_profile_mae_improvement_eur_mwh": _finite_mean(improvement),
        "median_profile_mae_improvement_eur_mwh": _finite_median(improvement),
        "positive_mae_improvement_fold_count": int((improvement > 0.0).sum()),
        "mean_baseline_profile_correlation": _finite_mean(numeric["baseline_profile_correlation"]),
        "mean_adjusted_profile_correlation": _finite_mean(numeric["adjusted_profile_correlation"]),
        "mean_train_eval_profile_mae_eur_mwh": _finite_mean(
            numeric["train_eval_profile_mae_eur_mwh"]
        ),
        "historical_fold_not_independent_count": int(
            _bool_series(numeric["historical_fold_not_independent_of_current_candidate_fit"]).sum()
        ),
        "post_valuation_fold_count": int(
            _bool_series(numeric["eval_after_current_valuation"]).sum()
        ),
    }


def _audit_aggregate_buckets(buckets: pd.DataFrame) -> dict[str, dict[str, Any]]:
    required = {
        "bucket",
        "metric_type",
        "hours",
        "baseline_mae_eur_mwh",
        "adjusted_mae_eur_mwh",
        "mae_improvement_eur_mwh",
        "baseline_rmse_eur_mwh",
        "adjusted_rmse_eur_mwh",
        "rmse_improvement_eur_mwh",
        "adjusted_error_p95_eur_mwh",
    }
    if buckets.empty or not required.issubset(buckets.columns):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for (bucket, metric_type), group in buckets.groupby(["bucket", "metric_type"], sort=True):
        improvement = pd.to_numeric(group["mae_improvement_eur_mwh"], errors="raise")
        result[f"{metric_type}:{bucket}"] = {
            "fold_count": int(len(group)),
            "total_hours": int(pd.to_numeric(group["hours"], errors="raise").sum()),
            "mean_baseline_mae_eur_mwh": _finite_mean(group["baseline_mae_eur_mwh"]),
            "mean_adjusted_mae_eur_mwh": _finite_mean(group["adjusted_mae_eur_mwh"]),
            "mean_mae_improvement_eur_mwh": _finite_mean(improvement),
            "positive_mae_improvement_fold_count": int((improvement > 0.0).sum()),
            "mean_baseline_rmse_eur_mwh": _finite_mean(group["baseline_rmse_eur_mwh"]),
            "mean_adjusted_rmse_eur_mwh": _finite_mean(group["adjusted_rmse_eur_mwh"]),
            "mean_rmse_improvement_eur_mwh": _finite_mean(group["rmse_improvement_eur_mwh"]),
            "mean_adjusted_error_p95_eur_mwh": _finite_mean(group["adjusted_error_p95_eur_mwh"]),
        }
    return result


def _aggregate_mapping_matches(actual: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return False
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, dict):
            if not _aggregate_mapping_matches(actual_value, expected_value):
                return False
        elif expected_value is None:
            if actual_value is not None:
                return False
        elif isinstance(expected_value, int):
            if isinstance(actual_value, bool) or not isinstance(actual_value, int):
                return False
            if actual_value != expected_value:
                return False
        else:
            try:
                if (
                    not pd.notna(actual_value)
                    or abs(float(actual_value) - float(expected_value)) > 1e-12
                ):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def _finite_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    finite = numeric[numeric.map(math.isfinite)]
    return float(finite.mean()) if not finite.empty else None


def _finite_median(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    finite = numeric[numeric.map(math.isfinite)]
    return float(finite.median()) if not finite.empty else None


def _fold_determinants_match(actual: pd.DataFrame, expected: pd.DataFrame) -> bool:
    if len(actual) != len(expected) or expected.empty:
        return False
    timestamp_columns = [
        "cutoff_utc",
        "train_start_utc",
        "train_end_exclusive_utc",
        "eval_start_utc",
        "eval_end_exclusive_utc",
        "valuation_timestamp_utc",
    ]
    integer_columns = [
        "train_hours",
        "eval_hours",
        "common_profile_cells",
        "train_eval_common_profile_cells",
    ]
    numeric_columns = [
        "baseline_profile_mae_eur_mwh",
        "adjusted_profile_mae_eur_mwh",
        "profile_mae_improvement_eur_mwh",
        "baseline_profile_rmse_eur_mwh",
        "adjusted_profile_rmse_eur_mwh",
        "profile_rmse_improvement_eur_mwh",
        "baseline_profile_correlation",
        "adjusted_profile_correlation",
        "train_eval_profile_mae_eur_mwh",
        "train_eval_profile_correlation",
    ]
    boolean_columns = [
        "no_temporal_leak",
        "fold_eligible",
        "eval_after_current_valuation",
        "historical_fold_not_independent_of_current_candidate_fit",
    ]
    try:
        for column in timestamp_columns:
            if [_utc_text(value) for value in actual[column]] != [
                _utc_text(value) for value in expected[column]
            ]:
                return False
        for column in integer_columns:
            if _exact_integer_values(actual[column]) != _exact_integer_values(expected[column]):
                return False
        for column in boolean_columns:
            if _bool_series(actual[column]).tolist() != _bool_series(expected[column]).tolist():
                return False
        for column in numeric_columns:
            pd.testing.assert_series_equal(
                pd.to_numeric(actual[column], errors="raise").reset_index(drop=True),
                pd.to_numeric(expected[column], errors="raise").reset_index(drop=True),
                check_names=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        return bool(
            all(
                _utc_timestamp(value) <= _utc_timestamp(expected.iloc[0]["valuation_timestamp_utc"])
                for value in actual["eval_end_exclusive_utc"]
            )
        )
    except (AssertionError, KeyError, TypeError, ValueError):
        return False


def _bucket_rows_match(actual: pd.DataFrame, expected: pd.DataFrame) -> bool:
    if len(actual) != len(expected) or expected.empty:
        return False
    key_columns = ["cutoff_utc", "bucket", "metric_type"]
    integer_columns = ["hours"]
    numeric_columns = [
        "baseline_mae_eur_mwh",
        "adjusted_mae_eur_mwh",
        "mae_improvement_eur_mwh",
        "baseline_rmse_eur_mwh",
        "adjusted_rmse_eur_mwh",
        "rmse_improvement_eur_mwh",
        "baseline_error_p95_eur_mwh",
        "adjusted_error_p95_eur_mwh",
    ]
    required = set([*key_columns, *integer_columns, *numeric_columns])
    if not required.issubset(actual.columns) or not required.issubset(expected.columns):
        return False
    try:
        actual_ordered = actual.sort_values(key_columns, kind="stable").reset_index(drop=True)
        expected_ordered = expected.sort_values(key_columns, kind="stable").reset_index(drop=True)
        if [_utc_text(value) for value in actual_ordered["cutoff_utc"]] != [
            _utc_text(value) for value in expected_ordered["cutoff_utc"]
        ]:
            return False
        for column in ["bucket", "metric_type"]:
            if actual_ordered[column].astype(str).tolist() != expected_ordered[column].astype(
                str
            ).tolist():
                return False
        for column in integer_columns:
            if _exact_integer_values(actual_ordered[column]) != _exact_integer_values(
                expected_ordered[column]
            ):
                return False
        for column in numeric_columns:
            pd.testing.assert_series_equal(
                pd.to_numeric(actual_ordered[column], errors="raise"),
                pd.to_numeric(expected_ordered[column], errors="raise"),
                check_names=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        return True
    except (AssertionError, KeyError, TypeError, ValueError):
        return False


def _exact_integer_values(series: pd.Series) -> list[int]:
    numeric = pd.to_numeric(series, errors="raise").astype(float)
    if not numeric.map(math.isfinite).all() or not (numeric == numeric.round()).all():
        raise ValueError("integer evidence column contains a fractional or non-finite value")
    return numeric.astype("int64").tolist()


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError("boolean evidence column is invalid")
    return normalized == "true"


def _evidence_path(value: Any, *, base: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return assert_absolute_path_has_no_links(Path(path.absolute()))


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _is_canonical_t057_audit_route(
    *,
    plan_json: Path,
    spot_backtest_summary: Path,
    output: Path | None,
) -> bool:
    canonical_runner = canonical_t057_output_path() / "locked_holdout_runner"
    return bool(
        plan_json == canonical_t057_plan_path()
        and _sha256(plan_json) == T057_PLAN_SHA256
        and spot_backtest_summary == canonical_runner / "spot_backtest_summary.json"
        and output == canonical_runner / "locked_holdout_audit.json"
    )


def _load_spot_for_audit(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(io.BytesIO(_capture_bytes(path, label="spot parquet")))
    if "price_eur_mwh" not in frame.columns or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("spot parquet audit schema invalid")
    index = frame.index
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    out = pd.DataFrame(
        {"price_eur_mwh": pd.to_numeric(frame["price_eur_mwh"], errors="raise").to_numpy()},
        index=index,
    ).sort_index()
    if out.index.has_duplicates or not out["price_eur_mwh"].notna().all():
        raise ValueError("spot parquet audit values invalid")
    return out


def _load_candidate_price_for_audit(path: Path, *, label: str) -> pd.Series:
    frame = pd.read_csv(io.BytesIO(_capture_bytes(path, label=f"{label} candidate CSV")))
    required = {"timestamp_ch", AUDIT_PRICE}
    if not required.issubset(frame.columns):
        raise ValueError(f"{label} candidate audit schema invalid")
    timestamps = _parse_timestamp_ch(frame["timestamp_ch"], frame.get("utc_offset_ch"))
    index = pd.DatetimeIndex(timestamps).tz_convert("UTC")
    values = pd.to_numeric(frame[AUDIT_PRICE], errors="raise").to_numpy(dtype=float)
    series = pd.Series(values, index=index, name=label).sort_index()
    if series.index.has_duplicates or not series.notna().all():
        raise ValueError(f"{label} candidate audit values invalid")
    return series


def _load_candidate_for_rolling_replay(path: Path, *, label: str) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(_capture_bytes(path, label=f"{label} candidate CSV")))
    missing = sorted(set(["timestamp_ch", *_BT_PRICE_COLUMNS]) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} candidate missing required columns: {missing}")
    timestamps = _parse_timestamp_ch(frame["timestamp_ch"], frame.get("utc_offset_ch"))
    out = frame.copy()
    for column in _BT_PRICE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="raise")
    out.index = pd.DatetimeIndex(timestamps)
    if out.index.has_duplicates:
        raise ValueError(f"{label} candidate has duplicate timestamps")
    out = out.sort_index()
    out.index.name = "timestamp_ch"
    return out[_BT_PRICE_COLUMNS]


def _independent_holdout_residual_replay(
    *,
    baseline_path: Path | None,
    adjusted_path: Path | None,
    spot_path: Path | None,
    reported_post: pd.DataFrame,
    start_utc: str,
    end_utc: str,
) -> tuple[pd.DataFrame, bool]:
    empty = pd.DataFrame()
    try:
        if baseline_path is None or adjusted_path is None or spot_path is None:
            return empty, False
        baseline = _load_candidate_price_for_audit(baseline_path, label="baseline")
        adjusted = _load_candidate_price_for_audit(adjusted_path, label="adjusted")
        spot = _load_spot_for_audit(spot_path)["price_eur_mwh"]
        start = _utc_timestamp(start_utc)
        end = _utc_timestamp(end_utc)
        baseline = baseline[(baseline.index >= start) & (baseline.index < end)]
        adjusted = adjusted[(adjusted.index >= start) & (adjusted.index < end)]
        spot = spot[(spot.index >= start) & (spot.index < end)]
        joined = pd.concat([baseline, adjusted, spot], axis=1, join="inner").sort_index()
        expected = pd.date_range(start, end, freq="h", inclusive="left")
        if not joined.index.equals(expected):
            return empty, False
        local_month = joined.index.tz_convert("Europe/Zurich").strftime("%Y-%m")
        residuals = joined.copy()
        for column in ("baseline", "adjusted", "price_eur_mwh"):
            residuals[column] = joined[column] - joined[column].groupby(local_month).transform(
                "mean"
            )
        replayed = pd.DataFrame(
            {
                "timestamp_utc": joined.index,
                "baseline_abs_error_eur_mwh": (residuals["baseline"] - residuals["price_eur_mwh"])
                .abs()
                .to_numpy(),
                "adjusted_abs_error_eur_mwh": (residuals["adjusted"] - residuals["price_eur_mwh"])
                .abs()
                .to_numpy(),
            }
        )
        required = set(replayed.columns)
        if not required.issubset(reported_post.columns):
            return replayed, False
        reported = reported_post[list(replayed.columns)].copy()
        reported["timestamp_utc"] = pd.to_datetime(reported["timestamp_utc"], utc=True)
        reported = reported.sort_values("timestamp_utc").reset_index(drop=True)
        replayed = replayed.sort_values("timestamp_utc").reset_index(drop=True)
        timestamps_match = reported["timestamp_utc"].equals(replayed["timestamp_utc"])
        values_match = all(
            pd.Series(reported[column], dtype=float).sub(replayed[column]).abs().le(1e-10).all()
            for column in ("baseline_abs_error_eur_mwh", "adjusted_abs_error_eur_mwh")
        )
        return replayed, bool(timestamps_match and values_match)
    except (OSError, ValueError, TypeError, KeyError):
        return empty, False


def _load_post_valuation(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(io.BytesIO(_capture_bytes(path, label="post-valuation CSV")))
    if "timestamp_utc" not in frame.columns:
        raise ValueError(f"post valuation CSV missing timestamp_utc: {path}")
    out = frame.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    return out


def _holdout_window(frame: pd.DataFrame, *, start_utc: str, end_utc: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    start = pd.Timestamp(start_utc)
    end = pd.Timestamp(end_utc)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    mask = (frame["timestamp_utc"] >= start) & (frame["timestamp_utc"] < end)
    return frame.loc[mask].copy()


def _expected_holdout_timestamps(*, start_utc: str, end_utc: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_utc)
    end = pd.Timestamp(end_utc)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    return pd.date_range(start, end, freq="h", inclusive="left")


def _window_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"hours": 0, "residual_mae_improvement_eur_mwh": None}
    baseline = pd.to_numeric(frame["baseline_abs_error_eur_mwh"], errors="raise")
    adjusted = pd.to_numeric(frame["adjusted_abs_error_eur_mwh"], errors="raise")
    return {
        "hours": int(len(frame)),
        "baseline_residual_mae_eur_mwh": float(baseline.mean()),
        "adjusted_residual_mae_eur_mwh": float(adjusted.mean()),
        "residual_mae_improvement_eur_mwh": float(baseline.mean() - adjusted.mean()),
    }


def _utc_text(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _utc_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    try:
        return _utc_text(left) == _utc_text(right)
    except (TypeError, ValueError):
        return False


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        _capture_bytes(path, label=f"JSON evidence {path.name}").decode("utf-8"),
        object_pairs_hook=_unique_json_mapping,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence must be an object: {path}")
    return payload


def _unique_json_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _resolved_path(path: Path) -> Path:
    selected = path.expanduser()
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    return assert_absolute_path_has_no_links(Path(selected.absolute()))


def _sha256(path: Path) -> str:
    return hashlib.sha256(_capture_bytes(path, label=f"SHA-256 evidence {path.name}")).hexdigest()


def _capture_bytes(
    path: Path,
    *,
    label: str,
    max_bytes: int = _MAX_EVIDENCE_BYTES,
) -> bytes:
    selected = _resolved_path(path)
    context = _AUDIT_CAPTURE.get()
    key = str(selected).casefold()
    if context is not None and key in context:
        return context[key]
    payload = read_stable_single_link_file(selected, label=label, max_bytes=max_bytes)
    if context is not None:
        context[key] = payload
    return payload


def _build_locked_plan_identity_from_capture(
    plan: dict[str, Any],
    plan_json: Path,
) -> dict[str, Any]:
    identity = build_locked_plan_identity(plan)
    identity["plan_json"] = str(plan_json)
    identity["plan_json_sha256"] = _sha256(plan_json)
    return identity


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--spot-backtest-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = audit_holdout(
        plan_json=args.plan_json,
        spot_backtest_summary=args.spot_backtest_summary,
        output=args.output,
    )
    print(json.dumps(_jsonable(audit), indent=2, sort_keys=True))
    return 0 if audit.get("holdout_pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())

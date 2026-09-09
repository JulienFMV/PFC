from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.build_t057_fold_integrity_correction as correction_module
from scripts.build_t057_fold_integrity_correction import build_correction


def test_t057_correction_is_canonically_bound_recomputed_and_append_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = {
        "hours": 336,
        "baseline_residual_mae_eur_mwh": 18.2,
        "adjusted_residual_mae_eur_mwh": 18.0,
        "residual_mae_improvement_eur_mwh": 0.2,
    }
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    spot = tmp_path / "spot.parquet"
    inner_run = tmp_path / "locked_holdout_run_summary.json"
    baseline.write_bytes(b"baseline-candidate")
    adjusted.write_bytes(b"adjusted-candidate")
    spot.write_bytes(b"spot-parquet")
    inner_run.write_bytes(b"locked-inner-run")
    plan = tmp_path / "plan.json"
    _write_json(
        plan,
        {
            "schema_version": "epex_lab_locked_holdout_plan.v1",
            "plan_id": "t057_locked_t056_future_holdout",
            "baseline_csv": str(baseline.resolve()),
            "baseline_csv_sha256": _sha256(baseline),
            "adjusted_csv": str(adjusted.resolve()),
            "adjusted_csv_sha256": _sha256(adjusted),
            "holdout_start_utc": "2026-07-10T00:00:00Z",
            "holdout_end_utc": "2026-07-24T00:00:00Z",
            "backtest": {
                "valuation_timestamp_utc": "2026-07-09T00:00:00Z",
                "lookback_years": 2,
                "eval_days": 14,
                "embargo_days": 1,
            },
        },
    )
    plan_sha = _sha256(plan)
    legacy_folds = tmp_path / "legacy-folds.csv"
    legacy_folds.write_text(
        _fold_csv(
            [
                (
                    "2026-06-24T00:00:00Z",
                    "2026-06-25T00:00:00Z",
                    "2026-07-09T00:00:00Z",
                )
            ]
            * 12
        ),
        encoding="utf-8",
    )
    legacy_spot = tmp_path / "legacy-spot.json"
    _write_json(
        legacy_spot,
        {
            "source_hashes": {
                "baseline_csv": _sha256(baseline),
                "adjusted_csv": _sha256(adjusted),
                "spot_parquet": _sha256(spot),
            },
            "outputs": {"rolling_spot_profile_folds_csv": str(legacy_folds.resolve())},
            "output_hashes": {"rolling_spot_profile_folds_csv": _sha256(legacy_folds)},
        },
    )
    original = tmp_path / "original.json"
    _write_json(
        original,
        {
            "schema_version": "energy_charts_epex_locked_holdout_run.v1",
            "production_approved": False,
            "actual_plan_json_sha256": plan_sha,
            "expected_plan_json_sha256": plan_sha,
            "plan_json": str(plan.resolve()),
            "locked_holdout_run_summary_sha256": _sha256(inner_run),
            "locked_holdout": {
                "actual_plan_json_sha256": plan_sha,
                "expected_plan_json_sha256": plan_sha,
                "locked_plan_identity": {
                    "plan_id": "t057_locked_t056_future_holdout",
                    "plan_json": str(plan.resolve()),
                    "plan_json_sha256": plan_sha,
                    "baseline_csv": str(baseline.resolve()),
                    "baseline_csv_sha256": _sha256(baseline),
                    "adjusted_csv": str(adjusted.resolve()),
                    "adjusted_csv_sha256": _sha256(adjusted),
                },
                "spot_backtest_summary": str(legacy_spot.resolve()),
                "spot_backtest_summary_sha256": _sha256(legacy_spot),
                "spot_parquet": str(spot.resolve()),
                "spot_provenance": {"spot_parquet_sha256": _sha256(spot)},
                "holdout_metrics": metrics,
            },
        },
    )
    quality = tmp_path / "quality.json"
    _write_json(
        quality,
        {
            "schema_version": "pfc_local_quality_report.v1",
            "status": "LOCAL_DIAGNOSTIC_PASS_NOT_PRODUCTION",
            "source": {
                "run_id": "t057_locked_t056_future_holdout",
                "plan_sha256": plan_sha,
                "run_summary_sha256": _sha256(original),
                "evidence": {
                    "spot backtest summary": {"sha256": _sha256(legacy_spot)},
                    "locked holdout run summary": {"sha256": _sha256(inner_run)},
                    "locked plan": {"sha256": plan_sha},
                    "baseline candidate": {"sha256": _sha256(baseline)},
                    "adjusted candidate": {"sha256": _sha256(adjusted)},
                    "spot parquet": {"sha256": _sha256(spot)},
                },
            },
        },
    )
    forensic_folds = tmp_path / "forensic-folds.csv"
    forensic_folds.write_text(
        _forensic_fold_csv(),
        encoding="utf-8",
    )
    post = tmp_path / "post.csv"
    post.write_text(
        "timestamp_utc,baseline_abs_error_eur_mwh,adjusted_abs_error_eur_mwh\n"
        + "".join(
            f"{timestamp.isoformat()},18.2,18.0\n"
            for timestamp in pd.date_range(
                "2026-07-10T00:00:00Z",
                periods=336,
                freq="h",
            )
        ),
        encoding="utf-8",
    )
    rolling_metrics = {
        "mean_baseline_profile_mae_eur_mwh": 27.5,
        "mean_adjusted_profile_mae_eur_mwh": 26.9,
        "mean_profile_mae_improvement_eur_mwh": 0.6,
        "median_profile_mae_improvement_eur_mwh": 0.6,
        "positive_mae_improvement_fold_count": 1,
        "mean_baseline_profile_correlation": 0.91,
        "mean_adjusted_profile_correlation": 0.92,
        "mean_train_eval_profile_mae_eur_mwh": 18.0,
        "historical_fold_not_independent_count": 1,
        "post_valuation_fold_count": 0,
    }
    forensic = tmp_path / "forensic.json"
    _write_json(
        forensic,
        {
            "schema_version": "epex_shape_lab_spot_backtest.v2",
            "production_approved": False,
            "baseline_csv": str(baseline.resolve()),
            "adjusted_csv": str(adjusted.resolve()),
            "spot_parquet": str(spot.resolve()),
            "source_hashes": {
                "baseline_csv": _sha256(baseline),
                "adjusted_csv": _sha256(adjusted),
                "spot_parquet": _sha256(spot),
            },
            "valuation_timestamp_utc": "2026-07-09T00:00:00Z",
            "lookback_years": 2,
            "eval_days": 14,
            "embargo_days": 1,
            "rolling_fold_count": 1,
            "eligible_rolling_fold_count": 1,
            "evaluation_window": {
                "start_utc": "2026-07-10T00:00:00Z",
                "end_utc": "2026-07-24T00:00:00Z",
                "end_exclusive": True,
            },
            "post_valuation_metrics": metrics,
            "rolling_metrics": rolling_metrics,
            "outputs": {
                "rolling_spot_profile_folds_csv": str(forensic_folds.resolve()),
                "post_valuation_timestamp_residuals_csv": str(post.resolve()),
            },
            "output_hashes": {
                "rolling_spot_profile_folds_csv": _sha256(forensic_folds),
                "post_valuation_timestamp_residuals_csv": _sha256(post),
            },
        },
    )
    monkeypatch.setattr(
        correction_module,
        "_canonical_source_contract",
        lambda: {
            "root": tmp_path,
            "original_run": original,
            "original_run_sha256": _sha256(original),
            "superseded_quality": quality,
            "superseded_quality_sha256": _sha256(quality),
            "forensic_summary": forensic,
            "forensic_summary_sha256": _sha256(forensic),
            "plan": plan,
            "plan_sha256": plan_sha,
        },
    )
    output = tmp_path / "correction" / "correction.json"

    result = build_correction(
        original_run_summary=original,
        expected_original_run_summary_sha256=_sha256(original),
        superseded_quality_report=quality,
        expected_superseded_quality_report_sha256=_sha256(quality),
        forensic_backtest_summary=forensic,
        expected_forensic_backtest_summary_sha256=_sha256(forensic),
        output=output,
    )

    assert result["status"] == "T057_HISTORICAL_ROLLING_EVIDENCE_REVOKED"
    assert result["schema_version"] == "t057_fold_integrity_correction.v2"
    assert result["source_contract"]["canonical_sources_only"] is True
    assert result["source_contract"]["metrics_recomputed_from_csv"] is True
    assert result["legacy_evidence"]["fold_integrity"]["duplicate_cutoff_count"] == 11
    assert result["forensic_replay"]["fold_integrity"]["row_count"] == 1
    assert result["retained_future_holdout"]["independent_episode_count"] == 1
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="append-only"):
        build_correction(
            original_run_summary=original,
            expected_original_run_summary_sha256=_sha256(original),
            superseded_quality_report=quality,
            expected_superseded_quality_report_sha256=_sha256(quality),
            forensic_backtest_summary=forensic,
            expected_forensic_backtest_summary_sha256=_sha256(forensic),
            output=output,
        )
    assert output.read_bytes() == before


def test_t057_correction_rejects_wrong_caller_anchor(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    _write_json(source, {})

    with pytest.raises(ValueError, match="canonical registered source"):
        build_correction(
            original_run_summary=source,
            expected_original_run_summary_sha256="0" * 64,
            superseded_quality_report=source,
            expected_superseded_quality_report_sha256=_sha256(source),
            forensic_backtest_summary=source,
            expected_forensic_backtest_summary_sha256=_sha256(source),
            output=tmp_path / "correction.json",
        )


def _fold_csv(rows: list[tuple[str, str, str]]) -> str:
    return "cutoff_utc,eval_start_utc,eval_end_exclusive_utc\n" + "".join(
        f"{cutoff},{start},{end}\n" for cutoff, start, end in rows
    )


def _forensic_fold_csv() -> str:
    return (
        "cutoff_utc,train_start_utc,train_end_exclusive_utc,eval_start_utc,"
        "eval_end_exclusive_utc,valuation_timestamp_utc,train_hours,eval_hours,"
        "common_profile_cells,train_eval_common_profile_cells,no_temporal_leak,"
        "eval_after_current_valuation,historical_fold_not_independent_of_current_candidate_fit,"
        "baseline_profile_mae_eur_mwh,adjusted_profile_mae_eur_mwh,"
        "profile_mae_improvement_eur_mwh,baseline_profile_rmse_eur_mwh,"
        "adjusted_profile_rmse_eur_mwh,profile_rmse_improvement_eur_mwh,"
        "baseline_profile_correlation,adjusted_profile_correlation,"
        "train_eval_profile_mae_eur_mwh,train_eval_profile_correlation,fold_eligible\n"
        "2026-06-24T00:00:00Z,2024-06-24T00:00:00Z,2026-06-24T00:00:00Z,"
        "2026-06-25T00:00:00Z,2026-07-09T00:00:00Z,2026-07-09T00:00:00Z,"
        "17520,336,96,96,True,False,True,27.5,26.9,0.6,34.0,33.0,1.0,"
        "0.91,0.92,18.0,0.90,True\n"
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

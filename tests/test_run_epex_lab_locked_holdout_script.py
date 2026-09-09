from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.check_epex_lab_locked_holdout_coverage import CANDIDATE_PRICE_COLUMNS
from scripts.run_epex_lab_locked_holdout import main, run_locked_holdout


def test_run_epex_lab_locked_holdout_waits_when_coverage_missing(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=2, freq="h"),
    ).to_parquet(spot)

    summary = run_locked_holdout(
        plan_json=plan,
        spot_parquet=spot,
        output_dir=tmp_path / "out",
        expected_plan_sha256=_sha256(plan),
    )

    assert summary["status"] == "WAITING_FOR_FULL_SPOT_COVERAGE"
    assert summary["coverage_ready"] is False
    assert summary["backtest_ran"] is False
    assert summary["audit_ran"] is False
    assert (tmp_path / "out" / "coverage_status.json").exists()
    assert (tmp_path / "out" / "locked_holdout_run_summary.json").exists()
    assert not (tmp_path / "out" / "spot_backtest_summary.json").exists()


def test_run_epex_lab_locked_holdout_rejects_existing_output_before_any_write(
    tmp_path: Path,
) -> None:
    plan = _write_plan(tmp_path)
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=2, freq="h"),
    ).to_parquet(spot)
    output = tmp_path / "out"
    run_locked_holdout(
        plan_json=plan,
        spot_parquet=spot,
        output_dir=output,
        expected_plan_sha256=_sha256(plan),
    )
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(FileExistsError):
        run_locked_holdout(
            plan_json=plan,
            spot_parquet=spot,
            output_dir=output,
            expected_plan_sha256=_sha256(plan),
        )

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_run_epex_lab_locked_holdout_cli_exits_nonzero_when_coverage_missing(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=2, freq="h"),
    ).to_parquet(spot)

    code = main(
        [
            "--plan-json",
            str(plan),
            "--expected-plan-sha256",
            _sha256(plan),
            "--spot-parquet",
            str(spot),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 1


def test_run_epex_lab_locked_holdout_runs_backtest_and_audit_when_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _write_plan(tmp_path)
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="h"),
    ).to_parquet(spot)

    def fake_backtest_against_spot(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        (output_dir / "post.csv").write_text(
            "timestamp_utc,baseline_abs_error_eur_mwh,adjusted_abs_error_eur_mwh\n"
            "2026-07-10T00:00:00Z,4,3\n"
            "2026-07-10T01:00:00Z,4,3\n"
            "2026-07-10T02:00:00Z,4,3\n"
            "2026-07-10T03:00:00Z,4,3\n",
            encoding="utf-8",
        )
        payload = {
            "schema_version": "epex_shape_lab_spot_backtest.v3",
            "read_only": True,
            "benchmark_policy": "rolling_origin_epex_spot_no_ompex_lab_only",
            "promotion_gate": False,
            "production_approved": False,
            "independent_production_evidence": False,
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
            "ompex_used_in_backtest": False,
            "strict_lab_gate_pass": True,
            "eligible_rolling_fold_count": 1,
            "strict_lab_checks": {
                "rolling_folds_present": True,
                "rolling_folds_no_temporal_leak": True,
            },
            "evaluation_window": {
                "start_utc": "2026-07-10T00:00:00Z",
                "end_utc": "2026-07-10T04:00:00Z",
                "end_exclusive": True,
                "spot_filtered_before_residual_centering": True,
            },
            "valuation_timestamp_utc": "2026-07-09T00:00:00+00:00",
            "status": "DIAGNOSTIC_PASS",
            "source_hashes": {
                "baseline_csv": _sha256(Path(kwargs["baseline_csv"])),
                "adjusted_csv": _sha256(Path(kwargs["adjusted_csv"])),
            },
            "post_valuation_metrics": {"residual_mae_improvement_eur_mwh": 1.0},
            "outputs": {"post_valuation_timestamp_residuals_csv": str(output_dir / "post.csv")},
            "output_hashes": {"post_valuation_timestamp_residuals_csv": _sha256(output_dir / "post.csv")},
        }
        (output_dir / "spot_backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("scripts.run_epex_lab_locked_holdout.backtest_against_spot", fake_backtest_against_spot)

    summary = run_locked_holdout(
        plan_json=plan,
        spot_parquet=spot,
        output_dir=tmp_path / "out",
        expected_plan_sha256=_sha256(plan),
    )

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_FAIL"
    assert summary["coverage_ready"] is True
    assert summary["backtest_ran"] is True
    assert summary["audit_ran"] is True
    assert summary["holdout_pass"] is False
    assert (tmp_path / "out" / "locked_holdout_audit.json").exists()


def test_run_epex_lab_locked_holdout_cli_exits_zero_when_holdout_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _write_plan(tmp_path)
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="h"),
    ).to_parquet(spot)

    def fake_backtest_against_spot(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        (output_dir / "post.csv").write_text(
            "timestamp_utc,baseline_abs_error_eur_mwh,adjusted_abs_error_eur_mwh\n"
            "2026-07-10T00:00:00Z,4,3\n"
            "2026-07-10T01:00:00Z,4,3\n"
            "2026-07-10T02:00:00Z,4,3\n"
            "2026-07-10T03:00:00Z,4,3\n",
            encoding="utf-8",
        )
        payload = {
            "schema_version": "epex_shape_lab_spot_backtest.v3",
            "read_only": True,
            "benchmark_policy": "rolling_origin_epex_spot_no_ompex_lab_only",
            "promotion_gate": False,
            "production_approved": False,
            "independent_production_evidence": False,
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
            "ompex_used_in_backtest": False,
            "strict_lab_gate_pass": True,
            "eligible_rolling_fold_count": 1,
            "strict_lab_checks": {
                "rolling_folds_present": True,
                "rolling_folds_no_temporal_leak": True,
            },
            "evaluation_window": {
                "start_utc": "2026-07-10T00:00:00Z",
                "end_utc": "2026-07-10T04:00:00Z",
                "end_exclusive": True,
                "spot_filtered_before_residual_centering": True,
            },
            "valuation_timestamp_utc": "2026-07-09T00:00:00+00:00",
            "status": "DIAGNOSTIC_PASS",
            "source_hashes": {
                "baseline_csv": _sha256(Path(kwargs["baseline_csv"])),
                "adjusted_csv": _sha256(Path(kwargs["adjusted_csv"])),
            },
            "post_valuation_metrics": {"residual_mae_improvement_eur_mwh": 1.0},
            "outputs": {"post_valuation_timestamp_residuals_csv": str(output_dir / "post.csv")},
            "output_hashes": {"post_valuation_timestamp_residuals_csv": _sha256(output_dir / "post.csv")},
        }
        (output_dir / "spot_backtest_summary.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("scripts.run_epex_lab_locked_holdout.backtest_against_spot", fake_backtest_against_spot)

    code = main(
        [
            "--plan-json",
            str(plan),
            "--expected-plan-sha256",
            _sha256(plan),
            "--spot-parquet",
            str(spot),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 1


def test_run_epex_lab_locked_holdout_rejects_plan_hash_mismatch(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="h"),
    ).to_parquet(spot)

    summary = run_locked_holdout(
        plan_json=plan,
        spot_parquet=spot,
        output_dir=tmp_path / "out",
        expected_plan_sha256="0" * 64,
    )

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_PLAN_HASH_MISMATCH"
    assert summary["expected_plan_json_sha256"] == "0" * 64
    assert summary["actual_plan_json_sha256"] == _sha256(plan)
    assert summary["backtest_ran"] is False
    assert summary["audit_ran"] is False
    assert summary["holdout_pass"] is False


def test_run_epex_lab_locked_holdout_real_pipeline_uses_history_and_exact_window(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    plan_payload["activation_status"] = "locked_future_holdout"
    plan.write_text(json.dumps(plan_payload), encoding="utf-8")
    spot = tmp_path / "spot.parquet"
    spot_index = pd.date_range("2024-07-01T00:00:00Z", "2026-07-10T03:00:00Z", freq="h")
    local = spot_index.tz_convert("Europe/Zurich")
    spot_frame = pd.DataFrame(
        {"price_eur_mwh": 50.0 + (local.hour.to_numpy(dtype=float) / 10.0)},
        index=spot_index,
    )
    spot_frame.to_parquet(spot)
    provider_raw = tmp_path / "provider-raw.json"
    provider_raw.write_text(
        json.dumps(
            {
                "unix_seconds": [int(timestamp.timestamp()) for timestamp in spot_index],
                "price": spot_frame["price_eur_mwh"].tolist(),
            }
        ),
        encoding="utf-8",
    )
    acquisition = tmp_path / "spot-acquisition.json"
    acquisition.write_text(
        json.dumps(
            {
                "schema_version": "energy_charts_epex_spot_hourly_fetch.v1",
                "status": "OK",
                "source": "energy-charts.info",
                "bzn": "CH",
                "acquired_at_utc": "2026-07-10T05:00:00Z",
                "start_utc": str(spot_index[0]),
                "end_utc": "2026-07-10T04:00:00Z",
                "full_window_covered": True,
                "expected_hour_count": len(spot_index),
                "observed_hour_count": len(spot_index),
                "missing_hour_count": 0,
                "raw_observation_count": len(spot_index),
                "output_parquet_sha256": _sha256(spot),
                "provider_raw_json": str(provider_raw),
                "provider_raw_json_sha256": _sha256(provider_raw),
            }
        ),
        encoding="utf-8",
    )
    capture_seal = tmp_path / "energy_charts_locked_holdout_capture_seal.json"
    capture_seal.write_text(
        json.dumps(
            {
                "schema_version": "energy_charts_epex_locked_holdout_capture.v1",
                "plan_id": "test_holdout",
                "plan_json_sha256": _sha256(plan),
                "bzn": "CH",
                "spot_parquet_sha256": _sha256(spot),
                "spot_fetch_summary_sha256": _sha256(acquisition),
                "provider_raw_json_sha256": _sha256(provider_raw),
                "acquired_at_utc": "2026-07-10T05:00:00Z",
                "production_authorization": False,
            }
        ),
        encoding="utf-8",
    )

    summary = run_locked_holdout(
        plan_json=plan,
        spot_parquet=spot,
        output_dir=tmp_path / "real-out",
        expected_plan_sha256=_sha256(plan),
        spot_acquisition_summary=acquisition,
        spot_capture_seal=capture_seal,
    )

    assert summary["status"] == "LOCKED_HOLDOUT_PASS"
    assert summary["strict_lab_gate_pass"] is True
    backtest = json.loads((tmp_path / "real-out" / "spot_backtest_summary.json").read_text(encoding="utf-8"))
    assert backtest["eligible_rolling_fold_count"] > 0
    assert backtest["spot_provenance"]["status"] == "SPOT_PROVIDER_RAW_BOUND"
    assert backtest["spot_provenance"]["capture_seal"]["pass"] is True
    assert backtest["evaluation_window"]["spot_filtered_before_residual_centering"] is True
    post = pd.read_csv(tmp_path / "real-out" / "post_valuation_timestamp_residuals.csv")
    assert len(post) == 4
    assert pd.to_datetime(post["timestamp_utc"], utc=True).min() == pd.Timestamp("2026-07-10T00:00:00Z")
    assert pd.to_datetime(post["timestamp_utc"], utc=True).max() == pd.Timestamp("2026-07-10T03:00:00Z")


def test_direct_runner_rejects_noncanonical_t057_before_backtest(tmp_path: Path, monkeypatch) -> None:
    plan = _write_plan(tmp_path)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["plan_id"] = "t057_locked_t056_future_holdout"
    payload["activation_status"] = "locked_future_holdout"
    plan.write_text(json.dumps(payload), encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        "scripts.run_epex_lab_locked_holdout.backtest_against_spot",
        lambda **kwargs: calls.append("backtest"),
    )

    summary = run_locked_holdout(
        plan_json=plan,
        spot_parquet=tmp_path / "spot.parquet",
        output_dir=tmp_path / "noncanonical",
        expected_plan_sha256=_sha256(plan),
        spot_acquisition_summary=tmp_path / "acquisition.json",
        spot_capture_seal=tmp_path / "seal.json",
    )

    assert summary["status"] == "NO_GO_T057_CANONICAL_ROUTE_MISMATCH"
    assert summary["backtest_ran"] is False
    assert calls == []
    assert not (tmp_path / "noncanonical").exists()


def test_run_epex_lab_locked_holdout_rejects_unprovenanced_locked_spot(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["activation_status"] = "locked_future_holdout"
    plan.write_text(json.dumps(payload), encoding="utf-8")
    spot = tmp_path / "spot.parquet"
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="h"),
    ).to_parquet(spot)

    summary = run_locked_holdout(
        plan_json=plan,
        spot_parquet=spot,
        output_dir=tmp_path / "out",
        expected_plan_sha256=_sha256(plan),
    )

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_SPOT_PROVENANCE_INVALID"
    assert summary["spot_provenance"]["status"] == "MISSING_SPOT_ACQUISITION_SUMMARY"
    assert summary["backtest_ran"] is False


def test_run_epex_lab_locked_holdout_rejects_raw_parquet_semantic_mismatch(
    tmp_path: Path,
) -> None:
    plan = _write_plan(tmp_path)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["activation_status"] = "locked_future_holdout"
    plan.write_text(json.dumps(payload), encoding="utf-8")
    spot = tmp_path / "spot.parquet"
    timestamps = pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="h")
    pd.DataFrame(
        {"price_eur_mwh": [1.0, 2.0, 3.0, 4.0]},
        index=timestamps,
    ).to_parquet(spot)
    provider_raw = tmp_path / "provider-raw.json"
    provider_raw.write_text(
        json.dumps(
            {
                "unix_seconds": [int(timestamp.timestamp()) for timestamp in timestamps],
                "price": [101.0, 102.0, 103.0, 104.0],
            }
        ),
        encoding="utf-8",
    )
    acquisition = tmp_path / "spot-acquisition.json"
    acquisition.write_text(
        json.dumps(
            {
                "schema_version": "energy_charts_epex_spot_hourly_fetch.v1",
                "status": "OK",
                "source": "energy-charts.info",
                "bzn": "CH",
                "acquired_at_utc": "2026-07-10T05:00:00Z",
                "start_utc": "2026-07-10T00:00:00Z",
                "end_utc": "2026-07-10T04:00:00Z",
                "full_window_covered": True,
                "expected_hour_count": 4,
                "observed_hour_count": 4,
                "missing_hour_count": 0,
                "raw_observation_count": 4,
                "output_parquet_sha256": _sha256(spot),
                "provider_raw_json": str(provider_raw),
                "provider_raw_json_sha256": _sha256(provider_raw),
            }
        ),
        encoding="utf-8",
    )

    summary = run_locked_holdout(
        plan_json=plan,
        spot_parquet=spot,
        output_dir=tmp_path / "out-mismatch",
        expected_plan_sha256=_sha256(plan),
        spot_acquisition_summary=acquisition,
    )

    assert summary["status"] == "NO_GO_LOCKED_HOLDOUT_SPOT_PROVENANCE_INVALID"
    assert summary["spot_provenance"]["checks"]["provider_raw_semantic_replay"] is False
    assert summary["backtest_ran"] is False


def _write_plan(tmp_path: Path) -> Path:
    baseline = tmp_path / "baseline.csv"
    adjusted = tmp_path / "adjusted.csv"
    _write_candidate_csv(baseline)
    _write_candidate_csv(adjusted)
    baseline_sha = _sha256(baseline)
    adjusted_sha = _sha256(adjusted)
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "epex_lab_locked_holdout_plan.v1",
                "plan_id": "test_holdout",
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
                    "max_auto_folds": 12,
                    "min_eval_hours": 4,
                },
                "pass_criteria": {
                    "baseline_csv_sha256": baseline_sha,
                    "adjusted_csv_sha256": adjusted_sha,
                    "strict_lab_gate_pass": True,
                    "min_holdout_hours": 4,
                    "min_residual_mae_improvement_eur_mwh": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_candidate_csv(path: Path) -> None:
    utc = pd.date_range("2026-07-10T00:00:00Z", periods=4, freq="h")
    local = utc.tz_convert("Europe/Zurich")
    offset = local.strftime("%z")
    frame = pd.DataFrame(
        {
            "timestamp_ch": local.strftime("%d.%m.%Y %H:%M"),
            "utc_offset_ch": "UTC" + offset.str.slice(0, 3) + ":" + offset.str.slice(3, 5),
        }
    )
    for index, column in enumerate(CANDIDATE_PRICE_COLUMNS):
        frame[column] = 50.0 + float(index)
    frame.to_csv(path, index=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

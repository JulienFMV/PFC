from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd

from scripts.audit_epex_lab_locked_holdout import audit_holdout, main


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
    summary = _write_summary(tmp_path, baseline_sha="base", adjusted_sha="adjusted")

    audit = audit_holdout(plan_json=plan, spot_backtest_summary=summary, output=tmp_path / "audit.json")

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
    summary = _write_summary(tmp_path, baseline_sha="base", adjusted_sha="adjusted")

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
    summary = _write_summary(tmp_path, baseline_sha="base", adjusted_sha="adjusted")

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
    summary = _write_summary(tmp_path, baseline_sha="base", adjusted_sha="adjusted", schema_version="other.v1")

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
    summary = _write_summary(tmp_path, baseline_sha="base", adjusted_sha="adjusted")
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
    summary = _write_summary(tmp_path, baseline_sha="base", adjusted_sha="adjusted")
    post.write_text(
        "timestamp_utc,baseline_abs_error_eur_mwh,adjusted_abs_error_eur_mwh\n"
        "2026-07-10T00:00:00Z,4,3\n",
        encoding="utf-8",
    )

    audit = audit_holdout(plan_json=plan, spot_backtest_summary=summary)

    assert audit["status"] == "NO_GO_LOCKED_HOLDOUT_FAIL"
    assert audit["holdout_pass"] is False
    assert audit["checks"]["post_valuation_csv_sha256_bound"] is False


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
    summary = _write_summary(tmp_path, baseline_sha="base", adjusted_sha="adjusted")

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
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "epex_lab_locked_holdout_plan.v1",
                "benchmark_policy": "locked_future_no_ompex_holdout",
                "ompex_used_in_model": False,
                "ompex_used_in_selection": False,
                "ompex_used_in_backtest": False,
                "holdout_start_utc": "2026-07-10T00:00:00Z",
                "holdout_end_utc": "2026-07-11T00:00:00Z",
                "backtest": {"valuation_timestamp_utc": "2026-07-09T00:00:00Z"},
                "pass_criteria": {
                    "baseline_csv_sha256": "base",
                    "adjusted_csv_sha256": "adjusted",
                    "strict_lab_gate_pass": True,
                    "min_holdout_hours": 4,
                    "min_residual_mae_improvement_eur_mwh": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_summary(
    tmp_path: Path,
    *,
    baseline_sha: str,
    adjusted_sha: str,
    schema_version: str = "epex_shape_lab_spot_backtest.v1",
) -> Path:
    path = tmp_path / "summary.json"
    post_csv = tmp_path / "post.csv"
    path.write_text(
        json.dumps(
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
                "valuation_timestamp_utc": "2026-07-09T00:00:00+00:00",
                "source_hashes": {
                    "baseline_csv": baseline_sha,
                    "adjusted_csv": adjusted_sha,
                },
                "outputs": {
                    "post_valuation_timestamp_residuals_csv": str(post_csv),
                },
                "output_hashes": {
                    "post_valuation_timestamp_residuals_csv": _sha256(post_csv) if post_csv.exists() else None,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

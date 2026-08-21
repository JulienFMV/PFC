"""Run a locked EPEX lab holdout only after spot coverage is complete.

The runner is intentionally conservative: it first writes a coverage report and
stops without running the backtest unless the pre-registered holdout window is
fully covered by the supplied EPEX spot parquet. It never approves production
promotion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.durable_artifact import write_durable_exact_bytes
from scripts.audit_epex_lab_locked_holdout import audit_holdout
from scripts.backtest_epex_shape_lab_against_spot import backtest_against_spot
from scripts.check_epex_lab_locked_holdout_coverage import check_coverage
from scripts.epex_lab_locked_holdout_policy import (
    RUN_SCHEMA_VERSION,
    T057_CAPTURE_SCHEMA_VERSION,
    T057_PLAN_ID,
    T057_PLAN_SHA256,
    build_locked_plan_identity,
    canonical_t057_output_path,
    canonical_t057_plan_path,
)
from scripts.fetch_energy_charts_epex_spot_hourly import replay_provider_raw_hourly


def run_locked_holdout(
    *,
    plan_json: Path,
    spot_parquet: Path,
    output_dir: Path,
    expected_plan_sha256: str,
    spot_acquisition_summary: Path | None = None,
    spot_capture_seal: Path | None = None,
) -> dict[str, Any]:
    plan_json = _resolved_path(plan_json)
    spot_parquet = _resolved_path(spot_parquet)
    output_dir = _resolved_path(output_dir)
    plan = _read_json(plan_json)
    actual_plan_sha256 = _sha256(plan_json)
    if plan.get("plan_id") == T057_PLAN_ID and not _is_canonical_t057_route(
        plan_json=plan_json,
        expected_plan_sha256=expected_plan_sha256,
        actual_plan_sha256=actual_plan_sha256,
        spot_parquet=spot_parquet,
        spot_acquisition_summary=spot_acquisition_summary,
        spot_capture_seal=spot_capture_seal,
        output_dir=output_dir,
    ):
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            "read_only": True,
            "promotion_gate": False,
            "production_approved": False,
            "plan_json": str(plan_json),
            "expected_plan_json_sha256": expected_plan_sha256,
            "actual_plan_json_sha256": actual_plan_sha256,
            "spot_parquet": str(spot_parquet),
            "output_dir": str(output_dir),
            "coverage_ready": False,
            "locked_plan_identity": build_locked_plan_identity(plan, plan_json=plan_json),
            "benchmark_policy": "locked_future_no_ompex_holdout",
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
            "ompex_used_in_backtest": False,
            "status": "NO_GO_T057_CANONICAL_ROUTE_MISMATCH",
            "backtest_ran": False,
            "audit_ran": False,
            "holdout_pass": False,
            "next_action": "Use the frozen T057 plan and unique canonical one-shot output route.",
        }
    output_dir.mkdir(parents=True, exist_ok=False)
    if expected_plan_sha256 != actual_plan_sha256:
        run_summary = {
            "schema_version": RUN_SCHEMA_VERSION,
            "read_only": True,
            "promotion_gate": False,
            "production_approved": False,
            "plan_json": str(plan_json),
            "expected_plan_json_sha256": expected_plan_sha256,
            "actual_plan_json_sha256": actual_plan_sha256,
            "spot_parquet": str(spot_parquet),
            "output_dir": str(output_dir),
            "coverage_ready": False,
            "locked_plan_identity": build_locked_plan_identity(plan, plan_json=plan_json),
            "benchmark_policy": "locked_future_no_ompex_holdout",
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
            "ompex_used_in_backtest": False,
            "status": "NO_GO_LOCKED_HOLDOUT_PLAN_HASH_MISMATCH",
            "backtest_ran": False,
            "audit_ran": False,
            "holdout_pass": False,
            "next_action": "Do not run holdout; regenerate or use the exact pre-registered locked plan hash.",
        }
        return _write_run_summary(output_dir, run_summary)
    spot_provenance = _spot_provenance_evidence(
        spot_parquet=spot_parquet,
        acquisition_summary=spot_acquisition_summary,
    )
    capture_seal = _spot_capture_seal_evidence(
        plan=plan,
        plan_sha256=actual_plan_sha256,
        spot_parquet=spot_parquet,
        acquisition_summary=spot_acquisition_summary,
        capture_seal=spot_capture_seal,
    )
    spot_provenance["capture_seal"] = capture_seal
    if capture_seal["required"] is True:
        spot_provenance["pass"] = bool(
            spot_provenance.get("pass") is True and capture_seal["pass"] is True
        )
        if spot_provenance["pass"] is not True:
            spot_provenance["status"] = "SPOT_CAPTURE_ROUTE_INVALID"
    if plan.get("activation_status") == "locked_future_holdout" and spot_provenance["pass"] is not True:
        run_summary = {
            "schema_version": RUN_SCHEMA_VERSION,
            "read_only": True,
            "promotion_gate": False,
            "production_approved": False,
            "plan_json": str(plan_json),
            "expected_plan_json_sha256": expected_plan_sha256,
            "actual_plan_json_sha256": actual_plan_sha256,
            "spot_parquet": str(spot_parquet),
            "output_dir": str(output_dir),
            "coverage_ready": False,
            "locked_plan_identity": build_locked_plan_identity(plan, plan_json=plan_json),
            "benchmark_policy": "locked_future_no_ompex_holdout",
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
            "ompex_used_in_backtest": False,
            "spot_provenance": spot_provenance,
            "status": "NO_GO_LOCKED_HOLDOUT_SPOT_PROVENANCE_INVALID",
            "backtest_ran": False,
            "audit_ran": False,
            "holdout_pass": False,
            "next_action": "Capture and bind immutable provider-raw spot evidence before running the holdout.",
        }
        return _write_run_summary(output_dir, run_summary)
    coverage_path = output_dir / "coverage_status.json"
    coverage = check_coverage(plan_json=plan_json, spot_parquet=spot_parquet, output=coverage_path)
    run_summary: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "plan_json": str(plan_json),
        "expected_plan_json_sha256": expected_plan_sha256,
        "actual_plan_json_sha256": actual_plan_sha256,
        "spot_parquet": str(spot_parquet),
        "output_dir": str(output_dir),
        "coverage_status": str(coverage_path),
        "coverage_status_sha256": _sha256(coverage_path),
        "coverage_ready": bool(coverage.get("ready_to_run_backtest")),
        "coverage": coverage,
        "locked_plan_identity": build_locked_plan_identity(plan, plan_json=plan_json),
        "benchmark_policy": "locked_future_no_ompex_holdout",
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "spot_provenance": spot_provenance,
    }
    if coverage.get("ready_to_run_backtest") is not True:
        run_summary.update(
            {
                "status": coverage.get("status") or "WAITING_FOR_FULL_SPOT_COVERAGE",
                "backtest_ran": False,
                "audit_ran": False,
                "next_action": coverage.get("next_action"),
            }
        )
        return _write_run_summary(output_dir, run_summary)

    backtest_cfg = plan.get("backtest") or {}
    backtest = backtest_against_spot(
        baseline_csv=_repo_evidence_path(plan["baseline_csv"]),
        adjusted_csv=_repo_evidence_path(plan["adjusted_csv"]),
        spot_parquet=spot_parquet,
        output_dir=output_dir,
        valuation_timestamp=str(backtest_cfg["valuation_timestamp_utc"]),
        lookback_years=int(backtest_cfg.get("lookback_years", 2)),
        eval_days=int(backtest_cfg.get("eval_days", 30)),
        embargo_days=int(backtest_cfg.get("embargo_days", 1)),
        max_auto_folds=int(backtest_cfg.get("max_auto_folds", 12)),
        min_eval_hours=int(backtest_cfg.get("min_eval_hours", 24)),
        evaluation_start_utc=str(plan["holdout_start_utc"]),
        evaluation_end_utc=str(plan["holdout_end_utc"]),
        spot_provenance=spot_provenance,
    )
    backtest_summary_path = output_dir / "spot_backtest_summary.json"
    audit_path = output_dir / "locked_holdout_audit.json"
    audit = audit_holdout(plan_json=plan_json, spot_backtest_summary=backtest_summary_path, output=audit_path)
    run_summary.update(
        {
            "status": "LOCKED_HOLDOUT_PASS" if audit.get("holdout_pass") is True else "NO_GO_LOCKED_HOLDOUT_FAIL",
            "backtest_ran": True,
            "audit_ran": True,
            "spot_backtest_summary": str(backtest_summary_path),
            "spot_backtest_summary_sha256": _sha256(backtest_summary_path),
            "locked_holdout_audit": str(audit_path),
            "locked_holdout_audit_sha256": _sha256(audit_path),
            "spot_backtest_status": backtest.get("status"),
            "holdout_audit_status": audit.get("status"),
            "holdout_pass": bool(audit.get("holdout_pass")),
            "strict_lab_gate_pass": bool(backtest.get("strict_lab_gate_pass")),
            "post_valuation_metrics": backtest.get("post_valuation_metrics"),
            "holdout_metrics": audit.get("holdout_metrics"),
            "next_action": (
                "Review holdout evidence; production still requires adjusted production/export/selected/capstone chain."
                if audit.get("holdout_pass") is True
                else "Do not promote; investigate holdout failure without retuning against this locked window."
            ),
        }
    )
    return _write_run_summary(output_dir, run_summary)


def _write_run_summary(output_dir: Path, run_summary: dict[str, Any]) -> dict[str, Any]:
    path = output_dir / "locked_holdout_run_summary.json"
    payload = (json.dumps(_jsonable(run_summary), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    write_durable_exact_bytes(path, payload, label="locked holdout run summary")
    run_summary["run_summary"] = str(_resolved_path(path))
    return run_summary


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _repo_evidence_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    return _resolved_path(path)


def _is_canonical_t057_route(
    *,
    plan_json: Path,
    expected_plan_sha256: str,
    actual_plan_sha256: str,
    spot_parquet: Path,
    spot_acquisition_summary: Path | None,
    spot_capture_seal: Path | None,
    output_dir: Path,
) -> bool:
    canonical_output = canonical_t057_output_path()
    return bool(
        plan_json == canonical_t057_plan_path()
        and expected_plan_sha256 == T057_PLAN_SHA256
        and actual_plan_sha256 == T057_PLAN_SHA256
        and spot_parquet == canonical_output / "energy_charts_epex_spot_hourly.parquet"
        and spot_acquisition_summary is not None
        and _resolved_path(spot_acquisition_summary)
        == canonical_output / "energy_charts_epex_spot_fetch_summary.json"
        and spot_capture_seal is not None
        and _resolved_path(spot_capture_seal)
        == canonical_output / "energy_charts_locked_holdout_capture_seal.json"
        and output_dir == canonical_output / "locked_holdout_runner"
    )


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spot_provenance_evidence(
    *,
    spot_parquet: Path,
    acquisition_summary: Path | None,
) -> dict[str, Any]:
    if acquisition_summary is None:
        return {
            "provided": False,
            "pass": False,
            "status": "MISSING_SPOT_ACQUISITION_SUMMARY",
            "checks": {},
        }
    acquisition_summary = _resolved_path(acquisition_summary)
    if not acquisition_summary.is_file():
        return {
            "provided": True,
            "pass": False,
            "status": "SPOT_ACQUISITION_SUMMARY_MISSING",
            "acquisition_summary": str(acquisition_summary),
            "checks": {},
        }
    try:
        summary = _read_json(acquisition_summary)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "provided": True,
            "pass": False,
            "status": "SPOT_ACQUISITION_SUMMARY_INVALID",
            "acquisition_summary": str(acquisition_summary),
            "error": f"{type(exc).__name__}: {exc}",
            "checks": {},
        }
    raw_text = summary.get("provider_raw_json")
    raw_path = Path(str(raw_text)) if raw_text else None
    if raw_path is not None and not raw_path.is_absolute():
        raw_path = acquisition_summary.parent / raw_path
    raw_path = _resolved_path(raw_path) if raw_path is not None else None
    spot_sha = _sha256(spot_parquet) if spot_parquet.is_file() else None
    raw_sha = _sha256(raw_path) if raw_path is not None and raw_path.is_file() else None
    semantic_replay = False
    acquisition_scope = False
    if raw_path is not None and raw_path.is_file() and spot_parquet.is_file():
        try:
            replayed, replayed_coverage = replay_provider_raw_hourly(
                raw_path,
                start=str(summary.get("start_utc")),
                end=str(summary.get("end_utc")),
            )
            derived = pd.read_parquet(spot_parquet)
            derived.index = pd.to_datetime(derived.index, utc=True)
            pd.testing.assert_frame_equal(
                derived[["price_eur_mwh"]].sort_index(),
                replayed[["price_eur_mwh"]].sort_index(),
                check_exact=True,
                check_names=False,
            )
            semantic_replay = True
            acquisition_scope = bool(
                summary.get("source") == "energy-charts.info"
                and summary.get("bzn") == "CH"
                and isinstance(summary.get("acquired_at_utc"), str)
                and replayed_coverage.get("full_window_covered") is True
                and summary.get("expected_hour_count")
                == replayed_coverage.get("expected_hour_count")
                and summary.get("observed_hour_count")
                == replayed_coverage.get("observed_hour_count")
                and summary.get("missing_hour_count") == 0
                and summary.get("raw_observation_count")
                == replayed_coverage.get("raw_observation_count")
            )
        except (AssertionError, OSError, ValueError, TypeError, KeyError):
            semantic_replay = False
            acquisition_scope = False
    checks = {
        "summary_schema": summary.get("schema_version") == "energy_charts_epex_spot_hourly_fetch.v1",
        "summary_status_ok": summary.get("status") == "OK",
        "full_window_covered": summary.get("full_window_covered") is True,
        "spot_parquet_present": spot_parquet.is_file(),
        "spot_parquet_sha256_bound": spot_sha is not None and summary.get("output_parquet_sha256") == spot_sha,
        "provider_raw_path_present": raw_path is not None,
        "provider_raw_file_present": raw_path is not None and raw_path.is_file(),
        "provider_raw_sha256_bound": raw_sha is not None and summary.get("provider_raw_json_sha256") == raw_sha,
        "provider_raw_semantic_replay": semantic_replay,
        "acquisition_scope_replayed": acquisition_scope,
    }
    passed = bool(all(checks.values()))
    return {
        "provided": True,
        "pass": passed,
        "status": "SPOT_PROVIDER_RAW_BOUND" if passed else "SPOT_PROVIDER_RAW_BINDING_INVALID",
        "acquisition_summary": str(acquisition_summary),
        "acquisition_summary_sha256": _sha256(acquisition_summary),
        "spot_parquet": str(spot_parquet),
        "spot_parquet_sha256": spot_sha,
        "provider_raw_json": str(raw_path) if raw_path is not None else None,
        "provider_raw_json_sha256": raw_sha,
        "checks": checks,
    }


def _spot_capture_seal_evidence(
    *,
    plan: dict[str, Any],
    plan_sha256: str,
    spot_parquet: Path,
    acquisition_summary: Path | None,
    capture_seal: Path | None,
) -> dict[str, Any]:
    if plan.get("plan_id") != T057_PLAN_ID:
        return {
            "required": False,
            "pass": True,
            "status": "NOT_REQUIRED_FOR_NON_T057_PLAN",
        }
    if acquisition_summary is None or capture_seal is None:
        return {
            "required": True,
            "pass": False,
            "status": "MISSING_T057_CAPTURE_SEAL",
        }
    acquisition_summary = _resolved_path(acquisition_summary)
    capture_seal = _resolved_path(capture_seal)
    failed = {
        "required": True,
        "pass": False,
        "status": "T057_CAPTURE_SEAL_INVALID",
        "capture_seal": str(capture_seal),
    }
    canonical_output = canonical_t057_output_path()
    if not (
        plan_sha256 == T057_PLAN_SHA256
        and spot_parquet == canonical_output / "energy_charts_epex_spot_hourly.parquet"
        and _resolved_path(acquisition_summary)
        == canonical_output / "energy_charts_epex_spot_fetch_summary.json"
        and _resolved_path(capture_seal)
        == canonical_output / "energy_charts_locked_holdout_capture_seal.json"
    ):
        return failed
    try:
        seal = _read_json(capture_seal)
        acquisition = _read_json(acquisition_summary)
        raw_text = acquisition.get("provider_raw_json")
        provider_raw = Path(str(raw_text)) if raw_text else None
        if provider_raw is not None and not provider_raw.is_absolute():
            provider_raw = acquisition_summary.parent / provider_raw
        provider_raw = _resolved_path(provider_raw) if provider_raw is not None else None
        expected_keys = {
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
        attempt_seal = canonical_output / "energy_charts_locked_holdout_attempt_seal.json"
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
        checks = {
            "schema": seal.get("schema_version") == T057_CAPTURE_SCHEMA_VERSION,
            "exact_schema_keys": set(seal) == expected_keys,
            "plan_id": seal.get("plan_id") == T057_PLAN_ID,
            "plan_sha256": seal.get("plan_json_sha256") == plan_sha256,
            "bzn_ch": seal.get("bzn") == "CH",
            "co_located": capture_seal.parent
            == acquisition_summary.parent
            == spot_parquet.parent,
            "spot_parquet_sha256": spot_parquet.is_file()
            and seal.get("spot_parquet_sha256") == _sha256(spot_parquet),
            "spot_fetch_summary_sha256": acquisition_summary.is_file()
            and seal.get("spot_fetch_summary_sha256") == _sha256(acquisition_summary),
            "provider_raw_sha256": provider_raw is not None
            and provider_raw.is_file()
            and seal.get("provider_raw_json_sha256") == _sha256(provider_raw),
            "attempt_seal_bound": attempt_seal.is_file()
            and seal.get("attempt_seal_sha256") == _sha256(attempt_seal)
            and set(attempt) == expected_attempt_keys
            and attempt.get("schema_version")
            == "energy_charts_epex_locked_holdout_attempt.v1"
            and attempt.get("plan_id") == T057_PLAN_ID
            and attempt.get("plan_json_sha256") == T057_PLAN_SHA256
            and attempt.get("bzn") == "CH"
            and _resolved_path(Path(str(attempt.get("output_dir", ""))))
            == canonical_output
            and isinstance(attempt.get("attempted_at_utc"), str)
            and attempt.get("production_authorization") is False,
            "acquired_at_utc": isinstance(seal.get("acquired_at_utc"), str)
            and seal.get("acquired_at_utc") == acquisition.get("acquired_at_utc"),
            "not_production_authority": seal.get("production_authorization") is False,
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return failed
    passed = bool(all(checks.values()))
    return {
        **failed,
        "pass": passed,
        "status": "T057_CAPTURE_SEAL_BOUND" if passed else "T057_CAPTURE_SEAL_INVALID",
        "capture_seal_sha256": _sha256(capture_seal),
        "checks": checks,
    }


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
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--spot-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spot-acquisition-summary", type=Path)
    parser.add_argument("--spot-capture-seal", type=Path)
    args = parser.parse_args(argv)
    summary = run_locked_holdout(
        plan_json=args.plan_json,
        spot_parquet=args.spot_parquet,
        output_dir=args.output_dir,
        expected_plan_sha256=args.expected_plan_sha256,
        spot_acquisition_summary=args.spot_acquisition_summary,
        spot_capture_seal=args.spot_capture_seal,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary.get("status") == "LOCKED_HOLDOUT_PASS" and summary.get("holdout_pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())

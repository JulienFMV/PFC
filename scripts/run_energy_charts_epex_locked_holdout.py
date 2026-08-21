"""Fetch observed Energy Charts spot and run a locked EPEX holdout if ready.

This operator wrapper is fail-closed: it first verifies the locked plan hash,
then fetches enough pre-valuation history for rolling-origin folds plus the
full pre-registered holdout window. Provider-raw JSON is retained beside the
derived parquet. If Energy Charts has not published every expected hour, no
spot parquet is written and the locked holdout runner is not called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.epex_lab_locked_holdout_policy import (
    ENERGY_CHARTS_RUN_SCHEMA_VERSION,
    T057_CAPTURE_SCHEMA_VERSION,
    T057_PLAN_ID,
    T057_PLAN_SHA256,
    build_locked_plan_identity,
    canonical_t057_output_path,
    canonical_t057_plan_path,
)
from scripts.fetch_energy_charts_epex_spot_hourly import fetch_hourly_spot
from scripts.run_epex_lab_locked_holdout import run_locked_holdout

SUMMARY_SCHEMA_VERSION = ENERGY_CHARTS_RUN_SCHEMA_VERSION
PLAN_SCHEMA_VERSION = "epex_lab_locked_holdout_plan.v1"
CAPTURE_SEAL_NAME = "energy_charts_locked_holdout_capture_seal.json"
CAPTURE_ATTEMPT_NAME = "energy_charts_locked_holdout_attempt_seal.json"
_CAPTURE_EVIDENCE_NAMES = (
    CAPTURE_ATTEMPT_NAME,
    "energy_charts_epex_spot_hourly.parquet",
    "energy_charts_epex_spot_fetch_summary.json",
    "energy_charts_epex_spot_provider_raw.json",
    CAPTURE_SEAL_NAME,
)


def run_energy_charts_locked_holdout(
    *,
    plan_json: Path,
    expected_plan_sha256: str,
    output_dir: Path,
    bzn: str = "CH",
    as_of_utc: str | None = None,
) -> dict[str, Any]:
    plan_json = _resolved_path(plan_json)
    output_dir = _resolved_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = _read_json(plan_json)
    actual_plan_sha256 = _sha256(plan_json)
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "plan_json": str(plan_json),
        "expected_plan_json_sha256": expected_plan_sha256,
        "actual_plan_json_sha256": actual_plan_sha256,
        "output_dir": str(output_dir),
        "bzn": bzn,
        "benchmark_policy": "locked_future_no_ompex_holdout",
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "locked_plan_identity": build_locked_plan_identity(plan, plan_json=plan_json),
        "spot_fetch_ran": False,
        "locked_holdout_ran": False,
        "holdout_pass": False,
    }
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        summary.update(
            {
                "status": "NO_GO_LOCKED_HOLDOUT_PLAN_SCHEMA",
                "next_action": f"Use a locked holdout plan with schema {PLAN_SCHEMA_VERSION}.",
            }
        )
        return _write_summary(output_dir, summary)
    if actual_plan_sha256 != expected_plan_sha256:
        summary.update(
            {
                "status": "NO_GO_LOCKED_HOLDOUT_PLAN_HASH_MISMATCH",
                "next_action": "Do not fetch spot or run holdout; use the exact pre-registered locked plan hash.",
            }
        )
        return _write_summary(output_dir, summary)
    if plan.get("plan_id") == T057_PLAN_ID:
        canonical_output = canonical_t057_output_path()
        if (
            plan_json != canonical_t057_plan_path()
            or expected_plan_sha256 != T057_PLAN_SHA256
            or actual_plan_sha256 != T057_PLAN_SHA256
            or output_dir != canonical_output
            or bzn != "CH"
            or as_of_utc is not None
        ):
            summary.update(
                {
                    "status": "NO_GO_T057_CANONICAL_ROUTE_MISMATCH",
                    "canonical_output_dir": str(canonical_output),
                    "next_action": (
                        "Use the unique T057 wrapper route, frozen plan SHA, CH bidding zone, "
                        "canonical output directory, and trusted system clock."
                    ),
                }
            )
            return _write_summary(output_dir, summary)

    holdout_start = _to_utc(plan["holdout_start_utc"])
    holdout_end = _to_utc(plan["holdout_end_utc"])
    latest_required = holdout_end - timedelta(hours=1)
    as_of = _to_utc(as_of_utc) if as_of_utc else datetime.now(tz=timezone.utc)
    backtest_cfg = plan.get("backtest") or {}
    lookback_years = int(backtest_cfg.get("lookback_years", 2))
    eval_days = int(backtest_cfg.get("eval_days", 30))
    embargo_days = int(backtest_cfg.get("embargo_days", 1))
    valuation = _to_utc(backtest_cfg.get("valuation_timestamp_utc") or plan.get("frozen_at_utc"))
    rolling_history_start = _years_before(valuation, lookback_years) - timedelta(
        days=eval_days + embargo_days
    )
    fetch_start = min(rolling_history_start, holdout_start)
    summary.update(
        {
            "as_of_utc": _iso(as_of),
            "holdout_start_utc": _iso(holdout_start),
            "holdout_end_utc": _iso(holdout_end),
            "latest_required_holdout_utc": _iso(latest_required),
            "holdout_complete_after_utc": _iso(holdout_end),
            "rolling_history_start_utc": _iso(rolling_history_start),
            "spot_fetch_start_utc": _iso(fetch_start),
        }
    )
    if holdout_end <= holdout_start:
        summary.update(
            {
                "status": "NO_GO_LOCKED_HOLDOUT_PLAN_WINDOW_INVALID",
                "next_action": "Fix the locked holdout plan window before fetching spot or running holdout.",
            }
        )
        return _write_summary(output_dir, summary)
    if as_of < holdout_end:
        summary.update(
            {
                "status": "LOCKED_HOLDOUT_WINDOW_NOT_COMPLETE",
                "next_action": "Wait until the locked holdout window is complete, then refresh Energy Charts spot.",
            }
        )
        return _write_summary(output_dir, summary)

    existing_capture_evidence = sorted(
        name for name in _CAPTURE_EVIDENCE_NAMES if (output_dir / name).exists()
    )
    if existing_capture_evidence:
        summary.update(
            {
                "status": "NO_GO_LOCKED_HOLDOUT_CAPTURE_ALREADY_EXISTS",
                "existing_capture_evidence": existing_capture_evidence,
                "next_action": (
                    "Do not overwrite, refresh, or cherry-pick locked holdout evidence; "
                    "audit the first capture or register a new future holdout plan."
                ),
            }
        )
        prior_summary = output_dir / "energy_charts_locked_holdout_run_summary.json"
        summary["run_summary"] = str(prior_summary) if prior_summary.exists() else None
        return summary

    attempt_seal = output_dir / CAPTURE_ATTEMPT_NAME
    _write_capture_seal(
        attempt_seal,
        {
            "schema_version": "energy_charts_epex_locked_holdout_attempt.v1",
            "plan_id": plan.get("plan_id"),
            "plan_json_sha256": actual_plan_sha256,
            "bzn": bzn,
            "output_dir": str(output_dir),
            "attempted_at_utc": _iso(datetime.now(tz=timezone.utc)),
            "production_authorization": False,
        },
    )
    summary.update(
        {
            "attempt_seal": str(attempt_seal),
            "attempt_seal_sha256": _sha256(attempt_seal),
        }
    )

    spot_parquet = output_dir / "energy_charts_epex_spot_hourly.parquet"
    spot_fetch_summary = output_dir / "energy_charts_epex_spot_fetch_summary.json"
    provider_raw_json = output_dir / "energy_charts_epex_spot_provider_raw.json"
    fetch_summary = fetch_hourly_spot(
        start=_iso(fetch_start),
        end=str(plan["holdout_end_utc"]),
        bzn=bzn,
        output_parquet=spot_parquet,
        summary_json=spot_fetch_summary,
        provider_raw_json=provider_raw_json,
        allow_partial=False,
    )
    summary.update(
        {
            "spot_fetch_ran": True,
            "spot_fetch_summary": str(spot_fetch_summary),
            "spot_fetch_summary_sha256": _sha256(spot_fetch_summary),
            "provider_raw_json": str(provider_raw_json),
            "provider_raw_json_sha256": _sha256(provider_raw_json) if provider_raw_json.exists() else None,
            "spot_fetch": fetch_summary,
        }
    )
    if fetch_summary.get("full_window_covered") is not True:
        summary.update(
            {
                "status": "LOCKED_HOLDOUT_SPOT_WAITING",
                "next_action": fetch_summary.get("next_action")
                or "Refresh after Energy Charts publishes every locked holdout hour.",
            }
        )
        return _write_summary(output_dir, summary)

    capture_seal = output_dir / CAPTURE_SEAL_NAME
    _write_capture_seal(
        capture_seal,
        {
            "schema_version": T057_CAPTURE_SCHEMA_VERSION,
            "plan_id": plan.get("plan_id"),
            "plan_json_sha256": actual_plan_sha256,
            "bzn": bzn,
            "spot_parquet_sha256": _sha256(spot_parquet),
            "spot_fetch_summary_sha256": _sha256(spot_fetch_summary),
            "provider_raw_json_sha256": _sha256(provider_raw_json),
            "attempt_seal_sha256": _sha256(attempt_seal),
            "acquired_at_utc": fetch_summary.get("acquired_at_utc"),
            "production_authorization": False,
        },
    )
    summary.update(
        {
            "capture_seal": str(capture_seal),
            "capture_seal_sha256": _sha256(capture_seal),
            "capture_policy": "FIRST_PROVIDER_CAPTURE_FAIL_CLOSED_NO_OVERWRITE",
        }
    )

    holdout_output_dir = output_dir / "locked_holdout_runner"
    run_summary = run_locked_holdout(
        plan_json=plan_json,
        spot_parquet=spot_parquet,
        output_dir=holdout_output_dir,
        expected_plan_sha256=expected_plan_sha256,
        spot_acquisition_summary=spot_fetch_summary,
        spot_capture_seal=capture_seal,
    )
    summary.update(
        {
            "locked_holdout_ran": True,
            "locked_holdout_output_dir": str(holdout_output_dir),
            "locked_holdout_run_summary": run_summary.get("run_summary"),
            "locked_holdout_run_summary_sha256": _sha256(Path(str(run_summary["run_summary"]))),
            "locked_holdout": run_summary,
            "holdout_pass": bool(run_summary.get("holdout_pass")),
            "status": run_summary.get("status"),
            "next_action": run_summary.get("next_action"),
        }
    )
    return _write_summary(output_dir, summary)


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    path = output_dir / "energy_charts_locked_holdout_run_summary.json"
    path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["run_summary"] = str(_resolved_path(path))
    return summary


def _write_capture_seal(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value)
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _years_before(value: datetime, years: int) -> datetime:
    if years <= 0:
        raise ValueError("lookback_years must be positive")
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bzn", default="CH")
    parser.add_argument(
        "--as-of-utc",
        default=None,
        help="Optional UTC timestamp used for pre-window guard testing. Defaults to current UTC time.",
    )
    args = parser.parse_args(argv)
    summary = run_energy_charts_locked_holdout(
        plan_json=args.plan_json,
        expected_plan_sha256=args.expected_plan_sha256,
        output_dir=args.output_dir,
        bzn=args.bzn,
        as_of_utc=args.as_of_utc,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary.get("status") == "LOCKED_HOLDOUT_PASS" and summary.get("holdout_pass") is True else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

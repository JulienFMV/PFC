"""Fetch observed Energy Charts spot and run a locked EPEX holdout if ready.

This operator wrapper is fail-closed: it first verifies the locked plan hash,
then fetches the full pre-registered holdout window as observed hourly spot.
If Energy Charts has not published every expected hour, no spot parquet is
written and the locked holdout runner is not called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.fetch_energy_charts_epex_spot_hourly import fetch_hourly_spot
from scripts.run_epex_lab_locked_holdout import run_locked_holdout


SUMMARY_SCHEMA_VERSION = "energy_charts_epex_locked_holdout_run.v1"
PLAN_SCHEMA_VERSION = "epex_lab_locked_holdout_plan.v1"


def run_energy_charts_locked_holdout(
    *,
    plan_json: Path,
    expected_plan_sha256: str,
    output_dir: Path,
    bzn: str = "CH",
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

    spot_parquet = output_dir / "energy_charts_epex_spot_hourly.parquet"
    spot_fetch_summary = output_dir / "energy_charts_epex_spot_fetch_summary.json"
    fetch_summary = fetch_hourly_spot(
        start=str(plan["holdout_start_utc"]),
        end=str(plan["holdout_end_utc"]),
        bzn=bzn,
        output_parquet=spot_parquet,
        summary_json=spot_fetch_summary,
        allow_partial=False,
    )
    summary.update(
        {
            "spot_fetch_ran": True,
            "spot_fetch_summary": str(spot_fetch_summary),
            "spot_fetch_summary_sha256": _sha256(spot_fetch_summary),
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

    holdout_output_dir = output_dir / "locked_holdout_runner"
    run_summary = run_locked_holdout(
        plan_json=plan_json,
        spot_parquet=spot_parquet,
        output_dir=holdout_output_dir,
        expected_plan_sha256=expected_plan_sha256,
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
    args = parser.parse_args(argv)
    summary = run_energy_charts_locked_holdout(
        plan_json=args.plan_json,
        expected_plan_sha256=args.expected_plan_sha256,
        output_dir=args.output_dir,
        bzn=args.bzn,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary.get("status") == "LOCKED_HOLDOUT_PASS" and summary.get("holdout_pass") is True else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

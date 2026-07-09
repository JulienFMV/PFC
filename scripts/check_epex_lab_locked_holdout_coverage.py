"""Check whether a locked EPEX lab holdout has enough spot coverage to run.

This helper reads a locked holdout plan and a candidate future EPEX spot parquet
and reports whether the full pre-registered holdout window is covered. It does
not run the holdout backtest and does not approve production promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


PLAN_SCHEMA_VERSION = "epex_lab_locked_holdout_plan.v1"


def check_coverage(
    *,
    plan_json: Path,
    spot_parquet: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    plan = _read_json(plan_json)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError(f"plan must be {PLAN_SCHEMA_VERSION}")
    start = _to_utc(plan["holdout_start_utc"])
    end = _to_utc(plan["holdout_end_utc"])
    if end <= start:
        raise ValueError("holdout_end_utc must be after holdout_start_utc")
    spot = _load_spot_index(spot_parquet)
    expected = pd.date_range(start, end, freq="h", inclusive="left")
    observed = spot[(spot >= start) & (spot < end)]
    observed_unique = pd.DatetimeIndex(observed.unique()).sort_values()
    missing = expected.difference(observed_unique)
    extra_duplicates = int(len(observed) - len(observed_unique))
    criteria = plan.get("pass_criteria") or {}
    min_hours = int(criteria.get("min_holdout_hours", len(expected)))
    coverage = {
        "schema_version": "epex_lab_locked_holdout_coverage.v1",
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "plan_json": str(plan_json),
        "spot_parquet": str(spot_parquet),
        "spot_parquet_sha256": _sha256(spot_parquet),
        "plan_id": plan.get("plan_id"),
        "holdout_start_utc": _iso(start),
        "holdout_end_utc": _iso(end),
        "expected_holdout_hours": int(len(expected)),
        "observed_holdout_hours": int(len(observed_unique)),
        "duplicate_holdout_rows": extra_duplicates,
        "missing_holdout_hours": int(len(missing)),
        "first_missing_holdout_utc": _iso(missing[0]) if len(missing) else None,
        "last_missing_holdout_utc": _iso(missing[-1]) if len(missing) else None,
        "min_required_holdout_hours": min_hours,
        "spot_min_utc": _iso(spot.min()) if len(spot) else None,
        "spot_max_utc": _iso(spot.max()) if len(spot) else None,
        "checks": {
            "plan_no_ompex": plan.get("ompex_used_in_model") is False
            and plan.get("ompex_used_in_selection") is False
            and plan.get("ompex_used_in_backtest") is False,
            "spot_parquet_non_empty": bool(len(spot)),
            "full_window_covered": int(len(missing)) == 0,
            "min_holdout_hours_met": int(len(observed_unique)) >= min_hours,
            "no_duplicate_holdout_rows": extra_duplicates == 0,
        },
    }
    checks = coverage["checks"]
    ready = bool(
        checks["plan_no_ompex"]
        and checks["spot_parquet_non_empty"]
        and checks["full_window_covered"]
        and checks["min_holdout_hours_met"]
        and checks["no_duplicate_holdout_rows"]
    )
    coverage["ready_to_run_backtest"] = ready
    coverage["status"] = "READY_TO_RUN_HOLDOUT_BACKTEST" if ready else "WAITING_FOR_FULL_SPOT_COVERAGE"
    coverage["next_action"] = (
        "Run the locked holdout backtest and audit commands from the plan."
        if ready
        else "Refresh the EPEX spot parquet after the holdout window is complete, then rerun this coverage check."
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_jsonable(coverage), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return coverage


def _load_spot_index(path: Path) -> pd.DatetimeIndex:
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("spot parquet must use a DatetimeIndex")
    idx = frame.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return pd.DatetimeIndex(idx).sort_values()


def _to_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _iso(value: Any) -> str:
    if value is None:
        return None
    return _to_utc(value).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    parser.add_argument("--spot-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = check_coverage(plan_json=args.plan_json, spot_parquet=args.spot_parquet, output=args.output)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary.get("ready_to_run_backtest") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())

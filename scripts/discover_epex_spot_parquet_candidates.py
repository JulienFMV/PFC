"""Discover candidate EPEX spot parquet files for a locked holdout run.

This helper is read-only. It scans one or more local roots for parquet files
that look like EPEX spot inputs, ranks them by latest timestamp, and writes
operator-ready commands for the locked holdout coverage check and runner. It
does not run a holdout backtest and does not approve production promotion.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


SPOT_PRICE_COLUMN = "price_eur_mwh"


def discover_candidates(
    *,
    plan_json: Path,
    search_roots: list[Path],
    output_json: Path,
    max_candidates: int = 20,
    pattern: str = "*.parquet",
    require_hourly_grid: bool = True,
) -> dict[str, Any]:
    if max_candidates <= 0:
        raise ValueError("--max-candidates must be positive")
    plan_json = plan_json.expanduser().resolve(strict=False)
    plan = _read_json(plan_json)
    holdout_start = _to_utc(plan["holdout_start_utc"])
    holdout_end = _to_utc(plan["holdout_end_utc"])
    expected = pd.date_range(holdout_start, holdout_end, freq="h", inclusive="left")
    latest_required = expected[-1] if len(expected) else None
    plan_sha = _sha256(plan_json)

    rows = []
    statuses = []
    scanned_file_count = 0
    for root in search_roots:
        root = root.expanduser().resolve(strict=False)
        if not root.exists():
            continue
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            scanned_file_count += 1
            status = _spot_status(
                path,
                expected=expected,
                latest_required=latest_required,
                require_hourly_grid=require_hourly_grid,
            )
            statuses.append(status)
            if status["is_candidate"]:
                rows.append(status)

    rows.sort(
        key=lambda row: (
            row["spot_max_utc"] or "",
            row["last_write_time_utc"] or "",
            row["path"],
        ),
        reverse=True,
    )
    limited = rows[:max_candidates]
    best = limited[0] if limited else None
    output_json = output_json.expanduser().resolve(strict=False)
    coverage_output = output_json.with_name("coverage_status_from_discovered_spot.json")
    run_output_dir = output_json.with_name("locked_holdout_runner_from_discovered_spot")
    summary = {
        "schema_version": "epex_spot_parquet_discovery.v1",
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "plan_json": str(plan_json),
        "plan_json_sha256": plan_sha,
        "plan_id": plan.get("plan_id"),
        "holdout_start_utc": _iso(holdout_start),
        "holdout_end_utc": _iso(holdout_end),
        "latest_required_holdout_utc": _iso(latest_required),
        "search_roots": [str(path.expanduser().resolve(strict=False)) for path in search_roots],
        "pattern": pattern,
        "require_hourly_grid": require_hourly_grid,
        "scanned_file_count": scanned_file_count,
        "candidate_count": len(rows),
        "rejected_file_count": int(len(statuses) - len(rows)),
        "spot_like_rejected_file_count": int(
            sum(
                1
                for status in statuses
                if not status["is_candidate"] and status["has_price_column"] and status["datetime_index"]
            )
        ),
        "rejection_reason_counts": _rejection_reason_counts(statuses),
        "returned_candidate_count": len(limited),
        "best_candidate": best,
        "candidates": limited,
        "recommended_commands": _recommended_commands(
            plan_json=plan_json,
            plan_sha=plan_sha,
            best=best,
            coverage_output=coverage_output,
            run_output_dir=run_output_dir,
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _rejection_reason_counts(statuses: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(status.get("rejection_reason") or "unknown")
        for status in statuses
        if not bool(status.get("is_candidate"))
    )
    return dict(sorted(counts.items()))


def _spot_status(
    path: Path,
    *,
    expected: pd.DatetimeIndex,
    latest_required: pd.Timestamp | None,
    require_hourly_grid: bool,
) -> dict[str, Any]:
    base = {
        "path": str(path.expanduser().resolve(strict=False)),
        "last_write_time_utc": _iso(pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")),
        "is_candidate": False,
        "rejection_reason": None,
        "spot_min_utc": None,
        "spot_max_utc": None,
        "row_count": None,
        "has_price_column": False,
        "datetime_index": False,
        "hourly_grid_compatible": False,
        "spot_hours_until_latest_required_holdout": None,
        "expected_holdout_hours": int(len(expected)),
        "observed_holdout_hours": 0,
        "missing_holdout_hours": int(len(expected)),
        "first_missing_holdout_utc": _iso(expected[0]) if len(expected) else None,
        "last_missing_holdout_utc": _iso(expected[-1]) if len(expected) else None,
        "full_window_covered": False,
    }
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - read-only discovery should skip bad files
        return {**base, "rejection_reason": f"parquet_read_failed:{type(exc).__name__}"}
    if not isinstance(frame.index, pd.DatetimeIndex):
        return {**base, "rejection_reason": "index_not_datetime", "row_count": int(len(frame))}
    has_price = SPOT_PRICE_COLUMN in frame.columns
    idx = frame.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    idx = pd.DatetimeIndex(idx).sort_values()
    spot_min = idx.min() if len(idx) else None
    spot_max = idx.max() if len(idx) else None
    coverage = _holdout_coverage(idx, expected)
    hourly_grid = _hourly_grid_compatible(idx)
    is_candidate = bool(has_price and len(idx) and (hourly_grid or not require_hourly_grid))
    rejection_reason = None
    if not has_price or not len(idx):
        rejection_reason = "missing_price_column_or_empty"
    elif require_hourly_grid and not hourly_grid:
        rejection_reason = "non_hourly_grid"
    return {
        **base,
        "is_candidate": is_candidate,
        "rejection_reason": rejection_reason,
        "spot_min_utc": _iso(spot_min),
        "spot_max_utc": _iso(spot_max),
        "row_count": int(len(frame)),
        "has_price_column": has_price,
        "datetime_index": True,
        "hourly_grid_compatible": hourly_grid,
        "spot_hours_until_latest_required_holdout": (
            _positive_hours_between(spot_max, latest_required)
            if spot_max is not None and latest_required is not None
            else None
        ),
        **coverage,
    }


def _holdout_coverage(index: pd.DatetimeIndex, expected: pd.DatetimeIndex) -> dict[str, Any]:
    if not len(expected):
        return {
            "expected_holdout_hours": 0,
            "observed_holdout_hours": 0,
            "missing_holdout_hours": 0,
            "first_missing_holdout_utc": None,
            "last_missing_holdout_utc": None,
            "full_window_covered": True,
        }
    observed = pd.DatetimeIndex(index.unique()).intersection(expected)
    missing = expected.difference(observed)
    return {
        "expected_holdout_hours": int(len(expected)),
        "observed_holdout_hours": int(len(observed)),
        "missing_holdout_hours": int(len(missing)),
        "first_missing_holdout_utc": _iso(missing[0]) if len(missing) else None,
        "last_missing_holdout_utc": _iso(missing[-1]) if len(missing) else None,
        "full_window_covered": len(missing) == 0,
    }


def _hourly_grid_compatible(index: pd.DatetimeIndex) -> bool:
    if not len(index):
        return False
    return bool(
        (
            (index.minute == 0)
            & (index.second == 0)
            & (index.microsecond == 0)
            & (index.nanosecond == 0)
        ).all()
    )


def _recommended_commands(
    *,
    plan_json: Path,
    plan_sha: str,
    best: dict[str, Any] | None,
    coverage_output: Path,
    run_output_dir: Path,
) -> dict[str, str | None]:
    if best is None:
        return {"coverage_check": None, "run_locked_holdout": None}
    spot = Path(str(best["path"]))
    return {
        "coverage_check": _join_command(
            [
                "python",
                "scripts/check_epex_lab_locked_holdout_coverage.py",
                "--plan-json",
                str(plan_json),
                "--spot-parquet",
                str(spot),
                "--output",
                str(coverage_output),
            ]
        ),
        "run_locked_holdout": _join_command(
            [
                "python",
                "scripts/run_epex_lab_locked_holdout.py",
                "--plan-json",
                str(plan_json),
                "--expected-plan-sha256",
                plan_sha,
                "--spot-parquet",
                str(spot),
                "--output-dir",
                str(run_output_dir),
            ]
        ),
    }


def _join_command(parts: list[str]) -> str:
    return " ".join(_quote(part) for part in parts)


def _quote(value: str) -> str:
    if not value or any(ch.isspace() for ch in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def _positive_hours_between(left: Any, right: Any) -> float:
    delta_hours = (_to_utc(right) - _to_utc(left)).total_seconds() / 3600.0
    return float(max(0.0, delta_hours))


def _to_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _iso(value: Any) -> str | None:
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
    parser.add_argument("--search-root", type=Path, action="append", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--pattern", default="*.parquet")
    parser.add_argument(
        "--include-non-hourly-spot",
        action="store_true",
        help="Also return parquet files whose DatetimeIndex is not on exact hourly timestamps.",
    )
    args = parser.parse_args(argv)
    summary = discover_candidates(
        plan_json=args.plan_json,
        search_roots=args.search_root or [Path("output/phase14")],
        output_json=args.output_json,
        max_candidates=args.max_candidates,
        pattern=args.pattern,
        require_hourly_grid=not args.include_non_hourly_spot,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if summary["candidate_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

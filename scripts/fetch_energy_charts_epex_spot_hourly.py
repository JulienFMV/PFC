"""Fetch Energy Charts EPEX prices as an observed hourly spot parquet.

This helper is intended for locked holdout coverage checks. It does not update
the project EPEX caches and it never forward-fills beyond timestamps returned
by Energy Charts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.data.ingest_energy_charts import API_BASE, _retry_get


PRICE_COLUMN = "price_eur_mwh"
SUMMARY_SCHEMA_VERSION = "energy_charts_epex_spot_hourly_fetch.v1"


def fetch_hourly_spot(
    *,
    start: str,
    end: str,
    bzn: str,
    output_parquet: Path,
    summary_json: Path | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    start_utc = _to_utc(start)
    end_utc = _to_utc(end)
    if end_utc <= start_utc:
        raise ValueError("end must be after start")

    api_start = _api_start_date(start_utc)
    api_end = _api_end_date(end_utc)
    output_parquet = output_parquet.resolve()
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "source": "energy-charts.info",
        "bzn": bzn,
        "start_utc": _iso(start_utc),
        "end_utc": _iso(end_utc),
        "api_start": api_start,
        "api_end": api_end,
        "output_parquet": str(output_parquet),
        "output_parquet_sha256": None,
    }
    try:
        raw = _fetch_raw_prices(start=api_start, end=api_end, bzn=bzn)
    except Exception as exc:
        summary.update(
            {
                "status": "SPOT_FETCH_ERROR",
                "full_window_covered": False,
                "fetch_error": f"{type(exc).__name__}: {exc}",
                **_empty_coverage(start_utc=start_utc, end_utc=end_utc),
                "next_action": "Refresh after Energy Charts publishes the requested window or verify API availability.",
            }
        )
        _write_summary(summary_json, summary)
        return summary

    hourly, coverage = _build_hourly_spot(
        raw,
        start_utc=start_utc,
        end_utc=end_utc,
    )
    status = "OK" if coverage["full_window_covered"] else "PARTIAL_COVERAGE"
    summary.update({"status": status, **coverage})

    if not coverage["full_window_covered"] and not allow_partial:
        summary["next_action"] = "Choose a narrower end date or refresh after Energy Charts publishes the missing hours."
        _write_summary(summary_json, summary)
        return summary

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(output_parquet, engine="pyarrow", compression="snappy")
    summary["output_parquet_sha256"] = _sha256(output_parquet)
    summary["next_action"] = (
        "Run the locked holdout coverage check with this parquet."
        if coverage["full_window_covered"]
        else "Use only for discovery diagnostics; do not run a locked holdout until full coverage exists."
    )
    _write_summary(summary_json, summary)
    return summary


def _fetch_raw_prices(*, start: str, end: str, bzn: str) -> pd.DataFrame:
    resp = _retry_get(f"{API_BASE}/price", params={"bzn": bzn, "start": start, "end": end})
    data = resp.json()
    timestamps = data.get("unix_seconds", [])
    prices = data.get("price", [])
    if len(timestamps) != len(prices):
        raise ValueError("Energy Charts timestamp and price arrays have different lengths")
    if not timestamps:
        return pd.DataFrame(columns=[PRICE_COLUMN], index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
            PRICE_COLUMN: prices,
        }
    ).set_index("timestamp")
    frame = frame.dropna(subset=[PRICE_COLUMN])
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def _empty_coverage(*, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> dict[str, Any]:
    expected = pd.date_range(start_utc, end_utc, freq="h", inclusive="left")
    latest_required = expected[-1] if len(expected) else None
    return {
        "expected_hour_count": int(len(expected)),
        "observed_hour_count": 0,
        "missing_hour_count": int(len(expected)),
        "first_missing_utc": _iso(expected[0]) if len(expected) else None,
        "last_missing_utc": _iso(expected[-1]) if len(expected) else None,
        "spot_min_utc": None,
        "spot_max_utc": None,
        "latest_required_utc": _iso(latest_required),
        "spot_hours_until_latest_required": None,
        "raw_observation_count": 0,
    }


def _build_hourly_spot(
    raw: pd.DataFrame,
    *,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if PRICE_COLUMN not in raw.columns:
        raise ValueError(f"raw prices must contain {PRICE_COLUMN}")
    raw = raw.copy()
    raw.index = _as_utc_index(raw.index)
    raw = raw.sort_index()
    raw = raw[(raw.index >= start_utc) & (raw.index < end_utc)]
    raw[PRICE_COLUMN] = pd.to_numeric(raw[PRICE_COLUMN], errors="coerce")
    raw = raw[np.isfinite(raw[PRICE_COLUMN].to_numpy(dtype="float64", na_value=np.nan))]

    expected = pd.date_range(start_utc, end_utc, freq="h", inclusive="left")
    if raw.empty:
        hourly = pd.DataFrame(columns=[PRICE_COLUMN], index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    else:
        hourly = raw[[PRICE_COLUMN]].groupby(raw.index.floor("h")).mean()
        hourly.index = _as_utc_index(hourly.index)
        hourly.index.name = "timestamp"
        hourly = hourly[(hourly.index >= start_utc) & (hourly.index < end_utc)]
        hourly = hourly[~hourly.index.duplicated(keep="last")].sort_index()

    observed = pd.DatetimeIndex(hourly.index.unique()).sort_values()
    missing = expected.difference(observed)
    latest_required = expected[-1] if len(expected) else None
    spot_max = observed[-1] if len(observed) else None
    lag = None
    if latest_required is not None and spot_max is not None:
        lag = max(0.0, (latest_required - spot_max) / pd.Timedelta(hours=1))
    coverage = {
        "expected_hour_count": int(len(expected)),
        "observed_hour_count": int(len(observed)),
        "missing_hour_count": int(len(missing)),
        "first_missing_utc": _iso(missing[0]) if len(missing) else None,
        "last_missing_utc": _iso(missing[-1]) if len(missing) else None,
        "spot_min_utc": _iso(observed[0]) if len(observed) else None,
        "spot_max_utc": _iso(spot_max),
        "latest_required_utc": _iso(latest_required),
        "spot_hours_until_latest_required": lag,
        "full_window_covered": bool(len(missing) == 0),
        "raw_observation_count": int(len(raw)),
    }
    return hourly, coverage


def _to_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _as_utc_index(index: pd.Index) -> pd.DatetimeIndex:
    dt_index = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    return dt_index.tz_convert("UTC")


def _api_start_date(start_utc: pd.Timestamp) -> str:
    return start_utc.date().isoformat()


def _api_end_date(end_utc: pd.Timestamp) -> str:
    if end_utc == end_utc.floor("D"):
        return end_utc.date().isoformat()
    return (end_utc.floor("D") + pd.Timedelta(days=1)).date().isoformat()


def _iso(ts: pd.Timestamp | None) -> str | None:
    if ts is None:
        return None
    return ts.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_summary(summary_json: Path | None, summary: dict[str, Any]) -> None:
    if summary_json is None:
        return
    summary_json = summary_json.resolve()
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC start date/time, inclusive")
    parser.add_argument("--end", required=True, help="UTC end date/time, exclusive")
    parser.add_argument("--bzn", default="CH", help="Energy Charts bidding zone")
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write partial observed hours instead of failing closed on missing hours.",
    )
    args = parser.parse_args(argv)

    summary = fetch_hourly_spot(
        start=args.start,
        end=args.end,
        bzn=args.bzn,
        output_parquet=args.output_parquet,
        summary_json=args.summary_json,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["full_window_covered"] or args.allow_partial else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

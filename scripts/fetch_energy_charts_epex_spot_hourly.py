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
from datetime import datetime, timezone
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
    provider_raw_json: Path | None = None,
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
        "acquired_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "bzn": bzn,
        "start_utc": _iso(start_utc),
        "end_utc": _iso(end_utc),
        "api_start": api_start,
        "api_end": api_end,
        "output_parquet": str(output_parquet),
        "output_parquet_sha256": None,
        "provider_raw_json": str(provider_raw_json.resolve()) if provider_raw_json is not None else None,
        "provider_raw_json_sha256": None,
    }
    try:
        raw = _fetch_raw_prices(start=api_start, end=api_end, bzn=bzn)
        if provider_raw_json is not None:
            raw_payload = raw.attrs.get("provider_raw_response_bytes")
            if not isinstance(raw_payload, bytes) or not raw_payload:
                raise ValueError("provider raw response bytes are unavailable")
            provider_raw_json = provider_raw_json.resolve()
            _write_immutable_bytes(provider_raw_json, raw_payload)
            summary["provider_raw_json"] = str(provider_raw_json)
            summary["provider_raw_json_sha256"] = _sha256(provider_raw_json)
        raw.attrs.pop("provider_raw_response_bytes", None)
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
    raw_response = bytes(resp.content)
    frame = _raw_frame_from_provider_bytes(raw_response)
    frame.attrs["provider_raw_response_bytes"] = raw_response
    return frame


def replay_provider_raw_hourly(
    provider_raw_json: Path,
    *,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild the exact derived hourly frame from retained provider bytes."""

    return replay_provider_raw_hourly_bytes(
        provider_raw_json.read_bytes(),
        start=start,
        end=end,
    )


def replay_provider_raw_hourly_bytes(
    provider_raw_bytes: bytes,
    *,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild hourly spot from caller-captured immutable provider bytes."""

    start_utc = _to_utc(start)
    end_utc = _to_utc(end)
    raw = _raw_frame_from_provider_bytes(provider_raw_bytes)
    return _build_hourly_spot(raw, start_utc=start_utc, end_utc=end_utc)


def _raw_frame_from_provider_bytes(raw_response: bytes) -> pd.DataFrame:
    data = json.loads(raw_response.decode("utf-8-sig"))
    timestamps = data.get("unix_seconds", [])
    prices = data.get("price", [])
    if not isinstance(timestamps, list) or not isinstance(prices, list):
        raise ValueError("Energy Charts timestamp and price arrays are invalid")
    if len(timestamps) != len(prices):
        raise ValueError("Energy Charts timestamp and price arrays have different lengths")
    if not timestamps:
        return pd.DataFrame(
            columns=[PRICE_COLUMN],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in timestamps):
        raise ValueError("Energy Charts timestamps must be integer epochs")
    numeric = pd.to_numeric(pd.Series(prices), errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype="float64", na_value=np.nan)).all():
        raise ValueError("Energy Charts prices must be finite numeric values")
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
            PRICE_COLUMN: numeric.to_numpy(dtype="float64"),
        }
    ).set_index("timestamp")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Energy Charts timestamps must be ordered and unique")
    if len(frame) == 1:
        cadence = 3600 if frame.index[0] == frame.index[0].floor("h") else None
    else:
        deltas = np.diff(frame.index.asi8) // 1_000_000_000
        unique = {int(value) for value in deltas}
        cadence = unique.pop() if len(unique) == 1 else None
    if cadence not in {900, 3600}:
        raise ValueError("Energy Charts source cadence is unsupported or incomplete")
    frame.attrs["source_cadence_seconds"] = cadence
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
        "source_cadence_seconds": None,
        "incomplete_source_observation_count": 0,
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
    declared_cadence = raw.attrs.get("source_cadence_seconds")
    raw.index = _as_utc_index(raw.index)
    raw = raw.sort_index()
    raw = raw[(raw.index >= start_utc) & (raw.index < end_utc)]
    raw[PRICE_COLUMN] = pd.to_numeric(raw[PRICE_COLUMN], errors="coerce")
    raw = raw[np.isfinite(raw[PRICE_COLUMN].to_numpy(dtype="float64", na_value=np.nan))]

    expected = pd.date_range(start_utc, end_utc, freq="h", inclusive="left")
    source_cadence = _source_cadence_seconds(
        raw,
        declared=declared_cadence,
        start_utc=start_utc,
        end_utc=end_utc,
    )
    if raw.empty:
        hourly = pd.DataFrame(columns=[PRICE_COLUMN], index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    elif source_cadence == 3600:
        valid = raw.index == raw.index.floor("h")
        hourly = raw.loc[valid, [PRICE_COLUMN]].copy()
        hourly.index.name = "timestamp"
    else:
        quarter_offsets = raw.index.minute.isin([0, 15, 30, 45]) & (
            raw.index.second == 0
        )
        eligible = raw.loc[quarter_offsets, [PRICE_COLUMN]].copy()
        groups = eligible.groupby(eligible.index.floor("h"))
        complete_hours = groups.size()
        complete_hours = complete_hours[complete_hours == 4].index
        hourly = groups.mean().loc[complete_hours]
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
        "source_cadence_seconds": source_cadence,
        "incomplete_source_observation_count": int(
            len(raw) - len(hourly) * (4 if source_cadence == 900 else 1)
        ),
    }
    return hourly, coverage


def _source_cadence_seconds(
    raw: pd.DataFrame,
    *,
    declared: object,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> int:
    if declared is not None:
        if declared not in {900, 3600}:
            raise ValueError("raw source cadence declaration is invalid")
        return int(declared)
    if raw.empty:
        return 3600
    if len(raw) == 1:
        if (
            end_utc - start_utc == pd.Timedelta(hours=1)
            and raw.index[0] == start_utc
            and raw.index[0] == raw.index[0].floor("h")
        ):
            return 3600
        return 3600
    if bool((raw.index == raw.index.floor("h")).all()):
        return 3600
    if bool(
        (
            raw.index.minute.isin([0, 15, 30, 45])
            & (raw.index.second == 0)
        ).all()
    ):
        return 900
    raise ValueError("raw source cadence cannot be classified")


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


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite different provider-raw evidence: {path}")
        return
    with path.open("xb") as handle:
        handle.write(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC start date/time, inclusive")
    parser.add_argument("--end", required=True, help="UTC end date/time, exclusive")
    parser.add_argument("--bzn", default="CH", help="Energy Charts bidding zone")
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--provider-raw-json", type=Path)
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
        provider_raw_json=args.provider_raw_json,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["full_window_covered"] or args.allow_partial else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

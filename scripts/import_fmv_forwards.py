"""scripts/import_fmv_forwards.py — build the Phase 10 forwards parquet from a
real EEX historical export (FMV office machine, H:\ share)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.data.ingest_forwards import load_forwards_timeseries

EXPECTED_VINTAGES = [
    pd.Timestamp(s, tz="UTC")
    for s in [
        "2024-01-31 17:00:00", "2024-02-29 17:00:00", "2024-03-29 17:00:00",
        "2024-04-30 16:00:00", "2024-05-31 16:00:00", "2024-06-28 16:00:00",
        "2024-07-31 16:00:00", "2024-08-30 16:00:00", "2024-09-30 16:00:00",
        "2024-10-31 17:00:00", "2024-11-29 17:00:00", "2024-12-31 17:00:00",
        "2025-01-31 17:00:00", "2025-02-28 17:00:00", "2025-03-31 16:00:00",
        "2025-04-30 16:00:00", "2025-05-30 16:00:00", "2025-06-30 16:00:00",
        "2025-07-31 16:00:00", "2025-08-29 16:00:00", "2025-09-30 16:00:00",
        "2025-10-31 17:00:00", "2025-11-28 17:00:00", "2025-12-31 17:00:00",
    ]
]


def build_forwards_frame(input_path, market, source_tag):
    ts = load_forwards_timeseries(input_path, market=market, include_week=False)
    base = ts[ts["load_type"] == "BASE"].copy()
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    sessions = pd.DatetimeIndex(sorted(base["date"].unique()))

    rows, empty = [], []
    for vintage in EXPECTED_VINTAGES:
        eligible = sessions[sessions <= pd.Timestamp(vintage.date())]
        if len(eligible) == 0:
            empty.append(str(vintage.date())); continue
        snap = base[base["date"] == eligible.max()]
        if snap.empty:
            empty.append(str(vintage.date())); continue
        rows.append(pd.DataFrame({
            "vintage": vintage,
            "key": snap["product"].astype(str).to_numpy(),
            "price": snap["price"].astype("float64").to_numpy(),
            "forwards_source": source_tag,
        }))

    if empty:
        raise ValueError(f"No EEX session at/before vintages (sheet={market}): {empty}")

    out = pd.concat(rows, ignore_index=True)
    return out.drop_duplicates(subset=["vintage", "key"], keep="last").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--market", default="CH")
    ap.add_argument("--output", default="data/forwards_history_phase10.parquet")
    ap.add_argument("--source-tag", default="real_eex_xlsx")
    args = ap.parse_args()

    out = build_forwards_frame(args.input, args.market, args.source_tag)

    found, expected = set(out["vintage"].unique()), set(EXPECTED_VINTAGES)
    if found != expected:
        raise ValueError(f"Vintage mismatch: missing={sorted(expected-found)} extra={sorted(found-expected)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    pv = out.groupby("vintage").size()
    print(f"Wrote {len(out)} rows to {output_path}")
    print(f"  Vintages       : {out['vintage'].nunique()} (expected 24)")
    print(f"  Keys/vintage   : min={pv.min()} max={pv.max()} mean={pv.mean():.1f}")
    print(f"  Price EUR/MWh  : min={out['price'].min():.1f} max={out['price'].max():.1f} mean={out['price'].mean():.1f}")
    print(f"  forwards_source: {out['forwards_source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()

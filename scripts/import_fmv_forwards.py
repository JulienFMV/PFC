"""scripts/import_fmv_forwards.py — build the Phase 10 forwards parquet from a
real EEX historical export (FMV office machine, H:\\ share).

Reuses the battle-tested parser ``pfc_shaping.data.ingest_forwards.
load_forwards_timeseries`` to read the EEX workbook (one sheet per market,
product codes Y01_/Q__/M___BASE in row 1, daily marks from row 4), then
snapshots, for each of the 24 hard-coded scorecard vintages, the BASE
Cal/Quarter/Month curve quoted at the last trading session on or before the
vintage's calendar date.

Output schema (consumed by pfc_shaping.validation.scorecard):
    vintage          datetime64[ns, UTC]
    key              str   ('2025' | '2025-Q1' | '2025-03')
    price            float64  (EUR/MWh, BASE)
    forwards_source  str   (default 'real_eex_xlsx' -> SC#1 gate-eligible)

Usage:
    python scripts/import_fmv_forwards.py \
        --input "H:\\...\\Price_Report_EEX_CH_DE_Hist.xlsx" \
        --market CH
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.data.ingest_forwards import load_forwards_timeseries

# 24 canonical vintage timestamps (UTC), last EEX session of each month.
# Must match pfc_shaping.validation.scorecard.list_vintages_2024_2025 exactly.
EXPECTED_VINTAGES: list[pd.Timestamp] = [
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


def build_forwards_frame(
    input_path: str | Path,
    market: str,
    source_tag: str,
) -> pd.DataFrame:
    ts = load_forwards_timeseries(input_path, market=market, include_week=False)
    base = ts[ts["load_type"] == "BASE"].copy()
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    sessions = pd.DatetimeIndex(sorted(base["date"].unique()))

    rows: list[pd.DataFrame] = []
    empty_vintages: list[str] = []
    for vintage in EXPECTED_VINTAGES:
        cutoff = pd.Timestamp(vintage.date())  # tz-naive calendar date
        eligible = sessions[sessions <= cutoff]
        if len(eligible) == 0:
            empty_vintages.append(str(vintage.date()))
            continue
        snap_date = eligible.max()
        snap = base[base["date"] == snap_date]
        if snap.empty:
            empty_vintages.append(str(vintage.date()))
            continue
        rows.append(
            pd.DataFrame(
                {
                    "vintage": vintage,
                    "key": snap["product"].astype(str).to_numpy(),
                    "price": snap["price"].astype("float64").to_numpy(),
                    "forwards_source": source_tag,
                }
            )
        )

    if empty_vintages:
        raise ValueError(
            f"No EEX session found at/before these vintages (sheet={market}): "
            f"{empty_vintages}. Use a history file covering these dates."
        )

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["vintage", "key"], keep="last").reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="EEX historical export (xlsx)")
    ap.add_argument("--market", default="CH", help="Sheet/market name (default CH)")
    ap.add_argument(
        "--output", default="data/forwards_history_phase10.parquet",
        help="Destination parquet (default data/forwards_history_phase10.parquet)",
    )
    ap.add_argument("--source-tag", default="real_eex_xlsx", help="forwards_source value")
    args = ap.parse_args()

    out = build_forwards_frame(args.input, args.market, args.source_tag)

    found = set(out["vintage"].unique())
    expected = set(EXPECTED_VINTAGES)
    if found != expected:
        raise ValueError(
            f"Vintage set mismatch.\n  missing={sorted(expected - found)}\n"
            f"  extra={sorted(found - expected)}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    per_vintage = out.groupby("vintage").size()
    print(f"Wrote {len(out)} rows to {output_path}")
    print(f"  Market sheet     : {args.market}")
    print(f"  Vintages         : {out['vintage'].nunique()} (expected 24)")
    print(f"  Keys/vintage     : min={per_vintage.min()} max={per_vintage.max()} "
          f"mean={per_vintage.mean():.1f}")
    print(f"  Price EUR/MWh    : min={out['price'].min():.1f} max={out['price'].max():.1f} "
          f"mean={out['price'].mean():.1f}")
    print(f"  forwards_source  : {out['forwards_source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Backfill the governed Swiss CT multi-country forecast cache in chunks."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.data.ingest_entso import fetch_and_cache_multi_country_forecasts  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CT_MULTI_COUNTRY_BACKFILL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Swiss CT multi-country forecasts from ENTSO-E.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD (exclusive)")
    parser.add_argument("--chunk-days", type=int, default=14, help="Chunk size in days")
    return parser.parse_args()


def iter_chunks(start: pd.Timestamp, end: pd.Timestamp, chunk_days: int):
    cursor = start
    step = pd.Timedelta(days=chunk_days)
    while cursor < end:
        nxt = min(cursor + step, end)
        yield cursor, nxt
        cursor = nxt


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    if end <= start:
        raise ValueError("end must be strictly after start")
    if args.chunk_days < 1:
        raise ValueError("chunk-days must be >= 1")

    total_rows = 0
    for chunk_start, chunk_end in iter_chunks(start, end, args.chunk_days):
        logger.info(
            "Backfilling multi-country forecasts: %s -> %s",
            chunk_start.date(),
            (chunk_end - timedelta(seconds=1)).date(),
        )
        df = fetch_and_cache_multi_country_forecasts(
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        )
        total_rows = len(df)
        logger.info(
            "  cache_rows=%d  min_ts=%s  max_ts=%s",
            len(df),
            df.index.min() if len(df) else None,
            df.index.max() if len(df) else None,
        )

    print(
        {
            "cache_rows": total_rows,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "chunk_days": args.chunk_days,
        }
    )


if __name__ == "__main__":
    main()

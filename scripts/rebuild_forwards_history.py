"""
rebuild_forwards_history.py
---------------------------
Phase 2 LT — rebuild data/eex_forwards_history.parquet from the two
canonical EEX desk files:

    Price_Report_EEX_Yearly.xlsx              # snapshot DE/CH/AT/IT/FR
    Price_Report_EEX_CH_DE_Historique2019.xlsx # depth CH/DE since 2019

The two files are ingested in priority order so that, when both contain
the same (date, product, load_type, market), the *Yearly* mark wins
(it is the most recently quoted source). Empty / non-market sheets and
Week products are filtered out automatically by ``ingest_forwards``
(Phase 1ter).

Output:
    data/eex_forwards_history.parquet   — long format with columns
                                          ``date, product, load_type,
                                          product_type, market, price``

This script is idempotent and safe to re-run. It does not call any
production pipeline; it only refreshes the historical Parquet that
``forward_proxy.load_base_prices`` reads as a longitudinal source.

Usage (PowerShell, FMV env):

    $env:PATH='C:\\Users\\jbattaglia\\.conda\\ppa_env;...' + $env:PATH
    C:\\Users\\jbattaglia\\.conda\\ppa_env\\python.exe scripts\\rebuild_forwards_history.py

Force a specific path:

    python scripts/rebuild_forwards_history.py `
      --yearly  "H:\\...\\Price_Report_EEX_Yearly.xlsx" `
      --history "H:\\...\\Price_Report_EEX_CH_DE_Historique2019.xlsx"

Skip the Yearly file (history-only refresh) or vice versa:

    python scripts/rebuild_forwards_history.py --skip-yearly
    python scripts/rebuild_forwards_history.py --skip-history
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow this script to import pfc_shaping when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.data.ingest_forwards import (  # noqa: E402
    load_forwards_timeseries,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("rebuild_forwards_history")

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_YEARLY_PATHS = [
    Path(r"H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_Yearly.xlsx"),
    Path(r"\\fmvfs1\data\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_Yearly.xlsx"),
]
DEFAULT_HISTORY_PATHS = [
    Path(r"H:\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_CH_DE_Historique2019.xlsx"),
    Path(r"\\fmvfs1\data\Energy\GeCom\MARCHE & NEGOCE\Prix\EEX - ER\Price_Report_EEX_CH_DE_Historique2019.xlsx"),
]
DEFAULT_OUT_PARQUET = REPO_ROOT / "data" / "eex_forwards_history.parquet"

# Markets to ingest per source file. Yearly carries the full panel;
# Historique2019 carries only CH/DE on its real (non-empty) sheets.
YEARLY_MARKETS = ["CH", "DE", "FR", "AT", "IT"]
HISTORY_MARKETS = ["CH", "DE"]


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        try:
            if p.exists():
                return p
        except OSError:
            continue
    return None


def _ingest_one_file(
    path: Path,
    markets: list[str],
    source_label: str,
) -> pd.DataFrame:
    """Ingest one EEX XLSX into a DataFrame in long format.

    Empty market sheets (FR/AT/IT in the historical workbook) raise
    inside ``load_forwards_timeseries`` and are skipped here without
    aborting the run.
    """
    frames: list[pd.DataFrame] = []
    for mkt in markets:
        try:
            ts = load_forwards_timeseries(path, market=mkt)
        except Exception as exc:
            logger.warning(
                "  [%s] sheet=%s: %s",
                source_label, mkt, exc,
            )
            continue
        ts["market"] = mkt
        ts["source"] = source_label
        frames.append(ts)
        logger.info(
            "  [%s] sheet=%s: %d rows, %d products, %s -> %s",
            source_label, mkt, len(ts), ts["product"].nunique(),
            ts["date"].min().date(), ts["date"].max().date(),
        )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def rebuild(
    yearly: Path | None,
    history: Path | None,
    out_parquet: Path,
) -> pd.DataFrame:
    """Combine Yearly + Historique into a single deduplicated Parquet."""
    if yearly is None and history is None:
        raise ValueError(
            "At least one of --yearly / --history must point at an existing file."
        )

    pieces: list[pd.DataFrame] = []

    if history is not None:
        logger.info("Ingesting historical workbook: %s", history)
        hist_df = _ingest_one_file(history, HISTORY_MARKETS, source_label="HISTORIQUE2019")
        if not hist_df.empty:
            pieces.append(hist_df)
        else:
            logger.warning("Historical workbook produced 0 rows.")

    if yearly is not None:
        logger.info("Ingesting yearly snapshot workbook: %s", yearly)
        yr_df = _ingest_one_file(yearly, YEARLY_MARKETS, source_label="YEARLY")
        if not yr_df.empty:
            pieces.append(yr_df)
        else:
            logger.warning("Yearly workbook produced 0 rows.")

    if not pieces:
        raise ValueError("No data ingested from any source — nothing to write.")

    combined = pd.concat(pieces, ignore_index=True)

    # Dedup priority: Yearly > Historique. We append Yearly *after*
    # Historique above, so ``keep="last"`` retains the Yearly mark on
    # any conflicting (date, product, load_type, market) tuple.
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["date", "product", "load_type", "market"],
        keep="last",
    )
    after = len(combined)
    if before != after:
        logger.info("Deduplicated %d overlapping rows (Yearly wins).", before - after)

    combined = combined.sort_values(
        ["market", "date", "product", "load_type"]
    ).reset_index(drop=True)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_parquet, index=False)
    logger.info("Wrote %s (%d rows)", out_parquet, len(combined))

    _log_summary(combined)
    return combined


def _log_summary(df: pd.DataFrame) -> None:
    """Per-market coverage report."""
    logger.info("---- Coverage summary ----")
    for mkt in sorted(df["market"].unique()):
        mkt_df = df[df["market"] == mkt]
        n_rows = len(mkt_df)
        n_products = mkt_df["product"].nunique()
        date_min = mkt_df["date"].min().date()
        date_max = mkt_df["date"].max().date()
        n_neg = int((mkt_df["price"] < 0).sum())
        ptype_counts = mkt_df["product_type"].value_counts().to_dict()
        logger.info(
            "  %-3s | %6d rows | %3d products | %s -> %s | neg=%d | %s",
            mkt, n_rows, n_products, date_min, date_max, n_neg, ptype_counts,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yearly",
        type=Path,
        default=None,
        help="Path to Price_Report_EEX_Yearly.xlsx (default: probe H: and UNC).",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="Path to Price_Report_EEX_CH_DE_Historique2019.xlsx (default: probe H: and UNC).",
    )
    parser.add_argument(
        "--skip-yearly",
        action="store_true",
        help="Don't ingest the Yearly snapshot workbook.",
    )
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="Don't ingest the Historique2019 workbook.",
    )
    parser.add_argument(
        "--out-parquet",
        type=Path,
        default=DEFAULT_OUT_PARQUET,
        help=f"Destination Parquet (default: {DEFAULT_OUT_PARQUET}).",
    )
    args = parser.parse_args()

    yearly = None if args.skip_yearly else (args.yearly or _first_existing(DEFAULT_YEARLY_PATHS))
    history = None if args.skip_history else (args.history or _first_existing(DEFAULT_HISTORY_PATHS))

    if yearly is None and not args.skip_yearly:
        logger.warning(
            "Yearly file not found in defaults; pass --yearly or use --skip-yearly."
        )
    if history is None and not args.skip_history:
        logger.warning(
            "Historical file not found in defaults; pass --history or use --skip-history."
        )

    rebuild(yearly=yearly, history=history, out_parquet=args.out_parquet)
    return 0


if __name__ == "__main__":
    sys.exit(main())

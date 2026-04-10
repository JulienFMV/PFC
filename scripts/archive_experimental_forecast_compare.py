#!/usr/bin/env python3
"""
Archive the latest experimental forecast comparison with dated JSON and a cumulative CSV summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_experimental_forecasts import build_experimental_compare_payload  # noqa: E402


DEFAULT_LEAR = ROOT / "pfc_shaping" / "output" / "lear_forecast_latest.csv"
DEFAULT_PRICEFM = ROOT / "pfc_shaping" / "output" / "pricefm_forecast_latest.csv"
DEFAULT_OUTPUT_DIR = ROOT / "pfc_shaping" / "output"
DEFAULT_SUMMARY = ROOT / "pfc_shaping" / "output" / "experimental_forecast_compare_history.csv"


def _extract_row(payload: dict, run_date: str) -> dict:
    summaries = {row["label"]: row for row in payload["summaries"]}
    deltas = {row["label"]: row for row in payload["deltas_vs_lear"]}
    fb_meta = payload.get("futureboost_meta", {})
    return {
        "run_date": run_date,
        "lear_mean": summaries["lear"]["mean"],
        "fixed_mean": summaries["fixed"]["mean"],
        "regime_mean": summaries["regime"]["mean"],
        "futureboost_mean": summaries["futureboost"]["mean"],
        "fixed_delta_mean": deltas["fixed"]["delta_mean"],
        "regime_delta_mean": deltas["regime"]["delta_mean"],
        "futureboost_delta_mean": deltas["futureboost"]["delta_mean"],
        "lear_peak_mean": summaries["lear"].get("peak_mean"),
        "fixed_peak_mean": summaries["fixed"].get("peak_mean"),
        "regime_peak_mean": summaries["regime"].get("peak_mean"),
        "futureboost_peak_mean": summaries["futureboost"].get("peak_mean"),
        "futureboost_alpha": fb_meta.get("alpha"),
        "futureboost_train_rows": fb_meta.get("train_rows"),
        "futureboost_future_rows": fb_meta.get("future_rows"),
        "futureboost_pricefm_rows": fb_meta.get("future_pricefm_rows"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive the latest experimental forecast comparison.")
    parser.add_argument("--lear", type=Path, default=DEFAULT_LEAR)
    parser.add_argument("--pricefm", type=Path, default=DEFAULT_PRICEFM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    payload = build_experimental_compare_payload(args.lear, args.pricefm)
    run_date = pd.Timestamp.now(tz="Europe/Zurich").strftime("%Y-%m-%d")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dated_json = args.output_dir / f"experimental_forecast_compare_{run_date}.json"
    latest_json = args.output_dir / "experimental_forecast_compare.json"
    dated_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    row = _extract_row(payload, run_date)
    if args.summary.exists():
        hist = pd.read_csv(args.summary)
        hist = hist[hist["run_date"] != run_date]
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    else:
        hist = pd.DataFrame([row])
    hist = hist.sort_values("run_date").reset_index(drop=True)
    hist.to_csv(args.summary, index=False)

    print(f"dated_json:{dated_json}")
    print(f"latest_json:{latest_json}")
    print(f"summary_csv:{args.summary}")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()

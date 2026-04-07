#!/usr/bin/env python3
"""
Fast blend evaluation using an existing LEAR backtest parquet instead of recomputing
the whole LEAR backtest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICEFM_HOURLY = ROOT / "pfc_shaping" / "output" / "pricefm_ch_full_e5_predictions_hourly.csv"
DEFAULT_BT = ROOT / "pfc_shaping" / "output" / "lear_backtest_latest.parquet"
DEFAULT_OUTPUT = ROOT / "pfc_shaping" / "output" / "pricefm_ch_full_e5_blend_eval.json"


def _score_frame(df: pd.DataFrame, forecast_col: str, actual_col: str = "actual") -> dict[str, float]:
    err = df[forecast_col] - df[actual_col]
    abs_err = err.abs()
    ape = abs_err / df[actual_col].abs().clip(lower=1) * 100.0
    mae = float(abs_err.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    mape = float(ape.mean())
    corr = float(df[forecast_col].corr(df[actual_col]))
    score = (
        0.35 * (mae / 15.0)
        + 0.30 * (rmse / 22.3)
        + 0.20 * (mape / 30.9)
        + 0.15 * (1.0 - corr)
    )
    return {"mae": mae, "rmse": rmse, "mape": mape, "corr": corr, "score": score}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast evaluation of PriceFM/LEAR blend from saved backtest.")
    parser.add_argument("--pricefm-hourly", type=Path, default=DEFAULT_PRICEFM_HOURLY)
    parser.add_argument("--lear-backtest", type=Path, default=DEFAULT_BT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.pricefm_hourly = args.pricefm_hourly.resolve()
    args.lear_backtest = args.lear_backtest.resolve()
    args.output = args.output.resolve()

    pricefm = pd.read_csv(args.pricefm_hourly)
    pricefm["timestamp"] = pd.to_datetime(pricefm["timestamp"], utc=True)

    bt = pd.read_parquet(args.lear_backtest).copy()
    if "timestamp" not in bt.columns:
        bt["timestamp"] = pd.to_datetime(bt["forecast_ts"], utc=True)
    bt = bt.rename(columns={"forecast": "lear"})
    bt = bt[["timestamp", "actual", "lear"]]

    merged = bt.merge(pricefm[["timestamp", "pricefm"]], on="timestamp", how="inner")
    if merged.empty:
        raise ValueError("No overlap between saved LEAR backtest and PriceFM predictions.")

    rows = []
    for w in np.linspace(0.0, 1.0, 21):
        col = f"blend_{w:.2f}"
        merged[col] = (1.0 - w) * merged["lear"] + w * merged["pricefm"]
        metrics = _score_frame(merged, col)
        metrics["weight_pricefm"] = float(w)
        rows.append(metrics)

    blend_df = pd.DataFrame(rows).sort_values("score", ascending=True).reset_index(drop=True)
    payload = {
        "pricefm_hourly": str(args.pricefm_hourly),
        "lear_backtest": str(args.lear_backtest),
        "overlap_start": str(merged["timestamp"].min()),
        "overlap_end": str(merged["timestamp"].max()),
        "n_hours_overlap": int(len(merged)),
        "lear": _score_frame(merged, "lear"),
        "pricefm": _score_frame(merged, "pricefm"),
        "best_blend": blend_df.iloc[0].to_dict(),
        "top5_blends": blend_df.head(5).to_dict(orient="records"),
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"output:{args.output}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

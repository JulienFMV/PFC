#!/usr/bin/env python3
"""
Evaluate exported hourly PriceFM predictions against the current LEAR backtest and
search a simple convex blend on the overlapping window.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICEFM_HOURLY = ROOT / "pfc_shaping" / "output" / "pricefm_ch_de_predictions_hourly.csv"
DEFAULT_OUTPUT = ROOT / "pfc_shaping" / "output" / "pricefm_ch_de_blend_eval.json"


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
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "corr": corr,
        "score": score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PriceFM hourly predictions vs LEAR and test blends.")
    parser.add_argument("--pricefm-hourly", type=Path, default=DEFAULT_PRICEFM_HOURLY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.pricefm_hourly = args.pricefm_hourly.resolve()
    args.output = args.output.resolve()

    if not args.pricefm_hourly.exists():
        raise FileNotFoundError(f"PriceFM hourly predictions not found: {args.pricefm_hourly}")

    sys.path.insert(0, str(ROOT))
    from pfc_shaping.model.lear_forecaster import LEARForecaster

    pricefm = pd.read_csv(args.pricefm_hourly)
    pricefm["timestamp"] = pd.to_datetime(pricefm["timestamp"], utc=True)

    epex_ch = pd.read_parquet(ROOT / "pfc_shaping" / "data" / "epex_15min.parquet")
    epex_de = pd.read_parquet(ROOT / "pfc_shaping" / "data" / "epex_de_15min.parquet")
    entso = pd.read_parquet(ROOT / "pfc_shaping" / "data" / "entso_15min.parquet")

    lear = LEARForecaster()
    lear.fit(epex_15min=epex_ch, entso_15min=entso, epex_de_15min=epex_de)
    bt = lear.backtest(n_days=30, horizon=1).copy()
    bt["timestamp"] = pd.to_datetime(bt["date"]) + pd.to_timedelta(bt["hour"], unit="h")
    bt["timestamp"] = bt["timestamp"].dt.tz_localize("Europe/Zurich").dt.tz_convert("UTC")
    bt = bt[["timestamp", "actual", "forecast"]].rename(columns={"forecast": "lear"})

    merged = bt.merge(pricefm[["timestamp", "pricefm"]], on="timestamp", how="inner")
    if merged.empty:
        raise ValueError("No timestamp overlap between LEAR backtest and PriceFM predictions.")

    blend_rows = []
    for w in np.linspace(0.0, 1.0, 21):
        col = f"blend_{w:.2f}"
        merged[col] = (1.0 - w) * merged["lear"] + w * merged["pricefm"]
        metrics = _score_frame(merged, col)
        metrics["weight_pricefm"] = float(w)
        blend_rows.append(metrics)

    blend_df = pd.DataFrame(blend_rows).sort_values("score", ascending=True).reset_index(drop=True)
    payload = {
        "pricefm_hourly": str(args.pricefm_hourly),
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

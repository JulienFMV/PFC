#!/usr/bin/env python3
"""Segment Swiss CT backtests into interpretable regime families."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_backtest(path: Path) -> pd.DataFrame:
    bt = pd.read_parquet(path).copy()
    if "forecast_ts" in bt.columns:
        bt["timestamp"] = pd.to_datetime(bt["forecast_ts"], utc=True)
    elif "date" in bt.columns and "hour" in bt.columns:
        local_midnight = pd.to_datetime(bt["date"]).dt.tz_localize("Europe/Zurich")
        bt["timestamp"] = (local_midnight + pd.to_timedelta(bt["hour"], unit="h")).dt.tz_convert("UTC")
    else:
        raise ValueError("Backtest must contain forecast_ts or date+hour.")
    bt["local_ts"] = bt["timestamp"].dt.tz_convert("Europe/Zurich")
    bt["date_local"] = bt["local_ts"].dt.normalize()
    bt["abs_error"] = (bt["forecast"] - bt["actual"]).abs()
    bt["signed_error"] = bt["forecast"] - bt["actual"]
    return bt


def _daily_regimes(bt: pd.DataFrame) -> pd.DataFrame:
    daily = bt.groupby("date_local").agg(
        mae=("abs_error", "mean"),
        rmse=("signed_error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        bias=("signed_error", "mean"),
        actual_mean=("actual", "mean"),
        actual_min=("actual", "min"),
        actual_max=("actual", "max"),
        forecast_mean=("forecast", "mean"),
    )
    midday = bt[bt["local_ts"].dt.hour.between(10, 15)].groupby("date_local")["actual"].mean()
    rest = bt[~bt["local_ts"].dt.hour.between(10, 15)].groupby("date_local")["actual"].mean()
    daily["midday_delta"] = (midday - rest).reindex(daily.index)

    ch_holidays = holidays.Switzerland(years=sorted({d.year for d in daily.index}))
    de_holidays = holidays.Germany(years=sorted({d.year for d in daily.index}))
    fr_holidays = holidays.France(years=sorted({d.year for d in daily.index}))

    dates = [d.date() for d in daily.index]
    daily["holiday_mismatch"] = [
        (d not in ch_holidays) and ((d in de_holidays) or (d in fr_holidays))
        for d in dates
    ]
    daily["negative_day"] = daily["actual_min"] < 0.0
    daily["deep_negative_day"] = daily["actual_min"] <= -100.0
    daily["midday_collapse"] = daily["midday_delta"] < -80.0
    daily["spike_day"] = daily["actual_max"] >= 200.0
    daily["low_mean_day"] = daily["actual_mean"] < 60.0
    daily["high_mean_day"] = daily["actual_mean"] > 130.0
    return daily


def _segment_summary(bt: pd.DataFrame, daily: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seg in [
        "negative_day",
        "deep_negative_day",
        "midday_collapse",
        "holiday_mismatch",
        "low_mean_day",
        "high_mean_day",
        "spike_day",
    ]:
        dates = daily.index[daily[seg]]
        subset = bt[bt["date_local"].isin(dates)]
        if subset.empty:
            continue
        rows.append(
            {
                "segment": seg,
                "days": int(len(dates)),
                "hour_share": float(len(subset) / len(bt)),
                "mae": float(subset["abs_error"].mean()),
                "rmse": float(np.sqrt(np.mean(np.square(subset["signed_error"])))),
                "bias": float(subset["signed_error"].mean()),
                "contribution_to_global_mae": float((len(subset) / len(bt)) * subset["abs_error"].mean()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Swiss CT backtest regime families.")
    parser.add_argument(
        "--backtest",
        type=Path,
        default=ROOT / "pfc_shaping" / "output" / "lear_foundation_ab_20260512_145501_297410_foundation.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "pfc_shaping" / "output")
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bt = _load_backtest(args.backtest)
    daily = _daily_regimes(bt)
    segments = _segment_summary(bt, daily)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backtest": str(args.backtest),
        "overall": {
            "mae": float(bt["abs_error"].mean()),
            "rmse": float(np.sqrt(np.mean(np.square(bt["signed_error"])))),
            "bias": float(bt["signed_error"].mean()),
            "n_hours": int(len(bt)),
            "n_days": int(daily.shape[0]),
        },
        "segments": segments,
        "worst_days": [
            {
                "date_local": str(idx),
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "bias": float(row["bias"]),
                "actual_mean": float(row["actual_mean"]),
                "actual_min": float(row["actual_min"]),
                "actual_max": float(row["actual_max"]),
                "holiday_mismatch": bool(row["holiday_mismatch"]),
                "negative_day": bool(row["negative_day"]),
                "deep_negative_day": bool(row["deep_negative_day"]),
                "midday_collapse": bool(row["midday_collapse"]),
                "spike_day": bool(row["spike_day"]),
            }
            for idx, row in daily.sort_values("mae", ascending=False).head(args.top_k).iterrows()
        ],
    }

    out_path = args.output_dir / "ct_regime_families_latest.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

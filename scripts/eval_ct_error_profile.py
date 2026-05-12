#!/usr/bin/env python3
"""Profile the current Swiss CT production path error structure."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.model.lear_forecaster import LEARForecaster  # noqa: E402


def _with_timestamp(bt: pd.DataFrame) -> pd.DataFrame:
    out = bt.copy()
    if "forecast_ts" in out.columns:
        out["timestamp"] = pd.to_datetime(out["forecast_ts"], utc=True)
    elif "date" in out.columns and "hour" in out.columns:
        local_midnight = pd.to_datetime(out["date"]).dt.tz_localize("Europe/Zurich")
        out["timestamp"] = (local_midnight + pd.to_timedelta(out["hour"], unit="h")).dt.tz_convert("UTC")
    else:
        raise ValueError("Backtest must contain forecast_ts or date+hour.")
    out["local_ts"] = out["timestamp"].dt.tz_convert("Europe/Zurich")
    out["abs_error"] = (out["forecast"] - out["actual"]).abs()
    out["signed_error"] = out["forecast"] - out["actual"]
    return out


def _score(df: pd.DataFrame) -> dict[str, float | int]:
    clean = df[["forecast", "actual"]].dropna()
    err = clean["forecast"] - clean["actual"]
    abs_err = err.abs()
    ape = abs_err / clean["actual"].abs().clip(lower=1.0) * 100.0
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mape": float(ape.mean()),
        "corr": float(clean["forecast"].corr(clean["actual"])),
        "n": int(len(clean)),
    }


def _label_segments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    local = out["local_ts"]
    price = out["actual"]
    out["segment"] = np.select(
        [
            local.dt.hour.between(10, 15),
            local.dt.hour.isin([7, 8, 9, 17, 18, 19, 20]),
            local.dt.dayofweek >= 5,
            price.abs() >= price.abs().quantile(0.90),
        ],
        ["solar_midday", "peak", "weekend", "tail_price"],
        default="other",
    )
    out["hour_bucket"] = np.select(
        [
            local.dt.hour.between(0, 5),
            local.dt.hour.between(6, 9),
            local.dt.hour.between(10, 15),
            local.dt.hour.between(16, 20),
        ],
        ["night", "morning_ramp", "midday", "evening_peak"],
        default="late_evening",
    )
    out["is_weekend"] = (local.dt.dayofweek >= 5).astype(int)
    out["date_local"] = local.dt.date.astype(str)
    return out


def _rank_table(df: pd.DataFrame, by: list[str]) -> list[dict[str, object]]:
    rows = []
    grouped = df.groupby(by, dropna=False)
    for key, sub in grouped:
        row: dict[str, object] = {}
        if not isinstance(key, tuple):
            key = (key,)
        for name, value in zip(by, key):
            row[name] = value
        row.update(_score(sub))
        row["bias"] = float(sub["signed_error"].mean())
        row["p95_abs_error"] = float(sub["abs_error"].quantile(0.95))
        rows.append(row)
    rows.sort(key=lambda r: (-float(r["mae"]), -int(r["n"])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile current Swiss CT error structure.")
    parser.add_argument("--n-days", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--use-foundation-model", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "pfc_shaping" / "output")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    epex_ch = pd.read_parquet(ROOT / "pfc_shaping" / "data" / "epex_15min.parquet")
    epex_de = pd.read_parquet(ROOT / "pfc_shaping" / "data" / "epex_de_15min.parquet")
    entso = pd.read_parquet(ROOT / "pfc_shaping" / "data" / "entso_15min.parquet")

    model = LEARForecaster(
        use_foundation_model=args.use_foundation_model,
        use_extended_physical_ch_features=False,
        use_gbm_blend=True,
        use_mlp_blend=False,
        gbm_blend_max_horizon_days=1,
    )
    model.fit(epex_15min=epex_ch, entso_15min=entso, epex_de_15min=epex_de)
    bt = model.backtest(n_days=args.n_days, horizon=args.horizon)
    prof = _label_segments(_with_timestamp(bt))

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_days": args.n_days,
        "horizon": args.horizon,
        "use_foundation_model": bool(args.use_foundation_model),
        "overall": _score(prof),
        "segment_rank": _rank_table(prof, ["segment"])[: args.top_k],
        "hour_rank": _rank_table(prof, ["hour"])[: args.top_k],
        "hour_bucket_rank": _rank_table(prof, ["hour_bucket"])[: args.top_k],
        "segment_hour_rank": _rank_table(prof, ["segment", "hour"])[: args.top_k],
        "worst_days": _rank_table(prof, ["date_local"])[: args.top_k],
    }

    out_path = args.output_dir / f"ct_error_profile_h{args.horizon}_{args.n_days}d.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(json.dumps({
        "overall": payload["overall"],
        "top_segment": payload["segment_rank"][0] if payload["segment_rank"] else None,
        "top_hour": payload["hour_rank"][0] if payload["hour_rank"] else None,
        "top_day": payload["worst_days"][0] if payload["worst_days"] else None,
        "output": str(out_path),
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Profile the worst Swiss CT days against daily system signals."""

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
    bt["date_local"] = bt["local_ts"].dt.date.astype(str)
    bt["abs_error"] = (bt["forecast"] - bt["actual"]).abs()
    bt["signed_error"] = bt["forecast"] - bt["actual"]
    return bt


def _daily_zscore(s: pd.Series, window: int = 28) -> pd.Series:
    shifted = s.shift(1)
    mean = shifted.rolling(window, min_periods=max(7, window // 4)).mean()
    std = shifted.rolling(window, min_periods=max(7, window // 4)).std().replace(0.0, np.nan)
    return ((s - mean) / std).replace([np.inf, -np.inf], np.nan)


def _daily_frame() -> pd.DataFrame:
    data_dir = ROOT / "pfc_shaping" / "data"
    entso = pd.read_parquet(data_dir / "entso_15min.parquet").copy()
    epex_ch = pd.read_parquet(data_dir / "epex_15min.parquet").copy()
    epex_de = pd.read_parquet(data_dir / "epex_de_15min.parquet").copy()
    hydro = pd.read_parquet(data_dir / "hydro_reservoir.parquet").copy()

    daily = pd.DataFrame()
    daily.index = (
        entso.index.tz_convert("Europe/Zurich").normalize().unique().sort_values()
        if isinstance(entso.index, pd.DatetimeIndex)
        else pd.DatetimeIndex([])
    )

    def add_daily_from_hourly(df: pd.DataFrame, col: str, agg: str, out_name: str) -> None:
        if col not in df.columns:
            return
        series = df[col]
        local = series.copy()
        local.index = local.index.tz_convert("Europe/Zurich")
        if agg == "mean":
            out = local.resample("D").mean()
        elif agg == "max":
            out = local.resample("D").max()
        elif agg == "sum":
            out = local.resample("D").sum()
        else:
            raise ValueError(agg)
        daily[out_name] = out

    entso_hourly = entso.resample("h").mean()
    epex_ch_hourly = epex_ch["price_eur_mwh"].resample("h").mean().to_frame("price_ch")
    epex_de_hourly = epex_de["price_eur_mwh"].resample("h").mean().to_frame("price_de")

    add_daily_from_hourly(epex_ch_hourly, "price_ch", "mean", "price_ch_mean")
    add_daily_from_hourly(epex_ch_hourly, "price_ch", "max", "price_ch_max")
    add_daily_from_hourly(epex_de_hourly, "price_de", "mean", "price_de_mean")
    add_daily_from_hourly(epex_de_hourly, "price_de", "max", "price_de_max")
    add_daily_from_hourly(entso_hourly, "load_mw", "mean", "load_ch_mean")
    add_daily_from_hourly(entso_hourly, "solar_mw", "mean", "solar_ch_mean")
    add_daily_from_hourly(entso_hourly, "wind_mw", "mean", "wind_ch_mean")
    add_daily_from_hourly(entso_hourly, "solar_de_mw", "mean", "solar_de_mean")
    add_daily_from_hourly(entso_hourly, "wind_de_mw", "mean", "wind_de_mean")
    add_daily_from_hourly(entso_hourly, "residual_load_de_mw", "mean", "residual_load_de_mean")
    add_daily_from_hourly(entso_hourly, "cross_border_mw", "mean", "cross_border_mean")
    add_daily_from_hourly(entso_hourly, "flow_deviation", "mean", "flow_deviation_mean")
    add_daily_from_hourly(entso_hourly, "load_deviation", "mean", "load_deviation_mean")
    add_daily_from_hourly(entso_hourly, "fr_nuclear_unavailable_mw", "mean", "fr_nuclear_unavailable_mean")

    for border in ["de", "fr", "at", "it"]:
        add_daily_from_hourly(entso_hourly, f"scheduled_net_export_ch_{border}_mw", "mean", f"sched_ch_{border}_mean")
        add_daily_from_hourly(entso_hourly, f"flow_net_export_ch_{border}_mw", "mean", f"flow_ch_{border}_mean")
        add_daily_from_hourly(entso_hourly, f"ntc_net_ch_{border}_mw", "mean", f"ntc_net_ch_{border}_mean")

    if "fill_pct" in hydro.columns:
        h = hydro.copy()
        if not isinstance(h.index, pd.DatetimeIndex):
            h.index = pd.to_datetime(h.index, utc=True)
        if h.index.tz is None:
            h.index = h.index.tz_localize("UTC")
        h.index = h.index.tz_convert("Europe/Zurich")
        daily["hydro_fill_pct"] = h["fill_pct"].resample("D").ffill()
    if "fill_deviation" in hydro.columns:
        h = hydro.copy()
        if not isinstance(h.index, pd.DatetimeIndex):
            h.index = pd.to_datetime(h.index, utc=True)
        if h.index.tz is None:
            h.index = h.index.tz_localize("UTC")
        h.index = h.index.tz_convert("Europe/Zurich")
        daily["hydro_fill_deviation"] = h["fill_deviation"].resample("D").ffill()

    for col in list(daily.columns):
        daily[f"{col}_z28"] = _daily_zscore(daily[col])

    daily.index = daily.index.date.astype(str)
    daily.index.name = "date_local"
    return daily.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile worst Swiss CT days vs daily exogenous signals.")
    parser.add_argument(
        "--backtest",
        type=Path,
        default=ROOT / "pfc_shaping" / "output" / "lear_foundation_ab_20260508_145701_120987_foundation.parquet",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "pfc_shaping" / "output")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bt = _load_backtest(args.backtest)
    daily_err = (
        bt.groupby("date_local", dropna=False)
        .agg(
            mae=("abs_error", "mean"),
            rmse=("signed_error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            bias=("signed_error", "mean"),
            price_mean=("actual", "mean"),
            price_max=("actual", "max"),
            forecast_mean=("forecast", "mean"),
        )
        .reset_index()
        .sort_values(["mae", "rmse"], ascending=[False, False])
        .reset_index(drop=True)
    )

    daily = _daily_frame()
    merged = daily_err.merge(daily, on="date_local", how="left")

    signal_cols = [
        c for c in merged.columns
        if c.endswith("_z28")
    ]

    worst_rows = []
    for _, row in merged.head(args.top_k).iterrows():
        signals = {
            col: float(row[col])
            for col in signal_cols
            if pd.notna(row[col]) and abs(float(row[col])) >= 1.0
        }
        signals = dict(sorted(signals.items(), key=lambda kv: -abs(kv[1]))[:12])
        worst_rows.append(
            {
                "date_local": row["date_local"],
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "bias": float(row["bias"]),
                "price_mean": float(row["price_mean"]),
                "price_max": float(row["price_max"]),
                "signals_abs_z_ge_1": signals,
            }
        )

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backtest": str(args.backtest),
        "top_k": args.top_k,
        "overall": {
            "mae_mean": float(daily_err["mae"].mean()),
            "mae_p90": float(daily_err["mae"].quantile(0.90)),
            "bias_mean": float(daily_err["bias"].mean()),
        },
        "worst_days": worst_rows,
    }

    out_path = args.output_dir / "ct_extreme_days_profile_latest.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

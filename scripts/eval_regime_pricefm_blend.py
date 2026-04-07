#!/usr/bin/env python3
"""
Evaluate a regime-aware PriceFM/LEAR blend on the common backtest window.

The router is intentionally simple and deployable:
- time split train/test on the overlap window
- regimes built from hour bucket, weekend flag, and disagreement bucket
- per-regime PriceFM weight selected on the train slice
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
DEFAULT_PRICEFM_HOURLY = ROOT / "pfc_shaping" / "output" / "pricefm_ch_full_predictions_hourly.csv"
DEFAULT_OUTPUT = ROOT / "pfc_shaping" / "output" / "pricefm_ch_full_regime_blend_eval.json"


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


def _hour_bucket(hour: int) -> str:
    if 10 <= hour <= 15:
        return "solar_midday"
    if 7 <= hour <= 9 or 16 <= hour <= 19:
        return "peak"
    return "offpeak"


def _choose_weight(train_df: pd.DataFrame, candidate_weights: np.ndarray) -> float:
    best_w = 0.0
    best_loss = float("inf")
    for w in candidate_weights:
        pred = (1.0 - w) * train_df["lear"] + w * train_df["pricefm"]
        loss = float(np.mean((pred - train_df["actual"]) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_w = float(w)
    return best_w


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a regime-aware LEAR/PriceFM blend.")
    parser.add_argument("--pricefm-hourly", type=Path, default=DEFAULT_PRICEFM_HOURLY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    args = parser.parse_args()

    args.pricefm_hourly = args.pricefm_hourly.resolve()
    args.output = args.output.resolve()

    if not args.pricefm_hourly.exists():
        raise FileNotFoundError(f"PriceFM hourly predictions not found: {args.pricefm_hourly}")

    sys.path.insert(0, str(ROOT))
    from pfc_shaping.model.lear_forecaster import LEARForecaster, _get_holidays

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

    merged = bt.merge(pricefm[["timestamp", "pricefm"]], on="timestamp", how="inner").sort_values("timestamp")
    if merged.empty:
        raise ValueError("No timestamp overlap between LEAR backtest and PriceFM predictions.")

    merged["hour_local"] = merged["timestamp"].dt.tz_convert("Europe/Zurich").dt.hour
    merged["date_local"] = merged["timestamp"].dt.tz_convert("Europe/Zurich").dt.date
    merged["weekday"] = merged["timestamp"].dt.tz_convert("Europe/Zurich").dt.weekday
    merged["is_weekend"] = (merged["weekday"] >= 5).astype(int)
    merged["hour_bucket"] = merged["hour_local"].apply(_hour_bucket)
    merged["abs_disagreement"] = (merged["pricefm"] - merged["lear"]).abs()
    merged["signed_disagreement"] = merged["pricefm"] - merged["lear"]

    hol_flags = []
    for d in merged["date_local"]:
        ch_hols, de_hols, fr_hols = _get_holidays(d.year)
        hol_flags.append(
            (
                int(d in ch_hols),
                int(d in de_hols),
                int(d in fr_hols),
            )
        )
    merged[["is_holiday_ch", "is_holiday_de", "is_holiday_fr"]] = pd.DataFrame(hol_flags, index=merged.index)

    split_idx = max(24, int(len(merged) * args.train_ratio))
    train = merged.iloc[:split_idx].copy()
    test = merged.iloc[split_idx:].copy()

    if len(test) < 48:
        raise ValueError("Test slice too short for a meaningful regime evaluation.")

    q1, q2 = train["abs_disagreement"].quantile([0.33, 0.66]).tolist()
    bins = [-np.inf, q1, q2, np.inf]
    labels = ["low", "mid", "high"]
    train["disagreement_bucket"] = pd.cut(train["abs_disagreement"], bins=bins, labels=labels, include_lowest=True)
    test["disagreement_bucket"] = pd.cut(test["abs_disagreement"], bins=bins, labels=labels, include_lowest=True)

    candidate_weights = np.arange(0.0, 0.41, 0.05)

    global_train_best = _choose_weight(train, candidate_weights)
    train["blend_global"] = (1.0 - global_train_best) * train["lear"] + global_train_best * train["pricefm"]
    test["blend_global"] = (1.0 - global_train_best) * test["lear"] + global_train_best * test["pricefm"]

    regime_cols = ["hour_bucket", "is_weekend", "is_holiday_ch", "disagreement_bucket"]
    weight_map: dict[str, float] = {}
    default_weight = global_train_best

    for key, grp in train.groupby(regime_cols, dropna=False):
        if len(grp) < 12:
            continue
        w = _choose_weight(grp, candidate_weights)
        weight_map["|".join(map(str, key))] = w

    def _lookup_weight(row: pd.Series) -> float:
        key = "|".join(map(str, [row[c] for c in regime_cols]))
        return float(weight_map.get(key, default_weight))

    train["weight_regime"] = train.apply(_lookup_weight, axis=1)
    test["weight_regime"] = test.apply(_lookup_weight, axis=1)
    train["blend_regime"] = (1.0 - train["weight_regime"]) * train["lear"] + train["weight_regime"] * train["pricefm"]
    test["blend_regime"] = (1.0 - test["weight_regime"]) * test["lear"] + test["weight_regime"] * test["pricefm"]

    payload = {
        "pricefm_hourly": str(args.pricefm_hourly),
        "overlap_start": str(merged["timestamp"].min()),
        "overlap_end": str(merged["timestamp"].max()),
        "n_hours_overlap": int(len(merged)),
        "train_hours": int(len(train)),
        "test_hours": int(len(test)),
        "lear_test": _score_frame(test, "lear"),
        "pricefm_test": _score_frame(test, "pricefm"),
        "global_blend_test": {
            **_score_frame(test, "blend_global"),
            "weight_pricefm": float(global_train_best),
        },
        "regime_blend_test": _score_frame(test, "blend_regime"),
        "default_weight_pricefm": float(default_weight),
        "n_regimes_fitted": int(len(weight_map)),
        "sample_regime_weights": [
            {"regime": k, "weight_pricefm": float(v)} for k, v in sorted(weight_map.items())[:12]
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"output:{args.output}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

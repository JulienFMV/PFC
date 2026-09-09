#!/usr/bin/env python3
"""
Evaluate a lightweight FutureBoosting-style meta-layer on the LEAR/PriceFM overlap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.ct.model.futureboost_experimental import (  # noqa: E402
    DEFAULT_FUTUREBOOST_EXPERIMENT,
    FutureBoostExperimentalRegressor,
)
from pfc_shaping.ct.model.pricefm_experimental import (  # noqa: E402
    blend_lear_with_pricefm,
)


DEFAULT_PRICEFM_HOURLY = ROOT / "pfc_shaping" / "output" / "pricefm_ch_full_e5_predictions_hourly.csv"
DEFAULT_LEAR_BACKTEST = ROOT / "pfc_shaping" / "output" / "lear_backtest_2026-03-20.parquet"
DEFAULT_OUTPUT = ROOT / "pfc_shaping" / "output" / "futureboost_experimental_eval.json"


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


def _load_saved_backtest(path: Path) -> pd.DataFrame:
    bt = pd.read_parquet(path).copy()
    if "horizon" in bt.columns:
        bt = bt[bt["horizon"] == 1].copy()
    if "date" in bt.columns and "hour" in bt.columns:
        local_midnight = pd.to_datetime(bt["date"]).dt.tz_localize("Europe/Zurich")
        bt["timestamp"] = local_midnight + pd.to_timedelta(bt["hour"], unit="h")
        bt["timestamp"] = bt["timestamp"].dt.tz_convert("UTC")
    elif "forecast_ts" in bt.columns:
        bt["timestamp"] = pd.to_datetime(bt["forecast_ts"], utc=True)
    else:
        raise ValueError("Unsupported saved LEAR backtest format.")

    if "forecast" in bt.columns:
        bt = bt.rename(columns={"forecast": "lear"})
    return bt[["timestamp", "actual", "lear"]].drop_duplicates(subset=["timestamp"], keep="last")


def _build_overlap(pricefm_hourly: Path, n_days: int, lear_backtest: Path | None, recompute_backtest: bool) -> pd.DataFrame:
    pricefm = pd.read_csv(pricefm_hourly)
    pricefm["timestamp"] = pd.to_datetime(pricefm["timestamp"], utc=True)

    if lear_backtest is not None and lear_backtest.exists() and not recompute_backtest:
        merged = _load_saved_backtest(lear_backtest)
    else:
        from pfc_shaping.ct.model.lear_forecaster import LEARForecaster  # noqa: WPS433

        epex_ch = pd.read_parquet(ROOT / "pfc_shaping" / "data" / "epex_15min.parquet")
        epex_de = pd.read_parquet(ROOT / "pfc_shaping" / "data" / "epex_de_15min.parquet")
        entso = pd.read_parquet(ROOT / "pfc_shaping" / "data" / "entso_15min.parquet")

        lear = LEARForecaster(use_foundation_model=True)
        lear.fit(epex_15min=epex_ch, entso_15min=entso, epex_de_15min=epex_de)
        bt = lear.backtest(n_days=n_days, horizon=1).copy()

        if "timestamp" not in bt.columns:
            local_midnight = pd.to_datetime(bt["date"]).dt.tz_localize("Europe/Zurich")
            bt["timestamp"] = local_midnight + pd.to_timedelta(bt["hour"], unit="h")
            bt["timestamp"] = bt["timestamp"].dt.tz_convert("UTC")

        merged = bt.rename(columns={"forecast": "lear"})[["timestamp", "actual", "lear"]]

    merged = merged.merge(pricefm[["timestamp", "pricefm"]], on="timestamp", how="inner").sort_values("timestamp")
    if merged.empty:
        raise ValueError("No timestamp overlap between LEAR backtest and PriceFM predictions.")
    return merged.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a FutureBoost-like meta-layer for LEAR + PriceFM.")
    parser.add_argument("--pricefm-hourly", type=Path, default=DEFAULT_PRICEFM_HOURLY)
    parser.add_argument("--n-days", type=int, default=30)
    parser.add_argument("--lear-backtest", type=Path, default=DEFAULT_LEAR_BACKTEST)
    parser.add_argument("--recompute-backtest", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    merged = _build_overlap(
        args.pricefm_hourly.resolve(),
        args.n_days,
        args.lear_backtest.resolve() if args.lear_backtest else None,
        args.recompute_backtest,
    )

    fixed = blend_lear_with_pricefm(
        merged.rename(columns={"lear": "price_lear"})[["timestamp", "price_lear"]],
        merged[["timestamp", "pricefm"]].rename(columns={"pricefm": "price_pricefm"}),
        weight_pricefm=0.15,
    ).rename(columns={"price_lear": "fixed_blend"})[["timestamp", "fixed_blend"]]

    regime = blend_lear_with_pricefm(
        merged.rename(columns={"lear": "price_lear"})[["timestamp", "price_lear"]],
        merged[["timestamp", "pricefm"]].rename(columns={"pricefm": "price_pricefm"}),
        weight_pricefm=None,
    ).rename(columns={"price_lear": "regime_blend"})[["timestamp", "regime_blend", "pricefm_weight"]]

    merged = merged.merge(fixed, on="timestamp", how="left").merge(regime, on="timestamp", how="left")

    split_idx = max(1, int(len(merged) * DEFAULT_FUTUREBOOST_EXPERIMENT.train_fraction))
    split_idx = min(split_idx, len(merged) - 1)
    train = merged.iloc[:split_idx].copy()
    test = merged.iloc[split_idx:].copy()

    meta = FutureBoostExperimentalRegressor()
    meta.fit(train)
    train["futureboost"] = meta.predict(train)
    test["futureboost"] = meta.predict(test)

    local = test["timestamp"].dt.tz_convert("Europe/Zurich")
    hour = local.dt.hour
    weekend = local.dt.dayofweek >= 5
    peak = hour.isin([7, 8, 9, 17, 18, 19, 20])
    solar_midday = hour.between(10, 15)
    test["segment"] = np.select(
        [solar_midday, peak, weekend],
        ["solar_midday", "peak", "weekend"],
        default="other",
    )

    payload = {
        "pricefm_hourly": str(args.pricefm_hourly.resolve()),
        "lear_backtest": str(args.lear_backtest.resolve()) if args.lear_backtest else None,
        "n_overlap_hours": int(len(merged)),
        "train_hours": int(len(train)),
        "test_hours": int(len(test)),
        "train_fraction": DEFAULT_FUTUREBOOST_EXPERIMENT.train_fraction,
        "meta_alpha": meta.alpha_,
        "lear_test": _score_frame(test, "lear"),
        "fixed_blend_test": _score_frame(test, "fixed_blend"),
        "regime_blend_test": _score_frame(test, "regime_blend"),
        "futureboost_test": _score_frame(test, "futureboost"),
        "regime_weights_test": {
            str(k): int(v) for k, v in regime.loc[split_idx:, "pricefm_weight"].round(2).value_counts().sort_index().items()
        },
        "segments_test": [],
    }

    for segment, seg_df in test.groupby("segment"):
        if len(seg_df) < 5:
            continue
        payload["segments_test"].append(
            {
                "segment": segment,
                "n": int(len(seg_df)),
                "lear": _score_frame(seg_df, "lear"),
                "regime_blend": _score_frame(seg_df, "regime_blend"),
                "futureboost": _score_frame(seg_df, "futureboost"),
            }
        )

    args.output.resolve().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"output:{args.output.resolve()}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

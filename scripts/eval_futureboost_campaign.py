#!/usr/bin/env python3
"""
Run a small multi-window campaign for the experimental FutureBoost-like meta layer.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.ct.model.futureboost_experimental import (  # noqa: E402
    DEFAULT_FUTUREBOOST_EXPERIMENT,
    FutureBoostExperimentalRegressor,
)
from pfc_shaping.ct.model.pricefm_experimental import blend_lear_with_pricefm  # noqa: E402
from scripts.eval_futureboost_experimental import _build_overlap, _score_frame  # noqa: E402


DEFAULT_PRICEFM_HOURLY = ROOT / "pfc_shaping" / "output" / "pricefm_ch_full_e5_predictions_hourly.csv"
DEFAULT_BACKTESTS = [
    ROOT / "pfc_shaping" / "output" / "lear_backtest_2026-03-15.parquet",
    ROOT / "pfc_shaping" / "output" / "lear_backtest_2026-03-20.parquet",
]
DEFAULT_OUTPUT = ROOT / "pfc_shaping" / "output" / "futureboost_campaign.json"
DEFAULT_TRAIN_FRACTIONS = [0.50, 0.60, 0.70, 0.80]


def _evaluate_one(
    pricefm_hourly: Path,
    lear_backtest: Path,
    train_fraction: float,
    use_cin: bool,
) -> dict:
    merged = _build_overlap(pricefm_hourly, n_days=30, lear_backtest=lear_backtest, recompute_backtest=False)

    fixed = blend_lear_with_pricefm(
        merged.rename(columns={"lear": "price_lear"})[["timestamp", "price_lear"]],
        merged[["timestamp", "pricefm"]].rename(columns={"pricefm": "price_pricefm"}),
        weight_pricefm=0.15,
    ).rename(columns={"price_lear": "fixed_blend"})[["timestamp", "fixed_blend"]]

    regime = blend_lear_with_pricefm(
        merged.rename(columns={"lear": "price_lear"})[["timestamp", "price_lear"]],
        merged[["timestamp", "pricefm"]].rename(columns={"pricefm": "price_pricefm"}),
        weight_pricefm=None,
    ).rename(columns={"price_lear": "regime_blend"})[["timestamp", "regime_blend"]]

    merged = merged.merge(fixed, on="timestamp", how="left").merge(regime, on="timestamp", how="left")

    split_idx = max(1, min(int(len(merged) * train_fraction), len(merged) - 1))
    train = merged.iloc[:split_idx].copy()
    test = merged.iloc[split_idx:].copy()

    cfg = replace(DEFAULT_FUTUREBOOST_EXPERIMENT, use_causal_instance_norm=use_cin)
    meta = FutureBoostExperimentalRegressor(config=cfg)
    meta.fit(train)
    test["futureboost"] = meta.predict(test)

    result = {
        "backtest": str(lear_backtest),
        "train_fraction": train_fraction,
        "n_overlap_hours": int(len(merged)),
        "train_hours": int(len(train)),
        "test_hours": int(len(test)),
        "use_cin": bool(use_cin),
        "meta_alpha": meta.alpha_,
        "lear_score": _score_frame(test, "lear"),
        "fixed_score": _score_frame(test, "fixed_blend"),
        "regime_score": _score_frame(test, "regime_blend"),
        "futureboost_score": _score_frame(test, "futureboost"),
    }
    result["delta_vs_lear"] = result["futureboost_score"]["score"] - result["lear_score"]["score"]
    result["delta_vs_regime"] = result["futureboost_score"]["score"] - result["regime_score"]["score"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-window campaign for the FutureBoost experimental layer.")
    parser.add_argument("--pricefm-hourly", type=Path, default=DEFAULT_PRICEFM_HOURLY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=["baseline", "cin", "both"],
        default="both",
        help="Evaluate FutureBoost with CIN disabled, enabled, or both.",
    )
    args = parser.parse_args()

    rows = []
    if args.mode == "baseline":
        cin_modes = [False]
    elif args.mode == "cin":
        cin_modes = [True]
    else:
        cin_modes = [False, True]

    for backtest in DEFAULT_BACKTESTS:
        if not backtest.exists():
            continue
        for use_cin in cin_modes:
            for train_fraction in DEFAULT_TRAIN_FRACTIONS:
                rows.append(
                    _evaluate_one(
                        args.pricefm_hourly.resolve(),
                        backtest.resolve(),
                        train_fraction,
                        use_cin=use_cin,
                    )
                )

    if not rows:
        raise FileNotFoundError("No backtest windows available for the FutureBoost campaign.")

    df = pd.DataFrame(
        {
            "backtest": [row["backtest"] for row in rows],
            "train_fraction": [row["train_fraction"] for row in rows],
            "use_cin": [row["use_cin"] for row in rows],
            "lear_score": [row["lear_score"]["score"] for row in rows],
            "fixed_score": [row["fixed_score"]["score"] for row in rows],
            "regime_score": [row["regime_score"]["score"] for row in rows],
            "futureboost_score": [row["futureboost_score"]["score"] for row in rows],
            "delta_vs_lear": [row["delta_vs_lear"] for row in rows],
            "delta_vs_regime": [row["delta_vs_regime"] for row in rows],
        }
    )

    grouped = {}
    for use_cin, sub in df.groupby("use_cin"):
        grouped[str(bool(use_cin))] = {
            "n_runs": int(len(sub)),
            "futureboost_better_than_lear": int((sub["futureboost_score"] < sub["lear_score"]).sum()),
            "futureboost_better_than_regime": int((sub["futureboost_score"] < sub["regime_score"]).sum()),
            "mean_delta_vs_lear": float(sub["delta_vs_lear"].mean()),
            "mean_delta_vs_regime": float(sub["delta_vs_regime"].mean()),
            "median_delta_vs_lear": float(sub["delta_vs_lear"].median()),
            "median_delta_vs_regime": float(sub["delta_vs_regime"].median()),
        }

    summary = {
        "n_runs": int(len(rows)),
        "futureboost_better_than_lear": int((df["futureboost_score"] < df["lear_score"]).sum()),
        "futureboost_better_than_regime": int((df["futureboost_score"] < df["regime_score"]).sum()),
        "mean_delta_vs_lear": float(df["delta_vs_lear"].mean()),
        "mean_delta_vs_regime": float(df["delta_vs_regime"].mean()),
        "median_delta_vs_lear": float(df["delta_vs_lear"].median()),
        "median_delta_vs_regime": float(df["delta_vs_regime"].median()),
        "by_use_cin": grouped,
    }

    payload = {
        "pricefm_hourly": str(args.pricefm_hourly.resolve()),
        "runs": rows,
        "summary": summary,
    }
    args.output.resolve().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"output:{args.output.resolve()}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

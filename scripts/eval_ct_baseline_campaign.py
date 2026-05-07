#!/usr/bin/env python3
"""Unified Swiss CT campaign for baseline-vs-neighbors-vs-FutureBoost decisions."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.model.futureboost_experimental import (  # noqa: E402
    DEFAULT_FUTUREBOOST_EXPERIMENT,
    FutureBoostExperimentalRegressor,
)
from pfc_shaping.model.lear_forecaster import LEARForecaster  # noqa: E402
from pfc_shaping.model.pricefm_experimental import blend_lear_with_pricefm  # noqa: E402


DATA_DIR = ROOT / "pfc_shaping" / "data"
OUTPUT_DIR = ROOT / "pfc_shaping" / "output"
DEFAULT_OUTPUT = OUTPUT_DIR / "ct_baseline_campaign.json"
DEFAULT_PRICEFM_HOURLY = OUTPUT_DIR / "pricefm_ch_full_e5_predictions_hourly.csv"

NEIGHBOR_PATHS = {
    "fr": DATA_DIR / "epex_fr_15min.parquet",
    "at": DATA_DIR / "epex_at_15min.parquet",
    "it": DATA_DIR / "epex_it_15min.parquet",
}


def _score(df: pd.DataFrame, forecast_col: str = "forecast", actual_col: str = "actual") -> dict[str, float | int]:
    clean = df[[forecast_col, actual_col]].dropna().copy()
    if clean.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "corr": float("nan"), "n": 0}
    err = clean[forecast_col] - clean[actual_col]
    abs_err = err.abs()
    ape = abs_err / clean[actual_col].abs().clip(lower=1.0) * 100.0
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mape": float(ape.mean()),
        "corr": float(clean[forecast_col].corr(clean[actual_col])),
        "n": int(len(clean)),
    }


def _overall_score(metrics: dict[str, float | int]) -> float:
    mae = float(metrics["mae"])
    rmse = float(metrics["rmse"])
    mape = float(metrics["mape"])
    corr = float(metrics["corr"])
    return (
        0.35 * (mae / 15.0)
        + 0.30 * (rmse / 22.3)
        + 0.20 * (mape / 30.9)
        + 0.15 * (1.0 - corr)
    )


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{path} must use a DatetimeIndex.")
    return frame


def _load_optional_neighbors() -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, object]]]:
    frames: dict[str, pd.DataFrame] = {}
    info: dict[str, dict[str, object]] = {}
    for code, path in NEIGHBOR_PATHS.items():
        exists = path.exists()
        info[code] = {"path": str(path), "exists": exists}
        if not exists:
            continue
        frame = _load_frame(path)
        frames[code] = frame
        info[code]["rows"] = int(len(frame))
        if len(frame.index) > 0:
            info[code]["start"] = str(frame.index.min())
            info[code]["end"] = str(frame.index.max())
    return frames, info


def _run_lear_variant(
    label: str,
    epex_ch: pd.DataFrame,
    entso: pd.DataFrame,
    hydro: pd.DataFrame,
    outages: pd.DataFrame | None,
    epex_de: pd.DataFrame | None,
    neighbors: dict[str, pd.DataFrame] | None,
    n_days: int,
    horizons: list[int],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    model = LEARForecaster(tz="Europe/Zurich")
    model.fit(
        epex_15min=epex_ch,
        entso_15min=entso,
        hydro=hydro,
        outages_15min=outages,
        epex_de_15min=epex_de,
        neighbor_price_15min=neighbors,
    )

    frames = []
    summary_rows: list[dict[str, object]] = []
    for horizon in horizons:
        backtest = model.backtest(n_days=n_days, horizon=horizon).copy()
        backtest["variant"] = label
        backtest["horizon"] = horizon
        frames.append(backtest)
        metrics = _score(backtest)
        summary_rows.append(
            {
                "variant": label,
                "horizon": horizon,
                **metrics,
                "score": _overall_score(metrics),
            }
        )
    return pd.concat(frames, ignore_index=True), summary_rows


def _load_pricefm_hourly(path: Path) -> pd.DataFrame:
    pricefm = pd.read_csv(path)
    pricefm["timestamp"] = pd.to_datetime(pricefm["timestamp"], utc=True)
    return pricefm[["timestamp", "pricefm"]].drop_duplicates(subset=["timestamp"], keep="last")


def _backtest_to_overlap(backtest: pd.DataFrame) -> pd.DataFrame:
    df = backtest.copy()
    if "forecast_ts" in df.columns:
        df["timestamp"] = pd.to_datetime(df["forecast_ts"], utc=True)
    elif "date" in df.columns and "hour" in df.columns:
        local_midnight = pd.to_datetime(df["date"]).dt.tz_localize("Europe/Zurich")
        df["timestamp"] = (local_midnight + pd.to_timedelta(df["hour"], unit="h")).dt.tz_convert("UTC")
    else:
        raise ValueError("Backtest must contain forecast_ts or date+hour.")
    return (
        df[["timestamp", "actual", "forecast"]]
        .rename(columns={"forecast": "lear"})
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _evaluate_futureboost(
    backtest_h1: pd.DataFrame,
    pricefm_hourly: pd.DataFrame,
    use_cin: bool,
) -> dict[str, object]:
    merged = _backtest_to_overlap(backtest_h1).merge(pricefm_hourly, on="timestamp", how="inner")
    if len(merged) < 48:
        raise ValueError("Not enough LEAR/PriceFM overlap to evaluate FutureBoost.")

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
    split_idx = max(1, min(int(len(merged) * DEFAULT_FUTUREBOOST_EXPERIMENT.train_fraction), len(merged) - 1))
    train = merged.iloc[:split_idx].copy()
    test = merged.iloc[split_idx:].copy()

    cfg = replace(DEFAULT_FUTUREBOOST_EXPERIMENT, use_causal_instance_norm=use_cin)
    meta = FutureBoostExperimentalRegressor(config=cfg)
    meta.fit(train)
    test["futureboost"] = meta.predict(test)

    lear_metrics = _score(test.rename(columns={"lear": "forecast"}))
    fixed_metrics = _score(test.rename(columns={"fixed_blend": "forecast"}))
    regime_metrics = _score(test.rename(columns={"regime_blend": "forecast"}))
    futureboost_metrics = _score(test.rename(columns={"futureboost": "forecast"}))

    return {
        "use_cin": bool(use_cin),
        "train_hours": int(len(train)),
        "test_hours": int(len(test)),
        "meta_alpha": meta.alpha_,
        "lear": {**lear_metrics, "score": _overall_score(lear_metrics)},
        "fixed_blend": {**fixed_metrics, "score": _overall_score(fixed_metrics)},
        "regime_blend": {**regime_metrics, "score": _overall_score(regime_metrics)},
        "futureboost": {**futureboost_metrics, "score": _overall_score(futureboost_metrics)},
        "delta_vs_lear": float(_overall_score(futureboost_metrics) - _overall_score(lear_metrics)),
        "delta_vs_regime": float(_overall_score(futureboost_metrics) - _overall_score(regime_metrics)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a unified Swiss CT baseline campaign.")
    parser.add_argument("--n-days", type=int, default=30)
    parser.add_argument("--horizons", type=str, default="1,3,5")
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Comma-separated subset of variants to run: ch_only,ch_de,ch_multi_neighbor or 'all'.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pricefm-hourly", type=Path, default=DEFAULT_PRICEFM_HOURLY)
    parser.add_argument(
        "--futureboost-mode",
        choices=["skip", "baseline", "cin", "both"],
        default="skip",
        help="Whether to evaluate FutureBoost if PriceFM artifacts are available.",
    )
    args = parser.parse_args()

    horizons = [int(item.strip()) for item in args.horizons.split(",") if item.strip()]
    if not horizons:
        raise ValueError("At least one horizon is required.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("ct_baseline_campaign_%Y%m%d_%H%M%S")

    epex_ch = _load_frame(DATA_DIR / "epex_15min.parquet")
    epex_de = _load_frame(DATA_DIR / "epex_de_15min.parquet")
    entso = _load_frame(DATA_DIR / "entso_15min.parquet")
    hydro = _load_frame(DATA_DIR / "hydro_reservoir.parquet")
    outages = _load_frame(DATA_DIR / "outages_15min.parquet") if (DATA_DIR / "outages_15min.parquet").exists() else None
    optional_neighbors, neighbor_info = _load_optional_neighbors()

    variants: list[tuple[str, pd.DataFrame | None, dict[str, pd.DataFrame] | None]] = [
        ("ch_only", None, None),
        ("ch_de", epex_de, None),
    ]
    if optional_neighbors:
        full_neighbors = {"de": epex_de, **optional_neighbors}
        variants.append(("ch_multi_neighbor", epex_de, full_neighbors))

    requested_variants = {item.strip() for item in args.variants.split(",") if item.strip()}
    if not requested_variants or "all" in requested_variants:
        requested_variants = {label for label, _, _ in variants}
    variants = [item for item in variants if item[0] in requested_variants]
    if not variants:
        raise ValueError(f"No matching variants selected from {sorted(requested_variants)}")

    backtest_paths: dict[str, str] = {}
    summary_rows: list[dict[str, object]] = []
    variant_backtests: dict[str, pd.DataFrame] = {}

    for label, de_frame, neighbor_frames in variants:
        backtest, rows = _run_lear_variant(
            label=label,
            epex_ch=epex_ch,
            entso=entso,
            hydro=hydro,
            outages=outages,
            epex_de=de_frame,
            neighbors=neighbor_frames,
            n_days=args.n_days,
            horizons=horizons,
        )
        variant_backtests[label] = backtest
        backtest_path = args.output.parent / f"{run_id}_{label}.parquet"
        backtest.to_parquet(backtest_path, index=False)
        backtest_paths[label] = str(backtest_path)
        summary_rows.extend(rows)

    summary_df = pd.DataFrame(summary_rows).sort_values(["horizon", "score", "variant"]).reset_index(drop=True)
    summary_csv = args.output.parent / f"{run_id}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    futureboost_payload: dict[str, object] = {"status": "skipped", "reason": "futureboost-mode=skip"}
    if args.futureboost_mode != "skip":
        if not args.pricefm_hourly.exists():
            futureboost_payload = {
                "status": "skipped",
                "reason": f"missing PriceFM hourly file: {args.pricefm_hourly}",
            }
        else:
            best_h1 = summary_df[summary_df["horizon"] == 1].sort_values("score").head(1)
            if best_h1.empty:
                futureboost_payload = {"status": "skipped", "reason": "no horizon=1 LEAR backtest available"}
            else:
                best_variant = str(best_h1.iloc[0]["variant"])
                backtest_h1 = variant_backtests[best_variant]
                backtest_h1 = backtest_h1[backtest_h1["horizon"] == 1].copy()
                pricefm_hourly = _load_pricefm_hourly(args.pricefm_hourly)
                cin_modes = [False] if args.futureboost_mode == "baseline" else [True] if args.futureboost_mode == "cin" else [False, True]
                runs = []
                for use_cin in cin_modes:
                    runs.append(_evaluate_futureboost(backtest_h1, pricefm_hourly, use_cin=use_cin))
                futureboost_payload = {
                    "status": "ok",
                    "source_variant": best_variant,
                    "pricefm_hourly": str(args.pricefm_hourly),
                    "runs": runs,
                }

    payload = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_days": args.n_days,
        "horizons": horizons,
        "requested_variants": sorted(requested_variants),
        "data": {
            "epex_ch_rows": int(len(epex_ch)),
            "epex_de_rows": int(len(epex_de)),
            "entso_rows": int(len(entso)),
            "hydro_rows": int(len(hydro)),
            "outages_rows": int(0 if outages is None else len(outages)),
            "optional_neighbors": neighbor_info,
        },
        "variants": summary_rows,
        "backtests": backtest_paths,
        "summary_csv": str(summary_csv),
        "futureboost": futureboost_payload,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"output:{args.output}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

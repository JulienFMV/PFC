#!/usr/bin/env python3
"""Reproducible A/B harness for LEAR GBM blend evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.model.lear_forecaster import LEARForecaster  # noqa: E402


DATA_FILES = {
    "epex_ch": ROOT / "pfc_shaping" / "data" / "epex_15min.parquet",
    "epex_de": ROOT / "pfc_shaping" / "data" / "epex_de_15min.parquet",
    "entso": ROOT / "pfc_shaping" / "data" / "entso_15min.parquet",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _score(df: pd.DataFrame, forecast_col: str = "forecast", actual_col: str = "actual") -> dict[str, float | int]:
    clean = df[[forecast_col, actual_col]].dropna()
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
    out["hour"] = out["local_ts"].dt.hour
    return out


def _segment_metrics(bt: pd.DataFrame) -> dict[str, object]:
    df = _with_timestamp(bt)
    df["abs_error"] = (df["forecast"] - df["actual"]).abs()
    hour_rows = []
    for hour, hour_df in df.groupby("hour"):
        row = {"hour": int(hour)}
        row.update(_score(hour_df))
        hour_rows.append(row)

    local = df["local_ts"]
    price = df["actual"]
    df["segment"] = np.select(
        [
            local.dt.hour.between(10, 15),
            local.dt.hour.isin([7, 8, 9, 17, 18, 19, 20]),
            local.dt.dayofweek >= 5,
            price.abs() >= price.abs().quantile(0.90),
        ],
        ["solar_midday", "peak", "weekend", "tail_price"],
        default="other",
    )
    segment_rows = []
    for segment, seg_df in df.groupby("segment"):
        row = {"segment": str(segment)}
        row.update(_score(seg_df))
        segment_rows.append(row)
    return {"by_hour": hour_rows, "by_segment": segment_rows}


def _block_bootstrap_delta(
    baseline: pd.DataFrame,
    experiment: pd.DataFrame,
    seed: int,
    n_boot: int,
    block_hours: int,
) -> dict[str, float]:
    b = _with_timestamp(baseline)[["timestamp", "actual", "forecast"]].rename(columns={"forecast": "baseline"})
    e = _with_timestamp(experiment)[["timestamp", "forecast"]].rename(columns={"forecast": "experiment"})
    merged = b.merge(e, on="timestamp", how="inner").dropna()
    if merged.empty:
        return {"mean_delta_mae": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}

    merged["baseline_abs"] = (merged["baseline"] - merged["actual"]).abs()
    merged["experiment_abs"] = (merged["experiment"] - merged["actual"]).abs()
    merged["delta"] = merged["experiment_abs"] - merged["baseline_abs"]
    deltas = merged["delta"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    block = max(1, int(block_hours))
    n = len(deltas)
    draws = []
    for _ in range(max(1, int(n_boot))):
        sample = []
        while len(sample) < n:
            start = int(rng.integers(0, max(1, n - block + 1)))
            sample.extend(deltas[start : start + block])
        draws.append(float(np.mean(sample[:n])))
    return {
        "mean_delta_mae": float(np.mean(deltas)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def _run_variant(
    label: str,
    use_gbm_blend: bool,
    epex_ch: pd.DataFrame,
    epex_de: pd.DataFrame,
    entso: pd.DataFrame,
    n_days: int,
    horizon: int,
) -> pd.DataFrame:
    model = LEARForecaster(
        use_foundation_model=False,
        use_extended_physical_ch_features=False,
        use_gbm_blend=use_gbm_blend,
    )
    model.fit(epex_15min=epex_ch, entso_15min=entso, epex_de_15min=epex_de)
    bt = model.backtest(n_days=n_days, horizon=horizon).copy()
    bt["variant"] = label
    return bt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible LEAR GBM A/B.")
    parser.add_argument("--n-days", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--block-hours", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "pfc_shaping" / "output")
    args = parser.parse_args()

    _set_seed(args.seed)
    run_id = datetime.now(timezone.utc).strftime("lear_gbm_ab_%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "args": vars(args),
        "inputs": {},
    }
    for name, path in DATA_FILES.items():
        manifest["inputs"][name] = {
            "path": str(path),
            "sha256": _sha256(path) if path.exists() else None,
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat() if path.exists() else None,
        }

    epex_ch = pd.read_parquet(DATA_FILES["epex_ch"])
    epex_de = pd.read_parquet(DATA_FILES["epex_de"])
    entso = pd.read_parquet(DATA_FILES["entso"])

    baseline = _run_variant(
        "lasso_only",
        False,
        epex_ch,
        epex_de,
        entso,
        args.n_days,
        args.horizon,
    )
    experiment = _run_variant(
        "gbm_blend",
        True,
        epex_ch,
        epex_de,
        entso,
        args.n_days,
        args.horizon,
    )

    baseline_path = args.output_dir / f"{run_id}_lasso_only.parquet"
    experiment_path = args.output_dir / f"{run_id}_gbm_blend.parquet"
    baseline.to_parquet(baseline_path, index=False)
    experiment.to_parquet(experiment_path, index=False)

    baseline_score = _score(baseline)
    experiment_score = _score(experiment)

    payload = {
        **manifest,
        "lasso_only_backtest": str(baseline_path),
        "gbm_blend_backtest": str(experiment_path),
        "lasso_only": baseline_score,
        "gbm_blend": experiment_score,
        "delta": {
            k: experiment_score[k] - baseline_score[k]
            for k in ["mae", "rmse", "mape", "corr"]
        },
        "bootstrap_delta_mae": _block_bootstrap_delta(
            baseline,
            experiment,
            seed=args.seed,
            n_boot=args.bootstrap,
            block_hours=args.block_hours,
        ),
        "lasso_only_breakdown": _segment_metrics(baseline),
        "gbm_blend_breakdown": _segment_metrics(experiment),
    }

    output_path = args.output_dir / f"{run_id}.json"
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload["delta"], indent=2))
    print(f"output:{output_path.resolve()}")


if __name__ == "__main__":
    main()

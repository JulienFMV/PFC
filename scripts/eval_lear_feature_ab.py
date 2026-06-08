#!/usr/bin/env python3
"""Reproducible A/B harness for LEAR feature experiments.

The script runs the same LEAR backtest protocol for a baseline model and an
experimental model, saves both backtests, records input file hashes, reports
hour/regime metrics, and estimates a block-bootstrap confidence interval for
the MAE delta.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfc_shaping.ct.model.lear_forecaster import LEARForecaster  # noqa: E402


DATA_FILES = {
    "epex_ch": ROOT / "pfc_shaping" / "data" / "epex_15min.parquet",
    "epex_de": ROOT / "pfc_shaping" / "data" / "epex_de_15min.parquet",
    "entso": ROOT / "pfc_shaping" / "data" / "entso_15min.parquet",
    "de_renewable_forecast": ROOT / "pfc_shaping" / "data" / "de_renewable_forecast.parquet",
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
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _score(df: pd.DataFrame, forecast_col: str = "forecast", actual_col: str = "actual") -> dict[str, float | int]:
    clean = df[[forecast_col, actual_col]].dropna()
    if clean.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "wape": float("nan"), "mape": float("nan"), "corr": float("nan"), "n": 0}
    err = clean[forecast_col] - clean[actual_col]
    abs_err = err.abs()
    ape = abs_err / clean[actual_col].abs().clip(lower=1.0) * 100.0
    denom = float(clean[actual_col].abs().sum())
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "wape": float(abs_err.sum() / denom) if denom > 0 else float("nan"),
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

    peak_rows = []
    df["peak_bucket"] = np.where(df["local_ts"].dt.hour.between(7, 19), "peak", "offpeak")
    for bucket, bucket_df in df.groupby("peak_bucket"):
        row = {"bucket": str(bucket)}
        row.update(_score(bucket_df))
        peak_rows.append(row)

    horizon_rows = []
    if "horizon" in df.columns:
        for horizon, horizon_df in df.groupby("horizon"):
            row = {"horizon": int(horizon)}
            row.update(_score(horizon_df))
            horizon_rows.append(row)

    return {
        "by_hour": hour_rows,
        "by_segment": segment_rows,
        "peak_offpeak": peak_rows,
        "by_horizon_day": horizon_rows,
    }


def _foundation_metrics(bt: pd.DataFrame) -> dict[str, object] | None:
    if "fm_raw_price" not in bt.columns:
        return None
    fm = bt.dropna(subset=["fm_raw_price", "actual"]).copy()
    if fm.empty:
        return None
    fm_eval = fm.copy()
    fm_eval["forecast"] = fm_eval["fm_raw_price"]
    return {
        "overall": _score(fm, forecast_col="fm_raw_price"),
        "breakdown": _segment_metrics(fm_eval),
    }


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


def _archive_inputs(run_id: str, manifest: dict[str, object]) -> None:
    archive_dir = ROOT / "pfc_shaping" / "data" / "archive" / run_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name, path in DATA_FILES.items():
        if path.exists():
            dest = archive_dir / path.name
            shutil.copy2(path, dest)
            manifest["inputs"][name]["archive"] = str(dest)


def _run_variant(
    label: str,
    physical_features: bool,
    use_future_covariates: bool,
    use_de_renewable_future: bool,
    epex_ch: pd.DataFrame,
    epex_de: pd.DataFrame,
    entso: pd.DataFrame,
    de_renewable_forecast: pd.DataFrame | None,
    n_days: int,
    horizon: int,
    use_foundation_model: bool,
) -> pd.DataFrame:
    model = LEARForecaster(
        use_foundation_model=use_foundation_model,
        use_extended_physical_ch_features=physical_features,
        use_future_covariates=use_future_covariates,
        use_de_renewable_future=use_de_renewable_future,
    )
    model.fit(
        epex_15min=epex_ch,
        entso_15min=entso,
        epex_de_15min=epex_de,
        de_renewable_forecast=de_renewable_forecast,
    )
    bt = model.backtest(n_days=n_days, horizon=horizon, include_fm_attribution=True).copy()
    bt["variant"] = label
    return bt


def _parse_horizons(value: str | None, fallback: int) -> list[int]:
    if not value:
        return [int(fallback)]
    horizons = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        horizons.append(int(part))
    return sorted(set(horizons))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible LEAR feature A/B.")
    parser.add_argument("--n-days", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--horizons", type=str, default=None, help="Comma-separated D+ horizons, e.g. 1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--block-hours", type=int, default=24)
    parser.add_argument("--use-foundation-model", action="store_true")
    parser.add_argument("--physical-features", action="store_true")
    parser.add_argument("--archive-inputs", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "pfc_shaping" / "output")
    args = parser.parse_args()

    _set_seed(args.seed)
    run_id = datetime.now(timezone.utc).strftime("lear_feature_ab_%Y%m%d_%H%M%S")
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
    if args.archive_inputs:
        _archive_inputs(run_id, manifest)

    epex_ch = pd.read_parquet(DATA_FILES["epex_ch"])
    epex_de = pd.read_parquet(DATA_FILES["epex_de"])
    entso = pd.read_parquet(DATA_FILES["entso"])
    de_renewable_forecast = (
        pd.read_parquet(DATA_FILES["de_renewable_forecast"])
        if DATA_FILES["de_renewable_forecast"].exists()
        else None
    )

    horizons = _parse_horizons(args.horizons, args.horizon)
    variants = {
        "A_baseline": (False, False),
        "B_calendar_future": (True, False),
        "C_calendar_de_future": (True, True),
    }
    backtests: dict[str, pd.DataFrame] = {}
    for label, (future_flag, de_future_flag) in variants.items():
        frames = []
        for horizon in horizons:
            frames.append(
                _run_variant(
                    label,
                    args.physical_features,
                    future_flag,
                    de_future_flag,
                    epex_ch,
                    epex_de,
                    entso,
                    de_renewable_forecast,
                    args.n_days,
                    horizon,
                    args.use_foundation_model,
                )
            )
        backtests[label] = pd.concat(frames, ignore_index=True)

    backtest_paths = {}
    for label, bt in backtests.items():
        path = args.output_dir / f"{run_id}_{label}.parquet"
        bt.to_parquet(path, index=False)
        backtest_paths[label] = str(path)

    scores = {label: _score(bt) for label, bt in backtests.items()}
    breakdowns = {label: _segment_metrics(bt) for label, bt in backtests.items()}
    foundation = {label: _foundation_metrics(bt) for label, bt in backtests.items()}

    payload = {
        **manifest,
        "horizons": horizons,
        "backtests": backtest_paths,
        "scores": scores,
        "delta_vs_A": {
            label: {
                k: scores[label][k] - scores["A_baseline"][k]
                for k in ["mae", "rmse", "wape", "mape", "corr"]
            }
            for label in ["B_calendar_future", "C_calendar_de_future"]
        },
        "bootstrap_delta_mae_vs_A": {
            label: _block_bootstrap_delta(
                backtests["A_baseline"],
                backtests[label],
                seed=args.seed,
                n_boot=args.bootstrap,
                block_hours=args.block_hours,
            )
            for label in ["B_calendar_future", "C_calendar_de_future"]
        },
        "breakdowns": breakdowns,
        "foundation_only": foundation,
    }

    output_path = args.output_dir / f"{run_id}.json"
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload["delta_vs_A"], indent=2))
    print(f"output:{output_path.resolve()}")


if __name__ == "__main__":
    main()

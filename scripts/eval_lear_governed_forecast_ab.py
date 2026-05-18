#!/usr/bin/env python3
"""Reproducible A/B harness for governed forecast feature evaluation."""

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
    "entso": ROOT / "pfc_shaping" / "data" / "entso_fundamentals_15min.parquet",
    "hydro": ROOT / "pfc_shaping" / "data" / "hydro_reservoir.parquet",
    "outages": ROOT / "pfc_shaping" / "data" / "outages_15min.parquet",
    "commodities": ROOT / "data" / "commodities_cache.parquet",
    "de_renewable_forecast": ROOT / "pfc_shaping" / "data" / "de_renewable_forecast.parquet",
    "multi_country_forecast": ROOT / "pfc_shaping" / "data" / "multi_country_forecast_15min.parquet",
    "weather_forecast": ROOT / "pfc_shaping" / "data" / "weather_forecast_hourly.parquet",
}


def _read_optional_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return None if df.empty else df


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
    return {"by_segment": segment_rows}


def _run_variant(
    label: str,
    use_governed_forecast_features: bool,
    epex_ch: pd.DataFrame,
    epex_de: pd.DataFrame,
    entso: pd.DataFrame,
    hydro: pd.DataFrame,
    outages: pd.DataFrame | None,
    commodities: pd.DataFrame | None,
    de_renewable_forecast: pd.DataFrame | None,
    multi_country_forecast: pd.DataFrame | None,
    weather_forecast: pd.DataFrame | None,
    n_days: int,
    horizon: int,
) -> pd.DataFrame:
    model = LEARForecaster(
        use_foundation_model=True,
        use_governed_forecast_features=use_governed_forecast_features,
    )
    model.fit(
        epex_15min=epex_ch,
        entso_15min=entso,
        outages_15min=outages,
        commodities=commodities,
        hydro=hydro,
        epex_de_15min=epex_de,
        de_renewable_forecast=de_renewable_forecast,
        multi_country_forecast=multi_country_forecast,
        weather_forecast=weather_forecast,
    )
    bt = model.backtest(n_days=n_days, horizon=horizon).copy()
    bt["variant"] = label
    return bt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run governed forecast feature A/B.")
    parser.add_argument("--n-days", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "pfc_shaping" / "output")
    args = parser.parse_args()

    _set_seed(args.seed)
    run_id = datetime.now(timezone.utc).strftime("lear_governed_forecast_ab_%Y%m%d_%H%M%S")
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
    hydro = pd.read_parquet(DATA_FILES["hydro"])
    outages = _read_optional_parquet(DATA_FILES["outages"])
    commodities = _read_optional_parquet(DATA_FILES["commodities"])
    de_renewable_forecast = _read_optional_parquet(DATA_FILES["de_renewable_forecast"])
    multi_country_forecast = _read_optional_parquet(DATA_FILES["multi_country_forecast"])
    weather_forecast = _read_optional_parquet(DATA_FILES["weather_forecast"])

    baseline = _run_variant(
        "baseline",
        False,
        epex_ch,
        epex_de,
        entso,
        hydro,
        outages,
        commodities,
        de_renewable_forecast,
        multi_country_forecast,
        weather_forecast,
        args.n_days,
        args.horizon,
    )
    experiment = _run_variant(
        "governed_forecast_features",
        True,
        epex_ch,
        epex_de,
        entso,
        hydro,
        outages,
        commodities,
        de_renewable_forecast,
        multi_country_forecast,
        weather_forecast,
        args.n_days,
        args.horizon,
    )

    baseline_path = args.output_dir / f"{run_id}_baseline.parquet"
    experiment_path = args.output_dir / f"{run_id}_governed_forecast_features.parquet"
    baseline.to_parquet(baseline_path, index=False)
    experiment.to_parquet(experiment_path, index=False)

    baseline_score = _score(baseline)
    experiment_score = _score(experiment)

    payload = {
        **manifest,
        "baseline_backtest": str(baseline_path),
        "governed_forecast_features_backtest": str(experiment_path),
        "baseline": baseline_score,
        "governed_forecast_features": experiment_score,
        "delta": {
            k: experiment_score[k] - baseline_score[k]
            for k in ["mae", "rmse", "mape", "corr"]
        },
        "baseline_breakdown": _segment_metrics(baseline),
        "governed_breakdown": _segment_metrics(experiment),
    }

    output_path = args.output_dir / f"{run_id}.json"
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload["delta"], indent=2))
    print(f"output:{output_path.resolve()}")


if __name__ == "__main__":
    main()

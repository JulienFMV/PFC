#!/usr/bin/env python3
"""Reproducible multi-horizon 2x2 A/B harness for Swiss CT governed features."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

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
    "fr_nuclear_forecast": ROOT / "pfc_shaping" / "data" / "fr_nuclear_forecast_15min.parquet",
    "weather_forecast": ROOT / "pfc_shaping" / "data" / "weather_forecast_hourly.parquet",
}

VARIANTS = (
    ("no_fm_baseline", False, False),
    ("fm_only", True, False),
    ("no_fm_governed", False, True),
    ("fm_governed", True, True),
)


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


def _parse_horizons(args_horizons: str | None, args_horizon: int | None) -> list[int]:
    if args_horizons:
        return [int(x.strip()) for x in args_horizons.split(",") if x.strip()]
    if args_horizon is not None:
        return [int(args_horizon)]
    return [1, 5, 10]


def _verify_vintage_schema(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path),
        "has_publication_ts": False,
        "has_delivery_ts": False,
        "vintage_schema_verified": False,
    }
    if not path.exists():
        return result
    try:
        df = pd.read_parquet(path)
    except Exception:
        return result
    cols = set(df.columns)
    has_publication_ts = "publication_ts" in cols
    has_delivery_ts = "delivery_ts" in cols
    result["has_publication_ts"] = has_publication_ts
    result["has_delivery_ts"] = has_delivery_ts
    result["vintage_schema_verified"] = bool(has_publication_ts and has_delivery_ts)
    return result


def _aligned_frame(bt_a: pd.DataFrame, bt_b: pd.DataFrame) -> pd.DataFrame:
    a = _with_timestamp(bt_a).rename(columns={"forecast": "forecast_a", "actual": "actual_a"})
    b = _with_timestamp(bt_b).rename(columns={"forecast": "forecast_b", "actual": "actual_b"})
    cols = ["timestamp", "date", "hour", "forecast_a", "actual_a"]
    a = a[cols]
    b = b[["timestamp", "forecast_b", "actual_b"]]
    merged = a.merge(b, on="timestamp", how="inner")
    if merged.empty:
        raise ValueError("No overlapping timestamps between compared variants.")
    merged["actual"] = merged["actual_a"]
    return merged


def _newey_west_variance_mean(d: np.ndarray, lag: int) -> float:
    n = len(d)
    if n <= 1:
        return float("nan")
    x = d - np.mean(d)
    gamma0 = float(np.dot(x, x) / n)
    var = gamma0
    for ell in range(1, min(lag, n - 1) + 1):
        gamma = float(np.dot(x[ell:], x[:-ell]) / n)
        weight = 1.0 - ell / (lag + 1.0)
        var += 2.0 * weight * gamma
    return max(var / n, 1e-12)


def _dm_test(bt_a: pd.DataFrame, bt_b: pd.DataFrame, horizon: int) -> dict[str, float | int]:
    aligned = _aligned_frame(bt_a, bt_b)
    err_a = aligned["forecast_a"] - aligned["actual"]
    err_b = aligned["forecast_b"] - aligned["actual"]
    out: dict[str, float | int] = {"n": int(len(aligned)), "lag": int(max(1, horizon))}
    for loss_name, loss_a, loss_b in [
        ("mae_loss", err_a.abs(), err_b.abs()),
        ("mse_loss", err_a.pow(2), err_b.pow(2)),
    ]:
        d = (loss_a - loss_b).to_numpy(dtype=float)
        mean_d = float(np.mean(d))
        var_mean = _newey_west_variance_mean(d, lag=max(1, horizon))
        stat = mean_d / math.sqrt(var_mean)
        p_value = float(1.0 - norm.cdf(stat))  # H1: model B better than model A
        out[loss_name] = {
            "dm_stat": float(stat),
            "p_value_one_sided": p_value,
            "mean_loss_diff": mean_d,
        }
    return out


def _bootstrap_metric_diff(
    bt_a: pd.DataFrame,
    bt_b: pd.DataFrame,
    metric: str,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    aligned = _aligned_frame(bt_a, bt_b)
    rng = np.random.default_rng(seed)
    n = len(aligned)
    idx = np.arange(n)
    diffs = []
    for _ in range(int(n_boot)):
        sample = rng.choice(idx, size=n, replace=True)
        sub = aligned.iloc[sample]
        err_a = sub["forecast_a"] - sub["actual"]
        err_b = sub["forecast_b"] - sub["actual"]
        if metric == "mae":
            val = float(err_b.abs().mean() - err_a.abs().mean())
        elif metric == "rmse":
            val = float(np.sqrt((err_b**2).mean()) - np.sqrt((err_a**2).mean()))
        else:
            raise ValueError(metric)
        diffs.append(val)
    arr = np.asarray(diffs, dtype=float)
    return {
        "point_estimate": float(arr.mean()),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def _pairwise_stats(
    variants_bt: dict[str, pd.DataFrame],
    horizon: int,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    pairwise: dict[str, object] = {}
    for a, b in itertools.combinations(variants_bt.keys(), 2):
        key = f"{a}__vs__{b}"
        pairwise[key] = {
            "dm_test": _dm_test(variants_bt[a], variants_bt[b], horizon=horizon),
            "bootstrap_diff_b_minus_a": {
                "mae": _bootstrap_metric_diff(variants_bt[a], variants_bt[b], "mae", n_boot=n_boot, seed=seed),
                "rmse": _bootstrap_metric_diff(variants_bt[a], variants_bt[b], "rmse", n_boot=n_boot, seed=seed + 17),
            },
        }
    return pairwise


def _decomposition(
    variants_bt: dict[str, pd.DataFrame],
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    base = variants_bt["no_fm_baseline"]
    fm = variants_bt["fm_only"]
    no_fm_gov = variants_bt["no_fm_governed"]
    fm_gov = variants_bt["fm_governed"]

    def mae(df: pd.DataFrame) -> float:
        return _score(df)["mae"]  # type: ignore[index]

    point_total = mae(fm_gov) - mae(base)
    point_from_fm = mae(fm) - mae(base)
    point_from_gov_given_fm = mae(fm_gov) - mae(fm)
    point_interaction = point_total - point_from_fm - point_from_gov_given_fm
    point_from_gov_without_fm = mae(no_fm_gov) - mae(base)

    aligned = {
        "base_fm": _aligned_frame(base, fm),
        "fm_fmgov": _aligned_frame(fm, fm_gov),
        "base_fmgov": _aligned_frame(base, fm_gov),
        "base_nofmgov": _aligned_frame(base, no_fm_gov),
    }
    rng = np.random.default_rng(seed)
    n = min(len(v) for v in aligned.values())
    boot = {
        "total_gain": [],
        "from_FM": [],
        "from_governed_given_FM": [],
        "from_governed_without_FM": [],
        "interaction": [],
    }
    for _ in range(int(n_boot)):
        sample = rng.choice(np.arange(n), size=n, replace=True)
        total = float((aligned["base_fmgov"].iloc[sample]["forecast_b"] - aligned["base_fmgov"].iloc[sample]["actual"]).abs().mean()
                      - (aligned["base_fmgov"].iloc[sample]["forecast_a"] - aligned["base_fmgov"].iloc[sample]["actual"]).abs().mean())
        from_fm = float((aligned["base_fm"].iloc[sample]["forecast_b"] - aligned["base_fm"].iloc[sample]["actual"]).abs().mean()
                        - (aligned["base_fm"].iloc[sample]["forecast_a"] - aligned["base_fm"].iloc[sample]["actual"]).abs().mean())
        from_gov_fm = float((aligned["fm_fmgov"].iloc[sample]["forecast_b"] - aligned["fm_fmgov"].iloc[sample]["actual"]).abs().mean()
                            - (aligned["fm_fmgov"].iloc[sample]["forecast_a"] - aligned["fm_fmgov"].iloc[sample]["actual"]).abs().mean())
        from_gov_nofm = float((aligned["base_nofmgov"].iloc[sample]["forecast_b"] - aligned["base_nofmgov"].iloc[sample]["actual"]).abs().mean()
                              - (aligned["base_nofmgov"].iloc[sample]["forecast_a"] - aligned["base_nofmgov"].iloc[sample]["actual"]).abs().mean())
        interaction = total - from_fm - from_gov_fm
        boot["total_gain"].append(total)
        boot["from_FM"].append(from_fm)
        boot["from_governed_given_FM"].append(from_gov_fm)
        boot["from_governed_without_FM"].append(from_gov_nofm)
        boot["interaction"].append(interaction)

    def summarize(name: str, point: float) -> dict[str, float]:
        arr = np.asarray(boot[name], dtype=float)
        pct = float(100.0 * point / abs(point_total)) if abs(point_total) > 1e-9 else 0.0
        return {
            "point_estimate": float(point),
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)),
            "pct_of_total_gain": pct,
        }

    return {
        "mae_decomposition": {
            "total_gain": summarize("total_gain", point_total),
            "from_FM": summarize("from_FM", point_from_fm),
            "from_governed_given_FM": summarize("from_governed_given_FM", point_from_gov_given_fm),
            "from_governed_without_FM": summarize("from_governed_without_FM", point_from_gov_without_fm),
            "interaction": summarize("interaction", point_interaction),
        }
    }


def _run_variant(
    label: str,
    use_foundation_model: bool,
    use_governed_forecast_features: bool,
    epex_ch: pd.DataFrame,
    epex_de: pd.DataFrame,
    entso: pd.DataFrame,
    hydro: pd.DataFrame,
    outages: pd.DataFrame | None,
    commodities: pd.DataFrame | None,
    de_renewable_forecast: pd.DataFrame | None,
    multi_country_forecast: pd.DataFrame | None,
    fr_nuclear_forecast: pd.DataFrame | None,
    weather_forecast: pd.DataFrame | None,
    n_days: int,
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    model = LEARForecaster(
        use_foundation_model=use_foundation_model,
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
        fr_nuclear_forecast=fr_nuclear_forecast,
        weather_forecast=weather_forecast,
    )
    governed_health = model.assess_governed_forecast_feature_health(
        horizon_days=max(1, int(horizon)),
        min_coverage_ratio=float(os.getenv("PFC_CT_MIN_GOVERNED_FORECAST_COVERAGE", "0.98")),
    )
    bt = model.backtest(n_days=n_days, horizon=horizon).copy()
    bt["variant"] = label
    return bt, governed_health


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Swiss CT governed feature 2x2 A/B.")
    parser.add_argument("--n-days", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--horizons", type=str, default="1,5,10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "pfc_shaping" / "output")
    args = parser.parse_args()

    _set_seed(args.seed)
    run_id = datetime.now(timezone.utc).strftime("lear_governed_forecast_ab_%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    horizons = _parse_horizons(args.horizons, args.horizon)

    manifest: dict[str, object] = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "seed": int(args.seed),
        "args": {
            **vars(args),
            "horizons": horizons,
        },
        "inputs": {},
    }
    for name, path in DATA_FILES.items():
        manifest["inputs"][name] = {
            "path": str(path),
            "sha256": _sha256(path) if path.exists() else None,
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat() if path.exists() else None,
        }
    vintage_checks = {
        "multi_country_forecast": _verify_vintage_schema(DATA_FILES["multi_country_forecast"]),
        "weather_forecast": _verify_vintage_schema(DATA_FILES["weather_forecast"]),
    }
    manifest["vintage_schema_verified"] = bool(
        vintage_checks["multi_country_forecast"]["vintage_schema_verified"]
        and vintage_checks["weather_forecast"]["vintage_schema_verified"]
    )
    manifest["vintage_schema_checks"] = vintage_checks

    epex_ch = pd.read_parquet(DATA_FILES["epex_ch"])
    epex_de = pd.read_parquet(DATA_FILES["epex_de"])
    entso = pd.read_parquet(DATA_FILES["entso"])
    hydro = pd.read_parquet(DATA_FILES["hydro"])
    outages = _read_optional_parquet(DATA_FILES["outages"])
    commodities = _read_optional_parquet(DATA_FILES["commodities"])
    de_renewable_forecast = _read_optional_parquet(DATA_FILES["de_renewable_forecast"])
    multi_country_forecast = _read_optional_parquet(DATA_FILES["multi_country_forecast"])
    fr_nuclear_forecast = _read_optional_parquet(DATA_FILES["fr_nuclear_forecast"])
    weather_forecast = _read_optional_parquet(DATA_FILES["weather_forecast"])

    horizons_payload: dict[str, object] = {}
    summary: dict[str, object] = {}

    for horizon in horizons:
        variants_bt: dict[str, pd.DataFrame] = {}
        variants_health: dict[str, dict[str, object]] = {}
        backtest_paths: dict[str, str] = {}
        scores: dict[str, dict[str, float | int]] = {}
        breakdowns: dict[str, object] = {}

        for label, use_fm, use_gov in VARIANTS:
            bt, health = _run_variant(
                label,
                use_fm,
                use_gov,
                epex_ch,
                epex_de,
                entso,
                hydro,
                outages,
                commodities,
                de_renewable_forecast,
                multi_country_forecast,
                fr_nuclear_forecast,
                weather_forecast,
                args.n_days,
                horizon,
            )
            variants_bt[label] = bt
            variants_health[label] = health
            bt_path = args.output_dir / f"{run_id}_h{horizon}_{label}.parquet"
            bt.to_parquet(bt_path, index=False)
            backtest_paths[label] = str(bt_path)
            scores[label] = _score(bt)
            breakdowns[label] = _segment_metrics(bt)

        pairwise = _pairwise_stats(
            variants_bt=variants_bt,
            horizon=horizon,
            n_boot=int(args.bootstrap_samples),
            seed=int(args.seed) + horizon * 101,
        )
        decomp = _decomposition(
            variants_bt=variants_bt,
            n_boot=int(args.bootstrap_samples),
            seed=int(args.seed) + horizon * 211,
        )

        horizon_key = f"h{horizon}"
        horizons_payload[horizon_key] = {
            "horizon": int(horizon),
            "variant_scores": scores,
            "variant_health": variants_health,
            "backtests": backtest_paths,
            "pairwise_stats": pairwise,
            "decomposition": decomp,
            "breakdowns": breakdowns,
        }
        summary[horizon_key] = {
            "no_fm_baseline_mae": scores["no_fm_baseline"]["mae"],
            "fm_only_mae": scores["fm_only"]["mae"],
            "no_fm_governed_mae": scores["no_fm_governed"]["mae"],
            "fm_governed_mae": scores["fm_governed"]["mae"],
            "dm_p_fm_governed_vs_fm_only_mae": pairwise["fm_only__vs__fm_governed"]["dm_test"]["mae_loss"]["p_value_one_sided"],
            "bootstrap_ci_fm_governed_vs_fm_only_mae": pairwise["fm_only__vs__fm_governed"]["bootstrap_diff_b_minus_a"]["mae"],
            "vintage_schema_verified": manifest["vintage_schema_verified"],
        }

    payload = {
        **manifest,
        "summary": summary,
        "horizons": horizons_payload,
    }

    output_path = args.output_dir / f"{run_id}.json"
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"output:{output_path.resolve()}")


if __name__ == "__main__":
    main()

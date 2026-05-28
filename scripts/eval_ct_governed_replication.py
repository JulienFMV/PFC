#!/usr/bin/env python3
"""Structured replication harness for Swiss CT governed feature evidence."""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_lear_governed_forecast_ab import (
    DATA_FILES,
    VARIANTS,
    _decomposition,
    _pairwise_stats,
    _parse_horizons,
    _read_optional_parquet,
    _run_variant,
    _score,
    _set_seed,
    _sha256,
    _verify_vintage_schema,
)


def _load_inputs() -> dict[str, pd.DataFrame | None]:
    return {
        "epex_ch": pd.read_parquet(DATA_FILES["epex_ch"]),
        "epex_de": pd.read_parquet(DATA_FILES["epex_de"]),
        "entso": pd.read_parquet(DATA_FILES["entso"]),
        "hydro": pd.read_parquet(DATA_FILES["hydro"]),
        "outages": _read_optional_parquet(DATA_FILES["outages"]),
        "commodities": _read_optional_parquet(DATA_FILES["commodities"]),
        "de_renewable_forecast": _read_optional_parquet(DATA_FILES["de_renewable_forecast"]),
        "multi_country_forecast": _read_optional_parquet(DATA_FILES["multi_country_forecast"]),
        "fr_nuclear_forecast": _read_optional_parquet(DATA_FILES["fr_nuclear_forecast"]),
        "weather_forecast": _read_optional_parquet(DATA_FILES["weather_forecast"]),
    }


def _slice_frame(df: pd.DataFrame | None, cutoff_utc: pd.Timestamp) -> pd.DataFrame | None:
    if df is None:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    if df.index.tz is None:
        return df.loc[df.index <= cutoff_utc.tz_localize(None)].copy()
    return df.loc[df.index <= cutoff_utc].copy()


def _derive_default_cutoffs(
    epex_ch: pd.DataFrame,
    first_governed_ts: pd.Timestamp,
    n_periods: int,
    freq: str,
) -> list[pd.Timestamp]:
    latest = epex_ch.index.max().tz_convert("UTC")
    start = max(first_governed_ts, epex_ch.index.min().tz_convert("UTC"))
    candidates = pd.date_range(start=start.normalize(), end=latest.normalize(), freq=freq, tz="UTC")
    cutoffs = [ts + pd.Timedelta(hours=23, minutes=45) for ts in candidates]
    cutoffs = [ts for ts in cutoffs if ts <= latest]
    return cutoffs[-int(n_periods):]


def _parse_cutoffs(args_cutoffs: str | None, epex_ch: pd.DataFrame, first_governed_ts: pd.Timestamp, n_periods: int, freq: str) -> list[pd.Timestamp]:
    if args_cutoffs:
        return [pd.Timestamp(item.strip(), tz="UTC") + pd.Timedelta(hours=23, minutes=45) for item in args_cutoffs.split(",") if item.strip()]
    return _derive_default_cutoffs(epex_ch=epex_ch, first_governed_ts=first_governed_ts, n_periods=n_periods, freq=freq)


def _replication_label(cutoff_utc: pd.Timestamp) -> str:
    return cutoff_utc.tz_convert("Europe/Zurich").strftime("%Y%m%d")


def _collect_manifest(output_dir: Path, args: argparse.Namespace, cutoffs: list[pd.Timestamp]) -> dict[str, object]:
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    manifest: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_result.stdout.strip() if git_result.returncode == 0 else "",
        "args": {
            **vars(args),
            "cutoffs_utc": [ts.isoformat() for ts in cutoffs],
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
    manifest["output_dir"] = str(output_dir)
    return manifest


def _wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * (((phat * (1.0 - phat)) / n + (z * z) / (4.0 * n * n)) ** 0.5)
    return max(0.0, center - margin), min(1.0, center + margin)


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured Swiss CT governed replications across cutoffs and seeds.")
    parser.add_argument("--horizons", type=str, default="2,3,4,5")
    parser.add_argument("--n-days", type=int, default=15)
    parser.add_argument("--bootstrap-samples", type=int, default=250)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--cutoffs", type=str, default=None, help="Comma-separated UTC dates (YYYY-MM-DD) used as replication cutoffs.")
    parser.add_argument("--n-periods", type=int, default=4)
    parser.add_argument("--freq", type=str, default="ME", help="Pandas date_range frequency for default cutoffs, e.g. ME or QE.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "pfc_shaping" / "output" / "governed_replications")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    horizons = _parse_horizons(args.horizons, None)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    loaded = _load_inputs()
    first_governed_ts = min(
        df.index.min().tz_convert("UTC")
        for name, df in loaded.items()
        if name in {"multi_country_forecast", "weather_forecast", "fr_nuclear_forecast"} and isinstance(df, pd.DataFrame) and not df.empty
    )
    cutoffs = _parse_cutoffs(args.cutoffs, loaded["epex_ch"], first_governed_ts, args.n_periods, args.freq)
    manifest = _collect_manifest(args.output_dir, args, cutoffs)

    replications: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for cutoff in cutoffs:
        sliced = {name: _slice_frame(df, cutoff) for name, df in loaded.items()}
        for seed in seeds:
            _set_seed(seed)
            for horizon in horizons:
                variants_bt: dict[str, pd.DataFrame] = {}
                variants_health: dict[str, dict[str, object]] = {}
                scores: dict[str, dict[str, float | int]] = {}
                for label, use_fm, use_gov in VARIANTS:
                    bt, health = _run_variant(
                        label,
                        use_fm,
                        use_gov,
                        sliced["epex_ch"],
                        sliced["epex_de"],
                        sliced["entso"],
                        sliced["hydro"],
                        sliced["outages"],
                        sliced["commodities"],
                        sliced["de_renewable_forecast"],
                        sliced["multi_country_forecast"],
                        sliced["fr_nuclear_forecast"],
                        sliced["weather_forecast"],
                        args.n_days,
                        horizon,
                        seed,
                    )
                    variants_bt[label] = bt
                    variants_health[label] = health
                    scores[label] = _score(bt)

                pairwise = _pairwise_stats(
                    variants_bt=variants_bt,
                    horizon=horizon,
                    n_boot=int(args.bootstrap_samples),
                    seed=int(seed) + horizon * 101,
                )
                decomp = _decomposition(
                    variants_bt=variants_bt,
                    n_boot=int(args.bootstrap_samples),
                    seed=int(seed) + horizon * 211,
                )

                gov_vs_base = pairwise["no_fm_baseline__vs__no_fm_governed"]["dm_test"]["mae_loss"]["p_value_one_sided"]
                fm_cond = pairwise["fm_only__vs__fm_governed"]["dm_test"]["mae_loss"]["p_value_one_sided"]
                gov_ci = pairwise["no_fm_baseline__vs__no_fm_governed"]["bootstrap_diff_b_minus_a"]["mae"]
                fm_ci = pairwise["fm_only__vs__fm_governed"]["bootstrap_diff_b_minus_a"]["mae"]

                replication_id = f"{_replication_label(cutoff)}_s{seed}_h{horizon}"
                payload = {
                    "replication_id": replication_id,
                    "cutoff_utc": cutoff.isoformat(),
                    "cutoff_local_date": cutoff.tz_convert("Europe/Zurich").date().isoformat(),
                    "seed": seed,
                    "horizon": horizon,
                    "scores": scores,
                    "variant_health": variants_health,
                    "pairwise_stats": pairwise,
                    "decomposition": decomp,
                    "vintage_schema_verified": manifest["vintage_schema_verified"],
                }
                out_path = args.output_dir / f"{replication_id}.json"
                out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
                replications.append(payload)
                summary_rows.append(
                    {
                        "replication_id": replication_id,
                        "cutoff_local_date": cutoff.tz_convert("Europe/Zurich").date().isoformat(),
                        "seed": seed,
                        "horizon": horizon,
                        "baseline_mae": scores["no_fm_baseline"]["mae"],
                        "governed_mae": scores["no_fm_governed"]["mae"],
                        "fm_only_mae": scores["fm_only"]["mae"],
                        "fm_governed_mae": scores["fm_governed"]["mae"],
                        "governed_vs_baseline_dm_p": gov_vs_base,
                        "governed_vs_baseline_ci_low": gov_ci["ci_low"],
                        "governed_vs_baseline_ci_high": gov_ci["ci_high"],
                        "fm_governed_vs_fm_only_dm_p": fm_cond,
                        "fm_governed_vs_fm_only_ci_low": fm_ci["ci_low"],
                        "fm_governed_vs_fm_only_ci_high": fm_ci["ci_high"],
                        "governed_nominal_significant": bool(gov_vs_base < 0.05 and gov_ci["ci_high"] < 0.0),
                        "fm_conditioned_nominal_significant": bool(fm_cond < 0.05 and fm_ci["ci_high"] < 0.0),
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(["horizon", "cutoff_local_date", "seed"]).reset_index(drop=True)
    summary_csv = args.output_dir / "replication_summary_latest.csv"
    summary_df.to_csv(summary_csv, index=False)

    aggregate_rows: list[dict[str, object]] = []
    for horizon, group in summary_df.groupby("horizon"):
        n_replications = int(len(group))
        k_gov = int(group["governed_nominal_significant"].sum())
        k_fm = int(group["fm_conditioned_nominal_significant"].sum())
        gov_ci_low, gov_ci_high = _wilson_interval(k_gov, n_replications)
        fm_ci_low, fm_ci_high = _wilson_interval(k_fm, n_replications)
        aggregate_rows.append(
            {
                "horizon": int(horizon),
                "n_replications": n_replications,
                "baseline_mae_mean": float(group["baseline_mae"].mean()),
                "governed_mae_mean": float(group["governed_mae"].mean()),
                "governed_mae_delta_mean": float((group["governed_mae"] - group["baseline_mae"]).mean()),
                "governed_mae_delta_median": float((group["governed_mae"] - group["baseline_mae"]).median()),
                "governed_nominal_significant_rate": float(group["governed_nominal_significant"].mean()),
                "governed_nominal_significant_count": k_gov,
                "governed_nominal_significant_wilson_ci_low": gov_ci_low,
                "governed_nominal_significant_wilson_ci_high": gov_ci_high,
                "governed_nominal_significant_binom_p": float(
                    binomtest(k_gov, n_replications, p=0.025, alternative="greater").pvalue
                ),
                "fm_conditioned_nominal_significant_rate": float(group["fm_conditioned_nominal_significant"].mean()),
                "fm_conditioned_nominal_significant_count": k_fm,
                "fm_conditioned_nominal_significant_wilson_ci_low": fm_ci_low,
                "fm_conditioned_nominal_significant_wilson_ci_high": fm_ci_high,
                "fm_conditioned_nominal_significant_binom_p": float(
                    binomtest(k_fm, n_replications, p=0.025, alternative="greater").pvalue
                ),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values("horizon").reset_index(drop=True)
    aggregate_json = args.output_dir / "replication_aggregate_latest.json"
    aggregate_payload = {
        **manifest,
        "horizons": horizons,
        "seeds": seeds,
        "cutoffs_utc": [ts.isoformat() for ts in cutoffs],
        "n_replications": int(len(summary_df)),
        "aggregate": aggregate.to_dict(orient="records"),
        "summary_csv": str(summary_csv),
    }
    aggregate_json.write_text(json.dumps(aggregate_payload, indent=2, default=str), encoding="utf-8")
    print(summary_csv)
    print(aggregate_json)


if __name__ == "__main__":
    main()

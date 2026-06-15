"""Audit local/test CH hourly PFC shape quality."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.export_local_test_ch_hourly_csv import (
    _latest_eex_base_prices,
    _parse_timestamp_ch,
)
from pfc_shaping.calibration.eex_contract_selection import calibration_buckets


PRICE = "price_weighted_mean_eur_mwh"


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "timestamp_ch",
        "timestamp_utc",
        "price_slow_eur_mwh",
        "price_central_eur_mwh",
        "price_fast_eur_mwh",
        PRICE,
        "structural_p10_eur_mwh",
        "structural_p50_eur_mwh",
        "structural_p90_eur_mwh",
        "structural_width_eur_mwh",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df["ts"] = _parse_timestamp_ch(df["timestamp_ch"], df.get("utc_offset_ch"))
    df["year"] = df["ts"].dt.year
    df["month"] = df["ts"].dt.month
    df["hour"] = df["ts"].dt.hour
    df["weekday"] = df["ts"].dt.weekday
    return df


def _product_residuals(df: pd.DataFrame, forward_prices: dict[str, float]) -> pd.DataFrame:
    products, targets = calibration_buckets(df["ts"], forward_prices)
    rows = []
    for product, idx in products.groupby(products).groups.items():
        if product is None or product not in targets:
            continue
        mean = float(df.loc[idx, PRICE].mean())
        target = float(targets[str(product)])
        rows.append(
            {
                "product": product,
                "target_eex_base_eur_mwh": target,
                "csv_mean_eur_mwh": mean,
                "abs_error_eur_mwh": abs(mean - target),
                "rows": len(idx),
            }
        )
    return pd.DataFrame(rows).sort_values("product").reset_index(drop=True)


def audit(csv_path: Path, forwards_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    df = _load_csv(csv_path)
    _, forward_prices = _latest_eex_base_prices(forwards_path, market="CH")
    residuals = _product_residuals(df, forward_prices)

    numeric_cols = [c for c in df.columns if c.endswith("_eur_mwh")]
    finite_ok = bool(np.isfinite(df[numeric_cols].to_numpy(dtype=float)).all())
    no_negative = bool((df[numeric_cols] >= 0.0).all().all())
    min_price = float(df[numeric_cols].min().min())
    weighted_negative_hours = int((df[PRICE] < 0.0).sum())
    p10_negative_hours = int((df["structural_p10_eur_mwh"] < 0.0).sum())
    bounded_negative_ok = bool(
        min_price >= -30.0
        and weighted_negative_hours == 0
        and p10_negative_hours <= max(1, int(len(df) * 0.01))
    )
    quantile_order = bool(
        (
            (df["structural_p10_eur_mwh"] <= df["structural_p50_eur_mwh"])
            & (df["structural_p50_eur_mwh"] <= df["structural_p90_eur_mwh"])
        ).all()
    )
    ramp = df[PRICE].diff().abs().dropna()
    boundary_jumps = []
    for i in range(1, len(df)):
        t = df["ts"].iloc[i]
        if t.day == 1 and t.hour == 0:
            boundary_jumps.append(abs(float(df[PRICE].iloc[i] - df[PRICE].iloc[i - 1])))
    boundary = pd.Series(boundary_jumps, dtype=float)

    rows = []
    for year, group in df.groupby("year"):
        midday = float(group[group["hour"].between(11, 15)][PRICE].mean())
        evening = float(group[group["hour"].between(17, 21)][PRICE].mean())
        weekday = float(group[group["weekday"] < 5][PRICE].mean())
        weekend = float(group[group["weekday"] >= 5][PRICE].mean())
        rows.append(
            {
                "year": int(year),
                "mean_eur_mwh": float(group[PRICE].mean()),
                "evening_minus_midday_eur_mwh": evening - midday,
                "weekend_minus_weekday_eur_mwh": weekend - weekday,
                "structural_width_mean_eur_mwh": float(group["structural_width_eur_mwh"].mean()),
                "structural_width_p95_eur_mwh": float(group["structural_width_eur_mwh"].quantile(0.95)),
            }
        )
    annual = pd.DataFrame(rows)

    max_eex_error = float(residuals["abs_error_eur_mwh"].max()) if not residuals.empty else float("inf")
    width_mean = float(df["structural_width_eur_mwh"].mean())
    width_p95 = float(df["structural_width_eur_mwh"].quantile(0.95))
    duck_2030 = float(annual.loc[annual["year"] == 2030, "evening_minus_midday_eur_mwh"].iloc[0])
    weekend_2030 = float(annual.loc[annual["year"] == 2030, "weekend_minus_weekday_eur_mwh"].iloc[0])
    ramp_p99 = float(ramp.quantile(0.99))
    ramp_max = float(ramp.max())
    boundary_p95 = float(boundary.quantile(0.95)) if len(boundary) else 0.0

    score = 0.0
    score += 1.5 if finite_ok and (no_negative or bounded_negative_ok) and quantile_order else 0.0
    score += 2.0 if max_eex_error <= 0.01 else 0.0
    score += 1.5 if 6.0 <= width_mean <= 10.0 and 18.0 <= width_p95 <= 30.0 else 0.75 if width_p95 >= 10.0 else 0.0
    score += 1.5 if 18.0 <= duck_2030 <= 30.0 else 0.75 if 12.0 <= duck_2030 <= 35.0 else 0.0
    score += 0.75 if weekend_2030 <= -4.0 else 0.25
    score += 1.0 if ramp_p99 <= 30.0 and ramp_max <= 70.0 else 0.5 if ramp_p99 <= 35.0 else 0.0
    score += 0.75 if boundary_p95 <= 20.0 else 0.25
    metrics = {
        "score_10": score,
        "finite_ok": float(finite_ok),
        "no_negative": float(no_negative),
        "bounded_negative_ok": float(bounded_negative_ok),
        "min_price_eur_mwh": min_price,
        "weighted_negative_hours": float(weighted_negative_hours),
        "p10_negative_hours": float(p10_negative_hours),
        "quantile_order": float(quantile_order),
        "max_eex_error_eur_mwh": max_eex_error,
        "structural_width_mean_eur_mwh": width_mean,
        "structural_width_p95_eur_mwh": width_p95,
        "duck_2030_evening_minus_midday_eur_mwh": duck_2030,
        "weekend_2030_minus_weekday_eur_mwh": weekend_2030,
        "ramp_abs_p99_eur_mwh": ramp_p99,
        "ramp_abs_max_eur_mwh": ramp_max,
        "boundary_jump_abs_p95_eur_mwh": boundary_p95,
    }
    return annual, residuals, metrics


def _md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "|" + "|".join("---" for _ in frame.columns) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path: Path, *, csv_path: Path, annual: pd.DataFrame, residuals: pd.DataFrame, metrics: dict[str, float]) -> None:
    lines = [
        "# CH Hourly PFC Shape Audit",
        "",
        f"* CSV: `{csv_path}`",
        f"* score: `{metrics['score_10']:.2f}/10`",
        "* scope: `local/test only`",
        "* production approval: `NO`",
        "",
        "## Metrics",
        "",
        _md_table(pd.DataFrame([metrics])),
        "",
        "## Annual Shape",
        "",
        _md_table(annual),
        "",
        "## EEX Residuals",
        "",
        _md_table(residuals),
        "",
        "## Notes",
        "",
        "* Score >= 8.5 is a local/test quality threshold, not production approval.",
        "* Remaining high boundary/ramp metrics require upstream smoothing before production use.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--forwards", default="data/eex_forwards_history.parquet")
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    annual, residuals, metrics = audit(Path(args.csv), Path(args.forwards))
    write_report(Path(args.report), csv_path=Path(args.csv), annual=annual, residuals=residuals, metrics=metrics)
    print(f"[shape-audit] score={metrics['score_10']:.2f}/10")
    print(f"[shape-audit] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

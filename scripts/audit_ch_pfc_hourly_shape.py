"""Audit local/test CH hourly PFC shape quality."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.export_local_test_ch_hourly_csv import (
    _eex_peak_mask,
    _latest_eex_prices_by_load_type,
    _parse_timestamp_ch,
)
from pfc_shaping.calibration.eex_contract_selection import calibration_buckets


PRICE = "price_weighted_mean_eur_mwh"


def _max_true_run(mask: pd.Series) -> int:
    max_run = 0
    current = 0
    for value in mask.astype(bool).to_numpy():
        if value:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return int(max_run)


def _negative_gate(df: pd.DataFrame, *, min_any_price: float, weighted_negative_hours: int) -> dict[str, float | str]:
    weighted = df[PRICE].astype(float)
    weighted_negative = weighted < 0.0
    weighted_negative_share_pct = 100.0 * float(weighted_negative.mean())
    allowed_zone = df["month"].between(4, 9) & df["hour"].between(10, 16)
    outside_allowed = int((weighted_negative & ~allowed_zone).sum())
    localization_pct = (
        100.0 * float((weighted_negative & allowed_zone).sum()) / float(weighted_negative_hours)
        if weighted_negative_hours
        else 100.0
    )
    max_per_month = (
        int(df.loc[weighted_negative].groupby(["year", "month"]).size().max())
        if weighted_negative_hours
        else 0
    )
    max_run = _max_true_run(weighted_negative)
    min_weighted = float(weighted.min())
    p10_negative_share_pct = 100.0 * float((df["structural_p10_eur_mwh"] < 0.0).mean())
    fast_negative_share_pct = 100.0 * float((df["price_fast_eur_mwh"] < 0.0).mean())
    max_allowed_hours = max(96, int(0.005 * len(df)))
    ok = (
        min_weighted >= -15.0
        and min_any_price >= -30.0
        and weighted_negative_hours <= max_allowed_hours
        and weighted_negative_share_pct <= 0.50
        and max_per_month <= 48
        and outside_allowed == 0
        and localization_pct >= 95.0
        and max_run <= 8
        and p10_negative_share_pct <= 2.0
        and fast_negative_share_pct <= 2.0
    )
    return {
        "negative_gate_status": "PASS" if ok else "FAIL",
        "min_weighted_eur_mwh": min_weighted,
        "weighted_negative_share_pct": weighted_negative_share_pct,
        "weighted_negative_outside_allowed_hours": float(outside_allowed),
        "negative_localization_pct": localization_pct,
        "max_weighted_negative_hours_per_month": float(max_per_month),
        "max_weighted_negative_run_hours": float(max_run),
        "p10_negative_share_pct": p10_negative_share_pct,
        "fast_negative_share_pct": fast_negative_share_pct,
    }


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


def _product_residuals(df: pd.DataFrame, forward_prices: dict[str, float], *, load_type: str) -> pd.DataFrame:
    products, targets = calibration_buckets(df["ts"], forward_prices)
    rows = []
    for product, idx in products.groupby(products).groups.items():
        if product is None or product not in targets:
            continue
        mean = float(df.loc[idx, PRICE].mean())
        target = float(targets[str(product)])
        rows.append(
            {
                "load_type": load_type,
                "product": product,
                "target_eex_eur_mwh": target,
                "csv_mean_eur_mwh": mean,
                "abs_error_eur_mwh": abs(mean - target),
                "rows": len(idx),
            }
        )
    return pd.DataFrame(rows).sort_values("product").reset_index(drop=True)


def _peak_product_residuals(df: pd.DataFrame, forward_prices: dict[str, float]) -> pd.DataFrame:
    peak_mask = _eex_peak_mask(df["ts"], country="CH")
    if not bool(peak_mask.any()) or not forward_prices:
        return pd.DataFrame()
    peak_df = df.loc[peak_mask].copy()
    residuals = _product_residuals(peak_df, forward_prices, load_type="PEAK")
    return residuals


def audit(csv_path: Path, forwards_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    df = _load_csv(csv_path)
    _, prices_by_load_type = _latest_eex_prices_by_load_type(forwards_path, market="CH")
    base_prices = prices_by_load_type.get("BASE", {})
    if not base_prices:
        raise ValueError("no EEX CH BASE prices in latest forward snapshot")
    base_residuals = _product_residuals(df, base_prices, load_type="BASE")
    peak_residuals = _peak_product_residuals(df, prices_by_load_type.get("PEAK", {}))
    residuals = pd.concat([base_residuals, peak_residuals], ignore_index=True)
    if not residuals.empty:
        residuals = residuals.sort_values(["load_type", "product"]).reset_index(drop=True)

    numeric_cols = [c for c in df.columns if c.endswith("_eur_mwh")]
    finite_ok = bool(np.isfinite(df[numeric_cols].to_numpy(dtype=float)).all())
    no_negative = bool((df[numeric_cols] >= 0.0).all().all())
    min_price = float(df[numeric_cols].min().min())
    weighted_negative_hours = int((df[PRICE] < 0.0).sum())
    p10_negative_hours = int((df["structural_p10_eur_mwh"] < 0.0).sum())
    negative_gate = _negative_gate(
        df,
        min_any_price=min_price,
        weighted_negative_hours=weighted_negative_hours,
    )
    bounded_negative_ok = bool(negative_gate["negative_gate_status"] == "PASS")
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
    max_eex_base_error = (
        float(residuals.loc[residuals["load_type"] == "BASE", "abs_error_eur_mwh"].max())
        if not residuals.empty and bool((residuals["load_type"] == "BASE").any())
        else float("inf")
    )
    max_eex_peak_error = (
        float(residuals.loc[residuals["load_type"] == "PEAK", "abs_error_eur_mwh"].max())
        if not residuals.empty and bool((residuals["load_type"] == "PEAK").any())
        else float("inf")
    )
    eex_peak_residual_count = (
        int((residuals["load_type"] == "PEAK").sum())
        if not residuals.empty and "load_type" in residuals
        else 0
    )
    width_mean = float(df["structural_width_eur_mwh"].mean())
    width_p95 = float(df["structural_width_eur_mwh"].quantile(0.95))
    duck_2030 = float(annual.loc[annual["year"] == 2030, "evening_minus_midday_eur_mwh"].iloc[0])
    weekend_2030 = float(annual.loc[annual["year"] == 2030, "weekend_minus_weekday_eur_mwh"].iloc[0])
    ramp_p99 = float(ramp.quantile(0.99))
    ramp_max = float(ramp.max())
    boundary_p95 = float(boundary.quantile(0.95)) if len(boundary) else 0.0

    score = 0.0
    score += 1.5 if finite_ok and bounded_negative_ok and quantile_order else 0.0
    score += 2.0 if max_eex_error <= 0.01 else 0.0
    score += 1.5 if 6.0 <= width_mean <= 11.0 and 18.0 <= width_p95 <= 32.0 else 0.75 if width_p95 >= 10.0 else 0.0
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
        "max_eex_base_error_eur_mwh": max_eex_base_error,
        "max_eex_peak_error_eur_mwh": max_eex_peak_error,
        "eex_peak_residual_count": float(eex_peak_residual_count),
        **negative_gate,
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

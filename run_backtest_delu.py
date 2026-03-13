#!/usr/bin/env python3
"""
Walk-forward backtest of the PFC 15min shaping model on DE-LU actual prices.

Methodology:
    - Training: all data before each test month (expanding window)
    - Test periods: Nov 2025, Dec 2025, Jan 2026, Feb 2026
    - For each test month:
        1. Fit ShapeHourly (CH data) for f_H, f_W
        2. Fit ShapeIntraday (DE-LU 15min) for f_Q
        3. Fit Uncertainty for p10/p90
        4. Assemble PFC using actual monthly mean as base price (to isolate shape skill)
        5. Compare predicted shape vs actual DE-LU 15min prices
    - KPIs: RMSE, MAE, bias, IC80% coverage, skill score vs naive (last-week repeat)

Author: FMV SA — Quant Desk
"""

import sys
import os
import logging
import warnings

warnings.filterwarnings("ignore")

# Ensure pfc_shaping is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.model.shape_hourly import ShapeHourly
from pfc_shaping.model.shape_intraday import ShapeIntraday
from pfc_shaping.model.uncertainty import Uncertainty
from pfc_shaping.model.assembler import PFCAssembler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest_delu")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = "pfc_shaping/data"
OUTPUT_DIR = "pfc_shaping/output"

# Test periods (each month predicted out-of-sample)
TEST_MONTHS = ["2025-11", "2025-12", "2026-01", "2026-02"]

# Minimum training lookback
MIN_TRAIN_MONTHS = 12


def load_data():
    """Load all parquet files."""
    logger.info("Loading data...")

    # CH hourly prices (for f_H, f_W calibration)
    ch = pd.read_parquet(f"{DATA_DIR}/epex_15min.parquet")
    logger.info("  CH data: %d rows, %s to %s", len(ch), ch.index.min().date(), ch.index.max().date())

    # DE-LU 15min actual prices (ground truth + f_Q calibration)
    de = pd.read_parquet(f"{DATA_DIR}/epex_de_15min.parquet")
    logger.info("  DE-LU data: %d rows, %s to %s", len(de), de.index.min().date(), de.index.max().date())

    # ENTSO-E fundamentals
    entso = pd.read_parquet(f"{DATA_DIR}/entso_15min.parquet")
    logger.info("  ENTSO-E data: %d rows", len(entso))

    return ch, de, entso


def build_naive_forecast(train_de: pd.DataFrame, test_idx: pd.DatetimeIndex) -> pd.Series:
    """
    Naive benchmark: repeat last week of training data cyclically.
    Maps each test timestamp to the same (weekday, hour, minute) from the last 7 days of training.
    """
    last_week = train_de.tail(96 * 7)  # last 7 days of 15min data
    if len(last_week) < 96 * 7:
        # Not enough data; use last available data repeated
        last_week = train_de.tail(96)

    # Build lookup: (weekday, hour, minute) -> price
    lw = last_week.copy()
    lw_zurich = lw.index.tz_convert("Europe/Zurich")
    lw["key"] = list(zip(lw_zurich.dayofweek, lw_zurich.hour, lw_zurich.minute))
    lookup = lw.groupby("key")["price_eur_mwh"].mean().to_dict()

    # Apply to test index
    test_zurich = test_idx.tz_convert("Europe/Zurich")
    keys = list(zip(test_zurich.dayofweek, test_zurich.hour, test_zurich.minute))
    naive_prices = pd.Series(
        [lookup.get(k, np.nan) for k in keys],
        index=test_idx,
        name="naive_price",
    )
    # Fill any missing with overall training mean
    naive_prices = naive_prices.fillna(train_de["price_eur_mwh"].mean())
    return naive_prices


def compute_kpis(
    pred: pd.Series,
    actual: pd.Series,
    naive: pd.Series,
    p10=None,
    p90=None,
) -> dict:
    """Compute all KPIs."""
    err = pred - actual
    err_naive = naive - actual

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    # Skill score vs naive (last week repeat)
    mse_model = float(np.mean(err ** 2))
    mse_naive = float(np.mean(err_naive ** 2))
    skill = float(1.0 - mse_model / max(mse_naive, 1e-10))

    # IC80% coverage
    ic80_cov = np.nan
    if p10 is not None and p90 is not None:
        valid = p10.notna() & p90.notna()
        if valid.sum() > 0:
            in_ic = (actual[valid] >= p10[valid]) & (actual[valid] <= p90[valid])
            ic80_cov = float(in_ic.mean())

    # Additional diagnostics
    rmse_naive = float(np.sqrt(mse_naive))
    corr = float(np.corrcoef(pred.values, actual.values)[0, 1]) if len(pred) > 2 else np.nan

    return {
        "RMSE_EUR": rmse,
        "MAE_EUR": mae,
        "Bias_EUR": bias,
        "IC80_coverage": ic80_cov,
        "Skill_vs_naive": skill,
        "RMSE_naive_EUR": rmse_naive,
        "Corr_pred_actual": corr,
        "N_obs": len(pred),
    }


def backtest_one_month(
    period: str,
    ch: pd.DataFrame,
    de: pd.DataFrame,
    entso: pd.DataFrame,
):
    """Run backtest for a single month."""
    period_start = pd.Timestamp(f"{period}-01", tz="UTC")
    period_end = (period_start + pd.offsets.MonthEnd(0)).replace(hour=23, minute=45)
    # Ensure period_end doesn't exceed available data
    period_end = min(period_end, de.index.max())

    if period_start > de.index.max():
        logger.warning("Period %s starts after data ends -- skipping", period)
        return None

    # Training data: everything before the test month
    train_ch = ch[ch.index < period_start]
    train_de = de[de.index < period_start]
    train_entso = entso[entso.index < period_start]

    # Test data: the month itself
    test_de = de[(de.index >= period_start) & (de.index <= period_end)]

    if len(train_de) < 96 * 30:
        logger.warning("Period %s: insufficient training data (%d) -- skipping", period, len(train_de))
        return None
    if len(test_de) < 96:
        logger.warning("Period %s: insufficient test data (%d) -- skipping", period, len(test_de))
        return None

    logger.info(
        "=== %s === Train CH: %d, Train DE: %d, Test DE: %d obs",
        period, len(train_ch), len(train_de), len(test_de),
    )

    # ── Step 1: Fit ShapeHourly on CH data (for f_H, f_W) ──────────────────
    cal_train_ch = enrich_15min_index(train_ch.index)
    sh = ShapeHourly()
    sh.fit(train_ch, cal_train_ch)
    logger.info("  ShapeHourly fitted: %d cells, f_W=%s",
                len(sh.factors_), {k: round(v, 3) for k, v in sh.f_W_.items()})

    # ── Step 2: Fit ShapeIntraday on DE-LU data (for f_Q) ──────────────────
    cal_train_de = enrich_15min_index(train_de.index)
    si = ShapeIntraday()
    si.fit(train_de, train_entso, cal_train_de)
    logger.info("  ShapeIntraday fitted: %d base cells, %d with exo correction",
                len(si.base_factors_), len(si.corrections_))

    # ── Step 3: Fit Uncertainty on DE-LU data ───────────────────────────────
    unc = Uncertainty(n_boot=200, seed=42)
    unc.fit(train_de, cal_train_de)

    # ── Step 4: Assemble PFC for test period ────────────────────────────────
    # Use actual monthly mean as base price to isolate shape quality
    actual_mean = test_de["price_eur_mwh"].mean()
    # Also compute from training as a realistic forward proxy
    train_mean = train_de.tail(96 * 30)["price_eur_mwh"].mean()  # last 30 days of training

    # Use training-based base price (realistic, no look-ahead)
    base_price = train_mean
    logger.info("  Base price (train last 30d mean): %.2f EUR/MWh  (actual test mean: %.2f)",
                base_price, actual_mean)

    assembler = PFCAssembler(sh, si, unc)
    year_key = period[:4]
    month_key = period  # e.g., '2025-11'

    test_start_str = period_start.strftime("%Y-%m-%d")
    n_days = (period_end - period_start).days + 1

    pfc = assembler.build(
        base_prices={month_key: base_price},
        start_date=test_start_str,
        horizon_days=n_days + 1,
    )

    # ── Step 5: Align and compare ───────────────────────────────────────────
    common_idx = pfc.index.intersection(test_de.index)
    if len(common_idx) == 0:
        logger.warning("  No common index between PFC and test data!")
        return None

    pred = pfc.loc[common_idx, "price_shape"]
    actual = test_de.loc[common_idx, "price_eur_mwh"]
    p10 = pfc.loc[common_idx, "p10"] if "p10" in pfc.columns else None
    p90 = pfc.loc[common_idx, "p90"] if "p90" in pfc.columns else None

    # Naive benchmark
    naive = build_naive_forecast(train_de, common_idx)

    # Rescale pred to match actual mean (since base price is imperfect)
    # This isolates shape quality from level prediction
    pred_rescaled = pred * (actual.mean() / pred.mean()) if pred.mean() != 0 else pred
    # Also rescale p10/p90
    scale = actual.mean() / pred.mean() if pred.mean() != 0 else 1.0
    if p10 is not None:
        p10_rescaled = p10 * scale
        p90_rescaled = p90 * scale
    else:
        p10_rescaled = p90_rescaled = None

    # KPIs on raw (level + shape)
    kpis_raw = compute_kpis(pred, actual, naive, p10, p90)

    # KPIs on rescaled (shape only, fair comparison)
    kpis_shape = compute_kpis(pred_rescaled, actual, naive, p10_rescaled, p90_rescaled)

    result = {"period": period}
    for k, v in kpis_raw.items():
        result[f"raw_{k}"] = v
    for k, v in kpis_shape.items():
        result[f"shape_{k}"] = v

    result["base_price_used"] = base_price
    result["actual_mean"] = actual_mean
    result["scale_factor"] = scale

    return result


def main():
    ch, de, entso = load_data()

    results = []
    for period in TEST_MONTHS:
        try:
            r = backtest_one_month(period, ch, de, entso)
            if r is not None:
                results.append(r)
        except Exception as e:
            logger.error("Period %s failed: %s", period, e, exc_info=True)

    if not results:
        logger.error("No results produced!")
        sys.exit(1)

    df = pd.DataFrame(results).set_index("period")

    # ── Print results ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  WALK-FORWARD BACKTEST -- PFC 15min Shaping Model (DE-LU)")
    print("=" * 80)

    print("\n--- RAW KPIs (level + shape, base price from training) ---")
    raw_cols = [c for c in df.columns if c.startswith("raw_")]
    print(df[raw_cols].to_string(float_format="{:.4f}".format))

    print("\n--- SHAPE-ONLY KPIs (rescaled to actual mean, isolates shape quality) ---")
    shape_cols = [c for c in df.columns if c.startswith("shape_")]
    print(df[shape_cols].to_string(float_format="{:.4f}".format))

    print("\n--- Level diagnostics ---")
    print(df[["base_price_used", "actual_mean", "scale_factor"]].to_string(
        float_format="{:.4f}".format))

    # Summary statistics across all periods
    print("\n" + "-" * 60)
    print("  SUMMARY (mean across test periods)")
    print("-" * 60)

    summary_keys = [
        ("Shape RMSE (EUR/MWh)", "shape_RMSE_EUR"),
        ("Shape MAE (EUR/MWh)", "shape_MAE_EUR"),
        ("Shape Bias (EUR/MWh)", "shape_Bias_EUR"),
        ("IC80% Coverage", "shape_IC80_coverage"),
        ("Skill vs Naive (last week)", "shape_Skill_vs_naive"),
        ("Correlation (pred vs actual)", "shape_Corr_pred_actual"),
        ("Raw RMSE (EUR/MWh)", "raw_RMSE_EUR"),
        ("Raw MAE (EUR/MWh)", "raw_MAE_EUR"),
        ("Naive RMSE (EUR/MWh)", "raw_RMSE_naive_EUR"),
    ]
    for label, col in summary_keys:
        if col in df.columns:
            val = df[col].mean()
            print(f"  {label:<40s} = {val:.4f}")

    # Target coverage check
    if "shape_IC80_coverage" in df.columns:
        avg_cov = df["shape_IC80_coverage"].mean()
        target = 0.80
        status = "OK" if abs(avg_cov - target) < 0.10 else "ATTENTION"
        print(f"\n  IC80% target: {target:.0%}, achieved: {avg_cov:.1%}  [{status}]")

    print("=" * 80)

    # ── Save to CSV ─────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "backtest_results.csv")
    df.to_csv(out_path)
    logger.info("Results saved to %s", out_path)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

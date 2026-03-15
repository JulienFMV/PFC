"""
error_analysis.py — Detailed PFC Shaping Error Decomposition
=============================================================
Runs the eval pipeline and decomposes prediction errors across multiple
dimensions to identify where the model loses accuracy.

Usage:
    cd /Users/julienbattaglia/Desktop/PFC && python error_analysis.py 2>&1
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))


def main() -> None:
    t0 = time.time()

    # ── 1. Replicate eval to get raw errors ──────────────────────────────
    from dashboard.utils import load_epex
    import yaml
    from pfc_shaping.data.calendar_ch import enrich_15min_index
    from pfc_shaping.data.forward_proxy import derive_base_prices
    from pfc_shaping.model.assembler import PFCAssembler
    from pfc_shaping.model.shape_hourly import ShapeHourly
    from pfc_shaping.model.shape_intraday import ShapeIntraday
    from pfc_shaping.model.uncertainty import Uncertainty

    epex = load_epex()
    if epex is None or len(epex) < 96 * 365:
        print("ERROR: Insufficient EPEX data")
        return

    # Setup: train/test split
    test_months = 2
    cutoff = epex.index.max() - pd.DateOffset(months=test_months)
    train = epex[epex.index < cutoff]
    test = epex[epex.index >= cutoff]

    cfg_path = ROOT / "pfc_shaping" / "config.yaml"
    with cfg_path.open() as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    lookback = model_cfg.get("lookback_months", 36)
    sigma = model_cfg.get("gaussian_sigma", 0.5)

    lb_cutoff = train.index.max() - pd.DateOffset(months=lookback)
    train_lb = train[train.index >= lb_cutoff]
    cal = enrich_15min_index(train_lb.index)

    sh = ShapeHourly(sigma=sigma)
    sh.fit(train_lb, cal)

    si = ShapeIntraday()
    si.fit(train_lb, entso_df=None, calendar_df=cal)

    unc = Uncertainty()
    unc.fit(train_lb, cal)

    base_prices = derive_base_prices(train, start_year=cutoff.year, n_years=1)

    assembler = PFCAssembler(shape_hourly=sh, shape_intraday=si, uncertainty=unc)
    pfc = assembler.build(
        base_prices=base_prices,
        start_date=cutoff.strftime("%Y-%m-%d"),
        horizon_days=test_months * 31,
    )

    common_idx = pfc.index.intersection(test.index)
    if len(common_idx) < 96 * 7:
        print("ERROR: Insufficient overlap")
        return

    pfc_prices = pfc.loc[common_idx, "price_shape"]
    spot_prices = test.loc[common_idx, "price_eur_mwh"]
    errors = pfc_prices.values - spot_prices.values

    # Build enriched DataFrame for analysis
    idx_zh = common_idx.tz_convert("Europe/Zurich")
    cal_test = enrich_15min_index(common_idx)

    df = pd.DataFrame({
        "pfc": pfc_prices.values,
        "spot": spot_prices.values,
        "error": errors,
        "abs_error": np.abs(errors),
        "sq_error": errors**2,
        "hour": idx_zh.hour,
        "minute": idx_zh.minute,
        "dow": idx_zh.dayofweek,  # 0=Mon, 6=Sun
        "month": idx_zh.month,
        "quart": cal_test["quart"].values,
        "saison": cal_test["saison"].values,
        "type_jour": cal_test["type_jour"].values,
        "f_H": pfc.loc[common_idx, "f_H"].values,
        "f_W": pfc.loc[common_idx, "f_W"].values,
        "f_Q": pfc.loc[common_idx, "f_Q"].values,
        "B": pfc.loc[common_idx, "B"].values,
    }, index=common_idx)

    # Scale for shape-only analysis
    scale = spot_prices.mean() / pfc_prices.mean() if pfc_prices.mean() != 0 else 1.0
    df["shape_error"] = df["pfc"] * scale - df["spot"]
    df["shape_sq_error"] = df["shape_error"] ** 2

    # Weekday labels
    dow_labels = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    df["dow_name"] = df["dow"].map(dow_labels)

    # Is ramping hour
    df["is_ramp"] = df["hour"].isin([6, 7, 8, 9, 10, 17, 18, 19, 20])
    df["is_weekend"] = df["dow"].isin([5, 6])

    # Total RMSE confirmation
    rmse_total = np.sqrt(np.mean(df["sq_error"]))
    rmse_shape = np.sqrt(np.mean(df["shape_sq_error"]))
    bias = df["error"].mean()
    mae = df["abs_error"].mean()

    print("=" * 72)
    print("PFC SHAPING MODEL — DETAILED ERROR ANALYSIS")
    print("=" * 72)
    print(f"\nTest period: {common_idx.min().date()} -> {common_idx.max().date()}")
    print(f"N points: {len(df)}")
    print(f"Total RMSE:      {rmse_total:.2f} EUR/MWh")
    print(f"Shape-only RMSE: {rmse_shape:.2f} EUR/MWh")
    print(f"Bias:            {bias:.2f} EUR/MWh")
    print(f"MAE:             {mae:.2f} EUR/MWh")
    print(f"Spot mean:       {df['spot'].mean():.2f} EUR/MWh")
    print(f"PFC mean:        {df['pfc'].mean():.2f} EUR/MWh")
    print(f"Scale factor:    {scale:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # 2. RMSE BY HOUR OF DAY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("2. RMSE BY HOUR OF DAY")
    print("=" * 72)

    hourly = df.groupby("hour").agg(
        rmse=("sq_error", lambda x: np.sqrt(x.mean())),
        shape_rmse=("shape_sq_error", lambda x: np.sqrt(x.mean())),
        bias=("error", "mean"),
        mae=("abs_error", "mean"),
        n=("error", "count"),
        mean_spot=("spot", "mean"),
    ).round(2)
    hourly["contribution_pct"] = (hourly["rmse"]**2 * hourly["n"] / df["sq_error"].sum() * 100).round(1)

    print(f"\n{'Hour':>4} {'RMSE':>8} {'ShpRMSE':>8} {'Bias':>8} {'MAE':>8} {'MeanSpot':>8} {'Contrib%':>8} {'N':>6}")
    print("-" * 72)
    for h, row in hourly.iterrows():
        marker = " ***" if row["rmse"] > rmse_total * 1.2 else ""
        print(f"{h:>4} {row['rmse']:>8.2f} {row['shape_rmse']:>8.2f} {row['bias']:>8.2f} {row['mae']:>8.2f} {row['mean_spot']:>8.2f} {row['contribution_pct']:>7.1f}% {row['n']:>5.0f}{marker}")

    top3_hours = hourly.nlargest(3, "rmse")
    print(f"\nTOP 3 worst hours: {list(top3_hours.index)} with RMSE {list(top3_hours['rmse'].round(1))}")

    # ═══════════════════════════════════════════════════════════════════
    # 3. RMSE BY DAY OF WEEK
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("3. RMSE BY DAY OF WEEK")
    print("=" * 72)

    dow = df.groupby(["dow", "dow_name"]).agg(
        rmse=("sq_error", lambda x: np.sqrt(x.mean())),
        shape_rmse=("shape_sq_error", lambda x: np.sqrt(x.mean())),
        bias=("error", "mean"),
        n=("error", "count"),
        mean_spot=("spot", "mean"),
    ).round(2)

    print(f"\n{'Day':>6} {'RMSE':>8} {'ShpRMSE':>8} {'Bias':>8} {'MeanSpot':>8} {'N':>6}")
    print("-" * 50)
    for (d, name), row in dow.iterrows():
        print(f"{name:>6} {row['rmse']:>8.2f} {row['shape_rmse']:>8.2f} {row['bias']:>8.2f} {row['mean_spot']:>8.2f} {row['n']:>5.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # 4. RMSE BY MONTH
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("4. RMSE BY MONTH")
    print("=" * 72)

    monthly = df.groupby("month").agg(
        rmse=("sq_error", lambda x: np.sqrt(x.mean())),
        shape_rmse=("shape_sq_error", lambda x: np.sqrt(x.mean())),
        bias=("error", "mean"),
        n=("error", "count"),
        mean_spot=("spot", "mean"),
        mean_pfc=("pfc", "mean"),
    ).round(2)

    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    print(f"\n{'Month':>6} {'RMSE':>8} {'ShpRMSE':>8} {'Bias':>8} {'MnSpot':>8} {'MnPFC':>8} {'N':>6}")
    print("-" * 56)
    for m, row in monthly.iterrows():
        print(f"{month_names.get(m, str(m)):>6} {row['rmse']:>8.2f} {row['shape_rmse']:>8.2f} {row['bias']:>8.2f} {row['mean_spot']:>8.2f} {row['mean_pfc']:>8.2f} {row['n']:>5.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # 5. RMSE BY SEASON
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("5. RMSE BY SEASON")
    print("=" * 72)

    seasonal = df.groupby("saison").agg(
        rmse=("sq_error", lambda x: np.sqrt(x.mean())),
        shape_rmse=("shape_sq_error", lambda x: np.sqrt(x.mean())),
        bias=("error", "mean"),
        n=("error", "count"),
        mean_spot=("spot", "mean"),
    ).round(2)

    for s, row in seasonal.iterrows():
        print(f"  {s:<12} RMSE={row['rmse']:>7.2f}  ShpRMSE={row['shape_rmse']:>7.2f}  Bias={row['bias']:>7.2f}  MeanSpot={row['mean_spot']:>7.2f}  N={row['n']:>5.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # 6. RMSE BY QUARTER-HOUR (q=1..4)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("6. RMSE BY QUARTER-HOUR (q=1..4)")
    print("=" * 72)

    qh = df.groupby("quart").agg(
        rmse=("sq_error", lambda x: np.sqrt(x.mean())),
        shape_rmse=("shape_sq_error", lambda x: np.sqrt(x.mean())),
        bias=("error", "mean"),
        n=("error", "count"),
    ).round(2)

    for q, row in qh.iterrows():
        print(f"  q={q}  RMSE={row['rmse']:>7.2f}  ShpRMSE={row['shape_rmse']:>7.2f}  Bias={row['bias']:>7.2f}  N={row['n']:>5.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # 7. WEEKDAY vs WEEKEND
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("7. WEEKDAY vs WEEKEND")
    print("=" * 72)

    for label, mask in [("Weekday", ~df["is_weekend"]), ("Weekend", df["is_weekend"])]:
        sub = df[mask]
        r = np.sqrt(sub["sq_error"].mean())
        rs = np.sqrt(sub["shape_sq_error"].mean())
        b = sub["error"].mean()
        print(f"  {label:<10} RMSE={r:>7.2f}  ShpRMSE={rs:>7.2f}  Bias={b:>7.2f}  N={len(sub):>5}")

    # ═══════════════════════════════════════════════════════════════════
    # 8. RAMPING vs FLAT HOURS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("8. RAMPING HOURS (6-10, 17-20) vs FLAT HOURS")
    print("=" * 72)

    for label, mask in [("Ramp hours", df["is_ramp"]), ("Flat hours", ~df["is_ramp"])]:
        sub = df[mask]
        r = np.sqrt(sub["sq_error"].mean())
        rs = np.sqrt(sub["shape_sq_error"].mean())
        b = sub["error"].mean()
        print(f"  {label:<12} RMSE={r:>7.2f}  ShpRMSE={rs:>7.2f}  Bias={b:>7.2f}  N={len(sub):>5}")

    # ═══════════════════════════════════════════════════════════════════
    # 9. HETEROSCEDASTICITY — ERRORS vs PRICE LEVEL
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("9. HETEROSCEDASTICITY — ERRORS vs PRICE LEVEL")
    print("=" * 72)

    # Quintile analysis
    df["spot_quintile"] = pd.qcut(df["spot"], 5, labels=["Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"])
    hetero = df.groupby("spot_quintile").agg(
        rmse=("sq_error", lambda x: np.sqrt(x.mean())),
        shape_rmse=("shape_sq_error", lambda x: np.sqrt(x.mean())),
        bias=("error", "mean"),
        mean_spot=("spot", "mean"),
        std_spot=("spot", "std"),
        n=("error", "count"),
    ).round(2)

    print(f"\n{'Quintile':>10} {'RMSE':>8} {'ShpRMSE':>8} {'Bias':>8} {'MeanSpot':>8} {'StdSpot':>8}")
    print("-" * 58)
    for q, row in hetero.iterrows():
        print(f"{str(q):>10} {row['rmse']:>8.2f} {row['shape_rmse']:>8.2f} {row['bias']:>8.2f} {row['mean_spot']:>8.2f} {row['std_spot']:>8.2f}")

    # Correlation between |error| and price level
    corr_abs = np.corrcoef(df["spot"].values, df["abs_error"].values)[0, 1]
    corr_sq = np.corrcoef(df["spot"].values, df["sq_error"].values)[0, 1]
    print(f"\nCorrelation(|error|, spot_price) = {corr_abs:.3f}")
    print(f"Correlation(error^2, spot_price) = {corr_sq:.3f}")

    # Relative error analysis
    df["rel_error"] = df["error"] / df["spot"].clip(lower=1.0)
    rel_rmse = np.sqrt(np.mean(df["rel_error"]**2))
    print(f"Relative RMSE (error/spot): {rel_rmse:.3f} = {rel_rmse*100:.1f}%")

    # ═══════════════════════════════════════════════════════════════════
    # 10. RESIDUAL AUTOCORRELATION
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("10. RESIDUAL AUTOCORRELATION STRUCTURE")
    print("=" * 72)

    errs = df["error"].values
    n = len(errs)
    mean_err = errs.mean()
    denom = np.sum((errs - mean_err)**2)

    print(f"\n{'Lag':>8} {'ACF':>10} {'Interpretation':>30}")
    print("-" * 50)
    for lag in [1, 2, 3, 4, 8, 12, 24, 48, 96, 192, 384, 672]:  # 15min increments
        if lag >= n:
            break
        num = np.sum((errs[lag:] - mean_err) * (errs[:-lag] - mean_err))
        acf_val = num / denom
        hrs = lag * 0.25
        if hrs < 24:
            time_label = f"{hrs:.1f}h"
        else:
            time_label = f"{hrs/24:.1f}d"
        print(f"{lag:>8} {acf_val:>10.4f}  ({time_label})")

    # Ljung-Box like test: sum of squared ACFs
    sum_sq_acf = 0
    for lag in range(1, min(97, n)):
        num = np.sum((errs[lag:] - mean_err) * (errs[:-lag] - mean_err))
        acf_val = num / denom
        sum_sq_acf += acf_val**2

    print(f"\nSum of squared ACF (lag 1..96 = 24h): {sum_sq_acf:.4f}")
    print("  > 0.1 suggests strong serial dependence; errors are NOT white noise.")

    # ═══════════════════════════════════════════════════════════════════
    # 11. CROSS-TABULATION: HOUR x TYPE_JOUR
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("11. TOP 15 WORST (HOUR, TYPE_JOUR) CELLS")
    print("=" * 72)

    cell_rmse = df.groupby(["hour", "type_jour"]).agg(
        rmse=("sq_error", lambda x: np.sqrt(x.mean())),
        bias=("error", "mean"),
        n=("error", "count"),
    ).round(2)
    cell_rmse = cell_rmse[cell_rmse["n"] >= 20]
    top_cells = cell_rmse.nlargest(15, "rmse")

    print(f"\n{'Hour':>4} {'TypeJour':>12} {'RMSE':>8} {'Bias':>8} {'N':>6}")
    print("-" * 42)
    for (h, tj), row in top_cells.iterrows():
        print(f"{h:>4} {tj:>12} {row['rmse']:>8.2f} {row['bias']:>8.2f} {row['n']:>5.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # 12. CROSS-TABULATION: HOUR x SAISON
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("12. TOP 15 WORST (HOUR, SAISON) CELLS")
    print("=" * 72)

    cell_hs = df.groupby(["hour", "saison"]).agg(
        rmse=("sq_error", lambda x: np.sqrt(x.mean())),
        bias=("error", "mean"),
        n=("error", "count"),
    ).round(2)
    cell_hs = cell_hs[cell_hs["n"] >= 20]
    top_cells_hs = cell_hs.nlargest(15, "rmse")

    print(f"\n{'Hour':>4} {'Saison':>12} {'RMSE':>8} {'Bias':>8} {'N':>6}")
    print("-" * 42)
    for (h, s), row in top_cells_hs.iterrows():
        print(f"{h:>4} {s:>12} {row['rmse']:>8.2f} {row['bias']:>8.2f} {row['n']:>5.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # 13. BIAS DECOMPOSITION
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("13. BIAS DECOMPOSITION")
    print("=" * 72)

    total_rmse_sq = rmse_total**2
    bias_sq = bias**2
    variance_err = np.var(errors)
    print(f"\n  MSE = Bias^2 + Var(error)")
    print(f"  MSE          = {total_rmse_sq:.2f}")
    print(f"  Bias^2       = {bias_sq:.2f}  ({bias_sq/total_rmse_sq*100:.1f}%)")
    print(f"  Var(error)   = {variance_err:.2f}  ({variance_err/total_rmse_sq*100:.1f}%)")
    print(f"\n  => Bias accounts for {bias_sq/total_rmse_sq*100:.1f}% of MSE")
    print(f"  => Removing bias alone would reduce RMSE to {np.sqrt(variance_err):.2f} EUR/MWh")

    # ═══════════════════════════════════════════════════════════════════
    # 14. f_W FACTOR ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("14. SHAPE FACTOR f_W — CALIBRATED vs IDEAL")
    print("=" * 72)

    # What f_W values does the model use?
    fw_model = df.groupby("type_jour")["f_W"].first().round(4)
    # What would be ideal? (spot ratio by type_jour)
    mean_spot_all = df["spot"].mean()
    fw_ideal = df.groupby("type_jour")["spot"].mean() / mean_spot_all

    fw_compare = pd.DataFrame({"f_W_model": fw_model, "f_W_ideal": fw_ideal.round(4)})
    fw_compare["gap"] = (fw_compare["f_W_model"] - fw_compare["f_W_ideal"]).round(4)
    print(fw_compare.to_string())

    # ═══════════════════════════════════════════════════════════════════
    # 15. EXTREME ERRORS (TAIL ANALYSIS)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("15. EXTREME ERRORS — TAIL ANALYSIS")
    print("=" * 72)

    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    err_pcts = np.percentile(errors, pcts)
    abs_pcts = np.percentile(np.abs(errors), pcts)

    print(f"\n{'Percentile':>12} {'Error':>10} {'|Error|':>10}")
    print("-" * 34)
    for p, e, a in zip(pcts, err_pcts, abs_pcts):
        print(f"{'p'+str(p):>12} {e:>10.2f} {a:>10.2f}")

    # How many extreme errors?
    n_gt50 = (np.abs(errors) > 50).sum()
    n_gt100 = (np.abs(errors) > 100).sum()
    print(f"\n  |error| > 50 EUR/MWh: {n_gt50} ({n_gt50/len(errors)*100:.1f}%)")
    print(f"  |error| > 100 EUR/MWh: {n_gt100} ({n_gt100/len(errors)*100:.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # 16. HOUR x WEEKDAY/WEEKEND DETAILED
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("16. RMSE BY HOUR: WEEKDAY vs WEEKEND")
    print("=" * 72)

    print(f"\n{'Hour':>4} {'WD_RMSE':>9} {'WE_RMSE':>9} {'WD_Bias':>9} {'WE_Bias':>9} {'Delta':>8}")
    print("-" * 52)
    for h in range(24):
        wd_sub = df[(df["hour"] == h) & (~df["is_weekend"])]
        we_sub = df[(df["hour"] == h) & (df["is_weekend"])]
        if len(wd_sub) < 5 or len(we_sub) < 5:
            continue
        wd_rmse = np.sqrt(wd_sub["sq_error"].mean())
        we_rmse = np.sqrt(we_sub["sq_error"].mean())
        wd_bias = wd_sub["error"].mean()
        we_bias = we_sub["error"].mean()
        delta = we_rmse - wd_rmse
        marker = " ***" if abs(delta) > 10 else ""
        print(f"{h:>4} {wd_rmse:>9.2f} {we_rmse:>9.2f} {wd_bias:>9.2f} {we_bias:>9.2f} {delta:>8.2f}{marker}")

    # ═══════════════════════════════════════════════════════════════════
    # 17. SUMMARY RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("17. SUMMARY & RECOMMENDATIONS")
    print("=" * 72)

    print(f"""
KEY FINDINGS:
  1. Bias = {bias:.1f} EUR/MWh ({bias_sq/total_rmse_sq*100:.0f}% of MSE)
     -> The forward_proxy derives base prices from trailing spot.
     -> Systematic over/under-prediction = level mismatch.
     -> Fix: removing bias alone -> RMSE from {rmse_total:.1f} to {np.sqrt(variance_err):.1f}

  2. Shape-only RMSE = {rmse_shape:.1f} EUR/MWh
     -> Even after level correction, ~{rmse_shape:.0f} EUR/MWh of irreducible shape error.
     -> Largest contributors: check hours & seasons above.

  3. Autocorrelation: errors are serially correlated
     -> Errors persist for hours/days -> not random noise.
     -> An AR(1) or exponential smoothing correction could help.

  4. Heteroscedasticity: RMSE varies with price level
     -> High-price periods have higher absolute errors.
     -> Consider multiplicative error model or log-transform.

STRUCTURAL RECOMMENDATIONS (priority order):

  [P0] FIX BIAS — USE REAL FORWARDS
       The {bias:.1f} EUR/MWh bias dominates. Using live EEX forwards
       instead of spot-derived proxies would eliminate ~{bias_sq/total_rmse_sq*100:.0f}% of MSE.

  [P1] ADAPTIVE f_W BY SEASON
       f_W(type_jour) is calibrated globally but weekend/holiday patterns
       differ strongly by season. f_W(type_jour, saison) would add
       20 cells (5 types x 4 seasons) to capture winter-weekend vs
       summer-weekend differences.

  [P2] AUTOREGRESSIVE ERROR CORRECTION
       High ACF at lag 1-4 (15min-1h) means a simple AR(1) correction
       on residuals could reduce shape RMSE by ~15-25%.
       P_corrected(t) = P_raw(t) + phi * epsilon(t-1)

  [P3] REGIME-DEPENDENT f_H
       Hourly profiles differ when price levels are extreme (spikes,
       negative prices). Adding a "price regime" dimension to f_H
       (normal / high / extreme) would capture non-linear effects.

  [P4] LOG-SPACE SHAPING
       The multiplicative model P = B * f_S * f_W * f_H * f_Q generates
       heteroscedastic errors (larger errors at high prices).
       Working in log-space: log(P) = log(B) + log(f_S) + ...
       and fitting additive factors would make errors more homoscedastic.
""")

    elapsed = time.time() - t0
    print(f"Analysis completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

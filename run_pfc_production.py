#!/usr/bin/env python3
"""
PFC 15min Production Run
========================
Full pipeline: load data, fit models, assemble N+3 PFC, save output.
Includes ENTSO-E climatology forecast for Layer 2 Ridge corrections.
"""

import sys
import os
import logging
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure project root on path
PROJECT_ROOT = "/Users/julienbattaglia/Desktop/PFC"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PFC_PROD")

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD ALL DATA
# ═══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("STEP 1: Loading data")
logger.info("=" * 70)

t0 = time.time()

DATA = "pfc_shaping/data"

epex_ch = pd.read_parquet(f"{DATA}/epex_15min.parquet")
epex_de = pd.read_parquet(f"{DATA}/epex_de_15min.parquet")
entso   = pd.read_parquet(f"{DATA}/entso_15min.parquet")
hydro   = pd.read_parquet(f"{DATA}/hydro_reservoir.parquet")

logger.info("  EPEX CH:  %d rows  [%s -> %s]", len(epex_ch), epex_ch.index.min().date(), epex_ch.index.max().date())
logger.info("  EPEX DE:  %d rows  [%s -> %s]", len(epex_de), epex_de.index.min().date(), epex_de.index.max().date())
logger.info("  ENTSO-E:  %d rows  [%s -> %s]", len(entso), entso.index.min().date(), entso.index.max().date())
logger.info("  Hydro:    %d rows  [%s -> %s]", len(hydro), hydro.index.min().date(), hydro.index.max().date())
logger.info("  Data loaded in %.1fs", time.time() - t0)

# ═══════════════════════════════════════════════════════════════════════
# 2. CALENDAR ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("STEP 2: Calendar enrichment")
logger.info("=" * 70)

t1 = time.time()
from pfc_shaping.data.calendar_ch import enrich_15min_index

cal_ch = enrich_15min_index(epex_ch.index)
cal_de = enrich_15min_index(epex_de.index)

logger.info("  CH calendar: %d rows, types: %s", len(cal_ch), dict(cal_ch["type_jour"].value_counts()))
logger.info("  DE calendar: %d rows", len(cal_de))
logger.info("  Calendar enriched in %.1fs", time.time() - t1)

# ═══════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("STEP 3: Feature engineering (solar_regime, load_deviation)")
logger.info("=" * 70)

# Features already in entso_15min.parquet
logger.info("  solar_regime stats: mean=%.2f, std=%.2f",
            entso["solar_regime"].mean(), entso["solar_regime"].std())
logger.info("  load_deviation stats: mean=%.2f, std=%.2f",
            entso["load_deviation"].mean(), entso["load_deviation"].std())

# ═══════════════════════════════════════════════════════════════════════
# 4. FIT ShapeHourly on CH data (full history)
# ═══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("STEP 4: Fitting ShapeHourly on CH EPEX (full history)")
logger.info("=" * 70)

t2 = time.time()
from pfc_shaping.model.shape_hourly import ShapeHourly

sh = ShapeHourly()
sh.fit(epex_ch, cal_ch)

logger.info("  Fitted %d (saison, type_jour) cells", len(sh.factors_))
logger.info("  f_W ratios: %s", {k: round(v, 4) for k, v in sh.f_W_.items()})
logger.info("  Sample Hiver/Ouvrable peak h=12: f_H=%.4f", sh.get("Hiver", "Ouvrable")[12])
logger.info("  ShapeHourly fitted in %.1fs", time.time() - t2)

# ═══════════════════════════════════════════════════════════════════════
# 5. FIT ShapeIntraday on DE-LU post Oct 2025 (real 15min auction)
# ═══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("STEP 5: Fitting ShapeIntraday on DE-LU post Oct 2025")
logger.info("=" * 70)

t3 = time.time()
from pfc_shaping.model.shape_intraday import ShapeIntraday

cutoff_de = pd.Timestamp("2025-10-01", tz="UTC")
epex_de_post = epex_de[epex_de.index >= cutoff_de]
cal_de_post = cal_de.loc[epex_de_post.index]
entso_de_post = entso.reindex(epex_de_post.index)

logger.info("  DE-LU post Oct 2025: %d rows", len(epex_de_post))

si = ShapeIntraday()
si.fit(epex_de_post, entso_de_post, cal_de_post)

logger.info("  Fitted %d (saison, type_jour, heure) cells", len(si.base_factors_))
logger.info("  Corrections (layer 2): %d cells", len(si.corrections_))
logger.info("  ShapeIntraday fitted in %.1fs", time.time() - t3)

# ═══════════════════════════════════════════════════════════════════════
# 6. FIT WaterValue correction
# ═══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("STEP 6: Fitting WaterValue correction")
logger.info("=" * 70)

t4 = time.time()
from pfc_shaping.model.water_value import WaterValueCorrection

wv = WaterValueCorrection()
wv.fit(epex_ch, hydro, cal_ch)

logger.info("  beta_WV = %.4f", wv.beta_wv_)
logger.info("  Season sensitivities: %s", {k: f"{v:.3f}" for k, v in wv.season_sensitivity_.items()})
logger.info("  Calibration obs: %d", wv.n_obs_)
logger.info("  WaterValue fitted in %.1fs", time.time() - t4)

# ═══════════════════════════════════════════════════════════════════════
# 7. FIT Uncertainty (bootstrap n_boot=500)
# ═══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("STEP 7: Fitting Uncertainty (n_boot=500, production quality)")
logger.info("=" * 70)

t5 = time.time()
from pfc_shaping.model.uncertainty import Uncertainty

unc = Uncertainty(n_boot=500, seed=42)
# Use DE-LU for 15min uncertainty (true 15min granularity)
unc.fit(epex_de_post, cal_de_post)

logger.info("  Bootstrap cells: %d", len(unc.boot_stats_))
logger.info("  Uncertainty fitted in %.1fs", time.time() - t5)

# ═══════════════════════════════════════════════════════════════════════
# 8. BUILD BASE_PRICES dict
# ═══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("STEP 8: Building base_prices (EEX forward levels)")
logger.info("=" * 70)

base_prices = {
    # Calendar years
    "2026": 72.0,
    "2027": 68.0,
    "2028": 65.0,
    "2029": 63.0,
    # 2026 Quarters
    "2026-Q1": 78.0,
    "2026-Q2": 65.0,
    "2026-Q3": 55.0,
    "2026-Q4": 80.0,
    # Monthly M+1..M+6 (Mar-Sep 2026)
    "2026-03": 75.0,
    "2026-04": 62.0,
    "2026-05": 58.0,
    "2026-06": 52.0,
    "2026-07": 48.0,
    "2026-08": 50.0,
    "2026-09": 60.0,
}

# Use ContractCascader to fill in missing months
from pfc_shaping.calibration.cascading import ContractCascader

cascader = ContractCascader()
cascader.fit_seasonal_ratios(epex_ch)

cascaded_prices = cascader.cascade(base_prices)

logger.info("  Input keys: %d", len(base_prices))
logger.info("  Cascaded keys: %d", len(cascaded_prices))
for k in sorted(cascaded_prices.keys()):
    logger.info("    %s: %.2f EUR/MWh", k, cascaded_prices[k])

# ═══════════════════════════════════════════════════════════════════════
# 9. ASSEMBLE PFC N+3 (1095 days from 2026-03-14)
# ═══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("STEP 9: Assembling PFC N+3 (2026-03-14, 1095 days)")
logger.info("=" * 70)

t6 = time.time()

# Try ArbitrageFreeCalibrator
calibrator = None
try:
    from pfc_shaping.calibration.arbitrage_free import ArbitrageFreeCalibrator
    calibrator = ArbitrageFreeCalibrator(smoothness_weight=1.0, tol=0.01)
    logger.info("  ArbitrageFreeCalibrator loaded OK")
except Exception as e:
    logger.warning("  ArbitrageFreeCalibrator unavailable: %s", e)

from pfc_shaping.model.assembler import PFCAssembler

# Build hydro forecast for forward horizon (use latest known fill_deviation, decay to 0)
latest_fill_dev = hydro["fill_deviation"].iloc[-1]
logger.info("  Latest hydro fill_deviation: %.3f (as of %s)", latest_fill_dev, hydro.index[-1].date())

# Create simple hydro forecast: latest value decaying linearly to 0 over 12 months
start_date = (pd.Timestamp.utcnow() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
horizon_days = 1095
future_start = pd.Timestamp(start_date, tz="UTC")
future_end = future_start + pd.Timedelta(days=horizon_days)
hydro_idx = pd.date_range(future_start, future_end, freq="W-MON", tz="UTC")
decay = np.linspace(latest_fill_dev, 0.0, len(hydro_idx))
hydro_forecast = pd.DataFrame({"fill_deviation": decay}, index=hydro_idx)

# ── Build ENTSO-E climatology forecast (solar_regime + load_deviation) ──
# Use historical seasonal patterns from entso_15min.parquet to project
# solar_regime and load_deviation for the N+3 horizon.
# This activates Layer 2 Ridge corrections at runtime.
logger.info("  Building ENTSO-E climatology forecast for N+3 horizon...")

future_idx = pd.date_range(future_start, future_end, freq="15min", inclusive="left", tz="UTC")
future_zurich = future_idx.tz_convert("Europe/Zurich")

# Compute historical climatology: median solar_regime and load_deviation
# per (month, hour, quarter-of-hour) from 3 years of history
entso_zurich = entso.copy()
entso_zurich["month"] = entso.index.tz_convert("Europe/Zurich").month
entso_zurich["hour"] = entso.index.tz_convert("Europe/Zurich").hour
entso_zurich["qh"] = (entso.index.minute // 15) + 1

clim = entso_zurich.groupby(["month", "hour", "qh"]).agg(
    solar_regime_median=("solar_regime", "median"),
    load_deviation_median=("load_deviation", "median"),
).reset_index()

# Map climatology onto the future index
future_keys = pd.DataFrame({
    "month": future_zurich.month,
    "hour": future_zurich.hour,
    "qh": (future_zurich.minute // 15) + 1,
}, index=future_idx)

entso_forecast = future_keys.merge(
    clim, on=["month", "hour", "qh"], how="left"
).set_index(future_idx)

entso_forecast = entso_forecast.rename(columns={
    "solar_regime_median": "solar_regime",
    "load_deviation_median": "load_deviation",
})[["solar_regime", "load_deviation"]]

# Fill any NaN with neutral values
entso_forecast["solar_regime"] = entso_forecast["solar_regime"].fillna(1.0)
entso_forecast["load_deviation"] = entso_forecast["load_deviation"].fillna(0.0)

logger.info("  ENTSO-E climatology forecast: %d rows", len(entso_forecast))
logger.info("  solar_regime: mean=%.2f  std=%.2f",
            entso_forecast["solar_regime"].mean(), entso_forecast["solar_regime"].std())
logger.info("  load_deviation: mean=%.3f  std=%.3f",
            entso_forecast["load_deviation"].mean(), entso_forecast["load_deviation"].std())

assembler = PFCAssembler(
    shape_hourly=sh,
    shape_intraday=si,
    uncertainty=unc,
    water_value=wv,
    cascader=cascader,
    calibrator=calibrator,
)

pfc = assembler.build(
    base_prices=cascaded_prices,
    start_date=start_date,
    horizon_days=horizon_days,
    entso_forecast=entso_forecast,
    hydro_forecast=hydro_forecast,
)

logger.info("  PFC assembled: %d rows in %.1fs", len(pfc), time.time() - t6)

# ═══════════════════════════════════════════════════════════════════════
# 10. SAVE OUTPUT
# ═══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("STEP 10: Saving output")
logger.info("=" * 70)

out_dir = "pfc_shaping/output"
today = pd.Timestamp.now().strftime("%Y-%m-%d")
out_base = f"{out_dir}/pfc_15min_{today}"

pfc.to_parquet(f"{out_base}.parquet")
logger.info("  Saved: %s.parquet (%d rows)", out_base, len(pfc))

pfc.to_csv(f"{out_base}.csv")
logger.info("  Saved: %s.csv", out_base)

# Save model artifacts
artifacts_dir = "pfc_shaping/model/artifacts"
os.makedirs(artifacts_dir, exist_ok=True)
sh.save(f"{artifacts_dir}/shape_hourly.parquet")
si.save(f"{artifacts_dir}/shape_intraday.parquet")
wv.save(f"{artifacts_dir}/water_value.parquet")
unc.save(f"{artifacts_dir}/uncertainty.parquet")

# ═══════════════════════════════════════════════════════════════════════
# 11. COMPREHENSIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("STEP 11: Comprehensive Statistics")
logger.info("=" * 70)

print("\n" + "=" * 80)
print("  PFC 15min PRODUCTION RUN — 2026-03-14")
print("  Horizon: %s -> %s (%d days)" % (pfc.index.min().date(), pfc.index.max().date(), horizon_days))
print("  Timestamps: %d (15min intervals)" % len(pfc))
print("=" * 80)

# --- Price distribution ---
print("\n--- PRICE DISTRIBUTION (EUR/MWh) ---")
print("  Mean:    %.2f" % pfc["price_shape"].mean())
print("  Median:  %.2f" % pfc["price_shape"].median())
print("  Std:     %.2f" % pfc["price_shape"].std())
print("  Min:     %.2f" % pfc["price_shape"].min())
print("  Max:     %.2f" % pfc["price_shape"].max())
print("  p5:      %.2f" % pfc["price_shape"].quantile(0.05))
print("  p95:     %.2f" % pfc["price_shape"].quantile(0.95))

# --- Factor ranges ---
print("\n--- FACTOR RANGES ---")
for col in ["f_S", "f_W", "f_H", "f_Q", "f_WV"]:
    s = pfc[col]
    print("  %-5s: mean=%.4f  min=%.4f  max=%.4f  std=%.4f" % (col, s.mean(), s.min(), s.max(), s.std()))

# --- f_Q detail ---
print("\n--- f_Q INTRADAY DETAIL ---")
print("  f_Q range: [%.4f, %.4f]" % (pfc["f_Q"].min(), pfc["f_Q"].max()))
print("  f_Q mean:  %.4f (should be ~1.0)" % pfc["f_Q"].mean())
pfc_zurich = pfc.copy()
pfc_zurich["hour"] = pfc.index.tz_convert("Europe/Zurich").hour
for h in [0, 6, 8, 12, 17, 20, 23]:
    mask = pfc_zurich["hour"] == h
    print("    h=%02d: f_Q mean=%.4f  std=%.4f" % (h, pfc_zurich.loc[mask, "f_Q"].mean(), pfc_zurich.loc[mask, "f_Q"].std()))

# --- IC width by horizon ---
print("\n--- CONFIDENCE INTERVAL WIDTH BY HORIZON ---")
if pfc["p10"].notna().any():
    pfc_zurich["ic_width"] = pfc["p90"] - pfc["p10"]
    for pt in ["M+1..M+6", "M+7..M+12", "Y+2/Y+3"]:
        mask = pfc["profile_type"] == pt
        if mask.any():
            w = pfc_zurich.loc[mask, "ic_width"]
            p = pfc.loc[mask, "price_shape"]
            rel_w = (w / p.abs().clip(lower=1.0)).mean() * 100
            print("  %-12s: mean_width=%.2f EUR/MWh  mean_price=%.2f  rel_width=%.1f%%" % (
                pt, w.mean(), p.mean(), rel_w))
else:
    print("  (No IC computed)")

# --- Energy consistency ---
print("\n--- ENERGY CONSISTENCY (mean PFC vs forward) ---")
idx_zurich = pfc.index.tz_convert("Europe/Zurich")
for key in sorted(cascaded_prices.keys()):
    base = cascaded_prices[key]
    # Parse key to get mask
    if len(key) == 4 and key.isdigit():
        mask = idx_zurich.year == int(key)
        label = f"Cal {key}"
    elif "Q" in key:
        year = int(key[:4])
        q = int(key[-1])
        q_months = {1: [1,2,3], 2: [4,5,6], 3: [7,8,9], 4: [10,11,12]}[q]
        mask = (idx_zurich.year == year) & (idx_zurich.month.isin(q_months))
        label = f"  {key}"
    elif len(key) == 7 and key[4] == "-":
        year = int(key[:4])
        month = int(key[5:])
        mask = (idx_zurich.year == year) & (idx_zurich.month == month)
        label = f"    {key}"
    else:
        continue

    n_pts = mask.sum()
    if n_pts == 0:
        continue
    mean_pfc = pfc.loc[mask, "price_shape"].mean()
    dev_pct = (mean_pfc - base) / abs(base) * 100
    marker = "OK" if abs(dev_pct) < 5.0 else "WARN"
    print("  %-14s: fwd=%.2f  pfc=%.2f  dev=%+.2f%%  n=%d  [%s]" % (
        label, base, mean_pfc, dev_pct, n_pts, marker))

# --- Calibration status ---
print("\n--- CALIBRATION STATUS ---")
if pfc["calibrated"].any():
    print("  ArbitrageFree calibration: APPLIED")
else:
    print("  ArbitrageFree calibration: NOT APPLIED (raw shape only)")

# --- Annual averages ---
print("\n--- ANNUAL AVERAGES ---")
for yr in sorted(idx_zurich.year.unique()):
    mask = idx_zurich.year == yr
    if mask.sum() > 0:
        p = pfc.loc[mask, "price_shape"]
        print("  %d: mean=%.2f  min=%.2f  max=%.2f  n=%d" % (yr, p.mean(), p.min(), p.max(), mask.sum()))

# --- Profile type distribution ---
print("\n--- PROFILE TYPE DISTRIBUTION ---")
for pt, cnt in pfc["profile_type"].value_counts().items():
    pct = cnt / len(pfc) * 100
    print("  %-12s: %d rows (%.1f%%)" % (pt, cnt, pct))

# --- Peak/Off-peak analysis ---
print("\n--- PEAK / OFF-PEAK ANALYSIS ---")
hour = idx_zurich.hour
dow = idx_zurich.dayofweek
is_peak = (hour >= 8) & (hour < 20) & (dow < 5)
for yr in sorted(idx_zurich.year.unique()):
    yr_mask = idx_zurich.year == yr
    if yr_mask.sum() == 0:
        continue
    pk = pfc.loc[yr_mask & is_peak, "price_shape"]
    op = pfc.loc[yr_mask & ~is_peak, "price_shape"]
    if len(pk) > 0 and len(op) > 0:
        spread = pk.mean() - op.mean()
        ratio = pk.mean() / op.mean() if op.mean() != 0 else float("nan")
        print("  %d: peak=%.2f  offpeak=%.2f  spread=%.2f  ratio=%.3f" % (
            yr, pk.mean(), op.mean(), spread, ratio))

total_time = time.time() - t0
print("\n" + "=" * 80)
print("  TOTAL EXECUTION TIME: %.1f seconds" % total_time)
print("  Output: %s.parquet + .csv" % out_base)
print("=" * 80)

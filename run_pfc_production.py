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

cal_ch = enrich_15min_index(epex_ch.index, country="CH")
cal_de = enrich_15min_index(epex_de.index, country="DE")

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

import yaml
with open("pfc_shaping/config.yaml") as _f:
    _config = yaml.safe_load(_f)
_model_cfg = _config.get("model", {})
_sh_mode = _model_cfg.get("shape_hourly_mode", "table")

if _sh_mode == "mlp":
    from pfc_shaping.model.shape_hourly_mlp import ShapeHourlyMLP
    sh = ShapeHourlyMLP()
    logger.info("  Using ShapeHourlyMLP (neural)")
else:
    from pfc_shaping.model.shape_hourly import ShapeHourly
    sh = ShapeHourly()
    logger.info("  Using ShapeHourly (table)")

sh.fit(epex_ch, cal_ch, hydro_df=hydro)

if _sh_mode == "mlp":
    logger.info("  MLP fitted (neural shape function)")
else:
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

from pfc_shaping.data.forward_proxy import load_base_prices as load_fwd_prices

base_prices, fwd_source = load_fwd_prices(
    epex_ch,
    eex_report_path="Price_Report_EEX.xlsx",
)
logger.info("  Forward source: %s", fwd_source)

# Use ContractCascader to fill in missing months
from pfc_shaping.calibration.cascading import ContractCascader

cascader = ContractCascader()
cascader.fit_seasonal_ratios(epex_ch)
cascader.fit_peak_ratios(epex_ch)

cascaded_prices = cascader.cascade(base_prices)
# Synthesize Peak forwards where only Base is quoted (e.g. Cal 2028+)
cascaded_prices = cascader.synthesize_peak_prices(cascaded_prices)

logger.info("  Input keys: %d", len(base_prices))
logger.info("  Cascaded keys: %d", len(cascaded_prices))
for k in sorted(cascaded_prices.keys()):
    logger.info("    %s: %.2f EUR/MWh", k, cascaded_prices[k])

# ═══════════════════════════════════════════════════════════════════════
# 9. ASSEMBLE PFC — full coverage of all EEX forward years
# ═══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("STEP 9: Assembling PFC")
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

# Compute horizon: cover all years with available forwards (through 31/12 of last year)
start_date = (pd.Timestamp.utcnow() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
max_fwd_year = max(
    int(k[:4]) for k in cascaded_prices.keys()
    if k[:4].isdigit() and len(k) >= 4
)
end_of_last_year = pd.Timestamp(f"{max_fwd_year}-12-31", tz="UTC")
future_start_ts = pd.Timestamp(start_date, tz="UTC")
horizon_days = (end_of_last_year - future_start_ts).days + 1
logger.info("  Horizon: %s -> 31/12/%d = %d days (%.1f years)",
            start_date, max_fwd_year, horizon_days, horizon_days / 365)
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

agg_dict = {
    "solar_regime_median": ("solar_regime", "median"),
    "load_deviation_median": ("load_deviation", "median"),
}
if "flow_deviation" in entso_zurich.columns:
    agg_dict["flow_deviation_median"] = ("flow_deviation", "median")

clim = entso_zurich.groupby(["month", "hour", "qh"]).agg(**agg_dict).reset_index()

# Map climatology onto the future index
future_keys = pd.DataFrame({
    "month": future_zurich.month,
    "hour": future_zurich.hour,
    "qh": (future_zurich.minute // 15) + 1,
}, index=future_idx)

entso_forecast = future_keys.merge(
    clim, on=["month", "hour", "qh"], how="left"
).set_index(future_idx)

rename_map = {
    "solar_regime_median": "solar_regime",
    "load_deviation_median": "load_deviation",
}
keep_cols = ["solar_regime", "load_deviation"]
if "flow_deviation_median" in entso_forecast.columns:
    rename_map["flow_deviation_median"] = "flow_deviation"
    keep_cols.append("flow_deviation")

entso_forecast = entso_forecast.rename(columns=rename_map)[keep_cols]

# Fill any NaN with neutral values
entso_forecast["solar_regime"] = entso_forecast["solar_regime"].fillna(1.0)
entso_forecast["load_deviation"] = entso_forecast["load_deviation"].fillna(0.0)
if "flow_deviation" in entso_forecast.columns:
    entso_forecast["flow_deviation"] = entso_forecast["flow_deviation"].fillna(0.0)

logger.info("  ENTSO-E climatology forecast: %d rows", len(entso_forecast))
logger.info("  solar_regime: mean=%.2f  std=%.2f",
            entso_forecast["solar_regime"].mean(), entso_forecast["solar_regime"].std())
logger.info("  load_deviation: mean=%.3f  std=%.3f",
            entso_forecast["load_deviation"].mean(), entso_forecast["load_deviation"].std())

# ── Load outages forecast (REMIT UMM) for shape adjustment ──
outages_forecast = None
outages_path = "pfc_shaping/data/outages_15min.parquet"
if os.path.exists(outages_path):
    outages_all = pd.read_parquet(outages_path)
    outages_forecast = outages_all[outages_all.index >= future_start]
    logger.info("  Outages forecast: %d rows, max unavail=%.0f MW",
                len(outages_forecast), outages_forecast["unavailable_mw"].max() if len(outages_forecast) > 0 else 0)

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
    outages_forecast=outages_forecast,
)

logger.info("  PFC assembled: %d rows in %.1fs", len(pfc), time.time() - t6)

# ═══════════════════════════════════════════════════════════════════════
# 9b. LEAR SHORT-TERM OVERLAY (D+1 to D+10)
# ═══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("STEP 9b: LEAR short-term forecast (D+1..D+10)")
logger.info("=" * 70)
t_lear = time.time()

try:
    from pfc_shaping.model.lear_forecaster import LEARForecaster

    # Load commodities for gas/CO2 features
    commodities_path = "data/commodities_cache.parquet"
    commodities_df = None
    if os.path.exists(commodities_path):
        commodities_df = pd.read_parquet(commodities_path)

    lear = LEARForecaster(tz="Europe/Zurich")
    lear.fit(
        epex_15min=epex_ch,
        entso_15min=entso,
        outages_15min=outages_all if outages_forecast is not None else None,
        commodities=commodities_df,
        hydro=hydro,
    )

    lear_forecast = lear.predict(horizon_days=10)
    logger.info("  LEAR forecast: %d hours, mean=%.1f EUR/MWh",
                len(lear_forecast), lear_forecast["price_lear"].mean())

    # Blend with PFC (D1-7 = LEAR, D8-10 = blend, D11+ = pure PFC)
    pfc = lear.blend_with_pfc(pfc, lear_forecast)

    # Save LEAR standalone forecast (parquet + CSV)
    lear_base = f"pfc_shaping/output/lear_forecast_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
    lear_forecast.to_parquet(f"{lear_base}.parquet", index=False)
    lear_forecast.to_csv(f"{lear_base}.csv", index=False)
    logger.info("  LEAR standalone saved: %s.parquet", lear_base)
    logger.info("  LEAR completed in %.1fs", time.time() - t_lear)

except Exception as exc:
    logger.warning("  LEAR overlay failed (PFC unchanged): %s", exc)
    import traceback
    traceback.print_exc()

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
if _sh_mode == "mlp":
    sh.save(f"{artifacts_dir}/shape_hourly_mlp.pkl")
else:
    sh.save(f"{artifacts_dir}/shape_hourly.parquet")
si.save(f"{artifacts_dir}/shape_intraday.parquet")
wv.save(f"{artifacts_dir}/water_value.parquet")
unc.save(f"{artifacts_dir}/uncertainty.parquet")

# ═══════════════════════════════════════════════════════════════════════
# 10b. BUILD DE PFC (reuses ShapeIntraday + Uncertainty from DE-LU)
# ═══════════════════════════════════════════════════════════════════════
logger.info("=" * 70)
logger.info("STEP 10b: Building DE PFC")
logger.info("=" * 70)

t7 = time.time()

# ── Fit ShapeHourly on DE EPEX (full history) ──
if _sh_mode == "mlp":
    from pfc_shaping.model.shape_hourly_mlp import ShapeHourlyMLP
    sh_de = ShapeHourlyMLP()
else:
    from pfc_shaping.model.shape_hourly import ShapeHourly
    sh_de = ShapeHourly()
sh_de.fit(epex_de, cal_de)
logger.info("  DE ShapeHourly fitted (%s mode)", _sh_mode)

# ── Load DE forwards ──
base_prices_de, fwd_source_de = load_fwd_prices(
    epex_de,
    eex_report_path="Price_Report_EEX.xlsx",
    market="DE",
)
logger.info("  DE forward source: %s", fwd_source_de)

# ── Cascade DE forwards ──
cascader_de = ContractCascader(tz="Europe/Berlin")
cascader_de.fit_seasonal_ratios(epex_de)
cascader_de.fit_peak_ratios(epex_de)
cascaded_prices_de = cascader_de.cascade(base_prices_de)
cascaded_prices_de = cascader_de.synthesize_peak_prices(cascaded_prices_de)

logger.info("  DE cascaded keys: %d", len(cascaded_prices_de))
for k in sorted(cascaded_prices_de.keys()):
    logger.info("    DE %s: %.2f EUR/MWh", k, cascaded_prices_de[k])

# ── Assemble DE PFC (no WaterValue — DE has no Swiss hydro) ──
assembler_de = PFCAssembler(
    shape_hourly=sh_de,
    shape_intraday=si,       # shared: trained on DE-LU
    uncertainty=unc,          # shared: trained on DE-LU
    water_value=None,         # no hydro correction for DE
    cascader=cascader_de,
    calibrator=calibrator,
)

pfc_de = assembler_de.build(
    base_prices=cascaded_prices_de,
    start_date=start_date,
    horizon_days=horizon_days,
    entso_forecast=entso_forecast,
    hydro_forecast=None,
    country="DE",
)

logger.info("  DE PFC assembled: %d rows in %.1fs", len(pfc_de), time.time() - t7)

# ── Save DE output ──
out_base_de = f"{out_dir}/pfc_de_15min_{today}"
pfc_de.to_parquet(f"{out_base_de}.parquet")
logger.info("  Saved: %s.parquet (%d rows)", out_base_de, len(pfc_de))
pfc_de.to_csv(f"{out_base_de}.csv")
logger.info("  Saved: %s.csv", out_base_de)

# Save DE model artifacts
if _sh_mode == "mlp":
    sh_de.save(f"{artifacts_dir}/shape_hourly_de_mlp.pkl")
else:
    sh_de.save(f"{artifacts_dir}/shape_hourly_de.parquet")

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
    elif key.endswith("-Peak"):
        continue  # Skip Peak keys in this summary (handled separately)
    elif "Q" in key:
        year = int(key[:4])
        q = int(key.split("Q")[1][0])
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

# ═══════════════════════════════════════════════════════════════════════
# 12. DE PFC SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  DE PFC SUMMARY")
print("=" * 80)
print("  Timestamps: %d" % len(pfc_de))
print("  Mean price: %.2f EUR/MWh" % pfc_de["price_shape"].mean())
print("  Min: %.2f  Max: %.2f" % (pfc_de["price_shape"].min(), pfc_de["price_shape"].max()))

idx_de_local = pfc_de.index.tz_convert("Europe/Berlin")
print("\n--- DE ANNUAL AVERAGES ---")
for yr in sorted(idx_de_local.year.unique()):
    mask = idx_de_local.year == yr
    if mask.sum() > 0:
        p = pfc_de.loc[mask, "price_shape"]
        print("  %d: mean=%.2f  min=%.2f  max=%.2f  n=%d" % (yr, p.mean(), p.min(), p.max(), mask.sum()))

print("\n--- DE PEAK / OFF-PEAK ---")
hour_de = idx_de_local.hour
dow_de = idx_de_local.dayofweek
is_peak_de = (hour_de >= 8) & (hour_de < 20) & (dow_de < 5)
for yr in sorted(idx_de_local.year.unique()):
    yr_mask = idx_de_local.year == yr
    if yr_mask.sum() == 0:
        continue
    pk = pfc_de.loc[yr_mask & is_peak_de, "price_shape"]
    op = pfc_de.loc[yr_mask & ~is_peak_de, "price_shape"]
    if len(pk) > 0 and len(op) > 0:
        spread = pk.mean() - op.mean()
        ratio = pk.mean() / op.mean() if op.mean() != 0 else float("nan")
        print("  %d: peak=%.2f  offpeak=%.2f  spread=%.2f  ratio=%.3f" % (
            yr, pk.mean(), op.mean(), spread, ratio))

total_time = time.time() - t0
print("\n" + "=" * 80)
print("  TOTAL EXECUTION TIME: %.1f seconds" % total_time)
print("  Output CH: %s.parquet + .csv" % out_base)
print("  Output DE: %s.parquet + .csv" % out_base_de)
print("=" * 80)

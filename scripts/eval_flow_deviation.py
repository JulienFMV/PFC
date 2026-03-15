"""
Evaluation de l'impact de flow_deviation sur le modèle ShapeIntraday.

Utilise les données DE-LU post-Oct 2025 (vrai 15min auction).
Compare baseline (solar + load) vs enrichi (+flow_deviation).
"""

import sys
import time
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.model.shape_intraday import ShapeIntraday

# ── Load data ──────────────────────────────────────────────────────────────
print("=" * 70)
print("LOADING DATA — DE-LU post Oct 2025 (real 15min auction)")
print("=" * 70)

epex_de = pd.read_parquet("pfc_shaping/data/epex_de_15min.parquet")
entso = pd.read_parquet("pfc_shaping/data/entso_15min.parquet")

# Post Oct 2025 only (real 15min)
cutoff = pd.Timestamp("2025-10-01", tz="UTC")
epex_de = epex_de[epex_de.index >= cutoff]

# Align entso to epex_de index
entso = entso.reindex(epex_de.index)

# Compute flow_deviation
print("Computing flow_deviation...")
if "cross_border_mw" in entso.columns:
    monthly = entso.groupby(entso.index.to_period("M"))["cross_border_mw"]
    entso["flow_deviation"] = monthly.transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    entso["flow_deviation"] = entso["flow_deviation"].fillna(0)
    print(f"  flow_deviation stats: mean={entso['flow_deviation'].mean():.3f}, "
          f"std={entso['flow_deviation'].std():.3f}, "
          f"non-zero={(entso['flow_deviation'] != 0).sum()}/{len(entso)}")
else:
    entso["flow_deviation"] = 0.0

# Calendar enrichment
cal = enrich_15min_index(epex_de.index)

print(f"Data: {len(epex_de)} rows, {epex_de.index.min().date()} → {epex_de.index.max().date()}")

# Verify 15min granularity
epex_check = epex_de.copy()
epex_check["hour_key"] = epex_check.index.floor("h")
within_std = epex_check.groupby("hour_key")["price_eur_mwh"].std()
pct_15min = (within_std > 0.01).mean() * 100
print(f"Actual 15min variation: {pct_15min:.1f}% of hours")

# ── Walk-forward: train on first 4 months, test on last ~1.5 months ──────
print("\n" + "=" * 70)
print("WALK-FORWARD BACKTEST")
print("=" * 70)

test_start = epex_de.index.max() - pd.Timedelta(days=45)
train_mask = epex_de.index < test_start
test_mask = epex_de.index >= test_start

train_epex = epex_de[train_mask]
test_epex = epex_de[test_mask]
train_entso = entso[train_mask]
test_entso = entso[test_mask]
train_cal = cal[train_mask]
test_cal = cal[test_mask]

print(f"Train: {len(train_epex)} rows ({train_epex.index.min().date()} → {train_epex.index.max().date()})")
print(f"Test:  {len(test_epex)} rows ({test_epex.index.min().date()} → {test_epex.index.max().date()})")

# ── Baseline ──────────────────────────────────────────────────────────────
print("\n--- Baseline (solar_regime + load_deviation) ---")
t0 = time.time()

entso_bl = train_entso[["solar_regime", "load_deviation"]].copy()
si_base = ShapeIntraday()
si_base.fit(train_epex, entso_bl, train_cal)

entso_test_bl = test_entso[["solar_regime", "load_deviation"]].copy()
fq_base = si_base.apply(test_epex.index, test_cal, entso_test_bl)

print(f"  Fitted in {time.time()-t0:.1f}s")
print(f"  Layer 2 corrections: {len(si_base.corrections_)}/{len(si_base.base_factors_)} cells")

# ── Enriched (+flow) ──────────────────────────────────────────────────────
print("\n--- Enriched (+flow_deviation) ---")
t0 = time.time()

entso_en = train_entso[["solar_regime", "load_deviation", "flow_deviation"]].copy()
si_flow = ShapeIntraday()
si_flow.fit(train_epex, entso_en, train_cal)

entso_test_en = test_entso[["solar_regime", "load_deviation", "flow_deviation"]].copy()
fq_flow = si_flow.apply(test_epex.index, test_cal, entso_test_en)

print(f"  Fitted in {time.time()-t0:.1f}s")
print(f"  Layer 2 corrections: {len(si_flow.corrections_)}/{len(si_flow.base_factors_)} cells")

# ── Compute actual f_Q ────────────────────────────────────────────────────
print("\n--- Computing actual f_Q ratios ---")

test_df = test_epex[["price_eur_mwh"]].copy()
test_df["hour_key"] = test_df.index.floor("h")
hour_means = test_df.groupby("hour_key")["price_eur_mwh"].transform("mean")
test_df["hour_mean"] = hour_means

valid = test_df["hour_mean"].abs() > 0.5
test_df = test_df[valid]
test_df["f_Q_actual"] = test_df["price_eur_mwh"] / test_df["hour_mean"]

fq_base_a = fq_base.reindex(test_df.index).dropna()
fq_flow_a = fq_flow.reindex(test_df.index).dropna()
common_idx = test_df.index.intersection(fq_base_a.index).intersection(fq_flow_a.index)

actual = test_df.loc[common_idx, "f_Q_actual"].values
pred_base = fq_base_a.loc[common_idx].values
pred_flow = fq_flow_a.loc[common_idx].values

# Also compute price-level errors (f_Q * hour_mean)
hour_means_test = test_df.loc[common_idx, "hour_mean"].values
price_actual = actual * hour_means_test
price_base = pred_base * hour_means_test
price_flow = pred_flow * hour_means_test

# ── Results ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS (f_Q shape ratio)")
print("=" * 70)

rmse_b = np.sqrt(np.mean((actual - pred_base) ** 2))
rmse_f = np.sqrt(np.mean((actual - pred_flow) ** 2))
mae_b = np.mean(np.abs(actual - pred_base))
mae_f = np.mean(np.abs(actual - pred_flow))
bias_b = np.mean(pred_base - actual)
bias_f = np.mean(pred_flow - actual)

# Price-level errors
rmse_p_b = np.sqrt(np.mean((price_actual - price_base) ** 2))
rmse_p_f = np.sqrt(np.mean((price_actual - price_flow) ** 2))

print(f"\n{'Metric':<30} {'Baseline':>12} {'+ Flow':>12} {'Gain %':>10}")
print("-" * 66)
print(f"{'RMSE f_Q (ratio)':<30} {rmse_b:>12.6f} {rmse_f:>12.6f} {(1-rmse_f/rmse_b)*100:>+9.2f}%")
print(f"{'MAE f_Q (ratio)':<30} {mae_b:>12.6f} {mae_f:>12.6f} {(1-mae_f/mae_b)*100:>+9.2f}%")
print(f"{'Bias f_Q':<30} {bias_b:>12.6f} {bias_f:>12.6f}")
print(f"{'RMSE price (EUR/MWh)':<30} {rmse_p_b:>12.2f} {rmse_p_f:>12.2f} {(1-rmse_p_f/rmse_p_b)*100:>+9.2f}%")
print(f"{'Layer 2 cells':<30} {len(si_base.corrections_):>12d} {len(si_flow.corrections_):>12d}")
print(f"{'N observations':<30} {len(common_idx):>12d}")

# ── Per-hour ──────────────────────────────────────────────────────────────
print("\n--- Per-hour RMSE (ratio) ---")
test_hours = test_cal.loc[common_idx, "heure_hce"].astype(int).values

print(f"\n{'Hour':<6} {'RMSE Base':>12} {'RMSE Flow':>12} {'Gain %':>10}  {'Price RMSE B':>12} {'Price RMSE F':>12}")
print("-" * 70)
for h in range(24):
    mask_h = test_hours == h
    if mask_h.sum() < 20:
        continue
    rb = np.sqrt(np.mean((actual[mask_h] - pred_base[mask_h]) ** 2))
    rf = np.sqrt(np.mean((actual[mask_h] - pred_flow[mask_h]) ** 2))
    pb = np.sqrt(np.mean((price_actual[mask_h] - price_base[mask_h]) ** 2))
    pf = np.sqrt(np.mean((price_actual[mask_h] - price_flow[mask_h]) ** 2))
    gain = (1 - rf / rb) * 100 if rb > 0 else 0
    marker = " <<<" if gain > 1.0 else (" !!!" if gain < -1.0 else "")
    print(f"  {h:>2}h   {rb:>12.6f} {rf:>12.6f} {gain:>+9.2f}%  {pb:>12.2f} {pf:>12.2f}{marker}")

# ── Flow coefficients ──────────────────────────────────────────────────────
print("\n--- Flow coefficients in Layer 2 ---")
flow_coefs_by_hour = {}
for key, corr in si_flow.corrections_.items():
    saison, tj, h = key
    for q in range(1, 5):
        bflow = corr.get(f"b_flow_q{q}", None)
        if bflow is not None:
            flow_coefs_by_hour.setdefault(h, []).append(bflow)

if flow_coefs_by_hour:
    print(f"\n{'Hour':<6} {'Mean b_flow':>12} {'Abs Mean':>10} {'N':>6}")
    print("-" * 36)
    for h in sorted(flow_coefs_by_hour.keys()):
        vals = flow_coefs_by_hour[h]
        print(f"  {h:>2}h   {np.mean(vals):>+12.6f} {np.mean(np.abs(vals)):>10.6f} {len(vals):>6d}"
              + (" ***" if np.mean(np.abs(vals)) > 0.005 else ""))
else:
    print("  No flow coefficients (all rejected by R² OOS gate)")
    print("\n  Investigating why — checking Ridge R² on training data...")

    # Debug: manually check if flow_deviation has signal
    from pfc_shaping.data.calendar_ch import enrich_15min_index

    train_debug = train_epex[["price_eur_mwh"]].copy()
    train_debug = train_debug.join(train_cal[["saison", "type_jour", "heure_hce", "quart"]])
    train_debug = train_debug.join(train_entso[["solar_regime", "load_deviation", "flow_deviation"]])
    train_debug = train_debug.dropna()

    train_debug["hour_key"] = train_debug.index.floor("h")
    hmeans = train_debug.groupby("hour_key")["price_eur_mwh"].transform("mean")
    train_debug["hour_mean"] = hmeans
    valid_train = train_debug["hour_mean"].abs() > 0.5
    train_debug = train_debug[valid_train]
    train_debug["ratio"] = train_debug["price_eur_mwh"] / train_debug["hour_mean"]

    # Check correlation between flow_deviation and ratio for key hours
    print(f"\n  Correlation(flow_deviation, f_Q_ratio) by hour:")
    for h in [6, 7, 8, 9, 10, 17, 18, 19, 20]:
        mask = train_debug["heure_hce"] == h
        if mask.sum() > 50:
            corr = train_debug.loc[mask, "flow_deviation"].corr(train_debug.loc[mask, "ratio"])
            n = mask.sum()
            print(f"    {h:>2}h: r={corr:+.4f} (n={n})")

    # Check overall
    corr_overall = train_debug["flow_deviation"].corr(train_debug["ratio"])
    print(f"    ALL: r={corr_overall:+.4f} (n={len(train_debug)})")

    # Check if data has enough variation
    print(f"\n  flow_deviation in train: non-NaN={train_debug['flow_deviation'].notna().sum()}, "
          f"std={train_debug['flow_deviation'].std():.3f}")
    print(f"  ratio in train: mean={train_debug['ratio'].mean():.4f}, "
          f"std={train_debug['ratio'].std():.4f}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

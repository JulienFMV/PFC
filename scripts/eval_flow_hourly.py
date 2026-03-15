"""
Evaluation du signal flow_deviation au niveau HORAIRE (f_H).

La flexibilité hydro CH (turbinage peak / pompage offpeak) impacte surtout
la FORME HORAIRE, pas la forme 15min intra-horaire.

Ce script mesure la corrélation entre flow_deviation et les écarts de
prix horaires par rapport au profil moyen, par heure et par saison.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from pfc_shaping.data.calendar_ch import enrich_15min_index

# ── Load ──────────────────────────────────────────────────────────────────
print("=" * 70)
print("HOURLY SHAPE ANALYSIS — Impact of cross-border flows")
print("=" * 70)

epex = pd.read_parquet("pfc_shaping/data/epex_15min.parquet")
entso = pd.read_parquet("pfc_shaping/data/entso_15min.parquet")

# Resample to hourly
epex_h = epex["price_eur_mwh"].resample("h").mean()
entso_h = entso.resample("h").mean()

# Compute flow_deviation at hourly level
monthly = entso_h.groupby(entso_h.index.to_period("M"))["cross_border_mw"]
entso_h["flow_deviation"] = monthly.transform(
    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
).fillna(0)

# Calendar
cal = enrich_15min_index(epex.index)
cal_h = cal.resample("h").first()

# Compute f_H_actual : price_hour / price_day_mean
df = pd.DataFrame({
    "price": epex_h,
    "flow_dev": entso_h["flow_deviation"],
    "load_dev": entso_h["load_deviation"] if "load_deviation" in entso_h else 0,
    "solar_regime": entso_h["solar_regime"] if "solar_regime" in entso_h else 1,
    "cross_border_mw": entso_h["cross_border_mw"],
    "saison": cal_h["saison"],
    "type_jour": cal_h["type_jour"],
    "hour": cal_h["heure_hce"],
}).dropna(subset=["price", "flow_dev"])

# Daily mean
df["day"] = df.index.date
daily_mean = df.groupby("day")["price"].transform("mean")
df["f_H_actual"] = df["price"] / daily_mean
df = df[daily_mean.abs() > 0.5]  # avoid div/0

print(f"Data: {len(df)} hourly obs, {df.index.min().date()} → {df.index.max().date()}")
print(f"flow_deviation: mean={df['flow_dev'].mean():.3f}, std={df['flow_dev'].std():.3f}")

# ── Correlations ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CORRELATION: flow_deviation vs f_H shape deviation")
print("=" * 70)

# f_H deviation = actual f_H - mean f_H for that (saison, type_jour, hour)
cell_means = df.groupby(["saison", "type_jour", "hour"])["f_H_actual"].transform("mean")
df["f_H_deviation"] = df["f_H_actual"] - cell_means

print(f"\n{'Hour':<6} {'Corr(flow, f_H_dev)':>20} {'Corr(load, f_H_dev)':>20} {'Corr(flow, price)':>20} {'N':>8}")
print("-" * 78)

for h in range(24):
    mask = df["hour"] == h
    sub = df[mask]
    if len(sub) < 100:
        continue
    r_flow = sub["flow_dev"].corr(sub["f_H_deviation"])
    r_load = sub["load_dev"].corr(sub["f_H_deviation"])
    r_price = sub["flow_dev"].corr(sub["price"])
    marker = " ***" if abs(r_flow) > 0.10 else ""
    print(f"  {h:>2}h   {r_flow:>+20.4f} {r_load:>+20.4f} {r_price:>+20.4f} {len(sub):>8d}{marker}")

r_overall = df["flow_dev"].corr(df["f_H_deviation"])
r_price_all = df["flow_dev"].corr(df["price"])
print(f"  ALL   {r_overall:>+20.4f} {'':>20} {r_price_all:>+20.4f} {len(df):>8d}")

# ── Per-season ────────────────────────────────────────────────────────────
print(f"\n--- Per-season correlations (flow_dev vs f_H_deviation) ---")
print(f"\n{'Season':<12} {'Peak (8-20h)':>15} {'Offpeak':>15} {'Ramp 6-10':>15} {'Ramp 17-20':>15}")
print("-" * 75)

for s in ["Hiver", "Printemps", "Ete", "Automne"]:
    mask_s = df["saison"] == s
    if mask_s.sum() < 200:
        continue
    sub = df[mask_s]

    peak = sub[sub["hour"].between(8, 20)]
    offpeak = sub[~sub["hour"].between(8, 20)]
    ramp_am = sub[sub["hour"].between(6, 10)]
    ramp_pm = sub[sub["hour"].between(17, 20)]

    r_peak = peak["flow_dev"].corr(peak["f_H_deviation"]) if len(peak) > 50 else np.nan
    r_off = offpeak["flow_dev"].corr(offpeak["f_H_deviation"]) if len(offpeak) > 50 else np.nan
    r_ram = ramp_am["flow_dev"].corr(ramp_am["f_H_deviation"]) if len(ramp_am) > 50 else np.nan
    r_rpm = ramp_pm["flow_dev"].corr(ramp_pm["f_H_deviation"]) if len(ramp_pm) > 50 else np.nan

    print(f"  {s:<10} {r_peak:>+15.4f} {r_off:>+15.4f} {r_ram:>+15.4f} {r_rpm:>+15.4f}")

# ── Direct price vs flow scatter stats ────────────────────────────────────
print(f"\n--- Cross-border flow vs price by hour ---")
print(f"\n{'Hour':<6} {'Mean price when':>35}    {'Price diff':>12}")
print(f"{'':>6} {'Export (flow<-1σ)':>17} {'Import (flow>+1σ)':>17}    {'(imp - exp)':>12}")
print("-" * 72)

for h in range(24):
    mask = df["hour"] == h
    sub = df[mask]
    if len(sub) < 100:
        continue
    export_heavy = sub[sub["flow_dev"] < -1]["price"]
    import_heavy = sub[sub["flow_dev"] > 1]["price"]
    if len(export_heavy) > 10 and len(import_heavy) > 10:
        p_exp = export_heavy.mean()
        p_imp = import_heavy.mean()
        diff = p_imp - p_exp
        marker = " ***" if abs(diff) > 10 else ""
        print(f"  {h:>2}h   {p_exp:>17.1f} {p_imp:>17.1f}    {diff:>+12.1f}{marker}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
Si les corrélations peak sont significatives (|r| > 0.1), le flow_deviation
devrait être intégré dans ShapeHourly (f_H) plutôt que ShapeIntraday (f_Q).

L'effet attendu : quand la CH exporte fort (flow_dev << 0, turbinage),
les heures de pointe ont un f_H plus "pointu" (plus d'énergie en peak).
Quand la CH importe fort (flow_dev >> 0), le profil est plus plat.
""")

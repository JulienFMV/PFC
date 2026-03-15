#!/usr/bin/env python3
"""
autoresearch_eval_lear.py
--------------------------
Fixed evaluation harness for LEAR short-term forecaster autoresearch.
DO NOT MODIFY — this is the immutable evaluation script.

Runs LEAR backtest on the last 30 days and outputs metrics.
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── Load data from parquet (same as run_pfc_production.py) ─────────────
t0 = time.time()

DATA = "pfc_shaping/data"

print("status:loading_data", flush=True)

try:
    epex_ch = pd.read_parquet(f"{DATA}/epex_15min.parquet")
    epex_de = pd.read_parquet(f"{DATA}/epex_de_15min.parquet")
    entso = pd.read_parquet(f"{DATA}/entso_15min.parquet")
except Exception as e:
    print(f"status:data_error:{e}")
    sys.exit(1)

# ── Fit LEAR model ─────────────────────────────────────────────────────
print("status:fitting_lear", flush=True)

sys.path.insert(0, ".")

try:
    from pfc_shaping.model.lear_forecaster import LEARForecaster

    lear = LEARForecaster()
    lear.fit(
        epex_15min=epex_ch,
        entso_15min=entso,
        epex_de_15min=epex_de,
    )
except Exception as e:
    print(f"status:fit_error:{e}")
    sys.exit(1)

# ── Run backtest ───────────────────────────────────────────────────────
print("status:running_backtest", flush=True)

try:
    bt = lear.backtest(n_days=30, horizon=1)
except Exception as e:
    print(f"status:backtest_error:{e}")
    sys.exit(1)

# ── Compute metrics ────────────────────────────────────────────────────
mae = float(bt["abs_error"].mean())
rmse = float(np.sqrt((bt["error"] ** 2).mean()))
mape = float(bt["ape"].mean())
corr = float(bt["forecast"].corr(bt["actual"]))
bias = float(bt["error"].mean())

# Per-period metrics
peak = bt[bt["hour"].between(7, 19)]
offpeak = bt[~bt["hour"].between(7, 19)]
mae_peak = float(peak["abs_error"].mean())
mae_offpeak = float(offpeak["abs_error"].mean())
corr_peak = float(peak["forecast"].corr(peak["actual"]))
corr_offpeak = float(offpeak["forecast"].corr(offpeak["actual"]))

# Composite score (lower is better): weighted combination
# Normalized to baseline: MAE=15.0, RMSE=22.3, MAPE=30.9, corr=0.629
score = (
    0.35 * (mae / 15.0)
    + 0.30 * (rmse / 22.3)
    + 0.20 * (mape / 30.9)
    + 0.15 * (1.0 - corr)
)

elapsed = time.time() - t0

# ── Output (grep-friendly) ────────────────────────────────────────────
print(f"mae:{mae:.2f}")
print(f"rmse:{rmse:.2f}")
print(f"mape:{mape:.2f}")
print(f"corr:{corr:.3f}")
print(f"bias:{bias:.2f}")
print(f"mae_peak:{mae_peak:.2f}")
print(f"mae_offpeak:{mae_offpeak:.2f}")
print(f"corr_peak:{corr_peak:.3f}")
print(f"corr_offpeak:{corr_offpeak:.3f}")
print(f"score:{score:.4f}")
print(f"elapsed:{elapsed:.1f}s")
print(f"status:ok")
print(f"n_hours:{len(bt)}")

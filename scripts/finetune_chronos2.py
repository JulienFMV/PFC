#!/usr/bin/env python3
"""
Fine-tune Chronos-2 with LoRA on CH+DE EPEX hourly electricity prices.

Uses AutoGluon TimeSeriesPredictor for streamlined fine-tuning.
The fine-tuned model is saved and used by FoundationForecaster.

Usage:
    python scripts/finetune_chronos2.py

Output:
    pfc_shaping/model/chronos2_finetuned/ — fine-tuned model checkpoint
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import pandas as pd
import numpy as np

DATA = "pfc_shaping/data"
MODEL_OUTPUT = "pfc_shaping/model/chronos2_finetuned"

print("=" * 60)
print("Chronos-2 LoRA Fine-Tuning on CH+DE EPEX Data")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────
print("\n[1/4] Loading data...")

epex_ch = pd.read_parquet(f"{DATA}/epex_15min.parquet")
epex_de = pd.read_parquet(f"{DATA}/epex_de_15min.parquet")

# Resample to hourly
ch_h = epex_ch["price_eur_mwh"].resample("h").mean().dropna()
de_col = "price_eur_mwh" if "price_eur_mwh" in epex_de.columns else epex_de.columns[0]
de_h = epex_de[de_col].resample("h").mean().dropna()

# Load DE renewable forecasts if available
de_re_path = f"{DATA}/de_renewable_forecast.parquet"
has_renewables = os.path.exists(de_re_path)
if has_renewables:
    de_re = pd.read_parquet(de_re_path)
    wind_h = de_re["forecast_wind_de_mw"].resample("h").mean()
    solar_h = de_re["forecast_solar_de_mw"].resample("h").mean()
    print(f"  DE renewable forecasts loaded: wind={wind_h.mean():.0f} MW, solar={solar_h.mean():.0f} MW")

# Align all series to common index
common_idx = ch_h.index.intersection(de_h.index)
if has_renewables:
    common_idx = common_idx.intersection(wind_h.dropna().index)

print(f"  CH prices: {len(ch_h)} hours")
print(f"  DE prices: {len(de_h)} hours")
print(f"  Common: {len(common_idx)} hours ({common_idx[0].date()} → {common_idx[-1].date()})")

# ── Prepare TimeSeriesDataFrame ────────────────────────────
print("\n[2/4] Preparing training data...")

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# Build DataFrame with both markets as separate items + covariates
records = []
for item_id, prices in [("CH", ch_h), ("DE", de_h)]:
    for ts in common_idx:
        row = {
            "item_id": item_id,
            "timestamp": ts.tz_localize(None) if ts.tzinfo else ts,
            "target": float(prices.get(ts, np.nan)),
        }
        if has_renewables:
            row["wind_de_gw"] = float(wind_h.get(ts, 0)) / 1000  # MW -> GW for scale
            row["solar_de_gw"] = float(solar_h.get(ts, 0)) / 1000
        records.append(row)

df = pd.DataFrame(records)
df = df.dropna(subset=["target"])

known_covariates = []
if has_renewables:
    known_covariates = ["wind_de_gw", "solar_de_gw"]

ts_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp",
)

# Train/test split: last 30 days for validation
prediction_length = 24  # 24 hours ahead
train_data, test_data = ts_data.train_test_split(prediction_length=prediction_length)

n_train = len(train_data)
n_test = len(test_data)
print(f"  Training: {n_train} rows, Test: {n_test} rows")
print(f"  Prediction length: {prediction_length} hours")
if known_covariates:
    print(f"  Covariates: {known_covariates}")

# ── Fine-tune ──────────────────────────────────────────────
print("\n[3/4] Fine-tuning Chronos-2 with LoRA...")
print("  This may take 10-30 minutes on CPU/MPS...")

hyperparameters = {
    "Chronos2": {
        "fine_tune": True,
        "fine_tune_mode": "lora",
        "fine_tune_lr": 1e-4,
        "fine_tune_steps": 1000,
        "fine_tune_batch_size": 16,
        "model_path": "amazon/chronos-2",
        "ag_args": {"name_suffix": "FineTuned"},
    }
}

predictor_kwargs = {
    "prediction_length": prediction_length,
    "target": "target",
    "eval_metric": "MASE",
    "freq": "h",
}

if known_covariates:
    predictor_kwargs["known_covariates_names"] = known_covariates

predictor = TimeSeriesPredictor(**predictor_kwargs)

predictor.fit(
    train_data=train_data,
    hyperparameters=hyperparameters,
    time_limit=1800,  # 30 min max
    enable_ensemble=False,
)

# ── Evaluate & Save ────────────────────────────────────────
print("\n[4/4] Evaluating and saving...")

# Evaluate on test set
leaderboard = predictor.leaderboard(test_data)
print("\nLeaderboard:")
print(leaderboard.to_string())

# Save
predictor.save(MODEL_OUTPUT)
print(f"\nModel saved to: {MODEL_OUTPUT}")

# Quick test prediction
predictions = predictor.predict(test_data)
print(f"\nSample predictions (CH, next 6h):")
ch_pred = predictions.loc["CH"].head(6)
print(ch_pred.to_string())

print("\nDone! The fine-tuned model can be loaded with:")
print(f'  predictor = TimeSeriesPredictor.load("{MODEL_OUTPUT}")')

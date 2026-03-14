"""
autoresearch_eval.py — PFC Autoresearch Evaluation Script
=========================================================
Equivalent of Karpathy's train.py for our PFC model.

Runs a walk-forward backtest of the PFC model against realized EPEX spot
prices and outputs a standardized metrics block.

Usage:
    python3 autoresearch_eval.py > eval.log 2>&1
    grep "^rmse:" eval.log

The agent reads these metrics to decide keep/revert.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Setup paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))


def main() -> None:
    t0 = time.time()

    # ── Load EPEX data ───────────────────────────────────────────────────
    from dashboard.utils import load_epex

    epex = load_epex()
    if epex is None or len(epex) < 96 * 365:
        print("ERROR: Insufficient EPEX data", file=sys.stderr)
        print("---")
        print("rmse:             999.000000")
        print("status:           crash")
        return

    # ── Setup: train/test split ──────────────────────────────────────────
    test_months = 2
    cutoff = epex.index.max() - pd.DateOffset(months=test_months)
    train = epex[epex.index < cutoff]
    test = epex[epex.index >= cutoff]

    if len(test) < 96 * 14:
        print("ERROR: Insufficient test data", file=sys.stderr)
        print("---")
        print("rmse:             999.000000")
        print("status:           crash")
        return

    # ── Fit model components ─────────────────────────────────────────────
    try:
        import yaml

        from pfc_shaping.data.calendar_ch import enrich_15min_index
        from pfc_shaping.data.forward_proxy import derive_base_prices
        from pfc_shaping.model.assembler import PFCAssembler
        from pfc_shaping.model.shape_hourly import ShapeHourly
        from pfc_shaping.model.shape_intraday import ShapeIntraday
        from pfc_shaping.model.uncertainty import Uncertainty

        # Load config
        cfg_path = ROOT / "pfc_shaping" / "config.yaml"
        with cfg_path.open() as f:
            config = yaml.safe_load(f)

        model_cfg = config.get("model", {})
        lookback = model_cfg.get("lookback_months", 36)
        sigma = model_cfg.get("gaussian_sigma", 0.5)

        # Lookback window
        lb_cutoff = train.index.max() - pd.DateOffset(months=lookback)
        train_lb = train[train.index >= lb_cutoff]
        cal = enrich_15min_index(train_lb.index)

        # Load ENTSO-E data (solar_regime, load_deviation)
        entso_path = ROOT / "pfc_shaping" / "data" / "entso_15min.parquet"
        entso_df = None
        if entso_path.exists():
            entso_full = pd.read_parquet(entso_path)
            entso_df = entso_full.reindex(train_lb.index)
            print(f"ENTSO-E loaded: {len(entso_df)} rows, cols={list(entso_df.columns)}", file=sys.stderr)

        # Fit shape models
        sh = ShapeHourly(sigma=sigma)
        sh.fit(train_lb, cal)

        si = ShapeIntraday()
        si.fit(train_lb, entso_df=entso_df, calendar_df=cal)

        unc = Uncertainty()
        unc.fit(train_lb, cal)

        # Base prices
        base_prices = derive_base_prices(train, start_year=cutoff.year, n_years=1)

        # Assemble PFC
        assembler = PFCAssembler(
            shape_hourly=sh,
            shape_intraday=si,
            uncertainty=unc,
        )

        # ENTSO-E forecast for the test period (use actual data as "perfect forecast")
        entso_forecast = None
        if entso_df is not None and entso_full is not None:
            entso_forecast = entso_full[entso_full.index >= cutoff]

        pfc = assembler.build(
            base_prices=base_prices,
            start_date=cutoff.strftime("%Y-%m-%d"),
            horizon_days=test_months * 31,
            entso_forecast=entso_forecast,
        )

        # ── Evaluate against spot ────────────────────────────────────────
        common_idx = pfc.index.intersection(test.index)

        if len(common_idx) < 96 * 7:
            print("ERROR: Insufficient overlap", file=sys.stderr)
            print("---")
            print("rmse:             999.000000")
            print("status:           crash")
            return

        pfc_prices = pfc.loc[common_idx, "price_shape"]
        spot_prices = test.loc[common_idx, "price_eur_mwh"]

        errors = pfc_prices.values - spot_prices.values
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        bias = float(np.mean(errors))

        # Shape RMSE (level-adjusted)
        scale = spot_prices.mean() / pfc_prices.mean() if pfc_prices.mean() != 0 else 1.0
        shape_errors = pfc_prices.values * scale - spot_prices.values
        rmse_shape = float(np.sqrt(np.mean(shape_errors**2)))

        # Hourly RMSE
        hourly_rmse = []
        idx_zh = common_idx.tz_convert("Europe/Zurich")
        for h in range(24):
            mask = idx_zh.hour == h
            if mask.sum() > 0:
                hourly_rmse.append(np.sqrt(np.mean(errors[mask] ** 2)))
        rmse_hourly_std = float(np.std(hourly_rmse))

        # IC coverage (if p10/p90 available)
        ic80_coverage = 0.0
        if "p10" in pfc.columns and "p90" in pfc.columns:
            p10 = pfc.loc[common_idx, "p10"]
            p90 = pfc.loc[common_idx, "p90"]
            covered = (spot_prices >= p10.values) & (spot_prices <= p90.values)
            ic80_coverage = float(covered.mean())

        elapsed = time.time() - t0

        # ── Output metrics (Karpathy format) ─────────────────────────────
        print("---")
        print(f"rmse:             {rmse:.6f}")
        print(f"rmse_shape:       {rmse_shape:.6f}")
        print(f"mae:              {mae:.6f}")
        print(f"bias:             {bias:.6f}")
        print(f"rmse_hourly_std:  {rmse_hourly_std:.6f}")
        print(f"ic80_coverage:    {ic80_coverage:.6f}")
        print(f"n_points:         {len(common_idx)}")
        print(f"test_period:      {common_idx.min().date()} -> {common_idx.max().date()}")
        print(f"eval_seconds:     {elapsed:.1f}")
        print(f"status:           ok")

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        print("---")
        print("rmse:             999.000000")
        print(f"eval_seconds:     {elapsed:.1f}")
        print("status:           crash")


if __name__ == "__main__":
    main()

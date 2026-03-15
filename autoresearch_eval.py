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
        from pfc_shaping.model.shape_intraday import ShapeIntraday
        from pfc_shaping.model.uncertainty import Uncertainty

        # Load config
        cfg_path = ROOT / "pfc_shaping" / "config.yaml"
        with cfg_path.open() as f:
            config = yaml.safe_load(f)

        model_cfg = config.get("model", {})
        lookback = model_cfg.get("lookback_months", 36)
        sigma = model_cfg.get("gaussian_sigma", 0.5)
        sh_mode = model_cfg.get("shape_hourly_mode", "table")

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

        # Load hydro reservoir data (KYOS analogue weighting)
        hydro_path = ROOT / "pfc_shaping" / "data" / "hydro_reservoir.parquet"
        hydro_df = None
        if hydro_path.exists():
            hydro_df = pd.read_parquet(hydro_path)
            print(f"Hydro reservoir loaded: {len(hydro_df)} weeks, fill range={hydro_df['fill_pct'].min():.1%}-{hydro_df['fill_pct'].max():.1%}", file=sys.stderr)

        # Fit shape models (with hydro analogue weighting)
        if sh_mode == "mlp":
            from pfc_shaping.model.shape_hourly_mlp import ShapeHourlyMLP
            sh = ShapeHourlyMLP()
            print("Using ShapeHourlyMLP (neural)", file=sys.stderr)
        else:
            from pfc_shaping.model.shape_hourly import ShapeHourly
            sh = ShapeHourly(sigma=sigma)
            print("Using ShapeHourly (table)", file=sys.stderr)
        sh.fit(train_lb, cal, hydro_df=hydro_df)

        si = ShapeIntraday()
        si.fit(train_lb, entso_df=entso_df, calendar_df=cal, hydro_df=hydro_df)

        unc = Uncertainty()
        unc.fit(train_lb, cal)

        # Base prices
        base_prices = derive_base_prices(train, start_year=cutoff.year, n_years=1)

        # Cascading + Calibrator
        from pfc_shaping.calibration.cascading import ContractCascader
        from pfc_shaping.calibration.arbitrage_free import ArbitrageFreeCalibrator

        cascader = ContractCascader()
        cascader.fit_seasonal_ratios(train)

        calibration_mode = model_cfg.get("calibration_mode", "multiplicative")
        calibrator = ArbitrageFreeCalibrator(
            smoothness_weight=model_cfg.get("smoothness_weight", 1.0),
            tol=model_cfg.get("calibration_tol", 0.01),
            regularisation=1e-6,  # higher regularisation for Base+Peak stability
            mode=calibration_mode,  # SOTA: multiplicative preserves model structure
        )

        # Assemble PFC (with calibrator + cascader)
        assembler = PFCAssembler(
            shape_hourly=sh,
            shape_intraday=si,
            uncertainty=unc,
            cascader=cascader,
            calibrator=calibrator,
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

        from scipy.stats import spearmanr

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

        # ── Shape quality metrics (Maciejowska et al. 2025) ────────────
        # Group by day for daily profile metrics
        dates_zh = idx_zh.date
        unique_dates = sorted(set(dates_zh))

        corr_f_list = []    # daily Pearson profile correlation
        spearman_list = []  # daily Spearman rank correlation
        mpd_list = []       # Min-Max Price Deviation (hours)
        mhd_list = []       # Max Hour Deviation (hours)
        cov_e_list = []     # Coefficient of Variation of daily errors

        for d in unique_dates:
            day_mask = dates_zh == d
            n_pts = day_mask.sum()
            if n_pts < 92:  # need ~full day (96 quarter-hours)
                continue

            pfc_day = pfc_prices.values[day_mask]
            spot_day = spot_prices.values[day_mask]
            err_day = errors[day_mask]

            # Aggregate to hourly for profile metrics
            n_h = n_pts // 4
            if n_h < 20:
                continue
            pfc_hourly = np.array([pfc_day[i*4:(i+1)*4].mean() for i in range(n_h)])
            spot_hourly = np.array([spot_day[i*4:(i+1)*4].mean() for i in range(n_h)])

            # Corr-f: Pearson correlation of daily profile
            if np.std(pfc_hourly) > 0 and np.std(spot_hourly) > 0:
                corr_f_list.append(float(np.corrcoef(pfc_hourly, spot_hourly)[0, 1]))

            # Spearman rank correlation
            if len(pfc_hourly) >= 3:
                rho, _ = spearmanr(pfc_hourly, spot_hourly)
                if np.isfinite(rho):
                    spearman_list.append(float(rho))

            # MPD: |argmax error| + |argmin error|
            h_max_pfc = int(np.argmax(pfc_hourly))
            h_max_spot = int(np.argmax(spot_hourly))
            h_min_pfc = int(np.argmin(pfc_hourly))
            h_min_spot = int(np.argmin(spot_hourly))
            mpd_list.append(abs(h_max_pfc - h_max_spot) + abs(h_min_pfc - h_min_spot))

            # MHD: |argmax error|
            mhd_list.append(abs(h_max_pfc - h_max_spot))

            # Cov-e: CV of intra-day errors
            err_hourly = np.array([err_day[i*4:(i+1)*4].mean() for i in range(n_h)])
            mean_abs_err = np.mean(np.abs(err_hourly))
            if mean_abs_err > 0.01:
                cov_e_list.append(float(np.std(err_hourly) / mean_abs_err))

        corr_f = float(np.median(corr_f_list)) if corr_f_list else 0.0
        spearman_rho = float(np.median(spearman_list)) if spearman_list else 0.0
        mpd = float(np.mean(mpd_list)) if mpd_list else 99.0
        mhd = float(np.mean(mhd_list)) if mhd_list else 99.0
        cov_e = float(np.median(cov_e_list)) if cov_e_list else 99.0

        # ── Peak/Offpeak spread accuracy ───────────────────────────────
        # Peak = weekday 08:00-20:00 Zurich time
        is_weekday = idx_zh.dayofweek < 5
        is_peak_hour = (idx_zh.hour >= 8) & (idx_zh.hour < 20)
        peak_mask = is_weekday & is_peak_hour
        offpeak_mask = ~peak_mask

        if peak_mask.sum() > 96 and offpeak_mask.sum() > 96:
            pfc_spread = pfc_prices.values[peak_mask].mean() - pfc_prices.values[offpeak_mask].mean()
            spot_spread = spot_prices.values[peak_mask].mean() - spot_prices.values[offpeak_mask].mean()
            spread_error = float(abs(pfc_spread - spot_spread))
            spread_ratio = float(pfc_spread / spot_spread) if abs(spot_spread) > 0.1 else 1.0
        else:
            spread_error = 99.0
            spread_ratio = 1.0

        # ── Dispatch revenue metric (simplified hydro DFL) ────────────
        # Simulate a 100 MW hydro plant dispatching 8h/day in the most
        # expensive hours according to the PFC vs the actual spot.
        # Revenue gap = how much revenue is lost due to shape errors.
        dispatch_hours = 8  # turbine 8h/day
        capacity_mw = 100
        revenue_pfc = 0.0
        revenue_perfect = 0.0
        revenue_flat = 0.0
        n_dispatch_days = 0

        for d in unique_dates:
            day_mask = dates_zh == d
            n_pts = day_mask.sum()
            if n_pts < 92:
                continue

            spot_day = spot_prices.values[day_mask]
            pfc_day = pfc_prices.values[day_mask]

            # Aggregate to hourly
            n_h = n_pts // 4
            if n_h < 20:
                continue
            spot_hourly = np.array([spot_day[i*4:(i+1)*4].mean() for i in range(n_h)])
            pfc_hourly = np.array([pfc_day[i*4:(i+1)*4].mean() for i in range(n_h)])

            # PFC-based dispatch: pick top dispatch_hours by PFC price
            pfc_top_hours = np.argsort(pfc_hourly)[-dispatch_hours:]
            # Perfect dispatch: pick top dispatch_hours by actual spot
            perfect_top_hours = np.argsort(spot_hourly)[-dispatch_hours:]
            # Flat dispatch: first dispatch_hours hours (baseline)
            flat_hours = list(range(dispatch_hours))

            revenue_pfc += spot_hourly[pfc_top_hours].sum() * capacity_mw
            revenue_perfect += spot_hourly[perfect_top_hours].sum() * capacity_mw
            revenue_flat += spot_hourly[flat_hours].sum() * capacity_mw
            n_dispatch_days += 1

        if n_dispatch_days > 0 and revenue_perfect > 0:
            # Capture ratio: how much of perfect-foresight revenue does PFC capture?
            dispatch_capture = float(revenue_pfc / revenue_perfect)
            # Improvement over flat dispatch
            dispatch_vs_flat = float((revenue_pfc - revenue_flat) / (revenue_perfect - revenue_flat)) if revenue_perfect != revenue_flat else 1.0
        else:
            dispatch_capture = 0.0
            dispatch_vs_flat = 0.0

        elapsed = time.time() - t0

        # ── Output metrics (Karpathy format) ─────────────────────────────
        print("---")
        print(f"corr_f:           {corr_f:.6f}")
        print(f"spearman_rho:     {spearman_rho:.6f}")
        print(f"mpd:              {mpd:.6f}")
        print(f"mhd:              {mhd:.6f}")
        print(f"cov_e:            {cov_e:.6f}")
        print(f"spread_error:     {spread_error:.6f}")
        print(f"spread_ratio:     {spread_ratio:.6f}")
        print(f"dispatch_capture: {dispatch_capture:.6f}")
        print(f"dispatch_vs_flat: {dispatch_vs_flat:.6f}")
        print(f"rmse:             {rmse:.6f}")
        print(f"rmse_shape:       {rmse_shape:.6f}")
        print(f"mae:              {mae:.6f}")
        print(f"bias:             {bias:.6f}")
        print(f"rmse_hourly_std:  {rmse_hourly_std:.6f}")
        print(f"ic80_coverage:    {ic80_coverage:.6f}")
        print(f"n_points:         {len(common_idx)}")
        print(f"n_days:           {len(unique_dates)}")
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

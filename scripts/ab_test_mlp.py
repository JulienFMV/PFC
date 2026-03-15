"""
A/B test: ShapeHourly (table) vs ShapeHourlyMLP (neural)
Runs the same eval pipeline with both and compares metrics.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))

import warnings
warnings.filterwarnings("ignore")


def run_eval(shape_hourly_cls, label: str, **kwargs):
    """Run evaluation with a given shape hourly implementation."""
    t0 = time.time()
    import yaml
    from dashboard.utils import load_epex
    from pfc_shaping.data.calendar_ch import enrich_15min_index
    from pfc_shaping.data.forward_proxy import derive_base_prices
    from pfc_shaping.model.assembler import PFCAssembler
    from pfc_shaping.model.shape_intraday import ShapeIntraday
    from pfc_shaping.model.uncertainty import Uncertainty
    from pfc_shaping.calibration.cascading import ContractCascader
    from pfc_shaping.calibration.arbitrage_free import ArbitrageFreeCalibrator

    epex = load_epex()
    cfg_path = ROOT / "pfc_shaping" / "config.yaml"
    with cfg_path.open() as f:
        config = yaml.safe_load(f)
    model_cfg = config.get("model", {})

    test_months = 2
    cutoff = epex.index.max() - pd.DateOffset(months=test_months)
    train = epex[epex.index < cutoff]
    test = epex[epex.index >= cutoff]

    lookback = model_cfg.get("lookback_months", 36)
    sigma = model_cfg.get("gaussian_sigma", 0.5)
    lb_cutoff = train.index.max() - pd.DateOffset(months=lookback)
    train_lb = train[train.index >= lb_cutoff]
    cal = enrich_15min_index(train_lb.index)

    entso_path = ROOT / "pfc_shaping" / "data" / "entso_15min.parquet"
    entso_df = None
    entso_full = None
    if entso_path.exists():
        entso_full = pd.read_parquet(entso_path)
        entso_df = entso_full.reindex(train_lb.index)

    hydro_path = ROOT / "pfc_shaping" / "data" / "hydro_reservoir.parquet"
    hydro_df = None
    if hydro_path.exists():
        hydro_df = pd.read_parquet(hydro_path)

    # Fit shape hourly (table or MLP)
    if "sigma" in kwargs:
        sh = shape_hourly_cls(sigma=kwargs["sigma"])
    else:
        sh = shape_hourly_cls()
    sh.fit(train_lb, cal, hydro_df=hydro_df)

    si = ShapeIntraday()
    si.fit(train_lb, entso_df=entso_df, calendar_df=cal, hydro_df=hydro_df)

    unc = Uncertainty()
    unc.fit(train_lb, cal)

    base_prices = derive_base_prices(train, start_year=cutoff.year, n_years=1)

    cascader = ContractCascader()
    cascader.fit_seasonal_ratios(train)

    calibrator = ArbitrageFreeCalibrator(
        smoothness_weight=model_cfg.get("smoothness_weight", 1.0),
        tol=model_cfg.get("calibration_tol", 0.01),
        regularisation=1e-6,
        mode=model_cfg.get("calibration_mode", "multiplicative"),
    )

    assembler = PFCAssembler(
        shape_hourly=sh,
        shape_intraday=si,
        uncertainty=unc,
        cascader=cascader,
        calibrator=calibrator,
    )

    entso_forecast = None
    if entso_full is not None:
        entso_forecast = entso_full[entso_full.index >= cutoff]

    pfc = assembler.build(
        base_prices=base_prices,
        start_date=cutoff.strftime("%Y-%m-%d"),
        horizon_days=test_months * 31,
        entso_forecast=entso_forecast,
    )

    # Evaluate
    common_idx = pfc.index.intersection(test.index)
    pfc_prices = pfc.loc[common_idx, "price_shape"]
    spot_prices = test.loc[common_idx, "price_eur_mwh"]

    errors = pfc_prices.values - spot_prices.values
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))

    scale = spot_prices.mean() / pfc_prices.mean() if pfc_prices.mean() != 0 else 1.0
    shape_errors = pfc_prices.values * scale - spot_prices.values
    rmse_shape = float(np.sqrt(np.mean(shape_errors**2)))

    # Profile correlation
    from scipy.stats import spearmanr
    idx_zh = common_idx.tz_convert("Europe/Zurich")
    dates_zh = idx_zh.date
    unique_dates = sorted(set(dates_zh))
    corr_list, spearman_list = [], []
    for d in unique_dates:
        day_mask = dates_zh == d
        n_pts = day_mask.sum()
        if n_pts < 92:
            continue
        n_h = n_pts // 4
        if n_h < 20:
            continue
        pfc_day = pfc_prices.values[day_mask]
        spot_day = spot_prices.values[day_mask]
        pfc_h = np.array([pfc_day[i*4:(i+1)*4].mean() for i in range(n_h)])
        spot_h = np.array([spot_day[i*4:(i+1)*4].mean() for i in range(n_h)])
        if np.std(pfc_h) > 0 and np.std(spot_h) > 0:
            corr_list.append(float(np.corrcoef(pfc_h, spot_h)[0, 1]))
        rho, _ = spearmanr(pfc_h, spot_h)
        if np.isfinite(rho):
            spearman_list.append(float(rho))

    elapsed = time.time() - t0

    return {
        "label": label,
        "rmse": rmse,
        "rmse_shape": rmse_shape,
        "mae": mae,
        "bias": bias,
        "corr_f": float(np.median(corr_list)) if corr_list else 0.0,
        "spearman": float(np.median(spearman_list)) if spearman_list else 0.0,
        "elapsed": elapsed,
    }


def main():
    from pfc_shaping.model.shape_hourly import ShapeHourly
    from pfc_shaping.model.shape_hourly_mlp import ShapeHourlyMLP

    print("=" * 60)
    print("A/B TEST: ShapeHourly (table) vs ShapeHourlyMLP (neural)")
    print("=" * 60)

    print("\n[A] Running TABLE-based ShapeHourly...")
    a = run_eval(ShapeHourly, "TABLE", sigma=0.5)

    print("\n[B] Running MLP-based ShapeHourlyMLP...")
    b = run_eval(ShapeHourlyMLP, "MLP")

    print("\n" + "=" * 60)
    print(f"{'Metric':<20} {'TABLE':>12} {'MLP':>12} {'Delta':>12}")
    print("-" * 60)
    for key in ["rmse", "rmse_shape", "mae", "bias", "corr_f", "spearman"]:
        va, vb = a[key], b[key]
        delta = vb - va
        better = "+" if (delta < 0 and key != "corr_f" and key != "spearman") or \
                        (delta > 0 and key in ("corr_f", "spearman")) else " "
        if key in ("corr_f", "spearman"):
            better = "+" if delta > 0 else " "
        print(f"{key:<20} {va:>12.4f} {vb:>12.4f} {delta:>+12.4f} {better}")
    print("-" * 60)
    print(f"{'Time (s)':<20} {a['elapsed']:>12.1f} {b['elapsed']:>12.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

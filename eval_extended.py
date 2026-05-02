"""
eval_extended.py — Extended PFC evaluation harness (Phase 0.3 of 2026 roadmap)
=============================================================================
Companion to ``autoresearch_eval.py`` (which must remain untouched per the
autoresearch program). Adds:

    * multi-window OOS backtests (winter 2026Q1, summer 2024Q3, long-horizon
      Y+1 covering all of 2025) so structural changes that only manifest in
      certain regimes (e.g. additive mode for negative prices in summer) are
      actually visible;
    * stratified RMSE/MAE/bias by:
        - horizon bucket  (DA J+1..J+7, M+1, M+2..M+3, Y+1+)
        - peak vs off-peak (EEX definition, weekday 08-20 Zurich)
        - low-price regime (|spot| < 5 EUR/MWh)
        - season (Hiver/Printemps/Ete/Automne)
        - hour bucket (night 0-6, morning ramp 6-9, midday 10-15, evening 16-21, late 22-23)

Usage:
    python3 eval_extended.py [--window {2026q1,2024q3,longhorizon,all}]

Output:
    JSON dump on stdout, plus a markdown summary table on stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))


@dataclass
class Window:
    """OOS backtest window definition."""

    name: str
    cutoff: pd.Timestamp        # Train ends just before this. PFC starts here.
    horizon_days: int           # PFC horizon (and OOS depth)
    description: str = ""


@dataclass
class StratifiedMetrics:
    """Aggregated metrics over a window, with per-stratum breakdown."""

    window: str
    n_points: int
    rmse: float
    mae: float
    bias: float
    rmse_shape: float
    ic80_coverage: float | None
    by_stratum: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Stratifier helpers
# --------------------------------------------------------------------------

SEASONS_BY_MONTH = {
    12: "Hiver", 1: "Hiver", 2: "Hiver",
    3: "Printemps", 4: "Printemps", 5: "Printemps",
    6: "Ete", 7: "Ete", 8: "Ete",
    9: "Automne", 10: "Automne", 11: "Automne",
}


def _hour_bucket(h: int) -> str:
    if h < 6: return "night_00_05"
    if h < 10: return "morning_06_09"
    if h < 16: return "midday_10_15"
    if h < 22: return "evening_16_21"
    return "late_22_23"


def _horizon_bucket(days_ahead: float) -> str:
    if days_ahead <= 7: return "DA_0_7d"
    if days_ahead <= 30: return "M+1"
    if days_ahead <= 90: return "M+2_M+3"
    return "Y+1+"


def _fine_horizon_bucket(d: float) -> str:
    if d <= 1: return "D+0_1"
    if d <= 3: return "D+2_3"
    if d <= 5: return "D+4_5"
    if d <= 10: return "D+6_10"
    if d <= 30: return "D+11_30"
    if d <= 90: return "D+31_90"
    return "D+91+"


def _build_strata(idx: pd.DatetimeIndex, cutoff: pd.Timestamp,
                  spot: np.ndarray) -> dict[str, np.ndarray]:
    idx_zh = idx.tz_convert("Europe/Zurich")
    days_ahead = (idx - cutoff).total_seconds() / 86400.0
    is_weekday = idx_zh.dayofweek < 5
    is_peak_hour = (idx_zh.hour >= 8) & (idx_zh.hour < 20)
    return {
        "peak_offpeak": np.where(is_weekday & is_peak_hour, "peak", "offpeak"),
        "season": np.array([SEASONS_BY_MONTH[m] for m in idx_zh.month]),
        "horizon": np.array([_horizon_bucket(d) for d in days_ahead]),
        "horizon_fine": np.array([_fine_horizon_bucket(d) for d in days_ahead]),
        "hour_bucket": np.array([_hour_bucket(h) for h in idx_zh.hour]),
        "low_price": np.where(np.abs(spot) < 5.0, "low_abs_lt_5", "normal"),
    }


def _agg(errors: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    sel = errors[mask]
    if len(sel) == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"), "bias": float("nan")}
    return {
        "n": int(len(sel)),
        "rmse": float(np.sqrt(np.mean(sel ** 2))),
        "mae": float(np.mean(np.abs(sel))),
        "bias": float(np.mean(sel)),
    }


# --------------------------------------------------------------------------
# Pipeline (mirrors autoresearch_eval.py minus the metric block)
# --------------------------------------------------------------------------

def _build_pfc(
    epex: pd.DataFrame,
    window: Window,
    with_lear: bool = False,
    blend_start_day: int = 8,
    blend_end_day: int = 11,
):
    """Fit shape models on data < cutoff and build a PFC over the window.

    If ``with_lear=True``, additionally fit ``LEARForecaster`` on the same
    train window and overlay its D+1..D+10 forecast onto the PFC via
    ``LEARForecaster.blend_with_pfc``.
    """
    import yaml
    from pfc_shaping.calibration.arbitrage_free import ArbitrageFreeCalibrator
    from pfc_shaping.calibration.cascading import ContractCascader
    from pfc_shaping.data.calendar_ch import enrich_15min_index
    from pfc_shaping.data.forward_proxy import derive_base_prices
    from pfc_shaping.shaping import PFCAssembler, ShapeIntraday, Uncertainty

    cfg_path = ROOT / "pfc_shaping" / "config.yaml"
    with cfg_path.open() as f:
        config = yaml.safe_load(f)
    model_cfg = config.get("model", {})
    lookback = model_cfg.get("lookback_months", 36)
    sigma = model_cfg.get("gaussian_sigma", 0.5)
    sh_mode = model_cfg.get("shape_hourly_mode", "table")

    train = epex[epex.index < window.cutoff]
    if len(train) < 96 * 365:
        raise RuntimeError(f"insufficient train data for window {window.name}: {len(train)} rows")

    lb_cutoff = train.index.max() - pd.DateOffset(months=lookback)
    train_lb = train[train.index >= lb_cutoff]
    cal = enrich_15min_index(train_lb.index)

    entso_path = ROOT / "pfc_shaping" / "data" / "entso_15min.parquet"
    entso_full = None
    entso_df = None
    if entso_path.exists():
        entso_full = pd.read_parquet(entso_path)
        entso_df = entso_full.reindex(train_lb.index)

    hydro_path = ROOT / "pfc_shaping" / "data" / "hydro_reservoir.parquet"
    hydro_df = pd.read_parquet(hydro_path) if hydro_path.exists() else None

    if sh_mode == "mlp":
        from pfc_shaping.shaping.shape_hourly_mlp import ShapeHourlyMLP
        sh = ShapeHourlyMLP()
    else:
        from pfc_shaping.shaping import ShapeHourly
        sh = ShapeHourly(sigma=sigma)
    sh.fit(train_lb, cal, hydro_df=hydro_df)

    si = ShapeIntraday()
    si.fit(train_lb, entso_df=entso_df, calendar_df=cal, hydro_df=hydro_df)

    unc = Uncertainty()
    unc.fit(train_lb, cal)

    # Phase 4.1: load commodities (TTF gas + EUA) and pass them to
    # derive_base_prices so the anchor level is blended with a
    # fundamental regression instead of pure trailing-spot mean.
    commodities_df = None
    for cpath in (
        ROOT / "data" / "commodities_cache.parquet",
        ROOT / "pfc_shaping" / "data" / "commodities_cache.parquet",
    ):
        if cpath.exists():
            cm = pd.read_parquet(cpath)
            if cm.index.tz is None:
                cm.index = cm.index.tz_localize("UTC")
            commodities_df = cm[cm.index < window.cutoff]
            break

    fundamental_blend = float(getattr(window, "_fundamental_blend", 0.4) or 0.4)
    anchor_months = int(getattr(window, "_anchor_months", 6) or 6)
    base_prices = derive_base_prices(
        train,
        start_year=window.cutoff.year,
        n_years=max(1, window.horizon_days // 365 + 1),
        anchor_months=anchor_months,
        commodities=commodities_df,
        fundamental_blend=fundamental_blend,
    )

    cascader = ContractCascader()
    cascader.fit_seasonal_ratios(train)
    calibrator = ArbitrageFreeCalibrator(
        smoothness_weight=model_cfg.get("smoothness_weight", 1.0),
        tol=model_cfg.get("calibration_tol", 0.01),
        regularisation=1e-6,
        mode=model_cfg.get("calibration_mode", "multiplicative"),
    )

    assembler = PFCAssembler(
        shape_hourly=sh, shape_intraday=si, uncertainty=unc,
        cascader=cascader, calibrator=calibrator,
    )

    # Climatological ENTSO-E forecast (no look-ahead)
    entso_forecast = None
    if entso_df is not None and entso_full is not None:
        hist = entso_full[entso_full.index < window.cutoff]
        if not hist.empty:
            idx_zh = hist.index.tz_convert("Europe/Zurich")
            hist_c = hist.copy()
            hist_c["_month"] = idx_zh.month
            hist_c["_hour"] = idx_zh.hour
            hist_c["_quarter"] = idx_zh.minute // 15
            numeric_cols = [
                c for c in hist_c.select_dtypes("number").columns
                if c not in ("_month", "_hour", "_quarter")
            ]
            clim = hist_c.groupby(["_month", "_hour", "_quarter"])[numeric_cols].median()
            fwd_idx = pd.date_range(
                window.cutoff,
                window.cutoff + pd.Timedelta(days=max(int(window.horizon_days), 31)),
                freq="15min", tz="UTC", inclusive="left",
            )
            fwd_zh = fwd_idx.tz_convert("Europe/Zurich")
            keys = list(zip(fwd_zh.month, fwd_zh.hour, fwd_zh.minute // 15))
            if numeric_cols and not clim.empty:
                rows = []
                empty = pd.Series(0.0, index=numeric_cols)
                for k in keys:
                    if k in clim.index:
                        rows.append(clim.loc[k])
                    else:
                        rows.append(empty)
                entso_forecast = pd.DataFrame(rows, index=fwd_idx).fillna(0.0)
            else:
                entso_forecast = None

    # Energy-consistency fix: monthly Peak forwards (e.g. 2026-01-Peak)
    # require a full-month PFC to be averageable at the right level. For
    # short evaluation windows (e.g. lear_only_window=5), build the PFC
    # over at least 31 days, then restrict the test mask to the requested
    # horizon. Without this, peak hours are over-predicted by 10-15 EUR
    # because the calibrator cannot reconcile a 5-day mean with a monthly
    # forward target.
    build_horizon = max(int(window.horizon_days), 31)
    pfc = assembler.build(
        base_prices=base_prices,
        start_date=window.cutoff.strftime("%Y-%m-%d"),
        horizon_days=build_horizon,
        entso_forecast=entso_forecast,
        reference_date=window.cutoff,
    )
    if build_horizon > window.horizon_days:
        eval_end = window.cutoff + pd.Timedelta(days=window.horizon_days)
        pfc = pfc[pfc.index < eval_end]

    if with_lear:
        try:
            from pfc_shaping.forecasting import LEARForecaster

            epex_de_path = ROOT / "pfc_shaping" / "data" / "epex_de_15min.parquet"
            epex_de = (
                pd.read_parquet(epex_de_path).sort_index() if epex_de_path.exists() else None
            )
            if epex_de is not None:
                epex_de = epex_de[epex_de.index < window.cutoff]

            lear = LEARForecaster(tz="Europe/Zurich", use_foundation_model=False)
            lear.fit(
                epex_15min=train,
                entso_15min=entso_full[entso_full.index < window.cutoff] if entso_full is not None else None,
                hydro=hydro_df,
                epex_de_15min=epex_de,
            )
            lear_horizon = min(14, window.horizon_days)
            lear_forecast = lear.predict(horizon_days=lear_horizon)
            pfc = lear.blend_with_pfc(
                pfc, lear_forecast,
                blend_start_day=blend_start_day,
                blend_end_day=blend_end_day,
            )
            print(
                f"[lear] {window.name}: fitted + blended over {lear_horizon}d "
                f"(forecast mean={lear_forecast['price_lear'].mean():.1f} EUR/MWh)",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[lear] {window.name}: failed ({exc}) — PFC-only", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    return pfc


def _apply_persistence_overlay(
    pfc: pd.DataFrame,
    epex: pd.DataFrame,
    cutoff: pd.Timestamp,
    days: int = 2,
    weight_d1: float = 0.55,
) -> pd.DataFrame:
    """Blend a persistence (last observed day's hourly pattern) into the PFC
    on D+1..D+`days`, with weight decaying linearly from ``weight_d1`` to 0
    at day ``days+1``. This compensates for the structural PFC's lack of
    lag-1 awareness: realized day D-1 mean is the strongest single
    predictor of D+1 in EPF literature (Lago 2021)."""
    if days <= 0 or weight_d1 <= 0:
        return pfc

    # Last observed day's hourly pattern (Zurich-time hour-of-day → price)
    last_end = cutoff.tz_convert(epex.index.tz)
    last_start = last_end - pd.Timedelta(days=1)
    last_day = epex[(epex.index >= last_start) & (epex.index < last_end)][
        "price_eur_mwh"
    ].copy()
    if len(last_day) < 80:  # need ~24h
        return pfc
    idx_zh = last_day.index.tz_convert("Europe/Zurich")
    pattern_by_hour = last_day.groupby(idx_zh.hour).mean()
    last_day_mean = float(pattern_by_hour.mean())

    out = pfc.copy()
    pfc_zh = pfc.index.tz_convert("Europe/Zurich")
    days_ahead = (pfc.index - cutoff).total_seconds() / 86400.0
    hours_zh = pfc_zh.hour
    pred = pfc["price_shape"].values.copy()

    for i in range(len(pfc)):
        d = days_ahead[i]
        if d < 0 or d > days:
            continue
        # weight decays linearly: d=0 → weight_d1, d=days → 0
        w = float(np.clip(weight_d1 * (1.0 - d / float(days)), 0.0, 1.0))
        if w <= 0:
            continue
        pattern_value = float(pattern_by_hour.get(hours_zh[i], last_day_mean))
        pred[i] = (1.0 - w) * pred[i] + w * pattern_value

    out["price_shape"] = pred
    return out


def _evaluate_window(
    epex: pd.DataFrame,
    window: Window,
    with_lear: bool = False,
    blend_start_day: int = 8,
    blend_end_day: int = 11,
    conformal_calib_days: int = 0,
    persistence_days: int = 2,
    persistence_weight: float = 0.55,
) -> StratifiedMetrics | None:
    pfc = _build_pfc(
        epex, window,
        with_lear=with_lear,
        blend_start_day=blend_start_day,
        blend_end_day=blend_end_day,
    )
    if persistence_days > 0:
        pfc = _apply_persistence_overlay(
            pfc, epex, window.cutoff,
            days=persistence_days, weight_d1=persistence_weight,
        )
    test = epex[epex.index >= window.cutoff]
    common = pfc.index.intersection(test.index)
    if len(common) < 96 * 3:
        print(f"[skip] {window.name}: insufficient overlap ({len(common)} pts)", file=sys.stderr)
        return None

    # Phase 5.1: split-conformal recalibration of the bootstrap intervals.
    # Use the first ``conformal_calib_days`` days of the test window to fit a
    # scalar correction so IC80 coverage hits the nominal 80% on that
    # holdout set; then drop those days and evaluate the metrics on the
    # remainder. ``conformal_calib_days=0`` skips recalibration entirely.
    #
    # Practical guardrails:
    #   * The calibration set must keep ≥ 30 days of test for evaluation
    #     (otherwise the metrics have too much variance).
    #   * If the calibration window is too short (<14 days) split-conformal
    #     fails because exchangeability breaks (early-window volatility
    #     not representative of the full test). For winter_2026q1 (60d test)
    #     this means recalibration is skipped; for longhorizon_2025 (365d
    #     test) the recommended setting is calib_days=60 which lands IC80
    #     close to the 0.80 nominal target.
    if conformal_calib_days > 0 and "p10" in pfc.columns and "p90" in pfc.columns:
        from pfc_shaping.shaping import Uncertainty
        calib_end = window.cutoff + pd.Timedelta(days=conformal_calib_days)
        calib_mask = (common < calib_end)
        test_mask = (common >= calib_end)
        n_calib = int(calib_mask.sum())
        n_test_remainder = int(test_mask.sum())
        if n_calib >= 96 * 14 and n_test_remainder >= 96 * 30:
            calib_idx = common[calib_mask]
            unc_local = Uncertainty()
            stats = unc_local.recalibrate(
                realized=test.loc[calib_idx, "price_eur_mwh"].values,
                prices=pfc.loc[calib_idx, "price_shape"].values,
                p10=pfc.loc[calib_idx, "p10"].values,
                p90=pfc.loc[calib_idx, "p90"].values,
            )
            half_lo = pfc.loc[common, "price_shape"].values - pfc.loc[common, "p10"].values
            half_hi = pfc.loc[common, "p90"].values - pfc.loc[common, "price_shape"].values
            scale = stats["scale"]
            pfc.loc[common, "p10"] = pfc.loc[common, "price_shape"].values - half_lo * scale
            pfc.loc[common, "p90"] = pfc.loc[common, "price_shape"].values + half_hi * scale
            common = common[test_mask]
            print(
                f"[conformal] {window.name}: calib_days={conformal_calib_days}, "
                f"scale={scale:.3f}, calib_cov={stats['after']:.3f}",
                file=sys.stderr,
            )
        else:
            print(
                f"[conformal] {window.name}: SKIPPED (calib={n_calib}, "
                f"remainder={n_test_remainder}). Need calib >= 1344 (14d) "
                f"AND remainder >= 2880 (30d).",
                file=sys.stderr,
            )

    pfc_p = pfc.loc[common, "price_shape"].values
    spot_p = test.loc[common, "price_eur_mwh"].values
    err = pfc_p - spot_p

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    scale = float(spot_p.mean() / pfc_p.mean()) if pfc_p.mean() != 0 else 1.0
    rmse_shape = float(np.sqrt(np.mean((pfc_p * scale - spot_p) ** 2)))

    ic80 = None
    if "p10" in pfc.columns and "p90" in pfc.columns:
        p10 = pfc.loc[common, "p10"].values
        p90 = pfc.loc[common, "p90"].values
        ic80 = float(((spot_p >= p10) & (spot_p <= p90)).mean())

    strata = _build_strata(common, window.cutoff, spot_p)
    by_stratum: dict[str, dict[str, dict[str, float]]] = {}
    for strat_name, labels in strata.items():
        per_label: dict[str, dict[str, float]] = {}
        for label in sorted(set(labels.tolist())):
            mask = labels == label
            per_label[str(label)] = _agg(err, mask)
        by_stratum[strat_name] = per_label

    return StratifiedMetrics(
        window=window.name, n_points=len(common),
        rmse=rmse, mae=mae, bias=bias, rmse_shape=rmse_shape,
        ic80_coverage=ic80, by_stratum=by_stratum,
    )


# --------------------------------------------------------------------------
# Window definitions
# --------------------------------------------------------------------------

def _windows(epex: pd.DataFrame, selection: str) -> list[Window]:
    """Build OOS windows that fit inside available EPEX history."""
    last = epex.index.max()
    out: list[Window] = []
    if selection in ("2026q1", "all"):
        out.append(Window(
            name="winter_2026q1",
            cutoff=last.normalize() - pd.DateOffset(months=2),
            horizon_days=62,
            description="Default autoresearch window (last 2 months, winter)",
        ))
    if selection in ("2024q3", "all"):
        # Cutoff at end of Q2 2024, test summer Q3 2024 (PV regime, neg-price exposure)
        out.append(Window(
            name="summer_2024q3",
            cutoff=pd.Timestamp("2024-07-01", tz="UTC"),
            horizon_days=92,
            description="Summer 2024 (PV regime, low/neg-price stress)",
        ))
    if selection in ("longhorizon", "all"):
        # Cutoff at end of 2024, test all of 2025 (Y+1 horizon)
        out.append(Window(
            name="longhorizon_2025",
            cutoff=pd.Timestamp("2025-01-01", tz="UTC"),
            horizon_days=365,
            description="Y+1 horizon, full year 2025",
        ))
    if not out:
        raise SystemExit(f"no windows selected (selection={selection!r})")
    # Filter windows whose test range goes beyond available data (keep partial)
    return [w for w in out if w.cutoff < last]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _md_table(results: list[StratifiedMetrics]) -> str:
    lines = []
    lines.append("| window | n_pts | RMSE | MAE | bias | RMSE_shape | IC80 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        ic = f"{r.ic80_coverage:.3f}" if r.ic80_coverage is not None else "n/a"
        lines.append(
            f"| {r.window} | {r.n_points} | {r.rmse:.3f} | {r.mae:.3f} | "
            f"{r.bias:+.3f} | {r.rmse_shape:.3f} | {ic} |"
        )
    return "\n".join(lines)


def _md_strata(r: StratifiedMetrics) -> str:
    out = [f"\n### {r.window} — by stratum"]
    for strat, per_label in r.by_stratum.items():
        out.append(f"\n**{strat}**")
        out.append("| label | n | RMSE | MAE | bias |")
        out.append("|---|---:|---:|---:|---:|")
        for label, m in per_label.items():
            if m["n"] == 0:
                continue
            out.append(
                f"| {label} | {m['n']} | {m['rmse']:.3f} | {m['mae']:.3f} | {m['bias']:+.3f} |"
            )
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--window", default="all",
        choices=["2026q1", "2024q3", "longhorizon", "all"],
        help="Which OOS window(s) to evaluate",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit JSON only on stdout (default: also write markdown summary to stderr)",
    )
    parser.add_argument(
        "--with-lear", action="store_true",
        help="Overlay LEAR D+1..D+10 forecast onto the PFC (Phase 2)",
    )
    parser.add_argument(
        "--lear-only-window", type=int, default=None, metavar="DAYS",
        help="Restrict evaluation to the first N days of each test window "
             "(useful with --with-lear to focus on the LEAR-blended segment).",
    )
    parser.add_argument(
        "--blend-start", type=int, default=5, metavar="DAY",
        help="Day at which LEAR-PFC blend begins (default: 5, post CT.1 sweep). "
             "LEAR pure before, linear blend in [start, end), PFC pure from end.",
    )
    parser.add_argument(
        "--blend-end", type=int, default=10, metavar="DAY",
        help="Day at which LEAR-PFC blend ends (default: 10, post CT.1 sweep).",
    )
    parser.add_argument(
        "--conformal-calib-days", type=int, default=0, metavar="DAYS",
        help="Phase 5.1: split-conformal recalibration. Use the first N days "
             "of the test window to fit a scalar correction on bootstrap "
             "intervals so empirical IC80 hits the nominal 80%%; evaluate on "
             "the remainder. 0 (default) disables recalibration.",
    )
    parser.add_argument(
        "--fundamental-blend", type=float, default=0.4, metavar="W",
        help="Weight of the gas+EUA fundamental anchor in derive_base_prices "
             "(0.0 = pure trailing spot, 1.0 = pure fundamentals). Default 0.4.",
    )
    parser.add_argument(
        "--anchor-months", type=int, default=6, metavar="M",
        help="Trailing months for the spot anchor in derive_base_prices "
             "(shorter = more responsive to recent regime; longer = smoother).",
    )
    parser.add_argument(
        "--persistence-days", type=int, default=2, metavar="D",
        help="Apply a persistence-overlay on the first D test days (default 2). "
             "Anchors near-term predictions on the last observed day's hourly "
             "pattern. Set 0 to disable.",
    )
    parser.add_argument(
        "--persistence-weight", type=float, default=0.55, metavar="W",
        help="Weight of the persistence overlay at D+1 (decays linearly to 0 "
             "at D+persistence_days+1). Default 0.55.",
    )
    args = parser.parse_args()

    from dashboard.utils import load_epex
    epex = load_epex()
    if epex is None or len(epex) < 96 * 365:
        print(json.dumps({"status": "crash", "reason": "insufficient_epex"}))
        sys.exit(1)

    t0 = time.time()
    windows = _windows(epex, args.window)
    results: list[StratifiedMetrics] = []
    for w in windows:
        if args.lear_only_window:
            w = Window(
                name=f"{w.name}_first{args.lear_only_window}d",
                cutoff=w.cutoff,
                horizon_days=args.lear_only_window,
                description=w.description,
            )
        # Pass tuneables through the Window dataclass via dynamic attrs
        object.__setattr__(w, "_fundamental_blend", args.fundamental_blend)
        object.__setattr__(w, "_anchor_months", args.anchor_months)
        print(f"[run] {w.name} cutoff={w.cutoff.date()} h={w.horizon_days}d "
              f"lear={args.with_lear} blend=[{args.blend_start},{args.blend_end})",
              file=sys.stderr)
        try:
            r = _evaluate_window(
                epex, w,
                with_lear=args.with_lear,
                blend_start_day=args.blend_start,
                blend_end_day=args.blend_end,
                conformal_calib_days=args.conformal_calib_days,
                persistence_days=args.persistence_days,
                persistence_weight=args.persistence_weight,
            )
        except Exception as exc:
            print(f"[fail] {w.name}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            continue
        if r is not None:
            results.append(r)

    elapsed = time.time() - t0
    payload = {
        "status": "ok" if results else "no_results",
        "elapsed_seconds": round(elapsed, 1),
        "results": [
            {
                "window": r.window, "n_points": r.n_points,
                "rmse": r.rmse, "mae": r.mae, "bias": r.bias,
                "rmse_shape": r.rmse_shape, "ic80_coverage": r.ic80_coverage,
                "by_stratum": r.by_stratum,
            }
            for r in results
        ],
    }
    print(json.dumps(payload, indent=2, default=float))

    if not args.json and results:
        print("\n## Extended evaluation summary\n", file=sys.stderr)
        print(_md_table(results), file=sys.stderr)
        for r in results:
            print(_md_strata(r), file=sys.stderr)
        print(f"\n_eval_seconds: {elapsed:.1f}_", file=sys.stderr)


if __name__ == "__main__":
    main()

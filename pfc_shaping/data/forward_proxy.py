"""
forward_proxy.py
----------------
Derive forward price estimates from spot history when live EEX forwards
are not available (e.g., no XLSX report or Databricks connection).

Method:
    1. Compute seasonal shape from recent 2-year EPEX spot history
       (monthly ratio vs global mean).
    2. Estimate Cal-year forward levels using:
       - Spot-based anchor: trailing 6-month average (more stable than 3m)
       - Term-structure decay: estimated from historical year-over-year
         price changes (mean reversion).
    3. Build full monthly, quarterly, and annual base prices.

This is a PROXY only. In production, base_prices should come from
live EEX forward quotes (XLSX desk report or API).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_monthly_seasonal_ratios(
    epex: pd.DataFrame,
    n_years: int = 3,
    weekend_pv_dummy: bool = True,
) -> dict[int, float]:
    """Rolling N-year monthly seasonality ratios from EPEX history.

    Returns a dict {month -> ratio} normalised to mean 1.0 over the 12
    months. Falls back gracefully on insufficient data.

    With ``weekend_pv_dummy=True`` (default), the regression is run separately
    on weekday and weekend observations and the two patterns are equal-weighted
    before normalisation. This dampens the peak/off-peak inversion seen on
    sunny CH weekends in 2024+ that distorts a naive monthly mean.
    """
    if "price_eur_mwh" not in epex.columns or len(epex) == 0:
        return dict(_FALLBACK_SEASONAL_RATIOS)

    cutoff = epex.index.max() - pd.DateOffset(years=n_years)
    recent = epex[epex.index >= cutoff]
    if len(recent) < 96 * 180:
        recent = epex

    idx_zh = recent.index.tz_convert("Europe/Zurich")
    is_weekend = idx_zh.dayofweek >= 5

    if weekend_pv_dummy:
        weekday_avg = recent[~is_weekend].groupby(idx_zh[~is_weekend].month)["price_eur_mwh"].mean()
        weekend_avg = recent[is_weekend].groupby(idx_zh[is_weekend].month)["price_eur_mwh"].mean()
        wd_ratio = weekday_avg / weekday_avg.mean() if weekday_avg.mean() > 0 else None
        we_ratio = weekend_avg / weekend_avg.mean() if weekend_avg.mean() > 0 else None
        if wd_ratio is None or we_ratio is None:
            monthly = recent.groupby(idx_zh.month)["price_eur_mwh"].mean()
            ratios = monthly / monthly.mean() if monthly.mean() > 0 else None
        else:
            combined = (wd_ratio.add(we_ratio, fill_value=wd_ratio.mean())) / 2.0
            ratios = combined / combined.mean()
    else:
        monthly = recent.groupby(idx_zh.month)["price_eur_mwh"].mean()
        ratios = monthly / monthly.mean() if monthly.mean() > 0 else None

    if ratios is None:
        return dict(_FALLBACK_SEASONAL_RATIOS)

    out = {int(m): float(v) for m, v in ratios.items()}
    # Sanity-clip and fill gaps from fallback table.
    for m in range(1, 13):
        v = out.get(m)
        if v is None or not (0.3 <= v <= 3.0):
            out[m] = _FALLBACK_SEASONAL_RATIOS[m]
    return out


_FALLBACK_SEASONAL_RATIOS = {
    1: 1.18, 2: 1.12, 3: 1.02, 4: 0.90, 5: 0.85, 6: 0.88,
    7: 0.90, 8: 0.92, 9: 0.95, 10: 1.02, 11: 1.10, 12: 1.16,
}


def _fit_fundamental_anchor(
    epex: pd.DataFrame,
    commodities: pd.DataFrame,
) -> tuple[float, dict] | None:
    """
    Phase 4.1: regress monthly spot mean on monthly TTF gas + EUA means and
    return the model's prediction for the most recent commodities snapshot.

    This produces an anchor level driven by **fundamentals** (gas + carbon)
    rather than by the rolling spot trail, which matters when:
      * spot is in a transient deviation from fuel-cost equilibrium;
      * the forecast horizon enters a regime (e.g. winter) different from
        the trailing window.

    Returns ``(anchor_level, info)`` or ``None`` if data is insufficient.

    References:
      * Hirsch & Ziel (2024), Energy Journal — gas/CO2 drivers explain >60%
        of the level variance on EU power 2022-2024.
      * Maciejowska, Nitka, Weron (2025), RSER.
    """
    if commodities is None or commodities.empty:
        return None

    cmap = {}
    for col in commodities.columns:
        c = col.lower()
        if "ttf" in c or "gas" in c:
            cmap["gas"] = col
        elif "eua" in c or "co2" in c or "carbon" in c:
            cmap["eua"] = col
    if "gas" not in cmap or "eua" not in cmap:
        return None

    cm = commodities[[cmap["gas"], cmap["eua"]]].rename(
        columns={cmap["gas"]: "gas", cmap["eua"]: "eua"}
    ).dropna()
    if len(cm) < 90:
        return None
    if cm.index.tz is None:
        cm.index = cm.index.tz_localize("UTC")

    # Monthly means: align spot and commodities on YYYY-MM
    spot_monthly = (
        epex["price_eur_mwh"].resample("MS").mean().rename("spot")
    )
    cm_monthly = cm.resample("MS").mean()
    df = spot_monthly.to_frame().join(cm_monthly, how="inner").dropna()
    # 9 months minimum: tight enough to keep OLS reasonable, loose enough to
    # activate on backtests where commodities history is shallow (Y+1 cutoff
    # in early 2025 for example).
    if len(df) < 9:
        return None

    X = df[["gas", "eua"]].values
    y = df["spot"].values
    # Solve OLS with intercept via lstsq (no sklearn dep required).
    Xa = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    a, b_gas, b_eua = float(coef[0]), float(coef[1]), float(coef[2])
    y_hat = a + b_gas * X[:, 0] + b_eua * X[:, 1]
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Predict using most recent gas + EUA snapshot (last available month).
    gas_now = float(cm_monthly["gas"].iloc[-1])
    eua_now = float(cm_monthly["eua"].iloc[-1])
    anchor = a + b_gas * gas_now + b_eua * eua_now

    info = {
        "intercept": a, "b_gas": b_gas, "b_eua": b_eua,
        "r2": r2, "n_months": int(len(df)),
        "gas_snapshot": gas_now, "eua_snapshot": eua_now,
    }
    logger.info(
        "Fundamental anchor: spot ~ %.2f + %.3f*gas + %.3f*EUA (R2=%.3f, "
        "n=%d months) → %.1f EUR/MWh",
        a, b_gas, b_eua, r2, info["n_months"], anchor,
    )
    return float(anchor), info


def derive_base_prices(
    epex: pd.DataFrame,
    start_year: int | None = None,
    n_years: int = 4,
    anchor_months: int = 6,
    commodities: pd.DataFrame | None = None,
    fundamental_blend: float = 0.4,
) -> dict[str, float]:
    """Derive base prices from EPEX spot history.

    Args:
        epex: DataFrame with 'price_eur_mwh' column, DatetimeIndex UTC.
        start_year: First delivery year. Default: current year.
        n_years: Number of forward years to generate.
        anchor_months: Months of trailing spot to anchor level.
        commodities: Optional daily commodities frame with TTF gas + EUA
            columns (auto-detected by name). When provided and the
            regression is well-behaved (R^2 >= 0.4), the anchor level is
            blended with the fundamental prediction.
        fundamental_blend: Weight on the fundamental anchor when blending.

    Returns:
        Dict with keys like '2026', '2026-Q1', '2026-01' etc.
    """
    if start_year is None:
        start_year = pd.Timestamp.now().year

    idx_zh = epex.index.tz_convert("Europe/Zurich")

    # ── 1. Trailing anchor level ──────────────────────────────────────
    n_rows = min(96 * 30 * anchor_months, len(epex))
    trailing_anchor = float(epex["price_eur_mwh"].iloc[-n_rows:].mean())

    # ── 1b. Optional fundamental anchor (Phase 4.1) ───────────────────
    fundamental = _fit_fundamental_anchor(epex, commodities) if commodities is not None else None
    if fundamental is not None and fundamental[1]["r2"] >= 0.40:
        f_anchor, info = fundamental
        w = float(np.clip(fundamental_blend, 0.0, 1.0))
        anchor = w * f_anchor + (1.0 - w) * trailing_anchor
        logger.info(
            "Forward proxy anchor: trailing=%.1f, fundamental=%.1f (R2=%.2f), "
            "blended (w=%.2f)=%.1f EUR/MWh",
            trailing_anchor, f_anchor, info["r2"], w, anchor,
        )
    else:
        anchor = trailing_anchor
        if fundamental is not None:
            logger.info(
                "Forward proxy anchor: trailing=%.1f (fundamental R2=%.2f below threshold)",
                trailing_anchor, fundamental[1]["r2"],
            )
        else:
            logger.info("Forward proxy anchor (trailing %dm): %.1f EUR/MWh",
                        anchor_months, trailing_anchor)

    # ── 2. Seasonal shape from last 2+ years ─────────────────────────
    # Anchor on epex.index.max() rather than pd.Timestamp.now() so that
    # historical backtests (cutoff far in the past) still pick up a
    # meaningful 2-year window aligned with the train data.
    anchor_end = epex.index.max()
    cutoff = anchor_end - pd.DateOffset(years=2)
    recent = epex[epex.index >= cutoff]
    if len(recent) < 96 * 180:  # less than 6 months
        recent = epex  # fallback to full history

    recent_zh = recent.index.tz_convert("Europe/Zurich")
    monthly_avg = recent.groupby(recent_zh.month)["price_eur_mwh"].mean()
    # Reindex to all 12 months so downstream code can safely index by month.
    monthly_avg = monthly_avg.reindex(range(1, 13))
    global_mean = float(monthly_avg.dropna().mean()) if monthly_avg.notna().any() else 0.0

    if abs(global_mean) < 1.0:
        seasonal_ratio = pd.Series(1.0, index=range(1, 13))
    else:
        seasonal_ratio = (monthly_avg / global_mean).fillna(1.0)

    # ── 2b. Peak/Base ratio from history ───────────────────────────
    # Peak = weekday 08:00-20:00 Zurich time
    is_weekday = recent_zh.dayofweek < 5
    is_peak_hour = (recent_zh.hour >= 8) & (recent_zh.hour < 20)
    peak_mask = is_weekday & is_peak_hour

    peak_avg_by_month = recent[peak_mask].groupby(recent_zh[peak_mask].month)["price_eur_mwh"].mean()
    base_avg_by_month = recent.groupby(recent_zh.month)["price_eur_mwh"].mean()

    peak_ratio_by_month = {}
    for m in range(1, 13):
        if m in peak_avg_by_month.index and m in base_avg_by_month.index and base_avg_by_month[m] > 0:
            peak_ratio_by_month[m] = float(peak_avg_by_month[m] / base_avg_by_month[m])
        else:
            peak_ratio_by_month[m] = 1.15  # default Swiss peak/base ratio

    logger.info(
        "Peak/Base ratios: winter=%.2f, summer=%.2f",
        np.mean([peak_ratio_by_month.get(m, 1.15) for m in [12, 1, 2]]),
        np.mean([peak_ratio_by_month.get(m, 1.15) for m in [6, 7, 8]]),
    )

    logger.info(
        "Seasonal ratios: winter=%.2f, summer=%.2f",
        seasonal_ratio[[12, 1, 2]].mean(),
        seasonal_ratio[[6, 7, 8]].mean(),
    )

    # ── 3. Term-structure decay (mean reversion) ─────────────────────
    # Estimate from year-over-year changes in the spot history
    yearly_avg = epex.groupby(idx_zh.year)["price_eur_mwh"].mean()
    if len(yearly_avg) >= 3:
        yoy_ratios = yearly_avg.pct_change().dropna()
        # Dampened mean reversion: converge toward long-term average
        annual_decay = float(np.clip(yoy_ratios.median(), -0.10, 0.0))
    else:
        annual_decay = -0.03  # conservative 3% backwardation per year

    logger.info("Estimated annual decay: %.1f%%", annual_decay * 100)

    # ── 4. Build Cal-year levels ─────────────────────────────────────
    base_prices: dict[str, float] = {}
    for i in range(n_years):
        year = start_year + i
        cal_level = anchor * (1 + annual_decay) ** i
        base_prices[str(year)] = round(cal_level, 1)

        # Monthly prices (base + peak)
        for month in range(1, 13):
            ratio = seasonal_ratio.get(month, 1.0)
            monthly_base = round(cal_level * ratio, 1)
            base_prices[f"{year}-{month:02d}"] = monthly_base
            # Peak price = base × peak/base ratio for that month
            pk_ratio = peak_ratio_by_month.get(month, 1.15)
            base_prices[f"{year}-{month:02d}-Peak"] = round(monthly_base * pk_ratio, 1)

        # Quarterly prices (hour-weighted average of months for energy conservation)
        import calendar
        for q, q_months in {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}.items():
            q_prices = [base_prices[f"{year}-{m:02d}"] for m in q_months]
            q_peak_prices = [base_prices[f"{year}-{m:02d}-Peak"] for m in q_months]
            hours = [calendar.monthrange(year, m)[1] * 24 for m in q_months]
            total_h = sum(hours)
            base_prices[f"{year}-Q{q}"] = round(
                sum(p * h for p, h in zip(q_prices, hours)) / total_h, 1)
            base_prices[f"{year}-Q{q}-Peak"] = round(
                sum(p * h for p, h in zip(q_peak_prices, hours)) / total_h, 1)

    logger.info(
        "Forward proxy: %d keys, Cal range: %s",
        len(base_prices),
        {k: f"{v:.0f}" for k, v in base_prices.items() if len(k) == 4},
    )

    return base_prices


def load_base_prices(
    epex: pd.DataFrame,
    eex_report_path: str | None = None,
    config: dict | None = None,
    market: str = "CH",
) -> tuple[dict[str, float], str]:
    """Load base prices: try EEX report first, then derive from spot.

    Args:
        epex: EPEX spot DataFrame for fallback proxy derivation.
        eex_report_path: Path to EEX XLSX report.
        config: Optional config dict.
        market: 'CH' or 'DE' — EEX sheet to load from.

    Returns:
        (base_prices dict, source description string)
    """
    # ── Try 1: EEX XLSX report (desk quotidien) ──────────────────────
    if eex_report_path or (config and config.get("forwards", {}).get("eex_report_path")):
        path = eex_report_path or config["forwards"]["eex_report_path"]
        try:
            from pfc_shaping.data.ingest_forwards import load_base_prices_from_eex_report

            # The explicit function argument takes precedence. The config
            # fallback is only used when callers do not pass a market.
            eex_market = market or "CH"
            if config and not market:
                eex_market = config.get("forwards", {}).get("eex_market", eex_market)

            prices = load_base_prices_from_eex_report(path, market=eex_market)
            if prices and len(prices) >= 3:
                logger.info("Loaded %d EEX forward prices from XLSX (%s): %s", len(prices), eex_market, path)
                # Also update historical Parquet
                try:
                    from pfc_shaping.data.ingest_forwards import update_forwards_parquet
                    markets = ["CH", "DE"]
                    if config:
                        markets = config.get("forwards", {}).get("eex_markets", markets)
                    update_forwards_parquet(path, markets=markets)
                except Exception as e:
                    logger.warning("Could not update forwards history: %s", e)
                return prices, f"EEX XLSX {eex_market} ({len(prices)} keys)"
        except Exception as exc:
            logger.warning("EEX XLSX not available: %s", exc)

    # ── Try 2: EEX report UNC path ───────────────────────────────────
    if config and config.get("forwards", {}).get("eex_report_path_unc"):
        unc_path = config["forwards"]["eex_report_path_unc"]
        try:
            from pfc_shaping.data.ingest_forwards import load_base_prices_from_eex_report

            eex_market = market or "CH"
            if not market:
                eex_market = config.get("forwards", {}).get("eex_market", eex_market)
            prices = load_base_prices_from_eex_report(unc_path, market=eex_market)
            if prices and len(prices) >= 3:
                logger.info("Loaded %d EEX forward prices from UNC: %s", len(prices), unc_path)
                return prices, f"EEX UNC {eex_market} ({len(prices)} keys)"
        except Exception as exc:
            logger.warning("EEX UNC not available: %s", exc)

    # ── Try 3: Databricks ────────────────────────────────────────────
    if config and config.get("databricks", {}).get("enabled"):
        try:
            from pfc_shaping.data.ingest_forwards import load_base_prices as load_db

            prices = load_db(db_config=config.get("databricks", {}))
            if prices and len(prices) >= 3:
                logger.info("Loaded %d EEX forward prices from Databricks", len(prices))
                return prices, f"Databricks ({len(prices)} keys)"
        except Exception as exc:
            logger.warning("Databricks not available: %s", exc)

    # ── Fallback: Derive from spot ───────────────────────────────────
    logger.info("No EEX forward source available — deriving from spot history")
    prices = derive_base_prices(epex)
    return prices, f"Spot-derived proxy ({len(prices)} keys)"

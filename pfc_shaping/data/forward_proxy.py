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


def derive_base_prices(
    epex: pd.DataFrame,
    start_year: int | None = None,
    n_years: int = 4,
    anchor_months: int = 6,
) -> dict[str, float]:
    """Derive base prices from EPEX spot history.

    Args:
        epex: DataFrame with 'price_eur_mwh' column, DatetimeIndex UTC.
        start_year: First delivery year. Default: current year.
        n_years: Number of forward years to generate.
        anchor_months: Months of trailing spot to anchor level.

    Returns:
        Dict with keys like '2026', '2026-Q1', '2026-01' etc.
    """
    if start_year is None:
        start_year = pd.Timestamp.now().year

    idx_zh = epex.index.tz_convert("Europe/Zurich")

    # ── 1. Trailing anchor level ──────────────────────────────────────
    n_rows = min(96 * 30 * anchor_months, len(epex))
    anchor = epex["price_eur_mwh"].iloc[-n_rows:].mean()
    logger.info("Forward proxy anchor (trailing %dm): %.1f EUR/MWh", anchor_months, anchor)

    # ── 2. Seasonal shape from last 2+ years ─────────────────────────
    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=2)
    recent = epex[epex.index >= cutoff]
    if len(recent) < 96 * 180:  # less than 6 months
        recent = epex  # fallback to full history

    recent_zh = recent.index.tz_convert("Europe/Zurich")
    monthly_avg = recent.groupby(recent_zh.month)["price_eur_mwh"].mean()
    global_mean = monthly_avg.mean()

    if abs(global_mean) < 1.0:
        seasonal_ratio = pd.Series(1.0, index=range(1, 13))
    else:
        seasonal_ratio = monthly_avg / global_mean

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

        # Quarterly prices (average of months)
        for q, months in {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}.items():
            q_prices = [base_prices[f"{year}-{m:02d}"] for m in months]
            q_peak_prices = [base_prices[f"{year}-{m:02d}-Peak"] for m in months]
            base_prices[f"{year}-Q{q}"] = round(float(np.mean(q_prices)), 1)
            base_prices[f"{year}-Q{q}-Peak"] = round(float(np.mean(q_peak_prices)), 1)

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
) -> tuple[dict[str, float], str]:
    """Load base prices: try EEX report first, then derive from spot.

    Returns:
        (base_prices dict, source description string)
    """
    # ── Try 1: EEX XLSX report (desk quotidien) ──────────────────────
    if eex_report_path or (config and config.get("forwards", {}).get("eex_report_path")):
        path = eex_report_path or config["forwards"]["eex_report_path"]
        try:
            from pfc_shaping.data.ingest_forwards import load_base_prices_from_eex_report

            market = "CH"
            if config:
                market = config.get("forwards", {}).get("eex_market", "CH")

            prices = load_base_prices_from_eex_report(path, market=market)
            if prices and len(prices) >= 3:
                logger.info("Loaded %d EEX forward prices from XLSX: %s", len(prices), path)
                return prices, f"EEX XLSX ({len(prices)} keys)"
        except Exception as exc:
            logger.warning("EEX XLSX not available: %s", exc)

    # ── Try 2: EEX report UNC path ───────────────────────────────────
    if config and config.get("forwards", {}).get("eex_report_path_unc"):
        unc_path = config["forwards"]["eex_report_path_unc"]
        try:
            from pfc_shaping.data.ingest_forwards import load_base_prices_from_eex_report

            market = config.get("forwards", {}).get("eex_market", "CH")
            prices = load_base_prices_from_eex_report(unc_path, market=market)
            if prices and len(prices) >= 3:
                logger.info("Loaded %d EEX forward prices from UNC: %s", len(prices), unc_path)
                return prices, f"EEX UNC ({len(prices)} keys)"
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

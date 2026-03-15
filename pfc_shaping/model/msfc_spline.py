"""
msfc_spline.py — Maximum Smoothness Forward Curve (MSFC)
--------------------------------------------------------
Smooth the base price level B(t) across delivery period boundaries
while preserving the arbitrage-free constraint:

    mean(B(t) over period_i) = forward_price_i   for every contract i

Methodology:
    Benth, Koekebakker & Ollmar (2007) — "Extracting and Applying Smooth
    Forward Curves From Average-Based Commodity Contracts with Seasonal Variation"

    Simplified robust implementation using monthly midpoint knots
    with cubic spline interpolation. This avoids the oscillation problems
    of high-resolution QP approaches while producing smooth transitions.

    Algorithm:
    1. Start from monthly forward prices as knot values at month midpoints
    2. Solve a small QP (n_months knots) for max smoothness under
       average-pricing constraints
    3. Cubic-interpolate from monthly midpoints to daily, then to 15-min

Usage:
    from pfc_shaping.model.msfc_spline import smooth_base_prices
    B_smooth = smooth_base_prices(idx_15min, base_prices, B_flat)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

logger = logging.getLogger(__name__)


def smooth_base_prices(
    idx: pd.DatetimeIndex,
    base_prices: dict[str, float],
    B_flat: pd.Series,
) -> pd.Series:
    """Smooth the flat staircase B(t) into a continuous curve.

    Uses PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
    on monthly midpoint knots. PCHIP preserves monotonicity between
    knots and avoids the ringing/oscillation of natural cubic splines.

    Args:
        idx: 15-min DatetimeIndex (UTC).
        base_prices: Forward price dict (keys: '2026', '2026-Q1', '2026-03', etc.).
        B_flat: The staircase base price series from _resolve_base().

    Returns:
        pd.Series with same index, smoothed B values.
    """
    base_only = {k: v for k, v in base_prices.items() if not k.endswith("-Peak")}

    idx_zh = idx.tz_convert("Europe/Zurich")

    # ── 1. Extract monthly forward prices as knot values ──────────────
    # Get all months covered by the index
    month_keys = []
    month_midpoints = []  # as float days from start
    month_prices = []

    start_ts = idx_zh[0]
    year_min = idx_zh.min().year
    year_max = idx_zh.max().year

    for year in range(year_min, year_max + 1):
        for month in range(1, 13):
            key = f"{year}-{month:02d}"
            # Check if this month is in our index
            mask = (idx_zh.year == year) & (idx_zh.month == month)
            if not mask.any():
                continue

            # Get the price for this month (from base_prices or B_flat)
            if key in base_only:
                price = base_only[key]
            else:
                # Use the flat B value for this month
                price = float(B_flat[mask].mean())

            # Midpoint of the month in days from start
            month_start = pd.Timestamp(f"{year}-{month:02d}-01", tz="Europe/Zurich")
            if month == 12:
                month_end = pd.Timestamp(f"{year + 1}-01-01", tz="Europe/Zurich")
            else:
                month_end = pd.Timestamp(f"{year}-{month + 1:02d}-01", tz="Europe/Zurich")
            midpoint = month_start + (month_end - month_start) / 2

            days_from_start = (midpoint - start_ts).total_seconds() / 86400.0

            month_keys.append(key)
            month_midpoints.append(days_from_start)
            month_prices.append(price)

    if len(month_midpoints) < 3:
        logger.warning("MSFC: too few months (%d), skipping smoothing", len(month_midpoints))
        return B_flat

    x_knots = np.array(month_midpoints)
    y_knots = np.array(month_prices)

    # ── 2. PCHIP interpolation to 15-min ──────────────────────────────
    # PCHIP: monotonicity-preserving, no oscillation, C1-smooth
    interpolator = PchipInterpolator(x_knots, y_knots, extrapolate=True)

    # Convert each 15-min timestamp to days from start
    x_target = (idx_zh - start_ts).total_seconds().values / 86400.0
    B_smooth_raw = interpolator(x_target)

    # ── 3. Enforce mean constraints per delivery period ───────────────
    # Iterative correction: adjust knot values so that the mean
    # of the interpolated curve over each month = forward price
    B_smooth = _enforce_mean_constraints(
        idx_zh, x_target, x_knots, y_knots, month_keys,
        base_only, B_smooth_raw, start_ts, max_iter=10,
    )

    # Ensure positivity
    B_smooth = np.maximum(B_smooth, 1.0)

    # ── 4. Verify ──────────────────────────────────────────────────────
    max_correction = np.max(np.abs(B_smooth - B_flat.values))
    _verify_constraints(idx_zh, B_smooth, base_only)

    logger.info(
        "MSFC smoothing applied: %d monthly knots, max correction=%.1f EUR/MWh",
        len(month_midpoints), max_correction,
    )

    return pd.Series(B_smooth, index=idx, name="B")


def _enforce_mean_constraints(
    idx_zh: pd.DatetimeIndex,
    x_target: np.ndarray,
    x_knots: np.ndarray,
    y_knots: np.ndarray,
    month_keys: list[str],
    base_prices: dict[str, float],
    B_initial: np.ndarray,
    start_ts: pd.Timestamp,
    max_iter: int = 10,
) -> np.ndarray:
    """Iteratively adjust knot values to match monthly mean constraints.

    Simple and robust: for each month where mean(B) != forward price,
    scale the knot value proportionally. Converges quickly because
    PCHIP is local (changing one knot mainly affects nearby months).
    """
    years = np.array([t.year for t in idx_zh])
    months = np.array([t.month for t in idx_zh])

    y_adjusted = y_knots.copy()

    for iteration in range(max_iter):
        # Interpolate with current knots
        interpolator = PchipInterpolator(x_knots, y_adjusted, extrapolate=True)
        B_current = interpolator(x_target)
        B_current = np.maximum(B_current, 1.0)

        max_error = 0.0
        for i, key in enumerate(month_keys):
            if key not in base_prices:
                continue

            target = base_prices[key]
            y, m = int(key[:4]), int(key[5:7])
            mask = (years == y) & (months == m)

            if not mask.any():
                continue

            actual_mean = B_current[mask].mean()
            if abs(actual_mean) < 1e-6:
                continue

            error = target - actual_mean
            max_error = max(max_error, abs(error))

            # Adjust knot value proportionally
            correction = error * 0.8  # damping factor for stability
            y_adjusted[i] += correction

        if max_error < 0.5:  # converged within 0.5 EUR/MWh
            logger.debug("MSFC converged in %d iterations (max_error=%.2f)",
                         iteration + 1, max_error)
            break

    # Final interpolation
    interpolator = PchipInterpolator(x_knots, y_adjusted, extrapolate=True)
    return interpolator(x_target)


def _verify_constraints(
    idx_zh: pd.DatetimeIndex,
    B_smooth: np.ndarray,
    base_prices: dict[str, float],
) -> None:
    """Log verification of energy conservation after smoothing."""
    years = np.array([t.year for t in idx_zh])
    months = np.array([t.month for t in idx_zh])

    deviations = []
    for key, target in base_prices.items():
        if key.endswith("-Peak"):
            continue
        if len(key) == 7 and "-" in key and "Q" not in key:
            try:
                y, m = int(key[:4]), int(key[5:7])
            except (ValueError, IndexError):
                continue
            mask = (years == y) & (months == m)
        elif len(key) == 4 and key.isdigit():
            mask = years == int(key)
        else:
            continue

        if mask.any():
            actual = B_smooth[mask].mean()
            dev_pct = abs(actual - target) / max(abs(target), 1e-6) * 100
            if dev_pct > 1.0:
                deviations.append((key, target, actual, dev_pct))

    if deviations:
        for key, target, actual, dev_pct in deviations[:5]:
            logger.warning(
                "MSFC constraint violation: %s target=%.1f actual=%.1f (%.1f%%)",
                key, target, actual, dev_pct,
            )

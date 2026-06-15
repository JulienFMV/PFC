"""Intraday amplitude correction for LT HPFC f_H.

This module is an additive, flag-gated post-processor for the hourly shape
factor. It compresses the EEX peak-vs-offpeak contrast implied by ``f_H`` toward
the leak-free peak/base spread already fitted on the cascader. The operation is
mean-preserving within each local month and then within each local day, so the
downstream level and weekly layers keep their energy semantics.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "compress_intraday_peak_amplitude",
    "compress_price_peak_amplitude",
    "DEFAULT_MIN_SHRINK",
]

DEFAULT_MIN_SHRINK = 0.20


def _month_key(index: pd.DatetimeIndex, tz: str) -> pd.Index:
    local = index.tz_convert(tz)
    return pd.Index([f"{t.year}-{t.month:02d}" for t in local], name="month")


def _day_key(index: pd.DatetimeIndex, tz: str) -> pd.Index:
    local = index.tz_convert(tz)
    return pd.Index([f"{t.year}-{t.month:02d}-{t.day:02d}" for t in local], name="day")


def _peak_mask(calendar_df: pd.DataFrame) -> np.ndarray:
    """Canonical shape-level peak mask from enriched calendar columns."""
    hour = calendar_df["heure_hce"].astype(int).to_numpy()
    type_jour = calendar_df["type_jour"].astype(str).to_numpy()
    return (type_jour == "Ouvrable") & (hour >= 8) & (hour < 20)


def compress_intraday_peak_amplitude(
    f_H_series: pd.Series,
    calendar_df: pd.DataFrame,
    base_level: pd.Series,
    peak_base_spreads: dict[int, float] | None,
    *,
    tz: str = "Europe/Zurich",
    min_shrink: float = DEFAULT_MIN_SHRINK,
) -> pd.Series:
    """Compress f_H peak/offpeak amplitude toward fitted hydro-aware spreads.

    Parameters
    ----------
    f_H_series
        Hourly or sub-hourly shape factor, indexed by tz-aware UTC timestamps.
    calendar_df
        Enriched calendar with ``type_jour`` and ``heure_hce``.
    base_level
        Smoothed base level ``B`` in EUR/MWh on the same index.
    peak_base_spreads
        Month -> target peak/base spread in EUR/MWh, fitted pre-vintage by the
        cascader. Missing months degrade to no change.
    tz
        Delivery timezone.
    min_shrink
        Lower bound on the multiplicative contrast shrinkage. Prevents a noisy
        month from flattening all peak structure.
    """
    if peak_base_spreads is None:
        return f_H_series
    if f_H_series.empty:
        return f_H_series

    cal = calendar_df.reindex(f_H_series.index)
    if {"type_jour", "heure_hce"} - set(cal.columns):
        raise KeyError("calendar_df must contain type_jour and heure_hce")

    out = f_H_series.astype(float).copy()
    base = base_level.reindex(out.index).astype(float)
    peak = pd.Series(_peak_mask(cal), index=out.index)
    months = _month_key(out.index, tz)

    diagnostics: list[tuple[str, float, float, float]] = []
    for month_label in pd.unique(months):
        mask = months == month_label
        p_mask = mask & peak.to_numpy()
        o_mask = mask & (~peak.to_numpy())
        n_peak = int(np.sum(p_mask))
        n_off = int(np.sum(o_mask))
        if n_peak == 0 or n_off == 0:
            continue

        f_peak = float(out.iloc[p_mask].mean())
        f_off = float(out.iloc[o_mask].mean())
        current_rel = f_peak - f_off
        if not np.isfinite(current_rel) or current_rel <= 0.0:
            continue

        month_num = int(str(month_label)[5:7])
        target_abs = peak_base_spreads.get(month_num)
        if target_abs is None or not np.isfinite(target_abs):
            continue
        base_ref = float(base.iloc[mask].median())
        if not np.isfinite(base_ref) or abs(base_ref) < 1e-9:
            continue
        target_rel = max(0.0, float(target_abs) / abs(base_ref))
        if current_rel <= target_rel:
            continue

        shrink = max(float(min_shrink), min(1.0, target_rel / current_rel))
        desired_rel = current_rel * shrink
        reduction = current_rel - desired_rel
        total = n_peak + n_off
        peak_shift = reduction * n_off / total
        off_shift = reduction * n_peak / total

        out.iloc[p_mask] = out.iloc[p_mask] - peak_shift
        out.iloc[o_mask] = out.iloc[o_mask] + off_shift
        diagnostics.append((str(month_label), current_rel, desired_rel, shrink))

    # Preserve mean_h f_H = 1 on every local day. This mirrors solar_modulation.
    days = _day_key(out.index, tz)
    day_mean = out.groupby(days).transform("mean").replace(0.0, 1.0)
    out = (out / day_mean).rename(f_H_series.name or "f_H")
    out = out.clip(lower=0.05)
    day_mean = out.groupby(days).transform("mean").replace(0.0, 1.0)
    out = (out / day_mean).rename(f_H_series.name or "f_H")
    if diagnostics:
        logger.info(
            "Intraday amplitude compression applied to %d months; median shrink=%.3f",
            len(diagnostics),
            float(np.median([d[3] for d in diagnostics])),
        )
    return out


def compress_price_peak_amplitude(
    price_series: pd.Series,
    calendar_df: pd.DataFrame,
    peak_base_spreads: dict[int, float] | None,
    *,
    tz: str = "Europe/Zurich",
    min_shrink: float = DEFAULT_MIN_SHRINK,
) -> pd.Series:
    """Compress final price peak/offpeak spreads while preserving monthly means."""
    if peak_base_spreads is None:
        return price_series
    if price_series.empty:
        return price_series

    cal = calendar_df.reindex(price_series.index)
    if {"type_jour", "heure_hce"} - set(cal.columns):
        raise KeyError("calendar_df must contain type_jour and heure_hce")

    out = price_series.astype(float).copy()
    peak = pd.Series(_peak_mask(cal), index=out.index)
    months = _month_key(out.index, tz)
    diagnostics: list[tuple[str, float, float, float]] = []

    for month_label in pd.unique(months):
        mask = months == month_label
        p_mask = mask & peak.to_numpy()
        o_mask = mask & (~peak.to_numpy())
        n_peak = int(np.sum(p_mask))
        n_off = int(np.sum(o_mask))
        if n_peak == 0 or n_off == 0:
            continue

        current = float(out.iloc[p_mask].mean() - out.iloc[o_mask].mean())
        if not np.isfinite(current) or current <= 0.0:
            continue

        month_num = int(str(month_label)[5:7])
        target = peak_base_spreads.get(month_num)
        if target is None or not np.isfinite(target) or current <= target:
            continue

        shrink = max(float(min_shrink), min(1.0, float(target) / current))
        desired = current * shrink
        reduction = current - desired
        total = n_peak + n_off
        peak_shift = reduction * n_off / total
        off_shift = reduction * n_peak / total

        out.iloc[p_mask] = out.iloc[p_mask] - peak_shift
        out.iloc[o_mask] = out.iloc[o_mask] + off_shift
        diagnostics.append((str(month_label), current, desired, shrink))

    if diagnostics:
        logger.info(
            "Final price peak/offpeak compression applied to %d months; median shrink=%.3f",
            len(diagnostics),
            float(np.median([d[3] for d in diagnostics])),
        )
    return out.rename(price_series.name or "price_shape")

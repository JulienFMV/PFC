"""Quote-aware EEX contract bucket selection.

The helpers in this module build non-overlapping calibration buckets from a
set of potentially overlapping EEX products.  Priority is always
``Month > Quarter > Calendar``.  If a coarser quoted product overlaps finer
quoted products, the unquoted remainder receives an explicit residual target
so that all quoted products can be respected simultaneously.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def quarter_product(ts: pd.Timestamp) -> str:
    """Return the EEX quarter key for a local timestamp."""
    return f"{ts.year}-Q{((ts.month - 1) // 3) + 1}"


def finest_product_for_timestamp(ts: pd.Timestamp, prices: Mapping[str, float]) -> str | None:
    """Return the finest directly quoted product covering ``ts``.

    This helper is intentionally quote-only.  It does not create residual
    buckets; use :func:`calibration_buckets` for full quote-aware calibration.
    """
    month = f"{ts.year}-{ts.month:02d}"
    quarter = quarter_product(ts)
    year = str(ts.year)
    if month in prices:
        return month
    if quarter in prices:
        return quarter
    if year in prices:
        return year
    return None


def calibration_buckets(
    ts_local: pd.Series,
    forward_prices: Mapping[str, float],
) -> tuple[pd.Series, dict[str, float]]:
    """Build row-level non-overlapping calibration buckets.

    Args:
        ts_local: Series of local delivery timestamps.  One row represents one
            equal-energy delivery interval, e.g. one hour in the hourly export
            or one 15-minute interval in a 15-minute curve.
        forward_prices: Mapping of EEX BASE prices keyed as ``YYYY``,
            ``YYYY-Qn`` or ``YYYY-MM``.

    Returns:
        ``(bucket_by_row, target_by_bucket)``.  Residual buckets are named
        ``YYYY-RESIDUAL`` or ``YYYY-Qn-RESIDUAL``.
    """
    ts = pd.Series(ts_local, copy=False)
    if ts.empty:
        return pd.Series(dtype=object), {}
    if not hasattr(ts.dt, "year"):
        raise TypeError("ts_local must be datetime-like")

    buckets = pd.Series([None] * len(ts), index=ts.index, dtype=object)
    targets: dict[str, float] = {}
    years = ts.dt.year.astype(int)
    quarters = ts.dt.quarter.astype(int)
    months = ts.dt.month.astype(int)
    prices = {str(k): float(v) for k, v in forward_prices.items()}

    for year in sorted(years.unique()):
        for month in range(1, 13):
            key = f"{year}-{month:02d}"
            if key not in prices:
                continue
            mask = (years == year) & (months == month)
            if not bool(mask.any()):
                continue
            buckets.loc[mask] = key
            targets[key] = prices[key]

    for year in sorted(years.unique()):
        for quarter in range(1, 5):
            key = f"{year}-Q{quarter}"
            if key not in prices:
                continue
            q_mask = (years == year) & (quarters == quarter)
            if not bool(q_mask.any()):
                continue
            free_mask = q_mask & buckets.isna()
            if not bool(free_mask.any()):
                continue
            target = _residual_target(
                parent_key=key,
                parent_mask=q_mask,
                free_mask=free_mask,
                buckets=buckets,
                targets=targets,
                parent_target=prices[key],
            )
            bucket = key if not bool((q_mask & buckets.notna()).any()) else f"{key}-RESIDUAL"
            buckets.loc[free_mask] = bucket
            targets[bucket] = target

    for year in sorted(years.unique()):
        key = str(year)
        if key not in prices:
            continue
        y_mask = years == year
        if not bool(y_mask.any()):
            continue
        free_mask = y_mask & buckets.isna()
        if not bool(free_mask.any()):
            continue
        target = _residual_target(
            parent_key=key,
            parent_mask=y_mask,
            free_mask=free_mask,
            buckets=buckets,
            targets=targets,
            parent_target=prices[key],
        )
        bucket = key if not bool((y_mask & buckets.notna()).any()) else f"{key}-RESIDUAL"
        buckets.loc[free_mask] = bucket
        targets[bucket] = target

    return buckets, targets


def _residual_target(
    *,
    parent_key: str,
    parent_mask: pd.Series,
    free_mask: pd.Series,
    buckets: pd.Series,
    targets: Mapping[str, float],
    parent_target: float,
) -> float:
    known_mask = parent_mask & buckets.notna()
    if not bool(known_mask.any()):
        return float(parent_target)

    known_energy = 0.0
    for bucket, idx in buckets.loc[known_mask].groupby(buckets.loc[known_mask]).groups.items():
        if bucket not in targets:
            raise ValueError(f"missing target for nested EEX bucket {bucket!r} under {parent_key}")
        known_energy += float(targets[str(bucket)]) * len(idx)

    residual_hours = int(free_mask.sum())
    if residual_hours <= 0:
        raise ValueError(f"empty residual EEX bucket under {parent_key}")
    residual_energy = float(parent_target) * int(parent_mask.sum()) - known_energy
    target = residual_energy / residual_hours
    if not np.isfinite(target):
        raise ValueError(f"non-finite residual EEX target under {parent_key}")
    return float(target)

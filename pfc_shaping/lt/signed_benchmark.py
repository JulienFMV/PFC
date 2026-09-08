"""Numerical preparation for the local signed-shape experiment, without authority."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.local_benchmark import utc_index
from pfc_shaping.lt.structural_readiness import center_signed_hourly_shape


def closed_month_targets(prices: pd.Series, origin: pd.Timestamp) -> pd.DataFrame:
    """Build both targets on one native-hour history closed before the origin.

    Monthly means of realized prices define training labels only. Prices after
    the origin, incomplete edge months and the open month cannot enter labels.
    Historical publication/revision availability remains the caller's concern.
    """
    index = utc_index(prices.index).as_unit("ns")
    if origin.tzinfo is None:
        raise ValueError("origin must be timezone-aware")
    if np.any(index.asi8 % pd.Timedelta(hours=1).value):
        raise ValueError("prices must be native hourly observations")
    history = pd.Series(prices.to_numpy(dtype=float), index=index).loc[index < origin]
    periods = history.index.tz_convert("Europe/Zurich").tz_localize(None).to_period("M")
    accepted = []
    for month in periods.unique():
        start = month.start_time.tz_localize("Europe/Zurich")
        end = (month + 1).start_time.tz_localize("Europe/Zurich")
        expected = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
        observed = history.loc[periods == month]
        if end <= origin and observed.index.equals(expected) and np.isfinite(observed).all():
            accepted.append(month)
    history = history.loc[periods.isin(accepted)]
    signed = center_signed_hourly_shape(history)  # Also rejects interior gaps.
    days = history.index.tz_convert("Europe/Zurich").normalize()
    daily = history.groupby(days).transform("mean")
    ratio = (history / daily.where(daily.gt(5))).clip(.2, 3.)
    return pd.DataFrame({"price_eur_mwh": history, "signed_target": signed, "ratio_target": ratio})


def calendar_cell_reference(
    target: pd.Series,
    delivery: pd.DatetimeIndex,
    *,
    sample_weight: pd.Series | None = None,
) -> tuple[np.ndarray, dict]:
    """Unclipped calendar mean with optional explicit historical weighting.

    Weights change cell means only; calendar backoffs and the default arithmetic
    reference stay unchanged. The caller owns the origin-safe weighting policy.
    """
    index = utc_index(target.index)
    delivery = utc_index(delivery)
    if target.empty or delivery.empty or not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("reference requires nonempty finite training targets and delivery")
    if index[-1] >= delivery[0]:
        raise ValueError("reference training must precede delivery")
    train = enrich_15min_index(index).copy()
    train["target"] = target.to_numpy(dtype=float)
    if sample_weight is not None:
        if (not isinstance(sample_weight, pd.Series)
                or not sample_weight.index.equals(target.index)
                or sample_weight.dtype.kind not in "fiu"
                or not np.isfinite(sample_weight).all()
                or not sample_weight.gt(0).all()):
            raise ValueError("sample weights must be exactly aligned, finite and positive")
        # Rescaling leaves weighted means invariant and avoids overflow in sums.
        train["weight"] = sample_weight.to_numpy(dtype=float) / sample_weight.max()
        train["weighted_target"] = train.target * train.weight
    future = enrich_15min_index(delivery)
    result = np.full(len(delivery), np.nan)
    counts = {}
    for keys in [("saison", "type_jour", "heure_hce"), ("saison", "heure_hce"), ("heure_hce",)]:
        if sample_weight is None:
            table = train.groupby(list(keys))["target"].mean()
        else:
            totals = train.groupby(list(keys))[["weighted_target", "weight"]].sum()
            table = totals.weighted_target / totals.weight
        query = pd.MultiIndex.from_frame(future[list(keys)]) if len(keys) > 1 else pd.Index(future[keys[0]])
        values = table.reindex(query).to_numpy()
        use = np.isnan(result) & np.isfinite(values)
        result[use] = values[use]
        counts["/".join(keys)] = int(use.sum())
    counts["global"] = int(np.isnan(result).sum())
    result[np.isnan(result)] = (float(target.mean()) if sample_weight is None
                               else float(train.weighted_target.sum() / train.weight.sum()))
    return result, counts

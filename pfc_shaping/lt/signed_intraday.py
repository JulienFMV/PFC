"""Additive seasonal intrahour reference for local composition experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.local_benchmark import utc_index


def intraday_cell_reference(
    prices: pd.Series, delivery: pd.DatetimeIndex, *, country: str = "CH",
) -> tuple[pd.Series, dict[str, int]]:
    """Learn DE quarter residuals and map them onto a CH or DE civil calendar.

    This unconditional reference learns absolute price deviations, with no
    division near zero and no change of sign when the hourly price is negative.
    It makes no claim about predictive validity of DE-to-CH transfer.
    """
    if (not pd.api.types.is_numeric_dtype(prices.dtype)
            or pd.api.types.is_bool_dtype(prices.dtype)
            or np.iscomplexobj(prices.to_numpy())):
        raise ValueError("intraday prices must be real numeric values")
    history = utc_index(prices.index)
    future = utc_index(delivery)
    for index in (history, future):
        if (index.empty or np.any(index.as_unit("ns").asi8 % pd.Timedelta(minutes=15).value)
                or not pd.Series(1, index=index).groupby(index.floor("h")).size().eq(4).all()):
            raise ValueError("intraday reference requires complete native parent hours")
    if history[-1] >= future[0] or not np.isfinite(prices.to_numpy()).all():
        raise ValueError("intraday history must be finite and strictly before delivery")
    train = enrich_15min_index(history, country="DE").copy()
    values = pd.Series(prices.to_numpy(dtype=float), index=history)
    train["target"] = values - values.groupby(history.floor("h")).transform("mean")
    query_frame = enrich_15min_index(future, country=country)
    result = np.full(len(future), np.nan)
    counts = {}
    for keys in [("saison", "type_jour", "heure_hce", "quart"),
                 ("saison", "heure_hce", "quart"), ("heure_hce", "quart"), ("quart",)]:
        table = train.groupby(list(keys))["target"].mean()
        query = (pd.MultiIndex.from_frame(query_frame[list(keys)]) if len(keys) > 1
                 else pd.Index(query_frame[keys[0]]))
        predictions = table.reindex(query).to_numpy()
        use = np.isnan(result) & np.isfinite(predictions)
        result[use] = predictions[use]
        counts["/".join(keys)] = int(use.sum())
    if not np.isfinite(result).all():
        raise ValueError("intraday reference has unsupported quarters")
    residual = pd.Series(result, index=future, name="signed_intraday_shape_eur_mwh")
    residual -= residual.groupby(future.floor("h")).transform("mean")
    return residual, counts

"""Holiday pressure features for CH long-term spot/HFC shaping.

The EEX Peakload contract includes weekday public holidays. Spot and HFC shape
pressure is nevertheless holiday-sensitive because CH prices are coupled
to neighbouring markets and large cantonal holidays.  This module keeps that
structural day-type logic in LT model code so scripts and future production
paths can share one implementation.
"""

from __future__ import annotations

from functools import lru_cache

import holidays
import pandas as pd

LOCAL_TZ = "Europe/Zurich"


@lru_cache(maxsize=None)
def ch_market_holiday_components(year: int) -> dict[object, float]:
    """Weighted holidays that can materially affect CH spot shape.

    Weights are a structural pressure proxy, not a binary exchange calendar.
    CH national holidays get full weight. Large cantonal holidays and holidays
    in coupled neighbouring markets add partial pressure, capped downstream.
    """
    y = int(year)
    weights: dict[object, float] = {}

    def add_dates(dates: set, weight: float) -> None:
        for d in dates:
            weights[d] = weights.get(d, 0.0) + float(weight)

    add_dates(set(holidays.Switzerland(years=y).keys()), 1.00)
    for subdiv, weight in {
        "ZH": 0.35,
        "VD": 0.30,
        "GE": 0.30,
        "TI": 0.30,
        "BE": 0.25,
        "BS": 0.25,
        "BL": 0.20,
        "AG": 0.20,
        "SG": 0.20,
        "VS": 0.20,
    }.items():
        add_dates(set(holidays.Switzerland(years=y, subdiv=subdiv).keys()), weight)
    add_dates(set(holidays.Germany(years=y).keys()), 0.55)
    add_dates(set(holidays.France(years=y).keys()), 0.45)
    add_dates(set(holidays.Italy(years=y).keys()), 0.35)
    add_dates(set(holidays.Austria(years=y).keys()), 0.20)
    return weights


def ch_market_holiday_pressure(ts: pd.Timestamp) -> float:
    """Return holiday pressure in ``[0, 1.25]`` for a CH local timestamp."""
    local = pd.Timestamp(ts)
    if local.tzinfo is not None:
        local = local.tz_convert(LOCAL_TZ)
    return float(min(1.25, ch_market_holiday_components(int(local.year)).get(local.date(), 0.0)))


def ch_market_nonworking_pressure(ts: pd.Timestamp) -> float:
    """Return max weekend/holiday pressure used by CH HFC shape overlays."""
    local = pd.Timestamp(ts)
    if local.tzinfo is not None:
        local = local.tz_convert(LOCAL_TZ)
    weekend_pressure = 1.0 if local.weekday() >= 5 else 0.0
    return float(max(weekend_pressure, ch_market_holiday_pressure(local)))


def is_ch_nonworking_day(ts: pd.Timestamp) -> bool:
    """Return whether the timestamp should shape like a nonworking day."""
    return ch_market_nonworking_pressure(ts) >= 0.50

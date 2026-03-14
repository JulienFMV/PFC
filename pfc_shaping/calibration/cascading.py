"""
cascading.py
------------
Cascading decomposition of EEX power futures contracts.

When the forward curve has gaps (e.g., only a Calendar year price but no
Quarters or Months), this module fills in the missing granularities by
cascading higher-level contracts into lower-level ones using historical
seasonal patterns from EPEX Spot data.

Cascading hierarchy (EEX convention):
    Cal Y  →  Q1 + Q2 + Q3 + Q4
    Qi     →  Month1 + Month2 + Month3

Energy conservation constraint (hour-weighted average):
    F_parent = Σ(F_child_i × h_child_i) / Σ(h_child_i)

where h_child_i is the number of delivery hours in child period i.

Peak / Off-Peak decomposition:
    Peak   = 08:00–20:00 Mon–Fri excl. Swiss national holidays (EEX standard)
    Base   = all hours
    OffPeak = (Base × total_h - Peak × peak_h) / offpeak_h
"""

from __future__ import annotations

import calendar
import logging
import re
from dataclasses import dataclass, field

import holidays
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quarter ↔ month mapping
# ---------------------------------------------------------------------------
QUARTER_MONTHS: dict[int, list[int]] = {
    1: [1, 2, 3],
    2: [4, 5, 6],
    3: [7, 8, 9],
    4: [10, 11, 12],
}

MONTH_TO_QUARTER: dict[int, int] = {
    m: q for q, months in QUARTER_MONTHS.items() for m in months
}


# ---------------------------------------------------------------------------
# ContractSpec dataclass
# ---------------------------------------------------------------------------
@dataclass
class ContractSpec:
    """Specification of a delivery period for a power futures contract.

    Attributes:
        key: Delivery period key, e.g. '2026', '2026-Q1', '2026-01'.
        start: Delivery period start (inclusive), timezone-aware UTC.
        end: Delivery period end (exclusive), timezone-aware UTC.
        product_type: One of 'Cal', 'Quarter', 'Month'.
        n_hours: Total delivery hours in the period.
        n_peak_hours: Peak delivery hours (08:00-20:00 Mon-Fri excl. holidays).
        n_offpeak_hours: Off-peak delivery hours.
        price: Forward price in EUR/MWh (if known).
    """

    key: str
    start: pd.Timestamp
    end: pd.Timestamp
    product_type: str
    n_hours: int
    n_peak_hours: int
    n_offpeak_hours: int
    price: float | None = None


# ---------------------------------------------------------------------------
# Key parsing utilities
# ---------------------------------------------------------------------------

_RE_YEAR = re.compile(r"^(\d{4})$")
_RE_QUARTER = re.compile(r"^(\d{4})-Q([1-4])$")
_RE_MONTH = re.compile(r"^(\d{4})-(\d{2})$")


def parse_key(key: str) -> tuple[str, int, int | None]:
    """Parse a delivery period key into (product_type, year, sub_index).

    Args:
        key: Delivery period key, e.g. '2026', '2026-Q1', '2026-03'.

    Returns:
        Tuple of (product_type, year, sub_index) where sub_index is
        quarter number (1-4) for quarters, month number (1-12) for months,
        or None for calendar years.

    Raises:
        ValueError: If the key format is not recognised.
    """
    m = _RE_YEAR.match(key)
    if m:
        return ("Cal", int(m.group(1)), None)

    m = _RE_QUARTER.match(key)
    if m:
        return ("Quarter", int(m.group(1)), int(m.group(2)))

    m = _RE_MONTH.match(key)
    if m:
        return ("Month", int(m.group(1)), int(m.group(2)))

    # Peak/product-type suffixed keys (e.g. '2026-01-Peak') are handled
    # by the assembler, not the cascader. Return a sentinel type.
    if "-Peak" in key or "-Offpeak" in key:
        # Strip suffix and parse base key, return product-suffixed type
        base_key = key.replace("-Peak", "").replace("-Offpeak", "")
        base_type, yr, sub = parse_key(base_key)
        return (f"{base_type}_Peak", yr, sub)

    raise ValueError(f"Unrecognised delivery period key: {key!r}")


def quarter_key(year: int, q: int) -> str:
    """Build a quarter key, e.g. '2026-Q1'."""
    return f"{year}-Q{q}"


def month_key(year: int, m: int) -> str:
    """Build a month key, e.g. '2026-03'."""
    return f"{year}-{m:02d}"


# ---------------------------------------------------------------------------
# Hour counting helpers
# ---------------------------------------------------------------------------

def _swiss_holidays_set(year: int) -> set:
    """Return a set of ``datetime.date`` for Swiss national public holidays.

    Uses national-level holidays (no cantonal subdivision) for consistency
    with EEX peak/off-peak contract definitions.  EEX counts peak hours
    using national holidays, not cantonal ones.

    Note: ``calendar_ch.py`` uses ``subdiv="VS"`` for *dispatch shape*
    classification (Ferie_CH), which is correct — FMV operates in Valais.
    The distinction matters: VS has extra holidays (e.g. St-Joseph, Ascension
    cantonale) that EEX does not recognise as non-peak.
    """
    return set(holidays.Switzerland(years=year).keys())


def _period_boundaries_utc(
    year: int,
    month_start: int,
    month_end: int,
    tz: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start_utc, end_utc) for a period spanning month_start to month_end inclusive.

    Args:
        year: Delivery year.
        month_start: First month of the period (1-12).
        month_end: Last month of the period (1-12).
        tz: Local timezone string (e.g. 'Europe/Zurich').

    Returns:
        Tuple of UTC timestamps: start (inclusive), end (exclusive).
    """
    start_local = pd.Timestamp(year=year, month=month_start, day=1, tz=tz)
    if month_end == 12:
        end_local = pd.Timestamp(year=year + 1, month=1, day=1, tz=tz)
    else:
        end_local = pd.Timestamp(year=year, month=month_end + 1, day=1, tz=tz)

    return start_local.tz_convert("UTC"), end_local.tz_convert("UTC")


def count_hours(
    year: int,
    month_start: int,
    month_end: int,
    tz: str = "Europe/Zurich",
) -> tuple[int, int, int]:
    """Count total, peak, and off-peak hours for a delivery period.

    Correctly handles DST transitions (CET ↔ CEST) and leap years by
    generating the full hourly index in local time.

    Peak hours: 08:00–20:00 on weekdays (Mon–Fri) excluding Swiss holidays
    (canton VS).

    Args:
        year: Delivery year.
        month_start: First month of the period (inclusive).
        month_end: Last month of the period (inclusive).
        tz: Local timezone.

    Returns:
        (total_hours, peak_hours, offpeak_hours)
    """
    start_utc, end_utc = _period_boundaries_utc(year, month_start, month_end, tz)

    # Generate hourly index in UTC then convert to local
    idx_utc = pd.date_range(start_utc, end_utc, freq="h", inclusive="left", tz="UTC")
    idx_local = idx_utc.tz_convert(tz)

    total_hours = len(idx_utc)

    # Collect holidays for all years that may be spanned
    hol_years = set(idx_local.year.unique())
    ch_holidays: set = set()
    for y in hol_years:
        ch_holidays |= _swiss_holidays_set(y)

    # Peak mask: hour 08..19 on weekdays, not a Swiss holiday
    is_weekday = idx_local.weekday < 5  # Mon=0 .. Fri=4
    is_peak_hour = (idx_local.hour >= 8) & (idx_local.hour < 20)
    is_holiday = pd.Series(idx_local.date, index=idx_utc).isin(ch_holidays).values

    peak_mask = is_weekday & is_peak_hour & ~is_holiday
    peak_hours = int(peak_mask.sum())
    offpeak_hours = total_hours - peak_hours

    return total_hours, peak_hours, offpeak_hours


def _month_range_for_year(year: int) -> tuple[int, int]:
    """Return (1, 12) — full year."""
    return 1, 12


def _month_range_for_quarter(q: int) -> tuple[int, int]:
    """Return (first_month, last_month) for quarter q (1-4)."""
    months = QUARTER_MONTHS[q]
    return months[0], months[-1]


def _month_range_for_month(m: int) -> tuple[int, int]:
    """Return (m, m) — single month."""
    return m, m


# ---------------------------------------------------------------------------
# ContractCascader
# ---------------------------------------------------------------------------

class ContractCascader:
    """Decomposes higher-granularity power futures into lower-granularity
    contracts using seasonal patterns derived from EPEX Spot history.

    The cascading preserves the energy conservation constraint:
        F_parent = sum(F_child_i * h_i) / sum(h_i)

    where h_i is the number of delivery hours in each child period.

    Attributes:
        tz: Local timezone for hour counting and peak/off-peak classification.
        seasonal_ratios_: Fitted seasonal ratios, populated by ``fit_seasonal_ratios``.
    """

    def __init__(self, tz: str = "Europe/Zurich") -> None:
        self.tz = tz
        self.seasonal_ratios_: dict[str, dict[int, float]] | None = None

    # ------------------------------------------------------------------
    # Fitting seasonal ratios
    # ------------------------------------------------------------------

    def fit_seasonal_ratios(
        self,
        spot_history: pd.DataFrame,
    ) -> "ContractCascader":
        """Learn seasonal ratios from EPEX Spot history.

        Computes average quarterly and monthly ratios relative to the annual
        mean, averaged over all available full calendar years.

        The ratios are stored in ``self.seasonal_ratios_`` as::

            {
                "quarter": {1: r_q1, 2: r_q2, 3: r_q3, 4: r_q4},
                "month":   {1: r_m1, 2: r_m2, ..., 12: r_m12},
            }

        where r_qi = mean(spot in quarter i) / mean(spot in year).

        Args:
            spot_history: DataFrame with a ``price_eur_mwh`` column and a
                timezone-aware ``DatetimeIndex`` (UTC). Typically 3+ years
                of EPEX Spot hourly or sub-hourly data.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If the input data is empty or missing required column.
        """
        if spot_history.empty:
            raise ValueError("spot_history is empty")
        if "price_eur_mwh" not in spot_history.columns:
            raise ValueError("spot_history must contain a 'price_eur_mwh' column")

        # Work in local time for quarter/month grouping
        df = spot_history[["price_eur_mwh"]].copy()
        if df.index.tz is None:
            raise ValueError("spot_history index must be timezone-aware (UTC)")

        idx_local = df.index.tz_convert(self.tz)
        df["year"] = idx_local.year
        df["month"] = idx_local.month
        df["quarter"] = idx_local.quarter

        # Only use full calendar years
        year_counts = df.groupby("year").size()
        # A full year has ~8760 quarter-hours (or hours); accept years with >95% coverage
        median_count = year_counts.median()
        full_years = year_counts[year_counts >= 0.95 * median_count].index.tolist()

        if len(full_years) == 0:
            raise ValueError("No full calendar years found in spot_history")

        df_full = df[df["year"].isin(full_years)]
        logger.info(
            "Fitting seasonal ratios from %d full years: %s",
            len(full_years),
            full_years,
        )

        # Compute per-year ratios, then average across years
        quarter_ratios_all: dict[int, list[float]] = {q: [] for q in range(1, 5)}
        month_ratios_all: dict[int, list[float]] = {m: [] for m in range(1, 13)}

        for yr in full_years:
            yr_data = df_full[df_full["year"] == yr]
            yr_mean = yr_data["price_eur_mwh"].mean()

            if yr_mean == 0 or np.isnan(yr_mean):
                logger.warning("Year %d has zero/NaN mean spot price, skipping", yr)
                continue

            for q in range(1, 5):
                q_mean = yr_data[yr_data["quarter"] == q]["price_eur_mwh"].mean()
                if not np.isnan(q_mean):
                    quarter_ratios_all[q].append(q_mean / yr_mean)

            for m in range(1, 13):
                m_mean = yr_data[yr_data["month"] == m]["price_eur_mwh"].mean()
                if not np.isnan(m_mean):
                    month_ratios_all[m].append(m_mean / yr_mean)

        quarter_ratios = {
            q: float(np.mean(vals)) for q, vals in quarter_ratios_all.items() if vals
        }
        month_ratios = {
            m: float(np.mean(vals)) for m, vals in month_ratios_all.items() if vals
        }

        self.seasonal_ratios_ = {
            "quarter": quarter_ratios,
            "month": month_ratios,
        }

        logger.info(
            "Seasonal ratios fitted — Q: [%.3f, %.3f, %.3f, %.3f], "
            "M range: [%.3f .. %.3f]",
            quarter_ratios.get(1, float("nan")),
            quarter_ratios.get(2, float("nan")),
            quarter_ratios.get(3, float("nan")),
            quarter_ratios.get(4, float("nan")),
            min(month_ratios.values()) if month_ratios else float("nan"),
            max(month_ratios.values()) if month_ratios else float("nan"),
        )
        return self

    # ------------------------------------------------------------------
    # Default seasonal ratios (fallback if no spot history available)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_seasonal_ratios() -> dict[str, dict[int, float]]:
        """Return typical Swiss power seasonal ratios as a fallback.

        These are representative long-term averages for Swiss EPEX Spot.
        Winter months are higher (heating demand, lower hydro), summer
        months lower (hydro production, lower demand).
        """
        return {
            "quarter": {1: 1.12, 2: 0.92, 3: 0.88, 4: 1.08},
            "month": {
                1: 1.18, 2: 1.14, 3: 1.04,
                4: 0.94, 5: 0.90, 6: 0.92,
                7: 0.88, 8: 0.86, 9: 0.90,
                10: 1.02, 11: 1.10, 12: 1.12,
            },
        }

    def _get_ratios(self) -> dict[str, dict[int, float]]:
        """Return fitted ratios if available, otherwise defaults."""
        if self.seasonal_ratios_ is not None:
            return self.seasonal_ratios_
        logger.warning(
            "No fitted seasonal ratios — using built-in defaults. "
            "Call fit_seasonal_ratios() with EPEX Spot history for better accuracy."
        )
        return self._default_seasonal_ratios()

    # ------------------------------------------------------------------
    # Cascading logic
    # ------------------------------------------------------------------

    def cascade(
        self,
        base_prices: dict[str, float],
    ) -> dict[str, float]:
        """Fill missing granularities by cascading down the contract hierarchy.

        Processing order:
            1. Year → Quarters (if year exists but quarters are missing)
            2. Quarter → Months (if quarter exists but months are missing)

        The energy conservation constraint is enforced at every level:
            F_parent = sum(F_child_i * h_i) / sum(h_i)

        If a child contract already exists in ``base_prices``, it is kept
        unchanged (market data takes priority over derived values).

        Args:
            base_prices: Dict of forward prices keyed by delivery period
                (e.g. ``{'2026': 75.0, '2026-Q1': 80.0, '2026-03': 82.0}``).

        Returns:
            Enriched copy of ``base_prices`` with all derived prices added.
        """
        result = dict(base_prices)
        ratios = self._get_ratios()

        # --- Step 1: Year → Quarters ---
        years = [
            (yr, price)
            for key, price in base_prices.items()
            if "-Peak" not in key and "-Offpeak" not in key
            for ptype, yr, _ in [parse_key(key)]
            if ptype == "Cal"
        ]

        for year, year_price in years:
            q_keys = [quarter_key(year, q) for q in range(1, 5)]
            existing_qs = [k for k in q_keys if k in result]

            if len(existing_qs) == 4:
                # All quarters already present — skip
                continue

            if len(existing_qs) > 0:
                # Partial quarters present — cascade only the missing ones
                self._cascade_year_partial(
                    year, year_price, result, ratios["quarter"]
                )
            else:
                # No quarters — full cascade
                self._cascade_year_full(
                    year, year_price, result, ratios["quarter"]
                )

        # --- Step 2: Quarter → Months ---
        # Re-scan after quarter generation
        quarters = [
            (yr, q, price)
            for key, price in list(result.items())
            if "-Peak" not in key and "-Offpeak" not in key
            for ptype, yr, q in [parse_key(key)]
            if ptype == "Quarter"
        ]

        for year, q, q_price in quarters:
            months = QUARTER_MONTHS[q]
            m_keys = [month_key(year, m) for m in months]
            existing_ms = [k for k in m_keys if k in result]

            if len(existing_ms) == 3:
                continue

            if len(existing_ms) > 0:
                self._cascade_quarter_partial(
                    year, q, q_price, result, ratios["month"]
                )
            else:
                self._cascade_quarter_full(
                    year, q, q_price, result, ratios["month"]
                )

        # --- Verify energy conservation ---
        self._verify_conservation(result)

        return result

    def _cascade_year_full(
        self,
        year: int,
        year_price: float,
        result: dict[str, float],
        q_ratios: dict[int, float],
    ) -> None:
        """Cascade a calendar year into 4 quarters (none pre-existing).

        Applies seasonal ratios with a correction factor to enforce the
        energy conservation constraint.
        """
        # Compute hours per quarter
        q_hours = {}
        for q in range(1, 5):
            ms, me = _month_range_for_quarter(q)
            total_h, _, _ = count_hours(year, ms, me, self.tz)
            q_hours[q] = total_h

        total_year_hours = sum(q_hours.values())

        # Raw prices from ratios
        raw_prices = {q: year_price * q_ratios.get(q, 1.0) for q in range(1, 5)}

        # Correction factor for energy conservation
        # F_year = sum(F_qi * h_qi) / sum(h_qi)
        weighted_sum = sum(raw_prices[q] * q_hours[q] for q in range(1, 5))
        target_sum = year_price * total_year_hours

        if weighted_sum == 0:
            correction = 1.0
        else:
            correction = target_sum / weighted_sum

        for q in range(1, 5):
            key = quarter_key(year, q)
            price = raw_prices[q] * correction
            result[key] = round(price, 6)
            logger.debug(
                "Cascaded %s → %s: %.4f €/MWh (ratio=%.4f, h=%d)",
                year, key, price, q_ratios.get(q, 1.0), q_hours[q],
            )

        logger.info(
            "Cascaded Cal %d (%.2f €/MWh) → Q1..Q4: [%.2f, %.2f, %.2f, %.2f]",
            year,
            year_price,
            result[quarter_key(year, 1)],
            result[quarter_key(year, 2)],
            result[quarter_key(year, 3)],
            result[quarter_key(year, 4)],
        )

    def _cascade_year_partial(
        self,
        year: int,
        year_price: float,
        result: dict[str, float],
        q_ratios: dict[int, float],
    ) -> None:
        """Cascade a calendar year when some quarters are already known.

        The known quarters are fixed; the missing quarters share the residual
        energy budget, distributed according to their seasonal ratios.
        """
        q_hours = {}
        for q in range(1, 5):
            ms, me = _month_range_for_quarter(q)
            total_h, _, _ = count_hours(year, ms, me, self.tz)
            q_hours[q] = total_h

        total_year_hours = sum(q_hours.values())

        # Energy budget: year_price * total_year_hours
        known_energy = 0.0
        missing_qs = []
        for q in range(1, 5):
            key = quarter_key(year, q)
            if key in result:
                known_energy += result[key] * q_hours[q]
            else:
                missing_qs.append(q)

        residual_energy = year_price * total_year_hours - known_energy

        # Distribute residual according to ratios
        ratio_sum = sum(q_ratios.get(q, 1.0) * q_hours[q] for q in missing_qs)

        for q in missing_qs:
            key = quarter_key(year, q)
            if ratio_sum == 0:
                price = residual_energy / sum(q_hours[qq] for qq in missing_qs)
            else:
                share = q_ratios.get(q, 1.0) * q_hours[q] / ratio_sum
                price = residual_energy * share / q_hours[q]
            result[key] = round(price, 6)
            logger.debug(
                "Cascaded %d (partial) → %s: %.4f €/MWh", year, key, price
            )

    def _cascade_quarter_full(
        self,
        year: int,
        q: int,
        q_price: float,
        result: dict[str, float],
        m_ratios: dict[int, float],
    ) -> None:
        """Cascade a quarter into 3 months (none pre-existing)."""
        months = QUARTER_MONTHS[q]

        # Hours per month
        m_hours = {}
        for m in months:
            total_h, _, _ = count_hours(year, m, m, self.tz)
            m_hours[m] = total_h

        total_q_hours = sum(m_hours.values())

        # Raw prices from ratios — need relative ratios within the quarter
        # Compute quarter-level ratio from monthly ratios
        q_ratio_sum = sum(m_ratios.get(m, 1.0) for m in months)
        if q_ratio_sum == 0:
            q_ratio_sum = len(months)

        raw_prices = {
            m: q_price * (m_ratios.get(m, 1.0) / (q_ratio_sum / len(months)))
            for m in months
        }

        # Correction factor for energy conservation within the quarter
        weighted_sum = sum(raw_prices[m] * m_hours[m] for m in months)
        target_sum = q_price * total_q_hours

        if weighted_sum == 0:
            correction = 1.0
        else:
            correction = target_sum / weighted_sum

        for m in months:
            key = month_key(year, m)
            price = raw_prices[m] * correction
            result[key] = round(price, 6)
            logger.debug(
                "Cascaded %s-Q%d → %s: %.4f €/MWh (h=%d)",
                year, q, key, price, m_hours[m],
            )

        logger.info(
            "Cascaded Q%d-%d (%.2f €/MWh) → [%s]",
            q,
            year,
            q_price,
            ", ".join(
                f"{month_key(year, m)}={result[month_key(year, m)]:.2f}"
                for m in months
            ),
        )

    def _cascade_quarter_partial(
        self,
        year: int,
        q: int,
        q_price: float,
        result: dict[str, float],
        m_ratios: dict[int, float],
    ) -> None:
        """Cascade a quarter when some months are already known."""
        months = QUARTER_MONTHS[q]

        m_hours = {}
        for m in months:
            total_h, _, _ = count_hours(year, m, m, self.tz)
            m_hours[m] = total_h

        total_q_hours = sum(m_hours.values())

        known_energy = 0.0
        missing_ms = []
        for m in months:
            key = month_key(year, m)
            if key in result:
                known_energy += result[key] * m_hours[m]
            else:
                missing_ms.append(m)

        residual_energy = q_price * total_q_hours - known_energy
        ratio_sum = sum(m_ratios.get(m, 1.0) * m_hours[m] for m in missing_ms)

        for m in missing_ms:
            key = month_key(year, m)
            if ratio_sum == 0:
                price = residual_energy / sum(m_hours[mm] for mm in missing_ms)
            else:
                share = m_ratios.get(m, 1.0) * m_hours[m] / ratio_sum
                price = residual_energy * share / m_hours[m]
            result[key] = round(price, 6)
            logger.debug(
                "Cascaded Q%d-%d (partial) → %s: %.4f €/MWh",
                q, year, key, price,
            )

    def _verify_conservation(self, prices: dict[str, float]) -> None:
        """Verify that energy conservation holds at every level.

        Logs warnings if the constraint is violated beyond the tolerance
        of 0.001 EUR/MWh.
        """
        tolerance = 0.001  # €/MWh

        # Check year → quarter conservation
        for key, year_price in list(prices.items()):
            if "-Peak" in key or "-Offpeak" in key:
                continue
            ptype, year, _ = parse_key(key)
            if ptype != "Cal":
                continue

            q_keys = [quarter_key(year, q) for q in range(1, 5)]
            if not all(k in prices for k in q_keys):
                continue

            q_hours = {}
            for q in range(1, 5):
                ms, me = _month_range_for_quarter(q)
                total_h, _, _ = count_hours(year, ms, me, self.tz)
                q_hours[q] = total_h

            total_hours = sum(q_hours.values())
            weighted_avg = sum(
                prices[quarter_key(year, q)] * q_hours[q] for q in range(1, 5)
            ) / total_hours

            error = abs(weighted_avg - year_price)
            if error > tolerance:
                logger.warning(
                    "Energy conservation VIOLATED for Cal %d: "
                    "year_price=%.4f, weighted_avg=%.4f, error=%.6f €/MWh",
                    year, year_price, weighted_avg, error,
                )
            else:
                logger.debug(
                    "Energy conservation OK for Cal %d (error=%.6f €/MWh)",
                    year, error,
                )

        # Check quarter → month conservation
        for key, q_price in list(prices.items()):
            if "-Peak" in key or "-Offpeak" in key:
                continue
            ptype, year, q = parse_key(key)
            if ptype != "Quarter":
                continue

            months = QUARTER_MONTHS[q]
            m_keys = [month_key(year, m) for m in months]
            if not all(k in prices for k in m_keys):
                continue

            m_hours = {}
            for m in months:
                total_h, _, _ = count_hours(year, m, m, self.tz)
                m_hours[m] = total_h

            total_hours = sum(m_hours.values())
            weighted_avg = sum(
                prices[month_key(year, m)] * m_hours[m] for m in months
            ) / total_hours

            error = abs(weighted_avg - q_price)
            if error > tolerance:
                logger.warning(
                    "Energy conservation VIOLATED for %s: "
                    "q_price=%.4f, weighted_avg=%.4f, error=%.6f €/MWh",
                    key, q_price, weighted_avg, error,
                )
            else:
                logger.debug(
                    "Energy conservation OK for %s (error=%.6f €/MWh)",
                    key, error,
                )

    # ------------------------------------------------------------------
    # Peak / Off-Peak decomposition
    # ------------------------------------------------------------------

    @staticmethod
    def offpeak_price(
        base_price: float,
        peak_price: float,
        year: int,
        month_start: int,
        month_end: int,
        tz: str = "Europe/Zurich",
    ) -> float:
        """Compute the off-peak price from base and peak prices.

        Uses the identity:
            Base × total_h = Peak × peak_h + OffPeak × offpeak_h

        Therefore:
            OffPeak = (Base × total_h - Peak × peak_h) / offpeak_h

        Args:
            base_price: Base (all-hours) forward price in EUR/MWh.
            peak_price: Peak forward price in EUR/MWh.
            year: Delivery year.
            month_start: First month of the period.
            month_end: Last month of the period.
            tz: Local timezone.

        Returns:
            Off-peak price in EUR/MWh.

        Raises:
            ValueError: If there are no off-peak hours in the period.
        """
        total_h, peak_h, offpeak_h = count_hours(year, month_start, month_end, tz)

        if offpeak_h == 0:
            raise ValueError(
                f"No off-peak hours in {year}-{month_start:02d}..{month_end:02d}"
            )

        return (base_price * total_h - peak_price * peak_h) / offpeak_h

    # ------------------------------------------------------------------
    # Contract specification builder
    # ------------------------------------------------------------------

    def build_contract_specs(
        self,
        base_prices: dict[str, float],
    ) -> list[ContractSpec]:
        """Convert a base_prices dict into typed ContractSpec objects.

        For each key in ``base_prices``, computes the delivery period
        boundaries, total/peak/off-peak hours, and attaches the price.

        Args:
            base_prices: Dict of forward prices, e.g.
                ``{'2026': 75.0, '2026-Q1': 80.0, '2026-01': 82.0}``.

        Returns:
            List of ContractSpec objects, sorted by delivery start date.
        """
        specs: list[ContractSpec] = []

        for key, price in base_prices.items():
            if "-Peak" in key or "-Offpeak" in key:
                continue
            ptype, year, sub = parse_key(key)

            if ptype == "Cal":
                ms, me = 1, 12
            elif ptype == "Quarter":
                ms, me = _month_range_for_quarter(sub)
            elif ptype == "Month":
                ms, me = sub, sub
            else:
                continue

            start_utc, end_utc = _period_boundaries_utc(year, ms, me, self.tz)
            total_h, peak_h, offpeak_h = count_hours(year, ms, me, self.tz)

            specs.append(
                ContractSpec(
                    key=key,
                    start=start_utc,
                    end=end_utc,
                    product_type=ptype,
                    n_hours=total_h,
                    n_peak_hours=peak_h,
                    n_offpeak_hours=offpeak_h,
                    price=price,
                )
            )

        specs.sort(key=lambda s: s.start)
        return specs

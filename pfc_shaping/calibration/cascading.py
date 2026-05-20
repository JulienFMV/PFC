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
import warnings
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

def _holidays_set(year: int, country: str = "CH") -> set:
    """Return a set of ``datetime.date`` for national public holidays.

    Args:
        year: Calendar year.
        country: 'CH' or 'DE'. For CH uses national-level holidays (no
                 cantonal subdivision) for consistency with EEX peak/off-peak
                 contract definitions.

    Note: ``calendar_ch.py`` uses ``subdiv="VS"`` for *dispatch shape*
    classification (Ferie_CH), which is correct — FMV operates in Valais.
    The distinction matters: VS has extra holidays (e.g. St-Joseph, Ascension
    cantonale) that EEX does not recognise as non-peak.
    """
    if country == "DE":
        return set(holidays.Germany(years=year).keys())
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
    country: str = "CH",
) -> tuple[int, int, int]:
    """Count total, peak, and off-peak hours for a delivery period.

    Correctly handles DST transitions (CET ↔ CEST) and leap years by
    generating the full hourly index in local time.

    Peak hours: 08:00–20:00 on weekdays (Mon–Fri) excluding national holidays.

    Args:
        year: Delivery year.
        month_start: First month of the period (inclusive).
        month_end: Last month of the period (inclusive).
        tz: Local timezone.
        country: 'CH' or 'DE' — controls which holidays are excluded from peak.

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
    hol_set: set = set()
    for y in hol_years:
        hol_set |= _holidays_set(y, country=country)

    # Peak mask: hour 08..19 on weekdays, not a holiday
    is_weekday = idx_local.weekday < 5  # Mon=0 .. Fri=4
    is_peak_hour = (idx_local.hour >= 8) & (idx_local.hour < 20)
    is_holiday = pd.Series(idx_local.date, index=idx_utc).isin(hol_set).values

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

    Phase 5 D-A4-1 / NEG-04 — spread-additive peak synthesis:
        When ``allow_negative_peak=True`` (default, Phase 5 negative-ready),
        ``synthesize_peak_prices`` uses the spread-additive formula
        ``peak = base + spread`` (sign-invariant, D-A4-1). When
        ``allow_negative_peak=False`` (legacy rollback per D-A2-3), uses the
        historical multiplicative ``peak = base × ratio`` with implicit
        ``ratio >= 1`` (i.e., ``Peak >= Base``).

        The ``fit_peak_spreads(spot_history)`` public method must be called to
        populate ``peak_base_spreads_`` before ``synthesize_peak_prices`` is
        invoked on the spread-additive path; absent that, a default spread of
        5.0 €/MWh is used with a WARN log (RESEARCH Pitfall 4).

        The legacy ``fit_peak_ratios`` method is DEPRECATED (Phase 5 D-A4-2)
        with the codex-action-#2 unified shim contract: it ALWAYS delegates to
        ``fit_peak_spreads`` AND populates ``peak_base_ratios_`` derived as
        ``{m: 1.0 + spread / max(base_price_m, 1.0)}`` during the deprecation
        window — legacy attribute readers continue to work.

    Attributes:
        tz: Local timezone for hour counting and peak/off-peak classification.
        seasonal_ratios_: Fitted seasonal ratios, populated by ``fit_seasonal_ratios``.
        allow_negative_peak: When True (default), use spread-additive peak synthesis.
    """

    def __init__(self, tz: str = "Europe/Zurich", allow_negative_peak: bool = True) -> None:
        self.tz = tz
        self.allow_negative_peak = bool(allow_negative_peak)
        self.seasonal_ratios_: dict[str, dict[int, float]] | None = None

    # ------------------------------------------------------------------
    # Fitting seasonal ratios
    # ------------------------------------------------------------------

    def fit_peak_spreads(
        self,
        spot_history: pd.DataFrame,
    ) -> "ContractCascader":
        """Calibrate per-month peak-vs-base spreads in €/MWh (Phase 5 D-A4-1, NEG-04).

        For each month m in 1..12, computes:
            spread_m = mean_t(spot_history[t] | t in peak hours of month m)
                     - mean_t(spot_history[t] | t in base hours of month m)
            base_price_m = mean_t(spot_history[t] | t in base hours of month m)
                           (cached on ``self._base_price_per_month_`` for the
                           codex-action-#2 ratios derivation in ``fit_peak_ratios``)

        Stores ``self.peak_base_spreads_: dict[int, float]`` (keys 1..12, values
        in €/MWh) AND ``self._base_price_per_month_: dict[int, float]``.

        Sign-invariant: when consumed by ``synthesize_peak_prices`` under
        ``allow_negative_peak=True``, the formula ``peak = base + spread``
        produces the correct peak even when ``base < 0`` (negative monthly
        forward), because the spread itself is computed in €/MWh, not as a ratio.

        Codex action #7 fallback behavior: when ``spot_history`` is empty
        (``.empty == True``) OR has fewer than 100 rows (insufficient data),
        emits ``UserWarning`` and populates defaults:
            peak_base_spreads_ = {m: 5.0 for m in 1..12}
            _base_price_per_month_ = {m: 50.0 for m in 1..12}
        The 100-row threshold represents roughly 4 days of hourly data —
        insufficient for reliable monthly statistics.

        Parameters
        ----------
        spot_history : pd.DataFrame
            Historical EPEX-style spot prices with a tz-aware DatetimeIndex
            and a ``price_eur_mwh`` column.

        Returns
        -------
        ContractCascader
            self, for method chaining (matches ``fit_peak_ratios`` API).

        References
        ----------
        D-A4-1, NEG-04, RESEARCH §Architecture Patterns §Pattern 4,
        RESEARCH §Common Pitfalls §Pitfall 4.
        """
        _DEFAULT_SPREAD = 5.0
        _DEFAULT_BASE = 50.0

        # Codex action #7: insufficient data fallback
        if spot_history.empty or len(spot_history) < 100 or "price_eur_mwh" not in spot_history.columns:
            warnings.warn(
                "fit_peak_spreads: spot_history is empty/insufficient — using default "
                f"{_DEFAULT_SPREAD} €/MWh for all months. Provide at least 100 rows of "
                "EPEX hourly data with a 'price_eur_mwh' column for calibrated spreads.",
                UserWarning,
                stacklevel=2,
            )
            self.peak_base_spreads_: dict[int, float] = {m: _DEFAULT_SPREAD for m in range(1, 13)}
            self._base_price_per_month_: dict[int, float] = {m: _DEFAULT_BASE for m in range(1, 13)}
            return self

        df = spot_history[["price_eur_mwh"]].copy()
        if df.index.tz is None:
            warnings.warn(
                "fit_peak_spreads: spot_history index has no timezone — using default spread.",
                UserWarning,
                stacklevel=2,
            )
            self.peak_base_spreads_ = {m: _DEFAULT_SPREAD for m in range(1, 13)}
            self._base_price_per_month_ = {m: _DEFAULT_BASE for m in range(1, 13)}
            return self

        idx_local = df.index.tz_convert(self.tz)
        df["year"] = idx_local.year
        df["month"] = idx_local.month
        is_weekday = idx_local.weekday < 5
        is_peak_hour = (idx_local.hour >= 8) & (idx_local.hour < 20)
        df["is_peak"] = is_weekday & is_peak_hour

        year_counts = df.groupby("year").size()
        median_count = year_counts.median()
        full_years = year_counts[year_counts >= 0.95 * median_count].index.tolist()

        spreads: dict[int, float] = {}
        base_prices_per_month: dict[int, float] = {}

        if full_years:
            df_full = df[df["year"].isin(full_years)]
            spread_acc: dict[int, list[float]] = {m: [] for m in range(1, 13)}
            base_acc: dict[int, list[float]] = {m: [] for m in range(1, 13)}

            for yr in full_years:
                yr_data = df_full[df_full["year"] == yr]
                for m in range(1, 13):
                    m_data = yr_data[yr_data["month"] == m]
                    if len(m_data) < 24:
                        continue
                    base_avg = float(m_data["price_eur_mwh"].mean())
                    peak_data = m_data.loc[m_data["is_peak"], "price_eur_mwh"]
                    if len(peak_data) < 10 or np.isnan(peak_data.mean()):
                        continue
                    peak_avg = float(peak_data.mean())
                    spread_acc[m].append(peak_avg - base_avg)
                    base_acc[m].append(base_avg)

            for m in range(1, 13):
                if spread_acc[m]:
                    spreads[m] = float(np.mean(spread_acc[m]))
                    base_prices_per_month[m] = float(np.mean(base_acc[m]))
                else:
                    spreads[m] = _DEFAULT_SPREAD
                    base_prices_per_month[m] = _DEFAULT_BASE
        else:
            for m in range(1, 13):
                spreads[m] = _DEFAULT_SPREAD
                base_prices_per_month[m] = _DEFAULT_BASE

        self.peak_base_spreads_ = spreads
        self._base_price_per_month_ = base_prices_per_month

        logger.info(
            "Peak/Base spreads fitted — range: [%.2f .. %.2f] €/MWh (mean=%.2f)",
            min(spreads.values()),
            max(spreads.values()),
            float(np.mean(list(spreads.values()))),
        )
        return self

    def fit_peak_ratios(
        self,
        spot_history: pd.DataFrame,
    ) -> "ContractCascader":
        """DEPRECATED — use ``fit_peak_spreads`` (Phase 5 D-A4-2, NEG-04).

        UNIFIED SHIM CONTRACT per codex review action #2 (2026-05-19,
        05-REVIEWS.md lines 96-97, 113-116): this method ALWAYS does THREE things:

          1. Emit ``DeprecationWarning`` pointing to the migration target.
          2. Delegate to ``fit_peak_spreads(spot_history)`` so
             ``peak_base_spreads_`` is populated.
          3. Derive ``peak_base_ratios_`` from ``peak_base_spreads_`` and
             ``self._base_price_per_month_`` (cached by ``fit_peak_spreads``)
             via ``{m: 1.0 + spread / max(base_price_m, 1.0)}``. This ensures
             legacy callers that read ``self.peak_base_ratios_`` directly
             (e.g., tests, dashboards) continue to work during the deprecation
             window — they do NOT see ``AttributeError``.

        Codex action #2 motivation: the original CONTEXT.md D-A4-2 proposed
        ``NotImplementedError`` if ``spot_history`` was absent; RESEARCH
        §Pattern 6 revised to a transparent shim WITHOUT the ratios derivation.
        The codex review noted that the two earlier proposals were inconsistent
        and that LEGACY ATTRIBUTE READERS would silently break without the
        ratios derivation. This UNIFIED SHIM is the authoritative
        deprecation-window behavior.

        Migration: replace ``cascader.fit_peak_ratios(spot)`` with
        ``cascader.fit_peak_spreads(spot)`` and rely on
        ``peak_base_spreads_`` (€/MWh) instead of ``peak_base_ratios_``
        (dimensionless multiplier).

        RESEARCH §Dry-Run §callsite audit identified two callsites in
        ``pfc_shaping/pipeline/production_phases.py`` (lines 344, 644). Both
        are migrated in Plan 05-03 Task 4 (this plan). This shim is the
        safety net for any callsite that has not yet been migrated.

        Args:
            spot_history: DataFrame with ``price_eur_mwh`` column, tz-aware UTC index.

        Returns:
            self, for method chaining.
        """
        warnings.warn(
            "ContractCascader.fit_peak_ratios() is deprecated (Phase 5 D-A4-2). "
            "Use fit_peak_spreads(spot_history) which calibrates spreads in "
            "€/MWh (sign-invariant for negative forwards). Migration: replace "
            "fit_peak_ratios(spot) with fit_peak_spreads(spot). The legacy "
            "peak_base_ratios_ attribute is still populated during the "
            "deprecation window (codex action #2 unified shim contract).",
            DeprecationWarning,
            stacklevel=2,
        )
        # Step 1+2: delegate transparently (populates peak_base_spreads_ AND
        # _base_price_per_month_).
        self.fit_peak_spreads(spot_history)
        # WR-08 (Phase 5 code review): the shim previously relied on
        # fit_peak_spreads always caching ``_base_price_per_month_`` as an
        # implementation detail of its private contract. A future refactor of
        # fit_peak_spreads that drops this cache would surface as an
        # AttributeError here. Defensive fallback to a typical EEX-scale base
        # price (50.0 €/MWh) when the cache is missing — same default as the
        # ``max(., 50.0)`` fallback inside the dict-comprehension below — so
        # the shim remains robust to implementation changes upstream.
        base_per_month = getattr(self, "_base_price_per_month_", None)
        if not base_per_month:
            logger.warning(
                "ContractCascader.fit_peak_ratios shim: _base_price_per_month_ "
                "cache missing after fit_peak_spreads — falling back to a flat "
                "50.0 €/MWh base for every month. Legacy peak_base_ratios_ "
                "values will be approximate (typical EEX scale)."
            )
            base_per_month = {m: 50.0 for m in range(1, 13)}
            # Persist the fallback so subsequent reads see a populated cache.
            self._base_price_per_month_ = base_per_month
        # Step 3 (codex action #2): derive peak_base_ratios_ during deprecation
        # window so legacy attribute readers do not see AttributeError.
        # Formula: ratio_m = 1.0 + spread_m / max(base_price_m, 1.0). The
        # max(., 1.0) avoids division by very small or zero base prices (which
        # would produce unbounded ratios). For typical EEX scales (10-100 €/MWh
        # base), this is a near-no-op safety floor.
        self.peak_base_ratios_: dict[int, float] = {
            m: 1.0 + spread / max(base_per_month.get(m, 50.0), 1.0)
            for m, spread in self.peak_base_spreads_.items()
        }
        return self

    def synthesize_peak_prices(
        self,
        base_prices: dict[str, float],
    ) -> dict[str, float]:
        """Add synthetic Peak forwards where only Base exists.

        Phase 5 D-A4-1 / D-A4-3 / NEG-04 — two paths:
          - ``allow_negative_peak=True`` (default, negative-ready): spread-additive
            formula ``peak = base + spread`` where spread is in €/MWh. Sign-invariant:
            with ``base=-10`` and ``spread=+5``, ``peak=-5`` (NOT the legacy
            ``-10 × 1.05 = -10.5`` which inverts the spread semantic for negative base).
            The modern fitter ``fit_peak_spreads`` does NOT constrain the sign of the
            spread — a negative fitted spread (rare in EEX history but possible) flows
            through unchanged.
          - ``allow_negative_peak=False`` (legacy rollback per D-A4-3): historical
            multiplicative ``peak = base × ratio``. The ratio is NOT enforced to be
            ``>= 1`` — the legacy shim ``fit_peak_ratios`` derives ratios from spreads
            via ``ratio = 1.0 + spread / max(base, 1.0)``, which produces ``ratio < 1``
            whenever the underlying fitted spread is negative. The "Peak >= Base"
            invariant is therefore informational only and CAN be violated by the
            legacy path even when both base and ratio are positive. Additionally, for
            negative base prices, ``peak = base × ratio`` inverts the spread semantic
            (a 1.05 multiplier on a -10 base yields -10.5, i.e. Peak < Base) — callers
            relying on operator rollback with negative base prices should be aware of
            this incompatibility. Existing ``peak_base_ratios_`` cache is honored.

        RESEARCH Pitfall 4: if ``fit_peak_spreads`` was never called on the
        spread-additive path, a default spread of 5.0 €/MWh is used with a WARN log.

        Codex action #2 unified shim: the ``fit_peak_ratios`` deprecation shim
        populates BOTH ``peak_base_spreads_`` AND ``peak_base_ratios_`` during the
        deprecation window, so both paths work regardless of which fit method was called.

        Args:
            base_prices: Dict of forward prices.

        Returns:
            The enriched dict with synthetic Peak entries added.

        References
        ----------
        D-A4-1, D-A4-3, NEG-04, RESEARCH §Architecture Patterns §Pattern 4,
        codex action #2 unified shim contract.
        """
        result = dict(base_prices)
        n_synth = 0

        if self.allow_negative_peak:
            # Phase 5 D-A4-1 spread-additive path (default, negative-ready).
            # RESEARCH Pitfall 4: fall back to default 5.0 €/MWh if fit_peak_spreads
            # was never called.
            spreads = getattr(self, "peak_base_spreads_", None)
            if not spreads:
                logger.warning(
                    "ContractCascader.synthesize_peak_prices: no fitted "
                    "peak_base_spreads_ — using default 5.0 €/MWh for all months. "
                    "Call fit_peak_spreads(spot_history) before synthesize to "
                    "calibrate per-month spreads (RESEARCH Pitfall 4)."
                )
                spreads = {m: 5.0 for m in range(1, 13)}

            for key, price in list(base_prices.items()):
                if "-Peak" in key or "-Offpeak" in key:
                    continue
                peak_key = f"{key}-Peak"
                if peak_key in result:
                    continue  # Already has market peak

                try:
                    ptype, year, sub = parse_key(key)
                except ValueError:
                    continue

                if ptype == "Month" and sub is not None:
                    spread = spreads.get(sub, 5.0)
                elif ptype == "Quarter" and sub is not None:
                    months = QUARTER_MONTHS.get(sub, [])
                    spread = float(np.mean([spreads.get(m, 5.0) for m in months]))
                elif ptype == "Cal":
                    spread = float(np.mean(list(spreads.values())))
                else:
                    continue

                result[peak_key] = round(price + spread, 6)
                n_synth += 1

            if n_synth > 0:
                logger.info("Synthesized %d Peak forwards (spread-additive, Phase 5 D-A4-1)", n_synth)
        else:
            # Legacy multiplicative path (operator rollback per D-A4-3).
            # peak = base * ratio with implicit Peak >= Base (ratio >= 1).
            if not hasattr(self, "peak_base_ratios_") or not self.peak_base_ratios_:
                default_ratio = 1.05  # conservative 5% peak premium
                logger.warning("No fitted peak ratios — using default %.2f", default_ratio)
                ratios = {m: default_ratio for m in range(1, 13)}
            else:
                ratios = self.peak_base_ratios_

            for key, price in list(base_prices.items()):
                if "-Peak" in key or "-Offpeak" in key:
                    continue
                peak_key = f"{key}-Peak"
                if peak_key in result:
                    continue  # Already has market peak

                try:
                    ptype, year, sub = parse_key(key)
                except ValueError:
                    continue

                if ptype == "Month" and sub is not None:
                    ratio = ratios.get(sub, 1.05)
                elif ptype == "Quarter" and sub is not None:
                    months = QUARTER_MONTHS.get(sub, [])
                    ratio = np.mean([ratios.get(m, 1.05) for m in months])
                elif ptype == "Cal":
                    ratio = np.mean(list(ratios.values()))
                else:
                    continue

                result[peak_key] = round(price * ratio, 6)
                n_synth += 1

            if n_synth > 0:
                logger.info("Synthesized %d Peak forwards from historical ratios", n_synth)

        return result

    def fit_seasonal_ratios(
        self,
        spot_history: pd.DataFrame,
    ) -> "ContractCascader":
        """Learn seasonal ratios from EPEX Spot history with optional trends.

        Computes average quarterly and monthly ratios relative to the annual
        mean. If 3+ full years are available, also fits a linear trend per
        month/quarter to capture structural evolution (e.g. increasing solar
        → summer compression, heat pumps → winter elevation).

        The ratios are stored in ``self.seasonal_ratios_`` as::

            {
                "quarter": {1: r_q1, 2: r_q2, 3: r_q3, 4: r_q4},
                "month":   {1: r_m1, 2: r_m2, ..., 12: r_m12},
            }

        Additionally, ``self.seasonal_trends_`` stores per-period linear
        trends (ratio change per year) for forward-looking ratio adjustment.

        Args:
            spot_history: DataFrame with a ``price_eur_mwh`` column and a
                timezone-aware ``DatetimeIndex`` (UTC). Typically 3+ years
                of EPEX Spot hourly or sub-hourly data.

        Returns:
            self, for method chaining.
        """
        if spot_history.empty:
            raise ValueError("spot_history is empty")
        if "price_eur_mwh" not in spot_history.columns:
            raise ValueError("spot_history must contain a 'price_eur_mwh' column")

        df = spot_history[["price_eur_mwh"]].copy()
        if df.index.tz is None:
            raise ValueError("spot_history index must be timezone-aware (UTC)")

        idx_local = df.index.tz_convert(self.tz)
        df["year"] = idx_local.year
        df["month"] = idx_local.month
        df["quarter"] = idx_local.quarter

        year_counts = df.groupby("year").size()
        median_count = year_counts.median()
        full_years = sorted(year_counts[year_counts >= 0.95 * median_count].index.tolist())

        if len(full_years) == 0:
            raise ValueError("No full calendar years found in spot_history")

        df_full = df[df["year"].isin(full_years)]
        logger.info(
            "Fitting seasonal ratios from %d full years: %s",
            len(full_years),
            full_years,
        )

        # Compute per-year ratios
        quarter_ratios_all: dict[int, list[tuple[int, float]]] = {q: [] for q in range(1, 5)}
        month_ratios_all: dict[int, list[tuple[int, float]]] = {m: [] for m in range(1, 13)}

        for yr in full_years:
            yr_data = df_full[df_full["year"] == yr]
            yr_mean = yr_data["price_eur_mwh"].mean()

            if yr_mean == 0 or np.isnan(yr_mean):
                logger.warning("Year %d has zero/NaN mean spot price, skipping", yr)
                continue

            for q in range(1, 5):
                q_mean = yr_data[yr_data["quarter"] == q]["price_eur_mwh"].mean()
                if not np.isnan(q_mean):
                    quarter_ratios_all[q].append((yr, q_mean / yr_mean))

            for m in range(1, 13):
                m_mean = yr_data[yr_data["month"] == m]["price_eur_mwh"].mean()
                if not np.isnan(m_mean):
                    month_ratios_all[m].append((yr, m_mean / yr_mean))

        # Average ratios (backward-compatible)
        quarter_ratios = {
            q: float(np.mean([r for _, r in vals])) for q, vals in quarter_ratios_all.items() if vals
        }
        month_ratios = {
            m: float(np.mean([r for _, r in vals])) for m, vals in month_ratios_all.items() if vals
        }

        self.seasonal_ratios_ = {
            "quarter": quarter_ratios,
            "month": month_ratios,
        }

        # Fit linear trends per period if 3+ years available
        self.seasonal_trends_: dict[str, dict[int, float]] | None = None
        self._reference_year_ = max(full_years) if full_years else None

        if len(full_years) >= 3:
            q_trends: dict[int, float] = {}
            m_trends: dict[int, float] = {}

            for q, vals in quarter_ratios_all.items():
                if len(vals) >= 3:
                    years_arr = np.array([y for y, _ in vals], dtype=float)
                    ratios_arr = np.array([r for _, r in vals], dtype=float)
                    slope = np.polyfit(years_arr, ratios_arr, 1)[0]
                    q_trends[q] = float(slope)

            for m, vals in month_ratios_all.items():
                if len(vals) >= 3:
                    years_arr = np.array([y for y, _ in vals], dtype=float)
                    ratios_arr = np.array([r for _, r in vals], dtype=float)
                    slope = np.polyfit(years_arr, ratios_arr, 1)[0]
                    m_trends[m] = float(slope)

            if q_trends or m_trends:
                self.seasonal_trends_ = {
                    "quarter": q_trends,
                    "month": m_trends,
                }
                logger.info(
                    "Seasonal trends fitted (ref_year=%d) — "
                    "Q trends: [%s], M trends range: [%.4f .. %.4f]",
                    self._reference_year_,
                    ", ".join(f"Q{q}:{t:+.4f}/yr" for q, t in sorted(q_trends.items())),
                    min(m_trends.values()) if m_trends else 0,
                    max(m_trends.values()) if m_trends else 0,
                )

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

    def _get_ratios(self, target_year: int | None = None) -> dict[str, dict[int, float]]:
        """Return fitted ratios, optionally adjusted for target_year trend.

        If ``target_year`` is provided and seasonal trends were fitted,
        adjusts ratios with linear trend extrapolation from the reference
        year (last year of training data). Trend is clamped to avoid
        unreasonable extrapolation (max ±20% change from base ratio).

        Args:
            target_year: Delivery year for trend adjustment (e.g. 2029).
                If None, returns base ratios without trend.
        """
        if self.seasonal_ratios_ is None:
            logger.warning(
                "No fitted seasonal ratios — using built-in defaults. "
                "Call fit_seasonal_ratios() with EPEX Spot history for better accuracy."
            )
            return self._default_seasonal_ratios()

        # No trend adjustment requested or no trends fitted
        if target_year is None or self.seasonal_trends_ is None or self._reference_year_ is None:
            return self.seasonal_ratios_

        years_ahead = target_year - self._reference_year_
        if years_ahead <= 0:
            return self.seasonal_ratios_

        # Apply trend with clamp (max ±20% deviation from base ratio)
        max_deviation = 0.20

        adjusted_q = {}
        for q, base_r in self.seasonal_ratios_["quarter"].items():
            trend = self.seasonal_trends_.get("quarter", {}).get(q, 0.0)
            delta = trend * years_ahead
            delta = np.clip(delta, -max_deviation * base_r, max_deviation * base_r)
            adjusted_q[q] = base_r + delta

        adjusted_m = {}
        for m, base_r in self.seasonal_ratios_["month"].items():
            trend = self.seasonal_trends_.get("month", {}).get(m, 0.0)
            delta = trend * years_ahead
            delta = np.clip(delta, -max_deviation * base_r, max_deviation * base_r)
            adjusted_m[m] = base_r + delta

        return {"quarter": adjusted_q, "month": adjusted_m}

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

        # --- Step 1: Year → Quarters ---
        years = [
            (yr, price)
            for key, price in base_prices.items()
            if "-Peak" not in key and "-Offpeak" not in key
            for ptype, yr, _ in [parse_key(key)]
            if ptype == "Cal"
        ]

        for year, year_price in years:
            ratios = self._get_ratios(target_year=year)
            q_keys = [quarter_key(year, q) for q in range(1, 5)]
            existing_qs = [k for k in q_keys if k in result]

            if len(existing_qs) == 4:
                continue

            if len(existing_qs) > 0:
                self._cascade_year_partial(
                    year, year_price, result, ratios["quarter"]
                )
            else:
                self._cascade_year_full(
                    year, year_price, result, ratios["quarter"]
                )

        # --- Step 2: Quarter → Months ---
        quarters = [
            (yr, q, price)
            for key, price in list(result.items())
            if "-Peak" not in key and "-Offpeak" not in key
            for ptype, yr, q in [parse_key(key)]
            if ptype == "Quarter"
        ]

        for year, q, q_price in quarters:
            ratios = self._get_ratios(target_year=year)
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

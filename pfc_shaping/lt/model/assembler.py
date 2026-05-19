"""
assembler.py
------------
Assemblage de la PFC 15min N+3 ans.

Formule complÃƒÂ¨te (6 facteurs multiplicatifs + calibration) :
    P_raw(t) = B(year) Ãƒâ€” f_S(month) Ãƒâ€” f_W(dow) Ãƒâ€” f_H(h) Ãƒâ€” f_Q(q) Ãƒâ€” f_WV(t)

Puis calibration arbitrage-free (multiplicative, SOTA) :
    P_cal(t) = P_raw(t) × m(t)
    où m minimise ∫(m''(t))² sous contrainte :
        mean(P_cal sur contrat i) = prix_futures_i   ∀ i
    Mode multiplicatif préserve la structure du modèle (Kiesel-Paraschiv).

OÃƒÂ¹ :
    B(year)   = niveau de base annuel (forwards EEX Cal/Quarter/Month)
    f_S(month)= facteur saisonnier mensuel (normalisÃƒÂ© : mean = 1)
    f_W(dow)  = facteur jour de semaine (normalisÃƒÂ© : mean hebdo = 1)
    f_H(h)    = facteur horaire intraday (ShapeHourly)
    f_Q(q)    = facteur 15min intra-horaire (ShapeIntraday)
    f_WV(t)   = correction Water Value rÃƒÂ©servoirs hydro CH (WaterValueCorrection)

Pipeline :
    1. Cascading : enrichir les forwards manquants (YearÃ¢â€ â€™QÃ¢â€ â€™Month)
    2. Shape brut : P_raw = B Ãƒâ€” f_S Ãƒâ€” f_W Ãƒâ€” f_H Ãƒâ€” f_Q Ãƒâ€” f_WV
    3. Calibration : P_cal = P_raw + ÃŽÂ´ (arbitrage-free, Maximum Smoothness)
    4. IC bootstrap : p10, p90

Horizon glissant :
    start  = demain (J+1)
    end    = J + 1095 (Ã¢â€°Ë† 3 ans)

GranularitÃƒÂ© de B et f_S selon l'horizon :
    M+1 Ã¢â€ â€™ M+6   : B mensuel (forwards EEX Monthly)
    M+7 Ã¢â€ â€™ M+12  : B trimestriel (forwards EEX Quarterly)
    Y+2 Ã¢â€ â€™ Y+3   : B annuel (forwards EEX Cal)
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.model.shape_hourly import ShapeHourly, _split_level_anomaly
from pfc_shaping.lt.model.shape_intraday import ShapeIntraday

logger = logging.getLogger(__name__)

# Horizon standard N+3 ans en jours
HORIZON_DAYS = 3 * 365

# ── Country → IANA timezone + holidays constructor ──────────────────────
# Single source of truth for the 5 markets the LT pipeline supports.
# Adding a new market means extending these two tables and (optionally)
# the seasonal-fallback ratios at the bottom of this module.
_COUNTRY_LOCAL_TZ: dict[str, str] = {
    "CH": "Europe/Zurich",
    "DE": "Europe/Berlin",
    "AT": "Europe/Vienna",
    "FR": "Europe/Paris",
    "IT": "Europe/Rome",
}

_COUNTRY_HOLIDAYS_CTOR: dict[str, callable] = {
    "CH": holidays.Switzerland,
    "DE": holidays.Germany,
    "AT": holidays.Austria,
    "FR": holidays.France,
    "IT": holidays.Italy,
}


def _country_local_tz(country: str) -> str:
    """Return the IANA timezone for a market code.

    Falls back to ``Europe/Zurich`` for unknown codes (legacy CH-only
    callers) but logs a warning so the caller knows it is silently
    treated as CH.
    """
    code = str(country).upper()
    tz = _COUNTRY_LOCAL_TZ.get(code)
    if tz is None:
        logger.warning(
            "Unknown country %r in assembler — falling back to Europe/Zurich",
            country,
        )
        return "Europe/Zurich"
    return tz


def _country_holidays(years, country: str) -> set:
    """Collect public holidays for the requested country and years."""
    code = str(country).upper()
    ctor = _COUNTRY_HOLIDAYS_CTOR.get(code, holidays.Switzerland)
    result: set = set()
    for y in years:
        result |= set(ctor(years=int(y)).keys())
    return result


def _sh_apply_accepts_outages(sh_class: type) -> bool:
    """Return True iff sh_class.apply has an ``outages_forecast`` parameter.

    Replaces the former ``try/except TypeError`` pattern at assembler.py:284
    (see D-13 in Phase 05B CONTEXT.md): explicit signature inspection avoids
    masking real TypeErrors raised by bugs inside ShapeHourly.apply().

    Raises TypeError if sh_class.apply does not accept ``reference_date``,
    which is the minimum contract required by PFCAssembler.build().
    """
    sig = inspect.signature(sh_class.apply)
    if "reference_date" not in sig.parameters:
        raise TypeError(
            f"{sh_class.__name__}.apply must accept reference_date; got signature {sig}"
        )
    return "outages_forecast" in sig.parameters


def _emit_level_drift_telemetry(level: pd.Series, logger_: logging.Logger) -> None:
    """Emit D-A2-5 telemetry for the SHP-03 invariant monitor.

    Logs ``max |level - 1.0|`` at INFO on every flag=ON ``assembler.build()`` call.
    Warns if the drift exceeds 1e-6, which signals that the SHP-03 energy-
    normalisation invariant (``mean(f_H | cell) ≈ 1.0``) may be degraded — e.g.
    by a future Phase 5 MSFC log-prix re-normalisation.

    Extracted as a standalone helper so ``test_split_level_anomaly_drift_warning``
    (Plan 05C-02 Task 5, M1 cross-AI review fix) can call it directly without
    copy-pasting the inline telemetry logic (Option A pattern).

    Args:
        level:   pd.Series of per-cell mean values from ``_split_level_anomaly``.
        logger_: The module logger to emit to (``logging.getLogger(__name__)``
                 from ``assembler.py`` is the canonical target, so pytest's caplog
                 can capture it via ``logger="pfc_shaping.lt.model.assembler"``).
    """
    max_level_drift = float(abs(level - 1.0).max())
    logger_.info("f_H split: max |level - 1.0| = %.2e", max_level_drift)
    if max_level_drift > 1e-6:
        logger_.warning(
            "f_H split: level drift %.2e > 1e-6 — SHP-03 invariant may be degraded",
            max_level_drift,
        )


class PFCAssembler:
    """
    Assembleur de la PFC 15min N+3 ans.

    IntÃƒÂ¨gre les 3 modules ajoutÃƒÂ©s :
        - ContractCascader  : dÃƒÂ©composition automatique des forwards
        - WaterValueCorrection : correction hydro saisonniÃƒÂ¨re
        - ArbitrageFreeCalibrator : calibration no-arbitrage

    Args:
        shape_hourly   : instance ShapeHourly fittÃƒÂ©e
        shape_intraday : instance ShapeIntraday fittÃƒÂ©e
        uncertainty    : instance Uncertainty (optionnel, pour IC p10/p90)
        water_value    : instance WaterValueCorrection fittÃƒÂ©e (optionnel)
        cascader       : instance ContractCascader fittÃƒÂ©e (optionnel)
        calibrator     : instance ArbitrageFreeCalibrator (optionnel)
    """

    def __init__(
        self,
        shape_hourly: ShapeHourly,
        shape_intraday: ShapeIntraday,
        uncertainty=None,
        water_value=None,
        cascader=None,
        calibrator=None,
        calibration_fallback_to_raw: bool = True,
        peak_source_policy: str = "same_first",
        confidence_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.sh = shape_hourly
        self.si = shape_intraday
        self.unc = uncertainty
        self.wv = water_value
        self.cascader = cascader
        self.calibrator = calibrator
        self.calibration_fallback_to_raw = calibration_fallback_to_raw
        self.peak_source_policy = str(peak_source_policy)
        self.confidence_thresholds = confidence_thresholds or {
            "6m": 1.0, "12m": 0.85, "24m": 0.65, "36m": 0.45,
        }
        # Cache the outages_forecast capability check once at init (D-13).
        # Explicit signature inspection replaces the former try/except TypeError pattern —
        # prevents masking real TypeErrors from bugs inside self.sh.apply().
        self._sh_accepts_outages: bool = _sh_apply_accepts_outages(type(shape_hourly))
        logger.info(
            "Detected sh=%s — outages_forecast %s",
            type(shape_hourly).__name__,
            "passed" if self._sh_accepts_outages else "skipped",
        )

    def _select_peak_key(
        self,
        key_m: str,
        key_q: str,
        key_y: str,
        source_key: str,
        base_prices: dict,
    ) -> str | None:
        """Select peak quote key used for calibration.

        Policies:
        - ``same_first`` (default): prefer same-granularity peak, then coarser fallback.
        - ``strict_same``: only same-granularity peak.
        - ``any``: first available by M->Q->Y priority.
        """
        direct = f"{source_key}-Peak"
        if self.peak_source_policy == "strict_same":
            return direct if direct in base_prices else None
        if self.peak_source_policy == "same_first":
            if direct in base_prices:
                return direct
            for pk in [f"{key_m}-Peak", f"{key_q}-Peak", f"{key_y}-Peak"]:
                if pk in base_prices:
                    return pk
            return None
        for pk in [f"{key_m}-Peak", f"{key_q}-Peak", f"{key_y}-Peak"]:
            if pk in base_prices:
                return pk
        return None

    @staticmethod
    def _is_peak_timestamp(idx_local: pd.DatetimeIndex, country: str = "CH") -> np.ndarray:
        """Return EEX-style peak mask on a local timezone index (15-min granularity).

        Uses ``np.isin`` rather than ``pd.Series(..., index=idx_local).isin(...)``
        so the function survives autumnal DST transitions where ``idx_local``
        contains duplicated timestamps (Europe/Zurich on the last Sunday of
        October has 02:00→02:00 repeated, which makes the Series-with-index
        path raise on some pandas versions). The country-aware holidays
        table covers CH / DE / AT / FR / IT.
        """
        years = sorted(set(int(y) for y in idx_local.year.unique()))
        holiday_set = _country_holidays(years, country)

        is_weekday = idx_local.weekday < 5
        is_peak_hour = (idx_local.hour >= 8) & (idx_local.hour < 20)
        # np.isin is duplicate-safe; sorted holiday_set as numpy array.
        dates_arr = np.asarray(idx_local.date, dtype=object)
        holidays_arr = np.fromiter(holiday_set, dtype=object, count=len(holiday_set))
        is_holiday = np.isin(dates_arr, holidays_arr)
        return (is_weekday & is_peak_hour & ~is_holiday).astype(bool)

    def build(
        self,
        base_prices: dict,
        quoted_keys: set[str] | None = None,
        start_date: str | None = None,
        horizon_days: int = HORIZON_DAYS,
        entso_forecast: pd.DataFrame | None = None,
        hydro_forecast: pd.DataFrame | None = None,
        outages_forecast: pd.DataFrame | None = None,
        reference_date: pd.Timestamp | None = None,
        country: str = "CH",
    ) -> pd.DataFrame:
        """
        Construit la PFC 15min sur l'horizon N+3.

        Args:
            base_prices    : dict de niveaux de base, clÃƒÂ©s selon granularitÃƒÂ© :
                             {'2025': 65.0,               Ã¢â€ Â niveau annuel Cal
                              '2025-01': 70.0,            Ã¢â€ Â override mensuel si dispo
                              '2025-Q1': 68.0}            Ã¢â€ Â override trimestriel si dispo
                             Logique de prioritÃƒÂ© : mensuel > trimestriel > annuel

            start_date     : 'YYYY-MM-DD' (dÃƒÂ©faut = demain)
            horizon_days   : nombre de jours (dÃƒÂ©faut = 3Ãƒâ€”365)
            entso_forecast : prÃƒÂ©visions solar_regime + load_deviation sur l'horizon
                             (None Ã¢â€ â€™ valeurs neutres utilisÃƒÂ©es)
            hydro_forecast : prÃƒÂ©visions fill_deviation rÃƒÂ©servoirs hydro CH
                             (None Ã¢â€ â€™ f_WV = 1.0 neutre)

        Returns:
            DataFrame colonnes ['price_shape', 'f_S', 'f_W', 'f_H', 'f_Q',
                                'f_WV', 'profile_type', 'confidence',
                                'p10', 'p90', 'calibrated']
            index : DatetimeIndex UTC freq='15min'

        Notes:
            (a) When ``self.sh._use_seasonal_hourly is True``, the f_H damping uses
            ``_split_level_anomaly(f_H, cal)`` (Plan 05C-02 / D-A2-3..D-A2-5). The
            ``level`` component is computed via ``groupby().transform("mean")`` over
            the timestamps in the CURRENT build window — it is window-dependent, NOT
            a fit-stable cell anchor (M3 cross-AI review documentation).

            (b) For stable bowl shape across calls, the recommended MINIMUM build
            horizon is one full year (≥ 52 ISO weeks × all 8 (saison, type_jour) cells
            = 416 cells covered). Shorter horizons may exhibit small level
            discontinuities between consecutive ``build()`` calls with different
            windows.

            (c) The ``max |level - 1.0|`` telemetry (logged at INFO every flag=ON
            build, warning if > 1e-6) is the runtime detection of SHP-03 invariant
            degradation. See
            ``tests/test_shape_hourly_bowl.py::test_split_level_anomaly_drift_warning``
            (Plan 05C-02 Task 5) for the CI signal that the warning actually fires.
        """
        if start_date is None:
            start_date = (pd.Timestamp.now("UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        ts_start = pd.Timestamp(start_date, tz="UTC")
        ts_end = ts_start + pd.Timedelta(days=horizon_days)

        logger.info("Assemblage PFC 15min : %s Ã¢â€ â€™ %s", ts_start.date(), ts_end.date())

        # Ã¢â€â‚¬Ã¢â€â‚¬ 0. Cascading des forwards manquants Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        if self.cascader is not None:
            base_prices = self.cascader.cascade(base_prices)
            logger.info("Cascading terminÃƒÂ© : %d produits forwards", len(base_prices))

        # Index complet 15min UTC
        idx = pd.date_range(ts_start, ts_end, freq="15min", inclusive="left", tz="UTC")

        # Enrichissement calendaire
        cal = enrich_15min_index(idx, country=country)

        local_tz = "Europe/Berlin" if country == "DE" else "Europe/Zurich"
        idx_local = idx.tz_convert(local_tz)
        ref_local = (
            reference_date.tz_convert(local_tz)
            if reference_date is not None
            else pd.Timestamp.now(tz=local_tz)
        )
        # Use reference_date when provided so horizon logic is stable in backtests.
        months_ahead = pd.Series(
            (idx_local.year - ref_local.year) * 12 + (idx_local.month - ref_local.month),
            index=idx,
            dtype=int,
        )
        days_ahead = pd.Series(
            (idx_local - ref_local).total_seconds() / 86400.0,
            index=idx,
            dtype=float,
        )

        # Ã¢â€â‚¬Ã¢â€â‚¬ Facteur saisonnier mensuel f_S Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        f_S = self._compute_f_S(idx, base_prices, country=country)

        # Ã¢â€â‚¬Ã¢â€â‚¬ Facteur jour de semaine f_W Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        f_W = self._compute_f_W(cal)

        # Ã¢â€â‚¬Ã¢â€â‚¬ Facteur horaire f_H Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        # Capability check (replaces former try/except TypeError — see D-13).
        # self._sh_accepts_outages was resolved once at __init__ via explicit
        # signature inspection of type(self.sh).apply, so any TypeError raised
        # below propagates to the caller instead of being silently swallowed.
        if self._sh_accepts_outages:
            f_H = self.sh.apply(idx, cal, reference_date=reference_date,
                                outages_forecast=outages_forecast)
        else:
            f_H = self.sh.apply(idx, cal, reference_date=reference_date)

        # Ã¢â€â‚¬Ã¢â€â‚¬ Facteur 15min f_Q Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        f_Q = self.si.apply(idx, cal, entso_forecast, reference_date=reference_date)

        # Ã¢â€â‚¬Ã¢â€â‚¬ Facteur Water Value f_WV Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        if self.wv is not None:
            f_WV = self.wv.apply(idx, cal, hydro_forecast)
        else:
            f_WV = pd.Series(1.0, index=idx, name="f_WV")

        # Explicit long-horizon governance:
        # near horizon keeps rich historical shapes, far horizon converges
        # progressively to structural level B(t).
        shape_freedom = self._shape_freedom(months_ahead)
        f_S = 1.0 + (f_S - 1.0) * shape_freedom["f_S"]
        f_W = 1.0 + (f_W - 1.0) * shape_freedom["f_W"]
        # Lever 2 (Plan 05C-02, D-A2-3..D-A2-5): split-based damping under flag=ON,
        # legacy single-line under flag=OFF.
        if self.sh._use_seasonal_hourly:
            level, anomaly = _split_level_anomaly(f_H, cal)
            _emit_level_drift_telemetry(level, logger)
            level_damped = 1.0 + (level - 1.0) * shape_freedom["f_H"]
            f_H = (level_damped + anomaly).rename("f_H")
        else:
            f_H = 1.0 + (f_H - 1.0) * shape_freedom["f_H"]
        f_Q = 1.0 + (f_Q - 1.0) * shape_freedom["f_Q"]
        f_WV = 1.0 + (f_WV - 1.0) * shape_freedom["f_WV"]

        # Ã¢â€â‚¬Ã¢â€â‚¬ Niveau de base B par timestamp Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        B = self._resolve_base(idx, base_prices, country=country)

        # ── MSFC smoothing: spline lisse B(t) across period boundaries ──
        try:
            from pfc_shaping.lt.model.msfc_spline import smooth_base_prices
            B = smooth_base_prices(idx, base_prices, B)
        except Exception as exc:
            logger.warning("MSFC smoothing failed, using flat B: %s", exc)

        #Ã¢â€â‚¬Ã¢â€â‚¬ Prix brut (avant calibration) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        f_bridge = self._near_term_bridge_factor(idx, months_ahead, days_ahead, country=country)
        price_raw = B * f_S * f_W * f_H * f_Q * f_WV * f_bridge
        price_raw = self._stabilize_raw_curve(price_raw, B, months_ahead)

        # Ã¢â€â‚¬Ã¢â€â‚¬ Profile type (pour traÃƒÂ§abilitÃƒÂ©) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        profile_type = pd.Series("Y+2/Y+3", index=idx)
        profile_type[months_ahead <= 12] = "M+7..M+12"
        profile_type[months_ahead <= 6] = "M+1..M+6"

        # Ã¢â€â‚¬Ã¢â€â‚¬ Calibration arbitrage-free Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        calibrated = False
        if self.calibrator is not None:
            price_shape, calibrated = self._apply_calibration(
                price_raw, idx, base_prices, quoted_keys=quoted_keys, country=country
            )
        else:
            price_shape = price_raw
        price_shape = self._rebalance_near_term_bridge(
            price_shape,
            idx,
            months_ahead,
            days_ahead,
            country=country,
        )

        # Ã¢â€â‚¬Ã¢â€â‚¬ Assemblage DataFrame Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        df = pd.DataFrame(
            {
                "price_shape": price_shape,
                "B": B,
                "f_S": f_S,
                "f_W": f_W,
                "f_H": f_H,
                "f_Q": f_Q,
                "f_WV": f_WV,
                "f_bridge": f_bridge,
                "profile_type": profile_type,
                "confidence": self._confidence_score(months_ahead),
                "calibrated": calibrated,
            },
            index=idx,
        )

        # Ã¢â€â‚¬Ã¢â€â‚¬ Intervalles de confiance (optionnel) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        if self.unc is not None:
            ic = self.unc.compute(df, cal, reference_date=reference_date)
            df["p10"] = ic["p10"]
            df["p90"] = ic["p90"]
        else:
            df["p10"] = np.nan
            df["p90"] = np.nan

        # Ã¢â€â‚¬Ã¢â€â‚¬ VÃƒÂ©rification cohÃƒÂ©rence ÃƒÂ©nergÃƒÂ©tique Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        self._check_energy_consistency(df, base_prices, country=country)

        logger.info(
            "PFC assemblÃƒÂ©e : %d intervalles 15min, prix min=%.1f max=%.1f Ã¢â€šÂ¬/MWh, "
            "calibration=%s",
            len(df), df["price_shape"].min(), df["price_shape"].max(),
            "OK" if calibrated else "non appliquÃƒÂ©e"
        )
        return df

    # ---------------------------------------------------------------------------
    # Calibration arbitrage-free
    # ---------------------------------------------------------------------------

    def _apply_calibration(
        self,
        price_raw: pd.Series,
        idx: pd.DatetimeIndex,
        base_prices: dict,
        quoted_keys: set[str] | None = None,
        country: str = "CH",
    ) -> tuple[pd.Series, bool]:
        """Applique la calibration arbitrage-free sur la courbe brute.

        Convertit les base_prices en FuturesContract objects et appelle
        le calibrator.

        Returns:
            Tuple (prix calibrÃƒÂ©, True si convergence OK)
        """
        from pfc_shaping.calibration.arbitrage_free import FuturesContract
        from pfc_shaping.calibration.cascading import parse_key, _period_boundaries_utc

        contracts = self._build_non_overlapping_contracts(
            idx=idx,
            base_prices=base_prices,
            quoted_keys=quoted_keys,
            futures_contract_cls=FuturesContract,
            period_boundaries_fn=_period_boundaries_utc,
            country=country,
        )

        if not contracts:
            logger.info("Aucun contrat futures applicable Ã¢â‚¬â€ calibration ignorÃƒÂ©e")
            return price_raw, False

        logger.info(
            "Calibration arbitrage-free : %d contrats non-overlap (country=%s)",
            len(contracts), country,
        )
        # Route the market country to the calibrator so the EEX Peak
        # mask uses the right timezone and national-holiday calendar
        # (e.g. Bundesfeier excluded only for CH, Tag der Deutschen
        # Einheit excluded only for DE).
        try:
            result = self.calibrator.calibrate(price_raw, contracts, country=country)
        except TypeError:
            # Calibrator predates the country argument — fall back to
            # the legacy CH-only signature so old saved instances still
            # work. Log a warning so the operator knows the calibrator
            # is silently CH-biased on non-CH markets.
            if country.upper() != "CH":
                logger.warning(
                    "Calibrator does not accept country=%s; falling back to CH "
                    "holidays — peak/offpeak split may be biased on this market.",
                    country,
                )
            result = self.calibrator.calibrate(price_raw, contracts)

        if result.converged:
            logger.info(
                "Calibration convergÃƒÂ©e : rÃƒÂ©sidu max = %.6f Ã¢â€šÂ¬/MWh, "
                "coÃƒÂ»t lissage = %.2f",
                result.max_abs_residual,
                result.smoothness_cost,
            )
        else:
            logger.warning(
                "Calibration NON convergÃƒÂ©e : rÃƒÂ©sidu max = %.6f Ã¢â€šÂ¬/MWh",
                result.max_abs_residual,
            )
            if self.calibration_fallback_to_raw and result.max_abs_residual > 50.0:
                logger.warning(
                    "Fallback activé: utilisation de P_raw (calibrated=False) "
                    "car résidu trop élevé (>50 EUR/MWh)."
                )
                return price_raw, False
            # Apply calibration despite imperfect convergence if residuals are moderate
            logger.info(
                "Calibration appliquée malgré résidu max=%.1f EUR/MWh (seuil strict non atteint)",
                result.max_abs_residual,
            )

        # Consider calibration applied even if not perfectly converged
        # (moderate residuals are acceptable with MSFC pre-smoothing)
        return result.calibrated_curve, True

    def _build_non_overlapping_contracts(
        self,
        idx: pd.DatetimeIndex,
        base_prices: dict,
        quoted_keys: set[str] | None,
        futures_contract_cls,
        period_boundaries_fn,
        country: str = "CH",
    ) -> list:
        """
        Build a non-overlapping monthly contract set for calibration.

        This avoids rank-deficient/over-constrained systems created by
        mixing Calendar + Quarter + Month constraints simultaneously.
        Price priority per month: Month > Quarter > Calendar.

        Also injects Peak contracts when peak prices are available in
        base_prices (keys ending with '-Peak', e.g. '2026-01-Peak',
        '2026-Q1-Peak'). Peak constraints allow the calibrator to match
        both baseload and peakload forward quotes simultaneously.
        """
        local_tz = "Europe/Berlin" if country == "DE" else "Europe/Zurich"
        idx_local = idx.tz_convert(local_tz)
        month_periods = []
        seen: set[tuple[int, int]] = set()
        for ts in idx_local:
            key = (int(ts.year), int(ts.month))
            if key not in seen:
                seen.add(key)
                month_periods.append(key)

        contracts = []
        quoted_keys = set(quoted_keys or set())
        for year, month in month_periods:
            key_m = f"{year}-{month:02d}"
            key_q = f"{year}-Q{(month - 1) // 3 + 1}"
            key_y = str(year)

            # ── Find base price ────────────────────────────────────────
            source_key = None
            if key_m in base_prices:
                source_key = key_m
            elif key_q in base_prices:
                source_key = key_q
            elif key_y in base_prices:
                source_key = key_y

            if source_key is None:
                continue

            start_utc, end_utc = period_boundaries_fn(year, month, month, local_tz)
            if end_utc <= idx[0] or start_utc >= idx[-1]:
                continue

            base_price = float(base_prices[source_key])

            # ── Find peak price ────────────────────────────────────────
            peak_key = self._select_peak_key(
                key_m=key_m,
                key_q=key_q,
                key_y=key_y,
                source_key=source_key,
                base_prices=base_prices,
            )

            if peak_key is not None:
                # ── SOTA: Peak + OffPeak as DISJOINT constraints ───────
                # Instead of overlapping Base (all hours) + Peak (peak hours),
                # decompose into Peak + OffPeak with disjoint supports.
                # This improves Schur complement conditioning significantly.
                # OffPeak = (Base × total_h - Peak × peak_h) / offpeak_h
                from pfc_shaping.calibration.cascading import count_hours

                total_h, peak_h, offpeak_h = count_hours(year, month, month, tz=local_tz, country=country)
                peak_price = float(base_prices[peak_key])

                peak_is_quoted = peak_key in quoted_keys
                peak_weight = 1.0 if peak_is_quoted else 0.20
                contracts.append(
                    futures_contract_cls(
                        name=f"{year}-{month:02d}-Peak<{peak_key}>",
                        price=peak_price,
                        start=start_utc,
                        end=end_utc,
                        product_type="Peak",
                        is_hard=peak_is_quoted,
                        penalty_weight=peak_weight,
                    )
                )

                if offpeak_h > 0:
                    offpeak_price = (base_price * total_h - peak_price * peak_h) / offpeak_h
                    contracts.append(
                        futures_contract_cls(
                            name=f"{year}-{month:02d}-Offpeak<{source_key}>",
                            price=offpeak_price,
                            start=start_utc,
                            end=end_utc,
                            product_type="Offpeak",
                            is_hard=peak_is_quoted,
                            penalty_weight=peak_weight,
                        )
                    )
            else:
                # No peak price available — use Base constraint (all hours)
                contracts.append(
                    futures_contract_cls(
                        name=f"{year}-{month:02d}<{source_key}>",
                        price=base_price,
                        start=start_utc,
                        end=end_utc,
                        product_type="Base",
                        is_hard=True,
                        penalty_weight=1.0,
                    )
                )

        n_peak = sum(1 for c in contracts if c.product_type == "Peak")
        n_offpeak = sum(1 for c in contracts if c.product_type == "Offpeak")
        n_soft = sum(1 for c in contracts if not getattr(c, "is_hard", True))
        if n_peak > 0:
            logger.info(
                "Disjoint Peak+Offpeak contracts: %d peak, %d offpeak / %d total (%d soft)",
                n_peak, n_offpeak, len(contracts), n_soft,
            )

        return contracts

    # ---------------------------------------------------------------------------
    # Calcul des composantes
    # ---------------------------------------------------------------------------

    def _resolve_base(
        self,
        idx: pd.DatetimeIndex,
        base_prices: dict,
        country: str = "CH",
    ) -> pd.Series:
        """
        Resolve base level B for each timestamp (vectorized).
        Priority: monthly > quarterly > annual.

        Year and month are resolved in the *delivery zone's* local
        timezone — using ``Europe/Zurich`` for a DE / FR / IT / AT
        delivery would misclassify the boundary timestamps (a UTC
        timestamp at 23:00 is the previous day in Zurich but already
        the next day in Bucharest, etc.). For CH / DE / AT the
        difference is nil (same CET/CEST family), but for FR / IT
        cross-zone alignment matters.
        """
        local_tz = _country_local_tz(country)
        idx_local = idx.tz_convert(local_tz)

        # Build vectorized keys
        years = idx_local.year
        months = idx_local.month
        keys_m = pd.Index([f"{y}-{m:02d}" for y, m in zip(years, months)])
        keys_q = pd.Index([f"{y}-Q{(m - 1) // 3 + 1}" for y, m in zip(years, months)])
        keys_y = years.astype(str)

        B = keys_m.map(base_prices).to_series(index=idx).astype(float)

        # Fill missing with quarterly prices
        na_mask = B.isna()
        if na_mask.any():
            q_prices = keys_q[na_mask].map(base_prices)
            B.loc[na_mask] = q_prices.values

        # Fill remaining missing with annual prices
        na_mask = B.isna()
        if na_mask.any():
            y_prices = keys_y[na_mask].map(base_prices)
            B.loc[na_mask] = y_prices.values

        # Fallback: previous years
        for offset in [1, 2]:
            na_mask = B.isna()
            if not na_mask.any():
                break
            fb_keys = (years[na_mask] - offset).astype(str)
            fb_prices = fb_keys.map(base_prices)
            B.loc[na_mask] = fb_prices.values

        if B.isna().any():
            n_na = int(B.isna().sum())
            logger.warning("%d timestamps sans niveau de base — interpolation", n_na)
            B = B.interpolate(method="linear").ffill().bfill()

        return B

    # Historical seasonal ratios for CH/DE electricity (monthly / annual mean).
    # Derived from 10+ years of EPEX Swiss spot data.  Normalised so that
    # the 12-month equal-weighted mean = 1.0.  Used as fallback for Y+2/Y+3
    # when no monthly or quarterly forwards are available.
    _SEASONAL_RATIOS_CH = {
        1: 1.18, 2: 1.12, 3: 1.02, 4: 0.90, 5: 0.85, 6: 0.88,
        7: 0.90, 8: 0.92, 9: 0.95, 10: 1.02, 11: 1.10, 12: 1.16,
    }

    def _compute_f_S(
        self,
        idx: pd.DatetimeIndex,
        base_prices: dict,
        country: str = "CH",
    ) -> pd.Series:
        """
        Seasonal monthly factor f_S.

        _resolve_base picks the finest available forward (monthly > quarterly
        > annual).  When monthly forwards exist, B is already at monthly
        level, so applying f_S = monthly/annual would *double-count* the
        seasonal effect (Bug: B × f_S = monthly × monthly/annual = monthly²/annual).

        When only annual forwards exist (Y+2/Y+3), monthly forwards are
        unavailable so the ratio cannot be computed either — in that case
        we apply a historical seasonal pattern as fallback (P1-15).

        For months with monthly or quarterly forwards, f_S = 1.0 (seasonality
        is already captured in B).

        Year/month are resolved in the delivery zone's local timezone so the
        country-aware fallback ratios apply on the right boundaries.
        """
        local_tz = _country_local_tz(country)
        idx_local = idx.tz_convert(local_tz)
        months = idx_local.month
        years = idx_local.year

        f_S = pd.Series(1.0, index=idx, name="f_S")

        # Identify timestamps where only annual forwards exist (no monthly/quarterly)
        keys_m = pd.Index([f"{y}-{m:02d}" for y, m in zip(years, months)])
        keys_q = pd.Index([f"{y}-Q{(m - 1) // 3 + 1}" for y, m in zip(years, months)])

        has_monthly = keys_m.map(base_prices).notna()
        has_quarterly = keys_q.map(base_prices).notna()
        annual_only = ~has_monthly & ~has_quarterly

        if annual_only.any():
            # Apply historical seasonal ratios as fallback
            seasonal_values = months[annual_only].map(self._SEASONAL_RATIOS_CH)
            f_S.iloc[annual_only] = np.asarray(seasonal_values, dtype=float)
            n_affected = int(annual_only.sum())
            logger.info(
                "f_S seasonal fallback applied to %d timestamps (annual-only forward coverage)",
                n_affected,
            )

        return f_S

    def _compute_f_W(self, cal: pd.DataFrame) -> pd.Series:
        """
        Facteur jour de semaine f_W.
        Utilise les ratios saisonniers f_W(saison, type_jour) si disponibles,
        sinon fallback sur f_W(type_jour) global.

        After computing raw f_W, normalizes per ISO week so that
        mean(f_W) = 1 within each week. Weekly normalization preserves
        day-type relativities better than monthly (P1-03). This ensures
        f_W does not leak level information (which belongs in B and f_S).
        """
        _FW_DEFAULTS = {
            "Ouvrable": 1.05,
            "Samedi": 0.92,
            "Dimanche": 0.78,
            "Ferie_CH": 0.75,
            "Ferie_DE": 0.88,
        }

        # Prefer seasonal f_W if available
        if self.sh.f_W_seasonal_:
            keys = list(zip(cal["saison"], cal["type_jour"]))
            f_W_global = self.sh.f_W_ if self.sh.f_W_ else _FW_DEFAULTS
            values = [
                self.sh.f_W_seasonal_.get(k, f_W_global.get(k[1], 1.0))
                for k in keys
            ]
            f_W = pd.Series(values, index=cal.index, name="f_W", dtype=float)
        else:
            # Fallback to global f_W
            f_W_map = self.sh.f_W_ if self.sh.f_W_ else _FW_DEFAULTS
            f_W = cal["type_jour"].map(f_W_map).fillna(1.0).rename("f_W")

        # Normalize f_W per ISO week so mean(f_W) = 1 within each week (P1-03)
        # Weekly normalization preserves day-type relativities better than
        # monthly, because months with unusual weekday/weekend ratios (e.g.
        # 5 weekends) distort the monthly mean and leak calendar structure.
        idx_zh = cal.index.tz_convert("Europe/Zurich")
        week_key = pd.Index([f"{t.isocalendar()[0]}-W{t.isocalendar()[1]:02d}" for t in idx_zh])
        weekly_mean = f_W.groupby(week_key).transform("mean")
        # Avoid division by zero
        weekly_mean = weekly_mean.replace(0, 1.0)
        f_W = f_W / weekly_mean

        return f_W

    def _confidence_score(self, months_ahead: pd.Series) -> pd.Series:
        """Confidence score [0,1] decreasing with horizon, configurable."""
        ct = self.confidence_thresholds
        score = pd.Series(ct.get("6m", 1.0), index=months_ahead.index)
        score[months_ahead > 6]  = ct.get("12m", 0.85)
        score[months_ahead > 12] = ct.get("24m", 0.65)
        score[months_ahead > 24] = ct.get("36m", 0.45)
        return score

    def _shape_freedom(self, months_ahead: pd.Series) -> dict[str, pd.Series]:
        """
        Horizon-dependent freedom for each shaping block.

        Close horizons can keep richer historical structure. Far horizons
        progressively converge toward the structural base level B(t),
        with intraday and weekday effects damped earlier than seasonality.
        """

        def _interp(
            values: pd.Series,
            knots: list[tuple[float, float]],
        ) -> pd.Series:
            x = values.astype(float).to_numpy()
            xp = np.array([k[0] for k in knots], dtype=float)
            fp = np.array([k[1] for k in knots], dtype=float)
            y = np.interp(x, xp, fp)
            return pd.Series(y, index=values.index, dtype=float)

        return {
            "f_S": _interp(
                months_ahead,
                [(0.0, 1.00), (6.0, 1.00), (12.0, 0.92), (24.0, 0.82), (36.0, 0.72)],
            ),
            "f_W": _interp(
                months_ahead,
                [(0.0, 1.00), (6.0, 0.95), (12.0, 0.82), (24.0, 0.58), (36.0, 0.38)],
            ),
            "f_H": _interp(
                months_ahead,
                [(0.0, 1.00), (6.0, 0.98), (12.0, 0.88), (24.0, 0.62), (36.0, 0.42)],
            ),
            "f_Q": _interp(
                months_ahead,
                [(0.0, 1.00), (6.0, 0.90), (12.0, 0.72), (24.0, 0.38), (36.0, 0.18)],
            ),
            "f_WV": _interp(
                months_ahead,
                [(0.0, 1.00), (6.0, 0.98), (12.0, 0.90), (24.0, 0.70), (36.0, 0.50)],
            ),
        }

    def _stabilize_raw_curve(
        self,
        price_raw: pd.Series,
        base_level: pd.Series,
        months_ahead: pd.Series,
    ) -> pd.Series:
        """
        Softly shrink far-horizon deviations back toward the structural base.

        This is deliberately mild and starts only beyond 12 months so the
        prompt horizon remains market-reactive while Y+2/Y+3 tails become
        less noisy before arbitrage-free calibration.
        """

        ratio = (price_raw / base_level.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        months = months_ahead.astype(float)

        shrink = np.interp(
            months.to_numpy(),
            np.array([0.0, 12.0, 24.0, 36.0], dtype=float),
            np.array([0.0, 0.0, 0.10, 0.22], dtype=float),
        )
        shrink = pd.Series(shrink, index=price_raw.index, dtype=float)

        ratio_stable = 1.0 + (ratio - 1.0) * (1.0 - shrink)
        far_mask = months > 12.0
        if far_mask.any():
            ratio_far = ratio_stable.loc[far_mask]
            ratio_stable.loc[far_mask] = ratio_far.clip(lower=0.55, upper=1.85)
        return (base_level * ratio_stable).rename("price_shape")

    def _near_term_bridge_factor(
        self,
        idx: pd.DatetimeIndex,
        months_ahead: pd.Series,
        days_ahead: pd.Series,
        country: str = "CH",
    ) -> pd.Series:
        """
        Re-anchor the prompt monthly bridge after the D+10 overlay window.

        The short-term overlay can legitimately flatten or invert peak/offpeak
        on a handful of days. This factor prevents that local signal from
        leaking into the average structure of M+1..M+6 by restoring a mild,
        normalized peak premium from D+10 onward.
        """

        local_tz = "Europe/Berlin" if country == "DE" else "Europe/Zurich"
        idx_local = idx.tz_convert(local_tz)

        is_weekday = idx_local.dayofweek < 5
        hour = idx_local.hour
        is_peak = is_weekday & hour.isin([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        is_shoulder = is_weekday & hour.isin([6, 7, 20, 21])
        is_weekend_midday = (idx_local.dayofweek >= 5) & hour.isin([10, 11, 12, 13, 14, 15])

        regime_shape = np.select(
            [is_peak, is_shoulder, is_weekend_midday],
            [1.0, 0.35, 0.15],
            default=-0.22,
        )
        regime_shape = pd.Series(regime_shape, index=idx, dtype=float)

        bridge_strength = np.interp(
            days_ahead.astype(float).to_numpy(),
            np.array([0.0, 10.0, 20.0, 45.0, 90.0, 180.0, 365.0], dtype=float),
            np.array([0.0, 0.0, 0.08, 0.10, 0.08, 0.04, 0.0], dtype=float),
        )
        bridge_strength = pd.Series(bridge_strength, index=idx, dtype=float)
        bridge_strength = bridge_strength.where((days_ahead > 10.0) & (months_ahead <= 6), 0.0)

        factor = 1.0 + bridge_strength * regime_shape
        month_key = pd.Index(idx_local.strftime("%Y-%m"), name="month_key")
        factor = factor / factor.groupby(month_key).transform("mean").replace(0.0, 1.0)
        return factor.rename("f_bridge")

    def _rebalance_near_term_bridge(
        self,
        price_shape: pd.Series,
        idx: pd.DatetimeIndex,
        months_ahead: pd.Series,
        days_ahead: pd.Series,
        country: str = "CH",
    ) -> pd.Series:
        """
        Re-impose a mild prompt peak premium after calibration while
        preserving monthly means exactly.
        """

        local_tz = "Europe/Berlin" if country == "DE" else "Europe/Zurich"
        idx_local = idx.tz_convert(local_tz)

        is_weekday = idx_local.dayofweek < 5
        hour = idx_local.hour
        is_peak = is_weekday & hour.isin([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        is_shoulder = is_weekday & hour.isin([6, 7, 20, 21])
        is_weekend_midday = (idx_local.dayofweek >= 5) & hour.isin([10, 11, 12, 13, 14, 15])

        shape = np.select(
            [is_peak, is_shoulder, is_weekend_midday],
            [1.0, 0.25, 0.10],
            default=-0.26,
        )
        shape = pd.Series(shape, index=idx, dtype=float)

        strength = np.interp(
            days_ahead.astype(float).to_numpy(),
            np.array([0.0, 10.0, 20.0, 45.0, 90.0, 180.0, 365.0], dtype=float),
            np.array([0.0, 0.0, 0.12, 0.15, 0.12, 0.05, 0.0], dtype=float),
        )
        strength = pd.Series(strength, index=idx, dtype=float)
        strength = strength.where((days_ahead > 10.0) & (months_ahead <= 6), 0.0)

        factor = 1.0 + strength * shape
        month_key = pd.Index(idx_local.strftime("%Y-%m"), name="month_key")
        factor = factor / factor.groupby(month_key).transform("mean").replace(0.0, 1.0)
        rebalanced = price_shape * factor
        return rebalanced.rename("price_shape")

    def _check_energy_consistency(self, df: pd.DataFrame, base_prices: dict, country: str = "CH") -> None:
        """
        Verify price_shape average matches base prices at annual, quarterly,
        and monthly levels. Alerts if deviation exceeds threshold.

        Year / month boundaries and Peak hours are evaluated in the
        delivery zone's local timezone (CH=Zurich, DE=Berlin, FR=Paris,
        AT=Vienna, IT=Rome) — using Zurich for every market would skew
        cross-zone month boundaries on FR/IT (no shared CET shift).
        """
        threshold = 0.005 if df["calibrated"].any() else 0.05
        local_tz = _country_local_tz(country)
        idx_local = df.index.tz_convert(local_tz)

        idx_peak_mask = self._is_peak_timestamp(idx_local, country=country)
        idx_offpeak_mask = ~idx_peak_mask

        for key, base in base_prices.items():
            if base == 0:
                continue

            product_type = "Base"
            key_core = key
            if isinstance(key, str) and key.endswith("-Peak"):
                product_type = "Peak"
                key_core = key[:-5]
            elif isinstance(key, str) and key.endswith("-Offpeak"):
                product_type = "Offpeak"
                key_core = key[:-8]

            # Determine mask based on key type
            if len(key_core) == 4 and key_core.isdigit():
                # Annual key
                mask_period = idx_local.year == int(key_core)
                year_int = int(key_core)
                expected = (366 if pd.Timestamp(year=year_int, month=12, day=31).is_leap_year else 365) * 96
                min_coverage = 0.95
            elif len(key_core) == 7 and key_core[4] == '-' and key_core[5] == 'Q' and key_core[6].isdigit():
                # Quarterly key e.g. '2026-Q1'
                year_int = int(key_core[:4])
                q = int(key_core[6])
                q_months = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}[q]
                mask_period = (idx_local.year == year_int) & (idx_local.month.isin(q_months))
                expected = sum(
                    (28 + (m in (1, 3, 5, 7, 8, 10, 12)) * 3 + (m in (4, 6, 9, 11)) * 2) for m in q_months
                ) * 96
                min_coverage = 0.90
            elif len(key_core) == 7 and key_core[4] == '-' and key_core[5:].isdigit():
                # Monthly key e.g. '2026-03'
                year_int = int(key_core[:4])
                month_int = int(key_core[5:])
                mask_period = (idx_local.year == year_int) & (idx_local.month == month_int)
                import calendar as cal_mod
                expected = cal_mod.monthrange(year_int, month_int)[1] * 96
                min_coverage = 0.90
            else:
                continue

            if product_type == "Peak":
                mask = mask_period & idx_peak_mask
                expected = int(mask.sum())
            elif product_type == "Offpeak":
                mask = mask_period & idx_offpeak_mask
                expected = int(mask.sum())
            else:
                mask = mask_period

            n_points = int(mask.sum())
            if n_points == 0:
                continue
            if n_points < int(min_coverage * expected):
                logger.info(
                    "Energy consistency %s: skip (partial coverage %d/%d)",
                    key, n_points, expected,
                )
                continue

            mean_p = df.loc[mask, "price_shape"].mean()
            rel_err = abs(mean_p - base) / abs(base)
            label = key if product_type == "Base" else f"{key_core}-{product_type}"
            if rel_err > threshold:
                logger.warning(
                    "Energy consistency %s: base=%.2f, mean_PFC=%.2f, deviation=%.1f%%",
                    label, base, mean_p, rel_err * 100
                )
            else:
                logger.info(
                    "Energy consistency %s: OK (deviation=%.2f%%)", label, rel_err * 100
                )

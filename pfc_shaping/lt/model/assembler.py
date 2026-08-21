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
import os
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.lt.model.shape_hourly import ShapeHourly, _split_level_anomaly
from pfc_shaping.lt.model.shape_intraday import ShapeIntraday

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase 5 D-A2-2 — master flag PFC_LT_ALLOW_NEGATIVE_PRICES
# ---------------------------------------------------------------------------
_ALLOW_NEG_ENV_VAR = "PFC_LT_ALLOW_NEGATIVE_PRICES"


def _resolve_allow_negative(explicit: bool | None) -> bool:
    """Resolve PFC_LT_ALLOW_NEGATIVE_PRICES — explicit arg wins, else env-var, else False.

    Phase 5 D-A2-2 — mirrors the 5bis-A ``_resolve_flag`` helper in
    ``shape_hourly.py`` for consistency. The flag is freeze-at-init:
    ``PFCAssembler.__init__`` reads the env-var exactly once and stores
    the resolved value on ``self._allow_negative_prices``. Subsequent
    env-var changes do not affect the running assembler.

    Audit-trail-only: this flag does NOT override the four ctor args of
    the sub-components (``enforce_positivity``, ``enforce_m_factor_floor``,
    ``enforce_floor``, ``allow_negative_peak``). It is logged INFO once at
    init for operator traceability (D-A2-2).
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.getenv(_ALLOW_NEG_ENV_VAR, "0")
    if raw == "1":
        return True
    if raw == "0":
        return False
    logger.warning(
        "%s=%r invalide — traité comme False (Phase 5 D-A2-2)",
        _ALLOW_NEG_ENV_VAR, raw,
    )
    return False


# Horizon standard N+3 ans en jours
HORIZON_DAYS = 3 * 365


def _profile_type_labels(
    months_ahead: np.ndarray,
    *,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Return truthful maturity buckets, including horizons beyond Y+3."""

    profile_type = pd.Series("Y+4+", index=index)
    profile_type[months_ahead <= 36] = "Y+2/Y+3"
    profile_type[months_ahead <= 12] = "M+7..M+12"
    profile_type[months_ahead <= 6] = "M+1..M+6"
    return profile_type

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

    Intègre les 3 modules ajoutés :
        - ContractCascader  : décomposition automatique des forwards
        - WaterValueCorrection : correction hydro saisonnière
        - ArbitrageFreeCalibrator : calibration no-arbitrage

    Phase 5 D-A2-2 / D-A2-3 — master flag PFC_LT_ALLOW_NEGATIVE_PRICES is an
    audit-trail INFO log only, NOT an override. The four EXPLICIT FLOOR KWARGS
    on PFCAssembler.__init__ are the authoritative API surface:
      - enforce_positivity: bool = False
          (forwarded to smooth_base_prices(...) at the MSFC callsite in build())
      - enforce_m_factor_floor: bool = False
          (forwarded to self.calibrator.enforce_m_factor_floor)
      - enforce_floor: bool = False
          (forwarded to self.wv.enforce_floor)
      - allow_negative_peak: bool = True
          (forwarded to self.cascader.allow_negative_peak)

    Defaults are negative-ready (D-A2-1). To roll back to legacy positive-only
    behavior, construct:
        PFCAssembler(
            ...,
            enforce_positivity=True,
            enforce_m_factor_floor=True,
            enforce_floor=True,
            allow_negative_peak=False,
        )
    This is the canonical D-A2-3 operator rollback path — testable via
    ``tests/test_phase05_negative_prices.py::test_phase05_baseline_5bisA_via_enforce_true``
    (Plan 05-03 Task 7), which uses these four explicit ctor args.

    Args:
        shape_hourly   : instance ShapeHourly fittée
        shape_intraday : instance ShapeIntraday fittée
        uncertainty    : instance Uncertainty (optionnel, pour IC p10/p90)
        water_value    : instance WaterValueCorrection fittée (optionnel)
        cascader       : instance ContractCascader fittée (optionnel)
        calibrator     : instance ArbitrageFreeCalibrator (optionnel)
        allow_negative_prices : master flag audit-trail (D-A2-2); INFO-log only
        enforce_positivity    : floor for MSFC smooth_base_prices (D-A2-1, default OFF)
        enforce_m_factor_floor: floor for ArbitrageFreeCalibrator (D-A2-1, default OFF)
        enforce_floor         : floor for WaterValueCorrection (D-A2-1, default OFF)
        allow_negative_peak   : spread-additive peak synthesis (D-A2-1, default ON)
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
        allow_negative_prices: bool | None = None,
        enforce_positivity: bool = False,
        enforce_m_factor_floor: bool = False,
        enforce_floor: bool = False,
        allow_negative_peak: bool = True,
        enable_solar_modulation: bool = False,
        solar_penetration_feature=None,
        enable_intraday_amplitude_shrinkage: bool = False,
        enable_electrification_shape: bool = False,
        electrification_scenario: str | None = None,
        electrification_scenario_path: str | Path | None = None,
        require_electrification_scenarios: bool = False,
        require_production_electrification_scenarios: bool = False,
        monthly_level_authority: str = "legacy",
        skip_legacy_level_cascade: bool = False,
        skip_legacy_base_smoothing: bool = False,
        monthly_constraint_tolerance: float = 1e-9,
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

        # Phase 5 — store the four explicit floor kwargs as attributes (B1/B4 Approach B).
        # These are the authoritative API surface for floor enforcement; sub-component
        # ctor args (e.g. wv.enforce_floor) are honored when explicitly set by the
        # caller, but PFCAssembler-level kwargs take precedence on its own forwarding
        # paths (the MSFC callsite at build() and the diagnostic audit log here).
        self.enforce_positivity: bool = bool(enforce_positivity)
        self.enforce_m_factor_floor: bool = bool(enforce_m_factor_floor)
        self.enforce_floor: bool = bool(enforce_floor)
        self.allow_negative_peak: bool = bool(allow_negative_peak)

        # Phase 10 §4quater — solar-aware intra-day shape correction (default OFF).
        # When True, build() applies a multiplicative, exogenous solar-penetration
        # layer on f_H immediately after ShapeHourly.apply() and re-normalises to
        # mean_h=1. Default OFF keeps the pre-solar pipeline byte-identical
        # (Phase-10 atol=1e-12 reproducibility contract). See
        # pfc_shaping/lt/model/solar_modulation.py.
        self.enable_solar_modulation: bool = bool(enable_solar_modulation)
        self.solar_penetration_feature = solar_penetration_feature
        self.enable_intraday_amplitude_shrinkage: bool = bool(
            enable_intraday_amplitude_shrinkage
        )
        self.enable_electrification_shape: bool = bool(enable_electrification_shape)
        self.electrification_scenario = electrification_scenario or "central"
        self.electrification_scenario_path = electrification_scenario_path
        self.require_electrification_scenarios: bool = bool(require_electrification_scenarios)
        self.require_production_electrification_scenarios: bool = bool(
            require_production_electrification_scenarios
        )
        self.monthly_level_authority = str(monthly_level_authority)
        self.skip_legacy_level_cascade = bool(skip_legacy_level_cascade)
        self.skip_legacy_base_smoothing = bool(skip_legacy_base_smoothing)
        self.monthly_constraint_tolerance = float(monthly_constraint_tolerance)

        # B1/B4 Approach B — forward the four floor kwargs to sub-components.
        # WR-03 (Phase 5 code review): the previous one-way mutation
        # (``if self.enforce_m_factor_floor: sub.enforce_m_factor_floor = True``)
        # was non-idempotent and broke shared sub-component scenarios. If a
        # second PFCAssembler was constructed with the floor OFF over the same
        # cascader/calibrator/wv, the sub-component kept the floor that the
        # FIRST assembler had turned ON — silently giving the wrong behaviour.
        #
        # Fix: propagate the kwarg bidirectionally (PFCAssembler is the
        # authoritative source) and emit a WARNING if the sub-component had a
        # different prior value, so operators sharing components across
        # assemblers can detect unintended overrides.
        def _sync_sub_attr(sub, attr_name: str, target: bool, sub_label: str) -> None:
            if sub is None:
                return
            prior = getattr(sub, attr_name, None)
            if prior is not None and bool(prior) != bool(target):
                logger.warning(
                    "PFCAssembler overrides %s.%s: %s -> %s (sub-component shared "
                    "across assemblers may now see the new value).",
                    sub_label, attr_name, prior, target,
                )
            setattr(sub, attr_name, bool(target))

        _sync_sub_attr(self.calibrator, "enforce_m_factor_floor", self.enforce_m_factor_floor, "calibrator")
        _sync_sub_attr(self.wv, "enforce_floor", self.enforce_floor, "wv")
        _sync_sub_attr(self.cascader, "allow_negative_peak", self.allow_negative_peak, "cascader")

        # Phase 5 D-A2-2 master flag — audit-trail INFO only, NOT an override.
        # The four ctor args above are the actual API surface — operator rollback
        # per D-A2-3 = pass enforce_*=True / allow_negative_peak=False explicitly.
        self._allow_negative_prices: bool = _resolve_allow_negative(allow_negative_prices)

        # The audit log resolves each floor state from the PFCAssembler-level
        # kwarg first (authoritative), falling back to the sub-component attribute
        # if explicit forwarding has not been wired. This keeps the log accurate
        # even if a caller bypasses PFCAssembler kwargs and constructs sub-components
        # with enforce_floor=True directly.
        msfc_floor_on = self.enforce_positivity
        af_floor_on = self.enforce_m_factor_floor or bool(getattr(self.calibrator, "enforce_m_factor_floor", False))
        wv_floor_on = self.enforce_floor or bool(getattr(self.wv, "enforce_floor", False))
        cascading_neg_peak_on = self.allow_negative_peak and bool(getattr(self.cascader, "allow_negative_peak", True))

        logger.info(
            "PFC_LT_ALLOW_NEGATIVE_PRICES=%s, floors_disabled={msfc:enforce_positivity=%s, "
            "af:m_factor_floor=%s, wv:floor=%s, cascading:allow_neg_peak=%s}",
            self._allow_negative_prices,
            not msfc_floor_on,      # "floors_disabled" semantic: True when floor is OFF
            not af_floor_on,
            not wv_floor_on,
            cascading_neg_peak_on,  # allow_negative_peak=True == floor OFF, no negation
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
        """Return European EEX Peakload on a local-time index."""
        if str(country).upper() not in _COUNTRY_LOCAL_TZ:
            raise ValueError(f"unsupported EEX peak country: {country!r}")
        is_weekday = idx_local.weekday < 5
        is_peak_hour = (idx_local.hour >= 8) & (idx_local.hour < 20)
        return (is_weekday & is_peak_hour).astype(bool)

    def build(
        self,
        base_prices: dict,
        quoted_keys: set[str] | None = None,
        start_date: str | None = None,
        horizon_days: int = HORIZON_DAYS,
        delivery_index: pd.DatetimeIndex | None = None,
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
        if delivery_index is not None:
            idx = pd.DatetimeIndex(delivery_index)
            if idx.tz is None:
                raise ValueError("delivery_index must be timezone-aware")
            idx = idx.tz_convert("UTC")
            if idx.empty or not idx.is_unique or not idx.is_monotonic_increasing:
                raise ValueError("delivery_index must be non-empty, unique, and sorted")
            if len(idx) > 1 and not bool(
                (idx.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all()
            ):
                raise ValueError("delivery_index must have an exact 15-minute cadence")
            ts_start = idx[0]
            ts_end = idx[-1] + pd.Timedelta(minutes=15)
        else:
            if start_date is None:
                start_date = (pd.Timestamp.now("UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            ts_start = pd.Timestamp(start_date, tz="UTC")
            ts_end = ts_start + pd.Timedelta(days=horizon_days)
            idx = pd.date_range(ts_start, ts_end, freq="15min", inclusive="left", tz="UTC")

        logger.info("Assemblage PFC 15min : %s Ã¢â€ â€™ %s", ts_start.date(), ts_end.date())

        # Ã¢â€â‚¬Ã¢â€â‚¬ 0. Cascading des forwards manquants Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        if self.cascader is not None and not self.skip_legacy_level_cascade:
            base_prices = self.cascader.cascade(base_prices)
            logger.info("Cascading terminÃƒÂ© : %d produits forwards", len(base_prices))

        elif self.skip_legacy_level_cascade:
            logger.info(
                "Legacy forward cascade skipped; monthly_level_authority=%s",
                self.monthly_level_authority,
            )

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
        solver_monthly_level = self.monthly_level_authority.lower() == "solver"
        if solver_monthly_level and (
            not self.skip_legacy_level_cascade or not self.skip_legacy_base_smoothing
        ):
            raise ValueError(
                "monthly_level_authority='solver' requires "
                "skip_legacy_level_cascade=True and skip_legacy_base_smoothing=True"
            )

        # Ã¢â€â‚¬Ã¢â€â‚¬ Facteur saisonnier mensuel f_S Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        if solver_monthly_level:
            f_S = pd.Series(1.0, index=idx, name="f_S")
        else:
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

        # ── Solar-aware intra-day shape correction (Phase 10 §4quater) ──
        # Pure post-processing layer between ShapeHourly.apply() and the rest of
        # the assembler (research §6). Multiplies f_H by
        # (1 + beta[saison, type_jour, block(h)] * (solar_pen_m - baseline)) and
        # re-normalises per local day to mean_h=1. Default OFF (flag False) ⇒
        # byte-identical to the pre-solar pipeline. Applied BEFORE the f_H damping /
        # level-anomaly split below so the SHP-03 invariant (per-cell mean == 1) is
        # preserved going into that stage. reference_date IS the vintage on the
        # Phase-10 build path (build_one passes reference_date=vintage), which the
        # solar layer uses as the strict leak-free training cut-off.
        if getattr(self, "enable_solar_modulation", False):
            from pfc_shaping.lt.model.solar_modulation import solar_modulate
            f_H = solar_modulate(
                f_H,
                cal,
                self.sh,
                vintage=reference_date,
                feature=self.solar_penetration_feature,
            )

        # Ã¢â€â‚¬Ã¢â€â‚¬ Facteur 15min f_Q Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        if getattr(self, "enable_electrification_shape", False):
            from pfc_shaping.lt.model.electrification_shape import electrification_modulate

            f_H = electrification_modulate(
                f_H,
                cal,
                self.sh,
                vintage=reference_date,
                scenario=self.electrification_scenario,
                scenario_path=self.electrification_scenario_path,
                require_scenario_data=self.require_electrification_scenarios,
                require_production_scenario_data=self.require_production_electrification_scenarios,
                country=country,
                tz=local_tz,
            )

        f_Q = self.si.apply(idx, cal, entso_forecast, reference_date=reference_date)

        # Ã¢â€â‚¬Ã¢â€â‚¬ Facteur Water Value f_WV Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        # Phase 5 D-A3-2: delta-additif by default when enforce_floor=False (D-A2-1).
        use_delta_additive_wv = (self.wv is not None) and (not self.wv.enforce_floor)
        if use_delta_additive_wv:
            # Phase 5 delta-additive path. f_WV stays at 1.0 (pass-through) in the
            # multiplicative product; the actual WV contribution is added separately
            # below via compute_delta_wv once B is in scope. The shape_freedom['f_WV']
            # damping below is BYPASSED on this path (RESEARCH Pitfall 2 — the
            # horizon_decay inside WaterValueCorrection.apply() is the single source
            # of truth for far-horizon shrinkage; double-damping would over-flatten
            # the WV correction). delta_wv is computed AFTER B is in scope below.
            f_WV = pd.Series(1.0, index=idx, name="f_WV")
            delta_wv_pending = True
        elif self.wv is None:
            f_WV = pd.Series(1.0, index=idx, name="f_WV")
            delta_wv_pending = False
        else:
            # Legacy multiplicative path (operator rollback via enforce_floor=True
            # per D-A2-3 — preserves the historical F_WV_FLOOR/CAP clip behavior
            # AND the shape_freedom['f_WV'] damping path).
            f_WV = self.wv.apply(idx, cal, hydro_forecast)
            delta_wv_pending = False

        # Explicit long-horizon governance:
        # near horizon keeps rich historical shapes, far horizon converges
        # progressively to structural level B(t).
        shape_freedom = self._shape_freedom(months_ahead)
        if not solver_monthly_level:
            f_S = 1.0 + (f_S - 1.0) * shape_freedom["f_S"]
        f_W = 1.0 + (f_W - 1.0) * shape_freedom["f_W"]
        # Lever 2 (Plan 05C-02, D-A2-3..D-A2-5): split-based damping under flag=ON,
        # legacy single-line under flag=OFF.
        if bool(getattr(self.sh, "_use_seasonal_hourly", False)):
            level, anomaly = _split_level_anomaly(f_H, cal)
            _emit_level_drift_telemetry(level, logger)
            level_damped = 1.0 + (level - 1.0) * shape_freedom["f_H"]
            f_H = (level_damped + anomaly).rename("f_H")
        else:
            f_H = 1.0 + (f_H - 1.0) * shape_freedom["f_H"]
        f_Q = 1.0 + (f_Q - 1.0) * shape_freedom["f_Q"]
        if not delta_wv_pending:
            # Only damp f_WV on the legacy multiplicative path; on the delta-additive
            # path, f_WV is the pass-through 1.0 set above and the damping would be
            # a no-op anyway (1 + (1-1)*sf = 1). The explicit guard documents the
            # intent and avoids RESEARCH Pitfall 2 (double-damping anti-pattern).
            f_WV = 1.0 + (f_WV - 1.0) * shape_freedom["f_WV"]

        # Ã¢â€â‚¬Ã¢â€â‚¬ Niveau de base B par timestamp Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        B = self._resolve_base(idx, base_prices, country=country)

        # ── MSFC smoothing: spline lisse B(t) across period boundaries ──
        # Phase 5 B1/B4 Approach B: enforce_positivity forwarded from PFCAssembler
        # kwarg to smooth_base_prices (propagates to _enforce_mean_constraints
        # per Plan 05-01 RESEARCH Pitfall 1 — TWO floors, both gated by the same flag).
        if self.skip_legacy_base_smoothing:
            logger.info(
                "Legacy BASE MSFC smoothing skipped; monthly_level_authority=%s",
                self.monthly_level_authority,
            )
        else:
            try:
                from pfc_shaping.lt.model.msfc_spline import smooth_base_prices
                B = smooth_base_prices(idx, base_prices, B, enforce_positivity=self.enforce_positivity)
            except Exception as exc:
                logger.warning("MSFC smoothing failed, using flat B: %s", exc)

        if getattr(self, "enable_intraday_amplitude_shrinkage", False):
            from pfc_shaping.lt.model.intraday_amplitude import (
                compress_intraday_peak_amplitude,
            )

            peak_spreads = getattr(self.cascader, "peak_base_spreads_", None)
            f_H = compress_intraday_peak_amplitude(
                f_H,
                cal,
                B,
                peak_spreads,
                tz=local_tz,
            )

        #Ã¢â€â‚¬Ã¢â€â‚¬ Prix brut (avant calibration) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        f_bridge = self._near_term_bridge_factor(idx, months_ahead, days_ahead, country=country)
        if delta_wv_pending:
            # Phase 5 delta-additive WV path (D-A3-2). B is the post-MSFC signed
            # signal (RESEARCH Pitfall 3 — pass B, not price_raw). The compute_delta_wv
            # method internally calls self.wv.apply(B.index, cal, hydro_forecast) so
            # the (unclipped) f_wv is computed identically to the legacy path,
            # then converted to a signed €/MWh delta via (f_wv - 1) * |B|.
            # Codex review action #1 (2026-05-19, see 05-REVIEWS.md):
            # KEYWORD-ONLY call style (fill_df=..., calendar_df=...) is required by
            # the * separator in compute_delta_wv's signature — prevents the easy
            # swap with apply(timestamps, calendar_df, hydro_forecast) order.
            delta_wv = self.wv.compute_delta_wv(B, fill_df=hydro_forecast, calendar_df=cal)
            # Codex action #1 precondition guard: an index misalignment between
            # delta_wv and B would either propagate NaN through arithmetic or
            # silently broadcast the indices, both producing wrong PFC outputs.
            # WR-02 (Phase 5 code review): use explicit ValueError rather than
            # ``assert`` because ``assert`` statements are stripped under
            # ``python -O`` — a common production setting in containerised
            # deployments where this precondition would silently disappear.
            if not delta_wv.index.equals(B.index):
                raise ValueError(
                    f"compute_delta_wv returned mismatched index: "
                    f"delta_wv.index has {len(delta_wv.index)} entries, "
                    f"B.index has {len(B.index)} entries; "
                    f"first 3 delta_wv: {list(delta_wv.index[:3])}; "
                    f"first 3 B: {list(B.index[:3])}"
                )
            price_raw = B * f_S * f_W * f_H * f_Q * f_bridge + delta_wv
            # D-A3-5 telemetry — emitted ONCE per build call on the delta-additive path.
            sign_flips = int((np.sign(B) != np.sign(B.shift(1))).fillna(False).sum())
            logger.info(
                "WV delta_wv: min=%.2f, max=%.2f, mean=%.2f €/MWh, sign(B) flips: %d",
                float(delta_wv.min()), float(delta_wv.max()), float(delta_wv.mean()),
                sign_flips,
            )
        else:
            # Legacy multiplicative path (wv is None OR wv.enforce_floor=True).
            delta_wv = pd.Series(0.0, index=idx, name="delta_wv")
            price_raw = B * f_S * f_W * f_H * f_Q * f_WV * f_bridge
        if solver_monthly_level:
            price_raw = self._preserve_monthly_base_means(price_raw, B, idx, country=country)
        else:
            price_raw = self._stabilize_raw_curve(price_raw, B, months_ahead)

        # Ã¢â€â‚¬Ã¢â€â‚¬ Profile type (pour traÃƒÂ§abilitÃƒÂ©) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        profile_type = _profile_type_labels(months_ahead, index=idx)

        # Ã¢â€â‚¬Ã¢â€â‚¬ Calibration arbitrage-free Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        calibrated = False
        if self.calibrator is not None and not solver_monthly_level:
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
        if getattr(self, "enable_intraday_amplitude_shrinkage", False):
            from pfc_shaping.lt.model.intraday_amplitude import compress_price_peak_amplitude

            peak_spreads = getattr(self.cascader, "peak_base_spreads_", None)
            price_shape = compress_price_peak_amplitude(
                price_shape,
                cal,
                peak_spreads,
                tz=local_tz,
            )

        if solver_monthly_level:
            price_shape = self._preserve_monthly_base_means(price_shape, B, idx, country=country)
            price_pre_final_projection = price_shape.copy()
            price_shape, calibrated = self._project_final_solver_products(
                price_shape,
                idx=idx,
                base_prices=base_prices,
                quoted_keys=quoted_keys,
                country=country,
            )
            delta_final_product_projection = (
                price_shape - price_pre_final_projection
            ).rename("delta_final_product_projection")

        df = pd.DataFrame(
            {
                "price_shape": price_shape,
                "B": B,
                "f_S": f_S,
                "f_W": f_W,
                "f_H": f_H,
                "f_Q": f_Q,
                "f_WV": f_WV,
                "delta_wv": delta_wv,
                "f_bridge": f_bridge,
                "profile_type": profile_type,
                "confidence": self._confidence_score(months_ahead),
                "calibrated": calibrated,
            },
            index=idx,
        )
        if solver_monthly_level:
            df["price_pre_final_projection"] = price_pre_final_projection
            df["delta_final_product_projection"] = (
                delta_final_product_projection
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
        from pfc_shaping.calibration.cascading import _period_boundaries_utc

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
        # Route the market country for the local EEX Peakload timezone.
        # Public holidays remain in the contractual delivery window.
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
                    "timezone — peak/offpeak split may be biased on this market.",
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

        # WR-07 (Phase 5 code review): propagate result.converged faithfully.
        # Previously this returned a hard-coded True whenever the curve was
        # applied (even if convergence was False under the moderate-residual
        # path), which broke the CalibrationResult.converged contract in
        # downstream consumers and masked the NEG-02 floor-induced
        # non-convergence signal. The two prior log paths (converged True
        # vs False) already inform the operator; let the boolean trace through.
        return result.calibrated_curve, bool(result.converged)

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
        if self.skip_legacy_level_cascade and quoted_keys:
            return self._build_monthly_solver_contracts(
                idx=idx,
                base_prices=base_prices,
                quoted_keys=quoted_keys,
                futures_contract_cls=futures_contract_cls,
                period_boundaries_fn=period_boundaries_fn,
                country=country,
                local_tz=local_tz,
            )
        for year, month in month_periods:
            key_m = f"{year}-{month:02d}"
            key_q = f"{year}-Q{(month - 1) // 3 + 1}"
            key_y = str(year)

            # ── Find base price ────────────────────────────────────────
            source_key = self._select_base_contract_key(
                key_m=key_m,
                key_q=key_q,
                key_y=key_y,
                base_prices=base_prices,
                quoted_keys=quoted_keys,
            )

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

    def _build_monthly_solver_contracts(
        self,
        *,
        idx: pd.DatetimeIndex,
        base_prices: dict,
        quoted_keys: set[str],
        futures_contract_cls,
        period_boundaries_fn,
        country: str,
        local_tz: str,
    ) -> list:
        from pfc_shaping.calibration.monthly_forward_curve import (
            MarketQuote,
            build_monthly_constraint_system,
            product_periods,
        )

        quote_products = []
        delivery_months: set[pd.Period] = set()
        for key in sorted(quoted_keys):
            suffix = str(key).rsplit("-", 1)[-1].lower()
            if key not in base_prices or suffix in {"peak", "offpeak"}:
                continue
            try:
                months = product_periods(str(key))
            except ValueError:
                continue
            delivery_months.update(months.tolist())
            quote_products.append(
                MarketQuote(
                    market=country,
                    product=str(key),
                    load_type="BASE",
                    price=float(base_prices[key]),
                    source="assembler_monthly_solver_quoted_key",
                )
            )
        if not quote_products or not delivery_months:
            return []

        constraints = build_monthly_constraint_system(
            pd.PeriodIndex(sorted(delivery_months), freq="M"),
            tuple(quote_products),
            timezone=local_tz,
            market=country,
            load_type="BASE",
            constraint_tolerance=getattr(self, "monthly_constraint_tolerance", 1e-9),
        )
        contracts = []
        idx_step = (idx[1] - idx[0]) if len(idx) > 1 else pd.Timedelta(minutes=15)
        idx_end_exclusive = idx[-1] + idx_step
        for _, row in constraints.rows.iterrows():
            product = str(row["product"])
            bucket_months = constraints.month_buckets[constraints.month_buckets.eq(product)].index
            if len(bucket_months) == 0:
                continue
            first = bucket_months.min()
            last = bucket_months.max()
            start_utc, _ = period_boundaries_fn(int(first.year), int(first.month), int(first.month), local_tz)
            _, end_utc = period_boundaries_fn(int(last.year), int(last.month), int(last.month), local_tz)
            if start_utc < idx[0] or end_utc > idx_end_exclusive:
                logger.info(
                    "Skipping partial monthly-solver calibration row %s: row=[%s,%s), index=[%s,%s)",
                    product,
                    start_utc,
                    end_utc,
                    idx[0],
                    idx_end_exclusive,
                )
                continue
            contracts.append(
                futures_contract_cls(
                    name=f"{product}<monthly_solver:{row['parent_product']}>",
                    price=float(row["target"]),
                    start=start_utc,
                    end=end_utc,
                    product_type="Base",
                    is_hard=True,
                    penalty_weight=1.0,
                )
            )
        logger.info("Monthly solver final calibration contracts: %d BASE rows", len(contracts))
        return contracts

    @staticmethod
    def _select_base_contract_key(
        *,
        key_m: str,
        key_q: str,
        key_y: str,
        base_prices: dict,
        quoted_keys: set[str],
    ) -> str | None:
        """Select the most granular BASE key on the legacy calibration path.

        Solver-owned levels are handled by ``_build_monthly_solver_contracts``
        before this helper is reached. Preserving Month > Quarter > Calendar
        here is therefore required for legacy cascaded-shape parity.
        """
        del quoted_keys
        candidates = [key_m, key_q, key_y]
        for key in candidates:
            if key in base_prices:
                return key
        return None

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

    def _preserve_monthly_base_means(
        self,
        price_raw: pd.Series,
        base_level: pd.Series,
        idx: pd.DatetimeIndex,
        country: str = "CH",
    ) -> pd.Series:
        """
        Recenter hourly shape so solver-owned monthly BASE means equal B.

        The monthly forward solver is the level authority when enabled. Hourly,
        weekday, intraday and water-value terms may shape within each month, but
        they must not change the all-hour monthly mean before final calibration
        to the original CH traded products.
        """

        local_tz = _country_local_tz(country)
        month_key = pd.Index(idx.tz_convert(local_tz).strftime("%Y-%m"), name="month_key")
        raw_month_mean = price_raw.groupby(month_key).transform("mean")
        base_month_mean = base_level.groupby(month_key).transform("mean")
        return (price_raw + base_month_mean - raw_month_mean).rename("price_shape")

    def _project_final_solver_products(
        self,
        price_shape: pd.Series,
        *,
        idx: pd.DatetimeIndex,
        base_prices: dict,
        quoted_keys: set[str] | None,
        country: str,
    ) -> tuple[pd.Series, bool]:
        """Project the final solver-owned shape onto hard BASE/PEAK products.

        This is deliberately the last mutation of ``price_shape``. Synthetic
        solver months are hard BASE level constraints; only PEAK keys present
        in ``quoted_keys`` are eligible market constraints. The disjoint
        PEAK/OFFPEAK representation preserves each monthly BASE mean while
        enforcing accepted PEAK quotes.
        """

        from pfc_shaping.lt.model.quant_shape_optimizer import QuantShapeOptimizer
        from pfc_shaping.lt.model.shape_constraints import (
            build_base_peak_offpeak_constraint_system,
        )

        monthly_base_prices: dict[str, float] = {}
        for raw_key, value in base_prices.items():
            key = str(raw_key)
            if len(key) == 7 and key[4] == "-" and key[5:].isdigit():
                monthly_base_prices[key] = float(value)
        if not monthly_base_prices:
            raise ValueError(
                "monthly_level_authority='solver' requires synthetic monthly BASE levels "
                "for final product projection"
            )

        accepted_keys = {str(key) for key in (quoted_keys or set())}
        peak_prices: dict[str, float] = {}
        partial_peak_quotes: list[str] = []
        out_of_scope_peak_quotes: list[str] = []
        for raw_key in accepted_keys:
            lowered = raw_key.lower()
            if lowered.endswith("-offpeak"):
                raise ValueError(
                    f"explicit OFFPEAK quote {raw_key!r} is not supported by the solver projection; "
                    "provide the approved BASE/PEAK hierarchy and audit implied OFFPEAK"
                )
            if not lowered.endswith("-peak") or raw_key not in base_prices:
                continue
            product = raw_key[:-5]
            if self._product_has_full_coverage(idx, product, country=country):
                peak_prices[product] = float(base_prices[raw_key])
            elif self._product_has_any_coverage(idx, product, country=country):
                partial_peak_quotes.append(raw_key)
            else:
                out_of_scope_peak_quotes.append(raw_key)

        if partial_peak_quotes:
            raise ValueError(
                "quoted PEAK products partially overlap the delivered artifact and cannot be "
                f"repriced exactly: {sorted(partial_peak_quotes)}"
            )

        constraints = build_base_peak_offpeak_constraint_system(
            idx,
            monthly_base_prices,
            peak_prices,
            country=country,
        )
        optimizer = QuantShapeOptimizer(
            lambda_prior=1.0,
            lambda_smooth_h=0.0,
            lambda_smooth_m=0.0,
            lambda_seam=0.0,
            epsilon_ridge=0.0,
            feasibility_tol=min(
                float(getattr(self, "monthly_constraint_tolerance", 1e-9)),
                1e-9,
            ),
            stationarity_tol=1e-7,
        )
        result = optimizer.solve(price_shape.rename("final_shape_prior"), constraints)
        max_abs_error = (
            float(result.constraint_residuals["abs_error"].max())
            if not result.constraint_residuals.empty
            else 0.0
        )
        if not np.isfinite(max_abs_error) or max_abs_error > 1e-6:
            raise ValueError(
                "final BASE/PEAK/OFFPEAK projection residual exceeds hard tolerance: "
                f"{max_abs_error:.3e} EUR/MWh"
            )
        self.final_product_projection_report_ = {
            "constraint_count": int(len(constraints.rows)),
            "base_constraint_count": int(sum(row.kind == "BASE" for row in constraints.rows)),
            "peak_constraint_count": int(sum(row.kind == "PEAK" for row in constraints.rows)),
            "offpeak_constraint_count": int(sum(row.kind == "OFFPEAK" for row in constraints.rows)),
            "max_abs_error_eur_mwh": max_abs_error,
            "primal_inf": float(result.kkt.primal_inf),
            "stationarity_inf": float(result.kkt.stationarity_inf),
            "partial_peak_quote_ids": sorted(partial_peak_quotes),
            "out_of_scope_peak_quote_ids": sorted(out_of_scope_peak_quotes),
        }
        logger.info(
            "Final solver product projection: constraints=%d, PEAK=%d, OFFPEAK=%d, max_error=%.3e",
            len(constraints.rows),
            self.final_product_projection_report_["peak_constraint_count"],
            self.final_product_projection_report_["offpeak_constraint_count"],
            max_abs_error,
        )
        return result.curve.rename("price_shape"), True

    @staticmethod
    def _product_has_full_coverage(
        idx: pd.DatetimeIndex,
        product: str,
        *,
        country: str,
    ) -> bool:
        from pfc_shaping.calibration.monthly_forward_curve import product_periods

        try:
            months = product_periods(str(product))
        except ValueError:
            return False
        if len(months) == 0 or len(idx) == 0:
            return False
        local_tz = _country_local_tz(country)
        first = months.min()
        last = months.max()
        start = pd.Timestamp(year=int(first.year), month=int(first.month), day=1, tz=local_tz).tz_convert("UTC")
        if int(last.month) == 12:
            end = pd.Timestamp(year=int(last.year) + 1, month=1, day=1, tz=local_tz).tz_convert("UTC")
        else:
            end = pd.Timestamp(year=int(last.year), month=int(last.month) + 1, day=1, tz=local_tz).tz_convert("UTC")
        step = idx[1] - idx[0] if len(idx) > 1 else pd.Timedelta(minutes=15)
        return bool(idx[0] <= start and idx[-1] + step >= end)

    @staticmethod
    def _product_has_any_coverage(
        idx: pd.DatetimeIndex,
        product: str,
        *,
        country: str,
    ) -> bool:
        from pfc_shaping.calibration.monthly_forward_curve import product_periods

        try:
            months = product_periods(str(product))
        except ValueError:
            return False
        if len(months) == 0 or len(idx) == 0:
            return False
        local_tz = _country_local_tz(country)
        first = months.min()
        last = months.max()
        start = pd.Timestamp(year=int(first.year), month=int(first.month), day=1, tz=local_tz).tz_convert("UTC")
        if int(last.month) == 12:
            end = pd.Timestamp(year=int(last.year) + 1, month=1, day=1, tz=local_tz).tz_convert("UTC")
        else:
            end = pd.Timestamp(
                year=int(last.year), month=int(last.month) + 1, day=1, tz=local_tz
            ).tz_convert("UTC")
        return bool(idx[-1] >= start and idx[0] < end)

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
        # ``f_Q`` is the sole 15-minute layer inside a parent hour.  Evaluating
        # the horizon interpolation independently at each quarter introduced a
        # small covariance between ``f_bridge`` and mean-one ``f_Q``.  The
        # resulting parent-hour price drift was measurable even though the
        # factor gate passed.  Freeze the bridge strength at the delivery-hour
        # start so the hierarchy is price-neutral before monthly projection.
        bridge_strength = bridge_strength.groupby(idx.floor("h")).transform("first")

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
        # Keep the post-calibration bridge on the same parent-hour grid as the
        # pre-calibration bridge.  This prevents a later layer from silently
        # reintroducing quarter-hour structure outside ``f_Q``.
        strength = strength.groupby(idx.floor("h")).transform("first")

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

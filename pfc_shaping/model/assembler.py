"""
assembler.py
------------
Assemblage de la PFC 15min N+3 ans.

Formule complÃƒÂ¨te (6 facteurs multiplicatifs + calibration) :
    P_raw(t) = B(year) Ãƒâ€” f_S(month) Ãƒâ€” f_W(dow) Ãƒâ€” f_H(h) Ãƒâ€” f_Q(q) Ãƒâ€” f_WV(t)

Puis calibration arbitrage-free :
    P_cal(t) = P_raw(t) + ÃŽÂ´(t)
    oÃƒÂ¹ ÃŽÂ´ minimise Ã¢Ë†Â«(ÃŽÂ´''(t))Ã‚Â² sous contrainte :
        mean(P_cal sur contrat i) = prix_futures_i   Ã¢Ë†â‚¬ i

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

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.model.shape_hourly import ShapeHourly
from pfc_shaping.model.shape_intraday import ShapeIntraday

logger = logging.getLogger(__name__)

# Horizon standard N+3 ans en jours
HORIZON_DAYS = 3 * 365


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
        confidence_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.sh = shape_hourly
        self.si = shape_intraday
        self.unc = uncertainty
        self.wv = water_value
        self.cascader = cascader
        self.calibrator = calibrator
        self.calibration_fallback_to_raw = calibration_fallback_to_raw
        self.confidence_thresholds = confidence_thresholds or {
            "6m": 1.0, "12m": 0.85, "24m": 0.65, "36m": 0.45,
        }

    def build(
        self,
        base_prices: dict,
        start_date: str | None = None,
        horizon_days: int = HORIZON_DAYS,
        entso_forecast: pd.DataFrame | None = None,
        hydro_forecast: pd.DataFrame | None = None,
        reference_date: pd.Timestamp | None = None,
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
        """
        if start_date is None:
            start_date = (pd.Timestamp.utcnow() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

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
        cal = enrich_15min_index(idx)

        # Ã¢â€â‚¬Ã¢â€â‚¬ Facteur saisonnier mensuel f_S Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        f_S = self._compute_f_S(idx, base_prices)

        # Ã¢â€â‚¬Ã¢â€â‚¬ Facteur jour de semaine f_W Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        f_W = self._compute_f_W(cal)

        # Ã¢â€â‚¬Ã¢â€â‚¬ Facteur horaire f_H Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        f_H = self.sh.apply(idx, cal)

        # Ã¢â€â‚¬Ã¢â€â‚¬ Facteur 15min f_Q Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        f_Q = self.si.apply(idx, cal, entso_forecast)

        # Ã¢â€â‚¬Ã¢â€â‚¬ Facteur Water Value f_WV Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        if self.wv is not None:
            f_WV = self.wv.apply(idx, cal, hydro_forecast)
        else:
            f_WV = pd.Series(1.0, index=idx, name="f_WV")

        # Ã¢â€â‚¬Ã¢â€â‚¬ Niveau de base B par timestamp Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        B = self._resolve_base(idx, base_prices)

        # Ã¢â€â‚¬Ã¢â€â‚¬ Prix brut (avant calibration) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        price_raw = B * f_S * f_W * f_H * f_Q * f_WV

        # Ã¢â€â‚¬Ã¢â€â‚¬ Profile type (pour traÃƒÂ§abilitÃƒÂ©) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        idx_zurich = idx.tz_convert("Europe/Zurich")
        now_zurich = pd.Timestamp.now(tz="Europe/Zurich")
        # Robust month offset computation compatible with modern pandas Index API.
        months_ahead = pd.Series(
            (idx_zurich.year - now_zurich.year) * 12 + (idx_zurich.month - now_zurich.month),
            index=idx,
            dtype=int,
        )

        profile_type = pd.Series("Y+2/Y+3", index=idx)
        profile_type[months_ahead <= 12] = "M+7..M+12"
        profile_type[months_ahead <= 6] = "M+1..M+6"

        # Ã¢â€â‚¬Ã¢â€â‚¬ Calibration arbitrage-free Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
        calibrated = False
        if self.calibrator is not None:
            price_shape, calibrated = self._apply_calibration(
                price_raw, idx, base_prices
            )
        else:
            price_shape = price_raw

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
        self._check_energy_consistency(df, base_prices)

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
            futures_contract_cls=FuturesContract,
            period_boundaries_fn=_period_boundaries_utc,
        )

        if not contracts:
            logger.info("Aucun contrat futures applicable Ã¢â‚¬â€ calibration ignorÃƒÂ©e")
            return price_raw, False

        logger.info(
            "Calibration arbitrage-free : %d contrats non-overlap", len(contracts)
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
            if self.calibration_fallback_to_raw:
                logger.warning(
                    "Fallback activÃ©: utilisation de P_raw (calibrated=False) "
                    "car calibration non convergÃ©e."
                )
                return price_raw, False

        return result.calibrated_curve, result.converged

    def _build_non_overlapping_contracts(
        self,
        idx: pd.DatetimeIndex,
        base_prices: dict,
        futures_contract_cls,
        period_boundaries_fn,
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
        idx_local = idx.tz_convert("Europe/Zurich")
        month_periods = []
        seen: set[tuple[int, int]] = set()
        for ts in idx_local:
            key = (int(ts.year), int(ts.month))
            if key not in seen:
                seen.add(key)
                month_periods.append(key)

        contracts = []
        for year, month in month_periods:
            key_m = f"{year}-{month:02d}"
            key_q = f"{year}-Q{(month - 1) // 3 + 1}"
            key_y = str(year)

            # ── Base contract ──────────────────────────────────────────
            source_key = None
            if key_m in base_prices:
                source_key = key_m
            elif key_q in base_prices:
                source_key = key_q
            elif key_y in base_prices:
                source_key = key_y

            if source_key is None:
                continue

            start_utc, end_utc = period_boundaries_fn(year, month, month, "Europe/Zurich")
            if end_utc <= idx[0] or start_utc >= idx[-1]:
                continue

            contracts.append(
                futures_contract_cls(
                    name=f"{year}-{month:02d}<{source_key}>",
                    price=float(base_prices[source_key]),
                    start=start_utc,
                    end=end_utc,
                    product_type="Base",
                )
            )

            # ── Peak contract (if available) ───────────────────────────
            peak_key = None
            for pk in [f"{key_m}-Peak", f"{key_q}-Peak", f"{key_y}-Peak"]:
                if pk in base_prices:
                    peak_key = pk
                    break

            if peak_key is not None:
                contracts.append(
                    futures_contract_cls(
                        name=f"{year}-{month:02d}-Peak<{peak_key}>",
                        price=float(base_prices[peak_key]),
                        start=start_utc,
                        end=end_utc,
                        product_type="Peak",
                    )
                )

        n_peak = sum(1 for c in contracts if c.product_type == "Peak")
        if n_peak > 0:
            logger.info("Peak contracts injected: %d / %d total", n_peak, len(contracts))

        return contracts

    # ---------------------------------------------------------------------------
    # Calcul des composantes
    # ---------------------------------------------------------------------------

    def _resolve_base(self, idx: pd.DatetimeIndex, base_prices: dict) -> pd.Series:
        """
        Resolve base level B for each timestamp (vectorized).
        Priority: monthly > quarterly > annual.
        """
        idx_zurich = idx.tz_convert("Europe/Zurich")

        # Build vectorized keys
        years = idx_zurich.year
        months = idx_zurich.month
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

    def _compute_f_S(self, idx: pd.DatetimeIndex, base_prices: dict) -> pd.Series:
        """
        Seasonal monthly factor f_S.

        _resolve_base picks the finest available forward (monthly > quarterly
        > annual).  When monthly forwards exist, B is already at monthly
        level, so applying f_S = monthly/annual would *double-count* the
        seasonal effect (Bug: B × f_S = monthly × monthly/annual = monthly²/annual).

        When only annual forwards exist (Y+2/Y+3), monthly forwards are
        unavailable so the ratio cannot be computed either.

        => f_S = 1.0 always in the current architecture.
        The arbitrage-free calibration step handles fine-grained level
        adjustments to match all available forward contracts.
        """
        return pd.Series(1.0, index=idx, name="f_S")

    def _compute_f_W(self, cal: pd.DataFrame) -> pd.Series:
        """
        Facteur jour de semaine f_W.
        Utilise les ratios saisonniers f_W(saison, type_jour) si disponibles,
        sinon fallback sur f_W(type_jour) global.

        After computing raw f_W, normalizes per calendar month so that
        mean(f_W) = 1 within each month. This ensures f_W does not leak
        level information (which belongs in B and f_S).
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

        # Normalize f_W per month so mean(f_W) = 1 within each month
        idx_zh = cal.index.tz_convert("Europe/Zurich")
        month_key = pd.Index([f"{t.year}-{t.month:02d}" for t in idx_zh])
        monthly_mean = f_W.groupby(month_key).transform("mean")
        # Avoid division by zero
        monthly_mean = monthly_mean.replace(0, 1.0)
        f_W = f_W / monthly_mean

        return f_W

    def _confidence_score(self, months_ahead: pd.Series) -> pd.Series:
        """Confidence score [0,1] decreasing with horizon, configurable."""
        ct = self.confidence_thresholds
        score = pd.Series(ct.get("6m", 1.0), index=months_ahead.index)
        score[months_ahead > 6]  = ct.get("12m", 0.85)
        score[months_ahead > 12] = ct.get("24m", 0.65)
        score[months_ahead > 24] = ct.get("36m", 0.45)
        return score

    def _check_energy_consistency(self, df: pd.DataFrame, base_prices: dict) -> None:
        """
        Verify price_shape average matches base prices at annual, quarterly,
        and monthly levels. Alerts if deviation exceeds threshold.
        """
        threshold = 0.005 if df["calibrated"].any() else 0.05
        idx_zurich = df.index.tz_convert("Europe/Zurich")

        for key, base in base_prices.items():
            if base == 0:
                continue

            # Determine mask based on key type
            if len(key) == 4 and key.isdigit():
                # Annual key
                mask = idx_zurich.year == int(key)
                year_int = int(key)
                expected = (366 if pd.Timestamp(year=year_int, month=12, day=31).is_leap_year else 365) * 96
                min_coverage = 0.95
            elif len(key) == 7 and key[4] == '-' and key[5] == 'Q' and key[6].isdigit():
                # Quarterly key e.g. '2026-Q1'
                year_int = int(key[:4])
                q = int(key[6])
                q_months = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}[q]
                mask = (idx_zurich.year == year_int) & (idx_zurich.month.isin(q_months))
                expected = sum(
                    (28 + (m in (1, 3, 5, 7, 8, 10, 12)) * 3 + (m in (4, 6, 9, 11)) * 2) for m in q_months
                ) * 96
                min_coverage = 0.90
            elif len(key) == 7 and key[4] == '-' and key[5:].isdigit():
                # Monthly key e.g. '2026-03'
                year_int = int(key[:4])
                month_int = int(key[5:])
                mask = (idx_zurich.year == year_int) & (idx_zurich.month == month_int)
                import calendar as cal_mod
                expected = cal_mod.monthrange(year_int, month_int)[1] * 96
                min_coverage = 0.90
            else:
                continue

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
            if rel_err > threshold:
                logger.warning(
                    "Energy consistency %s: base=%.2f, mean_PFC=%.2f, deviation=%.1f%%",
                    key, base, mean_p, rel_err * 100
                )
            else:
                logger.info(
                    "Energy consistency %s: OK (deviation=%.2f%%)", key, rel_err * 100
                )

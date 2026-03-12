"""
assembler.py
------------
Assemblage de la PFC 15min N+3 ans.

Formule complÃ¨te (6 facteurs multiplicatifs + calibration) :
    P_raw(t) = B(year) Ã— f_S(month) Ã— f_W(dow) Ã— f_H(h) Ã— f_Q(q) Ã— f_WV(t)

Puis calibration arbitrage-free :
    P_cal(t) = P_raw(t) + Î´(t)
    oÃ¹ Î´ minimise âˆ«(Î´''(t))Â² sous contrainte :
        mean(P_cal sur contrat i) = prix_futures_i   âˆ€ i

OÃ¹ :
    B(year)   = niveau de base annuel (forwards EEX Cal/Quarter/Month)
    f_S(month)= facteur saisonnier mensuel (normalisÃ© : mean = 1)
    f_W(dow)  = facteur jour de semaine (normalisÃ© : mean hebdo = 1)
    f_H(h)    = facteur horaire intraday (ShapeHourly)
    f_Q(q)    = facteur 15min intra-horaire (ShapeIntraday)
    f_WV(t)   = correction Water Value rÃ©servoirs hydro CH (WaterValueCorrection)

Pipeline :
    1. Cascading : enrichir les forwards manquants (Yearâ†’Qâ†’Month)
    2. Shape brut : P_raw = B Ã— f_S Ã— f_W Ã— f_H Ã— f_Q Ã— f_WV
    3. Calibration : P_cal = P_raw + Î´ (arbitrage-free, Maximum Smoothness)
    4. IC bootstrap : p10, p90

Horizon glissant :
    start  = demain (J+1)
    end    = J + 1095 (â‰ˆ 3 ans)

GranularitÃ© de B et f_S selon l'horizon :
    M+1 â†’ M+6   : B mensuel (forwards EEX Monthly)
    M+7 â†’ M+12  : B trimestriel (forwards EEX Quarterly)
    Y+2 â†’ Y+3   : B annuel (forwards EEX Cal)
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

    IntÃ¨gre les 3 modules ajoutÃ©s :
        - ContractCascader  : dÃ©composition automatique des forwards
        - WaterValueCorrection : correction hydro saisonniÃ¨re
        - ArbitrageFreeCalibrator : calibration no-arbitrage

    Args:
        shape_hourly   : instance ShapeHourly fittÃ©e
        shape_intraday : instance ShapeIntraday fittÃ©e
        uncertainty    : instance Uncertainty (optionnel, pour IC p10/p90)
        water_value    : instance WaterValueCorrection fittÃ©e (optionnel)
        cascader       : instance ContractCascader fittÃ©e (optionnel)
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
    ) -> None:
        self.sh = shape_hourly
        self.si = shape_intraday
        self.unc = uncertainty
        self.wv = water_value
        self.cascader = cascader
        self.calibrator = calibrator

    def build(
        self,
        base_prices: dict,
        start_date: str | None = None,
        horizon_days: int = HORIZON_DAYS,
        entso_forecast: pd.DataFrame | None = None,
        hydro_forecast: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Construit la PFC 15min sur l'horizon N+3.

        Args:
            base_prices    : dict de niveaux de base, clÃ©s selon granularitÃ© :
                             {'2025': 65.0,               â† niveau annuel Cal
                              '2025-01': 70.0,            â† override mensuel si dispo
                              '2025-Q1': 68.0}            â† override trimestriel si dispo
                             Logique de prioritÃ© : mensuel > trimestriel > annuel

            start_date     : 'YYYY-MM-DD' (dÃ©faut = demain)
            horizon_days   : nombre de jours (dÃ©faut = 3Ã—365)
            entso_forecast : prÃ©visions solar_regime + load_deviation sur l'horizon
                             (None â†’ valeurs neutres utilisÃ©es)
            hydro_forecast : prÃ©visions fill_deviation rÃ©servoirs hydro CH
                             (None â†’ f_WV = 1.0 neutre)

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

        logger.info("Assemblage PFC 15min : %s â†’ %s", ts_start.date(), ts_end.date())

        # â”€â”€ 0. Cascading des forwards manquants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.cascader is not None:
            base_prices = self.cascader.cascade(base_prices)
            logger.info("Cascading terminÃ© : %d produits forwards", len(base_prices))

        # Index complet 15min UTC
        idx = pd.date_range(ts_start, ts_end, freq="15min", inclusive="left", tz="UTC")

        # Enrichissement calendaire
        cal = enrich_15min_index(idx)

        # â”€â”€ Facteur saisonnier mensuel f_S â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        f_S = self._compute_f_S(idx, base_prices)

        # â”€â”€ Facteur jour de semaine f_W â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        f_W = self._compute_f_W(cal)

        # â”€â”€ Facteur horaire f_H â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        f_H = self.sh.apply(idx, cal)

        # â”€â”€ Facteur 15min f_Q â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        f_Q = self.si.apply(idx, cal, entso_forecast)

        # â”€â”€ Facteur Water Value f_WV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.wv is not None:
            f_WV = self.wv.apply(idx, cal, hydro_forecast)
        else:
            f_WV = pd.Series(1.0, index=idx, name="f_WV")

        # â”€â”€ Niveau de base B par timestamp â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        B = self._resolve_base(idx, base_prices)

        # â”€â”€ Prix brut (avant calibration) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        price_raw = B * f_S * f_W * f_H * f_Q * f_WV

        # â”€â”€ Profile type (pour traÃ§abilitÃ©) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        idx_zurich = idx.tz_convert("Europe/Zurich")
        months_ahead = ((idx_zurich.to_period("M") - pd.Timestamp.now(tz="Europe/Zurich").to_period("M"))
                        .apply(lambda x: x.n if hasattr(x, 'n') else 0))

        profile_type = pd.Series("Y+2/Y+3", index=idx)
        profile_type[months_ahead <= 12] = "M+7..M+12"
        profile_type[months_ahead <= 6] = "M+1..M+6"

        # â”€â”€ Calibration arbitrage-free â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        calibrated = False
        if self.calibrator is not None:
            price_shape, calibrated = self._apply_calibration(
                price_raw, idx, base_prices
            )
        else:
            price_shape = price_raw

        # â”€â”€ Assemblage DataFrame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

        # â”€â”€ Intervalles de confiance (optionnel) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.unc is not None:
            ic = self.unc.compute(df, cal)
            df["p10"] = ic["p10"]
            df["p90"] = ic["p90"]
        else:
            df["p10"] = np.nan
            df["p90"] = np.nan

        # â”€â”€ VÃ©rification cohÃ©rence Ã©nergÃ©tique â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self._check_energy_consistency(df, base_prices)

        logger.info(
            "PFC assemblÃ©e : %d intervalles 15min, prix min=%.1f max=%.1f â‚¬/MWh, "
            "calibration=%s",
            len(df), df["price_shape"].min(), df["price_shape"].max(),
            "OK" if calibrated else "non appliquÃ©e"
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
            Tuple (prix calibrÃ©, True si convergence OK)
        """
        from pfc_shaping.calibration.arbitrage_free import FuturesContract
        from pfc_shaping.calibration.cascading import parse_key, _period_boundaries_utc

        contracts = []
        for key, price in base_prices.items():
            try:
                ptype, year, sub = parse_key(key)
            except ValueError:
                continue

            if ptype == "Cal":
                ms, me = 1, 12
            elif ptype == "Quarter":
                ms, me = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}[sub]
            elif ptype == "Month":
                ms, me = sub, sub
            else:
                continue

            start_utc, end_utc = _period_boundaries_utc(
                year, ms, me, "Europe/Zurich"
            )

            # Ne calibrer que les contrats qui chevauchent notre courbe
            if end_utc <= idx[0] or start_utc >= idx[-1]:
                continue

            contracts.append(FuturesContract(
                name=key,
                price=price,
                start=start_utc,
                end=end_utc,
                product_type="Base",
            ))

        if not contracts:
            logger.info("Aucun contrat futures applicable â€” calibration ignorÃ©e")
            return price_raw, False

        logger.info(
            "Calibration arbitrage-free : %d contrats futures", len(contracts)
        )
        result = self.calibrator.calibrate(price_raw, contracts)

        if result.converged:
            logger.info(
                "Calibration convergÃ©e : rÃ©sidu max = %.6f â‚¬/MWh, "
                "coÃ»t lissage = %.2f",
                result.max_abs_residual,
                result.smoothness_cost,
            )
        else:
            logger.warning(
                "Calibration NON convergÃ©e : rÃ©sidu max = %.6f â‚¬/MWh",
                result.max_abs_residual,
            )

        return result.calibrated_curve, result.converged

    # ---------------------------------------------------------------------------
    # Calcul des composantes
    # ---------------------------------------------------------------------------

    def _resolve_base(self, idx: pd.DatetimeIndex, base_prices: dict) -> pd.Series:
        """
        RÃ©sout le niveau de base B pour chaque timestamp.
        PrioritÃ© : mensuel > trimestriel > annuel.
        """
        B = pd.Series(np.nan, index=idx)
        idx_zurich = idx.tz_convert("Europe/Zurich")

        for i, ts in enumerate(idx_zurich):
            key_m = ts.strftime("%Y-%m")          # ex: '2025-03'
            key_q = f"{ts.year}-Q{(ts.month-1)//3+1}"  # ex: '2025-Q1'
            key_y = str(ts.year)                  # ex: '2025'

            if key_m in base_prices:
                B.iloc[i] = base_prices[key_m]
            elif key_q in base_prices:
                B.iloc[i] = base_prices[key_q]
            elif key_y in base_prices:
                B.iloc[i] = base_prices[key_y]
            else:
                # Fallback : dernier niveau annuel connu
                for y in [str(ts.year - 1), str(ts.year - 2)]:
                    if y in base_prices:
                        B.iloc[i] = base_prices[y]
                        break

        if B.isna().any():
            logger.warning("%d timestamps sans niveau de base â€” interpolation", B.isna().sum())
            B = B.interpolate(method="linear").ffill().bfill()

        return B

    def _compute_f_S(self, idx: pd.DatetimeIndex, base_prices: dict) -> pd.Series:
        """
        Facteur saisonnier mensuel f_S = niveau_mensuel / niveau_annuel.
        Si pas de dÃ©composition mensuelle disponible â†’ f_S = 1.
        """
        f_S = pd.Series(1.0, index=idx)
        idx_zurich = idx.tz_convert("Europe/Zurich")

        for i, ts in enumerate(idx_zurich):
            key_m = ts.strftime("%Y-%m")
            key_y = str(ts.year)
            if key_m in base_prices and key_y in base_prices and base_prices[key_y] != 0:
                f_S.iloc[i] = base_prices[key_m] / base_prices[key_y]

        return f_S

    def _compute_f_W(self, cal: pd.DataFrame) -> pd.Series:
        “””
        Facteur jour de semaine f_W.
        Utilise les ratios empiriques calibrés par ShapeHourly.fit() sur
        l'historique EPEX. Fallback sur des défauts si non disponible.
        “””
        _FW_DEFAULTS = {
            “Ouvrable”: 1.05,
            “Samedi”:   0.92,
            “Dimanche”: 0.78,
            “Ferie_CH”: 0.75,
            “Ferie_DE”: 0.88,
        }
        f_W_map = self.sh.f_W_ if self.sh.f_W_ else _FW_DEFAULTS
        return cal[“type_jour”].map(f_W_map).fillna(1.0).rename(“f_W”)

    def _confidence_score(self, months_ahead: pd.Series) -> pd.Series:
        """Score de confiance [0,1] dÃ©croissant avec l'horizon."""
        score = pd.Series(1.0, index=months_ahead.index)
        score[months_ahead > 6]  = 0.85
        score[months_ahead > 12] = 0.65
        score[months_ahead > 24] = 0.45
        return score

    def _check_energy_consistency(self, df: pd.DataFrame, base_prices: dict) -> None:
        """
        VÃ©rifie que la moyenne annuelle de price_shape â‰ˆ niveau de base annuel.
        LÃ¨ve une alerte si l'Ã©cart > 5% (ou > 0.5% si calibration appliquÃ©e).
        """
        threshold = 0.005 if df["calibrated"].any() else 0.05

        for year, base in base_prices.items():
            if len(year) != 4 or not year.isdigit():
                continue
            mask = df.index.tz_convert("Europe/Zurich").year == int(year)
            if mask.sum() == 0:
                continue
            mean_p = df.loc[mask, "price_shape"].mean()
            if base != 0:
                rel_err = abs(mean_p - base) / abs(base)
                if rel_err > threshold:
                    logger.warning(
                        "CohÃ©rence Ã©nergie %s : base=%.2f, mean_PFC=%.2f, Ã©cart=%.1f%%",
                        year, base, mean_p, rel_err * 100
                    )
                else:
                    logger.info(
                        "CohÃ©rence Ã©nergie %s : OK (Ã©cart=%.2f%%)", year, rel_err * 100
                    )

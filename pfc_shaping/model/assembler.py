"""
assembler.py
------------
Assemblage de la PFC 15min N+3 ans.

Formule complète :
    P(t) = B(year) × f_S(month) × f_W(dow) × f_H(h | saison, dow) × f_Q(q | h, saison, dow)

Où :
    B(year)   = niveau de base annuel fourni en input (€/MWh)
                (issu des forwards EEX Cal Y+1, Y+2, Y+3 ou d'une estimation)
    f_S(month)= facteur saisonnier mensuel (normalisé : mean mensuel annuel = 1)
    f_W(dow)  = facteur jour de semaine (normalisé : mean hebdo = 1)
    f_H(h)    = facteur horaire intraday (ShapeHourly)
    f_Q(q)    = facteur 15min intra-horaire (ShapeIntraday)

Contrainte globale de cohérence énergétique (vérifiée en fin d'assemblage) :
    mean_annuel[P(t)] ≈ B(year)   pour chaque année

Le modèle produit également les colonnes p10 / p90 (intervalles de confiance)
si un objet Uncertainty est fourni.

Horizon glissant :
    start  = demain (J+1)
    end    = J + 1095 (≈ 3 ans)

Granularité de B et f_S selon l'horizon :
    M+1 → M+6   : B mensuel (forwards EEX Monthly)
    M+7 → M+12  : B trimestriel (forwards EEX Quarterly)
    Y+2 → Y+3   : B annuel (forwards EEX Cal)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.calendar_ch import enrich_15min_index
from model.shape_hourly import ShapeHourly
from model.shape_intraday import ShapeIntraday

logger = logging.getLogger(__name__)

# Horizon standard N+3 ans en jours
HORIZON_DAYS = 3 * 365


class PFCAssembler:
    """
    Assembleur de la PFC 15min N+3 ans.

    Args:
        shape_hourly   : instance ShapeHourly fittée
        shape_intraday : instance ShapeIntraday fittée
        uncertainty    : instance Uncertainty (optionnel, pour IC p10/p90)
    """

    def __init__(
        self,
        shape_hourly: ShapeHourly,
        shape_intraday: ShapeIntraday,
        uncertainty=None,
    ) -> None:
        self.sh = shape_hourly
        self.si = shape_intraday
        self.unc = uncertainty

    def build(
        self,
        base_prices: dict,
        start_date: str | None = None,
        horizon_days: int = HORIZON_DAYS,
        entso_forecast: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Construit la PFC 15min sur l'horizon N+3.

        Args:
            base_prices    : dict de niveaux de base, clés selon granularité :
                             {'2025': 65.0,               ← niveau annuel Cal
                              '2025-01': 70.0,            ← override mensuel si dispo
                              '2025-Q1': 68.0}            ← override trimestriel si dispo
                             Logique de priorité : mensuel > trimestriel > annuel

            start_date     : 'YYYY-MM-DD' (défaut = demain)
            horizon_days   : nombre de jours (défaut = 3×365)
            entso_forecast : prévisions solar_regime + load_deviation sur l'horizon
                             (None → valeurs neutres utilisées)

        Returns:
            DataFrame colonnes ['price_shape', 'f_S', 'f_W', 'f_H', 'f_Q',
                                'profile_type', 'confidence', 'p10', 'p90']
            index : DatetimeIndex UTC freq='15min'
        """
        if start_date is None:
            start_date = (pd.Timestamp.utcnow() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        ts_start = pd.Timestamp(start_date, tz="UTC")
        ts_end = ts_start + pd.Timedelta(days=horizon_days)

        logger.info("Assemblage PFC 15min : %s → %s", ts_start.date(), ts_end.date())

        # Index complet 15min UTC
        idx = pd.date_range(ts_start, ts_end, freq="15min", inclusive="left", tz="UTC")

        # Enrichissement calendaire
        cal = enrich_15min_index(idx)

        # ── Facteur saisonnier mensuel f_S ─────────────────────────────────────
        f_S = self._compute_f_S(idx, base_prices)

        # ── Facteur jour de semaine f_W ────────────────────────────────────────
        f_W = self._compute_f_W(cal)

        # ── Facteur horaire f_H ────────────────────────────────────────────────
        f_H = self.sh.apply(idx, cal)

        # ── Facteur 15min f_Q ──────────────────────────────────────────────────
        f_Q = self.si.apply(idx, cal, entso_forecast)

        # ── Niveau de base B par timestamp ─────────────────────────────────────
        B = self._resolve_base(idx, base_prices)

        # ── Prix final ─────────────────────────────────────────────────────────
        price = B * f_S * f_W * f_H * f_Q

        # ── Profile type (pour traçabilité) ────────────────────────────────────
        idx_zurich = idx.tz_convert("Europe/Zurich")
        months_ahead = ((idx_zurich.to_period("M") - pd.Timestamp.now(tz="Europe/Zurich").to_period("M"))
                        .apply(lambda x: x.n if hasattr(x, 'n') else 0))

        profile_type = pd.Series("Y+2/Y+3", index=idx)
        profile_type[months_ahead <= 12] = "M+7..M+12"
        profile_type[months_ahead <= 6] = "M+1..M+6"

        # ── Assemblage DataFrame ───────────────────────────────────────────────
        df = pd.DataFrame(
            {
                "price_shape": price,
                "B": B,
                "f_S": f_S,
                "f_W": f_W,
                "f_H": f_H,
                "f_Q": f_Q,
                "profile_type": profile_type,
                "confidence": self._confidence_score(months_ahead),
            },
            index=idx,
        )

        # ── Intervalles de confiance (optionnel) ───────────────────────────────
        if self.unc is not None:
            ic = self.unc.compute(df, cal)
            df["p10"] = ic["p10"]
            df["p90"] = ic["p90"]
        else:
            df["p10"] = np.nan
            df["p90"] = np.nan

        # ── Vérification cohérence énergétique ────────────────────────────────
        self._check_energy_consistency(df, base_prices)

        logger.info(
            "PFC assemblée : %d intervalles 15min, prix min=%.1f max=%.1f €/MWh",
            len(df), df["price_shape"].min(), df["price_shape"].max()
        )
        return df

    # ---------------------------------------------------------------------------
    # Calcul des composantes
    # ---------------------------------------------------------------------------

    def _resolve_base(self, idx: pd.DatetimeIndex, base_prices: dict) -> pd.Series:
        """
        Résout le niveau de base B pour chaque timestamp.
        Priorité : mensuel > trimestriel > annuel.
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
            logger.warning("%d timestamps sans niveau de base — interpolation", B.isna().sum())
            B = B.interpolate(method="linear").ffill().bfill()

        return B

    def _compute_f_S(self, idx: pd.DatetimeIndex, base_prices: dict) -> pd.Series:
        """
        Facteur saisonnier mensuel f_S = niveau_mensuel / niveau_annuel.
        Si pas de décomposition mensuelle disponible → f_S = 1.
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
        """
        Facteur jour de semaine f_W.
        Basé sur les ratios empiriques moyens EPEX CH par type de jour.
        (Calibré empiriquement — peut être remplacé par des facteurs fittés)
        """
        # Ratios standard marché CH (workday=1.0 baseline)
        _FW_DEFAULTS = {
            "Ouvrable": 1.05,
            "Samedi":   0.92,
            "Dimanche": 0.78,
            "Ferie_CH": 0.75,
            "Ferie_DE": 0.88,
        }
        return cal["type_jour"].map(_FW_DEFAULTS).fillna(1.0).rename("f_W")

    def _confidence_score(self, months_ahead: pd.Series) -> pd.Series:
        """Score de confiance [0,1] décroissant avec l'horizon."""
        score = pd.Series(1.0, index=months_ahead.index)
        score[months_ahead > 6]  = 0.85
        score[months_ahead > 12] = 0.65
        score[months_ahead > 24] = 0.45
        return score

    def _check_energy_consistency(self, df: pd.DataFrame, base_prices: dict) -> None:
        """
        Vérifie que la moyenne annuelle de price_shape ≈ niveau de base annuel.
        Lève une alerte si l'écart > 5%.
        """
        for year, base in base_prices.items():
            if len(year) != 4 or not year.isdigit():
                continue
            mask = df.index.tz_convert("Europe/Zurich").year == int(year)
            if mask.sum() == 0:
                continue
            mean_p = df.loc[mask, "price_shape"].mean()
            if base != 0:
                rel_err = abs(mean_p - base) / abs(base)
                if rel_err > 0.05:
                    logger.warning(
                        "Cohérence énergie %s : base=%.2f, mean_PFC=%.2f, écart=%.1f%%",
                        year, base, mean_p, rel_err * 100
                    )
                else:
                    logger.info(
                        "Cohérence énergie %s : OK (écart=%.2f%%)", year, rel_err * 100
                    )

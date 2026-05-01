"""
pfc_flavors.py
--------------
Déclinaison de la PFC Mid-Market en 3 variantes métier pour FMV SA :

    1. Mid-Market (neutre)  — référence P&L, suivi de risque, reporting
    2. Offre Clients (GC)   — pricing offres grands consommateurs
    3. Valorisation Prod     — dispatch hydro, water value, optionalité

Principe fondamental :
    La FORME (duck curve, f_H, f_Q) est IDENTIQUE pour les 3 variantes.
    Seul le NIVEAU change via des primes additives :
        P_client(t) = P_mid(t) + spread(t) + risk_premium(t)
        P_prod(t)   = P_mid(t) + capture_premium(t)

Pourquoi ne pas modifier la shape :
    Modifier la duck curve créerait un biais systématique dans les décisions
    de dispatch et de pricing. La shape reflète la meilleure estimation du
    marché — elle est objective et commune aux 3 usages.

Architecture :
    PFCFlavors prend un DataFrame PFC mid-market (sortie de PFCAssembler)
    et retourne un dict de 3 DataFrames, un par variante.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Paramètres par défaut ──────────────────────────────────────────────────────

# Spread bid/ask par horizon (EUR/MWh)
# Plus l'horizon est long, plus le spread est large (moins liquide)
DEFAULT_SPREAD = {
    6:  1.0,    # M+1..M+6  : 1 EUR/MWh
    12: 1.5,    # M+7..M+12 : 1.5 EUR/MWh
    24: 2.5,    # Y+2       : 2.5 EUR/MWh
    36: 3.5,    # Y+3       : 3.5 EUR/MWh
}

# Prime de risque volumétrique par heure (EUR/MWh)
# Plus élevée aux heures volatiles (rampes solaires, pointe)
# Calibrée sur la volatilité historique intra-horaire EPEX CH
DEFAULT_HOURLY_RISK_PREMIUM = {
    # Nuit (stable) → prime faible
    0: 0.3, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.3, 5: 0.4,
    # Rampe matin (volatile) → prime élevée
    6: 1.2, 7: 1.5, 8: 1.8, 9: 1.5, 10: 1.0,
    # Jour (solaire, modérément volatile)
    11: 0.8, 12: 0.7, 13: 0.6, 14: 0.6, 15: 0.7, 16: 0.8,
    # Rampe soir (très volatile) → prime maximale
    17: 1.5, 18: 2.0, 19: 2.2, 20: 1.8,
    # Soirée (décroissante)
    21: 1.0, 22: 0.6, 23: 0.4,
}

# Prime de capture hydro (EUR/MWh)
# La flexibilité hydro permet de capturer un spread peak/off-peak
# Prime positive = la production vaut plus que le mid-market
# car on peut choisir QUAND turbiner
DEFAULT_CAPTURE_PREMIUM = {
    6:  0.8,    # M+1..M+6
    12: 0.6,    # M+7..M+12
    24: 0.4,    # Y+2
    36: 0.3,    # Y+3
}


class PFCFlavors:
    """
    Déclinaison de la PFC mid-market en 3 variantes métier.

    Attributs configurables :
        spread_by_horizon      : spread bid/ask par horizon (mois)
        hourly_risk_premium    : prime de risque par heure (EUR/MWh)
        capture_premium        : prime de capture hydro par horizon
        client_percentile      : percentile IC pour le pricing client
                                 ('p90' = conservateur vendeur)
    """

    def __init__(
        self,
        spread_by_horizon: dict[int, float] | None = None,
        hourly_risk_premium: dict[int, float] | None = None,
        capture_premium: dict[int, float] | None = None,
        client_percentile: str = "p90",
    ) -> None:
        self.spread_by_horizon = spread_by_horizon or DEFAULT_SPREAD.copy()
        self.hourly_risk_premium = hourly_risk_premium or DEFAULT_HOURLY_RISK_PREMIUM.copy()
        self.capture_premium = capture_premium or DEFAULT_CAPTURE_PREMIUM.copy()
        self.client_percentile = client_percentile

    def generate(
        self,
        pfc_mid: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        Génère les 3 variantes à partir de la PFC mid-market.

        Args:
            pfc_mid : DataFrame PFC mid-market (sortie de PFCAssembler.build())
                      Colonnes attendues : ['price_shape', 'p10', 'p90',
                      'profile_type', 'confidence', ...]

        Returns:
            dict avec 3 clés :
                'mid_market'  : PFC neutre (inchangée, colonne 'pfc_mid')
                'client'      : PFC offre clients (colonne 'pfc_client')
                'production'  : PFC valorisation production (colonne 'pfc_prod')
        """
        now = pd.Timestamp.now(tz="UTC")
        idx = pfc_mid.index
        n = len(idx)

        # ── Horizon en mois pour chaque timestamp ──
        days_ahead = (idx - now).total_seconds() / 86400
        months_ahead = np.maximum(0, np.round(days_ahead / 30)).astype(int)

        # ── Heures locales (Europe/Zurich) ──
        hours_local = idx.tz_convert("Europe/Zurich").hour

        # ══════════════════════════════════════════════════════════════════
        # 1. MID-MARKET (neutre — référence)
        # ══════════════════════════════════════════════════════════════════
        mid = pfc_mid.copy()
        mid["pfc_flavor"] = "mid_market"

        # ══════════════════════════════════════════════════════════════════
        # 2. OFFRE CLIENTS (prix vendeur conservateur)
        # ══════════════════════════════════════════════════════════════════
        # Composante 1 : spread bid/ask (dépend de l'horizon)
        spread_arr = np.vectorize(self._lookup_spread)(months_ahead)

        # Composante 2 : prime de risque horaire (dépend de l'heure)
        risk_arr = np.array([
            self.hourly_risk_premium.get(int(h), 0.5)
            for h in hours_local
        ])

        # Prix client = mid + spread + risk_premium
        # On utilise aussi le percentile haut des IC comme plancher
        client_price = pfc_mid["price_shape"].values + spread_arr + risk_arr

        if self.client_percentile == "p90" and "p90" in pfc_mid.columns:
            p90 = pfc_mid["p90"].values
            valid = ~np.isnan(p90)
            # Le prix client est au minimum le p90 (conservateur vendeur)
            client_price[valid] = np.maximum(client_price[valid], p90[valid])

        client = pfc_mid.copy()
        client["price_shape"] = client_price
        client["spread"] = spread_arr
        client["risk_premium"] = risk_arr
        client["pfc_flavor"] = "client"

        # Recalculer p10/p90 pour le client (shift by spread+risk)
        if "p10" in client.columns:
            client["p10"] = pfc_mid["p10"].values + spread_arr + risk_arr
            client["p90"] = pfc_mid["p90"].values + spread_arr + risk_arr

        # ══════════════════════════════════════════════════════════════════
        # 3. VALORISATION PRODUCTION (dispatch hydro)
        # ══════════════════════════════════════════════════════════════════
        # La prime de capture reflète la valeur de la flexibilité hydro :
        # un producteur avec réservoirs peut turbiner aux heures de pointe
        # et stocker aux heures creuses → il "capture" un prix moyen
        # supérieur au prix mid-market baseload
        capture_arr = np.vectorize(self._lookup_capture)(months_ahead)

        prod_price = pfc_mid["price_shape"].values + capture_arr

        prod = pfc_mid.copy()
        prod["price_shape"] = prod_price
        prod["capture_premium"] = capture_arr
        prod["pfc_flavor"] = "production"

        # p10/p90 shifted
        if "p10" in prod.columns:
            prod["p10"] = pfc_mid["p10"].values + capture_arr
            prod["p90"] = pfc_mid["p90"].values + capture_arr

        # ── Statistiques ──
        mid_mean = pfc_mid["price_shape"].mean()
        client_mean = client["price_shape"].mean()
        prod_mean = prod["price_shape"].mean()

        logger.info(
            "PFC Flavors générées : mid=%.2f, client=%.2f (+%.2f), "
            "prod=%.2f (+%.2f) EUR/MWh",
            mid_mean,
            client_mean, client_mean - mid_mean,
            prod_mean, prod_mean - mid_mean,
        )

        return {
            "mid_market": mid,
            "client": client,
            "production": prod,
        }

    # ── Lookups internes ──────────────────────────────────────────────────

    def _lookup_spread(self, months: int) -> float:
        """Spread bid/ask par horizon."""
        for threshold in sorted(self.spread_by_horizon.keys()):
            if months <= threshold:
                return self.spread_by_horizon[threshold]
        return self.spread_by_horizon[max(self.spread_by_horizon.keys())]

    def _lookup_capture(self, months: int) -> float:
        """Prime de capture hydro par horizon."""
        for threshold in sorted(self.capture_premium.keys()):
            if months <= threshold:
                return self.capture_premium[threshold]
        return self.capture_premium[max(self.capture_premium.keys())]

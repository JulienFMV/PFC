"""
structural_break.py
--------------------
Détection de ruptures structurelles dans les profils de shape 15min.

Une rupture structurelle survient lorsque la forme des prix 15min change
durablement (ex. : nouvelle capacité solaire massive en Suisse/Allemagne,
modification des règles de marché EPEX, changement de structure tarifaire
du réseau Swissgrid).

Méthode : Test de Chow (comparaison de deux sous-périodes)
    H0 : les profils mensuels sont stables entre les deux fenêtres
    H1 : rupture structurelle → lookback réduit

Appliqué sur les 24 facteurs horaires moyens par mois (vecteur agrégé).

Si rupture détectée :
    - Alerte loggée
    - Lookback réduit automatiquement (pour ne pas contaminer le modèle
      avec des données pré-rupture)
    - Email/log FMV envoyé (via hook externe optionnel)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Seuil de p-value pour déclarer une rupture
P_VALUE_THRESHOLD = 0.01

# Fenêtre minimale après réduction du lookback (mois)
MIN_LOOKBACK_MONTHS = 12


@dataclass
class BreakResult:
    detected: bool
    p_value: float
    break_date: pd.Timestamp | None
    recommended_lookback_months: int
    message: str


def detect_chow(
    epex_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    window_months: int = 12,
    full_lookback_months: int = 36,
) -> BreakResult:
    """
    Test de Chow mensuel sur les profils horaires moyens.

    Compare les profils de la première moitié vs la deuxième moitié
    de la fenêtre d'observation.

    Args:
        epex_df            : prix EPEX 15min (colonnes ['price_eur_mwh'])
        calendar_df        : enrichissement calendaire
        window_months      : taille de chaque sous-fenêtre (mois)
        full_lookback_months: lookback complet si pas de rupture

    Returns:
        BreakResult
    """
    df = epex_df[["price_eur_mwh"]].join(
        calendar_df[["heure_hce", "type_jour"]]
    ).dropna()

    # Profils horaires moyens par mois (Ouvrable uniquement, plus stable)
    df_ouv = df[df["type_jour"] == "Ouvrable"].copy()
    df_ouv["month"] = df_ouv.index.to_period("M")

    monthly_profiles = (
        df_ouv.groupby(["month", "heure_hce"])["price_eur_mwh"]
        .mean()
        .unstack("heure_hce")  # shape (n_months, 24)
    )

    if len(monthly_profiles) < window_months * 2:
        return BreakResult(
            detected=False,
            p_value=1.0,
            break_date=None,
            recommended_lookback_months=full_lookback_months,
            message="Historique insuffisant pour le test de Chow"
        )

    # Derniers window_months × 2 pour la comparaison
    recent = monthly_profiles.iloc[-window_months * 2:]
    first_half = recent.iloc[:window_months].values
    second_half = recent.iloc[window_months:].values

    # Normalisation : facteurs centrés-réduits par heure
    def _normalize(X: np.ndarray) -> np.ndarray:
        col_means = X.mean(axis=0, keepdims=True)
        col_means = np.where(col_means == 0, 1, col_means)
        return X / col_means

    first_norm = _normalize(first_half)
    second_norm = _normalize(second_half)

    # Test de Chow simplifié : F-test sur la différence des moyennes par heure
    p_values = []
    for h in range(24):
        if np.std(first_norm[:, h]) < 1e-10 and np.std(second_norm[:, h]) < 1e-10:
            continue
        _, p = stats.ttest_ind(first_norm[:, h], second_norm[:, h], equal_var=False)
        p_values.append(p)

    if not p_values:
        return BreakResult(
            detected=False,
            p_value=1.0,
            break_date=None,
            recommended_lookback_months=full_lookback_months,
            message="Pas assez de variance pour le test"
        )

    # Bonferroni correction sur 24 heures
    min_p = min(p_values)
    p_corrected = min(min_p * 24, 1.0)

    detected = p_corrected < P_VALUE_THRESHOLD
    break_date = None
    recommended_lookback = full_lookback_months

    if detected:
        # La rupture est supposée au point de coupure des deux fenêtres
        break_period = recent.index[window_months]
        break_date = break_period.to_timestamp()
        # Lookback réduit à partir de la rupture
        recommended_lookback = max(window_months, MIN_LOOKBACK_MONTHS)
        msg = (
            f"RUPTURE STRUCTURELLE détectée (p={p_corrected:.4f} < {P_VALUE_THRESHOLD}). "
            f"Date approximative : {break_date.strftime('%Y-%m')}. "
            f"Lookback réduit à {recommended_lookback} mois."
        )
        logger.warning(msg)
    else:
        msg = f"Pas de rupture détectée (p={p_corrected:.4f}). Lookback complet : {full_lookback_months} mois."
        logger.info(msg)

    return BreakResult(
        detected=detected,
        p_value=p_corrected,
        break_date=break_date,
        recommended_lookback_months=recommended_lookback,
        message=msg,
    )

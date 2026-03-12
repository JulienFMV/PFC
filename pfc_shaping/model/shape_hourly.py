"""
shape_hourly.py
---------------
Estimation des facteurs de shape horaire f_H(h | saison, type_jour).

Définition :
    f_H(h | saison, type_jour) = prix moyen heure h / prix moyen journée
                                  pour la cellule (saison, type_jour)

Contrainte :
    mean_h[ f_H(h | saison, type_jour) ] = 1   ∀ (saison, type_jour)

Lissage :
    Convolution gaussienne (σ = 0.5 heure) sur les 24 valeurs de chaque cellule
    pour éviter les discontinuités inter-heures artificielles.

Le résultat est un dictionnaire indexé (saison, type_jour) → array[24] de
facteurs normalisés.

Usage :
    from model.shape_hourly import ShapeHourly
    sh = ShapeHourly()
    sh.fit(epex_df, calendar_df)
    f_h = sh.get(saison="Hiver", type_jour="Ouvrable")   # array shape (24,)
    sh.save("model/shape_hourly.parquet")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

# Paramètre du lissage gaussien en unités d'heures
GAUSSIAN_SIGMA = 0.5

SAISONS = ["Hiver", "Printemps", "Ete", "Automne"]
TYPES_JOUR = ["Ouvrable", "Samedi", "Dimanche", "Ferie_CH", "Ferie_DE"]


class ShapeHourly:
    """
    Modèle de facteurs de forme horaire f_H.

    Attributs publics après fit() :
        factors_ : dict[(saison, type_jour)] -> np.ndarray shape (24,)
        n_obs_   : dict[(saison, type_jour)] -> int (nombre d'obs utilisées)
    """

    def __init__(self, sigma: float = GAUSSIAN_SIGMA) -> None:
        self.sigma = sigma
        self.factors_: dict[tuple[str, str], np.ndarray] = {}
        self.n_obs_: dict[tuple[str, str], int] = {}
        self.f_W_: dict[str, float] = {}  # ratios empiriques par type_jour

    def fit(self, epex_df: pd.DataFrame, calendar_df: pd.DataFrame) -> "ShapeHourly":
        """
        Estime les facteurs de forme sur l'historique EPEX 15min.

        Args:
            epex_df    : DataFrame issu de ingest_epex, colonnes ['price_eur_mwh'],
                         index DatetimeIndex UTC freq≈15min
            calendar_df: DataFrame issu de calendar_ch.enrich_15min_index(),
                         colonnes ['type_jour', 'saison', 'heure_hce', 'quart']

        Returns:
            self
        """
        df = epex_df[["price_eur_mwh"]].copy()
        df = df.join(calendar_df[["saison", "type_jour", "heure_hce"]])
        df = df.dropna(subset=["saison", "type_jour", "heure_hce", "price_eur_mwh"])

        # Calcul empirique de f_W : ratio prix moyen par type_jour / prix moyen global
        self._fit_f_W(df)

        for saison in SAISONS:
            for type_jour in TYPES_JOUR:
                mask = (df["saison"] == saison) & (df["type_jour"] == type_jour)
                subset = df.loc[mask]

                if len(subset) < 96:  # moins d'un jour — cellule vide
                    logger.warning(
                        "Cellule (%s, %s) : %d obs insuffisantes — fallback sur Ouvrable",
                        saison, type_jour, len(subset)
                    )
                    continue

                # Prix moyen par heure (aggrège les 4 quarts)
                hourly_mean = subset.groupby("heure_hce")["price_eur_mwh"].mean()
                hourly_mean = hourly_mean.reindex(range(24)).interpolate(method="linear")

                # Normalisation : f_H moyen = 1
                raw_factors = hourly_mean.values
                daily_mean = raw_factors.mean()
                if daily_mean == 0:
                    normalized = np.ones(24)
                else:
                    normalized = raw_factors / daily_mean

                # Lissage gaussien (σ en unités d'heures, circular pour continuité 23h→0h)
                smoothed = _gaussian_smooth_circular(normalized, sigma=self.sigma)

                # Re-normalisation après lissage
                smoothed = smoothed / smoothed.mean()

                self.factors_[(saison, type_jour)] = smoothed
                self.n_obs_[(saison, type_jour)] = len(subset)

        # Fallback : remplir les cellules vides avec la moyenne des cellules existantes
        self._fill_missing_cells()

        logger.info(
            "ShapeHourly fitted : %d cellules, sigma=%.1f",
            len(self.factors_), self.sigma
        )
        return self

    def get(self, saison: str, type_jour: str) -> np.ndarray:
        """
        Retourne le vecteur de 24 facteurs horaires pour une cellule donnée.

        Fallback automatique si la cellule est absente :
            Ferie_DE → Ferie_CH → Dimanche → (erreur)
        """
        key = (saison, type_jour)
        if key in self.factors_:
            return self.factors_[key]

        for fallback in ["Ferie_CH", "Dimanche"]:
            fb_key = (saison, fallback)
            if fb_key in self.factors_:
                logger.debug("Fallback (%s,%s) → (%s,%s)", saison, type_jour, saison, fallback)
                return self.factors_[fb_key]

        raise KeyError(f"Aucun facteur disponible pour {key} et ses fallbacks")

    def apply(self, timestamps: pd.DatetimeIndex, calendar_df: pd.DataFrame) -> pd.Series:
        """
        Applique les facteurs f_H sur un index 15min futur.

        Args:
            timestamps  : DatetimeIndex UTC (futur N+3 ans)
            calendar_df : enrichissement calendaire de timestamps

        Returns:
            pd.Series de f_H pour chaque timestamp, index=timestamps
        """
        result = pd.Series(index=timestamps, dtype=float, name="f_H")
        for (saison, type_jour), group in calendar_df.groupby(["saison", "type_jour"]):
            factors = self.get(saison, type_jour)
            for h in range(24):
                mask = (group["heure_hce"] == h)
                idx = group.index[mask]
                if len(idx) > 0:
                    result.loc[idx] = factors[h]
        return result

    def save(self, path: str | Path) -> None:
        """Sauvegarde les facteurs f_H et f_W en Parquet."""
        records = []
        for (saison, type_jour), factors in self.factors_.items():
            for h, v in enumerate(factors):
                records.append(
                    {"saison": saison, "type_jour": type_jour, "heure": h, "f_H": v,
                     "n_obs": self.n_obs_.get((saison, type_jour), 0)}
                )
        pd.DataFrame(records).to_parquet(path, index=False)

        # Sauvegarder f_W à côté (même répertoire)
        fw_path = Path(path).with_name("f_W.parquet")
        fw_records = [{"type_jour": k, "f_W": v} for k, v in self.f_W_.items()]
        if fw_records:
            pd.DataFrame(fw_records).to_parquet(fw_path, index=False)

        logger.info("ShapeHourly sauvegardé : %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ShapeHourly":
        """Charge depuis un fichier Parquet."""
        df = pd.read_parquet(path)
        obj = cls()
        for (saison, type_jour), grp in df.groupby(["saison", "type_jour"]):
            grp = grp.sort_values("heure")
            obj.factors_[(saison, type_jour)] = grp["f_H"].values
            obj.n_obs_[(saison, type_jour)] = int(grp["n_obs"].iloc[0])

        # Charger f_W si disponible
        fw_path = Path(path).with_name("f_W.parquet")
        if fw_path.exists():
            fw_df = pd.read_parquet(fw_path)
            obj.f_W_ = dict(zip(fw_df["type_jour"], fw_df["f_W"]))
        return obj

    # ---------------------------------------------------------------------------
    # Interne
    # ---------------------------------------------------------------------------

    def _fit_f_W(self, df: pd.DataFrame) -> None:
        """
        Calcule les ratios empiriques f_W par type_jour depuis l'historique EPEX.

        f_W(type_jour) = prix_moyen(type_jour) / prix_moyen(global)
        Normalisé pour que la moyenne pondérée hebdomadaire ≈ 1.
        """
        overall_mean = df["price_eur_mwh"].mean()
        if overall_mean == 0:
            self.f_W_ = {tj: 1.0 for tj in TYPES_JOUR}
            return

        for tj in TYPES_JOUR:
            mask = df["type_jour"] == tj
            subset = df.loc[mask, "price_eur_mwh"]
            if len(subset) >= 96:  # au moins 1 jour complet
                self.f_W_[tj] = subset.mean() / overall_mean
            else:
                self.f_W_[tj] = 1.0
                logger.warning("f_W(%s) : données insuffisantes — défaut 1.0", tj)

        # Fallback : Ferie_DE → Ferie_CH si pas assez de données
        if self.f_W_.get("Ferie_DE", 1.0) == 1.0 and "Ferie_CH" in self.f_W_:
            self.f_W_["Ferie_DE"] = self.f_W_["Ferie_CH"]

        logger.info(
            "f_W empiriques : %s",
            {k: round(v, 3) for k, v in self.f_W_.items()},
        )

    def _fill_missing_cells(self) -> None:
        """Remplit les cellules vides par interpolation depuis les cellules existantes."""
        for saison in SAISONS:
            for type_jour in TYPES_JOUR:
                key = (saison, type_jour)
                if key not in self.factors_:
                    # Cherche une cellule de la même saison
                    for tj_fallback in ["Ouvrable", "Samedi", "Dimanche"]:
                        fb = (saison, tj_fallback)
                        if fb in self.factors_:
                            self.factors_[key] = self.factors_[fb].copy()
                            logger.info(
                                "Cellule (%s,%s) remplie depuis (%s,%s)",
                                saison, type_jour, saison, tj_fallback
                            )
                            break


# ---------------------------------------------------------------------------
# Utilitaire
# ---------------------------------------------------------------------------

def _gaussian_smooth_circular(x: np.ndarray, sigma: float) -> np.ndarray:
    """
    Lissage gaussien en mode circulaire (24h → continuité 23h-0h).
    Implémenté par triplication de l'array et extraction du centre.
    """
    n = len(x)
    tiled = np.tile(x, 3)
    smoothed = gaussian_filter1d(tiled, sigma=sigma)
    return smoothed[n: 2 * n]

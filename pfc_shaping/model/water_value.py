"""
water_value.py
--------------
Correction de la courbe saisonnière basée sur le Water Value (coût d'opportunité
de l'eau stockée dans les réservoirs hydroélectriques suisses).

Principe :
    Si les réservoirs sont SOUS leur niveau historique moyen :
        → prix hiver ↑ (risque de pénurie, importations nécessaires)
        → prix été stable ou ↑ légèrement

    Si les réservoirs sont AU-DESSUS de leur niveau historique :
        → prix hiver ↓ (excédent d'eau disponible)
        → prix été ↓ (déstockage anticipé)

Modèle :
    f_WV(t) = 1 + β_WV × fill_deviation(t) × season_sensitivity(t)

    où :
        fill_deviation = (fill_actual - fill_historical_mean) / fill_historical_std
        season_sensitivity = facteur saisonnier qui amplifie l'effet en hiver
                            (Hiver: -0.8, Printemps: -0.3, Été: -0.1, Automne: -0.5)
        β_WV = coefficient calibré sur l'historique (typiquement -0.02 à -0.05)

    Le signe négatif signifie : réservoirs pleins → prix plus bas

Calibration :
    Régression linéaire des prix EPEX moyens mensuels sur fill_deviation,
    contrôlé par la saison et la tendance.

Contrainte :
    mean(f_WV) ≈ 1 sur l'horizon complet (facteur neutre en moyenne)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

# Sensibilités saisonnières par défaut (avant calibration)
# Le signe négatif signifie : fill_deviation positif → prix plus bas
DEFAULT_SEASON_SENSITIVITY = {
    "Hiver": -0.8,
    "Printemps": -0.3,
    "Ete": -0.1,
    "Automne": -0.5,
}

# Bornes du coefficient β_WV pour éviter les valeurs aberrantes
BETA_WV_MIN = -0.10
BETA_WV_MAX = -0.001

# Bornes du facteur f_WV pour éviter les valeurs extrêmes
F_WV_FLOOR = 0.80
F_WV_CAP = 1.20

# Mapping mois → saison (identique à calendar_ch.py)
_MONTH_TO_SAISON = {
    1: "Hiver", 2: "Hiver", 3: "Hiver",
    4: "Printemps", 5: "Printemps",
    6: "Ete", 7: "Ete", 8: "Ete", 9: "Ete",
    10: "Automne",
    11: "Hiver", 12: "Hiver",
}


class WaterValueCorrection:
    """Correction multiplicative de la PFC basée sur les niveaux de réservoirs.

    Attributs publics après fit() :
        beta_wv_             : coefficient calibré (négatif)
        season_sensitivity_  : dict[saison -> float] sensibilité par saison
        n_obs_               : nombre d'observations utilisées pour la calibration
    """

    def __init__(self) -> None:
        self.beta_wv_: float = 0.0
        self.season_sensitivity_: dict[str, float] = {}
        self.n_obs_: int = 0

    def fit(
        self,
        epex_df: pd.DataFrame,
        hydro_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
    ) -> "WaterValueCorrection":
        """Calibre β_WV et les sensibilités saisonnières sur l'historique.

        Méthode :
            1. Agrège les prix EPEX en moyenne mensuelle.
            2. Agrège le fill_deviation en moyenne mensuelle (forward-fill
               des données hebdomadaires vers le 15min, puis moyenne mensuelle).
            3. Régression linéaire :
                   prix_mensuel ~ β_0 + β_trend × t + Σ_s (β_s × fill_deviation × 1_{saison=s})
               où le terme d'interaction fill_deviation × saison capture la
               sensibilité différenciée hiver/été.

        Args:
            epex_df:     DataFrame EPEX 15min, colonnes ['price_eur_mwh'],
                         index DatetimeIndex UTC.
            hydro_df:    DataFrame hydro hebdomadaire, colonnes ['fill_deviation'],
                         index DatetimeIndex UTC (fréquence ~W).
            calendar_df: DataFrame calendaire (colonnes ['saison']),
                         index DatetimeIndex UTC.

        Returns:
            self
        """
        if hydro_df.empty or "fill_deviation" not in hydro_df.columns:
            logger.warning(
                "Données hydro absentes ou sans fill_deviation — "
                "calibration impossible, β_WV fixé à défaut"
            )
            self.beta_wv_ = -0.03
            self.season_sensitivity_ = DEFAULT_SEASON_SENSITIVITY.copy()
            self.n_obs_ = 0
            return self

        # ── Préparer les données mensuelles ──────────────────────────────────
        # Forward-fill hydro hebdomadaire vers le 15min, puis moyenne mensuelle
        hydro_15min = hydro_df[["fill_deviation"]].resample("15min").ffill()
        hydro_15min = hydro_15min.reindex(epex_df.index, method="ffill")

        df = epex_df[["price_eur_mwh"]].copy()
        df["fill_deviation"] = hydro_15min["fill_deviation"]
        df["saison"] = calendar_df["saison"].reindex(df.index)
        df = df.dropna(subset=["price_eur_mwh", "fill_deviation", "saison"])

        if len(df) == 0:
            logger.warning("Aucune donnée jointe EPEX/hydro — calibration par défaut")
            self.beta_wv_ = -0.03
            self.season_sensitivity_ = DEFAULT_SEASON_SENSITIVITY.copy()
            self.n_obs_ = 0
            return self

        # Agrégation mensuelle
        df["period"] = df.index.to_period("M")
        monthly = df.groupby("period").agg(
            price_mean=("price_eur_mwh", "mean"),
            fill_dev_mean=("fill_deviation", "mean"),
            saison=("saison", "first"),
        )
        monthly.index = monthly.index.to_timestamp()

        if len(monthly) < 12:
            logger.warning(
                "Moins de 12 mois de données jointes (%d) — "
                "calibration peu fiable, utilisation des défauts",
                len(monthly),
            )
            self.beta_wv_ = -0.03
            self.season_sensitivity_ = DEFAULT_SEASON_SENSITIVITY.copy()
            self.n_obs_ = len(monthly)
            return self

        # ── Régression avec interactions saisonnières ────────────────────────
        # Features : trend + fill_deviation × dummy_saison
        saisons = list(DEFAULT_SEASON_SENSITIVITY.keys())
        X_cols = []

        # Tendance temporelle normalisée [0, 1]
        t_num = (monthly.index - monthly.index[0]).total_seconds()
        t_norm = t_num / t_num.max() if t_num.max() > 0 else t_num
        X_df = pd.DataFrame({"trend": t_norm}, index=monthly.index)
        X_cols.append("trend")

        # Interactions fill_deviation × saison
        for s in saisons:
            col = f"fd_{s}"
            X_df[col] = monthly["fill_dev_mean"] * (monthly["saison"] == s).astype(float)
            X_cols.append(col)

        X = X_df[X_cols].values
        y = monthly["price_mean"].values

        try:
            reg = LinearRegression()
            reg.fit(X, y)

            # Extraire les coefficients d'interaction saisonnière
            # Normaliser par le prix moyen pour obtenir un effet relatif
            price_mean_global = y.mean()
            if price_mean_global == 0:
                price_mean_global = 1.0

            raw_sensitivities = {}
            for i, s in enumerate(saisons):
                coef_idx = 1 + i  # index 0 = trend
                raw_sensitivities[s] = reg.coef_[coef_idx] / price_mean_global

            # β_WV = moyenne pondérée des sensibilités saisonnières
            # (pondération par le nombre de mois dans chaque saison)
            weights = {s: (monthly["saison"] == s).sum() for s in saisons}
            total_w = sum(weights.values())
            if total_w > 0:
                beta_raw = sum(
                    raw_sensitivities[s] * weights[s] for s in saisons
                ) / total_w
            else:
                beta_raw = -0.03

            # Clamper β_WV dans les bornes raisonnables
            self.beta_wv_ = float(np.clip(beta_raw, BETA_WV_MIN, BETA_WV_MAX))

            # Sensibilités saisonnières relatives (normalisées par β_WV)
            if abs(self.beta_wv_) > 1e-6:
                self.season_sensitivity_ = {
                    s: float(
                        np.clip(raw_sensitivities[s] / abs(self.beta_wv_), -2.0, 0.0)
                    )
                    for s in saisons
                }
            else:
                self.season_sensitivity_ = DEFAULT_SEASON_SENSITIVITY.copy()

            self.n_obs_ = len(monthly)

            logger.info(
                "WaterValueCorrection calibré : β_WV=%.4f, n_obs=%d, "
                "sensibilités=%s",
                self.beta_wv_,
                self.n_obs_,
                {s: f"{v:.2f}" for s, v in self.season_sensitivity_.items()},
            )

        except Exception as exc:
            logger.error(
                "Erreur lors de la régression WaterValue : %s — valeurs par défaut",
                exc,
            )
            self.beta_wv_ = -0.03
            self.season_sensitivity_ = DEFAULT_SEASON_SENSITIVITY.copy()
            self.n_obs_ = 0

        return self

    def apply(
        self,
        timestamps: pd.DatetimeIndex,
        calendar_df: pd.DataFrame,
        hydro_forecast: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Retourne le facteur correctif f_WV pour chaque timestamp.

        Le facteur est calculé comme :
            f_WV(t) = 1 + β_WV × fill_deviation(t) × season_sensitivity(saison(t))

        puis renormalisé pour que mean(f_WV) ≈ 1 sur l'horizon complet.

        Les données hydro hebdomadaires sont interpolées en 15min par
        forward-fill (les niveaux de réservoirs changent hebdomadairement).

        Args:
            timestamps:     DatetimeIndex UTC du futur (horizon N+3).
            calendar_df:    Enrichissement calendaire avec colonne 'saison',
                            index aligné sur timestamps.
            hydro_forecast: DataFrame avec colonne 'fill_deviation' (et/ou
                            'water_value_proxy'), index DatetimeIndex UTC
                            (fréquence hebdomadaire ou plus fine).
                            Si None → f_WV = 1.0 partout (neutre).

        Returns:
            pd.Series de f_WV, index=timestamps, name='f_WV'
        """
        f_wv = pd.Series(1.0, index=timestamps, dtype=float, name="f_WV")

        if hydro_forecast is None or hydro_forecast.empty:
            logger.info("Pas de prévision hydro — f_WV neutre (1.0)")
            return f_wv

        if "fill_deviation" not in hydro_forecast.columns:
            logger.warning(
                "hydro_forecast sans colonne 'fill_deviation' — f_WV neutre"
            )
            return f_wv

        # ── Forward-fill des données hebdomadaires vers le 15min ─────────
        fill_dev = hydro_forecast[["fill_deviation"]].copy()

        # Resample vers 15min avec forward-fill
        fill_dev_15min = fill_dev.resample("15min").ffill()

        # Aligner sur les timestamps demandés (forward-fill pour les valeurs
        # en dehors de la plage des données hydro)
        fill_dev_aligned = fill_dev_15min.reindex(timestamps, method="ffill")

        # Backward-fill si les premiers timestamps sont avant les données hydro
        fill_dev_aligned = fill_dev_aligned.bfill()

        # ── Récupérer la saison pour chaque timestamp ────────────────────
        if "saison" in calendar_df.columns:
            saison = calendar_df["saison"].reindex(timestamps)
        else:
            # Fallback : dériver la saison du mois
            idx_zurich = timestamps.tz_convert("Europe/Zurich")
            saison = pd.Series(
                [_MONTH_TO_SAISON[m] for m in idx_zurich.month],
                index=timestamps,
            )

        # ── Calcul du facteur f_WV ───────────────────────────────────────
        sensitivity = self.season_sensitivity_ or DEFAULT_SEASON_SENSITIVITY
        beta = self.beta_wv_ if abs(self.beta_wv_) > 1e-8 else -0.03

        fill_dev_vals = fill_dev_aligned["fill_deviation"].fillna(0.0)
        season_sens = saison.map(sensitivity).fillna(-0.3).astype(float)

        raw_f_wv = 1.0 + beta * fill_dev_vals * season_sens

        # ── Clamping pour éviter les valeurs aberrantes ──────────────────
        raw_f_wv = raw_f_wv.clip(lower=F_WV_FLOOR, upper=F_WV_CAP)

        # ── Renormalisation : mean(f_WV) = 1 sur l'horizon ──────────────
        mean_f = raw_f_wv.mean()
        if abs(mean_f) > 1e-8:
            f_wv = (raw_f_wv / mean_f).rename("f_WV")
        else:
            f_wv = raw_f_wv.rename("f_WV")

        # Re-clamping après renormalisation
        f_wv = f_wv.clip(lower=F_WV_FLOOR, upper=F_WV_CAP)

        logger.info(
            "f_WV appliqué : mean=%.4f, min=%.4f, max=%.4f, β_WV=%.4f",
            f_wv.mean(), f_wv.min(), f_wv.max(), beta,
        )
        return f_wv

    def save(self, path: str | Path) -> None:
        """Sauvegarde les paramètres calibrés en Parquet.

        Args:
            path: chemin du fichier Parquet de sortie.
        """
        records = []
        for saison, sens in self.season_sensitivity_.items():
            records.append({
                "saison": saison,
                "season_sensitivity": sens,
                "beta_wv": self.beta_wv_,
                "n_obs": self.n_obs_,
            })

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_parquet(path, index=False)
        logger.info("WaterValueCorrection sauvegardé : %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "WaterValueCorrection":
        """Charge un modèle calibré depuis un fichier Parquet.

        Args:
            path: chemin du fichier Parquet.

        Returns:
            Instance WaterValueCorrection avec paramètres restaurés.
        """
        df = pd.read_parquet(path)
        obj = cls()
        obj.beta_wv_ = float(df["beta_wv"].iloc[0])
        obj.n_obs_ = int(df["n_obs"].iloc[0])
        obj.season_sensitivity_ = dict(
            zip(df["saison"], df["season_sensitivity"])
        )
        return obj

"""
uncertainty.py
--------------
Quantification de l'incertitude des facteurs de shape par bootstrap.

Méthode :
    Bootstrap non-paramétrique sur l'historique EPEX 15min.
    Pour chaque cellule (saison, type_jour, heure, quart), on tire N_BOOT
    échantillons avec remise et on calcule les percentiles des ratios f_Q.

    L'incertitude croît avec l'horizon (facteur d'élargissement calibré).

Sorties :
    Pour chaque timestamp t de la PFC :
        p10(t) = price_shape(t) × ratio_p10 × horizon_factor
        p90(t) = price_shape(t) × ratio_p90 × horizon_factor

Usage :
    from model.uncertainty import Uncertainty
    unc = Uncertainty()
    unc.fit(epex_df, calendar_df)
    ic = unc.compute(pfc_df, calendar_df)
    # ic['p10'], ic['p90'] : pd.Series alignées sur pfc_df.index
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

N_BOOT = 500  # tirages bootstrap
SEED = 42

# Facteur d'élargissement de l'IC selon l'horizon
# (l'incertitude des profiles augmente sur l'horizon long car les marchés
# et mix énergétique peuvent changer structurellement)
HORIZON_WIDENING = {
    6:  1.00,   # M+1..M+6   : IC calibré
    12: 1.15,   # M+7..M+12  : +15%
    24: 1.40,   # Y+2        : +40%
    36: 1.70,   # Y+3        : +70%
}


class Uncertainty:
    """
    Bootstrap non-paramétrique pour les IC p10/p90 de la PFC 15min.

    Attributs après fit() :
        boot_stats_ : dict[(saison, type_jour, heure, quart)] ->
                      {'p10': float, 'p50': float, 'p90': float}
                      (percentiles du ratio f_Q bootstrappé)
    """

    def __init__(self, n_boot: int = N_BOOT, seed: int = SEED) -> None:
        self.n_boot = n_boot
        self.seed = seed
        self.boot_stats_: dict[tuple, dict] = {}

    def fit(self, epex_df: pd.DataFrame, calendar_df: pd.DataFrame) -> "Uncertainty":
        """
        Calibre les IC sur l'historique EPEX 15min.

        Args:
            epex_df    : prix EPEX 15min ('price_eur_mwh')
            calendar_df: enrichissement calendaire ('saison','type_jour','heure_hce','quart')
        """
        rng = np.random.default_rng(self.seed)

        df = epex_df[["price_eur_mwh"]].join(
            calendar_df[["saison", "type_jour", "heure_hce", "quart"]]
        ).dropna()

        groups = df.groupby(["saison", "type_jour", "heure_hce", "quart"])
        total = len(groups)
        logger.info("Bootstrap IC : %d cellules × %d tirages", total, self.n_boot)

        for (saison, type_jour, h, q), grp in groups:
            # Construction des ratios f_Q observés
            hour_data = df[
                (df["saison"] == saison) &
                (df["type_jour"] == type_jour) &
                (df["heure_hce"] == h)
            ]
            if len(grp) < 10:
                self.boot_stats_[(saison, type_jour, h, q)] = {"p10": 0.9, "p50": 1.0, "p90": 1.1}
                continue

            # Ratio observé : prix_quart / moyenne_heure
            grp = grp.copy()
            grp["hour_key"] = grp.index.floor("h")
            hour_means = hour_data.groupby(hour_data.index.floor("h"))["price_eur_mwh"].mean()
            grp["hour_mean"] = grp["hour_key"].map(hour_means)
            grp = grp[grp["hour_mean"].abs() > 0.1]

            if len(grp) < 5:
                self.boot_stats_[(saison, type_jour, h, q)] = {"p10": 0.9, "p50": 1.0, "p90": 1.1}
                continue

            ratios = (grp["price_eur_mwh"] / grp["hour_mean"]).values

            # Bootstrap : mediane de chaque tirage
            boot_medians = np.array([
                np.median(rng.choice(ratios, size=len(ratios), replace=True))
                for _ in range(self.n_boot)
            ])

            self.boot_stats_[(saison, type_jour, h, q)] = {
                "p10": float(np.percentile(boot_medians, 10)),
                "p50": float(np.percentile(boot_medians, 50)),
                "p90": float(np.percentile(boot_medians, 90)),
            }

        logger.info("Bootstrap IC terminé : %d cellules calibrées", len(self.boot_stats_))
        return self

    def compute(self, pfc_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule p10 et p90 pour chaque timestamp de la PFC.

        Args:
            pfc_df     : DataFrame PFC avec colonnes ['price_shape', 'f_Q', 'profile_type']
            calendar_df: enrichissement calendaire aligné sur pfc_df.index

        Returns:
            DataFrame colonnes ['p10', 'p90'] aligné sur pfc_df.index
        """
        result = pd.DataFrame({"p10": np.nan, "p90": np.nan}, index=pfc_df.index)

        now = pd.Timestamp.now(tz="UTC")
        idx_zurich = pfc_df.index.tz_convert("Europe/Zurich")

        for i, (ts, row) in enumerate(pfc_df.iterrows()):
            cal_row = calendar_df.loc[ts] if ts in calendar_df.index else None
            if cal_row is None:
                continue

            saison = cal_row["saison"]
            type_jour = cal_row["type_jour"]
            h = int(cal_row["heure_hce"])
            q = int(cal_row["quart"])

            key = (saison, type_jour, h, q)
            if key not in self.boot_stats_:
                # Fallback : ±10% autour du prix central
                result.at[ts, "p10"] = row["price_shape"] * 0.90
                result.at[ts, "p90"] = row["price_shape"] * 1.10
                continue

            stats = self.boot_stats_[key]

            # Facteur d'élargissement selon horizon
            months_ahead = max(0, round((ts - now).days / 30))
            widen = self._widening_factor(months_ahead)

            center = row["price_shape"]
            half_width_p10 = abs(center - center * stats["p10"] / max(stats["p50"], 0.01))
            half_width_p90 = abs(center * stats["p90"] / max(stats["p50"], 0.01) - center)

            result.at[ts, "p10"] = center - half_width_p10 * widen
            result.at[ts, "p90"] = center + half_width_p90 * widen

        return result

    def save(self, path: str | Path) -> None:
        records = []
        for (saison, tj, h, q), stats in self.boot_stats_.items():
            records.append({
                "saison": saison, "type_jour": tj, "heure": h, "quart": q,
                **stats
            })
        pd.DataFrame(records).to_parquet(path, index=False)
        logger.info("Uncertainty sauvegardée : %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "Uncertainty":
        df = pd.read_parquet(path)
        obj = cls()
        for _, row in df.iterrows():
            key = (row["saison"], row["type_jour"], int(row["heure"]), int(row["quart"]))
            obj.boot_stats_[key] = {
                "p10": row["p10"], "p50": row["p50"], "p90": row["p90"]
            }
        return obj

    @staticmethod
    def _widening_factor(months_ahead: int) -> float:
        """Retourne le facteur d'élargissement pour un horizon donné."""
        for threshold in sorted(HORIZON_WIDENING.keys()):
            if months_ahead <= threshold:
                return HORIZON_WIDENING[threshold]
        return HORIZON_WIDENING[max(HORIZON_WIDENING.keys())]

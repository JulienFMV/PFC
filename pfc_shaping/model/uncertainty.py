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
    6:  2.50,   # M+1..M+6   : widen to capture forward-spot basis risk
    12: 2.80,   # M+7..M+12  : additional term-structure uncertainty
    24: 3.50,   # Y+2        : wider for structural market changes
    36: 4.20,   # Y+3        : widest for long-term energy transition
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
        Calibre les IC sur l'historique EPEX.

        Uses hourly price variation (day-to-day) rather than intra-hourly f_Q
        ratios, because EPEX data is hourly-resolution (identical prices within
        each hour). For each (saison, type_jour, heure) cell, computes the
        distribution of normalized price deviations:
            dev = (price - cell_mean) / cell_mean

        This captures how much the actual price deviates from the typical
        level for that cell, providing meaningful prediction intervals.

        Args:
            epex_df    : prix EPEX ('price_eur_mwh')
            calendar_df: enrichissement calendaire ('saison','type_jour','heure_hce','quart')
        """
        rng = np.random.default_rng(self.seed)

        df = epex_df[["price_eur_mwh"]].join(
            calendar_df[["saison", "type_jour", "heure_hce", "quart"]]
        ).dropna()

        # ── Hourly-level uncertainty (one entry per hour, not per quarter) ──
        # Group by (saison, type_jour, heure) — 480 cells max
        hourly_groups = df.groupby(["saison", "type_jour", "heure_hce"])
        logger.info("Bootstrap IC : %d hourly cells × %d tirages", len(hourly_groups), self.n_boot)

        hourly_stats: dict[tuple, dict] = {}
        for (saison, type_jour, h), grp in hourly_groups:
            if len(grp) < 20:
                hourly_stats[(saison, type_jour, h)] = {"p10": 0.85, "p50": 1.0, "p90": 1.15}
                continue

            cell_mean = grp["price_eur_mwh"].mean()
            if abs(cell_mean) < 1.0:
                hourly_stats[(saison, type_jour, h)] = {"p10": 0.85, "p50": 1.0, "p90": 1.15}
                continue

            # Normalized price ratio: actual / cell_mean
            ratios = (grp["price_eur_mwh"] / cell_mean).values

            # Bootstrap prediction intervals on the ratio distribution
            boot_p10 = np.empty(self.n_boot)
            boot_p50 = np.empty(self.n_boot)
            boot_p90 = np.empty(self.n_boot)
            for b in range(self.n_boot):
                sample = rng.choice(ratios, size=len(ratios), replace=True)
                boot_p10[b] = np.percentile(sample, 10)
                boot_p50[b] = np.percentile(sample, 50)
                boot_p90[b] = np.percentile(sample, 90)

            hourly_stats[(saison, type_jour, h)] = {
                "p10": float(np.mean(boot_p10)),
                "p50": float(np.mean(boot_p50)),
                "p90": float(np.mean(boot_p90)),
            }

        # ── Propagate to (saison, type_jour, heure, quart) keys ──────────
        # All 4 quarters within an hour get the same hourly uncertainty
        for (saison, type_jour, h), stats in hourly_stats.items():
            for q in range(1, 5):
                self.boot_stats_[(saison, type_jour, h, q)] = stats

        logger.info("Bootstrap IC terminé : %d hourly cells → %d quarter cells",
                     len(hourly_stats), len(self.boot_stats_))
        return self

    def compute(
        self,
        pfc_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
        reference_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Calcule p10 et p90 pour chaque timestamp de la PFC (vectorisé).

        Args:
            pfc_df         : DataFrame PFC avec colonnes ['price_shape', 'f_Q', 'profile_type']
            calendar_df    : enrichissement calendaire aligné sur pfc_df.index
            reference_date : date de référence pour le calcul de l'horizon
                             (défaut = now). Passer la date "as-of" pour le
                             backtest afin que l'horizon widening soit correct.

        Returns:
            DataFrame colonnes ['p10', 'p90'] aligné sur pfc_df.index
        """
        from collections import defaultdict

        n = len(pfc_df)
        prices = pfc_df["price_shape"].values

        now = reference_date if reference_date is not None else pd.Timestamp.now(tz="UTC")
        days_ahead = (pfc_df.index - now).total_seconds() / 86400
        months_ahead = np.maximum(0, np.round(days_ahead / 30)).astype(int)
        widen_arr = np.vectorize(self._widening_factor)(months_ahead)

        saisons = calendar_df["saison"].values
        types_jour = calendar_df["type_jour"].values
        heures = calendar_df["heure_hce"].astype(int).values
        quarts = calendar_df["quart"].astype(int).values

        # Default to ±10% fallback; override for known cells
        stat_p10 = np.full(n, 0.9)
        stat_p50 = np.full(n, 1.0)
        stat_p90 = np.full(n, 1.1)

        # Group indices by (saison, type_jour, heure, quart) — max 1920 keys
        key_groups: dict[tuple, list[int]] = defaultdict(list)
        for i in range(n):
            key_groups[(saisons[i], types_jour[i], heures[i], quarts[i])].append(i)

        for key, indices in key_groups.items():
            if key in self.boot_stats_:
                s = self.boot_stats_[key]
                idx_arr = np.array(indices)
                stat_p10[idx_arr] = s["p10"]
                stat_p50[idx_arr] = s["p50"]
                stat_p90[idx_arr] = s["p90"]

        # Vectorised IC computation
        safe_p50 = np.maximum(stat_p50, 0.01)
        half_lo = np.abs(prices - prices * stat_p10 / safe_p50)
        half_hi = np.abs(prices * stat_p90 / safe_p50 - prices)

        p10_out = prices - half_lo * widen_arr
        p90_out = prices + half_hi * widen_arr

        return pd.DataFrame({"p10": p10_out, "p90": p90_out}, index=pfc_df.index)

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

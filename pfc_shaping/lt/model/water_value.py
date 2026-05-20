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

# L'information hydro perd en fiabilité sur les horizons éloignés.
DEFAULT_HORIZON_HALFLIFE_DAYS = 270.0

# Les déficits ont en pratique un effet plus violent que les surplus.
SCARCITY_MULTIPLIER_MIN = 1.0
SCARCITY_MULTIPLIER_MAX = 2.5
ABUNDANCE_MULTIPLIER_MIN = 0.6
ABUNDANCE_MULTIPLIER_MAX = 1.4

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

    Phase 5 D-A2-1 / D-A3-1 / NEG-03:
        When ``enforce_floor=False`` (default, negative-ready), the multiplicative
        F_WV_FLOOR (0.80) / F_WV_CAP (1.20) clips at lines 394 and 407 are bypassed
        and f_wv flows through unconstrained from ``beta_wv_`` × ``season_sensitivity_``
        × ``horizon_decay`` — no re-fit needed. When ``enforce_floor=True`` (legacy
        rollback path per D-A2-3), the original multiplicative clip behavior is
        preserved exactly. The delta-additive API ``compute_delta_wv()`` is incompatible
        with ``enforce_floor=True`` and raises ValueError per D-A3-4.
    """

    def __init__(self, enforce_floor: bool = False) -> None:
        self.enforce_floor: bool = bool(enforce_floor)
        self.beta_wv_: float = 0.0
        self.season_sensitivity_: dict[str, float] = {}
        self.n_obs_: int = 0
        self.scarcity_multiplier_: float = 1.35
        self.abundance_multiplier_: float = 1.0
        self.horizon_halflife_days_: float = DEFAULT_HORIZON_HALFLIFE_DAYS

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
            self.scarcity_multiplier_ = 1.35
            self.abundance_multiplier_ = 1.0
            self.n_obs_ = len(monthly)
            return self

        # ── Régression sur prix RELATIFS (stationnarité) ──────────────────
        # Régresse prix/moyenne_glissante_12m ~ fill_deviation × saison.
        # Élimine le biais de non-stationnarité (crise 2021-2022).
        #
        # IMPORTANT : la moyenne glissante est strictement *causale*
        # (12 mois passés, mois courant exclu via ``closed="left"``).
        # La version initiale utilisait ``center=True``, qui mélangeait
        # 6 mois passés et 6 mois futurs pour normaliser le prix mensuel
        # — le ratio régressé voyait donc des prix du futur. Sur un fit
        # 2019-2023, l'automne 2022 (crise + niveau de remplissage bas)
        # se retrouvait normalisé par les 6 mois 2023 (post-crise),
        # biaisant β_WV vers le bas et sous-estimant l'effet hydro en
        # production.
        saisons = list(DEFAULT_SEASON_SENSITIVITY.keys())
        X_cols = []

        # Prix relatif : ratio vs moyenne glissante 12 mois (causale).
        rolling_mean = monthly["price_mean"].rolling(
            12, min_periods=6, center=False, closed="left"
        ).mean()
        # Fill edges with expanding mean (strictly past too: the
        # expanding mean at t aggregates 1..t-1 only when ``closed="left"``
        # — but pandas' expanding default includes t. We accept the
        # 1-month overlap on the very early rows; the alternative
        # (NaN-rows) breaks the regression on small histories).
        rolling_mean = rolling_mean.fillna(monthly["price_mean"].expanding().mean())
        rolling_mean = rolling_mean.replace(0, 1.0)  # guard div-by-zero
        monthly["price_ratio"] = monthly["price_mean"] / rolling_mean

        X_df = pd.DataFrame(index=monthly.index)

        # Interactions fill_deviation × saison (no trend needed on relative prices)
        for s in saisons:
            col = f"fd_{s}"
            X_df[col] = monthly["fill_dev_mean"] * (monthly["saison"] == s).astype(float)
            X_cols.append(col)

        X = X_df[X_cols].values
        y = monthly["price_ratio"].values

        try:
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X, y)

            # Coefficients are already in relative space (ratio effect per % fill)
            raw_sensitivities = {}
            for i, s in enumerate(saisons):
                raw_sensitivities[s] = reg.coef_[i]

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

            neg_mask = monthly["fill_dev_mean"] < 0
            pos_mask = monthly["fill_dev_mean"] > 0
            scarcity_slopes = []
            abundance_slopes = []
            for s in saisons:
                season_mask = monthly["saison"] == s

                neg_subset = monthly.loc[season_mask & neg_mask, ["fill_dev_mean", "price_ratio"]].dropna()
                if len(neg_subset) >= 3 and neg_subset["fill_dev_mean"].std() > 1e-8:
                    scarcity_slopes.append(abs(float(np.polyfit(
                        neg_subset["fill_dev_mean"].values.astype(float),
                        neg_subset["price_ratio"].values.astype(float),
                        1,
                    )[0])))

                pos_subset = monthly.loc[season_mask & pos_mask, ["fill_dev_mean", "price_ratio"]].dropna()
                if len(pos_subset) >= 3 and pos_subset["fill_dev_mean"].std() > 1e-8:
                    abundance_slopes.append(abs(float(np.polyfit(
                        pos_subset["fill_dev_mean"].values.astype(float),
                        pos_subset["price_ratio"].values.astype(float),
                        1,
                    )[0])))

            if scarcity_slopes and abundance_slopes:
                self.scarcity_multiplier_ = float(np.clip(
                    np.mean(scarcity_slopes) / max(np.mean(abundance_slopes), 1e-6),
                    SCARCITY_MULTIPLIER_MIN,
                    SCARCITY_MULTIPLIER_MAX,
                ))
                self.abundance_multiplier_ = float(np.clip(
                    np.mean(abundance_slopes) / max(np.mean(scarcity_slopes), 1e-6),
                    ABUNDANCE_MULTIPLIER_MIN,
                    ABUNDANCE_MULTIPLIER_MAX,
                ))
            else:
                self.scarcity_multiplier_ = 1.35
                self.abundance_multiplier_ = 1.0

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
        idx_zurich = timestamps.tz_convert("Europe/Zurich")
        months_ahead = (
            (idx_zurich.year - idx_zurich[0].year) * 12
            + (idx_zurich.month - idx_zurich[0].month)
        ).astype(float)
        horizon_decay = np.exp(
            -np.log(2) * (months_ahead * 30.0) / max(self.horizon_halflife_days_, 1.0)
        )

        asym_multiplier = pd.Series(self.abundance_multiplier_, index=timestamps, dtype=float)
        asym_multiplier.loc[fill_dev_vals < 0] = self.scarcity_multiplier_

        raw_f_wv = 1.0 + beta * fill_dev_vals * season_sens * asym_multiplier * horizon_decay

        # ── Clamping pour éviter les valeurs aberrantes ──────────────────
        if self.enforce_floor:
            raw_f_wv = raw_f_wv.clip(lower=F_WV_FLOOR, upper=F_WV_CAP)
        # Else (Phase 5 default, negative-ready, D-A2-1 / NEG-03): f_wv flows
        # unclipped — sign of (f_wv - 1) determines scarcity (>0) / abundance
        # (<0), converted to a signed €/MWh delta in compute_delta_wv via |B_smooth|.

        # ── Renormalisation par année de livraison : preserve annual forwards ──
        raw_f_wv = pd.Series(raw_f_wv, index=timestamps, name="f_WV")
        f_wv = raw_f_wv.copy()
        delivery_year = idx_zurich.year
        for year in np.unique(delivery_year):
            year_mask = delivery_year == year
            mean_f = float(raw_f_wv.loc[year_mask].mean())
            # WR-06 (Phase 5 code review): the previous guard ``abs(mean_f) > 1e-8``
            # was numerically too lax — for mean_f ~ 1e-7 the division still produces
            # factors of order 1e7, blowing up delta_wv = (f_wv - 1) * |B|. The
            # pathological case is enforce_floor=False (new default) combined with a
            # raw_f_wv ~ 0 fit (improbable in EEX practice but possible). Use a
            # 1e-6 threshold — well above floating-point precision for ratio
            # arithmetic at EEX scales — and skip renormalisation (raw_f_wv stays)
            # with a logged warning when the mean is suspicious.
            if abs(mean_f) > 1e-6:
                f_wv.loc[year_mask] = raw_f_wv.loc[year_mask] / mean_f
            else:
                logger.warning(
                    "WV renormalisation skipped for year %d: mean(raw_f_wv)=%.2e "
                    "below 1e-6 threshold — dividing would amplify noise. "
                    "raw_f_wv kept unrenormalised for this year.",
                    int(year), mean_f,
                )

        # Re-clamping après renormalisation
        if self.enforce_floor:
            f_wv = f_wv.clip(lower=F_WV_FLOOR, upper=F_WV_CAP)
        # Else: see comment at the analogous block above (raw_f_wv).

        logger.info(
            "f_WV appliqué : mean=%.4f, min=%.4f, max=%.4f, β_WV=%.4f",
            f_wv.mean(), f_wv.min(), f_wv.max(), beta,
        )
        return f_wv

    def compute_delta_wv(
        self,
        B_smooth: pd.Series,
        *,
        fill_df: "pd.DataFrame | None",
        calendar_df: pd.DataFrame,
    ) -> pd.Series:
        """Return the delta-additive WaterValue correction in €/MWh.

        Phase 5 D-A3-1 / NEG-03. Sign-invariant by construction:

          delta_wv = (f_wv - 1.0) * |B_smooth|

        Scarcity (f_wv > 1) yields delta_wv > 0 regardless of sign(B),
        meaning the corrected price is shifted UP (in €/MWh) — for B<0 this
        means LESS negative (correct directional semantic). Abundance
        (f_wv < 1) yields delta_wv < 0 — for B<0 this means MORE negative
        (correct directional semantic). The legacy multiplicative path
        ``f_wv × B`` would invert these semantics under B<0.

        The (unclipped) ``f_wv`` is computed via
        ``self.apply(B_smooth.index, calendar_df, fill_df)``, which carries the
        calibrated ``beta_wv_``, ``season_sensitivity_``, ``horizon_decay``
        (no re-fit needed — RESEARCH §Don't Hand-Roll).

        Codex review action item #1 (2026-05-19, see 05-REVIEWS.md):
        ``fill_df`` and ``calendar_df`` are KEYWORD-ONLY (enforced by the ``*``
        separator). The legacy ``apply(timestamps, calendar_df, hydro_forecast)``
        positional signature is easy to swap at call sites — keyword-only
        kwargs prevent silent index-misalignment bugs.

        Raises
        ------
        ValueError
            If ``self.enforce_floor=True``. The delta-additive semantic is
            incompatible with the multiplicative F_WV_FLOOR clip (D-A3-4) —
            an operator who wants the legacy multiplicative behavior must
            use ``apply()`` directly and the assembler's legacy path.
        TypeError
            If ``fill_df`` or ``calendar_df`` are passed positionally (the ``*``
            separator after ``B_smooth`` requires keyword form — codex action #1).

        Parameters
        ----------
        B_smooth : pd.Series
            Post-MSFC smoothed signal in €/MWh. May be signed
            (Phase 5 NEG-01). POSITIONAL (no ``*`` separator before this arg).
        fill_df : pd.DataFrame | None
            Hydro reservoir fill forecast — forwarded to ``apply()``.
            KEYWORD-ONLY (codex action #1).
        calendar_df : pd.DataFrame
            Calendar DataFrame (saison/type_jour/heure/month) — forwarded.
            KEYWORD-ONLY (codex action #1).

        Returns
        -------
        pd.Series
            delta_wv in €/MWh, indexed by ``B_smooth.index``, name='delta_wv'.
            The caller in assembler.build() is expected to assert
            ``delta_wv.index.equals(B_smooth.index)`` per codex action #1
            (precondition guard on the additive use).
        """
        if self.enforce_floor:
            raise ValueError(
                "compute_delta_wv() incompatible avec enforce_floor=True. "
                "Utiliser apply() pour le comportement multiplicatif legacy."
            )
        f_wv = self.apply(B_smooth.index, calendar_df, fill_df)
        delta = (f_wv - 1.0) * B_smooth.abs()
        delta.name = "delta_wv"
        return delta

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
                "scarcity_multiplier": self.scarcity_multiplier_,
                "abundance_multiplier": self.abundance_multiplier_,
                "horizon_halflife_days": self.horizon_halflife_days_,
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
        if "scarcity_multiplier" in df.columns:
            obj.scarcity_multiplier_ = float(df["scarcity_multiplier"].iloc[0])
        if "abundance_multiplier" in df.columns:
            obj.abundance_multiplier_ = float(df["abundance_multiplier"].iloc[0])
        if "horizon_halflife_days" in df.columns:
            obj.horizon_halflife_days_ = float(df["horizon_halflife_days"].iloc[0])
        obj.season_sensitivity_ = dict(
            zip(df["saison"], df["season_sensitivity"])
        )
        return obj

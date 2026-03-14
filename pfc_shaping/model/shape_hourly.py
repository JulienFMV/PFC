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

    def __init__(
        self,
        sigma: float = GAUSSIAN_SIGMA,
        halflife_days: float = 180.0,
        hydro_weight_sigma: float = 0.25,
    ) -> None:
        self.sigma = sigma
        self.halflife_days = halflife_days  # exponential decay half-life
        self.hydro_weight_sigma = hydro_weight_sigma  # kernel bandwidth for reservoir analogue weighting
        self.factors_: dict[tuple[str, str], np.ndarray] = {}
        self.n_obs_: dict[tuple[str, str], int] = {}
        self.f_W_: dict[str, float] = {}  # ratios empiriques par type_jour (global)
        self.f_W_seasonal_: dict[tuple[str, str], float] = {}  # (saison, type_jour) -> ratio
        self.global_factors_: np.ndarray | None = None
        # Horizon-dependent: per-year factors and linear trends
        self.factors_by_year_: dict[tuple[str, str, int], np.ndarray] = {}
        self.trend_per_hour_: dict[tuple[str, str], np.ndarray] = {}  # slope per hour
        # Hydro reservoir analogue data (set by fit if hydro_df is provided)
        self._hydro_fill_weekly: pd.Series | None = None

    def fit(
        self,
        epex_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
        hydro_df: pd.DataFrame | None = None,
    ) -> "ShapeHourly":
        """
        Estime les facteurs de forme sur l'historique EPEX 15min.

        Args:
            epex_df    : DataFrame issu de ingest_epex, colonnes ['price_eur_mwh'],
                         index DatetimeIndex UTC freq≈15min
            calendar_df: DataFrame issu de calendar_ch.enrich_15min_index(),
                         colonnes ['type_jour', 'saison', 'heure_hce', 'quart']
            hydro_df   : Optional weekly hydro reservoir data with 'fill_pct' column.
                         If provided, historical days are weighted by reservoir
                         fill similarity to the most recent level (KYOS analogue
                         approach). This improves shape estimation for Swiss hydro-
                         dominated markets.

        Returns:
            self
        """
        df = epex_df[["price_eur_mwh"]].copy()
        df = df.join(calendar_df[["saison", "type_jour", "heure_hce"]])
        df = df.dropna(subset=["saison", "type_jour", "heure_hce", "price_eur_mwh"])

        # Compute exponential decay weights (recent data counts more)
        t_max = df.index.max()
        days_ago = (t_max - df.index).total_seconds() / 86400.0
        decay_rate = np.log(2) / self.halflife_days
        df["_weight"] = np.exp(-decay_rate * days_ago)

        # ── Analogue-based hydro reservoir weighting (KYOS approach) ──
        if hydro_df is not None and "fill_pct" in hydro_df.columns:
            self._apply_hydro_analogue_weights(df, hydro_df)

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

                # Prix moyen pondéré par heure (exponential decay)
                hourly_mean = (
                    subset.groupby("heure_hce")
                    .apply(
                        lambda g: np.average(g["price_eur_mwh"], weights=g["_weight"]),
                        include_groups=False,
                    )
                )
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

        # Compute per-year profiles and linear trends for horizon-dependent shaping
        self._fit_trends(df)

        logger.info(
            "ShapeHourly fitted : %d cellules, sigma=%.1f, %d trends",
            len(self.factors_), self.sigma, len(self.trend_per_hour_)
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

    def get_for_horizon(self, saison: str, type_jour: str, years_ahead: float = 0.0) -> np.ndarray:
        """
        Retourne f_H ajusté par la tendance pour un horizon donné.

        Pour years_ahead=0, identique à get(). Pour years_ahead>0,
        applique la tendance linéaire estimée sur les profils historiques
        par année (ex: duck curve solaire se creusant).

        Args:
            saison: Saison cible
            type_jour: Type de jour cible
            years_ahead: Nombre d'années dans le futur (0 = court terme)

        Returns:
            np.ndarray shape (24,) — facteurs normalisés à mean=1
        """
        base = self.get(saison, type_jour)
        if years_ahead <= 0.5 or (saison, type_jour) not in self.trend_per_hour_:
            return base

        trend = self.trend_per_hour_[(saison, type_jour)]
        adjusted = base + trend * years_ahead
        # Ensure no negative factors and re-normalize to mean=1
        adjusted = np.maximum(adjusted, 0.1)
        adjusted = adjusted / adjusted.mean()
        return adjusted

    def apply(self, timestamps: pd.DatetimeIndex, calendar_df: pd.DataFrame,
              reference_date: pd.Timestamp | None = None) -> pd.Series:
        """
        Applique les facteurs f_H sur un index 15min futur.

        Si reference_date est fourni, applique des shapes horizon-dépendants
        (trend-adjusted) pour les timestamps éloignés (>1 an).

        Args:
            timestamps  : DatetimeIndex UTC (futur N+3 ans)
            calendar_df : enrichissement calendaire de timestamps
            reference_date : date de référence pour calculer years_ahead

        Returns:
            pd.Series de f_H pour chaque timestamp, index=timestamps
        """
        if reference_date is None:
            reference_date = pd.Timestamp.now(tz="UTC")

        result = pd.Series(index=timestamps, dtype=float, name="f_H")

        for (saison, type_jour), group in calendar_df.groupby(["saison", "type_jour"]):
            for h in range(24):
                mask = (group["heure_hce"] == h)
                idx = group.index[mask]
                if len(idx) == 0:
                    continue

                if self.trend_per_hour_:
                    # Apply horizon-dependent factors per timestamp
                    years_ahead = (idx - reference_date).total_seconds() / (365.25 * 86400)
                    factors_arr = self.get(saison, type_jour)
                    trend = self.trend_per_hour_.get((saison, type_jour))

                    if trend is not None:
                        # Vectorized: base + trend * years_ahead (clamped for short-term)
                        ya = np.maximum(years_ahead.values.astype(float), 0.0)
                        adjusted = factors_arr[h] + trend[h] * ya
                        adjusted = np.maximum(adjusted, 0.1)
                        result.loc[idx] = adjusted
                    else:
                        result.loc[idx] = factors_arr[h]
                else:
                    factors = self.get(saison, type_jour)
                    result.loc[idx] = factors[h]

        # Re-normalize per day to preserve mean=1 constraint
        if self.trend_per_hour_:
            idx_zh = timestamps.tz_convert("Europe/Zurich")
            day_key = pd.Index([f"{t.year}-{t.month:02d}-{t.day:02d}" for t in idx_zh])
            daily_mean = result.groupby(day_key).transform("mean")
            daily_mean = daily_mean.replace(0, 1.0)
            result = result / daily_mean

        if result.isna().any():
            fallback = np.ones(24) if self.global_factors_ is None else self.global_factors_
            na_mask = result.isna()
            missing_cal = calendar_df.loc[na_mask, "heure_hce"].astype(int)
            result.loc[na_mask] = missing_cal.map(lambda h: float(fallback[h])).values
            logger.warning("f_H fallback applied to %d missing timestamps", int(na_mask.sum()))
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
        obj.global_factors_ = obj._compute_global_fallback()
        return obj

    # ---------------------------------------------------------------------------
    # Interne
    # ---------------------------------------------------------------------------

    def _fit_trends(self, df: pd.DataFrame) -> None:
        """
        Compute per-year f_H profiles and fit linear trends per hour.

        This enables horizon-dependent shaping: for Y+2/Y+3, the solar
        duck curve deepens, evening peaks may shift, etc. The trend is
        fitted as a linear regression of f_H(h) across calendar years.

        Only cells with >= 3 years of data get trends.
        """
        df_year = df.index.tz_convert("Europe/Zurich").year if df.index.tz is not None else df.index.year
        years = sorted(set(df_year))
        if len(years) < 3:
            logger.info("Trends: need >= 3 years, have %d — skipping", len(years))
            return

        for saison in SAISONS:
            for type_jour in TYPES_JOUR:
                yearly_profiles = {}
                for yr in years:
                    mask = (
                        (df["saison"] == saison) &
                        (df["type_jour"] == type_jour) &
                        (df_year == yr)
                    )
                    subset = df.loc[mask]
                    if len(subset) < 96:
                        continue
                    hourly_mean = (
                        subset.groupby("heure_hce")["price_eur_mwh"]
                        .mean()
                        .reindex(range(24))
                        .interpolate(method="linear")
                    )
                    daily_mean = hourly_mean.mean()
                    if daily_mean == 0:
                        continue
                    profile = hourly_mean.values / daily_mean
                    yearly_profiles[yr] = profile
                    self.factors_by_year_[(saison, type_jour, yr)] = profile

                if len(yearly_profiles) < 3:
                    continue

                # Fit linear trend per hour across years
                yr_arr = np.array(sorted(yearly_profiles.keys()))
                profiles = np.array([yearly_profiles[y] for y in yr_arr])  # (n_years, 24)

                # Normalize years to "years since last year"
                yr_centered = yr_arr - yr_arr[-1]  # last year = 0

                slopes = np.zeros(24)
                for h in range(24):
                    # Simple linear regression: f_H(h) = a + b * year
                    coeffs = np.polyfit(yr_centered, profiles[:, h], 1)
                    slopes[h] = coeffs[0]  # slope per year

                # Only keep trends that are meaningful (|slope| > 0.001 for at least some hours)
                if np.max(np.abs(slopes)) > 0.001:
                    self.trend_per_hour_[(saison, type_jour)] = slopes
                    logger.debug(
                        "Trend (%s,%s): max_slope=%.4f at h=%d",
                        saison, type_jour, np.max(np.abs(slopes)), np.argmax(np.abs(slopes))
                    )

        logger.info(
            "Trends fitted: %d cells with significant trends",
            len(self.trend_per_hour_)
        )

    def _fit_f_W(self, df: pd.DataFrame) -> None:
        """
        Calcule les ratios empiriques f_W par type_jour depuis l'historique EPEX.

        f_W(type_jour) = prix_moyen(type_jour) / prix_moyen(global)

        Aussi calcule f_W_seasonal_ par (saison, type_jour) pour capturer
        la différence weekend hiver vs été.
        """
        # Use exponential decay weights (consistent with f_H estimation)
        overall_mean = np.average(df["price_eur_mwh"], weights=df["_weight"])
        if overall_mean == 0:
            self.f_W_ = {tj: 1.0 for tj in TYPES_JOUR}
            return

        # ── Global f_W (fallback) ──────────────────────────────────────
        for tj in TYPES_JOUR:
            mask = df["type_jour"] == tj
            subset = df.loc[mask]
            if len(subset) >= 96:  # au moins 1 jour complet
                self.f_W_[tj] = float(
                    np.average(subset["price_eur_mwh"], weights=subset["_weight"])
                    / overall_mean
                )
            else:
                self.f_W_[tj] = 1.0
                logger.warning("f_W(%s) : données insuffisantes — défaut 1.0", tj)

        # Fallback : Ferie_DE → Ferie_CH si pas assez de données
        if self.f_W_.get("Ferie_DE", 1.0) == 1.0 and "Ferie_CH" in self.f_W_:
            self.f_W_["Ferie_DE"] = self.f_W_["Ferie_CH"]

        # ── Seasonal f_W : f_W(saison, type_jour) ─────────────────────
        for saison in SAISONS:
            mask_s = df["saison"] == saison
            season_data = df.loc[mask_s]
            if len(season_data) > 0:
                season_mean = float(np.average(
                    season_data["price_eur_mwh"], weights=season_data["_weight"]
                ))
            else:
                season_mean = overall_mean

            if season_mean == 0 or len(season_data) < 96:
                for tj in TYPES_JOUR:
                    self.f_W_seasonal_[(saison, tj)] = self.f_W_.get(tj, 1.0)
                continue

            for tj in TYPES_JOUR:
                mask_tj = season_data["type_jour"] == tj
                subset = season_data.loc[mask_tj]
                if len(subset) >= 96:
                    self.f_W_seasonal_[(saison, tj)] = float(
                        np.average(subset["price_eur_mwh"], weights=subset["_weight"])
                        / season_mean
                    )
                else:
                    # Fallback to global f_W for this type_jour
                    self.f_W_seasonal_[(saison, tj)] = self.f_W_.get(tj, 1.0)

            # Fallback fériés
            key_de = (saison, "Ferie_DE")
            key_ch = (saison, "Ferie_CH")
            if self.f_W_seasonal_.get(key_de, 1.0) == 1.0 and key_ch in self.f_W_seasonal_:
                self.f_W_seasonal_[key_de] = self.f_W_seasonal_[key_ch]

        logger.info(
            "f_W global : %s",
            {k: round(v, 3) for k, v in self.f_W_.items()},
        )
        logger.info(
            "f_W seasonal : %d cellules calibrées",
            len(self.f_W_seasonal_),
        )

    def _fill_missing_cells(self) -> None:
        """Remplit les cellules vides par interpolation depuis les cellules existantes."""
        self.global_factors_ = self._compute_global_fallback()
        if self.global_factors_ is None:
            self.global_factors_ = np.ones(24)

        for saison in SAISONS:
            for type_jour in TYPES_JOUR:
                key = (saison, type_jour)
                if key in self.factors_:
                    continue

                filled = False

                # 1) Same-season fallback
                for tj_fallback in ["Ouvrable", "Samedi", "Dimanche", "Ferie_CH", "Ferie_DE"]:
                    fb = (saison, tj_fallback)
                    if fb in self.factors_:
                        self.factors_[key] = self.factors_[fb].copy()
                        logger.info(
                            "Cellule (%s,%s) remplie depuis (%s,%s)",
                            saison, type_jour, saison, tj_fallback
                        )
                        filled = True
                        break

                # 2) Same day-type fallback across seasons
                if not filled:
                    for saison_fb in SAISONS:
                        fb = (saison_fb, type_jour)
                        if fb in self.factors_:
                            self.factors_[key] = self.factors_[fb].copy()
                            logger.info(
                                "Cellule (%s,%s) remplie depuis (%s,%s)",
                                saison, type_jour, saison_fb, type_jour
                            )
                            filled = True
                            break

                # 3) Global fallback
                if not filled:
                    self.factors_[key] = self.global_factors_.copy()
                    logger.warning(
                        "Cellule (%s,%s) remplie avec fallback global",
                        saison, type_jour
                    )

    def _apply_hydro_analogue_weights(
        self, df: pd.DataFrame, hydro_df: pd.DataFrame
    ) -> None:
        """
        Multiply existing temporal weights by reservoir-similarity kernel.

        KYOS approach: for each historical timestamp, find the contemporary
        reservoir fill level, compute a Gaussian kernel weight based on
        distance to the most recent fill level, and multiply with the
        temporal decay weight. This gives higher weight to historical
        periods with similar hydro conditions.

        Modifies df["_weight"] in-place.
        """
        fill = hydro_df["fill_pct"].dropna()
        if len(fill) < 10:
            logger.info("Hydro analogue: insufficient data (%d weeks) — skipping", len(fill))
            return

        # Normalize to [0, 1] if stored as percentage (0-100)
        if fill.max() > 1.5:
            fill = fill / 100.0

        # Store for forecast-time analogue lookup
        self._hydro_fill_weekly = fill

        # Current (most recent) fill level
        current_fill = float(fill.iloc[-1])
        logger.info("Hydro analogue: current fill=%.1f%%, σ=%.2f", current_fill * 100, self.hydro_weight_sigma)

        # Map each timestamp in df to its weekly fill level
        # Create daily fill series by forward-filling weekly data
        date_range = pd.date_range(fill.index.min(), df.index.max(), freq="D", tz="UTC")
        fill_daily = fill.reindex(date_range, method="ffill")

        # Map: for each day in df, look up fill_pct
        df_dates = df.index.normalize()
        fill_at_date = fill_daily.reindex(df_dates)

        valid_mask = fill_at_date.notna()
        if valid_mask.sum() == 0:
            logger.info("Hydro analogue: no overlap with EPEX data — skipping")
            return

        fill_values = fill_at_date.values.astype(float)
        # Gaussian kernel: w = exp(-0.5 * ((fill - current) / sigma)^2)
        # Floor at 0.3 to prevent over-aggressive downweighting of
        # dissimilar reservoir states (preserves seasonal diversity)
        sigma = self.hydro_weight_sigma
        hydro_weight = np.exp(-0.5 * ((fill_values - current_fill) / sigma) ** 2)
        hydro_weight = np.where(np.isnan(hydro_weight), 1.0, hydro_weight)
        hydro_weight = np.maximum(hydro_weight, 0.3)  # floor

        # Combine: multiply existing temporal decay weight with hydro analogue weight
        df["_weight"] = df["_weight"].values * hydro_weight

        n_boosted = int((hydro_weight > 0.5).sum())
        logger.info(
            "Hydro analogue: %d/%d timestamps boosted (weight > 0.5)",
            n_boosted, len(hydro_weight),
        )

    def _compute_global_fallback(self) -> np.ndarray | None:
        """Average profile across all available cells; normalized to mean 1."""
        if not self.factors_:
            return None
        stack = np.vstack([v for v in self.factors_.values()])
        mean_profile = np.nanmean(stack, axis=0)
        if mean_profile.shape[0] != 24 or not np.isfinite(mean_profile).all():
            return None
        if mean_profile.mean() == 0:
            return np.ones(24)
        return mean_profile / mean_profile.mean()


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

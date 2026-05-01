"""
shape_intraday.py
-----------------
Estimation des facteurs de shape 15min intra-horaire f_Q(q | h, saison, type_jour).

C'est le COEUR du modèle FMV.

Définition :
    f_Q(q | h, saison, type_jour) =
        prix moyen du quart q dans l'heure h / prix moyen de l'heure h

Contrainte d'énergie :
    mean_q[ f_Q(q | h, ...) ] = 1   ∀ (h, saison, type_jour)
    => Σ f_Q(q) / 4 = 1

Architecture en 2 couches :
    Couche 1 (déterministe) : profil de base historique par régression robuste Huber
    Couche 2 (correction)   : correction additive Ridge sur variables exogènes
                              solar_regime, load_deviation et flow_deviation,
                              appliquée sur toutes les heures (gate R² OOS > 0
                              rejette automatiquement les heures inutiles).
                              flow_deviation capture la flexibilité hydro CH :
                              export à prix haut (turbinage peak), import à
                              prix bas (pompage / baseload).

Grille de calibration :
    4 saisons × 5 types_jour × 24 heures = 480 cellules
    Chaque cellule : 4 poids normalisés (q1..q4)

Usage :
    from model.shape_intraday import ShapeIntraday
    si = ShapeIntraday()
    si.fit(epex_df, entso_df, calendar_df)
    f_q = si.get(saison="Hiver", type_jour="Ouvrable", heure=8)  # array[4]
    si.save("model/shape_intraday.parquet")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, RidgeCV

logger = logging.getLogger(__name__)

SAISONS = ["Hiver", "Printemps", "Ete", "Automne"]
TYPES_JOUR = ["Ouvrable", "Samedi", "Dimanche", "Ferie_CH", "Ferie_DE"]

# Correction exogène sur TOUTES les heures (le gate R² > 0 OOS
# rejette automatiquement les heures où ça n'aide pas)
HEURES_RAMPE = set(range(24))

# Nombre minimum d'observations pour estimer une cellule
MIN_OBS_COUCHE1 = 8   # ≥ 8 occurrences de l'heure dans la cellule
MIN_OBS_COUCHE2 = 50  # pour la régression Ridge sur features exogènes (augmenté de 30)

# Seuil R² OOS minimal pour retenir une correction Layer 2
# Avec peu de données (< 1 an), R² > 0 passe souvent par chance.
# 0.02 = la correction doit expliquer au moins 2% de la variance OOS.
MIN_R2_OOS = 0.02

# Amplitude maximale des coefficients Layer 2 (borne les corrections extrêmes)
MAX_COEF_ABS = 0.05


class ShapeIntraday:
    """
    Modèle de facteurs de forme 15min intra-horaire.

    Attributs après fit() :
        base_factors_  : dict[(saison, type_jour, heure)] -> np.ndarray shape (4,)
        corrections_   : dict[(saison, type_jour, heure)] -> OLS coefficients dict
                         {'intercept': float, 'b_solar': float, 'b_load': float}
        n_obs_         : dict[(saison, type_jour, heure)] -> int
    """

    # Phase 3.1: Structural break — DA 15-min go-live on EPEX (FfE 2025
    # documented a sawtooth pattern emerging on quarter-hourly DA prices
    # post-Oct-2025). The 180-day exponential decay alone is too gradual
    # to isolate the rupture; pre-rupture observations get an additional
    # multiplicative downweight so the post-rupture regime dominates the
    # f_Q profile estimation.
    DA15MIN_RUPTURE_DATE = pd.Timestamp("2025-10-01", tz="UTC")
    PRE_RUPTURE_WEIGHT_FACTOR = 0.25  # extra weight applied to pre-rupture obs

    def __init__(
        self,
        halflife_days: float = 180.0,
        hydro_weight_sigma: float = 0.25,
        apply_da15min_rupture: bool = True,
    ) -> None:
        self.halflife_days = halflife_days
        self.hydro_weight_sigma = hydro_weight_sigma
        self.apply_da15min_rupture = apply_da15min_rupture
        self.base_factors_: dict[tuple, np.ndarray] = {}
        self.corrections_: dict[tuple, dict] = {}
        self.n_obs_: dict[tuple, int] = {}
        self._climatological_fill: pd.Series | None = None  # mean fill per week-of-year

    @staticmethod
    def _flatten_strength(years_ahead: np.ndarray) -> np.ndarray:
        """Intrahour profiles should flatten faster than hourly shapes."""
        ya = np.maximum(np.asarray(years_ahead, dtype=float), 0.0)
        return np.clip((ya - 0.5) / 2.5, 0.0, 0.40)

    def fit(
        self,
        epex_df: pd.DataFrame,
        entso_df: pd.DataFrame | None,
        calendar_df: pd.DataFrame,
        hydro_df: pd.DataFrame | None = None,
    ) -> "ShapeIntraday":
        """
        Estime les facteurs f_Q sur l'historique.

        Args:
            epex_df    : DataFrame EPEX Spot 15min (colonnes ['price_eur_mwh'])
            entso_df   : DataFrame ENTSO-E (colonnes ['solar_regime','load_deviation'])
                         Peut être None → seule la couche 1 sera calibrée
            calendar_df: Enrichissement calendaire (colonnes ['saison','type_jour',
                         'heure_hce','quart'])
            hydro_df   : Optional weekly hydro reservoir data with 'fill_pct' column.
                         If provided, weights historical data by reservoir similarity.
        """
        df = epex_df[["price_eur_mwh"]].copy()
        df = df.join(calendar_df[["saison", "type_jour", "heure_hce", "quart"]])

        if entso_df is not None:
            exo_cols = [c for c in ["solar_regime", "load_deviation", "flow_deviation"] if c in entso_df.columns]
            df = df.join(entso_df[exo_cols])
        else:
            df["solar_regime"] = np.nan
            df["load_deviation"] = np.nan

        if "flow_deviation" not in df.columns:
            df["flow_deviation"] = np.nan

        df = df.dropna(subset=["saison", "type_jour", "heure_hce", "quart", "price_eur_mwh"])

        # ── Temporal decay + hydro analogue weighting ─────────────────
        t_max = df.index.max()
        days_ago = (t_max - df.index).total_seconds() / 86400.0
        decay_rate = np.log(2) / self.halflife_days
        df["_weight"] = np.exp(-decay_rate * days_ago)

        # Phase 3.1: Structural break Oct-2025 (DA 15-min go-live, FfE 2025).
        # Apply only when training data straddles the rupture date and we have
        # at least 30 days of post-rupture observations — otherwise the
        # downweighting would simply throw away most of the data.
        if self.apply_da15min_rupture and df.index.min() < self.DA15MIN_RUPTURE_DATE:
            post_mask = df.index >= self.DA15MIN_RUPTURE_DATE
            n_post = int(post_mask.sum())
            if n_post >= 96 * 30:
                pre_mask = ~post_mask
                df.loc[pre_mask, "_weight"] = (
                    df.loc[pre_mask, "_weight"] * self.PRE_RUPTURE_WEIGHT_FACTOR
                )
                logger.info(
                    "DA15min rupture: downweighted %d pre-2025-10-01 obs by %.2fx "
                    "(%d post-rupture obs kept at full weight)",
                    int(pre_mask.sum()), self.PRE_RUPTURE_WEIGHT_FACTOR, n_post,
                )

        if hydro_df is not None and "fill_pct" in hydro_df.columns:
            self._apply_hydro_weights(df, hydro_df)

        for saison in SAISONS:
            for type_jour in TYPES_JOUR:
                mask_cell = (df["saison"] == saison) & (df["type_jour"] == type_jour)
                cell = df.loc[mask_cell]

                if len(cell) == 0:
                    continue

                for h in range(24):
                    key = (saison, type_jour, h)
                    hour_data = cell[cell["heure_hce"] == h]

                    if len(hour_data) < MIN_OBS_COUCHE1 * 4:
                        continue

                    # ── Couche 1 : profil de base Huber par quart ──────────────
                    base = self._fit_base(hour_data)
                    if base is None:
                        continue
                    self.base_factors_[key] = base
                    self.n_obs_[key] = len(hour_data)

                    # ── Couche 2 : correction exogène (heures de rampe) ────────
                    if h in HEURES_RAMPE:
                        corr = self._fit_correction(hour_data)
                        if corr is not None:
                            self.corrections_[key] = corr

        self._fill_missing_cells()

        logger.info(
            "ShapeIntraday fitted : %d cellules (h,saison,type), "
            "%d avec correction exogène",
            len(self.base_factors_), len(self.corrections_)
        )
        return self

    def get(
        self,
        saison: str,
        type_jour: str,
        heure: int,
        solar_regime: float = 1.0,
        load_deviation: float = 0.0,
        flow_deviation: float = 0.0,
    ) -> np.ndarray:
        """
        Retourne les 4 facteurs f_Q normalisés pour une cellule.

        Args:
            solar_regime  : 0=Faible, 1=Moyen, 2=Fort
            load_deviation: z-score de la charge (0 = charge normale)
            flow_deviation: z-score du flux transfrontalier (0 = flux normal)
                           Positif = import supérieur à la moyenne,
                           Négatif = export supérieur à la moyenne (turbinage)

        Returns:
            np.ndarray shape (4,) avec mean = 1  (contrainte énergie)
        """
        key = (saison, type_jour, heure)

        if key not in self.base_factors_:
            key = self._fallback_key(saison, type_jour, heure)

        factors = self.base_factors_[key].copy()

        # Correction exogène additive
        if key in self.corrections_:
            corr = self.corrections_[key]
            delta = np.array([
                corr.get(f"b_solar_q{q}", 0.0) * solar_regime
                + corr.get(f"b_load_q{q}", 0.0) * load_deviation
                + corr.get(f"b_flow_q{q}", 0.0) * flow_deviation
                + corr.get(f"intercept_q{q}", 0.0)
                for q in range(1, 5)
            ])
            factors = factors + delta

        # Re-normalisation garantie
        factors = factors / factors.mean()
        return factors

    def apply(
        self,
        timestamps: pd.DatetimeIndex,
        calendar_df: pd.DataFrame,
        entso_df: pd.DataFrame | None = None,
        reference_date: pd.Timestamp | None = None,
    ) -> pd.Series:
        """
        Applique f_Q sur un index 15min futur (horizon N+3) — vectorisé.

        Pour l'horizon futur, entso_df contient les prévisions de solar_regime
        et load_deviation (ou NaN → valeur neutre utilisée).

        Returns:
            pd.Series de f_Q, index=timestamps
        """
        from collections import defaultdict

        df_cal = calendar_df.copy()
        if entso_df is not None:
            for col in ["solar_regime", "load_deviation", "flow_deviation"]:
                if col in entso_df.columns:
                    df_cal[col] = entso_df[col].reindex(timestamps)
        else:
            df_cal["solar_regime"] = 1.0
            df_cal["load_deviation"] = 0.0

        df_cal["solar_regime"] = df_cal["solar_regime"].fillna(1.0)
        df_cal["load_deviation"] = df_cal["load_deviation"].fillna(0.0)
        if "flow_deviation" not in df_cal.columns:
            df_cal["flow_deviation"] = 0.0
        df_cal["flow_deviation"] = df_cal["flow_deviation"].fillna(0.0)

        n = len(timestamps)
        f_q_values = np.ones(n)
        if reference_date is None:
            reference_date = pd.Timestamp.now(tz="UTC")
        saisons = df_cal["saison"].values
        types_jour = df_cal["type_jour"].values
        heures = df_cal["heure_hce"].values.astype(int)
        quarts = df_cal["quart"].values.astype(int) - 1  # 0-indexed

        # Group indices by (saison, type_jour, heure) — max 480 groups
        key_groups: dict[tuple, list[int]] = defaultdict(list)
        for i in range(n):
            key_groups[(saisons[i], types_jour[i], heures[i])].append(i)

        for grp_key, indices in key_groups.items():
            saison, type_jour, heure = grp_key
            actual_key = (saison, type_jour, int(heure))
            if actual_key not in self.base_factors_:
                try:
                    actual_key = self._fallback_key(saison, type_jour, int(heure))
                except KeyError:
                    continue

            base = self.base_factors_[actual_key]
            idx_arr = np.array(indices)
            q_vals = quarts[idx_arr]
            years_ahead = (
                (timestamps[idx_arr] - reference_date).total_seconds() / (365.25 * 86400.0)
            ).astype(float)
            flatten = self._flatten_strength(years_ahead)

            if actual_key not in self.corrections_:
                # Simple lookup with horizon-aware flattening
                factors = np.tile(base, (len(idx_arr), 1))
                factors = 1.0 + (factors - 1.0) * (1.0 - flatten[:, None])
                factors = factors / factors.mean(axis=1, keepdims=True)
                f_q_values[idx_arr] = factors[np.arange(len(idx_arr)), q_vals]
            else:
                # Vectorised correction for ramp hours
                corr = self.corrections_[actual_key]
                solar_vals = df_cal["solar_regime"].values[idx_arr]
                load_vals = df_cal["load_deviation"].values[idx_arr]
                flow_vals = df_cal["flow_deviation"].values[idx_arr]
                n_grp = len(idx_arr)

                # Compute all 4 corrected factors per row: (n_grp, 4)
                factors = np.tile(base, (n_grp, 1))
                for q_idx in range(4):
                    factors[:, q_idx] += (
                        corr.get(f"b_solar_q{q_idx+1}", 0.0) * solar_vals
                        + corr.get(f"b_load_q{q_idx+1}", 0.0) * load_vals
                        + corr.get(f"b_flow_q{q_idx+1}", 0.0) * flow_vals
                        + corr.get(f"intercept_q{q_idx+1}", 0.0)
                    )
                # Re-normalize each row (mean of 4 quarters = 1)
                row_means = factors.mean(axis=1, keepdims=True)
                row_means[row_means == 0] = 1.0
                factors /= row_means
                factors = 1.0 + (factors - 1.0) * (1.0 - flatten[:, None])
                factors = factors / factors.mean(axis=1, keepdims=True)

                # Pick the correct quarter per row
                f_q_values[idx_arr] = factors[np.arange(n_grp), q_vals]

        return pd.Series(f_q_values, index=timestamps, name="f_Q")

    def save(self, path: str | Path) -> None:
        """Sauvegarde base_factors_ et corrections_ en deux Parquet."""
        path = Path(path)
        # Facteurs de base
        records = []
        for (saison, tj, h), arr in self.base_factors_.items():
            for q, v in enumerate(arr, start=1):
                records.append({
                    "saison": saison, "type_jour": tj, "heure": h,
                    "quart": q, "f_Q_base": v,
                    "n_obs": self.n_obs_.get((saison, tj, h), 0),
                })
        pd.DataFrame(records).to_parquet(path, index=False)

        # Corrections exogènes
        corr_path = path.with_stem(path.stem + "_corrections")
        if self.corrections_:
            corr_records = []
            for (saison, tj, h), corr in self.corrections_.items():
                row = {"saison": saison, "type_jour": tj, "heure": h}
                row.update(corr)
                corr_records.append(row)
            pd.DataFrame(corr_records).to_parquet(corr_path, index=False)

        logger.info("ShapeIntraday sauvegardé : %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ShapeIntraday":
        """Charge depuis Parquet."""
        path = Path(path)
        df = pd.read_parquet(path)
        obj = cls()
        for (saison, tj, h), grp in df.groupby(["saison", "type_jour", "heure"]):
            grp = grp.sort_values("quart")
            obj.base_factors_[(saison, tj, int(h))] = grp["f_Q_base"].values
            obj.n_obs_[(saison, tj, int(h))] = int(grp["n_obs"].iloc[0])

        corr_path = path.with_stem(path.stem + "_corrections")
        if corr_path.exists():
            cdf = pd.read_parquet(corr_path)
            for _, row in cdf.iterrows():
                key = (row["saison"], row["type_jour"], int(row["heure"]))
                obj.corrections_[key] = row.drop(["saison", "type_jour", "heure"]).to_dict()

        return obj

    # ---------------------------------------------------------------------------
    # Estimation interne
    # ---------------------------------------------------------------------------

    def _fit_base(self, hour_data: pd.DataFrame) -> np.ndarray | None:
        """
        Couche 1 : profil de base par régression robuste Huber.

        Pour chaque quart q ∈ {1,2,3,4} :
            multiplicative regime : ratio(q) = price(q) / mean_hour_price
            additive regime       : delta(q) = price(q) - mean_hour_price,
                                    factor(q) = 1 + delta(q) / scale,
                                    avec scale = median(|hour_mean|).

        Le régime additif est activé automatiquement sur une cellule lorsqu'une
        part significative des heures a un prix moyen proche de zéro (≥3 % avec
        |mean| < 5 €/MWh, ou n'importe quelle heure avec |mean| < 1 €/MWh). Cela évite l'explosion du ratio price/hour_mean
        documentée pendant les régimes de prix négatifs PV (28 % des heures
        DE 2024-2025, ref. pv-magazine; Latini-Piccirilli-Vargiolu 2019).

        Huber downweighte les spikes extrêmes. Sample_weight = decay temporel
        × analogue hydro si disponible.
        """
        has_weights = "_weight" in hour_data.columns

        # Hour means computed once
        hour_means_series = (
            hour_data.groupby(hour_data.index.floor("h"))["price_eur_mwh"].mean()
        )
        if len(hour_means_series) == 0:
            return None
        hm_abs = np.abs(hour_means_series.values)
        scale = float(max(np.median(hm_abs), 1.0))  # floor at 1 €/MWh
        low_price_share = float(np.mean(hm_abs < 5.0)) if len(hm_abs) else 0.0
        any_subscale = bool(np.any(hm_abs < 1.0))
        use_additive = (low_price_share >= 0.03) or any_subscale

        ratios = []
        for q in range(1, 5):
            q_data = hour_data[hour_data["quart"] == q].copy()
            q_data["hour_key"] = q_data.index.floor("h")
            q_data["hour_mean"] = q_data["hour_key"].map(hour_means_series)
            q_data = q_data.dropna(subset=["hour_mean"])

            if use_additive:
                if len(q_data) < MIN_OBS_COUCHE1:
                    return None
                delta = (q_data["price_eur_mwh"] - q_data["hour_mean"]).values
                w = q_data["_weight"].values if has_weights else None
                X = np.ones((len(delta), 1))
                try:
                    hub = HuberRegressor(epsilon=1.35, max_iter=200, fit_intercept=False)
                    hub.fit(X, delta, sample_weight=w)
                    d_q = float(hub.coef_[0])
                except Exception:
                    d_q = (
                        float(np.average(delta, weights=w))
                        if w is not None else float(np.median(delta))
                    )
                ratios.append(1.0 + d_q / scale)
            else:
                # Multiplicative regime (legacy): filter near-zero hour_mean
                q_data2 = q_data[q_data["hour_mean"].abs() > 0.1]
                if len(q_data2) < MIN_OBS_COUCHE1:
                    return None
                q_data2 = q_data2.copy()
                q_data2["ratio"] = q_data2["price_eur_mwh"] / q_data2["hour_mean"]
                X = np.ones((len(q_data2), 1))
                y = q_data2["ratio"].values
                w = q_data2["_weight"].values if has_weights else None
                try:
                    hub = HuberRegressor(epsilon=1.35, max_iter=200, fit_intercept=False)
                    hub.fit(X, y, sample_weight=w)
                    ratios.append(float(hub.coef_[0]))
                except Exception:
                    if w is not None:
                        ratios.append(float(np.average(y, weights=w)))
                    else:
                        ratios.append(float(np.median(y)))

        arr = np.array(ratios)
        if arr.mean() == 0:
            return np.ones(4)
        return arr / arr.mean()  # normalisation

    def _fit_correction(self, hour_data: pd.DataFrame) -> dict | None:
        """
        Couche 2 : correction Ridge par quart sur solar_regime et load_deviation.

        Ridge (α=1.0) au lieu d'OLS pour régulariser et limiter l'overfitting
        sur les heures de rampe avec peu d'observations.

        Validation : le modèle n'est retenu que si le R² out-of-sample
        (50/50 split temporel) est > 0 (i.e. mieux que la moyenne).

        Uses sample_weight from temporal decay + hydro analogue if available.
        """
        cols = ["solar_regime", "load_deviation", "flow_deviation"]
        available = [c for c in cols if c in hour_data.columns]
        if not available:
            return None

        has_weights = "_weight" in hour_data.columns
        result = {}
        for q in range(1, 5):
            q_data = hour_data[hour_data["quart"] == q].dropna(subset=available).copy()
            if len(q_data) < MIN_OBS_COUCHE2:
                continue

            q_data["hour_key"] = q_data.index.floor("h")
            hour_means = hour_data.groupby(hour_data.index.floor("h"))["price_eur_mwh"].mean()
            q_data["hour_mean"] = q_data["hour_key"].map(hour_means)
            q_data = q_data[q_data["hour_mean"].abs() > 0.1]
            q_data["ratio_obs"] = q_data["price_eur_mwh"] / q_data["hour_mean"]

            X = q_data[available].values
            y = q_data["ratio_obs"].values
            w = q_data["_weight"].values if has_weights else None

            try:
                # Validation temporelle : train sur 1ère moitié, test sur 2ème
                split = len(X) // 2
                if split < MIN_OBS_COUCHE2 // 2:
                    continue

                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]
                w_train = w[:split] if w is not None else None

                ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
                ridge.fit(X_train, y_train, sample_weight=w_train)

                # R² out-of-sample : n'accepter que si > 0 (mieux que la moyenne)
                y_pred = ridge.predict(X_test)
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - y_test.mean()) ** 2)
                r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                if r2_oos <= MIN_R2_OOS:
                    continue  # correction n'apporte pas assez → rejet

                # Refit sur toutes les données si validation passée
                ridge.fit(X, y, sample_weight=w)

                # Clip coefficients to prevent extreme corrections
                intercept = float(np.clip(ridge.intercept_, -MAX_COEF_ABS, MAX_COEF_ABS))
                result[f"intercept_q{q}"] = intercept
                for i, col in enumerate(available):
                    if "solar" in col:
                        short = "solar"
                    elif "flow" in col:
                        short = "flow"
                    else:
                        short = "load"
                    coef = float(np.clip(ridge.coef_[i], -MAX_COEF_ABS, MAX_COEF_ABS))
                    result[f"b_{short}_q{q}"] = coef
            except Exception:
                pass

        return result if result else None

    # ---------------------------------------------------------------------------
    # Hydro analogue weighting
    # ---------------------------------------------------------------------------

    def _apply_hydro_weights(self, df: pd.DataFrame, hydro_df: pd.DataFrame) -> None:
        """Multiply temporal weights by reservoir-fill similarity kernel (KYOS)."""
        fill = hydro_df["fill_pct"].dropna()
        if len(fill) < 10:
            return

        if fill.max() > 1.5:
            fill = fill / 100.0

        # Store climatological fill curve for horizon-dependent use
        if hasattr(fill.index, 'isocalendar'):
            week_of_year = fill.index.isocalendar().week.values
        else:
            week_of_year = fill.index.to_series().dt.isocalendar().week.values
        self._climatological_fill = pd.Series(
            fill.values, index=week_of_year,
        ).groupby(level=0).mean()

        current_fill = float(fill.iloc[-1])
        date_range = pd.date_range(fill.index.min(), df.index.max(), freq="D", tz="UTC")
        fill_daily = fill.reindex(date_range, method="ffill")
        df_dates = df.index.normalize()
        fill_at_date = fill_daily.reindex(df_dates)

        fill_values = fill_at_date.values.astype(float)
        sigma = self.hydro_weight_sigma
        hydro_weight = np.exp(-0.5 * ((fill_values - current_fill) / sigma) ** 2)
        hydro_weight = np.where(np.isnan(hydro_weight), 1.0, hydro_weight)
        hydro_weight = np.maximum(hydro_weight, 0.3)
        df["_weight"] = df["_weight"].values * hydro_weight

    # ---------------------------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------------------------

    def _fallback_key(self, saison: str, type_jour: str, heure: int) -> tuple:
        """Retourne la clé de fallback la plus proche."""
        fallback_tj = {"Ferie_DE": "Ferie_CH", "Ferie_CH": "Dimanche", "Dimanche": "Samedi", "Samedi": "Ouvrable"}
        tj = type_jour
        while tj in fallback_tj:
            tj = fallback_tj[tj]
            key = (saison, tj, heure)
            if key in self.base_factors_:
                return key
        # Dernier recours : même heure, Ouvrable, n'importe quelle saison
        for s in SAISONS:
            key = (s, "Ouvrable", heure)
            if key in self.base_factors_:
                return key
        raise KeyError(f"Aucun facteur trouvé pour ({saison}, {type_jour}, h={heure})")

    def _fill_missing_cells(self) -> None:
        """Propage les cellules vides par ordre de priorité."""
        for saison in SAISONS:
            for type_jour in TYPES_JOUR:
                for h in range(24):
                    key = (saison, type_jour, h)
                    if key not in self.base_factors_:
                        try:
                            fb = self._fallback_key(saison, type_jour, h)
                            self.base_factors_[key] = self.base_factors_[fb].copy()
                        except KeyError:
                            self.base_factors_[key] = np.ones(4)
                            logger.warning(
                                "Cellule (%s,%s,h=%d) : fallback → flat [1,1,1,1]",
                                saison, type_jour, h
                            )

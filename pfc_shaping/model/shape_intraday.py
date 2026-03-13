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
    Couche 2 (correction)   : correction additive OLS sur variables exogènes
                              solar_regime et load_deviation, appliquée
                              uniquement sur les heures à forte variance
                              intra-horaire (heures de rampe solaire : 6h-10h, 17h-20h)

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
from sklearn.linear_model import HuberRegressor, Ridge

logger = logging.getLogger(__name__)

SAISONS = ["Hiver", "Printemps", "Ete", "Automne"]
TYPES_JOUR = ["Ouvrable", "Samedi", "Dimanche", "Ferie_CH", "Ferie_DE"]

# Heures à forte variance intra-horaire → correction exogène activée
HEURES_RAMPE = set(range(6, 11)) | set(range(17, 21))

# Nombre minimum d'observations pour estimer une cellule
MIN_OBS_COUCHE1 = 8   # ≥ 8 occurrences de l'heure dans la cellule
MIN_OBS_COUCHE2 = 30  # pour la régression OLS sur features exogènes


class ShapeIntraday:
    """
    Modèle de facteurs de forme 15min intra-horaire.

    Attributs après fit() :
        base_factors_  : dict[(saison, type_jour, heure)] -> np.ndarray shape (4,)
        corrections_   : dict[(saison, type_jour, heure)] -> OLS coefficients dict
                         {'intercept': float, 'b_solar': float, 'b_load': float}
        n_obs_         : dict[(saison, type_jour, heure)] -> int
    """

    def __init__(self) -> None:
        self.base_factors_: dict[tuple, np.ndarray] = {}
        self.corrections_: dict[tuple, dict] = {}
        self.n_obs_: dict[tuple, int] = {}

    def fit(
        self,
        epex_df: pd.DataFrame,
        entso_df: pd.DataFrame | None,
        calendar_df: pd.DataFrame,
    ) -> "ShapeIntraday":
        """
        Estime les facteurs f_Q sur l'historique.

        Args:
            epex_df    : DataFrame EPEX Spot 15min (colonnes ['price_eur_mwh'])
            entso_df   : DataFrame ENTSO-E (colonnes ['solar_regime','load_deviation'])
                         Peut être None → seule la couche 1 sera calibrée
            calendar_df: Enrichissement calendaire (colonnes ['saison','type_jour',
                         'heure_hce','quart'])
        """
        df = epex_df[["price_eur_mwh"]].copy()
        df = df.join(calendar_df[["saison", "type_jour", "heure_hce", "quart"]])

        if entso_df is not None:
            exo_cols = [c for c in ["solar_regime", "load_deviation"] if c in entso_df.columns]
            df = df.join(entso_df[exo_cols])
        else:
            df["solar_regime"] = np.nan
            df["load_deviation"] = np.nan

        df = df.dropna(subset=["saison", "type_jour", "heure_hce", "quart", "price_eur_mwh"])

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
    ) -> np.ndarray:
        """
        Retourne les 4 facteurs f_Q normalisés pour une cellule.

        Args:
            solar_regime  : 0=Faible, 1=Moyen, 2=Fort
            load_deviation: z-score de la charge (0 = charge normale)

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
            for col in ["solar_regime", "load_deviation"]:
                if col in entso_df.columns:
                    df_cal[col] = entso_df[col].reindex(timestamps)
        else:
            df_cal["solar_regime"] = 1.0
            df_cal["load_deviation"] = 0.0

        df_cal["solar_regime"] = df_cal["solar_regime"].fillna(1.0)
        df_cal["load_deviation"] = df_cal["load_deviation"].fillna(0.0)

        n = len(timestamps)
        f_q_values = np.ones(n)
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

            if actual_key not in self.corrections_:
                # Simple lookup — base factors already normalized
                f_q_values[idx_arr] = base[q_vals]
            else:
                # Vectorised correction for ramp hours
                corr = self.corrections_[actual_key]
                solar_vals = df_cal["solar_regime"].values[idx_arr]
                load_vals = df_cal["load_deviation"].values[idx_arr]
                n_grp = len(idx_arr)

                # Compute all 4 corrected factors per row: (n_grp, 4)
                factors = np.tile(base, (n_grp, 1))
                for q_idx in range(4):
                    factors[:, q_idx] += (
                        corr.get(f"b_solar_q{q_idx+1}", 0.0) * solar_vals
                        + corr.get(f"b_load_q{q_idx+1}", 0.0) * load_vals
                        + corr.get(f"intercept_q{q_idx+1}", 0.0)
                    )
                # Re-normalize each row (mean of 4 quarters = 1)
                row_means = factors.mean(axis=1, keepdims=True)
                row_means[row_means == 0] = 1.0
                factors /= row_means

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
            ratio(q) = price(q) / mean_hour_price
        Puis normalisation : ratio / mean(ratio).

        Huber est préféré à la moyenne simple car il downweighte les spikes
        extrêmes (prix négatifs profonds, pointes de réglage secondaire).
        """
        ratios = []
        for q in range(1, 5):
            q_prices = hour_data[hour_data["quart"] == q]["price_eur_mwh"].values
            # Grouper par "occurrence d'heure" pour calculer le ratio vs la moyenne horaire
            # On reconstitue les groupes par (date, heure)
            q_data = hour_data[hour_data["quart"] == q].copy()

            # Indice temporel sans le quart → clé de l'heure parente
            q_data["hour_key"] = q_data.index.floor("h")
            hour_means = hour_data.groupby(hour_data.index.floor("h"))["price_eur_mwh"].mean()
            q_data["hour_mean"] = q_data["hour_key"].map(hour_means)
            q_data = q_data[q_data["hour_mean"].abs() > 0.1]  # évite div/0 autour de 0

            if len(q_data) < MIN_OBS_COUCHE1:
                return None

            q_data["ratio"] = q_data["price_eur_mwh"] / q_data["hour_mean"]

            # Régression Huber : ratio ~ 1 (intercept only, sans features)
            X = np.ones((len(q_data), 1))
            y = q_data["ratio"].values
            try:
                hub = HuberRegressor(epsilon=1.35, max_iter=200)
                hub.fit(X, y)
                ratios.append(hub.intercept_ + hub.coef_[0])
            except Exception:
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

        Modèle par quart q :
            résidu(q) = f_Q_observed(q) - f_Q_base(q)
                      = α_q + β_sol_q × solar_regime + β_load_q × load_deviation

        Retourne un dict avec les coefficients par quart.
        """
        cols = ["solar_regime", "load_deviation"]
        available = [c for c in cols if c in hour_data.columns]
        if not available:
            return None

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

            try:
                # Validation temporelle : train sur 1ère moitié, test sur 2ème
                split = len(X) // 2
                if split < MIN_OBS_COUCHE2 // 2:
                    continue

                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                ridge = Ridge(alpha=1.0)
                ridge.fit(X_train, y_train)

                # R² out-of-sample : n'accepter que si > 0 (mieux que la moyenne)
                y_pred = ridge.predict(X_test)
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - y_test.mean()) ** 2)
                r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                if r2_oos <= 0:
                    continue  # correction n'apporte rien → on la rejette

                # Refit sur toutes les données si validation passée
                ridge.fit(X, y)
                result[f"intercept_q{q}"] = float(ridge.intercept_)
                for i, col in enumerate(available):
                    short = "solar" if "solar" in col else "load"
                    result[f"b_{short}_q{q}"] = float(ridge.coef_[i])
            except Exception:
                pass

        return result if result else None

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

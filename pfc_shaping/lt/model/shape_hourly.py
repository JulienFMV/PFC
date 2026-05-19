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

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

# Paramètre du lissage gaussien en unités d'heures
GAUSSIAN_SIGMA = 0.5

# Defaults for hydro kernel bandwidth — flag-aware (Plan 05C-01, D-A1-4 / D-A3-2).
# These are the canonical defaults used by __init__ and _apply_hydro_analogue_weights.
# NEVER compare against the received params directly — always compare against these
# constants (RESEARCH Pitfall 3: conflict detection must not fire on default combos).
# flag=OFF: 0.25 (legacy value — kernel on ±30pp anomalies from current_fill scalar)
# flag=ON:  0.08 (calibrated for ±10pp anomalies from per-timestamp clim target; dry-run
#           on N(0,0.10) distribution gives CV=0.393, equivalent sélectivité to legacy
#           0.384 on ±30pp — see 05C-RESEARCH.md §Lever 1 for full calibration table)
_HYDRO_WEIGHT_SIGMA_OFF_DEFAULT: float = 0.25
_HYDRO_WEIGHT_SIGMA_ON_DEFAULT: float = 0.08

# Per cross-AI review consensus (Plan 05B-02): use a model-specific stem instead of a
# generic `_meta.parquet` to avoid sidecar-name collisions when sibling components
# (e.g. future ShapeIntraday) save into the same directory. The convention is
# `${stem}.meta.parquet` derived from the main artifact filename.
_META_SIDECAR_SUFFIX = ".meta.parquet"

# Feature flag: enable seasonal-hourly shape behaviour introduced in Phase 5bis-B.
# Read once at __init__ time and stored in self._use_seasonal_hourly (freeze-at-init).
# Accepted values in os.environ: "1" → True, "0" (or unset) → False (default off).
# Any other value emits a logger.warning and is treated as False.
_FLAG_ENV_VAR = "PFC_LT_USE_SEASONAL_HOURLY_SHAPE"


def _resolve_flag(explicit: bool | None) -> bool:
    """Resolve the use_seasonal_hourly feature flag from constructor arg or env var.

    Precedence (D-06):
    1. If explicit is not None → use it directly (constructor wins).
    2. Otherwise read os.getenv(_FLAG_ENV_VAR, "0") exactly once:
       - "1" → True
       - "0" (or unset) → False
       - Any other value → log warning, treat as False (default off).

    The env var is NEVER re-read after __init__; calling code must not call
    _resolve_flag() again after construction.
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.getenv(_FLAG_ENV_VAR, "0")
    if raw == "1":
        return True
    if raw == "0":
        return False
    logger.warning(
        "Invalid value %r for %s; treating as '0' (default off)",
        raw,
        _FLAG_ENV_VAR,
    )
    return False


def _meta_path(main_path) -> Path:
    """Return the path of the meta sidecar for a given main artifact path.

    Convention: `${stem}.meta.parquet` derived from the main artifact filename.
    E.g. "shape_hourly.parquet" -> "shape_hourly.meta.parquet".
    Centralizes the path computation so save and load cannot drift.
    """
    p = Path(main_path)
    return p.with_name(p.stem + _META_SIDECAR_SUFFIX)

SAISONS = ["Hiver", "Printemps", "Ete", "Automne"]
TYPES_JOUR = ["Ouvrable", "Samedi", "Dimanche", "Ferie_CH", "Ferie_DE"]


class _Factors3DView(Mapping):
    """Read-only Mapping facade over the nested factors_ dict.

    Keys are 3-tuples (saison: str, type_jour: str, hour: int) where
    hour ∈ [0, 24).  Values are the underlying float from the array at
    that hour index.

    The view stores a *reference* to the parent factors_ dict (no copy),
    so mutations to factors_ (adding cells or mutating arrays in-place)
    are immediately reflected in the view — intentional liveness behavior
    (per SHP-01 literal, Plan 05B-04 D-01).

    This class is intentionally NOT part of the public API; access it via
    the ``ShapeHourly.factors_3d_`` property.
    """

    __slots__ = ("_factors",)

    def __init__(self, factors: dict) -> None:
        self._factors = factors  # reference, not copy — live facade

    def __getitem__(self, key) -> float:
        if not (isinstance(key, tuple) and len(key) == 3):
            raise KeyError(key)
        saison, type_jour, hour = key
        if not (isinstance(hour, int) and 0 <= hour < 24):
            raise KeyError(key)
        try:
            arr = self._factors[(saison, type_jour)]
        except KeyError:
            raise KeyError(key)
        return float(arr[hour])

    def __iter__(self) -> Iterator:
        for (saison, type_jour) in self._factors:
            for h in range(24):
                yield (saison, type_jour, h)

    def __len__(self) -> int:
        return len(self._factors) * 24

    def __contains__(self, key) -> bool:
        if not (isinstance(key, tuple) and len(key) == 3):
            return False
        saison, type_jour, hour = key
        if not (isinstance(hour, int) and 0 <= hour < 24):
            return False
        return (saison, type_jour) in self._factors

    def __setitem__(self, key, value) -> None:
        raise TypeError(f"{type(self).__name__} is read-only")

    def __repr__(self) -> str:  # pragma: no cover
        return f"_Factors3DView({len(self)} entries across {len(self._factors)} cells)"


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
        hydro_weight_sigma: float | None = None,
        hydro_weight_sigma_off: float = _HYDRO_WEIGHT_SIGMA_OFF_DEFAULT,
        hydro_weight_sigma_on: float = _HYDRO_WEIGHT_SIGMA_ON_DEFAULT,
        use_seasonal_hourly: bool | None = None,
    ) -> None:
        self.sigma = sigma
        self.halflife_days = halflife_days  # exponential decay half-life
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
        self._climatological_fill: pd.Series | None = None  # mean fill per week-of-year
        # Feature flag: resolved once at __init__ and frozen.
        # Use the constructor kwarg or set env var PFC_LT_USE_SEASONAL_HOURLY_SHAPE BEFORE calling __init__.
        # Persisted into the ${stem}.meta.parquet sidecar by save() and overwritten by load()
        # (parquet wins over env — prevents train/serve skew, D-07).
        # In Phase 5bis-A this flag gates NO behavior; Phase 5bis-B uses it to gate the
        # per-timestamp climatological kernel target (D-A1-2).
        self._use_seasonal_hourly: bool = _resolve_flag(use_seasonal_hourly)

        # Hydro kernel bandwidth — flag-aware resolution (D-A1-4 / D-A3-2, Plan 05C-01).
        # Resolution precedence: legacy hydro_weight_sigma wins if not None (backward-compat
        # for existing callsites: ShapeHourly(hydro_weight_sigma=0.25) etc.).
        # When both hydro_weight_sigma (legacy) AND hydro_weight_sigma_off/_on are supplied
        # simultaneously (i.e. caller passed non-default off/on values), warn and let legacy win.
        if hydro_weight_sigma is not None:
            # Legacy single-sigma caller: applies to both flag states (bit-pour-bit preservation)
            if (hydro_weight_sigma_off != _HYDRO_WEIGHT_SIGMA_OFF_DEFAULT
                    or hydro_weight_sigma_on != _HYDRO_WEIGHT_SIGMA_ON_DEFAULT):
                logger.warning(
                    "ShapeHourly: hydro_weight_sigma=%r (legacy) AND "
                    "hydro_weight_sigma_off=%r/hydro_weight_sigma_on=%r both passed; "
                    "legacy hydro_weight_sigma wins for both flag states (D-A3-2)",
                    hydro_weight_sigma,
                    hydro_weight_sigma_off,
                    hydro_weight_sigma_on,
                )
            self._hydro_weight_sigma_off: float = float(hydro_weight_sigma)
            self._hydro_weight_sigma_on: float = float(hydro_weight_sigma)
        else:
            self._hydro_weight_sigma_off = float(hydro_weight_sigma_off)
            self._hydro_weight_sigma_on = float(hydro_weight_sigma_on)

        # Active-value: the runtime kernel bandwidth read by _apply_hydro_analogue_weights.
        # Resolved from the flag state at init (freeze-at-init pattern, D-06).
        self.hydro_weight_sigma: float = (
            self._hydro_weight_sigma_on if self._use_seasonal_hourly
            else self._hydro_weight_sigma_off
        )

    @property
    def factors_3d_(self) -> Mapping:
        """Read-only 3D view on factors_ keyed by (saison, type_jour, hour). SHP-01 literal.

        Returns a live Mapping backed by self.factors_ (no copy).  Accessing an
        element is equivalent to ``self.factors_[(saison, type_jour)][hour]`` cast to
        float.  The view reflects any subsequent mutation of self.factors_.

        This property constructs a new _Factors3DView wrapper on every access, but
        each wrapper shares the same underlying dict reference, so multiple views
        are always consistent with each other and with the authoritative state.
        """
        return _Factors3DView(self.factors_)

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
        trend_strength = float(self._trend_strength(np.array([years_ahead]))[0])
        flatten_strength = float(self._flatten_strength(np.array([years_ahead]))[0])
        adjusted = base + trend * trend_strength
        adjusted = 1.0 + (adjusted - 1.0) * (1.0 - flatten_strength)
        # Clamp to physically reasonable range and re-normalize to mean=1
        adjusted = np.clip(adjusted, 0.4, 2.0)
        adjusted = adjusted / adjusted.mean()
        return adjusted

    @staticmethod
    def _trend_strength(years_ahead: np.ndarray) -> np.ndarray:
        """Trend matters most at medium horizon, not infinitely far away."""
        ya = np.maximum(np.asarray(years_ahead, dtype=float), 0.0)
        ramp = np.clip(ya / 1.0, 0.0, 1.0)
        decay = np.where(ya <= 1.0, 1.0, np.exp(-(ya - 1.0) / 1.5))
        return ramp * decay

    @staticmethod
    def _flatten_strength(years_ahead: np.ndarray) -> np.ndarray:
        """Far-horizon hourly shapes should gradually converge to a stable profile."""
        ya = np.maximum(np.asarray(years_ahead, dtype=float), 0.0)
        return np.clip((ya - 1.0) / 3.0, 0.0, 0.25)

    def get_climatological_fill(self, week: int) -> float:
        """Return the climatological (long-term mean) fill level for a given ISO week."""
        if self._climatological_fill is None:
            return 0.5  # neutral default
        if week in self._climatological_fill.index:
            return float(self._climatological_fill[week])
        # Interpolate nearest
        return float(self._climatological_fill.iloc[
            (self._climatological_fill.index - week).abs().argmin()
        ])

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
                        # Vectorized: medium-horizon trend, then convergence to a stable profile
                        ya = np.maximum(years_ahead.values.astype(float), 0.0)
                        trend_strength = self._trend_strength(ya)
                        flatten_strength = self._flatten_strength(ya)
                        adjusted = factors_arr[h] + trend[h] * trend_strength
                        adjusted = 1.0 + (adjusted - 1.0) * (1.0 - flatten_strength)
                        adjusted = np.clip(adjusted, 0.4, 2.0)
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
            # Re-clip after normalization
            result = result.clip(lower=0.4, upper=2.0)

        if result.isna().any():
            fallback = np.ones(24) if self.global_factors_ is None else self.global_factors_
            na_mask = result.isna()
            missing_cal = calendar_df.loc[na_mask, "heure_hce"].astype(int)
            result.loc[na_mask] = missing_cal.map(lambda h: float(fallback[h])).values
            logger.warning("f_H fallback applied to %d missing timestamps", int(na_mask.sum()))
        return result

    def save(self, path: str | Path) -> None:
        """Sauvegarde les facteurs f_H et f_W en Parquet, plus un sidecar meta."""
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

        # ── Sidecar ${stem}.meta.parquet — persist all currently-lost trained attributes ──
        # (Plan 05B-02: fix pre-existing save/load bug — factors_by_year_, trend_per_hour_,
        # f_W_seasonal_, _climatological_fill, and scalar hyperparams were silently lost.)
        meta_records: list[dict] = []

        # All values stored as strings in the `value` column to allow heterogeneous types
        # (floats for numerical attrs, JSON for hyperparams) in a single parquet file.
        # The load() method casts back to the appropriate Python type per `attr`.

        # factors_by_year_: long format (saison, type_jour, year, heure, value)
        for (saison, type_jour, year), arr in self.factors_by_year_.items():
            for h, v in enumerate(arr):
                meta_records.append({
                    "attr": "factors_by_year_",
                    "saison": saison,
                    "type_jour": type_jour,
                    "year": int(year),
                    "heure": int(h),
                    "value": repr(float(v)),
                })

        # trend_per_hour_: long format (saison, type_jour, heure, value)
        for (saison, type_jour), arr in self.trend_per_hour_.items():
            for h, v in enumerate(arr):
                meta_records.append({
                    "attr": "trend_per_hour_",
                    "saison": saison,
                    "type_jour": type_jour,
                    "heure": int(h),
                    "value": repr(float(v)),
                })

        # f_W_seasonal_: 1 row per (saison, type_jour)
        for (saison, type_jour), v in self.f_W_seasonal_.items():
            meta_records.append({
                "attr": "f_W_seasonal_",
                "saison": saison,
                "type_jour": type_jour,
                "value": repr(float(v)),
            })

        # _climatological_fill: week -> value (absent if None — sentinel = absence)
        if self._climatological_fill is not None:
            for week, v in self._climatological_fill.items():
                meta_records.append({
                    "attr": "_climatological_fill",
                    "week": int(week),
                    "value": repr(float(v)),
                })

        # _hydro_fill_weekly: timestamp (ISO-8601 str) -> value (absent if None)
        if self._hydro_fill_weekly is not None:
            for ts, v in self._hydro_fill_weekly.items():
                meta_records.append({
                    "attr": "_hydro_fill_weekly",
                    "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                    "value": repr(float(v)),
                })

        # Scalar hyperparams — always present (JSON-string with sorted keys for determinism)
        # use_seasonal_hourly included to prevent train/serve skew (D-07): parquet wins over env at load.
        # Plan 05C-01 (D-A3-3): adds hydro_weight_sigma_off/_on/_resolved for flag-aware persistence.
        # Legacy key hydro_weight_sigma carries the resolved/active value for 5bis-A reader compat.
        meta_records.append({
            "attr": "hyperparams",
            "value": json.dumps(
                {
                    "halflife_days": self.halflife_days,
                    "hydro_weight_sigma": self.hydro_weight_sigma,            # resolved/active (compat)
                    "hydro_weight_sigma_off": self._hydro_weight_sigma_off,   # flag=OFF bandwidth
                    "hydro_weight_sigma_on": self._hydro_weight_sigma_on,     # flag=ON bandwidth
                    "hydro_weight_sigma_resolved": self.hydro_weight_sigma,   # explicit alias for resolved
                    "sigma": self.sigma,
                    "use_seasonal_hourly": bool(self._use_seasonal_hourly),
                },
                sort_keys=True,
            ),
        })

        # NOTE: global_factors_ is intentionally NOT written to the meta sidecar.
        # It is a deterministic function of factors_ (reconstructed via
        # _compute_global_fallback at load time). Persisting it would duplicate state
        # and risk drift. See Plan 05B-02 review consensus.

        meta_path = _meta_path(path)
        pd.DataFrame(meta_records).to_parquet(meta_path, index=False)

        logger.info("ShapeHourly sauvegardé : %s (+ %s sidecar)", path, meta_path.name)

    @classmethod
    def load(cls, path: str | Path) -> "ShapeHourly":
        """Charge depuis un fichier Parquet + ${stem}.meta.parquet sidecar (Plan 05B-02)."""
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

        # ── Restore from ${stem}.meta.parquet sidecar ──
        meta_path = _meta_path(path)
        if meta_path.exists():
            meta_df = pd.read_parquet(meta_path)

            # -- Scalar hyperparams --
            hp_rows = meta_df[meta_df["attr"] == "hyperparams"]
            if len(hp_rows) > 0:
                hp = json.loads(hp_rows["value"].iloc[0])
                obj.sigma = hp.get("sigma", obj.sigma)
                obj.halflife_days = hp.get("halflife_days", obj.halflife_days)
                if "use_seasonal_hourly" in hp:
                    obj._use_seasonal_hourly = bool(hp["use_seasonal_hourly"])
                    # parquet wins over env — prevents train/serve skew (D-07)
                # If "use_seasonal_hourly" key is absent (cross-plan compat — parquet written by
                # 05B-02 before 05B-03 was merged), leave obj._use_seasonal_hourly at the value
                # already set by cls() which read the env via _resolve_flag(None).

                # hydro_weight_sigma cross-plan fallback (Plan 05C-01, D-A3-3):
                # Sidecars written by 5bis-A have only "hydro_weight_sigma" (legacy single-σ).
                # Sidecars written by 5bis-B have "hydro_weight_sigma_off/_on/_resolved".
                if "hydro_weight_sigma_off" in hp:
                    obj._hydro_weight_sigma_off = float(hp["hydro_weight_sigma_off"])
                    obj._hydro_weight_sigma_on = float(hp["hydro_weight_sigma_on"])
                else:
                    # Pre-5bis-B sidecar: only legacy "hydro_weight_sigma" key present.
                    # Apply as single-σ to both flag states (D-A1-5 backward-compat).
                    legacy_hws = float(hp.get("hydro_weight_sigma", _HYDRO_WEIGHT_SIGMA_OFF_DEFAULT))
                    obj._hydro_weight_sigma_off = legacy_hws
                    obj._hydro_weight_sigma_on = legacy_hws
                # Active-value: prefer explicit "hydro_weight_sigma_resolved" key if present,
                # else fall back to legacy "hydro_weight_sigma" (compat), else default.
                obj.hydro_weight_sigma = float(
                    hp.get(
                        "hydro_weight_sigma_resolved",
                        hp.get("hydro_weight_sigma", obj.hydro_weight_sigma),
                    )
                )

            # -- factors_by_year_ --
            # (value column is stored as repr(float) string — cast back to float array)
            fby_rows = meta_df[meta_df["attr"] == "factors_by_year_"]
            if len(fby_rows) > 0:
                for (saison, type_jour, year), grp in fby_rows.groupby(
                    ["saison", "type_jour", "year"]
                ):
                    obj.factors_by_year_[(saison, type_jour, int(year))] = (
                        grp.sort_values("heure")["value"].apply(float).to_numpy()
                    )

            # -- trend_per_hour_ --
            tph_rows = meta_df[meta_df["attr"] == "trend_per_hour_"]
            if len(tph_rows) > 0:
                for (saison, type_jour), grp in tph_rows.groupby(["saison", "type_jour"]):
                    obj.trend_per_hour_[(saison, type_jour)] = (
                        grp.sort_values("heure")["value"].apply(float).to_numpy()
                    )

            # -- f_W_seasonal_ --
            fws_rows = meta_df[meta_df["attr"] == "f_W_seasonal_"]
            if len(fws_rows) > 0:
                for _, row in fws_rows.iterrows():
                    obj.f_W_seasonal_[(row["saison"], row["type_jour"])] = float(row["value"])

            # -- _climatological_fill --
            cf_rows = meta_df[meta_df["attr"] == "_climatological_fill"]
            if len(cf_rows) > 0:
                obj._climatological_fill = pd.Series(
                    cf_rows["value"].apply(float).to_numpy(),
                    index=cf_rows["week"].astype(int).to_numpy(),
                    name="fill_pct",
                )
            else:
                obj._climatological_fill = None

            # -- _hydro_fill_weekly --
            hfw_rows = meta_df[meta_df["attr"] == "_hydro_fill_weekly"]
            if len(hfw_rows) > 0:
                obj._hydro_fill_weekly = pd.Series(
                    hfw_rows["value"].apply(float).to_numpy(),
                    index=pd.to_datetime(hfw_rows["timestamp"], utc=True),
                )
            else:
                obj._hydro_fill_weekly = None

        else:
            # Legacy parquet without meta sidecar — emit one warning, use constructor defaults
            logger.warning(
                "Loading legacy %s without %s sidecar — factors_by_year_, trend_per_hour_, "
                "f_W_seasonal_, _climatological_fill, _hydro_fill_weekly, and scalar hyperparams "
                "(sigma, halflife_days, hydro_weight_sigma) unavailable; falling back to "
                "constructor defaults",
                path,
                meta_path.name,
            )

        # global_factors_ is intentionally NOT persisted to the meta sidecar —
        # reconstructed deterministically from factors_ (see Plan 05B-02 review consensus).
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

                # Exponential decay weights consistent with base profile halflife=180d (P1-05)
                # More recent years get higher weight; yr_centered has last year = 0
                # so -yr_centered gives "years ago" (positive values)
                trend_weights = np.exp(yr_centered * 365 * np.log(2) / 180)

                slopes = np.zeros(24)
                for h in range(24):
                    # Weighted linear regression: f_H(h) = a + b * year
                    coeffs = np.polyfit(yr_centered, profiles[:, h], 1, w=trend_weights)
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

        # Store for forecast-time analogue lookup (includes climatological curve)
        self._hydro_fill_weekly = fill

        # Compute climatological fill curve (mean fill per week-of-year)
        # Used for horizon-dependent analogue weighting: for Y+2/Y+3,
        # use expected fill for that time of year instead of current fill.
        if hasattr(fill.index, 'isocalendar'):
            week_of_year = fill.index.isocalendar().week.values
        else:
            week_of_year = fill.index.to_series().dt.isocalendar().week.values
        self._climatological_fill = pd.Series(
            fill.values, index=week_of_year,
        ).groupby(level=0).mean()

        # Current (most recent) fill level — used as kernel target under flag=OFF (legacy).
        # Under flag=ON (Plan 05C-01, D-A1-1), replaced by per-timestamp clim target.
        current_fill = float(fill.iloc[-1])
        if not self._use_seasonal_hourly:
            logger.info(
                "Hydro analogue: flag=OFF, current fill=%.1f%%, σ=%.2f, clim weeks=%d",
                current_fill * 100,
                self.hydro_weight_sigma,
                len(self._climatological_fill),
            )

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
        # Gaussian kernel: w = exp(-0.5 * ((fill - target) / sigma)^2)
        # Floor at 0.3 to prevent over-aggressive downweighting of
        # dissimilar reservoir states (preserves seasonal diversity) — D-A1-3.
        sigma = self.hydro_weight_sigma

        if self._use_seasonal_hourly:
            # Lever 1 (Plan 05C-01, D-A1-1): per-timestamp climatological fill target.
            # Kernel target = get_climatological_fill(woy(t)) for each historical timestamp,
            # replacing the legacy global scalar current_fill.  This preserves seasonal
            # diversity of the bowl by measuring "anomaly vs climatological norm" instead of
            # "distance from most recent fill level".
            #
            # Scale invariant: get_climatological_fill() returns [0,1] fill (normalized from
            # _climatological_fill which is built from the already-normalized fill Series at
            # lines ~860-875). fill_values is also in [0,1] after the normalize step above.
            # No additional unit conversion required.
            #
            # Vectorized lookup with nearest-neighbor fallback via get_climatological_fill()
            # — safe for fixtures with partial WOY coverage (RESEARCH Pitfall 1).
            # At most 52 dict lookups (unique WOY values), then O(N) array fill.
            if hasattr(df.index, "isocalendar"):
                woy_arr = df.index.isocalendar().week.values
            else:
                woy_arr = df.index.to_series().dt.isocalendar().week.values
            unique_woy = np.unique(woy_arr)
            clim_map = {int(w): self.get_climatological_fill(int(w)) for w in unique_woy}
            clim_target = np.array([clim_map[int(w)] for w in woy_arr], dtype=float)
            logger.info(
                "Hydro analogue: flag=ON, per-timestamp clim target "
                "(mean=%.1f%%), σ=%.2f, clim weeks=%d",
                np.mean(clim_target) * 100,
                sigma,
                len(self._climatological_fill),
            )
        else:
            # Legacy path (flag=OFF): global scalar current_fill — bit-pour-bit identical
            # to 5bis-A baseline (D-A1-2, SC #4).
            clim_target = current_fill  # scalar broadcast in kernel below

        # Private debug attribute: populated by both branches for test verification (D-A4-3).
        # NOT part of the public API. NOT persisted in the sidecar.
        # flag=ON: clim_target is a NumPy array of per-timestamp clim fill values.
        # flag=OFF: clim_target is the scalar float current_fill.
        self._last_clim_target_ = clim_target

        hydro_weight = np.exp(-0.5 * ((fill_values - clim_target) / sigma) ** 2)
        hydro_weight = np.where(np.isnan(hydro_weight), 1.0, hydro_weight)
        hydro_weight = np.maximum(hydro_weight, 0.3)  # floor (D-A1-3)

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

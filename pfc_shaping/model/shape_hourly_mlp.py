"""
shape_hourly_mlp.py
-------------------
Neural shape function for f_H: a small MLP that maps continuous features
to hourly price shape factors, replacing discrete (season, day_type) lookups.

Architecture:
    Input (12): hour_sin, hour_cos, month_sin, month_cos,
                dow_sin, dow_cos, is_holiday, hydro_fill, years_ahead,
                unavailable_nuclear, unavailable_hydro, unavailable_thermal
    Hidden:     2 × 64 neurons, ReLU (sklearn MLPRegressor with Adam)
    Output (1): f_H (unnormalized; post-normalized to mean=1 per day)

Training: sklearn MLPRegressor (Adam optimizer) on historical EPEX spot
          hourly ratios (price_h / price_day_mean).
Inference: sklearn predict (numpy under the hood).

Same interface as ShapeHourly (fit / apply / save / load) for drop-in swap.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

logger = logging.getLogger(__name__)


def _encode_features(
    hour: np.ndarray,
    month: np.ndarray,
    dow: np.ndarray,
    is_holiday: np.ndarray,
    hydro_fill: np.ndarray,
    years_ahead: np.ndarray,
    unavail_nuclear: np.ndarray | None = None,
    unavail_hydro: np.ndarray | None = None,
    unavail_thermal: np.ndarray | None = None,
) -> np.ndarray:
    """Encode raw features into MLP input (n, 12)."""
    n = len(hour)
    two_pi = 2 * np.pi
    # Normalize outage MW to GW for numerical stability
    u_nuc = (unavail_nuclear / 1000.0) if unavail_nuclear is not None else np.zeros(n)
    u_hyd = (unavail_hydro / 1000.0) if unavail_hydro is not None else np.zeros(n)
    u_thm = (unavail_thermal / 1000.0) if unavail_thermal is not None else np.zeros(n)
    X = np.column_stack([
        np.sin(two_pi * hour / 24),
        np.cos(two_pi * hour / 24),
        np.sin(two_pi * (month - 1) / 12),
        np.cos(two_pi * (month - 1) / 12),
        np.sin(two_pi * dow / 7),
        np.cos(two_pi * dow / 7),
        is_holiday.astype(float),
        hydro_fill.astype(float),
        years_ahead.astype(float),
        u_nuc.astype(float),
        u_hyd.astype(float),
        u_thm.astype(float),
    ])
    return X


class ShapeHourlyMLP:
    """
    Neural shape function f_H — drop-in replacement for ShapeHourly.

    Uses a small MLP (9 → 64 → 64 → 1) trained on historical spot price
    hourly ratios. Produces smooth, continuous f_H factors without discrete
    season/day-type bins.
    """

    def __init__(self, halflife_days: float = 180.0) -> None:
        self.halflife_days = halflife_days
        self.mlp_: MLPRegressor | None = None
        self.f_W_: dict[str, float] = {}
        self.f_W_seasonal_: dict[tuple[str, str], float] = {}
        self._hydro_fill_weekly: pd.Series | None = None
        self._climatological_fill: pd.Series | None = None
        # Interface compat with ShapeHourly
        self.factors_: dict = {}
        self.trend_per_hour_: dict = {}

    def fit(
        self,
        epex_df: pd.DataFrame,
        calendar_df: pd.DataFrame,
        hydro_df: pd.DataFrame | None = None,
        outages_df: pd.DataFrame | None = None,
    ) -> "ShapeHourlyMLP":
        """Train the MLP on historical EPEX spot data."""
        df = epex_df[["price_eur_mwh"]].copy()
        df = df.join(calendar_df[["saison", "type_jour", "heure_hce"]])
        df = df.dropna(subset=["saison", "type_jour", "heure_hce", "price_eur_mwh"])

        # Auto-load outages if not provided
        if outages_df is None:
            outages_df = self._try_load_outages()

        # Temporal decay weights
        t_max = df.index.max()
        days_ago = (t_max - df.index).total_seconds() / 86400.0
        decay_rate = np.log(2) / self.halflife_days
        weights = np.exp(-decay_rate * days_ago).values

        # Compute target: f_H = price_h / price_day (hourly ratio)
        idx_zh = df.index.tz_convert("Europe/Zurich")
        day_key = pd.Index([f"{t.year}-{t.month:02d}-{t.day:02d}" for t in idx_zh])
        daily_mean = df["price_eur_mwh"].groupby(day_key).transform("mean")
        # Skip days with zero or negative mean
        valid = (daily_mean > 5.0).values
        df = df[valid]
        weights = weights[valid]
        daily_mean = daily_mean[valid]
        day_key = day_key[valid]
        idx_zh = idx_zh[valid]

        target = (df["price_eur_mwh"] / daily_mean).values
        # Clip extreme ratios (spikes)
        target = np.clip(target, 0.2, 3.0)

        # Encode features
        hour = np.array([t.hour for t in idx_zh], dtype=float)
        month = np.array([t.month for t in idx_zh], dtype=float)
        dow = np.array([t.weekday() for t in idx_zh], dtype=float)
        is_holiday = df["type_jour"].isin(["Ferie_CH", "Ferie_DE"]).values.astype(float)

        # Hydro fill
        hydro_fill = np.full(len(df), 0.5)
        if hydro_df is not None and "fill_pct" in hydro_df.columns:
            self._setup_hydro(hydro_df)
            hydro_fill = self._map_hydro_fill(df.index)

        years_ahead = np.zeros(len(df))  # historical data = 0

        # Outages (unavailable capacity in MW)
        unavail_nuc, unavail_hyd, unavail_thm = self._map_outages(df.index, outages_df)

        X = _encode_features(
            hour, month, dow, is_holiday, hydro_fill, years_ahead,
            unavail_nuc, unavail_hyd, unavail_thm,
        )

        # Aggregate to hourly (average the 4 quarter-hours per hour)
        hour_key = day_key.values.astype(str) + "-" + hour.astype(int).astype(str)
        unique_hours, inverse = np.unique(hour_key, return_inverse=True)
        n_unique = len(unique_hours)

        X_hourly = np.zeros((n_unique, X.shape[1]))
        y_hourly = np.zeros(n_unique)
        w_hourly = np.zeros(n_unique)

        for col in range(X.shape[1]):
            X_hourly[:, col] = np.bincount(inverse, weights=X[:, col] * weights, minlength=n_unique)
        w_hourly = np.bincount(inverse, weights=weights, minlength=n_unique)
        y_hourly = np.bincount(inverse, weights=target * weights, minlength=n_unique)

        valid_h = w_hourly > 0
        X_hourly[valid_h] /= w_hourly[valid_h, None]
        y_hourly[valid_h] /= w_hourly[valid_h]

        # Apply sample weights by repeating high-weight samples
        # sklearn MLPRegressor doesn't support sample_weight natively,
        # so we use the weighted data directly (already weighted-averaged)

        logger.info(
            "MLP training: %d hourly samples from %d raw 15min obs",
            n_unique, len(df),
        )

        # Train sklearn MLP
        self.mlp_ = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            alpha=1e-5,  # light L2 regularization
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=2000,
            tol=1e-6,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            random_state=42,
            batch_size=min(256, n_unique),
        )
        self.mlp_.fit(X_hourly[valid_h], y_hourly[valid_h])

        # Evaluate fit quality
        pred = self.mlp_.predict(X_hourly[valid_h])
        rmse = np.sqrt(np.mean((pred - y_hourly[valid_h]) ** 2))
        logger.info(
            "MLP trained: %d iterations, RMSE(f_H)=%.4f, loss=%.6f",
            self.mlp_.n_iter_, rmse, self.mlp_.loss_,
        )

        # Fit f_W for compatibility
        self._fit_f_W(df, weights, calendar_df)

        return self

    def apply(
        self,
        timestamps: pd.DatetimeIndex,
        calendar_df: pd.DataFrame,
        reference_date: pd.Timestamp | None = None,
        outages_forecast: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Apply MLP to produce f_H for future timestamps."""
        if self.mlp_ is None:
            raise RuntimeError("MLP not fitted — call fit() first")

        if reference_date is None:
            reference_date = pd.Timestamp.now(tz="UTC")

        idx_zh = timestamps.tz_convert("Europe/Zurich")

        hour = np.array([t.hour for t in idx_zh], dtype=float)
        month = np.array([t.month for t in idx_zh], dtype=float)
        dow = np.array([t.weekday() for t in idx_zh], dtype=float)
        is_holiday = calendar_df["type_jour"].isin(
            ["Ferie_CH", "Ferie_DE"]
        ).values.astype(float)

        # Hydro fill: use climatological curve for forecast
        hydro_fill = np.full(len(timestamps), 0.5)
        if self._climatological_fill is not None:
            weeks = np.array([t.isocalendar()[1] for t in idx_zh])
            for w in np.unique(weeks):
                mask = weeks == w
                if w in self._climatological_fill.index:
                    hydro_fill[mask] = float(self._climatological_fill[w])

        # Years ahead
        ya = (timestamps - reference_date).total_seconds() / (365.25 * 86400)
        years_ahead = np.maximum(ya.values.astype(float), 0.0)

        # Outages: use forecast if provided, else zeros (no outage = neutral)
        unavail_nuc, unavail_hyd, unavail_thm = self._map_outages(
            timestamps, outages_forecast,
        )

        X = _encode_features(
            hour, month, dow, is_holiday, hydro_fill, years_ahead,
            unavail_nuc, unavail_hyd, unavail_thm,
        )
        raw = self.mlp_.predict(X)

        # Ensure positivity
        raw = np.maximum(raw, 0.1)

        # Normalize per day to mean=1
        result = pd.Series(raw, index=timestamps, name="f_H")
        day_key = pd.Index([f"{t.year}-{t.month:02d}-{t.day:02d}" for t in idx_zh])
        daily_mean = result.groupby(day_key).transform("mean")
        daily_mean = daily_mean.replace(0, 1.0)
        result = result / daily_mean
        result = result.clip(lower=0.4, upper=2.0)

        return result

    def get(self, saison: str, type_jour: str) -> np.ndarray:
        """Backward-compat: generate 24h profile for a (season, day_type) cell."""
        return self.get_for_horizon(saison, type_jour, years_ahead=0.0)

    def get_for_horizon(self, saison: str, type_jour: str, years_ahead: float = 0.0) -> np.ndarray:
        """Backward-compat: horizon-adjusted profile (MLP handles this natively)."""
        if self.mlp_ is None:
            return np.ones(24)

        month_map = {"Hiver": 1, "Printemps": 4, "Ete": 7, "Automne": 10}
        dow_map = {"Ouvrable": 2, "Samedi": 5, "Dimanche": 6,
                   "Ferie_CH": 6, "Ferie_DE": 6}

        m = month_map.get(saison, 1)
        d = dow_map.get(type_jour, 2)
        is_hol = 1.0 if type_jour in ("Ferie_CH", "Ferie_DE") else 0.0

        hour = np.arange(24, dtype=float)
        month = np.full(24, m, dtype=float)
        dow = np.full(24, d, dtype=float)
        holiday = np.full(24, is_hol)
        hydro = np.full(24, 0.5)
        ya = np.full(24, years_ahead)

        X = _encode_features(hour, month, dow, holiday, hydro, ya)
        raw = self.mlp_.predict(X)
        raw = np.maximum(raw, 0.1)
        raw = raw / raw.mean()
        return raw

    def get_climatological_fill(self, week: int) -> float:
        if self._climatological_fill is None:
            return 0.5
        if week in self._climatological_fill.index:
            return float(self._climatological_fill[week])
        return float(self._climatological_fill.iloc[
            (self._climatological_fill.index - week).abs().argmin()
        ])

    def save(self, path: str | Path) -> None:
        """Save MLP model to pickle."""
        path = Path(path).with_suffix(".pkl")
        save_dict = {
            "mlp": self.mlp_,
            "f_W": self.f_W_,
            "f_W_seasonal": self.f_W_seasonal_,
            "climatological_fill": self._climatological_fill,
        }
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

        logger.info("ShapeHourlyMLP saved: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ShapeHourlyMLP":
        """Load MLP model from pickle."""
        path = Path(path).with_suffix(".pkl")
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        obj = cls()
        obj.mlp_ = save_dict["mlp"]
        obj.f_W_ = save_dict.get("f_W", {})
        obj.f_W_seasonal_ = save_dict.get("f_W_seasonal", {})
        obj._climatological_fill = save_dict.get("climatological_fill")

        logger.info("ShapeHourlyMLP loaded: %s", path)
        return obj

    # ── Internal ─────────────────────────────────────────────────────

    def _setup_hydro(self, hydro_df: pd.DataFrame) -> None:
        fill = hydro_df["fill_pct"].dropna()
        if len(fill) < 10:
            return
        if fill.max() > 1.5:
            fill = fill / 100.0
        self._hydro_fill_weekly = fill
        if hasattr(fill.index, "isocalendar"):
            week_of_year = fill.index.isocalendar().week.values
        else:
            week_of_year = fill.index.to_series().dt.isocalendar().week.values
        self._climatological_fill = pd.Series(
            fill.values, index=week_of_year,
        ).groupby(level=0).mean()

    def _map_hydro_fill(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Map timestamps to hydro fill levels."""
        result = np.full(len(timestamps), 0.5)
        if self._hydro_fill_weekly is None:
            return result

        fill = self._hydro_fill_weekly
        date_range = pd.date_range(fill.index.min(), timestamps.max(), freq="D", tz="UTC")
        fill_daily = fill.reindex(date_range, method="ffill")
        df_dates = timestamps.normalize()
        fill_at_date = fill_daily.reindex(df_dates)
        valid = fill_at_date.notna()
        result[valid.values] = fill_at_date[valid].values.astype(float)
        return result

    @staticmethod
    def _try_load_outages() -> pd.DataFrame | None:
        """Try to load outages parquet if it exists."""
        outage_path = Path(__file__).resolve().parent.parent / "data" / "outages_15min.parquet"
        if outage_path.exists():
            try:
                df = pd.read_parquet(outage_path)
                logger.info("Outages loaded: %d rows, max unavail=%.0f MW",
                            len(df), df["unavailable_mw"].max() if "unavailable_mw" in df.columns else 0)
                return df
            except Exception as e:
                logger.warning("Failed to load outages: %s", e)
        return None

    @staticmethod
    def _map_outages(
        timestamps: pd.DatetimeIndex,
        outages_df: pd.DataFrame | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map outage data onto timestamps. Returns (nuclear, hydro, thermal) in MW."""
        n = len(timestamps)
        zeros = np.zeros(n)
        if outages_df is None or outages_df.empty:
            return zeros.copy(), zeros.copy(), zeros.copy()

        # Reindex outages to match timestamps (forward fill for gaps)
        cols = ["unavailable_nuclear", "unavailable_hydro", "unavailable_thermal"]
        for c in cols:
            if c not in outages_df.columns:
                outages_df[c] = 0.0

        aligned = outages_df[cols].reindex(timestamps, method="ffill").fillna(0.0)
        return (
            aligned["unavailable_nuclear"].values,
            aligned["unavailable_hydro"].values,
            aligned["unavailable_thermal"].values,
        )

    def _fit_f_W(self, df, weights, calendar_df) -> None:
        """Fit f_W ratios (same logic as ShapeHourly for compat)."""
        overall_mean = np.average(df["price_eur_mwh"], weights=weights)
        if overall_mean == 0:
            return
        TYPES_JOUR = ["Ouvrable", "Samedi", "Dimanche", "Ferie_CH", "Ferie_DE"]
        for tj in TYPES_JOUR:
            mask = df["type_jour"] == tj
            subset = df.loc[mask]
            if len(subset) >= 96:
                self.f_W_[tj] = float(
                    np.average(subset["price_eur_mwh"], weights=weights[mask.values])
                    / overall_mean
                )
            else:
                self.f_W_[tj] = 1.0

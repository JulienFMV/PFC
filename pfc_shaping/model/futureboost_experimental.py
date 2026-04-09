"""
Experimental FutureBoosting-like meta-corrector for Swiss short-term forecasting.

The goal is pragmatic: combine the robust LEAR baseline (already including
Chronos-style foundation signals) with an external graph-aware PriceFM forecast
through a lightweight downstream regression layer trained on recent overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pfc_shaping.model.pricefm_experimental import compute_pricefm_regime_weight


@dataclass(frozen=True)
class FutureBoostExperimentalConfig:
    ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0)
    train_fraction: float = 0.70


DEFAULT_FUTUREBOOST_EXPERIMENT = FutureBoostExperimentalConfig()


def build_futureboost_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact, regime-aware feature matrix from LEAR and PriceFM."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    local = out["timestamp"].dt.tz_convert("Europe/Zurich")
    hour = local.dt.hour
    dow = local.dt.dayofweek

    out["pricefm_weight_regime"] = compute_pricefm_regime_weight(out["timestamp"])
    out["forecast_mean"] = 0.5 * (out["lear"] + out["pricefm"])
    out["forecast_diff"] = out["pricefm"] - out["lear"]
    out["forecast_abs_diff"] = out["forecast_diff"].abs()
    out["forecast_ratio"] = out["pricefm"] / out["lear"].clip(lower=1.0)

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["is_weekend"] = (dow >= 5).astype(float)
    out["is_peak"] = hour.isin([7, 8, 9, 17, 18, 19, 20]).astype(float)
    out["is_solar_midday"] = hour.between(10, 15).astype(float)

    # A few targeted interactions are enough; avoid a large unstable design.
    out["diff_x_peak"] = out["forecast_diff"] * out["is_peak"]
    out["diff_x_weekend"] = out["forecast_diff"] * out["is_weekend"]
    out["diff_x_regime"] = out["forecast_diff"] * out["pricefm_weight_regime"]
    out["pricefm_x_regime"] = out["pricefm"] * out["pricefm_weight_regime"]

    feature_cols = [
        "lear",
        "pricefm",
        "forecast_mean",
        "forecast_diff",
        "forecast_abs_diff",
        "forecast_ratio",
        "pricefm_weight_regime",
        "hour_sin",
        "hour_cos",
        "is_weekend",
        "is_peak",
        "is_solar_midday",
        "diff_x_peak",
        "diff_x_weekend",
        "diff_x_regime",
        "pricefm_x_regime",
    ]
    return out[feature_cols]


class FutureBoostExperimentalRegressor:
    """Small ridge meta-model trained on recent LEAR/PriceFM overlap."""

    def __init__(self, config: FutureBoostExperimentalConfig = DEFAULT_FUTUREBOOST_EXPERIMENT):
        self.config = config
        self.pipeline = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("ridge", RidgeCV(alphas=np.asarray(config.ridge_alphas, dtype=float))),
            ]
        )
        self._feature_columns: list[str] | None = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "FutureBoostExperimentalRegressor":
        X = build_futureboost_features(df)
        y = df["actual"].astype(float).to_numpy()
        self.pipeline.fit(X, y)
        self._feature_columns = X.columns.tolist()
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self._feature_columns is None:
            raise RuntimeError("Call fit() before predict().")
        X = build_futureboost_features(df)[self._feature_columns]
        return self.pipeline.predict(X)

    @property
    def alpha_(self) -> float | None:
        if not self._fitted:
            return None
        return float(self.pipeline.named_steps["ridge"].alpha_)

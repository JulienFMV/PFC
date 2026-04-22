"""
Experimental FutureBoosting-like meta-corrector for Swiss short-term forecasting.

The goal is pragmatic: combine the robust LEAR baseline (already including
Chronos-style foundation signals) with an external graph-aware PriceFM forecast
through a lightweight downstream regression layer trained on recent overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pfc_shaping.model.pricefm_experimental import compute_pricefm_regime_weight, load_pricefm_forecast


@dataclass(frozen=True)
class FutureBoostExperimentalConfig:
    ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0)
    train_fraction: float = 0.70
    use_causal_instance_norm: bool = False
    use_qra_quantiles: bool = True
    qra_quantiles: tuple[float, ...] = (0.10, 0.50, 0.90)
    qra_alpha: float = 0.001
    qra_residual_floor: float = 0.5
    history_pricefm_path: str = "pfc_shaping/output/pricefm_ch_full_e5_predictions_hourly.csv"
    history_lear_backtests: tuple[str, ...] = (
        "pfc_shaping/output/lear_backtest_2026-03-15.parquet",
        "pfc_shaping/output/lear_backtest_2026-03-20.parquet",
    )


DEFAULT_FUTUREBOOST_EXPERIMENT = FutureBoostExperimentalConfig()


def _causal_rolling_zscore(
    series: pd.Series,
    window: int = 24 * 7,
    min_periods: int = 24,
) -> pd.Series:
    """Compute a causal rolling z-score (uses only past information)."""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    shifted = s.shift(1)
    mean = shifted.rolling(window=window, min_periods=min_periods).mean()
    std = shifted.rolling(window=window, min_periods=min_periods).std()
    denom = std.replace(0.0, np.nan)
    z = (s - mean) / denom
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _holiday_flags(local_ts: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    years = sorted(set(local_ts.dt.year.astype(int).tolist()))
    ch_holidays: set = set()
    de_holidays: set = set()
    fr_holidays: set = set()
    for year in years:
        ch_holidays |= set(holidays.Switzerland(years=year, subdiv="VS").keys())
        de_holidays |= set(holidays.Germany(years=year).keys())
        fr_holidays |= set(holidays.France(years=year).keys())
    dates = local_ts.dt.date
    is_holiday_ch = dates.isin(ch_holidays).astype(float)
    is_holiday_de = dates.isin(de_holidays).astype(float)
    is_holiday_fr = dates.isin(fr_holidays).astype(float)
    return is_holiday_ch, is_holiday_de, is_holiday_fr


def _load_futureboost_regime_features(
    timestamps: pd.Series | pd.DatetimeIndex,
    entso_path: str | Path = "pfc_shaping/data/entso_15min.parquet",
    outages_path: str | Path = "pfc_shaping/data/outages_15min.parquet",
) -> pd.DataFrame:
    """Load compact regime features aligned on hourly timestamps for FutureBoost."""
    ts = pd.to_datetime(timestamps, utc=True)
    path = Path(entso_path)
    aligned = pd.DataFrame({"timestamp": ts})

    if path.exists():
        entso = pd.read_parquet(path)
        if not entso.empty:
            if not isinstance(entso.index, pd.DatetimeIndex):
                entso.index = pd.to_datetime(entso.index, utc=True)
            elif entso.index.tz is None:
                entso.index = entso.index.tz_localize("UTC")
            else:
                entso.index = entso.index.tz_convert("UTC")

            cols = [
                c for c in [
                    "fr_nuclear_unavailable_mw",
                    "fr_nuclear_unavailability_ratio",
                    "fr_nuclear_stress_flag",
                ]
                if c in entso.columns
            ]
            if cols:
                hourly = entso[cols].resample("h").mean().reset_index().rename(columns={"index": "timestamp"})
                aligned = aligned.merge(hourly, on="timestamp", how="left")

    if "fr_nuclear_unavailable_mw" not in aligned.columns:
        out_path = Path(outages_path)
        if out_path.exists():
            outages = pd.read_parquet(out_path)
            if not outages.empty and "unavailable_nuclear" in outages.columns:
                if not isinstance(outages.index, pd.DatetimeIndex):
                    outages.index = pd.to_datetime(outages.index, utc=True)
                elif outages.index.tz is None:
                    outages.index = outages.index.tz_localize("UTC")
                else:
                    outages.index = outages.index.tz_convert("UTC")
                hourly_outages = (
                    outages[["unavailable_nuclear"]]
                    .rename(columns={"unavailable_nuclear": "fr_nuclear_unavailable_mw"})
                    .resample("h")
                    .mean()
                    .reset_index()
                    .rename(columns={"index": "timestamp"})
                )
                aligned = aligned.merge(hourly_outages, on="timestamp", how="left")

    if "fr_nuclear_unavailable_mw" in aligned.columns:
        unavailable = pd.to_numeric(aligned["fr_nuclear_unavailable_mw"], errors="coerce").fillna(0.0).clip(lower=0.0)
        q75 = float(unavailable.quantile(0.75)) if len(unavailable) else 0.0
        q90 = float(unavailable.quantile(0.90)) if len(unavailable) else 0.0
        scale = max(q90, 1.0)
        stress_threshold = max(5000.0, q75)
        aligned["fr_nuclear_unavailable_mw"] = unavailable
        if "fr_nuclear_unavailability_ratio" not in aligned.columns:
            aligned["fr_nuclear_unavailability_ratio"] = (unavailable / scale).clip(upper=1.5)
        if "fr_nuclear_stress_flag" not in aligned.columns:
            aligned["fr_nuclear_stress_flag"] = (unavailable >= stress_threshold).astype(float)

    for col in [
        "fr_nuclear_unavailable_mw",
        "fr_nuclear_unavailability_ratio",
        "fr_nuclear_stress_flag",
    ]:
        if col not in aligned.columns:
            aligned[col] = 0.0
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0.0)
    return aligned


def build_futureboost_features(
    df: pd.DataFrame,
    use_causal_instance_norm: bool = False,
) -> pd.DataFrame:
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
    if use_causal_instance_norm:
        out["lear_cinz"] = _causal_rolling_zscore(out["lear"])
        out["pricefm_cinz"] = _causal_rolling_zscore(out["pricefm"])
        out["diff_cinz"] = _causal_rolling_zscore(out["forecast_diff"])

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["is_weekend"] = (dow >= 5).astype(float)
    out["is_peak"] = hour.isin([7, 8, 9, 17, 18, 19, 20]).astype(float)
    out["is_solar_midday"] = hour.between(10, 15).astype(float)
    out["is_holiday_ch"], out["is_holiday_de"], out["is_holiday_fr"] = _holiday_flags(local)
    for col in [
        "fr_nuclear_unavailable_mw",
        "fr_nuclear_unavailability_ratio",
        "fr_nuclear_stress_flag",
    ]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    # A few targeted interactions are enough; avoid a large unstable design.
    out["diff_x_peak"] = out["forecast_diff"] * out["is_peak"]
    out["diff_x_weekend"] = out["forecast_diff"] * out["is_weekend"]
    out["diff_x_regime"] = out["forecast_diff"] * out["pricefm_weight_regime"]
    out["pricefm_x_regime"] = out["pricefm"] * out["pricefm_weight_regime"]
    out["diff_lag1"] = out["forecast_diff"].shift(1).ffill().fillna(0.0)
    out["diff_lag24"] = out["forecast_diff"].shift(24).ffill().fillna(0.0)
    out["abs_diff_lag1"] = out["forecast_abs_diff"].shift(1).ffill().fillna(0.0)
    out["abs_diff_lag24"] = out["forecast_abs_diff"].shift(24).ffill().fillna(0.0)
    out["diff_change_1h"] = out["forecast_diff"] - out["diff_lag1"]
    out["diff_x_holiday_ch"] = out["forecast_diff"] * out["is_holiday_ch"]
    out["diff_x_holiday_de"] = out["forecast_diff"] * out["is_holiday_de"]
    out["diff_x_holiday_fr"] = out["forecast_diff"] * out["is_holiday_fr"]
    out["diff_x_fr_nuclear_stress"] = out["forecast_diff"] * out["fr_nuclear_stress_flag"]
    out["pricefm_x_fr_nuclear_stress"] = out["pricefm"] * out["fr_nuclear_stress_flag"]
    if use_causal_instance_norm:
        out["diff_cinz_x_peak"] = out["diff_cinz"] * out["is_peak"]
        out["diff_cinz_x_regime"] = out["diff_cinz"] * out["pricefm_weight_regime"]

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
        "is_holiday_ch",
        "is_holiday_de",
        "is_holiday_fr",
        "fr_nuclear_unavailable_mw",
        "fr_nuclear_unavailability_ratio",
        "fr_nuclear_stress_flag",
        "diff_x_peak",
        "diff_x_weekend",
        "diff_x_regime",
        "pricefm_x_regime",
        "diff_lag1",
        "diff_lag24",
        "abs_diff_lag1",
        "abs_diff_lag24",
        "diff_change_1h",
        "diff_x_holiday_ch",
        "diff_x_holiday_de",
        "diff_x_holiday_fr",
        "diff_x_fr_nuclear_stress",
        "pricefm_x_fr_nuclear_stress",
    ]
    if use_causal_instance_norm:
        feature_cols.extend(
            [
                "lear_cinz",
                "pricefm_cinz",
                "diff_cinz",
                "diff_cinz_x_peak",
                "diff_cinz_x_regime",
            ]
        )
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
        self._qra_models: dict[float, Pipeline] = {}
        self._qra_enabled = False
        self._qra_residual_quantiles: dict[float, float] = {}

    def fit(self, df: pd.DataFrame) -> "FutureBoostExperimentalRegressor":
        X = build_futureboost_features(
            df,
            use_causal_instance_norm=self.config.use_causal_instance_norm,
        )
        y = df["actual"].astype(float).to_numpy()
        self.pipeline.fit(X, y)
        self._feature_columns = X.columns.tolist()
        point_preds = self.pipeline.predict(X)
        residuals = y - point_preds
        for q in self.config.qra_quantiles:
            self._qra_residual_quantiles[float(q)] = float(np.quantile(residuals, q))

        self._qra_models = {}
        self._qra_enabled = False
        if self.config.use_qra_quantiles:
            try:
                for q in self.config.qra_quantiles:
                    qr = Pipeline(
                        steps=[
                            ("scale", StandardScaler()),
                            ("quantile", QuantileRegressor(quantile=float(q), alpha=self.config.qra_alpha)),
                        ]
                    )
                    qr.fit(X, y)
                    self._qra_models[float(q)] = qr
                self._qra_enabled = len(self._qra_models) == len(self.config.qra_quantiles)
            except Exception:
                # Keep a robust fallback based on residual quantiles around the point model.
                self._qra_models = {}
                self._qra_enabled = False
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self._feature_columns is None:
            raise RuntimeError("Call fit() before predict().")
        X = build_futureboost_features(
            df,
            use_causal_instance_norm=self.config.use_causal_instance_norm,
        )[self._feature_columns]
        return self.pipeline.predict(X)

    def predict_quantiles(self, df: pd.DataFrame) -> dict[float, np.ndarray]:
        if not self._fitted or self._feature_columns is None:
            raise RuntimeError("Call fit() before predict_quantiles().")
        X = build_futureboost_features(
            df,
            use_causal_instance_norm=self.config.use_causal_instance_norm,
        )[self._feature_columns]
        if self._qra_enabled and self._qra_models:
            out = {q: self._qra_models[q].predict(X) for q in sorted(self._qra_models)}
        else:
            point = self.pipeline.predict(X)
            out = {
                float(q): point + self._qra_residual_quantiles.get(float(q), 0.0)
                for q in self.config.qra_quantiles
            }

        # Enforce monotonic quantiles row-wise.
        qs_sorted = sorted(out)
        stacked = np.column_stack([out[q] for q in qs_sorted])
        stacked = np.sort(stacked, axis=1)
        return {q: stacked[:, i] for i, q in enumerate(qs_sorted)}

    @property
    def alpha_(self) -> float | None:
        if not self._fitted:
            return None
        return float(self.pipeline.named_steps["ridge"].alpha_)

    @property
    def qra_enabled_(self) -> bool:
        return bool(self._qra_enabled)


def _load_saved_backtest(path: str | Path) -> pd.DataFrame:
    bt = pd.read_parquet(path).copy()
    if "horizon" in bt.columns:
        bt = bt[bt["horizon"] == 1].copy()
    if "date" in bt.columns and "hour" in bt.columns:
        local_midnight = pd.to_datetime(bt["date"]).dt.tz_localize("Europe/Zurich")
        bt["timestamp"] = local_midnight + pd.to_timedelta(bt["hour"], unit="h")
        bt["timestamp"] = bt["timestamp"].dt.tz_convert("UTC")
    elif "forecast_ts" in bt.columns:
        bt["timestamp"] = pd.to_datetime(bt["forecast_ts"], utc=True)
    else:
        raise ValueError(f"Unsupported backtest format: {path}")
    if "forecast" in bt.columns:
        bt = bt.rename(columns={"forecast": "lear"})
    return bt[["timestamp", "actual", "lear"]].drop_duplicates(subset=["timestamp"], keep="last")


def load_futureboost_training_overlap(
    config: FutureBoostExperimentalConfig = DEFAULT_FUTUREBOOST_EXPERIMENT,
) -> pd.DataFrame:
    history_pricefm_path = Path(config.history_pricefm_path)
    if not history_pricefm_path.exists():
        raise FileNotFoundError(f"Historical PriceFM predictions not found: {history_pricefm_path}")

    pricefm = pd.read_csv(history_pricefm_path)
    pricefm["timestamp"] = pd.to_datetime(pricefm["timestamp"], utc=True)
    pricefm = pricefm[["timestamp", "pricefm"]].drop_duplicates(subset=["timestamp"], keep="last")

    bt_frames: list[pd.DataFrame] = []
    for backtest_path in config.history_lear_backtests:
        p = Path(backtest_path)
        if p.exists():
            bt_frames.append(_load_saved_backtest(p))
    if not bt_frames:
        raise FileNotFoundError("No saved LEAR backtest windows available for FutureBoost training.")

    bt = (
        pd.concat(bt_frames, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
    )
    merged = bt.merge(pricefm, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    regime = _load_futureboost_regime_features(merged["timestamp"])
    merged = merged.merge(regime, on="timestamp", how="left")
    if merged.empty:
        raise ValueError("No overlap between saved LEAR backtests and historical PriceFM predictions.")
    return merged


def apply_futureboost_experimental(
    lear_forecast: pd.DataFrame,
    pricefm_forecast: pd.DataFrame,
    config: FutureBoostExperimentalConfig = DEFAULT_FUTUREBOOST_EXPERIMENT,
) -> tuple[pd.DataFrame, dict[str, float | int | str | None]]:
    """Train the experimental meta-layer on saved overlaps, then apply to future forecasts."""
    if "timestamp" not in lear_forecast.columns or "price_lear" not in lear_forecast.columns:
        raise ValueError("LEAR forecast must contain 'timestamp' and 'price_lear'.")

    training = load_futureboost_training_overlap(config=config)
    meta = FutureBoostExperimentalRegressor(config=config).fit(training)

    result = lear_forecast.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    cleanup_cols = [
        "price_pricefm",
        "price_pricefm_x",
        "price_pricefm_y",
        "futureboost_used",
        "price_lear_base",
        "futureboost_alpha",
    ]
    result = result.drop(columns=[c for c in cleanup_cols if c in result.columns], errors="ignore")

    pfm = pricefm_forecast.copy()
    if "price_pricefm" not in pfm.columns:
        pfm = load_pricefm_forecast(pfm)
    else:
        pfm["timestamp"] = pd.to_datetime(pfm["timestamp"], utc=True)

    merged = result.merge(pfm[["timestamp", "price_pricefm"]], on="timestamp", how="left")
    regime = _load_futureboost_regime_features(merged["timestamp"])
    merged = merged.merge(regime, on="timestamp", how="left")
    merged["price_lear_base"] = merged["price_lear"]
    merged = merged.rename(columns={"price_lear": "lear", "price_pricefm": "pricefm"})

    has_pricefm = merged["pricefm"].notna()
    if has_pricefm.any():
        overlap = merged.loc[has_pricefm, ["timestamp", "lear", "pricefm"]]
        preds = meta.predict(overlap)
        merged.loc[has_pricefm, "lear"] = preds
        quantiles = meta.predict_quantiles(overlap)
        q10 = quantiles.get(0.10)
        q50 = quantiles.get(0.50)
        q90 = quantiles.get(0.90)
        if q10 is None or q90 is None:
            # Fallback if custom quantiles are configured differently.
            qs = sorted(quantiles)
            q10 = quantiles[qs[0]]
            q90 = quantiles[qs[-1]]
            q50 = quantiles[qs[len(qs) // 2]]
        merged.loc[has_pricefm, "futureboost_p10"] = q10
        merged.loc[has_pricefm, "futureboost_p50"] = q50
        merged.loc[has_pricefm, "futureboost_p90"] = q90
        merged.loc[has_pricefm, "lear"] = q50
    merged = merged.rename(columns={"lear": "price_lear", "pricefm": "price_pricefm"})
    if "futureboost_p50" in merged.columns:
        spread_floor = max(float(config.qra_residual_floor), float(np.nanstd(training["actual"])))
        merged["futureboost_p50"] = merged["futureboost_p50"].fillna(merged["price_lear"])
        merged["futureboost_p10"] = merged["futureboost_p10"].fillna(
            merged["futureboost_p50"] - spread_floor
        )
        merged["futureboost_p90"] = merged["futureboost_p90"].fillna(
            merged["futureboost_p50"] + spread_floor
        )
        merged["futureboost_p10"] = np.minimum(merged["futureboost_p10"], merged["futureboost_p50"])
        merged["futureboost_p90"] = np.maximum(merged["futureboost_p90"], merged["futureboost_p50"])
    merged["futureboost_used"] = has_pricefm.astype(bool)
    merged["futureboost_alpha"] = meta.alpha_

    metadata = {
        "train_rows": int(len(training)),
        "alpha": meta.alpha_,
        "qra_enabled": bool(meta.qra_enabled_),
        "history_pricefm_path": str(Path(config.history_pricefm_path).resolve()),
        "history_backtests": len([p for p in config.history_lear_backtests if Path(p).exists()]),
        "future_rows": int(len(merged)),
        "future_pricefm_rows": int(has_pricefm.sum()),
    }
    return merged, metadata

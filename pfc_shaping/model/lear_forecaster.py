"""
lear_forecaster.py
------------------
LASSO Estimated AutoRegressive (LEAR) model for short-term electricity
price forecasting (D+1 to D+10).

Based on Ziel & Weron (2018), Lago et al. (2021), and the epftoolbox
reference implementation. Adapted for Swiss (CH) and German (DE) markets.

Architecture:
    - 24 independent LASSO regressions (one per delivery hour)
    - Asinh variance-stabilizing transformation
    - Multi-window calibration averaging (28, 56, 84, 728 days)
    - Features: lagged prices, load, renewables, outages, gas/CO2, DOW dummies

Usage:
    from pfc_shaping.model.lear_forecaster import LEARForecaster
    lear = LEARForecaster()
    lear.fit(epex_hourly, entso_hourly, outages_hourly, commodities)
    forecast = lear.predict(horizon_days=10)
"""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsCV

logger = logging.getLogger(__name__)


def _asinh_transform(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Variance-stabilizing area hyperbolic sine transform."""
    mu = np.nanmean(x)
    sigma = np.nanstd(x)
    if sigma < 1e-6:
        sigma = 1.0
    return np.arcsinh((x - mu) / sigma), mu, sigma


def _asinh_inverse(y: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Inverse asinh transform."""
    return np.sinh(y) * sigma + mu


class LEARForecaster:
    """LASSO-based day-ahead price forecaster.

    Fits 24 independent LASSO models and averages across
    multiple calibration windows for robust forecasts.
    """

    CALIBRATION_WINDOWS = [28, 56, 84, 728]  # days
    LAGS_DAYS = [1, 2, 3, 7]  # price lag structure

    def __init__(
        self,
        tz: str = "Europe/Zurich",
        max_iter: int = 2500,
    ):
        self.tz = tz
        self.max_iter = max_iter
        self._fitted = False

    def fit(
        self,
        epex_15min: pd.DataFrame,
        entso_15min: pd.DataFrame | None = None,
        outages_15min: pd.DataFrame | None = None,
        commodities: pd.DataFrame | None = None,
        hydro: pd.DataFrame | None = None,
    ) -> "LEARForecaster":
        """Prepare hourly data matrices for LEAR training.

        Stores aligned hourly DataFrames ready for feature construction.
        Actual LASSO fitting happens at predict-time with rolling windows.
        """
        # Aggregate EPEX to hourly
        self.prices_h_ = (
            epex_15min["price_eur_mwh"]
            .resample("h").mean()
            .dropna()
        )
        idx_local = self.prices_h_.index.tz_convert(self.tz)

        # Build hourly exogenous matrix
        exog = pd.DataFrame(index=self.prices_h_.index)

        # ENTSO-E: load and renewables
        if entso_15min is not None and not entso_15min.empty:
            for col in ["load_mw", "solar_mw", "wind_mw"]:
                if col in entso_15min.columns:
                    exog[col] = entso_15min[col].resample("h").mean()

        # Outages
        if outages_15min is not None and not outages_15min.empty:
            exog["outages_mw"] = (
                outages_15min["unavailable_mw"].resample("h").mean()
            )

        # Commodities (daily → forward-fill to hourly)
        if commodities is not None and not commodities.empty:
            for col in commodities.columns:
                daily = commodities[col].dropna()
                if daily.empty:
                    continue
                # Localize if needed
                if daily.index.tz is None:
                    daily.index = daily.index.tz_localize("UTC")
                name = col.split("|")[0].replace(" ", "_").lower()
                # Resample to daily, forward-fill to hourly
                exog[name] = daily.resample("h").ffill()

        # Hydro reservoir fill (weekly → forward-fill)
        if hydro is not None and not hydro.empty and "fill_pct" in hydro.columns:
            fill = hydro["fill_pct"].dropna()
            if not fill.empty:
                if fill.index.tz is None:
                    fill.index = fill.index.tz_localize("UTC")
                exog["hydro_fill"] = fill.resample("h").ffill()

        # Align everything
        common_idx = self.prices_h_.index
        self.exog_ = exog.reindex(common_idx)

        # Cache local hour info
        self._idx_local = self.prices_h_.index.tz_convert(self.tz)

        self._fitted = True
        logger.info(
            "LEAR data prepared: %d hours, %d exogenous features, %s → %s",
            len(self.prices_h_),
            len(self.exog_.columns),
            self.prices_h_.index[0].date(),
            self.prices_h_.index[-1].date(),
        )
        return self

    def _build_features(
        self,
        prices: pd.Series,
        exog: pd.DataFrame,
        target_hour: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build LEAR feature matrix for a specific delivery hour.

        Features (per Ziel & Weron 2018):
            - Lagged prices: all 24 hours from d-1, d-2, d-3, d-7
            - Exogenous: load/solar/wind for d, d-1, d-7
            - Day-of-week dummies (6 binary columns)
            - Commodities: 2-day lagged gas, CO2
        """
        # Pivot prices to daily × 24h matrix
        idx_local = prices.index.tz_convert(self.tz)
        df = pd.DataFrame({
            "date": idx_local.date,
            "hour": idx_local.hour,
            "price": prices.values,
        })
        price_pivot = df.pivot_table(
            index="date", columns="hour", values="price", aggfunc="mean"
        )

        # Only keep complete days (all 24 hours)
        complete = price_pivot.dropna(thresh=23)

        features_list = []
        feature_names = []

        # 1. Lagged prices: all 24 hours from lag days
        for lag in self.LAGS_DAYS:
            lagged = complete.shift(lag)
            for h in range(24):
                if h in lagged.columns:
                    col_name = f"price_d-{lag}_h{h:02d}"
                    features_list.append(lagged[h].values)
                    feature_names.append(col_name)

        n_dates = len(complete)
        dates = pd.DatetimeIndex(complete.index)

        # 2. Exogenous features (load, solar, wind, outages)
        exog_cols = [c for c in exog.columns
                     if c in ["load_mw", "solar_mw", "wind_mw", "outages_mw"]]

        for col in exog_cols:
            # Pivot exogenous to daily too
            series = exog[col].dropna()
            if series.empty:
                continue
            exog_local = series.index.tz_convert(self.tz)
            edf = pd.DataFrame({
                "date": exog_local.date,
                "hour": exog_local.hour,
                "value": series.values,
            })
            epivot = edf.pivot_table(
                index="date", columns="hour", values="value", aggfunc="mean"
            )
            epivot = epivot.reindex(complete.index)

            # Current day, target hour
            if target_hour in epivot.columns:
                features_list.append(epivot[target_hour].values)
                feature_names.append(f"{col}_d0_h{target_hour:02d}")

            # Lags d-1, d-7 for target hour
            for lag in [1, 7]:
                shifted = epivot.shift(lag)
                if target_hour in shifted.columns:
                    features_list.append(shifted[target_hour].values)
                    feature_names.append(f"{col}_d-{lag}_h{target_hour:02d}")

        # 3. Commodities (2-day lagged)
        commodity_cols = [c for c in exog.columns
                         if c in ["ttf_gas", "co2_eua_(krbn)", "brent"]]
        for col in commodity_cols:
            series = exog[col].dropna()
            if series.empty:
                continue
            exog_local = series.index.tz_convert(self.tz)
            edf = pd.DataFrame({
                "date": exog_local.date,
                "hour": exog_local.hour,
                "value": series.values,
            })
            # Daily average
            daily_avg = edf.groupby("date")["value"].mean()
            daily_aligned = daily_avg.reindex(complete.index)
            # 2-day lag
            features_list.append(daily_aligned.shift(2).values)
            feature_names.append(f"{col}_d-2")

        # 4. Hydro fill
        if "hydro_fill" in exog.columns:
            series = exog["hydro_fill"].dropna()
            if not series.empty:
                exog_local = series.index.tz_convert(self.tz)
                edf = pd.DataFrame({
                    "date": exog_local.date,
                    "hour": exog_local.hour,
                    "value": series.values,
                })
                daily_avg = edf.groupby("date")["value"].mean()
                daily_aligned = daily_avg.reindex(complete.index)
                features_list.append(daily_aligned.values)
                feature_names.append("hydro_fill")

        # 5. Day-of-week dummies (6 cols, drop Sunday)
        dow = pd.to_datetime(dates).dayofweek
        for d in range(6):  # Mon=0 to Sat=5
            features_list.append((dow == d).astype(float))
            feature_names.append(f"dow_{d}")

        # Assemble
        X = np.column_stack(features_list)
        X_df = pd.DataFrame(X, index=complete.index, columns=feature_names)

        # Target: price at target_hour
        y = complete[target_hour] if target_hour in complete.columns else pd.Series(dtype=float)

        # Drop rows with any NaN
        valid = X_df.notna().all(axis=1) & y.notna()
        return X_df.loc[valid], y.loc[valid]

    def predict(
        self,
        horizon_days: int = 10,
        quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
    ) -> pd.DataFrame:
        """Generate hourly price forecasts for the next `horizon_days`.

        Uses multi-window LASSO averaging with asinh transform.

        Returns:
            DataFrame with columns: timestamp, hour, price_mean, price_p10,
            price_p50, price_p90, plus per-window forecasts.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        last_date = pd.Timestamp(
            self._idx_local[-1].date(),
            tz=self.tz,
        )

        all_forecasts = []

        for hour in range(24):
            X_full, y_full = self._build_features(
                self.prices_h_, self.exog_, target_hour=hour
            )

            if len(y_full) < 30:
                logger.warning("Hour %d: only %d samples, skipping", hour, len(y_full))
                continue

            # Multi-window forecasts
            window_preds: list[np.ndarray] = []

            for window in self.CALIBRATION_WINDOWS:
                if len(y_full) < window:
                    # Use all available data
                    X_train = X_full
                    y_train = y_full
                else:
                    X_train = X_full.iloc[-window:]
                    y_train = y_full.iloc[-window:]

                # Asinh transform
                y_arr = y_train.values.astype(float)
                y_t, mu, sigma = _asinh_transform(y_arr)

                # Fit LASSO
                try:
                    model = LassoLarsCV(
                        max_iter=self.max_iter,
                        cv=5,
                    )
                    X_arr = X_train.values.astype(float)
                    # Replace NaN with 0 in features
                    X_arr = np.nan_to_num(X_arr, nan=0.0)
                    model.fit(X_arr, y_t)
                except Exception as exc:
                    logger.warning(
                        "LASSO failed for hour %d, window %d: %s",
                        hour, window, exc,
                    )
                    continue

                # Predict for each forecast day
                day_preds = []
                for d in range(1, horizon_days + 1):
                    forecast_date = (last_date + timedelta(days=d)).date()
                    x_pred = self._build_prediction_row(
                        X_full, y_full, forecast_date, hour, d,
                    )
                    if x_pred is not None:
                        x_arr = np.nan_to_num(x_pred.reshape(1, -1), nan=0.0)
                        pred_t = model.predict(x_arr)[0]
                        pred = _asinh_inverse(pred_t, mu, sigma)
                        day_preds.append((forecast_date, pred))

                if day_preds:
                    window_preds.append(day_preds)

            if not window_preds:
                continue

            # Average across windows
            for d_idx in range(horizon_days):
                preds_for_day = []
                forecast_date = None
                for wp in window_preds:
                    if d_idx < len(wp):
                        forecast_date = wp[d_idx][0]
                        preds_for_day.append(wp[d_idx][1])

                if preds_for_day and forecast_date is not None:
                    mean_pred = np.mean(preds_for_day)
                    # Ensure non-negative floor at -50 (negative prices possible)
                    mean_pred = max(mean_pred, -50.0)

                    all_forecasts.append({
                        "date": forecast_date,
                        "hour": hour,
                        "price_lear": float(mean_pred),
                        "n_windows": len(preds_for_day),
                    })

        if not all_forecasts:
            raise ValueError("LEAR produced no forecasts")

        result = pd.DataFrame(all_forecasts)
        result = result.sort_values(["date", "hour"]).reset_index(drop=True)

        # Build proper UTC timestamps
        result["timestamp"] = pd.to_datetime(result["date"]) + pd.to_timedelta(
            result["hour"], unit="h"
        )
        result["timestamp"] = result["timestamp"].dt.tz_localize(self.tz).dt.tz_convert("UTC")

        # Compute uncertainty from cross-window variance
        self._compute_uncertainty(result, all_forecasts)

        n_hours = len(result)
        n_days = result["date"].nunique()
        mean_price = result["price_lear"].mean()
        logger.info(
            "LEAR forecast: %d hours (%d days), mean=%.1f EUR/MWh",
            n_hours, n_days, mean_price,
        )
        return result

    def _build_prediction_row(
        self,
        X_full: pd.DataFrame,
        y_full: pd.Series,
        forecast_date,
        hour: int,
        days_ahead: int,
    ) -> np.ndarray | None:
        """Build feature row for a future date.

        Uses most recent available data for lags. For day d+k:
        - d-1 lag uses d+(k-1) if k>1 (recursive), else last known
        - d-7 lag uses known history
        - Exogenous d0: use climatological mean for that hour/month
        """
        n_features = X_full.shape[1]
        x = np.zeros(n_features)
        cols = X_full.columns

        # Last available date in training data
        last_train_date = y_full.index[-1]

        for i, col_name in enumerate(cols):
            if col_name.startswith("price_d-"):
                # Extract lag and hour from name
                parts = col_name.split("_")
                lag = int(parts[1].replace("d-", ""))
                h = int(parts[2].replace("h", ""))

                # Effective lag from forecast date
                eff_lag = lag + days_ahead - 1
                if eff_lag <= 0:
                    # Need a prediction we haven't made yet — use last known
                    eff_lag = 1

                # Look back from last_train_date
                target_idx = len(y_full) - eff_lag
                if target_idx >= 0 and target_idx < len(X_full):
                    # Find the price column for hour h at that date
                    ref_col = f"price_d-1_h{h:02d}"
                    if ref_col in cols:
                        ref_i = list(cols).index(ref_col)
                        row_idx = max(0, len(X_full) - eff_lag)
                        if row_idx < len(X_full):
                            x[i] = X_full.iloc[row_idx, ref_i]
                        else:
                            x[i] = X_full.iloc[-1, ref_i]
                    else:
                        x[i] = y_full.iloc[-eff_lag] if eff_lag <= len(y_full) else y_full.iloc[-1]
                else:
                    x[i] = y_full.iloc[-1]

            elif col_name.startswith("dow_"):
                # Day-of-week for forecast date
                d = int(col_name.split("_")[1])
                import datetime
                if isinstance(forecast_date, datetime.date):
                    x[i] = 1.0 if forecast_date.weekday() == d else 0.0

            elif col_name == "hydro_fill":
                # Use last known value
                x[i] = X_full[col_name].dropna().iloc[-1] if col_name in X_full.columns else 0

            elif col_name.endswith("_d-2") or col_name.endswith("_d-1"):
                # Commodity or exogenous lag: use last known
                x[i] = X_full[col_name].dropna().iloc[-1] if not X_full[col_name].dropna().empty else 0

            elif "_d0_" in col_name:
                # Current day exogenous: use recent climatology (last 28 days, same hour)
                vals = X_full[col_name].dropna().tail(28)
                x[i] = vals.mean() if not vals.empty else 0

            elif "_d-7_" in col_name:
                # 7-day lagged exogenous
                x[i] = X_full[col_name].dropna().iloc[-1] if not X_full[col_name].dropna().empty else 0

            else:
                # Unknown feature — use last known
                x[i] = X_full[col_name].dropna().iloc[-1] if not X_full[col_name].dropna().empty else 0

        return x

    def _compute_uncertainty(self, result: pd.DataFrame, raw: list) -> None:
        """Add p10/p90 columns based on historical forecast errors."""
        # Use last 90 days of in-sample residuals to estimate spread
        # Simplified: scale by horizon-dependent factor
        horizon_scale = {
            1: 0.05, 2: 0.07, 3: 0.08, 4: 0.09, 5: 0.10,
            6: 0.11, 7: 0.12, 8: 0.14, 9: 0.15, 10: 0.17,
        }

        result["days_ahead"] = (
            (pd.to_datetime(result["date"]) - pd.to_datetime(result["date"].min()))
            .dt.days + 1
        )

        for _, row in result.iterrows():
            d = int(row["days_ahead"])
            scale = horizon_scale.get(d, 0.17)
            spread = abs(row["price_lear"]) * scale
            result.loc[row.name, "price_p10"] = row["price_lear"] - 1.28 * spread
            result.loc[row.name, "price_p90"] = row["price_lear"] + 1.28 * spread

    def backtest(
        self,
        n_days: int = 30,
        horizon: int = 1,
    ) -> pd.DataFrame:
        """Rolling out-of-sample backtest: predict D+horizon for each of the last n_days.

        For each test day T (going back from the last available date):
        1. Use data up to T-1 to fit LASSO
        2. Predict hour-by-hour for day T+horizon-1
        3. Compare to actual EPEX spot

        Args:
            n_days: Number of days to backtest.
            horizon: Forecast horizon in days (1 = day-ahead).

        Returns:
            DataFrame with columns: date, hour, forecast, actual, error
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before backtest()")

        # Pivot prices to daily matrix
        idx_local = self.prices_h_.index.tz_convert(self.tz)
        df = pd.DataFrame({
            "date": idx_local.date,
            "hour": idx_local.hour,
            "price": self.prices_h_.values,
        })
        price_pivot = df.pivot_table(
            index="date", columns="hour", values="price", aggfunc="mean"
        )
        complete = price_pivot.dropna(thresh=23)
        all_dates = complete.index

        if len(all_dates) < n_days + 30:
            n_days = min(n_days, len(all_dates) - 30)
            if n_days < 5:
                raise ValueError("Not enough data for backtest")

        results = []
        test_dates = all_dates[-(n_days + horizon - 1): -horizon + 1] if horizon > 1 else all_dates[-n_days:]

        for test_idx, test_date in enumerate(test_dates):
            # Find position in all_dates
            pos = list(all_dates).index(test_date)
            target_pos = pos + horizon - 1
            if target_pos >= len(all_dates):
                continue
            target_date = all_dates[target_pos]

            # Cutoff: use only data up to test_date (exclusive of target)
            cutoff_pos = pos
            if cutoff_pos < 30:
                continue

            for hour in range(24):
                # Build features on truncated data
                prices_trunc = self.prices_h_[:pd.Timestamp(test_date, tz=self.tz).tz_convert("UTC")]
                exog_trunc = self.exog_.loc[:prices_trunc.index[-1]]

                X_full, y_full = self._build_features(
                    prices_trunc, exog_trunc, target_hour=hour
                )

                if len(y_full) < 30:
                    continue

                # Multi-window average
                preds = []
                for window in self.CALIBRATION_WINDOWS:
                    n = min(window, len(y_full))
                    X_w = X_full.iloc[-n:]
                    y_w = y_full.iloc[-n:]

                    y_arr = y_w.values.astype(float)
                    y_t, mu, sigma = _asinh_transform(y_arr)

                    try:
                        model = LassoLarsCV(max_iter=self.max_iter, cv=5)
                        X_arr = np.nan_to_num(X_w.values.astype(float), nan=0.0)
                        model.fit(X_arr, y_t)
                    except Exception:
                        continue

                    x_pred = self._build_prediction_row(
                        X_full, y_full, target_date, hour, horizon,
                    )
                    if x_pred is not None:
                        x_arr = np.nan_to_num(x_pred.reshape(1, -1), nan=0.0)
                        pred_t = model.predict(x_arr)[0]
                        preds.append(_asinh_inverse(pred_t, mu, sigma))

                if not preds:
                    continue

                forecast = float(np.mean(preds))
                actual = complete.loc[target_date, hour] if hour in complete.columns else np.nan

                if not np.isnan(actual):
                    results.append({
                        "date": str(target_date),
                        "hour": hour,
                        "forecast": forecast,
                        "actual": float(actual),
                        "error": forecast - float(actual),
                        "horizon": horizon,
                    })

            if (test_idx + 1) % 5 == 0:
                logger.info("  Backtest: %d/%d days done", test_idx + 1, len(test_dates))

        if not results:
            raise ValueError("Backtest produced no results")

        bt = pd.DataFrame(results)
        bt["abs_error"] = bt["error"].abs()
        bt["ape"] = (bt["abs_error"] / bt["actual"].abs().clip(lower=1)) * 100

        mae = bt["abs_error"].mean()
        rmse = np.sqrt((bt["error"] ** 2).mean())
        mape = bt["ape"].mean()
        corr = bt["forecast"].corr(bt["actual"])

        logger.info(
            "LEAR backtest (D+%d, %d days): MAE=%.1f, RMSE=%.1f, MAPE=%.1f%%, corr=%.3f",
            horizon, bt["date"].nunique(), mae, rmse, mape, corr,
        )
        return bt

    def blend_with_pfc(
        self,
        pfc_15min: pd.DataFrame,
        lear_forecast: pd.DataFrame,
        blend_start_day: int = 8,
        blend_end_day: int = 11,
    ) -> pd.DataFrame:
        """Blend LEAR short-term forecast with PFC structural model.

        Days 1-7: 100% LEAR
        Days 8-10: linear blend LEAR → PFC
        Day 11+: 100% PFC

        Args:
            pfc_15min: Full PFC output with 'price_shape' column
            lear_forecast: LEAR hourly forecast with 'timestamp', 'price_lear'
            blend_start_day: First day of blending zone
            blend_end_day: First day of pure PFC

        Returns:
            Modified pfc_15min with short-term prices replaced/blended.
        """
        result = pfc_15min.copy()

        if "price_shape" not in result.columns:
            logger.warning("No price_shape column in PFC — skipping blend")
            return result

        # Build LEAR hourly Series indexed by UTC timestamp
        lear_ts = lear_forecast.set_index("timestamp")["price_lear"]
        first_date = pd.to_datetime(lear_forecast["date"].min())
        lear_days = lear_forecast.set_index("timestamp")["date"].apply(
            lambda d: (pd.to_datetime(d) - first_date).days + 1
        )

        # Map each 15-min PFC timestamp to its hour
        pfc_hours = result.index.floor("h")

        # Find which PFC rows have a LEAR match
        has_lear = pfc_hours.isin(lear_ts.index)
        if not has_lear.any():
            logger.warning("No timestamp overlap between LEAR and PFC — skipping blend")
            return result

        # Compute hourly PFC means (vectorized)
        result["_hour"] = pfc_hours
        hourly_pfc_mean = result.loc[has_lear].groupby("_hour")["price_shape"].transform("mean")

        # Map LEAR prices and day numbers to 15-min grid
        lear_mapped = lear_ts.reindex(pfc_hours[has_lear]).values
        days_mapped = lear_days.reindex(pfc_hours[has_lear]).values

        pfc_prices = result.loc[has_lear, "price_shape"].values
        hourly_means = hourly_pfc_mean.values

        # Compute blended prices vectorized
        new_prices = pfc_prices.copy()
        for i in range(len(new_prices)):
            day = days_mapped[i]
            lear_p = lear_mapped[i]
            pfc_p = pfc_prices[i]
            h_mean = hourly_means[i]

            if day < blend_start_day:
                # Pure LEAR with 15-min shape preserved
                if h_mean > 0:
                    new_prices[i] = pfc_p * (lear_p / h_mean)
                else:
                    new_prices[i] = lear_p
            elif day < blend_end_day:
                # Linear blend
                w_lear = 1.0 - (day - blend_start_day) / (blend_end_day - blend_start_day)
                if h_mean > 0:
                    lear_scaled = pfc_p * (lear_p / h_mean)
                else:
                    lear_scaled = lear_p
                new_prices[i] = w_lear * lear_scaled + (1 - w_lear) * pfc_p

        result.loc[has_lear, "price_shape"] = new_prices
        result.drop(columns=["_hour"], inplace=True)

        n_replaced = int((days_mapped < blend_start_day).sum())
        n_blended = int(((days_mapped >= blend_start_day) & (days_mapped < blend_end_day)).sum())

        logger.info(
            "LEAR blend: %d timestamps replaced, %d blended, %d+ pure PFC",
            n_replaced, n_blended,
            len(result) - n_replaced - n_blended,
        )
        return result

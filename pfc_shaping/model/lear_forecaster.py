"""
lear_forecaster.py
------------------
Hybrid LEAR+MLP model for short-term electricity price forecasting (D+1 to D+10).

Based on Ziel & Weron (2018), Lago et al. (2021), El Mahtout & Ziel (2026),
and the epftoolbox reference implementation. Adapted for Swiss (CH) and
German (DE) markets with cross-border price features.

Architecture:
    - 24 independent LASSO regressions (one per delivery hour)
    - Asinh variance-stabilizing transformation
    - Multi-window calibration averaging (42, 56, 84, 365 days)
    - Reduced feature set (~40 vs 120+) to prevent overfitting
    - Cross-border DE prices as exogenous features
    - MLP ensemble member for nonlinear interactions
    - AR error correction (autocorrelation of forecast errors)
    - Per-hour variance recalibration
    - Conformal prediction for calibrated forecast intervals

Usage:
    from pfc_shaping.model.lear_forecaster import LEARForecaster
    lear = LEARForecaster()
    lear.fit(epex_ch, entso, outages, commodities, hydro, epex_de_15min=epex_de)
    forecast = lear.predict(horizon_days=10)
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, QuantileRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.outliers_influence import variance_inflation_factor

from pfc_shaping.model.foundation_forecaster import FoundationForecaster

logger = logging.getLogger(__name__)
PRICE_FLOOR_EUR_MWH = -500.0
PRICE_CAP_EUR_MWH = 1000.0

# Holiday sets for Swiss, German and French markets (lazy-initialized)
_HOLIDAY_CACHE: dict[int, tuple[set, set, set]] = {}


def _get_holidays(year: int) -> tuple[set, set, set]:
    """Return (CH holidays, DE holidays, FR holidays) for a given year. Cached."""
    if year not in _HOLIDAY_CACHE:
        try:
            import holidays as hol
            ch = set(hol.Switzerland(years=year, subdiv="VS").keys())
            de = set(hol.Germany(years=year).keys())
            fr = set(hol.France(years=year).keys())
        except ImportError:
            ch, de, fr = set(), set(), set()
        _HOLIDAY_CACHE[year] = (ch, de, fr)
    return _HOLIDAY_CACHE[year]

# Peak hours (CET) for Swiss market
PEAK_HOURS = list(range(7, 20))  # h07-h19
OFFPEAK_HOURS = [h for h in range(24) if h not in PEAK_HOURS]


def _asinh_transform(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Variance-stabilizing area hyperbolic sine transform."""
    mu = np.nanmean(x)
    sigma = np.nanstd(x)
    if sigma < 1e-6:
        sigma = 1.0
    return np.arcsinh((x - mu) / sigma), mu, sigma


def _asinh_domain_bounds(mu: float, sigma: float) -> tuple[float, float]:
    """Return transformed bounds implied by physical Swiss price limits."""
    sigma_eff = max(float(sigma), 1e-6)
    lower_t = float(np.arcsinh((PRICE_FLOOR_EUR_MWH - mu) / sigma_eff))
    upper_t = float(np.arcsinh((PRICE_CAP_EUR_MWH - mu) / sigma_eff))
    return (min(lower_t, upper_t), max(lower_t, upper_t))


def _asinh_inverse(y: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Inverse asinh transform clipped to physical Swiss price bounds."""
    sigma_eff = max(float(sigma), 1e-6)
    lower_t, upper_t = _asinh_domain_bounds(mu, sigma_eff)
    y_arr = np.asarray(y, dtype=float)
    y_safe = np.clip(y_arr, lower_t, upper_t)
    return np.sinh(y_safe) * sigma_eff + mu


def _causal_asinh_transform(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Causal instance normalization + asinh (inspired by Toto).

    Unlike fixed asinh, each observation x[t] is normalized by the
    expanding-window mean and std computed causally from x[0:t+1].
    This adapts to regime shifts in real-time.

    The final mu/sigma (from the full window) are returned for
    inverse transform at prediction time.
    """
    n = len(x)
    result = np.zeros(n)

    # Expanding-window causal normalization
    for t in range(n):
        window = x[:t + 1]
        valid = window[np.isfinite(window)]
        if len(valid) < 2:
            mu_t = valid[0] if len(valid) == 1 else 0.0
            sigma_t = 1.0
        else:
            mu_t = np.mean(valid)
            sigma_t = np.std(valid) + 0.1  # Toto: +0.1 for numerical stability
        result[t] = np.arcsinh((x[t] - mu_t) / sigma_t)

    # Return final mu/sigma for inverse at prediction time
    mu_final = float(np.nanmean(x))
    sigma_final = float(np.nanstd(x))
    if sigma_final < 1e-6:
        sigma_final = 1.0
    return result, mu_final, sigma_final


class LEARForecaster:
    """LASSO-based day-ahead price forecaster with expert-level refinements.

    Key improvements over vanilla LEAR:
    - Reduced feature set (~40 features) prevents overfitting on short windows
    - Per-hour variance recalibration corrects LASSO shrinkage
    - AR error correction exploits error autocorrelation (lag-1 ≈ 0.50)
    - MLP ensemble member captures nonlinear price dynamics
    """

    CALIBRATION_WINDOWS = [42, 56, 84, 180, 365]  # days
    LAGS_DAYS = [1, 2, 3, 7, 14]  # price lag structure
    GOVERNED_FORECAST_COLUMNS = (
        "forecast_load_ch_mw",
        "forecast_load_de_mw",
        "forecast_load_fr_mw",
        "forecast_load_it_mw",
        "forecast_solar_ch_mw",
        "forecast_solar_de_mw",
        "forecast_solar_fr_mw",
        "forecast_solar_it_mw",
        "forecast_wind_de_mw",
        "forecast_wind_fr_mw",
        "forecast_wind_at_mw",
        "forecast_wind_it_mw",
        "forecast_fr_nuclear_available_mw",
        "forecast_fr_nuclear_unavailability_ratio",
    )
    GOVERNED_WEATHER_COLUMNS = (
        "wx_ch_zurich_shortwave_radiation",
        "wx_ch_zurich_cloud_cover",
        "wx_ch_zurich_temperature_2m",
        "wx_de_hamburg_wind_speed_10m",
        "wx_de_munich_shortwave_radiation",
        "wx_fr_lyon_cloud_cover",
        "wx_at_vienna_wind_speed_10m",
        "wx_it_milan_shortwave_radiation",
    )
    GOVERNED_FORECAST_CORE_COLUMNS = tuple(
        col for col in GOVERNED_FORECAST_COLUMNS if not col.startswith("forecast_fr_nuclear_")
    )
    GOVERNED_LONG_HORIZON_REQUIRED_COLUMNS = (
        "forecast_load_ch_mw",
        "forecast_load_de_mw",
        "forecast_load_fr_mw",
        "forecast_load_it_mw",
    )
    GOVERNED_WEATHER_PROXY_MAP = {
        "forecast_solar_ch_mw": ("wx_ch_zurich_shortwave_radiation", "solar"),
        "forecast_solar_de_mw": ("wx_de_munich_shortwave_radiation", "solar"),
        "forecast_solar_fr_mw": ("wx_fr_lyon_shortwave_radiation", "solar"),
        "forecast_solar_it_mw": ("wx_it_milan_shortwave_radiation", "solar"),
        "forecast_solar_at_mw": ("wx_at_vienna_shortwave_radiation", "solar"),
        "forecast_wind_de_mw": ("wx_de_hamburg_wind_speed_10m", "wind"),
        "forecast_wind_fr_mw": ("wx_fr_lyon_wind_speed_10m", "wind"),
        "forecast_wind_at_mw": ("wx_at_vienna_wind_speed_10m", "wind"),
        "forecast_wind_it_mw": ("wx_it_milan_wind_speed_10m", "wind"),
    }

    def __init__(
        self,
        tz: str = "Europe/Zurich",
        max_iter: int = 2500,
        random_state: int = 42,
        use_foundation_model: bool = False,
        foundation_blend_max_horizon_days: int = 1,
        use_extended_physical_ch_features: bool = False,
        use_gbm_blend: bool = True,
        use_mlp_blend: bool = False,
        gbm_blend_max_horizon_days: int = 1,
        use_regime_short_window_weighting: bool = True,
        gbm_weight_cap_peak_block: float = 0.55,
        use_governed_forecast_features: bool = False,
        use_governed_d7_features: bool = False,
    ):
        self.tz = tz
        self.max_iter = max_iter
        self.random_state = random_state
        self.use_foundation_model = use_foundation_model
        self.foundation_blend_max_horizon_days = max(0, int(foundation_blend_max_horizon_days))
        self.use_extended_physical_ch_features = use_extended_physical_ch_features
        self.use_gbm_blend = use_gbm_blend
        self.use_mlp_blend = use_mlp_blend
        self.gbm_blend_max_horizon_days = max(0, int(gbm_blend_max_horizon_days))
        self.use_regime_short_window_weighting = use_regime_short_window_weighting
        self.gbm_weight_cap_peak_block = float(gbm_weight_cap_peak_block)
        self.use_governed_forecast_features = use_governed_forecast_features
        self.use_governed_d7_features = use_governed_d7_features
        self.elasticnet_l1_ratio_base = 0.1
        self.elasticnet_l1_ratio_effective = 0.1
        self._fitted = False
        self._fm = FoundationForecaster() if use_foundation_model else None
        self._lgbm_device_type = "gpu"
        self._governed_feature_health: dict[str, object] = {
            "requested": bool(use_governed_forecast_features),
            "enabled_for_run": bool(use_governed_forecast_features),
            "coverage_ratio": 0.0,
            "min_coverage_ratio": 1.0,
            "max_supported_horizon_days": 0,
            "missing_sources": [],
            "source_coverage": {},
            "vintage_schema_verified": False,
        }
        self._governed_vintage_status: dict[str, dict[str, Any]] = {}
        self._governed_vif_report: dict[str, Any] = {
            "enabled": False,
            "standard_scaler_applied": True,
            "base_l1_ratio": self.elasticnet_l1_ratio_base,
            "effective_l1_ratio": self.elasticnet_l1_ratio_effective,
            "features_over_5": [],
            "feature_max_vif": {},
            "per_hour": {},
        }

    def fit(
        self,
        epex_15min: pd.DataFrame,
        entso_15min: pd.DataFrame | None = None,
        outages_15min: pd.DataFrame | None = None,
        commodities: pd.DataFrame | None = None,
        hydro: pd.DataFrame | None = None,
        epex_de_15min: pd.DataFrame | None = None,
        neighbor_price_15min: dict[str, pd.DataFrame] | None = None,
        de_renewable_forecast: pd.DataFrame | None = None,
        multi_country_forecast: pd.DataFrame | None = None,
        fr_nuclear_forecast: pd.DataFrame | None = None,
        weather_forecast: pd.DataFrame | None = None,
    ) -> "LEARForecaster":
        """Prepare hourly data matrices for LEAR training."""
        # Aggregate EPEX CH to hourly
        self.prices_h_ = (
            epex_15min["price_eur_mwh"]
            .resample("h").mean()
            .dropna()
        )

        # Build hourly exogenous matrix
        exog = pd.DataFrame(index=self.prices_h_.index)
        self._forecast_feature_pivots: dict[str, pd.DataFrame] = {}
        self._all_forecast_feature_pivots: dict[str, pd.DataFrame] = {}
        self._governed_raw_series: dict[str, pd.Series] = {}

        # Cross-border prices
        self._has_de_prices = False
        self._neighbor_price_series_h_: dict[str, pd.Series] = {}
        if epex_de_15min is not None and not epex_de_15min.empty:
            de_col = "price_eur_mwh" if "price_eur_mwh" in epex_de_15min.columns else epex_de_15min.columns[0]
            self.prices_de_h_ = (
                epex_de_15min[de_col]
                .resample("h").mean()
                .dropna()
            )
            exog["de_price"] = self.prices_de_h_
            self._neighbor_price_series_h_["de"] = self.prices_de_h_
            self._has_de_prices = True
            logger.info("  DE cross-border prices loaded: %d hours", len(self.prices_de_h_))
        if neighbor_price_15min:
            for code, frame in neighbor_price_15min.items():
                code = str(code).lower()
                if code == "de" or frame is None or frame.empty:
                    continue
                nb_col = "price_eur_mwh" if "price_eur_mwh" in frame.columns else frame.columns[0]
                nb_series_h = frame[nb_col].resample("h").mean().dropna()
                exog[f"{code}_price"] = nb_series_h
                self._neighbor_price_series_h_[code] = nb_series_h
                logger.info("  %s cross-border prices loaded: %d hours", code.upper(), len(nb_series_h))

        # ENTSO-E
        if entso_15min is not None and not entso_15min.empty:
            entso_cols = [
                "load_mw", "solar_mw", "wind_mw",
                "load_de_mw", "solar_de_mw", "wind_de_mw", "residual_load_de_mw",
            ]
            if self.use_extended_physical_ch_features:
                entso_cols.extend([
                    "nuclear_mw",
                    "hydro_ror_mw",
                    "hydro_reservoir_mw",
                    "hydro_pumped_mw",
                ])
            for col in entso_cols:
                if col in entso_15min.columns:
                    exog[col] = entso_15min[col].resample("h").mean()

        # Outages
        if outages_15min is not None and not outages_15min.empty:
            exog["outages_mw"] = (
                outages_15min["unavailable_mw"].resample("h").mean()
            )

        # Commodities
        if commodities is not None and not commodities.empty:
            for col in commodities.columns:
                daily = commodities[col].dropna()
                if daily.empty:
                    continue
                if daily.index.tz is None:
                    daily.index = daily.index.tz_localize("UTC")
                name = col.split("|")[0].replace(" ", "_").lower()
                exog[name] = daily.resample("h").ffill()

        # Hydro fill
        if hydro is not None and not hydro.empty and "fill_pct" in hydro.columns:
            fill = hydro["fill_pct"].dropna()
            if not fill.empty:
                if fill.index.tz is None:
                    fill.index = fill.index.tz_localize("UTC")
                exog["hydro_fill"] = fill.resample("h").ffill()

        # DE renewable forecasts (wind + solar day-ahead from ENTSO-E)
        if de_renewable_forecast is None:
            # Auto-load from parquet if available
            _de_re_path = Path(__file__).resolve().parent.parent / "data" / "de_renewable_forecast.parquet"
            if _de_re_path.exists():
                de_renewable_forecast = pd.read_parquet(_de_re_path)
                logger.info("DE renewable forecasts auto-loaded: %d rows", len(de_renewable_forecast))

        if de_renewable_forecast is not None and not de_renewable_forecast.empty:
            for col in ["forecast_wind_de_mw", "forecast_solar_de_mw"]:
                if col in de_renewable_forecast.columns:
                    s = de_renewable_forecast[col].dropna()
                    if not s.empty:
                        if s.index.tz is None:
                            s.index = s.index.tz_localize("UTC")
                        exog[col] = s.resample("h").mean()
            # Total DE renewable forecast
            if "forecast_wind_de_mw" in exog.columns and "forecast_solar_de_mw" in exog.columns:
                exog["forecast_renewable_de_mw"] = (
                    exog["forecast_wind_de_mw"].fillna(0) + exog["forecast_solar_de_mw"].fillna(0)
                )
            logger.info("  DE renewable forecasts: wind=%.0f MW avg, solar=%.0f MW avg",
                        exog.get("forecast_wind_de_mw", pd.Series([0])).mean(),
                        exog.get("forecast_solar_de_mw", pd.Series([0])).mean())

        # Governed multi-country load/renewable forecasts
        if self.use_governed_forecast_features and multi_country_forecast is not None and not multi_country_forecast.empty:
            for col in multi_country_forecast.columns:
                s, publication_ts = self._extract_governed_forecast_series(
                    multi_country_forecast,
                    col,
                    dataset_name="multi_country_forecast",
                )
                if s.empty:
                    continue
                self._governed_raw_series[col] = s
                self._forecast_feature_pivots[col] = self._series_to_forecast_pivot(
                    s,
                    publication_ts=publication_ts,
                    source_name=col,
                )
                exog[col] = s.resample("h").mean()

        if self.use_governed_forecast_features and fr_nuclear_forecast is not None and not fr_nuclear_forecast.empty:
            for col in fr_nuclear_forecast.columns:
                s, publication_ts = self._extract_governed_forecast_series(
                    fr_nuclear_forecast,
                    col,
                    dataset_name="fr_nuclear_forecast",
                )
                if s.empty:
                    continue
                self._governed_raw_series[col] = s
                self._forecast_feature_pivots[col] = self._series_to_forecast_pivot(
                    s,
                    publication_ts=publication_ts,
                    source_name=col,
                )
                exog[col] = s.resample("h").mean()

        # Governed weather forecast layer
        if self.use_governed_forecast_features and weather_forecast is not None and not weather_forecast.empty:
            for col in weather_forecast.columns:
                s, publication_ts = self._extract_governed_forecast_series(
                    weather_forecast,
                    col,
                    dataset_name="weather_forecast",
                )
                if s.empty:
                    continue
                self._governed_raw_series[col] = s
                self._forecast_feature_pivots[col] = self._series_to_forecast_pivot(
                    s,
                    publication_ts=publication_ts,
                    source_name=col,
                )
                exog[col] = s.resample("h").mean()

        if self.use_governed_forecast_features:
            self._fill_governed_forecast_gaps_from_weather(exog)

        # Align
        common_idx = self.prices_h_.index
        self.exog_ = exog.reindex(common_idx)
        self._idx_local = self.prices_h_.index.tz_convert(self.tz)

        # Pre-compute daily price pivot (used by _build_features)
        idx_local = self._idx_local
        df = pd.DataFrame({
            "date": idx_local.date,
            "hour": idx_local.hour,
            "price": self.prices_h_.values,
        })
        self._price_pivot = df.pivot_table(
            index="date", columns="hour", values="price", aggfunc="mean"
        )
        self._complete_days = self._price_pivot.dropna(thresh=23)
        if not self._forecast_feature_pivots:
            self._forecast_feature_pivots = self._build_forecast_feature_pivots(exog)
        self._all_forecast_feature_pivots = dict(self._forecast_feature_pivots)
        self._governed_vif_report = self._analyze_governed_feature_vif()
        self.elasticnet_l1_ratio_effective = self._select_effective_l1_ratio(self._governed_vif_report)

        # Pre-compute per-hour variance calibration (last 90 days)
        self._var_calib = self._compute_variance_calibration()

        # Storage
        self._mlp_models: dict = {}
        self._conformal_residuals: dict[int, np.ndarray] = {}

        self._fitted = True
        logger.info(
            "LEAR data prepared: %d hours, %d exog features (DE=%s, neighbors=%s), %s → %s",
            len(self.prices_h_),
            len(self.exog_.columns),
            self._has_de_prices,
            sorted(self._neighbor_price_series_h_.keys()),
            self.prices_h_.index[0].date(),
            self.prices_h_.index[-1].date(),
        )
        return self

    def _build_weather_proxy_series(
        self,
        forecast_series: pd.Series | None,
        weather_series: pd.Series,
        mode: str,
    ) -> pd.Series:
        """Build a conservative weather-derived proxy for missing forecast horizons."""
        weather = pd.to_numeric(weather_series, errors="coerce").dropna().sort_index()
        if weather.empty:
            return pd.Series(dtype=float)

        forecast = (
            pd.to_numeric(forecast_series, errors="coerce").dropna().sort_index()
            if forecast_series is not None else pd.Series(dtype=float)
        )
        overlap = pd.concat(
            [forecast.rename("forecast"), weather.rename("weather")],
            axis=1,
            join="inner",
        ).dropna()
        overlap = overlap.tail(24 * 90)
        if overlap.empty:
            return pd.Series(dtype=float)

        if mode == "solar":
            overlap = overlap[overlap["weather"] > 10.0]
            if overlap.empty:
                return pd.Series(dtype=float)
            scale = float((overlap["forecast"] / overlap["weather"]).replace([np.inf, -np.inf], np.nan).dropna().median())
            if not np.isfinite(scale) or scale <= 0:
                return pd.Series(dtype=float)
            proxy = weather * scale
            proxy = proxy.where(weather > 5.0, 0.0)
        else:
            overlap = overlap[overlap["weather"] > 0.1]
            if overlap.empty:
                return pd.Series(dtype=float)
            scale = float((overlap["forecast"] / overlap["weather"]).replace([np.inf, -np.inf], np.nan).dropna().median())
            if not np.isfinite(scale) or scale <= 0:
                return pd.Series(dtype=float)
            proxy = weather * scale
            proxy = proxy.clip(lower=0.0)

        return proxy.rename(forecast_series.name if forecast_series is not None else weather_series.name)

    def _fill_governed_forecast_gaps_from_weather(self, exog: pd.DataFrame) -> None:
        """Fill missing governed solar/wind forecasts with conservative weather-based proxies."""
        raw_series_map = getattr(self, "_governed_raw_series", {})
        for forecast_col, (weather_col, mode) in self.GOVERNED_WEATHER_PROXY_MAP.items():
            weather_series = raw_series_map.get(weather_col)
            if (weather_series is None or weather_series.empty) and weather_col in exog.columns:
                weather_series = exog[weather_col].dropna()
            if weather_series is None or weather_series.empty:
                continue
            forecast_series = raw_series_map.get(forecast_col)
            if forecast_series is None and forecast_col in exog.columns:
                forecast_series = exog[forecast_col].dropna()
            proxy = self._build_weather_proxy_series(forecast_series, weather_series, mode=mode)
            if proxy.empty:
                continue
            if forecast_series is None:
                filled = proxy
            else:
                filled = forecast_series.combine_first(proxy)
            raw_series_map[forecast_col] = filled
            self._governed_raw_series = raw_series_map
            exog[forecast_col] = filled
            self._forecast_feature_pivots[forecast_col] = self._series_to_forecast_pivot(
                filled.dropna(),
                source_name=f"{forecast_col}_weather_proxy",
            )

    def _build_forecast_feature_pivots(self, exog: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Store hourly pivots for true forecast-time exogenous series."""
        pivots: dict[str, pd.DataFrame] = {}
        for col in exog.columns:
            if not (col.startswith("forecast_") or col.startswith("wx_")):
                continue
            series = exog[col].dropna()
            if series.empty:
                continue
            pivots[col] = self._series_to_forecast_pivot(series)
        return pivots

    def _frame_has_vintage_schema(self, frame: pd.DataFrame) -> bool:
        cols = set(frame.columns)
        return "publication_ts" in cols and "delivery_ts" in cols

    def _record_governed_vintage_status(
        self,
        dataset_name: str,
        *,
        verified: bool,
        invalid_rows_dropped: int = 0,
        total_rows: int = 0,
    ) -> None:
        status = self._governed_vintage_status.setdefault(
            dataset_name,
            {
                "vintage_schema_verified": verified,
                "invalid_rows_dropped": 0,
                "total_rows_seen": 0,
            },
        )
        status["vintage_schema_verified"] = bool(status["vintage_schema_verified"] and verified) if status["total_rows_seen"] else bool(verified)
        status["invalid_rows_dropped"] = int(status["invalid_rows_dropped"]) + int(invalid_rows_dropped)
        status["total_rows_seen"] = int(status["total_rows_seen"]) + int(total_rows)

    def _extract_governed_forecast_series(
        self,
        frame: pd.DataFrame,
        value_col: str,
        dataset_name: str,
    ) -> tuple[pd.Series, pd.Series | None]:
        if value_col not in frame.columns:
            return pd.Series(dtype=float), None

        if self._frame_has_vintage_schema(frame):
            subset = frame[[value_col, "publication_ts", "delivery_ts"]].dropna(subset=[value_col, "publication_ts", "delivery_ts"]).copy()
            if subset.empty:
                self._record_governed_vintage_status(dataset_name, verified=True, total_rows=0)
                return pd.Series(dtype=float), None
            publication_ts = pd.to_datetime(subset["publication_ts"], utc=True, errors="coerce")
            delivery_ts = pd.to_datetime(subset["delivery_ts"], utc=True, errors="coerce")
            subset = subset.assign(publication_ts=publication_ts, delivery_ts=delivery_ts).dropna(subset=["publication_ts", "delivery_ts"])
            values = subset[value_col].astype(float)
            series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(subset["delivery_ts"], tz="UTC"), name=value_col).sort_index()
            publication_series = pd.Series(
                subset["publication_ts"].to_numpy(),
                index=series.index,
                name="publication_ts",
            )
            self._record_governed_vintage_status(dataset_name, verified=True, total_rows=len(subset))
            return series, publication_series

        if dataset_name not in self._governed_vintage_status:
            logger.warning(
                "Governed dataset '%s' has no publication_ts/delivery_ts schema; governed A/B remains an upper-bound until vintage is verified upstream.",
                dataset_name,
            )
        if "forecast_for_ts_utc" in frame.columns:
            delivery_index = pd.to_datetime(frame["forecast_for_ts_utc"], utc=True, errors="coerce")
        else:
            delivery_index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        series = pd.Series(frame[value_col].to_numpy(), index=delivery_index, name=value_col).dropna().sort_index()
        self._record_governed_vintage_status(dataset_name, verified=False, total_rows=len(frame))
        return series, None

    def _series_to_forecast_pivot(
        self,
        series: pd.Series,
        publication_ts: pd.Series | None = None,
        source_name: str | None = None,
    ) -> pd.DataFrame:
        """Build a local-hour forecast pivot from delivery-indexed values.

        Expected governed schema:
        - `series.index`: delivery timestamp in UTC
        - `publication_ts`: timestamp in UTC when the forecast was published

        When `publication_ts` is provided, rows are kept only if:
        `publication_ts <= delivery_ts - 13h`
        which approximates the SDAC day-ahead causality cutoff.
        Violations are dropped and logged as warnings.
        """
        series = series.dropna().sort_index()
        if publication_ts is not None:
            pub = publication_ts.reindex(series.index)
            pub = pd.to_datetime(pub, utc=True, errors="coerce")
            valid = pub <= (series.index - pd.Timedelta(hours=13))
            invalid_rows = int((~valid.fillna(False)).sum())
            if invalid_rows:
                logger.warning(
                    "Dropping %d governed forecast rows failing vintage cutoff%s.",
                    invalid_rows,
                    f" for {source_name}" if source_name else "",
                )
            series = series.loc[valid.fillna(False)]
        if series.empty:
            return pd.DataFrame()
        local_idx = series.index.tz_convert(self.tz)
        df = pd.DataFrame({
            "date": local_idx.date,
            "hour": local_idx.hour,
            "value": series.values,
        })
        return df.pivot_table(index="date", columns="hour", values="value", aggfunc="mean")

    def assess_governed_forecast_feature_health(
        self,
        horizon_days: int,
        min_coverage_ratio: float = 0.98,
    ) -> dict[str, object]:
        """Measure forward coverage of governed forecast/weather pivots.

        This is used to fail loud in production when future-known inputs are
        not actually available for the requested horizon.
        """
        requested = bool(self.use_governed_forecast_features)
        health: dict[str, object] = {
            "requested": requested,
            "enabled_for_run": requested,
            "coverage_ratio": 1.0 if not requested else 0.0,
            "min_coverage_ratio": float(min_coverage_ratio),
            "max_supported_horizon_days": 0 if requested else int(horizon_days),
            "missing_sources": [],
            "source_coverage": {},
            "forecast_dates_local": [],
            "vintage_schema_verified": bool(
                self._governed_vintage_status
                and all(bool(v.get("vintage_schema_verified", False)) for v in self._governed_vintage_status.values())
            ),
        }
        if not requested or not self._fitted:
            self._governed_feature_health = health
            return health

        last_date = pd.Timestamp(self._idx_local[-1].date(), tz=self.tz)
        forecast_dates = [(last_date + timedelta(days=d)).date() for d in range(1, int(horizon_days) + 1)]
        health["forecast_dates_local"] = [str(d) for d in forecast_dates]

        expected_sources = list(self.GOVERNED_FORECAST_COLUMNS) + list(self.GOVERNED_WEATHER_COLUMNS)
        total_expected = len(forecast_dates) * 24 * len(expected_sources)
        total_available = 0
        source_coverage: dict[str, float] = {}
        dropped_sources: list[str] = []
        max_supported_horizon_days = 0
        source_pivots = getattr(self, "_all_forecast_feature_pivots", None) or self._forecast_feature_pivots

        for source_col in expected_sources:
            pivot = source_pivots.get(source_col)
            if pivot is None:
                source_coverage[source_col] = 0.0
                dropped_sources.append(source_col)
                continue
            future = pivot.reindex(index=forecast_dates, columns=range(24))
            available = int(future.notna().sum().sum())
            expected = int(future.shape[0] * future.shape[1])
            coverage = float(available / expected) if expected else 0.0
            source_coverage[source_col] = coverage
            total_available += available
            if coverage < float(min_coverage_ratio):
                dropped_sources.append(source_col)

        retained_sources = [
            source_col
            for source_col in expected_sources
            if source_coverage.get(source_col, 0.0) >= float(min_coverage_ratio)
        ]
        self._forecast_feature_pivots = {
            source_col: source_pivots[source_col]
            for source_col in retained_sources
            if source_col in source_pivots
        }

        if retained_sources:
            for d in range(1, int(horizon_days) + 1):
                candidate_dates = forecast_dates[:d]
                candidate_ok = True
                for source_col in retained_sources:
                    pivot = self._forecast_feature_pivots.get(source_col)
                    if pivot is None:
                        candidate_ok = False
                        break
                    future = pivot.reindex(index=candidate_dates, columns=range(24))
                    expected = int(future.shape[0] * future.shape[1])
                    available = int(future.notna().sum().sum())
                    coverage = float(available / expected) if expected else 0.0
                    if coverage < float(min_coverage_ratio):
                        candidate_ok = False
                        break
                if candidate_ok:
                    max_supported_horizon_days = d

        retained_core_forecast = [
            source_col for source_col in retained_sources
            if source_col in self.GOVERNED_FORECAST_CORE_COLUMNS
        ]
        retained_weather = [
            source_col for source_col in retained_sources
            if source_col in self.GOVERNED_WEATHER_COLUMNS
        ]

        if retained_sources and int(horizon_days) > 5:
            long_horizon_required = [
                source_col
                for source_col in self.GOVERNED_LONG_HORIZON_REQUIRED_COLUMNS
                if source_col in source_pivots
            ]
            if long_horizon_required:
                long_horizon_supported_days = 0
                for d in range(1, int(horizon_days) + 1):
                    candidate_dates = forecast_dates[:d]
                    candidate_ok = True
                    for source_col in long_horizon_required:
                        pivot = source_pivots.get(source_col)
                        if pivot is None:
                            candidate_ok = False
                            break
                        future = pivot.reindex(index=candidate_dates, columns=range(24))
                        expected = int(future.shape[0] * future.shape[1])
                        available = int(future.notna().sum().sum())
                        coverage = float(available / expected) if expected else 0.0
                        if coverage < float(min_coverage_ratio):
                            candidate_ok = False
                            break
                    if candidate_ok:
                        long_horizon_supported_days = d
                max_supported_horizon_days = min(max_supported_horizon_days, long_horizon_supported_days)

        coverage_ratio = float(total_available / total_expected) if total_expected else 1.0
        enabled_for_run = (
            bool(retained_sources)
            and bool(retained_core_forecast)
            and bool(retained_weather)
            and max_supported_horizon_days > 0
            and coverage_ratio >= float(min_coverage_ratio)
        )
        if not enabled_for_run:
            max_supported_horizon_days = 0

        health["coverage_ratio"] = coverage_ratio
        health["max_supported_horizon_days"] = max_supported_horizon_days
        health["missing_sources"] = sorted(set(dropped_sources))
        health["source_coverage"] = source_coverage
        health["enabled_for_run"] = enabled_for_run
        self._governed_feature_health = health
        return health

    def _select_effective_l1_ratio(self, vif_report: dict[str, Any]) -> float:
        features_over_5 = list(vif_report.get("features_over_5", []))
        if len(features_over_5) > 3:
            return 0.5
        return float(self.elasticnet_l1_ratio_base)

    def _analyze_governed_feature_vif(
        self,
        hours: tuple[int, ...] = (7, 12, 18),
        max_rows: int = 180,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "enabled": bool(self.use_governed_forecast_features),
            "standard_scaler_applied": True,
            "base_l1_ratio": float(self.elasticnet_l1_ratio_base),
            "effective_l1_ratio": float(self.elasticnet_l1_ratio_base),
            "threshold": 5.0,
            "hours_evaluated": list(hours),
            "features_over_5": [],
            "feature_max_vif": {},
            "per_hour": {},
        }
        if not self.use_governed_forecast_features:
            return report

        feature_max_vif: dict[str, float] = {}
        features_over_5: set[str] = set()
        for hour in hours:
            X_hour, _ = self._build_features(self.prices_h_, self.exog_, target_hour=hour)
            governed_cols = [
                c for c in X_hour.columns
                if c.startswith("forecast_") or c.startswith("wx_")
            ]
            if len(governed_cols) < 2:
                report["per_hour"][str(hour)] = {"n_features": len(governed_cols), "n_rows": int(len(X_hour))}
                continue
            X_sub = X_hour[governed_cols].tail(max_rows).copy()
            std = X_sub.std(axis=0)
            X_sub = X_sub.loc[:, std.fillna(0.0) > 1e-12]
            if X_sub.shape[1] < 2 or X_sub.shape[0] <= X_sub.shape[1] + 1:
                report["per_hour"][str(hour)] = {
                    "n_features": int(X_sub.shape[1]),
                    "n_rows": int(X_sub.shape[0]),
                    "skipped": True,
                }
                continue
            X_arr = np.nan_to_num(X_sub.to_numpy(dtype=float), nan=0.0)
            vif_rows: list[dict[str, float | str]] = []
            for i, col in enumerate(X_sub.columns):
                try:
                    vif = float(variance_inflation_factor(X_arr, i))
                except Exception:
                    vif = float("inf")
                vif_rows.append({"feature": col, "vif": vif})
                if np.isfinite(vif):
                    feature_max_vif[col] = max(feature_max_vif.get(col, 0.0), vif)
                    if vif > 5.0:
                        features_over_5.add(col)
                else:
                    feature_max_vif[col] = float("inf")
                    features_over_5.add(col)
            vif_rows.sort(key=lambda row: float(row["vif"]) if np.isfinite(float(row["vif"])) else 1e12, reverse=True)
            report["per_hour"][str(hour)] = {
                "n_features": int(X_sub.shape[1]),
                "n_rows": int(X_sub.shape[0]),
                "top_vif": vif_rows[:15],
            }

        report["feature_max_vif"] = {
            feature: (value if np.isfinite(value) else "inf")
            for feature, value in sorted(feature_max_vif.items(), key=lambda item: (-(item[1] if np.isfinite(item[1]) else 1e12), item[0]))
        }
        report["features_over_5"] = sorted(features_over_5)
        report["effective_l1_ratio"] = float(self._select_effective_l1_ratio(report))
        return report

    def get_governed_vif_report(self) -> dict[str, Any]:
        return dict(self._governed_vif_report)

    def _compute_variance_calibration(self) -> dict[int, tuple[float, float]]:
        """Compute per-hour recent mean and std for variance recalibration.

        Returns dict[hour] = (recent_mean, recent_std) from last 90 days.
        """
        calib = {}
        complete = self._complete_days
        n_recent = min(90, len(complete) - 10)
        if n_recent < 20:
            return calib

        recent = complete.iloc[-n_recent:]
        for h in range(24):
            if h in recent.columns:
                vals = recent[h].dropna()
                if len(vals) > 10:
                    calib[h] = (float(vals.mean()), float(vals.std()))
        return calib

    def _build_features(
        self,
        prices: pd.Series,
        exog: pd.DataFrame,
        target_hour: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build REDUCED feature matrix for a specific delivery hour.

        Expert feature reduction: ~40 features instead of 120+.
        Only use: target hour, peak mean, off-peak mean, daily mean
        per lag day, instead of all 24 hours × 4 lags = 96 price features.
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
        complete = price_pivot.dropna(thresh=23)

        features_list = []
        feature_names = []

        # ── 1. CH price aggregates per lag (REDUCED: 5 features × 4 lags = 20) ──
        peak_cols = [h for h in PEAK_HOURS if h in complete.columns]
        offpeak_cols = [h for h in OFFPEAK_HOURS if h in complete.columns]

        for lag in self.LAGS_DAYS:
            lagged = complete.shift(lag)

            # Target hour price (most important)
            if target_hour in lagged.columns:
                features_list.append(lagged[target_hour].values)
                feature_names.append(f"price_d-{lag}_h{target_hour:02d}")

            # Daily mean
            features_list.append(lagged.mean(axis=1).values)
            feature_names.append(f"price_mean_d-{lag}")

            # Peak mean
            if peak_cols:
                features_list.append(lagged[peak_cols].mean(axis=1).values)
                feature_names.append(f"price_peak_d-{lag}")

            # Off-peak mean
            if offpeak_cols:
                features_list.append(lagged[offpeak_cols].mean(axis=1).values)
                feature_names.append(f"price_offpeak_d-{lag}")

            # Min and max of the day (captures volatility/spikes)
            features_list.append(lagged.max(axis=1).values)
            feature_names.append(f"price_max_d-{lag}")

        # Same-day-of-week memory: stabilize long horizons on recurring weekday/weekend regimes.
        weekly_target_frames = []
        weekly_mean_frames = []
        for lag in [7, 14, 21, 28]:
            lagged = complete.shift(lag)
            if target_hour in lagged.columns:
                weekly_target_frames.append(lagged[target_hour])
            weekly_mean_frames.append(lagged.mean(axis=1))

        if weekly_target_frames:
            same_dow_target = pd.concat(weekly_target_frames, axis=1).mean(axis=1)
            features_list.append(same_dow_target.values)
            feature_names.append(f"price_same_dow_mean_h{target_hour:02d}")

        if weekly_mean_frames:
            same_dow_daily_mean = pd.concat(weekly_mean_frames, axis=1).mean(axis=1)
            features_list.append(same_dow_daily_mean.values)
            feature_names.append("price_same_dow_daily_mean")

        # Price momentum: d-1 vs d-2 change (trend signal)
        if target_hour in complete.columns:
            d1 = complete.shift(1)[target_hour]
            d2 = complete.shift(2)[target_hour]
            features_list.append((d1 - d2).values)
            feature_names.append(f"price_momentum_h{target_hour:02d}")

        # Recent volatility (std of last 7 days, target hour)
        if target_hour in complete.columns:
            rolling_std = complete[target_hour].rolling(7, min_periods=3).std().shift(1)
            features_list.append(rolling_std.values)
            feature_names.append(f"price_vol7d_h{target_hour:02d}")

        n_dates = len(complete)
        dates = pd.DatetimeIndex(complete.index)

        # ── 2. Cross-border price features ──
        neighbor_price_cols = [c for c in exog.columns if c.endswith("_price")]
        for price_col in neighbor_price_cols:
            neighbor_code = price_col.replace("_price", "")
            nb_series = exog[price_col].dropna()
            if nb_series.empty:
                continue
            nb_local = nb_series.index.tz_convert(self.tz)
            nb_df = pd.DataFrame({
                "date": nb_local.date,
                "hour": nb_local.hour,
                "price": nb_series.values,
            })
            nb_pivot = nb_df.pivot_table(
                index="date", columns="hour", values="price", aggfunc="mean"
            )
            nb_pivot = nb_pivot.reindex(complete.index)

            for lag in [1, 2, 7]:
                nb_lagged = nb_pivot.shift(lag)
                if target_hour in nb_lagged.columns:
                    features_list.append(nb_lagged[target_hour].values)
                    feature_names.append(f"{neighbor_code}_price_d-{lag}_h{target_hour:02d}")
                features_list.append(nb_lagged.mean(axis=1).values)
                feature_names.append(f"{neighbor_code}_price_mean_d-{lag}")

            if neighbor_code == "de" and target_hour in nb_pivot.columns and target_hour in complete.columns:
                spread = complete.shift(1)[target_hour] - nb_pivot.shift(1)[target_hour]
                features_list.append(spread.values)
                feature_names.append(f"ch_de_spread_d-1_h{target_hour:02d}")

                congestion = (spread.abs() > 2.0).astype(float)
                features_list.append(congestion.values)
                feature_names.append("ch_de_congestion_d-1")

        # ── 3. DE renewable forecasts (day-ahead wind + solar) ──
        for re_col in ["forecast_wind_de_mw", "forecast_solar_de_mw", "forecast_renewable_de_mw"]:
            if re_col in exog.columns:
                re_series = exog[re_col].dropna()
                if not re_series.empty:
                    re_local = re_series.index.tz_convert(self.tz)
                    re_df = pd.DataFrame({
                        "date": re_local.date, "hour": re_local.hour,
                        "value": re_series.values,
                    })
                    re_pivot = re_df.pivot_table(
                        index="date", columns="hour", values="value", aggfunc="mean"
                    )
                    re_pivot = re_pivot.reindex(complete.index)

                    # D+0 forecast (available at auction time)
                    if target_hour in re_pivot.columns:
                        features_list.append(re_pivot[target_hour].values)
                        feature_names.append(f"{re_col}_d0_h{target_hour:02d}")

                    # D-1 forecast (yesterday's same hour)
                    if target_hour in re_pivot.columns:
                        features_list.append(re_pivot.shift(1)[target_hour].values)
                        feature_names.append(f"{re_col}_d-1_h{target_hour:02d}")

        if self.use_governed_forecast_features:
            # ── 3a. Compact multi-country forecast block ──
            compact_forecast_cols = [
                "forecast_load_ch_mw",
                "forecast_load_de_mw",
                "forecast_load_fr_mw",
                "forecast_load_it_mw",
                "forecast_solar_ch_mw",
                "forecast_solar_de_mw",
                "forecast_solar_fr_mw",
                "forecast_solar_it_mw",
                "forecast_wind_de_mw",
                "forecast_wind_fr_mw",
                "forecast_wind_at_mw",
                "forecast_wind_it_mw",
                "forecast_fr_nuclear_available_mw",
                "forecast_fr_nuclear_unavailability_ratio",
            ]
            for fc_col in compact_forecast_cols:
                if fc_col not in exog.columns:
                    continue
                fc_series = exog[fc_col].dropna()
                if fc_series.empty:
                    continue
                fc_local = fc_series.index.tz_convert(self.tz)
                fc_df = pd.DataFrame({
                    "date": fc_local.date,
                    "hour": fc_local.hour,
                    "value": fc_series.values,
                })
                fc_pivot = fc_df.pivot_table(index="date", columns="hour", values="value", aggfunc="mean")
                fc_pivot = fc_pivot.reindex(complete.index)
                if target_hour in fc_pivot.columns:
                    self._append_feature_if_sufficient_history(
                        features_list, feature_names, fc_pivot[target_hour],
                        f"{fc_col}_d0_h{target_hour:02d}",
                        min_non_null=60,
                        reference_len=len(complete),
                        min_coverage_ratio=0.15,
                        max_required_non_null=180,
                    )
                    self._append_feature_if_sufficient_history(
                        features_list, feature_names, fc_pivot.shift(1)[target_hour],
                        f"{fc_col}_d-1_h{target_hour:02d}",
                        min_non_null=60,
                        reference_len=len(complete),
                        min_coverage_ratio=0.15,
                        max_required_non_null=180,
                    )
                    if self.use_governed_d7_features:
                        self._append_feature_if_sufficient_history(
                            features_list, feature_names, fc_pivot.shift(7)[target_hour],
                            f"{fc_col}_d-7_h{target_hour:02d}",
                            min_non_null=60,
                            reference_len=len(complete),
                            min_coverage_ratio=0.15,
                            max_required_non_null=180,
                        )
                midday_cols = [h for h in range(10, 16) if h in fc_pivot.columns]
                if midday_cols and ("solar" in fc_col or "load" in fc_col or "nuclear" in fc_col):
                    midday_mean = fc_pivot[midday_cols].mean(axis=1)
                    self._append_feature_if_sufficient_history(
                        features_list, feature_names, midday_mean,
                        f"{fc_col}_d0_midday_mean",
                        min_non_null=60,
                        reference_len=len(complete),
                        min_coverage_ratio=0.15,
                        max_required_non_null=180,
                    )
                    self._append_feature_if_sufficient_history(
                        features_list, feature_names, midday_mean.shift(1),
                        f"{fc_col}_d-1_midday_mean",
                        min_non_null=60,
                        reference_len=len(complete),
                        min_coverage_ratio=0.15,
                        max_required_non_null=180,
                    )
                    if self.use_governed_d7_features:
                        self._append_feature_if_sufficient_history(
                            features_list, feature_names, midday_mean.shift(7),
                            f"{fc_col}_d-7_midday_mean",
                            min_non_null=60,
                            reference_len=len(complete),
                            min_coverage_ratio=0.15,
                            max_required_non_null=180,
                        )

            # ── 3b. Compact weather forecast block ──
            compact_weather_cols = [
                "wx_ch_zurich_shortwave_radiation",
                "wx_ch_zurich_cloud_cover",
                "wx_ch_zurich_temperature_2m",
                "wx_de_hamburg_wind_speed_10m",
                "wx_de_munich_shortwave_radiation",
                "wx_fr_lyon_cloud_cover",
                "wx_at_vienna_wind_speed_10m",
                "wx_it_milan_shortwave_radiation",
            ]
            for wx_col in compact_weather_cols:
                if wx_col not in exog.columns:
                    continue
                wx_series = exog[wx_col].dropna()
                if wx_series.empty:
                    continue
                wx_local = wx_series.index.tz_convert(self.tz)
                wx_df = pd.DataFrame({
                    "date": wx_local.date,
                    "hour": wx_local.hour,
                    "value": wx_series.values,
                })
                wx_pivot = wx_df.pivot_table(index="date", columns="hour", values="value", aggfunc="mean")
                wx_pivot = wx_pivot.reindex(complete.index)
                if target_hour in wx_pivot.columns:
                    self._append_feature_if_sufficient_history(
                        features_list, feature_names, wx_pivot[target_hour],
                        f"{wx_col}_d0_h{target_hour:02d}",
                        min_non_null=60,
                        reference_len=len(complete),
                        min_coverage_ratio=0.15,
                        max_required_non_null=180,
                    )
                    self._append_feature_if_sufficient_history(
                        features_list, feature_names, wx_pivot.shift(1)[target_hour],
                        f"{wx_col}_d-1_h{target_hour:02d}",
                        min_non_null=60,
                        reference_len=len(complete),
                        min_coverage_ratio=0.15,
                        max_required_non_null=180,
                    )
                    if self.use_governed_d7_features:
                        self._append_feature_if_sufficient_history(
                            features_list, feature_names, wx_pivot.shift(7)[target_hour],
                            f"{wx_col}_d-7_h{target_hour:02d}",
                            min_non_null=60,
                            reference_len=len(complete),
                            min_coverage_ratio=0.15,
                            max_required_non_null=180,
                        )
                midday_cols = [h for h in range(10, 16) if h in wx_pivot.columns]
                if midday_cols and ("shortwave_radiation" in wx_col or "cloud_cover" in wx_col):
                    midday_mean = wx_pivot[midday_cols].mean(axis=1)
                    self._append_feature_if_sufficient_history(
                        features_list, feature_names, midday_mean,
                        f"{wx_col}_d0_midday_mean",
                        min_non_null=60,
                        reference_len=len(complete),
                        min_coverage_ratio=0.15,
                        max_required_non_null=180,
                    )
                    self._append_feature_if_sufficient_history(
                        features_list, feature_names, midday_mean.shift(1),
                        f"{wx_col}_d-1_midday_mean",
                        min_non_null=60,
                        reference_len=len(complete),
                        min_coverage_ratio=0.15,
                        max_required_non_null=180,
                    )
                    if self.use_governed_d7_features:
                        self._append_feature_if_sufficient_history(
                            features_list, feature_names, midday_mean.shift(7),
                            f"{wx_col}_d-7_midday_mean",
                            min_non_null=60,
                            reference_len=len(complete),
                            min_coverage_ratio=0.15,
                            max_required_non_null=180,
                        )

        # ── 4. Exogenous features (CH) ──
        exog_cols = [c for c in exog.columns
                     if c in [
                         "load_mw", "solar_mw", "wind_mw", "outages_mw",
                         "nuclear_mw", "hydro_ror_mw", "hydro_reservoir_mw", "hydro_pumped_mw",
                     ]]

        for col in exog_cols:
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

            # d-1 and d-7 only (d+0 removed: train/inference mismatch
            # — realized values in training but 28d avg in production)
            for lag in [1, 7]:
                shifted = epivot.shift(lag)
                if target_hour in shifted.columns:
                    features_list.append(shifted[target_hour].values)
                    feature_names.append(f"{col}_d-{lag}_h{target_hour:02d}")

        # ── 3b. Enriched d-1 aggregate features (compensate d+0 removal) ──
        ch_exog_cols = [c for c in exog.columns
                        if c in [
                            "load_mw", "solar_mw", "wind_mw",
                            "nuclear_mw", "hydro_ror_mw", "hydro_reservoir_mw", "hydro_pumped_mw",
                        ]]
        for col in ch_exog_cols:
            series = exog[col].dropna()
            if series.empty:
                continue
            exog_local = series.index.tz_convert(self.tz)
            edf_agg = pd.DataFrame({
                "date": exog_local.date,
                "value": series.values,
            })
            daily_agg = edf_agg.groupby("date")["value"].agg(["mean", "max", "min"])
            daily_agg.index = pd.to_datetime(daily_agg.index)
            daily_agg = daily_agg.reindex(complete.index)
            # d-1 daily mean, max, range
            for lag in [1]:
                shifted_agg = daily_agg.shift(lag)
                features_list.append(shifted_agg["mean"].values)
                feature_names.append(f"{col}_daily_mean_d-{lag}")
                features_list.append(shifted_agg["max"].values)
                feature_names.append(f"{col}_daily_max_d-{lag}")
                features_list.append((shifted_agg["max"] - shifted_agg["min"]).values)
                feature_names.append(f"{col}_daily_range_d-{lag}")

        # ── 3c. Compact DE regime signals ──
        if "solar_de_mw" in exog.columns:
            solar_de = exog["solar_de_mw"].dropna()
            if not solar_de.empty:
                solar_local = solar_de.index.tz_convert(self.tz)
                solar_df = pd.DataFrame({
                    "date": solar_local.date,
                    "hour": solar_local.hour,
                    "value": solar_de.values,
                })
                solar_pivot = solar_df.pivot_table(
                    index="date", columns="hour", values="value", aggfunc="mean"
                ).reindex(complete.index)
                midday_cols = [h for h in range(10, 16) if h in solar_pivot.columns]
                if midday_cols:
                    solar_midday = solar_pivot[midday_cols].mean(axis=1)
                    solar_midday_d1 = solar_midday.shift(1)
                    solar_midday_roll = solar_midday.shift(1).rolling(28, min_periods=7)
                    solar_midday_mean = solar_midday_roll.mean()
                    solar_midday_std = solar_midday_roll.std().replace(0, np.nan)
                    solar_midday_z = ((solar_midday_d1 - solar_midday_mean) / solar_midday_std).fillna(0.0)
                    features_list.append(solar_midday_z.values)
                    feature_names.append("solar_de_midday_z_d-1")
                    features_list.append((solar_midday_z > 0.75).astype(float).values)
                    feature_names.append("solar_de_midday_high_d-1")

        if "residual_load_de_mw" in exog.columns:
            residual_de = exog["residual_load_de_mw"].dropna()
            if not residual_de.empty:
                residual_local = residual_de.index.tz_convert(self.tz)
                residual_df = pd.DataFrame({
                    "date": residual_local.date,
                    "hour": residual_local.hour,
                    "value": residual_de.values,
                })
                residual_pivot = residual_df.pivot_table(
                    index="date", columns="hour", values="value", aggfunc="mean"
                ).reindex(complete.index)
                if target_hour in residual_pivot.columns:
                    residual_hour = residual_pivot[target_hour]
                    residual_hour_d1 = residual_hour.shift(1)
                    residual_roll = residual_hour.shift(1).rolling(28, min_periods=7)
                    residual_mean = residual_roll.mean()
                    residual_std = residual_roll.std().replace(0, np.nan)
                    residual_z = ((residual_hour_d1 - residual_mean) / residual_std).fillna(0.0)
                    features_list.append(residual_z.values)
                    feature_names.append(f"residual_load_de_z_d-1_h{target_hour:02d}")
                    features_list.append((residual_z < -0.75).astype(float).values)
                    feature_names.append(f"residual_load_de_low_d-1_h{target_hour:02d}")

                residual_daily = residual_df.groupby("date")["value"].mean().reindex(complete.index)
                residual_daily_d1 = residual_daily.shift(1)
                residual_daily_roll = residual_daily.shift(1).rolling(28, min_periods=7)
                residual_daily_mean = residual_daily_roll.mean()
                residual_daily_std = residual_daily_roll.std().replace(0, np.nan)
                residual_daily_z = ((residual_daily_d1 - residual_daily_mean) / residual_daily_std).fillna(0.0)
                features_list.append(residual_daily_z.values)
                feature_names.append("residual_load_de_daily_z_d-1")

        # ── 4. Commodities ──
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
            daily_avg = edf.groupby("date")["value"].mean()
            daily_aligned = daily_avg.reindex(complete.index)
            self._append_feature_if_sufficient_history(
                features_list,
                feature_names,
                daily_aligned.shift(2),
                f"{col}_d-2",
                min_non_null=60,
                reference_len=len(complete),
                min_coverage_ratio=0.15,
                max_required_non_null=180,
            )

        # ── 5. Hydro fill ──
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
                hydro_fill_available = daily_aligned.shift(2)
                self._append_feature_if_sufficient_history(
                    features_list,
                    feature_names,
                    hydro_fill_available,
                    "hydro_fill",
                    min_non_null=60,
                    reference_len=len(complete),
                    min_coverage_ratio=0.15,
                    max_required_non_null=180,
                )

                # Swiss Market Intelligence: hydro fill delta (rate of change)
                # Captures filling (snowmelt) vs emptying (winter drawdown)
                hydro_delta_7d = hydro_fill_available - hydro_fill_available.shift(7)
                self._append_feature_if_sufficient_history(
                    features_list,
                    feature_names,
                    hydro_delta_7d,
                    "hydro_fill_delta_7d",
                    min_non_null=60,
                    reference_len=len(complete),
                    min_coverage_ratio=0.15,
                    max_required_non_null=180,
                )

        # CH physical stack compact ratios: make hydro / nuclear informative without
        # requiring the model to infer relative scale from raw MW levels alone.
        if self.use_extended_physical_ch_features and "load_mw" in exog.columns:
            load_h = exog["load_mw"].replace(0, np.nan)
            physical_ratio_cols = [
                ("nuclear_mw", "nuclear_share_ch"),
                ("hydro_ror_mw", "hydro_ror_share_ch"),
                ("hydro_reservoir_mw", "hydro_reservoir_share_ch"),
                ("hydro_pumped_mw", "hydro_pumped_share_ch"),
            ]
            for src_col, feat_name in physical_ratio_cols:
                if src_col not in exog.columns:
                    continue
                ratio = (exog[src_col] / load_h).replace([np.inf, -np.inf], np.nan)
                ratio = ratio.dropna()
                if ratio.empty:
                    continue
                ratio_local = ratio.index.tz_convert(self.tz)
                ratio_df = pd.DataFrame({
                    "date": ratio_local.date,
                    "hour": ratio_local.hour,
                    "value": ratio.values,
                })
                ratio_pivot = ratio_df.pivot_table(
                    index="date", columns="hour", values="value", aggfunc="mean"
                ).reindex(complete.index)
                if target_hour in ratio_pivot.columns:
                    features_list.append(ratio_pivot.shift(1)[target_hour].values)
                    feature_names.append(f"{feat_name}_d-1_h{target_hour:02d}")

        # ── 5b. Swiss-specific features (ALPINE v2 quick wins) ──
        # --- Wallis-specific reservoir fill (55% of CH storage, FMV's region) ---
        if "wallis_gwh" in exog.columns:
            ws = exog["wallis_gwh"].dropna()
            if not ws.empty:
                ws_local = ws.index.tz_convert(self.tz)
                ws_df = pd.DataFrame({"date": ws_local.date, "value": ws.values})
                ws_daily = ws_df.groupby("date")["value"].mean()
                ws_aligned = ws_daily.reindex(complete.index).ffill()
                features_list.append(ws_aligned.shift(2).values)
                feature_names.append("wallis_fill_gwh")

        # --- Hydro sub-types from ENTSO-E (pumped, RoR, reservoir generation) ---
        for hydro_col in ["hydro_pumped_mw", "hydro_ror_mw"]:
            if hydro_col in exog.columns:
                hs = exog[hydro_col].dropna()
                if not hs.empty:
                    hs_local = hs.index.tz_convert(self.tz)
                    hs_df = pd.DataFrame({
                        "date": hs_local.date, "hour": hs_local.hour,
                        "value": hs.values,
                    })
                    hs_pivot = hs_df.pivot_table(
                        index="date", columns="hour", values="value", aggfunc="mean"
                    ).reindex(complete.index)
                    if target_hour in hs_pivot.columns:
                        features_list.append(hs_pivot.shift(1)[target_hour].values)
                        feature_names.append(f"{hydro_col}_direct_d-1_h{target_hour:02d}")

        # --- Nuclear available capacity (2960 MW total - unavailable) ---
        if "unavailable_nuclear" in exog.columns:
            nuc = exog["unavailable_nuclear"].dropna()
            if not nuc.empty:
                nuc_local = nuc.index.tz_convert(self.tz)
                nuc_df = pd.DataFrame({"date": nuc_local.date, "value": nuc.values})
                nuc_daily = nuc_df.groupby("date")["value"].mean()
                nuc_aligned = nuc_daily.reindex(complete.index)
                nuclear_avail = 2960.0 - nuc_aligned.shift(1).fillna(0)
                features_list.append(nuclear_avail.values)
                feature_names.append("nuclear_available_ch_mw")

        # --- Nant de Drance structural break (900MW pump-storage, July 2022) ---
        dt_idx = pd.to_datetime(complete.index)
        ndd_online = (dt_idx >= pd.Timestamp("2022-07-01")).astype(float)
        features_list.append(ndd_online)
        feature_names.append("nant_de_drance_online")

        # --- Winterreserve active (~500 GWh withheld from market, Oct 15 - Apr 15) ---
        wr_month = dt_idx.month
        wr_day = dt_idx.day
        winterreserve = (
            ((wr_month >= 11) | (wr_month <= 3)) |
            ((wr_month == 10) & (wr_day >= 15)) |
            ((wr_month == 4) & (wr_day <= 15))
        ).astype(float)
        # Only active since winter 2022/23
        winterreserve = winterreserve * (dt_idx >= pd.Timestamp("2022-10-15")).astype(float)
        features_list.append(winterreserve)
        feature_names.append("winterreserve_active")

        # --- Directional congestion (import vs export, replaces simple binary) ---
        if "ch_de_spread_d-1_h{:02d}".format(target_hour) in feature_names:
            spread_idx = feature_names.index(
                "ch_de_spread_d-1_h{:02d}".format(target_hour)
            )
            spread_vals = features_list[spread_idx]
            # CH importing (CH price > DE): spread > 2
            features_list.append((spread_vals > 2.0).astype(float))
            feature_names.append("ch_importing_congested")
            # CH exporting (CH price < DE): spread < -2
            features_list.append((spread_vals < -2.0).astype(float))
            feature_names.append("ch_exporting_congested")

        # ── 5c. Price dynamics features (from fmv_epex analysis) ──
        # --- Rolling quantile distances (spike/regime detection) ---
        if target_hour in complete.columns:
            spot_h = complete[target_hour]
            q90_168 = spot_h.shift(1).rolling(7, min_periods=4).quantile(0.90)
            q10_168 = spot_h.shift(1).rolling(7, min_periods=4).quantile(0.10)
            features_list.append((spot_h.shift(1) - q90_168).values)
            feature_names.append("spot_dist_to_q90_7d")
            features_list.append((spot_h.shift(1) - q10_168).values)
            feature_names.append("spot_dist_to_q10_7d")

            # --- Price ramps (first difference and abs rolling mean) ---
            spot_d1 = spot_h.diff()
            features_list.append(spot_d1.shift(1).values)
            feature_names.append("spot_ramp_d-1")
            ramp_abs_7d = spot_d1.abs().rolling(7, min_periods=3).mean()
            features_list.append(ramp_abs_7d.shift(1).values)
            feature_names.append("spot_ramp_abs_7d_mean")

            # --- Rolling std 7d and 14d (volatility regimes) ---
            vol_7d = spot_h.rolling(7, min_periods=4).std()
            features_list.append(vol_7d.shift(1).values)
            feature_names.append("spot_vol_7d")

        # ── 6. Calendar features ──
        dow = pd.to_datetime(dates).dayofweek
        for d in range(6):  # Mon=0 to Sat=5
            features_list.append((dow == d).astype(float))
            feature_names.append(f"dow_{d}")

        # Weekend flag (stronger signal than individual dummies)
        is_weekend = (dow >= 5).astype(float)
        features_list.append(is_weekend)
        feature_names.append("is_weekend")

        # Hour-specific regime dummies for the main long-horizon failure modes.
        if target_hour in {10, 11, 12, 13, 14}:
            features_list.append(is_weekend)
            feature_names.append("weekend_midday_regime")
        if target_hour in {7, 18, 19}:
            features_list.append((dow < 5).astype(float))
            feature_names.append("weekday_peak_regime")

        # Month sin/cos (captures seasonality)
        month = pd.to_datetime(dates).month
        features_list.append(np.sin(2 * np.pi * month / 12))
        feature_names.append("month_sin")
        features_list.append(np.cos(2 * np.pi * month / 12))
        feature_names.append("month_cos")

        # ── 7. Holiday features (CH + DE) ──
        dt_dates = pd.to_datetime(dates)
        is_holiday_ch = np.zeros(n_dates)
        is_holiday_de = np.zeros(n_dates)
        is_holiday_fr = np.zeros(n_dates)
        for i, d in enumerate(dt_dates):
            ch_hols, de_hols, fr_hols = _get_holidays(d.year)
            if d.date() in ch_hols:
                is_holiday_ch[i] = 1.0
            if d.date() in de_hols:
                is_holiday_de[i] = 1.0
            if d.date() in fr_hols:
                is_holiday_fr[i] = 1.0
        features_list.append(is_holiday_ch)
        feature_names.append("is_holiday_ch")
        features_list.append(is_holiday_de)
        feature_names.append("is_holiday_de")
        features_list.append(is_holiday_fr)
        feature_names.append("is_holiday_fr")
        neighbor_holiday_mismatch = (((is_holiday_de + is_holiday_fr) > 0.0) & (is_holiday_ch < 0.5)).astype(float)
        features_list.append(neighbor_holiday_mismatch)
        feature_names.append("neighbor_holiday_mismatch")
        if 8 <= int(target_hour) <= 17:
            features_list.append(neighbor_holiday_mismatch)
            feature_names.append("neighbor_holiday_mismatch_daytime")

        # ── 8. Fuel stack proxy (marginal cost of gas plants) ──
        # gas_price * heat_rate + co2_price * emission_factor
        gas_col = next((c for c in exog.columns if "ttf" in c.lower() or "gas" in c.lower()), None)
        co2_col = next((c for c in exog.columns if "co2" in c.lower() or "eua" in c.lower()), None)
        if gas_col and co2_col:
            gas_daily = exog[gas_col].dropna()
            co2_daily = exog[co2_col].dropna()
            if not gas_daily.empty and not co2_daily.empty:
                gas_local = gas_daily.index.tz_convert(self.tz)
                gas_df = pd.DataFrame({"date": gas_local.date, "value": gas_daily.values})
                gas_by_date = gas_df.groupby("date")["value"].mean()
                co2_local = co2_daily.index.tz_convert(self.tz)
                co2_df = pd.DataFrame({"date": co2_local.date, "value": co2_daily.values})
                co2_by_date = co2_df.groupby("date")["value"].mean()
                fuel_proxy = (gas_by_date * 2.0 + co2_by_date * 0.37).reindex(complete.index)
                self._append_feature_if_sufficient_history(
                    features_list,
                    feature_names,
                    fuel_proxy.shift(1),
                    "fuel_stack_proxy_d-1",
                    min_non_null=60,
                    reference_len=len(complete),
                    min_coverage_ratio=0.15,
                    max_required_non_null=180,
                )

        # Assemble
        X = np.column_stack(features_list)
        X_df = pd.DataFrame(X, index=complete.index, columns=feature_names)

        # Target
        y = complete[target_hour] if target_hour in complete.columns else pd.Series(dtype=float)

        # Drop rows with any NaN
        valid = X_df.notna().all(axis=1) & y.notna()
        return X_df.loc[valid], y.loc[valid]

    @staticmethod
    def _time_series_cv(n_samples: int) -> TimeSeriesSplit:
        """Build leakage-safe temporal CV folds for ElasticNetCV."""
        n_splits = max(2, min(5, n_samples // 14))
        if n_samples < 30:
            n_splits = 2
        return TimeSeriesSplit(n_splits=n_splits)

    @staticmethod
    def _append_feature_if_sufficient_history(
        features_list: list,
        feature_names: list[str],
        values: pd.Series | np.ndarray,
        name: str,
        min_non_null: int = 30,
        reference_len: int | None = None,
        min_coverage_ratio: float = 0.0,
        max_required_non_null: int | None = None,
    ) -> None:
        if name in feature_names:
            return
        arr = values.values if isinstance(values, pd.Series) else np.asarray(values)
        required_non_null = int(min_non_null)
        if reference_len is not None and float(min_coverage_ratio) > 0.0:
            coverage_required = int(np.ceil(float(reference_len) * float(min_coverage_ratio)))
            required_non_null = max(required_non_null, coverage_required)
        if max_required_non_null is not None:
            required_non_null = min(required_non_null, int(max_required_non_null))
        if int(pd.Series(arr).notna().sum()) < required_non_null:
            return
        features_list.append(arr)
        feature_names.append(name)

    @staticmethod
    def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.size == 0:
            return float("inf")
        return float(np.mean(np.abs(y_true - y_pred)))

    def _weighted_model_average(
        self,
        preds: list[float],
        val_maes: list[float],
        window_lengths: list[int] | None = None,
        prefer_short_windows: bool = False,
    ) -> float:
        """Weighted mean of model predictions via inverse-MAE weighting.

        Note: window pruning (MAE > 1.5x best) was tested and REVERTED
        because it degraded MAE. Simple inverse-MAE weighting is kept.
        """
        if not preds:
            return float("nan")
        if len(preds) == 1 or len(val_maes) != len(preds):
            return float(np.mean(preds))

        maes = np.array([m if np.isfinite(m) and m > 1e-6 else 1e-6 for m in val_maes], dtype=float)
        weights = 1.0 / maes
        if prefer_short_windows and window_lengths is not None and len(window_lengths) == len(preds):
            lengths = np.array([max(1, int(w)) for w in window_lengths], dtype=float)
            lengths = lengths / lengths.min()
            short_bonus = 1.0 / np.sqrt(lengths)
            weights = weights * short_bonus
        weights = weights / weights.sum()
        return float(np.dot(np.array(preds, dtype=float), weights))

    def _fit_qra(
        self,
        X_full: pd.DataFrame,
        y_full: pd.Series,
        lasso_models: list,
        n_cal: int = 60,
    ) -> dict[float, "QuantileRegressor"] | None:
        """Fit QRA (Quantile Regression Averaging) on recent calibration data.

        Nowotarski & Weron (2015): treat per-window predictions as features
        in a quantile regression. Learns optimal combination weights that
        may vary across quantiles, providing calibrated probabilistic forecasts.

        Returns dict mapping quantile level -> fitted QuantileRegressor, or None.
        """
        n_cal = min(n_cal, len(y_full) - 30)
        if n_cal < 20 or len(lasso_models) < 2:
            return None

        X_cal = X_full.iloc[-n_cal:]
        y_cal = y_full.iloc[-n_cal:].values.astype(float)

        # Generate per-window predictions on calibration set
        window_preds = []
        for model, mu, sigma, scaler, _, _, _ in lasso_models:
            X_arr = np.nan_to_num(X_cal.values.astype(float), nan=0.0)
            X_scaled = scaler.transform(X_arr)
            pred_t = model.predict(X_scaled)
            window_preds.append(_asinh_inverse(pred_t, mu, sigma))

        if not window_preds:
            return None

        # Feature matrix: each column is one window's predictions
        Z = np.column_stack(window_preds)  # (n_cal, n_windows)

        qra_models = {}
        for q in [0.10, 0.50, 0.90]:
            try:
                qr = QuantileRegressor(
                    quantile=q,
                    alpha=0.01,  # light regularization
                    solver="highs",
                )
                qr.fit(Z, y_cal)
                qra_models[q] = qr
            except Exception:
                pass

        return qra_models if len(qra_models) >= 2 else None

    def _qra_predict(
        self,
        qra_models: dict[float, "QuantileRegressor"],
        window_preds: list[float],
    ) -> dict[float, float]:
        """Apply fitted QRA to get quantile predictions."""
        Z = np.array(window_preds).reshape(1, -1)
        result = {}
        for q, model in qra_models.items():
            result[q] = float(model.predict(Z)[0])
        return result

    def _recent_bias_by_regime(
        self,
        X_full: pd.DataFrame,
        y_full: pd.Series,
        lasso_models: list,
        recent_n: int = 14,
    ) -> tuple[float, float, float]:
        """Estimate recent bias from the held-out validation tails of LASSO windows."""
        pred_frames = []
        target_frames = []
        weight_map = {}
        for idx, (model, mu, sigma, scaler, X_w, y_w, val_mae) in enumerate(lasso_models):
            n_val = max(5, min(14, len(y_w) // 4))
            if n_val < 5:
                continue
            X_val = X_w.iloc[-n_val:]
            y_val = y_w.iloc[-n_val:].astype(float)
            X_arr = np.nan_to_num(X_val.values.astype(float), nan=0.0)
            pred_t = model.predict(scaler.transform(X_arr))
            col = f"m{idx}"
            pred_frames.append(pd.Series(_asinh_inverse(pred_t, mu, sigma), index=X_val.index, name=col))
            target_frames.append(y_val.rename("actual"))
            weight_map[col] = 1.0 / max(1e-6, val_mae) if np.isfinite(val_mae) else 1.0

        if not pred_frames:
            return 0.0, 0.0, 0.0

        pred_df = pd.concat(pred_frames, axis=1).sort_index().tail(recent_n)
        y_recent_s = pd.concat(target_frames, axis=1).bfill(axis=1).iloc[:, 0].reindex(pred_df.index)
        if pred_df.empty or y_recent_s.dropna().empty:
            return 0.0, 0.0, 0.0

        weights = pd.Series(weight_map, dtype=float).reindex(pred_df.columns).fillna(1.0)
        weights = weights / weights.sum()
        pred_recent = pred_df.mul(weights, axis=1).sum(axis=1).values.astype(float)
        y_recent = y_recent_s.values.astype(float)

        err = np.array(pred_recent, dtype=float) - y_recent
        overall = float(np.nanmean(err)) if np.isfinite(err).any() else 0.0
        idx_recent = pd.to_datetime(pred_df.index)
        weekend_mask = idx_recent.dayofweek >= 5
        if weekend_mask.any():
            weekend = float(np.nanmean(err[weekend_mask]))
        else:
            weekend = overall
        if (~weekend_mask).any():
            weekday = float(np.nanmean(err[~weekend_mask]))
        else:
            weekday = overall
        return overall, weekday, weekend

    def _detect_regime(self, y_full: pd.Series) -> str:
        """Detect current price regime from recent data.

        Returns 'spike', 'volatile', or 'normal' based on recent price
        behavior. Used to adapt model parameters (calibration windows,
        bias correction strength, etc.).
        """
        if len(y_full) < 30:
            return "normal"

        recent_7d = y_full.iloc[-min(7, len(y_full)):].values.astype(float)
        recent_30d = y_full.iloc[-min(30, len(y_full)):].values.astype(float)
        recent_90d = y_full.iloc[-min(90, len(y_full)):].values.astype(float)

        # Volatility ratio: 7d std vs 90d std
        std_7d = float(np.nanstd(recent_7d))
        std_90d = float(np.nanstd(recent_90d))
        vol_ratio = std_7d / max(std_90d, 0.1)

        # Spike detection: any of last 7 days > Q95 of 90d
        q95 = float(np.nanquantile(recent_90d, 0.95))
        recent_spikes = (recent_7d > q95).sum()

        # Price level shift: 7d mean vs 30d mean
        mean_7d = float(np.nanmean(recent_7d))
        mean_30d = float(np.nanmean(recent_30d))
        level_shift = abs(mean_7d - mean_30d) / max(abs(mean_30d), 1.0)

        if recent_spikes >= 2 or vol_ratio > 2.0:
            return "spike"
        elif vol_ratio > 1.3 or level_shift > 0.15:
            return "volatile"
        else:
            return "normal"

    def _regime_adjusted_params(
        self, regime: str
    ) -> dict:
        """Return regime-dependent parameter adjustments.

        In spike regime: shorter effective windows, stronger bias correction,
        higher FM weight. In normal: use defaults.
        """
        if regime == "spike":
            return {
                "bias_strength": 0.85,     # stronger (default 0.70)
                "ar1_coeff": 0.7,          # stronger (default 0.6)
                "fm_weight_boost": 0.05,   # give FM more weight
                "prefer_short_windows": True,
            }
        elif regime == "volatile":
            return {
                "bias_strength": 0.75,
                "ar1_coeff": 0.70,
                "fm_weight_boost": 0.03,
                "prefer_short_windows": False,
            }
        else:
            return {
                "bias_strength": 0.70,
                "ar1_coeff": 0.65,
                "fm_weight_boost": 0.0,
                "prefer_short_windows": False,
            }

    def _is_chronos2_finetuned(self) -> bool:
        if self._fm is None:
            return False
        fm_model_name = getattr(self._fm, "CHRONOS2_MODEL", "")
        return "chronos2_finetuned" in str(fm_model_name).lower()

    def _foundation_blend_weight(
        self,
        horizon: int,
        hour: int,
        regime_params: dict | None = None,
    ) -> float:
        if self._fm is None or horizon > self.foundation_blend_max_horizon_days:
            return 0.0

        regime_boost = 0.0
        if regime_params is not None:
            regime_boost = float(regime_params.get("fm_weight_boost", 0.0))

        if self._is_chronos2_finetuned():
            if horizon <= 1:
                w_fm = 0.18
            elif horizon == 2:
                w_fm = 0.35
            elif horizon == 3:
                w_fm = 0.38
            else:
                w_fm = max(0.14, 0.38 - 0.03 * float(horizon - 3))
            if hour in PEAK_HOURS:
                w_fm += 0.03
            # Current Swiss CT failures are dominated by daytime solar-collapse regimes
            # (spring holidays/weekends with deep midday price crashes). Give the
            # finetuned Chronos path more authority on J+1 daylight hours, where it
            # consistently improves recent OOS backtests on the live snapshot.
            if horizon <= 1 and 8 <= int(hour) <= 17:
                w_fm += 0.16
            w_fm += min(0.04, regime_boost)
            return float(min(0.48, max(0.12, w_fm)))

        w_fm = max(0.05, 0.20 - 0.02 * float(horizon - 1))
        w_fm += min(0.03, regime_boost)
        return float(min(0.28, max(0.05, w_fm)))

    def _apply_spike_uplift(
        self,
        pred: float,
        y_full: pd.Series,
        days_ahead: int,
    ) -> float:
        """Add a controlled uplift in high-price regimes to reduce tail underprediction."""
        if len(y_full) < 30:
            return pred
        recent = y_full.iloc[-min(90, len(y_full)) :].values.astype(float)
        q75 = float(np.quantile(recent, 0.75))
        q90 = float(np.quantile(recent, 0.90))
        spread = max(1.0, q90 - q75)
        if pred <= q75:
            return pred
        intensity = min(1.2, (pred - q75) / spread)
        horizon_scale = 1.0 + 0.04 * max(days_ahead - 1, 0)
        uplift = intensity * spread * 0.05 * horizon_scale
        uplift = min(4.0, uplift)
        return float(pred + uplift)

    def _fit_gbm_for_hour(
        self,
        hour: int,
        X_full: pd.DataFrame,
        y_full: pd.Series,
    ) -> tuple | None:
        """Fit LightGBM for one hour (captures nonlinearities).

        GBM is used as an ensemble member alongside LASSO
        where price dynamics are nonlinear (load ramp, solar decline, etc.).
        Lago et al. (2021) showed LEAR+DNN reduces MAE 5-10% on peak.
        """
        n = min(365, len(y_full))
        if n < 60:
            return None

        X_train = X_full.iloc[-n:]
        y_train = y_full.iloc[-n:]

        X_arr = np.nan_to_num(X_train.values.astype(float), nan=0.0)
        y_arr = y_train.values.astype(float)
        n_val = max(7, int(len(y_arr) * 0.15))
        if len(y_arr) - n_val < 30:
            return None
        X_fit, X_val = X_arr[:-n_val], X_arr[-n_val:]
        y_fit, y_val = y_arr[:-n_val], y_arr[-n_val:]

        try:
            gbm = lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=6,
                num_leaves=31,
                learning_rate=0.05,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                reg_alpha=0.1,
                device_type=self._lgbm_device_type,
                n_jobs=-1,
                verbosity=-1,
                random_state=self.random_state,
            )
            gbm.fit(
                X_fit,
                y_fit,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
            return gbm
        except Exception as exc:
            if self._lgbm_device_type == "gpu":
                logger.warning("  LightGBM GPU unavailable for h=%d, falling back to CPU: %s", hour, exc)
                self._lgbm_device_type = "cpu"
                return self._fit_gbm_for_hour(hour, X_full, y_full)
            logger.debug("  GBM h=%d failed: %s", hour, exc)
            return None

    def _gbm_weight_cap_for_hour(self, hour: int) -> float:
        """Return the maximum allowed GBM blend weight for a delivery hour."""
        if 7 <= int(hour) <= 20:
            return max(0.15, self.gbm_weight_cap_peak_block)
        return 0.40

    def _fit_lasso_for_hour(
        self,
        X_full: pd.DataFrame,
        y_full: pd.Series,
        hour: int,
    ) -> list[tuple]:
        """Fit multi-window LASSO models for one delivery hour."""
        fitted = []
        for window in self.CALIBRATION_WINDOWS:
            n = min(window, len(y_full))
            if n < 30:
                continue
            X_w = X_full.iloc[-n:]
            y_w = y_full.iloc[-n:]

            y_arr = y_w.values.astype(float)
            # Fixed asinh transform for all windows (consistent forward/inverse)
            # Causal variant was removed: inverse transform was inconsistent
            # (per-timestep mu/sigma at training, global mu/sigma at prediction)
            y_t, mu, sigma = _asinh_transform(y_arr)

            try:
                X_arr = np.nan_to_num(X_w.values.astype(float), nan=0.0)
                # Hold out last n_val observations for out-of-time validation
                n_val = max(5, min(14, n // 4))
                n_train = n - n_val
                scaler = StandardScaler()
                scaler.fit(X_arr[:n_train])
                X_scaled = scaler.transform(X_arr)
                cv = self._time_series_cv(n_train)
                effective_max_iter = self.max_iter
                if self.use_governed_forecast_features and float(self.elasticnet_l1_ratio_effective) >= 0.5:
                    effective_max_iter = max(int(effective_max_iter), 5000)
                model = ElasticNetCV(
                    l1_ratio=self.elasticnet_l1_ratio_effective,
                    max_iter=effective_max_iter,
                    cv=cv,
                    random_state=self.random_state,
                )

                # Exponential decay weighting: recent obs matter more
                # Half-life 180 days (Paraschiv et al., Swiss market)
                days_ago_train = np.arange(n_train)[::-1].astype(float) + n_val
                sample_weights = np.exp(-days_ago_train * (np.log(2) / 180))
                model.fit(X_scaled[:n_train], y_t[:n_train],
                          sample_weight=sample_weights)

                # Out-of-time validation on held-out recent data
                Xv_scaled = scaler.transform(
                    np.nan_to_num(X_w.iloc[-n_val:].values.astype(float), nan=0.0)
                )
                y_val = y_w.iloc[-n_val:].values.astype(float)
                pred_t_val = model.predict(Xv_scaled)
                lower_t, upper_t = _asinh_domain_bounds(mu, sigma)
                if (
                    not np.all(np.isfinite(pred_t_val))
                    or float(np.nanmax(pred_t_val)) > upper_t + 0.5
                    or float(np.nanmin(pred_t_val)) < lower_t - 0.5
                ):
                    logger.debug(
                        "Discard unstable LASSO h=%d w=%d: transformed preds outside physical bounds",
                        hour,
                        window,
                    )
                    continue
                pred_val = _asinh_inverse(pred_t_val, mu, sigma)
                val_mae = self._safe_mae(y_val, pred_val)

                fitted.append((model, mu, sigma, scaler, X_w, y_w, val_mae))
            except Exception as exc:
                logger.warning("LASSO h=%d w=%d: %s", hour, window, exc)
        return fitted

    def _fit_mlp_ensemble(
        self,
        hour: int,
        X_full: pd.DataFrame,
        y_full: pd.Series,
    ) -> tuple[MLPRegressor, StandardScaler] | None:
        """Train MLP as ensemble member alongside LASSO."""
        n = min(365, len(y_full))
        if n < 100:
            return None

        X_train = X_full.iloc[-n:]
        y_train = y_full.iloc[-n:]

        X_arr = np.nan_to_num(X_train.values.astype(float), nan=0.0)
        y_arr = y_train.values.astype(float)

        mlp_scaler = StandardScaler()
        X_scaled = mlp_scaler.fit_transform(X_arr)

        try:
            # Chronological split: last 15% for validation (no random leak)
            mlp = MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=500,
                early_stopping=False,
                random_state=self.random_state,
                learning_rate_init=0.001,
                alpha=0.01,
            )
            mlp.fit(X_scaled, y_arr)
            return mlp, mlp_scaler
        except Exception as exc:
            logger.debug("  MLP h=%d failed: %s", hour, exc)
            return None

    def _build_gbm_cache(
        self,
        prices_ref: pd.Series,
        exog_ref: pd.DataFrame,
    ) -> dict[int, tuple]:
        """Pre-train per-hour GBM side models and their blend weights."""
        gbm_cache: dict[int, tuple] = {}
        if not self.use_gbm_blend:
            return gbm_cache
        for hour in range(24):
            X_pre, y_pre = self._build_features(
                prices_ref, exog_ref, target_hour=hour
            )
            if len(y_pre) < 60:
                continue
            gbm_model = self._fit_gbm_for_hour(hour, X_pre, y_pre)
            if gbm_model is None:
                continue
            n_eval = min(14, len(y_pre) // 4)
            if n_eval < 5:
                continue
            X_ev = X_pre.iloc[-n_eval:].copy()
            X_ev_arr = np.nan_to_num(X_ev.values.astype(float), nan=0.0)
            X_ev_for_gbm = pd.DataFrame(X_ev_arr, columns=X_pre.columns, index=X_ev.index)
            y_ev = y_pre.iloc[-n_eval:].values.astype(float)
            gbm_mae = self._safe_mae(y_ev, gbm_model.predict(X_ev_for_gbm))
            lasso_pre = self._fit_lasso_for_hour(X_pre, y_pre, hour)
            if lasso_pre:
                lasso_eval_preds = []
                for mdl, mu, sigma, scl, _, _, _ in lasso_pre:
                    pred_t = mdl.predict(scl.transform(X_ev_arr))
                    lasso_eval_preds.append(_asinh_inverse(pred_t, mu, sigma))
                lasso_mae_ev = self._safe_mae(y_ev, np.mean(lasso_eval_preds, axis=0))
                inv_gbm = 1.0 / max(1e-6, gbm_mae)
                inv_lasso = 1.0 / max(1e-6, lasso_mae_ev)
                w_gbm = inv_gbm / (inv_gbm + inv_lasso)
                w_gbm = max(0.15, min(self._gbm_weight_cap_for_hour(hour), w_gbm))
            else:
                w_gbm = 0.20
            gbm_cache[hour] = (gbm_model, w_gbm, list(X_pre.columns))
        if gbm_cache:
            logger.info("Pre-trained GBM for %d hours (device=%s)", len(gbm_cache), self._lgbm_device_type)
        return gbm_cache

    def _compute_conformal_residuals(
        self,
        hour: int,
        X_full: pd.DataFrame,
        y_full: pd.Series,
        lasso_models: list,
    ) -> np.ndarray:
        """Compute conformal residuals on held-out validation tails."""
        pred_frames = []
        target_frames = []
        weights_raw = {}
        for idx, (model, mu, sigma, scaler, X_w, y_w, val_mae) in enumerate(lasso_models):
            n_val = max(5, min(14, len(y_w) // 4))
            if n_val < 5:
                continue
            X_cal = X_w.iloc[-n_val:]
            y_cal = y_w.iloc[-n_val:].astype(float)
            X_arr = np.nan_to_num(X_cal.values.astype(float), nan=0.0)
            X_scaled = scaler.transform(X_arr)
            pred_t = model.predict(X_scaled)
            col = f"m{idx}"
            pred_frames.append(pd.Series(_asinh_inverse(pred_t, mu, sigma), index=X_cal.index, name=col))
            target_frames.append(y_cal.rename("actual"))
            weights_raw[col] = 1.0 / max(1e-6, val_mae) if np.isfinite(val_mae) else 1.0

        if not pred_frames:
            return np.array([])

        pred_df = pd.concat(pred_frames, axis=1).sort_index()
        y_cal_s = pd.concat(target_frames, axis=1).bfill(axis=1).iloc[:, 0].reindex(pred_df.index)
        weights = pd.Series(weights_raw, dtype=float).reindex(pred_df.columns).fillna(1.0)
        weights = weights / weights.sum()
        combined = pred_df.mul(weights, axis=1).sum(axis=1)
        residuals = (y_cal_s - combined).abs().dropna()
        return residuals.values.astype(float)

    def _recalibrate_variance(
        self,
        raw_pred: float,
        hour: int,
        lasso_models: list,
        X_full: pd.DataFrame,
        y_full: pd.Series,
        days_ahead: int = 1,
    ) -> float:
        """Per-hour variance recalibration.

        Corrects LASSO shrinkage by scaling predictions to match
        recent actual variance. Compression ratio at h12 is 0.36 —
        the model only captures 36% of actual variance. This expands it.
        """
        if hour not in self._var_calib:
            return raw_pred

        actual_mean, actual_std = self._var_calib[hour]

        # Compute LASSO OOS std on last 14 days (held out from shortest
        # calibration window to avoid in-sample variance underestimate).
        n_oos = min(14, len(X_full) - 30)
        if n_oos < 7:
            return raw_pred

        X_oos = X_full.iloc[-n_oos:]
        lasso_oos = []
        val_maes = []
        for model, mu, sigma, scaler, _, _, val_mae in lasso_models:
            X_arr = np.nan_to_num(X_oos.values.astype(float), nan=0.0)
            X_scaled = scaler.transform(X_arr)
            pred_t = model.predict(X_scaled)
            lasso_oos.append(_asinh_inverse(pred_t, mu, sigma))
            val_maes.append(val_mae)

        if not lasso_oos:
            return raw_pred

        preds_mat = np.vstack(lasso_oos)
        weights = np.array(
            [1.0 / max(1e-6, v) if np.isfinite(v) else 1.0 for v in val_maes],
            dtype=float,
        )
        weights = weights / weights.sum()
        lasso_avg = np.average(preds_mat, axis=0, weights=weights)
        lasso_mean = float(np.mean(lasso_avg))
        lasso_std = float(np.std(lasso_avg))

        if lasso_std < 1e-6:
            return raw_pred

        # Expand the deviation from mean to match actual std
        # Cap at 1.0 — no expansion (LASSO compression preserved)
        var_ratio = min(actual_std / lasso_std, 1.0)
        recalibrated = actual_mean + (raw_pred - lasso_mean) * var_ratio

        # Quantile-based clamp: keep extremes while avoiding unstable tails.
        n_clamp = min(28, len(y_full) - 5)
        y_recent = y_full.iloc[-n_clamp:].values.astype(float)
        q_low = float(np.quantile(y_recent, 0.03))
        q_high = float(np.quantile(y_recent, 0.97))
        horizon_widen = 1.0 + 0.03 * max(days_ahead - 1, 0)
        lower = min(q_low, actual_mean - 2.0 * actual_std) * horizon_widen
        upper = max(q_high, actual_mean + 2.0 * actual_std) * horizon_widen
        if lower > upper:
            lower, upper = upper, lower
        recalibrated = float(np.clip(recalibrated, lower, upper))

        return recalibrated

    def predict(
        self,
        horizon_days: int = 10,
        quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
    ) -> pd.DataFrame:
        """Generate hourly price forecasts for the next `horizon_days`."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        last_date = pd.Timestamp(
            self._idx_local[-1].date(),
            tz=self.tz,
        )

        # Foundation model: zero-shot forecast with covariates (if Chronos-2)
        fm_preds = None
        if self._fm is not None and self._fm.available:
            # Build covariate dict from exogenous data
            fm_covariates = {}
            for code, series in getattr(self, "_neighbor_price_series_h_", {}).items():
                if len(series) > 48:
                    fm_covariates[f"{code}_price"] = series
            for col in ["load_mw", "solar_mw", "wind_mw"]:
                if col in self.exog_.columns:
                    s = self.exog_[col].dropna()
                    if len(s) > 48:
                        fm_covariates[col] = s

            fm_result = self._fm.forecast(
                self.prices_h_,
                horizon=24 * horizon_days,
                covariates=fm_covariates if fm_covariates else None,
            )
            if fm_result is not None:
                fm_preds = fm_result
                logger.info(
                    "Foundation model forecast: %d hours, %d covariates, backend=%s",
                    len(fm_result["median"]),
                    len(fm_covariates),
                    self._fm.backend,
                )

        gbm_cache = {}
        if self.use_gbm_blend and self.gbm_blend_max_horizon_days > 0:
            gbm_cache = self._build_gbm_cache(self.prices_h_, self.exog_)

        all_forecasts = []
        n_mlp_used = 0

        for hour in range(24):
            X_full, y_full = self._build_features(
                self.prices_h_, self.exog_, target_hour=hour
            )

            if len(y_full) < 30:
                logger.warning("Hour %d: only %d samples, skipping", hour, len(y_full))
                continue

            lasso_models = self._fit_lasso_for_hour(X_full, y_full, hour)
            if not lasso_models:
                continue

            mlp_model, mlp_scaler = None, None
            if self.use_mlp_blend:
                mlp_result = self._fit_mlp_ensemble(hour, X_full, y_full)
                if mlp_result is not None:
                    mlp_model, mlp_scaler = mlp_result
                    self._mlp_models[hour] = (mlp_model, mlp_scaler)
                    n_mlp_used += 1

            recent_bias, weekday_bias, weekend_bias = self._recent_bias_by_regime(
                X_full, y_full, lasso_models, recent_n=14
            )

            regime = self._detect_regime(y_full)
            rp = self._regime_adjusted_params(regime)

            # Conformal residuals
            conf_residuals = self._compute_conformal_residuals(
                hour, X_full, y_full, lasso_models
            )
            if len(conf_residuals) > 0:
                self._conformal_residuals[hour] = conf_residuals

            # QRA: fit quantile regression on per-window predictions
            qra_models = self._fit_qra(X_full, y_full, lasso_models)

            for d in range(1, horizon_days + 1):
                forecast_date = (last_date + timedelta(days=d)).date()

                # LASSO predictions
                window_preds = []
                window_maes = []
                window_lengths = []
                x_pred = None
                for model, mu, sigma, scaler, X_w, _, val_mae in lasso_models:
                    x_pred = self._build_prediction_row(
                        X_full, y_full, forecast_date, hour, d,
                    )
                    if x_pred is not None:
                        x_arr = np.nan_to_num(x_pred.reshape(1, -1), nan=0.0)
                        x_scaled = scaler.transform(x_arr)
                        pred_t = model.predict(x_scaled)[0]
                        window_preds.append(_asinh_inverse(pred_t, mu, sigma))
                        window_maes.append(val_mae)
                        window_lengths.append(len(X_w))

                if not window_preds:
                    continue

                lasso_mean = self._weighted_model_average(
                    window_preds,
                    window_maes,
                    window_lengths=window_lengths,
                    prefer_short_windows=self.use_regime_short_window_weighting and rp.get("prefer_short_windows", False),
                )

                # Variance recalibration
                recalibrated = self._recalibrate_variance(
                    lasso_mean, hour, lasso_models, X_full, y_full, d
                )

                # Bias correction with horizon damping.
                horizon_bias_weight = max(0.25, 1.0 - 0.06 * (d - 1))
                recalibrated = recalibrated - horizon_bias_weight * recent_bias
                regime_bias = weekend_bias if pd.Timestamp(forecast_date).dayofweek >= 5 else weekday_bias
                recalibrated = recalibrated - 0.70 * horizon_bias_weight * regime_bias

                # Ensemble: adaptive average with MLP (reduce MLP weight on long horizon).
                if mlp_model is not None and x_pred is not None:
                    x_arr = np.nan_to_num(x_pred.reshape(1, -1), nan=0.0)
                    x_mlp_scaled = mlp_scaler.transform(x_arr)
                    mlp_pred = float(mlp_model.predict(x_mlp_scaled)[0])
                    n_eval = max(8, min(14, len(y_full) // 4))
                    X_eval = X_full.iloc[-n_eval:]
                    y_eval = y_full.iloc[-n_eval:].values.astype(float)
                    X_eval_arr = np.nan_to_num(X_eval.values.astype(float), nan=0.0)
                    lasso_eval = []
                    for model, mu, sigma, scaler, _, _, _ in lasso_models:
                        pred_t = model.predict(scaler.transform(X_eval_arr))
                        lasso_eval.append(_asinh_inverse(pred_t, mu, sigma))
                    lasso_eval_pred = np.mean(lasso_eval, axis=0)
                    lasso_mae = self._safe_mae(y_eval, lasso_eval_pred)
                    mlp_eval_pred = mlp_model.predict(mlp_scaler.transform(X_eval_arr))
                    mlp_mae = self._safe_mae(y_eval, mlp_eval_pred)
                    inv_lasso = 1.0 / max(1e-6, lasso_mae)
                    inv_mlp = 1.0 / max(1e-6, mlp_mae)
                    w_mlp = inv_mlp / (inv_lasso + inv_mlp)
                    w_mlp = max(0.15, min(0.45, w_mlp))
                    w_mlp *= max(0.4, 1.0 - 0.08 * (d - 1))
                    w_lasso = 1.0 - w_mlp
                    final_pred = w_lasso * recalibrated + w_mlp * mlp_pred
                else:
                    final_pred = recalibrated

                if d <= self.gbm_blend_max_horizon_days and hour in gbm_cache and x_pred is not None:
                    gbm_model, w_gbm, gbm_feature_names = gbm_cache[hour]
                    x_gbm = np.nan_to_num(x_pred.reshape(1, -1), nan=0.0)
                    x_gbm_df = pd.DataFrame(x_gbm, columns=gbm_feature_names)
                    gbm_pred = float(gbm_model.predict(x_gbm_df)[0])
                    final_pred = (1.0 - w_gbm) * final_pred + w_gbm * gbm_pred

                # Tail regime correction to reduce underprediction of spikes.
                final_pred = self._apply_spike_uplift(final_pred, y_full, d)
                final_pred = max(final_pred, -50.0)

                # Foundation model blending AFTER all LEAR post-processing
                fm_raw = None
                if fm_preds is not None:
                    fm_idx = (d - 1) * 24 + hour
                    if fm_idx < len(fm_preds["median"]):
                        fm_raw = float(fm_preds["median"][fm_idx])
                        if np.isfinite(fm_raw):
                            w_fm = self._foundation_blend_weight(d, hour)
                            if w_fm > 0.0:
                                final_pred = (1.0 - w_fm) * final_pred + w_fm * fm_raw

                # Prediction intervals: QRA (preferred) or conformal (fallback)
                if qra_models is not None and len(window_preds) == len(lasso_models):
                    # QRA: calibrated quantile predictions from per-window outputs
                    qra_result = self._qra_predict(qra_models, window_preds)
                    p10 = qra_result.get(0.10, final_pred - 15.0)
                    p90 = qra_result.get(0.90, final_pred + 15.0)
                    # Widen intervals for longer horizons
                    horizon_scale = 1.0 + 0.05 * (d - 1)
                    mid = (p10 + p90) / 2
                    half = (p90 - p10) / 2 * horizon_scale
                    p10 = mid - half
                    p90 = mid + half
                else:
                    # Conformal fallback
                    q_low = float(min(quantiles)) if len(quantiles) > 0 else 0.10
                    q_high = float(max(quantiles)) if len(quantiles) > 0 else 0.90
                    tail_q = max(q_high, 1.0 - q_low)
                    if len(conf_residuals) > 0:
                        horizon_scale = 1.0 + 0.05 * (d - 1)
                        scaled_residuals = conf_residuals * horizon_scale
                        spread_q = float(np.quantile(scaled_residuals, tail_q))
                        p10 = final_pred - spread_q
                        p90 = final_pred + spread_q
                    else:
                        spread = abs(final_pred) * (0.05 + 0.01 * d)
                        z = 1.28 if tail_q <= 0.90 else 1.64
                        p10 = final_pred - z * spread
                        p90 = final_pred + z * spread

                all_forecasts.append({
                    "date": forecast_date,
                    "hour": hour,
                    "price_lear": final_pred,
                    "price_p10": p10,
                    "price_p90": p90,
                    "n_windows": len(window_preds),
                    "mlp_used": mlp_model is not None,
                    "fm_used": fm_raw is not None,
                    "fm_raw_price": fm_raw,
                })

        if not all_forecasts:
            raise ValueError("LEAR produced no forecasts")

        result = pd.DataFrame(all_forecasts)
        result = result.sort_values(["date", "hour"]).reset_index(drop=True)

        result["timestamp"] = pd.to_datetime(result["date"]) + pd.to_timedelta(
            result["hour"], unit="h"
        )
        result["timestamp"] = result["timestamp"].dt.tz_localize(
            self.tz,
            nonexistent="shift_forward",
            ambiguous="infer",
        ).dt.tz_convert("UTC")

        result["days_ahead"] = (
            (pd.to_datetime(result["date"]) - pd.to_datetime(result["date"].min()))
            .dt.days + 1
        )

        n_hours = len(result)
        n_days = result["date"].nunique()
        mean_price = result["price_lear"].mean()
        logger.info(
            "LEAR forecast: %d hours (%d days), mean=%.1f EUR/MWh, MLP=%d/24h",
            n_hours, n_days, mean_price, n_mlp_used,
        )
        return result

    @staticmethod
    def _feature_tail_mean(X_full: pd.DataFrame, col_name: str, n: int = 28) -> float:
        data = X_full.loc[:, col_name]
        if isinstance(data, pd.DataFrame):
            arr = data.tail(n).to_numpy(dtype=float).ravel()
        else:
            arr = data.dropna().tail(n).to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else 0.0

    @staticmethod
    def _feature_last_value(X_full: pd.DataFrame, col_name: str) -> float:
        data = X_full.loc[:, col_name]
        if isinstance(data, pd.DataFrame):
            arr = data.tail(1).to_numpy(dtype=float).ravel()
        else:
            arr = data.dropna().tail(1).to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr[-1]) if arr.size else 0.0

    def _build_prediction_row(
        self,
        X_full: pd.DataFrame,
        y_full: pd.Series,
        forecast_date,
        hour: int,
        days_ahead: int,
    ) -> np.ndarray | None:
        """Build feature row for a future date.

        Simplified and more robust than the original:
        - Price lags: use last known row shifted by effective lag
        - Aggregates: recompute from known data
        - Calendar: compute directly from forecast_date
        """
        import datetime

        n_features = X_full.shape[1]
        x = np.zeros(n_features)
        cols = X_full.columns

        for i, col_name in enumerate(cols):
            # ── Price features ──
            if col_name.startswith("price_same_dow_mean_h"):
                vals = X_full[col_name].dropna()
                x[i] = vals.iloc[-1] if not vals.empty else 0.0

            elif col_name.startswith("price_") and "_d-" in col_name:
                if "mean" in col_name or "peak" in col_name or "offpeak" in col_name or "max" in col_name:
                    # Aggregate: use last known value (shifted if needed)
                    lag_str = col_name.split("_d-")[1]
                    lag = int(lag_str)
                    eff_lag = lag + days_ahead - 1
                    row_idx = max(0, len(X_full) - max(eff_lag, 1))
                    x[i] = X_full.iloc[row_idx][col_name] if row_idx < len(X_full) else X_full[col_name].dropna().iloc[-1]
                elif "_h" in col_name:
                    # Specific hour price
                    parts = col_name.split("_")
                    lag = int(parts[1].replace("d-", ""))
                    eff_lag = lag + days_ahead - 1
                    if eff_lag <= 0:
                        eff_lag = 1
                    row_idx = max(0, len(X_full) - eff_lag)
                    x[i] = X_full.iloc[min(row_idx, len(X_full) - 1)][col_name]
                else:
                    # Momentum, volatility, etc.
                    x[i] = X_full[col_name].dropna().iloc[-1] if not X_full[col_name].dropna().empty else 0

            # ── Neighbor price features ──
            elif any(col_name.startswith(f"{code}_price_") for code in ["de", "fr", "at", "it"]):
                if "_d-" in col_name:
                    parts = col_name.split("_d-")
                    lag = int(parts[1].split("_")[0]) if "_" in parts[1] else int(parts[1])
                    eff_lag = lag + days_ahead - 1
                    if eff_lag <= 0:
                        eff_lag = 1
                    row_idx = max(0, len(X_full) - eff_lag)
                    x[i] = X_full.iloc[min(row_idx, len(X_full) - 1)][col_name]
                else:
                    x[i] = X_full[col_name].dropna().iloc[-1] if not X_full[col_name].dropna().empty else 0

            elif col_name.startswith("ch_de_spread_"):
                x[i] = self._feature_last_value(X_full, col_name)

            # ── Calendar features ──
            elif col_name.startswith("dow_"):
                d = int(col_name.split("_")[1])
                if isinstance(forecast_date, datetime.date):
                    x[i] = 1.0 if forecast_date.weekday() == d else 0.0

            elif col_name == "is_weekend":
                if isinstance(forecast_date, datetime.date):
                    x[i] = 1.0 if forecast_date.weekday() >= 5 else 0.0

            elif col_name == "weekend_midday_regime":
                if isinstance(forecast_date, datetime.date):
                    x[i] = 1.0 if forecast_date.weekday() >= 5 else 0.0

            elif col_name == "weekday_peak_regime":
                if isinstance(forecast_date, datetime.date):
                    x[i] = 1.0 if forecast_date.weekday() < 5 else 0.0

            elif col_name == "month_sin":
                if isinstance(forecast_date, datetime.date):
                    x[i] = np.sin(2 * np.pi * forecast_date.month / 12)

            elif col_name == "month_cos":
                if isinstance(forecast_date, datetime.date):
                    x[i] = np.cos(2 * np.pi * forecast_date.month / 12)

            elif col_name == "is_holiday_ch":
                if isinstance(forecast_date, datetime.date):
                    ch_hols, _, _ = _get_holidays(forecast_date.year)
                    x[i] = 1.0 if forecast_date in ch_hols else 0.0

            elif col_name == "is_holiday_de":
                if isinstance(forecast_date, datetime.date):
                    _, de_hols, _ = _get_holidays(forecast_date.year)
                    x[i] = 1.0 if forecast_date in de_hols else 0.0

            elif col_name == "is_holiday_fr":
                if isinstance(forecast_date, datetime.date):
                    _, _, fr_hols = _get_holidays(forecast_date.year)
                    x[i] = 1.0 if forecast_date in fr_hols else 0.0

            elif col_name == "neighbor_holiday_mismatch":
                if isinstance(forecast_date, datetime.date):
                    ch_hols, de_hols, fr_hols = _get_holidays(forecast_date.year)
                    x[i] = 1.0 if (
                        (forecast_date not in ch_hols)
                        and ((forecast_date in de_hols) or (forecast_date in fr_hols))
                    ) else 0.0

            elif col_name == "neighbor_holiday_mismatch_daytime":
                if isinstance(forecast_date, datetime.date):
                    ch_hols, de_hols, fr_hols = _get_holidays(forecast_date.year)
                    x[i] = 1.0 if (
                        (forecast_date not in ch_hols)
                        and ((forecast_date in de_hols) or (forecast_date in fr_hols))
                    ) else 0.0

            # ── Exogenous ──
            elif (
                (col_name.startswith("forecast_") or col_name.startswith("wx_"))
                and "_d0_h" in col_name
            ):
                source_col = col_name.split("_d0_h")[0]
                pivot = getattr(self, "_forecast_feature_pivots", {}).get(source_col)
                if pivot is not None and forecast_date in pivot.index:
                    hour_col = int(col_name.split("_d0_h")[1])
                    x[i] = float(pivot.loc[forecast_date, hour_col]) if hour_col in pivot.columns else 0.0
                else:
                    x[i] = self._feature_tail_mean(X_full, col_name, n=28)

            elif (
                (col_name.startswith("forecast_") or col_name.startswith("wx_"))
                and col_name.endswith("_d0_midday_mean")
            ):
                source_col = col_name.replace("_d0_midday_mean", "")
                pivot = getattr(self, "_forecast_feature_pivots", {}).get(source_col)
                if pivot is not None and forecast_date in pivot.index:
                    midday_cols = [h for h in range(10, 16) if h in pivot.columns]
                    x[i] = float(pivot.loc[forecast_date, midday_cols].mean()) if midday_cols else 0.0
                else:
                    x[i] = self._feature_tail_mean(X_full, col_name, n=28)

            elif "_d0_" in col_name:
                x[i] = self._feature_tail_mean(X_full, col_name, n=28)

            elif col_name == "hydro_fill":
                x[i] = self._feature_last_value(X_full, col_name)

            else:
                # Commodity lags, exogenous lags — use last known
                x[i] = X_full[col_name].dropna().iloc[-1] if not X_full[col_name].dropna().empty else 0

        return x

    def backtest(
        self,
        n_days: int = 30,
        horizon: int = 1,
    ) -> pd.DataFrame:
        """Rolling out-of-sample backtest with AR error correction.

        Key improvement: uses error autocorrelation (lag-1 ≈ 0.50)
        to correct each day's forecast based on previous day's error.
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

        # Foundation model: pre-generate forecasts with covariates
        fm_daily_cache: dict[str, dict[str, np.ndarray]] = {}
        if self._fm is not None and self._fm.available:
            for test_date in test_dates:
                cutoff = pd.Timestamp(test_date, tz=self.tz).tz_convert("UTC")
                prices_trunc = self.prices_h_[:cutoff]
                if len(prices_trunc) >= 168:
                    # Build truncated covariates
                    fm_cov = {}
                    for code, series in getattr(self, "_neighbor_price_series_h_", {}).items():
                        trunc = series[:cutoff]
                        if len(trunc) > 48:
                            fm_cov[f"{code}_price"] = trunc
                    for col in ["load_mw", "solar_mw", "wind_mw"]:
                        if col in self.exog_.columns:
                            s = self.exog_[col][:cutoff].dropna()
                            if len(s) > 48:
                                fm_cov[col] = s

                    fm_result = self._fm.forecast(
                        prices_trunc,
                        horizon=24 * horizon,
                        covariates=fm_cov if fm_cov else None,
                    )
                    if fm_result is not None:
                        fm_daily_cache[str(test_date)] = fm_result
            if fm_daily_cache:
                logger.info(
                    "Foundation model backtest: %d/%d days, backend=%s",
                    len(fm_daily_cache), len(test_dates),
                    self._fm.backend,
                )

        gbm_cache: dict[int, tuple] = {}
        if self.use_gbm_blend and horizon <= self.gbm_blend_max_horizon_days:
            first_test_date = test_dates[0]
            prices_pretrain = self.prices_h_[:pd.Timestamp(first_test_date, tz=self.tz).tz_convert("UTC")]
            exog_pretrain = self.exog_.loc[:prices_pretrain.index[-1]]
            gbm_cache = self._build_gbm_cache(prices_pretrain, exog_pretrain)

        # Track previous day's errors for AR correction (lag-1, lag-2, lag-7)
        prev_day_errors: dict[int, float] = {}  # hour -> error (lag-1)
        prev2_day_errors: dict[int, float] = {}  # hour -> error (lag-2)
        # Expanding-window bias correction + lag-7 AR
        error_history: dict[int, list[float]] = {h: [] for h in range(24)}

        for test_idx, test_date in enumerate(test_dates):
            pos = list(all_dates).index(test_date)
            target_pos = pos + horizon - 1
            if target_pos >= len(all_dates):
                continue
            target_date = all_dates[target_pos]

            cutoff_pos = pos
            if cutoff_pos < 30:
                continue

            day_errors: dict[int, float] = {}

            for hour in range(24):
                prices_trunc = self.prices_h_[:pd.Timestamp(test_date, tz=self.tz).tz_convert("UTC")]
                exog_trunc = self.exog_.loc[:prices_trunc.index[-1]]

                X_full, y_full = self._build_features(
                    prices_trunc, exog_trunc, target_hour=hour
                )

                if len(y_full) < 30:
                    continue

                lasso_models = self._fit_lasso_for_hour(X_full, y_full, hour)
                if not lasso_models:
                    continue

                # Regime detection: adapt parameters to current market state
                regime = self._detect_regime(y_full)
                rp = self._regime_adjusted_params(regime)

                recent_bias, weekday_bias, weekend_bias = self._recent_bias_by_regime(
                    X_full, y_full, lasso_models, recent_n=14
                )

                # LASSO predictions
                preds = []
                pred_maes = []
                pred_window_lengths = []
                x_pred = None
                for model, mu, sigma, scaler, X_w, _, val_mae in lasso_models:
                    x_pred = self._build_prediction_row(
                        X_full, y_full, target_date, hour, horizon,
                    )
                    if x_pred is not None:
                        x_arr = np.nan_to_num(x_pred.reshape(1, -1), nan=0.0)
                        x_scaled = scaler.transform(x_arr)
                        pred_t = model.predict(x_scaled)[0]
                        preds.append(_asinh_inverse(pred_t, mu, sigma))
                        pred_maes.append(val_mae)
                        pred_window_lengths.append(len(X_w))

                if not preds:
                    continue

                raw_forecast = self._weighted_model_average(
                    preds,
                    pred_maes,
                    window_lengths=pred_window_lengths,
                    prefer_short_windows=self.use_regime_short_window_weighting and rp.get("prefer_short_windows", False),
                )
                forecast = self._recalibrate_variance(
                    raw_forecast,
                    hour,
                    lasso_models,
                    X_full,
                    y_full,
                    horizon,
                )
                horizon_bias_weight = max(0.25, 1.0 - 0.06 * (horizon - 1))
                forecast = forecast - horizon_bias_weight * recent_bias
                regime_bias = weekend_bias if pd.Timestamp(target_date).dayofweek >= 5 else weekday_bias
                forecast = forecast - 0.70 * horizon_bias_weight * regime_bias

                # GBM ensemble for peak hours (reuse pre-trained model)
                if hour in gbm_cache and x_pred is not None:
                    gbm_model, w_gbm, gbm_feature_names = gbm_cache[hour]
                    x_gbm = np.nan_to_num(x_pred.reshape(1, -1), nan=0.0)
                    x_gbm_df = pd.DataFrame(x_gbm, columns=gbm_feature_names)
                    gbm_pred = float(gbm_model.predict(x_gbm_df)[0])
                    forecast = (1.0 - w_gbm) * forecast + w_gbm * gbm_pred

                forecast = self._apply_spike_uplift(forecast, y_full, horizon)

                # AR error correction with regime-adaptive coefficients
                if hour in prev_day_errors:
                    prev_err = prev_day_errors[hour]
                    if np.isfinite(prev_err):
                        ar1 = max(0.25, rp["ar1_coeff"] - 0.05 * (horizon - 1))
                        correction = ar1 * np.clip(prev_err, -100, 100)
                        forecast = forecast - correction
                if hour in prev2_day_errors:
                    prev2_err = prev2_day_errors[hour]
                    if np.isfinite(prev2_err):
                        ar2 = max(0.05, 0.15 - 0.02 * (horizon - 1))
                        correction2 = ar2 * np.clip(prev2_err, -100, 100)
                        forecast = forecast - correction2
                # Lag-7: same-day-of-week error (strong weekly autocorrelation)
                if len(error_history[hour]) >= 7:
                    lag7_err = error_history[hour][-7]
                    if np.isfinite(lag7_err):
                        ar7 = max(0.03, 0.10 - 0.01 * (horizon - 1))
                        forecast = forecast - ar7 * np.clip(lag7_err, -100, 100)

                # NOTE: Expanding-window bias correction REMOVED (was double-counting
                # with recent_bias from _recent_bias_by_regime + AR correction).
                # Three auditors confirmed this causes over-correction.

                # Clamp forecast to physical bounds
                forecast = float(np.clip(forecast, -500, 1000))

                # Foundation model blending AFTER all LEAR post-processing
                # (consistent with predict() order of operations)
                fm_key = str(test_date)
                if fm_key in fm_daily_cache:
                    fm_data = fm_daily_cache[fm_key]
                    fm_idx = (horizon - 1) * 24 + hour
                    if fm_idx < len(fm_data["median"]):
                        fm_price = float(fm_data["median"][fm_idx])
                        if np.isfinite(fm_price):
                            w_fm = self._foundation_blend_weight(horizon, hour, rp)
                            if w_fm > 0.0:
                                forecast = (1.0 - w_fm) * forecast + w_fm * fm_price

                actual = complete.loc[target_date, hour] if hour in complete.columns else np.nan

                if not np.isnan(actual):
                    error = forecast - float(actual)
                    day_errors[hour] = error
                    error_history[hour].append(error)

                    results.append({
                        "date": str(target_date),
                        "hour": hour,
                        "forecast": forecast,
                        "actual": float(actual),
                        "error": error,
                        "horizon": horizon,
                    })

            # Update prev_day_errors for next iteration
            prev2_day_errors = prev_day_errors.copy()
            prev_day_errors = day_errors

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
        """Blend LEAR short-term forecast with PFC structural model."""
        result = pfc_15min.copy()

        if "price_shape" not in result.columns:
            logger.warning("No price_shape column in PFC — skipping blend")
            return result

        lear_ts = lear_forecast.set_index("timestamp")["price_lear"]
        first_date = pd.to_datetime(lear_forecast["date"].min())
        lear_days = lear_forecast.set_index("timestamp")["date"].apply(
            lambda d: (pd.to_datetime(d) - first_date).days + 1
        )

        pfc_hours = result.index.floor("h")

        has_lear = pfc_hours.isin(lear_ts.index)
        if not has_lear.any():
            logger.warning("No timestamp overlap between LEAR and PFC — skipping blend")
            return result

        result["_hour"] = pfc_hours
        hourly_pfc_mean = result.loc[has_lear].groupby("_hour")["price_shape"].transform("mean")

        lear_mapped = lear_ts.reindex(pfc_hours[has_lear]).values
        days_mapped = lear_days.reindex(pfc_hours[has_lear]).values

        pfc_prices = result.loc[has_lear, "price_shape"].values
        hourly_means = hourly_pfc_mean.values

        new_prices = pfc_prices.copy()
        for i in range(len(new_prices)):
            day = days_mapped[i]
            lear_p = lear_mapped[i]
            pfc_p = pfc_prices[i]
            h_mean = hourly_means[i]

            if day < blend_start_day:
                if h_mean > 0:
                    new_prices[i] = pfc_p * (lear_p / h_mean)
                else:
                    new_prices[i] = lear_p
            elif day < blend_end_day:
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

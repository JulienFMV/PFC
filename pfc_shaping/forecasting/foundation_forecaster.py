"""
foundation_forecaster.py
------------------------
Zero-shot time-series foundation model wrapper for electricity price forecasting.

Supports two backends (auto-detected):
- Chronos-2 (v2.0+): native covariates (DE prices, load, wind, solar), quantiles
- Chronos-Bolt (v1.x): univariate only, fallback

Soft dependency: if torch/chronos are not installed, all methods gracefully
return None and LEAR operates standalone.

Install:
    pip install "chronos-forecasting>=2.0" torch   # for Chronos-2
    pip install chronos-forecasting torch            # for Chronos-Bolt fallback
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Soft imports — detect which Chronos version is available
_CHRONOS2_AVAILABLE = False
_CHRONOS_BOLT_AVAILABLE = False

try:
    import torch

    try:
        from chronos import Chronos2Pipeline
        _CHRONOS2_AVAILABLE = True
        logger.debug("Chronos-2 available")
    except ImportError:
        pass

    if not _CHRONOS2_AVAILABLE:
        try:
            from chronos import ChronosBoltPipeline
            _CHRONOS_BOLT_AVAILABLE = True
            logger.debug("Chronos-Bolt available (v1.x fallback)")
        except ImportError:
            pass
except ImportError:
    pass

_BOLT_MAX_PREDICTION_LENGTH = 64


class FoundationForecaster:
    """Zero-shot foundation model for electricity price forecasting.

    Auto-selects the best available backend:
    1. Chronos-2 (preferred): supports covariates + native quantiles
    2. Chronos-Bolt (fallback): univariate only

    The model is loaded lazily on first use and cached for the session.
    """

    _LOCAL_FINETUNED_CHRONOS2 = Path(__file__).resolve().parent / "chronos2_finetuned"
    _LOCAL_CHRONOS2 = Path(__file__).resolve().parents[2] / "models" / "chronos-2"
    if _LOCAL_FINETUNED_CHRONOS2.exists():
        CHRONOS2_MODEL = str(_LOCAL_FINETUNED_CHRONOS2)
    elif _LOCAL_CHRONOS2.exists():
        CHRONOS2_MODEL = str(_LOCAL_CHRONOS2)
    else:
        CHRONOS2_MODEL = "amazon/chronos-2"
    BOLT_MODEL = "amazon/chronos-bolt-base"
    _BOLT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(self, device: str | None = None):
        if device is None:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self._device = device
        self._pipeline = None
        self._backend = None  # "chronos2" or "bolt"
        self._loaded = False

    @property
    def available(self) -> bool:
        return _CHRONOS2_AVAILABLE or _CHRONOS_BOLT_AVAILABLE

    @property
    def backend(self) -> str | None:
        """Which backend is active: 'chronos2', 'bolt', or None."""
        if not self._loaded:
            self._ensure_loaded()
        return self._backend

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return self._pipeline is not None
        self._loaded = True

        if _CHRONOS2_AVAILABLE:
            try:
                logger.info("Loading Chronos-2: %s", self.CHRONOS2_MODEL)
                self._pipeline = Chronos2Pipeline.from_pretrained(
                    self.CHRONOS2_MODEL,
                    device_map=self._device,
                    dtype=torch.float32,
                )
                self._backend = "chronos2"
                logger.info("Chronos-2 loaded (covariates supported)")
                return True
            except Exception:
                logger.warning("Chronos-2 load failed, trying Bolt", exc_info=True)

        if _CHRONOS_BOLT_AVAILABLE:
            try:
                logger.info("Loading Chronos-Bolt: %s", self.BOLT_MODEL)
                self._pipeline = ChronosBoltPipeline.from_pretrained(
                    self.BOLT_MODEL,
                    device_map=self._device,
                    torch_dtype=torch.float32,
                )
                self._backend = "bolt"
                logger.info("Chronos-Bolt loaded (univariate only)")
                return True
            except Exception:
                logger.warning("Chronos-Bolt load failed", exc_info=True)

        logger.debug("No foundation model available")
        return False

    def _prepare_series(self, s: pd.Series | np.ndarray) -> np.ndarray:
        """Clean a series: interpolate NaN, return float32."""
        if isinstance(s, pd.Series):
            values = (
                s.interpolate(method="linear", limit_direction="both")
                .ffill().bfill()
                .values.astype(np.float32)
            )
        else:
            arr = np.asarray(s, dtype=np.float32)
            nans = np.isnan(arr)
            if nans.any():
                x = np.arange(len(arr))
                arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
            values = arr
        return values

    def forecast(
        self,
        price_history: pd.Series | np.ndarray,
        horizon: int = 24,
        covariates: dict[str, pd.Series] | None = None,
    ) -> Optional[dict[str, np.ndarray]]:
        """Generate zero-shot probabilistic forecast.

        Args:
            price_history: Historical hourly CH prices (>= 48 hours).
            horizon: Number of hours to forecast.
            covariates: Optional dict of covariate series (e.g. DE prices, load).
                        Only used by Chronos-2. Ignored by Bolt.

        Returns:
            dict with keys 'median', 'q10', 'q90', 'mean' or None.
        """
        if not self._ensure_loaded():
            return None

        if self._backend == "chronos2":
            return self._forecast_chronos2(price_history, horizon, covariates)
        else:
            return self._forecast_bolt(price_history, horizon)

    def _forecast_chronos2(
        self,
        price_history: pd.Series | np.ndarray,
        horizon: int,
        covariates: dict[str, pd.Series] | None = None,
    ) -> Optional[dict[str, np.ndarray]]:
        """Forecast using Chronos-2 with DataFrame API (supports covariates)."""
        try:
            # Build DataFrame with target + covariates
            prices = self._prepare_series(price_history)
            if len(prices) < 48:
                return None

            n = len(prices)
            # Use synthetic regular timestamps — real timestamps have DST gaps
            # that cause Chronos-2 to fail with "Could not infer frequency"
            ts = pd.date_range(end="2025-01-01", periods=n, freq="h")
            df = pd.DataFrame({
                "timestamp": ts,
                "item_id": "ch_price",
                "target": prices,
            })

            # Add past-only covariates (aligned with price history)
            if covariates:
                for name, series in covariates.items():
                    vals = self._prepare_series(series)
                    # Align length: take the last n values
                    if len(vals) >= n:
                        df[name] = vals[-n:]
                    else:
                        # Pad with NaN at start, then fill
                        padded = np.full(n, np.nan)
                        padded[-len(vals):] = vals
                        df[name] = pd.Series(padded).ffill().bfill().values

            result_df = self._pipeline.predict_df(
                df,
                target="target",
                timestamp_column="timestamp",
                id_column="item_id",
                prediction_length=horizon,
                quantile_levels=[0.1, 0.5, 0.9],
            )

            # Extract quantiles from result DataFrame
            q10_col = [c for c in result_df.columns if str(c) == "0.1"]
            q50_col = [c for c in result_df.columns if str(c) == "0.5"]
            q90_col = [c for c in result_df.columns if str(c) == "0.9"]

            if not q50_col:
                logger.warning("Chronos-2 returned unexpected columns: %s", result_df.columns.tolist())
                return None

            median = result_df[q50_col[0]].values[:horizon].astype(np.float64)
            q10 = result_df[q10_col[0]].values[:horizon].astype(np.float64) if q10_col else median - 15
            q90 = result_df[q90_col[0]].values[:horizon].astype(np.float64) if q90_col else median + 15

            if not np.all(np.isfinite(median)):
                n_bad = (~np.isfinite(median)).sum()
                logger.warning("Chronos-2 returned %d non-finite values", n_bad)
                return None

            n_cov = len(covariates) if covariates else 0
            logger.info("Chronos-2 forecast: %d hours, %d covariates", horizon, n_cov)

            return {
                "q10": q10, "median": median, "q90": q90,
                "mean": median,
            }

        except Exception:
            logger.warning("Chronos-2 forecast failed", exc_info=True)
            return None

    def _forecast_bolt(
        self,
        price_history: pd.Series | np.ndarray,
        horizon: int,
    ) -> Optional[dict[str, np.ndarray]]:
        """Forecast using Chronos-Bolt (univariate).

        Uses Bolt's built-in chunking for long horizons (no custom logic needed).
        """
        values = self._prepare_series(price_history)
        if len(values) < 48:
            return None

        try:
            context = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
            # Let Bolt handle chunking internally (it preserves quantile
            # calibration across chunks, unlike our previous custom logic)
            forecast = self._pipeline.predict(
                context, prediction_length=horizon
            )
            # forecast shape: (1, num_quantiles, horizon)
            q_np = forecast[0].numpy()
            n_q = q_np.shape[0]

            if not np.all(np.isfinite(q_np)):
                n_bad = (~np.isfinite(q_np)).sum()
                logger.warning("Chronos-Bolt returned %d non-finite values", n_bad)
                return None

            # Dynamically pick quantile indices (default 9: [0.1..0.9])
            idx_q10 = 0
            idx_q50 = n_q // 2
            idx_q90 = n_q - 1
            return {
                "q10": q_np[idx_q10], "median": q_np[idx_q50],
                "q90": q_np[idx_q90], "mean": q_np[idx_q50],
            }
        except Exception:
            logger.warning("Chronos-Bolt forecast failed", exc_info=True)
            return None

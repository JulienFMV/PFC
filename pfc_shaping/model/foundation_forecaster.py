"""
foundation_forecaster.py
------------------------
Zero-shot time-series foundation model wrapper for electricity price forecasting.

Provides an optional ensemble member alongside LEAR. Uses pretrained models
(Chronos-Bolt, TimesFM, etc.) that require no training — they generate forecasts
purely from price history, capturing nonlinear temporal patterns that LASSO misses.

Soft dependency: if torch/chronos are not installed, all methods gracefully
return None and LEAR operates standalone.

Install:
    pip install torch chronos-forecasting

Usage:
    from pfc_shaping.model.foundation_forecaster import FoundationForecaster
    fm = FoundationForecaster()
    if fm.available:
        preds = fm.forecast(price_history, horizon=24)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Soft imports — foundation models are optional
_CHRONOS_AVAILABLE = False

try:
    import torch
    from chronos import ChronosBoltPipeline

    _CHRONOS_AVAILABLE = True
except ImportError:
    pass

# Chronos-Bolt native prediction length before variance collapse
_BOLT_MAX_PREDICTION_LENGTH = 64


class FoundationForecaster:
    """Zero-shot foundation model for electricity price forecasting.

    Currently supports:
    - Chronos-Bolt (Amazon): fast, CPU-friendly, probabilistic
    - More backends can be added (Timer-XL, TimesFM, Toto, TiRex)

    The model is loaded lazily on first use and cached for the session.

    IMPORTANT: Chronos-Bolt returns pre-computed quantile predictions at
    fixed levels [0.1, 0.2, ..., 0.9], NOT samples. Shape is
    (batch, 9, prediction_length). We index directly into the quantile
    dimension rather than computing quantiles of quantiles.
    """

    DEFAULT_MODEL = "amazon/chronos-bolt-base"
    # Chronos-Bolt quantile levels (fixed by the model architecture)
    _BOLT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cpu",
    ):
        self._model_id = model_id or self.DEFAULT_MODEL
        self._device = device
        self._pipeline = None
        self._loaded = False

    @property
    def available(self) -> bool:
        """Check if foundation model dependencies are installed."""
        return _CHRONOS_AVAILABLE

    def _ensure_loaded(self) -> bool:
        """Lazy-load the model on first use."""
        if self._loaded:
            return self._pipeline is not None

        self._loaded = True

        if not _CHRONOS_AVAILABLE:
            logger.debug("Foundation model unavailable: chronos not installed")
            return False

        try:
            logger.info("Loading foundation model: %s", self._model_id)
            self._pipeline = ChronosBoltPipeline.from_pretrained(
                self._model_id,
                device_map=self._device,
                torch_dtype=torch.float32,
            )
            logger.info("Foundation model loaded successfully")
            return True
        except Exception:
            logger.warning("Failed to load foundation model", exc_info=True)
            self._pipeline = None
            return False

    def _prepare_context(
        self, price_history: pd.Series | np.ndarray
    ) -> Optional[np.ndarray]:
        """Convert price history to clean float32 array, preserving temporal continuity."""
        if isinstance(price_history, pd.Series):
            # Interpolate interior NaNs to preserve temporal continuity,
            # then forward/backward fill edges
            values = (
                price_history
                .interpolate(method="linear", limit_direction="both")
                .ffill()
                .bfill()
                .values.astype(np.float32)
            )
        else:
            arr = np.asarray(price_history, dtype=np.float32)
            # Replace NaN with linear interpolation
            nans = np.isnan(arr)
            if nans.any():
                x = np.arange(len(arr))
                arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
            values = arr

        if len(values) < 48:
            logger.warning(
                "Foundation model needs >= 48 hours of history, got %d", len(values)
            )
            return None

        return values

    def _predict_chunked(
        self, context: "torch.Tensor", horizon: int
    ) -> Optional["torch.Tensor"]:
        """Run Chronos-Bolt prediction, chunking if horizon > 64.

        Chronos-Bolt has a native prediction length of 64 steps. For longer
        horizons, we iterate: predict 64 steps, append the median to context,
        and predict the next 64 steps, etc.

        Returns tensor of shape (9, horizon) — 9 quantile levels.
        """
        chunks = []
        remaining = horizon
        ctx = context  # (1, T)

        while remaining > 0:
            chunk_len = min(remaining, _BOLT_MAX_PREDICTION_LENGTH)

            forecast_chunk = self._pipeline.predict(
                ctx, prediction_length=chunk_len
            )
            # forecast_chunk shape: (1, 9, chunk_len)
            chunks.append(forecast_chunk[0])  # (9, chunk_len)

            if remaining > chunk_len:
                # Extend context with median (index 4 = q50) for next chunk
                median_chunk = forecast_chunk[0, 4:5, :]  # (1, chunk_len)
                ctx = torch.cat([ctx, median_chunk], dim=-1)

            remaining -= chunk_len

        # Concatenate all chunks along the horizon dimension
        return torch.cat(chunks, dim=-1)  # (9, horizon)

    def forecast(
        self,
        price_history: pd.Series | np.ndarray,
        horizon: int = 24,
    ) -> Optional[dict[str, np.ndarray]]:
        """Generate zero-shot probabilistic forecast from price history.

        Args:
            price_history: Historical hourly prices (>= 48 hours).
            horizon: Number of hours to forecast.

        Returns:
            dict with keys 'median', 'mean', 'q10', 'q90' (or None if unavailable).
            Each value is a np.ndarray of shape (horizon,).
        """
        if not self._ensure_loaded():
            return None

        values = self._prepare_context(price_history)
        if values is None:
            return None

        try:
            context = torch.tensor(values, dtype=torch.float32).unsqueeze(0)

            # Chronos-Bolt returns (batch, 9_quantiles, horizon)
            # Quantile levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            quantile_preds = self._predict_chunked(context, horizon)
            # quantile_preds shape: (9, horizon)

            q_np = quantile_preds.numpy()

            # Validate: check for NaN/Inf
            if not np.all(np.isfinite(q_np)):
                n_bad = (~np.isfinite(q_np)).sum()
                logger.warning(
                    "Foundation model returned %d non-finite values, skipping", n_bad
                )
                return None

            result = {
                "q10": q_np[0],       # index 0 = 10th percentile
                "q20": q_np[1],       # index 1 = 20th percentile
                "median": q_np[4],    # index 4 = 50th percentile
                "q80": q_np[7],       # index 7 = 80th percentile
                "q90": q_np[8],       # index 8 = 90th percentile
                "mean": q_np[4],      # best proxy without samples
            }
            return result

        except Exception:
            logger.warning("Foundation model forecast failed", exc_info=True)
            return None

    def forecast_multi_horizon(
        self,
        price_history: pd.Series | np.ndarray,
        max_days: int = 10,
    ) -> Optional[pd.DataFrame]:
        """Generate multi-day hourly forecasts.

        Returns DataFrame with columns: day, hour, fm_price, fm_q10, fm_q90.
        """
        result = self.forecast(price_history, horizon=24 * max_days)
        if result is None:
            return None

        rows = []
        for i in range(24 * max_days):
            rows.append({
                "day": i // 24 + 1,
                "hour": i % 24,
                "fm_price": float(result["median"][i]),
                "fm_q10": float(result["q10"][i]),
                "fm_q90": float(result["q90"][i]),
            })

        return pd.DataFrame(rows)

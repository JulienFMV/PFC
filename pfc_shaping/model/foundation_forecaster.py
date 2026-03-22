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
_CHRONOS_PIPELINE = None

try:
    import torch
    from chronos import ChronosBoltPipeline

    _CHRONOS_AVAILABLE = True
except ImportError:
    pass


class FoundationForecaster:
    """Zero-shot foundation model for electricity price forecasting.

    Currently supports:
    - Chronos-Bolt (Amazon): fast, CPU-friendly, probabilistic
    - More backends can be added (Timer-XL, TimesFM, Toto, TiRex)

    The model is loaded lazily on first use and cached for the session.
    """

    # Model ID on HuggingFace — Bolt is 10x faster than base Chronos
    DEFAULT_MODEL = "amazon/chronos-bolt-base"

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
        except Exception as exc:
            logger.warning("Failed to load foundation model: %s", exc)
            self._pipeline = None
            return False

    def forecast(
        self,
        price_history: pd.Series | np.ndarray,
        horizon: int = 24,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> Optional[dict[str, np.ndarray]]:
        """Generate zero-shot probabilistic forecast from price history.

        Args:
            price_history: Historical hourly prices (at least 168 hours recommended).
            horizon: Number of hours to forecast (default: 24 = one day).
            quantiles: Quantile levels for prediction intervals.

        Returns:
            dict with keys 'median', 'mean', 'q10', 'q90' (or None if unavailable).
            Each value is a np.ndarray of shape (horizon,).
        """
        if not self._ensure_loaded():
            return None

        # Convert to tensor
        if isinstance(price_history, pd.Series):
            values = price_history.dropna().values.astype(np.float32)
        else:
            values = np.asarray(price_history, dtype=np.float32)

        if len(values) < 48:
            logger.warning("Foundation model needs >= 48 hours of history, got %d", len(values))
            return None

        try:
            context = torch.tensor(values).unsqueeze(0)  # (1, T)

            # Chronos-Bolt returns (batch, num_samples, horizon) for quantiles
            forecast = self._pipeline.predict(
                context,
                prediction_length=horizon,
            )
            # forecast shape: (1, num_samples, horizon)

            result = {}
            for q in quantiles:
                q_key = f"q{int(q * 100):02d}"
                q_val = np.quantile(forecast[0].numpy(), q, axis=0)
                result[q_key] = q_val

            result["median"] = np.quantile(forecast[0].numpy(), 0.5, axis=0)
            result["mean"] = forecast[0].numpy().mean(axis=0)

            return result

        except Exception as exc:
            logger.warning("Foundation model forecast failed: %s", exc)
            return None

    def forecast_multi_horizon(
        self,
        price_history: pd.Series | np.ndarray,
        max_days: int = 10,
    ) -> Optional[pd.DataFrame]:
        """Generate multi-day hourly forecasts.

        Returns DataFrame with columns: day, hour, fm_price, fm_q10, fm_q90.
        """
        result = self.forecast(
            price_history,
            horizon=24 * max_days,
            quantiles=(0.1, 0.5, 0.9),
        )
        if result is None:
            return None

        rows = []
        for i in range(24 * max_days):
            day = i // 24 + 1
            hour = i % 24
            rows.append({
                "day": day,
                "hour": hour,
                "fm_price": float(result["median"][i]),
                "fm_q10": float(result["q10"][i]),
                "fm_q90": float(result["q90"][i]),
            })

        return pd.DataFrame(rows)

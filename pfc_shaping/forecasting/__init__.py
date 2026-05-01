"""
pfc_shaping.forecasting — short-term DA / IDA price forecasters.

Production stack (D+1..D+10):

    LEARForecaster        — LASSO multi-window + MLP + QRA + Conformal +
                            AR-bias correction. Optionally ensembles with
                            Chronos-2 if torch + chronos-forecasting are
                            available (default ON, soft-fails to standalone).

    FoundationForecaster  — wrapper around Chronos-2 / Chronos-Bolt zero-shot
                            time-series foundation models.

    chronos2_finetuned/   — local LoRA adapter (rank=8) for Chronos-2,
                            fine-tuned on Swiss/EU price data.

Experimental challengers (OFF by default in production, gated by env vars):

    PriceFM (graph-aware)     blend_lear_with_pricefm — gated by
                              ``PFC_ENABLE_PRICEFM_EXPERIMENT=1``
    FutureBoost meta-corrector apply_futureboost_experimental — gated by
                              the same env switch

Soft dependencies — install with:
    pip install torch chronos-forecasting peft transformers
to activate the Chronos-2 ensemble member. Without these, LEAR runs
standalone and gracefully reports the missing backend.
"""

from pfc_shaping.forecasting.lear_forecaster import LEARForecaster
from pfc_shaping.forecasting.foundation_forecaster import FoundationForecaster
from pfc_shaping.forecasting.pricefm_experimental import (
    BEST_PRICEFM_EXPERIMENT,
    PriceFMExperimentalConfig,
    blend_lear_with_pricefm,
    load_pricefm_forecast,
    compute_pricefm_regime_weight,
)
from pfc_shaping.forecasting.futureboost_experimental import (
    DEFAULT_FUTUREBOOST_EXPERIMENT,
    FutureBoostExperimentalConfig,
    apply_futureboost_experimental,
)

__all__ = [
    "LEARForecaster",
    "FoundationForecaster",
    "BEST_PRICEFM_EXPERIMENT",
    "PriceFMExperimentalConfig",
    "blend_lear_with_pricefm",
    "load_pricefm_forecast",
    "compute_pricefm_regime_weight",
    "DEFAULT_FUTUREBOOST_EXPERIMENT",
    "FutureBoostExperimentalConfig",
    "apply_futureboost_experimental",
]

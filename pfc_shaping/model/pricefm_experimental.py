"""Backward-compat shim. New code: ``from pfc_shaping.forecasting.pricefm_experimental import *``"""
from pfc_shaping.forecasting.pricefm_experimental import *  # noqa: F401,F403
from pfc_shaping.forecasting.pricefm_experimental import (  # noqa: F401
    BEST_PRICEFM_EXPERIMENT, PriceFMExperimentalConfig,
    blend_lear_with_pricefm, load_pricefm_forecast,
    compute_pricefm_regime_weight,
)

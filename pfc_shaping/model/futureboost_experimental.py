"""Backward-compat shim. New code: ``from pfc_shaping.forecasting.futureboost_experimental import *``"""
from pfc_shaping.forecasting.futureboost_experimental import *  # noqa: F401,F403
from pfc_shaping.forecasting.futureboost_experimental import (  # noqa: F401
    DEFAULT_FUTUREBOOST_EXPERIMENT, FutureBoostExperimentalConfig,
    apply_futureboost_experimental,
)

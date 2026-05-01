"""Backward-compat shim. New code: ``from pfc_shaping.forecasting.lear_forecaster import *``"""
from pfc_shaping.forecasting.lear_forecaster import *  # noqa: F401,F403
from pfc_shaping.forecasting.lear_forecaster import LEARForecaster  # noqa: F401
try:  # pragma: no cover - private helper used by some scripts
    from pfc_shaping.forecasting.lear_forecaster import _get_holidays  # noqa: F401
except ImportError:
    pass

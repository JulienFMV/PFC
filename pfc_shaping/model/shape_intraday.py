"""Backward-compat shim. New code: ``from pfc_shaping.shaping.shape_intraday import *``"""
from pfc_shaping.shaping.shape_intraday import *  # noqa: F401,F403
from pfc_shaping.shaping.shape_intraday import (  # noqa: F401
    ShapeIntraday, MIN_OBS_COUCHE1, MIN_OBS_COUCHE2, MIN_R2_OOS,
    MAX_COEF_ABS, RIDGE_ALPHAS, SAISONS, TYPES_JOUR, HEURES_RAMPE,
)

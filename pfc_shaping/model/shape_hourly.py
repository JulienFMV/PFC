"""Backward-compat shim. New code: ``from pfc_shaping.shaping.shape_hourly import *``"""
from pfc_shaping.shaping.shape_hourly import *  # noqa: F401,F403
from pfc_shaping.shaping.shape_hourly import (  # noqa: F401
    ShapeHourly, GAUSSIAN_SIGMA, SAISONS, TYPES_JOUR,
)

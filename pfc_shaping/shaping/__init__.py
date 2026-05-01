"""
pfc_shaping.shaping — long-term Price Forward Curve structural model.

Multiplicative decomposition over the M+1..Y+3 horizon:

    P(t) = B(year/quarter/month) × f_S × f_W × f_H × f_Q × f_WV

with arbitrage-free Maximum Smoothness calibration on top.

Public API:
    PFCAssembler           — main assembler
    ShapeHourly            — f_H + f_W estimator
    ShapeIntraday          — f_Q estimator (15-min intra-hour)
    Uncertainty            — bootstrap p10/p90 intervals
    WaterValueCorrection   — f_WV (CH hydro reservoir)
    smooth_base_prices     — MSFC spline smoothing for B(t)
"""

from pfc_shaping.shaping.assembler import PFCAssembler, HORIZON_DAYS
from pfc_shaping.shaping.shape_hourly import (
    ShapeHourly, GAUSSIAN_SIGMA, SAISONS, TYPES_JOUR,
)
from pfc_shaping.shaping.shape_intraday import (
    ShapeIntraday, MIN_OBS_COUCHE1, MIN_OBS_COUCHE2,
    MIN_R2_OOS, MAX_COEF_ABS, RIDGE_ALPHAS, HEURES_RAMPE,
)
from pfc_shaping.shaping.uncertainty import Uncertainty
from pfc_shaping.shaping.water_value import WaterValueCorrection
from pfc_shaping.shaping.msfc_spline import smooth_base_prices

__all__ = [
    "PFCAssembler", "HORIZON_DAYS",
    "ShapeHourly", "GAUSSIAN_SIGMA", "SAISONS", "TYPES_JOUR",
    "ShapeIntraday", "MIN_OBS_COUCHE1", "MIN_OBS_COUCHE2",
    "MIN_R2_OOS", "MAX_COEF_ABS", "RIDGE_ALPHAS", "HEURES_RAMPE",
    "Uncertainty", "WaterValueCorrection", "smooth_base_prices",
]

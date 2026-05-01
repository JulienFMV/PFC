"""
pfc_shaping.model — DEPRECATED legacy namespace.

The model package was split into two clearly-scoped sub-packages:

    pfc_shaping.shaping       — long-term PFC structural model
                                (assembler, shape_hourly, shape_intraday,
                                 msfc_spline, uncertainty, water_value,
                                 pfc_flavors)
    pfc_shaping.forecasting   — short-term DA/IDA forecasters
                                (lear_forecaster, foundation_forecaster,
                                 pricefm_experimental, futureboost_experimental,
                                 chronos2_finetuned/)

This shim namespace is kept for backward compatibility with the **frozen**
evaluation harnesses (autoresearch_eval.py, autoresearch_eval_lear.py)
and any pickled artefacts that reference the old paths. New code MUST
import from ``pfc_shaping.shaping`` or ``pfc_shaping.forecasting`` directly.
"""

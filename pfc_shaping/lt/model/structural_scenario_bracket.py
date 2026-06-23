"""Structural scenario bracket helpers for LT local/test curves.

The slow/central/fast structural scenarios are governed scenario points, not a
calibrated probability distribution.  These helpers therefore compute an
ordered scenario bracket and only populate legacy p10/p50/p90 names as explicit
compatibility aliases.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


SCENARIO_PRICE_COLUMNS = {
    "slow": "price_slow_eur_mwh",
    "central": "price_central_eur_mwh",
    "fast": "price_fast_eur_mwh",
}

STRUCTURAL_SCENARIO_COLUMNS = [
    "structural_scenario_low_eur_mwh",
    "structural_scenario_central_eur_mwh",
    "structural_scenario_high_eur_mwh",
    "structural_scenario_spread_eur_mwh",
]

LEGACY_STRUCTURAL_ALIAS_COLUMNS = [
    "structural_p10_eur_mwh",
    "structural_p50_eur_mwh",
    "structural_p90_eur_mwh",
    "structural_width_eur_mwh",
]

PRICE_COLUMNS = [
    *SCENARIO_PRICE_COLUMNS.values(),
    "price_weighted_mean_eur_mwh",
    *STRUCTURAL_SCENARIO_COLUMNS,
    *LEGACY_STRUCTURAL_ALIAS_COLUMNS,
]


def recompute_structural_scenario_bracket(
    frame: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    include_legacy_aliases: bool = True,
) -> pd.DataFrame:
    """Recompute weighted mean and ordered structural scenario bracket.

    Legacy p10/p50/p90 columns, when requested, are compatibility aliases:
    p10=ordered low, p50=central scenario, p90=ordered high.  No probabilistic
    quantile is inferred from the three scenario points.
    """
    out = frame.copy()
    labels = tuple(SCENARIO_PRICE_COLUMNS)
    missing = [SCENARIO_PRICE_COLUMNS[label] for label in labels if SCENARIO_PRICE_COLUMNS[label] not in out.columns]
    if missing:
        raise ValueError(f"structural scenario bracket missing scenario columns: {missing}")
    matrix = np.column_stack([out[SCENARIO_PRICE_COLUMNS[label]].to_numpy(dtype=float) for label in labels])
    weight_vector = np.array([float(weights[label]) for label in labels], dtype=float)
    total = float(weight_vector.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("structural scenario weights must sum to a finite positive value")
    weight_vector = weight_vector / total

    out["price_weighted_mean_eur_mwh"] = matrix @ weight_vector
    out["structural_scenario_low_eur_mwh"] = np.nanmin(matrix, axis=1)
    out["structural_scenario_central_eur_mwh"] = out[SCENARIO_PRICE_COLUMNS["central"]].astype(float)
    out["structural_scenario_high_eur_mwh"] = np.nanmax(matrix, axis=1)
    out["structural_scenario_spread_eur_mwh"] = (
        out["structural_scenario_high_eur_mwh"] - out["structural_scenario_low_eur_mwh"]
    )
    if include_legacy_aliases:
        out["structural_p10_eur_mwh"] = out["structural_scenario_low_eur_mwh"]
        out["structural_p50_eur_mwh"] = out["structural_scenario_central_eur_mwh"]
        out["structural_p90_eur_mwh"] = out["structural_scenario_high_eur_mwh"]
        out["structural_width_eur_mwh"] = out["structural_scenario_spread_eur_mwh"]
    return out


def ensure_structural_scenario_bracket_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Populate bracket aliases, preferring canonical scenario columns."""
    out = frame.copy()
    mapping = {
        "structural_p10_eur_mwh": "structural_scenario_low_eur_mwh",
        "structural_p50_eur_mwh": "structural_scenario_central_eur_mwh",
        "structural_p90_eur_mwh": "structural_scenario_high_eur_mwh",
        "structural_width_eur_mwh": "structural_scenario_spread_eur_mwh",
    }
    for legacy, canonical in mapping.items():
        if canonical not in out.columns and legacy in out.columns:
            out[canonical] = out[legacy]
        if canonical in out.columns:
            out[legacy] = out[canonical]
    return out

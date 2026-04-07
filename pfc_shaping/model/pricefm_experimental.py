"""
Experimental PriceFM helpers for Swiss short-term research.

This module intentionally keeps PriceFM outside the default production path.
It freezes the current best research setup and exposes a lightweight blend
utility that can be enabled explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PriceFMExperimentalConfig:
    dataset_countries: tuple[str, ...] = ("AT", "CH", "DE_LU", "FR", "IT_NORD")
    graph_degree: int = 1
    epochs: int = 5
    blend_weight: float = 0.15
    dataset_path: str = "pfc_shaping/output/pricefm_ch_full_probe.csv"
    forecast_latest_path: str = "pfc_shaping/output/pricefm_forecast_latest.csv"


BEST_PRICEFM_EXPERIMENT = PriceFMExperimentalConfig()


def load_pricefm_forecast(path: str | Path) -> pd.DataFrame:
    """Load a PriceFM forecast file and normalize the expected columns."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("PriceFM forecast must contain a 'timestamp' column.")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if "pricefm" in df.columns:
        price_col = "pricefm"
    elif "price_pricefm" in df.columns:
        price_col = "price_pricefm"
    else:
        numeric_cols = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise ValueError("No numeric forecast column found in PriceFM forecast.")
        price_col = numeric_cols[0]

    return (
        df[["timestamp", price_col]]
        .rename(columns={price_col: "price_pricefm"})
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
    )


def blend_lear_with_pricefm(
    lear_forecast: pd.DataFrame,
    pricefm_forecast: pd.DataFrame,
    weight_pricefm: float = BEST_PRICEFM_EXPERIMENT.blend_weight,
) -> pd.DataFrame:
    """Blend a LEAR forecast dataframe with a timestamp-aligned PriceFM forecast."""
    if "timestamp" not in lear_forecast.columns or "price_lear" not in lear_forecast.columns:
        raise ValueError("LEAR forecast must contain 'timestamp' and 'price_lear'.")

    if not 0.0 <= weight_pricefm <= 1.0:
        raise ValueError("weight_pricefm must be between 0 and 1.")

    result = lear_forecast.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)

    pfm = pricefm_forecast.copy()
    if "price_pricefm" not in pfm.columns:
        pfm = load_pricefm_forecast(pfm)
    else:
        pfm["timestamp"] = pd.to_datetime(pfm["timestamp"], utc=True)

    merged = result.merge(pfm[["timestamp", "price_pricefm"]], on="timestamp", how="left")
    merged["price_lear_base"] = merged["price_lear"]

    has_pricefm = merged["price_pricefm"].notna()
    merged.loc[has_pricefm, "price_lear"] = (
        (1.0 - weight_pricefm) * merged.loc[has_pricefm, "price_lear_base"]
        + weight_pricefm * merged.loc[has_pricefm, "price_pricefm"]
    )
    merged["pricefm_weight"] = 0.0
    merged.loc[has_pricefm, "pricefm_weight"] = float(weight_pricefm)
    merged["pricefm_used"] = has_pricefm.astype(bool)
    return merged

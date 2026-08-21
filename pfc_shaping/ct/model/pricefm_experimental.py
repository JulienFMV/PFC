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
    weekday_offpeak_weight: float = 0.05
    peak_weight: float = 0.40
    weekend_weight: float = 0.40
    solar_midday_weight: float = 0.00
    dataset_path: str = "pfc_shaping/output/pricefm_ch_full_probe.csv"
    forecast_latest_path: str = "pfc_shaping/output/pricefm_forecast_latest.csv"


BEST_PRICEFM_EXPERIMENT = PriceFMExperimentalConfig()


def compute_pricefm_regime_weight(
    timestamps: pd.Series,
    config: PriceFMExperimentalConfig = BEST_PRICEFM_EXPERIMENT,
) -> pd.Series:
    """Simple Swiss market regime weights learned from the recent A/B campaign."""
    ts = pd.to_datetime(timestamps, utc=True)
    local = ts.dt.tz_convert("Europe/Zurich")
    hour = local.dt.hour
    weekend = local.dt.dayofweek >= 5
    peak = hour.isin([7, 8, 9, 17, 18, 19, 20])
    solar_midday = hour.between(10, 15)

    weights = pd.Series(config.weekday_offpeak_weight, index=ts.index, dtype=float)
    weights.loc[peak] = config.peak_weight
    weights.loc[weekend] = config.weekend_weight
    weights.loc[solar_midday] = config.solar_midday_weight
    return weights.clip(lower=0.0, upper=1.0)


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
    weight_pricefm: float | None = BEST_PRICEFM_EXPERIMENT.blend_weight,
) -> pd.DataFrame:
    """Blend a LEAR forecast dataframe with a timestamp-aligned PriceFM forecast."""
    if "timestamp" not in lear_forecast.columns or "price_lear" not in lear_forecast.columns:
        raise ValueError("LEAR forecast must contain 'timestamp' and 'price_lear'.")

    if weight_pricefm is not None and not 0.0 <= weight_pricefm <= 1.0:
        raise ValueError("weight_pricefm must be between 0 and 1.")

    result = lear_forecast.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    cleanup_cols = [
        "price_pricefm",
        "price_pricefm_x",
        "price_pricefm_y",
        "pricefm_weight",
        "pricefm_used",
        "price_lear_base",
    ]
    result = result.drop(columns=[c for c in cleanup_cols if c in result.columns], errors="ignore")

    pfm = pricefm_forecast.copy()
    if "price_pricefm" not in pfm.columns:
        pfm = load_pricefm_forecast(pfm)
    else:
        pfm["timestamp"] = pd.to_datetime(pfm["timestamp"], utc=True)

    merged = result.merge(pfm[["timestamp", "price_pricefm"]], on="timestamp", how="left")
    merged["price_lear_base"] = merged["price_lear"]

    has_pricefm = merged["price_pricefm"].notna()
    if weight_pricefm is None:
        merged["pricefm_weight"] = compute_pricefm_regime_weight(merged["timestamp"])
    else:
        merged["pricefm_weight"] = float(weight_pricefm)

    merged.loc[has_pricefm, "price_lear"] = (
        (1.0 - merged.loc[has_pricefm, "pricefm_weight"]) * merged.loc[has_pricefm, "price_lear_base"]
        + merged.loc[has_pricefm, "pricefm_weight"] * merged.loc[has_pricefm, "price_pricefm"]
    )
    merged.loc[~has_pricefm, "pricefm_weight"] = 0.0
    merged["pricefm_used"] = has_pricefm.astype(bool)
    return merged

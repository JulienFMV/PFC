"""
ingest_entso.py
---------------
Ingestion des données réseau et renewables depuis l'API ENTSO-E Transparency
(via entsoe-py) :
  - Charge réseau Swissgrid 15min
  - Production solaire 15min
  - Production éolienne 15min

Clé API : variable d'environnement ENTSOE_API_KEY (ou fichier .env à la racine).

Format de sortie canonique (Parquet local) :
    index : DatetimeIndex UTC freq='15min'
    colonnes :
        load_mw         — charge totale CH [MW]
        solar_mw        — production solaire CH [MW]
        wind_mw         — production éolienne CH [MW]
        solar_regime    — {0=Faible, 1=Moyen, 2=Fort} (tertiles mensuels)
        load_deviation  — écart normalisé vs moyenne mensuelle
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_PARQUET = Path(__file__).resolve().parent.parent / "data" / "entso_15min.parquet"

# Charger .env depuis la racine du repo
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

MAX_RETRIES = 3
BASE_DELAY = 5


def _get_client():
    """Crée un client ENTSO-E. Lève ValueError si pas de clé API."""
    from entsoe import EntsoePandasClient

    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise ValueError(
            "Clé API ENTSO-E non trouvée. "
            "Définir ENTSOE_API_KEY dans l'environnement ou dans .env"
        )
    return EntsoePandasClient(api_key=api_key)


def _retry(func, *args, max_retries: int = MAX_RETRIES, **kwargs):
    """Appel avec retry + backoff exponentiel."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = BASE_DELAY * (2 ** attempt)
            logger.warning(
                "ENTSO-E tentative %d/%d échouée (%s), retry dans %ds",
                attempt + 1, max_retries, e, delay,
            )
            time.sleep(delay)


def load_from_api(
    start: str,
    end: str,
    country_code: str = "CH",
) -> pd.DataFrame:
    """
    Charge load + renewables depuis l'API ENTSO-E pour une période donnée.

    Args:
        start / end  : 'YYYY-MM-DD'
        country_code : code zone ENTSO-E (défaut 'CH')

    Returns:
        DataFrame colonnes ['load_mw', 'solar_mw', 'wind_mw']
        index : DatetimeIndex UTC
    """
    client = _get_client()

    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")

    logger.info("ENTSO-E API load + generation : %s → %s (zone=%s)", start, end, country_code)

    # --- Load ---
    df_load_raw = _retry(client.query_load, country_code, ts_start, ts_end)
    if isinstance(df_load_raw, pd.DataFrame):
        # query_load peut retourner DataFrame avec colonnes Forecasted/Actual
        if "Actual Load" in df_load_raw.columns:
            df_load = df_load_raw[["Actual Load"]].rename(columns={"Actual Load": "load_mw"})
        else:
            # Prendre la dernière colonne comme load
            df_load = df_load_raw.iloc[:, -1].to_frame("load_mw")
    else:
        df_load = df_load_raw.to_frame("load_mw")

    # --- Generation par type ---
    df_gen_raw = _retry(client.query_generation, country_code, ts_start, ts_end)

    # Extraire solar et wind depuis les colonnes multi-level ou flat
    solar_mw = _extract_generation_column(df_gen_raw, "Solar")
    wind_mw = _extract_generation_column(df_gen_raw, "Wind Onshore") + \
              _extract_generation_column(df_gen_raw, "Wind Offshore")

    df_gen = pd.DataFrame({"solar_mw": solar_mw, "wind_mw": wind_mw})

    # --- Resample à 15min et joindre ---
    for df_ in [df_load, df_gen]:
        if df_.index.tz is None:
            df_.index = df_.index.tz_localize("UTC")

    # Resample si nécessaire (certaines séries sont horaires)
    df_load_15 = _resample_to_15min(df_load)
    df_gen_15 = _resample_to_15min(df_gen)

    df = df_load_15.join(df_gen_15, how="outer").sort_index()
    df[["solar_mw", "wind_mw"]] = df[["solar_mw", "wind_mw"]].fillna(0.0)

    logger.info(
        "ENTSO-E chargé : %d lignes, load [%.0f-%.0f] MW",
        len(df), df["load_mw"].min(), df["load_mw"].max(),
    )
    return df


def _extract_generation_column(df_gen: pd.DataFrame, fuel_type: str) -> pd.Series:
    """
    Extrait une colonne de génération par type de combustible.
    Gère les colonnes multi-level (Actual Aggregated, Actual Consumption)
    et les colonnes flat.
    """
    if df_gen.empty:
        return pd.Series(0.0, index=df_gen.index, dtype=float)

    # Multi-level columns: ('Solar', 'Actual Aggregated'), etc.
    if isinstance(df_gen.columns, pd.MultiIndex):
        matching = [col for col in df_gen.columns if col[0] == fuel_type]
        if not matching:
            return pd.Series(0.0, index=df_gen.index, dtype=float)
        # Préférer 'Actual Aggregated' sur 'Actual Consumption'
        for col in matching:
            if "Aggregated" in str(col[1]):
                return df_gen[col].fillna(0.0)
        return df_gen[matching[0]].fillna(0.0)

    # Flat columns
    matching = [c for c in df_gen.columns if fuel_type in c]
    if not matching:
        return pd.Series(0.0, index=df_gen.index, dtype=float)
    return df_gen[matching[0]].fillna(0.0)


def _resample_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si la fréquence est horaire (ou autre > 15min), forward-fill vers 15min.
    Si déjà 15min ou plus fin, retourner tel quel.
    """
    if len(df) < 2:
        return df

    median_delta = df.index.to_series().diff().median()
    if median_delta > pd.Timedelta(minutes=15):
        df = df.resample("15min").ffill()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit le DataFrame avec solar_regime et load_deviation.

    solar_regime :
        Tertiles mensuels sur solar_mw → 0=Faible, 1=Moyen, 2=Fort

    load_deviation :
        (load_mw - mean_mensuel) / std_mensuel
    """
    df = df.copy()

    def _solar_regime(x: pd.Series) -> pd.Series:
        q33, q66 = np.nanpercentile(x, [33, 66])
        return pd.cut(x, bins=[-np.inf, q33, q66, np.inf], labels=[0, 1, 2]).astype(float)

    df["solar_regime"] = df.groupby(df.index.to_period("M"))["solar_mw"].transform(_solar_regime)

    monthly_mean = df.groupby(df.index.to_period("M"))["load_mw"].transform("mean")
    monthly_std = df.groupby(df.index.to_period("M"))["load_mw"].transform("std")
    df["load_deviation"] = (df["load_mw"] - monthly_mean) / monthly_std.replace(0, np.nan)

    return df


def load_parquet(path: str | Path = DEFAULT_PARQUET) -> pd.DataFrame:
    """Charge le cache Parquet local."""
    return pd.read_parquet(path)


def fetch_and_cache(
    start: str,
    end: str,
    parquet_path: str | Path = DEFAULT_PARQUET,
    country_code: str = "CH",
) -> pd.DataFrame:
    """
    Télécharge depuis l'API ENTSO-E, fusionne avec le cache local et sauvegarde.
    Recalcule les features sur l'ensemble.

    Returns:
        DataFrame canonique complet mis à jour
    """
    new_raw = load_from_api(start, end, country_code)

    parquet_path = Path(parquet_path)
    if parquet_path.exists():
        existing = load_parquet(parquet_path)
        raw_cols = ["load_mw", "solar_mw", "wind_mw"]
        existing_raw = existing[[c for c in raw_cols if c in existing.columns]]
        combined = pd.concat([existing_raw, new_raw])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_raw

    combined = build_features(combined)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    logger.info("Cache ENTSO-E mis à jour : %s (%d lignes)", parquet_path, len(combined))
    return combined

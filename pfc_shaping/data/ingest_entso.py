"""
ingest_entso.py
---------------
Ingestion des donnÃ©es rÃ©seau et renewables depuis Databricks :
  - Charge rÃ©seau Swissgrid 15min
  - Production solaire 15min
  - Production Ã©olienne 15min

Les tables Databricks (config.yaml â†’ databricks.tables) attendues :
    swissgrid_load  : timestamp_utc TIMESTAMP, load_mw DOUBLE
    renewables      : timestamp_utc TIMESTAMP, solar_mw DOUBLE, wind_mw DOUBLE

Format de sortie canonique (Parquet local) :
    index : DatetimeIndex UTC freq='15min'
    colonnes :
        load_mw         â€” charge totale CH [MW]
        solar_mw        â€” production solaire CH [MW]
        wind_mw         â€” production Ã©olienne CH [MW]
        solar_regime    â€” {0=Faible, 1=Moyen, 2=Fort} (tertiles mensuels)
        load_deviation  â€” Ã©cart normalisÃ© vs moyenne mensuelle
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.databricks_client import query_to_df, table_fqn

logger = logging.getLogger(__name__)

DEFAULT_PARQUET = Path(__file__).parent.parent / "data" / "entso_15min.parquet"


def load_from_databricks(
    start: str,
    end: str,
    db_config: dict | None = None,
) -> pd.DataFrame:
    """
    Charge load + renewables depuis Databricks pour une pÃ©riode donnÃ©e.

    Args:
        start / end : 'YYYY-MM-DD'
        db_config   : config Databricks (si None, lit config.yaml)

    Returns:
        DataFrame colonnes ['load_mw', 'solar_mw', 'wind_mw']
        index : DatetimeIndex UTC
    """
    load_fqn = table_fqn("swissgrid_load", db_config)
    ren_fqn  = table_fqn("renewables", db_config)

    sql_load = f"""
        SELECT timestamp_utc, load_mw
        FROM {load_fqn}
        WHERE timestamp_utc >= '{start}'
          AND timestamp_utc <  '{end}'
        ORDER BY timestamp_utc
    """
    sql_ren = f"""
        SELECT timestamp_utc, solar_mw, wind_mw
        FROM {ren_fqn}
        WHERE timestamp_utc >= '{start}'
          AND timestamp_utc <  '{end}'
        ORDER BY timestamp_utc
    """

    logger.info("Swissgrid load + renewables Databricks : %s â†’ %s", start, end)
    df_load = query_to_df(sql_load, config=db_config)
    df_ren  = query_to_df(sql_ren,  config=db_config)

    for df_, label in [(df_load, "load"), (df_ren, "renewables")]:
        if df_.empty:
            logger.warning("Aucune donnÃ©e %s retournÃ©e pour %s â†’ %s", label, start, end)

    df_load["timestamp_utc"] = pd.to_datetime(df_load["timestamp_utc"], utc=True)
    df_load = df_load.set_index("timestamp_utc")

    df_ren["timestamp_utc"] = pd.to_datetime(df_ren["timestamp_utc"], utc=True)
    df_ren = df_ren.set_index("timestamp_utc")

    df = df_load.join(df_ren, how="outer").sort_index()
    df[["solar_mw", "wind_mw"]] = df[["solar_mw", "wind_mw"]].fillna(0.0)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit le DataFrame avec solar_regime et load_deviation.

    solar_regime :
        Tertiles mensuels sur solar_mw â†’ 0=Faible, 1=Moyen, 2=Fort
        (permet au modÃ¨le de shape de capturer l'impact du solaire sur les
        prix intra-horaires en heures de midi)

    load_deviation :
        (load_mw - mean_mensuel_saisonnier) / std_mensuel_saisonnier
        Capte les dÃ©viations de charge qui modifient la pression tarifaire
        en particulier en heures de pointe (8h-9h, 18h-19h)
    """
    df = df.copy()

    # solar_regime par mois calendaire
    def _solar_regime(x: pd.Series) -> pd.Series:
        q33, q66 = np.nanpercentile(x, [33, 66])
        return pd.cut(x, bins=[-np.inf, q33, q66, np.inf], labels=[0, 1, 2]).astype(float)

    df["solar_regime"] = df.groupby(df.index.to_period("M"))["solar_mw"].transform(_solar_regime)

    # load_deviation normalisÃ©e (z-score mensuel)
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
    db_config: dict | None = None,
) -> pd.DataFrame:
    """
    TÃ©lÃ©charge depuis Databricks, fusionne avec le cache local et sauvegarde.
    Recalcule les features sur l'ensemble (les tertiles mensuels peuvent changer).

    Returns:
        DataFrame canonique complet mis Ã  jour
    """
    new_raw = load_from_databricks(start, end, db_config)

    parquet_path = Path(parquet_path)
    if parquet_path.exists():
        existing = load_parquet(parquet_path)
        raw_cols = ["load_mw", "solar_mw", "wind_mw"]
        existing_raw = existing[[c for c in raw_cols if c in existing.columns]]
        combined = pd.concat([existing_raw, new_raw])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_raw

    # Recalcul features sur l'ensemble complet
    combined = build_features(combined)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    logger.info("Cache ENTSO-E mis Ã  jour : %s (%d lignes)", parquet_path, len(combined))
    return combined

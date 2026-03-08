"""
ingest_epex.py
--------------
Ingestion des prix EPEX Spot 15 minutes depuis Databricks.

La table Databricks (config.yaml → databricks.tables.epex_15min) doit exposer :
    timestamp_utc   TIMESTAMP   — horodatage UTC du début du quart d'heure
    price_eur_mwh   DOUBLE      — prix en €/MWh
    area            STRING      — zone de prix ('CH', 'DE-AT', …)

Un cache Parquet local est maintenu pour limiter les requêtes répétées
et permettre le fonctionnement hors-ligne (backtest, dev).

Nettoyage appliqué :
  - Conservation des prix négatifs (information de marché)
  - Flagging des spikes > percentile 99.9 mensuel (sans suppression)
  - Alerte si trous > 15min détectés
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.databricks_client import query_to_df, table_fqn

logger = logging.getLogger(__name__)

DEFAULT_PARQUET = Path(__file__).parent.parent / "data" / "epex_15min.parquet"


def load_from_databricks(
    start: str,
    end: str,
    area: str = "CH",
    db_config: dict | None = None,
) -> pd.DataFrame:
    """
    Charge les prix EPEX 15min depuis Databricks pour une période donnée.

    Args:
        start    : date de début 'YYYY-MM-DD'
        end      : date de fin   'YYYY-MM-DD' (exclu)
        area     : zone de prix ('CH' ou 'DE-AT')
        db_config: config Databricks (si None, lit config.yaml)

    Returns:
        DataFrame canonique :
            index   : DatetimeIndex UTC
            colonnes: ['price_eur_mwh', 'spike_flag']
    """
    fqn = table_fqn("epex_15min", db_config)
    sql = f"""
        SELECT timestamp_utc, price_eur_mwh
        FROM {fqn}
        WHERE area          = '{area}'
          AND timestamp_utc >= '{start}'
          AND timestamp_utc <  '{end}'
        ORDER BY timestamp_utc
    """
    logger.info("EPEX Databricks : %s → %s (zone=%s)", start, end, area)
    raw = query_to_df(sql, config=db_config)

    if raw.empty:
        raise ValueError(f"Aucune donnée EPEX retournée pour {area} entre {start} et {end}")

    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True)
    raw = raw.set_index("timestamp_utc").sort_index()
    df = raw[["price_eur_mwh"]].copy()
    df = _clean(df)

    logger.info("EPEX chargé : %d lignes (%.1f%% négatifs)",
                len(df), (df["price_eur_mwh"] < 0).mean() * 100)
    return df


def load_parquet(path: str | Path = DEFAULT_PARQUET) -> pd.DataFrame:
    """Charge le cache Parquet local."""
    return pd.read_parquet(path)


def fetch_and_cache(
    start: str,
    end: str,
    area: str = "CH",
    parquet_path: str | Path = DEFAULT_PARQUET,
    db_config: dict | None = None,
) -> pd.DataFrame:
    """
    Télécharge depuis Databricks, fusionne avec le cache Parquet local
    (déduplique), et sauvegarde.

    Utilisé lors du cycle de mise à jour hebdomadaire.

    Returns:
        DataFrame complet mis à jour
    """
    new = load_from_databricks(start, end, area, db_config)

    parquet_path = Path(parquet_path)
    if parquet_path.exists():
        existing = load_parquet(parquet_path)
        combined = pd.concat([existing, new])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    logger.info("Cache EPEX mis à jour : %s (%d lignes)", parquet_path, len(combined))
    return combined


# ---------------------------------------------------------------------------
# Nettoyage interne
# ---------------------------------------------------------------------------

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Vérification complétude
    if len(df) > 1:
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq="15min", tz="UTC")
        missing = full_idx.difference(df.index)
        if len(missing) > 0:
            logger.warning("%d intervalles 15min manquants", len(missing))

    # Spike flag par mois (prix absolus — on garde les négatifs)
    monthly_p999 = df.groupby(df.index.to_period("M"))["price_eur_mwh"].transform(
        lambda x: np.nanpercentile(np.abs(x), 99.9)
    )
    df["spike_flag"] = np.abs(df["price_eur_mwh"]) > monthly_p999

    if df["spike_flag"].sum() > 0:
        logger.info("%d spikes extrêmes flaggés (conservés)", df["spike_flag"].sum())

    return df

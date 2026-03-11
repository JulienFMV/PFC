"""
ingest_epex.py
--------------
Ingestion des prix EPEX Spot 15 minutes depuis l'API ENTSO-E Transparency.

L'API ENTSO-E expose les prix day-ahead (résolution horaire ou 15min selon
la zone). Pour la Suisse (CH), la résolution native est horaire ; le module
effectue un forward-fill vers 15min pour homogénéité avec le reste du pipeline.

Clé API : variable d'environnement ENTSOE_API_KEY (ou fichier .env à la racine).

Cache Parquet local maintenu pour limiter les appels API et permettre le
fonctionnement hors-ligne.

Nettoyage appliqué :
  - Conservation des prix négatifs (information de marché)
  - Flagging des spikes > percentile 99.9 mensuel (sans suppression)
  - Alerte si trous > 15min détectés
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

DEFAULT_PARQUET = Path(__file__).resolve().parent.parent / "data" / "epex_15min.parquet"

_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

MAX_RETRIES = 3
BASE_DELAY = 5


def _get_client():
    """Crée un client ENTSO-E."""
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
    Charge les prix day-ahead depuis l'API ENTSO-E.

    Args:
        start / end    : 'YYYY-MM-DD'
        country_code   : zone de prix ('CH', 'DE_LU', 'FR', …)

    Returns:
        DataFrame canonique :
            index   : DatetimeIndex UTC freq='15min'
            colonnes: ['price_eur_mwh', 'spike_flag']
    """
    client = _get_client()

    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")

    logger.info("ENTSO-E API prix DA : %s → %s (zone=%s)", start, end, country_code)

    prices = _retry(client.query_day_ahead_prices, country_code, ts_start, ts_end)

    # query_day_ahead_prices retourne une Series
    if isinstance(prices, pd.Series):
        df = prices.to_frame("price_eur_mwh")
    else:
        df = prices.rename(columns={prices.columns[0]: "price_eur_mwh"})

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Resample vers 15min si horaire (forward-fill : prix DA constant sur l'heure)
    if len(df) > 1:
        median_delta = df.index.to_series().diff().median()
        if median_delta > pd.Timedelta(minutes=15):
            # Créer index 15min complet et forward-fill
            full_idx = pd.date_range(df.index.min(), df.index.max(), freq="15min", tz="UTC")
            df = df.reindex(full_idx).ffill()
            # Supprimer les 3 quarts d'heure ajoutés après le dernier point horaire
            # qui dépasseraient la plage demandée
            df = df[df.index < ts_end]

    df = _clean(df)

    logger.info(
        "EPEX chargé : %d lignes (%.1f%% négatifs)",
        len(df), (df["price_eur_mwh"] < 0).mean() * 100,
    )
    return df


def load_parquet(path: str | Path = DEFAULT_PARQUET) -> pd.DataFrame:
    """Charge le cache Parquet local."""
    return pd.read_parquet(path)


def fetch_and_cache(
    start: str,
    end: str,
    country_code: str = "CH",
    parquet_path: str | Path = DEFAULT_PARQUET,
) -> pd.DataFrame:
    """
    Télécharge depuis l'API ENTSO-E, fusionne avec le cache Parquet local
    (déduplique), et sauvegarde.

    Returns:
        DataFrame complet mis à jour
    """
    new = load_from_api(start, end, country_code)

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


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Spike flagging + vérification complétude."""
    df = df.copy()

    # Vérification complétude
    if len(df) > 1:
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq="15min", tz="UTC")
        missing = full_idx.difference(df.index)
        if len(missing) > 0:
            logger.warning("%d intervalles 15min manquants", len(missing))

    # Spike flag par mois
    monthly_p999 = df.groupby(df.index.to_period("M"))["price_eur_mwh"].transform(
        lambda x: np.nanpercentile(np.abs(x), 99.9)
    )
    df["spike_flag"] = np.abs(df["price_eur_mwh"]) > monthly_p999

    if df["spike_flag"].sum() > 0:
        logger.info("%d spikes extrêmes flaggés (conservés)", df["spike_flag"].sum())

    return df

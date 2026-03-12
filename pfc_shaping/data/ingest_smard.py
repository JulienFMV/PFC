"""
ingest_smard.py
---------------
Ingestion des prix EPEX Spot day-ahead depuis l'API SMARD.de
(Bundesnetzagentur — données publiques, sans authentification).

SMARD expose les prix day-ahead pour la Suisse (filter 259) en résolution
horaire. Le module forward-fill vers 15min pour homogénéité avec le pipeline.

Avantages vs ENTSO-E API :
    - Pas de clé API nécessaire
    - Très stable et fiable (hébergé par le régulateur allemand)
    - Données mises à jour quasi temps-réel

Limitations :
    - Prix CH uniquement en résolution horaire (pas 15min natif)
    - Load et generation CH non disponibles (seulement DE)

Filtres SMARD utilisés :
    259  — Prix day-ahead Suisse (EUR/MWh)
    4169 — Prix day-ahead DE-LU (EUR/MWh) — fallback

Cache Parquet local maintenu pour limiter les appels API.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

SMARD_BASE_URL = "https://www.smard.de/app/chart_data"
DEFAULT_PARQUET = Path(__file__).resolve().parent.parent / "data" / "epex_15min.parquet"

# Filtres SMARD
FILTER_CH_PRICE = 259      # Prix day-ahead Suisse
FILTER_DELU_PRICE = 4169   # Prix day-ahead DE-LU (fallback)

MAX_RETRIES = 3
BASE_DELAY = 2
REQUEST_TIMEOUT = 30


def _retry_get(url: str, max_retries: int = MAX_RETRIES) -> requests.Response:
    """GET avec retry + backoff exponentiel."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = BASE_DELAY * (2 ** attempt)
            logger.warning(
                "SMARD tentative %d/%d échouée (%s), retry dans %ds",
                attempt + 1, max_retries, e, delay,
            )
            time.sleep(delay)


def _get_index(filter_id: int, region: str = "DE", resolution: str = "hour") -> list[int]:
    """Récupère la liste des timestamps disponibles (début de chaque chunk hebdo)."""
    url = f"{SMARD_BASE_URL}/{filter_id}/{region}/index_{resolution}.json"
    resp = _retry_get(url)
    return resp.json()["timestamps"]


def _get_chunk(
    filter_id: int,
    region: str,
    resolution: str,
    timestamp: int,
) -> list[list]:
    """Récupère un chunk de données SMARD (1 semaine)."""
    url = (
        f"{SMARD_BASE_URL}/{filter_id}/{region}/"
        f"{filter_id}_{region}_{resolution}_{timestamp}.json"
    )
    resp = _retry_get(url)
    return resp.json()["series"]


def load_from_smard(
    start: str,
    end: str,
    country_code: str = "CH",
) -> pd.DataFrame:
    """
    Charge les prix day-ahead depuis l'API SMARD.

    Args:
        start / end    : 'YYYY-MM-DD'
        country_code   : 'CH' (défaut) ou 'DE-LU'

    Returns:
        DataFrame canonique :
            index   : DatetimeIndex UTC freq='15min'
            colonnes: ['price_eur_mwh', 'spike_flag']
    """
    filter_id = FILTER_CH_PRICE if country_code == "CH" else FILTER_DELU_PRICE
    region = "DE" if country_code == "CH" else "DE-LU"

    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")
    start_ms = int(ts_start.timestamp() * 1000)
    end_ms = int(ts_end.timestamp() * 1000)

    logger.info("SMARD API prix DA : %s → %s (filter=%d, zone=%s)", start, end, filter_id, country_code)

    # Récupérer l'index des chunks disponibles
    index_timestamps = _get_index(filter_id, region, "hour")

    # Sélectionner les chunks qui couvrent notre période
    relevant_chunks = [
        ts for ts in index_timestamps
        if ts >= start_ms - 7 * 24 * 3600 * 1000  # chunk précédent pour couverture
        and ts <= end_ms
    ]

    if not relevant_chunks:
        raise ValueError(f"Aucune donnée SMARD disponible pour {start} → {end}")

    # Charger tous les chunks pertinents
    all_series = []
    for chunk_ts in relevant_chunks:
        try:
            series = _get_chunk(filter_id, region, "hour", chunk_ts)
            all_series.extend(series)
        except Exception as e:
            logger.warning("Chunk SMARD %d échoué : %s", chunk_ts, e)

    if not all_series:
        raise ValueError(f"Aucune donnée SMARD récupérée pour {start} → {end}")

    # Convertir en DataFrame
    df = pd.DataFrame(all_series, columns=["timestamp_ms", "price_eur_mwh"])
    df = df.dropna(subset=["price_eur_mwh"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.set_index("timestamp").drop(columns=["timestamp_ms"])
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Filtrer sur la période demandée
    df = df[(df.index >= ts_start) & (df.index < ts_end)]

    # Resample vers 15min (forward-fill : prix DA constant sur l'heure)
    if not df.empty:
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq="15min", tz="UTC")
        df = df.reindex(full_idx).ffill()
        df = df[(df.index >= ts_start) & (df.index < ts_end)]

    # Spike flagging
    df = _clean(df)

    logger.info(
        "SMARD chargé : %d lignes 15min (%.1f%% négatifs)",
        len(df), (df["price_eur_mwh"] < 0).mean() * 100 if len(df) > 0 else 0,
    )
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Spike flagging + vérification complétude."""
    df = df.copy()

    if df.empty:
        df["spike_flag"] = pd.Series(dtype=bool)
        return df

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
    Télécharge depuis SMARD, fusionne avec le cache Parquet local
    (déduplique), et sauvegarde.

    Returns:
        DataFrame complet mis à jour
    """
    new = load_from_smard(start, end, country_code)

    parquet_path = Path(parquet_path)
    if parquet_path.exists():
        existing = load_parquet(parquet_path)
        combined = pd.concat([existing, new])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    logger.info("Cache EPEX mis à jour (SMARD) : %s (%d lignes)", parquet_path, len(combined))
    return combined

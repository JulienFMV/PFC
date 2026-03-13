"""
ingest_energy_charts.py
-----------------------
Source de données unifiée via l'API energy-charts.info (Fraunhofer ISE).

Remplace ingest_smard.py (prix uniquement) et ingest_entso.py (load/gen)
par une source unique couvrant TOUTES les données nécessaires :
    - Prix day-ahead EPEX CH (bzn=CH) — résolution horaire, ffill → 15min
    - Prix day-ahead EPEX DE-LU (bzn=DE-LU) — résolution 15min native
      (depuis transition SDAC le 1er oct 2025, utilisé pour calibrer f_Q)
    - Charge réseau CH (Load)
    - Production solaire, éolienne, hydraulique
    - Flux transfrontaliers (optionnel)

Avantages :
    - API publique, sans authentification
    - Source unique = moins de points de défaillance
    - Données Fraunhofer ISE très fiables (agrègent ENTSO-E + SMARD)
    - DE-LU 15min natif depuis oct 2025 (marché le plus liquide d'Europe)

API docs : https://api.energy-charts.info
Licence  : CC BY 4.0 (Bundesnetzagentur / SMARD.de)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

API_BASE = "https://api.energy-charts.info"

MAX_RETRIES = 3
BASE_DELAY = 2
REQUEST_TIMEOUT = 30

DEFAULT_EPEX_PARQUET = Path(__file__).resolve().parent.parent / "data" / "epex_15min.parquet"
DEFAULT_EPEX_DE_PARQUET = Path(__file__).resolve().parent.parent / "data" / "epex_de_15min.parquet"
DEFAULT_ENTSO_PARQUET = Path(__file__).resolve().parent.parent / "data" / "entso_15min.parquet"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _retry_get(url: str, params: dict | None = None) -> requests.Response:
    """GET avec retry + backoff exponentiel."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = BASE_DELAY * (2 ** attempt)
            logger.warning(
                "energy-charts tentative %d/%d échouée (%s), retry dans %ds",
                attempt + 1, MAX_RETRIES, e, delay,
            )
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Prix day-ahead
# ---------------------------------------------------------------------------

def load_prices(start: str, end: str, bzn: str = "CH") -> pd.DataFrame:
    """
    Charge les prix day-ahead depuis energy-charts.info.

    Args:
        start / end : 'YYYY-MM-DD'
        bzn         : zone de prix ('CH', 'DE-LU', 'AT', ...)

    Returns:
        DataFrame canonique :
            index   : DatetimeIndex UTC freq='15min'
            colonnes: ['price_eur_mwh', 'spike_flag']
    """
    logger.info("energy-charts prix DA : %s → %s (bzn=%s)", start, end, bzn)

    resp = _retry_get(f"{API_BASE}/price", params={
        "bzn": bzn,
        "start": start,
        "end": end,
    })
    data = resp.json()

    timestamps = data.get("unix_seconds", [])
    prices = data.get("price", [])

    if not timestamps or not prices:
        raise ValueError(f"Aucune donnée prix energy-charts pour {start} → {end} (bzn={bzn})")

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
        "price_eur_mwh": prices,
    }).set_index("timestamp")

    df = df.dropna(subset=["price_eur_mwh"])
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Filtrer sur la période demandée
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")
    df = df[(df.index >= ts_start) & (df.index < ts_end)]

    # Resample vers 15min (forward-fill : prix DA constant sur l'heure)
    if not df.empty:
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq="15min", tz="UTC")
        df = df.reindex(full_idx).ffill()
        df = df[(df.index >= ts_start) & (df.index < ts_end)]

    # Spike flagging
    df = _spike_flag(df)

    logger.info(
        "energy-charts prix chargés : %d lignes 15min (%.1f%% négatifs)",
        len(df), (df["price_eur_mwh"] < 0).mean() * 100 if len(df) > 0 else 0,
    )
    return df


# ---------------------------------------------------------------------------
# Load + Generation (public_power)
# ---------------------------------------------------------------------------

# Mapping des noms energy-charts → colonnes canoniques
_PRODUCTION_TYPE_MAP = {
    "Load": "load_mw",
    "Solar": "solar_mw",
    "Wind onshore": "wind_mw",
    "Hydro Run-of-River": "hydro_ror_mw",
    "Hydro water reservoir": "hydro_reservoir_mw",
    "Hydro pumped storage": "hydro_pumped_mw",
    "Nuclear": "nuclear_mw",
    "Cross border electricity trading": "cross_border_mw",
}


def load_power(start: str, end: str, country: str = "ch") -> pd.DataFrame:
    """
    Charge load + generation depuis energy-charts.info.

    Args:
        start / end : 'YYYY-MM-DD'
        country     : code pays ('ch', 'de', ...)

    Returns:
        DataFrame canonique :
            index   : DatetimeIndex UTC freq='15min'
            colonnes: ['load_mw', 'solar_mw', 'wind_mw',
                        'hydro_ror_mw', 'hydro_reservoir_mw', 'hydro_pumped_mw',
                        'nuclear_mw', 'cross_border_mw']
    """
    logger.info("energy-charts public_power : %s → %s (country=%s)", start, end, country)

    resp = _retry_get(f"{API_BASE}/public_power", params={
        "country": country,
        "start": start,
        "end": end,
    })
    data = resp.json()

    timestamps = data.get("unix_seconds", [])
    production_types = data.get("production_types", [])

    if not timestamps:
        raise ValueError(f"Aucune donnée power energy-charts pour {start} → {end}")

    # Construire le DataFrame à partir des production_types
    df = pd.DataFrame(index=pd.to_datetime(timestamps, unit="s", utc=True))
    df.index.name = "timestamp"

    for pt in production_types:
        name = pt.get("name", "")
        col = _PRODUCTION_TYPE_MAP.get(name)
        if col and "data" in pt:
            values = pt["data"]
            # Remplacer None par NaN
            values = [v if v is not None else np.nan for v in values]
            if len(values) == len(df):
                df[col] = values

    # S'assurer que les colonnes canoniques existent
    for col in ["load_mw", "solar_mw", "wind_mw"]:
        if col not in df.columns:
            df[col] = np.nan
            logger.warning("Colonne %s absente des données energy-charts", col)

    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Filtrer sur la période demandée
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")
    df = df[(df.index >= ts_start) & (df.index < ts_end)]

    # Resample vers 15min (forward-fill)
    if not df.empty:
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq="15min", tz="UTC")
        df = df.reindex(full_idx).ffill()
        df = df[(df.index >= ts_start) & (df.index < ts_end)]

    logger.info(
        "energy-charts power chargé : %d lignes 15min, colonnes=%s",
        len(df), list(df.columns),
    )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les features exogènes dérivées (compatibles avec ingest_entso).

    Colonnes ajoutées :
        - solar_regime : {0, 1, 2} — terciles mensuels de production solaire
        - load_deviation : écart normalisé charge vs moyenne mensuelle
    """
    df = df.copy()

    # Solar regime : terciles mensuels
    if "solar_mw" in df.columns:
        def _solar_tercile(x):
            try:
                return pd.qcut(x, q=3, labels=[0, 1, 2], duplicates="drop")
            except (ValueError, TypeError):
                # Pas assez de valeurs distinctes (ex: nuit = tout à 0)
                return pd.Series(1, index=x.index)

        monthly = df.groupby(df.index.to_period("M"))["solar_mw"]
        df["solar_regime"] = monthly.transform(_solar_tercile)
        df["solar_regime"] = df["solar_regime"].fillna(1).astype(int)
    else:
        df["solar_regime"] = 1

    # Load deviation : z-score mensuel
    if "load_mw" in df.columns:
        monthly = df.groupby(df.index.to_period("M"))["load_mw"]
        df["load_deviation"] = monthly.transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        df["load_deviation"] = df["load_deviation"].fillna(0)
    else:
        df["load_deviation"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Spike flagging (identique à ingest_smard / ingest_epex)
# ---------------------------------------------------------------------------

def _spike_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Spike flagging par mois (percentile 99.9)."""
    df = df.copy()

    if df.empty:
        df["spike_flag"] = pd.Series(dtype=bool)
        return df

    monthly_p999 = df.groupby(df.index.to_period("M"))["price_eur_mwh"].transform(
        lambda x: np.nanpercentile(np.abs(x), 99.9)
    )
    df["spike_flag"] = np.abs(df["price_eur_mwh"]) > monthly_p999

    if df["spike_flag"].sum() > 0:
        logger.info("%d spikes extrêmes flaggés (conservés)", df["spike_flag"].sum())

    return df


# ---------------------------------------------------------------------------
# Cache Parquet
# ---------------------------------------------------------------------------

def load_epex_parquet(path: str | Path = DEFAULT_EPEX_PARQUET) -> pd.DataFrame:
    """Charge le cache Parquet local des prix EPEX."""
    return pd.read_parquet(path)


def load_power_parquet(path: str | Path = DEFAULT_ENTSO_PARQUET) -> pd.DataFrame:
    """Charge le cache Parquet local load/generation."""
    return pd.read_parquet(path)


def fetch_and_cache_prices(
    start: str,
    end: str,
    bzn: str = "CH",
    parquet_path: str | Path = DEFAULT_EPEX_PARQUET,
) -> pd.DataFrame:
    """
    Télécharge les prix depuis energy-charts, fusionne avec le cache Parquet,
    et sauvegarde.
    """
    new = load_prices(start, end, bzn)

    parquet_path = Path(parquet_path)
    if parquet_path.exists():
        existing = load_epex_parquet(parquet_path)
        combined = pd.concat([existing, new])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    logger.info("Cache EPEX mis à jour (energy-charts) : %s (%d lignes)", parquet_path, len(combined))
    return combined


def fetch_and_cache_prices_de(
    start: str,
    end: str,
    parquet_path: str | Path = DEFAULT_EPEX_DE_PARQUET,
) -> pd.DataFrame:
    """
    Télécharge les prix DE-LU 15min depuis energy-charts et met à jour le cache.

    Les prix DE-LU sont en résolution 15min native depuis la transition SDAC
    (1er octobre 2025). Avant cette date, les données sont horaires ffill.
    Ces prix sont utilisés pour calibrer les facteurs f_Q (profil intra-horaire)
    du modèle de shaping CH.
    """
    new = load_prices(start, end, bzn="DE-LU")

    parquet_path = Path(parquet_path)
    if parquet_path.exists():
        existing = load_epex_parquet(parquet_path)
        combined = pd.concat([existing, new])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    logger.info("Cache EPEX DE-LU mis à jour (energy-charts) : %s (%d lignes)", parquet_path, len(combined))
    return combined


def fetch_and_cache_power(
    start: str,
    end: str,
    country: str = "ch",
    parquet_path: str | Path = DEFAULT_ENTSO_PARQUET,
) -> pd.DataFrame:
    """
    Télécharge load/generation depuis energy-charts, fusionne avec le cache
    Parquet, et sauvegarde. Recalcule les features sur le dataset complet.
    """
    new = load_power(start, end, country)

    parquet_path = Path(parquet_path)
    if parquet_path.exists():
        existing = load_power_parquet(parquet_path)
        combined = pd.concat([existing, new])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new

    # Recalculer les features sur le dataset complet
    combined = build_features(combined)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    logger.info("Cache power mis à jour (energy-charts) : %s (%d lignes)", parquet_path, len(combined))
    return combined

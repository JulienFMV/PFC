"""
ingest_outages.py
-----------------
Ingestion des indisponibilités de production depuis l'API ENTSO-E Transparency
(REMIT UMM — Urgent Market Messages).

Données récupérées :
    - Indisponibilités planifiées et forcées des unités de production
    - Agrégées en capacité indisponible (MW) par pas 15min

Format de sortie canonique (Parquet local) :
    index : DatetimeIndex UTC freq='15min'
    colonnes :
        unavailable_mw      — capacité indisponible totale [MW]
        unavailable_nuclear  — capacité nucléaire indisponible [MW]
        unavailable_hydro    — capacité hydro indisponible [MW]
        unavailable_thermal  — capacité thermique indisponible [MW]
        n_outages            — nombre d'unités en arrêt
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

DEFAULT_PARQUET = Path(__file__).resolve().parent.parent / "data" / "outages_15min.parquet"

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
                "ENTSO-E outages tentative %d/%d échouée (%s), retry dans %ds",
                attempt + 1, max_retries, e, delay,
            )
            time.sleep(delay)


def _classify_fuel(fuel_type: str) -> str:
    """Classifie le type de combustible en catégorie agrégée."""
    fuel_lower = str(fuel_type).lower()
    if "nuclear" in fuel_lower:
        return "nuclear"
    if any(k in fuel_lower for k in ("hydro", "water", "pump", "reservoir", "run")):
        return "hydro"
    return "thermal"


def load_outages_from_api(
    start: str,
    end: str,
    country_code: str = "CH",
) -> pd.DataFrame:
    """
    Charge les indisponibilités depuis l'API ENTSO-E.

    Args:
        start / end  : 'YYYY-MM-DD'
        country_code : code zone ENTSO-E (défaut 'CH')

    Returns:
        DataFrame avec capacité indisponible par pas 15min
    """
    client = _get_client()

    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")

    logger.info("ENTSO-E outages: %s → %s (zone=%s)", start, end, country_code)

    # query_unavailability_of_generation_units retourne un DataFrame
    # avec colonnes: Available/Nominal Capacity, fuel type, etc.
    try:
        outages = _retry(
            client.query_unavailability_of_generation_units,
            country_code, ts_start, ts_end,
        )
    except Exception as e:
        logger.warning("Impossible de charger les outages ENTSO-E: %s", e)
        # Retourner un DataFrame vide avec les bonnes colonnes
        idx = pd.date_range(ts_start, ts_end, freq="15min", inclusive="left", tz="UTC")
        return pd.DataFrame({
            "unavailable_mw": 0.0,
            "unavailable_nuclear": 0.0,
            "unavailable_hydro": 0.0,
            "unavailable_thermal": 0.0,
            "n_outages": 0,
        }, index=idx)

    if outages is None or (isinstance(outages, pd.DataFrame) and outages.empty):
        logger.info("Aucune indisponibilité trouvée pour la période")
        idx = pd.date_range(ts_start, ts_end, freq="15min", inclusive="left", tz="UTC")
        return pd.DataFrame({
            "unavailable_mw": 0.0,
            "unavailable_nuclear": 0.0,
            "unavailable_hydro": 0.0,
            "unavailable_thermal": 0.0,
            "n_outages": 0,
        }, index=idx)

    # L'API retourne des événements avec start/end et capacité nominale/disponible.
    # On doit les transformer en time series 15min.
    return _events_to_timeseries(outages, ts_start, ts_end)


def _events_to_timeseries(
    outages: pd.DataFrame,
    ts_start: pd.Timestamp,
    ts_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Convertit les événements d'indisponibilité en time series 15min.

    entsoe-py retourne un DataFrame avec colonnes :
        nominal_power, avail_qty, start, end, plant_type, docstatus,
        production_resource_name, businesstype, ...
    Index: created_doc_time
    """
    idx = pd.date_range(ts_start, ts_end, freq="15min", inclusive="left", tz="UTC")
    result = pd.DataFrame(index=idx)
    result["unavailable_mw"] = 0.0
    result["unavailable_nuclear"] = 0.0
    result["unavailable_hydro"] = 0.0
    result["unavailable_thermal"] = 0.0
    result["n_outages"] = 0

    if not isinstance(outages, pd.DataFrame) or outages.empty:
        return result

    # Filter out cancelled events
    if "docstatus" in outages.columns:
        active = outages[outages["docstatus"] != "Cancelled"].copy()
    else:
        active = outages.copy()

    # Deduplicate: keep latest revision per (mrid, production_resource_name)
    if "revision" in active.columns and "mrid" in active.columns:
        active["revision"] = pd.to_numeric(active["revision"], errors="coerce").fillna(1)
        active = active.sort_values("revision").drop_duplicates(
            subset=["mrid"], keep="last",
        )

    n_applied = 0
    for _, row in active.iterrows():
        try:
            # Column names from entsoe-py
            nominal = pd.to_numeric(row.get("nominal_power", 0), errors="coerce") or 0
            available = pd.to_numeric(row.get("avail_qty", 0), errors="coerce") or 0
            unavail_mw = max(0, float(nominal) - float(available))

            if unavail_mw < 1:
                continue

            # Parse start/end timestamps
            evt_start = pd.Timestamp(row["start"])
            evt_end = pd.Timestamp(row["end"])

            if evt_start.tz is None:
                evt_start = evt_start.tz_localize("UTC")
            else:
                evt_start = evt_start.tz_convert("UTC")
            if evt_end.tz is None:
                evt_end = evt_end.tz_localize("UTC")
            else:
                evt_end = evt_end.tz_convert("UTC")

            # Classify fuel type
            fuel_raw = str(row.get("plant_type", row.get("psrType", "unknown")))
            fuel_cat = _classify_fuel(fuel_raw)

            # Apply to time series
            mask = (idx >= evt_start) & (idx < evt_end)
            if mask.any():
                result.loc[mask, "unavailable_mw"] += unavail_mw
                result.loc[mask, f"unavailable_{fuel_cat}"] += unavail_mw
                result.loc[mask, "n_outages"] += 1
                n_applied += 1

        except Exception as e:
            logger.debug("Outage event parsing error: %s — row: %s", e, dict(row))
            continue

    logger.info(
        "Outages: %d events active (%d cancelled), %d applied to timeseries, "
        "max unavail=%.0f MW, mean n_outages=%.1f",
        len(active), len(outages) - len(active), n_applied,
        result["unavailable_mw"].max(), result["n_outages"].mean(),
    )
    return result


def fetch_and_cache(
    start: str,
    end: str,
    parquet_path: str | Path = DEFAULT_PARQUET,
    country_code: str = "CH",
) -> pd.DataFrame:
    """
    Télécharge les outages, fusionne avec le cache local et sauvegarde.
    """
    new_data = load_outages_from_api(start, end, country_code)

    parquet_path = Path(parquet_path)
    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_data

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    logger.info("Cache outages mis à jour: %s (%d lignes)", parquet_path, len(combined))
    return combined


def load_parquet(path: str | Path = DEFAULT_PARQUET) -> pd.DataFrame | None:
    """Charge le cache Parquet local, retourne None si inexistant."""
    p = Path(path)
    if p.exists():
        return pd.read_parquet(p)
    return None

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

from pfc_shaping.data.ct_datasets import resolve_local_cache_path

logger = logging.getLogger(__name__)

DEFAULT_PARQUET = Path(__file__).resolve().parent.parent / "data" / "entso_15min.parquet"
DEFAULT_DE_RENEWABLE_FORECAST_PARQUET = (
    Path(__file__).resolve().parent.parent / "data" / "de_renewable_forecast.parquet"
)
DEFAULT_MULTI_COUNTRY_FORECAST_PARQUET = (
    Path(__file__).resolve().parent.parent / "data" / "multi_country_forecast_15min.parquet"
)
DEFAULT_FR_NUCLEAR_FORECAST_PARQUET = (
    Path(__file__).resolve().parent.parent / "data" / "fr_nuclear_forecast_15min.parquet"
)

# Charger .env depuis la racine du worktree, puis fallback vers le repo PFC
_ENV_CANDIDATES = [
    Path(__file__).resolve().parent.parent.parent / ".env",
    Path(__file__).resolve().parent.parent.parent.parent / "PFC" / ".env",
]
for _env_path in _ENV_CANDIDATES:
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path)
        break

MAX_RETRIES = 3
BASE_DELAY = 5
SWISS_BORDERS = {
    "de": "DE_LU",
    "fr": "FR",
    "at": "AT",
    "it": "IT_NORD",
}
NEIGHBOR_ZONES = {
    "at": "AT",
    "de": "DE_LU",
    "fr": "FR",
    "it": "IT_NORD",
}
FORECAST_ZONES = {
    "ch": "CH",
    "de": "DE_LU",
    "fr": "FR",
    "at": "AT",
    "it": "IT_NORD",
}
FORECAST_PROCESS_TYPES = ("A01", "A31")


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
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
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
    Charge load + generation détaillée depuis l'API ENTSO-E pour une période donnée.

    Args:
        start / end  : 'YYYY-MM-DD'
        country_code : code zone ENTSO-E (défaut 'CH')

    Returns:
        DataFrame colonnes ['load_mw', 'solar_mw', 'wind_mw',
                            'nuclear_mw', 'hydro_ror_mw',
                            'hydro_reservoir_mw', 'hydro_pumped_mw']
        index : DatetimeIndex UTC
    """
    client = _get_client()

    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")

    logger.info("ENTSO-E API load + generation : %s → %s (zone=%s)", start, end, country_code)

    # --- Load ---
    df_load_raw = _retry(client.query_load, country_code, start=ts_start, end=ts_end)
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
    df_gen_raw = _retry(client.query_generation, country_code, start=ts_start, end=ts_end)

    # Extraire les technologies clés depuis les colonnes multi-level ou flat.
    solar_mw = _extract_generation_column(df_gen_raw, "Solar")
    wind_mw = (
        _extract_generation_column(df_gen_raw, "Wind Onshore")
        + _extract_generation_column(df_gen_raw, "Wind Offshore")
    )
    nuclear_mw = _extract_generation_column(df_gen_raw, "Nuclear")
    hydro_ror_mw = _extract_generation_column(df_gen_raw, "Hydro Run-of-river and poundage")
    hydro_reservoir_mw = _extract_generation_column(df_gen_raw, "Hydro Water Reservoir")
    hydro_pumped_mw = _extract_generation_column(df_gen_raw, "Hydro Pumped Storage")

    df_gen = pd.DataFrame(
        {
            "solar_mw": solar_mw,
            "wind_mw": wind_mw,
            "nuclear_mw": nuclear_mw,
            "hydro_ror_mw": hydro_ror_mw,
            "hydro_reservoir_mw": hydro_reservoir_mw,
            "hydro_pumped_mw": hydro_pumped_mw,
        }
    )

    # --- Resample à 15min et joindre ---
    for df_ in [df_load, df_gen]:
        if df_.index.tz is None:
            df_.index = df_.index.tz_localize("UTC")

    # Resample si nécessaire (certaines séries sont horaires)
    df_load_15 = _resample_to_15min(df_load)
    df_gen_15 = _resample_to_15min(df_gen)

    df = df_load_15.join(df_gen_15, how="outer").sort_index()
    fill_zero_cols = [
        c for c in [
            "solar_mw",
            "wind_mw",
            "nuclear_mw",
            "hydro_ror_mw",
            "hydro_reservoir_mw",
            "hydro_pumped_mw",
        ]
        if c in df.columns
    ]
    if fill_zero_cols:
        df[fill_zero_cols] = df[fill_zero_cols].fillna(0.0)

    neighbor_df = _load_neighbor_power_features(client, ts_start, ts_end)
    if not neighbor_df.empty:
        df = df.join(neighbor_df, how="outer").sort_index()

    border_df = _load_swiss_border_features(client, ts_start, ts_end)
    if not border_df.empty:
        df = df.join(border_df, how="outer").sort_index()

    fr_nuclear_df = _load_fr_nuclear_outage_features(client, ts_start, ts_end)
    if not fr_nuclear_df.empty:
        df = df.join(fr_nuclear_df, how="outer").sort_index()

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

    fuel_norm = str(fuel_type).strip().lower()

    # Multi-level columns: ('Solar', 'Actual Aggregated'), etc.
    if isinstance(df_gen.columns, pd.MultiIndex):
        matching = [col for col in df_gen.columns if str(col[0]).strip().lower() == fuel_norm]
        if not matching:
            return pd.Series(0.0, index=df_gen.index, dtype=float)
        # Préférer 'Actual Aggregated' sur 'Actual Consumption'
        for col in matching:
            if "Aggregated" in str(col[1]):
                return df_gen[col].fillna(0.0)
        return df_gen[matching[0]].fillna(0.0)

    # Flat columns
    matching = [c for c in df_gen.columns if fuel_norm in str(c).strip().lower()]
    if not matching:
        return pd.Series(0.0, index=df_gen.index, dtype=float)
    return df_gen[matching[0]].fillna(0.0)


def _extract_forecast_column(df_forecast: pd.DataFrame, fuel_type: str) -> pd.Series:
    """
    Extract a wind/solar forecast series from ENTSO-E forecast output.

    The entsoe-py payload shape can vary by zone and endpoint; re-use the
    generation-column matching logic and fall back to a broad text match.
    """
    if df_forecast is None or len(df_forecast) == 0:
        return pd.Series(dtype=float)

    series = _extract_generation_column(df_forecast, fuel_type)
    if not series.empty and not series.isna().all():
        return series.astype(float)

    fuel_norm = str(fuel_type).strip().lower()
    if isinstance(df_forecast.columns, pd.MultiIndex):
        matching = [
            col for col in df_forecast.columns
            if fuel_norm in " ".join(map(str, col)).strip().lower()
        ]
        if matching:
            return df_forecast[matching[0]].fillna(0.0).astype(float)
    else:
        matching = [c for c in df_forecast.columns if fuel_norm in str(c).strip().lower()]
        if matching:
            return df_forecast[matching[0]].fillna(0.0).astype(float)

    return pd.Series(0.0, index=df_forecast.index, dtype=float)


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


def _to_15min_series(series: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    """Normalize ENTSO-E outputs to a named UTC 15min series."""
    if isinstance(series, pd.DataFrame):
        if series.empty:
            return pd.Series(dtype=float, name=name)
        series = series.iloc[:, 0]

    series = series.rename(name).sort_index()
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")

    if len(series) > 1:
        median_delta = series.index.to_series().diff().median()
        if median_delta > pd.Timedelta(minutes=15):
            series = series.resample("15min").ffill()
    return series.astype(float)


def _query_series_or_empty(func, *args, name: str, **kwargs) -> pd.Series:
    """Run ENTSO-E query and degrade gracefully if a border is unavailable."""
    try:
        raw = _retry(func, *args, **kwargs)
        return _to_15min_series(raw, name)
    except Exception as exc:
        logger.warning("ENTSO-E query failed for %s: %s", name, exc)
        return pd.Series(dtype=float, name=name)


def _query_load_forecast_or_empty(
    client,
    zone_code: str,
    ts_start: pd.Timestamp,
    ts_end: pd.Timestamp,
    name: str,
    process_type: str = "A01",
) -> pd.Series:
    """Run ENTSO-E load forecast query and degrade gracefully if unavailable."""
    query_func = getattr(client, "query_load_forecast", None)
    if query_func is None:
        logger.warning("ENTSO-E client has no query_load_forecast method for %s", name)
        return pd.Series(dtype=float, name=name)
    try:
        raw = _retry(query_func, zone_code, start=ts_start, end=ts_end, process_type=process_type)
    except Exception as exc:
        logger.warning("ENTSO-E load forecast failed for %s (%s): %s", name, process_type, exc)
        return pd.Series(dtype=float, name=name)

    if isinstance(raw, pd.DataFrame):
        if raw.empty:
            return pd.Series(dtype=float, name=name)
        if "Forecasted Load" in raw.columns:
            series = raw["Forecasted Load"]
        else:
            series = raw.iloc[:, -1]
    else:
        series = raw
    return _to_15min_series(series, name)


def _merge_preferred_forecast_series(
    preferred: pd.Series,
    fallback: pd.Series,
    name: str,
) -> pd.Series:
    """Prefer the higher-fidelity series and fill remaining gaps from the fallback horizon."""
    if preferred.empty:
        return fallback.rename(name)
    if fallback.empty:
        return preferred.rename(name)
    merged = preferred.combine_first(fallback).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.rename(name)


def _merge_preferred_forecast_frames(
    preferred: pd.DataFrame,
    fallback: pd.DataFrame,
) -> pd.DataFrame:
    """Prefer the higher-fidelity frame and fill remaining gaps from fallback columns."""
    if preferred.empty:
        return fallback
    if fallback.empty:
        return preferred
    merged = preferred.combine_first(fallback).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def load_multi_country_forecasts(
    start: str,
    end: str,
    zones: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Load J+ forecasts for CH/DE/FR/AT/IT from ENTSO-E.

    Returns a 15min UTC dataframe indexed by forecast_for_ts_utc with columns such as:
      - forecast_load_ch_mw
      - forecast_load_de_mw
      - forecast_solar_de_mw
      - forecast_wind_de_mw
      - ...
    """
    client = _get_client()
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")
    zones = zones or FORECAST_ZONES

    frames: list[pd.Series] = []
    for key, zone_code in zones.items():
        load_series_da = _query_load_forecast_or_empty(
            client,
            zone_code,
            ts_start,
            ts_end,
            name=f"forecast_load_{key}_mw",
            process_type="A01",
        )
        load_series_wa = _query_load_forecast_or_empty(
            client,
            zone_code,
            ts_start,
            ts_end,
            name=f"forecast_load_{key}_mw",
            process_type="A31",
        )
        load_series = _merge_preferred_forecast_series(
            preferred=load_series_da,
            fallback=load_series_wa,
            name=f"forecast_load_{key}_mw",
        )
        if not load_series.empty:
            frames.append(load_series)

        forecast_frames_by_process: list[pd.DataFrame] = []
        for process_type in FORECAST_PROCESS_TYPES:
            try:
                forecast_raw = _retry(
                    client.query_wind_and_solar_forecast,
                    zone_code,
                    start=ts_start,
                    end=ts_end,
                    process_type=process_type,
                )
                solar = _extract_forecast_column(forecast_raw, "Solar").rename(f"forecast_solar_{key}_mw")
                wind = _extract_forecast_column(forecast_raw, "Wind").rename(f"forecast_wind_{key}_mw")
                forecast_df = pd.concat([solar, wind], axis=1)
                if forecast_df.empty:
                    continue
                if forecast_df.index.tz is None:
                    forecast_df.index = forecast_df.index.tz_localize("UTC")
                forecast_df = _resample_to_15min(forecast_df).sort_index()
                forecast_df = forecast_df[(forecast_df.index >= ts_start) & (forecast_df.index < ts_end)]
                for col in forecast_df.columns:
                    forecast_df[col] = forecast_df[col].astype(float).fillna(0.0)
                forecast_frames_by_process.append(forecast_df)
            except Exception as exc:
                logger.warning(
                    "ENTSO-E wind/solar forecast failed for %s (%s): %s",
                    key,
                    process_type,
                    exc,
                )
        if forecast_frames_by_process:
            forecast_df = forecast_frames_by_process[0]
            for extra_frame in forecast_frames_by_process[1:]:
                forecast_df = _merge_preferred_forecast_frames(
                    preferred=forecast_df,
                    fallback=extra_frame,
                )
            frames.extend(forecast_df[col] for col in forecast_df.columns)

    if not frames:
        return pd.DataFrame()

    forecast = pd.concat(frames, axis=1).sort_index()
    forecast = forecast[~forecast.index.duplicated(keep="last")]
    forecast.index.name = "forecast_for_ts_utc"
    return forecast


def load_fr_nuclear_forecast(
    start: str,
    end: str,
    total_capacity_mw: float = 61000.0,
) -> pd.DataFrame:
    """Load future French nuclear availability as a governed Swiss CT signal."""
    client = _get_client()
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")
    unavailable_df = _load_fr_nuclear_outage_features(client, ts_start, ts_end)
    if unavailable_df.empty or "fr_nuclear_unavailable_mw" not in unavailable_df.columns:
        return pd.DataFrame()

    unavailable = pd.to_numeric(
        unavailable_df["fr_nuclear_unavailable_mw"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    out = pd.DataFrame(index=unavailable_df.index.copy())
    out["forecast_fr_nuclear_unavailable_mw"] = unavailable
    out["forecast_fr_nuclear_available_mw"] = (float(total_capacity_mw) - unavailable).clip(lower=0.0)
    out["forecast_fr_nuclear_unavailability_ratio"] = (
        unavailable / float(total_capacity_mw)
    ).clip(lower=0.0, upper=1.5)
    out.index.name = "forecast_for_ts_utc"
    return out.sort_index()


def _load_fr_nuclear_outage_features(
    client,
    ts_start: pd.Timestamp,
    ts_end: pd.Timestamp,
) -> pd.DataFrame:
    """Load French nuclear unavailability as a compact Swiss regime signal."""
    cache_candidates = [
        Path(__file__).resolve().parent.parent / "data" / "outages_fr_15min.parquet",
        Path(__file__).resolve().parent.parent / "data" / "outages_15min.parquet",
    ]
    for cache_path in cache_candidates:
        if not cache_path.exists():
            continue
        try:
            cached = pd.read_parquet(cache_path)
            if "unavailable_nuclear" not in cached.columns or cached.empty:
                continue
            if not isinstance(cached.index, pd.DatetimeIndex):
                cached.index = pd.to_datetime(cached.index, utc=True)
            elif cached.index.tz is None:
                cached.index = cached.index.tz_localize("UTC")
            else:
                cached.index = cached.index.tz_convert("UTC")
            window = cached.loc[(cached.index >= ts_start) & (cached.index < ts_end), ["unavailable_nuclear"]]
            if window.empty:
                continue
            logger.info("Using cached FR nuclear outage signal from %s", cache_path)
            return window.rename(columns={"unavailable_nuclear": "fr_nuclear_unavailable_mw"})
        except Exception as exc:
            logger.warning("Failed to read cached FR nuclear outages from %s: %s", cache_path, exc)

    try:
        from pfc_shaping.data.ingest_outages import _events_to_timeseries  # noqa: WPS433
    except Exception as exc:
        logger.warning("Failed to import outage helper for FR nuclear outages: %s", exc)
        return pd.DataFrame()

    try:
        outages = _retry(
            client.query_unavailability_of_generation_units,
            "FR",
            ts_start,
            ts_end,
        )
    except Exception as exc:
        logger.warning("ENTSO-E FR outages failed: %s", exc)
        return pd.DataFrame()

    if outages is None or (isinstance(outages, pd.DataFrame) and outages.empty):
        return pd.DataFrame()

    outage_ts = _events_to_timeseries(outages, ts_start, ts_end)
    if outage_ts.empty or "unavailable_nuclear" not in outage_ts.columns:
        return pd.DataFrame()

    return outage_ts[["unavailable_nuclear"]].rename(
        columns={"unavailable_nuclear": "fr_nuclear_unavailable_mw"}
    )


def load_de_renewable_forecast(
    start: str,
    end: str,
    country_code: str = "DE_LU",
) -> pd.DataFrame:
    """
    Load ENTSO-E day-ahead renewable forecasts for Germany.

    Returns a 15min UTC dataframe with:
      - forecast_wind_de_mw
      - forecast_solar_de_mw
    """
    client = _get_client()

    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")

    logger.info(
        "ENTSO-E DE renewable forecast: %s -> %s (zone=%s)",
        start,
        end,
        country_code,
    )

    forecast_raw = _retry(
        client.query_wind_and_solar_forecast,
        country_code,
        start=ts_start,
        end=ts_end,
    )

    wind = _extract_forecast_column(forecast_raw, "Wind").rename("forecast_wind_de_mw")
    solar = _extract_forecast_column(forecast_raw, "Solar").rename("forecast_solar_de_mw")

    forecast_df = pd.concat([wind, solar], axis=1)
    if forecast_df.index.tz is None:
        forecast_df.index = forecast_df.index.tz_localize("UTC")

    forecast_df = _resample_to_15min(forecast_df).sort_index()
    forecast_df = forecast_df[(forecast_df.index >= ts_start) & (forecast_df.index < ts_end)]
    for col in ["forecast_wind_de_mw", "forecast_solar_de_mw"]:
        if col in forecast_df.columns:
            forecast_df[col] = forecast_df[col].astype(float).fillna(0.0)

    return forecast_df


def _load_swiss_border_features(
    client,
    ts_start: pd.Timestamp,
    ts_end: pd.Timestamp,
) -> pd.DataFrame:
    """Load CH bilateral schedule / flow / NTC series for key neighbors."""
    frames: list[pd.Series] = []

    for border_key, border_code in SWISS_BORDERS.items():
        sched_export = _query_series_or_empty(
            client.query_scheduled_exchanges,
            "CH",
            border_code,
            start=ts_start,
            end=ts_end,
            dayahead=True,
            name=f"scheduled_export_ch_{border_key}_mw",
        )
        sched_import = _query_series_or_empty(
            client.query_scheduled_exchanges,
            border_code,
            "CH",
            start=ts_start,
            end=ts_end,
            dayahead=True,
            name=f"scheduled_import_ch_{border_key}_mw",
        )
        if not sched_export.empty or not sched_import.empty:
            sched_idx = sched_export.index.union(sched_import.index)
            sched_export = sched_export.reindex(sched_idx).fillna(0.0)
            sched_import = sched_import.reindex(sched_idx).fillna(0.0)
            frames.append((sched_export - sched_import).rename(f"scheduled_net_export_ch_{border_key}_mw"))

        flow_export = _query_series_or_empty(
            client.query_crossborder_flows,
            "CH",
            border_code,
            start=ts_start,
            end=ts_end,
            name=f"flow_export_ch_{border_key}_mw",
        )
        flow_import = _query_series_or_empty(
            client.query_crossborder_flows,
            border_code,
            "CH",
            start=ts_start,
            end=ts_end,
            name=f"flow_import_ch_{border_key}_mw",
        )
        if not flow_export.empty or not flow_import.empty:
            flow_idx = flow_export.index.union(flow_import.index)
            flow_export = flow_export.reindex(flow_idx).fillna(0.0)
            flow_import = flow_import.reindex(flow_idx).fillna(0.0)
            frames.append((flow_export - flow_import).rename(f"flow_net_export_ch_{border_key}_mw"))

        ntc_export = _query_series_or_empty(
            client.query_net_transfer_capacity_dayahead,
            "CH",
            border_code,
            start=ts_start,
            end=ts_end,
            name=f"ntc_export_ch_{border_key}_mw",
        )
        ntc_import = _query_series_or_empty(
            client.query_net_transfer_capacity_dayahead,
            border_code,
            "CH",
            start=ts_start,
            end=ts_end,
            name=f"ntc_import_ch_{border_key}_mw",
        )
        if not ntc_export.empty or not ntc_import.empty:
            ntc_idx = ntc_export.index.union(ntc_import.index)
            ntc_export = ntc_export.reindex(ntc_idx).fillna(0.0)
            ntc_import = ntc_import.reindex(ntc_idx).fillna(0.0)
            frames.extend([
                ntc_export.rename(f"ntc_export_ch_{border_key}_mw"),
                ntc_import.rename(f"ntc_import_ch_{border_key}_mw"),
                (ntc_export - ntc_import).rename(f"ntc_net_ch_{border_key}_mw"),
                (ntc_export + ntc_import).rename(f"ntc_total_ch_{border_key}_mw"),
            ])

    if not frames:
        return pd.DataFrame()

    border_df = pd.concat(frames, axis=1).sort_index()
    return border_df[~border_df.index.duplicated(keep="last")]


def _load_neighbor_power_features(
    client,
    ts_start: pd.Timestamp,
    ts_end: pd.Timestamp,
) -> pd.DataFrame:
    """Load DE/IT load and renewable signals useful for Swiss CT forecasting."""
    frames: list[pd.DataFrame] = []

    for key, zone_code in NEIGHBOR_ZONES.items():
        try:
            load_raw = _retry(client.query_load, zone_code, start=ts_start, end=ts_end)
            if isinstance(load_raw, pd.DataFrame):
                if "Actual Load" in load_raw.columns:
                    load_df = load_raw[["Actual Load"]].rename(columns={"Actual Load": f"load_{key}_mw"})
                else:
                    load_df = load_raw.iloc[:, -1].to_frame(f"load_{key}_mw")
            else:
                load_df = load_raw.to_frame(f"load_{key}_mw")
            if load_df.index.tz is None:
                load_df.index = load_df.index.tz_localize("UTC")
            load_df = _resample_to_15min(load_df)
        except Exception as exc:
            logger.warning("ENTSO-E neighbor load failed for %s: %s", key, exc)
            load_df = pd.DataFrame()

        try:
            gen_raw = _retry(client.query_generation, zone_code, start=ts_start, end=ts_end)
            solar = _extract_generation_column(gen_raw, "Solar").rename(f"solar_{key}_mw")
            wind = (
                _extract_generation_column(gen_raw, "Wind Onshore")
                + _extract_generation_column(gen_raw, "Wind Offshore")
            ).rename(f"wind_{key}_mw")
            gen_df = pd.concat([solar, wind], axis=1)
            if gen_df.index.tz is None:
                gen_df.index = gen_df.index.tz_localize("UTC")
            gen_df = _resample_to_15min(gen_df)
        except Exception as exc:
            logger.warning("ENTSO-E neighbor generation failed for %s: %s", key, exc)
            gen_df = pd.DataFrame()

        if load_df.empty and gen_df.empty:
            continue

        zone_df = load_df.join(gen_df, how="outer").sort_index()
        for col in [f"solar_{key}_mw", f"wind_{key}_mw"]:
            if col in zone_df.columns:
                zone_df[col] = zone_df[col].fillna(0.0)

        if key == "de":
            load_col = "load_de_mw"
            solar_col = "solar_de_mw"
            wind_col = "wind_de_mw"
            if all(col in zone_df.columns for col in [load_col, solar_col, wind_col]):
                zone_df["residual_load_de_mw"] = (
                    zone_df[load_col] - zone_df[solar_col].fillna(0.0) - zone_df[wind_col].fillna(0.0)
                )

        frames.append(zone_df)

    if not frames:
        return pd.DataFrame()

    neighbor_df = pd.concat(frames, axis=1).sort_index()
    return neighbor_df[~neighbor_df.index.duplicated(keep="last")]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit le DataFrame avec solar_regime, load_deviation et flow_deviation.

    solar_regime :
        Tertiles mensuels sur solar_mw → 0=Faible, 1=Moyen, 2=Fort

    load_deviation :
        (load_mw - mean_mensuel) / std_mensuel

    flow_deviation :
        (cross_border_mw - mean_mensuel) / std_mensuel
        Capture la flexibilité hydro CH (export peak / import offpeak).
    """
    df = df.copy()

    def _solar_regime(x: pd.Series) -> pd.Series:
        q33, q66 = np.nanpercentile(x, [33, 66])
        if not np.isfinite(q33) or not np.isfinite(q66) or q33 >= q66:
            return pd.Series(1.0, index=x.index)
        return pd.cut(
            x,
            bins=[-np.inf, q33, q66, np.inf],
            labels=[0, 1, 2],
            duplicates="drop",
        ).astype(float)

    df["solar_regime"] = df.groupby(df.index.to_period("M"))["solar_mw"].transform(_solar_regime)

    monthly_mean = df.groupby(df.index.to_period("M"))["load_mw"].transform("mean")
    monthly_std = df.groupby(df.index.to_period("M"))["load_mw"].transform("std")
    df["load_deviation"] = (df["load_mw"] - monthly_mean) / monthly_std.replace(0, np.nan)

    # Flow deviation : z-score mensuel du flux transfrontalier
    if "cross_border_mw" in df.columns:
        flow_mean = df.groupby(df.index.to_period("M"))["cross_border_mw"].transform("mean")
        flow_std = df.groupby(df.index.to_period("M"))["cross_border_mw"].transform("std")
        df["flow_deviation"] = (df["cross_border_mw"] - flow_mean) / flow_std.replace(0, np.nan)
        df["flow_deviation"] = df["flow_deviation"].fillna(0)
    else:
        df["flow_deviation"] = 0.0

    for border_key in SWISS_BORDERS:
        sched_col = f"scheduled_net_export_ch_{border_key}_mw"
        if sched_col in df.columns:
            monthly = df.groupby(df.index.to_period("M"))[sched_col]
            df[f"{sched_col}_zscore"] = monthly.transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            ).fillna(0.0)

        ntc_total_col = f"ntc_total_ch_{border_key}_mw"
        if ntc_total_col in df.columns:
            monthly = df.groupby(df.index.to_period("M"))[ntc_total_col]
            df[f"{ntc_total_col}_zscore"] = monthly.transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            ).fillna(0.0)

        ntc_net_col = f"ntc_net_ch_{border_key}_mw"
        if ntc_net_col in df.columns:
            monthly = df.groupby(df.index.to_period("M"))[ntc_net_col]
            df[f"{ntc_net_col}_zscore"] = monthly.transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            ).fillna(0.0)

        export_col = f"ntc_export_ch_{border_key}_mw"
        import_col = f"ntc_import_ch_{border_key}_mw"
        if export_col in df.columns and import_col in df.columns:
            denom = (df[export_col].fillna(0.0) + df[import_col].fillna(0.0)).replace(0, np.nan)
            df[f"ntc_balance_ch_{border_key}"] = (
                (df[export_col].fillna(0.0) - df[import_col].fillna(0.0)) / denom
            ).fillna(0.0)

    if "fr_nuclear_unavailable_mw" in df.columns:
        unavailable = df["fr_nuclear_unavailable_mw"].fillna(0.0).clip(lower=0.0)
        q75 = float(unavailable.quantile(0.75)) if len(unavailable) else 0.0
        q90 = float(unavailable.quantile(0.90)) if len(unavailable) else 0.0
        scale = max(q90, 1.0)
        stress_threshold = max(5000.0, q75)
        df["fr_nuclear_unavailable_mw"] = unavailable
        df["fr_nuclear_unavailability_ratio"] = (unavailable / scale).clip(upper=1.5)
        df["fr_nuclear_stress_flag"] = (unavailable >= stress_threshold).astype(float)

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
        raw_cols = [
            c for c in existing.columns
            if c in {"load_mw", "solar_mw", "wind_mw", "cross_border_mw", "nuclear_mw", "hydro_ror_mw",
                     "hydro_reservoir_mw", "hydro_pumped_mw", "fr_nuclear_unavailable_mw"}
            or c.startswith(("load_", "solar_", "wind_", "residual_load_"))
            or c.startswith(("scheduled_", "flow_net_export_", "ntc_export_", "ntc_import_", "ntc_net_", "ntc_total_"))
        ]
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


def fetch_and_cache_de_renewable_forecast(
    start: str,
    end: str,
    parquet_path: str | Path | None = None,
    country_code: str = "DE_LU",
) -> pd.DataFrame:
    """
    Refresh the cached DE day-ahead renewable forecast dataset used by Swiss CT.

    Fresh values win on overlapping timestamps; older cached rows are preserved
    outside the refreshed window.
    """
    new = load_de_renewable_forecast(start, end, country_code=country_code)

    parquet_path = Path(parquet_path) if parquet_path is not None else resolve_local_cache_path(
        Path(__file__).resolve().parents[2], "ct_forecast_de_renewables_15min"
    )
    if parquet_path is None:
        raise ValueError("Missing cache path for ct_forecast_de_renewables_15min")
    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        combined = new.combine_first(existing).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = new

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    logger.info(
        "Cache DE renewable forecast updated: %s (%d rows, max_ts=%s)",
        parquet_path,
        len(combined),
        combined.index.max() if len(combined) else None,
    )
    return combined


def fetch_and_cache_multi_country_forecasts(
    start: str,
    end: str,
    parquet_path: str | Path | None = None,
    zones: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Refresh the cached multi-country day-ahead forecast dataset for Swiss CT.

    The cache stores forecast-for timestamps on a normalized 15min UTC grid.
    """
    new = load_multi_country_forecasts(start, end, zones=zones)
    parquet_path = Path(parquet_path) if parquet_path is not None else resolve_local_cache_path(
        Path(__file__).resolve().parents[2], "ct_forecast_multi_country_15min"
    )
    if parquet_path is None:
        raise ValueError("Missing cache path for ct_forecast_multi_country_15min")

    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        combined = new.combine_first(existing).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = new

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    logger.info(
        "Cache multi-country forecast updated: %s (%d rows, max_ts=%s)",
        parquet_path,
        len(combined),
        combined.index.max() if len(combined) else None,
    )
    return combined


def fetch_and_cache_fr_nuclear_forecast(
    start: str,
    end: str,
    parquet_path: str | Path | None = None,
) -> pd.DataFrame:
    """Refresh the cached FR nuclear future-availability dataset for Swiss CT."""
    new = load_fr_nuclear_forecast(start, end)
    parquet_path = Path(parquet_path) if parquet_path is not None else resolve_local_cache_path(
        Path(__file__).resolve().parents[2], "ct_forecast_fr_nuclear_15min"
    )
    if parquet_path is None:
        raise ValueError("Missing cache path for ct_forecast_fr_nuclear_15min")

    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        combined = new.combine_first(existing).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = new

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    logger.info(
        "Cache FR nuclear forecast updated: %s (%d rows, max_ts=%s)",
        parquet_path,
        len(combined),
        combined.index.max() if len(combined) else None,
    )
    return combined

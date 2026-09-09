"""
ingest_entso.py
---------------
Ingestion legacy des données réseau et renouvelables ENTSO-E (via entsoe-py).

La cadence source est préservée. Une série horaire reste horaire ; ce module
ne fabrique jamais quatre quarts d'heure par forward-fill. Dans un DataFrame
mixte, les timestamps absents restent ``NaN`` et ne deviennent jamais des
zéros économiques. L'admission modèle exige séparément le sidecar de régimes
de cadence effectifs D260/D261.

Clé API : variable d'environnement ENTSOE_API_KEY (ou fichier .env à la racine).

Format de sortie legacy local (Parquet local) :
    index : union triée des timestamps natifs, en UTC
    colonnes :
        load_mw         — charge totale CH [MW]
        solar_mw        — production solaire CH [MW]
        wind_mw         — production éolienne CH [MW]
        solar_regime    — feature causale dérivée par le transform versionné
        load_deviation  — feature causale dérivée par le transform versionné

Le nom historique ``entso_15min.parquet`` est conservé uniquement pour
compatibilité de chemin ; il ne prouve aucune granularité.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_PARQUET = Path(__file__).resolve().parent.parent / "data" / "entso_15min.parquet"

# Charger .env depuis la racine du repo
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

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
        index : union UTC des timestamps natifs, sans upsampling
    """
    client = _get_client()

    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")

    logger.info("ENTSO-E API load + generation : %s → %s (zone=%s)", start, end, country_code)

    # --- Load ---
    df_load_raw = _retry(client.query_load, country_code, start=ts_start, end=ts_end)
    df_load = _extract_actual_load(df_load_raw, "load_mw")

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

    # --- Préserver les grilles natives et joindre sur leur union ---
    for df_ in [df_load, df_gen]:
        _normalize_native_frame_in_place(df_)

    df = df_load.join(df_gen, how="outer").sort_index()

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
        return pd.Series(index=df_gen.index, dtype=float)

    fuel_norm = str(fuel_type).strip().lower()

    # Multi-level columns: ('Solar', 'Actual Aggregated'), etc.
    if isinstance(df_gen.columns, pd.MultiIndex):
        matching = [col for col in df_gen.columns if str(col[0]).strip().lower() == fuel_norm]
        if not matching:
            return pd.Series(index=df_gen.index, dtype=float)
        # Préférer 'Actual Aggregated' sur 'Actual Consumption'
        for col in matching:
            if "Aggregated" in str(col[1]):
                return df_gen[col].astype(float)
        return df_gen[matching[0]].astype(float)

    # Flat columns
    matching = [c for c in df_gen.columns if fuel_norm in str(c).strip().lower()]
    if not matching:
        return pd.Series(index=df_gen.index, dtype=float)
    return df_gen[matching[0]].astype(float)


def _extract_actual_load(
    raw: pd.Series | pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    """Select actual load explicitly; reject ambiguous multi-column responses."""

    if isinstance(raw, pd.Series):
        return raw.rename(name).to_frame()
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("ENTSO-E load response must be a Series or DataFrame")
    if "Actual Load" in raw.columns:
        return raw[["Actual Load"]].rename(columns={"Actual Load": name})
    if raw.shape[1] == 1:
        return raw.iloc[:, 0].rename(name).to_frame()
    raise ValueError("ENTSO-E load response has no unambiguous Actual Load column")


def _normalize_native_frame_in_place(frame: pd.DataFrame) -> None:
    """Normalize only timezone/order; never change the source observation grid."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("ENTSO-E source requires a DatetimeIndex")
    if frame.index.has_duplicates:
        raise ValueError("ENTSO-E source timestamps must be unique")
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame.sort_index(inplace=True)


def _normalize_native_series(series: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    """Return one named UTC series while preserving its exact native timestamps."""

    if isinstance(series, pd.DataFrame):
        if series.empty:
            return pd.Series(dtype=float, name=name)
        if series.shape[1] != 1:
            raise ValueError(f"ENTSO-E {name} response has ambiguous columns")
        series = series.iloc[:, 0]

    frame = series.rename(name).to_frame()
    _normalize_native_frame_in_place(frame)
    return frame[name].astype(float)


def _query_series_or_empty(func, *args, name: str, **kwargs) -> pd.Series:
    """Run ENTSO-E query and degrade gracefully if a border is unavailable."""
    try:
        raw = _retry(func, *args, **kwargs)
        return _normalize_native_series(raw, name)
    except Exception as exc:
        logger.warning("ENTSO-E query failed for %s: %s", name, exc)
        return pd.Series(dtype=float, name=name)


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
        if not sched_export.empty:
            frames.append(sched_export)
        if not sched_import.empty:
            frames.append(sched_import)
        if not sched_export.empty and not sched_import.empty:
            frames.append(
                (sched_export - sched_import).rename(
                    f"scheduled_net_export_ch_{border_key}_mw"
                )
            )

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
        if not flow_export.empty:
            frames.append(flow_export)
        if not flow_import.empty:
            frames.append(flow_import)
        if not flow_export.empty and not flow_import.empty:
            frames.append(
                (flow_export - flow_import).rename(
                    f"flow_net_export_ch_{border_key}_mw"
                )
            )

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
        if not ntc_export.empty:
            frames.append(ntc_export)
        if not ntc_import.empty:
            frames.append(ntc_import)
        if not ntc_export.empty and not ntc_import.empty:
            frames.extend(
                [
                    (ntc_export - ntc_import).rename(
                        f"ntc_net_ch_{border_key}_mw"
                    ),
                    (ntc_export + ntc_import).rename(
                        f"ntc_total_ch_{border_key}_mw"
                    ),
                ]
            )

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
            load_df = _extract_actual_load(load_raw, f"load_{key}_mw")
            _normalize_native_frame_in_place(load_df)
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
            _normalize_native_frame_in_place(gen_df)
        except Exception as exc:
            logger.warning("ENTSO-E neighbor generation failed for %s: %s", key, exc)
            gen_df = pd.DataFrame()

        if load_df.empty and gen_df.empty:
            continue

        zone_df = load_df.join(gen_df, how="outer").sort_index()

        if key == "de":
            load_col = "load_de_mw"
            solar_col = "solar_de_mw"
            wind_col = "wind_de_mw"
            if all(col in zone_df.columns for col in [load_col, solar_col, wind_col]):
                zone_df["residual_load_de_mw"] = (
                    zone_df[load_col] - zone_df[solar_col] - zone_df[wind_col]
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
    from pfc_shaping.data.lt_replay_transforms import build_entso_features

    return build_entso_features(df)


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
    Télécharge une capture ENTSO-E neuve et la sauvegarde localement.

    Le chemin doit être absent. Un ancien cache peut contenir des quarts
    d'heure fabriqués par l'implémentation historique et ne peut pas être
    distingué d'une capture native sans sidecar D260/D261. La fusion échoue
    donc avant tout appel réseau ; utiliser un nouveau chemin immuable.

    Returns:
        DataFrame canonique complet mis à jour
    """
    parquet_path = Path(parquet_path)
    if parquet_path.exists():
        raise FileExistsError(
            "existing ENTSO-E cache has unproven native cadence; "
            "immutable capture path must be new"
        )

    combined = load_from_api(start, end, country_code)

    combined = build_features(combined)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    logger.info("Cache ENTSO-E mis à jour : %s (%d lignes)", parquet_path, len(combined))
    return combined

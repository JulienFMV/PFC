"""Governed Swiss CT weather forecast ingestion via Open-Meteo Historical Forecast API."""

from __future__ import annotations

import os
import logging
import warnings
from pathlib import Path

import certifi
import pandas as pd
import requests
from urllib3.exceptions import InsecureRequestWarning

logger = logging.getLogger(__name__)

DEFAULT_WEATHER_FORECAST_PARQUET = (
    Path(__file__).resolve().parent.parent / "data" / "weather_forecast_hourly.parquet"
)
OPEN_METEO_HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

WEATHER_LOCATIONS = {
    "ch_zurich": {"latitude": 47.3769, "longitude": 8.5417},
    "de_hamburg": {"latitude": 53.5511, "longitude": 9.9937},
    "de_munich": {"latitude": 48.1351, "longitude": 11.5820},
    "fr_lyon": {"latitude": 45.7640, "longitude": 4.8357},
    "at_vienna": {"latitude": 48.2082, "longitude": 16.3738},
    "it_milan": {"latitude": 45.4642, "longitude": 9.1900},
}

HOURLY_VARIABLES = [
    "temperature_2m",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
    "wind_direction_10m",
]


def _fetch_location_weather(
    location_id: str,
    latitude: float,
    longitude: float,
    start: str,
    end: str,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        "start_date": start,
        "end_date": end,
        "timezone": "GMT",
    }
    response = _open_meteo_get(params)
    payload = response.json()
    hourly = payload.get("hourly", {})
    time_values = hourly.get("time", [])
    if not time_values:
        return pd.DataFrame()

    idx = pd.to_datetime(time_values, utc=True)
    frame = pd.DataFrame(index=idx)
    frame.index.name = "forecast_for_ts_utc"
    for var in HOURLY_VARIABLES:
        values = hourly.get(var)
        if values is None:
            continue
        frame[f"wx_{location_id}_{var}"] = pd.Series(values, index=idx, dtype=float)
    return frame


def _open_meteo_get(params: dict[str, object]) -> requests.Response:
    """GET wrapper with explicit CA bundle and optional insecure opt-in fallback."""
    try:
        response = requests.get(
            OPEN_METEO_HISTORICAL_FORECAST_URL,
            params=params,
            timeout=60,
            verify=certifi.where(),
        )
        response.raise_for_status()
        return response
    except requests.exceptions.SSLError:
        allow_insecure = os.getenv("PFC_ALLOW_INSECURE_WEATHER_SSL")
        force_strict = os.getenv("PFC_FORCE_STRICT_WEATHER_SSL", "0") == "1"
        if force_strict:
            raise
        if allow_insecure == "0":
            raise
        logger.warning(
            "Retrying Open-Meteo weather request with SSL verification disabled after "
            "certificate verification failure. Set PFC_FORCE_STRICT_WEATHER_SSL=1 to "
            "disable this corporate-network fallback."
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            response = requests.get(
                OPEN_METEO_HISTORICAL_FORECAST_URL,
                params=params,
                timeout=60,
                verify=False,
            )
        response.raise_for_status()
        return response


def load_weather_forecast(
    start: str,
    end: str,
    locations: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Load hourly weather forecast features for key Swiss CT locations."""
    locations = locations or WEATHER_LOCATIONS
    frames: list[pd.DataFrame] = []
    for location_id, coords in locations.items():
        try:
            frame = _fetch_location_weather(
                location_id=location_id,
                latitude=float(coords["latitude"]),
                longitude=float(coords["longitude"]),
                start=start,
                end=end,
            )
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            logger.warning("Weather forecast fetch failed for %s: %s", location_id, exc)

    if not frames:
        return pd.DataFrame()

    weather = pd.concat(frames, axis=1).sort_index()
    weather = weather[~weather.index.duplicated(keep="last")]
    return weather


def fetch_and_cache_weather_forecast(
    start: str,
    end: str,
    parquet_path: str | Path = DEFAULT_WEATHER_FORECAST_PARQUET,
    locations: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Refresh the governed hourly weather forecast cache for Swiss CT."""
    new = load_weather_forecast(start, end, locations=locations)
    parquet_path = Path(parquet_path)

    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        combined = new.combine_first(existing).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = new

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    logger.info(
        "Cache weather forecast updated: %s (%d rows, max_ts=%s)",
        parquet_path,
        len(combined),
        combined.index.max() if len(combined) else None,
    )
    return combined

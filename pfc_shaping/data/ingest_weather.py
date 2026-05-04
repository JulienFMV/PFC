"""Open-Meteo hourly weather ingestion for short-term power forecasting."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_PARQUET = Path(__file__).resolve().parent / "weather_open_meteo_hourly.parquet"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
]

WEATHER_LOCATIONS = {
    "CH_Zurich": {"group": "CH", "lat": 47.3769, "lon": 8.5417, "weight": 0.30},
    "CH_Bern": {"group": "CH", "lat": 46.9480, "lon": 7.4474, "weight": 0.25},
    "CH_Geneva": {"group": "CH", "lat": 46.2044, "lon": 6.1432, "weight": 0.20},
    "CH_Lugano": {"group": "CH", "lat": 46.0037, "lon": 8.9511, "weight": 0.15},
    "CH_Visp": {"group": "CH", "lat": 46.2932, "lon": 7.8819, "weight": 0.10},
    "DE_Freiburg": {"group": "DE", "lat": 47.9990, "lon": 7.8421, "weight": 0.35},
    "DE_Stuttgart": {"group": "DE", "lat": 48.7758, "lon": 9.1829, "weight": 0.35},
    "DE_Munich": {"group": "DE", "lat": 48.1351, "lon": 11.5820, "weight": 0.30},
}


def _open_json_url(base_url: str, params: dict[str, object]) -> dict:
    query = urlencode(params, doseq=True)
    with urlopen(f"{base_url}?{query}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _hourly_payload_to_frame(payload: dict, location: str) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    if df.empty:
        return df
    df["location"] = location
    return df


def _fetch_location_archive(location: str, meta: dict, start_date: date, end_date: date) -> pd.DataFrame:
    payload = _open_json_url(
        "https://archive-api.open-meteo.com/v1/archive",
        {
            "latitude": meta["lat"],
            "longitude": meta["lon"],
            "hourly": HOURLY_VARS,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "timezone": "UTC",
        },
    )
    return _hourly_payload_to_frame(payload, location)


def _fetch_location_forecast(location: str, meta: dict, forecast_days: int) -> pd.DataFrame:
    payload = _open_json_url(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": meta["lat"],
            "longitude": meta["lon"],
            "hourly": HOURLY_VARS,
            "forecast_days": int(forecast_days),
            "timezone": "UTC",
        },
    )
    return _hourly_payload_to_frame(payload, location)


def _weighted_group_aggregate(df: pd.DataFrame, group: str) -> pd.DataFrame:
    group_locs = {name: meta for name, meta in WEATHER_LOCATIONS.items() if meta["group"] == group}
    pieces = []
    for location, meta in group_locs.items():
        loc = df[df["location"] == location].copy()
        if loc.empty:
            continue
        weight = float(meta["weight"])
        value_cols = [c for c in HOURLY_VARS if c in loc.columns]
        for col in value_cols:
            loc[col] = pd.to_numeric(loc[col], errors="coerce") * weight
        loc["weight"] = weight
        pieces.append(loc[["location", "weight", *value_cols]])

    if not pieces:
        return pd.DataFrame()

    stacked = pd.concat(pieces).sort_index()
    value_cols = [c for c in HOURLY_VARS if c in stacked.columns]
    grouped = stacked.groupby(stacked.index)
    agg = grouped[value_cols].sum()
    weights = grouped["weight"].sum().replace(0, pd.NA)
    agg = agg.div(weights, axis=0)

    prefix = group.lower()
    out = pd.DataFrame(index=agg.index)
    if "temperature_2m" in agg.columns:
        out[f"temp_{prefix}_c"] = agg["temperature_2m"]
        out[f"hdd_{prefix}"] = (18.0 - agg["temperature_2m"]).clip(lower=0.0)
        out[f"cdd_{prefix}"] = (agg["temperature_2m"] - 22.0).clip(lower=0.0)
    if "relative_humidity_2m" in agg.columns:
        out[f"humidity_{prefix}_pct"] = agg["relative_humidity_2m"]
    if "cloud_cover" in agg.columns:
        out[f"cloud_{prefix}_pct"] = agg["cloud_cover"]
    if "wind_speed_10m" in agg.columns:
        out[f"wind_{prefix}_ms"] = agg["wind_speed_10m"]
    if "wind_gusts_10m" in agg.columns:
        out[f"gust_{prefix}_ms"] = agg["wind_gusts_10m"]
    if "shortwave_radiation" in agg.columns:
        out[f"radiation_{prefix}_wm2"] = agg["shortwave_radiation"]
    return out


def _derive_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"temp_ch_c", "temp_de_c"}.issubset(out.columns):
        out["temp_spread_ch_minus_de_c"] = out["temp_ch_c"] - out["temp_de_c"]
    if {"hdd_ch", "hdd_de"}.issubset(out.columns):
        out["hdd_spread_ch_minus_de"] = out["hdd_ch"] - out["hdd_de"]
    if {"radiation_ch_wm2", "radiation_de_wm2"}.issubset(out.columns):
        out["radiation_spread_ch_minus_de_wm2"] = out["radiation_ch_wm2"] - out["radiation_de_wm2"]
    return out


def fetch_and_cache(start: str, end: str, parquet_path: str | Path = DEFAULT_PARQUET) -> pd.DataFrame:
    """Fetch hourly historical + forecast weather and cache it to parquet."""
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    today_utc = datetime.now(timezone.utc).date()
    yesterday_utc = today_utc - timedelta(days=1)

    frames = []
    for location, meta in WEATHER_LOCATIONS.items():
        loc_frames = []
        if start_date <= yesterday_utc:
            archive_end = min(end_date, yesterday_utc)
            if archive_end >= start_date:
                loc_frames.append(_fetch_location_archive(location, meta, start_date, archive_end))
        if end_date >= today_utc:
            forecast_start = max(start_date, today_utc)
            forecast_days = (end_date - forecast_start).days + 1
            if forecast_days > 0:
                forecast = _fetch_location_forecast(location, meta, forecast_days)
                if not forecast.empty:
                    forecast = forecast[forecast.index >= pd.Timestamp(forecast_start, tz="UTC")]
                loc_frames.append(forecast)

        loc_df = pd.concat([f for f in loc_frames if not f.empty]).sort_index() if loc_frames else pd.DataFrame()
        if loc_df.empty:
            logger.warning("Weather location %s returned no data", location)
            continue
        loc_df = loc_df[~loc_df.index.duplicated(keep="last")]
        frames.append(loc_df)

    if not frames:
        raise RuntimeError("Weather ingestion returned no rows from Open-Meteo")

    raw = pd.concat(frames).sort_index()
    ch = _weighted_group_aggregate(raw, "CH")
    de = _weighted_group_aggregate(raw, "DE")
    weather = pd.concat([ch, de], axis=1).sort_index()
    weather = _derive_cross_features(weather)
    weather = weather[~weather.index.duplicated(keep="last")]
    weather = weather.sort_index()

    path = Path(parquet_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(path)
    logger.info("Weather cached to %s (%d rows, %d cols)", path, len(weather), len(weather.columns))
    return weather


def load_parquet(parquet_path: str | Path = DEFAULT_PARQUET) -> pd.DataFrame:
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()

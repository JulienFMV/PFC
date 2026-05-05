"""DWD ICON weather ingestion for short-term power forecasting.

This module is intended for production-grade, issue-time-safe weather inputs
from the DWD Open Data server. It focuses on a compact set of physically
meaningful variables and interpolates them over representative CH/DE points.

Scope:
- ICON-D2 for forecast hours 0..48
- ICON-EU for forecast hours 49..120

This is primarily a live/prod ingestion path. Deep historical backfills still
require a local archive or DWD Pamore access.
"""

from __future__ import annotations

import bz2
import json
import logging
import math
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import xarray as xr
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_PARQUET = Path(__file__).resolve().parent / "weather_dwd_hourly.parquet"
DEFAULT_METADATA = Path(__file__).resolve().parent / "weather_dwd_hourly.metadata.json"

VERIFY_SSL = False

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

VAR_SPECS = {
    "t_2m": {"out": "temp_c", "units": "K_to_C"},
    "clct": {"out": "cloud_pct"},
    "u_10m": {"out": "u10_ms"},
    "v_10m": {"out": "v10_ms"},
    "aswdir_s": {"out": "rad_direct_wm2"},
    "aswdifd_s": {"out": "rad_diffuse_wm2"},
    "h_snow": {"out": "snow_depth_m"},
}

MODEL_CONFIG = {
    "icon-d2": {
        "dir": "icon-d2",
        "horizons": range(0, 49),
        "run_hours": [0, 3, 6, 9, 12, 15, 18, 21],
    },
    "icon-eu": {
        "dir": "icon-eu",
        "horizons": list(range(49, 79)) + list(range(81, 121, 3)),
        "run_hours": [0, 6, 12, 18],
    },
}


def _make_session() -> requests.Session:
    session = requests.Session()
    session.verify = VERIFY_SSL
    return session


def _latest_run_timestamp(now_utc: datetime, run_hours: list[int], lag_hours: int = 4) -> datetime:
    anchor = now_utc - timedelta(hours=lag_hours)
    for hour in sorted(run_hours, reverse=True):
        candidate = anchor.replace(hour=hour, minute=0, second=0, microsecond=0)
        if candidate <= anchor:
            return candidate
    prev_day = (anchor - timedelta(days=1)).date()
    return datetime(prev_day.year, prev_day.month, prev_day.day, max(run_hours), tzinfo=timezone.utc)


def _list_variable_files(session: requests.Session, model_dir: str, run_hour: int, var_dir: str) -> list[str]:
    url = f"https://opendata.dwd.de/weather/nwp/{model_dir}/grib/{run_hour:02d}/{var_dir}/"
    response = session.get(url, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    files = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if not href or href.endswith("/") or not href.endswith(".grib2.bz2"):
            continue
        if "regular-lat-lon" not in href:
            continue
        files.append(href)
    return sorted(files)


def _select_file_for_horizon(files: list[str], run_stamp: datetime, horizon_hour: int) -> str:
    run_token = run_stamp.strftime("%Y%m%d%H")
    horizon_token = f"_{horizon_hour:03d}_"
    for name in files:
        if run_token in name and horizon_token in name:
            return name
    raise FileNotFoundError(f"No GRIB file for run={run_token} horizon={horizon_hour:03d}")


def _download_and_open_dataset(
    session: requests.Session,
    model_dir: str,
    run_hour: int,
    var_dir: str,
    filename: str,
) -> xr.Dataset:
    url = f"https://opendata.dwd.de/weather/nwp/{model_dir}/grib/{run_hour:02d}/{var_dir}/{filename}"
    response = session.get(url, timeout=120)
    response.raise_for_status()
    tmp = Path(tempfile.gettempdir()) / filename.replace(".bz2", "")
    tmp.write_bytes(bz2.decompress(response.content))
    return xr.open_dataset(tmp, engine="cfgrib")


def _interpolate_points(ds: xr.Dataset, out_name: str) -> pd.DataFrame:
    var_name = next(iter(ds.data_vars))
    arr = ds[var_name]
    valid_time_values = pd.to_datetime(ds["valid_time"].values, utc=True)
    if getattr(valid_time_values, "ndim", 0) > 0:
        valid_time = pd.Timestamp(valid_time_values[0]).floor("h")
    else:
        valid_time = pd.Timestamp(valid_time_values).floor("h")
    if "step" in arr.dims:
        arr = arr.mean(dim="step")
    rows = []
    for location, meta in WEATHER_LOCATIONS.items():
        value = float(arr.interp(latitude=meta["lat"], longitude=meta["lon"]).values)
        rows.append(
            {
                "valid_time": valid_time,
                "location": location,
                "group": meta["group"],
                "weight": float(meta["weight"]),
                out_name: value,
            }
        )
    return pd.DataFrame(rows)


def _fetch_model_horizon_block(
    session: requests.Session,
    *,
    model_key: str,
    run_stamp: datetime,
    horizon_hours: list[int],
) -> pd.DataFrame:
    config = MODEL_CONFIG[model_key]
    model_dir = config["dir"]
    run_hour = run_stamp.hour
    listings = {var_dir: _list_variable_files(session, model_dir, run_hour, var_dir) for var_dir in VAR_SPECS}

    blocks: list[pd.DataFrame] = []
    for horizon_hour in horizon_hours:
        merged: pd.DataFrame | None = None
        for var_dir, spec in VAR_SPECS.items():
            filename = _select_file_for_horizon(listings[var_dir], run_stamp, horizon_hour)
            ds = _download_and_open_dataset(session, model_dir, run_hour, var_dir, filename)
            block = _interpolate_points(ds, spec["out"])
            if merged is None:
                merged = block
            else:
                merged = merged.merge(block, on=["valid_time", "location", "group", "weight"], how="outer")
        if merged is not None:
            merged["source_model"] = model_key
            merged["run_time_utc"] = run_stamp
            merged["horizon_hour"] = int(horizon_hour)
            blocks.append(merged)
    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True)


def _postprocess_point_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "temp_c" in out.columns:
        out["temp_c"] = out["temp_c"] - 273.15
        out["hdd"] = (18.0 - out["temp_c"]).clip(lower=0.0)
        out["cdd"] = (out["temp_c"] - 22.0).clip(lower=0.0)
    if {"u10_ms", "v10_ms"}.issubset(out.columns):
        out["wind_ms"] = (out["u10_ms"] ** 2 + out["v10_ms"] ** 2).pow(0.5)
    if {"rad_direct_wm2", "rad_diffuse_wm2"}.issubset(out.columns):
        out["radiation_wm2"] = out["rad_direct_wm2"] + out["rad_diffuse_wm2"]
    return out


def _weighted_group_aggregate(df: pd.DataFrame, group: str) -> pd.DataFrame:
    subset = df[df["group"] == group].copy()
    if subset.empty:
        return pd.DataFrame()
    feature_cols = [
        c
        for c in subset.columns
        if c not in {"valid_time", "location", "group", "weight", "source_model", "run_time_utc", "horizon_hour"}
    ]
    for col in feature_cols:
        subset[col] = pd.to_numeric(subset[col], errors="coerce") * subset["weight"]
    grouped = subset.groupby("valid_time")
    agg = grouped[feature_cols].sum()
    weights = grouped["weight"].sum().replace(0, pd.NA)
    agg = agg.div(weights, axis=0)

    prefix = group.lower()
    out = pd.DataFrame(index=agg.index)
    rename_map = {
        "temp_c": f"temp_{prefix}_c",
        "hdd": f"hdd_{prefix}",
        "cdd": f"cdd_{prefix}",
        "cloud_pct": f"cloud_{prefix}_pct",
        "wind_ms": f"wind_{prefix}_ms",
        "radiation_wm2": f"radiation_{prefix}_wm2",
        "snow_depth_m": f"snow_depth_{prefix}_m",
    }
    for src, dst in rename_map.items():
        if src in agg.columns:
            out[dst] = agg[src]
    return out


def _derive_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"temp_ch_c", "temp_de_c"}.issubset(out.columns):
        out["temp_spread_ch_minus_de_c"] = out["temp_ch_c"] - out["temp_de_c"]
    if {"hdd_ch", "hdd_de"}.issubset(out.columns):
        out["hdd_spread_ch_minus_de"] = out["hdd_ch"] - out["hdd_de"]
    if {"radiation_ch_wm2", "radiation_de_wm2"}.issubset(out.columns):
        out["radiation_spread_ch_minus_de_wm2"] = out["radiation_ch_wm2"] - out["radiation_de_wm2"]
    if {"wind_ch_ms", "wind_de_ms"}.issubset(out.columns):
        out["wind_spread_ch_minus_de_ms"] = out["wind_ch_ms"] - out["wind_de_ms"]
    if {"snow_depth_ch_m", "snow_depth_de_m"}.issubset(out.columns):
        out["snow_depth_spread_ch_minus_de_m"] = out["snow_depth_ch_m"] - out["snow_depth_de_m"]
    return out


def _write_metadata(path: Path, *, run_d2: datetime, run_eu: datetime, rows: int, cols: int) -> None:
    payload = {
        "source": "dwd-open-data",
        "models": {
            "icon-d2_run_utc": run_d2.isoformat(),
            "icon-eu_run_utc": run_eu.isoformat(),
        },
        "variables": list(VAR_SPECS),
        "locations": WEATHER_LOCATIONS,
        "rows": rows,
        "cols": cols,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fetch_and_cache(
    *,
    parquet_path: str | Path = DEFAULT_PARQUET,
    metadata_path: str | Path = DEFAULT_METADATA,
    now_utc: datetime | None = None,
) -> pd.DataFrame:
    """Fetch the latest DWD ICON-D2/EU weather forecasts and cache them to parquet."""
    now_utc = now_utc or datetime.now(timezone.utc)
    session = _make_session()
    run_d2 = _latest_run_timestamp(now_utc, MODEL_CONFIG["icon-d2"]["run_hours"], lag_hours=3)
    run_eu = _latest_run_timestamp(now_utc, MODEL_CONFIG["icon-eu"]["run_hours"], lag_hours=4)

    d2 = _fetch_model_horizon_block(
        session,
        model_key="icon-d2",
        run_stamp=run_d2,
        horizon_hours=list(MODEL_CONFIG["icon-d2"]["horizons"]),
    )
    eu = _fetch_model_horizon_block(
        session,
        model_key="icon-eu",
        run_stamp=run_eu,
        horizon_hours=list(MODEL_CONFIG["icon-eu"]["horizons"]),
    )
    raw = pd.concat([d2, eu], ignore_index=True)
    if raw.empty:
        raise RuntimeError("DWD weather ingestion returned no rows")

    raw = _postprocess_point_values(raw)
    ch = _weighted_group_aggregate(raw, "CH")
    de = _weighted_group_aggregate(raw, "DE")
    weather = pd.concat([ch, de], axis=1).sort_index()
    weather = _derive_cross_features(weather)
    weather = weather[~weather.index.duplicated(keep="last")].sort_index()
    weather = weather.resample("h").interpolate(method="time").ffill().bfill()

    path = Path(parquet_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(path)
    _write_metadata(Path(metadata_path), run_d2=run_d2, run_eu=run_eu, rows=len(weather), cols=len(weather.columns))
    logger.info("DWD weather cached to %s (%d rows, %d cols)", path, len(weather), len(weather.columns))
    return weather


def load_parquet(parquet_path: str | Path = DEFAULT_PARQUET) -> pd.DataFrame:
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()

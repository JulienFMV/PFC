"""Enrich governed electrification scenarios with explicit proxy assumptions.

The input scenario table remains the source of truth for official fields. This
script only fills missing recommended production columns, stamps the resulting
rows with an explicit assumption publication date, and marks quality flags as
proxy-enriched. It is intended for local production-readiness smoke runs while
TYNDP/MaStR/Pronovo/NTC feeds are being connected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.electrification_acquisition import (
    DEFAULT_ELECTRIFICATION_SCENARIO_PATH,
)
from pfc_shaping.data.electrification_scenarios import (
    RECOMMENDED_COLUMNS,
    validate_electrification_scenarios,
)

PROFILE_NAME = "ch_first_pfc_proxy_v0"

_SCHEDULES = {
    "battery_power_gw": {2025: 0.4, 2030: 2.0, 2035: 3.5},
    "battery_energy_gwh": {2025: 0.8, 2030: 5.0, 2035: 10.0},
    "managed_charging_share": {2025: 0.20, 2030: 0.35, 2035: 0.50},
    "ntc_ch_de_gw": {2025: 4.0, 2035: 4.0},
    "ntc_ch_fr_gw": {2025: 4.0, 2035: 4.0},
    "ntc_ch_it_gw": {2025: 3.0, 2035: 3.0},
    "dispatchable_gw": {2025: 0.0, 2035: 0.0},
    "gas_gw": {2025: 0.0, 2035: 0.0},
    "coal_gw": {2025: 0.0, 2035: 0.0},
}


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def _interp(year: int, schedule: dict[int, float]) -> float:
    xs = np.array(sorted(schedule), dtype=float)
    ys = np.array([schedule[int(x)] for x in xs], dtype=float)
    return float(np.interp(float(year), xs, ys))


def _fill_if_missing(df: pd.DataFrame, column: str, values: pd.Series) -> None:
    if column not in df.columns:
        df[column] = np.nan
    mask = df[column].isna()
    if mask.any():
        df.loc[mask, column] = values.loc[mask].to_numpy()


def _ensure_source(df: pd.DataFrame) -> None:
    if "source" not in df.columns:
        df["source"] = "scenario"
    df["source"] = df["source"].fillna("scenario").astype(str)


def _winter_demand_from_hourly(hourly_path: str | Path | None) -> pd.DataFrame:
    if hourly_path is None:
        return pd.DataFrame(columns=["scenario", "delivery_year", "winter_demand_twh"])
    hourly = pd.read_parquet(hourly_path)
    needed = {"scenario", "year", "datetime", "technology", "gwh"}
    if not needed <= set(hourly.columns):
        missing = sorted(needed - set(hourly.columns))
        raise KeyError(f"hourly EP2050 table missing columns {missing}")
    sub = hourly[hourly["technology"].astype(str) == "total_consumption"].copy()
    dt = pd.to_datetime(sub["datetime"])
    sub = sub[dt.dt.month.isin([1, 2, 12])].copy()
    out = (
        sub.groupby(["scenario", "year"], as_index=False)["gwh"]
        .sum()
        .rename(columns={"year": "delivery_year"})
    )
    out["winter_demand_twh"] = out["gwh"].astype(float) / 1000.0
    return out[["scenario", "delivery_year", "winter_demand_twh"]]


def enrich_scenarios(
    frame: pd.DataFrame,
    *,
    assumption_publication_date: str | pd.Timestamp,
    hourly_path: str | Path | None = None,
    profile: str = PROFILE_NAME,
) -> pd.DataFrame:
    """Fill missing recommended fields with explicit CH proxy assumptions."""
    if profile != PROFILE_NAME:
        raise ValueError(f"unknown enrichment profile {profile!r}")
    df = validate_electrification_scenarios(frame).copy()
    _ensure_source(df)
    assumption_ts = pd.Timestamp(assumption_publication_date)
    if assumption_ts.tz is None:
        assumption_ts = assumption_ts.tz_localize("UTC")
    else:
        assumption_ts = assumption_ts.tz_convert("UTC")

    years = df["delivery_year"].astype(int)
    for column, schedule in _SCHEDULES.items():
        _fill_if_missing(df, column, years.map(lambda year: _interp(int(year), schedule)).astype(float))

    winter = _winter_demand_from_hourly(hourly_path)
    if not winter.empty:
        df = df.merge(winter, on=["scenario", "delivery_year"], how="left", suffixes=("", "_from_hourly"))
        if "winter_demand_twh_from_hourly" in df.columns:
            _fill_if_missing(df, "winter_demand_twh", df["winter_demand_twh_from_hourly"])
            df = df.drop(columns=["winter_demand_twh_from_hourly"])
    if "winter_demand_twh" not in df.columns:
        df["winter_demand_twh"] = np.nan
    _fill_if_missing(df, "winter_demand_twh", pd.to_numeric(df["demand_twh"], errors="coerce") * 0.28)

    if "hydro_reservoir_twh" not in df.columns:
        df["hydro_reservoir_twh"] = np.nan
    hydro_proxy = pd.to_numeric(df.get("winter_demand_twh", 0.0), errors="coerce").fillna(0.0) * 0.20
    _fill_if_missing(df, "hydro_reservoir_twh", hydro_proxy.clip(lower=0.0, upper=12.0))

    if "scenario_weight" not in df.columns:
        df["scenario_weight"] = np.nan
    group_cols = ["publication_date", "country", "delivery_year"]
    if "delivery_month" in df.columns:
        group_cols.append("delivery_month")
    weights = 1.0 / df.groupby(group_cols, dropna=False)["scenario"].transform("nunique").astype(float)
    _fill_if_missing(df, "scenario_weight", weights)

    if "delivery_month" not in df.columns:
        df["delivery_month"] = pd.NA
    if "quality_flag" not in df.columns:
        df["quality_flag"] = "official"
    df["quality_flag"] = df["quality_flag"].fillna("official").astype(str).map(
        lambda value: value if "proxy_enriched" in value else f"{value}_proxy_enriched"
    )
    df["assumption_profile"] = profile
    df["source"] = df["source"].map(
        lambda value: value if profile in value else f"{value}+{profile}"
    )
    base_publication = pd.to_datetime(df["publication_date"], utc=True)
    df["publication_date"] = pd.Series(
        np.maximum(base_publication.to_numpy(dtype="datetime64[ns]"), assumption_ts.to_datetime64()),
        index=df.index,
    ).dt.tz_localize("UTC")

    missing = sorted(RECOMMENDED_COLUMNS - set(df.columns))
    if missing:
        raise KeyError(f"enrichment did not create recommended columns {missing}")
    return validate_electrification_scenarios(df, require_recommended=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_ELECTRIFICATION_SCENARIO_PATH))
    parser.add_argument("--output", required=True)
    parser.add_argument("--hourly", default=None, help="Optional silver EP2050 hourly parquet for winter demand.")
    parser.add_argument("--profile", default=PROFILE_NAME)
    parser.add_argument(
        "--assumption-publication-date",
        required=True,
        help="Governance date for the internal assumptions, e.g. 2026-06-05.",
    )
    args = parser.parse_args(argv)

    enriched = enrich_scenarios(
        _read_table(args.input),
        assumption_publication_date=args.assumption_publication_date,
        hourly_path=args.hourly,
        profile=args.profile,
    )
    _write_table(enriched, args.output)
    print(f"[enrich] rows={len(enriched)}")
    print(f"[enrich] profile={args.profile}")
    print(f"[enrich] output -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

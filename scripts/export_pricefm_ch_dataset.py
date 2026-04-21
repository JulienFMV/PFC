#!/usr/bin/env python3
"""
Export a minimal CH-centric dataset in the tabular format expected by PriceFM.

The exported CSV contains quarter-hourly UTC rows with columns like:
    time_utc, CH-price, CH-load, CH-solar, CH-wind, DE_LU-price, ...

This script only includes countries for which the required quartet
{price, load, solar, wind} is sufficiently complete.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "pfc_shaping" / "data"
DEFAULT_OUTPUT = ROOT / "pfc_shaping" / "output" / "pricefm_ch_poc.csv"

COUNTRY_SPECS = {
    "CH": {
        "price_path": DATA_DIR / "epex_15min.parquet",
        "price_col": "price_eur_mwh",
        "entso": {
            "load": "load_mw",
            "solar": "solar_mw",
            "wind": "wind_mw",
        },
    },
    "DE_LU": {
        "price_path": DATA_DIR / "epex_de_15min.parquet",
        "price_col": "price_eur_mwh",
        "entso": {
            "load": "load_de_mw",
            "solar": "solar_de_mw",
            "wind": "wind_de_mw",
        },
    },
    "AT": {
        "price_path": DATA_DIR / "epex_at_15min.parquet",
        "price_col": "price_eur_mwh",
        "entso": {
            "load": "load_at_mw",
            "solar": "solar_at_mw",
            "wind": "wind_at_mw",
        },
    },
    "FR": {
        "price_path": DATA_DIR / "epex_fr_15min.parquet",
        "price_col": "price_eur_mwh",
        "entso": {
            "load": "load_fr_mw",
            "solar": "solar_fr_mw",
            "wind": "wind_fr_mw",
        },
    },
    "IT_NORD": {
        "price_path": DATA_DIR / "epex_it_15min.parquet",
        "price_col": "price_eur_mwh",
        "entso": {
            "load": "load_it_mw",
            "solar": "solar_it_mw",
            "wind": "wind_it_mw",
        },
    },
}


def _load_price_series(path: Path, col: str) -> pd.Series:
    df = pd.read_parquet(path)
    series = df[col].copy()
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    return series.sort_index()


def build_dataset(
    countries: list[str],
    min_coverage: float = 0.85,
    include_fr_nuclear: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    entso = pd.read_parquet(DATA_DIR / "entso_15min.parquet").sort_index()
    if entso.index.tz is None:
        entso.index = entso.index.tz_localize("UTC")

    coverage_report: dict[str, float] = {}
    country_frames: list[pd.DataFrame] = []

    for country in countries:
        spec = COUNTRY_SPECS[country]
        price = _load_price_series(spec["price_path"], spec["price_col"]).rename(f"{country}-price")

        cols = {}
        for feature_name, entso_col in spec["entso"].items():
            if entso_col not in entso.columns:
                continue
            cols[f"{country}-{feature_name}"] = entso[entso_col]

        if len(cols) != 3:
            coverage_report[country] = 0.0
            continue

        frame = pd.concat([price, pd.DataFrame(cols)], axis=1).sort_index()
        feature_cols = [f"{country}-price", f"{country}-load", f"{country}-solar", f"{country}-wind"]
        valid_bounds = []
        for feature_col in feature_cols:
            valid_idx = frame.index[frame[feature_col].notna()]
            if len(valid_idx) == 0:
                valid_bounds = []
                break
            valid_bounds.append((valid_idx[0], valid_idx[-1]))

        if not valid_bounds:
            coverage_report[country] = 0.0
            continue

        first_valid = max(bound[0] for bound in valid_bounds)
        last_valid = min(bound[1] for bound in valid_bounds)
        if first_valid > last_valid:
            coverage_report[country] = 0.0
            continue
        frame = frame.loc[first_valid:last_valid].copy()
        coverage = float(frame[feature_cols].notna().all(axis=1).mean())
        coverage_report[country] = coverage
        if coverage < min_coverage:
            continue

        country_frames.append(frame[feature_cols])

    if not country_frames:
        raise ValueError("No country met the minimum coverage threshold.")

    dataset = pd.concat(country_frames, axis=1).sort_index()
    dataset = dataset.dropna(how="all")
    dataset = dataset.dropna(how="any")

    # Optional Swiss/Alpine exogenous covariate.
    # We export it per country namespace so PriceFM can consume it as an optional
    # additional channel without changing the base quartet requirement.
    if include_fr_nuclear:
        outages_path = DATA_DIR / "outages_15min.parquet"
        if outages_path.exists():
            outages = pd.read_parquet(outages_path).sort_index()
            if not outages.empty and "unavailable_nuclear" in outages.columns:
                if outages.index.tz is None:
                    outages.index = outages.index.tz_localize("UTC")
                fr_nuclear = pd.to_numeric(outages["unavailable_nuclear"], errors="coerce")
                fr_nuclear = fr_nuclear.ffill().fillna(0.0).clip(lower=0.0)
                kept_countries = sorted({c.split("-")[0] for c in dataset.columns})
                for country in kept_countries:
                    col = f"{country}-fr_nuclear"
                    dataset[col] = fr_nuclear.reindex(dataset.index).ffill().fillna(0.0).astype(float)

    dataset = dataset.reset_index().rename(columns={"index": "time_utc", "timestamp": "time_utc"})
    if "time_utc" not in dataset.columns:
        dataset = dataset.rename(columns={dataset.columns[0]: "time_utc"})
    dataset["time_utc"] = pd.to_datetime(dataset["time_utc"], utc=True)
    return dataset, coverage_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CH-centric PriceFM dataset.")
    parser.add_argument(
        "--countries",
        nargs="+",
        default=["CH", "DE_LU", "AT", "FR", "IT_NORD"],
        help="Candidate countries to include.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.85,
        help="Minimum row-wise coverage required per country.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--no-fr-nuclear",
        action="store_true",
        help="Disable optional CH-fr_nuclear_unavailable_mw covariate export.",
    )
    args = parser.parse_args()

    dataset, coverage_report = build_dataset(
        args.countries,
        min_coverage=args.min_coverage,
        include_fr_nuclear=not args.no_fr_nuclear,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)

    kept_countries = sorted({c.split("-")[0] for c in dataset.columns if c != "time_utc"})
    print(f"output:{args.output}")
    print(f"rows:{len(dataset)}")
    print(f"countries_kept:{','.join(kept_countries)}")
    for country, coverage in coverage_report.items():
        print(f"coverage_{country}:{coverage:.3f}")


if __name__ == "__main__":
    main()

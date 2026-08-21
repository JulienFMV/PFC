"""Import official TYNDP 2024 Supply Inputs into the LT scenario contract.

The ENTSO-E/ENTSOG supply workbook contains governed supply-side trajectories,
but it does not by itself contain the full production scenario contract needed
by the HPFC model. This importer therefore writes a partial official inventory:
no synthetic zeros are added for missing demand, NTC or flexibility fields.
Strict production validation must continue to fail until those feeds are merged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.electrification_scenarios import (
    CRITICAL_PRODUCTION_VALUE_COLUMNS,
    PRODUCTION_COLUMNS,
    validate_electrification_scenarios,
)

DEFAULT_LOCAL_ROOT = Path(r"C:\Users\jbattaglia\pfc_local_data\scenarios\tyndp_2024")
DEFAULT_WORKBOOK = (
    DEFAULT_LOCAL_ROOT
    / "extracted"
    / "20231103-Final-Supply-Inputs-for-TYNDP-2024-Scenarios.xlsx"
    / "20231103 - Final Supply Inputs for TYNDP 2024 Scenarios.xlsx"
)
DEFAULT_OUTPUT = Path("data/electrification_scenarios_tyndp2024_supply.parquet")
DEFAULT_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/TYNDP2024-SUPPLY-IMPORT.md")

COUNTRY_CODES = {
    "CH": ["CH"],
    "DE": ["DE00"],
    "FR": ["FR00"],
    "AT": ["AT00"],
    "IT": ["IT"],
}

SCENARIO_LABELS = {
    "slow": "LOW",
    "central": "BEST_ESTIMATE",
    "fast": "HIGH",
}

SCENARIO_WEIGHTS = {
    "slow": 0.25,
    "central": 0.50,
    "fast": 0.25,
}


@dataclass(frozen=True)
class SheetSpec:
    sheet: str
    unit: str
    code_col: int
    group_year_cols: dict[str, dict[int, int]]


PV_SPEC = SheetSpec(
    sheet="1.1.",
    unit="MW",
    code_col=1,
    group_year_cols={
        "LOW": {2030: 2, 2040: 3, 2050: 4},
        "BEST_ESTIMATE": {2030: 5, 2040: 6},
        "HIGH": {2030: 7, 2040: 8, 2050: 9},
    },
)
WIND_ONSHORE_SPEC = SheetSpec(
    sheet="1.2.",
    unit="MW",
    code_col=1,
    group_year_cols={
        "LOW": {2030: 2, 2040: 3, 2050: 4},
        "BEST_ESTIMATE": {2030: 5, 2040: 6},
        "HIGH": {2030: 7, 2040: 8, 2050: 9},
    },
)
WIND_OFFSHORE_SPEC = SheetSpec(
    sheet="1.3.",
    unit="MW",
    code_col=1,
    group_year_cols={
        "LOW": {2030: 2, 2035: 3, 2040: 4, 2045: 5, 2050: 6},
        "BEST_ESTIMATE": {2030: 7, 2035: 8, 2040: 9, 2045: 10, 2050: 11},
        "HIGH": {2030: 12, 2040: 13, 2050: 14},
    },
)
NUCLEAR_SPEC = SheetSpec(
    sheet="1.4.",
    unit="MW",
    code_col=1,
    group_year_cols={
        "LOW": {2030: 2, 2040: 3, 2050: 4},
        "BEST_ESTIMATE": {2030: 5, 2040: 6},
        "HIGH": {2030: 7, 2040: 8, 2050: 9},
    },
)
PROSUMER_BATTERY_SPEC = SheetSpec(
    sheet="1.5.",
    unit="MWh",
    code_col=2,
    group_year_cols={
        "LOW": {2030: 3, 2040: 4, 2050: 5},
        "BEST_ESTIMATE": {2030: 6, 2040: 7},
        "HIGH": {2030: 8, 2040: 9, 2050: 10},
    },
)
UTILITY_BATTERY_SPEC = SheetSpec(
    sheet="1.6.",
    unit="MWh",
    code_col=2,
    group_year_cols={
        "LOW": {2030: 3, 2040: 4, 2050: 5},
        "BEST_ESTIMATE": {2030: 6, 2040: 7},
        "HIGH": {2030: 8, 2040: 9, 2050: 10},
    },
)


def _country_mask(codes: pd.Series, country: str) -> pd.Series:
    labels = COUNTRY_CODES[country]
    text = codes.fillna("").astype(str).str.upper()
    if labels == ["IT"]:
        return text.str.startswith("IT")
    if labels == ["CH"]:
        return text.eq("CH00")
    return text.isin(labels)


def _sum_sheet_value(
    frames: dict[str, pd.DataFrame],
    spec: SheetSpec,
    *,
    country: str,
    scenario_group: str,
    year: int,
) -> float | None:
    col = spec.group_year_cols.get(scenario_group, {}).get(int(year))
    if col is None:
        return None
    df = frames[spec.sheet]
    if col >= df.shape[1]:
        return None
    mask = _country_mask(df.iloc[:, spec.code_col], country)
    values = pd.to_numeric(df.loc[mask, col], errors="coerce").dropna()
    if values.empty:
        return None
    total = float(values.sum())
    if not np.isfinite(total):
        return None
    return total


def _fuel_prices(frames: dict[str, pd.DataFrame], year: int) -> dict[str, float | None]:
    df = frames["3.1."]
    header = df.iloc[2, :].tolist()
    year_cols = {
        int(value): idx
        for idx, value in enumerate(header)
        if isinstance(value, (int, float)) and not pd.isna(value)
    }
    col = year_cols.get(int(year))
    if col is None:
        return {"gas_eur_mwh": None, "coal_eur_mwh": None, "co2_eur_t": None}
    labels = df.iloc[:, 1].fillna("").astype(str).str.lower()

    def value_for(label: str, multiplier: float = 1.0) -> float | None:
        rows = df.loc[labels == label.lower(), col]
        if rows.empty:
            return None
        value = pd.to_numeric(rows, errors="coerce").dropna()
        if value.empty:
            return None
        return float(value.iloc[0]) * multiplier

    return {
        "gas_eur_mwh": value_for("Natural Gas", 3.6),
        "coal_eur_mwh": value_for("Hard coal", 3.6),
        "co2_eur_t": value_for("CO2 price", 1.0),
    }


def build_tyndp2024_supply_inventory(
    workbook: Path,
    *,
    countries: list[str],
    scenarios: list[str],
    years: list[int],
    publication_date: str,
    ingested_at_utc: str,
) -> pd.DataFrame:
    if not workbook.exists():
        raise FileNotFoundError(f"TYNDP 2024 supply workbook not found: {workbook}")

    sheet_names = {
        spec.sheet
        for spec in [
            PV_SPEC,
            WIND_ONSHORE_SPEC,
            WIND_OFFSHORE_SPEC,
            NUCLEAR_SPEC,
            PROSUMER_BATTERY_SPEC,
            UTILITY_BATTERY_SPEC,
        ]
    } | {"3.1."}
    frames = {
        sheet: pd.read_excel(workbook, sheet_name=sheet, header=None)
        for sheet in sorted(sheet_names)
    }

    rows: list[dict[str, object]] = []
    for country in countries:
        for scenario in scenarios:
            scenario_group = SCENARIO_LABELS[scenario]
            for year in years:
                pv_mw = _sum_sheet_value(frames, PV_SPEC, country=country, scenario_group=scenario_group, year=year)
                wind_onshore_mw = _sum_sheet_value(
                    frames, WIND_ONSHORE_SPEC, country=country, scenario_group=scenario_group, year=year
                )
                wind_offshore_mw = _sum_sheet_value(
                    frames, WIND_OFFSHORE_SPEC, country=country, scenario_group=scenario_group, year=year
                )
                nuclear_mw = _sum_sheet_value(
                    frames, NUCLEAR_SPEC, country=country, scenario_group=scenario_group, year=year
                )
                prosumer_battery_mwh = _sum_sheet_value(
                    frames, PROSUMER_BATTERY_SPEC, country=country, scenario_group=scenario_group, year=year
                )
                utility_battery_mwh = _sum_sheet_value(
                    frames, UTILITY_BATTERY_SPEC, country=country, scenario_group=scenario_group, year=year
                )
                fuel = _fuel_prices(frames, year)
                rows.append(
                    {
                        "publication_date": pd.Timestamp(publication_date, tz="UTC"),
                        "measurement_date": pd.NaT,
                        "track": "scenario",
                        "source": "ENTSO-E/ENTSOG TYNDP 2024 Supply Inputs",
                        "scenario_edition": "TYNDP2024",
                        "scenario": scenario,
                        "country": country,
                        "delivery_year": int(year),
                        "delivery_month": np.nan,
                        "scenario_weight": SCENARIO_WEIGHTS[scenario],
                        "quality_flag": "official_tyndp_supply_partial",
                        "ingested_at_utc": pd.Timestamp(ingested_at_utc, tz="UTC"),
                        "pv_gw": None if pv_mw is None else pv_mw / 1000.0,
                        "wind_gw": None
                        if wind_onshore_mw is None or wind_offshore_mw is None
                        else (wind_onshore_mw + wind_offshore_mw) / 1000.0,
                        "battery_energy_gwh": None
                        if prosumer_battery_mwh is None or utility_battery_mwh is None
                        else (prosumer_battery_mwh + utility_battery_mwh) / 1000.0,
                        "nuclear_gw": None if nuclear_mw is None else nuclear_mw / 1000.0,
                        **fuel,
                    }
                )

    out = pd.DataFrame(rows)
    for col in sorted(PRODUCTION_COLUMNS):
        if col not in out.columns:
            out[col] = np.nan
    ordered = [
        "publication_date",
        "measurement_date",
        "track",
        "source",
        "scenario_edition",
        "scenario",
        "country",
        "delivery_year",
        "delivery_month",
        "scenario_weight",
        "quality_flag",
        "ingested_at_utc",
    ]
    remaining = [col for col in sorted(out.columns) if col not in ordered]
    return validate_electrification_scenarios(out[ordered + remaining], require_recommended=True)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_years(value: str) -> list[int]:
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _write_report(path: Path, frame: pd.DataFrame, *, workbook: Path) -> None:
    non_null = frame[list(sorted(CRITICAL_PRODUCTION_VALUE_COLUMNS & set(frame.columns)))].notna().sum()
    summary = pd.DataFrame(
        [
            {
                "rows": len(frame),
                "countries": ", ".join(sorted(frame["country"].unique())),
                "scenarios": ", ".join(sorted(frame["scenario"].unique())),
                "years": ", ".join(str(y) for y in sorted(frame["delivery_year"].unique())),
            }
        ]
    )
    non_null_df = non_null.reset_index(name="non_null_rows").rename(columns={"index": "column"})

    def to_md(df: pd.DataFrame) -> str:
        if df.empty:
            return "_none_"
        cols = [str(c) for c in df.columns]
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
        return "\n".join(lines)

    lines = [
        "# TYNDP 2024 Supply Import",
        "",
        f"* workbook: `{workbook}`",
        "* status: `PARTIAL - not production-complete`",
        "* source: ENTSO-E/ENTSOG TYNDP 2024 Supply Inputs workbook",
        "* rule: missing demand, NTC, battery power and flexibility fields remain null; no silent zero filling",
        "",
        "## Coverage",
        "",
        to_md(summary),
        "",
        "## Non-null Critical Fields",
        "",
        to_md(non_null_df),
        "",
        "## Production Limitation",
        "",
        "This file is an official supply-side component. It must be merged with governed demand, NTC, hydro, EV, heat-pump and flexibility feeds before `--require-production` can pass.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", default=str(DEFAULT_WORKBOOK))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--countries", default="CH,DE,FR,IT,AT")
    parser.add_argument("--scenarios", default="slow,central,fast")
    parser.add_argument("--years", default="2030,2040")
    parser.add_argument("--publication-date", default="2024-05-31")
    parser.add_argument("--ingested-at-utc", default="2026-06-11")
    args = parser.parse_args(argv)

    frame = build_tyndp2024_supply_inventory(
        Path(args.workbook),
        countries=_parse_csv(args.countries),
        scenarios=_parse_csv(args.scenarios),
        years=_parse_years(args.years),
        publication_date=args.publication_date,
        ingested_at_utc=args.ingested_at_utc,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    _write_report(Path(args.report), frame, workbook=Path(args.workbook))
    print(f"[tyndp2024] supply rows={len(frame)} -> {output}")
    print(f"[tyndp2024] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

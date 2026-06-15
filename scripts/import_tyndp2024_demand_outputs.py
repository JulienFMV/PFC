"""Import official TYNDP 2024 Demand Outputs into the LT scenario contract.

The demand workbook is an official ENTSO-E/ENTSOG component, but it is not a
complete HPFC production inventory on its own: Switzerland is absent from the
output table and only Distributed Energy / Global Ambition are provided for
2040/2050. The importer therefore writes raw official scenario labels and marks
the rows as partial.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.electrification_scenarios import (
    CRITICAL_PRODUCTION_VALUE_COLUMNS,
    PRODUCTION_COLUMNS,
    validate_electrification_scenarios,
)

DEFAULT_WORKBOOK = (
    Path(r"C:\Users\jbattaglia\pfc_local_data\scenarios\tyndp_2024")
    / "extracted"
    / "Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb"
    / "Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb"
)
DEFAULT_OUTPUT = Path("data/electrification_scenarios_tyndp2024_demand.parquet")
DEFAULT_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/TYNDP2024-DEMAND-IMPORT.md")

SCENARIO_MAP = {
    "DE": "tyndp_distributed_energy",
    "GA": "tyndp_global_ambition",
}


def _sum_or_nan(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.sum())


def build_tyndp2024_demand_inventory(
    demand_output: pd.DataFrame,
    *,
    countries: list[str],
    publication_date: str,
    ingested_at_utc: str,
) -> pd.DataFrame:
    header_scenario = demand_output.iloc[0]
    header_year = demand_output.iloc[1]
    value_columns: list[tuple[int, str, int]] = []
    for idx in range(10, demand_output.shape[1]):
        scenario_label = str(header_scenario.iloc[idx]).strip()
        if scenario_label not in SCENARIO_MAP:
            continue
        year_value = pd.to_numeric(pd.Series([header_year.iloc[idx]]), errors="coerce").iloc[0]
        if pd.isna(year_value):
            continue
        value_columns.append((idx, SCENARIO_MAP[scenario_label], int(year_value)))

    body = demand_output.iloc[2:].copy()
    mask = (
        body.iloc[:, 1].astype(str).isin(countries)
        & body.iloc[:, 3].astype(str).str.contains("_final_demand_electricity_", regex=False)
        & body.iloc[:, 6].astype(str).str.lower().eq("electricity")
        & body.iloc[:, 7].astype(str).str.lower().eq("energy demand")
        & body.iloc[:, 9].astype(str).str.lower().eq("twh")
    )
    body = body.loc[mask]

    rows: list[dict[str, object]] = []
    for country in countries:
        country_rows = body[body.iloc[:, 1].astype(str) == country]
        if country_rows.empty:
            continue
        for col, scenario, year in value_columns:
            value = _sum_or_nan(country_rows.iloc[:, col])
            rows.append(
                {
                    "publication_date": pd.Timestamp(publication_date, tz="UTC"),
                    "measurement_date": pd.NaT,
                    "track": "scenario",
                    "source": "ENTSO-E/ENTSOG TYNDP 2024 Demand Outputs",
                    "scenario_edition": "TYNDP2024",
                    "scenario": scenario,
                    "country": country,
                    "delivery_year": int(year),
                    "delivery_month": np.nan,
                    "scenario_weight": np.nan,
                    "quality_flag": "official_tyndp_demand_partial",
                    "ingested_at_utc": pd.Timestamp(ingested_at_utc, tz="UTC"),
                    "demand_twh": float(value),
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


def _to_md(df: pd.DataFrame) -> str:
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


def _write_report(path: Path, frame: pd.DataFrame, *, workbook: Path) -> None:
    non_null = frame[list(sorted(CRITICAL_PRODUCTION_VALUE_COLUMNS & set(frame.columns)))].notna().sum()
    summary = pd.DataFrame(
        [
            {
                "rows": len(frame),
                "countries": ", ".join(sorted(frame["country"].unique())) if len(frame) else "",
                "scenarios": ", ".join(sorted(frame["scenario"].unique())) if len(frame) else "",
                "years": ", ".join(str(y) for y in sorted(frame["delivery_year"].unique())) if len(frame) else "",
            }
        ]
    )
    lines = [
        "# TYNDP 2024 Demand Import",
        "",
        f"* workbook: `{workbook}`",
        "* status: `PARTIAL - not production-complete`",
        "* source: ENTSO-E/ENTSOG TYNDP 2024 Demand Outputs workbook",
        "* rule: raw official DE/GA scenario labels are preserved; no slow/central/fast mapping is inferred",
        "",
        "## Coverage",
        "",
        _to_md(summary),
        "",
        "## Non-null Critical Fields",
        "",
        _to_md(non_null.reset_index(name="non_null_rows").rename(columns={"index": "column"})),
        "",
        "## Production Limitation",
        "",
        "This component supplies demand for available neighbouring countries only. CH demand remains sourced from OFEN/EP2050, and final production still requires governed scenario mapping, NTC, hydro, EV, heat-pump and flexibility feeds.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", default=str(DEFAULT_WORKBOOK))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--countries", default="AT,DE,FR,IT")
    parser.add_argument("--publication-date", default="2024-05-31")
    parser.add_argument("--ingested-at-utc", default="2026-06-11")
    args = parser.parse_args(argv)

    workbook = Path(args.workbook)
    if not workbook.exists():
        raise FileNotFoundError(f"TYNDP 2024 demand workbook not found: {workbook}")
    try:
        demand_output = pd.read_excel(
            workbook,
            sheet_name="3_DEMAND_OUTPUT",
            engine="pyxlsb",
            header=None,
        )
    except ImportError as exc:
        raise ImportError("Reading TYNDP Demand requires pyxlsb. Install requirements.txt.") from exc

    frame = build_tyndp2024_demand_inventory(
        demand_output,
        countries=_parse_csv(args.countries),
        publication_date=args.publication_date,
        ingested_at_utc=args.ingested_at_utc,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    _write_report(Path(args.report), frame, workbook=workbook)
    print(f"[tyndp2024] demand rows={len(frame)} -> {output}")
    print(f"[tyndp2024] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

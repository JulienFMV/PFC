"""Build a non-production 2030 neighbour demand bridge from TYNDP Demand.

TYNDP 2024 Demand Outputs provide REF 2019 plus DE/GA 2040/2050, but no 2030
row and no CH rows. This bridge creates slow/central/fast neighbour rows for
2030 by linearly interpolating REF 2019 to each official 2040 scenario and
mapping the lower/higher 2030 demand to slow/fast. Historical ENTSO-E load
ratios provide peak-load and winter-demand scalers.

The output is explicitly partial/proxy and must fail strict production gates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.electrification_scenarios import (
    PRODUCTION_COLUMNS,
    validate_electrification_scenarios,
)

DEFAULT_WORKBOOK = (
    Path(r"C:\Users\jbattaglia\pfc_local_data\scenarios\tyndp_2024")
    / "extracted"
    / "Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb"
    / "Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb"
)
DEFAULT_ENTSO = Path("pfc_shaping/data/entso_15min.parquet")
DEFAULT_OUTPUT = Path("data/electrification_scenarios_tyndp2024_neighbor_demand_bridge_2030.parquet")
DEFAULT_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/TYNDP2024-NEIGHBOR-DEMAND-BRIDGE.md")

LOAD_COLUMNS = {
    "AT": "load_at_mw",
    "DE": "load_de_mw",
    "FR": "load_fr_mw",
    "IT": "load_it_mw",
}

SCENARIO_WEIGHTS = {
    "slow": 0.25,
    "central": 0.50,
    "fast": 0.25,
}


def _sum_or_nan(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.sum())


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _demand_output_final_rows(demand_output: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    body = demand_output.iloc[2:].copy()
    return body[
        body.iloc[:, 1].astype(str).isin(countries)
        & body.iloc[:, 3].astype(str).str.contains("_final_demand_electricity_", regex=False)
        & body.iloc[:, 6].astype(str).str.lower().eq("electricity")
        & body.iloc[:, 7].astype(str).str.lower().eq("energy demand")
        & body.iloc[:, 9].astype(str).str.lower().eq("twh")
    ].copy()


def _official_demand_totals(demand_output: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    rows = _demand_output_final_rows(demand_output, countries)
    out: list[dict[str, object]] = []
    for country in countries:
        sub = rows[rows.iloc[:, 1].astype(str) == country]
        if sub.empty:
            continue
        out.append(
            {
                "country": country,
                "ref_2019_twh": _sum_or_nan(sub.iloc[:, 10]),
                "de_2040_twh": _sum_or_nan(sub.iloc[:, 11]),
                "ga_2040_twh": _sum_or_nan(sub.iloc[:, 13]),
            }
        )
    return pd.DataFrame(out)


def _interpolate_2030(ref_2019: float, scenario_2040: float) -> float:
    return float(ref_2019 + (scenario_2040 - ref_2019) * ((2030 - 2019) / (2040 - 2019)))


def _historical_load_ratios(
    entso: pd.DataFrame,
    *,
    countries: list[str],
    vintage: str,
    allow_missing: bool,
) -> pd.DataFrame:
    df = entso.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("ENTSO physical dataframe must be indexed by timestamps")
    df = df.sort_index()
    vintage_ts = pd.Timestamp(vintage)
    vintage_ts = vintage_ts.tz_localize("UTC") if vintage_ts.tz is None else vintage_ts.tz_convert("UTC")
    df = df[df.index <= vintage_ts]
    if df.empty:
        raise ValueError("no ENTSO load data available at or before vintage")

    candidate_years = sorted(set(int(y) for y in df.index.year if int(y) < vintage_ts.year), reverse=True)
    out: list[dict[str, object]] = []
    for country in countries:
        col = LOAD_COLUMNS[country]
        if col not in df.columns:
            if not allow_missing:
                raise KeyError(f"missing ENTSO load column {col}")
            out.append(
                {
                    "country": country,
                    "historical_year": pd.NA,
                    "historical_annual_twh": np.nan,
                    "peak_to_annual_gw_per_twh": np.nan,
                    "winter_share": np.nan,
                    "ratio_quality": f"missing_column:{col}",
                }
            )
            continue
        selected = None
        selected_year = None
        for year in candidate_years:
            sub = pd.to_numeric(df.loc[df.index.year == year, col], errors="coerce").dropna()
            expected = 8784 * 4 if pd.Timestamp(f"{year}-12-31").is_leap_year else 8760 * 4
            if len(sub) >= 0.95 * expected:
                selected = sub
                selected_year = year
                break
        if selected is None or selected.empty:
            if not allow_missing:
                raise ValueError(f"no sufficiently complete historical load year for {country}")
            out.append(
                {
                    "country": country,
                    "historical_year": pd.NA,
                    "historical_annual_twh": np.nan,
                    "peak_to_annual_gw_per_twh": np.nan,
                    "winter_share": np.nan,
                    "ratio_quality": "missing_complete_historical_load_year",
                }
            )
            continue
        annual_twh = float(selected.sum() * 0.25 / 1_000_000.0)
        peak_gw = float(selected.max() / 1000.0)
        idx = selected.index
        winter = selected[(idx.month <= 3) | (idx.month >= 10)]
        winter_twh = float(winter.sum() * 0.25 / 1_000_000.0)
        if annual_twh <= 0.0:
            raise ValueError(f"non-positive historical annual demand for {country}")
        out.append(
            {
                "country": country,
                "historical_year": int(selected_year),
                "historical_annual_twh": annual_twh,
                "peak_to_annual_gw_per_twh": peak_gw / annual_twh,
                "winter_share": winter_twh / annual_twh,
                "ratio_quality": "historical_ratio",
            }
        )
    return pd.DataFrame(out)


def build_neighbor_demand_bridge(
    *,
    demand_output: pd.DataFrame,
    entso: pd.DataFrame,
    countries: list[str],
    vintage: str,
    publication_date: str,
    ingested_at_utc: str,
    allow_missing_load_ratios: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    totals = _official_demand_totals(demand_output, countries)
    ratios = _historical_load_ratios(
        entso,
        countries=countries,
        vintage=vintage,
        allow_missing=allow_missing_load_ratios,
    )
    merged = totals.merge(ratios, on="country", how="inner", validate="one_to_one")
    if set(merged["country"]) != set(countries):
        missing = sorted(set(countries) - set(merged["country"]))
        raise ValueError(f"missing demand bridge inputs for countries {missing}")

    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        de_2030 = _interpolate_2030(float(row["ref_2019_twh"]), float(row["de_2040_twh"]))
        ga_2030 = _interpolate_2030(float(row["ref_2019_twh"]), float(row["ga_2040_twh"]))
        slow = min(de_2030, ga_2030)
        fast = max(de_2030, ga_2030)
        scenario_values = {
            "slow": slow,
            "central": 0.5 * (slow + fast),
            "fast": fast,
        }
        for scenario, demand_twh in scenario_values.items():
            peak_ratio = float(row["peak_to_annual_gw_per_twh"]) if pd.notna(row["peak_to_annual_gw_per_twh"]) else np.nan
            winter_share = float(row["winter_share"]) if pd.notna(row["winter_share"]) else np.nan
            rows.append(
                {
                    "publication_date": pd.Timestamp(publication_date, tz="UTC"),
                    "measurement_date": pd.NaT,
                    "track": "scenario",
                    "source": "TYNDP2024 Demand REF2019-to-2040 bridge + ENTSO-E historical load ratios",
                    "scenario_edition": "TYNDP2024_DEMAND_BRIDGE_2030",
                    "scenario": scenario,
                    "country": row["country"],
                    "delivery_year": 2030,
                    "delivery_month": np.nan,
                    "scenario_weight": SCENARIO_WEIGHTS[scenario],
                    "quality_flag": "internal_tyndp_demand_bridge_partial_proxy",
                    "ingested_at_utc": pd.Timestamp(ingested_at_utc, tz="UTC"),
                    "demand_twh": float(demand_twh),
                    "peak_load_gw": np.nan if not np.isfinite(peak_ratio) else float(demand_twh) * peak_ratio,
                    "winter_demand_twh": np.nan if not np.isfinite(winter_share) else float(demand_twh) * winter_share,
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
    return validate_electrification_scenarios(out[ordered + remaining], require_recommended=True), merged


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_report(path: Path, frame: pd.DataFrame, diagnostics: pd.DataFrame, *, output: Path) -> None:
    preview = frame[
        ["country", "scenario", "delivery_year", "demand_twh", "peak_load_gw", "winter_demand_twh", "quality_flag"]
    ].sort_values(["country", "scenario"])
    lines = [
        "# TYNDP 2024 Neighbour Demand Bridge",
        "",
        f"* output: `{output}`",
        "* status: `PARTIAL - production gate must fail`",
        "* method: interpolate REF 2019 to official DE/GA 2040, then map lower/midpoint/higher to slow/central/fast",
        "* peak and winter demand: scaled with latest complete historical ENTSO-E load year available before vintage",
        "* rule: this is an internal bridge, not a governed production scenario",
        "",
        "## Output Preview",
        "",
        _md_table(preview),
        "",
        "## Diagnostics",
        "",
        _md_table(diagnostics),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", default=str(DEFAULT_WORKBOOK))
    parser.add_argument("--entso", default=str(DEFAULT_ENTSO))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--countries", default="AT,DE,FR,IT")
    parser.add_argument("--vintage", default="2026-06-11")
    parser.add_argument("--publication-date", default="2026-06-11")
    parser.add_argument("--ingested-at-utc", default="2026-06-11")
    args = parser.parse_args(argv)

    workbook = Path(args.workbook)
    if not workbook.exists():
        raise FileNotFoundError(f"TYNDP Demand workbook not found: {workbook}")
    demand_output = pd.read_excel(workbook, sheet_name="3_DEMAND_OUTPUT", engine="pyxlsb", header=None)
    entso = pd.read_parquet(args.entso)
    frame, diagnostics = build_neighbor_demand_bridge(
        demand_output=demand_output,
        entso=entso,
        countries=_parse_csv(args.countries),
        vintage=args.vintage,
        publication_date=args.publication_date,
        ingested_at_utc=args.ingested_at_utc,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    _write_report(Path(args.report), frame, diagnostics, output=output)
    print(f"[demand-bridge] rows={len(frame)} -> {output}")
    print(f"[demand-bridge] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

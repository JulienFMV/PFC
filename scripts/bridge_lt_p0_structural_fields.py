"""Bridge selected P0 LT structural fields with explicit non-prod provenance.

This script closes missing numeric P0 fields only where a traceable bridge is
available from local official components or historical CH ratios. It does not
claim production approval: output rows are marked partial/proxy and must still
fail strict governance until official/gouverned feeds replace these bridges.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.electrification_scenarios import (
    derive_hpfc_scenario_features,
    load_electrification_scenarios,
    validate_electrification_scenarios,
)

DEFAULT_INPUT = Path("data/electrification_scenarios_composed_partial_2030.parquet")
DEFAULT_OUTPUT = Path("data/electrification_scenarios_composed_p0_bridge_2030.parquet")
DEFAULT_FEATURE_OUTPUT = Path("data/hpfc_scenario_features_composed_p0_bridge_2030.parquet")
DEFAULT_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/P0-STRUCTURAL-BRIDGE.md")
DEFAULT_ENTSO = Path("pfc_shaping/data/entso_15min.parquet")
DEFAULT_DEMAND_WORKBOOK = (
    Path(r"C:\Users\jbattaglia\pfc_local_data\scenarios\tyndp_2024")
    / "extracted"
    / "Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb"
    / "Demand_Scenarios_TYNDP_2024_After_Public_Consultation.xlsb"
)

PV_CAPACITY_FACTORS = {
    "CH": 0.12,
    "AT": 0.12,
    "DE": 0.11,
    "FR": 0.14,
    "IT": 0.16,
}
WIND_CAPACITY_FACTORS = {
    "CH": 0.20,
    "AT": 0.25,
    "DE": 0.28,
    "FR": 0.27,
    "IT": 0.23,
}


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _infer_step_hours(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 0.25
    diffs = index.to_series().diff().dropna().dt.total_seconds() / 3600.0
    if diffs.empty:
        return 0.25
    return float(diffs.median())


def historical_ch_load_ratios(entso_path: Path) -> dict[str, float]:
    entso = pd.read_parquet(entso_path)
    if "load_mw" not in entso.columns:
        raise KeyError("ENTSO frame lacks load_mw")
    load = pd.to_numeric(entso["load_mw"], errors="coerce").dropna()
    if load.empty:
        raise ValueError("ENTSO CH load_mw is empty")
    step_h = _infer_step_hours(entso.index)
    avg_gw = float(load.mean() / 1000.0)
    peak_gw = float(load.max() / 1000.0)
    if avg_gw <= 0:
        raise ValueError("ENTSO CH average load is non-positive")
    peak_to_average = peak_gw / avg_gw
    if isinstance(entso.index, pd.DatetimeIndex):
        winter = load.loc[entso.index.month.isin([1, 2, 3, 11, 12])]
        annual_twh = float(load.sum() * step_h / 1_000_000.0)
        winter_twh = float(winter.sum() * step_h / 1_000_000.0)
        winter_share = winter_twh / annual_twh if annual_twh > 0 else np.nan
    else:
        winter_share = 5.0 / 12.0
    if not np.isfinite(winter_share):
        winter_share = 5.0 / 12.0
    return {
        "peak_to_average": float(peak_to_average),
        "winter_share": float(winter_share),
    }


def _sum_or_nan(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.sum())


def _demand_component_bridge(
    workbook: Path,
    *,
    countries: list[str],
    variables: list[str],
) -> pd.DataFrame:
    if not workbook.exists():
        return pd.DataFrame(columns=["country", "slow", "central", "fast"])
    demand = pd.read_excel(workbook, sheet_name="3_DEMAND_OUTPUT", engine="pyxlsb", header=None)
    header_scenario = demand.iloc[0]
    header_year = demand.iloc[1]
    ref_cols: list[int] = []
    de_2040_cols: list[int] = []
    ga_2040_cols: list[int] = []
    for idx in range(10, demand.shape[1]):
        scenario_label = str(header_scenario.iloc[idx]).strip()
        year_value = pd.to_numeric(pd.Series([header_year.iloc[idx]]), errors="coerce").iloc[0]
        if pd.isna(year_value):
            continue
        year = int(year_value)
        if scenario_label == "REF" and year == 2019:
            ref_cols.append(idx)
        elif scenario_label == "DE" and year == 2040:
            de_2040_cols.append(idx)
        elif scenario_label == "GA" and year == 2040:
            ga_2040_cols.append(idx)
    body = demand.iloc[2:].copy()
    variable_s = body.iloc[:, 3].fillna("").astype(str)
    mask = (
        body.iloc[:, 1].astype(str).isin(countries)
        & variable_s.isin(variables)
        & body.iloc[:, 6].astype(str).str.lower().eq("electricity")
        & body.iloc[:, 7].astype(str).str.lower().eq("energy demand")
        & body.iloc[:, 9].astype(str).str.lower().eq("twh")
    )
    body = body.loc[mask]
    rows = []
    for country in countries:
        sub = body[body.iloc[:, 1].astype(str) == country]
        if sub.empty or not ref_cols or not de_2040_cols or not ga_2040_cols:
            continue
        ref = _sum_or_nan(sub.iloc[:, ref_cols].stack())
        de = _sum_or_nan(sub.iloc[:, de_2040_cols].stack())
        ga = _sum_or_nan(sub.iloc[:, ga_2040_cols].stack())
        if not np.isfinite(ref) or not np.isfinite(de) or not np.isfinite(ga):
            continue
        de_2030 = ref + (de - ref) * (2030 - 2019) / (2040 - 2019)
        ga_2030 = ref + (ga - ref) * (2030 - 2019) / (2040 - 2019)
        slow = min(de_2030, ga_2030)
        fast = max(de_2030, ga_2030)
        rows.append({"country": country, "slow": slow, "central": 0.5 * (slow + fast), "fast": fast})
    return pd.DataFrame(rows)


def build_p0_bridge(
    frame: pd.DataFrame,
    *,
    entso_path: Path,
    demand_workbook: Path,
    countries: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = validate_electrification_scenarios(frame, require_recommended=True).copy()
    ratios = historical_ch_load_ratios(entso_path)
    ev_bridge = _demand_component_bridge(
        demand_workbook,
        countries=[c for c in countries if c != "CH"],
        variables=["transport_final_demand_electricity_energetic"],
    )
    heat_bridge = _demand_component_bridge(
        demand_workbook,
        countries=[c for c in countries if c != "CH"],
        variables=[
            "households_space_heating_final_demand_electricity",
            "buildings_space_heating_final_demand_electricity",
        ],
    )
    ev_lookup = {
        (row.country, scenario): float(getattr(row, scenario))
        for row in ev_bridge.itertuples(index=False)
        for scenario in ["slow", "central", "fast"]
    }
    heat_lookup = {
        (row.country, scenario): float(getattr(row, scenario))
        for row in heat_bridge.itertuples(index=False)
        for scenario in ["slow", "central", "fast"]
    }
    changes: list[dict[str, object]] = []

    def fill(idx, column: str, value: float, method: str) -> None:
        if column not in out.columns or not np.isfinite(value) or value < 0:
            return
        if pd.notna(out.at[idx, column]):
            return
        out.at[idx, column] = float(value)
        changes.append(
            {
                "country": out.at[idx, "country"],
                "scenario": out.at[idx, "scenario"],
                "delivery_year": int(out.at[idx, "delivery_year"]),
                "column": column,
                "value": float(value),
                "method": method,
            }
        )

    for idx, row in out.iterrows():
        country = str(row["country"])
        scenario = str(row["scenario"])
        demand_twh = pd.to_numeric(pd.Series([row.get("demand_twh")]), errors="coerce").iloc[0]
        if np.isfinite(demand_twh) and demand_twh > 0:
            avg_gw = demand_twh / 8.76
            fill(idx, "peak_load_gw", avg_gw * ratios["peak_to_average"], "CH historical peak/average load ratio")
            fill(idx, "winter_demand_twh", demand_twh * ratios["winter_share"], "CH historical winter demand share")
        pv_gw = pd.to_numeric(pd.Series([row.get("pv_gw")]), errors="coerce").iloc[0]
        if np.isfinite(pv_gw):
            fill(idx, "pv_twh", pv_gw * PV_CAPACITY_FACTORS.get(country, 0.12) * 8.76, "country PV capacity factor bridge")
        wind_gw = pd.to_numeric(pd.Series([row.get("wind_gw")]), errors="coerce").iloc[0]
        if np.isfinite(wind_gw):
            fill(idx, "wind_twh", wind_gw * WIND_CAPACITY_FACTORS.get(country, 0.25) * 8.76, "country wind capacity factor bridge")
        battery_energy = pd.to_numeric(pd.Series([row.get("battery_energy_gwh")]), errors="coerce").iloc[0]
        if np.isfinite(battery_energy):
            fill(idx, "battery_power_gw", battery_energy / 4.0, "4h battery duration bridge")
        nuclear = pd.to_numeric(pd.Series([row.get("nuclear_gw")]), errors="coerce").iloc[0]
        if np.isfinite(nuclear) and nuclear > 0:
            fill(idx, "dispatchable_gw", nuclear, "nuclear-only firm dispatchable lower-bound bridge")
        fill(idx, "ev_twh", ev_lookup.get((country, scenario), np.nan), "TYNDP Demand transport electricity REF2019-to-2040 bridge")
        fill(idx, "heatpump_twh", heat_lookup.get((country, scenario), np.nan), "TYNDP Demand electric space-heating REF2019-to-2040 bridge")

    if changes:
        out["quality_flag"] = out["quality_flag"].astype(str) + "_p0_structural_bridge_proxy"
        out["source"] = out["source"].astype(str) + " + P0 structural bridge"
        if "component_ids" in out.columns:
            out["component_ids"] = out["component_ids"].fillna("").astype(str).map(
                lambda value: value + ",internal_p0_structural_bridge"
                if value
                else "internal_p0_structural_bridge"
            )
    out = validate_electrification_scenarios(out, require_recommended=True)
    return out, pd.DataFrame(changes)


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        values = [str(row[c]).replace("|", "\\|").replace("\n", " ") for c in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path: Path, *, frame: pd.DataFrame, changes: pd.DataFrame, output: Path) -> None:
    by_column = (
        changes.groupby("column").size().reset_index(name="filled_rows").sort_values("column")
        if not changes.empty
        else pd.DataFrame(columns=["column", "filled_rows"])
    )
    lines = [
        "# P0 Structural Bridge",
        "",
        f"* output: `{output}`",
        "* status: `NON-PRODUCTION / PARTIAL / PROXY`",
        "* rule: fills only traceable P0 numeric gaps; does not fill NTC, hydro or cross-border balance without source",
        "",
        "## Filled Rows By Column",
        "",
        _md_table(by_column),
        "",
        "## Methodology",
        "",
        "* `peak_load_gw`: CH historical peak/average load ratio from local ENTSO-E cache.",
        "* `winter_demand_twh`: CH historical Nov-Mar demand share from local ENTSO-E cache.",
        "* `pv_twh` / `wind_twh`: explicit country capacity-factor bridge; proxy only.",
        "* `battery_power_gw`: four-hour duration bridge from battery energy.",
        "* `dispatchable_gw`: nuclear-only lower-bound bridge where missing.",
        "* neighbour `ev_twh` / `heatpump_twh`: TYNDP Demand REF2019-to-2040 bridge.",
        "",
        "## Changed Cells",
        "",
        _md_table(changes),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--features-output", default=str(DEFAULT_FEATURE_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--entso", default=str(DEFAULT_ENTSO))
    parser.add_argument("--demand-workbook", default=str(DEFAULT_DEMAND_WORKBOOK))
    parser.add_argument("--countries", default="CH,DE,FR,IT,AT")
    args = parser.parse_args(argv)

    frame = load_electrification_scenarios(Path(args.input), require_recommended=True)
    out, changes = build_p0_bridge(
        frame,
        entso_path=Path(args.entso),
        demand_workbook=Path(args.demand_workbook),
        countries=_parse_csv(args.countries),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    features = derive_hpfc_scenario_features(out)
    features_output = Path(args.features_output)
    features_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(features_output, index=False)
    write_report(Path(args.report), frame=out, changes=changes, output=output)
    print(f"[p0-bridge] rows={len(out)} changed_cells={len(changes)} -> {output}")
    print(f"[p0-bridge] features -> {features_output}")
    print(f"[p0-bridge] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

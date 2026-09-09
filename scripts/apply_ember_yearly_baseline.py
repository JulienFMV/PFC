"""Apply Ember yearly electricity baselines to LT scenario inventories.

Ember yearly data is a public historical/current baseline source. It is useful
to close traceable numeric gaps, but it is not a governed 2030 scenario path.
Rows touched by this script remain marked as proxy and must not pass strict
production governance without accountable approval.
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

DEFAULT_INPUT = Path("data/electrification_scenarios_composed_p0_real_sources_2030.parquet")
DEFAULT_OUTPUT = Path("data/electrification_scenarios_composed_p0_public_sources_2030.parquet")
DEFAULT_FEATURE_OUTPUT = Path("data/hpfc_scenario_features_composed_p0_public_sources_2030.parquet")
DEFAULT_COMPONENT_OUTPUT = Path("data/electrification_scenarios_ember_yearly_baseline_2026.parquet")
DEFAULT_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/EMBER-YEARLY-BASELINE.md")
DEFAULT_EMBER_CSV = (
    Path(r"C:\Users\jbattaglia\pfc_local_data\scenarios\ember_yearly")
    / "yearly_full_release_long_format.csv"
)

COMPONENT_ID = "ember_yearly_2026_baseline"
PUBLICATION_DATE = pd.Timestamp("2026-04-24", tz="UTC")
COUNTRY_MAP = {
    "AT": "Austria",
    "CH": "Switzerland",
    "DE": "Germany",
    "FR": "France",
    "IT": "Italy",
}


def _latest_value(
    frame: pd.DataFrame,
    *,
    area: str,
    category: str,
    variable: str,
    unit: str,
) -> tuple[float, int] | None:
    sub = frame[
        frame["Area"].astype(str).eq(area)
        & frame["Category"].astype(str).eq(category)
        & frame["Variable"].astype(str).eq(variable)
        & frame["Unit"].astype(str).eq(unit)
    ].copy()
    sub["Value"] = pd.to_numeric(sub["Value"], errors="coerce")
    sub["Year"] = pd.to_numeric(sub["Year"], errors="coerce")
    sub = sub.dropna(subset=["Value", "Year"]).sort_values("Year")
    if sub.empty:
        return None
    row = sub.iloc[-1]
    return float(row["Value"]), int(row["Year"])


def _positive_or_none(value_year: tuple[float, int] | None) -> tuple[float, int] | None:
    if value_year is None:
        return None
    value, year = value_year
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value, year


def load_ember_yearly_baseline(path: Path, *, countries: list[str]) -> pd.DataFrame:
    usecols = ["Area", "ISO 3 code", "Year", "Category", "Subcategory", "Variable", "Unit", "Value"]
    raw = pd.read_csv(path, usecols=usecols)
    rows: list[dict[str, object]] = []
    for country in countries:
        area = COUNTRY_MAP[country]
        field_specs = [
            ("hydro_twh", "Electricity generation", "Hydro", "TWh", True),
            ("hydro_capacity_gw", "Capacity", "Hydro", "GW", True),
            ("net_import_twh", "Electricity imports", "Net Imports", "TWh", False),
            ("gas_gw", "Capacity", "Gas", "GW", True),
            ("coal_gw", "Capacity", "Coal", "GW", True),
        ]
        values: dict[str, float] = {}
        source_years: dict[str, int] = {}
        for field, category, variable, unit, require_positive in field_specs:
            value_year = _latest_value(raw, area=area, category=category, variable=variable, unit=unit)
            if require_positive:
                value_year = _positive_or_none(value_year)
            elif value_year is not None and (not np.isfinite(value_year[0]) or value_year[0] == 0.0):
                value_year = None
            if value_year is None:
                continue
            values[field], source_years[field] = value_year

        dispatchable_parts = [
            values.get("gas_gw"),
            values.get("coal_gw"),
        ]
        nuclear = _positive_or_none(
            _latest_value(raw, area=area, category="Capacity", variable="Nuclear", unit="GW")
        )
        if nuclear is not None:
            dispatchable_parts.append(nuclear[0])
            source_years["dispatchable_gw"] = max(source_years.get("dispatchable_gw", 0), nuclear[1])
        positive_parts = [value for value in dispatchable_parts if value is not None and value > 0.0]
        if positive_parts:
            values["dispatchable_gw"] = float(sum(positive_parts))
            source_years["dispatchable_gw"] = max(
                [source_years.get(field, 0) for field in ["gas_gw", "coal_gw"] if field in source_years]
                + [source_years.get("dispatchable_gw", 0)]
            )

        for field, value in values.items():
            rows.append(
                {
                    "publication_date": PUBLICATION_DATE,
                    "source": "Ember yearly electricity data",
                    "component_id": COMPONENT_ID,
                    "country": country,
                    "area": area,
                    "field": field,
                    "value": float(value),
                    "source_year": int(source_years[field]),
                    "method": (
                        "latest positive gas+coal+nuclear capacity lower-bound"
                        if field == "dispatchable_gw"
                        else "latest reported Ember yearly value"
                    ),
                    "quality_flag": "official_historical_baseline_proxy",
                }
            )
    return pd.DataFrame(rows)


def _append_component_ids(value: object) -> str:
    components = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if COMPONENT_ID not in components:
        components.append(COMPONENT_ID)
    return ",".join(components)


def apply_ember_baseline(
    frame: pd.DataFrame,
    baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = validate_electrification_scenarios(frame, require_recommended=True).copy()
    changes: list[dict[str, object]] = []
    for row in baseline.itertuples(index=False):
        field = str(row.field)
        country = str(row.country)
        value = float(row.value)
        if field not in out.columns:
            raise KeyError(f"scenario inventory lacks {field}")
        country_mask = out["country"].astype(str).eq(country)
        missing = country_mask & out[field].isna()
        changed = out.loc[missing, ["country", "scenario", "delivery_year"]].copy()
        out.loc[missing, field] = value
        for changed_row in changed.itertuples(index=False):
            changes.append(
                {
                    "country": changed_row.country,
                    "scenario": changed_row.scenario,
                    "delivery_year": int(changed_row.delivery_year),
                    "column": field,
                    "value": value,
                    "source_year": int(row.source_year),
                    "method": row.method,
                }
            )
    if changes:
        out["quality_flag"] = out["quality_flag"].astype(str) + "_ember_yearly_baseline_proxy"
        out["source"] = out["source"].astype(str) + " + Ember yearly baseline"
        if "component_ids" in out.columns:
            out["component_ids"] = out["component_ids"].map(_append_component_ids)
    return validate_electrification_scenarios(out, require_recommended=True), pd.DataFrame(changes)


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        values = [str(row[c]).replace("|", "\\|").replace("\n", " ") for c in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    csv_path: Path,
    output: Path,
    component_output: Path,
    baseline: pd.DataFrame,
    changes: pd.DataFrame,
) -> None:
    lines = [
        "# Ember Yearly Baseline",
        "",
        f"* raw source: `{csv_path}`",
        f"* scenario output: `{output}`",
        f"* component output: `{component_output}`",
        "* status: `NON-PRODUCTION / OFFICIAL HISTORICAL BASELINE / PROXY`",
        "* publication date used for vintage checks: `2026-04-24`",
        "* filled fields: `hydro_twh`, `hydro_capacity_gw`, `net_import_twh`, `gas_gw`, `coal_gw`, `dispatchable_gw` where source values are present and non-zero.",
        "* deliberately not filled: `import_twh`, `export_twh`, `hydro_reservoir_twh` because Ember yearly data does not provide governed gross flows or reservoir energy capacity.",
        "",
        "## Baseline Values",
        "",
        _md_table(baseline.sort_values(["country", "field"])),
        "",
        "## Changed Scenario Cells",
        "",
        _md_table(changes),
        "",
        "## Production Interpretation",
        "",
        "This reduces numeric gaps with a true public source, but it remains a baseline projection proxy. "
        "Strict production governance must still fail until 2030 scenario values or explicit committee neutralisations are approved.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--ember-csv", default=str(DEFAULT_EMBER_CSV))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--features-output", default=str(DEFAULT_FEATURE_OUTPUT))
    parser.add_argument("--component-output", default=str(DEFAULT_COMPONENT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--countries", default="CH,DE,FR,IT,AT")
    args = parser.parse_args(argv)

    countries = [item.strip() for item in args.countries.split(",") if item.strip()]
    baseline = load_ember_yearly_baseline(Path(args.ember_csv), countries=countries)
    frame = load_electrification_scenarios(Path(args.input), require_recommended=True)
    out, changes = apply_ember_baseline(frame, baseline)

    component_output = Path(args.component_output)
    component_output.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_parquet(component_output, index=False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)

    features = derive_hpfc_scenario_features(out)
    features_output = Path(args.features_output)
    features_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(features_output, index=False)

    write_report(
        Path(args.report),
        csv_path=Path(args.ember_csv),
        output=output,
        component_output=component_output,
        baseline=baseline,
        changes=changes,
    )
    print(f"[ember-baseline] baseline_rows={len(baseline)} changed_cells={len(changes)} -> {output}")
    print(f"[ember-baseline] component -> {component_output}")
    print(f"[ember-baseline] features -> {features_output}")
    print(f"[ember-baseline] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

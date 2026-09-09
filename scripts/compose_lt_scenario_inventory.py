"""Compose available LT scenario components into one audited partial inventory.

This is a data-engineering bridge, not a production override. It merges the
official TYNDP 2024 supply component with the local CH EP2050 component on a
country/scenario/year key. Missing critical values remain null and the output is
marked partial so strict production gates continue to fail until the missing
governed feeds are loaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.data.electrification_scenarios import (
    CRITICAL_PRODUCTION_VALUE_COLUMNS,
    PRODUCTION_COLUMNS,
    asof_electrification_scenarios,
    assert_scenario_coverage,
    derive_hpfc_scenario_features,
    interpolate_electrification_scenario_years,
    load_electrification_scenarios,
    validate_electrification_scenarios,
)

DEFAULT_TYNDP_SUPPLY = Path("data/electrification_scenarios_tyndp2024_supply.parquet")
DEFAULT_CH_EP2050 = Path("data/electrification_scenarios_ep2050_enriched_slow_central_fast.parquet")
DEFAULT_NEIGHBOR_DEMAND = Path("data/electrification_scenarios_tyndp2024_neighbor_demand_bridge_2030.parquet")
DEFAULT_OUTPUT = Path("data/electrification_scenarios_composed_partial_2030.parquet")
DEFAULT_FEATURE_OUTPUT = Path("data/hpfc_scenario_features_composed_partial_2030.parquet")
DEFAULT_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/COMPOSED-PARTIAL-INVENTORY.md")

KEY_COLUMNS = ["country", "scenario", "delivery_year", "delivery_month"]
METADATA_COLUMNS = {
    "publication_date",
    "measurement_date",
    "track",
    "source",
    "component_ids",
    "scenario_edition",
    "quality_flag",
    "ingested_at_utc",
}
CH_EP2050_OVERLAY_COLUMNS = {
    "demand_twh",
    "peak_load_gw",
    "winter_demand_twh",
    "battery_power_gw",
    "ev_twh",
    "heatpump_twh",
    "hydro_twh",
    "hydro_capacity_gw",
    "hydro_reservoir_twh",
    "import_twh",
    "export_twh",
    "net_import_twh",
}
NEIGHBOR_DEMAND_OVERLAY_COLUMNS = {
    "demand_twh",
    "peak_load_gw",
    "winter_demand_twh",
}


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_years(value: str) -> list[int]:
    years: set[int] = set()
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            years.update(range(int(start), int(end) + 1))
        else:
            years.add(int(part))
    return sorted(years)


def _periods(years: list[int]) -> list[pd.Period]:
    return [pd.Period(f"{year}-{month:02d}", "M") for year in years for month in range(1, 13)]


def _dedupe(frame: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["publication_date", "ingested_at_utc"] if c in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols)
    return frame.drop_duplicates(KEY_COLUMNS, keep="last").reset_index(drop=True)


def _prepare_component(path: Path, *, years: list[int], allow_clamp: bool) -> pd.DataFrame:
    frame = load_electrification_scenarios(path, require_recommended=False)
    frame = interpolate_electrification_scenario_years(frame, years, allow_clamp=allow_clamp)
    frame = validate_electrification_scenarios(frame, require_recommended=False)
    for col in sorted(PRODUCTION_COLUMNS):
        if col not in frame.columns:
            frame[col] = np.nan
    return frame


def compose_partial_inventory(
    *,
    tyndp_supply_path: Path,
    ch_ep2050_path: Path,
    neighbor_demand_path: Path | None = None,
    years: list[int],
    vintage: str,
    countries: list[str],
    scenarios: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = _prepare_component(tyndp_supply_path, years=years, allow_clamp=False)
    base = base[
        base["country"].astype(str).isin(countries)
        & base["scenario"].astype(str).isin(scenarios)
        & base["delivery_year"].astype(int).isin(years)
    ].copy()
    base = _dedupe(base)

    ch = _prepare_component(ch_ep2050_path, years=years, allow_clamp=False)
    ch = ch[
        (ch["country"].astype(str) == "CH")
        & ch["scenario"].astype(str).isin(scenarios)
        & ch["delivery_year"].astype(int).isin(years)
    ].copy()
    ch = _dedupe(ch)
    neighbor_demand = None
    if neighbor_demand_path is not None:
        neighbor_demand = _prepare_component(neighbor_demand_path, years=years, allow_clamp=True)
        neighbor_demand = neighbor_demand[
            neighbor_demand["country"].astype(str).isin([c for c in countries if c != "CH"])
            & neighbor_demand["scenario"].astype(str).isin(scenarios)
            & neighbor_demand["delivery_year"].astype(int).isin(years)
        ].copy()
        neighbor_demand = _dedupe(neighbor_demand)

    merged = base.set_index(KEY_COLUMNS, drop=False).sort_index().copy()
    provenance_rows: list[dict[str, object]] = []
    for _, row in ch.iterrows():
        key = tuple(row[col] for col in KEY_COLUMNS)
        if key not in merged.index:
            continue
        for col in sorted(CH_EP2050_OVERLAY_COLUMNS & set(row.index)):
            value = row[col]
            if pd.isna(value):
                continue
            merged.loc[key, col] = value
            provenance_rows.append(
                {
                    "country": row["country"],
                    "scenario": row["scenario"],
                    "delivery_year": int(row["delivery_year"]),
                    "column": col,
                    "source": "CH EP2050 enriched slow/central/fast",
                }
            )
        merged.loc[key, "publication_date"] = max(
            pd.Timestamp(merged.loc[key, "publication_date"]),
            pd.Timestamp(row["publication_date"]),
        )
        merged.loc[key, "source"] = "ENTSO-E/ENTSOG TYNDP 2024 Supply Inputs + CH EP2050 enriched"
        merged.loc[key, "component_ids"] = "tyndp2024_supply,ch_ep2050_enriched"
        merged.loc[key, "scenario_edition"] = "TYNDP2024+EP2050_LOCAL"
        merged.loc[key, "quality_flag"] = "official_component_partial_ch_ep2050_proxy"
        merged.loc[key, "track"] = "scenario"
        if "ingested_at_utc" in merged.columns and pd.isna(merged.loc[key, "ingested_at_utc"]):
            merged.loc[key, "ingested_at_utc"] = pd.Timestamp(vintage, tz="UTC")

    if neighbor_demand is not None:
        for _, row in neighbor_demand.iterrows():
            key = tuple(row[col] for col in KEY_COLUMNS)
            if key not in merged.index:
                continue
            for col in sorted(NEIGHBOR_DEMAND_OVERLAY_COLUMNS & set(row.index)):
                value = row[col]
                if pd.isna(value):
                    continue
                merged.loc[key, col] = value
                provenance_rows.append(
                    {
                        "country": row["country"],
                        "scenario": row["scenario"],
                        "delivery_year": int(row["delivery_year"]),
                        "column": col,
                        "source": "TYNDP2024 neighbour demand bridge",
                    }
                )
            merged.loc[key, "publication_date"] = max(
                pd.Timestamp(merged.loc[key, "publication_date"]),
                pd.Timestamp(row["publication_date"]),
            )
            merged.loc[key, "source"] = "ENTSO-E/ENTSOG TYNDP 2024 Supply Inputs + demand bridge"
            merged.loc[key, "component_ids"] = "tyndp2024_supply,tyndp2024_neighbor_demand_bridge"
            merged.loc[key, "scenario_edition"] = "TYNDP2024_SUPPLY_DEMAND_BRIDGE"
            merged.loc[key, "quality_flag"] = "official_component_partial_neighbor_demand_proxy"
            merged.loc[key, "track"] = "scenario"

    out = merged.reset_index(drop=True)
    out["component_ids"] = out.get("component_ids", pd.Series(index=out.index, dtype="object")).fillna("tyndp2024_supply")
    out["quality_flag"] = out["quality_flag"].fillna("official_component_partial")
    out["track"] = out["track"].fillna("scenario")
    out["scenario_edition"] = out["scenario_edition"].fillna("TYNDP2024+EP2050_LOCAL")
    out["ingested_at_utc"] = pd.to_datetime(
        out["ingested_at_utc"].fillna(pd.Timestamp(vintage, tz="UTC")),
        utc=True,
    )
    out = validate_electrification_scenarios(out, require_recommended=True)

    asof = asof_electrification_scenarios(out, vintage)
    assert_scenario_coverage(
        asof,
        country="CH",
        scenarios=scenarios,
        periods=_periods(years),
    )
    for country in countries:
        assert_scenario_coverage(
            asof,
            country=country,
            scenarios=scenarios,
            periods=_periods(years),
        )

    features = derive_hpfc_scenario_features(out)
    provenance = pd.DataFrame(provenance_rows)
    return out, features, provenance


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        vals = [str(row[c]).replace("|", "\\|").replace("\n", " ") for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_report(
    path: Path,
    *,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    provenance: pd.DataFrame,
    output: Path,
    features_output: Path,
) -> None:
    critical = sorted(CRITICAL_PRODUCTION_VALUE_COLUMNS & set(frame.columns))
    gaps = frame[critical].isna().sum().reset_index()
    gaps.columns = ["column", "missing_rows"]
    gaps = gaps[gaps["missing_rows"] > 0].sort_values(["missing_rows", "column"], ascending=[False, True])
    coverage = (
        frame.groupby(["country", "scenario"])["delivery_year"]
        .apply(lambda s: ", ".join(str(int(y)) for y in sorted(s.unique())))
        .reset_index(name="years")
    )
    lines = [
        "# Composed Partial LT Scenario Inventory",
        "",
        f"* scenario output: `{output}`",
        f"* feature output: `{features_output}`",
        f"* rows: `{len(frame)}`",
        f"* feature rows: `{len(features)}`",
        "* status: `PARTIAL - production gate must fail`",
        "* rule: no missing critical value is converted to zero",
        "",
        "## Coverage",
        "",
        _md_table(coverage),
        "",
        "## Remaining Critical Gaps",
        "",
        _md_table(gaps),
        "",
        "## Overlay Provenance",
        "",
        _md_table(
            provenance.groupby(["country", "scenario", "delivery_year", "source"])["column"]
            .apply(lambda s: ", ".join(sorted(set(s))))
            .reset_index()
            if len(provenance)
            else provenance
        ),
        "",
        "## Decision",
        "",
        "This composed file is a stronger smoke/prod-readiness inventory than the CH-only proxy because it carries official multi-country TYNDP supply rows, bridged neighbour 2030 demand when provided, and CH EP2050 overlays. It remains non-production because neighbour peak/winter load, NTC, dispatchable capacity, hydro, EV/heat-pump/flex and committee-approved scenario mapping are still incomplete.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tyndp-supply", default=str(DEFAULT_TYNDP_SUPPLY))
    parser.add_argument("--ch-ep2050", default=str(DEFAULT_CH_EP2050))
    parser.add_argument("--neighbor-demand", default=str(DEFAULT_NEIGHBOR_DEMAND))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--features-output", default=str(DEFAULT_FEATURE_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--years", default="2030")
    parser.add_argument("--vintage", default="2026-06-11")
    parser.add_argument("--countries", default="CH,DE,FR,IT,AT")
    parser.add_argument("--scenarios", default="slow,central,fast")
    args = parser.parse_args(argv)

    frame, features, provenance = compose_partial_inventory(
        tyndp_supply_path=Path(args.tyndp_supply),
        ch_ep2050_path=Path(args.ch_ep2050),
        neighbor_demand_path=Path(args.neighbor_demand) if args.neighbor_demand else None,
        years=_parse_years(args.years),
        vintage=args.vintage,
        countries=_parse_csv(args.countries),
        scenarios=_parse_csv(args.scenarios),
    )
    output = Path(args.output)
    features_output = Path(args.features_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    features.to_parquet(features_output, index=False)
    _write_report(
        Path(args.report),
        frame=frame,
        features=features,
        provenance=provenance,
        output=output,
        features_output=features_output,
    )
    print(f"[compose] scenarios rows={len(frame)} -> {output}")
    print(f"[compose] features rows={len(features)} -> {features_output}")
    print(f"[compose] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

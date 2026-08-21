"""Build a governed LT scenario data gap register.

The register turns the strict scenario-governance failures into an actionable
data backlog. It does not fill values and never converts critical missing data
to zero; it only classifies blockers by data family, priority and expected
source owner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scripts.validate_scenario_governance import _parse_csv, _parse_years, validate_governance
from pfc_shaping.data.electrification_scenarios import load_electrification_scenarios

DEFAULT_MANIFEST = Path(".planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-MANIFEST.yaml")
DEFAULT_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/LT-DATA-GAP-REGISTER.md")
DEFAULT_CSV = Path("data/lt_scenario_governance_gap_register.csv")


FIELD_MAP: dict[str, dict[str, str]] = {
    "demand_twh": {
        "family": "demand",
        "priority": "P0",
        "owner": "scenario_data",
        "source": "OFEN EP2050+ for CH; TYNDP Demand or national TSO demand scenario for neighbours",
        "action": "Load governed annual demand by country/scenario/year.",
    },
    "peak_load_gw": {
        "family": "demand_shape",
        "priority": "P0",
        "owner": "entsoe_timeseries",
        "source": "ENTSO-E realised load plus governed LT demand scaling",
        "action": "Derive peak-load ratio from non-empty historical load and apply to governed demand.",
    },
    "winter_demand_twh": {
        "family": "demand_shape",
        "priority": "P0",
        "owner": "entsoe_timeseries",
        "source": "ENTSO-E realised load plus governed LT demand scaling",
        "action": "Derive winter demand share from non-empty historical load and apply to governed demand.",
    },
    "pv_twh": {
        "family": "renewables_energy",
        "priority": "P0",
        "owner": "renewables_scenarios",
        "source": "TYNDP Supply, OFEN EP2050+, national renewable statistics",
        "action": "Add governed PV energy, not only installed capacity.",
    },
    "wind_twh": {
        "family": "renewables_energy",
        "priority": "P0",
        "owner": "renewables_scenarios",
        "source": "TYNDP Supply and national renewable statistics",
        "action": "Add governed wind energy, not only installed capacity.",
    },
    "battery_power_gw": {
        "family": "storage_flex",
        "priority": "P0",
        "owner": "flex_scenarios",
        "source": "TYNDP Supply, MaStR/BNetzA, Pronovo or internal approved storage scenario",
        "action": "Add governed battery power and align with battery energy duration.",
    },
    "battery_energy_gwh": {
        "family": "storage_flex",
        "priority": "P0",
        "owner": "flex_scenarios",
        "source": "TYNDP Supply, MaStR/BNetzA, Pronovo or internal approved storage scenario",
        "action": "Add governed battery energy and align with battery power duration.",
    },
    "ev_twh": {
        "family": "electrification",
        "priority": "P0",
        "owner": "electrification_scenarios",
        "source": "OFEN EP2050+, TYNDP Demand transport components, national mobility scenario",
        "action": "Add governed EV electricity demand by country/scenario/year.",
    },
    "heatpump_twh": {
        "family": "electrification",
        "priority": "P0",
        "owner": "electrification_scenarios",
        "source": "OFEN EP2050+, TYNDP Demand heating components, national heat scenario",
        "action": "Add governed heat-pump electricity demand by country/scenario/year.",
    },
    "managed_charging_share": {
        "family": "electrification_flex",
        "priority": "P1",
        "owner": "flex_scenarios",
        "source": "risk committee assumption or published smart-charging scenario",
        "action": "Approve bounded [0,1] managed charging share; do not infer silently.",
    },
    "hydro_twh": {
        "family": "hydro",
        "priority": "P0",
        "owner": "hydro_scenarios",
        "source": "OFEN EP2050+, Swissgrid/ENTSO-E hydro statistics",
        "action": "Add governed hydro annual energy by country/scenario/year.",
    },
    "hydro_capacity_gw": {
        "family": "hydro",
        "priority": "P0",
        "owner": "hydro_scenarios",
        "source": "OFEN EP2050+, Swissgrid/ENTSO-E installed capacity",
        "action": "Add governed hydro capacity by country/scenario/year.",
    },
    "hydro_reservoir_twh": {
        "family": "hydro",
        "priority": "P0",
        "owner": "hydro_scenarios",
        "source": "OFEN EP2050+, Swiss reservoir statistics",
        "action": "Add governed reservoir energy capacity or explicitly approved neutralisation.",
    },
    "nuclear_gw": {
        "family": "firm_capacity",
        "priority": "P0",
        "owner": "supply_scenarios",
        "source": "TYNDP Supply, national phase-out plans, OFEN EP2050+",
        "action": "Add governed nuclear capacity by country/scenario/year.",
    },
    "dispatchable_gw": {
        "family": "firm_capacity",
        "priority": "P0",
        "owner": "supply_scenarios",
        "source": "TYNDP Supply, ENTSO-E installed capacity, national adequacy reports",
        "action": "Add governed dispatchable capacity excluding intermittent renewables.",
    },
    "gas_gw": {
        "family": "thermal_capacity",
        "priority": "P1",
        "owner": "supply_scenarios",
        "source": "TYNDP Supply and national adequacy reports",
        "action": "Add governed gas capacity for thermal-set diagnostics.",
    },
    "coal_gw": {
        "family": "thermal_capacity",
        "priority": "P1",
        "owner": "supply_scenarios",
        "source": "TYNDP Supply and national coal phase-out plans",
        "action": "Add governed coal capacity for thermal-set diagnostics.",
    },
    "import_twh": {
        "family": "cross_border_balance",
        "priority": "P0",
        "owner": "cross_border_scenarios",
        "source": "OFEN EP2050+, TYNDP, Swissgrid/TSO adequacy assumptions",
        "action": "Add governed import energy by country/scenario/year.",
    },
    "export_twh": {
        "family": "cross_border_balance",
        "priority": "P0",
        "owner": "cross_border_scenarios",
        "source": "OFEN EP2050+, TYNDP, Swissgrid/TSO adequacy assumptions",
        "action": "Add governed export energy by country/scenario/year.",
    },
    "net_import_twh": {
        "family": "cross_border_balance",
        "priority": "P0",
        "owner": "cross_border_scenarios",
        "source": "OFEN EP2050+, TYNDP, Swissgrid/TSO adequacy assumptions",
        "action": "Add governed net import or derive explicitly from governed import/export.",
    },
    "cross_border_balance": {
        "family": "cross_border_balance",
        "priority": "P0",
        "owner": "cross_border_scenarios",
        "source": "OFEN EP2050+, TYNDP, Swissgrid/TSO adequacy assumptions",
        "action": "Provide governed net_import_twh or both governed import_twh and export_twh.",
    },
    "gas_eur_mwh": {
        "family": "commodities",
        "priority": "P1",
        "owner": "market_data",
        "source": "market forward curves or approved commodity scenario",
        "action": "Add gas scenario for thermal consistency diagnostics.",
    },
    "coal_eur_mwh": {
        "family": "commodities",
        "priority": "P1",
        "owner": "market_data",
        "source": "market forward curves or approved commodity scenario",
        "action": "Add coal scenario for thermal consistency diagnostics.",
    },
    "co2_eur_t": {
        "family": "commodities",
        "priority": "P1",
        "owner": "market_data",
        "source": "EUA market curve or approved CO2 scenario",
        "action": "Add CO2 scenario for thermal consistency diagnostics.",
    },
    "electrolysis_twh": {
        "family": "flex_demand",
        "priority": "P1",
        "owner": "flex_scenarios",
        "source": "TYNDP hydrogen/electrolysis assumptions or approved internal scenario",
        "action": "Add governed electrolysis demand or explicit approved neutralisation.",
    },
    "p2x_gw": {
        "family": "flex_demand",
        "priority": "P1",
        "owner": "flex_scenarios",
        "source": "TYNDP hydrogen/electrolysis assumptions or approved internal scenario",
        "action": "Add governed P2X capacity or explicit approved neutralisation.",
    },
    "dsm_gw": {
        "family": "flex_demand",
        "priority": "P1",
        "owner": "flex_scenarios",
        "source": "TYNDP DSM/flex assumptions or approved internal scenario",
        "action": "Add governed DSM capacity or explicit approved neutralisation.",
    },
}


def _field_meta(column: str) -> dict[str, str]:
    if column.startswith("ntc_"):
        return {
            "family": "interconnection",
            "priority": "P0",
            "owner": "cross_border_scenarios",
            "source": "Swissgrid, TYNDP, JAO/TSO long-term capacity assumptions",
            "action": "Add governed NTC assumption; local ENTSO-E cache currently has no usable NTC observations.",
        }
    return FIELD_MAP.get(
        column,
        {
            "family": "unknown",
            "priority": "P1",
            "owner": "scenario_data",
            "source": "governed source required",
            "action": "Classify and populate this governed production field.",
        },
    )


def build_gap_register(issues: list[dict[str, object]], *, effective_rows: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for issue in issues:
        name = str(issue["issue"])
        detail = str(issue["detail"])
        issue_rows = list(issue.get("rows") or [])
        if name == "missing_critical_values":
            for item in issue_rows:
                column = str(item["column"])
                meta = _field_meta(column)
                rows.append(
                    {
                        "priority": meta["priority"],
                        "issue": name,
                        "family": meta["family"],
                        "field": column,
                        "missing_rows": int(item.get("missing_rows", 0)),
                        "effective_rows": effective_rows,
                        "country": "",
                        "scenario": "",
                        "delivery_year": "",
                        "owner": meta["owner"],
                        "source_candidate": meta["source"],
                        "required_action": meta["action"],
                        "prod_status": "BLOCKING",
                    }
                )
            continue
        if name == "missing_conditional_critical_groups":
            for item in issue_rows:
                group = str(item.get("group", "conditional_group"))
                meta = _field_meta(group)
                rows.append(
                    {
                        "priority": meta["priority"],
                        "issue": name,
                        "family": meta["family"],
                        "field": group,
                        "missing_rows": 1,
                        "effective_rows": effective_rows,
                        "country": item.get("country", ""),
                        "scenario": item.get("scenario", ""),
                        "delivery_year": item.get("delivery_year", ""),
                        "owner": meta["owner"],
                        "source_candidate": meta["source"],
                        "required_action": (
                            "Populate one complete governed alternative for this group: "
                            f"{item.get('alternatives', '')}."
                        ),
                        "prod_status": "BLOCKING",
                    }
                )
            continue
        if name == "missing_production_columns":
            for item in issue_rows:
                column = str(item["column"])
                meta = _field_meta(column)
                rows.append(
                    {
                        "priority": meta["priority"],
                        "issue": name,
                        "family": meta["family"],
                        "field": column,
                        "missing_rows": effective_rows,
                        "effective_rows": effective_rows,
                        "country": "",
                        "scenario": "",
                        "delivery_year": "",
                        "owner": meta["owner"],
                        "source_candidate": meta["source"],
                        "required_action": f"Add required production column `{column}`. {meta['action']}",
                        "prod_status": "BLOCKING",
                    }
                )
            continue
        if name == "unacceptable_quality_flags":
            for item in issue_rows:
                rows.append(
                    {
                        "priority": "P0",
                        "issue": name,
                        "family": "data_quality",
                        "field": "quality_flag",
                        "missing_rows": 0,
                        "effective_rows": effective_rows,
                        "country": item.get("country", ""),
                        "scenario": item.get("scenario", ""),
                        "delivery_year": item.get("delivery_year", ""),
                        "owner": "scenario_data",
                        "source_candidate": "official governed scenario component",
                        "required_action": f"Replace or approve source so quality_flag is production-governed: {item.get('quality_flag', '')}",
                        "prod_status": "BLOCKING",
                    }
                )
            continue
        if name == "missing_inventory_coverage":
            for item in issue_rows:
                rows.append(
                    {
                        "priority": "P0",
                        "issue": name,
                        "family": "coverage",
                        "field": "row",
                        "missing_rows": 1,
                        "effective_rows": effective_rows,
                        "country": item.get("country", ""),
                        "scenario": item.get("scenario", ""),
                        "delivery_year": item.get("delivery_year", ""),
                        "owner": "scenario_data",
                        "source_candidate": "governed scenario inventory",
                        "required_action": "Add the missing country/scenario/year row with publication_date <= vintage.",
                        "prod_status": "BLOCKING",
                    }
                )
            continue
        if name == "source_component_issues":
            for item in issue_rows:
                rows.append(
                    {
                        "priority": "P0",
                        "issue": name,
                        "family": "source_metadata",
                        "field": str(item.get("issue", "")),
                        "missing_rows": 0,
                        "effective_rows": effective_rows,
                        "country": "",
                        "scenario": "",
                        "delivery_year": "",
                        "owner": "data_engineering",
                        "source_candidate": str(item.get("source_id", "")),
                        "required_action": f"Fix governed source component metadata/path: {item.get('local_path', '')}",
                        "prod_status": "BLOCKING",
                    }
                )
            continue
        if name in {"missing_source_component_links", "source_component_link_issues"}:
            if not issue_rows:
                issue_rows = [{}]
            for item in issue_rows:
                rows.append(
                    {
                        "priority": "P0",
                        "issue": name,
                        "family": "source_provenance",
                        "field": "component_ids",
                        "missing_rows": 1 if item else effective_rows,
                        "effective_rows": effective_rows,
                        "country": item.get("country", ""),
                        "scenario": item.get("scenario", ""),
                        "delivery_year": item.get("delivery_year", ""),
                        "owner": "data_engineering",
                        "source_candidate": str(item.get("unknown_component_ids", "governed source_components")),
                        "required_action": "Add row-level source_id/component_ids that reference manifest source_components.",
                        "prod_status": "BLOCKING",
                    }
                )
            continue
        if name == "unjustified_zero_values":
            for item in issue_rows:
                column = str(item.get("column", ""))
                meta = _field_meta(column)
                rows.append(
                    {
                        "priority": meta["priority"],
                        "issue": name,
                        "family": meta["family"],
                        "field": column,
                        "missing_rows": 0,
                        "effective_rows": effective_rows,
                        "country": item.get("country", ""),
                        "scenario": item.get("scenario", ""),
                        "delivery_year": item.get("delivery_year", ""),
                        "owner": meta["owner"],
                        "source_candidate": meta["source"],
                        "required_action": "Provide explicit zero justification or add the field to governed zero_allowed_fields.",
                        "prod_status": "BLOCKING",
                    }
                )
            continue
        if name == "physical_bounds_violations":
            for item in issue_rows:
                column = str(item.get("column", ""))
                meta = _field_meta(column)
                rows.append(
                    {
                        "priority": meta["priority"],
                        "issue": name,
                        "family": meta["family"],
                        "field": column,
                        "missing_rows": 0,
                        "effective_rows": effective_rows,
                        "country": item.get("country", ""),
                        "scenario": item.get("scenario", ""),
                        "delivery_year": item.get("delivery_year", ""),
                        "owner": meta["owner"],
                        "source_candidate": meta["source"],
                        "required_action": f"Review physical bound breach value={item.get('value', '')} min={item.get('min', '')} max={item.get('max', '')}.",
                        "prod_status": "BLOCKING",
                    }
                )
            continue
        rows.append(
            {
                "priority": "P0",
                "issue": name,
                "family": "governance_decision",
                "field": "",
                "missing_rows": 0,
                "effective_rows": effective_rows,
                "country": "",
                "scenario": "",
                "delivery_year": "",
                "owner": "risk_committee",
                "source_candidate": "scenario governance manifest",
                "required_action": detail,
                "prod_status": "BLOCKING",
            }
        )
    columns = [
        "priority",
        "issue",
        "family",
        "field",
        "missing_rows",
        "effective_rows",
        "country",
        "scenario",
        "delivery_year",
        "owner",
        "source_candidate",
        "required_action",
        "prod_status",
    ]
    return pd.DataFrame(rows, columns=columns)


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    lines = [
        "| " + " | ".join(str(c) for c in df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[c]).replace("|", "\\|").replace("\n", " ") for c in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path: Path, *, register: pd.DataFrame, inventory_path: Path, manifest_path: Path, vintage: str) -> None:
    grouped = (
        register.groupby(["priority", "family"], dropna=False)
        .size()
        .reset_index(name="gap_count")
        .sort_values(["priority", "family"])
        if not register.empty
        else pd.DataFrame(columns=["priority", "family", "gap_count"])
    )
    top_fields = (
        register.loc[register["field"].astype(str) != ""]
        .groupby(["priority", "field", "owner", "source_candidate", "required_action"], dropna=False)
        .agg(missing_rows=("missing_rows", "max"))
        .reset_index()
        .sort_values(["priority", "field"])
        if not register.empty
        else pd.DataFrame(columns=["priority", "field", "owner", "source_candidate", "required_action", "missing_rows"])
    )
    lines = [
        "# LT Scenario Data Gap Register",
        "",
        f"* inventory: `{inventory_path}`",
        f"* manifest: `{manifest_path}`",
        f"* vintage: `{vintage}`",
        f"* blocking gaps: `{len(register)}`",
        "",
        "## Gap Summary",
        "",
        _md_table(grouped),
        "",
        "## Field-Level Actions",
        "",
        _md_table(top_fields),
        "",
        "## Full Register",
        "",
        _md_table(register),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--vintage", required=True)
    parser.add_argument("--countries", default="CH,DE,FR,IT,AT")
    parser.add_argument("--scenarios", default="slow,central,fast")
    parser.add_argument("--years", default="2030")
    parser.add_argument("--csv-output", default=str(DEFAULT_CSV))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args(argv)

    inventory_path = Path(args.inventory)
    manifest_path = Path(args.manifest)
    inventory = load_electrification_scenarios(inventory_path, require_recommended=True)
    manifest: dict[str, Any] = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    issues, effective = validate_governance(
        inventory=inventory,
        manifest=manifest,
        vintage=args.vintage,
        countries=_parse_csv(args.countries),
        scenarios=_parse_csv(args.scenarios),
        years=_parse_years(args.years),
    )
    register = build_gap_register(issues, effective_rows=len(effective))
    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    register.to_csv(csv_path, index=False)
    write_report(
        Path(args.report),
        register=register,
        inventory_path=inventory_path,
        manifest_path=manifest_path,
        vintage=args.vintage,
    )
    print(f"[gap-register] rows={len(register)}")
    print(f"[gap-register] csv -> {csv_path}")
    print(f"[gap-register] report -> {args.report}")
    return 0 if register.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())

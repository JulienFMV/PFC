"""Validate committee-style governance for LT scenario inventories.

This is stricter than schema validation. It checks that a scenario inventory is
covered by an approved governance manifest: scope, weights, source components,
approval date, quality flags and critical production values.

Production mode requires accountable human approval and rejects proxy/partial
quality flags. Local-test mode can accept documented expert-agent approval and
traceable proxy/partial flags, but it is explicitly not a production approval.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pfc_shaping.data.electrification_scenarios import (
    CONDITIONAL_PRODUCTION_FIELD_GROUPS,
    COUNTRY_SCOPED_CRITICAL_VALUE_COLUMNS,
    CRITICAL_PRODUCTION_VALUE_COLUMNS,
    PRODUCTION_COLUMNS,
    UNACCEPTABLE_PRODUCTION_QUALITY_TOKENS,
    asof_electrification_scenarios,
    load_electrification_scenarios,
)

DEFAULT_MANIFEST = Path(".planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-MANIFEST.yaml")
DEFAULT_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-VALIDATION.md")
REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _to_utc(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _effective_rows(
    inventory: pd.DataFrame,
    *,
    vintage: str,
    countries: list[str],
    scenarios: list[str],
    years: list[int],
) -> pd.DataFrame:
    asof = asof_electrification_scenarios(inventory, vintage)
    scoped = asof[
        asof["country"].astype(str).isin(countries)
        & asof["scenario"].astype(str).isin(scenarios)
        & asof["delivery_year"].astype(int).isin(years)
    ].copy()
    if scoped.empty:
        return scoped
    return (
        scoped.sort_values("publication_date")
        .drop_duplicates(["country", "scenario", "delivery_year"], keep="last")
        .reset_index(drop=True)
    )


def _issue(name: str, detail: str, rows: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {"issue": name, "detail": detail, "rows": rows or []}


def _resolve_local_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _split_component_ids(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _country_scoped_critical_fields(rules: dict[str, Any]) -> dict[str, set[str]]:
    scoped = {
        str(field): {str(country) for country in countries}
        for field, countries in COUNTRY_SCOPED_CRITICAL_VALUE_COLUMNS.items()
    }
    for field, countries in (rules.get("country_scoped_critical_fields") or {}).items():
        scoped[str(field)] = {str(country) for country in (countries or [])}
    return scoped


def _conditional_field_groups(rules: dict[str, Any]) -> dict[str, tuple[tuple[str, ...], ...]]:
    groups = {
        str(name): tuple(tuple(str(field) for field in option) for option in alternatives)
        for name, alternatives in CONDITIONAL_PRODUCTION_FIELD_GROUPS.items()
    }
    for name, rule in (rules.get("conditional_critical_groups") or {}).items():
        alternatives = rule.get("require_any", rule) if isinstance(rule, dict) else rule
        groups[str(name)] = tuple(tuple(str(field) for field in option) for option in (alternatives or []))
    return groups


def _conditional_group_fields(groups: dict[str, tuple[tuple[str, ...], ...]]) -> set[str]:
    return {
        field
        for alternatives in groups.values()
        for option in alternatives
        for field in option
    }


def validate_governance(
    *,
    inventory: pd.DataFrame,
    manifest: dict[str, Any],
    vintage: str,
    countries: list[str],
    scenarios: list[str],
    years: list[int],
    mode: str = "production",
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    issues: list[dict[str, object]] = []
    vintage_ts = _to_utc(vintage)
    if vintage_ts is None:
        raise ValueError("vintage is required")
    if mode not in {"production", "local-test"}:
        raise ValueError("mode must be one of ['production', 'local-test']")

    effective = _effective_rows(
        inventory,
        vintage=vintage,
        countries=countries,
        scenarios=scenarios,
        years=years,
    )

    if mode == "production" and (
        manifest.get("status") != "approved" or manifest.get("approved_for_production") is not True
    ):
        issues.append(
            _issue(
                "manifest_not_approved",
                "manifest status must be approved and approved_for_production must be true",
            )
        )
    if mode == "local-test":
        if manifest.get("status") not in {"agent_approved_local_test", "approved_local_test"}:
            issues.append(
                _issue(
                    "local_test_manifest_not_approved",
                    "local-test mode requires status agent_approved_local_test or approved_local_test",
                )
            )
        if manifest.get("approved_for_production") is True:
            issues.append(
                _issue(
                    "local_test_manifest_claims_production",
                    "local-test manifest must not set approved_for_production=true",
                )
            )
        if manifest.get("approved_for_local_test") is not True:
            issues.append(
                _issue(
                    "local_test_not_approved",
                    "local-test mode requires approved_for_local_test=true",
                )
            )

    approval_date = _to_utc(manifest.get("approval_date"))
    if approval_date is None:
        issues.append(_issue("missing_approval_date", "approval_date is required"))
    elif approval_date > vintage_ts:
        issues.append(_issue("approval_after_vintage", f"{approval_date} > {vintage_ts}"))

    approved_by = manifest.get("approved_by") or []
    if mode == "production" and (not isinstance(approved_by, list) or not approved_by):
        issues.append(_issue("missing_approvers", "approved_by must contain at least one approver"))
    if mode == "local-test":
        agent_approvers = manifest.get("local_test_approved_by_agents") or []
        if not isinstance(agent_approvers, list) or not agent_approvers:
            issues.append(
                _issue(
                    "missing_agent_approvers",
                    "local-test mode requires local_test_approved_by_agents",
                )
            )
        else:
            rows = []
            for approver in agent_approvers:
                if not isinstance(approver, dict) or not approver.get("role") or not approver.get("verdict"):
                    rows.append({"approver": str(approver)})
                elif "production" in str(approver.get("verdict", "")).lower() and "no_go" not in str(approver.get("verdict", "")).lower():
                    rows.append({"approver": str(approver), "issue": "agent verdict must not claim production approval"})
            if rows:
                issues.append(
                    _issue(
                        "invalid_agent_approvers",
                        "agent approvers must have role/verdict and cannot claim production approval",
                        rows,
                    )
                )

    scope = manifest.get("scope") or {}
    for label, expected in [("countries", countries), ("scenarios", scenarios), ("years", years)]:
        declared = scope.get(label) or []
        declared_s = {str(v) for v in declared}
        expected_s = {str(v) for v in expected}
        if declared_s != expected_s:
            issues.append(
                _issue(
                    f"scope_{label}_mismatch",
                    f"declared={sorted(declared_s)} expected={sorted(expected_s)}",
                )
            )

    missing_coverage: list[dict[str, object]] = []
    for country in countries:
        for scenario in scenarios:
            for year in years:
                has_row = bool(
                    (
                        (effective["country"].astype(str) == str(country))
                        & (effective["scenario"].astype(str) == str(scenario))
                        & (effective["delivery_year"].astype(int) == int(year))
                    ).any()
                ) if len(effective) else False
                if not has_row:
                    missing_coverage.append({"country": country, "scenario": scenario, "delivery_year": int(year)})
    if missing_coverage:
        issues.append(_issue("missing_inventory_coverage", "inventory lacks governed rows", missing_coverage))

    weights = manifest.get("scenario_weights") or {}
    missing_weights = [scenario for scenario in scenarios if scenario not in weights]
    if missing_weights:
        issues.append(_issue("missing_scenario_weights", ", ".join(missing_weights)))
    else:
        weight_sum = sum(float(weights[scenario]) for scenario in scenarios)
        if abs(weight_sum - 1.0) > 1e-9:
            issues.append(_issue("scenario_weights_not_one", f"weight_sum={weight_sum:.12f}"))
        if len(effective) and "scenario_weight" in effective.columns:
            rows: list[dict[str, object]] = []
            for scenario in scenarios:
                expected = float(weights[scenario])
                values = pd.to_numeric(
                    effective.loc[effective["scenario"].astype(str) == scenario, "scenario_weight"],
                    errors="coerce",
                ).dropna()
                if values.empty or (values.sub(expected).abs() > 1e-9).any():
                    rows.append({"scenario": scenario, "expected_weight": expected})
            if rows:
                issues.append(_issue("inventory_weight_mismatch", "inventory weights differ from manifest", rows))

    rules = manifest.get("rules") or {}
    country_scoped_critical = _country_scoped_critical_fields(rules)
    conditional_groups = _conditional_field_groups(rules)
    component_ids = {
        str(source.get("source_id"))
        for source in (manifest.get("source_components") or [])
        if source.get("source_id")
    }
    if bool(rules.get("require_source_component_links", False)):
        if len(effective) and not ({"source_id", "component_ids"} & set(effective.columns)):
            issues.append(
                _issue(
                    "missing_source_component_links",
                    "inventory rows must carry source_id or component_ids",
                )
            )
        elif len(effective):
            rows: list[dict[str, object]] = []
            for _, row in effective.iterrows():
                row_components = []
                if "component_ids" in effective.columns:
                    row_components.extend(_split_component_ids(row.get("component_ids")))
                if "source_id" in effective.columns:
                    row_components.extend(_split_component_ids(row.get("source_id")))
                missing = sorted({component for component in row_components if component not in component_ids})
                if not row_components or missing:
                    rows.append(
                        {
                            "country": row.get("country", ""),
                            "scenario": row.get("scenario", ""),
                            "delivery_year": row.get("delivery_year", ""),
                            "component_ids": ",".join(row_components),
                            "unknown_component_ids": ",".join(missing),
                        }
                    )
            if rows:
                issues.append(
                    _issue(
                        "source_component_link_issues",
                        "effective inventory rows must reference declared source_components",
                        rows,
                    )
                )

    allow_bad_quality = bool(rules.get("allow_proxy_or_partial_quality_flags", False))
    if not allow_bad_quality and len(effective) and "quality_flag" in effective.columns:
        flags = effective["quality_flag"].fillna("").astype(str).str.lower()
        bad = flags.map(lambda value: any(token in value for token in UNACCEPTABLE_PRODUCTION_QUALITY_TOKENS))
        if bool(bad.any()):
            rows = (
                effective.loc[bad, ["country", "scenario", "delivery_year", "quality_flag"]]
                .drop_duplicates()
                .to_dict("records")
            )
            issues.append(_issue("unacceptable_quality_flags", "proxy/partial/internal/fallback/synthetic flags are not production-governed", rows))

    if bool(rules.get("require_complete_critical_values", True)):
        missing_production_columns = sorted(PRODUCTION_COLUMNS - set(effective.columns))
        if missing_production_columns:
            issues.append(
                _issue(
                    "missing_production_columns",
                    "production columns are absent from the governed inventory",
                    [{"column": column} for column in missing_production_columns],
                )
            )
        critical = sorted(CRITICAL_PRODUCTION_VALUE_COLUMNS & set(effective.columns))
        if critical and len(effective):
            rows = [
                {"column": column, "missing_rows": int(mask.sum())}
                for column in critical
                for mask in [
                    effective[column].isna()
                    & (
                        effective["country"].astype(str).isin(country_scoped_critical[column])
                        if column in country_scoped_critical
                        else True
                    )
                ]
                if int(mask.sum()) > 0
            ]
            if rows:
                issues.append(_issue("missing_critical_values", "critical production fields contain nulls", rows))
        if conditional_groups and len(effective):
            rows = []
            for group_name, alternatives in conditional_groups.items():
                for _, row in effective.iterrows():
                    ok = any(
                        all(field in effective.columns and pd.notna(row.get(field)) for field in option)
                        for option in alternatives
                    )
                    if ok:
                        continue
                    rows.append(
                        {
                            "group": group_name,
                            "country": row.get("country", ""),
                            "scenario": row.get("scenario", ""),
                            "delivery_year": row.get("delivery_year", ""),
                            "alternatives": " OR ".join("+".join(option) for option in alternatives),
                        }
                    )
            if rows:
                issues.append(
                    _issue(
                        "missing_conditional_critical_groups",
                        "conditional production field groups have no complete alternative",
                        rows,
                    )
                )

    if bool(rules.get("require_zero_justification", False)) and len(effective):
        zero_allowed_fields = set(rules.get("zero_allowed_fields") or [])
        conditional_fields = _conditional_group_fields(conditional_groups)
        critical_numeric = sorted(
            ((CRITICAL_PRODUCTION_VALUE_COLUMNS | conditional_fields) & set(effective.columns))
            - {"track", "source", "scenario_edition", "quality_flag", "ingested_at_utc"}
        )
        rows = []
        for column in critical_numeric:
            if column in zero_allowed_fields:
                continue
            values = pd.to_numeric(effective[column], errors="coerce")
            zero_mask = values.notna() & values.eq(0.0)
            if not bool(zero_mask.any()):
                continue
            justification_col = f"{column}_zero_justification"
            if justification_col in effective.columns:
                justified = effective.loc[zero_mask, justification_col].fillna("").astype(str).str.strip().ne("")
                zero_rows = effective.loc[zero_mask].loc[~justified]
            else:
                zero_rows = effective.loc[zero_mask]
            for _, row in zero_rows.iterrows():
                rows.append(
                    {
                        "country": row.get("country", ""),
                        "scenario": row.get("scenario", ""),
                        "delivery_year": row.get("delivery_year", ""),
                        "column": column,
                    }
                )
        if rows:
            issues.append(
                _issue(
                    "unjustified_zero_values",
                    "critical zero values require explicit justification or zero_allowed_fields",
                    rows,
                )
            )

    physical_bounds = rules.get("physical_bounds") or {}
    if physical_bounds and len(effective):
        rows = []
        for column, bounds in physical_bounds.items():
            if column not in effective.columns:
                continue
            values = pd.to_numeric(effective[column], errors="coerce")
            min_value = bounds.get("min") if isinstance(bounds, dict) else None
            max_value = bounds.get("max") if isinstance(bounds, dict) else None
            bad = pd.Series(False, index=effective.index)
            if min_value is not None:
                bad = bad | (values.notna() & values.lt(float(min_value)))
            if max_value is not None:
                bad = bad | (values.notna() & values.gt(float(max_value)))
            for _, row in effective.loc[bad].iterrows():
                rows.append(
                    {
                        "country": row.get("country", ""),
                        "scenario": row.get("scenario", ""),
                        "delivery_year": row.get("delivery_year", ""),
                        "column": column,
                        "value": row.get(column, ""),
                        "min": "" if min_value is None else min_value,
                        "max": "" if max_value is None else max_value,
                    }
                )
        if rows:
            issues.append(_issue("physical_bounds_violations", "physical plausibility bounds failed", rows))

    source_rows: list[dict[str, object]] = []
    for source in manifest.get("source_components") or []:
        path_value = source.get("local_path")
        if path_value and not _resolve_local_path(path_value).exists():
            source_rows.append({"source_id": source.get("source_id", ""), "local_path": path_value, "issue": "missing_local_path"})
        source_date = _to_utc(source.get("publication_date"))
        if source_date is None:
            source_rows.append({"source_id": source.get("source_id", ""), "local_path": path_value or "", "issue": "missing_publication_date"})
        elif source_date > vintage_ts:
            source_rows.append({"source_id": source.get("source_id", ""), "local_path": path_value or "", "issue": "publication_after_vintage"})
    if source_rows:
        issues.append(_issue("source_component_issues", "source component metadata is not production-governed", source_rows))

    return issues, effective


def _md_table(rows: list[dict[str, object]] | pd.DataFrame) -> str:
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty:
        return "_none_"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[c]).replace("|", "\\|").replace("\n", " ") for c in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    manifest_path: Path,
    inventory_path: Path,
    vintage: str,
    mode: str,
    issues: list[dict[str, object]],
    effective: pd.DataFrame,
) -> None:
    lines = [
        "# Scenario Governance Validation",
        "",
        f"* manifest: `{manifest_path}`",
        f"* inventory: `{inventory_path}`",
        f"* vintage: `{vintage}`",
        f"* mode: `{mode}`",
        f"* governance gate: `{'OK' if not issues else 'FAILED'}`",
        f"* effective rows: `{len(effective)}`",
        "",
        "## Issues",
        "",
    ]
    if issues:
        for issue in issues:
            lines.extend(
                [
                    f"### {issue['issue']}",
                    "",
                    str(issue["detail"]),
                    "",
                    _md_table(issue["rows"]),
                    "",
                ]
            )
    else:
        lines.extend(["_none_", ""])
    lines.extend(
        [
            "## Effective Inventory Rows",
            "",
            _md_table(
                effective[["country", "scenario", "delivery_year", "publication_date", "quality_flag"]]
                if len(effective)
                else effective
            ),
            "",
        ]
    )
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
    parser.add_argument("--mode", choices=["production", "local-test"], default="production")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args(argv)

    inventory_path = Path(args.inventory)
    manifest_path = Path(args.manifest)
    inventory = load_electrification_scenarios(inventory_path, require_recommended=True)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    issues, effective = validate_governance(
        inventory=inventory,
        manifest=manifest,
        vintage=args.vintage,
        countries=_parse_csv(args.countries),
        scenarios=_parse_csv(args.scenarios),
        years=_parse_years(args.years),
        mode=args.mode,
    )
    write_report(
        Path(args.report),
        manifest_path=manifest_path,
        inventory_path=inventory_path,
        vintage=args.vintage,
        mode=args.mode,
        issues=issues,
        effective=effective,
    )
    print(f"[governance] gate={'OK' if not issues else 'FAILED'}")
    print(f"[governance] report -> {args.report}")
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())

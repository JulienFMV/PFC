"""Validate the LT electrification scenario inventory.

Examples
--------
Local parquet/csv:
    python scripts/validate_electrification_scenarios.py --path data/electrification_scenarios.parquet

Databricks table from config.yaml:
    python scripts/validate_electrification_scenarios.py --databricks --vintage 2026-01-15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pfc_shaping.data.electrification_acquisition import (
    DEFAULT_ELECTRIFICATION_SCENARIO_PATH,
    load_electrification_scenarios_from_databricks,
)
from pfc_shaping.data.electrification_scenarios import (
    CONDITIONAL_PRODUCTION_FIELD_GROUPS,
    COUNTRY_SCOPED_CRITICAL_VALUE_COLUMNS,
    CRITICAL_PRODUCTION_VALUE_COLUMNS,
    PRODUCTION_COLUMNS,
    RECOMMENDED_COLUMNS,
    UNACCEPTABLE_PRODUCTION_QUALITY_TOKENS,
    asof_electrification_scenarios,
    assert_scenario_coverage,
    load_electrification_scenarios,
)


def _parse_years(value: str) -> list[int]:
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_\n"

    def fmt(value: object) -> str:
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item) for item in value)
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value)

    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for column in df.columns:
            text = fmt(row[column])
            values.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _summary(frame: pd.DataFrame, asof: pd.DataFrame, *, country: str) -> dict[str, object]:
    out: dict[str, object] = {
        "rows_total": int(len(frame)),
        "rows_asof": int(len(asof)),
        "countries": sorted(frame["country"].astype(str).unique().tolist()) if len(frame) else [],
        "scenarios": sorted(frame["scenario"].astype(str).unique().tolist()) if len(frame) else [],
    }
    if len(asof):
        out["publication_min"] = str(asof["publication_date"].min())
        out["publication_max"] = str(asof["publication_date"].max())
        sub = asof[asof["country"].astype(str) == str(country)]
        out["delivery_years_country"] = sorted(sub["delivery_year"].astype(int).unique().tolist())
    return out


def _production_issues(
    frame: pd.DataFrame,
    asof: pd.DataFrame,
    *,
    country: str,
    scenarios: list[str],
    years: list[int],
    required_countries: list[str],
    require_production: bool,
    forbid_proxy: bool,
) -> dict[str, pd.DataFrame]:
    issues: dict[str, pd.DataFrame] = {}
    if len(asof) and {"country", "scenario", "delivery_year"}.issubset(asof.columns):
        scoped = asof[
            asof["scenario"].astype(str).isin(scenarios)
            & asof["delivery_year"].astype(int).isin(years)
        ].copy()
        if required_countries:
            scoped = scoped[scoped["country"].astype(str).isin(required_countries)]
        else:
            scoped = scoped[scoped["country"].astype(str) == str(country)]
        effective = (
            scoped.sort_values("publication_date")
            .drop_duplicates(["country", "scenario", "delivery_year"], keep="last")
            .reset_index(drop=True)
        )
    else:
        effective = asof.iloc[0:0].copy()

    if require_production:
        missing = sorted(PRODUCTION_COLUMNS - set(frame.columns))
        if missing:
            issues["missing_production_columns"] = pd.DataFrame({"column": missing})
        critical_value_columns = sorted(CRITICAL_PRODUCTION_VALUE_COLUMNS & set(effective.columns))
        if critical_value_columns and len(effective):
            missing_values = [
                {"column": column, "missing_rows": int(mask.sum())}
                for column in critical_value_columns
                for mask in [
                    effective[column].isna()
                    & (
                        effective["country"].astype(str).isin(COUNTRY_SCOPED_CRITICAL_VALUE_COLUMNS[column])
                        if column in COUNTRY_SCOPED_CRITICAL_VALUE_COLUMNS
                        else True
                    )
                ]
                if int(mask.sum()) > 0
            ]
            if missing_values:
                issues["missing_critical_production_values"] = pd.DataFrame(missing_values)
        missing_groups: list[dict[str, object]] = []
        for group_name, alternatives in CONDITIONAL_PRODUCTION_FIELD_GROUPS.items():
            for _, row in effective.iterrows():
                ok = any(
                    all(field in effective.columns and pd.notna(row.get(field)) for field in option)
                    for option in alternatives
                )
                if ok:
                    continue
                missing_groups.append(
                    {
                        "group": group_name,
                        "country": row.get("country", ""),
                        "scenario": row.get("scenario", ""),
                        "delivery_year": row.get("delivery_year", ""),
                    }
                )
        if missing_groups:
            issues["missing_conditional_production_values"] = pd.DataFrame(missing_groups)

    if required_countries:
        present = set(asof["country"].astype(str).unique().tolist()) if len(asof) else set()
        missing_countries = [c for c in required_countries if c not in present]
        if missing_countries:
            issues["missing_required_countries"] = pd.DataFrame({"country": missing_countries})
        if scenarios and years and {"country", "scenario", "delivery_year"}.issubset(asof.columns):
            country_s = asof["country"].astype(str)
            scenario_s = asof["scenario"].astype(str)
            delivery_year_s = asof["delivery_year"].astype(int)
            missing_coverage: list[dict[str, object]] = []
            for required_country in required_countries:
                for scenario in scenarios:
                    for year in years:
                        has_row = bool(
                            (
                                (country_s == str(required_country))
                                & (scenario_s == str(scenario))
                                & (delivery_year_s == int(year))
                            ).any()
                        )
                        if not has_row:
                            missing_coverage.append(
                                {
                                    "country": str(required_country),
                                    "scenario": str(scenario),
                                    "delivery_year": int(year),
                                }
                            )
            if missing_coverage:
                issues["missing_required_country_scenario_years"] = pd.DataFrame(missing_coverage)

    if forbid_proxy and "quality_flag" in effective.columns and len(effective):
        flags = effective["quality_flag"].fillna("").astype(str).str.lower()
        bad = flags.map(
            lambda value: any(token in value for token in UNACCEPTABLE_PRODUCTION_QUALITY_TOKENS)
        )
        if bad.any():
            issues["unacceptable_quality_flags"] = (
                effective.loc[bad, ["country", "scenario", "delivery_year", "quality_flag"]]
                .value_counts(dropna=False)
                .reset_index(name="rows")
                .sort_values(["country", "scenario", "delivery_year", "quality_flag"])
            )

    if "scenario_weight" in asof.columns and scenarios:
        bad_weights: list[dict[str, object]] = []
        sub = asof[
            (asof["country"].astype(str) == str(country))
            & (asof["scenario"].astype(str).isin(scenarios))
            & (asof["delivery_year"].astype(int).isin(years))
        ].copy()
        for year, g in sub.groupby("delivery_year"):
            latest = g.sort_values("publication_date").drop_duplicates("scenario", keep="last")
            if int(latest["scenario"].nunique()) != len(scenarios):
                continue
            weight_sum = float(pd.to_numeric(latest["scenario_weight"], errors="coerce").sum())
            if abs(weight_sum - 1.0) > 1e-9:
                bad_weights.append(
                    {
                        "country": country,
                        "delivery_year": int(year),
                        "scenario_count": int(latest["scenario"].nunique()),
                        "weight_sum": weight_sum,
                    }
                )
        if bad_weights:
            issues["scenario_weight_sum_not_one"] = pd.DataFrame(bad_weights)
    return issues


def _write_report(
    path: Path,
    *,
    vintage: str,
    country: str,
    scenarios: list[str],
    years: list[int],
    summary: dict[str, object],
    missing_recommended: list[str],
    production_issues: dict[str, pd.DataFrame],
    coverage_ok: bool,
) -> None:
    production_ok = not production_issues
    if len(years) == 1:
        years_label = str(years[0])
    else:
        contiguous = years == list(range(years[0], years[-1] + 1))
        years_label = f"{years[0]}-{years[-1]}" if contiguous else ", ".join(str(year) for year in years)
    lines = [
        "# Electrification Scenario Inventory Validation",
        "",
        f"* vintage: `{vintage}`",
        f"* country: `{country}`",
        f"* scenarios: `{', '.join(scenarios)}`",
        f"* delivery years: `{years_label}`",
        f"* coverage: `{'OK' if coverage_ok else 'FAILED'}`",
        f"* production gate: `{'OK' if production_ok else 'FAILED'}`",
        "",
        "## Summary",
        "",
        _df_to_md(pd.DataFrame([summary])),
        "",
        "## Missing Recommended Columns",
        "",
        _df_to_md(pd.DataFrame({"column": missing_recommended})) if missing_recommended else "_none_",
        "",
        "## Production Gate Issues",
        "",
    ]
    if production_issues:
        for name, df in production_issues.items():
            lines.extend([f"### {name}", "", _df_to_md(df), ""])
    else:
        lines.extend(["_none_", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(DEFAULT_ELECTRIFICATION_SCENARIO_PATH))
    parser.add_argument("--databricks", action="store_true", help="Load from Databricks config table.")
    parser.add_argument("--table-key", default="electrification_scenarios")
    parser.add_argument("--where-sql", default=None)
    parser.add_argument("--vintage", required=True, help="Pricing vintage, e.g. 2026-01-15.")
    parser.add_argument("--country", default="CH")
    parser.add_argument("--scenarios", default="slow,central,fast")
    parser.add_argument("--years", default="2027-2030", help="Range `2027-2030` or list `2027,2028`.")
    parser.add_argument("--require-recommended", action="store_true")
    parser.add_argument("--require-production", action="store_true")
    parser.add_argument(
        "--required-countries",
        default=None,
        help="Comma-separated countries that must have as-of rows, e.g. CH,DE,FR,IT,AT.",
    )
    parser.add_argument(
        "--forbid-proxy",
        action="store_true",
        help="Reject quality flags containing proxy, internal, fallback or synthetic.",
    )
    parser.add_argument("--report", default=".planning/phases/13-lt-electrification-scenario-shape/scenario_inventory_validation.md")
    args = parser.parse_args(argv)

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    years = _parse_years(args.years)
    required_countries = _parse_csv(args.required_countries)
    periods = [pd.Period(f"{year}-{month:02d}", "M") for year in years for month in range(1, 13)]

    if args.databricks:
        frame = load_electrification_scenarios_from_databricks(
            table_key=args.table_key,
            where_sql=args.where_sql,
            require_recommended=args.require_recommended,
        )
    else:
        frame = load_electrification_scenarios(
            args.path,
            require_recommended=args.require_recommended,
        )
    asof = asof_electrification_scenarios(frame, args.vintage)
    missing_recommended = sorted(RECOMMENDED_COLUMNS - set(frame.columns))

    coverage_ok = True
    try:
        assert_scenario_coverage(
            asof,
            country=args.country,
            scenarios=scenarios,
            periods=periods,
        )
    except KeyError as exc:
        coverage_ok = False
        print(f"[scenario] coverage FAILED: {exc}")

    summary = _summary(frame, asof, country=args.country)
    production_issues = _production_issues(
        frame,
        asof,
        country=args.country,
        scenarios=scenarios,
        years=years,
        required_countries=required_countries,
        require_production=args.require_production,
        forbid_proxy=bool(args.forbid_proxy or args.require_production),
    )
    _write_report(
        Path(args.report),
        vintage=args.vintage,
        country=args.country,
        scenarios=scenarios,
        years=years,
        summary=summary,
        missing_recommended=missing_recommended,
        production_issues=production_issues,
        coverage_ok=coverage_ok,
    )

    print(f"[scenario] rows total={summary['rows_total']} asof={summary['rows_asof']}")
    print(f"[scenario] coverage={'OK' if coverage_ok else 'FAILED'}")
    print(f"[scenario] production_gate={'OK' if not production_issues else 'FAILED'}")
    print(f"[scenario] report -> {args.report}")
    return 0 if coverage_ok and not production_issues else 2


if __name__ == "__main__":
    raise SystemExit(main())

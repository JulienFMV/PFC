"""Audit the full BFE/SFOE opendata.swiss catalogue for PFC LT relevance."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_OUTPUT = Path("data/bfe_opendata_catalog_audit.csv")
DEFAULT_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/BFE-OPENDATA-CATALOG-AUDIT.md")

CKAN_SEARCH_URL = "https://ckan.opendata.swiss/api/3/action/package_search"
ORG = "bundesamt-fur-energie-bfe"

FAMILY_RULES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("hydro_reservoir", ("speicherseen", "reservoir content", "wasta", "wasserkraftanlagen", "hydropower plants", "dams under federal"), "P0"),
    ("installed_capacity", ("elektrizitatsproduktionsanlagen", "electricity production plants", "nuclear power plants", "wind energy plants", "photovoltaik-grossanlagen"), "P0"),
    ("electricity_balance", ("elektrizitaetsbilanz", "elektrizitätsbilanz", "electricity balance", "aussenhandel", "physical import", "landesverbrauch", "endverbrauch", "stromproduktion", "stromverbrauch", "nettoimport"), "P0"),
    ("pv", ("photovoltaik", "photovoltaic", "winterstrom", "sonnendach", "solar energy potential"), "P0"),
    ("ev_mobility", ("elektromobilitaet", "elektromobilität", "electric vehicle", "plug-in vehicles", "ladestationen", "charging requirement"), "P1"),
    ("demand_buildings", ("gebaeude", "gebäude", "waerme", "wärme", "heat pump", "heating", "minergie"), "P1"),
    ("prices_market", ("strompreis", "marktpreis", "day-ahead", "referenz-marktpreis"), "P1"),
    ("energiedashboard", ("energiedashboard", "energy-dashboard"), "P0"),
    ("grid_infrastructure", ("transmission line", "uebertragungsleitungen", "übertragungsleitungen", "high-voltage", "nominal voltage"), "P2"),
)


def _as_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values() if v)
    if isinstance(value, list):
        return " ".join(_as_text(v) for v in value)
    return "" if value is None else str(value)


def fetch_bfe_packages(rows: int = 200) -> list[dict[str, Any]]:
    query = f"organization:{ORG}"
    params = urllib.parse.urlencode({"q": query, "rows": rows})
    request = urllib.request.Request(
        f"{CKAN_SEARCH_URL}?{params}",
        headers={"User-Agent": "PFC-LT BFE opendata catalogue audit"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    return list(payload["result"]["results"])


def classify_package(package: dict[str, Any]) -> tuple[str, str]:
    haystack = " ".join(
        [
            _as_text(package.get("name")),
            _as_text(package.get("title")),
            _as_text(package.get("description")),
            _as_text(package.get("keywords")),
            _as_text([tag.get("name") for tag in package.get("tags", [])]),
        ]
    ).lower()
    matched: list[str] = []
    priority = "P3"
    for family, needles, rule_priority in FAMILY_RULES:
        if any(needle.lower() in haystack for needle in needles):
            matched.append(family)
            priority = min(priority, rule_priority)
    return ",".join(dict.fromkeys(matched)) if matched else "other", priority


def build_catalog_audit(packages: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for package in packages:
        family, priority = classify_package(package)
        resources = package.get("resources", []) or []
        formats = sorted({str(resource.get("format") or "").upper() for resource in resources if resource.get("format")})
        urls = [
            str(resource.get("download_url") or resource.get("url") or "")
            for resource in resources
            if resource.get("download_url") or resource.get("url")
        ]
        title = package.get("title")
        rows.append(
            {
                "priority": priority,
                "family": family,
                "dataset_name": package.get("name", ""),
                "title_en": title.get("en", "") if isinstance(title, dict) else str(title or ""),
                "issued": package.get("issued", ""),
                "metadata_modified": package.get("metadata_modified", ""),
                "formats": ",".join(formats),
                "resource_count": len(resources),
                "first_resource_url": urls[0] if urls else "",
                "catalog_url": f"https://opendata.swiss/en/dataset/{package.get('name', '')}",
            }
        )
    return pd.DataFrame(rows).sort_values(["priority", "family", "dataset_name"])


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


def write_report(path: Path, *, audit: pd.DataFrame, output: Path) -> None:
    summary = (
        audit.groupby(["priority", "family"], dropna=False)
        .size()
        .reset_index(name="datasets")
        .sort_values(["priority", "family"])
    )
    top = audit[audit["priority"].isin(["P0", "P1"])].head(40)
    lines = [
        "# BFE opendata.swiss Catalogue Audit",
        "",
        f"* organization: `{ORG}`",
        f"* datasets scanned: `{len(audit)}`",
        f"* csv: `{output}`",
        "",
        "## Summary",
        "",
        _md_table(summary),
        "",
        "## P0/P1 Candidates",
        "",
        _md_table(
            top[
                [
                    "priority",
                    "family",
                    "dataset_name",
                    "title_en",
                    "formats",
                    "first_resource_url",
                ]
            ]
        ),
        "",
        "## Production Use",
        "",
        "This is a discovery register. A dataset becomes model-usable only after a dedicated importer, local raw cache, schema validation, vintage metadata and production gate documentation.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--rows", type=int, default=200)
    args = parser.parse_args(argv)

    audit = build_catalog_audit(fetch_bfe_packages(rows=args.rows))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False)
    write_report(Path(args.report), audit=audit, output=output)
    print(f"[bfe-catalog] datasets={len(audit)}")
    print(f"[bfe-catalog] output -> {output}")
    print(f"[bfe-catalog] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

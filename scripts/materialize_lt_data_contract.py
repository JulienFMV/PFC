"""Materialize the local LT data contract used by the PFC model.

This script does not pretend that local proxy data is final production data. It
creates the local cache folders, checks the model-critical parquet files, expands
the best available Phase 13 slow/central/fast scenario inventory to the requested
delivery years, writes the canonical model paths, and emits a Markdown readiness
report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

import pandas as pd

from pfc_shaping.data.electrification_scenarios import (
    asof_electrification_scenarios,
    assert_scenario_coverage,
    derive_hpfc_scenario_features,
    interpolate_electrification_scenario_years,
    load_electrification_scenarios,
    validate_electrification_scenarios,
)


DEFAULT_LOCAL_ROOT = Path(r"C:\Users\jbattaglia\pfc_local_data")
DEFAULT_SCENARIO_SOURCE = Path("data/electrification_scenarios_ep2050_enriched_slow_central_fast.parquet")
DEFAULT_SCENARIO_OUTPUT = Path("data/electrification_scenarios.parquet")
DEFAULT_FEATURE_OUTPUT = Path("data/hpfc_scenario_features.parquet")
DEFAULT_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/LT-DATA-CONTRACT-LOCAL.md")

LOCAL_CACHE_DIRS = [
    "entsoe/raw_xml",
    "entsoe/bronze_timeseries",
    "entsoe/bronze_exchanges",
    "entsoe/bronze_forecasts",
    "scenarios/ofen_ep2050",
    "scenarios/tyndp_2024",
    "scenarios/pronovo",
    "scenarios/mastr",
    "scenarios/swissgrid_ntc",
    "market/eex",
    "market/epex",
    "market/commodities",
]

MODEL_ASSETS = [
    ("EPEX CH spot", Path("pfc_shaping/data/epex_15min.parquet"), "historical price shape calibration"),
    ("EPEX DE spot", Path("pfc_shaping/data/epex_de_15min.parquet"), "cross-border/CT diagnostics"),
    ("EPEX CH hourly", Path("data/epex_hourly.parquet"), "LT perfect-foresight and local PFC runner"),
    ("ENTSO-E wide physicals", Path("pfc_shaping/data/entso_15min.parquet"), "LT/CT physical features"),
    ("Hydro reservoir", Path("pfc_shaping/data/hydro_reservoir.parquet"), "water-value diagnostics"),
    ("Generation outages", Path("pfc_shaping/data/outages_15min.parquet"), "availability diagnostics"),
    ("EEX forwards", Path("data/eex_forwards_history.parquet"), "market anchors"),
    ("Commodities", Path("data/commodities_cache.parquet"), "thermal/basis diagnostics"),
]

EXTERNAL_GAPS = [
    ("TYNDP 2024 multi-country scenarios", "missing official DE/FR/IT/AT backbone in canonical local table"),
    ("Pronovo CH PV actuals", "missing vintage actualization of CH PV trajectory"),
    ("MaStR DE PV/battery actuals", "missing DE PV/battery commissioning actualization"),
    ("ENTSO-E forecast snapshots", "current DE renewable forecast lacks as_of_utc and is not vintage-safe"),
    ("Swissgrid/TYNDP/JAO long-term NTC", "CH-AT and governed NTC scenario feed not loaded"),
    ("Fuel/CO2/electrolysis/DSM scenarios", "thermal-set and belly-refill structural drivers not loaded"),
]


@dataclass(frozen=True)
class AssetStatus:
    name: str
    path: Path
    purpose: str
    exists: bool
    rows: int | None = None
    columns: int | None = None
    error: str | None = None


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
    if not years:
        raise ValueError("at least one delivery year is required")
    return sorted(years)


def _periods(years: Iterable[int]) -> list[pd.Period]:
    return [pd.Period(f"{int(year)}-{month:02d}", "M") for year in years for month in range(1, 13)]


def _ensure_local_dirs(local_root: Path) -> list[Path]:
    created: list[Path] = []
    for rel in LOCAL_CACHE_DIRS:
        path = local_root / rel
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created


def _asset_status(path: Path, name: str, purpose: str) -> AssetStatus:
    if not path.exists():
        return AssetStatus(name=name, path=path, purpose=purpose, exists=False)
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - defensive report path
        return AssetStatus(name=name, path=path, purpose=purpose, exists=True, error=str(exc))
    return AssetStatus(name=name, path=path, purpose=purpose, exists=True, rows=len(df), columns=len(df.columns))


def audit_model_assets() -> list[AssetStatus]:
    return [_asset_status(path, name, purpose) for name, path, purpose in MODEL_ASSETS]


def materialize_scenario_contract(
    *,
    source_path: Path,
    scenario_output: Path,
    features_output: Path,
    years: list[int],
    vintage: str,
    country: str,
    scenarios: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = load_electrification_scenarios(source_path, require_recommended=True)
    expanded = interpolate_electrification_scenario_years(source, years, allow_clamp=False)
    expanded = validate_electrification_scenarios(expanded, require_recommended=True)
    asof = asof_electrification_scenarios(expanded, vintage)
    assert_scenario_coverage(
        asof,
        country=country,
        scenarios=scenarios,
        periods=_periods(years),
    )
    scenario_output.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_parquet(scenario_output, index=False)

    features = derive_hpfc_scenario_features(expanded)
    features_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(features_output, index=False)
    return expanded, features


def _md_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        rendered = []
        for value in row:
            if value is None:
                rendered.append("")
            elif isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def write_report(
    path: Path,
    *,
    local_root: Path,
    created_dirs: list[Path],
    asset_status: list[AssetStatus],
    scenario_output: Path,
    features_output: Path,
    scenarios: list[str],
    years: list[int],
    vintage: str,
    scenario_rows: int,
    feature_rows: int,
) -> None:
    missing_assets = [a for a in asset_status if not a.exists or a.error]
    prod_complete = False
    lines = [
        "# LT Data Contract Local Materialization",
        "",
        f"* local root: `{local_root}`",
        f"* vintage: `{vintage}`",
        f"* scenarios: `{', '.join(scenarios)}`",
        f"* delivery years: `{years[0]}-{years[-1]}`",
        f"* canonical scenario output: `{scenario_output}`",
        f"* canonical feature output: `{features_output}`",
        f"* production complete: `{'YES' if prod_complete else 'NO - local proxy only'}`",
        "",
        "## Local Cache Directories",
        "",
        *_md_table(["path"], [[p] for p in created_dirs]),
        "",
        "## Model Asset Audit",
        "",
        *_md_table(
            ["asset", "path", "exists", "rows", "columns", "purpose", "error"],
            [
                [a.name, a.path, "yes" if a.exists else "no", a.rows, a.columns, a.purpose, a.error or ""]
                for a in asset_status
            ],
        ),
        "",
        "## Canonical Scenario Contract",
        "",
        *_md_table(
            ["file", "rows", "status"],
            [
                [scenario_output, scenario_rows, "written and vintage-gated"],
                [features_output, feature_rows, "written from canonical scenario file"],
            ],
        ),
        "",
        "## External Data Still Required For 10/10 Production",
        "",
        *_md_table(["source", "gap"], [[name, gap] for name, gap in EXTERNAL_GAPS]),
        "",
        "## Decision",
        "",
        "The local project is now wired to the canonical Phase 13 model paths using the best available governed local EP2050 proxy inventory. "
        "This is sufficient for local smoke/prod-readiness runs with `require_electrification_scenarios=True`, but it is not sufficient to enable Phase 13 as final production signal until the external gaps above are loaded and validated.",
        "",
    ]
    if missing_assets:
        lines.extend(
            [
                "## Missing Local Model Assets",
                "",
                *_md_table(
                    ["asset", "path", "issue"],
                    [[a.name, a.path, a.error or "missing"] for a in missing_assets],
                ),
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", default=str(DEFAULT_LOCAL_ROOT))
    parser.add_argument("--scenario-source", default=str(DEFAULT_SCENARIO_SOURCE))
    parser.add_argument("--scenario-output", default=str(DEFAULT_SCENARIO_OUTPUT))
    parser.add_argument("--features-output", default=str(DEFAULT_FEATURE_OUTPUT))
    parser.add_argument("--years", default="2027-2031")
    parser.add_argument("--vintage", default="2026-06-05")
    parser.add_argument("--country", default="CH")
    parser.add_argument("--scenarios", default="slow,central,fast")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args(argv)

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    years = _parse_years(args.years)
    local_root = Path(args.local_root)
    created_dirs = _ensure_local_dirs(local_root)
    asset_status = audit_model_assets()
    scenario_frame, features = materialize_scenario_contract(
        source_path=Path(args.scenario_source),
        scenario_output=Path(args.scenario_output),
        features_output=Path(args.features_output),
        years=years,
        vintage=args.vintage,
        country=args.country,
        scenarios=scenarios,
    )
    write_report(
        Path(args.report),
        local_root=local_root,
        created_dirs=created_dirs,
        asset_status=asset_status,
        scenario_output=Path(args.scenario_output),
        features_output=Path(args.features_output),
        scenarios=scenarios,
        years=years,
        vintage=args.vintage,
        scenario_rows=len(scenario_frame),
        feature_rows=len(features),
    )
    print(f"[lt-data] local root -> {local_root}")
    print(f"[lt-data] scenario -> {args.scenario_output} rows={len(scenario_frame)}")
    print(f"[lt-data] features -> {args.features_output} rows={len(features)}")
    print(f"[lt-data] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Apply Swissgrid cross-border NTC baseline values to LT scenarios.

The Swissgrid annual cross-border CSV is an observed/current-year capacity
source, not a governed 2030 expansion plan. This script therefore fills NTC
columns as a traceable baseline proxy and keeps the scenario inventory
non-production until governance approves a long-term NTC assumption.
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

DEFAULT_INPUT = Path("data/electrification_scenarios_composed_p0_bridge_2030.parquet")
DEFAULT_OUTPUT = Path("data/electrification_scenarios_composed_p0_real_sources_2030.parquet")
DEFAULT_FEATURE_OUTPUT = Path("data/hpfc_scenario_features_composed_p0_real_sources_2030.parquet")
DEFAULT_COMPONENT_OUTPUT = Path("data/electrification_scenarios_swissgrid_ntc_baseline_2026.parquet")
DEFAULT_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/SWISSGRID-NTC-BASELINE.md")
DEFAULT_SWISSGRID_CSV = (
    Path(r"C:\Users\jbattaglia\pfc_local_data\scenarios\swissgrid_ntc")
    / "Grenzfluesse-2026.csv"
)

COMPONENT_ID = "swissgrid_ntc_2026_baseline"
PUBLICATION_DATE = pd.Timestamp("2026-06-08", tz="UTC")
BORDER_COLUMNS = {
    "AT": {
        "field": "ntc_ch_at_gw",
        "export_token": "Schweiz > Österreich",
        "import_token": "Österreich > Schweiz",
    },
    "DE": {
        "field": "ntc_ch_de_gw",
        "export_token": "Schweiz > Deutschland",
        "import_token": "Deutschland > Schweiz",
    },
    "FR": {
        "field": "ntc_ch_fr_gw",
        "export_token": "Schweiz > Frankreich",
        "import_token": "Frankreich > Schweiz",
    },
    "IT": {
        "field": "ntc_ch_it_gw",
        "export_token": "Schweiz > Italien",
        "import_token": "Italien > Schweiz",
    },
}


def _find_column(columns: list[str], token: str) -> str:
    matches = [column for column in columns if token in column]
    if len(matches) != 1:
        raise KeyError(f"expected exactly one Swissgrid column containing {token!r}, got {matches}")
    return matches[0]


def _capacity_stats(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").abs()
    values = values[np.isfinite(values) & values.gt(0.0)]
    if values.empty:
        raise ValueError("Swissgrid NTC series has no positive numeric capacity values")
    return {
        "p05_mw": float(values.quantile(0.05)),
        "median_mw": float(values.median()),
        "p95_mw": float(values.quantile(0.95)),
        "observations": int(values.count()),
    }


def load_swissgrid_ntc_baseline(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    columns = list(raw.columns)
    rows: list[dict[str, object]] = []
    for border, spec in BORDER_COLUMNS.items():
        export_col = _find_column(columns, spec["export_token"])
        import_col = _find_column(columns, spec["import_token"])
        export_stats = _capacity_stats(raw[export_col])
        import_stats = _capacity_stats(raw[import_col])
        symmetric_median_mw = min(export_stats["median_mw"], import_stats["median_mw"])
        rows.append(
            {
                "publication_date": PUBLICATION_DATE,
                "source": "Swissgrid cross-border load flows 2026",
                "component_id": COMPONENT_ID,
                "border": f"CH-{border}",
                "ntc_field": spec["field"],
                "export_column": export_col,
                "import_column": import_col,
                "export_median_gw": export_stats["median_mw"] / 1000.0,
                "import_median_gw": import_stats["median_mw"] / 1000.0,
                "symmetric_baseline_gw": symmetric_median_mw / 1000.0,
                "export_p05_gw": export_stats["p05_mw"] / 1000.0,
                "import_p05_gw": import_stats["p05_mw"] / 1000.0,
                "export_p95_gw": export_stats["p95_mw"] / 1000.0,
                "import_p95_gw": import_stats["p95_mw"] / 1000.0,
                "observations": min(export_stats["observations"], import_stats["observations"]),
                "method": "min(import/export median absolute NTC MW)",
                "quality_flag": "official_observed_baseline_proxy",
            }
        )
    return pd.DataFrame(rows)


def _append_component_ids(value: object) -> str:
    components = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if COMPONENT_ID not in components:
        components.append(COMPONENT_ID)
    return ",".join(components)


def apply_ntc_baseline(
    frame: pd.DataFrame,
    baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = validate_electrification_scenarios(frame, require_recommended=True).copy()
    changes: list[dict[str, object]] = []
    for row in baseline.itertuples(index=False):
        field = str(row.ntc_field)
        value = float(row.symmetric_baseline_gw)
        if field not in out.columns:
            raise KeyError(f"scenario inventory lacks {field}")
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"invalid Swissgrid NTC baseline for {field}: {value}")
        missing = out[field].isna()
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
                    "border": row.border,
                    "method": row.method,
                }
            )
    if changes:
        out["quality_flag"] = out["quality_flag"].astype(str) + "_swissgrid_ntc_baseline_proxy"
        out["source"] = out["source"].astype(str) + " + Swissgrid NTC baseline"
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
        "# Swissgrid NTC Baseline",
        "",
        f"* raw source: `{csv_path}`",
        f"* scenario output: `{output}`",
        f"* component output: `{component_output}`",
        "* status: `NON-PRODUCTION / OFFICIAL OBSERVED BASELINE / PROXY`",
        "* publication date used for vintage checks: `2026-06-08`",
        "* source method: Swissgrid 2026 annual cross-border NTC columns.",
        "* model method: generic `ntc_ch_*_gw` uses `min(median export, median import)` in GW.",
        "",
        "## Baseline Values",
        "",
        _md_table(
            baseline[
                [
                    "border",
                    "ntc_field",
                    "export_median_gw",
                    "import_median_gw",
                    "symmetric_baseline_gw",
                    "observations",
                ]
            ]
        ),
        "",
        "## Changed Scenario Cells",
        "",
        _md_table(changes),
        "",
        "## Production Interpretation",
        "",
        "This closes the numeric NTC P0 gap with a true Swissgrid source, but it is not a 2030 governed expansion assumption. "
        "Strict production governance must therefore remain failed until the risk committee approves long-term NTC values or a TYNDP/TSO LT capacity path.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--swissgrid-csv", default=str(DEFAULT_SWISSGRID_CSV))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--features-output", default=str(DEFAULT_FEATURE_OUTPUT))
    parser.add_argument("--component-output", default=str(DEFAULT_COMPONENT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args(argv)

    baseline = load_swissgrid_ntc_baseline(Path(args.swissgrid_csv))
    frame = load_electrification_scenarios(Path(args.input), require_recommended=True)
    out, changes = apply_ntc_baseline(frame, baseline)

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
        csv_path=Path(args.swissgrid_csv),
        output=output,
        component_output=component_output,
        baseline=baseline,
        changes=changes,
    )
    print(f"[swissgrid-ntc] baseline_rows={len(baseline)} changed_cells={len(changes)} -> {output}")
    print(f"[swissgrid-ntc] component -> {component_output}")
    print(f"[swissgrid-ntc] features -> {features_output}")
    print(f"[swissgrid-ntc] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

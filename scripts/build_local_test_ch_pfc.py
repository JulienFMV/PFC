"""Build a local-test CH LT PFC from the governed Phase 13 candidate inventory.

This runner is intentionally non-production. It requires the local-test
governance gate to pass, then builds one CH PFC per scenario and a weighted
structural fan chart. Production approval remains a separate human-signoff gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pfc_shaping.data.electrification_scenarios import (
    derive_hpfc_scenario_features,
    interpolate_electrification_scenario_years,
    load_electrification_scenarios,
)
from pfc_shaping.lt.model.electrification_shape import structural_fan_chart
from scripts.build_ep2050_multi_scenario_pfc import (
    _build_one_curve,
    _delivery_years,
    _df_to_md,
    _fit_components,
    _parse_csv,
    _parse_weights,
    _safe_slug,
)
from scripts.build_first_ep2050_pfc import _latest_forwards, _load_epex_hourly
from scripts.validate_scenario_governance import validate_governance, write_report

DEFAULT_INVENTORY = Path("data/electrification_scenarios_prod_candidate_neutralized_2030.parquet")
DEFAULT_MANIFEST = Path(".planning/phases/13-lt-electrification-scenario-shape/SCENARIO-GOVERNANCE-LOCAL-TEST-MANIFEST.yaml")
DEFAULT_GOVERNANCE_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/LOCAL-TEST-GOVERNANCE-GATE.md")
DEFAULT_SUMMARY = Path(".planning/phases/13-lt-electrification-scenario-shape/LOCAL-TEST-CH-PFC.md")


def _write_parquet(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=index)


def _expand_inventory(
    inventory: pd.DataFrame,
    *,
    years: list[int],
    scenarios: list[str],
    output: Path,
) -> pd.DataFrame:
    expanded = interpolate_electrification_scenario_years(inventory, years, allow_clamp=True)
    expanded = expanded[
        (expanded["country"].astype(str) == "CH")
        & expanded["scenario"].astype(str).isin(scenarios)
    ].copy()
    if expanded.empty:
        raise ValueError("expanded CH scenario inventory is empty")
    _write_parquet(expanded, output)
    return expanded


def _price_summary(frame: pd.DataFrame, *, market: str) -> dict[str, float]:
    tz = "Europe/Zurich" if market == "CH" else "Europe/Berlin"
    local = frame.index.tz_convert(tz)
    price = frame["price_shape"].astype(float)
    return {
        "mean": float(price.mean()),
        "min": float(price.min()),
        "p05": float(price.quantile(0.05)),
        "p95": float(price.quantile(0.95)),
        "max": float(price.max()),
        "midday_mean": float(price[(local.hour >= 10) & (local.hour <= 15)].mean()),
        "evening_mean": float(price[(local.hour >= 18) & (local.hour <= 21)].mean()),
        "night_mean": float(price[(local.hour <= 5) | (local.hour >= 22)].mean()),
    }


def _build_fan(pfc_by_scenario: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    curves = {scenario: frame["price_shape"].astype(float) for scenario, frame in pfc_by_scenario.items()}
    fan = structural_fan_chart(curves, weights=weights)
    out = pd.DataFrame(index=fan.index)
    for scenario, frame in pfc_by_scenario.items():
        out[f"curve_{scenario}"] = frame["price_shape"].astype(float)
    out["weighted_mean"] = fan["weighted_mean"]
    out["structural_p10"] = fan["q10"]
    out["structural_p50"] = fan["q50"]
    out["structural_p90"] = fan["q90"]
    out["structural_width"] = fan["structural_width"]
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError("fan chart contains non-finite values")
    if (out["structural_p10"] > out["structural_p90"]).any():
        raise ValueError("fan chart violates p10 <= p90")
    return out


def _write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    latest_forward_date: pd.Timestamp,
    scenarios: list[str],
    weights: dict[str, float],
    expanded: pd.DataFrame,
    pfc_paths: dict[str, Path],
    fan: pd.DataFrame,
    fan_path: Path,
    governance_report: Path,
) -> None:
    metrics = pd.DataFrame(
        [
            {"scenario": scenario, **_price_summary(pd.read_parquet(pfc_paths[scenario]), market=args.market)}
            for scenario in scenarios
        ]
    )
    fan_metrics = pd.DataFrame(
        [
            {
                "rows": len(fan),
                "weighted_mean": float(fan["weighted_mean"].mean()),
                "structural_width_mean": float(fan["structural_width"].mean()),
                "structural_width_p95": float(fan["structural_width"].quantile(0.95)),
                "structural_width_max": float(fan["structural_width"].max()),
            }
        ]
    )
    display_cols = [
        "scenario",
        "delivery_year",
        "scenario_weight",
        "quality_flag",
        "demand_twh",
        "peak_load_gw",
        "pv_gw",
        "pv_twh",
        "wind_gw",
        "wind_twh",
        "battery_power_gw",
        "battery_energy_gwh",
        "ev_twh",
        "heatpump_twh",
        "hydro_reservoir_twh",
        "net_import_twh",
        "ntc_ch_de_gw",
        "ntc_ch_fr_gw",
        "ntc_ch_it_gw",
        "ntc_ch_at_gw",
    ]
    display_cols = [col for col in display_cols if col in expanded.columns]
    lines = [
        "# Local-Test CH PFC 2030",
        "",
        "* status: `agent-approved local/test only`",
        "* production approval: `NO`",
        "* production activation allowed: `NO`",
        f"* governance report: `{governance_report}`",
        f"* source inventory: `{args.inventory}`",
        f"* expanded scenario path: `{args.expanded_output}`",
        f"* feature output: `{args.features_output}`",
        f"* fan chart: `{fan_path}`",
        f"* market: `{args.market}`",
        f"* start date UTC: `{args.start_date}`",
        f"* horizon days: `{args.horizon_days}`",
        f"* EEX forward date: `{latest_forward_date.date()}`",
        f"* scenario weights: `{', '.join(f'{k}={v:.4f}' for k, v in weights.items())}`",
        "",
        "## PFC Outputs",
        "",
        *_df_to_md(pd.DataFrame({"scenario": list(pfc_paths), "path": [str(p) for p in pfc_paths.values()]})),
        "",
        "## Price Summary",
        "",
        *_df_to_md(metrics),
        "",
        "## Structural Fan Chart",
        "",
        *_df_to_md(fan_metrics),
        "",
        "## Expanded CH Scenario Rows",
        "",
        *_df_to_md(expanded[display_cols].sort_values(["scenario", "delivery_year"])),
        "",
        "## Limitations",
        "",
        "* This curve is suitable for local validation, diagnostics and model review only.",
        "* Agent approval replaces human approval only for local/test work.",
        "* Production FMV use still requires the production governance gate with accountable human sign-off.",
        "* Proxy/partial/internal quality flags remain visible and are not relabelled as production-governed.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--governance-report", default=str(DEFAULT_GOVERNANCE_REPORT))
    parser.add_argument("--vintage", default="2026-06-12")
    parser.add_argument("--scenarios", default="slow,central,fast")
    parser.add_argument("--weights", default="0.25,0.50,0.25")
    parser.add_argument("--market", default="CH")
    parser.add_argument("--start-date", default="2030-01-01")
    parser.add_argument("--horizon-days", type=int, default=365)
    parser.add_argument("--epex-hourly", default="data/epex_hourly.parquet")
    parser.add_argument("--forwards", default="data/eex_forwards_history.parquet")
    parser.add_argument("--expanded-output", default="output/local_test_ch_pfc_2030_scenario_expanded.parquet")
    parser.add_argument("--features-output", default="data/hpfc_scenario_features_local_test_2030.parquet")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--output-prefix", default="local_test_ch_pfc_2030")
    parser.add_argument("--fan-chart-output", default="output/local_test_ch_pfc_2030_structural_fan_chart.parquet")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument(
        "--disable-cascade-trend-for-annual-only",
        action="store_true",
        help=(
            "Local/test guard: do not extrapolate fitted seasonal trends when a delivery "
            "year has only a Cal quote. Quoted months/quarters are still preserved."
        ),
    )
    args = parser.parse_args(argv)

    scenarios = _parse_csv(args.scenarios)
    weights = _parse_weights(args.weights, scenarios)
    inventory_path = Path(args.inventory)
    manifest_path = Path(args.manifest)
    inventory = load_electrification_scenarios(inventory_path, require_recommended=True)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}

    issues, effective = validate_governance(
        inventory=inventory,
        manifest=manifest,
        vintage=args.vintage,
        countries=["CH", "DE", "FR", "IT", "AT"],
        scenarios=scenarios,
        years=[2030],
        mode="local-test",
    )
    write_report(
        Path(args.governance_report),
        manifest_path=manifest_path,
        inventory_path=inventory_path,
        vintage=args.vintage,
        mode="local-test",
        issues=issues,
        effective=effective,
    )
    if issues:
        raise ValueError(f"local-test governance gate failed with {len(issues)} issues")

    years = _delivery_years(args.start_date, args.horizon_days, args.market)
    expanded = _expand_inventory(
        inventory,
        years=years,
        scenarios=scenarios,
        output=Path(args.expanded_output),
    )
    _write_parquet(derive_hpfc_scenario_features(expanded), Path(args.features_output))

    epex = _load_epex_hourly(args.epex_hourly)
    latest, base_prices = _latest_forwards(args.forwards, market=args.market)
    reference_date = pd.Timestamp(latest, tz="UTC")
    sh, si, cascader, calibrator = _fit_components(
        epex,
        args.market,
        disable_cascade_trend_for_annual_only=bool(args.disable_cascade_trend_for_annual_only),
    )

    pfc_by_scenario: dict[str, pd.DataFrame] = {}
    pfc_paths: dict[str, Path] = {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        pfc = _build_one_curve(
            scenario=scenario,
            scenario_path=Path(args.expanded_output),
            base_prices=base_prices,
            reference_date=reference_date,
            market=args.market,
            start_date=args.start_date,
            horizon_days=args.horizon_days,
            sh=sh,
            si=si,
            cascader=cascader,
            calibrator=calibrator,
        )
        path = output_dir / f"{args.output_prefix}_{_safe_slug(scenario)}.parquet"
        _write_parquet(pfc, path, index=True)
        pfc_by_scenario[scenario] = pfc
        pfc_paths[scenario] = path

    fan = _build_fan(pfc_by_scenario, weights)
    fan_path = Path(args.fan_chart_output)
    _write_parquet(fan, fan_path, index=True)
    _write_summary(
        Path(args.summary),
        args=args,
        latest_forward_date=latest,
        scenarios=scenarios,
        weights=weights,
        expanded=expanded,
        pfc_paths=pfc_paths,
        fan=fan,
        fan_path=fan_path,
        governance_report=Path(args.governance_report),
    )
    print(f"[local-test-pfc] governance -> {args.governance_report}")
    for scenario, path in pfc_paths.items():
        print(f"[local-test-pfc] {scenario} -> {path}")
    print(f"[local-test-pfc] fan chart -> {fan_path}")
    print(f"[local-test-pfc] summary -> {args.summary}")
    print(f"[local-test-pfc] weighted_mean={fan['weighted_mean'].mean():.2f} width_mean={fan['structural_width'].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build EP2050 slow/central/fast LT PFCs and a structural fan chart.

The script is a local production-readiness workflow for Phase 13. It derives a
governed three-scenario inventory from the enriched OFEN EP2050 rows, builds one
2030 PFC per scenario with the electrification layer explicitly enabled and
fail-fast scenario coverage, then writes a weighted structural fan chart.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pfc_shaping.calibration.arbitrage_free import ArbitrageFreeCalibrator
from pfc_shaping.calibration.cascading import ContractCascader
from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.data.electrification_scenarios import (
    derive_hpfc_scenario_features,
    interpolate_electrification_scenario_years,
    load_electrification_scenarios,
    validate_electrification_scenarios,
)
from pfc_shaping.lt.model.assembler import PFCAssembler
from pfc_shaping.lt.model.electrification_shape import structural_fan_chart
from pfc_shaping.lt.model.shape_hourly import ShapeHourly
from pfc_shaping.lt.model.shape_intraday import ShapeIntraday
from pfc_shaping.pipeline.monthly_curve_authority import MonthlyLevelAuthority
from scripts.build_first_ep2050_pfc import _latest_forwards, _load_epex_hourly, _validate_pfc

PROFILE_NAME = "ep2050_slow_central_fast_mapping_v0"
DEFAULT_WEIGHTS = {"slow": 0.25, "central": 0.50, "fast": 0.25}


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_weights(value: str, scenarios: Sequence[str]) -> dict[str, float]:
    raw = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(raw) != len(scenarios):
        raise ValueError(f"expected {len(scenarios)} weights, got {len(raw)}")
    if any(weight < 0.0 or not np.isfinite(weight) for weight in raw):
        raise ValueError("scenario weights must be finite and non-negative")
    total = float(sum(raw))
    if total <= 0.0:
        raise ValueError("scenario weights must sum to a positive value")
    return {scenario: weight / total for scenario, weight in zip(scenarios, raw)}


def _safe_slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _to_utc(value: str | pd.Timestamp | None, fallback: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value) if value is not None else fallback
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _append_token(value: object, token: str) -> str:
    text = "scenario" if pd.isna(value) else str(value)
    return text if token in text else f"{text}+{token}"


def _quality_with_suffix(value: object, suffix: str) -> str:
    text = "internal" if pd.isna(value) else str(value)
    return text if suffix in text else f"{text}_{suffix}"


def _latest_per_scenario(frame: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["country", "delivery_year", "scenario", "publication_date"]
    if "delivery_month" in frame.columns:
        sort_cols.insert(2, "delivery_month")
    df = frame.sort_values(sort_cols).copy()
    key_cols = ["country", "delivery_year", "scenario"]
    if "delivery_month" in df.columns:
        key_cols.insert(2, "delivery_month")
    return df.drop_duplicates(key_cols, keep="last")


def derive_slow_central_fast_inventory(
    frame: pd.DataFrame,
    *,
    slow_source: str = "WWB",
    fast_source: str = "ZERO_Basis",
    scenarios: Sequence[str] = ("slow", "central", "fast"),
    weights: Mapping[str, float] | None = None,
    central_blend: float = 0.5,
    mapping_publication_date: str | pd.Timestamp | None = None,
    profile: str = PROFILE_NAME,
) -> pd.DataFrame:
    """Map enriched OFEN rows to governed slow/central/fast scenarios.

    ``slow`` and ``fast`` are aliases of the chosen official EP2050 paths.
    ``central`` is an explicit bounded midpoint between those paths. No fields
    are filled with zero; missing source values remain missing and are visible in
    downstream validation.
    """
    if len(scenarios) != 3:
        raise ValueError("exactly three scenario labels are required")
    slow_label, central_label, fast_label = [str(s) for s in scenarios]
    if not (0.0 <= float(central_blend) <= 1.0):
        raise ValueError("central_blend must be in [0, 1]")
    weights = dict(weights or DEFAULT_WEIGHTS)

    df = _latest_per_scenario(validate_electrification_scenarios(frame, require_recommended=True))
    group_cols = ["country", "delivery_year"]
    if "delivery_month" in df.columns:
        group_cols.append("delivery_month")
    numeric_cols = [
        col
        for col in df.columns
        if col not in {"publication_date", "scenario", "source", "quality_flag", "assumption_profile"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    numeric_cols = [col for col in numeric_cols if col not in {"scenario_weight"}]

    rows: list[dict[str, object]] = []
    for group_key, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        slow_rows = sub[sub["scenario"].astype(str) == str(slow_source)]
        fast_rows = sub[sub["scenario"].astype(str) == str(fast_source)]
        if slow_rows.empty or fast_rows.empty:
            raise KeyError(
                f"missing source scenarios {slow_source!r}/{fast_source!r} for "
                f"{dict(zip(group_cols, group_key))}"
            )
        slow = slow_rows.sort_values("publication_date").iloc[-1].copy()
        fast = fast_rows.sort_values("publication_date").iloc[-1].copy()
        pub = _to_utc(mapping_publication_date, max(slow["publication_date"], fast["publication_date"]))

        for label, source_row, original in [
            (slow_label, slow, slow_source),
            (fast_label, fast, fast_source),
        ]:
            row = source_row.copy()
            row["scenario"] = label
            row["scenario_weight"] = float(weights.get(label, 0.0))
            row["original_scenario"] = original
            row["source"] = _append_token(row.get("source", "scenario"), profile)
            row["quality_flag"] = _quality_with_suffix(row.get("quality_flag", "official"), "scenario_mapped")
            row["assumption_profile"] = profile
            row["publication_date"] = max(_to_utc(row["publication_date"], pub), pub)
            rows.append(row.to_dict())

        central = slow.copy()
        central["scenario"] = central_label
        central["scenario_weight"] = float(weights.get(central_label, 0.0))
        central["original_scenario"] = f"midpoint({slow_source},{fast_source})"
        central["source"] = f"{profile}:{slow_source}:{fast_source}"
        central["quality_flag"] = "internal_midpoint_proxy_enriched"
        central["assumption_profile"] = profile
        central["publication_date"] = pub
        for col in numeric_cols:
            a = pd.to_numeric(pd.Series([slow.get(col)]), errors="coerce").iloc[0]
            b = pd.to_numeric(pd.Series([fast.get(col)]), errors="coerce").iloc[0]
            if pd.isna(a) and pd.isna(b):
                central[col] = np.nan
            elif pd.isna(a):
                central[col] = float(b)
            elif pd.isna(b):
                central[col] = float(a)
            else:
                central[col] = (1.0 - float(central_blend)) * float(a) + float(central_blend) * float(b)
        rows.append(central.to_dict())

    out = pd.DataFrame(rows)
    order = {label: i for i, label in enumerate([slow_label, central_label, fast_label])}
    out["_scenario_order"] = out["scenario"].map(order).fillna(99)
    sort_cols = ["country", "delivery_year", "_scenario_order"]
    if "delivery_month" in out.columns:
        sort_cols.insert(2, "delivery_month")
    out = out.sort_values(sort_cols).drop(columns=["_scenario_order"]).reset_index(drop=True)
    return validate_electrification_scenarios(out, require_recommended=True)


def _write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _delivery_years(start_date: str, horizon_days: int, market: str) -> list[int]:
    tz = "Europe/Berlin" if market == "DE" else "Europe/Zurich"
    idx = pd.date_range(pd.Timestamp(start_date, tz="UTC"), periods=max(int(horizon_days), 1) * 96, freq="15min")
    return sorted(idx.tz_convert(tz).year.unique().astype(int).tolist())


def _expanded_scenario_path(
    scenario_path: str | Path,
    *,
    output_dir: str | Path,
    output_prefix: str,
    years: Sequence[int],
    scenarios: Sequence[str],
) -> Path:
    frame = load_electrification_scenarios(scenario_path, require_recommended=True)
    expanded = interpolate_electrification_scenario_years(frame, years, allow_clamp=False)
    expanded = expanded[expanded["scenario"].astype(str).isin([str(s) for s in scenarios])].copy()
    if expanded.empty:
        raise ValueError("expanded scenario table is empty")
    path = Path(output_dir) / f"{output_prefix}_scenario_expanded.parquet"
    _write_table(expanded, path)
    return path


def _fit_components(
    epex: pd.DataFrame,
    market: str,
    *,
    intraday_epex: pd.DataFrame | None = None,
    intraday_market: str | None = None,
    intraday_cutoff: str | pd.Timestamp | None = None,
    disable_cascade_trend_for_annual_only: bool = False,
    enable_sparse_intraday_regularization: bool = False,
):
    intraday = epex if intraday_epex is None else intraday_epex
    intraday_country = market if intraday_market is None else intraday_market
    if intraday_cutoff is not None:
        cutoff = pd.Timestamp(intraday_cutoff)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        intraday = intraday.loc[intraday.index >= cutoff]
    if intraday.empty:
        raise ValueError("intraday EPEX calibration frame is empty")
    if intraday_epex is not None:
        if not isinstance(intraday.index, pd.DatetimeIndex):
            raise TypeError("intraday EPEX calibration index must be a DatetimeIndex")
        if not intraday.index.is_monotonic_increasing or intraday.index.has_duplicates:
            raise ValueError("intraday EPEX calibration index must be ordered and unique")
        deltas = intraday.index.to_series().diff().dropna()
        if not deltas.eq(pd.Timedelta(minutes=15)).all():
            raise ValueError("intraday EPEX calibration grid must be contiguous 15-minute UTC")
        if set(intraday.index.minute.unique()) != {0, 15, 30, 45}:
            raise ValueError("intraday EPEX calibration grid must contain all four quarter-hours")

    cal = enrich_15min_index(epex.index, country=market)
    sh = ShapeHourly(use_seasonal_hourly=True).fit(epex, cal)
    sh._solar_epex_hist = epex
    intraday_cal = enrich_15min_index(intraday.index, country=intraday_country)
    intraday_options = (
        {"enable_sparse_support_regularization": True}
        if enable_sparse_intraday_regularization
        else {}
    )
    si = ShapeIntraday(**intraday_options).fit(intraday, None, intraday_cal)
    cascader = ContractCascader(
        disable_trend_for_annual_only=disable_cascade_trend_for_annual_only
    )
    cascader.fit_seasonal_ratios(epex)
    cascader.fit_peak_spreads(epex)
    calibrator = ArbitrageFreeCalibrator(smoothness_weight=1.0, tol=0.01)
    return sh, si, cascader, calibrator


def _build_one_curve(
    *,
    scenario: str,
    scenario_path: Path,
    base_prices: dict[str, float],
    reference_date: pd.Timestamp,
    market: str,
    start_date: str,
    horizon_days: int,
    sh,
    si,
    cascader,
    calibrator,
    monthly_authority: MonthlyLevelAuthority | None = None,
    delivery_index: pd.DatetimeIndex | None = None,
    peak_prices: dict[str, float] | None = None,
) -> pd.DataFrame:
    assembler_base_prices = dict(
        monthly_authority.assembler_base_prices
        if monthly_authority is not None
        else base_prices
    )
    assembler_base_prices.update(peak_prices or {})
    quoted_keys = set(
        monthly_authority.quoted_keys
        if monthly_authority is not None
        else base_prices
    )
    quoted_keys.update(peak_prices or {})
    monthly_constraint_tolerance = 1e-9
    if monthly_authority is not None:
        solver_config = dict(monthly_authority.manifest.get("solver_config", {}) or {})
        monthly_constraint_tolerance = float(solver_config.get("constraint_tolerance", 1e-9))
    assembler = PFCAssembler(
        shape_hourly=sh,
        shape_intraday=si,
        cascader=None if monthly_authority is not None else cascader,
        calibrator=calibrator,
        enable_electrification_shape=True,
        electrification_scenario=scenario,
        electrification_scenario_path=scenario_path,
        require_electrification_scenarios=True,
        enable_intraday_amplitude_shrinkage=True,
        monthly_level_authority="solver" if monthly_authority is not None else "legacy",
        skip_legacy_level_cascade=monthly_authority is not None,
        skip_legacy_base_smoothing=monthly_authority is not None,
        monthly_constraint_tolerance=monthly_constraint_tolerance,
    )
    pfc = assembler.build(
        base_prices=assembler_base_prices,
        quoted_keys=quoted_keys,
        start_date=start_date,
        horizon_days=horizon_days,
        reference_date=reference_date,
        country=market,
        delivery_index=delivery_index,
    )
    _validate_pfc(
        pfc,
        args=SimpleNamespace(start_date=start_date, horizon_days=horizon_days),
        expected_index=delivery_index,
    )
    return pfc


def build_weighted_fan_chart(
    pfc_by_scenario: Mapping[str, pd.DataFrame],
    *,
    weights: Mapping[str, float],
) -> pd.DataFrame:
    curves = {label: frame["price_shape"].astype(float) for label, frame in pfc_by_scenario.items()}
    fan = structural_fan_chart(curves, weights=weights)
    out = pd.DataFrame(index=fan.index)
    for label, frame in pfc_by_scenario.items():
        out[f"curve_{label}"] = frame["price_shape"].astype(float)
    out["weighted_mean"] = fan["weighted_mean"]
    out["structural_scenario_low"] = fan["scenario_low"]
    out["structural_scenario_central"] = fan["scenario_central"]
    out["structural_scenario_high"] = fan["scenario_high"]
    out["structural_scenario_spread"] = fan["scenario_spread"]
    bad = (
        out["structural_scenario_low"].notna()
        & out["structural_scenario_high"].notna()
        & (out["structural_scenario_low"] > out["structural_scenario_high"])
    )
    if bad.any():
        raise ValueError("structural scenario bracket violates low <= high")
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError("structural fan chart contains non-finite values")
    return out


def _block_summary(frame: pd.DataFrame, *, market: str) -> dict[str, float]:
    tz = "Europe/Berlin" if market == "DE" else "Europe/Zurich"
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


def _df_to_md(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["_empty_"]
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |"]
    lines.append("|" + "|".join("---" for _ in df.columns) + "|")
    for _, row in df.iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_summary(
    path: str | Path,
    *,
    args: argparse.Namespace,
    latest: pd.Timestamp,
    weights: Mapping[str, float],
    scenario_inventory: pd.DataFrame,
    expanded_path: Path,
    pfc_paths: Mapping[str, Path],
    fan_chart_path: Path,
    pfc_by_scenario: Mapping[str, pd.DataFrame],
    fan: pd.DataFrame,
) -> None:
    scenario_rows = scenario_inventory[
        scenario_inventory["delivery_year"].astype(int).isin([2030, 2031])
    ].copy()
    display_cols = [
        "scenario",
        "original_scenario",
        "delivery_year",
        "scenario_weight",
        "quality_flag",
        "demand_twh",
        "pv_twh",
        "wind_twh",
        "battery_power_gw",
        "battery_energy_gwh",
        "ev_twh",
        "heatpump_twh",
        "managed_charging_share",
        "hydro_reservoir_twh",
        "net_import_twh",
    ]
    display_cols = [c for c in display_cols if c in scenario_rows.columns]
    metrics = pd.DataFrame(
        [
            {"scenario": scenario, **_block_summary(frame, market=args.market)}
            for scenario, frame in pfc_by_scenario.items()
        ]
    )
    fan_metrics = pd.DataFrame(
        [
            {
                "rows": len(fan),
                "weighted_mean": float(fan["weighted_mean"].mean()),
                "scenario_low_mean": float(fan["structural_scenario_low"].mean()),
                "scenario_high_mean": float(fan["structural_scenario_high"].mean()),
                "scenario_spread_mean": float(fan["structural_scenario_spread"].mean()),
                "scenario_spread_p95": float(fan["structural_scenario_spread"].quantile(0.95)),
                "max_scenario_spread": float(fan["structural_scenario_spread"].max()),
            }
        ]
    )
    lines = [
        "# EP2050 Multi-Scenario PFC 2030",
        "",
        f"* market: `{args.market}`",
        f"* delivery start: `{args.start_date}`",
        f"* horizon days: `{args.horizon_days}`",
        f"* EEX forward date: `{latest.date()}`",
        f"* source enriched inventory: `{args.scenario_input}`",
        f"* mapped inventory: `{args.scenario_output}`",
        f"* gold features: `{args.features_output}`",
        f"* expanded scenario path: `{expanded_path}`",
        f"* fan chart: `{fan_chart_path}`",
        "* electrification shape: `ON`",
        "* intraday amplitude shrinkage: `ON`",
        "* missing scenario behavior: `fail-fast`",
        f"* scenario weights: `{', '.join(f'{k}={v:.4f}' for k, v in weights.items())}`",
        f"* mapping profile: `{PROFILE_NAME}`",
        "",
        "## PFC Outputs",
        "",
        *_df_to_md(pd.DataFrame({"scenario": list(pfc_paths), "path": [str(p) for p in pfc_paths.values()]})),
        "",
        "## Price Summary",
        "",
        *_df_to_md(metrics),
        "",
        "## Structural Fan Chart Summary",
        "",
        *_df_to_md(fan_metrics),
        "",
        "## Scenario Inventory Used",
        "",
        *_df_to_md(scenario_rows[display_cols].sort_values(["scenario", "delivery_year"])),
        "",
        "## Governance Notes",
        "",
        "* `slow` aliases OFEN `WWB` after the local proxy enrichment.",
        "* `fast` aliases OFEN `ZERO_Basis` after the local proxy enrichment.",
        "* `central` is an explicit midpoint between `WWB` and `ZERO_Basis`, stamped as an internal proxy assumption.",
        "* All scenario rows keep `publication_date <= vintage`; production/smoke builds use `require_electrification_scenarios=True`.",
        "* This is still a local proxy workflow until TYNDP, Pronovo, MaStR and governed NTC feeds are connected.",
        "",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _legacy_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-input", default="data/electrification_scenarios_ep2050_enriched.parquet")
    parser.add_argument("--scenario-output", default="data/electrification_scenarios_ep2050_enriched_slow_central_fast.parquet")
    parser.add_argument("--features-output", default="data/hpfc_scenario_features_ep2050_enriched_slow_central_fast.parquet")
    parser.add_argument("--epex-hourly", default="data/epex_hourly.parquet")
    parser.add_argument("--forwards", default="data/eex_forwards_history.parquet")
    parser.add_argument("--market", default="CH")
    parser.add_argument("--start-date", default="2030-01-01")
    parser.add_argument("--horizon-days", type=int, default=365)
    parser.add_argument("--scenarios", default="slow,central,fast")
    parser.add_argument("--weights", default="0.25,0.50,0.25")
    parser.add_argument("--slow-source", default="WWB")
    parser.add_argument("--fast-source", default="ZERO_Basis")
    parser.add_argument("--central-blend", type=float, default=0.5)
    parser.add_argument("--mapping-publication-date", default="2026-06-05")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--output-prefix", default="ep2050_pfc_2030")
    parser.add_argument("--fan-chart-output", default="output/ep2050_pfc_2030_structural_fan_chart.parquet")
    parser.add_argument("--summary", default=".planning/phases/13-lt-electrification-scenario-shape/MULTI-SCENARIO-PFC.md")
    args = parser.parse_args(argv)

    scenarios = _parse_csv(args.scenarios)
    weights = _parse_weights(args.weights, scenarios)
    source = load_electrification_scenarios(args.scenario_input, require_recommended=True)
    inventory = derive_slow_central_fast_inventory(
        source,
        slow_source=args.slow_source,
        fast_source=args.fast_source,
        scenarios=scenarios,
        weights=weights,
        central_blend=args.central_blend,
        mapping_publication_date=args.mapping_publication_date,
    )
    _write_table(inventory, args.scenario_output)
    _write_table(derive_hpfc_scenario_features(inventory), args.features_output)

    years = _delivery_years(args.start_date, args.horizon_days, args.market)
    expanded_path = _expanded_scenario_path(
        args.scenario_output,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        years=years,
        scenarios=scenarios,
    )

    epex = _load_epex_hourly(args.epex_hourly)
    latest, base_prices = _latest_forwards(args.forwards, market=args.market)
    reference_date = pd.Timestamp(latest, tz="UTC")
    sh, si, cascader, calibrator = _fit_components(epex, args.market)

    pfc_by_scenario: dict[str, pd.DataFrame] = {}
    pfc_paths: dict[str, Path] = {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        pfc = _build_one_curve(
            scenario=scenario,
            scenario_path=expanded_path,
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
        pfc.to_parquet(path)
        pfc_by_scenario[scenario] = pfc
        pfc_paths[scenario] = path

    fan = build_weighted_fan_chart(pfc_by_scenario, weights=weights)
    fan_chart_path = Path(args.fan_chart_output)
    fan_chart_path.parent.mkdir(parents=True, exist_ok=True)
    fan.to_parquet(fan_chart_path)
    _write_summary(
        args.summary,
        args=args,
        latest=latest,
        weights=weights,
        scenario_inventory=inventory,
        expanded_path=expanded_path,
        pfc_paths=pfc_paths,
        fan_chart_path=fan_chart_path,
        pfc_by_scenario=pfc_by_scenario,
        fan=fan,
    )

    print(f"[multi-pfc] scenarios={','.join(scenarios)}")
    print(f"[multi-pfc] inventory -> {args.scenario_output}")
    print(f"[multi-pfc] features -> {args.features_output}")
    for scenario, path in pfc_paths.items():
        print(f"[multi-pfc] {scenario} -> {path}")
    print(f"[multi-pfc] fan chart -> {fan_chart_path}")
    print(f"[multi-pfc] summary -> {args.summary}")
    print(
        f"[multi-pfc] weighted_mean={fan['weighted_mean'].mean():.2f} "
        f"scenario_spread_mean={fan['structural_scenario_spread'].mean():.2f}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    del argv
    raise RuntimeError(
        "direct multi-scenario publication is disabled: use "
        "python -m scripts.build_local_test_ch_pfc for an explicit local/test NO-GO run"
    )


if __name__ == "__main__":
    raise SystemExit(main())

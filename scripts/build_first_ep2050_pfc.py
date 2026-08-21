"""Build a first local LT PFC using EEX forwards and OFEN EP2050+ scenarios.

This is a production-readiness smoke runner: it does not replace the daily
pipeline, but produces an inspectable first curve with the Phase 13 structural
scenario layer explicitly enabled and fail-fast scenario coverage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pfc_shaping.calibration.arbitrage_free import ArbitrageFreeCalibrator
from pfc_shaping.calibration.cascading import ContractCascader
from pfc_shaping.data.calendar_ch import enrich_15min_index
from pfc_shaping.data.electrification_scenarios import (
    interpolate_electrification_scenario_years,
    load_electrification_scenarios,
)
from pfc_shaping.lt.model.assembler import PFCAssembler
from pfc_shaping.lt.model.shape_hourly import ShapeHourly
from pfc_shaping.lt.model.shape_intraday import ShapeIntraday


def _load_epex_hourly(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "price_eur_mwh" in df.columns:
        out = df[["price_eur_mwh"]].astype(float).copy()
    elif len(df.columns) == 1:
        out = df.iloc[:, [0]].astype(float).copy()
        out.columns = ["price_eur_mwh"]
    else:
        raise KeyError(f"cannot identify EPEX price column in {path}")
    if out.index.tz is None:
        out.index = pd.to_datetime(out.index, utc=True)
    else:
        out.index = out.index.tz_convert("UTC")
    return out.sort_index()


def _latest_forwards(path: str | Path, *, market: str) -> tuple[pd.Timestamp, dict[str, float]]:
    df = pd.read_parquet(path)
    sub = df[(df["market"].astype(str) == str(market)) & (df["load_type"].astype(str) == "BASE")].copy()
    if sub.empty:
        raise ValueError(f"no BASE forwards for market={market!r} in {path}")
    latest = pd.Timestamp(sub["date"].max())
    snap = sub[sub["date"] == latest]
    base_prices = dict(zip(snap["product"].astype(str), snap["price"].astype(float)))
    if not base_prices:
        raise ValueError(f"empty latest forward snapshot for market={market!r}, date={latest}")
    return latest, base_prices


def _scenario_lines(path: Path) -> list[str]:
    cols = [
        "scenario",
        "delivery_year",
        "quality_flag",
        "demand_twh",
        "pv_twh",
        "wind_twh",
        "battery_power_gw",
        "battery_energy_gwh",
        "managed_charging_share",
        "winter_demand_twh",
        "hydro_twh",
        "hydro_reservoir_twh",
        "net_import_twh",
        "ntc_ch_de_gw",
        "ntc_ch_fr_gw",
        "ntc_ch_it_gw",
        "scenario_weight",
    ]
    try:
        scenario = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - summary must not hide a valid PFC
        return [f"Could not read expanded scenario table `{path}`: `{exc}`."]
    keep = [c for c in cols if c in scenario.columns]
    if not keep:
        return [f"Expanded scenario table `{path}` contains no displayable columns."]

    lines = ["| " + " | ".join(keep) + " |", "|" + "|".join("---" for _ in keep) + "|"]
    for _, row in scenario[keep].sort_values([c for c in ["scenario", "delivery_year"] if c in keep]).iterrows():
        rendered = []
        for col in keep:
            value = row[col]
            if pd.isna(value):
                rendered.append("")
            elif isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def _validate_pfc(
    pfc: pd.DataFrame,
    *,
    args: argparse.Namespace,
    expected_index: pd.DatetimeIndex | None = None,
) -> None:
    if expected_index is None:
        expected_rows = max(int(args.horizon_days), 1) * 96
        expected_index = pd.date_range(
            pd.Timestamp(args.start_date, tz="UTC"),
            periods=expected_rows,
            freq="15min",
        )
    else:
        expected_index = pd.DatetimeIndex(expected_index)
        expected_rows = len(expected_index)
    if len(pfc) != expected_rows:
        raise ValueError(f"PFC row count mismatch: expected {expected_rows}, got {len(pfc)}")
    if not isinstance(pfc.index, pd.DatetimeIndex):
        raise TypeError("PFC index must be a DatetimeIndex")
    if pfc.index.tz is None:
        raise ValueError("PFC index must be timezone-aware")
    if not pfc.index.equals(expected_index):
        raise ValueError("PFC index must be a complete 15-min UTC grid for the requested horizon")
    if not pfc.index.is_monotonic_increasing or not pfc.index.is_unique:
        raise ValueError("PFC index must be sorted and unique")
    if "price_shape" not in pfc.columns:
        raise KeyError("PFC output missing price_shape")
    price = pfc["price_shape"].to_numpy(dtype=float)
    if not np.isfinite(price).all():
        raise ValueError("PFC price_shape contains NaN or infinite values")
    if {"p10", "p90"} <= set(pfc.columns):
        bad = pfc["p10"].notna() & pfc["p90"].notna() & (pfc["p10"] > pfc["p90"])
        if bad.any():
            raise ValueError("PFC structural fan chart violates p10 <= p90")


def _write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    pfc: pd.DataFrame,
    latest: pd.Timestamp,
    base_prices: dict[str, float],
    expanded_scenario_path: Path,
) -> None:
    local = pfc.index.tz_convert("Europe/Zurich")
    summary = {
        "rows": len(pfc),
        "start": str(pfc.index.min()),
        "end": str(pfc.index.max()),
        "mean": float(pfc["price_shape"].mean()),
        "min": float(pfc["price_shape"].min()),
        "max": float(pfc["price_shape"].max()),
        "p05": float(pfc["price_shape"].quantile(0.05)),
        "p95": float(pfc["price_shape"].quantile(0.95)),
        "midday_mean": float(pfc.loc[(local.hour >= 10) & (local.hour <= 15), "price_shape"].mean()),
        "evening_mean": float(pfc.loc[(local.hour >= 18) & (local.hour <= 21), "price_shape"].mean()),
        "night_mean": float(pfc.loc[(local.hour <= 5) | (local.hour >= 22), "price_shape"].mean()),
    }
    lines = [
        "# First EP2050 PFC",
        "",
        f"* market: `{args.market}`",
        f"* scenario: `{args.scenario}`",
        f"* delivery start: `{args.start_date}`",
        f"* horizon days: `{args.horizon_days}`",
        f"* EEX forward date: `{latest.date()}`",
        f"* forward keys: `{len(base_prices)}`",
        f"* source scenario path: `{args.scenario_path}`",
        f"* expanded scenario path: `{expanded_scenario_path}`",
        "* electrification shape: `ON`",
        "* intraday amplitude shrinkage: `ON`",
        "* missing scenario behavior: `fail-fast`",
        f"* output parquet: `{args.output}`",
        "",
        "## Price Summary",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    lines.extend(f"| `{k}` | {v:.4f} |" if isinstance(v, float) else f"| `{k}` | {v} |" for k, v in summary.items())
    lines.extend(
        [
            "",
            "## Forward Keys",
            "",
            ", ".join(sorted(base_prices)),
            "",
            "## Expanded Scenario Used",
            "",
            *_scenario_lines(expanded_scenario_path),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _expanded_scenario_path(args: argparse.Namespace) -> Path:
    tz = "Europe/Berlin" if args.market == "DE" else "Europe/Zurich"
    idx = pd.date_range(
        pd.Timestamp(args.start_date, tz="UTC"),
        periods=max(int(args.horizon_days), 1) * 96,
        freq="15min",
    )
    years = sorted(idx.tz_convert(tz).year.unique().astype(int).tolist())
    frame = load_electrification_scenarios(args.scenario_path)
    expanded = interpolate_electrification_scenario_years(frame, years, allow_clamp=False)
    expanded = expanded[expanded["scenario"].astype(str) == str(args.scenario)].copy()
    if expanded.empty:
        raise ValueError(f"scenario {args.scenario!r} not found in {args.scenario_path}")
    path = Path(args.output).with_name(Path(args.output).stem + "_scenario_expanded.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_parquet(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epex-hourly", default="data/epex_hourly.parquet")
    parser.add_argument("--forwards", default="data/eex_forwards_history.parquet")
    parser.add_argument("--scenario-path", default="data/electrification_scenarios_ep2050.parquet")
    parser.add_argument("--market", default="CH")
    parser.add_argument("--scenario", default="ZERO_Basis")
    parser.add_argument("--start-date", default="2030-01-01")
    parser.add_argument("--horizon-days", type=int, default=365)
    parser.add_argument("--output", default="output/first_ep2050_pfc_2030_zero_basis.parquet")
    parser.add_argument("--summary", default=".planning/phases/13-lt-electrification-scenario-shape/FIRST-PFC.md")
    args = parser.parse_args(argv)

    epex = _load_epex_hourly(args.epex_hourly)
    latest, base_prices = _latest_forwards(args.forwards, market=args.market)
    reference_date = pd.Timestamp(latest, tz="UTC")
    cal = enrich_15min_index(epex.index, country=args.market)

    sh = ShapeHourly(use_seasonal_hourly=True).fit(epex, cal)
    sh._solar_epex_hist = epex
    si = ShapeIntraday().fit(epex, None, cal)
    cascader = ContractCascader()
    cascader.fit_seasonal_ratios(epex)
    cascader.fit_peak_spreads(epex)
    calibrator = ArbitrageFreeCalibrator(smoothness_weight=1.0, tol=0.01)

    scenario_path = _expanded_scenario_path(args)
    assembler = PFCAssembler(
        shape_hourly=sh,
        shape_intraday=si,
        cascader=cascader,
        calibrator=calibrator,
        enable_electrification_shape=True,
        electrification_scenario=args.scenario,
        electrification_scenario_path=scenario_path,
        require_electrification_scenarios=True,
        enable_intraday_amplitude_shrinkage=True,
    )
    pfc = assembler.build(
        base_prices=base_prices,
        quoted_keys=set(base_prices),
        start_date=args.start_date,
        horizon_days=args.horizon_days,
        reference_date=reference_date,
        country=args.market,
    )
    _validate_pfc(pfc, args=args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pfc.to_parquet(output)
    _write_summary(
        Path(args.summary),
        args=args,
        pfc=pfc,
        latest=latest,
        base_prices=base_prices,
        expanded_scenario_path=scenario_path,
    )

    print(f"[first-pfc] rows={len(pfc)}")
    print(f"[first-pfc] output -> {output}")
    print(f"[first-pfc] summary -> {args.summary}")
    print(f"[first-pfc] mean={pfc['price_shape'].mean():.2f} min={pfc['price_shape'].min():.2f} max={pfc['price_shape'].max():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

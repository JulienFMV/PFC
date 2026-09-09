"""Compare baseline vs adjusted EPEX shape-lab candidates without OMPEX.

This diagnostic is intentionally independent from OMPEX/HFC.  It checks that
an adjusted lab candidate keeps timestamps aligned, preserves monthly levels,
does not rebuild fan width unexpectedly, and records the calendar shape effects
introduced by the A/B delta.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.export_local_test_ch_hourly_csv import _eex_peak_mask, _parse_timestamp_ch  # noqa: E402


PRICE = "price_weighted_mean_eur_mwh"
PRICE_COLUMNS = [
    "price_slow_eur_mwh",
    "price_central_eur_mwh",
    "price_fast_eur_mwh",
    PRICE,
    "structural_p10_eur_mwh",
    "structural_p50_eur_mwh",
    "structural_p90_eur_mwh",
    "structural_width_eur_mwh",
]


def compare_ab(
    *,
    baseline_csv: Path,
    adjusted_csv: Path,
    output_dir: Path,
) -> dict[str, Any]:
    baseline = _load_hourly(baseline_csv)
    adjusted = _load_hourly(adjusted_csv)
    joined = baseline.join(adjusted, how="inner", lsuffix="_baseline", rsuffix="_adjusted")
    if len(joined) != len(baseline) or len(joined) != len(adjusted):
        raise ValueError(
            "baseline and adjusted candidates must have identical timestamp sets: "
            f"baseline={len(baseline)}, adjusted={len(adjusted)}, overlap={len(joined)}"
        )
    joined = _add_calendar(joined)
    joined["delta_eur_mwh"] = joined[f"{PRICE}_adjusted"] - joined[f"{PRICE}_baseline"]
    joined["width_delta_eur_mwh"] = (
        joined["structural_width_eur_mwh_adjusted"] - joined["structural_width_eur_mwh_baseline"]
    )
    joined["implied_width_eur_mwh_baseline"] = (
        joined["structural_p90_eur_mwh_baseline"] - joined["structural_p10_eur_mwh_baseline"]
    )
    joined["implied_width_eur_mwh_adjusted"] = (
        joined["structural_p90_eur_mwh_adjusted"] - joined["structural_p10_eur_mwh_adjusted"]
    )
    joined["implied_width_delta_eur_mwh"] = (
        joined["implied_width_eur_mwh_adjusted"] - joined["implied_width_eur_mwh_baseline"]
    )
    joined["reported_minus_implied_width_eur_mwh_baseline"] = (
        joined["structural_width_eur_mwh_baseline"] - joined["implied_width_eur_mwh_baseline"]
    )
    joined["reported_minus_implied_width_eur_mwh_adjusted"] = (
        joined["structural_width_eur_mwh_adjusted"] - joined["implied_width_eur_mwh_adjusted"]
    )

    monthly = _monthly_summary(joined)
    annual = _annual_summary(joined)
    calendar = _calendar_summary(joined)
    load_type = _load_type_summary(joined)
    month_hour = _month_hour_delta_summary(joined)
    peak_offpeak = _peak_offpeak_monthly_summary(joined)
    boundaries = _boundary_delta_jumps(joined)
    summary = _summary(
        joined,
        monthly=monthly,
        month_hour=month_hour,
        peak_offpeak=peak_offpeak,
        boundaries=boundaries,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    joined.reset_index(names="timestamp_ch").to_csv(output_dir / "aligned_baseline_adjusted.csv", index=False)
    monthly.to_csv(output_dir / "monthly_summary.csv", index=False)
    annual.to_csv(output_dir / "annual_summary.csv", index=False)
    calendar.to_csv(output_dir / "calendar_delta_summary.csv", index=False)
    load_type.to_csv(output_dir / "load_type_delta_summary.csv", index=False)
    month_hour.to_csv(output_dir / "month_hour_delta_summary.csv", index=False)
    peak_offpeak.to_csv(output_dir / "peak_offpeak_monthly_summary.csv", index=False)
    boundaries.to_csv(output_dir / "boundary_delta_jumps.csv", index=False)
    _write_plots(
        month_hour=month_hour,
        peak_offpeak=peak_offpeak,
        boundaries=boundaries,
        output_dir=output_dir,
    )
    summary.update(
        {
            "benchmark_policy": "independent_no_ompex",
            "ompex_used_in_model": False,
            "ompex_used_in_selection": False,
            "baseline_csv": str(baseline_csv),
            "adjusted_csv": str(adjusted_csv),
            "outputs": {
                "aligned_csv": str(output_dir / "aligned_baseline_adjusted.csv"),
                "monthly_summary_csv": str(output_dir / "monthly_summary.csv"),
                "annual_summary_csv": str(output_dir / "annual_summary.csv"),
                "calendar_delta_summary_csv": str(output_dir / "calendar_delta_summary.csv"),
                "load_type_delta_summary_csv": str(output_dir / "load_type_delta_summary.csv"),
                "month_hour_delta_summary_csv": str(output_dir / "month_hour_delta_summary.csv"),
                "peak_offpeak_monthly_summary_csv": str(output_dir / "peak_offpeak_monthly_summary.csv"),
                "boundary_delta_jumps_csv": str(output_dir / "boundary_delta_jumps.csv"),
                "delta_heatmap_png_dir": str(output_dir),
                "peak_offpeak_spread_png": str(output_dir / "peak_offpeak_spread_delta_by_month.png"),
                "boundary_delta_jumps_png": str(output_dir / "boundary_delta_jumps.png"),
            },
        }
    )
    (output_dir / "ab_comparison_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_hourly(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(["timestamp_ch", *PRICE_COLUMNS]) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    ts = _parse_timestamp_ch(frame["timestamp_ch"], frame.get("utc_offset_ch"))
    out = frame.copy()
    for column in PRICE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="raise")
    out.index = pd.DatetimeIndex(ts)
    if out.index.has_duplicates:
        raise ValueError(f"{path} has duplicate timestamps")
    return out[PRICE_COLUMNS].sort_index()


def _add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    idx = out.index
    out["year"] = idx.year.astype(int)
    out["month"] = idx.month.astype(int)
    out["year_month"] = pd.PeriodIndex(idx.strftime("%Y-%m"), freq="M").astype(str)
    out["hour"] = idx.hour.astype(int)
    out["weekday"] = idx.weekday.astype(int)
    out["is_weekend"] = out["weekday"] >= 5
    out["is_eex_peak"] = _eex_peak_mask(pd.Series(idx, index=out.index), country="CH").to_numpy(dtype=bool)
    out["is_solar_tail"] = out["month"].between(3, 10) & out["hour"].between(10, 16)
    out["is_evening_ramp"] = out["hour"].between(17, 21)
    out["is_midday"] = out["hour"].between(11, 15)
    out["is_night"] = out["hour"].between(0, 5)
    return out


def _monthly_summary(joined: pd.DataFrame) -> pd.DataFrame:
    grouped = joined.groupby("year_month", sort=True)
    rows = []
    for month, group in grouped:
        rows.append(
            {
                "year_month": month,
                "rows": int(len(group)),
                "baseline_mean_eur_mwh": float(group[f"{PRICE}_baseline"].mean()),
                "adjusted_mean_eur_mwh": float(group[f"{PRICE}_adjusted"].mean()),
                "mean_delta_eur_mwh": float(group["delta_eur_mwh"].mean()),
                "max_abs_delta_eur_mwh": float(group["delta_eur_mwh"].abs().max()),
                "width_delta_mean_eur_mwh": float(group["width_delta_eur_mwh"].mean()),
                "implied_width_delta_mean_eur_mwh": float(group["implied_width_delta_eur_mwh"].mean()),
                "max_abs_reported_minus_implied_width_adjusted_eur_mwh": float(
                    group["reported_minus_implied_width_eur_mwh_adjusted"].abs().max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _annual_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in joined.groupby("year", sort=True):
        baseline_midday = float(group.loc[group["is_midday"], f"{PRICE}_baseline"].mean())
        adjusted_midday = float(group.loc[group["is_midday"], f"{PRICE}_adjusted"].mean())
        baseline_evening = float(group.loc[group["is_evening_ramp"], f"{PRICE}_baseline"].mean())
        adjusted_evening = float(group.loc[group["is_evening_ramp"], f"{PRICE}_adjusted"].mean())
        baseline_weekend = float(group.loc[group["is_weekend"], f"{PRICE}_baseline"].mean())
        adjusted_weekend = float(group.loc[group["is_weekend"], f"{PRICE}_adjusted"].mean())
        baseline_weekday = float(group.loc[~group["is_weekend"], f"{PRICE}_baseline"].mean())
        adjusted_weekday = float(group.loc[~group["is_weekend"], f"{PRICE}_adjusted"].mean())
        rows.append(
            {
                "year": int(year),
                "baseline_evening_minus_midday_eur_mwh": baseline_evening - baseline_midday,
                "adjusted_evening_minus_midday_eur_mwh": adjusted_evening - adjusted_midday,
                "evening_minus_midday_change_eur_mwh": (adjusted_evening - adjusted_midday)
                - (baseline_evening - baseline_midday),
                "baseline_weekend_minus_weekday_eur_mwh": baseline_weekend - baseline_weekday,
                "adjusted_weekend_minus_weekday_eur_mwh": adjusted_weekend - adjusted_weekday,
                "weekend_minus_weekday_change_eur_mwh": (adjusted_weekend - adjusted_weekday)
                - (baseline_weekend - baseline_weekday),
            }
        )
    return pd.DataFrame(rows)


def _calendar_summary(joined: pd.DataFrame) -> pd.DataFrame:
    buckets = {
        "all": pd.Series(True, index=joined.index),
        "weekend": joined["is_weekend"],
        "weekday": ~joined["is_weekend"],
        "solar_tail_mar_oct_10_16": joined["is_solar_tail"],
        "evening_ramp_17_21": joined["is_evening_ramp"],
        "midday_11_15": joined["is_midday"],
        "night_00_05": joined["is_night"],
    }
    rows = []
    for name, mask in buckets.items():
        group = joined.loc[mask.to_numpy(dtype=bool)]
        if group.empty:
            continue
        rows.append(
            {
                "bucket": name,
                "rows": int(len(group)),
                "mean_delta_eur_mwh": float(group["delta_eur_mwh"].mean()),
                "p05_delta_eur_mwh": float(group["delta_eur_mwh"].quantile(0.05)),
                "p50_delta_eur_mwh": float(group["delta_eur_mwh"].quantile(0.50)),
                "p95_delta_eur_mwh": float(group["delta_eur_mwh"].quantile(0.95)),
                "max_abs_delta_eur_mwh": float(group["delta_eur_mwh"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def _load_type_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    buckets = {
        "BASE": pd.Series(True, index=joined.index),
        "PEAK": joined["is_eex_peak"],
        "OFFPEAK": ~joined["is_eex_peak"],
    }
    for load_type, mask in buckets.items():
        group = joined.loc[mask.to_numpy(dtype=bool)]
        if group.empty:
            continue
        ramp_baseline = group[f"{PRICE}_baseline"].diff().abs().dropna()
        ramp_adjusted = group[f"{PRICE}_adjusted"].diff().abs().dropna()
        rows.append(
            {
                "load_type": load_type,
                "rows": int(len(group)),
                "baseline_mean_eur_mwh": float(group[f"{PRICE}_baseline"].mean()),
                "adjusted_mean_eur_mwh": float(group[f"{PRICE}_adjusted"].mean()),
                "mean_delta_eur_mwh": float(group["delta_eur_mwh"].mean()),
                "p05_delta_eur_mwh": float(group["delta_eur_mwh"].quantile(0.05)),
                "p50_delta_eur_mwh": float(group["delta_eur_mwh"].quantile(0.50)),
                "p95_delta_eur_mwh": float(group["delta_eur_mwh"].quantile(0.95)),
                "max_abs_delta_eur_mwh": float(group["delta_eur_mwh"].abs().max()),
                "ramp_abs_p99_baseline_eur_mwh": (
                    float(ramp_baseline.quantile(0.99)) if not ramp_baseline.empty else 0.0
                ),
                "ramp_abs_p99_adjusted_eur_mwh": (
                    float(ramp_adjusted.quantile(0.99)) if not ramp_adjusted.empty else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _month_hour_delta_summary(joined: pd.DataFrame) -> pd.DataFrame:
    grouped = joined.groupby(["year", "month", "hour"], sort=True)
    rows = []
    for (year, month, hour), group in grouped:
        rows.append(
            {
                "year": int(year),
                "month": int(month),
                "hour": int(hour),
                "rows": int(len(group)),
                "baseline_mean_eur_mwh": float(group[f"{PRICE}_baseline"].mean()),
                "adjusted_mean_eur_mwh": float(group[f"{PRICE}_adjusted"].mean()),
                "mean_delta_eur_mwh": float(group["delta_eur_mwh"].mean()),
                "max_abs_delta_eur_mwh": float(group["delta_eur_mwh"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def _peak_offpeak_monthly_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year_month, group in joined.groupby("year_month", sort=True):
        peak = group.loc[group["is_eex_peak"].to_numpy(dtype=bool)]
        offpeak = group.loc[(~group["is_eex_peak"]).to_numpy(dtype=bool)]
        if peak.empty or offpeak.empty:
            continue
        baseline_peak = float(peak[f"{PRICE}_baseline"].mean())
        adjusted_peak = float(peak[f"{PRICE}_adjusted"].mean())
        baseline_offpeak = float(offpeak[f"{PRICE}_baseline"].mean())
        adjusted_offpeak = float(offpeak[f"{PRICE}_adjusted"].mean())
        rows.append(
            {
                "year_month": str(year_month),
                "peak_rows": int(len(peak)),
                "offpeak_rows": int(len(offpeak)),
                "baseline_peak_mean_eur_mwh": baseline_peak,
                "adjusted_peak_mean_eur_mwh": adjusted_peak,
                "peak_mean_delta_eur_mwh": adjusted_peak - baseline_peak,
                "baseline_offpeak_mean_eur_mwh": baseline_offpeak,
                "adjusted_offpeak_mean_eur_mwh": adjusted_offpeak,
                "offpeak_mean_delta_eur_mwh": adjusted_offpeak - baseline_offpeak,
                "baseline_peak_offpeak_spread_eur_mwh": baseline_peak - baseline_offpeak,
                "adjusted_peak_offpeak_spread_eur_mwh": adjusted_peak - adjusted_offpeak,
                "peak_offpeak_spread_delta_eur_mwh": (adjusted_peak - adjusted_offpeak)
                - (baseline_peak - baseline_offpeak),
            }
        )
    return pd.DataFrame(rows)


def _boundary_delta_jumps(joined: pd.DataFrame) -> pd.DataFrame:
    ordered = joined.sort_index().copy()
    ordered["previous_year_month"] = ordered["year_month"].shift(1)
    ordered["previous_delta_eur_mwh"] = ordered["delta_eur_mwh"].shift(1)
    ordered["previous_baseline_price_eur_mwh"] = ordered[f"{PRICE}_baseline"].shift(1)
    ordered["previous_adjusted_price_eur_mwh"] = ordered[f"{PRICE}_adjusted"].shift(1)
    ordered["is_month_boundary"] = ordered["year_month"].ne(ordered["previous_year_month"])
    ordered.iloc[0, ordered.columns.get_loc("is_month_boundary")] = False
    boundaries = ordered.loc[ordered["is_month_boundary"]].copy()
    if boundaries.empty:
        return pd.DataFrame(
            columns=[
                "timestamp_ch",
                "year_month",
                "previous_year_month",
                "is_year_boundary",
                "delta_eur_mwh",
                "previous_delta_eur_mwh",
                "delta_jump_eur_mwh",
                "baseline_price_jump_eur_mwh",
                "adjusted_price_jump_eur_mwh",
            ]
        )
    boundaries["delta_jump_eur_mwh"] = boundaries["delta_eur_mwh"] - boundaries["previous_delta_eur_mwh"]
    boundaries["baseline_price_jump_eur_mwh"] = (
        boundaries[f"{PRICE}_baseline"] - boundaries["previous_baseline_price_eur_mwh"]
    )
    boundaries["adjusted_price_jump_eur_mwh"] = (
        boundaries[f"{PRICE}_adjusted"] - boundaries["previous_adjusted_price_eur_mwh"]
    )
    boundaries["is_year_boundary"] = boundaries.index.month == 1
    return boundaries.reset_index(names="timestamp_ch")[
        [
            "timestamp_ch",
            "year_month",
            "previous_year_month",
            "is_year_boundary",
            "delta_eur_mwh",
            "previous_delta_eur_mwh",
            "delta_jump_eur_mwh",
            "baseline_price_jump_eur_mwh",
            "adjusted_price_jump_eur_mwh",
        ]
    ]


def _summary(
    joined: pd.DataFrame,
    *,
    monthly: pd.DataFrame,
    month_hour: pd.DataFrame,
    peak_offpeak: pd.DataFrame,
    boundaries: pd.DataFrame,
) -> dict[str, Any]:
    numeric = joined[[f"{col}_adjusted" for col in PRICE_COLUMNS if col != "structural_width_eur_mwh"]]
    quantile_order = bool(
        (
            (joined["structural_p10_eur_mwh_adjusted"] <= joined["structural_p50_eur_mwh_adjusted"])
            & (joined["structural_p50_eur_mwh_adjusted"] <= joined["structural_p90_eur_mwh_adjusted"])
        ).all()
    )
    ramp_baseline = joined[f"{PRICE}_baseline"].diff().abs().dropna()
    ramp_adjusted = joined[f"{PRICE}_adjusted"].diff().abs().dropna()
    weighted_negative_mask = joined[f"{PRICE}_adjusted"] < 0.0
    p10_negative_mask = joined["structural_p10_eur_mwh_adjusted"] < 0.0
    return {
        "n_hours": int(len(joined)),
        "finite_adjusted_ok": bool(np.isfinite(numeric.to_numpy(dtype=float)).all()),
        "quantile_order_adjusted_ok": quantile_order,
        "weighted_negative_hours_adjusted": int(weighted_negative_mask.sum()),
        "p10_negative_hours_adjusted": int(p10_negative_mask.sum()),
        "p50_negative_hours_adjusted": int((joined["structural_p50_eur_mwh_adjusted"] < 0.0).sum()),
        "p90_negative_hours_adjusted": int((joined["structural_p90_eur_mwh_adjusted"] < 0.0).sum()),
        "slow_negative_hours_adjusted": int((joined["price_slow_eur_mwh_adjusted"] < 0.0).sum()),
        "central_negative_hours_adjusted": int((joined["price_central_eur_mwh_adjusted"] < 0.0).sum()),
        "fast_negative_hours_adjusted": int((joined["price_fast_eur_mwh_adjusted"] < 0.0).sum()),
        "weighted_negative_cluster_max_hours": _max_true_cluster(weighted_negative_mask),
        "p10_negative_cluster_max_hours": _max_true_cluster(p10_negative_mask),
        "min_adjusted_price_eur_mwh": float(numeric.min().min()),
        "max_abs_delta_eur_mwh": float(joined["delta_eur_mwh"].abs().max()),
        "mean_delta_eur_mwh": float(joined["delta_eur_mwh"].mean()),
        "max_abs_monthly_mean_delta_eur_mwh": float(monthly["mean_delta_eur_mwh"].abs().max()),
        "max_abs_month_hour_mean_delta_eur_mwh": _max_abs_column(month_hour, "mean_delta_eur_mwh"),
        "max_abs_peak_offpeak_spread_delta_eur_mwh": _max_abs_column(
            peak_offpeak,
            "peak_offpeak_spread_delta_eur_mwh",
        ),
        "max_abs_boundary_delta_jump_eur_mwh": _max_abs_column(boundaries, "delta_jump_eur_mwh"),
        "max_abs_width_delta_eur_mwh": float(joined["width_delta_eur_mwh"].abs().max()),
        "max_abs_implied_width_delta_eur_mwh": float(joined["implied_width_delta_eur_mwh"].abs().max()),
        "max_abs_reported_minus_implied_width_baseline_eur_mwh": float(
            joined["reported_minus_implied_width_eur_mwh_baseline"].abs().max()
        ),
        "max_abs_reported_minus_implied_width_adjusted_eur_mwh": float(
            joined["reported_minus_implied_width_eur_mwh_adjusted"].abs().max()
        ),
        "ramp_abs_p99_baseline_eur_mwh": float(ramp_baseline.quantile(0.99)),
        "ramp_abs_p99_adjusted_eur_mwh": float(ramp_adjusted.quantile(0.99)),
        "ramp_abs_max_baseline_eur_mwh": float(ramp_baseline.max()),
        "ramp_abs_max_adjusted_eur_mwh": float(ramp_adjusted.max()),
    }


def _max_abs_column(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(frame[column].abs().max())


def _max_true_cluster(mask: pd.Series) -> int:
    if mask.empty:
        return 0
    values = mask.to_numpy(dtype=bool)
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _write_plots(
    *,
    month_hour: pd.DataFrame,
    peak_offpeak: pd.DataFrame,
    boundaries: pd.DataFrame,
    output_dir: Path,
) -> None:
    _plot_delta_heatmaps(month_hour, output_dir)
    _plot_peak_offpeak_spread(peak_offpeak, output_dir)
    _plot_boundary_delta_jumps(boundaries, output_dir)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_delta_heatmaps(month_hour: pd.DataFrame, output_dir: Path) -> None:
    if month_hour.empty:
        return
    for year, group in month_hour.groupby("year", sort=True):
        heat = group.pivot_table(index="month", columns="hour", values="mean_delta_eur_mwh", aggfunc="mean")
        if heat.empty:
            continue
        fig, ax = plt.subplots(figsize=(13, 5))
        max_abs = float(np.nanmax(np.abs(heat.to_numpy(dtype=float)))) if heat.size else 0.0
        vmax = max(max_abs, 0.1)
        im = ax.imshow(heat.values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{int(year)} baseline vs adjusted delta: month x hour")
        ax.set_xlabel("Hour CH")
        ax.set_ylabel("Month")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels(range(0, 24, 2))
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels([f"M{int(month):02d}" for month in heat.index])
        fig.colorbar(im, ax=ax, label="Adjusted - baseline (EUR/MWh)")
        _save(fig, output_dir / f"delta_heatmap_month_hour_{int(year)}.png")


def _plot_peak_offpeak_spread(peak_offpeak: pd.DataFrame, output_dir: Path) -> None:
    if peak_offpeak.empty:
        return
    plot = peak_offpeak.copy()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        plot["year_month"],
        plot["peak_offpeak_spread_delta_eur_mwh"],
        marker="o",
        linewidth=1.8,
        color="#7c3aed",
    )
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_title("Baseline vs adjusted PEAK-OFFPEAK spread delta")
    ax.set_xlabel("Delivery month")
    ax.set_ylabel("Spread delta (EUR/MWh)")
    tick_positions = np.arange(0, len(plot), max(1, len(plot) // 18), dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        plot["year_month"].iloc[tick_positions].astype(str),
        rotation=70,
        ha="right",
        fontsize=7,
    )
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, output_dir / "peak_offpeak_spread_delta_by_month.png")


def _plot_boundary_delta_jumps(boundaries: pd.DataFrame, output_dir: Path) -> None:
    if boundaries.empty:
        return
    plot = boundaries.copy()
    colors = np.where(plot["is_year_boundary"].astype(bool), "#dc2626", "#2563eb")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(np.arange(len(plot)), plot["delta_jump_eur_mwh"], color=colors)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_title("Adjusted delta jumps at month boundaries")
    ax.set_xlabel("Delivery month")
    ax.set_ylabel("Delta jump (EUR/MWh)")
    ax.set_xticks(np.arange(len(plot)))
    ax.set_xticklabels(plot["year_month"].astype(str), rotation=70, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, output_dir / "boundary_delta_jumps.png")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--adjusted-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = compare_ab(
        baseline_csv=args.baseline_csv,
        adjusted_csv=args.adjusted_csv,
        output_dir=args.output_dir,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

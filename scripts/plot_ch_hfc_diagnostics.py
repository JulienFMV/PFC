"""Generate PNG diagnostics for the local/test CH HFC hourly curve."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfc_shaping.calibration.eex_contract_selection import calibration_buckets  # noqa: E402
from scripts.export_local_test_ch_hourly_csv import (  # noqa: E402
    _eex_peak_mask,
    _latest_eex_prices_by_load_type,
    _parse_timestamp_ch,
)


PRICE = "price_weighted_mean_eur_mwh"


def _load(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame["ts_ch"] = _parse_timestamp_ch(frame["timestamp_ch"], frame.get("utc_offset_ch"))
    frame["year"] = frame["ts_ch"].dt.year.astype(int)
    frame["quarter"] = frame["ts_ch"].dt.quarter.astype(int)
    frame["month"] = frame["ts_ch"].dt.month.astype(int)
    frame["hour"] = frame["ts_ch"].dt.hour.astype(int)
    frame["is_eex_peak"] = _eex_peak_mask(frame["ts_ch"], country="CH").astype(bool)
    return frame


def _add_eex_buckets(frame: pd.DataFrame, forwards_path: Path) -> pd.DataFrame:
    _, by_load = _latest_eex_prices_by_load_type(forwards_path, market="CH")
    base = by_load.get("BASE", {})
    if not base:
        raise ValueError("no CH BASE forwards available for bucket diagnostics")
    buckets, _ = calibration_buckets(frame["ts_ch"], base)
    out = frame.copy()
    out["eex_base_bucket"] = buckets.to_numpy()
    return out


def _eex_residuals(frame: pd.DataFrame, forwards_path: Path) -> pd.DataFrame:
    _, by_load = _latest_eex_prices_by_load_type(forwards_path, market="CH")
    rows: list[dict[str, object]] = []
    for load_type, prices in by_load.items():
        if load_type == "PEAK":
            scoped = frame.loc[frame["is_eex_peak"]].copy()
        else:
            scoped = frame
        if scoped.empty or not prices:
            continue
        buckets, targets = calibration_buckets(scoped["ts_ch"], prices)
        for product, idx in buckets.groupby(buckets, dropna=True).groups.items():
            if product is None or str(product) not in targets:
                continue
            mean = float(scoped.loc[idx, PRICE].mean())
            target = float(targets[str(product)])
            rows.append(
                {
                    "load_type": load_type,
                    "product": str(product),
                    "target_eex_eur_mwh": target,
                    "csv_mean_eur_mwh": mean,
                    "residual_eur_mwh": mean - target,
                    "abs_error_eur_mwh": abs(mean - target),
                    "rows": int(len(idx)),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["load_type", "product"]).reset_index(drop=True)


def _monthly(frame: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        frame.groupby(["year", "month"], as_index=False)
        .agg(
            mean_eur_mwh=(PRICE, "mean"),
            min_eur_mwh=(PRICE, "min"),
            max_eur_mwh=(PRICE, "max"),
            fast_negative_hours=("price_fast_eur_mwh", lambda s: int((s < 0.0).sum())),
            p10_negative_hours=("structural_p10_eur_mwh", lambda s: int((s < 0.0).sum())),
            eex_base_bucket=("eex_base_bucket", lambda s: str(s.mode().iloc[0]) if not s.mode().empty else ""),
        )
        .sort_values(["year", "month"])
    )
    monthly["mom_delta_eur_mwh"] = monthly.groupby("year")["mean_eur_mwh"].diff()
    monthly["month_label"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    return monthly


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_monthly_means(monthly: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for year, group in monthly.groupby("year"):
        ax.plot(group["month"], group["mean_eur_mwh"], marker="o", linewidth=2, label=str(year))
    for x in (3.5, 6.5, 9.5):
        ax.axvline(x, color="#9ca3af", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("CH HFC monthly mean by delivery year")
    ax.set_xlabel("Month")
    ax.set_ylabel("EUR/MWh")
    ax.set_xticks(range(1, 13))
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=5, fontsize=9)
    _save(fig, output / "01_monthly_means_by_year.png")


def _plot_focus_2027_2028(monthly: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, year in zip(axes, (2027, 2028)):
        group = monthly[monthly["year"] == year].copy()
        ax.plot(group["month"], group["mean_eur_mwh"], marker="o", color="#1f77b4", linewidth=2)
        for _, row in group.iterrows():
            ax.annotate(
                row["eex_base_bucket"],
                (row["month"], row["mean_eur_mwh"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color="#374151",
            )
        ax.set_title(f"{year}: monthly means with active EEX BASE bucket")
        ax.set_ylabel("EUR/MWh")
        ax.grid(True, alpha=0.25)
        for x in (3.5, 6.5, 9.5):
            ax.axvline(x, color="#9ca3af", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[-1].set_xlabel("Month")
    axes[-1].set_xticks(range(1, 13))
    _save(fig, output / "02_focus_2027_2028_eex_buckets.png")


def _plot_mom_deltas(monthly: pd.DataFrame, output: Path) -> None:
    years = [y for y in sorted(monthly["year"].unique()) if y >= 2027]
    fig, axes = plt.subplots(len(years), 1, figsize=(12, 2.2 * len(years)), sharex=True)
    if len(years) == 1:
        axes = [axes]
    for ax, year in zip(axes, years):
        group = monthly[monthly["year"] == year]
        colors = np.where(group["mom_delta_eur_mwh"].fillna(0.0) >= 0.0, "#2ca25f", "#de2d26")
        ax.bar(group["month"], group["mom_delta_eur_mwh"], color=colors)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"{year}: month-to-month delta")
        ax.set_ylabel("EUR/MWh")
        ax.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xticks(range(1, 13))
    axes[-1].set_xlabel("Month")
    _save(fig, output / "03_month_to_month_deltas.png")


def _plot_duck_curves(frame: pd.DataFrame, output: Path) -> None:
    for year in (2027, 2028, 2030):
        sub = frame[frame["year"] == year]
        if sub.empty:
            continue
        duck = sub.groupby(["month", "hour"], as_index=False)[PRICE].mean()
        fig, ax = plt.subplots(figsize=(12, 6))
        for month, group in duck.groupby("month"):
            ax.plot(group["hour"], group[PRICE], linewidth=1.6, label=f"M{month:02d}")
        ax.set_title(f"{year}: monthly duck curves - weighted mean")
        ax.set_xlabel("Hour CH")
        ax.set_ylabel("EUR/MWh")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=6, fontsize=8)
        _save(fig, output / f"04_duck_curves_{year}.png")


def _plot_heatmap(frame: pd.DataFrame, output: Path) -> None:
    for year in (2028, 2030):
        sub = frame[frame["year"] == year]
        if sub.empty:
            continue
        heat = sub.pivot_table(index="month", columns="hour", values=PRICE, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(13, 5))
        im = ax.imshow(heat.values, aspect="auto", cmap="viridis")
        ax.set_title(f"{year}: month x hour heatmap - weighted mean")
        ax.set_xlabel("Hour CH")
        ax.set_ylabel("Month")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels(range(0, 24, 2))
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels([f"M{m:02d}" for m in heat.index])
        fig.colorbar(im, ax=ax, label="EUR/MWh")
        _save(fig, output / f"05_heatmap_month_hour_{year}.png")


def _plot_negative_tail(frame: pd.DataFrame, output: Path) -> None:
    neg = (
        frame.groupby(["year", "month", "hour"], as_index=False)
        .agg(
            fast_negative_hours=("price_fast_eur_mwh", lambda s: int((s < 0.0).sum())),
            p10_negative_hours=("structural_p10_eur_mwh", lambda s: int((s < 0.0).sum())),
        )
    )
    for col in ("fast_negative_hours", "p10_negative_hours"):
        pivot = neg.pivot_table(index=["year", "month"], columns="hour", values=col, aggfunc="sum").fillna(0.0)
        fig, ax = plt.subplots(figsize=(13, 6))
        im = ax.imshow(pivot.values, aspect="auto", cmap="Reds")
        ax.set_title(f"Negative tail heatmap - {col}")
        ax.set_xlabel("Hour CH")
        ax.set_ylabel("Year-month")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels(range(0, 24, 2))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{y}-{m:02d}" for y, m in pivot.index], fontsize=7)
        fig.colorbar(im, ax=ax, label="Hours")
        _save(fig, output / f"06_negative_tail_{col}.png")


def _plot_eex_residuals(residuals: pd.DataFrame, output: Path) -> None:
    if residuals.empty:
        return
    plot = residuals.copy()
    plot["label"] = plot["load_type"] + " " + plot["product"].astype(str)
    colors = np.where(plot["load_type"].eq("BASE"), "#2563eb", "#7c3aed")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(np.arange(len(plot)), plot["residual_eur_mwh"], color=colors)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.axhline(0.01, color="#dc2626", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(-0.01, color="#dc2626", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("EEX BASE/PEAK residuals by active calibration product")
    ax.set_ylabel("CSV mean - EEX target (EUR/MWh)")
    ax.set_xticks(np.arange(len(plot)))
    ax.set_xticklabels(plot["label"], rotation=70, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    max_abs = float(plot["abs_error_eur_mwh"].max())
    ax.text(
        0.01,
        0.95,
        f"max abs residual = {max_abs:.6f} EUR/MWh",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d1d5db"},
    )
    _save(fig, output / "07_eex_residuals_by_product.png")


def _plot_boundary_jumps(frame: pd.DataFrame, output: Path, baseline_csv: Path | None = None) -> None:
    current = frame[["ts_ch", PRICE]].copy()
    current["month_key"] = current["ts_ch"].dt.strftime("%Y-%m")
    current["is_boundary"] = current["month_key"].ne(current["month_key"].shift(1))
    current.loc[current.index[0], "is_boundary"] = False
    current["price_jump_eur_mwh"] = current[PRICE].diff()
    current["delta_jump_eur_mwh"] = np.nan

    title_suffix = "price jumps"
    value_col = "price_jump_eur_mwh"
    ylabel = "Boundary price jump (EUR/MWh)"
    if baseline_csv is not None and baseline_csv.exists():
        baseline = _load(baseline_csv)[["ts_ch", PRICE]].rename(columns={PRICE: "baseline_price"})
        merged = current.merge(baseline, on="ts_ch", how="left")
        merged["applied_delta"] = merged[PRICE] - merged["baseline_price"]
        merged["delta_jump_eur_mwh"] = merged["applied_delta"].diff()
        current = merged
        title_suffix = "applied-delta jumps vs no-smoothing baseline"
        value_col = "delta_jump_eur_mwh"
        ylabel = "Boundary applied-delta jump (EUR/MWh)"

    boundaries = current.loc[current["is_boundary"]].copy()
    if boundaries.empty:
        return
    boundaries["label"] = boundaries["ts_ch"].dt.strftime("%Y-%m")
    boundaries["is_year_boundary"] = boundaries["ts_ch"].dt.month.eq(1)
    colors = np.where(boundaries["is_year_boundary"], "#dc2626", "#2563eb")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(np.arange(len(boundaries)), boundaries[value_col], color=colors)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_title(f"Month/year boundary {title_suffix}")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(boundaries)))
    ax.set_xticklabels(boundaries["label"], rotation=70, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    max_abs = float(boundaries[value_col].abs().max())
    ax.text(
        0.01,
        0.95,
        f"max abs boundary jump = {max_abs:.3f} EUR/MWh",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d1d5db"},
    )
    _save(fig, output / "08_boundary_delta_jumps.png")


def _plot_executive_summary(frame: pd.DataFrame, monthly: pd.DataFrame, residuals: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    for year, group in monthly.groupby("year"):
        ax.plot(group["month"], group["mean_eur_mwh"], marker="o", linewidth=1.8, label=str(year))
    ax.set_title("Monthly means")
    ax.set_xlabel("Month")
    ax.set_ylabel("EUR/MWh")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8)

    ax = axes[0, 1]
    sub = frame[frame["year"] == 2030]
    heat = sub.pivot_table(index="month", columns="hour", values=PRICE, aggfunc="mean")
    im = ax.imshow(heat.values, aspect="auto", cmap="viridis")
    ax.set_title("2030 month x hour")
    ax.set_xlabel("Hour CH")
    ax.set_ylabel("Month")
    ax.set_xticks(range(0, 24, 4))
    ax.set_xticklabels(range(0, 24, 4))
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels([f"M{m:02d}" for m in heat.index])
    fig.colorbar(im, ax=ax, label="EUR/MWh")

    ax = axes[1, 0]
    duck = sub.groupby(["month", "hour"], as_index=False)[PRICE].mean()
    for month in (1, 4, 7, 10, 12):
        group = duck[duck["month"] == month]
        ax.plot(group["hour"], group[PRICE], linewidth=1.8, label=f"M{month:02d}")
    ax.set_title("2030 selected duck curves")
    ax.set_xlabel("Hour CH")
    ax.set_ylabel("EUR/MWh")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=5, fontsize=8)

    ax = axes[1, 1]
    if residuals.empty:
        ax.text(0.5, 0.5, "No EEX residuals", ha="center", va="center")
    else:
        grouped = residuals.groupby("load_type")["abs_error_eur_mwh"].max()
        bars = ax.bar(grouped.index, grouped.values, color=["#2563eb", "#7c3aed"][: len(grouped)])
        ax.axhline(0.01, color="#dc2626", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.set_ylim(0.0, max(0.011, float(grouped.max()) * 1.25))
        for bar, value in zip(bars, grouped.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                max(float(value), 0.00015),
                f"{float(value):.6f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title("Max EEX residual by load type")
        ax.set_ylabel("EUR/MWh")
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("CH HFC QA executive diagnostic", fontsize=16)
    fig.tight_layout()
    _save(fig, output / "09_executive_qa_summary.png")


def build_plots(csv_path: Path, forwards_path: Path, output_dir: Path, baseline_csv: Path | None = None) -> list[Path]:
    frame = _add_eex_buckets(_load(csv_path), forwards_path)
    monthly = _monthly(frame)
    residuals = _eex_residuals(frame, forwards_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_dir / "monthly_diagnostics.csv", index=False)
    residuals.to_csv(output_dir / "eex_residual_diagnostics.csv", index=False)
    _plot_monthly_means(monthly, output_dir)
    _plot_focus_2027_2028(monthly, output_dir)
    _plot_mom_deltas(monthly, output_dir)
    _plot_duck_curves(frame, output_dir)
    _plot_heatmap(frame, output_dir)
    _plot_negative_tail(frame, output_dir)
    _plot_eex_residuals(residuals, output_dir)
    _plot_boundary_jumps(frame, output_dir, baseline_csv)
    _plot_executive_summary(frame, monthly, residuals, output_dir)
    return sorted(output_dir.glob("*.png"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="output/ch_hfc_hourly.csv")
    parser.add_argument("--forwards", default="data/eex_forwards_history.parquet")
    parser.add_argument("--output-dir", default="output/hfc_diagnostics")
    parser.add_argument(
        "--baseline-csv",
        default=None,
        help="Optional no-smoothing baseline CSV used to plot applied-delta boundary jumps.",
    )
    args = parser.parse_args(argv)
    baseline = Path(args.baseline_csv) if args.baseline_csv else None
    paths = build_plots(Path(args.csv), Path(args.forwards), Path(args.output_dir), baseline_csv=baseline)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def build_plots(csv_path: Path, forwards_path: Path, output_dir: Path) -> list[Path]:
    frame = _add_eex_buckets(_load(csv_path), forwards_path)
    monthly = _monthly(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_dir / "monthly_diagnostics.csv", index=False)
    _plot_monthly_means(monthly, output_dir)
    _plot_focus_2027_2028(monthly, output_dir)
    _plot_mom_deltas(monthly, output_dir)
    _plot_duck_curves(frame, output_dir)
    _plot_heatmap(frame, output_dir)
    _plot_negative_tail(frame, output_dir)
    return sorted(output_dir.glob("*.png"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="output/ch_hfc_hourly.csv")
    parser.add_argument("--forwards", default="data/eex_forwards_history.parquet")
    parser.add_argument("--output-dir", default="output/hfc_diagnostics")
    args = parser.parse_args(argv)
    paths = build_plots(Path(args.csv), Path(args.forwards), Path(args.output_dir))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

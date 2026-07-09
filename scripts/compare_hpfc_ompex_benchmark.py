"""Read-only HPFC vs OMPEX benchmark diagnostics.

OMPEX is an external advisory benchmark, not a model input and not a target
curve. This script compares an HPFC CSV against an OMPEX XLSX and writes
diagnostics that are explicitly read-only evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.export_local_test_ch_hourly_csv import _parse_timestamp_ch  # noqa: E402


DEFAULT_HPFC_PRICE = "price_weighted_mean_eur_mwh"
ALIGNMENT_SHIFTS: dict[str, pd.Timedelta] = {
    "direct": pd.Timedelta(0),
    "ompex_minus_1h_hourending": -pd.Timedelta(hours=1),
    "ompex_plus_1h": pd.Timedelta(hours=1),
}


def load_hpfc(path: Path, *, price_col: str = DEFAULT_HPFC_PRICE) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=None, engine="python")
    if "timestamp_ch" in frame.columns:
        ts = _parse_timestamp_ch(frame["timestamp_ch"], frame.get("utc_offset_ch"))
    elif "timestamp_local" in frame.columns:
        ts = pd.to_datetime(frame["timestamp_local"], errors="coerce")
    else:
        raise ValueError(f"{path} has no timestamp_ch or timestamp_local column")
    if price_col not in frame.columns:
        if price_col == DEFAULT_HPFC_PRICE and "price_shape" in frame.columns:
            price_col = "price_shape"
        else:
            raise ValueError(f"{path} missing price column {price_col!r}")
    out = frame.copy()
    out["timestamp"] = ts
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    numeric_cols = [
        column
        for column in out.columns
        if column.endswith("_eur_mwh") and column != price_col
    ]
    for column in numeric_cols:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    keep = ["timestamp", price_col] + numeric_cols
    out = out[keep].dropna(subset=["timestamp", price_col]).sort_values("timestamp")
    if out.empty:
        raise ValueError(f"{path} has no usable HPFC rows")
    grouped = out.set_index("timestamp").groupby(level=0).mean(numeric_only=True)
    if isinstance(grouped.index, pd.DatetimeIndex) and grouped.index.tz is not None:
        grouped.index = grouped.index.tz_localize(None)
    return grouped.rename(columns={price_col: "hpfc_eur_mwh"})


def load_ompex(path: Path) -> pd.Series:
    frame = pd.read_excel(path, sheet_name=0)
    if "Date" not in frame.columns or "EUR/MWh" not in frame.columns:
        raise ValueError(f"{path} must contain Date and EUR/MWh columns")
    ts = pd.to_datetime(frame["Date"], dayfirst=True, errors="coerce")
    price = pd.to_numeric(frame["EUR/MWh"], errors="coerce")
    series = pd.Series(price.to_numpy(), index=ts, name="ompex_eur_mwh")
    series = series[~series.index.isna()].dropna().sort_index()
    if series.empty:
        raise ValueError(f"{path} has no usable OMPEX rows")
    return series.groupby(level=0).mean()


def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["year"] = out.index.year.astype(int)
    out["month"] = out.index.month.astype(int)
    out["year_month"] = out.index.to_period("M").astype(str)
    out["hour"] = out.index.hour.astype(int)
    out["weekday"] = out.index.dayofweek.astype(int)
    out["is_weekend"] = out["weekday"].isin([5, 6])
    out["is_peak"] = (~out["is_weekend"]) & out["hour"].between(8, 19)
    out["bucket"] = np.where(out["is_peak"], "peak_weekday_08_19", "offpeak_or_weekend")
    if {"structural_p10_eur_mwh", "structural_p90_eur_mwh"}.issubset(out.columns):
        out["ompex_inside_p10_p90"] = out["ompex_eur_mwh"].between(
            out["structural_p10_eur_mwh"],
            out["structural_p90_eur_mwh"],
        )
    else:
        out["ompex_inside_p10_p90"] = np.nan
    return out


def _metrics(joined: pd.DataFrame) -> dict[str, float | int | str]:
    err = joined["hpfc_eur_mwh"] - joined["ompex_eur_mwh"]
    ramp = _ramp_frame(joined)
    boundary = _boundary_jumps(joined)
    return {
        "n_points": int(len(joined)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
        "median_error": float(err.median()),
        "p05_error": float(err.quantile(0.05)),
        "p95_error": float(err.quantile(0.95)),
        "p95_abs_error": float(err.abs().quantile(0.95)),
        "max_abs_error": float(err.abs().max()),
        "correlation": float(joined["hpfc_eur_mwh"].corr(joined["ompex_eur_mwh"])),
        "window_start": str(joined.index.min()),
        "window_end": str(joined.index.max()),
        "hpfc_mean": float(joined["hpfc_eur_mwh"].mean()),
        "ompex_mean": float(joined["ompex_eur_mwh"].mean()),
        "ompex_inside_p10_p90_rate": float(joined["ompex_inside_p10_p90"].mean())
        if joined["ompex_inside_p10_p90"].notna().any()
        else np.nan,
        "ramp_n_points": int(len(ramp)),
        "ramp_mae": float(ramp["abs_ramp_diff"].mean()) if not ramp.empty else np.nan,
        "ramp_p95_abs_error": float(ramp["abs_ramp_diff"].quantile(0.95)) if not ramp.empty else np.nan,
        "boundary_jump_n_points": int(len(boundary)),
        "boundary_jump_mae": float(boundary["abs_jump_diff"].mean()) if not boundary.empty else np.nan,
        "boundary_jump_p95_abs_error": float(boundary["abs_jump_diff"].quantile(0.95)) if not boundary.empty else np.nan,
    }


def _join_with_shift(hpfc: pd.DataFrame, ompex: pd.Series, shift: pd.Timedelta) -> pd.DataFrame:
    shifted = ompex.copy()
    shifted.index = shifted.index + shift
    shifted = shifted.groupby(level=0).mean()
    joined = hpfc.join(shifted.to_frame(), how="inner").dropna(subset=["hpfc_eur_mwh", "ompex_eur_mwh"])
    if joined.empty:
        raise ValueError("no overlapping timestamps between HPFC and OMPEX")
    joined["diff_hpfc_minus_ompex"] = joined["hpfc_eur_mwh"] - joined["ompex_eur_mwh"]
    joined["abs_diff"] = joined["diff_hpfc_minus_ompex"].abs()
    return add_calendar(joined)


def compare(
    *,
    hpfc_csv: Path,
    ompex_xlsx: Path,
    output_dir: Path,
    price_col: str = DEFAULT_HPFC_PRICE,
    alignment: str = "auto",
    write_plots: bool = True,
) -> dict[str, object]:
    hpfc = load_hpfc(hpfc_csv, price_col=price_col)
    ompex = load_ompex(ompex_xlsx)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    for name, shift in ALIGNMENT_SHIFTS.items():
        joined = _join_with_shift(hpfc, ompex, shift)
        candidates[name] = joined
        metric = _metrics(joined)
        metric["alignment"] = name
        rows.append(metric)
    sensitivity = pd.DataFrame(rows).sort_values(["mae", "rmse"], ascending=[True, True])
    sensitivity.to_csv(output_dir / "alignment_sensitivity.csv", index=False)

    if alignment == "auto":
        selected = str(sensitivity.iloc[0]["alignment"])
    elif alignment in candidates:
        selected = alignment
    else:
        raise ValueError(f"unknown alignment {alignment!r}; expected auto or one of {sorted(candidates)}")
    joined = candidates[selected]
    summary = {
        **_metrics(joined),
        "alignment": selected,
        "hpfc_file": str(hpfc_csv),
        "ompex_file": str(ompex_xlsx),
        "benchmark_policy": "advisory",
        "read_only": True,
        "promotion_gate": False,
        "production_approved": False,
        "ompex_used_in_model": False,
        "ompex_used_in_selection": False,
        "ompex_used_in_backtest": False,
        "ompex_quality_caveat": (
            "OMPEX is an external imperfect benchmark, not ground truth, "
            "not an optimizer target, and not a production promotion authority."
        ),
    }

    joined.reset_index(names="timestamp").to_csv(output_dir / "aligned_hourly_comparison.csv", index=False)
    _aggregate(joined, ["year"]).to_csv(output_dir / "by_year.csv", index=False)
    _aggregate(joined, ["year_month"]).to_csv(output_dir / "by_year_month.csv", index=False)
    _aggregate(joined, ["hour"]).to_csv(output_dir / "by_hour.csv", index=False)
    _aggregate(joined, ["bucket"]).to_csv(output_dir / "by_bucket.csv", index=False)
    _aggregate(joined, ["is_weekend"]).to_csv(output_dir / "by_weekend.csv", index=False)
    _ramp_metrics(joined).to_csv(output_dir / "ramp_metrics.csv", index=False)
    _boundary_jumps(joined).to_csv(output_dir / "boundary_jumps.csv", index=False)
    _month_hour_matrix(joined, value="diff_hpfc_minus_ompex", agg="mean").to_csv(
        output_dir / "month_hour_bias_matrix.csv"
    )
    _month_hour_matrix(joined, value="abs_diff", agg="mean").to_csv(output_dir / "month_hour_mae_matrix.csv")
    joined.sort_values("abs_diff", ascending=False).head(100).reset_index(names="timestamp").to_csv(
        output_dir / "top_abs_differences.csv",
        index=False,
    )
    (output_dir / "benchmark_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    if write_plots:
        _write_plots(joined, output_dir)
    return summary


def _aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(keys, dropna=False)
        .agg(
            n=("diff_hpfc_minus_ompex", "size"),
            hpfc_mean=("hpfc_eur_mwh", "mean"),
            ompex_mean=("ompex_eur_mwh", "mean"),
            bias=("diff_hpfc_minus_ompex", "mean"),
            mae=("abs_diff", "mean"),
            p95_abs_error=("abs_diff", lambda s: s.quantile(0.95)),
            corr=(
                "hpfc_eur_mwh",
                lambda s: s.corr(frame.loc[s.index, "ompex_eur_mwh"]) if len(s) > 1 else np.nan,
            ),
            inside_p10_p90=("ompex_inside_p10_p90", "mean"),
        )
        .reset_index()
    )


def _ramp_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_index()
    out = ordered[["hpfc_eur_mwh", "ompex_eur_mwh"]].diff().dropna()
    if out.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "hour",
                "hpfc_ramp_eur_mwh",
                "ompex_ramp_eur_mwh",
                "ramp_diff_hpfc_minus_ompex",
                "abs_ramp_diff",
            ]
        )
    out = out.rename(columns={"hpfc_eur_mwh": "hpfc_ramp_eur_mwh", "ompex_eur_mwh": "ompex_ramp_eur_mwh"})
    out["ramp_diff_hpfc_minus_ompex"] = out["hpfc_ramp_eur_mwh"] - out["ompex_ramp_eur_mwh"]
    out["abs_ramp_diff"] = out["ramp_diff_hpfc_minus_ompex"].abs()
    out["hour"] = out.index.hour.astype(int)
    return out.reset_index(names="timestamp")


def _ramp_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    ramp = _ramp_frame(frame)
    if ramp.empty:
        return pd.DataFrame(
            [
                {
                    "bucket": "all",
                    "n": 0,
                    "hpfc_ramp_mean": np.nan,
                    "ompex_ramp_mean": np.nan,
                    "ramp_bias": np.nan,
                    "ramp_mae": np.nan,
                    "ramp_p95_abs_error": np.nan,
                    "ramp_max_abs_error": np.nan,
                }
            ]
        )
    rows = [
        {
            "bucket": "all",
            "n": int(len(ramp)),
            "hpfc_ramp_mean": float(ramp["hpfc_ramp_eur_mwh"].mean()),
            "ompex_ramp_mean": float(ramp["ompex_ramp_eur_mwh"].mean()),
            "ramp_bias": float(ramp["ramp_diff_hpfc_minus_ompex"].mean()),
            "ramp_mae": float(ramp["abs_ramp_diff"].mean()),
            "ramp_p95_abs_error": float(ramp["abs_ramp_diff"].quantile(0.95)),
            "ramp_max_abs_error": float(ramp["abs_ramp_diff"].max()),
        }
    ]
    by_hour = (
        ramp.groupby("hour", dropna=False)
        .agg(
            n=("abs_ramp_diff", "size"),
            hpfc_ramp_mean=("hpfc_ramp_eur_mwh", "mean"),
            ompex_ramp_mean=("ompex_ramp_eur_mwh", "mean"),
            ramp_bias=("ramp_diff_hpfc_minus_ompex", "mean"),
            ramp_mae=("abs_ramp_diff", "mean"),
            ramp_p95_abs_error=("abs_ramp_diff", lambda s: s.quantile(0.95)),
            ramp_max_abs_error=("abs_ramp_diff", "max"),
        )
        .reset_index()
    )
    by_hour.insert(0, "bucket", "hour_" + by_hour["hour"].astype(str).str.zfill(2))
    by_hour = by_hour.drop(columns=["hour"])
    return pd.concat([pd.DataFrame(rows), by_hour], ignore_index=True)


def _boundary_jumps(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_index()
    if len(ordered) < 2:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "previous_timestamp",
                "hpfc_jump_eur_mwh",
                "ompex_jump_eur_mwh",
                "jump_diff_hpfc_minus_ompex",
                "abs_jump_diff",
            ]
        )
    prev = ordered.shift(1)
    periods = pd.Series(ordered.index.to_period("M"), index=ordered.index)
    previous_timestamps = pd.Series(ordered.index, index=ordered.index).shift(1)
    boundary_mask = (periods != periods.shift(1)) & prev["hpfc_eur_mwh"].notna()
    boundary = ordered.loc[boundary_mask, ["hpfc_eur_mwh", "ompex_eur_mwh"]].copy()
    if boundary.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "previous_timestamp",
                "hpfc_jump_eur_mwh",
                "ompex_jump_eur_mwh",
                "jump_diff_hpfc_minus_ompex",
                "abs_jump_diff",
            ]
        )
    previous = prev.loc[boundary.index, ["hpfc_eur_mwh", "ompex_eur_mwh"]]
    out = pd.DataFrame(index=boundary.index)
    out["previous_timestamp"] = previous_timestamps.loc[boundary.index].to_numpy()
    out["hpfc_jump_eur_mwh"] = boundary["hpfc_eur_mwh"] - previous["hpfc_eur_mwh"]
    out["ompex_jump_eur_mwh"] = boundary["ompex_eur_mwh"] - previous["ompex_eur_mwh"]
    out["jump_diff_hpfc_minus_ompex"] = out["hpfc_jump_eur_mwh"] - out["ompex_jump_eur_mwh"]
    out["abs_jump_diff"] = out["jump_diff_hpfc_minus_ompex"].abs()
    return out.reset_index(names="timestamp")


def _month_hour_matrix(frame: pd.DataFrame, *, value: str, agg: str) -> pd.DataFrame:
    pivot = frame.pivot_table(index="year_month", columns="hour", values=value, aggfunc=agg)
    pivot = pivot.reindex(columns=list(range(24)))
    pivot.columns = [f"hour_{int(column):02d}" for column in pivot.columns]
    return pivot


def _write_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    monthly = frame[["hpfc_eur_mwh", "ompex_eur_mwh"]].resample("MS").mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly.index, monthly["hpfc_eur_mwh"], label="HPFC", linewidth=2)
    ax.plot(monthly.index, monthly["ompex_eur_mwh"], label="OMPEX", linewidth=2)
    ax.set_title("Monthly mean: HPFC vs OMPEX")
    ax.set_ylabel("EUR/MWh")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "01_monthly_mean_hpfc_vs_ompex.png", dpi=140)
    plt.close(fig)

    by_hour = _aggregate(frame, ["hour"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(by_hour["hour"], by_hour["bias"], marker="o", label="Bias")
    ax.plot(by_hour["hour"], by_hour["mae"], marker="o", label="MAE")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Error by hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("EUR/MWh")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "02_error_by_hour.png", dpi=140)
    plt.close(fig)

    matrix = _month_hour_matrix(frame, value="diff_hpfc_minus_ompex", agg="mean")
    fig, ax = plt.subplots(figsize=(12, max(5, min(14, 0.18 * len(matrix)))))
    im = ax.imshow(matrix.to_numpy(dtype="float64"), aspect="auto", cmap="coolwarm")
    ax.set_title("Mean error by month and hour: HPFC minus OMPEX")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Month")
    ax.set_xticks(range(24))
    ax.set_xticklabels([str(hour) for hour in range(24)])
    step = max(1, len(matrix) // 18)
    y_positions = list(range(0, len(matrix), step))
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(matrix.index[pos]) for pos in y_positions])
    fig.colorbar(im, ax=ax, label="EUR/MWh")
    fig.tight_layout()
    fig.savefig(output_dir / "03_month_hour_bias_heatmap.png", dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpfc-csv", type=Path, required=True)
    parser.add_argument("--ompex-xlsx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--price-col", default=DEFAULT_HPFC_PRICE)
    parser.add_argument("--alignment", default="auto", choices=["auto", *ALIGNMENT_SHIFTS.keys()])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    summary = compare(
        hpfc_csv=args.hpfc_csv,
        ompex_xlsx=args.ompex_xlsx,
        output_dir=args.output_dir,
        price_col=args.price_col,
        alignment=args.alignment,
        write_plots=not args.no_plots,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

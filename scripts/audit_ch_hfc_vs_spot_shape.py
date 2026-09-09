"""Compare local/test CH HFC shape against historical CH spot shape."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.export_local_test_ch_hourly_csv import _parse_timestamp_ch  # noqa: E402

LOCAL_TZ = "Europe/Zurich"
HFC_PRICE = "price_weighted_mean_eur_mwh"
SPOT_PRICE = "price_eur_mwh"


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _add_calendar(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out[ts_col].dt.year.astype(int)
    out["month"] = out[ts_col].dt.month.astype(int)
    out["hour"] = out[ts_col].dt.hour.astype(int)
    out["weekday"] = out[ts_col].dt.weekday.astype(int)
    out["season"] = out["month"].map(_season)
    out["is_peak"] = (out["weekday"] < 5) & out["hour"].between(8, 19)
    out["is_weekend"] = out["weekday"] >= 5
    return out


def _load_hfc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp_ch", HFC_PRICE, "structural_p10_eur_mwh", "price_fast_eur_mwh"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"HFC CSV missing required columns: {missing}")
    df["ts"] = _parse_timestamp_ch(df["timestamp_ch"], df.get("utc_offset_ch"))
    return _add_calendar(df, "ts")


def _load_spot(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if SPOT_PRICE not in df.columns:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            raise ValueError(f"spot source {path} has no numeric price column")
        df = df.rename(columns={numeric[0]: SPOT_PRICE})
    if isinstance(df.index, pd.DatetimeIndex):
        ts = pd.to_datetime(df.index, utc=True).tz_convert(LOCAL_TZ)
    elif "timestamp_utc" in df.columns:
        ts = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(LOCAL_TZ)
    else:
        raise ValueError(f"spot source {path} has no timestamp index/column")
    out = pd.DataFrame({"ts": ts, SPOT_PRICE: pd.to_numeric(df[SPOT_PRICE], errors="coerce").to_numpy()})
    out = out.dropna(subset=[SPOT_PRICE]).sort_values("ts")
    hourly = out.set_index("ts").resample("h")[SPOT_PRICE].mean().dropna().reset_index()
    return _add_calendar(hourly, "ts")


def _shape_metrics(df: pd.DataFrame, price_col: str, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, g in df.groupby("year"):
        peak = float(g.loc[g["is_peak"], price_col].mean())
        offpeak = float(g.loc[~g["is_peak"], price_col].mean())
        midday = float(g.loc[g["hour"].between(11, 15), price_col].mean())
        evening = float(g.loc[g["hour"].between(17, 21), price_col].mean())
        morning = float(g.loc[g["hour"].between(7, 9), price_col].mean())
        night = float(g.loc[g["hour"].between(0, 5), price_col].mean())
        weekday = float(g.loc[~g["is_weekend"], price_col].mean())
        weekend = float(g.loc[g["is_weekend"], price_col].mean())
        season = g.groupby("season")[price_col].mean()
        month = g.groupby("month")[price_col].mean()
        rows.append(
            {
                "source": label,
                "year": int(year),
                "mean_eur_mwh": float(g[price_col].mean()),
                "min_eur_mwh": float(g[price_col].min()),
                "negative_hours": int((g[price_col] < 0.0).sum()),
                "negative_share_pct": float((g[price_col] < 0.0).mean() * 100.0),
                "peak_offpeak_spread_eur_mwh": peak - offpeak,
                "evening_midday_spread_eur_mwh": evening - midday,
                "morning_night_spread_eur_mwh": morning - night,
                "weekend_weekday_spread_eur_mwh": weekend - weekday,
                "winter_summer_spread_eur_mwh": float(season.get("winter", np.nan) - season.get("summer", np.nan)),
                "spring_autumn_spread_eur_mwh": float(season.get("spring", np.nan) - season.get("autumn", np.nan)),
                "jan_oct_spread_eur_mwh": float(month.get(1, np.nan) - month.get(10, np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _normalized_profile(df: pd.DataFrame, price_col: str) -> pd.Series:
    cell = df.groupby(["month", "hour"])[price_col].mean()
    month_mean = df.groupby("month")[price_col].mean()
    full_index = pd.MultiIndex.from_product([range(1, 13), range(24)], names=["month", "hour"])
    profile = cell.reindex(full_index)
    centered = []
    for month, hour in full_index:
        value = profile.loc[(month, hour)]
        if pd.isna(value) or month not in month_mean.index:
            centered.append(np.nan)
        else:
            centered.append(float(value) - float(month_mean.loc[month]))
    return pd.Series(centered, index=full_index, dtype=float).dropna()


def _profile_comparison(hfc: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    spot_profile = _normalized_profile(spot, SPOT_PRICE)
    rows: list[dict[str, object]] = []
    for year, hfc_year in hfc.groupby("year"):
        hfc_profile = _normalized_profile(hfc_year, HFC_PRICE)
        aligned = pd.concat([hfc_profile.rename("hfc"), spot_profile.rename("spot")], axis=1).dropna()
        if aligned.empty:
            continue
        corr = float(aligned["hfc"].corr(aligned["spot"]))
        mae = float((aligned["hfc"] - aligned["spot"]).abs().mean())
        rows.append(
            {
                "year": int(year),
                "month_hour_shape_corr_vs_spot": corr,
                "month_hour_shape_mae_vs_spot_eur_mwh": mae,
                "cells": len(aligned),
            }
        )
    return pd.DataFrame(rows)


def _negative_diagnostics(hfc: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in ["price_fast_eur_mwh", "structural_p10_eur_mwh", HFC_PRICE]:
        g = hfc[hfc[column] < 0.0]
        rows.append(
            {
                "series": column,
                "negative_hours": int(len(g)),
                "min_eur_mwh": float(hfc[column].min()),
                "spring_summer_midday_share_pct": (
                    float(
                        (
                            g["month"].isin([4, 5, 6, 7, 8, 9])
                            & g["hour"].between(10, 15)
                        ).mean()
                        * 100.0
                    )
                    if len(g)
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def audit(hfc_csv: Path, spot_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    hfc = _load_hfc(hfc_csv)
    spot = _load_spot(spot_path)
    spot_metrics = _shape_metrics(spot, SPOT_PRICE, "spot_actual")
    hfc_metrics = _shape_metrics(hfc, HFC_PRICE, "hfc_weighted")
    metrics = pd.concat([spot_metrics, hfc_metrics], ignore_index=True)
    profile = _profile_comparison(hfc, spot)
    negatives = _negative_diagnostics(hfc)

    hfc_latest = hfc_metrics[hfc_metrics["year"] == hfc_metrics["year"].max()].iloc[0]
    score = 0.0
    score += 1.5 if hfc_latest["peak_offpeak_spread_eur_mwh"] > 0.0 else 0.0
    score += 1.5 if hfc_latest["evening_midday_spread_eur_mwh"] >= 12.0 else 0.75
    score += 1.0 if hfc_latest["weekend_weekday_spread_eur_mwh"] <= -3.0 else 0.25
    score += 1.5 if hfc_latest["winter_summer_spread_eur_mwh"] > 0.0 else 0.0
    score += 1.0 if hfc_latest["jan_oct_spread_eur_mwh"] > 0.0 else 0.0
    corr_latest = float(profile.loc[profile["year"] == hfc_latest["year"], "month_hour_shape_corr_vs_spot"].iloc[0])
    score += 1.5 if corr_latest >= 0.45 else 0.75 if corr_latest >= 0.25 else 0.0
    fast_neg = int(negatives.loc[negatives["series"] == "price_fast_eur_mwh", "negative_hours"].iloc[0])
    weighted_neg = int(negatives.loc[negatives["series"] == HFC_PRICE, "negative_hours"].iloc[0])
    score += 1.0 if fast_neg > 0 and weighted_neg == 0 else 0.25
    summary = {
        "score_10": score,
        "spot_start": str(spot["ts"].min()),
        "spot_end": str(spot["ts"].max()),
        "spot_rows": float(len(spot)),
        "hfc_start": str(hfc["ts"].min()),
        "hfc_end": str(hfc["ts"].max()),
        "hfc_rows": float(len(hfc)),
        "latest_hfc_year": float(hfc_latest["year"]),
        "latest_hfc_peak_offpeak_spread_eur_mwh": float(hfc_latest["peak_offpeak_spread_eur_mwh"]),
        "latest_hfc_evening_midday_spread_eur_mwh": float(hfc_latest["evening_midday_spread_eur_mwh"]),
        "latest_hfc_winter_summer_spread_eur_mwh": float(hfc_latest["winter_summer_spread_eur_mwh"]),
        "latest_hfc_jan_oct_spread_eur_mwh": float(hfc_latest["jan_oct_spread_eur_mwh"]),
        "latest_hfc_shape_corr_vs_spot": corr_latest,
        "fast_negative_hours": float(fast_neg),
        "weighted_negative_hours": float(weighted_neg),
    }
    return metrics, profile, negatives, spot.tail(1), summary


def _md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "|" + "|".join("---" for _ in frame.columns) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    hfc_csv: Path,
    spot_path: Path,
    metrics: pd.DataFrame,
    profile: pd.DataFrame,
    negatives: pd.DataFrame,
    summary: dict[str, float],
) -> None:
    lines = [
        "# CH HFC vs Spot Shape Audit",
        "",
        f"* HFC CSV: `{hfc_csv}`",
        f"* spot source: `{spot_path}`",
        f"* score: `{summary['score_10']:.2f}/10`",
        "* scope: `local/test quality audit`",
        "* production approval: `NO`",
        "",
        "## Summary",
        "",
        _md_table(pd.DataFrame([summary])),
        "",
        "## Annual Metrics",
        "",
        _md_table(metrics),
        "",
        "## Normalized Month-Hour Profile vs Spot",
        "",
        _md_table(profile),
        "",
        "## Negative-Price Diagnostics",
        "",
        _md_table(negatives),
        "",
        "## Interpretation",
        "",
        "* Shape correlation is computed on month-hour profiles de-meaned by month, not on outright price levels.",
        "* Historical spot is used as a plausibility anchor; EEX forwards remain the level anchor for the HFC.",
        "* A positive peak/offpeak spread, positive winter/summer spread, positive January/October spread and evening-over-midday duck spread are expected for a credible CH LT HFC.",
        "* Negative prices are expected mainly in the lower structural tail, not necessarily in the weighted FMV curve.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--spot", default="data/epex_hourly.parquet")
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    metrics, profile, negatives, _, summary = audit(Path(args.csv), Path(args.spot))
    write_report(
        Path(args.report),
        hfc_csv=Path(args.csv),
        spot_path=Path(args.spot),
        metrics=metrics,
        profile=profile,
        negatives=negatives,
        summary=summary,
    )
    print(f"[hfc-vs-spot] score={summary['score_10']:.2f}/10")
    print(f"[hfc-vs-spot] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

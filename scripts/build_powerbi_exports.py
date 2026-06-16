"""Build lightweight CSV sidecars for the Power BI CH HFC dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_ch_hfc_vs_spot_shape import audit as audit_vs_spot  # noqa: E402
from scripts.audit_ch_pfc_hourly_shape import audit as audit_shape  # noqa: E402
from scripts.export_local_test_ch_hourly_csv import _eex_peak_mask, _parse_timestamp_ch  # noqa: E402


PRICE_COLUMNS = [
    "price_slow_eur_mwh",
    "price_central_eur_mwh",
    "price_fast_eur_mwh",
    "price_weighted_mean_eur_mwh",
    "structural_p10_eur_mwh",
    "structural_p50_eur_mwh",
    "structural_p90_eur_mwh",
    "structural_width_eur_mwh",
]

DISPLAY_PRICE_COLUMNS = [
    "price_weighted_mean_eur_mwh",
    "price_central_eur_mwh",
    "price_slow_eur_mwh",
    "price_fast_eur_mwh",
    "structural_p10_eur_mwh",
    "structural_p50_eur_mwh",
    "structural_p90_eur_mwh",
    "structural_width_eur_mwh",
]


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def load_hourly(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame["ts_ch"] = _parse_timestamp_ch(frame["timestamp_ch"], frame.get("utc_offset_ch"))
    frame["timestamp_ch_iso"] = frame["ts_ch"].dt.strftime("%Y-%m-%d %H:%M:%S")
    frame["timestamp_ch_text"] = frame["ts_ch"].dt.strftime("%d.%m.%Y %H:%M")
    frame["timestamp_utc_text"] = pd.to_datetime(frame["timestamp_utc"], dayfirst=True).dt.strftime("%d.%m.%Y %H:%M")
    frame["date_ch"] = frame["ts_ch"].dt.strftime("%Y-%m-%d")
    frame["year"] = frame["ts_ch"].dt.year.astype(int)
    frame["quarter"] = frame["ts_ch"].dt.quarter.astype(int)
    frame["month"] = frame["ts_ch"].dt.month.astype(int)
    frame["month_code"] = frame["ts_ch"].dt.strftime("M%m")
    frame["month_label"] = frame["ts_ch"].dt.strftime("%Y-%m")
    frame["hour"] = frame["ts_ch"].dt.hour.astype(int)
    frame["hour_code"] = frame["ts_ch"].dt.strftime("H%H")
    frame["weekday"] = frame["ts_ch"].dt.weekday.astype(int)
    frame["season"] = frame["month"].map(_season)
    frame["is_weekend"] = frame["weekday"] >= 5
    frame["is_eex_peak"] = _eex_peak_mask(frame["ts_ch"], country="CH").astype(bool)
    frame["is_weighted_negative"] = frame["price_weighted_mean_eur_mwh"] < 0.0
    frame["is_fast_negative"] = frame["price_fast_eur_mwh"] < 0.0
    frame["is_p10_negative"] = frame["structural_p10_eur_mwh"] < 0.0
    return frame


def build_exports(csv_path: Path, forwards_path: Path, spot_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    hourly = load_hourly(csv_path)

    hourly_cols = [
        "timestamp_ch_text",
        "timestamp_ch_iso",
        "timestamp_ch",
        "utc_offset_ch",
        "timestamp_utc",
        "timestamp_utc_text",
        "date_ch",
        "year",
        "quarter",
        "month",
        "month_code",
        "month_label",
        "hour",
        "hour_code",
        "weekday",
        "season",
        "is_weekend",
        "is_eex_peak",
        "is_weighted_negative",
        "is_fast_negative",
        "is_p10_negative",
        *PRICE_COLUMNS,
    ]
    hourly_export = hourly[hourly_cols].copy()
    hourly_export[DISPLAY_PRICE_COLUMNS] = hourly_export[DISPLAY_PRICE_COLUMNS].round(6)
    hourly_export.to_csv(output_dir / "hfc_hourly_powerbi.csv", index=False)

    period = hourly.groupby(["year", "quarter", "month"], as_index=False)[PRICE_COLUMNS].mean().round(6)
    period.to_csv(output_dir / "period_means.csv", index=False)

    duck_month = hourly.groupby(["year", "month", "hour"], as_index=False)[PRICE_COLUMNS].mean().round(6)
    duck_month["month_code"] = "M" + duck_month["month"].astype(str).str.zfill(2)
    duck_month["hour_code"] = "H" + duck_month["hour"].astype(str).str.zfill(2)
    duck_month.to_csv(output_dir / "duck_month_hour.csv", index=False)

    duck_month_long = duck_month.melt(
        id_vars=["year", "month", "month_code", "hour", "hour_code"],
        value_vars=[
            "price_weighted_mean_eur_mwh",
            "price_central_eur_mwh",
            "price_fast_eur_mwh",
            "structural_p10_eur_mwh",
            "structural_p90_eur_mwh",
        ],
        var_name="series",
        value_name="price_eur_mwh",
    )
    duck_month_long["series_label"] = duck_month_long["series"].map(
        {
            "price_weighted_mean_eur_mwh": "Weighted mean",
            "price_central_eur_mwh": "Central",
            "price_fast_eur_mwh": "Fast",
            "structural_p10_eur_mwh": "P10",
            "structural_p90_eur_mwh": "P90",
        }
    )
    duck_month_long.to_csv(output_dir / "duck_month_hour_long.csv", index=False)

    duck_season = hourly.groupby(["year", "season", "hour"], as_index=False)[PRICE_COLUMNS].mean().round(6)
    duck_season["hour_code"] = "H" + duck_season["hour"].astype(str).str.zfill(2)
    duck_season.to_csv(output_dir / "duck_season_hour.csv", index=False)

    duck_season_long = duck_season.melt(
        id_vars=["year", "season", "hour", "hour_code"],
        value_vars=["price_weighted_mean_eur_mwh", "price_fast_eur_mwh", "structural_p10_eur_mwh", "structural_p90_eur_mwh"],
        var_name="series",
        value_name="price_eur_mwh",
    )
    duck_season_long["series_label"] = duck_season_long["series"].map(
        {
            "price_weighted_mean_eur_mwh": "Weighted mean",
            "price_fast_eur_mwh": "Fast",
            "structural_p10_eur_mwh": "P10",
            "structural_p90_eur_mwh": "P90",
        }
    )
    duck_season_long.to_csv(output_dir / "duck_season_hour_long.csv", index=False)

    heatmap = hourly.pivot_table(
        index=["year", "month"],
        columns="hour",
        values="price_weighted_mean_eur_mwh",
        aggfunc="mean",
    ).reset_index()
    heatmap.columns = [str(column) for column in heatmap.columns]
    heatmap.round(6).to_csv(output_dir / "heatmap_month_hour.csv", index=False)

    negative = (
        hourly.groupby(["year", "month", "hour"], as_index=False)
        .agg(
            rows=("timestamp_ch", "size"),
            weighted_negative_hours=("is_weighted_negative", "sum"),
            fast_negative_hours=("is_fast_negative", "sum"),
            p10_negative_hours=("is_p10_negative", "sum"),
            min_weighted_eur_mwh=("price_weighted_mean_eur_mwh", "min"),
            min_fast_eur_mwh=("price_fast_eur_mwh", "min"),
            min_p10_eur_mwh=("structural_p10_eur_mwh", "min"),
        )
        .round(6)
    )
    negative["month_code"] = "M" + negative["month"].astype(str).str.zfill(2)
    negative["hour_code"] = "H" + negative["hour"].astype(str).str.zfill(2)
    negative.to_csv(output_dir / "negative_low_hours.csv", index=False)

    annual, residuals, shape_metrics = audit_shape(csv_path, forwards_path)
    derived_annual = []
    for year, group in hourly.groupby("year", sort=True):
        winter = group.loc[group["season"] == "Winter", "price_weighted_mean_eur_mwh"].mean()
        summer = group.loc[group["season"] == "Summer", "price_weighted_mean_eur_mwh"].mean()
        peak = group.loc[group["is_eex_peak"], "price_weighted_mean_eur_mwh"].mean()
        offpeak = group.loc[~group["is_eex_peak"], "price_weighted_mean_eur_mwh"].mean()
        derived_annual.append(
            {
                "year": int(year),
                "winter_summer_spread_eur_mwh": float(winter - summer),
                "peak_minus_offpeak_eur_mwh": float(peak - offpeak),
                "negative_hours_weighted": int((group["price_weighted_mean_eur_mwh"] < 0.0).sum()),
                "negative_hours_fast": int((group["price_fast_eur_mwh"] < 0.0).sum()),
                "negative_hours_p10": int((group["structural_p10_eur_mwh"] < 0.0).sum()),
                "min_weighted_eur_mwh": float(group["price_weighted_mean_eur_mwh"].min()),
                "min_fast_eur_mwh": float(group["price_fast_eur_mwh"].min()),
                "min_p10_eur_mwh": float(group["structural_p10_eur_mwh"].min()),
            }
        )
    annual = annual.merge(pd.DataFrame(derived_annual), on="year", how="left")
    annual_cols = [
        "year",
        "mean_eur_mwh",
        "winter_summer_spread_eur_mwh",
        "evening_minus_midday_eur_mwh",
        "peak_minus_offpeak_eur_mwh",
        "weekend_minus_weekday_eur_mwh",
        "negative_hours_weighted",
        "negative_hours_fast",
        "negative_hours_p10",
        "structural_width_mean_eur_mwh",
        "structural_width_p95_eur_mwh",
        "min_weighted_eur_mwh",
        "min_fast_eur_mwh",
        "min_p10_eur_mwh",
    ]
    annual[annual_cols].round(6).to_csv(output_dir / "annual_shape.csv", index=False)
    residuals.to_csv(output_dir / "eex_residuals.csv", index=False)

    _, _, _, _, spot_summary = audit_vs_spot(csv_path, spot_path)
    def metric_value(value: object) -> str:
        if isinstance(value, (int, float)):
            if abs(value) < 1e-4:
                return f"{value:.6f}"
            if abs(value - round(value)) < 1e-9:
                return f"{value:.0f}"
            return f"{value:.2f}"
        return str(value)

    summary_rows = [
        {"metric": "source_csv", "value": str(csv_path)},
        {"metric": "shape_score_10", "value": shape_metrics["score_10"]},
        {"metric": "hfc_vs_spot_score_10", "value": spot_summary["score_10"]},
        {"metric": "max_eex_base_error_eur_mwh", "value": shape_metrics["max_eex_base_error_eur_mwh"]},
        {"metric": "max_eex_peak_error_eur_mwh", "value": shape_metrics["max_eex_peak_error_eur_mwh"]},
        {"metric": "weighted_negative_hours", "value": shape_metrics["weighted_negative_hours"]},
        {"metric": "p10_negative_hours", "value": shape_metrics["p10_negative_hours"]},
        {"metric": "min_price_eur_mwh", "value": shape_metrics["min_price_eur_mwh"]},
        {
            "metric": "latest_hfc_peak_offpeak_spread_eur_mwh",
            "value": spot_summary["latest_hfc_peak_offpeak_spread_eur_mwh"],
        },
        {
            "metric": "latest_hfc_evening_midday_spread_eur_mwh",
            "value": spot_summary["latest_hfc_evening_midday_spread_eur_mwh"],
        },
        {
            "metric": "latest_hfc_winter_summer_spread_eur_mwh",
            "value": spot_summary["latest_hfc_winter_summer_spread_eur_mwh"],
        },
        {"metric": "latest_hfc_shape_corr_vs_spot", "value": spot_summary["latest_hfc_shape_corr_vs_spot"]},
    ]
    summary = pd.DataFrame(
        [
            {"metric": row["metric"], "value": metric_value(row["value"])}
            for row in summary_rows
        ]
    )
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="output/ch_hfc_hourly.csv")
    parser.add_argument("--forwards", default="data/eex_forwards_history.parquet")
    parser.add_argument("--spot", default="data/epex_hourly.parquet")
    parser.add_argument("--output-dir", default="powerbi/data")
    args = parser.parse_args(argv)
    build_exports(Path(args.csv), Path(args.forwards), Path(args.spot), Path(args.output_dir))
    print(f"[powerbi] exports -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

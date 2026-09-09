"""Generate a compact PBIR report layout for the PFC_QA Power BI project.

The script writes native Power BI report pages into
``powerbi/PFC_QA.Report/definition/pages``.  It intentionally keeps visuals
simple: cards, tables, line charts, column charts and a negative-tail matrix.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


VISUAL_SCHEMA = "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/visualContainer/2.9.0/schema.json"
PAGE_SCHEMA = "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/page/2.1.0/schema.json"
PAGES_SCHEMA = "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/pagesMetadata/1.1.0/schema.json"


def src(table: str) -> dict:
    return {"SourceRef": {"Entity": table}}


def col(table: str, column: str) -> dict:
    return {
        "field": {"Column": {"Expression": src(table), "Property": column}},
        "queryRef": f"{table}.{column}",
        "nativeQueryRef": column,
    }


def meas(table: str, name: str) -> dict:
    return {
        "field": {"Measure": {"Expression": src(table), "Property": name}},
        "queryRef": f"{table}.{name}",
        "nativeQueryRef": name,
    }


def literal(value: str) -> dict:
    return {"expr": {"Literal": {"Value": value}}}


def visual(
    name: str,
    visual_type: str,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    roles: dict[str, list[dict]] | None = None,
) -> dict:
    out = {
        "$schema": VISUAL_SCHEMA,
        "name": name,
        "position": {"x": x, "y": y, "z": 0, "width": w, "height": h},
        "visual": {
            "visualType": visual_type,
            "visualContainerObjects": {
                "title": [
                    {
                        "properties": {
                            "show": literal("true"),
                            "text": literal("'" + title.replace("'", "''") + "'"),
                            "fontSize": literal("11D"),
                            "bold": literal("true"),
                        }
                    }
                ]
            },
        },
    }
    if roles:
        out["visual"]["query"] = {"queryState": {k: {"projections": v} for k, v in roles.items()}}
    return out


def card(name: str, x: int, y: int, title: str, field: dict) -> dict:
    return visual(name, "cardVisual", x, y, 190, 95, title, {"Data": [field]})


def slicer(name: str, x: int, y: int, title: str, field: dict) -> dict:
    return visual(name, "slicer", x, y, 170, 80, title, {"Values": [field]})


def table(name: str, x: int, y: int, w: int, h: int, title: str, fields: list[dict]) -> dict:
    return visual(name, "tableEx", x, y, w, h, title, {"Values": fields})


def line(name: str, x: int, y: int, w: int, h: int, title: str, category: dict, values: list[dict], series: dict | None = None) -> dict:
    roles = {"Category": [category], "Y": values}
    if series is not None:
        roles["Series"] = [series]
    return visual(name, "lineChart", x, y, w, h, title, roles)


def bar(name: str, x: int, y: int, w: int, h: int, title: str, category: dict, values: list[dict], series: dict | None = None) -> dict:
    roles = {"Category": [category], "Y": values}
    if series is not None:
        roles["Series"] = [series]
    return visual(name, "clusteredColumnChart", x, y, w, h, title, roles)


def matrix(name: str, x: int, y: int, w: int, h: int, title: str, rows: list[dict], cols: list[dict], vals: list[dict]) -> dict:
    return visual(name, "pivotTable", x, y, w, h, title, {"Rows": rows, "Columns": cols, "Values": vals})


def page_json(name: str, display: str) -> dict:
    return {
        "$schema": PAGE_SCHEMA,
        "name": name,
        "displayName": display,
        "displayOption": "FitToPage",
        "height": 720,
        "width": 1280,
    }


def pages() -> dict[str, tuple[str, list[dict]]]:
    return {
        "Executive_QA": (
            "Executive QA",
            [
                slicer("slicer_year", 20, 20, "Delivery year", col("HFC_Hourly", "year")),
                slicer("slicer_season", 210, 20, "Season", col("HFC_Hourly", "season")),
                slicer("slicer_month", 400, 20, "Month", col("HFC_Hourly", "month_label")),
                card("avg_price", 20, 125, "Avg weighted price", meas("HFC_Hourly", "Avg Price Weighted")),
                card("min_price", 220, 125, "Min weighted price", meas("HFC_Hourly", "Min Weighted Price")),
                card("weighted_neg", 420, 125, "Weighted negative hours", meas("HFC_Hourly", "Weighted Negative Hours")),
                card("fast_neg", 620, 125, "Fast negative hours", meas("HFC_Hourly", "Fast Negative Hours")),
                card("base_res", 820, 125, "EEX BASE residual", meas("EEX_Residuals", "EEX Max BASE Residual Text")),
                card("peak_res", 1020, 125, "EEX PEAK residual", meas("EEX_Residuals", "EEX Max PEAK Residual Text")),
                card("peak_spread", 20, 235, "Peak/offpeak spread", meas("HFC_Hourly", "Peak Offpeak Spread")),
                card("duck_spread", 220, 235, "Evening/midday spread", meas("HFC_Hourly", "Evening Midday Spread")),
                table("summary", 20, 360, 520, 310, "Summary metrics", [col("Summary_Metrics", "metric"), col("Summary_Metrics", "value")]),
                table("annual", 580, 310, 660, 360, "Annual shape", [
                    col("Annual_Shape", "year"),
                    col("Annual_Shape", "mean_eur_mwh"),
                    col("Annual_Shape", "evening_minus_midday_eur_mwh"),
                    col("Annual_Shape", "weekend_minus_weekday_eur_mwh"),
                    col("Annual_Shape", "structural_width_mean_eur_mwh"),
                    col("Annual_Shape", "structural_width_p95_eur_mwh"),
                ]),
            ],
        ),
        "EEX_Calibration": (
            "EEX Calibration",
            [
                table("eex_table", 20, 30, 760, 620, "Quoted EEX BASE/PEAK residuals", [
                    col("EEX_Residuals", "load_type"),
                    col("EEX_Residuals", "product"),
                    col("EEX_Residuals", "target_eex_eur_mwh"),
                    col("EEX_Residuals", "csv_mean_eur_mwh"),
                    col("EEX_Residuals", "abs_error_eur_mwh"),
                    col("EEX_Residuals", "rows"),
                ]),
                table("eex_peak_products", 820, 40, 420, 330, "PEAK/BASE residual detail", [
                    col("EEX_Residuals", "load_type"),
                    col("EEX_Residuals", "product"),
                    col("EEX_Residuals", "abs_error_eur_mwh"),
                ]),
                card("eex_max", 850, 430, "Max EEX residual", meas("EEX_Residuals", "EEX Max Abs Residual Text")),
                card("eex_base", 850, 540, "Max BASE residual", meas("EEX_Residuals", "EEX Max BASE Residual Text")),
                card("eex_peak", 1050, 540, "Max PEAK residual", meas("EEX_Residuals", "EEX Max PEAK Residual Text")),
            ],
        ),
        "Duck_Curves": (
            "Duck Curves",
            [
                slicer("duck_year", 20, 20, "Year", col("Duck_Month_Hour", "year")),
                line("monthly_duck", 20, 115, 600, 330, "Monthly duck curve - weighted mean", col("Duck_Month_Hour", "hour"), [meas("Duck_Month_Hour", "Duck Month Avg Weighted Price")], col("Duck_Month_Hour", "month_code")),
                line("season_duck", 650, 115, 590, 330, "Seasonal duck curve - weighted mean", col("Duck_Season_Hour", "hour"), [meas("Duck_Season_Hour", "Duck Season Avg Weighted Price")], col("Duck_Season_Hour", "season")),
                table("monthly_duck_data", 20, 470, 600, 190, "Monthly duck data", [
                    col("Duck_Month_Hour", "year"),
                    col("Duck_Month_Hour", "month_code"),
                    col("Duck_Month_Hour", "hour_code"),
                    col("Duck_Month_Hour", "price_weighted_mean_eur_mwh"),
                    col("Duck_Month_Hour", "price_fast_eur_mwh"),
                    col("Duck_Month_Hour", "structural_p10_eur_mwh"),
                ]),
                table("season_duck_data", 650, 470, 590, 190, "Seasonal duck data", [
                    col("Duck_Season_Hour", "year"),
                    col("Duck_Season_Hour", "season"),
                    col("Duck_Season_Hour", "hour_code"),
                    col("Duck_Season_Hour", "price_weighted_mean_eur_mwh"),
                    col("Duck_Season_Hour", "price_fast_eur_mwh"),
                    col("Duck_Season_Hour", "structural_p10_eur_mwh"),
                ]),
            ],
        ),
        "Seasonality": (
            "Seasonality",
            [
                line("monthly_means", 20, 40, 620, 320, "Monthly means by year", col("Duck_Month_Hour", "month"), [meas("Duck_Month_Hour", "Duck Month Avg Weighted Price")], col("Duck_Month_Hour", "year")),
                table("annual_shape", 670, 40, 570, 350, "Annual shape KPIs", [
                    col("Annual_Shape", "year"),
                    col("Annual_Shape", "mean_eur_mwh"),
                    col("Annual_Shape", "evening_minus_midday_eur_mwh"),
                    col("Annual_Shape", "weekend_minus_weekday_eur_mwh"),
                    col("Annual_Shape", "structural_width_p95_eur_mwh"),
                ]),
                table("annual_spreads", 20, 410, 620, 260, "Annual spreads and negative tails", [
                    col("Annual_Shape", "year"),
                    col("Annual_Shape", "winter_summer_spread_eur_mwh"),
                    col("Annual_Shape", "peak_minus_offpeak_eur_mwh"),
                    col("Annual_Shape", "evening_minus_midday_eur_mwh"),
                    col("Annual_Shape", "weekend_minus_weekday_eur_mwh"),
                    col("Annual_Shape", "negative_hours_fast"),
                    col("Annual_Shape", "negative_hours_p10"),
                ]),
            ],
        ),
        "Negative_Tail": (
            "Negative Tail",
            [
                matrix("neg_heatmap", 20, 40, 620, 520, "Fast negative hours heatmap", [col("Negative_Low_Hours", "month_code")], [col("Negative_Low_Hours", "hour_code")], [col("Negative_Low_Hours", "fast_negative_hours")]),
                table("neg_table", 670, 40, 570, 520, "Low-price diagnostics", [
                    col("Negative_Low_Hours", "year"),
                    col("Negative_Low_Hours", "month"),
                    col("Negative_Low_Hours", "hour"),
                    col("Negative_Low_Hours", "fast_negative_hours"),
                    col("Negative_Low_Hours", "p10_negative_hours"),
                    col("Negative_Low_Hours", "min_fast_eur_mwh"),
                    col("Negative_Low_Hours", "min_p10_eur_mwh"),
                ]),
            ],
        ),
        "Data_Drilldown": (
            "Data Drilldown",
            [
                table("hourly_table", 20, 30, 1220, 620, "Hourly records", [
                    col("HFC_Hourly", "timestamp_ch_text"),
                    col("HFC_Hourly", "timestamp_utc_text"),
                    col("HFC_Hourly", "hour_code"),
                    col("HFC_Hourly", "price_weighted_mean_eur_mwh"),
                    col("HFC_Hourly", "price_central_eur_mwh"),
                    col("HFC_Hourly", "price_fast_eur_mwh"),
                    col("HFC_Hourly", "structural_p10_eur_mwh"),
                    col("HFC_Hourly", "structural_p90_eur_mwh"),
                    col("HFC_Hourly", "is_eex_peak"),
                    col("HFC_Hourly", "is_fast_negative"),
                ])
            ],
        ),
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build(report_dir: Path, *, backup: bool = True) -> None:
    pages_root = report_dir / "definition" / "pages"
    if backup:
        backup_dir = report_dir.with_name(report_dir.name + f".bak_{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.copytree(report_dir, backup_dir)
        print(f"[powerbi-layout] backup -> {backup_dir}")

    for child in pages_root.iterdir() if pages_root.exists() else []:
        if child.is_dir():
            shutil.rmtree(child)
        elif child.name == "pages.json":
            child.unlink()

    page_defs = pages()
    order = list(page_defs)
    for page_name, (display, visuals) in page_defs.items():
        page_dir = pages_root / page_name
        write_json(page_dir / "page.json", page_json(page_name, display))
        visuals_root = page_dir / "visuals"
        for idx, item in enumerate(visuals, start=1):
            write_json(visuals_root / f"{idx:02d}_{item['name']}" / "visual.json", item)

    write_json(
        pages_root / "pages.json",
        {"$schema": PAGES_SCHEMA, "pageOrder": order, "activePageName": order[0]},
    )
    print(f"[powerbi-layout] pages={len(order)} visuals={sum(len(v) for _, v in page_defs.values())}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="powerbi/PFC_QA.Report")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args(argv)
    build(Path(args.report_dir), backup=not args.no_backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

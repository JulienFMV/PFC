"""Build the Power BI semantic model files for the PFC QA project."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


ROOT_TABLES = {
    "HFC_Hourly": {
        "csv": "hfc_hourly_powerbi.csv",
        "columns": [
            ("timestamp_ch_text", "string", "type text", None),
            ("timestamp_ch_iso", "dateTime", "type datetime", "dd.mm.yyyy hh:nn"),
            ("timestamp_ch", "string", "type text", None),
            ("utc_offset_ch", "string", "type text", None),
            ("timestamp_utc", "string", "type text", None),
            ("timestamp_utc_text", "string", "type text", None),
            ("date_ch", "dateTime", "type date", "dd.mm.yyyy"),
            ("year", "int64", "Int64.Type", "0"),
            ("quarter", "int64", "Int64.Type", "0"),
            ("month", "int64", "Int64.Type", "0"),
            ("month_code", "string", "type text", None),
            ("month_label", "string", "type text", None),
            ("hour", "int64", "Int64.Type", "0"),
            ("hour_code", "string", "type text", None),
            ("weekday", "int64", "Int64.Type", "0"),
            ("season", "string", "type text", None),
            ("is_weekend", "boolean", "type logical", None),
            ("is_eex_peak", "boolean", "type logical", None),
            ("is_weighted_negative", "boolean", "type logical", None),
            ("is_fast_negative", "boolean", "type logical", None),
            ("is_p10_negative", "boolean", "type logical", None),
            ("price_slow_eur_mwh", "double", "type number", "0.00"),
            ("price_central_eur_mwh", "double", "type number", "0.00"),
            ("price_fast_eur_mwh", "double", "type number", "0.00"),
            ("price_weighted_mean_eur_mwh", "double", "type number", "0.00"),
            ("structural_p10_eur_mwh", "double", "type number", "0.00"),
            ("structural_p50_eur_mwh", "double", "type number", "0.00"),
            ("structural_p90_eur_mwh", "double", "type number", "0.00"),
            ("structural_width_eur_mwh", "double", "type number", "0.00"),
        ],
        "measures": [
            ("Avg Price Weighted", "AVERAGE ( HFC_Hourly[price_weighted_mean_eur_mwh] )", "0.00"),
            ("Avg Price Central", "AVERAGE ( HFC_Hourly[price_central_eur_mwh] )", "0.00"),
            ("Avg Price Fast", "AVERAGE ( HFC_Hourly[price_fast_eur_mwh] )", "0.00"),
            ("Avg Price Slow", "AVERAGE ( HFC_Hourly[price_slow_eur_mwh] )", "0.00"),
            ("Min Weighted Price", "MIN ( HFC_Hourly[price_weighted_mean_eur_mwh] )", "0.00"),
            (
                "Weighted Negative Hours",
                "CALCULATE ( COUNTROWS ( HFC_Hourly ), HFC_Hourly[price_weighted_mean_eur_mwh] < 0 )",
                "0",
            ),
            (
                "Fast Negative Hours",
                "CALCULATE ( COUNTROWS ( HFC_Hourly ), HFC_Hourly[price_fast_eur_mwh] < 0 )",
                "0",
            ),
            (
                "P10 Negative Hours",
                "CALCULATE ( COUNTROWS ( HFC_Hourly ), HFC_Hourly[structural_p10_eur_mwh] < 0 )",
                "0",
            ),
            (
                "Peak Avg Price",
                "CALCULATE ( [Avg Price Weighted], HFC_Hourly[is_eex_peak] = TRUE () )",
                "0.00",
            ),
            (
                "Offpeak Avg Price",
                "CALCULATE ( [Avg Price Weighted], HFC_Hourly[is_eex_peak] = FALSE () )",
                "0.00",
            ),
            ("Peak Offpeak Spread", "[Peak Avg Price] - [Offpeak Avg Price]", "0.00"),
            (
                "Midday Avg Price",
                "CALCULATE ( [Avg Price Weighted], HFC_Hourly[hour] >= 11, HFC_Hourly[hour] <= 15 )",
                "0.00",
            ),
            (
                "Evening Avg Price",
                "CALCULATE ( [Avg Price Weighted], HFC_Hourly[hour] >= 17, HFC_Hourly[hour] <= 21 )",
                "0.00",
            ),
            ("Evening Midday Spread", "[Evening Avg Price] - [Midday Avg Price]", "0.00"),
            (
                "Weekend Avg Price",
                "CALCULATE ( [Avg Price Weighted], HFC_Hourly[is_weekend] = TRUE () )",
                "0.00",
            ),
            (
                "Weekday Avg Price",
                "CALCULATE ( [Avg Price Weighted], HFC_Hourly[is_weekend] = FALSE () )",
                "0.00",
            ),
            ("Weekend Weekday Spread", "[Weekend Avg Price] - [Weekday Avg Price]", "0.00"),
        ],
    },
    "EEX_Residuals": {
        "csv": "eex_residuals.csv",
        "columns": [
            ("load_type", "string", "type text", None),
            ("product", "string", "type text", None),
            ("target_eex_eur_mwh", "double", "type number", "0.00"),
            ("csv_mean_eur_mwh", "double", "type number", "0.00"),
            ("abs_error_eur_mwh", "double", "type number", "0.000000"),
            ("rows", "int64", "Int64.Type", "0"),
        ],
        "measures": [
            ("EEX Max Abs Residual", "MAX ( EEX_Residuals[abs_error_eur_mwh] )", "0.000000"),
            (
                "EEX Max BASE Residual",
                'CALCULATE ( MAX ( EEX_Residuals[abs_error_eur_mwh] ), EEX_Residuals[load_type] = "BASE" )',
                "0.000000",
            ),
            (
                "EEX Max PEAK Residual",
                'CALCULATE ( MAX ( EEX_Residuals[abs_error_eur_mwh] ), EEX_Residuals[load_type] = "PEAK" )',
                "0.000000",
            ),
            ("EEX Max Abs Residual Text", 'FORMAT ( [EEX Max Abs Residual], "0.000000" )', None),
            ("EEX Max BASE Residual Text", 'FORMAT ( [EEX Max BASE Residual], "0.000000" )', None),
            ("EEX Max PEAK Residual Text", 'FORMAT ( [EEX Max PEAK Residual], "0.000000" )', None),
        ],
    },
    "Duck_Month_Hour": {
        "csv": "duck_month_hour.csv",
        "columns": [
            ("year", "int64", "Int64.Type", "0"),
            ("month", "int64", "Int64.Type", "0"),
            ("hour", "int64", "Int64.Type", "0"),
            ("price_slow_eur_mwh", "double", "type number", "0.00"),
            ("price_central_eur_mwh", "double", "type number", "0.00"),
            ("price_fast_eur_mwh", "double", "type number", "0.00"),
            ("price_weighted_mean_eur_mwh", "double", "type number", "0.00"),
            ("structural_p10_eur_mwh", "double", "type number", "0.00"),
            ("structural_p50_eur_mwh", "double", "type number", "0.00"),
            ("structural_p90_eur_mwh", "double", "type number", "0.00"),
            ("structural_width_eur_mwh", "double", "type number", "0.00"),
            ("month_code", "string", "type text", None),
            ("hour_code", "string", "type text", None),
        ],
        "measures": [
            ("Duck Month Avg Weighted Price", "AVERAGE ( Duck_Month_Hour[price_weighted_mean_eur_mwh] )", "0.00"),
            ("Duck Month Avg Central Price", "AVERAGE ( Duck_Month_Hour[price_central_eur_mwh] )", "0.00"),
            ("Duck Month Avg Fast Price", "AVERAGE ( Duck_Month_Hour[price_fast_eur_mwh] )", "0.00"),
            ("Duck Month Avg Structural Width", "AVERAGE ( Duck_Month_Hour[structural_width_eur_mwh] )", "0.00"),
        ],
    },
    "Duck_Month_Hour_Long": {
        "csv": "duck_month_hour_long.csv",
        "columns": [
            ("year", "int64", "Int64.Type", "0"),
            ("month", "int64", "Int64.Type", "0"),
            ("month_code", "string", "type text", None),
            ("hour", "int64", "Int64.Type", "0"),
            ("hour_code", "string", "type text", None),
            ("series", "string", "type text", None),
            ("price_eur_mwh", "double", "type number", "0.00"),
            ("series_label", "string", "type text", None),
        ],
        "measures": [("Duck Month Long Avg Price", "AVERAGE ( Duck_Month_Hour_Long[price_eur_mwh] )", "0.00")],
    },
    "Duck_Season_Hour": {
        "csv": "duck_season_hour.csv",
        "columns": [
            ("year", "int64", "Int64.Type", "0"),
            ("season", "string", "type text", None),
            ("hour", "int64", "Int64.Type", "0"),
            ("price_slow_eur_mwh", "double", "type number", "0.00"),
            ("price_central_eur_mwh", "double", "type number", "0.00"),
            ("price_fast_eur_mwh", "double", "type number", "0.00"),
            ("price_weighted_mean_eur_mwh", "double", "type number", "0.00"),
            ("structural_p10_eur_mwh", "double", "type number", "0.00"),
            ("structural_p50_eur_mwh", "double", "type number", "0.00"),
            ("structural_p90_eur_mwh", "double", "type number", "0.00"),
            ("structural_width_eur_mwh", "double", "type number", "0.00"),
            ("hour_code", "string", "type text", None),
        ],
        "measures": [
            ("Duck Season Avg Weighted Price", "AVERAGE ( Duck_Season_Hour[price_weighted_mean_eur_mwh] )", "0.00"),
            ("Duck Season Avg Fast Price", "AVERAGE ( Duck_Season_Hour[price_fast_eur_mwh] )", "0.00"),
            ("Duck Season Avg Structural Width", "AVERAGE ( Duck_Season_Hour[structural_width_eur_mwh] )", "0.00"),
        ],
    },
    "Duck_Season_Hour_Long": {
        "csv": "duck_season_hour_long.csv",
        "columns": [
            ("year", "int64", "Int64.Type", "0"),
            ("season", "string", "type text", None),
            ("hour", "int64", "Int64.Type", "0"),
            ("hour_code", "string", "type text", None),
            ("series", "string", "type text", None),
            ("price_eur_mwh", "double", "type number", "0.00"),
            ("series_label", "string", "type text", None),
        ],
        "measures": [("Duck Season Long Avg Price", "AVERAGE ( Duck_Season_Hour_Long[price_eur_mwh] )", "0.00")],
    },
    "Annual_Shape": {
        "csv": "annual_shape.csv",
        "columns": [
            ("year", "int64", "Int64.Type", "0"),
            ("mean_eur_mwh", "double", "type number", "0.00"),
            ("winter_summer_spread_eur_mwh", "double", "type number", "0.00"),
            ("evening_minus_midday_eur_mwh", "double", "type number", "0.00"),
            ("peak_minus_offpeak_eur_mwh", "double", "type number", "0.00"),
            ("weekend_minus_weekday_eur_mwh", "double", "type number", "0.00"),
            ("negative_hours_weighted", "int64", "Int64.Type", "0"),
            ("negative_hours_fast", "int64", "Int64.Type", "0"),
            ("negative_hours_p10", "int64", "Int64.Type", "0"),
            ("structural_width_mean_eur_mwh", "double", "type number", "0.00"),
            ("structural_width_p95_eur_mwh", "double", "type number", "0.00"),
            ("min_weighted_eur_mwh", "double", "type number", "0.00"),
            ("min_fast_eur_mwh", "double", "type number", "0.00"),
            ("min_p10_eur_mwh", "double", "type number", "0.00"),
        ],
        "measures": [],
    },
    "Negative_Low_Hours": {
        "csv": "negative_low_hours.csv",
        "columns": [
            ("year", "int64", "Int64.Type", "0"),
            ("month", "int64", "Int64.Type", "0"),
            ("hour", "int64", "Int64.Type", "0"),
            ("rows", "int64", "Int64.Type", "0"),
            ("weighted_negative_hours", "int64", "Int64.Type", "0"),
            ("fast_negative_hours", "int64", "Int64.Type", "0"),
            ("p10_negative_hours", "int64", "Int64.Type", "0"),
            ("min_weighted_eur_mwh", "double", "type number", "0.00"),
            ("min_fast_eur_mwh", "double", "type number", "0.00"),
            ("min_p10_eur_mwh", "double", "type number", "0.00"),
            ("month_code", "string", "type text", None),
            ("hour_code", "string", "type text", None),
        ],
        "measures": [],
    },
    "Summary_Metrics": {
        "csv": "summary_metrics.csv",
        "columns": [("metric", "string", "type text", None), ("value", "string", "type text", None)],
        "measures": [],
    },
}


def _format_path(path: Path) -> str:
    return str(path).replace("\\", "\\")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _column_tmdl(name: str, dtype: str, fmt: str | None) -> str:
    lines = [f"\tcolumn {name}", f"\t\tdataType: {dtype}"]
    if fmt is not None:
        lines.append(f"\t\tformatString: {fmt}")
    lines.extend(["\t\tsummarizeBy: none", f"\t\tsourceColumn: {name}", ""])
    return "\n".join(lines)


def _measure_tmdl(name: str, expression: str, fmt: str | None) -> str:
    lines = [f"\tmeasure '{name}' = {expression}"]
    if fmt is not None:
        lines.append(f"\t\tformatString: {fmt}")
    lines.append("")
    return "\n".join(lines)


def _table_tmdl(table: str, spec: dict, data_dir: Path) -> str:
    csv_path = _format_path(data_dir / spec["csv"])
    columns = spec["columns"]
    typed = ", ".join(f'{{"{name}", {m_type}}}' for name, _, m_type, _ in columns)
    lines = [f"table {table}", ""]
    for name, expression, fmt in spec.get("measures", []):
        lines.append(_measure_tmdl(name, expression, fmt))
    for name, dtype, _, fmt in columns:
        lines.append(_column_tmdl(name, dtype, fmt))
    lines.extend(
        [
            f"\tpartition {table} = m",
            "\t\tmode: import",
            "\t\tsource =",
            "\t\t\t\tlet",
            f'\t\t\t\t    Source = Csv.Document(File.Contents("{csv_path}"), [Delimiter=",", Columns={len(columns)}, Encoding=65001, QuoteStyle=QuoteStyle.Csv]),',
            "\t\t\t\t    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),",
            f"\t\t\t\t    Typed = Table.TransformColumnTypes(Promoted, {{{typed}}})",
            "\t\t\t\tin",
            "\t\t\t\t    Typed",
            "",
        ]
    )
    return "\n".join(lines)


def build(model_dir: Path, data_dir: Path, *, backup: bool = True) -> None:
    definition = model_dir / "definition"
    tables_dir = definition / "tables"
    if backup:
        backup_dir = model_dir.with_name(model_dir.name + f".bak_{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.copytree(model_dir, backup_dir)
        print(f"[powerbi-model] backup -> {backup_dir}")

    tables_dir.mkdir(parents=True, exist_ok=True)
    for path in tables_dir.glob("*.tmdl"):
        path.unlink()

    order = list(ROOT_TABLES)
    for table, spec in ROOT_TABLES.items():
        _write(tables_dir / f"{table}.tmdl", _table_tmdl(table, spec, data_dir))

    refs = "\n".join(f"ref table {name}" for name in order)
    model = f"""model Model
\tculture: en-US
\tdefaultPowerBIDataSourceVersion: powerBI_V3
\tsourceQueryCulture: fr-CH

annotation __PBI_TimeIntelligenceEnabled = 0

annotation PBI_ProTooling = ["DevMode"]

annotation PBI_QueryOrder = {order!r}

{refs}

ref cultureInfo en-US
"""
    _write(definition / "model.tmdl", model.replace("'", '"'))
    _write(definition / "relationships.tmdl", "")
    for cache_name in ("cache.abf",):
        cache_path = model_dir / ".pbi" / cache_name
        if cache_path.exists():
            try:
                cache_path.unlink()
                print(f"[powerbi-model] removed stale cache -> {cache_path}")
            except PermissionError:
                print(f"[powerbi-model] stale cache locked by Power BI -> {cache_path}")
    print(f"[powerbi-model] tables={len(order)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="powerbi/PFC_QA.SemanticModel")
    parser.add_argument("--data-dir", default="powerbi/data")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args(argv)
    build(Path(args.model_dir), Path(args.data_dir).resolve(), backup=not args.no_backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Import Swiss BFE Energiedashboard daily CSV feeds.

The Energiedashboard feeds are useful public daily actuals for CH production,
consumption, cross-border flows and spot base prices. They are not long-term
scenario paths and the forecast feed is not vintage-safe unless this script is
run as a daily snapshot archive.
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_LOCAL_ROOT = Path.home() / "pfc_local_data" / "scenarios" / "bfe_energiedashboard"
DEFAULT_OUTPUT = Path("data/bfe_energiedashboard_daily.parquet")
DEFAULT_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/BFE-ENERGYDASHBOARD.md")


@dataclass(frozen=True)
class BfeFeed:
    dataset_id: str
    url: str
    kind: str
    source_name: str
    issued_date: str


FEEDS: tuple[BfeFeed, ...] = (
    BfeFeed(
        "BFE-DS-0093",
        "https://www.bfe-ogd.ch/ogd104_stromproduktion_swissgrid.csv",
        "production",
        "BFE Energiedashboard electricity production Swissgrid",
        "2022-12-14",
    ),
    BfeFeed(
        "BFE-DS-0096",
        "https://www.bfe-ogd.ch/ogd103_stromverbrauch_swissgrid_lv_und_endv.csv",
        "consumption",
        "BFE Energiedashboard national and final consumption",
        "2022-12-14",
    ),
    BfeFeed(
        "BFE-DS-0094",
        "https://www.bfe-ogd.ch/ogd107_strom_import_export.csv",
        "power_flows",
        "BFE Energiedashboard daily electricity import/export flows",
        "2022-12-14",
    ),
    BfeFeed(
        "BFE-DS-0087",
        "https://www.bfe-ogd.ch/ogd106_preise_strom_boerse.csv",
        "spot_base",
        "BFE Energiedashboard day-ahead base spot price",
        "2022-12-14",
    ),
    BfeFeed(
        "BFE-DS-0082",
        "https://www.bfe-ogd.ch/ogd110_strom_verbrauch_prognose.csv",
        "consumption_forecast",
        "BFE Energiedashboard SDSC power consumption forecast",
        "2023-01-11",
    ),
    BfeFeed(
        "BFE-DS-0095",
        "https://www.bfe-ogd.ch/ogd103_stromverbrauch_geschaetzt_swissgrid.csv",
        "estimated_consumption",
        "BFE Energiedashboard model-based estimated national consumption",
        "2022-12-14",
    ),
)

PRODUCTION_CARRIER_MAP = {
    "flusskraft": "production_hydro_ror_gwh",
    "kernkraft": "production_nuclear_gwh",
    "photovoltaik": "production_solar_gwh",
    "speicherkraft": "production_hydro_storage_gwh",
    "thermische": "production_thermal_gwh",
    "wind": "production_wind_gwh",
}

COLUMN_MAP = {
    "Landesverbrauch_GWh": "national_consumption_gwh",
    "Endverbrauch_GWh": "final_consumption_gwh",
    "Landesverbrauch_geschaetzt_SG_GWh": "national_consumption_estimated_gwh",
    "Baseload_EUR_MWh": "spot_baseload_eur_mwh",
    "Endverbrauch_Prognose_Min_GWh": "final_consumption_forecast_p025_gwh",
    "Endverbrauch_Prognose_Mittel_GWh": "final_consumption_forecast_mean_gwh",
    "Endverbrauch_Prognose_Max_GWh": "final_consumption_forecast_p975_gwh",
    "AT_CH_GWh": "flow_at_to_ch_gwh",
    "DE_CH_GWh": "flow_de_to_ch_gwh",
    "FR_CH_GWh": "flow_fr_to_ch_gwh",
    "IT_CH_GWh": "flow_it_to_ch_gwh",
    "CH_AT_GWh": "flow_ch_to_at_gwh",
    "CH_DE_GWh": "flow_ch_to_de_gwh",
    "CH_FR_GWh": "flow_ch_to_fr_gwh",
    "CH_IT_GWh": "flow_ch_to_it_gwh",
    "Nettoimport": "net_import_gwh",
}


def _slug(text: object) -> str:
    value = str(text).strip().lower()
    value = value.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    value = value.replace("é", "e").replace("è", "e").replace("à", "a")
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def _read_csv(feed: BfeFeed, *, source_dir: Path | None) -> pd.DataFrame:
    if source_dir is not None:
        path = source_dir / Path(feed.url).name
        return pd.read_csv(path)
    return pd.read_csv(feed.url)


def download_feeds(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for feed in FEEDS:
        request = urllib.request.Request(feed.url, headers={"User-Agent": "PFC-LT BFE Energiedashboard importer"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        (raw_dir / Path(feed.url).name).write_bytes(payload)


def _date_column(frame: pd.DataFrame) -> str:
    for candidate in ("Datum", "Date"):
        if candidate in frame.columns:
            return candidate
    raise KeyError("BFE feed missing Datum/Date column")


def _normalize_feed(feed: BfeFeed, frame: pd.DataFrame, *, ingested_at_utc: pd.Timestamp) -> pd.DataFrame:
    date_col = _date_column(frame)
    df = frame.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    rows: list[dict[str, object]] = []

    if feed.kind == "production":
        for _, row in df.iterrows():
            carrier = _slug(row["Energietraeger"])
            variable = PRODUCTION_CARRIER_MAP.get(carrier, f"production_{carrier}_gwh")
            rows.append(
                {
                    "date_utc": row[date_col],
                    "dataset_id": feed.dataset_id,
                    "source": feed.source_name,
                    "source_url": feed.url,
                    "track": "actual",
                    "variable": variable,
                    "value": pd.to_numeric(row["Produktion_GWh"], errors="coerce"),
                    "unit": "GWh",
                    "publication_date": ingested_at_utc,
                    "measurement_date": row[date_col],
                    "as_of_utc": pd.NaT,
                    "quality_flag": "official_daily_actual_no_row_publication_time",
                    "ingested_at_utc": ingested_at_utc,
                }
            )
    else:
        value_columns = [column for column in df.columns if column != date_col]
        for _, row in df.iterrows():
            for column in value_columns:
                variable = COLUMN_MAP.get(column, _slug(column))
                unit = "EUR/MWh" if variable.endswith("eur_mwh") else "GWh"
                track = "forecast" if feed.kind == "consumption_forecast" else "actual"
                rows.append(
                    {
                        "date_utc": row[date_col],
                        "dataset_id": feed.dataset_id,
                        "source": feed.source_name,
                        "source_url": feed.url,
                        "track": track,
                        "variable": variable,
                        "value": pd.to_numeric(row[column], errors="coerce"),
                        "unit": unit,
                        "publication_date": ingested_at_utc,
                        "measurement_date": row[date_col] if track == "actual" else pd.NaT,
                        "as_of_utc": ingested_at_utc if track == "forecast" else pd.NaT,
                        "quality_flag": (
                            "official_forecast_snapshot_ingest_time_only"
                            if track == "forecast"
                            else "official_daily_actual_no_row_publication_time"
                        ),
                        "ingested_at_utc": ingested_at_utc,
                    }
                )
    out = pd.DataFrame(rows)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["date_utc", "variable", "value"]).reset_index(drop=True)


def build_bfe_energiedashboard_daily(
    *,
    source_dir: Path | None = None,
    ingested_at_utc: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    stamp = pd.Timestamp.utcnow() if ingested_at_utc is None else pd.Timestamp(ingested_at_utc)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    frames = [
        _normalize_feed(feed, _read_csv(feed, source_dir=source_dir), ingested_at_utc=stamp)
        for feed in FEEDS
    ]
    return pd.concat(frames, ignore_index=True).sort_values(["date_utc", "dataset_id", "variable"])


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    lines = [
        "| " + " | ".join(str(c) for c in df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[c]).replace("|", "\\|").replace("\n", " ") for c in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path: Path, *, output: Path, raw_dir: Path, frame: pd.DataFrame) -> None:
    summary = (
        frame.groupby(["dataset_id", "source", "track"], dropna=False)
        .agg(
            rows=("value", "size"),
            variables=("variable", "nunique"),
            first_date=("date_utc", "min"),
            last_date=("date_utc", "max"),
            quality_flag=("quality_flag", "first"),
        )
        .reset_index()
        .sort_values("dataset_id")
    )
    variables = (
        frame.groupby(["dataset_id", "variable", "unit"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["dataset_id", "variable"])
    )
    lines = [
        "# BFE Energiedashboard Import",
        "",
        f"* output: `{output}`",
        f"* raw cache: `{raw_dir}`",
        f"* rows: `{len(frame)}`",
        "",
        "## Source Summary",
        "",
        _md_table(summary),
        "",
        "## Variables",
        "",
        _md_table(variables),
        "",
        "## Production Use",
        "",
        "These feeds are official daily actuals useful for CH historical calibration and diagnostics.",
        "They are not long-term 2030 scenario paths. The SDSC forecast feed is only vintage-safe",
        "from the ingestion timestamp unless this importer is scheduled daily without overwriting snapshots.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(DEFAULT_LOCAL_ROOT))
    parser.add_argument("--source-dir", default=None, help="Read already downloaded CSVs from this directory.")
    parser.add_argument("--no-download", action="store_true", help="Skip download and read from --source-dir/--raw-dir.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--ingested-at-utc", default=None)
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    source_dir = Path(args.source_dir) if args.source_dir else raw_dir
    if not args.no_download:
        download_feeds(raw_dir)
    frame = build_bfe_energiedashboard_daily(
        source_dir=source_dir,
        ingested_at_utc=args.ingested_at_utc,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    write_report(Path(args.report), output=output, raw_dir=raw_dir, frame=frame)
    print(f"[bfe-energiedashboard] rows={len(frame)}")
    print(f"[bfe-energiedashboard] output -> {output}")
    print(f"[bfe-energiedashboard] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

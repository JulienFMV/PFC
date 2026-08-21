"""Import BFE/SFOE structural actuals for Swiss LT PFC modelling.

The imported feeds are official BFE actual/baseline datasets:

* weekly Swiss storage-lake reservoir content;
* in-operation Swiss electricity production plants;
* WASTA Swiss hydropower plant technical statistics.

They are useful for calibration and scenario actualization. They are not
forward-looking 2030 scenario paths and must not by themselves approve the LT
scenario inventory for production.
"""

from __future__ import annotations

import argparse
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_LOCAL_ROOT = Path.home() / "pfc_local_data" / "scenarios" / "bfe_structural_actuals"
DEFAULT_RESERVOIR_OUTPUT = Path("data/bfe_hydro_reservoir_weekly.parquet")
DEFAULT_PLANTS_OUTPUT = Path("data/bfe_ch_production_plants.parquet")
DEFAULT_CAPACITY_OUTPUT = Path("data/bfe_ch_installed_capacity_actuals.parquet")
DEFAULT_WASTA_PLANTS_OUTPUT = Path("data/bfe_wasta_hydro_plants.parquet")
DEFAULT_WASTA_SUMMARY_OUTPUT = Path("data/bfe_wasta_hydro_summary.parquet")
DEFAULT_STRUCTURAL_OUTPUT = Path("data/bfe_ch_structural_actuals.parquet")
DEFAULT_REPORT = Path(".planning/phases/13-lt-electrification-scenario-shape/BFE-STRUCTURAL-ACTUALS.md")


@dataclass(frozen=True)
class SourceFile:
    source_id: str
    dataset_id: str
    url: str
    filename: str
    source_name: str


RESERVOIR_SOURCE = SourceFile(
    "bfe_speicherseen_weekly",
    "ogd17-bfe",
    "https://www.uvek-gis.admin.ch/BFE/ogd/17/ogd17_fuellungsgrad_speicherseen.csv",
    "ogd17_fuellungsgrad_speicherseen.csv",
    "BFE weekly Swiss storage-lake reservoir content",
)
PRODUCTION_PLANTS_SOURCE = SourceFile(
    "bfe_electricity_production_plants",
    "ch.bfe.elektrizitaetsproduktionsanlagen",
    "https://data.geo.admin.ch/ch.bfe.elektrizitaetsproduktionsanlagen/csv/2056/ch.bfe.elektrizitaetsproduktionsanlagen.zip",
    "ch.bfe.elektrizitaetsproduktionsanlagen.zip",
    "BFE electricity production plants in operation",
)
WASTA_SOURCE = SourceFile(
    "bfe_wasta_hydropower_plants",
    "ch.bfe.statistik-wasserkraftanlagen",
    "https://data.geo.admin.ch/ch.bfe.statistik-wasserkraftanlagen/statistik-wasserkraftanlagen/statistik-wasserkraftanlagen_2056.csv.zip",
    "statistik-wasserkraftanlagen_2056.csv.zip",
    "BFE WASTA hydropower plant statistics",
)
SOURCES = (RESERVOIR_SOURCE, PRODUCTION_PLANTS_SOURCE, WASTA_SOURCE)

REGION_LABELS = {
    "Wallis": "valais",
    "Graubuenden": "grisons",
    "Tessin": "ticino",
    "UebrigCH": "other_ch",
    "TotalCH": "total_ch",
}


def _stamp(value: str | pd.Timestamp | None) -> pd.Timestamp:
    stamp = pd.Timestamp.utcnow() if value is None else pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _slug(text: object) -> str:
    value = str(text or "").strip().lower()
    value = (
        value.replace("Ã¤", "ae")
        .replace("Ã¶", "oe")
        .replace("Ã¼", "ue")
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("é", "e")
        .replace("è", "e")
        .replace("à", "a")
    )
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def download_sources(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        request = urllib.request.Request(source.url, headers={"User-Agent": "PFC-LT BFE structural importer"})
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
        (raw_dir / source.filename).write_bytes(payload)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_zip_member(path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        with archive.open(member) as handle:
            return pd.read_csv(handle)


def _catalog_labels(frame: pd.DataFrame, *, key: str = "Catalogue_id", prefix: str) -> pd.DataFrame:
    columns = [key]
    for column in ("de", "fr", "it", "en", "DE", "FR", "IT", "ID"):
        if column in frame.columns:
            columns.append(column)
    out = frame[columns].copy()
    return out.rename(columns={column: f"{prefix}_{column.lower()}" for column in out.columns if column != key})


def build_reservoir_weekly(
    source_dir: Path,
    *,
    ingested_at_utc: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    stamp = _stamp(ingested_at_utc)
    raw = _read_csv(source_dir / RESERVOIR_SOURCE.filename)
    if "Datum" not in raw.columns:
        raise KeyError("BFE reservoir file missing Datum column")
    raw["Datum"] = pd.to_datetime(raw["Datum"], utc=True)
    rows: list[dict[str, object]] = []
    for _, row in raw.iterrows():
        for prefix, region in REGION_LABELS.items():
            content_col = f"{prefix}_speicherinhalt_gwh"
            capacity_col = f"{prefix}_max_speicherinhalt_gwh"
            if content_col not in raw.columns or capacity_col not in raw.columns:
                continue
            content = pd.to_numeric(row[content_col], errors="coerce")
            capacity = pd.to_numeric(row[capacity_col], errors="coerce")
            base = {
                "date_utc": row["Datum"],
                "country": "CH",
                "region": region,
                "dataset_id": RESERVOIR_SOURCE.dataset_id,
                "source_id": RESERVOIR_SOURCE.source_id,
                "source": RESERVOIR_SOURCE.source_name,
                "source_url": RESERVOIR_SOURCE.url,
                "track": "actual",
                "publication_date": stamp,
                "measurement_date": row["Datum"],
                "quality_flag": "official_weekly_actual_no_row_publication_time",
                "ingested_at_utc": stamp,
            }
            rows.append({**base, "variable": "hydro_reservoir_content_gwh", "value": content, "unit": "GWh"})
            rows.append({**base, "variable": "hydro_reservoir_capacity_gwh", "value": capacity, "unit": "GWh"})
            fill_ratio = content / capacity if pd.notna(content) and pd.notna(capacity) and capacity else pd.NA
            rows.append({**base, "variable": "hydro_reservoir_fill_ratio", "value": fill_ratio, "unit": "ratio"})
    out = pd.DataFrame(rows)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["date_utc", "variable", "value"]).sort_values(["date_utc", "region", "variable"])


def _production_technology(row: pd.Series) -> str:
    main = str(row.get("main_en", "")).lower()
    sub = str(row.get("sub_en", "")).lower()
    plant = str(row.get("plant_en", "")).lower()
    text = " ".join([main, sub, plant])
    if "photovoltaic" in text:
        return "pv"
    if "wind" in text:
        return "wind"
    if "nuclear" in text:
        return "nuclear"
    if "hydro" in text:
        return "hydro"
    if "fossil" in text or "natural gas" in text or "coal" in text or "oil" in text:
        return "thermal"
    if "biomass" in text:
        return "biomass"
    if "waste" in text:
        return "waste"
    return _slug(sub or main or plant or "other")


def build_production_plants(source_dir: Path, *, ingested_at_utc: str | pd.Timestamp | None = None) -> pd.DataFrame:
    stamp = _stamp(ingested_at_utc)
    archive = source_dir / PRODUCTION_PLANTS_SOURCE.filename
    plants = _read_zip_member(archive, "ElectricityProductionPlant.csv")
    required = {"xtf_id", "BeginningOfOperation", "TotalPower", "MainCategory", "SubCategory", "PlantCategory"}
    missing = required.difference(plants.columns)
    if missing:
        raise KeyError(f"BFE production plant file missing columns: {sorted(missing)}")
    main = _catalog_labels(_read_zip_member(archive, "MainCategoryCatalogue.csv"), prefix="main")
    sub = _catalog_labels(_read_zip_member(archive, "SubCategoryCatalogue.csv"), prefix="sub")
    plant_cat = _catalog_labels(_read_zip_member(archive, "PlantCategoryCatalogue.csv"), prefix="plant")
    out = (
        plants.merge(main, left_on="MainCategory", right_on="Catalogue_id", how="left")
        .drop(columns=["Catalogue_id"], errors="ignore")
        .merge(sub, left_on="SubCategory", right_on="Catalogue_id", how="left")
        .drop(columns=["Catalogue_id"], errors="ignore")
        .merge(plant_cat, left_on="PlantCategory", right_on="Catalogue_id", how="left")
        .drop(columns=["Catalogue_id"], errors="ignore")
    )
    out["technology"] = out.apply(_production_technology, axis=1)
    out["beginning_of_operation"] = pd.to_datetime(out["BeginningOfOperation"], errors="coerce", utc=True)
    out["total_power_gw_ac_equivalent"] = pd.to_numeric(out["TotalPower"], errors="coerce") / 1_000_000.0
    out["initial_power_gw_ac_equivalent"] = pd.to_numeric(out.get("InitialPower"), errors="coerce") / 1_000_000.0
    out["country"] = "CH"
    out["dataset_id"] = PRODUCTION_PLANTS_SOURCE.dataset_id
    out["source_id"] = PRODUCTION_PLANTS_SOURCE.source_id
    out["source"] = PRODUCTION_PLANTS_SOURCE.source_name
    out["source_url"] = PRODUCTION_PLANTS_SOURCE.url
    out["publication_date"] = stamp
    out["measurement_date"] = stamp
    out["quality_flag"] = "official_register_actual_snapshot_no_row_publication_time"
    out["ingested_at_utc"] = stamp
    columns = [
        "xtf_id",
        "country",
        "Canton",
        "Municipality",
        "beginning_of_operation",
        "technology",
        "main_en",
        "sub_en",
        "plant_en",
        "total_power_gw_ac_equivalent",
        "initial_power_gw_ac_equivalent",
        "dataset_id",
        "source_id",
        "source",
        "source_url",
        "publication_date",
        "measurement_date",
        "quality_flag",
        "ingested_at_utc",
    ]
    return out[columns].rename(columns={"Canton": "canton", "Municipality": "municipality"})


def build_installed_capacity_summary(plants: pd.DataFrame) -> pd.DataFrame:
    summary = (
        plants.groupby(["country", "technology"], dropna=False)
        .agg(
            value=("total_power_gw_ac_equivalent", "sum"),
            plant_count=("xtf_id", "count"),
            first_operation=("beginning_of_operation", "min"),
            latest_operation=("beginning_of_operation", "max"),
            publication_date=("publication_date", "first"),
            measurement_date=("measurement_date", "first"),
            source_id=("source_id", "first"),
            source=("source", "first"),
            source_url=("source_url", "first"),
            quality_flag=("quality_flag", "first"),
            ingested_at_utc=("ingested_at_utc", "first"),
        )
        .reset_index()
    )
    summary["variable"] = summary["technology"].map(
        {
            "pv": "pv_gw",
            "wind": "wind_gw",
            "hydro": "hydro_capacity_gw",
            "nuclear": "nuclear_gw",
            "thermal": "thermal_capacity_gw",
            "biomass": "biomass_capacity_gw",
            "waste": "waste_capacity_gw",
        }
    )
    summary["variable"] = summary["variable"].fillna(summary["technology"].astype(str) + "_capacity_gw")
    summary["unit"] = "GW_ac_equivalent"
    return summary.sort_values(["country", "technology"])


def build_wasta_hydro(
    source_dir: Path,
    *,
    ingested_at_utc: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stamp = _stamp(ingested_at_utc)
    archive = source_dir / WASTA_SOURCE.filename
    plants = _read_zip_member(archive, "HydropowerPlant.csv")
    tech = _read_zip_member(archive, "TechnicalSpecification.csv")
    type_cat = _read_zip_member(archive, "HydropowerPlantTypeCatalogue.csv")
    status_cat = _read_zip_member(archive, "HydropowerPlantOperationalStatusCatalogue.csv")
    required = {"Year", "hydropowerPlantR", "PerformanceGeneratorMaximum", "ProductionExpected"}
    missing = required.difference(tech.columns)
    if missing:
        raise KeyError(f"BFE WASTA technical file missing columns: {sorted(missing)}")

    latest = tech.sort_values(["hydropowerPlantR", "Year"]).groupby("hydropowerPlantR", as_index=False).tail(1)
    latest = latest.merge(plants, left_on="hydropowerPlantR", right_on="WASTANumber", how="left")
    latest = latest.merge(
        type_cat.rename(columns={"ID": "TypeCode", "DE": "type_de", "FR": "type_fr", "IT": "type_it"}),
        on="TypeCode",
        how="left",
    )
    latest = latest.merge(
        status_cat.rename(
            columns={
                "ID": "OperationalStatusCode",
                "DE": "operational_status_de",
                "FR": "operational_status_fr",
                "IT": "operational_status_it",
            }
        ),
        on="OperationalStatusCode",
        how="left",
    )
    latest["country"] = "CH"
    latest["measurement_date"] = pd.to_datetime(latest["DateOfStatistic"], errors="coerce", utc=True)
    latest["hydro_capacity_gw"] = pd.to_numeric(latest["PerformanceGeneratorMaximum"], errors="coerce") / 1000.0
    latest["hydro_twh"] = pd.to_numeric(latest["ProductionExpected"], errors="coerce") / 1000.0
    latest["pumped_storage_power_gw"] = pd.to_numeric(latest.get("PumpsPowerInputMaximum"), errors="coerce") / 1000.0
    latest["dataset_id"] = WASTA_SOURCE.dataset_id
    latest["source_id"] = WASTA_SOURCE.source_id
    latest["source"] = WASTA_SOURCE.source_name
    latest["source_url"] = WASTA_SOURCE.url
    latest["publication_date"] = stamp
    latest["quality_flag"] = "official_hydro_register_actual_snapshot_no_row_publication_time"
    latest["ingested_at_utc"] = stamp
    plant_columns = [
        "WASTANumber",
        "Name",
        "Location",
        "Canton",
        "TypeCode",
        "type_de",
        "BeginningOfOperation",
        "Year",
        "measurement_date",
        "hydro_capacity_gw",
        "hydro_twh",
        "pumped_storage_power_gw",
        "operational_status_de",
        "country",
        "dataset_id",
        "source_id",
        "source",
        "source_url",
        "publication_date",
        "quality_flag",
        "ingested_at_utc",
    ]
    plant_frame = latest[plant_columns].rename(
        columns={
            "WASTANumber": "wasta_number",
            "Name": "name",
            "Location": "location",
            "Canton": "canton",
            "TypeCode": "type_code",
            "BeginningOfOperation": "beginning_of_operation_year",
            "Year": "statistic_year",
        }
    )
    summary = (
        plant_frame.groupby(["country", "type_code", "type_de"], dropna=False)
        .agg(
            hydro_capacity_gw=("hydro_capacity_gw", "sum"),
            hydro_twh=("hydro_twh", "sum"),
            pumped_storage_power_gw=("pumped_storage_power_gw", "sum"),
            plant_count=("wasta_number", "count"),
            measurement_date=("measurement_date", "max"),
            publication_date=("publication_date", "first"),
            source_id=("source_id", "first"),
            source=("source", "first"),
            source_url=("source_url", "first"),
            quality_flag=("quality_flag", "first"),
            ingested_at_utc=("ingested_at_utc", "first"),
        )
        .reset_index()
        .sort_values(["country", "type_code"])
    )
    return plant_frame, summary


def build_structural_actuals(
    reservoir: pd.DataFrame,
    capacity: pd.DataFrame,
    wasta_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    latest_reservoir_date = reservoir["date_utc"].max()
    latest_reservoir = reservoir[
        (reservoir["date_utc"] == latest_reservoir_date) & (reservoir["region"] == "total_ch")
    ]
    for variable, target_variable, divisor, unit in (
        ("hydro_reservoir_content_gwh", "hydro_reservoir_twh", 1000.0, "TWh"),
        ("hydro_reservoir_capacity_gwh", "hydro_reservoir_capacity_twh", 1000.0, "TWh"),
        ("hydro_reservoir_fill_ratio", "hydro_reservoir_fill_ratio", 1.0, "ratio"),
    ):
        match = latest_reservoir[latest_reservoir["variable"] == variable]
        if not match.empty:
            row = match.iloc[0]
            rows.append(
                {
                    "country": "CH",
                    "track": "actual",
                    "measurement_date": row["measurement_date"],
                    "publication_date": row["publication_date"],
                    "source_id": row["source_id"],
                    "source": row["source"],
                    "source_url": row["source_url"],
                    "variable": target_variable,
                    "value": row["value"] / divisor,
                    "unit": unit,
                    "quality_flag": row["quality_flag"],
                    "ingested_at_utc": row["ingested_at_utc"],
                }
            )

    for _, row in capacity.iterrows():
        if row["variable"] in {"pv_gw", "wind_gw", "nuclear_gw", "thermal_capacity_gw", "biomass_capacity_gw", "waste_capacity_gw"}:
            rows.append(
                {
                    "country": "CH",
                    "track": "actual",
                    "measurement_date": row["measurement_date"],
                    "publication_date": row["publication_date"],
                    "source_id": row["source_id"],
                    "source": row["source"],
                    "source_url": row["source_url"],
                    "variable": row["variable"],
                    "value": row["value"],
                    "unit": "GW",
                    "quality_flag": row["quality_flag"],
                    "ingested_at_utc": row["ingested_at_utc"],
                }
            )

    first_hydro = wasta_summary.iloc[0]
    total_hydro = {
        "hydro_capacity_gw": wasta_summary["hydro_capacity_gw"].sum(),
        "hydro_twh": wasta_summary["hydro_twh"].sum(),
        "pumped_storage_power_gw": wasta_summary["pumped_storage_power_gw"].sum(),
        "measurement_date": wasta_summary["measurement_date"].max(),
        "publication_date": first_hydro["publication_date"],
        "source_id": first_hydro["source_id"],
        "source": first_hydro["source"],
        "source_url": first_hydro["source_url"],
        "quality_flag": first_hydro["quality_flag"],
        "ingested_at_utc": first_hydro["ingested_at_utc"],
    }
    for variable, unit in (
        ("hydro_capacity_gw", "GW"),
        ("hydro_twh", "TWh"),
        ("pumped_storage_power_gw", "GW"),
    ):
        rows.append(
            {
                "country": "CH",
                "track": "actual",
                "measurement_date": total_hydro["measurement_date"],
                "publication_date": total_hydro["publication_date"],
                "source_id": total_hydro["source_id"],
                "source": total_hydro["source"],
                "source_url": total_hydro["source_url"],
                "variable": variable,
                "value": total_hydro[variable],
                "unit": unit,
                "quality_flag": total_hydro["quality_flag"],
                "ingested_at_utc": total_hydro["ingested_at_utc"],
            }
        )
    out = pd.DataFrame(rows)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["variable", "value"]).sort_values(["country", "variable", "source_id"])


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


def write_report(
    path: Path,
    *,
    reservoir: pd.DataFrame,
    plants: pd.DataFrame,
    capacity: pd.DataFrame,
    wasta_plants: pd.DataFrame,
    wasta_summary: pd.DataFrame,
    structural: pd.DataFrame,
    outputs: dict[str, Path],
) -> None:
    reservoir_summary = (
        reservoir.groupby(["region", "variable", "unit"], dropna=False)
        .agg(rows=("value", "size"), first_date=("date_utc", "min"), last_date=("date_utc", "max"))
        .reset_index()
    )
    capacity_summary = capacity[["technology", "variable", "value", "unit", "plant_count"]].sort_values("value", ascending=False)
    wasta_type_summary = wasta_summary[
        ["type_code", "type_de", "hydro_capacity_gw", "hydro_twh", "pumped_storage_power_gw", "plant_count"]
    ]
    structural_summary = structural[["variable", "value", "unit", "measurement_date", "source_id"]]
    lines = [
        "# BFE Structural Actuals Import",
        "",
        "## Outputs",
        "",
        _md_table(pd.DataFrame([{"name": key, "path": str(value)} for key, value in outputs.items()])),
        "",
        "## Reservoir Weekly Actuals",
        "",
        _md_table(reservoir_summary),
        "",
        "## Installed Capacity Actuals",
        "",
        _md_table(capacity_summary),
        "",
        "## WASTA Hydro Summary",
        "",
        _md_table(wasta_type_summary),
        "",
        "## Model-Facing Structural Actuals",
        "",
        _md_table(structural_summary),
        "",
        "## Production Use",
        "",
        "These are official actual/baseline datasets for CH calibration and scenario actualization.",
        "They do not create a governed 2030 scenario path and must not flip Phase 13 production gates to GO.",
        "Use them only behind explicit actualization logic with publication/measurement date checks.",
        "",
        "## Source URLs",
        "",
        _md_table(
            pd.DataFrame(
                [
                    {"source_id": source.source_id, "dataset_id": source.dataset_id, "url": source.url}
                    for source in SOURCES
                ]
            )
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(DEFAULT_LOCAL_ROOT))
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--reservoir-output", default=str(DEFAULT_RESERVOIR_OUTPUT))
    parser.add_argument("--plants-output", default=str(DEFAULT_PLANTS_OUTPUT))
    parser.add_argument("--capacity-output", default=str(DEFAULT_CAPACITY_OUTPUT))
    parser.add_argument("--wasta-plants-output", default=str(DEFAULT_WASTA_PLANTS_OUTPUT))
    parser.add_argument("--wasta-summary-output", default=str(DEFAULT_WASTA_SUMMARY_OUTPUT))
    parser.add_argument("--structural-output", default=str(DEFAULT_STRUCTURAL_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--ingested-at-utc", default=None)
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    source_dir = Path(args.source_dir) if args.source_dir else raw_dir
    if not args.no_download:
        download_sources(raw_dir)

    reservoir = build_reservoir_weekly(source_dir, ingested_at_utc=args.ingested_at_utc)
    plants = build_production_plants(source_dir, ingested_at_utc=args.ingested_at_utc)
    capacity = build_installed_capacity_summary(plants)
    wasta_plants, wasta_summary = build_wasta_hydro(source_dir, ingested_at_utc=args.ingested_at_utc)
    structural = build_structural_actuals(reservoir, capacity, wasta_summary)

    outputs = {
        "reservoir": Path(args.reservoir_output),
        "plants": Path(args.plants_output),
        "capacity": Path(args.capacity_output),
        "wasta_plants": Path(args.wasta_plants_output),
        "wasta_summary": Path(args.wasta_summary_output),
        "structural": Path(args.structural_output),
    }
    frames = {
        "reservoir": reservoir,
        "plants": plants,
        "capacity": capacity,
        "wasta_plants": wasta_plants,
        "wasta_summary": wasta_summary,
        "structural": structural,
    }
    for name, output in outputs.items():
        output.parent.mkdir(parents=True, exist_ok=True)
        frames[name].to_parquet(output, index=False)

    write_report(
        Path(args.report),
        reservoir=reservoir,
        plants=plants,
        capacity=capacity,
        wasta_plants=wasta_plants,
        wasta_summary=wasta_summary,
        structural=structural,
        outputs=outputs,
    )
    print(f"[bfe-structural] reservoir_rows={len(reservoir)}")
    print(f"[bfe-structural] plants={len(plants)}")
    print(f"[bfe-structural] wasta_plants={len(wasta_plants)}")
    print(f"[bfe-structural] structural_rows={len(structural)}")
    print(f"[bfe-structural] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

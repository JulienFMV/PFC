"""Local ingestion helpers for SFOE/BFE Energy Perspectives 2050+ hourly data."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

EP2050_HOURLY_URL = "https://pubdb.bfe.admin.ch/de/publication/download/11142"
EP2050_PUBLICATION_DATE = pd.Timestamp("2022-10-01", tz="UTC")
DEFAULT_LOCAL_EP2050_DIR = Path.home() / "pfc_local_data" / "ep2050"

TECHNOLOGY_MAP = {
    "Laufwasserkraft 1)": "run_of_river_hydro",
    "Speicherkraft-werke": "storage_hydro",
    "Pumpspeicher gesamt (PSW)": "pumped_storage_generation",
    "Kernkraftwerke": "nuclear",
    "Photovoltaik 2)": "pv",
    "Wind": "wind",
    "Biomasse 3)": "biomass",
    "Sonstige 4)": "other_generation",
    "Importe": "imports",
    "Exporte": "exports",
    "Importsaldo": "net_imports",
    "EE-Abregelung": "renewable_curtailment",
    "Nachfrage \ninflexibel 5)": "inflexible_demand",
    "Nachfrage inflexibel 5)": "inflexible_demand",
    "Elektrolyse flexibel": "flexible_electrolysis",
    "PSW Verbrauch 6)": "pumped_storage_consumption",
    "E-Mobilität flexibel": "flexible_ev",
    "Wärmepumpe flexibel": "flexible_heatpump",
    "Summe Verbrauch": "total_consumption",
}

__all__ = [
    "EP2050_HOURLY_URL",
    "EP2050_PUBLICATION_DATE",
    "DEFAULT_LOCAL_EP2050_DIR",
    "download_ep2050_hourly_zip",
    "extract_ep2050_hourly_zip",
    "parse_ep2050_hourly_workbook",
    "build_ep2050_hourly",
    "derive_ep2050_annual_scenarios",
]


def download_ep2050_hourly_zip(
    root: str | Path = DEFAULT_LOCAL_EP2050_DIR,
    *,
    url: str = EP2050_HOURLY_URL,
) -> Path:
    """Download the official EP2050+ hourly ZIP to a local cache directory."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "ep2050_hourly_power_generation_consumption_11142.zip"
    if not path.exists():
        urlretrieve(url, path)
    return path


def extract_ep2050_hourly_zip(zip_path: str | Path, output_dir: str | Path | None = None) -> Path:
    """Extract the EP2050+ hourly ZIP and return the extraction directory."""
    zip_path = Path(zip_path)
    output_dir = Path(output_dir) if output_dir is not None else zip_path.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / ".extracted"
    if not marker.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(output_dir)
        marker.write_text(str(zip_path), encoding="utf-8")
    return output_dir


def _scenario_from_filename(path: Path) -> str:
    name = path.stem
    if "_WWB_" in name:
        return "WWB"
    if "_ZERO Basis_" in name:
        if "_KKW60_" in name:
            return "ZERO_Basis_KKW60"
        return "ZERO_Basis"
    return name


def _sheet_years(path: Path) -> list[str]:
    xl = pd.ExcelFile(path)
    return [s for s in xl.sheet_names if re.fullmatch(r"\d{4}", str(s))]


def _parse_year_sheet(path: Path, sheet: str, *, scenario: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    header_rows = raw.index[raw.apply(lambda r: r.astype(str).str.strip().eq("Stunde").any(), axis=1)]
    if len(header_rows) == 0:
        raise ValueError(f"Could not find EP2050 hourly header row in {path.name} sheet={sheet}")
    header_row = int(header_rows[0])
    header = raw.iloc[header_row].copy()
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = [str(c).strip() if pd.notna(c) else "" for c in header]
    data = data.loc[:, [c for c in data.columns if c and c != "nan"]]
    if "Stunde" not in data.columns:
        raise ValueError(f"Missing Stunde column in {path.name} sheet={sheet}")
    data["Stunde"] = pd.to_numeric(data["Stunde"], errors="coerce")
    data = data[data["Stunde"].notna()].copy()
    data = data[(data["Stunde"] >= 1) & (data["Stunde"] <= 8784)]
    year = int(sheet)
    base = pd.Timestamp(f"{year}-01-01 00:00:00", tz="Europe/Zurich")
    data["datetime"] = [base + pd.Timedelta(hours=int(h) - 1) for h in data["Stunde"]]

    value_cols = [c for c in data.columns if c not in {"Stunde", "datetime"}]
    long = data.melt(
        id_vars=["datetime"],
        value_vars=value_cols,
        var_name="technology_raw",
        value_name="gwh",
    )
    long["technology"] = long["technology_raw"].map(lambda x: TECHNOLOGY_MAP.get(str(x), str(x)))
    long["gwh"] = pd.to_numeric(long["gwh"], errors="coerce")
    long = long[long["gwh"].notna()].copy()
    long["mwh"] = long["gwh"] * 1000.0
    long["scenario"] = scenario
    long["year"] = year
    long["publication_date"] = EP2050_PUBLICATION_DATE
    long["source"] = "OFEN_EP2050_hourly_11142"
    return long[
        [
            "publication_date",
            "source",
            "scenario",
            "year",
            "datetime",
            "technology",
            "technology_raw",
            "gwh",
            "mwh",
        ]
    ].reset_index(drop=True)


def parse_ep2050_hourly_workbook(path: str | Path, *, years: list[int] | None = None) -> pd.DataFrame:
    """Parse one EP2050+ hourly workbook into long format."""
    path = Path(path)
    scenario = _scenario_from_filename(path)
    sheets = _sheet_years(path)
    if years is not None:
        wanted = {str(y) for y in years}
        sheets = [s for s in sheets if s in wanted]
    frames = [_parse_year_sheet(path, sheet, scenario=scenario) for sheet in sheets]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_ep2050_hourly(
    extracted_dir: str | Path,
    *,
    years: list[int] | None = None,
    scenarios: list[str] | None = None,
) -> pd.DataFrame:
    """Parse complete-structure EP2050+ hourly workbooks under `extracted_dir`."""
    extracted_dir = Path(extracted_dir)
    frames: list[pd.DataFrame] = []
    for path in sorted(extracted_dir.rglob("*.xlsx")):
        scenario = _scenario_from_filename(path)
        if scenario.endswith("KKW60"):
            continue
        if scenarios is not None and scenario not in set(scenarios):
            continue
        frames.append(parse_ep2050_hourly_workbook(path, years=years))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def derive_ep2050_annual_scenarios(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate EP2050+ hourly data into the scenario inventory schema."""
    if hourly.empty:
        return pd.DataFrame()
    sums = (
        hourly.groupby(["publication_date", "source", "scenario", "year", "technology"])["gwh"]
        .sum()
        .unstack("technology")
        .fillna(0.0)
    )
    peaks = (
        hourly.groupby(["publication_date", "source", "scenario", "year", "technology"])["gwh"]
        .max()
        .unstack("technology")
        .fillna(0.0)
    )
    rows: list[dict[str, object]] = []
    for idx, row in sums.iterrows():
        publication_date, source, scenario, year = idx
        peak_row = peaks.loc[idx]
        demand_twh = float(row.get("total_consumption", 0.0)) / 1000.0
        if demand_twh <= 0.0:
            continue
        hydro_peak_gw = float(
            peak_row.get("run_of_river_hydro", 0.0)
            + peak_row.get("storage_hydro", 0.0)
            + peak_row.get("pumped_storage_generation", 0.0)
        )
        net_import_twh = float(row.get("net_imports", row.get("imports", 0.0) - row.get("exports", 0.0))) / 1000.0
        rows.append(
            {
                "publication_date": publication_date,
                "source": source,
                "scenario": scenario,
                "country": "CH",
                "delivery_year": int(year),
                "delivery_month": pd.NA,
                "quality_flag": "official",
                "demand_twh": demand_twh,
                "peak_load_gw": float(peak_row.get("total_consumption", 0.0)),
                "pv_twh": float(row.get("pv", 0.0)) / 1000.0,
                "pv_gw": float(peak_row.get("pv", 0.0)),
                "wind_twh": float(row.get("wind", 0.0)) / 1000.0,
                "wind_gw": float(peak_row.get("wind", 0.0)),
                "hydro_twh": float(
                    row.get("run_of_river_hydro", 0.0)
                    + row.get("storage_hydro", 0.0)
                    + row.get("pumped_storage_generation", 0.0)
                ) / 1000.0,
                "hydro_capacity_gw": hydro_peak_gw,
                "nuclear_twh": float(row.get("nuclear", 0.0)) / 1000.0,
                "nuclear_gw": float(peak_row.get("nuclear", 0.0)),
                "ev_twh": float(row.get("flexible_ev", 0.0)) / 1000.0,
                "heatpump_twh": float(row.get("flexible_heatpump", 0.0)) / 1000.0,
                "import_twh": float(row.get("imports", 0.0)) / 1000.0,
                "export_twh": float(row.get("exports", 0.0)) / 1000.0,
                "net_import_twh": net_import_twh,
            }
        )
    return pd.DataFrame(rows)

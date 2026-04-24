"""Gemeinsame Hilfsfunktionen und Datenlader für die FMV Deviwa-Demo.

Alles bleibt lokal und dateibasiert, damit die Demo offline läuft.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PFC_DATA_DIR = ROOT / "pfc_shaping" / "data"
OUTPUT_DIR = ROOT / "pfc_shaping" / "output"

FMV_BLUE = "#0F52CC"
FMV_NAVY = "#0E1F3D"
FMV_ACCENT = "#F5B700"
FMV_GREEN = "#1E8A4C"
FMV_RED = "#C0392B"
FMV_GREY = "#6B7A99"


# ---------------------------------------------------------------------------
# Formatierung (Schweizer Konvention)
# ---------------------------------------------------------------------------

def fmt_chf(value: float, decimals: int = 0) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    sign = "-" if value < 0 else ""
    abs_val = abs(float(value))
    whole, frac = divmod(abs_val, 1.0)
    whole_str = f"{int(whole):,}".replace(",", "'")
    if decimals == 0:
        return f"{sign}{whole_str}"
    frac_str = f"{frac:.{decimals}f}"[2:]
    return f"{sign}{whole_str}.{frac_str}"


def fmt_eur_mwh(value: float, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{fmt_chf(value, decimals=decimals)} EUR/MWh"


def fmt_mwh(value: float, decimals: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{fmt_chf(value, decimals=decimals)} MWh"


def fmt_delta(value: float, suffix: str = "") -> str:
    if value is None or not np.isfinite(value):
        return "—"
    sign = "+" if value >= 0 else ""
    return f"{sign}{fmt_chf(value, decimals=2)}{suffix}"


# ---------------------------------------------------------------------------
# Daten-Loader (mit Caching)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_epex_ch() -> pd.DataFrame:
    path = PFC_DATA_DIR / "epex_15min.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df.rename(columns={"price_eur_mwh": "price"})
    df = df[["price"]]
    df.index = df.index.tz_convert("Europe/Zurich")
    return df


@st.cache_data(show_spinner=False)
def load_epex_de() -> pd.DataFrame:
    path = PFC_DATA_DIR / "epex_de_15min.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = df.index.tz_convert("Europe/Zurich")
    return df


@st.cache_data(show_spinner=False)
def load_entso() -> pd.DataFrame:
    path = PFC_DATA_DIR / "entso_15min.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = df.index.tz_convert("Europe/Zurich")
    return df


@st.cache_data(show_spinner=False)
def load_hydro() -> pd.DataFrame:
    path = PFC_DATA_DIR / "hydro_reservoir.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = df.index.tz_convert("Europe/Zurich")
    return df


@st.cache_data(show_spinner=False)
def load_eex_forwards() -> pd.DataFrame:
    path = DATA_DIR / "eex_forwards_history.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_commodities() -> pd.DataFrame:
    path = DATA_DIR / "commodities_cache.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.columns = [c.split("|")[0].strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_lear_forecast() -> pd.DataFrame:
    path = OUTPUT_DIR / "lear_forecast_latest.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Zurich")
    return df


@st.cache_data(show_spinner=False)
def load_lear_backtest() -> pd.DataFrame:
    path = OUTPUT_DIR / "lear_backtest_latest.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["forecast_ts"] = pd.to_datetime(df["forecast_ts"], utc=True)
    df["forecast_ts"] = df["forecast_ts"].dt.tz_convert("Europe/Zurich")
    return df


@st.cache_data(show_spinner=False)
def load_pfc() -> pd.DataFrame:
    """Letzte verfügbare PFC 15-Minuten-Kurve."""
    candidates = sorted(OUTPUT_DIR.glob("pfc_15min_*.parquet"))
    if not candidates:
        return pd.DataFrame()
    df = pd.read_parquet(candidates[-1])
    df.index = pd.to_datetime(df.index).tz_convert("Europe/Zurich")
    return df


# ---------------------------------------------------------------------------
# Gemeinsame UI-Bausteine
# ---------------------------------------------------------------------------

def render_header(page_title: str) -> None:
    st.markdown(
        f"""
        <div style='display:flex; align-items:center; justify-content:space-between;
                    border-bottom: 3px solid {FMV_BLUE}; padding-bottom: 0.6rem;
                    margin-bottom: 1.2rem;'>
            <div>
                <div style='font-size:0.85rem; letter-spacing:0.12em; color:{FMV_GREY};
                            text-transform:uppercase;'>FMV – Forces Motrices Valaisannes</div>
                <div style='font-size:1.8rem; font-weight:700; color:{FMV_NAVY};'>{page_title}</div>
            </div>
            <div style='text-align:right; font-size:0.8rem; color:{FMV_GREY};'>
                Deviwa Energiepool · Demo<br/>
                Stand: {pd.Timestamp.now(tz='Europe/Zurich').strftime('%d.%m.%Y %H:%M')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str | None = None,
             delta_color: str = FMV_NAVY, help_text: str | None = None) -> str:
    delta_html = (
        f"<div style='font-size:0.9rem; color:{delta_color}; margin-top:0.2rem;'>{delta}</div>"
        if delta else ""
    )
    tooltip = f" title='{help_text}'" if help_text else ""
    return f"""
    <div{tooltip} style='background:{"#FFFFFF"}; border:1px solid #E5EBF4;
                border-radius:10px; padding:0.9rem 1.1rem; height:100%;
                box-shadow: 0 1px 2px rgba(14,31,61,0.04);'>
        <div style='font-size:0.78rem; color:{FMV_GREY}; text-transform:uppercase;
                    letter-spacing:0.08em;'>{label}</div>
        <div style='font-size:1.55rem; font-weight:700; color:{FMV_NAVY}; margin-top:0.25rem;'>{value}</div>
        {delta_html}
    </div>
    """


# ---------------------------------------------------------------------------
# Gemeinsamer Deviwa-Loader und Akteur-Auswahl
# ---------------------------------------------------------------------------

DEVIWA_CANDIDATES = [
    "Deviwa.xlsx", "Deviwa.csv", "deviwa.xlsx", "deviwa.csv",
    "Deviwa_Deals.xlsx", "deviwa_deals.xlsx",
]


def find_deviwa_file() -> Path | None:
    for name in DEVIWA_CANDIDATES:
        p = DATA_DIR / name
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def _load_deviwa_cached(path_str: str, mtime: float) -> dict[str, pd.DataFrame]:
    from deviwa_parser import load_deviwa_file  # type: ignore
    return load_deviwa_file(path_str)


def load_deviwa_auto() -> dict[str, pd.DataFrame]:
    """Lädt Deviwa-Datei automatisch aus /data. Leer wenn nicht vorhanden."""
    p = find_deviwa_file()
    if p is None:
        return {}
    data = _load_deviwa_cached(str(p), p.stat().st_mtime)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def render_actor_selector(
    data: dict[str, pd.DataFrame],
    label: str = "Akteur auswählen",
    include_pool: bool = True,
    key: str | None = None,
) -> tuple[str, pd.DataFrame]:
    """Zeigt einen konsistenten Akteur-Selector und gibt (choice, filtered_df) zurück."""
    actors = [a for a in data.keys() if not a.startswith("_")]
    options = (["Gesamter Pool"] if include_pool else []) + actors
    if not options:
        st.warning("Keine Akteure gefunden.")
        return ("", pd.DataFrame())
    choice = st.radio(label, options=options, horizontal=True, key=key)
    if choice == "Gesamter Pool":
        frames = [v.assign(_actor=k) if "_actor" not in v.columns else v
                  for k, v in data.items() if v is not None and not v.empty]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        df = data.get(choice, pd.DataFrame()).copy()
    return choice, df


def freshness_badge(df: pd.DataFrame, label: str = "Datenstand") -> str:
    if df is None or df.empty:
        return f"<span style='color:{FMV_RED};'>❌ {label}: keine Daten</span>"
    if isinstance(df.index, pd.DatetimeIndex):
        last = df.index.max()
    else:
        ts_col = df.get("timestamp", df.get("date"))
        last = pd.to_datetime(ts_col.max()) if ts_col is not None else None
    if last is None or pd.isna(last):
        return f"<span style='color:{FMV_RED};'>❌ {label}: keine Daten</span>"
    ts = pd.Timestamp(last)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Europe/Zurich")
    else:
        ts = ts.tz_convert("Europe/Zurich")
    age_hours = (pd.Timestamp.now(tz="Europe/Zurich") - ts).total_seconds() / 3600
    color = FMV_GREEN if age_hours < 36 else (FMV_ACCENT if age_hours < 96 else FMV_RED)
    return (f"<span style='color:{color};'>● {label}: "
            f"{ts.strftime('%d.%m.%Y %H:%M')}</span>")

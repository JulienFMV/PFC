"""utils.py - Shared data loading, caching, and chart styling for the PFC dashboard."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

logger = logging.getLogger("pfc_dashboard")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

COLORS = {
    "bg": "#F3F6FB",
    "card": "#FFFFFF",
    "text": "#0E1F3D",
    "muted": "#5A6B8A",
    "navy": "#0B2E6F",
    "blue": "#0F52CC",
    "blue_soft": "#2E7BFF",
    "amber": "#D28B1A",
    "green": "#1F9D55",
    "red": "#D64545",
    "gray": "#CBD5E1",
    "band": "rgba(15, 82, 204, 0.12)",
    "band_border": "rgba(15, 82, 204, 0.35)",
}

PFC_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["card"],
        font=dict(family="Arial, sans-serif", color=COLORS["text"], size=13),
        title=dict(font=dict(size=20, color=COLORS["navy"]), x=0, xanchor="left"),
        xaxis=dict(
            gridcolor="#D9E2F1",
            zerolinecolor="#D9E2F1",
            showgrid=True,
            gridwidth=1,
        ),
        yaxis=dict(
            gridcolor="#D9E2F1",
            zerolinecolor="#D9E2F1",
            showgrid=True,
            gridwidth=1,
            title_standoff=10,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            bordercolor="rgba(255,255,255,0)",
            font=dict(size=12, color=COLORS["text"]),
        ),
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#EDF3FF", font_color=COLORS["text"]),
    )
)
pio.templates["pfc_fmv"] = PFC_TEMPLATE
pio.templates.default = "pfc_fmv"

DATA_DIR = PROJECT_ROOT / "pfc_shaping" / "data"
OUTPUT_DIR = PROJECT_ROOT / "pfc_shaping" / "output"

EPEX_PARQUET = DATA_DIR / "epex_15min.parquet"
ENTSO_PARQUET = DATA_DIR / "entso_15min.parquet"
HYDRO_PARQUET = DATA_DIR / "hydro_reservoir.parquet"
PFC_PARQUET = OUTPUT_DIR / "pfc_15min.parquet"


def _safe_read_parquet(path: Path, name: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            logger.warning("%s Parquet exists but is empty: %s", name, path)
            return None
        return df
    except Exception as exc:
        logger.error("Failed to read %s Parquet at %s: %s", name, path, exc)
        return None


@st.cache_data(ttl=3600, show_spinner="Chargement prix EPEX...")
def load_epex() -> pd.DataFrame | None:
    return _safe_read_parquet(EPEX_PARQUET, "EPEX")


@st.cache_data(ttl=3600, show_spinner="Chargement load/gen...")
def load_entso() -> pd.DataFrame | None:
    return _safe_read_parquet(ENTSO_PARQUET, "ENTSO")


@st.cache_data(ttl=3600, show_spinner="Chargement reservoirs hydro...")
def load_hydro() -> pd.DataFrame | None:
    df = _safe_read_parquet(HYDRO_PARQUET, "Hydro")
    if df is not None:
        return df
    try:
        from pfc_shaping.data.ingest_hydro import load_from_sfoe, build_water_value

        df = load_from_sfoe()
        return build_water_value(df)
    except Exception as exc:
        logger.error("SFOE hydro fallback failed: %s", exc)
        return None


@st.cache_data(ttl=3600, show_spinner="Chargement PFC...")
def load_pfc() -> pd.DataFrame | None:
    return _safe_read_parquet(PFC_PARQUET, "PFC")


@st.cache_data(ttl=86400, show_spinner="Chargement config...")
def load_config() -> dict:
    import yaml

    cfg_path = PROJECT_ROOT / "pfc_shaping" / "config.yaml"
    try:
        with open(cfg_path, encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        return {}


def data_freshness() -> dict[str, str]:
    result = {}
    for name, path in [
        ("EPEX", EPEX_PARQUET),
        ("Load/Gen", ENTSO_PARQUET),
        ("Hydro", HYDRO_PARQUET),
        ("PFC", PFC_PARQUET),
    ]:
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            result[name] = mtime.strftime("%d/%m/%Y %H:%M")
        else:
            result[name] = "--"
    return result


def show_freshness_sidebar() -> None:
    freshness = data_freshness()
    with st.sidebar:
        st.markdown("##### Derniere mise a jour")
        for name, ts in freshness.items():
            color = COLORS["green"] if ts != "--" else COLORS["red"]
            st.markdown(
                f'<span style="color:{color};">●</span> **{name}** : {ts}',
                unsafe_allow_html=True,
            )
        if st.button("Rafraichir le cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


def format_eur(v: float, decimals: int = 1) -> str:
    return f"{v:,.{decimals}f} EUR/MWh"


def format_pct(v: float, decimals: int = 1) -> str:
    return f"{v:,.{decimals}f}%"


def format_gwh(v: float, decimals: int = 0) -> str:
    return f"{v:,.{decimals}f} GWh"


def add_range_slider(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(
        rangeslider=dict(visible=True, bgcolor="#E7EEF9", thickness=0.05),
        rangeselector=dict(
            buttons=[
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(label="All", step="all"),
            ],
            bgcolor="#E7EEF9",
            activecolor=COLORS["blue"],
            font=dict(color=COLORS["text"], size=11),
        ),
    )
    return fig


def no_data_warning(name: str = "donnees") -> None:
    st.warning(
        f"Aucune {name} disponible. Lance `python -m pfc_shaping.pipeline.rolling_update` "
        "pour ingerer les donnees.",
        icon=":material/warning:",
    )


def export_csv_button(df: pd.DataFrame, filename: str, label: str = "Export CSV") -> None:
    csv = df.to_csv()
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )

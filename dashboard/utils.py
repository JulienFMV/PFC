"""
utils.py — Shared data loading, caching, and chart styling for the PFC dashboard.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

logger = logging.getLogger("pfc_dashboard")

# ── Project path ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Plotly template ───────────────────────────────────────────────────────
COLORS = {
    "bg": "#0B0E11",
    "card": "#161B22",
    "text": "#E6EDF3",
    "amber": "#FF9F1C",
    "blue": "#58A6FF",
    "green": "#3FB950",
    "red": "#F85149",
    "gray": "#484F58",
    "muted": "#8B949E",
    "band": "rgba(88,166,255,0.12)",
    "band_border": "rgba(88,166,255,0.3)",
}

PFC_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(family="Inter, sans-serif", color=COLORS["text"], size=13),
        title=dict(font=dict(size=18, color=COLORS["text"]), x=0, xanchor="left"),
        xaxis=dict(
            gridcolor="#21262D", zerolinecolor="#21262D",
            showgrid=True, gridwidth=1,
        ),
        yaxis=dict(
            gridcolor="#21262D", zerolinecolor="#21262D",
            showgrid=True, gridwidth=1,
            title_standoff=10,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=COLORS["card"], font_color=COLORS["text"]),
    )
)
pio.templates["pfc_dark"] = PFC_TEMPLATE
pio.templates.default = "pfc_dark"

# ── Parquet paths ─────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "pfc_shaping" / "data"
OUTPUT_DIR = PROJECT_ROOT / "pfc_shaping" / "output"

EPEX_PARQUET = DATA_DIR / "epex_15min.parquet"
ENTSO_PARQUET = DATA_DIR / "entso_15min.parquet"
HYDRO_PARQUET = DATA_DIR / "hydro_reservoir.parquet"
PFC_PARQUET = OUTPUT_DIR / "pfc_15min.parquet"


# ── Safe Parquet reader ───────────────────────────────────────────────────

def _safe_read_parquet(path: Path, name: str) -> pd.DataFrame | None:
    """Read a Parquet file with error handling for corruption."""
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            logger.warning("%s Parquet exists but is empty: %s", name, path)
            return None
        return df
    except Exception as e:
        logger.error("Failed to read %s Parquet at %s: %s", name, path, e)
        return None


# ── Cached loaders ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Chargement prix EPEX...")
def load_epex() -> pd.DataFrame | None:
    return _safe_read_parquet(EPEX_PARQUET, "EPEX")


@st.cache_data(ttl=3600, show_spinner="Chargement load/gen...")
def load_entso() -> pd.DataFrame | None:
    return _safe_read_parquet(ENTSO_PARQUET, "ENTSO")


@st.cache_data(ttl=3600, show_spinner="Chargement réservoirs hydro...")
def load_hydro() -> pd.DataFrame | None:
    df = _safe_read_parquet(HYDRO_PARQUET, "Hydro")
    if df is not None:
        return df
    # Fallback: download live from SFOE
    try:
        from pfc_shaping.data.ingest_hydro import load_from_sfoe, build_water_value
        df = load_from_sfoe()
        return build_water_value(df)
    except Exception as e:
        logger.error("SFOE hydro fallback failed: %s", e)
        return None


@st.cache_data(ttl=3600, show_spinner="Chargement PFC...")
def load_pfc() -> pd.DataFrame | None:
    return _safe_read_parquet(PFC_PARQUET, "PFC")


@st.cache_data(ttl=86400, show_spinner="Chargement config...")
def load_config() -> dict:
    import yaml
    cfg_path = PROJECT_ROOT / "pfc_shaping" / "config.yaml"
    try:
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        return {}


# ── Data freshness ────────────────────────────────────────────────────────

def data_freshness() -> dict[str, str]:
    """Return last-modified timestamps for each data file."""
    result = {}
    for name, path in [("EPEX", EPEX_PARQUET), ("Load/Gen", ENTSO_PARQUET),
                        ("Hydro", HYDRO_PARQUET), ("PFC", PFC_PARQUET)]:
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            result[name] = mtime.strftime("%d/%m/%Y %H:%M")
        else:
            result[name] = "—"
    return result


def show_freshness_sidebar() -> None:
    """Display data freshness info in sidebar."""
    freshness = data_freshness()
    with st.sidebar:
        st.markdown("##### Dernière mise à jour")
        for name, ts in freshness.items():
            color = COLORS["green"] if ts != "—" else COLORS["red"]
            st.markdown(
                f'<span style="color:{color};">●</span> **{name}** : {ts}',
                unsafe_allow_html=True,
            )
        if st.button("Rafraîchir le cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


# ── Chart helpers ─────────────────────────────────────────────────────────

def format_eur(v: float, decimals: int = 1) -> str:
    return f"{v:,.{decimals}f} EUR/MWh"


def format_pct(v: float, decimals: int = 1) -> str:
    return f"{v:,.{decimals}f}%"


def format_gwh(v: float, decimals: int = 0) -> str:
    return f"{v:,.{decimals}f} GWh"


def add_range_slider(fig: go.Figure) -> go.Figure:
    """Add a range slider to the x-axis for time navigation."""
    fig.update_xaxes(
        rangeslider=dict(visible=True, bgcolor=COLORS["card"], thickness=0.04),
        rangeselector=dict(
            buttons=[
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(label="All", step="all"),
            ],
            bgcolor=COLORS["card"],
            activecolor=COLORS["amber"],
            font=dict(color=COLORS["text"], size=11),
        ),
    )
    return fig


def no_data_warning(name: str = "données") -> None:
    st.warning(
        f"Aucune {name} disponible. Lance `python -m pfc_shaping.pipeline.rolling_update` "
        "pour ingérer les données.",
        icon="⚠️",
    )


def export_csv_button(df: pd.DataFrame, filename: str, label: str = "Export CSV") -> None:
    """Add a CSV download button for the given DataFrame."""
    csv = df.to_csv()
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )

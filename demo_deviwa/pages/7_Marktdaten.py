"""Seite 4 – Marktdaten CH/DE (Last, Erneuerbare, Grenzflüsse, Hydrospeicher)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    FMV_ACCENT,
    FMV_BLUE,
    FMV_GREEN,
    FMV_GREY,
    FMV_NAVY,
    FMV_RED,
    freshness_badge,
    kpi_card,
    load_entso,
    load_hydro,
    render_header,
)

st.set_page_config(
    page_title="FMV · Marktdaten",
    page_icon="🌍",
    layout="wide",
)

render_header("Marktdaten · ENTSO-E und Hydrospeicher")

entso = load_entso()
hydro = load_hydro()

if entso.empty:
    st.error("ENTSO-E-Daten nicht verfügbar.")
    st.stop()

# ---------------------------------------------------------------------------
# Zeitraumfilter
# ---------------------------------------------------------------------------
max_ts = entso.index.max()
range_choice = st.radio(
    "Zeitraum",
    options=["Letzte 24 h", "Letzte 7 Tage", "Letzte 30 Tage", "Letztes Jahr"],
    horizontal=True,
    index=1,
)
lookback = {
    "Letzte 24 h": pd.Timedelta(hours=24),
    "Letzte 7 Tage": pd.Timedelta(days=7),
    "Letzte 30 Tage": pd.Timedelta(days=30),
    "Letztes Jahr": pd.Timedelta(days=365),
}[range_choice]
start = max_ts - lookback
entso_sl = entso.loc[entso.index >= start]

# ---------------------------------------------------------------------------
# KPI-Zeile
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
avg_load = float(entso_sl["load_mw"].mean()) if "load_mw" in entso_sl.columns else np.nan
peak_load = float(entso_sl["load_mw"].max()) if "load_mw" in entso_sl.columns else np.nan
solar_share = np.nan
if "solar_mw" in entso_sl.columns and "load_mw" in entso_sl.columns:
    solar_share = 100.0 * entso_sl["solar_mw"].sum() / max(entso_sl["load_mw"].sum(), 1e-9)

fill_pct = np.nan
wallis_gwh = np.nan
if not hydro.empty:
    if "fill_pct" in hydro.columns:
        fill_pct = float(hydro["fill_pct"].dropna().iloc[-1])
    if "wallis_gwh" in hydro.columns:
        wallis_gwh = float(hydro["wallis_gwh"].dropna().iloc[-1])

c1.markdown(kpi_card("Last CH (Ø)", f"{avg_load:,.0f} MW".replace(",", "'") if np.isfinite(avg_load) else "—",
                     delta=f"Spitze: {peak_load:,.0f} MW".replace(",", "'") if np.isfinite(peak_load) else None,
                     delta_color=FMV_NAVY),
            unsafe_allow_html=True)
c2.markdown(kpi_card("Solar-Anteil CH", f"{solar_share:.1f} %" if np.isfinite(solar_share) else "—",
                     delta=range_choice, delta_color=FMV_ACCENT),
            unsafe_allow_html=True)
c3.markdown(kpi_card("CH Speicher Füllstand",
                     f"{fill_pct:.1f} %" if np.isfinite(fill_pct) else "—",
                     delta="gesamt", delta_color=FMV_BLUE),
            unsafe_allow_html=True)
c4.markdown(kpi_card("Wallis Speicher",
                     f"{wallis_gwh:,.0f} GWh".replace(",", "'") if np.isfinite(wallis_gwh) else "—",
                     delta="55 % der CH-Kapazität", delta_color=FMV_GREEN),
            unsafe_allow_html=True)

st.markdown("")

# ---------------------------------------------------------------------------
# Chart: Last + Erneuerbare CH
# ---------------------------------------------------------------------------
st.subheader("Last und Erzeugung CH")
plot_df = entso_sl[[c for c in ["load_mw", "solar_mw", "wind_mw", "hydro_ror_mw", "nuclear_mw"]
                   if c in entso_sl.columns]].resample("h").mean()

fig = go.Figure()
if "load_mw" in plot_df.columns:
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df["load_mw"],
        mode="lines", name="Last",
        line=dict(color=FMV_NAVY, width=2.5),
    ))
stacks = []
for col, name, color in [
    ("nuclear_mw", "Kernkraft", "#7B4EA8"),
    ("hydro_ror_mw", "Laufwasser", FMV_BLUE),
    ("wind_mw", "Wind", FMV_GREEN),
    ("solar_mw", "Solar", FMV_ACCENT),
]:
    if col in plot_df.columns:
        stacks.append((col, name, color))
for col, name, color in stacks:
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df[col],
        mode="lines", name=name,
        stackgroup="gen",
        line=dict(color=color, width=0.5),
        fillcolor=color,
    ))

fig.update_layout(
    height=420, margin=dict(l=20, r=20, t=10, b=20),
    plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(title="MW", gridcolor="#EEF1F6"),
    xaxis=dict(gridcolor="#EEF1F6"),
    legend=dict(orientation="h", y=-0.15),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Grenzflüsse
# ---------------------------------------------------------------------------
st.subheader("Grenzflüsse CH (Netto-Export positiv, Import negativ)")
flows = [c for c in ["scheduled_net_export_ch_de_mw", "scheduled_net_export_ch_fr_mw",
                     "scheduled_net_export_ch_at_mw", "scheduled_net_export_ch_it_mw"]
         if c in entso_sl.columns]

if flows:
    flow_df = entso_sl[flows].resample("h").mean()
    fig_flow = go.Figure()
    for col, name, color in [
        ("scheduled_net_export_ch_de_mw", "CH→DE", FMV_BLUE),
        ("scheduled_net_export_ch_fr_mw", "CH→FR", FMV_ACCENT),
        ("scheduled_net_export_ch_at_mw", "CH→AT", FMV_GREEN),
        ("scheduled_net_export_ch_it_mw", "CH→IT", "#8E44AD"),
    ]:
        if col in flow_df.columns:
            fig_flow.add_trace(go.Scatter(
                x=flow_df.index, y=flow_df[col],
                mode="lines", name=name,
                line=dict(color=color, width=1.8),
            ))
    fig_flow.add_hline(y=0, line_width=1, line_dash="dash", line_color=FMV_GREY)
    fig_flow.update_layout(
        height=340, margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="MW", gridcolor="#EEF1F6"),
        xaxis=dict(gridcolor="#EEF1F6"),
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )
    st.plotly_chart(fig_flow, use_container_width=True)

# ---------------------------------------------------------------------------
# Hydrospeicher
# ---------------------------------------------------------------------------
if not hydro.empty:
    st.subheader("CH Hydrospeicher · Füllstand")
    col_left, col_right = st.columns(2)
    with col_left:
        if "fill_pct" in hydro.columns:
            s = hydro["fill_pct"].dropna().tail(365)
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines",
                name="Füllstand %", line=dict(color=FMV_BLUE, width=2.2),
                fill="tozeroy", fillcolor="rgba(15,82,204,0.15)",
            ))
            fig_h.update_layout(
                height=320, margin=dict(l=20, r=20, t=10, b=20),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="%", gridcolor="#EEF1F6", range=[0, 100]),
                xaxis=dict(gridcolor="#EEF1F6"),
            )
            st.plotly_chart(fig_h, use_container_width=True)
    with col_right:
        regional = [c for c in ["wallis_gwh", "graubuenden_gwh", "tessin_gwh"] if c in hydro.columns]
        if regional:
            last_row = hydro[regional].dropna(how="all").iloc[-1]
            fig_pie = go.Figure(go.Pie(
                labels=["Wallis", "Graubünden", "Tessin"][:len(regional)],
                values=[float(last_row[c]) for c in regional],
                marker=dict(colors=[FMV_BLUE, FMV_ACCENT, FMV_GREEN]),
                hole=0.45,
            ))
            fig_pie.update_layout(
                height=320, margin=dict(l=20, r=20, t=10, b=20),
                paper_bgcolor="white",
                showlegend=True,
                legend=dict(orientation="h", y=-0.1),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

st.divider()
cols = st.columns(2)
cols[0].markdown(freshness_badge(entso, "ENTSO-E"), unsafe_allow_html=True)
cols[1].markdown(freshness_badge(hydro, "Hydro (BFE)"), unsafe_allow_html=True)

st.caption(
    "Quelle: ENTSO-E Transparency Platform · BFE Stauseen-Statistik · "
    "Aggregation und Aufbereitung durch FMV."
)

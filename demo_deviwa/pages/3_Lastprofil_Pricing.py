"""Seite 2 – Lastprofil-Pricing gegen PFC."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pricing import parse_load_file, price_load_against_pfc
from utils import (
    FMV_ACCENT,
    FMV_BLUE,
    FMV_GREEN,
    FMV_GREY,
    FMV_NAVY,
    FMV_RED,
    fmt_chf,
    fmt_eur_mwh,
    fmt_mwh,
    kpi_card,
    load_pfc,
    render_header,
)

st.set_page_config(
    page_title="FMV · Lastprofil-Pricing",
    page_icon="📊",
    layout="wide",
)

render_header("Lastprofil-Pricing gegen PFC")

st.markdown(
    "Laden Sie eine Lastkurve hoch (CSV oder Excel mit Zeitstempel und Last in MWh). "
    "Das Profil wird gegen die aktuelle **FMV Price Forward Curve** bepreist."
)

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Lastprofil hochladen (.csv / .xlsx)",
    type=["csv", "xlsx", "xls"],
    key="load_upload",
)

use_example = st.checkbox(
    "Beispielprofil verwenden (typischer Industrieverbraucher Haut-Valais, 20 GWh/Jahr)",
    value=uploaded is None,
)

pfc = load_pfc()
if pfc.empty:
    st.error("PFC nicht verfügbar. Bitte PFC-Lauf ausführen.")
    st.stop()

# ---------------------------------------------------------------------------
# Profil aufbereiten
# ---------------------------------------------------------------------------
load_hourly: pd.Series | None = None
meta: dict = {}

if uploaded is not None:
    try:
        load_hourly, meta = parse_load_file(uploaded)
        st.success(f"Datei geladen: {uploaded.name} · {meta['rows']} Zeilen · "
                   f"{meta['start'].strftime('%d.%m.%Y')} bis {meta['end'].strftime('%d.%m.%Y')}")
    except Exception as exc:
        st.error(f"Fehler beim Parsen: {exc}")
        st.stop()
elif use_example:
    # Synthetisches Beispielprofil: 20 GWh/Jahr, typischer Industrielastgang
    pfc_start = pfc.index.min()
    idx = pd.date_range(start=pfc_start, periods=8760, freq="h", tz="Europe/Zurich")
    # Basis 2.0 MW · + Tagesgang · + Wochengang
    base_mw = 2.0
    hour_shape = np.array([
        0.65, 0.60, 0.58, 0.58, 0.60, 0.75, 0.95, 1.15, 1.25, 1.30,
        1.30, 1.28, 1.25, 1.22, 1.22, 1.22, 1.25, 1.20, 1.10, 1.00,
        0.90, 0.80, 0.72, 0.68,
    ])
    values = np.array([
        base_mw * hour_shape[t.hour] * (0.75 if t.dayofweek >= 5 else 1.0)
        for t in idx
    ])
    # Saisonalität (Winter höher)
    seasonal = 1.0 + 0.18 * np.cos(2 * np.pi * (idx.dayofyear - 15) / 365)
    values = values * seasonal
    load_hourly = pd.Series(values, index=idx, name="load_mwh")
    meta = {
        "rows": len(load_hourly),
        "start": load_hourly.index.min(),
        "end": load_hourly.index.max(),
        "annual_mwh": float(load_hourly.sum()),
        "unit": "MWh (synthetisch)",
        "inferred_freq": "h",
    }
else:
    st.info("Bitte laden Sie ein Lastprofil hoch oder nutzen Sie das Beispielprofil.")
    st.stop()

# ---------------------------------------------------------------------------
# Bepreisung
# ---------------------------------------------------------------------------
result = price_load_against_pfc(load_hourly, pfc)

# ---------------------------------------------------------------------------
# KPI-Zeile
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(
    kpi_card("Jahresvolumen",
             fmt_mwh(result["total_mwh"], decimals=0),
             delta=f"{result['coverage_hours']:,} Stunden".replace(",", "'"),
             delta_color=FMV_GREY),
    unsafe_allow_html=True,
)
c2.markdown(
    kpi_card("Profilpreis",
             fmt_eur_mwh(result["profile_price_eur_mwh"]),
             delta=f"Baseload: {fmt_eur_mwh(result['baseload_price_eur_mwh'])}",
             delta_color=FMV_BLUE),
    unsafe_allow_html=True,
)
c3.markdown(
    kpi_card("Profilaufschlag",
             fmt_eur_mwh(result["profile_vs_baseload_eur_mwh"]),
             delta="ggü. Baseload",
             delta_color=FMV_RED if result["profile_vs_baseload_eur_mwh"] > 0 else FMV_GREEN),
    unsafe_allow_html=True,
)
c4.markdown(
    kpi_card("Gesamtkosten",
             f"{fmt_chf(result['cost_total_eur'], decimals=0)} EUR",
             delta=f"Ø {fmt_eur_mwh(result['profile_price_eur_mwh'])}",
             delta_color=FMV_NAVY),
    unsafe_allow_html=True,
)
c5.markdown(
    kpi_card("Peak-Anteil",
             f"{result['peak_share_pct']:.1f} %",
             delta=f"Peak {fmt_eur_mwh(result['peak_price_eur_mwh'])} / Offpeak {fmt_eur_mwh(result['offpeak_price_eur_mwh'])}",
             delta_color=FMV_GREY),
    unsafe_allow_html=True,
)

if result.get("projection_used"):
    st.warning(
        "ℹ️ Das Lastprofil liegt nicht im PFC-Zeitraum. Das typische Wochenprofil "
        "wurde auf das erste PFC-Jahr projiziert und auf das geschätzte Jahresvolumen skaliert."
    )

st.markdown("")

# ---------------------------------------------------------------------------
# Monatliche Aufstellung
# ---------------------------------------------------------------------------
st.subheader("Monatliche Kosten und Volumen")

monthly = result["monthly"]
col_left, col_right = st.columns([1.3, 1])

with col_left:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["month"], y=monthly["load_mwh"],
        name="Volumen (MWh)",
        marker_color=FMV_BLUE, opacity=0.8,
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["profile_price"],
        name="Profilpreis (EUR/MWh)",
        line=dict(color=FMV_ACCENT, width=3),
        yaxis="y2",
        mode="lines+markers",
    ))
    fig.update_layout(
        height=380, margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#EEF1F6", title=""),
        yaxis=dict(title="Volumen (MWh)", gridcolor="#EEF1F6", side="left"),
        yaxis2=dict(title="Preis (EUR/MWh)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=-0.18),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    table = monthly.copy()
    table["Monat"] = table["month"].dt.strftime("%b %Y")
    table["Volumen (MWh)"] = table["load_mwh"].round(0)
    table["Profilpreis (EUR/MWh)"] = table["profile_price"].round(2)
    table["Kosten (EUR)"] = table["cost_eur"].round(0)
    show = table[["Monat", "Volumen (MWh)", "Profilpreis (EUR/MWh)", "Kosten (EUR)"]]
    st.dataframe(show, use_container_width=True, hide_index=True, height=380)

# ---------------------------------------------------------------------------
# Profil × PFC Overlay
# ---------------------------------------------------------------------------
st.subheader("Profil-Overlay: Last und PFC stündlich (erste 14 Tage)")

common = load_hourly.index.intersection(pfc["price_shape"].resample("h").mean().index)
if len(common) > 0:
    window_start = common.min()
    window_end = window_start + pd.Timedelta(days=14)
    sl_load = load_hourly.loc[(load_hourly.index >= window_start) & (load_hourly.index <= window_end)]
    sl_pfc = pfc["price_shape"].resample("h").mean()
    sl_pfc.index = sl_pfc.index.tz_convert("Europe/Zurich")
    sl_pfc = sl_pfc.loc[(sl_pfc.index >= window_start) & (sl_pfc.index <= window_end)]

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=sl_load.index, y=sl_load.values,
        name="Last (MWh/h)",
        marker_color=FMV_BLUE, opacity=0.6,
        yaxis="y1",
    ))
    fig3.add_trace(go.Scatter(
        x=sl_pfc.index, y=sl_pfc.values,
        name="PFC (EUR/MWh)",
        line=dict(color=FMV_NAVY, width=2),
        yaxis="y2",
    ))
    fig3.update_layout(
        height=360, margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#EEF1F6"),
        yaxis=dict(title="Last (MWh/h)", gridcolor="#EEF1F6"),
        yaxis2=dict(title="PFC (EUR/MWh)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=-0.18),
        hovermode="x unified",
    )
    st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.caption(
    "Bepreisung: jede Stunde der Last wird mit dem entsprechenden PFC-Stundenpreis multipliziert. "
    "Peak = 08:00–20:00 Mo–Fr (Schweizer Konvention). "
    "Der Profilaufschlag ist der Unterschied zwischen profilgewichtetem Preis und einfachem Baseload-Durchschnitt."
)

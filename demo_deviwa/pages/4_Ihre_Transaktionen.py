"""Seite 4 – Transaktionsübersicht Deviwa-Pool."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deviwa_parser import (
    deals_table,
    load_deviwa_file,
    monthly_breakdown,
    summarize_actor,
)
from utils import (
    FMV_ACCENT,
    FMV_BLUE,
    FMV_GREEN,
    FMV_GREY,
    FMV_NAVY,
    FMV_RED,
    fmt_chf,
    fmt_mwh,
    kpi_card,
    render_header,
    ROOT,
)

st.set_page_config(
    page_title="FMV · Ihre Transaktionen",
    page_icon="📒",
    layout="wide",
)

render_header("Ihre Transaktionen · Deviwa-Energiepool")


# ---------------------------------------------------------------------------
# Datei laden
# ---------------------------------------------------------------------------
DEFAULT_PATHS = [
    ROOT / "data" / "Deviwa.xlsx",
    ROOT / "data" / "Deviwa.csv",
    ROOT / "data" / "deviwa.xlsx",
    ROOT / "data" / "deviwa.csv",
]

found_path: Path | None = None
for p in DEFAULT_PATHS:
    if p.exists():
        found_path = p
        break

uploaded = None
with st.sidebar:
    st.markdown("**Datenquelle**")
    if found_path is not None:
        st.success(f"Automatisch geladen: `{found_path.name}`")
    else:
        st.info("Keine Deviwa-Datei in /data gefunden. Bitte hochladen.")
    uploaded = st.file_uploader(
        "Optional: andere Datei hochladen",
        type=["xlsx", "xls", "csv"],
        key="deviwa_upload",
    )

data_by_actor: dict[str, pd.DataFrame] = {}
if uploaded is not None:
    tmp = Path("/tmp") / uploaded.name
    tmp.write_bytes(uploaded.read())
    data_by_actor = load_deviwa_file(tmp)
elif found_path is not None:
    data_by_actor = load_deviwa_file(found_path)

# HPFC-Sheet aussortieren (wird hier nicht angezeigt)
data_by_actor = {k: v for k, v in data_by_actor.items() if not k.startswith("_")}

if not data_by_actor:
    st.warning(
        "Keine Transaktionen geladen. Bitte legen Sie die Datei `Deviwa.xlsx` in den Ordner `/data` "
        "oder laden Sie sie oben hoch."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Akteur-Auswahl
# ---------------------------------------------------------------------------
actors = list(data_by_actor.keys())
choice = st.radio(
    "Akteur auswählen",
    options=["Gesamter Pool"] + actors,
    horizontal=True,
)

frames_all = [v for v in data_by_actor.values() if v is not None and not v.empty]
if choice == "Gesamter Pool":
    if not frames_all:
        st.info("Keine Daten zu aggregieren.")
        st.stop()
    df = pd.concat(frames_all, ignore_index=True)
    title = "Gesamter Deviwa-Pool"
else:
    df = data_by_actor.get(choice)
    if df is None or df.empty:
        st.info(f"Keine Daten für {choice}.")
        st.stop()
    title = f"Akteur: {choice}"

st.markdown(f"### {title}")

# ---------------------------------------------------------------------------
# KPI-Zeile
# ---------------------------------------------------------------------------
summary = summarize_actor(df)

c1, c2, c3, c4 = st.columns(4)
c1.markdown(
    kpi_card("Anzahl Deals",
             f"{summary['n_deals']}",
             delta=f"Zeitraum: {summary['period_start'].strftime('%m/%Y') if summary['period_start'] is not None else '—'}"
                   f" – {summary['period_end'].strftime('%m/%Y') if summary['period_end'] is not None else '—'}",
             delta_color=FMV_GREY),
    unsafe_allow_html=True,
)
c2.markdown(
    kpi_card("Volumen gesamt",
             fmt_mwh(summary["total_volume_mwh"], decimals=0),
             delta=f"Einspeisung: {fmt_mwh(summary['intake_mwh'],0)} · "
                   f"Bezug: {fmt_mwh(summary['withdrawal_mwh'],0)}",
             delta_color=FMV_NAVY),
    unsafe_allow_html=True,
)
c3.markdown(
    kpi_card("PnL gesamt",
             f"{fmt_chf(summary['total_pnl_eur'], decimals=0)} EUR",
             delta="Summe aller Deals",
             delta_color=FMV_GREEN if summary["total_pnl_eur"] >= 0 else FMV_RED),
    unsafe_allow_html=True,
)
c4.markdown(
    kpi_card("Netto Position",
             fmt_mwh(summary["intake_mwh"] - summary["withdrawal_mwh"], decimals=0),
             delta="Einspeisung − Bezug",
             delta_color=FMV_BLUE),
    unsafe_allow_html=True,
)

st.markdown("")

# ---------------------------------------------------------------------------
# Monatliche Aufteilung
# ---------------------------------------------------------------------------
monthly = monthly_breakdown(df)

if not monthly.empty:
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Volumen pro Monat (Einspeisung vs. Bezug)")
        fig_vol = go.Figure()
        for scope, color in [("Einspeisung", FMV_GREEN), ("Bezug", FMV_BLUE)]:
            sub = monthly[monthly["scope_clean"] == scope]
            if not sub.empty:
                fig_vol.add_trace(go.Bar(
                    x=sub["month"], y=sub["volume_mwh"],
                    name=scope, marker_color=color,
                ))
        fig_vol.update_layout(
            barmode="group",
            height=360, margin=dict(l=20, r=20, t=10, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="MWh", gridcolor="#EEF1F6"),
            xaxis=dict(gridcolor="#EEF1F6"),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_r:
        st.subheader("PnL pro Monat")
        pnl_monthly = monthly.groupby("month")["pnl_eur"].sum().reset_index()
        colors = [FMV_GREEN if v >= 0 else FMV_RED for v in pnl_monthly["pnl_eur"]]
        fig_pnl = go.Figure(go.Bar(
            x=pnl_monthly["month"], y=pnl_monthly["pnl_eur"],
            marker_color=colors,
        ))
        fig_pnl.update_layout(
            height=360, margin=dict(l=20, r=20, t=10, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="EUR", gridcolor="#EEF1F6"),
            xaxis=dict(gridcolor="#EEF1F6"),
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

# ---------------------------------------------------------------------------
# Deal-Tabelle
# ---------------------------------------------------------------------------
st.subheader("Deal-Details")
table = deals_table(df)
if table.empty:
    st.info("Keine Deal-Details verfügbar.")
else:
    # Filter: Produkt + Richtung
    with st.expander("Filter"):
        fc1, fc2 = st.columns(2)
        prod_opts = ["Alle"] + sorted(table["Produkt"].dropna().unique().tolist()) if "Produkt" in table.columns else ["Alle"]
        scope_opts = ["Alle"] + sorted(table["Richtung"].dropna().unique().tolist()) if "Richtung" in table.columns else ["Alle"]
        sel_prod = fc1.selectbox("Produkt", prod_opts)
        sel_scope = fc2.selectbox("Richtung", scope_opts)
    filtered = table.copy()
    if "Produkt" in filtered.columns and sel_prod != "Alle":
        filtered = filtered[filtered["Produkt"] == sel_prod]
    if "Richtung" in filtered.columns and sel_scope != "Alle":
        filtered = filtered[filtered["Richtung"] == sel_scope]

    for datecol in ["Handelsdatum", "Lieferung ab", "Lieferung bis", "Monat"]:
        if datecol in filtered.columns:
            filtered[datecol] = pd.to_datetime(filtered[datecol], errors="coerce").dt.strftime("%d.%m.%Y")

    st.dataframe(
        filtered, use_container_width=True, hide_index=True, height=420,
        column_config={
            "Volumen (MWh)": st.column_config.NumberColumn(format="%.2f"),
            "Marktwert (EUR)": st.column_config.NumberColumn(format="%.0f"),
            "PnL (EUR)": st.column_config.NumberColumn(format="%.0f"),
        },
    )

    # Export
    csv = filtered.to_csv(index=False, sep=";").encode("utf-8")
    st.download_button(
        "📥 Gefilterte Deals als CSV exportieren",
        csv,
        file_name=f"deviwa_{choice.replace(' ','_').lower()}_deals.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Transaktionsdaten aus dem Deviwa-Energiepool · interne Demo. "
    "Volumen in MWh, Beträge in EUR. "
    "Peak = 08:00–20:00 Mo–Fr."
)

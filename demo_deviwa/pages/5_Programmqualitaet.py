"""Seite 5 – Programmqualität: Plan (Programm) vs. Realität (Ist)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deviwa_parser import extract_programme_data
from utils import (
    FMV_ACCENT,
    FMV_BLUE,
    FMV_GREEN,
    FMV_GREY,
    FMV_NAVY,
    FMV_RED,
    find_deviwa_file,
    fmt_chf,
    fmt_mwh,
    kpi_card,
    render_header,
)

st.set_page_config(
    page_title="FMV · Programmqualität",
    page_icon="🎚️",
    layout="wide",
)

render_header("Programmqualität · Plan gegen Realität")

st.markdown(
    "Wie genau entspricht die **geplante Fahrkurve (Programm)** der **realisierten "
    "Einspeisung/Bezug (Ist)**? Abweichungen erzeugen Ausgleichsenergiekosten."
)

# ---------------------------------------------------------------------------
# Daten laden
# ---------------------------------------------------------------------------
path = find_deviwa_file()
if path is None:
    st.warning("Keine Deviwa-Datei in `/data` gefunden.")
    st.stop()

prog_data = extract_programme_data(path)
prog_data = {k: v for k, v in prog_data.items() if v is not None and not v.empty
             and "programme_mw" in v.columns and "actual_mw" in v.columns
             and (v["programme_mw"].notna().any() and v["actual_mw"].notna().any())}

if not prog_data:
    st.info(
        "Keine Sheets mit Programm- und Ist-Werten gefunden. "
        "Erwartet werden Spalten mit 'Programm/Plan/Geplant' und 'Ist/Actual/Spot' pro Zeitstempel. "
        "Sie können die Datei dennoch anzeigen – aktuell ist nur EDSH mit Programme&Spot-Daten ausgestattet."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Akteur-Auswahl (nur die mit Programme-Daten)
# ---------------------------------------------------------------------------
actors = list(prog_data.keys())
options = ["Gesamter Pool"] + actors if len(actors) > 1 else actors
choice = st.radio("Akteur auswählen", options=options, horizontal=True, key="prog_actor")

if choice == "Gesamter Pool":
    frames = [v.assign(_actor=k) for k, v in prog_data.items()]
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    title_suffix = "Gesamter Pool"
else:
    df = prog_data[choice].copy()
    title_suffix = choice

# ---------------------------------------------------------------------------
# Zeitraumfilter
# ---------------------------------------------------------------------------
max_ts = df["timestamp"].max()
min_ts = df["timestamp"].min()
window_choice = st.radio(
    "Zeitraum",
    options=["Letzte 7 Tage", "Letzte 30 Tage", "Letzte 90 Tage", "Gesamter Zeitraum"],
    horizontal=True, index=1, key="prog_range",
)
if window_choice == "Letzte 7 Tage":
    start = max_ts - pd.Timedelta(days=7)
elif window_choice == "Letzte 30 Tage":
    start = max_ts - pd.Timedelta(days=30)
elif window_choice == "Letzte 90 Tage":
    start = max_ts - pd.Timedelta(days=90)
else:
    start = min_ts
df_sl = df[df["timestamp"] >= start].copy()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
clean = df_sl.dropna(subset=["programme_mw", "actual_mw"])
mae_mw = float((clean["actual_mw"] - clean["programme_mw"]).abs().mean()) if not clean.empty else np.nan
rmse_mw = float(np.sqrt(((clean["actual_mw"] - clean["programme_mw"]) ** 2).mean())) if not clean.empty else np.nan
bias_mw = float((clean["actual_mw"] - clean["programme_mw"]).mean()) if not clean.empty else np.nan
total_prog = float(clean["programme_mw"].sum())
total_actual = float(clean["actual_mw"].sum())
# Rough accuracy: 100% - MAPE
ape = (clean["actual_mw"] - clean["programme_mw"]).abs() / clean["programme_mw"].abs().clip(lower=1.0) * 100
accuracy_pct = max(0.0, 100.0 - float(ape.mean())) if not ape.empty else np.nan

# Schätzung Ausgleichsenergiekosten (typischer Preis ~150 EUR/MWh im Mittel)
assumed_imbalance_price = 150.0
imbalance_mwh = float((clean["actual_mw"] - clean["programme_mw"]).abs().sum())
imbalance_cost_est = imbalance_mwh * assumed_imbalance_price / len(clean) * 1.0  # je Stunde
# Korrekte Schätzung: Summe |delta_mwh| × preis / anzahl stunden → einfach Summe × typ. preis
imbalance_cost_est = imbalance_mwh * assumed_imbalance_price

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(
    kpi_card("Genauigkeit",
             f"{accuracy_pct:.1f} %" if np.isfinite(accuracy_pct) else "—",
             delta=f"{len(clean):,} Datenpunkte".replace(",", "'"),
             delta_color=FMV_GREEN if accuracy_pct >= 90 else (FMV_ACCENT if accuracy_pct >= 80 else FMV_RED)),
    unsafe_allow_html=True,
)
c2.markdown(
    kpi_card("MAE",
             f"{mae_mw:.2f} MW" if np.isfinite(mae_mw) else "—",
             delta=f"RMSE: {rmse_mw:.2f} MW" if np.isfinite(rmse_mw) else None,
             delta_color=FMV_NAVY),
    unsafe_allow_html=True,
)
c3.markdown(
    kpi_card("Bias",
             f"{bias_mw:+.2f} MW" if np.isfinite(bias_mw) else "—",
             delta="positiv = mehr Ist als Plan",
             delta_color=FMV_BLUE if abs(bias_mw or 0) < 0.5 else FMV_RED),
    unsafe_allow_html=True,
)
c4.markdown(
    kpi_card("Summe Abweichung",
             fmt_mwh(imbalance_mwh, 0),
             delta=f"über {window_choice.lower()}",
             delta_color=FMV_GREY),
    unsafe_allow_html=True,
)
c5.markdown(
    kpi_card("Geschätzte Ausgleichskosten",
             f"{fmt_chf(imbalance_cost_est, 0)} EUR",
             delta=f"bei Ø {assumed_imbalance_price:.0f} EUR/MWh",
             delta_color=FMV_RED),
    unsafe_allow_html=True,
)

st.markdown("")

# ---------------------------------------------------------------------------
# Zeitreihe Programm vs Ist
# ---------------------------------------------------------------------------
st.subheader(f"Programm gegen Ist · {title_suffix}")

plot_df = df_sl.set_index("timestamp")[["programme_mw", "actual_mw"]].resample("h").mean()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df["programme_mw"],
    mode="lines", name="Programm (Plan)",
    line=dict(color=FMV_BLUE, width=2.2, dash="solid"),
))
fig.add_trace(go.Scatter(
    x=plot_df.index, y=plot_df["actual_mw"],
    mode="lines", name="Ist (realisiert)",
    line=dict(color=FMV_NAVY, width=2.2),
))
# Füllung der Abweichung
delta = (plot_df["actual_mw"] - plot_df["programme_mw"]).fillna(0)
fig.add_trace(go.Scatter(
    x=plot_df.index, y=delta,
    mode="lines", name="Abweichung (Ist − Plan)",
    line=dict(color=FMV_RED, width=1.3, dash="dot"),
    yaxis="y2",
))
fig.update_layout(
    height=440, margin=dict(l=20, r=20, t=10, b=20),
    plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(title="MW (Plan / Ist)", gridcolor="#EEF1F6"),
    yaxis2=dict(title="Abweichung (MW)", overlaying="y", side="right", showgrid=False),
    xaxis=dict(gridcolor="#EEF1F6"),
    legend=dict(orientation="h", y=-0.15),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Scatter Plan vs Ist
# ---------------------------------------------------------------------------
col_l, col_r = st.columns([1, 1])

with col_l:
    st.subheader("Streudiagramm Plan vs. Ist")
    if not clean.empty:
        sample = clean.sample(min(3000, len(clean)), random_state=42)
        lim = float(max(sample["programme_mw"].max(), sample["actual_mw"].max()) or 1)
        lim_low = float(min(sample["programme_mw"].min(), sample["actual_mw"].min()) or 0)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=sample["programme_mw"], y=sample["actual_mw"],
            mode="markers", name="Stundenwerte",
            marker=dict(size=5, color=FMV_BLUE, opacity=0.45),
        ))
        fig_sc.add_trace(go.Scatter(
            x=[lim_low, lim], y=[lim_low, lim],
            mode="lines", name="Perfekte Übereinstimmung",
            line=dict(color=FMV_RED, width=1.2, dash="dash"),
        ))
        fig_sc.update_layout(
            height=380, margin=dict(l=20, r=20, t=10, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="Programm (MW)", gridcolor="#EEF1F6"),
            yaxis=dict(title="Ist (MW)", gridcolor="#EEF1F6"),
            legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

with col_r:
    st.subheader("Abweichungsverteilung")
    if not clean.empty:
        err = (clean["actual_mw"] - clean["programme_mw"])
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=err, nbinsx=40,
            marker_color=FMV_BLUE, opacity=0.75,
        ))
        fig_h.add_vline(x=0, line_width=2, line_color=FMV_NAVY)
        fig_h.add_vline(x=float(err.mean()), line_width=2, line_dash="dash", line_color=FMV_RED,
                        annotation_text=f"Ø {float(err.mean()):.2f} MW",
                        annotation_position="top right")
        fig_h.update_layout(
            height=380, margin=dict(l=20, r=20, t=10, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="Ist − Plan (MW)", gridcolor="#EEF1F6"),
            yaxis=dict(title="Häufigkeit", gridcolor="#EEF1F6"),
        )
        st.plotly_chart(fig_h, use_container_width=True)

# ---------------------------------------------------------------------------
# Monatliche Genauigkeit
# ---------------------------------------------------------------------------
st.subheader("Monatliche Genauigkeit")
if "month" in df_sl.columns and not clean.empty:
    g = df_sl.dropna(subset=["programme_mw", "actual_mw"]).copy()
    g["abs_err"] = (g["actual_mw"] - g["programme_mw"]).abs()
    g["ape"] = g["abs_err"] / g["programme_mw"].abs().clip(lower=1.0) * 100
    monthly_acc = g.groupby("month").agg(
        mae_mw=("abs_err", "mean"),
        bias_mw=("actual_mw", lambda s: (s - g.loc[s.index, "programme_mw"]).mean()),
        ape_pct=("ape", "mean"),
        n=("abs_err", "count"),
    ).reset_index()
    monthly_acc["accuracy_pct"] = (100.0 - monthly_acc["ape_pct"]).clip(lower=0)

    fig_m = go.Figure()
    colors = [
        FMV_GREEN if v >= 90 else (FMV_ACCENT if v >= 80 else FMV_RED)
        for v in monthly_acc["accuracy_pct"]
    ]
    fig_m.add_trace(go.Bar(
        x=monthly_acc["month"], y=monthly_acc["accuracy_pct"],
        marker_color=colors,
        text=[f"{v:.1f}%" for v in monthly_acc["accuracy_pct"]],
        textposition="outside",
    ))
    fig_m.add_hline(y=90, line_width=1, line_dash="dash", line_color=FMV_GREEN,
                    annotation_text="Ziel 90%", annotation_position="top left")
    fig_m.update_layout(
        height=340, margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="Genauigkeit (%)", gridcolor="#EEF1F6", range=[0, 105]),
        xaxis=dict(gridcolor="#EEF1F6"),
    )
    st.plotly_chart(fig_m, use_container_width=True)

st.divider()
st.caption(
    "Quelle: Programme-Sheets der Deviwa-Datei. "
    "Geschätzte Ausgleichsenergiekosten basieren auf einem typischen Durchschnittspreis "
    "von 150 EUR/MWh (Schweizer Markt 2025). Tatsächliche Kosten abhängig vom Richtungsfaktor."
)

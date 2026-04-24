"""Seite 6 – Value-at-Risk und Stresstests.

Parametrischer 1-Tages-VaR (95 %) + historische Verteilung + Stressszenarien
auf die offene Netto-Position.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deviwa_parser import monthly_breakdown
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
    load_deviwa_auto,
    load_epex_ch,
    load_pfc,
    render_actor_selector,
    render_header,
)

st.set_page_config(
    page_title="FMV · VaR & Stresstest",
    page_icon="🛡️",
    layout="wide",
)

render_header("VaR & Stresstest · Risikobild der offenen Position")

st.markdown(
    "Wie viel Geld steht bei einer ungünstigen Preisbewegung auf dem Spiel? "
    "Die Bewertung erfolgt mark-to-market gegen die aktuelle PFC der FMV."
)

# ---------------------------------------------------------------------------
# Daten
# ---------------------------------------------------------------------------
data_by_actor = load_deviwa_auto()
pfc = load_pfc()
spot = load_epex_ch()

if not data_by_actor:
    st.warning("Keine Deviwa-Datei in `/data` gefunden.")
    st.stop()
if pfc.empty:
    st.warning("Keine PFC verfügbar – Bewertung nicht möglich.")
    st.stop()

choice, df = render_actor_selector(data_by_actor, key="var_actor")
if df is None or df.empty:
    st.stop()

st.markdown(
    f"<div style='color:{FMV_GREY};'>Ansicht: <strong style='color:{FMV_NAVY};'>{choice}</strong></div>",
    unsafe_allow_html=True,
)
st.markdown("")

# ---------------------------------------------------------------------------
# Offene Netto-Position je Monat (MWh)
# ---------------------------------------------------------------------------
monthly = monthly_breakdown(df)
if monthly.empty:
    st.info("Keine monatlichen Daten für Positionsbewertung.")
    st.stop()

pivot = monthly.pivot_table(
    index="month", columns="scope_clean", values="volume_mwh", aggfunc="sum"
).fillna(0.0)
pivot["net_mwh"] = pivot.get("Einspeisung", 0) - pivot.get("Bezug", 0)

# PFC je Monat mitteln
pfc_h = pfc["price_shape"].resample("h").mean()
pfc_h.index = pfc_h.index.tz_convert("Europe/Zurich")
pfc_monthly = pfc_h.resample("MS").mean()
pfc_monthly.index = pfc_monthly.index.tz_localize(None)

pos = pivot[["net_mwh"]].copy()
pos.index = pd.to_datetime(pos.index)
pos = pos.join(pfc_monthly.rename("pfc_eur_mwh"), how="left")
# Fallback: Monate ausserhalb des PFC-Zeitraums mit Gesamtdurchschnitt
pos["pfc_eur_mwh"] = pos["pfc_eur_mwh"].fillna(float(pfc_monthly.mean()))
pos["mtm_eur"] = pos["net_mwh"] * pos["pfc_eur_mwh"]
pos["abs_mwh"] = pos["net_mwh"].abs()

total_net_mwh = float(pos["net_mwh"].sum())
total_mtm_eur = float(pos["mtm_eur"].sum())
total_gross_mwh = float(pos["abs_mwh"].sum())

# ---------------------------------------------------------------------------
# Preisvolatilität aus dem Spot schätzen (log-Renditen, tägliche Mittel)
# ---------------------------------------------------------------------------
daily_returns: np.ndarray = np.array([])
sigma_daily = np.nan
if not spot.empty:
    daily = spot["price"].resample("D").mean().dropna().clip(lower=1.0)
    if len(daily) > 60:
        returns = np.log(daily.values[1:] / daily.values[:-1])
        daily_returns = returns[np.isfinite(returns)]
        if len(daily_returns) > 60:
            sigma_daily = float(np.std(daily_returns, ddof=1))

# ---------------------------------------------------------------------------
# VaR-Parameter
# ---------------------------------------------------------------------------
st.subheader("VaR-Parameter")
c_params = st.columns(4)
conf_level = c_params[0].selectbox("Konfidenzniveau", [0.95, 0.975, 0.99], index=0)
horizon_days = c_params[1].selectbox("Horizont (Handelstage)", [1, 5, 10], index=1)
method = c_params[2].selectbox("Methode", ["Parametrisch (Normal)", "Historisch (Simulation)"], index=0)
c_params[3].metric("Tagesvolatilität (Spot)",
                   f"{sigma_daily*100:.2f} %" if np.isfinite(sigma_daily) else "—")

z = float(norm.ppf(conf_level))
sqrt_t = float(np.sqrt(horizon_days))

# ---------------------------------------------------------------------------
# VaR-Berechnung
# ---------------------------------------------------------------------------
var_param_eur = np.nan
var_hist_eur = np.nan
es_eur = np.nan

abs_mtm = abs(total_mtm_eur)
if np.isfinite(sigma_daily):
    # Parametrisch: VaR = |value| × σ × z × √t  (Annahme: 1 Hauptrisiko = Preis)
    var_param_eur = abs_mtm * sigma_daily * z * sqrt_t

if len(daily_returns) > 60:
    # Historische Simulation: P&L = MtM × (1 − e^(return × √t))
    scaled_returns = daily_returns * sqrt_t
    simulated_pnl = total_mtm_eur * (np.exp(-scaled_returns) - 1.0)  # Preis nach unten = Verlust bei Long
    if total_mtm_eur < 0:
        simulated_pnl = -simulated_pnl  # Short: Preisanstieg = Verlust
    var_hist_eur = float(-np.quantile(simulated_pnl, 1 - conf_level))
    es_eur = float(-simulated_pnl[simulated_pnl <= np.quantile(simulated_pnl, 1 - conf_level)].mean())

active_var = var_param_eur if method.startswith("Parametrisch") else var_hist_eur

c1, c2, c3, c4 = st.columns(4)
c1.markdown(
    kpi_card("Offene Netto-Position",
             fmt_mwh(total_net_mwh, 0),
             delta=f"Brutto: {fmt_mwh(total_gross_mwh,0)}",
             delta_color=FMV_NAVY),
    unsafe_allow_html=True,
)
c2.markdown(
    kpi_card("Mark-to-Market",
             f"{fmt_chf(total_mtm_eur, 0)} EUR",
             delta=f"Ø PFC: {fmt_eur_mwh(float(pos['pfc_eur_mwh'].mean()))}",
             delta_color=FMV_BLUE),
    unsafe_allow_html=True,
)
c3.markdown(
    kpi_card(
        f"VaR {horizon_days}T {int(conf_level*100)} %",
        f"{fmt_chf(active_var, 0)} EUR" if np.isfinite(active_var) else "—",
        delta=method,
        delta_color=FMV_RED,
    ),
    unsafe_allow_html=True,
)
c4.markdown(
    kpi_card(
        "Expected Shortfall",
        f"{fmt_chf(es_eur, 0)} EUR" if np.isfinite(es_eur) else "—",
        delta=f"Ø Verlust jenseits VaR",
        delta_color=FMV_RED,
    ),
    unsafe_allow_html=True,
)

if not np.isfinite(sigma_daily):
    st.warning("Zu wenige Spot-Daten für Volatilitätsschätzung.")

st.markdown("")

# ---------------------------------------------------------------------------
# Offene Position je Monat + Bewertungsband
# ---------------------------------------------------------------------------
st.subheader("Monatliche offene Position und Bewertungsband")

pos_plot = pos.reset_index().rename(columns={"index": "month"}).copy()
if "month" not in pos_plot.columns:
    pos_plot["month"] = pos.index

# Unsicherheitsband: ±1 σ √t auf den MtM-Preis
if np.isfinite(sigma_daily):
    band_pct = sigma_daily * z * sqrt_t
    pos_plot["mtm_low"] = pos_plot["mtm_eur"] * (1 - band_pct)
    pos_plot["mtm_high"] = pos_plot["mtm_eur"] * (1 + band_pct)

fig_pos = go.Figure()
colors = [FMV_GREEN if v >= 0 else FMV_RED for v in pos_plot["net_mwh"]]
fig_pos.add_trace(go.Bar(
    x=pos_plot["month"], y=pos_plot["net_mwh"],
    marker_color=colors,
    name="Netto-Position (MWh)",
    yaxis="y1",
    opacity=0.85,
))
fig_pos.add_trace(go.Scatter(
    x=pos_plot["month"], y=pos_plot["mtm_eur"],
    mode="lines+markers", name="MtM (EUR)",
    line=dict(color=FMV_NAVY, width=2.5),
    yaxis="y2",
))
if "mtm_low" in pos_plot.columns:
    fig_pos.add_trace(go.Scatter(
        x=pos_plot["month"], y=pos_plot["mtm_high"],
        mode="lines", line=dict(width=0), showlegend=False, yaxis="y2",
        hoverinfo="skip",
    ))
    fig_pos.add_trace(go.Scatter(
        x=pos_plot["month"], y=pos_plot["mtm_low"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(15,82,204,0.15)",
        name=f"{int(conf_level*100)} % Bewertungsband",
        yaxis="y2",
    ))

fig_pos.add_hline(y=0, line_width=1, line_dash="dot", line_color=FMV_GREY)
fig_pos.update_layout(
    height=440, margin=dict(l=20, r=20, t=10, b=20),
    plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(title="Netto-Position (MWh)", gridcolor="#EEF1F6"),
    yaxis2=dict(title="MtM (EUR)", overlaying="y", side="right", showgrid=False),
    xaxis=dict(gridcolor="#EEF1F6"),
    legend=dict(orientation="h", y=-0.18),
    hovermode="x unified",
)
st.plotly_chart(fig_pos, use_container_width=True)

# ---------------------------------------------------------------------------
# Stresstests
# ---------------------------------------------------------------------------
st.subheader("Stresstests · Auswirkung auf den Portfolio-Wert")

scenarios = [
    {"name": "Preis +20 %", "price_shock": 0.20,
     "description": "Angebotsengpass, Gas-Rallye"},
    {"name": "Preis −20 %", "price_shock": -0.20,
     "description": "Milder Winter, Überangebot erneuerbar"},
    {"name": "Extremwinter (Preis +50 %)", "price_shock": 0.50,
     "description": "Gasmangel + Kältewelle"},
    {"name": "Trockenes Jahr (Preis +15 %)", "price_shock": 0.15,
     "description": "Hydro −30 %, Importabhängigkeit steigt"},
    {"name": "Spitze ohne Kernkraft (+30 %)", "price_shock": 0.30,
     "description": "Unplanmässiger Nuklear-Ausfall"},
    {"name": "CO₂-Sprung (Preis +10 %)", "price_shock": 0.10,
     "description": "EUA +20 EUR/t"},
]

rows = []
for sc in scenarios:
    shock = sc["price_shock"]
    # Long-Position (Netto > 0): Preisanstieg = Gewinn, Preisrückgang = Verlust
    impact_eur = total_mtm_eur * shock
    rows.append({
        "Szenario": sc["name"],
        "Beschreibung": sc["description"],
        "Preisbewegung": f"{shock*100:+.0f} %",
        "P&L (EUR)": impact_eur,
        "Neuer Portfolio-Wert": total_mtm_eur + impact_eur,
    })

stress_df = pd.DataFrame(rows)

# Anzeige
st.dataframe(
    stress_df.assign(**{
        "P&L (EUR)": stress_df["P&L (EUR)"].round(0),
        "Neuer Portfolio-Wert": stress_df["Neuer Portfolio-Wert"].round(0),
    }),
    use_container_width=True,
    hide_index=True,
    column_config={
        "P&L (EUR)": st.column_config.NumberColumn(format="%.0f"),
        "Neuer Portfolio-Wert": st.column_config.NumberColumn(format="%.0f"),
    },
)

# Bar chart P&L
fig_sc = go.Figure()
fig_sc.add_trace(go.Bar(
    x=stress_df["Szenario"],
    y=stress_df["P&L (EUR)"],
    marker_color=[FMV_GREEN if v >= 0 else FMV_RED for v in stress_df["P&L (EUR)"]],
    text=[f"{fmt_chf(v,0)} EUR" for v in stress_df["P&L (EUR)"]],
    textposition="outside",
))
fig_sc.add_hline(y=0, line_width=1, line_color=FMV_GREY)
fig_sc.update_layout(
    height=380, margin=dict(l=20, r=20, t=30, b=40),
    plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(title="P&L (EUR)", gridcolor="#EEF1F6"),
    xaxis=dict(tickangle=-15),
)
st.plotly_chart(fig_sc, use_container_width=True)

# ---------------------------------------------------------------------------
# Historische Simulation (Verteilung)
# ---------------------------------------------------------------------------
if len(daily_returns) > 60:
    st.subheader(f"Historische P&L-Verteilung · {horizon_days}-Tage-Horizont")
    scaled = daily_returns * sqrt_t
    sim_pnl = total_mtm_eur * (np.exp(-scaled) - 1.0) if total_mtm_eur >= 0 else -total_mtm_eur * (np.exp(-scaled) - 1.0)
    sim_pnl = -sim_pnl if total_mtm_eur < 0 else sim_pnl

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=sim_pnl, nbinsx=60,
        marker_color=FMV_BLUE, opacity=0.8,
    ))
    q = float(np.quantile(sim_pnl, 1 - conf_level))
    fig_hist.add_vline(x=q, line_width=2, line_dash="dash", line_color=FMV_RED,
                       annotation_text=f"VaR {int(conf_level*100)} %",
                       annotation_position="top right")
    fig_hist.add_vline(x=0, line_width=1, line_color=FMV_NAVY)
    fig_hist.update_layout(
        height=340, margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Simulierter P&L (EUR)", gridcolor="#EEF1F6"),
        yaxis=dict(title="Häufigkeit", gridcolor="#EEF1F6"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()
st.caption(
    "Methodik: Parametrisch VaR = |MtM| × σ × z × √t. Historische Simulation auf Basis "
    "täglicher Spot-Renditen. Stressszenarien sind exemplarisch und nicht vorhersagend. "
    "Die offene Position verwendet die Netto-Differenz Einspeisung − Bezug je Monat × aktuelle PFC."
)

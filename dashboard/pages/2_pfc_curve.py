"""
Page 2 â€” Courbe PFC
"La term structure complÃ¨te"
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    COLORS, add_range_slider, export_csv_button, load_pfc, no_data_warning,
    show_freshness_sidebar,
)

st.header("Courbe PFC N+3 ans")
st.caption("Price Forward Curve 15min â€” base, peak, off-peak avec intervalles de confiance")

show_freshness_sidebar()

pfc = load_pfc()

if pfc is None or "price_shape" not in (pfc.columns if pfc is not None else []):
    no_data_warning("PFC")
    st.stop()

# â”€â”€ Sidebar controls â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    st.subheader("RÃ©solution")
    resolution = st.radio("AgrÃ©gation", ["15min", "Horaire", "Journalier", "Mensuel"],
                          index=1, horizontal=True)
    resample_map = {"15min": None, "Horaire": "h", "Journalier": "D", "Mensuel": "MS"}

    st.subheader("Produits")
    show_base = st.checkbox("Base", value=True)
    show_peak = st.checkbox("Peak (08-20 Lu-Ve)", value=True)
    show_offpeak = st.checkbox("Off-Peak", value=False)
    show_bands = st.checkbox("Bandes IC 80%", value=True)

# â”€â”€ Resample â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
freq = resample_map[resolution]

if freq:
    pfc_r = pfc[["price_shape"]].resample(freq).mean()
    if "p10" in pfc.columns:
        pfc_r["p10"] = pfc["p10"].resample(freq).mean()
        pfc_r["p90"] = pfc["p90"].resample(freq).mean()
else:
    pfc_r = pfc

# â”€â”€ Peak / Off-Peak decomposition â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
idx_zurich = pfc.index.tz_convert("Europe/Zurich")
is_peak = (idx_zurich.hour >= 8) & (idx_zurich.hour < 20) & (idx_zurich.dayofweek < 5)

if freq:
    pfc_peak = pfc.loc[is_peak, "price_shape"].resample(freq).mean()
    pfc_offpeak = pfc.loc[~is_peak, "price_shape"].resample(freq).mean()
else:
    pfc_peak = pfc.loc[is_peak, "price_shape"]
    pfc_offpeak = pfc.loc[~is_peak, "price_shape"]

# â”€â”€ KPI Row â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Base avg", f"{pfc['price_shape'].mean():.1f} EUR/MWh")
with k2:
    st.metric("Peak avg", f"{pfc.loc[is_peak, 'price_shape'].mean():.1f} EUR/MWh")
with k3:
    spread = pfc.loc[is_peak, "price_shape"].mean() - pfc.loc[~is_peak, "price_shape"].mean()
    st.metric("Peak spread", f"{spread:+.1f} EUR/MWh")
with k4:
    if "calibrated" in pfc.columns:
        cal_pct = pfc["calibrated"].mean() * 100
        st.metric("CalibrÃ©", f"{cal_pct:.0f}%")
    else:
        st.metric("Horizon", f"{len(pfc) // 96} jours")

st.divider()

# â”€â”€ Main PFC Chart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig = go.Figure()

if show_base:
    fig.add_trace(go.Scatter(
        x=pfc_r.index, y=pfc_r["price_shape"],
        name="Base",
        line=dict(color=COLORS["amber"], width=2),
        hovertemplate="%{y:.1f}<extra>Base</extra>",
    ))

if show_peak and not pfc_peak.empty:
    fig.add_trace(go.Scatter(
        x=pfc_peak.index, y=pfc_peak.values,
        name="Peak",
        line=dict(color=COLORS["red"], width=1.5),
        hovertemplate="%{y:.1f}<extra>Peak</extra>",
    ))

if show_offpeak and not pfc_offpeak.empty:
    fig.add_trace(go.Scatter(
        x=pfc_offpeak.index, y=pfc_offpeak.values,
        name="Off-Peak",
        line=dict(color=COLORS["green"], width=1.5),
        hovertemplate="%{y:.1f}<extra>Off-Peak</extra>",
    ))

if show_bands and "p10" in pfc_r.columns:
    fig.add_trace(go.Scatter(
        x=pfc_r.index, y=pfc_r["p90"],
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=pfc_r.index, y=pfc_r["p10"],
        name="IC 80%", line=dict(width=0),
        fill="tonexty", fillcolor=COLORS["band"],
        hoverinfo="skip",
    ))

fig.update_layout(
    yaxis_title="EUR/MWh",
    height=500,
    legend=dict(orientation="h", y=1.02, x=0),
)
fig = add_range_slider(fig)
st.plotly_chart(fig, width="stretch")

# â”€â”€ Monthly term structure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.subheader("Term Structure mensuelle")

monthly = pfc["price_shape"].resample("MS").mean()
monthly_peak = pfc.loc[is_peak, "price_shape"].resample("MS").mean()

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=monthly.index, y=monthly.values,
    name="Base", marker_color=COLORS["blue"], opacity=0.7,
    hovertemplate="%{x|%b %Y}<br>%{y:.1f} EUR/MWh<extra>Base</extra>",
))
fig_bar.add_trace(go.Scatter(
    x=monthly_peak.index, y=monthly_peak.values,
    name="Peak", mode="lines+markers",
    line=dict(color=COLORS["red"], width=2),
    marker=dict(size=5),
    hovertemplate="%{y:.1f}<extra>Peak</extra>",
))

fig_bar.update_layout(
    yaxis_title="EUR/MWh", height=350,
    legend=dict(orientation="h", y=1.05, x=0),
    bargap=0.15,
)
st.plotly_chart(fig_bar, width="stretch")

# â”€â”€ Export â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.expander("Export"):
    export_csv_button(pfc_r, "pfc_curve.csv", "Export PFC (rÃ©solution sÃ©lectionnÃ©e)")

# â”€â”€ Confidence profile â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if "confidence" in pfc.columns:
    with st.expander("Score de confiance par horizon"):
        conf = pfc.groupby("profile_type")["confidence"].first().reset_index()
        conf.columns = ["Horizon", "Confiance"]
        st.dataframe(conf, hide_index=True, width="stretch")

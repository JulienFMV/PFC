"""
Page 7 - CH vs DE
Spread and term-structure comparison between CH and DE PFCs.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import COLORS, add_range_slider, load_pfc_market, no_data_warning, show_freshness_sidebar

st.header("CH vs DE Spread")
st.caption("Comparaison des courbes PFC CH et DE, plus spread CH-DE")

show_freshness_sidebar()

pfc_ch = load_pfc_market("CH")
pfc_de = load_pfc_market("DE")

if pfc_ch is None or "price_shape" not in pfc_ch.columns:
    no_data_warning("PFC CH")
    st.stop()
if pfc_de is None or "price_shape" not in pfc_de.columns:
    no_data_warning("PFC DE")
    st.stop()

joined = pd.concat(
    [
        pfc_ch["price_shape"].rename("ch"),
        pfc_de["price_shape"].rename("de"),
    ],
    axis=1,
    join="inner",
).dropna()

if joined.empty:
    st.warning("Pas de recouvrement temporel entre PFC CH et DE.")
    st.stop()

joined["spread_ch_de"] = joined["ch"] - joined["de"]

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("CH moy.", f"{joined['ch'].mean():.1f}")
with k2:
    st.metric("DE moy.", f"{joined['de'].mean():.1f}")
with k3:
    st.metric("Spread moy.", f"{joined['spread_ch_de'].mean():+.1f}")
with k4:
    front_n = min(96 * 30, len(joined))
    st.metric("Spread M+1", f"{joined['spread_ch_de'].iloc[:front_n].mean():+.1f}")
st.caption("EUR/MWh")

st.divider()

st.subheader("Courbes horaires et spread")
view = joined.resample("h").mean()

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=view.index,
        y=view["ch"],
        name="CH",
        line=dict(color=COLORS["amber"], width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=view.index,
        y=view["de"],
        name="DE",
        line=dict(color=COLORS["blue"], width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=view.index,
        y=view["spread_ch_de"],
        name="Spread CH-DE",
        yaxis="y2",
        line=dict(color=COLORS["red"], width=1.5, dash="dot"),
    )
)
fig.update_layout(
    height=500,
    yaxis=dict(title="Prix EUR/MWh"),
    yaxis2=dict(title="Spread EUR/MWh", overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h", y=1.04, x=0),
)
fig = add_range_slider(fig)
st.plotly_chart(fig, width="stretch")

st.subheader("Term Structure mensuelle")
m = joined.resample("MS").mean()

fig_m = go.Figure()
fig_m.add_trace(go.Bar(x=m.index, y=m["ch"], name="CH", marker_color=COLORS["amber"], opacity=0.8))
fig_m.add_trace(go.Bar(x=m.index, y=m["de"], name="DE", marker_color=COLORS["blue"], opacity=0.8))
fig_m.add_trace(
    go.Scatter(
        x=m.index,
        y=m["spread_ch_de"],
        name="Spread CH-DE",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color=COLORS["red"], width=2),
    )
)
fig_m.update_layout(
    barmode="group",
    height=420,
    yaxis=dict(title="Prix EUR/MWh"),
    yaxis2=dict(title="Spread EUR/MWh", overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h", y=1.04, x=0),
)
st.plotly_chart(fig_m, width="stretch")

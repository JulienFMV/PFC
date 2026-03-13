"""
Page 1 - Overview
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    COLORS,
    add_range_slider,
    export_csv_button,
    format_eur,
    format_gwh,
    format_pct,
    load_entso,
    load_epex,
    load_hydro,
    load_pfc,
    no_data_warning,
    show_freshness_sidebar,
)

st.header("Overview")
st.caption("Desk view: prix EPEX, momentum, flux systeme, hydro")

show_freshness_sidebar()

epex = load_epex()
entso = load_entso()
hydro = load_hydro()
pfc = load_pfc()

has_epex = epex is not None and "price_eur_mwh" in epex.columns
has_entso = entso is not None and "load_mw" in entso.columns
has_hydro = hydro is not None and "fill_pct" in hydro.columns
has_pfc = pfc is not None and "price_shape" in (pfc.columns if pfc is not None else [])

# Top controls bar (style FMV)
ctrl1, ctrl2, ctrl3, ctrl4, ctrl5, ctrl6 = st.columns([1, 1, 1, 1, 1, 1.2])
with ctrl1:
    country = st.selectbox("Country", ["CH", "DE", "FR", "AT"], index=0)
with ctrl2:
    base_peak = st.selectbox("Base/Peak", ["BASE", "PEAK"], index=0)
with ctrl3:
    period = st.selectbox("Period", ["YEAR", "QUARTER", "MONTH"], index=0)
with ctrl4:
    product = st.selectbox("Product", ["CAL-26", "CAL-27", "CAL-28", "CAL-29"], index=3)
with ctrl5:
    date_mode = st.selectbox("Date", ["Dernier", "Historique"], index=0)
with ctrl6:
    weeks = st.selectbox("Fenetre", ["13 semaines", "26 semaines", "50 semaines"], index=2)

st.markdown(
    f"<div style='padding:0.2rem 0 0.8rem 0;color:{COLORS['muted']};font-weight:600'>"
    f"{country} | {base_peak} | {period} | {product} | {weeks}</div>",
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    if has_epex:
        last_price = float(epex["price_eur_mwh"].iloc[-1])
        prev_day = float(epex["price_eur_mwh"].iloc[-96]) if len(epex) > 96 else last_price
        st.metric("Last Price", format_eur(last_price), delta=f"{last_price - prev_day:+.2f}", delta_color="inverse")
    else:
        st.metric("Last Price", "--")

with k2:
    if has_epex and len(epex) > 96 * 7:
        avg_7d = float(epex["price_eur_mwh"].iloc[-96 * 7 :].mean())
        st.metric("Moyenne 7j", format_eur(avg_7d))
    else:
        st.metric("Moyenne 7j", "--")

with k3:
    if has_epex and len(epex) > 96 * 30:
        std_30d = float(epex["price_eur_mwh"].iloc[-96 * 30 :].std())
        st.metric("Volatilite 30j", f"{std_30d:.2f} EUR/MWh")
    else:
        st.metric("Volatilite 30j", "--")

with k4:
    if has_hydro:
        fill = float(hydro["fill_pct"].iloc[-1])
        fill_prev = float(hydro["fill_pct"].iloc[-2]) if len(hydro) > 1 else fill
        st.metric("Hydro CH", format_pct(fill), delta=f"{fill - fill_prev:+.2f}pp")
    else:
        st.metric("Hydro CH", "--")

with k5:
    if has_entso:
        load_now = float(entso["load_mw"].iloc[-1])
        load_prev = float(entso["load_mw"].iloc[-96]) if len(entso) > 96 else load_now
        st.metric("Load CH", f"{load_now:,.0f} MW", delta=f"{load_now - load_prev:+,.0f} MW")
    else:
        st.metric("Load CH", "--")

st.divider()

left, right = st.columns([2.1, 1])

with left:
    st.subheader("Settlement Price (EUR/MWh)")

    if has_epex:
        px_h = epex["price_eur_mwh"].resample("h").mean().dropna()
        df = pd.DataFrame({"price": px_h})
        df["sma20"] = df["price"].rolling(24 * 20, min_periods=24).mean()
        df["sma50"] = df["price"].rolling(24 * 50, min_periods=24).mean()
        df["sma200"] = df["price"].rolling(24 * 200, min_periods=24).mean()
        roll_std = df["price"].rolling(24 * 20, min_periods=24).std()
        df["boll_low"] = df["sma20"] - 2 * roll_std
        df["boll_up"] = df["sma20"] + 2 * roll_std

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["sma20"], name="SMA 20", line=dict(color="#B33A3A", width=1.6)))
        fig.add_trace(go.Scatter(x=df.index, y=df["sma50"], name="SMA 50", line=dict(color="#9B8A3C", width=1.6)))
        fig.add_trace(go.Scatter(x=df.index, y=df["sma200"], name="SMA 200", line=dict(color="#111827", width=1.4)))
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["boll_low"],
                name="Bollinger Lower",
                line=dict(color="#6B7280", width=1.2, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["boll_up"],
                name="Bollinger Upper",
                line=dict(color="#6B7280", width=1.2, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["price"],
                name="Price Settlement",
                line=dict(color=COLORS["navy"], width=2.8),
            )
        )

        fig.update_layout(
            yaxis_title="EUR/MWh",
            height=510,
            legend=dict(orientation="h", y=1.03, x=0),
        )
        fig = add_range_slider(fig)
        st.plotly_chart(fig, width="stretch")
    else:
        no_data_warning("prix EPEX")

with right:
    st.subheader("Market Snapshot")

    if has_epex:
        px_h = epex["price_eur_mwh"].resample("h").mean().dropna()
        win = px_h.iloc[-24 * 7 :] if len(px_h) > 24 * 7 else px_h
        high = float(win.max())
        low = float(win.min())
        rng = high - low
        last = float(win.iloc[-1])
        pos = ((last - low) / rng * 100) if rng > 0 else 50.0

        snapshot = pd.DataFrame(
            {
                "Label": ["High", "Low", "Range", "Position % in Range", "Last Price"],
                "Value": [f"{high:.2f}", f"{low:.2f}", f"{rng:.2f}", f"{pos:.0f}%", f"{last:.2f}"],
            }
        )
        st.dataframe(snapshot, hide_index=True, width="stretch")

        zone = (
            "Lower part of the range"
            if pos < 40
            else "Mid-range"
            if pos < 60
            else "Upper part of the range"
        )
        st.markdown(
            f"<div style='border:1px solid #94B5FF;background:#F5F9FF;padding:0.7rem;"
            f"text-align:center;font-weight:700;color:{COLORS['blue']}'>"
            f"{zone}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Pas de snapshot sans prix EPEX.")

st.divider()

b1, b2 = st.columns([1.3, 1])

with b1:
    st.subheader("System Flows")
    if has_entso:
        ent_h = entso.resample("h").mean().dropna(how="all")
        fig_flow = go.Figure()

        if "load_mw" in ent_h.columns:
            fig_flow.add_trace(
                go.Scatter(
                    x=ent_h.index,
                    y=ent_h["load_mw"],
                    name="Load MW",
                    line=dict(color=COLORS["navy"], width=2),
                )
            )
        if "solar_mw" in ent_h.columns:
            fig_flow.add_trace(
                go.Scatter(
                    x=ent_h.index,
                    y=ent_h["solar_mw"],
                    name="Solar MW",
                    line=dict(color="#F59E0B", width=1.5),
                )
            )
        if "wind_mw" in ent_h.columns:
            fig_flow.add_trace(
                go.Scatter(
                    x=ent_h.index,
                    y=ent_h["wind_mw"],
                    name="Wind MW",
                    line=dict(color="#10B981", width=1.5),
                )
            )
        if "cross_border_mw" in ent_h.columns:
            fig_flow.add_trace(
                go.Scatter(
                    x=ent_h.index,
                    y=ent_h["cross_border_mw"],
                    name="Cross-border MW",
                    line=dict(color=COLORS["blue_soft"], width=1.5, dash="dash"),
                )
            )

        fig_flow.update_layout(height=360, yaxis_title="MW", legend=dict(orientation="h", y=1.02, x=0))
        st.plotly_chart(fig_flow, width="stretch")
    else:
        no_data_warning("load/generation")

with b2:
    st.subheader("Hydro Reservoirs - 5Y Position")
    if has_hydro:
        h = hydro.copy()
        h["week"] = h.index.isocalendar().week.values.astype(int)
        h["year"] = h.index.year
        current_year = int(h["year"].max())
        hist = h[h["year"].between(current_year - 5, current_year - 1)]

        if not hist.empty:
            env = hist.groupby("week")["fill_pct"].agg(["min", "mean", "max"])
            curr = h[h["year"] == current_year].sort_values("week")

            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(x=env.index, y=env["max"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_h.add_trace(
                go.Scatter(
                    x=env.index,
                    y=env["min"],
                    name="Min 5Y",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(100,130,190,0.14)",
                    hoverinfo="skip",
                )
            )
            fig_h.add_trace(
                go.Scatter(
                    x=env.index,
                    y=env["mean"],
                    name="Moyenne 5Y",
                    line=dict(color=COLORS["blue_soft"], width=2, dash="dot"),
                )
            )
            fig_h.add_trace(
                go.Scatter(
                    x=curr["week"],
                    y=curr["fill_pct"],
                    name=f"{current_year}",
                    line=dict(color=COLORS["navy"], width=3),
                )
            )
            fig_h.update_layout(height=360, xaxis_title="Semaine ISO", yaxis_title="Remplissage (%)", yaxis_range=[0, 100])
            st.plotly_chart(fig_h, width="stretch")

            latest = curr.iloc[-1] if not curr.empty else None
            if latest is not None:
                st.markdown(
                    f"**Niveau actuel:** {latest['fill_pct']:.2f}%  |  "
                    f"**Stock:** {format_gwh(float(hydro['fill_gwh'].iloc[-1]))}"
                )
        else:
            st.info("Historique insuffisant pour la bande 5 ans.")
    else:
        no_data_warning("hydro")

if has_epex:
    with st.expander("Export"):
        export_csv_button(epex.tail(96 * 30), "epex_last_30d.csv", "Export EPEX 30 jours")
        if has_entso:
            export_csv_button(entso.tail(96 * 30), "entso_last_30d.csv", "Export ENTSO 30 jours")

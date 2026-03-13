"""
Page 1 — Overview
"Où vont les prix ?"
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    COLORS, add_range_slider, export_csv_button, format_eur, format_gwh,
    format_pct, latest_run_summary, load_benchmarks, load_epex, load_hydro, load_pfc,
    no_data_warning, show_freshness_sidebar,
)

st.header("Overview")
st.caption("Vue d'ensemble du marche et de la PFC - mise a jour quotidienne")

show_freshness_sidebar()

# ── Load data ─────────────────────────────────────────────────────────────
epex = load_epex()
hydro = load_hydro()
pfc = load_pfc()

has_epex = epex is not None and "price_eur_mwh" in epex.columns
has_hydro = hydro is not None and "fill_pct" in hydro.columns
has_pfc = pfc is not None and "price_shape" in (pfc.columns if pfc is not None else [])

# ── KPI Row ───────────────────────────────────────────────────────────────
bench = load_benchmarks(limit=1)
run_meta = latest_run_summary()

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    if has_epex:
        last_price = epex["price_eur_mwh"].iloc[-1]
        prev_price = epex["price_eur_mwh"].iloc[-96] if len(epex) > 96 else last_price
        st.metric("Spot EPEX (dernier)", format_eur(last_price),
                  delta=f"{last_price - prev_price:+.1f}", delta_color="inverse")
    else:
        st.metric("Spot EPEX", "—")

with k2:
    if has_epex and len(epex) > 96 * 7:
        avg_7d = epex["price_eur_mwh"].iloc[-96*7:].mean()
        avg_prev_7d = epex["price_eur_mwh"].iloc[-96*14:-96*7].mean() if len(epex) > 96*14 else avg_7d
        st.metric("Moyenne 7j", format_eur(avg_7d),
                  delta=f"{avg_7d - avg_prev_7d:+.1f} vs sem. préc.")
    else:
        st.metric("Moyenne 7j", "—")

with k3:
    if has_pfc:
        front_month = pfc["price_shape"].iloc[:96*30].mean()
        st.metric("PFC Front-Month", format_eur(front_month))
    else:
        st.metric("PFC Front-Month", "—")

with k4:
    if has_hydro:
        fill = hydro["fill_pct"].iloc[-1]
        fill_prev = hydro["fill_pct"].iloc[-2] if len(hydro) > 1 else fill
        st.metric("Réservoirs CH", format_pct(fill),
                  delta=f"{fill - fill_prev:+.1f}pp", delta_color="normal")
    else:
        st.metric("Réservoirs CH", "—")

with k5:
    if has_hydro:
        gwh = hydro["fill_gwh"].iloc[-1]
        max_gwh = hydro["max_capacity_gwh"].iloc[-1]
        st.metric("Stock hydro", format_gwh(gwh), delta=f"/ {max_gwh:.0f} GWh max")
    else:
        st.metric("Stock hydro", "—")

with k6:
    if not bench.empty and "mae" in bench.columns:
        b = bench.iloc[0]
        st.metric("MAE vs HFC", f"{float(b['mae']):.2f} EUR/MWh", delta=f"run {run_meta['run_id']}")
    else:
        st.metric("MAE vs HFC", "—")

st.divider()

# ── Main chart: EPEX spot + PFC overlay ───────────────────────────────────
st.subheader("Prix spot EPEX + PFC prévisionnelle")

if has_epex:
    # Downsample to hourly for overview
    epex_h = epex["price_eur_mwh"].resample("h").mean()

    fig = go.Figure()

    # Spot historique
    fig.add_trace(go.Scatter(
        x=epex_h.index, y=epex_h.values,
        name="EPEX Spot",
        line=dict(color=COLORS["blue"], width=1.5),
        hovertemplate="%{y:.1f} EUR/MWh<extra>Spot</extra>",
    ))

    # PFC overlay
    if has_pfc:
        pfc_h = pfc["price_shape"].resample("h").mean()
        fig.add_trace(go.Scatter(
            x=pfc_h.index, y=pfc_h.values,
            name="PFC Shape",
            line=dict(color=COLORS["amber"], width=2),
            hovertemplate="%{y:.1f} EUR/MWh<extra>PFC</extra>",
        ))

        # Bandes IC p10/p90
        if "p10" in pfc.columns and "p90" in pfc.columns:
            p10_h = pfc["p10"].resample("h").mean()
            p90_h = pfc["p90"].resample("h").mean()
            fig.add_trace(go.Scatter(
                x=p90_h.index, y=p90_h.values,
                name="p90", line=dict(width=0), showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=p10_h.index, y=p10_h.values,
                name="IC 80%", line=dict(width=0),
                fill="tonexty", fillcolor=COLORS["band"],
                hoverinfo="skip",
            ))

    fig.update_layout(
        yaxis_title="EUR/MWh",
        height=450,
        legend=dict(orientation="h", y=1.02, x=0),
    )
    fig = add_range_slider(fig)
    st.plotly_chart(fig, width="stretch")
else:
    no_data_warning("prix EPEX")

# ── Bottom row: Hydro + recent stats ─────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Réservoirs hydro CH")
    if has_hydro:
        current_year = pd.Timestamp.now().year
        hydro_plot = hydro[["fill_pct"]].copy()
        hydro_plot["week"] = hydro_plot.index.isocalendar().week.values.astype(int)
        hydro_plot["year"] = hydro_plot.index.year

        hist = hydro_plot[hydro_plot["year"] < current_year]
        if not hist.empty:
            envelope = hist.groupby("week")["fill_pct"].agg(["min", "median", "max"])
            curr = hydro_plot[hydro_plot["year"] == current_year].sort_values("week")

            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(
                x=envelope.index, y=envelope["max"],
                name="Max historique", line=dict(width=0), showlegend=False,
                hoverinfo="skip",
            ))
            fig_h.add_trace(go.Scatter(
                x=envelope.index, y=envelope["min"],
                name="Plage historique", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(88,166,255,0.08)",
                hoverinfo="skip",
            ))
            fig_h.add_trace(go.Scatter(
                x=envelope.index, y=envelope["median"],
                name="Médiane hist.", line=dict(color=COLORS["muted"], width=1, dash="dot"),
                hovertemplate="%{y:.1f}%<extra>Médiane</extra>",
            ))
            if not curr.empty:
                fig_h.add_trace(go.Scatter(
                    x=curr["week"], y=curr["fill_pct"],
                    name=str(current_year),
                    line=dict(color=COLORS["amber"], width=3),
                    hovertemplate="%{y:.1f}%<extra>" + str(current_year) + "</extra>",
                ))
            fig_h.update_layout(
                xaxis_title="Semaine ISO", yaxis_title="Remplissage %",
                yaxis_range=[0, 100], height=350,
                legend=dict(orientation="h", y=1.05, x=0),
            )
            st.plotly_chart(fig_h, width="stretch")
    else:
        no_data_warning("réservoirs hydro")

with col_right:
    st.subheader("Statistiques récentes")
    if has_epex:
        n_30d = min(96 * 30, len(epex))
        last_30d = epex["price_eur_mwh"].iloc[-n_30d:]
        stats = {
            "Période": f"{last_30d.index.min().strftime('%d/%m')} — {last_30d.index.max().strftime('%d/%m/%Y')}",
            "Moyenne": f"{last_30d.mean():.1f} EUR/MWh",
            "Médiane": f"{last_30d.median():.1f} EUR/MWh",
            "Min": f"{last_30d.min():.1f} EUR/MWh",
            "Max": f"{last_30d.max():.1f} EUR/MWh",
            "Volatilité (std)": f"{last_30d.std():.1f} EUR/MWh",
            "Nb heures négatives": f"{(last_30d < 0).sum()}",
        }
        for k, v in stats.items():
            st.markdown(f"**{k}** : {v}")

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=last_30d.values, nbinsx=50,
            marker_color=COLORS["blue"], opacity=0.7,
            hovertemplate="%{x:.0f} EUR/MWh<br>%{y} obs<extra></extra>",
        ))
        fig_dist.update_layout(
            xaxis_title="EUR/MWh", yaxis_title="Fréquence",
            height=220, margin=dict(l=40, r=10, t=10, b=40),
            bargap=0.05,
        )
        st.plotly_chart(fig_dist, width="stretch")

        export_csv_button(last_30d.to_frame(), "epex_30d.csv", "Export 30j CSV")
    else:
        no_data_warning("statistiques")


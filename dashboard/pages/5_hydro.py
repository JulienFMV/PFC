"""
Page 5 — Hydro & Fondamentaux
"Les drivers physiques"
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    COLORS, export_csv_button, format_gwh, format_pct, load_hydro,
    no_data_warning, show_freshness_sidebar,
)

st.header("Hydro & Fondamentaux")
st.caption("Réservoirs hydroélectriques suisses — données SFOE (opendata.swiss)")

show_freshness_sidebar()

hydro = load_hydro()

if hydro is None or "fill_pct" not in (hydro.columns if hydro is not None else []):
    no_data_warning("réservoirs hydro")
    st.stop()

# ── KPI Row ───────────────────────────────────────────────────────────────
latest = hydro.iloc[-1]
prev = hydro.iloc[-2] if len(hydro) > 1 else latest

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Remplissage CH", format_pct(latest["fill_pct"]),
              delta=f"{latest['fill_pct'] - prev['fill_pct']:+.1f}pp")
with k2:
    st.metric("Stock total", format_gwh(latest["fill_gwh"]),
              delta=f"{latest['fill_gwh'] - prev['fill_gwh']:+.0f} GWh")
with k3:
    st.metric("Capacité max", format_gwh(latest["max_capacity_gwh"]))
with k4:
    if "fill_deviation" in hydro.columns:
        dev = latest["fill_deviation"]
        st.metric("Fill deviation", f"{dev:+.2f} σ",
                  delta="Sous moyenne" if dev < 0 else "Sur moyenne",
                  delta_color="inverse" if dev < 0 else "normal")
with k5:
    if "wallis_gwh" in hydro.columns:
        st.metric("Valais", format_gwh(latest.get("wallis_gwh", 0)))

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Options")
    show_regions = st.checkbox("Détail par région", value=False)
    available_years = sorted(hydro.index.year.unique(), reverse=True)
    current_year = pd.Timestamp.now().year
    default_years = [y for y in [current_year, current_year - 1] if y in available_years]
    year_compare = st.multiselect(
        "Années à comparer", available_years, default=default_years,
    )

# ── Main chart: Fan chart ─────────────────────────────────────────────────
st.subheader("Niveaux de remplissage — fan chart historique")

hydro_plot = hydro[["fill_pct"]].copy()
hydro_plot["week"] = hydro_plot.index.isocalendar().week.values.astype(int)
hydro_plot["year"] = hydro_plot.index.year

hist = hydro_plot[hydro_plot["year"] < current_year - 1]

if not hist.empty:
    envelope = hist.groupby("week")["fill_pct"].agg(
        p10=lambda x: np.percentile(x, 10),
        p25=lambda x: np.percentile(x, 25),
        median="median",
        p75=lambda x: np.percentile(x, 75),
        p90=lambda x: np.percentile(x, 90),
    )

    fig = go.Figure()

    # p10-p90 band
    fig.add_trace(go.Scatter(
        x=envelope.index, y=envelope["p90"],
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=envelope.index, y=envelope["p10"],
        name="P10-P90 hist.", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(88,166,255,0.06)",
        hoverinfo="skip",
    ))

    # p25-p75 band
    fig.add_trace(go.Scatter(
        x=envelope.index, y=envelope["p75"],
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=envelope.index, y=envelope["p25"],
        name="P25-P75 hist.", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(88,166,255,0.12)",
        hoverinfo="skip",
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=envelope.index, y=envelope["median"],
        name="Médiane hist.",
        line=dict(color=COLORS["muted"], width=1.5, dash="dot"),
        hovertemplate="Sem %{x}: %{y:.1f}%<extra>Médiane</extra>",
    ))

    # Selected years
    year_colors = [COLORS["amber"], COLORS["blue"], COLORS["green"],
                   COLORS["red"], "#A855F7", "#EC4899"]
    for i, year in enumerate(sorted(year_compare)):
        yr_data = hydro_plot[hydro_plot["year"] == year].sort_values("week")
        if yr_data.empty:
            continue
        color = year_colors[i % len(year_colors)]
        width = 3 if year == current_year else 2
        fig.add_trace(go.Scatter(
            x=yr_data["week"], y=yr_data["fill_pct"],
            name=str(year),
            line=dict(color=color, width=width),
            hovertemplate=f"Sem %{{x}}: %{{y:.1f}}%<extra>{year}</extra>",
        ))

    fig.update_layout(
        xaxis_title="Semaine ISO",
        yaxis_title="Remplissage (%)",
        yaxis_range=[0, 100],
        height=500,
        legend=dict(orientation="h", y=1.05, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Pas assez de données historiques pour le fan chart.")

# ── Fill deviation time series ────────────────────────────────────────────
if "fill_deviation" in hydro.columns:
    st.subheader("Fill deviation (z-score vs historique)")

    dev = hydro["fill_deviation"].dropna()

    if not dev.empty:
        fig_dev = go.Figure()

        fig_dev.add_trace(go.Scatter(
            x=dev.index, y=dev.values,
            line=dict(color=COLORS["amber"], width=1.5),
            hovertemplate="%{x|%b %Y}: %{y:+.2f}σ<extra></extra>",
            showlegend=False,
        ))

        # Color positive/negative differently
        pos = dev[dev >= 0]
        neg = dev[dev < 0]
        fig_dev.add_trace(go.Bar(
            x=pos.index, y=pos.values,
            marker_color=COLORS["green"], opacity=0.4,
            showlegend=False, hoverinfo="skip",
        ))
        fig_dev.add_trace(go.Bar(
            x=neg.index, y=neg.values,
            marker_color=COLORS["red"], opacity=0.4,
            showlegend=False, hoverinfo="skip",
        ))

        fig_dev.add_hline(y=0, line_color=COLORS["muted"], line_width=1)
        fig_dev.add_hline(y=-1, line_dash="dot", line_color=COLORS["red"], opacity=0.5,
                          annotation_text="-1σ")
        fig_dev.add_hline(y=1, line_dash="dot", line_color=COLORS["green"], opacity=0.5,
                          annotation_text="+1σ")
        fig_dev.update_layout(
            yaxis_title="Z-score",
            height=350,
            barmode="overlay",
        )
        st.plotly_chart(fig_dev, use_container_width=True)

        st.markdown(
            "> **Lecture** : Négatif = réservoirs sous la moyenne historique pour cette "
            "semaine → pression haussière sur les prix. Positif = surplus hydro → pression baissière."
        )

# ── Regional breakdown ────────────────────────────────────────────────────
if show_regions and "wallis_gwh" in hydro.columns:
    st.subheader("Détail régional")

    regions = {
        "Valais": "wallis_gwh",
        "Grisons": "graubuenden_gwh",
        "Tessin": "tessin_gwh",
    }

    fig_reg = go.Figure()
    colors = [COLORS["amber"], COLORS["blue"], COLORS["green"]]
    for i, (name, col) in enumerate(regions.items()):
        if col in hydro.columns:
            fig_reg.add_trace(go.Scatter(
                x=hydro.index, y=hydro[col],
                name=name, stackgroup="one",
                line=dict(width=0.5, color=colors[i]),
                hovertemplate=f"%{{x|%b %Y}}: %{{y:.0f}} GWh<extra>{name}</extra>",
            ))

    if all(c in hydro.columns for c in regions.values()):
        reste = hydro["fill_gwh"] - sum(hydro[c] for c in regions.values())
        fig_reg.add_trace(go.Scatter(
            x=hydro.index, y=reste,
            name="Reste CH", stackgroup="one",
            line=dict(width=0.5, color=COLORS["muted"]),
        ))

    fig_reg.update_layout(
        yaxis_title="GWh", height=400,
        legend=dict(orientation="h", y=1.05, x=0),
    )
    st.plotly_chart(fig_reg, use_container_width=True)

    # Current snapshot
    st.markdown("**Snapshot actuel**")
    snapshot_data = {}
    for name, col in regions.items():
        if col in hydro.columns:
            snapshot_data[name] = f"{latest[col]:.0f} GWh"
    if all(c in hydro.columns for c in regions.values()):
        reste_val = latest["fill_gwh"] - sum(latest[c] for c in regions.values())
        snapshot_data["Reste CH"] = f"{reste_val:.0f} GWh"
    snapshot_data["Total CH"] = f"**{latest['fill_gwh']:.0f} GWh**"

    cols = st.columns(len(snapshot_data))
    for i, (k, v) in enumerate(snapshot_data.items()):
        cols[i].markdown(f"**{k}**\n\n{v}")

# ── Export ────────────────────────────────────────────────────────────────
with st.expander("Export"):
    export_csv_button(hydro, "hydro_reservoirs.csv", "Export données hydro complètes")

"""
Page 10 — Indisponibilites
"Qui est a l'arret ?"

Visualise les indisponibilites de production (REMIT UMM) :
- KPIs clairs avec deltas semaine precedente
- Graphe principal: stacked area par type (nucleaire, hydro, thermique)
- Graphes individuels par type pour une lecture facile
- Correlation outages-prix
- Statistiques mensuelles
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils import (
    COLORS, add_range_slider, export_csv_button, load_epex,
    no_data_warning, show_freshness_sidebar,
)

st.header("Indisponibilites de Production")
st.caption("REMIT UMM — Centrales nucleaires, hydrauliques et thermiques (ENTSO-E)")

show_freshness_sidebar()


# ── Load data ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_outages():
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent.parent / "pfc_shaping" / "data" / "outages_15min.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return df


outages = load_outages()
epex = load_epex()

if outages is None or outages.empty:
    no_data_warning("outages")
    st.info(
        "Les donnees d'indisponibilites ne sont pas encore disponibles. "
        "Lancez `python scripts/run_daily.py` pour les ingerer depuis ENTSO-E."
    )
    st.stop()

has_epex = epex is not None and "price_eur_mwh" in (epex.columns if epex is not None else [])

# ── Resample to daily for cleaner charts ──────────────────────────────
outages_h = outages.resample("h").mean()
outages_d = outages.resample("D").mean()

# ── KPI Row with deltas ──────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

latest = outages.iloc[-1] if len(outages) > 0 else pd.Series()

# Compute 7-day averages for deltas
now_7d = outages.last("7D").mean() if len(outages) > 96 * 7 else latest
prev_7d = outages.iloc[:-96*7].last("7D").mean() if len(outages) > 96 * 14 else now_7d

with k1:
    total_mw = latest.get("unavailable_mw", 0)
    delta_total = now_7d.get("unavailable_mw", 0) - prev_7d.get("unavailable_mw", 0)
    st.metric("Total", f"{total_mw:,.0f} MW",
              delta=f"{delta_total:+,.0f}", delta_color="inverse")

with k2:
    nuc_mw = latest.get("unavailable_nuclear", 0)
    delta_nuc = now_7d.get("unavailable_nuclear", 0) - prev_7d.get("unavailable_nuclear", 0)
    st.metric("Nucleaire", f"{nuc_mw:,.0f} MW",
              delta=f"{delta_nuc:+,.0f}", delta_color="inverse")

with k3:
    hyd_mw = latest.get("unavailable_hydro", 0)
    delta_hyd = now_7d.get("unavailable_hydro", 0) - prev_7d.get("unavailable_hydro", 0)
    st.metric("Hydro", f"{hyd_mw:,.0f} MW",
              delta=f"{delta_hyd:+,.0f}", delta_color="inverse")

with k4:
    n_out = latest.get("n_outages", 0)
    st.metric("En arret", f"{int(n_out)}")

st.divider()

# ── Time range selector ──────────────────────────────────────────────
range_options = {"3 mois": 90, "6 mois": 180, "1 an": 365, "Tout": 0}
selected_range = st.radio(
    "Periode", list(range_options.keys()),
    horizontal=True, index=1, label_visibility="collapsed",
)
n_days = range_options[selected_range]
if n_days > 0:
    cutoff = outages_d.index.max() - pd.Timedelta(days=n_days)
    outages_plot = outages_d[outages_d.index >= cutoff]
    outages_h_plot = outages_h[outages_h.index >= cutoff]
else:
    outages_plot = outages_d
    outages_h_plot = outages_h

# ── Tab layout ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Vue d'ensemble", "Detail par type", "Correlation Prix", "Statistiques",
])

# ════════════════════════════════════════════════════════════════════════
# TAB 1: Vue d'ensemble — stacked area propre
# ════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Capacite indisponible totale (MW)")

    fig = go.Figure()

    # Properly stacked area (stackgroup ensures correct stacking)
    if "unavailable_nuclear" in outages_plot.columns:
        fig.add_trace(go.Scatter(
            x=outages_plot.index, y=outages_plot["unavailable_nuclear"],
            name="Nucleaire",
            stackgroup="one",
            fillcolor="rgba(220, 60, 60, 0.6)",
            line=dict(color="#DC3C3C", width=0.5),
            hovertemplate="%{y:,.0f} MW<extra>Nucleaire</extra>",
        ))

    if "unavailable_hydro" in outages_plot.columns:
        fig.add_trace(go.Scatter(
            x=outages_plot.index, y=outages_plot["unavailable_hydro"],
            name="Hydro",
            stackgroup="one",
            fillcolor="rgba(50, 120, 190, 0.6)",
            line=dict(color="#3278BE", width=0.5),
            hovertemplate="%{y:,.0f} MW<extra>Hydro</extra>",
        ))

    if "unavailable_thermal" in outages_plot.columns:
        fig.add_trace(go.Scatter(
            x=outages_plot.index, y=outages_plot["unavailable_thermal"],
            name="Thermique",
            stackgroup="one",
            fillcolor="rgba(150, 150, 150, 0.5)",
            line=dict(color="#969696", width=0.5),
            hovertemplate="%{y:,.0f} MW<extra>Thermique</extra>",
        ))

    fig.update_layout(
        height=400,
        margin=dict(l=60, r=20, t=10, b=40),
        yaxis_title="MW indisponible",
        legend=dict(orientation="h", y=1.05, x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Nombre d'unites — petit bar chart en dessous
    st.caption("Nombre d'unites en arret")
    fig_n = go.Figure()
    if "n_outages" in outages_plot.columns:
        fig_n.add_trace(go.Bar(
            x=outages_plot.index, y=outages_plot["n_outages"],
            marker_color="rgba(90, 107, 138, 0.6)",
            hovertemplate="%{y:.0f} unites<extra></extra>",
        ))
    fig_n.update_layout(
        height=150,
        margin=dict(l=60, r=20, t=5, b=30),
        yaxis_title="N",
        showlegend=False,
    )
    st.plotly_chart(fig_n, use_container_width=True)

    export_csv_button(outages_plot, "outages_daily")

# ════════════════════════════════════════════════════════════════════════
# TAB 2: Detail par type — un graphe par categorie
# ════════════════════════════════════════════════════════════════════════
with tab2:
    fuel_types = [
        ("Nucleaire", "unavailable_nuclear", "#DC3C3C", "rgba(220, 60, 60, 0.15)"),
        ("Hydro", "unavailable_hydro", "#3278BE", "rgba(50, 120, 190, 0.15)"),
        ("Thermique", "unavailable_thermal", "#969696", "rgba(150, 150, 150, 0.15)"),
    ]

    for label, col, color, fill_color in fuel_types:
        if col not in outages_plot.columns:
            continue

        series = outages_plot[col]
        avg_val = series.mean()
        max_val = series.max()
        current = series.iloc[-1] if len(series) > 0 else 0

        # Mini KPI + chart
        st.subheader(f"{label}")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Actuel", f"{current:,.0f} MW")
        with m2:
            st.metric("Moyenne", f"{avg_val:,.0f} MW")
        with m3:
            st.metric("Max", f"{max_val:,.0f} MW")

        fig_detail = go.Figure()

        # Daily line
        fig_detail.add_trace(go.Scatter(
            x=series.index, y=series.values,
            fill="tozeroy",
            fillcolor=fill_color,
            line=dict(color=color, width=1.5),
            hovertemplate="%{y:,.0f} MW<extra></extra>",
        ))

        # Moving average 7d
        ma7 = series.rolling(7, min_periods=1).mean()
        fig_detail.add_trace(go.Scatter(
            x=ma7.index, y=ma7.values,
            line=dict(color=color, width=2.5, dash="dot"),
            name="Moy. 7j",
            hovertemplate="%{y:,.0f} MW (7j)<extra></extra>",
        ))

        fig_detail.update_layout(
            height=250,
            margin=dict(l=60, r=20, t=10, b=30),
            yaxis_title="MW",
            showlegend=False,
        )
        st.plotly_chart(fig_detail, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# TAB 3: Correlation outages vs prix spot
# ════════════════════════════════════════════════════════════════════════
with tab3:
    if not has_epex:
        st.warning("Pas de donnees EPEX pour la correlation")
    else:
        epex_h = epex[["price_eur_mwh"]].resample("h").mean()
        common = outages_h_plot.index.intersection(epex_h.index)

        if len(common) < 24:
            st.warning("Pas assez de donnees communes outages/EPEX")
        else:
            merged = outages_h_plot.loc[common].join(epex_h.loc[common])

            # Dual-axis: Prix + Indisponibilite
            st.subheader("Prix spot vs Indisponibilite")

            fig2 = make_subplots(specs=[[{"secondary_y": True}]])

            fig2.add_trace(go.Scatter(
                x=merged.index, y=merged["price_eur_mwh"],
                name="Prix EPEX",
                line=dict(color=COLORS.get("blue", "#0F52CC"), width=1),
                opacity=0.6,
                hovertemplate="%{y:.1f} EUR/MWh<extra>Prix</extra>",
            ), secondary_y=False)

            fig2.add_trace(go.Scatter(
                x=merged.index, y=merged["unavailable_mw"],
                name="Indisponible",
                line=dict(color="#DC3C3C", width=2),
                hovertemplate="%{y:,.0f} MW<extra>Indispo</extra>",
            ), secondary_y=True)

            fig2.update_yaxes(title_text="Prix (EUR/MWh)", secondary_y=False)
            fig2.update_yaxes(title_text="Indisponible (MW)", secondary_y=True)
            fig2.update_layout(
                height=400,
                margin=dict(l=60, r=60, t=10, b=40),
                legend=dict(orientation="h", y=1.05, x=0),
                hovermode="x unified",
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Scatter plots
            st.subheader("Correlations")
            col1, col2 = st.columns(2)

            daily = merged.resample("D").agg({
                "unavailable_mw": "mean",
                "price_eur_mwh": "mean",
            }).dropna()

            with col1:
                if len(daily) > 10:
                    corr = daily["unavailable_mw"].corr(daily["price_eur_mwh"])
                    fig_scatter = go.Figure()
                    fig_scatter.add_trace(go.Scatter(
                        x=daily["unavailable_mw"],
                        y=daily["price_eur_mwh"],
                        mode="markers",
                        marker=dict(
                            size=6, color=COLORS.get("blue", "#0F52CC"),
                            opacity=0.4, line=dict(width=0),
                        ),
                    ))
                    # Trend line
                    z = np.polyfit(daily["unavailable_mw"], daily["price_eur_mwh"], 1)
                    x_line = np.linspace(daily["unavailable_mw"].min(), daily["unavailable_mw"].max(), 50)
                    fig_scatter.add_trace(go.Scatter(
                        x=x_line, y=np.polyval(z, x_line),
                        mode="lines",
                        line=dict(color=COLORS.get("red", "#C63D3D"), width=2, dash="dash"),
                        showlegend=False,
                    ))
                    fig_scatter.update_layout(
                        xaxis_title="Indisponibilite (MW)",
                        yaxis_title="Prix (EUR/MWh)",
                        title=f"Total — r = {corr:.3f}",
                        height=350,
                        margin=dict(l=50, r=20, t=40, b=50),
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

            with col2:
                if "unavailable_nuclear" in merged.columns:
                    daily_nuc = merged.resample("D").agg({
                        "unavailable_nuclear": "mean",
                        "price_eur_mwh": "mean",
                    }).dropna()

                    if len(daily_nuc) > 10:
                        corr_nuc = daily_nuc["unavailable_nuclear"].corr(daily_nuc["price_eur_mwh"])
                        fig_nuc = go.Figure()
                        fig_nuc.add_trace(go.Scatter(
                            x=daily_nuc["unavailable_nuclear"],
                            y=daily_nuc["price_eur_mwh"],
                            mode="markers",
                            marker=dict(
                                size=6, color="#DC3C3C",
                                opacity=0.4, line=dict(width=0),
                            ),
                        ))
                        z_nuc = np.polyfit(daily_nuc["unavailable_nuclear"], daily_nuc["price_eur_mwh"], 1)
                        x_nuc = np.linspace(daily_nuc["unavailable_nuclear"].min(), daily_nuc["unavailable_nuclear"].max(), 50)
                        fig_nuc.add_trace(go.Scatter(
                            x=x_nuc, y=np.polyval(z_nuc, x_nuc),
                            mode="lines",
                            line=dict(color=COLORS.get("red", "#C63D3D"), width=2, dash="dash"),
                            showlegend=False,
                        ))
                        fig_nuc.update_layout(
                            xaxis_title="Nucleaire indisponible (MW)",
                            yaxis_title="Prix (EUR/MWh)",
                            title=f"Nucleaire — r = {corr_nuc:.3f}",
                            height=350,
                            margin=dict(l=50, r=20, t=40, b=50),
                        )
                        st.plotly_chart(fig_nuc, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# TAB 4: Statistiques
# ════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Statistiques des indisponibilites")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Resume global**")
        stats = {
            "Periode": f"{outages.index.min().date()} — {outages.index.max().date()}",
            "Indisponibilite moyenne": f"{outages['unavailable_mw'].mean():,.0f} MW",
            "Indisponibilite max": f"{outages['unavailable_mw'].max():,.0f} MW",
            "Nucleaire moyen": f"{outages['unavailable_nuclear'].mean():,.0f} MW",
            "Hydro moyen": f"{outages['unavailable_hydro'].mean():,.0f} MW",
            "Outages simultanes (moy)": f"{outages['n_outages'].mean():.1f}",
            "Outages simultanes (max)": f"{int(outages['n_outages'].max())}",
        }
        for k, v in stats.items():
            st.text(f"{k}: {v}")

    with col2:
        st.markdown("**Indisponibilite moyenne par mois (MW)**")
        outages_monthly = outages.copy()
        idx_local = outages_monthly.index.tz_convert("Europe/Zurich")
        outages_monthly["month"] = idx_local.month
        outages_monthly["year"] = idx_local.year
        pivot = outages_monthly.groupby(["year", "month"])["unavailable_mw"].mean().unstack()
        if not pivot.empty:
            pivot.columns = [f"M{m:02d}" for m in pivot.columns]
            st.dataframe(pivot.style.format("{:.0f}"), height=200)

    # Raw data expander
    with st.expander("Donnees brutes (dernieres 48h)"):
        last_48h = outages.last("48h")
        st.dataframe(last_48h, height=300)

    export_csv_button(outages, "outages_raw")

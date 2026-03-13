"""
Page 3 â€” Shape Factors
"Comment le prix varie intra-jour / semaine / saison"
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    COLORS, export_csv_button, load_epex, no_data_warning, show_freshness_sidebar,
)

st.header("Shape Factors")
st.caption("DÃ©composition multiplicative : saisonnier, jour, horaire, 15min")

show_freshness_sidebar()

epex = load_epex()

if epex is None or "price_eur_mwh" not in (epex.columns if epex is not None else []):
    no_data_warning("prix EPEX (nÃ©cessaire pour calculer les shape factors)")
    st.stop()

# â”€â”€ Enrich with calendar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    from pfc_shaping.data.calendar_ch import enrich_15min_index
    cal = enrich_15min_index(epex.index)
except ImportError:
    st.error(
        "Module `holidays` non installÃ©. "
        "Lance `pip install holidays` puis relance le dashboard."
    )
    st.stop()
except Exception as e:
    st.error(f"Erreur enrichissement calendrier : {e}")
    st.stop()

df = epex[["price_eur_mwh"]].join(cal)

# â”€â”€ Sidebar filters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    st.subheader("Filtres")
    saisons = sorted(df["saison"].unique().tolist())
    sel_saisons = st.multiselect("Saisons", saisons, default=saisons)
    types_jour = sorted(df["type_jour"].unique().tolist())
    sel_types = st.multiselect("Types de jour", types_jour, default=types_jour)

mask = df["saison"].isin(sel_saisons) & df["type_jour"].isin(sel_types)
df_f = df[mask]

if df_f.empty:
    st.warning("Aucune donnÃ©e pour cette combinaison de filtres.")
    st.stop()

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
tab1, tab2, tab3, tab4 = st.tabs(["Profil horaire (f_H)", "Jour semaine (f_W)",
                                   "Intra-horaire (f_Q)", "Heatmap"])

# â”€â”€ TAB 1: f_H â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tab1:
    st.subheader("Facteur horaire f_H par saison x type de jour")

    hourly = (
        df_f.groupby(["saison", "type_jour", "heure_hce"])["price_eur_mwh"]
        .mean()
        .reset_index()
    )
    daily_mean = (
        df_f.groupby(["saison", "type_jour"])["price_eur_mwh"]
        .mean()
        .reset_index()
        .rename(columns={"price_eur_mwh": "daily_mean"})
    )
    hourly = hourly.merge(daily_mean, on=["saison", "type_jour"])
    hourly["f_H"] = np.where(
        hourly["daily_mean"].abs() > 0.1,
        hourly["price_eur_mwh"] / hourly["daily_mean"],
        1.0,
    )

    fig_fh = px.line(
        hourly, x="heure_hce", y="f_H",
        color="saison", facet_col="type_jour", facet_col_wrap=3,
        labels={"heure_hce": "Heure", "f_H": "f_H (ratio)", "saison": "Saison"},
        color_discrete_sequence=[COLORS["amber"], COLORS["blue"],
                                 COLORS["green"], COLORS["red"]],
    )
    fig_fh.add_hline(y=1.0, line_dash="dot", line_color=COLORS["muted"], opacity=0.5)
    fig_fh.update_layout(height=450)
    st.plotly_chart(fig_fh, width="stretch")

    with st.expander("Export"):
        export_csv_button(hourly, "shape_f_H.csv", "Export f_H")

# â”€â”€ TAB 2: f_W â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tab2:
    st.subheader("Facteur jour de semaine f_W")

    weekly_mean = df_f["price_eur_mwh"].mean()
    if abs(weekly_mean) < 0.1:
        st.warning("Moyenne des prix proche de zÃ©ro â€” f_W non calculable.")
    else:
        fw = (
            df_f.groupby("type_jour")["price_eur_mwh"]
            .mean()
            .reset_index()
        )
        fw["f_W"] = fw["price_eur_mwh"] / weekly_mean
        fw = fw.sort_values("f_W", ascending=True)

        fw["color"] = fw["f_W"].apply(
            lambda v: COLORS["green"] if v >= 1 else COLORS["red"]
        )

        fig_fw = go.Figure()
        fig_fw.add_trace(go.Bar(
            x=fw["f_W"], y=fw["type_jour"],
            orientation="h",
            marker_color=fw["color"],
            text=fw["f_W"].apply(lambda v: f"{v:.3f}"),
            textposition="outside",
            hovertemplate="%{y}: f_W = %{x:.3f}<extra></extra>",
        ))
        fig_fw.add_vline(x=1.0, line_dash="dot", line_color=COLORS["muted"])
        fig_fw.update_layout(
            xaxis_title="f_W (ratio vs moyenne)",
            height=300,
            xaxis_range=[0.6, 1.2],
        )
        st.plotly_chart(fig_fw, width="stretch")

        st.markdown(
            "> **Lecture** : Ouvrable > 1 = prix plus elevÃ©s en semaine. "
            "Ferie_CH < 1 = forte baisse les jours fÃ©riÃ©s suisses."
        )

# â”€â”€ TAB 3: f_Q â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tab3:
    st.subheader("Facteur intra-horaire f_Q (15min)")

    sel_hour = st.slider("Heure Ã  analyser", 0, 23, 8)

    q_data = df_f[df_f["heure_hce"] == sel_hour].copy()
    if not q_data.empty:
        hour_mean = q_data.groupby(q_data.index.floor("h"))["price_eur_mwh"].transform("mean")
        valid = hour_mean.abs() > 0.5
        q_data = q_data[valid]
        if not q_data.empty:
            q_data["f_Q"] = q_data["price_eur_mwh"] / hour_mean[valid]

            q_stats = (
                q_data.groupby(["saison", "quart"])["f_Q"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )

            fig_q = px.bar(
                q_stats, x="quart", y="mean", color="saison",
                barmode="group", error_y="std",
                labels={"quart": "Quart-d'heure", "mean": "f_Q moyen", "saison": "Saison"},
                color_discrete_sequence=[COLORS["amber"], COLORS["blue"],
                                         COLORS["green"], COLORS["red"]],
            )
            fig_q.add_hline(y=1.0, line_dash="dot", line_color=COLORS["muted"])
            fig_q.update_layout(height=350)
            st.plotly_chart(fig_q, width="stretch")

            st.markdown(
                f"> **Heure {sel_hour}h** â€” Q1=:00, Q2=:15, Q3=:30, Q4=:45. "
                "Les heures de rampe (7-10h, 17-20h) montrent les plus grands Ã©carts."
            )
        else:
            st.info("Pas assez de donnÃ©es valides pour cette heure.")
    else:
        st.info("Pas de donnÃ©es pour cette sÃ©lection.")

# â”€â”€ TAB 4: Heatmap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with tab4:
    st.subheader("Heatmap prix moyen : Mois x Heure")

    df_f_heat = df_f.copy()
    df_f_heat["month"] = df_f_heat.index.month
    heat_data = df_f_heat.groupby(["month", "heure_hce"])["price_eur_mwh"].mean().reset_index()
    heat_pivot = heat_data.pivot(index="month", columns="heure_hce", values="price_eur_mwh")

    month_labels = ["Jan", "FÃ©v", "Mar", "Avr", "Mai", "Jun",
                    "Jul", "AoÃ»", "Sep", "Oct", "Nov", "DÃ©c"]
    # Only use labels for months present
    y_labels = [month_labels[m-1] for m in heat_pivot.index]

    fig_heat = go.Figure(data=go.Heatmap(
        z=heat_pivot.values,
        x=list(range(24)),
        y=y_labels,
        colorscale=[
            [0, "#0D4A6B"],
            [0.3, "#1A7FA8"],
            [0.5, "#F0E442"],
            [0.7, "#E8850C"],
            [1.0, "#D62728"],
        ],
        colorbar=dict(title="EUR/MWh", titleside="right"),
        hovertemplate="Mois: %{y}<br>Heure: %{x}h<br>Prix: %{z:.1f} EUR/MWh<extra></extra>",
    ))
    fig_heat.update_layout(
        xaxis_title="Heure du jour (HCE)",
        yaxis=dict(autorange="reversed"),
        height=450,
    )
    st.plotly_chart(fig_heat, width="stretch")

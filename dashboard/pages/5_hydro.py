"""
Page 5 — Hydro & Fondamentaux
"Les drivers physiques"
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    COLORS, add_range_slider, export_csv_button, format_gwh, format_pct,
    load_entso, load_epex, load_hydro, no_data_warning, show_freshness_sidebar,
)

st.header("Hydro & Fondamentaux")
st.caption("Drivers physiques du marché suisse — SFOE, ENTSO-E, EPEX")

show_freshness_sidebar()

hydro = load_hydro()
entso = load_entso()
epex = load_epex()

has_hydro = hydro is not None and "fill_pct" in (hydro.columns if hydro is not None else [])
has_entso = entso is not None and not entso.empty
has_epex = epex is not None and "price_eur_mwh" in (epex.columns if epex is not None else [])

if not has_hydro and not has_entso:
    no_data_warning("fondamentaux")
    st.stop()

# ── KPI Row ───────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)

if has_hydro:
    latest = hydro.iloc[-1]
    prev = hydro.iloc[-2] if len(hydro) > 1 else latest
    with k1:
        st.metric("Remplissage CH", format_pct(latest["fill_pct"]),
                  delta=f"{latest['fill_pct'] - prev['fill_pct']:+.1f}pp")
    with k2:
        st.metric("Stock hydro", format_gwh(latest["fill_gwh"]),
                  delta=f"{latest['fill_gwh'] - prev['fill_gwh']:+.0f} GWh")
    with k3:
        if "fill_deviation" in hydro.columns:
            dev = latest["fill_deviation"]
            st.metric("Fill deviation", f"{dev:+.2f} σ",
                      delta="Sous moyenne" if dev < 0 else "Sur moyenne",
                      delta_color="inverse" if dev < 0 else "normal")

if has_entso:
    last_week = entso.iloc[-96*7:]
    with k4:
        avg_load = last_week["load_mw"].mean()
        st.metric("Charge moy. 7j", f"{avg_load/1000:.1f} GW")
    with k5:
        avg_solar = last_week["solar_mw"].mean()
        st.metric("Solaire moy. 7j", f"{avg_solar:.0f} MW")
    with k6:
        avg_xb = last_week["cross_border_mw"].mean()
        direction = "Import" if avg_xb > 0 else "Export"
        st.metric(f"{direction} moy. 7j", f"{abs(avg_xb):.0f} MW")

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────
current_year = pd.Timestamp.now().year
with st.sidebar:
    st.subheader("Options")
    show_regions = st.checkbox("Détail par région", value=False) if has_hydro else False
    if has_hydro:
        available_years = sorted(hydro.index.year.unique(), reverse=True)
        default_years = [y for y in [current_year, current_year - 1] if y in available_years]
        year_compare = st.multiselect(
            "Années à comparer", available_years, default=default_years,
        )
    else:
        year_compare = []
    entso_window = st.selectbox("Fenêtre ENTSO-E", ["30j", "90j", "1an", "Tout"], index=1)

entso_lookback = {"30j": 96 * 30, "90j": 96 * 90, "1an": 96 * 365, "Tout": len(entso) if has_entso else 0}
entso_n = entso_lookback.get(entso_window, 96 * 90)

# ── Tab layout ────────────────────────────────────────────────────────────
tab_hydro, tab_gen, tab_load, tab_corr = st.tabs([
    "Réservoirs hydro", "Mix de production", "Charge & échanges", "Corrélations",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Réservoirs hydro
# ══════════════════════════════════════════════════════════════════════════
with tab_hydro:
    if not has_hydro:
        no_data_warning("réservoirs hydro")
    else:
        # Fan chart
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
            fig.add_trace(go.Scatter(
                x=envelope.index, y=envelope["median"],
                name="Médiane hist.",
                line=dict(color=COLORS["muted"], width=1.5, dash="dot"),
                hovertemplate="Sem %{x}: %{y:.1f}%<extra>Médiane</extra>",
            ))

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
                xaxis_title="Semaine ISO", yaxis_title="Remplissage (%)",
                yaxis_range=[0, 100], height=500,
                legend=dict(orientation="h", y=1.05, x=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Water value proxy
        if "water_value_proxy" in hydro.columns:
            st.subheader("Water value proxy")
            wv = hydro["water_value_proxy"].dropna()
            if not wv.empty:
                fig_wv = go.Figure()
                fig_wv.add_trace(go.Scatter(
                    x=wv.index, y=wv.values,
                    line=dict(color=COLORS["amber"], width=1.5),
                    hovertemplate="%{x|%b %Y}: %{y:+.3f}<extra></extra>",
                    showlegend=False,
                ))
                fig_wv.add_hline(y=0, line_color=COLORS["muted"], line_width=1)
                fig_wv.update_layout(yaxis_title="Proxy (normalisé)", height=300)
                st.plotly_chart(fig_wv, use_container_width=True)
                st.markdown(
                    "> **Lecture** : Positif = eau rare (valeur haute → prix soutenus). "
                    "Négatif = surplus hydro (pression baissière)."
                )

        # Fill deviation
        if "fill_deviation" in hydro.columns:
            st.subheader("Fill deviation (z-score vs historique)")
            dev = hydro["fill_deviation"].dropna()
            if not dev.empty:
                fig_dev = go.Figure()
                pos = dev[dev >= 0]
                neg = dev[dev < 0]
                fig_dev.add_trace(go.Bar(
                    x=pos.index, y=pos.values,
                    marker_color=COLORS["green"], opacity=0.5,
                    showlegend=False, hoverinfo="skip",
                ))
                fig_dev.add_trace(go.Bar(
                    x=neg.index, y=neg.values,
                    marker_color=COLORS["red"], opacity=0.5,
                    showlegend=False, hoverinfo="skip",
                ))
                fig_dev.add_trace(go.Scatter(
                    x=dev.index, y=dev.values,
                    line=dict(color=COLORS["amber"], width=1.5),
                    hovertemplate="%{x|%b %Y}: %{y:+.2f}σ<extra></extra>",
                    showlegend=False,
                ))
                fig_dev.add_hline(y=0, line_color=COLORS["muted"], line_width=1)
                fig_dev.add_hline(y=-1, line_dash="dot", line_color=COLORS["red"], opacity=0.5,
                                  annotation_text="-1σ")
                fig_dev.add_hline(y=1, line_dash="dot", line_color=COLORS["green"], opacity=0.5,
                                  annotation_text="+1σ")
                fig_dev.update_layout(yaxis_title="Z-score", height=300, barmode="overlay")
                st.plotly_chart(fig_dev, use_container_width=True)

        # Regional breakdown
        if show_regions and "wallis_gwh" in hydro.columns:
            st.subheader("Détail régional")
            regions = {"Valais": "wallis_gwh", "Grisons": "graubuenden_gwh", "Tessin": "tessin_gwh"}
            fig_reg = go.Figure()
            reg_colors = [COLORS["amber"], COLORS["blue"], COLORS["green"]]
            for i, (name, col) in enumerate(regions.items()):
                if col in hydro.columns:
                    fig_reg.add_trace(go.Scatter(
                        x=hydro.index, y=hydro[col], name=name, stackgroup="one",
                        line=dict(width=0.5, color=reg_colors[i]),
                        hovertemplate=f"%{{x|%b %Y}}: %{{y:.0f}} GWh<extra>{name}</extra>",
                    ))
            if all(c in hydro.columns for c in regions.values()):
                reste = hydro["fill_gwh"] - sum(hydro[c] for c in regions.values())
                fig_reg.add_trace(go.Scatter(
                    x=hydro.index, y=reste, name="Reste CH", stackgroup="one",
                    line=dict(width=0.5, color=COLORS["muted"]),
                ))
            fig_reg.update_layout(yaxis_title="GWh", height=400,
                                  legend=dict(orientation="h", y=1.05, x=0))
            st.plotly_chart(fig_reg, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Mix de production
# ══════════════════════════════════════════════════════════════════════════
with tab_gen:
    if not has_entso:
        no_data_warning("données ENTSO-E")
    else:
        st.subheader("Mix de production CH")
        entso_view = entso.iloc[-entso_n:]

        # Resample to daily for readability
        gen_daily = entso_view[["nuclear_mw", "hydro_ror_mw", "hydro_reservoir_mw",
                                "hydro_pumped_mw", "solar_mw", "wind_mw"]].resample("D").mean()

        gen_map = {
            "nuclear_mw": ("Nucléaire", "#6366F1"),
            "hydro_ror_mw": ("Hydro fil eau", COLORS["blue"]),
            "hydro_reservoir_mw": ("Hydro réservoir", "#0EA5E9"),
            "hydro_pumped_mw": ("Pompage-turbinage", "#22D3EE"),
            "solar_mw": ("Solaire", COLORS["amber"]),
            "wind_mw": ("Éolien", COLORS["green"]),
        }

        fig_gen = go.Figure()
        for col, (label, color) in gen_map.items():
            if col in gen_daily.columns:
                fig_gen.add_trace(go.Scatter(
                    x=gen_daily.index, y=gen_daily[col],
                    name=label, stackgroup="one",
                    line=dict(width=0.5, color=color),
                    hovertemplate=f"%{{x|%d/%m/%Y}}: %{{y:.0f}} MW<extra>{label}</extra>",
                ))

        fig_gen.update_layout(
            yaxis_title="MW (moyenne journalière)", height=450,
            legend=dict(orientation="h", y=1.05, x=0),
        )
        fig_gen = add_range_slider(fig_gen)
        st.plotly_chart(fig_gen, use_container_width=True)

        # Generation shares (pie)
        st.subheader("Répartition moyenne")
        avg_gen = gen_daily[[c for c in gen_map if c in gen_daily.columns]].mean()
        avg_gen = avg_gen[avg_gen > 0]

        col_pie, col_stats = st.columns([1, 1])
        with col_pie:
            fig_pie = go.Figure(go.Pie(
                labels=[gen_map[c][0] for c in avg_gen.index],
                values=avg_gen.values,
                marker=dict(colors=[gen_map[c][1] for c in avg_gen.index]),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:.0f} MW (%{percent})<extra></extra>",
            ))
            fig_pie.update_layout(height=350, showlegend=False,
                                  margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_stats:
            total_gen = avg_gen.sum()
            st.markdown(f"**Production totale moyenne** : {total_gen:.0f} MW")
            for col_name in avg_gen.index:
                label, _ = gen_map[col_name]
                val = avg_gen[col_name]
                pct = val / total_gen * 100
                st.markdown(f"- **{label}** : {val:.0f} MW ({pct:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Charge & échanges transfrontaliers
# ══════════════════════════════════════════════════════════════════════════
with tab_load:
    if not has_entso:
        no_data_warning("données ENTSO-E")
    else:
        entso_view = entso.iloc[-entso_n:]

        # Load
        st.subheader("Charge réseau CH")
        load_daily = entso_view["load_mw"].resample("D").agg(["mean", "min", "max"])

        fig_load = go.Figure()
        fig_load.add_trace(go.Scatter(
            x=load_daily.index, y=load_daily["max"],
            name="Max", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_load.add_trace(go.Scatter(
            x=load_daily.index, y=load_daily["min"],
            name="Plage jour", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(15,82,204,0.1)",
            hoverinfo="skip",
        ))
        fig_load.add_trace(go.Scatter(
            x=load_daily.index, y=load_daily["mean"],
            name="Charge moy.", line=dict(color=COLORS["blue"], width=2),
            hovertemplate="%{x|%d/%m}: %{y:.0f} MW<extra>Charge</extra>",
        ))
        fig_load.update_layout(yaxis_title="MW", height=350,
                               legend=dict(orientation="h", y=1.05, x=0))
        fig_load = add_range_slider(fig_load)
        st.plotly_chart(fig_load, use_container_width=True)

        # Cross-border flows
        st.subheader("Flux transfrontaliers CH")
        xb_daily = entso_view["cross_border_mw"].resample("D").mean()

        fig_xb = go.Figure()
        pos_xb = xb_daily.clip(lower=0)
        neg_xb = xb_daily.clip(upper=0)
        fig_xb.add_trace(go.Bar(
            x=pos_xb.index, y=pos_xb.values,
            name="Import", marker_color=COLORS["red"], opacity=0.7,
            hovertemplate="%{x|%d/%m}: %{y:.0f} MW<extra>Import</extra>",
        ))
        fig_xb.add_trace(go.Bar(
            x=neg_xb.index, y=neg_xb.values,
            name="Export", marker_color=COLORS["green"], opacity=0.7,
            hovertemplate="%{x|%d/%m}: %{y:.0f} MW<extra>Export</extra>",
        ))
        fig_xb.add_hline(y=0, line_color=COLORS["muted"], line_width=1)
        fig_xb.update_layout(yaxis_title="MW (positif = import)", height=350,
                             barmode="relative",
                             legend=dict(orientation="h", y=1.05, x=0))
        st.plotly_chart(fig_xb, use_container_width=True)

        # Load deviation heatmap
        st.subheader("Écart de charge par heure")
        ld = entso_view[["load_deviation"]].copy()
        ld["hour"] = ld.index.hour
        ld["date"] = ld.index.date
        pivot = ld.pivot_table(values="load_deviation", index="hour", columns="date", aggfunc="mean")
        # Subsample columns if too many
        if pivot.shape[1] > 90:
            step = pivot.shape[1] // 90
            pivot = pivot.iloc[:, ::step]

        fig_hm = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[str(d) for d in pivot.columns],
            y=pivot.index,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="σ"),
            hovertemplate="Heure %{y}h<br>%{x}<br>%{z:+.2f}σ<extra></extra>",
        ))
        fig_hm.update_layout(
            yaxis_title="Heure", yaxis=dict(dtick=2),
            height=350,
        )
        st.plotly_chart(fig_hm, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — Corrélations prix / fondamentaux
# ══════════════════════════════════════════════════════════════════════════
with tab_corr:
    if not has_entso or not has_epex:
        st.info("Les données EPEX et ENTSO-E sont nécessaires pour les corrélations.")
    else:
        st.subheader("Corrélations prix spot vs fondamentaux")

        # Merge on common index
        common = epex[["price_eur_mwh"]].join(entso, how="inner")
        if len(common) > 96 * 365:
            common = common.iloc[-96 * 365:]  # last year

        corr_cols = ["load_mw", "nuclear_mw", "solar_mw", "hydro_reservoir_mw",
                     "cross_border_mw", "wind_mw"]
        corr_cols = [c for c in corr_cols if c in common.columns]

        corr_labels = {
            "load_mw": "Charge",
            "nuclear_mw": "Nucléaire",
            "solar_mw": "Solaire",
            "hydro_reservoir_mw": "Hydro réservoir",
            "cross_border_mw": "Flux transfrontalier",
            "wind_mw": "Éolien",
        }

        # Hourly aggregation for cleaner correlation
        hourly = common.resample("h").mean()

        corrs = hourly[["price_eur_mwh"] + corr_cols].corr()["price_eur_mwh"].drop("price_eur_mwh")

        fig_corr = go.Figure(go.Bar(
            x=[corr_labels.get(c, c) for c in corrs.index],
            y=corrs.values,
            marker_color=[COLORS["red"] if v > 0 else COLORS["blue"] for v in corrs.values],
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        ))
        fig_corr.update_layout(
            yaxis_title="Corrélation avec prix spot",
            height=350, yaxis_range=[-1, 1],
        )
        fig_corr.add_hline(y=0, line_color=COLORS["muted"], line_width=1)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Scatter: price vs load
        st.subheader("Prix vs Charge")
        sample = hourly.sample(min(2000, len(hourly)), random_state=42)
        fig_sc = go.Figure(go.Scattergl(
            x=sample["load_mw"], y=sample["price_eur_mwh"],
            mode="markers",
            marker=dict(size=3, color=COLORS["blue"], opacity=0.3),
            hovertemplate="Charge: %{x:.0f} MW<br>Prix: %{y:.1f} EUR/MWh<extra></extra>",
        ))
        fig_sc.update_layout(
            xaxis_title="Charge (MW)", yaxis_title="Prix spot (EUR/MWh)",
            height=400,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # Scatter: price vs solar
        if "solar_mw" in hourly.columns:
            st.subheader("Prix vs Production solaire")
            fig_sol = go.Figure(go.Scattergl(
                x=sample["solar_mw"], y=sample["price_eur_mwh"],
                mode="markers",
                marker=dict(size=3, color=COLORS["amber"], opacity=0.3),
                hovertemplate="Solaire: %{x:.0f} MW<br>Prix: %{y:.1f} EUR/MWh<extra></extra>",
            ))
            fig_sol.update_layout(
                xaxis_title="Production solaire (MW)", yaxis_title="Prix spot (EUR/MWh)",
                height=400,
            )
            st.plotly_chart(fig_sol, use_container_width=True)

# ── Export ────────────────────────────────────────────────────────────────
with st.expander("Export"):
    if has_hydro:
        export_csv_button(hydro, "hydro_reservoirs.csv", "Export hydro")
    if has_entso:
        export_csv_button(entso.iloc[-entso_n:], "entso_fondamentaux.csv", "Export ENTSO-E")


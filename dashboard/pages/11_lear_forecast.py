"""
Page 11 — LEAR Short-Term Forecast
"Prevision court terme D+1..D+10 — LASSO state of the art"
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils import (
    COLORS, add_range_slider, export_csv_button, load_epex,
    load_lear_forecast, load_pfc, show_freshness_sidebar,
)

st.header("Prevision Court Terme (LEAR)")
st.caption("LASSO Estimated AutoRegressive — D+1 a D+10")

show_freshness_sidebar()

# ── Load data ─────────────────────────────────────────────────────────────
lear = load_lear_forecast()
epex = load_epex()
pfc = load_pfc()

if lear is None:
    st.warning("Aucun forecast LEAR disponible. Lancez `python run_pfc_production.py`.")
    st.stop()

TZ = "Europe/Zurich"

# ── Prepare LEAR data ────────────────────────────────────────────────────
lear_ts = lear.set_index("timestamp").sort_index()
lear_local = lear_ts.index.tz_convert(TZ)

forecast_start = lear_ts.index.min()
forecast_end = lear_ts.index.max()

# ── Prepare EPEX (actuals if available for backtest) ─────────────────────
has_actuals = False
if epex is not None and not epex.empty:
    epex_h = epex["price_eur_mwh"].resample("h").mean()
    # Check overlap
    common = epex_h.index.intersection(lear_ts.index)
    if len(common) > 0:
        has_actuals = True
        actuals = epex_h.loc[common]

# ── KPI Row ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Horizon", f"D+1..D+10")
    st.caption(f"{forecast_start.strftime('%d/%m')} → {forecast_end.strftime('%d/%m/%Y')}")
with k2:
    mean_p = float(lear_ts["price_lear"].mean())
    st.metric("Prix moyen", f"{mean_p:.1f}")
    st.caption("EUR/MWh")
with k3:
    peak_mask = (lear_local.hour >= 8) & (lear_local.hour < 20) & (lear_local.weekday < 5)
    if peak_mask.any():
        peak_avg = float(lear_ts.loc[peak_mask, "price_lear"].mean())
        offpeak_avg = float(lear_ts.loc[~peak_mask, "price_lear"].mean())
        st.metric("Peak", f"{peak_avg:.1f}")
        st.caption(f"Offpeak: {offpeak_avg:.1f}")
    else:
        st.metric("Peak", "—")
with k4:
    spread = float(lear_ts["price_lear"].max() - lear_ts["price_lear"].min())
    st.metric("Spread", f"{spread:.0f}")
    st.caption("Max - Min EUR/MWh")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Forecast 10 jours", "Profil journalier", "Backtest", "Metriques",
])

# ════════════════════════════════════════════════════════════════════════
# TAB 1: 10-day forecast chart
# ════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Prevision horaire D+1 a D+10")

    fig = go.Figure()

    # Historical EPEX (last 7 days for context)
    if epex is not None and not epex.empty:
        epex_h_all = epex["price_eur_mwh"].resample("h").mean()
        lookback = forecast_start - pd.Timedelta(days=7)
        hist = epex_h_all.loc[lookback:forecast_start]
        if not hist.empty:
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist.values,
                name="EPEX Spot (historique)",
                line=dict(color=COLORS["blue"], width=1.5),
                hovertemplate="%{y:.1f} EUR/MWh<extra>Spot</extra>",
            ))

        # Actuals overlapping forecast period
        if has_actuals and len(actuals) > 0:
            fig.add_trace(go.Scatter(
                x=actuals.index, y=actuals.values,
                name="EPEX Spot (actuel)",
                line=dict(color=COLORS["blue"], width=2.5),
                hovertemplate="%{y:.1f} EUR/MWh<extra>Spot actuel</extra>",
            ))

    # LEAR forecast
    fig.add_trace(go.Scatter(
        x=lear_ts.index, y=lear_ts["price_lear"],
        name="LEAR Forecast",
        line=dict(color=COLORS["amber"], width=2.5),
        hovertemplate="%{y:.1f} EUR/MWh<extra>LEAR</extra>",
    ))

    # Confidence bands
    if "price_p10" in lear_ts.columns and "price_p90" in lear_ts.columns:
        fig.add_trace(go.Scatter(
            x=lear_ts.index, y=lear_ts["price_p90"],
            name="p90", line=dict(width=0), showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=lear_ts.index, y=lear_ts["price_p10"],
            name="IC 80%",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(192,138,43,0.12)",
            hoverinfo="skip",
        ))

    # PFC structural (for comparison)
    if pfc is not None and not pfc.empty and "price_shape" in pfc.columns:
        pfc_overlap = pfc.loc[
            (pfc.index >= forecast_start) & (pfc.index <= forecast_end),
            "price_shape",
        ]
        if not pfc_overlap.empty:
            pfc_h = pfc_overlap.resample("h").mean()
            fig.add_trace(go.Scatter(
                x=pfc_h.index, y=pfc_h.values,
                name="PFC structurel",
                line=dict(color=COLORS["muted"], width=1, dash="dot"),
                hovertemplate="%{y:.1f} EUR/MWh<extra>PFC</extra>",
            ))

    # Vertical line: now
    fig.add_shape(
        type="line",
        x0=forecast_start, x1=forecast_start,
        y0=0, y1=1, yref="paper",
        line=dict(color=COLORS["muted"], width=1, dash="dash"),
    )

    fig.update_layout(
        yaxis_title="EUR/MWh",
        height=500,
        legend=dict(orientation="h", y=1.05, x=0),
        hovermode="x unified",
    )
    fig = add_range_slider(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Daily summary table
    st.subheader("Resume journalier")
    daily_summary = lear.groupby("date").agg(
        Moyenne=("price_lear", "mean"),
        Min=("price_lear", "min"),
        Max=("price_lear", "max"),
    ).round(1)
    daily_summary.index.name = "Date"

    # Add day name
    daily_summary["Jour"] = pd.to_datetime(daily_summary.index).strftime("%A")
    day_map = {
        "Monday": "Lun", "Tuesday": "Mar", "Wednesday": "Mer",
        "Thursday": "Jeu", "Friday": "Ven", "Saturday": "Sam", "Sunday": "Dim",
    }
    daily_summary["Jour"] = daily_summary["Jour"].map(day_map)
    daily_summary = daily_summary[["Jour", "Moyenne", "Min", "Max"]]
    st.dataframe(daily_summary, use_container_width=True)

    export_csv_button(lear, "lear_forecast.csv", "Export forecast")


# ════════════════════════════════════════════════════════════════════════
# TAB 2: Hourly profile comparison
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Profil horaire moyen")

    # LEAR average profile
    lear_profile = lear.groupby("hour")["price_lear"].mean()

    fig_prof = go.Figure()
    fig_prof.add_trace(go.Bar(
        x=lear_profile.index, y=lear_profile.values,
        name="LEAR moyen",
        marker_color=COLORS["amber"],
        opacity=0.7,
        hovertemplate="H%{x:02d}: %{y:.1f} EUR/MWh<extra>LEAR</extra>",
    ))

    # Historical EPEX average profile (last 30 days)
    if epex is not None and not epex.empty:
        epex_h_all = epex["price_eur_mwh"].resample("h").mean()
        last_30d = epex_h_all.last("30D")
        if not last_30d.empty:
            epex_local = last_30d.index.tz_convert(TZ)
            epex_profile = last_30d.groupby(epex_local.hour).mean()
            fig_prof.add_trace(go.Scatter(
                x=epex_profile.index, y=epex_profile.values,
                name="EPEX 30j (ref)",
                line=dict(color=COLORS["blue"], width=2.5),
                mode="lines+markers",
                hovertemplate="H%{x:02d}: %{y:.1f} EUR/MWh<extra>EPEX 30j</extra>",
            ))

    fig_prof.update_layout(
        xaxis_title="Heure (CET/CEST)",
        yaxis_title="EUR/MWh",
        height=400,
        xaxis=dict(dtick=1),
        legend=dict(orientation="h", y=1.05, x=0),
    )
    st.plotly_chart(fig_prof, use_container_width=True)

    # Per-day profiles
    st.subheader("Profil par jour")
    dates = sorted(lear["date"].unique())

    # Heatmap: hour × day
    pivot = lear.pivot_table(index="hour", columns="date", values="price_lear")
    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(d) for d in pivot.columns],
        y=[f"H{h:02d}" for h in pivot.index],
        colorscale="YlOrRd",
        hovertemplate="Jour %{x}<br>%{y}: %{z:.1f} EUR/MWh<extra></extra>",
        colorbar=dict(title="EUR/MWh"),
    ))
    fig_heat.update_layout(height=450, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_heat, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 3: Backtest (if actuals available)
# ════════════════════════════════════════════════════════════════════════
with tab3:
    if not has_actuals or len(actuals) < 2:
        st.info(
            "Backtest disponible apres realisation des prix spot.\n\n"
            "Les heures deja realisees seront comparees automatiquement."
        )

        # Show PFC vs LEAR comparison instead
        if pfc is not None and not pfc.empty and "price_shape" in pfc.columns:
            st.subheader("LEAR vs PFC structurel")

            pfc_h = pfc.loc[
                (pfc.index >= forecast_start) & (pfc.index <= forecast_end),
                "price_shape",
            ].resample("h").mean()

            common_pfc = pfc_h.index.intersection(lear_ts.index)
            if len(common_pfc) > 10:
                comp = pd.DataFrame({
                    "LEAR": lear_ts.loc[common_pfc, "price_lear"],
                    "PFC": pfc_h.loc[common_pfc],
                })
                comp["Ecart"] = comp["LEAR"] - comp["PFC"]

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Ecart moyen", f"{comp['Ecart'].mean():+.1f}")
                    st.caption("LEAR - PFC (EUR/MWh)")
                with c2:
                    st.metric("Correl.", f"{comp['LEAR'].corr(comp['PFC']):.3f}")
                with c3:
                    rmse = np.sqrt((comp["Ecart"] ** 2).mean())
                    st.metric("RMSE", f"{rmse:.1f}")
                    st.caption("EUR/MWh")

                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    x=comp.index, y=comp["LEAR"],
                    name="LEAR", line=dict(color=COLORS["amber"], width=2),
                ))
                fig_comp.add_trace(go.Scatter(
                    x=comp.index, y=comp["PFC"],
                    name="PFC", line=dict(color=COLORS["muted"], width=1.5, dash="dot"),
                ))
                fig_comp.update_layout(
                    yaxis_title="EUR/MWh", height=350,
                    legend=dict(orientation="h", y=1.05, x=0),
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                # Scatter LEAR vs PFC
                fig_scat = go.Figure()
                fig_scat.add_trace(go.Scatter(
                    x=comp["PFC"], y=comp["LEAR"],
                    mode="markers",
                    marker=dict(size=5, color=COLORS["amber"], opacity=0.5),
                    showlegend=False,
                    hovertemplate="PFC: %{x:.1f}<br>LEAR: %{y:.1f}<extra></extra>",
                ))
                # 45-degree line
                mn = min(comp["PFC"].min(), comp["LEAR"].min())
                mx = max(comp["PFC"].max(), comp["LEAR"].max())
                fig_scat.add_trace(go.Scatter(
                    x=[mn, mx], y=[mn, mx],
                    mode="lines", line=dict(color=COLORS["muted"], dash="dash"),
                    showlegend=False,
                ))
                fig_scat.update_layout(
                    xaxis_title="PFC structurel (EUR/MWh)",
                    yaxis_title="LEAR (EUR/MWh)",
                    height=350,
                    title="LEAR vs PFC — scatter",
                )
                st.plotly_chart(fig_scat, use_container_width=True)
    else:
        st.subheader("Backtest LEAR vs Spot realise")

        comp = pd.DataFrame({
            "Forecast": lear_ts.loc[common, "price_lear"],
            "Actuel": actuals,
        }).dropna()

        if len(comp) < 2:
            st.warning("Pas assez de donnees pour le backtest")
        else:
            comp["Erreur"] = comp["Forecast"] - comp["Actuel"]
            comp["Erreur_abs"] = comp["Erreur"].abs()
            comp["APE"] = (comp["Erreur_abs"] / comp["Actuel"].abs().clip(lower=1)) * 100

            # KPIs
            b1, b2, b3, b4, b5 = st.columns(5)
            mae = comp["Erreur_abs"].mean()
            rmse = np.sqrt((comp["Erreur"] ** 2).mean())
            mape = comp["APE"].mean()
            bias = comp["Erreur"].mean()
            corr = comp["Forecast"].corr(comp["Actuel"])

            with b1:
                st.metric("MAE", f"{mae:.1f}")
                st.caption("EUR/MWh")
            with b2:
                st.metric("RMSE", f"{rmse:.1f}")
                st.caption("EUR/MWh")
            with b3:
                st.metric("MAPE", f"{mape:.1f}%")
            with b4:
                st.metric("Biais", f"{bias:+.1f}")
                st.caption("EUR/MWh")
            with b5:
                st.metric("Correl.", f"{corr:.3f}")

            st.divider()

            # Time series comparison
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=comp.index, y=comp["Actuel"],
                name="Spot realise",
                line=dict(color=COLORS["blue"], width=2),
            ))
            fig_bt.add_trace(go.Scatter(
                x=comp.index, y=comp["Forecast"],
                name="LEAR Forecast",
                line=dict(color=COLORS["amber"], width=2),
            ))
            fig_bt.update_layout(
                yaxis_title="EUR/MWh", height=400,
                legend=dict(orientation="h", y=1.05, x=0),
                hovermode="x unified",
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            # Error distribution
            fig_err = go.Figure(go.Histogram(
                x=comp["Erreur"].values,
                nbinsx=40,
                marker_color=COLORS["amber"],
                opacity=0.7,
                hovertemplate="%{x:.1f} EUR/MWh<br>%{y} obs<extra></extra>",
            ))
            fig_err.add_vline(x=0, line_color=COLORS["muted"], line_dash="dash")
            fig_err.update_layout(
                xaxis_title="Erreur (EUR/MWh)",
                yaxis_title="Frequence",
                height=250,
                title="Distribution des erreurs",
            )
            st.plotly_chart(fig_err, use_container_width=True)

            # Hourly MAE
            st.subheader("MAE par heure de livraison")
            comp_local = comp.copy()
            comp_local["hour"] = comp_local.index.tz_convert(TZ).hour
            hourly_mae = comp_local.groupby("hour")["Erreur_abs"].mean()

            fig_hmae = go.Figure(go.Bar(
                x=hourly_mae.index, y=hourly_mae.values,
                marker_color=[
                    COLORS["red"] if v > mae * 1.3 else COLORS["amber"]
                    for v in hourly_mae.values
                ],
                hovertemplate="H%{x:02d}: MAE=%{y:.1f} EUR/MWh<extra></extra>",
            ))
            fig_hmae.add_hline(y=mae, line_color=COLORS["muted"], line_dash="dot",
                               annotation_text=f"MAE global: {mae:.1f}")
            fig_hmae.update_layout(
                xaxis_title="Heure (CET/CEST)",
                yaxis_title="MAE (EUR/MWh)",
                height=300,
                xaxis=dict(dtick=1),
            )
            st.plotly_chart(fig_hmae, use_container_width=True)

            export_csv_button(comp, "lear_backtest.csv", "Export backtest")


# ════════════════════════════════════════════════════════════════════════
# TAB 4: Model details
# ════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Details du modele LEAR")

    st.markdown("""
**Architecture** : LASSO Estimated AutoRegressive (Ziel & Weron, 2018)

| Composant | Detail |
|-----------|--------|
| Modeles | 24 LASSO independants (1 par heure) |
| Features | Prix lagues (d-1/2/3/7), load, solaire, eolien, outages, TTF, CO2, hydro |
| Transformation | Asinh (variance-stabilizing) |
| Calibration | Moyenne sur 4 fenetres (28/56/84/728 jours) |
| Blend PFC | D1-7 = LEAR, D8-10 = transition, D11+ = PFC structurel |

**Pourquoi LEAR ?**
- Gagnant consistant des competitions EPF (Electricity Price Forecasting)
- LASSO selectionne automatiquement les features pertinentes (~30-50 sur 250+)
- Transformation asinh gere les prix negatifs et les spikes
- Moyenne multi-fenetres : robuste aux changements de regime de marche
""")

    # Feature importance (if we can compute it)
    st.subheader("Nombre de fenetres utilisees")
    if "n_windows" in lear.columns:
        win_stats = lear.groupby("hour")["n_windows"].mean()
        fig_win = go.Figure(go.Bar(
            x=win_stats.index, y=win_stats.values,
            marker_color=COLORS["blue"],
            hovertemplate="H%{x:02d}: %{y:.1f} fenetres<extra></extra>",
        ))
        fig_win.update_layout(
            xaxis_title="Heure", yaxis_title="Fenetres moyennes",
            height=250, xaxis=dict(dtick=1),
        )
        st.plotly_chart(fig_win, use_container_width=True)

    # Price stats per horizon day
    st.subheader("Statistiques par jour d'horizon")
    if "days_ahead" in lear.columns:
        day_stats = lear.groupby("days_ahead")["price_lear"].agg(
            ["mean", "std", "min", "max"]
        ).round(1)
        day_stats.columns = ["Moyenne", "Ecart-type", "Min", "Max"]
        day_stats.index.name = "D+"
        st.dataframe(day_stats, use_container_width=True)

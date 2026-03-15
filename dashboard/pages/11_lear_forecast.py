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
    load_lear_backtest, load_lear_forecast, load_pfc, show_freshness_sidebar,
)

st.header("Prevision Court Terme (LEAR)")
st.caption("Hybrid LEAR+MLP — D+1 a D+10 — Prix CH+DE cross-border")

show_freshness_sidebar()

# ── Load data ─────────────────────────────────────────────────────────────
lear = load_lear_forecast()
epex = load_epex()
pfc = load_pfc()
backtest = load_lear_backtest()

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
    st.plotly_chart(fig, width="stretch")

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
    st.plotly_chart(fig_prof, width="stretch")

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
    st.plotly_chart(fig_heat, width="stretch")


# ════════════════════════════════════════════════════════════════════════
# TAB 3: Backtest vs Spot (rolling out-of-sample)
# ════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Backtest LEAR vs Spot EPEX (out-of-sample)")
    st.caption("Test retroactif : que predisait le LEAR chaque jour passe ?")

    if backtest is None or backtest.empty:
        st.warning(
            "Aucun backtest disponible.\n\n"
            "Le backtest est genere automatiquement par `run_pfc_production.py` "
            "(30 jours, D+1 rolling)."
        )
    else:
        bt = backtest.copy()

        # KPIs
        mae = bt["abs_error"].mean()
        rmse = np.sqrt((bt["error"] ** 2).mean())
        mape = bt["ape"].mean()
        bias = bt["error"].mean()
        corr = bt["forecast"].corr(bt["actual"])

        b1, b2, b3, b4, b5 = st.columns(5)
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

        # Time series: forecast vs actual
        # Build hourly timestamps from date + hour
        bt["ts"] = pd.to_datetime(bt["date"]) + pd.to_timedelta(bt["hour"], unit="h")
        bt_sorted = bt.sort_values("ts")

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=bt_sorted["ts"], y=bt_sorted["actual"],
            name="Spot EPEX (realise)",
            line=dict(color=COLORS["blue"], width=2),
            hovertemplate="%{y:.1f} EUR/MWh<extra>Spot</extra>",
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt_sorted["ts"], y=bt_sorted["forecast"],
            name="LEAR (D+1 prevu)",
            line=dict(color=COLORS["amber"], width=2),
            hovertemplate="%{y:.1f} EUR/MWh<extra>LEAR</extra>",
        ))
        fig_bt.update_layout(
            yaxis_title="EUR/MWh", height=450,
            legend=dict(orientation="h", y=1.05, x=0),
            hovermode="x unified",
        )
        fig_bt = add_range_slider(fig_bt)
        st.plotly_chart(fig_bt, width="stretch")

        # Scatter: forecast vs actual
        col_scat, col_err = st.columns(2)

        with col_scat:
            st.subheader("Prevu vs Realise")
            fig_scat = go.Figure()
            fig_scat.add_trace(go.Scatter(
                x=bt["actual"], y=bt["forecast"],
                mode="markers",
                marker=dict(size=4, color=COLORS["amber"], opacity=0.4),
                showlegend=False,
                hovertemplate="Spot: %{x:.1f}<br>LEAR: %{y:.1f}<extra></extra>",
            ))
            mn = min(bt["actual"].min(), bt["forecast"].min())
            mx = max(bt["actual"].max(), bt["forecast"].max())
            fig_scat.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx],
                mode="lines", line=dict(color=COLORS["muted"], dash="dash"),
                showlegend=False,
            ))
            fig_scat.update_layout(
                xaxis_title="Spot realise (EUR/MWh)",
                yaxis_title="LEAR prevu (EUR/MWh)",
                height=350,
            )
            st.plotly_chart(fig_scat, width="stretch")

        with col_err:
            st.subheader("Distribution erreurs")
            fig_err = go.Figure(go.Histogram(
                x=bt["error"].values,
                nbinsx=50,
                marker_color=COLORS["amber"],
                opacity=0.7,
                hovertemplate="%{x:.1f} EUR/MWh<br>%{y} obs<extra></extra>",
            ))
            fig_err.add_shape(
                type="line", x0=0, x1=0, y0=0, y1=1, yref="paper",
                line=dict(color=COLORS["muted"], dash="dash"),
            )
            fig_err.update_layout(
                xaxis_title="Erreur (EUR/MWh)",
                yaxis_title="Frequence",
                height=350,
            )
            st.plotly_chart(fig_err, width="stretch")

        # MAE per hour
        st.subheader("MAE par heure de livraison")
        hourly_mae = bt.groupby("hour")["abs_error"].mean()

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
        st.plotly_chart(fig_hmae, width="stretch")

        # Daily MAE
        st.subheader("MAE par jour")
        daily_mae = bt.groupby("date")["abs_error"].mean()
        fig_dmae = go.Figure(go.Bar(
            x=daily_mae.index, y=daily_mae.values,
            marker_color=COLORS["blue"],
            hovertemplate="%{x}: MAE=%{y:.1f}<extra></extra>",
        ))
        fig_dmae.add_hline(y=mae, line_color=COLORS["muted"], line_dash="dot")
        fig_dmae.update_layout(
            yaxis_title="MAE (EUR/MWh)", height=250,
        )
        st.plotly_chart(fig_dmae, width="stretch")

        export_csv_button(bt, "lear_backtest.csv", "Export backtest")


# ════════════════════════════════════════════════════════════════════════
# TAB 4: Model details
# ════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Details du modele LEAR")

    st.markdown("""
**Architecture** : Hybrid LEAR+MLP (Ziel & Weron 2018, El Mahtout & Ziel 2026)

| Composant | Detail |
|-----------|--------|
| Modeles lineaires | 24 LASSO independants (1 par heure) |
| Modeles non-lineaires | 24 MLP (128-64 neurones) en ensemble |
| Features CH | Prix lagues (d-1/2/3/7), load, solaire, eolien, outages, TTF, CO2, hydro |
| Features DE | Prix DE cross-border (d-1/2/7), spread CH-DE |
| Transformation | Asinh (variance-stabilizing) + StandardScaler |
| Calibration | Moyenne sur 4 fenetres (42/56/84/365 jours) |
| Ensemble | 60% LASSO + 40% MLP |
| Intervalles | Prediction conforme (IC calibres sur residus OOS) |
| Blend PFC | D1-7 = LEAR, D8-10 = transition, D11+ = PFC structurel |

**Innovations** (El Mahtout & Ziel, 2026)
- Prix DE cross-border : le couplage CH-DE explique ~22% de la variance
- Ensemble hybrid LASSO+MLP : capture les interactions non-lineaires
- Prediction conforme : intervalles calibres (couverture garantie)
- StandardScaler : normalisation pour une meilleure convergence LASSO
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
        st.plotly_chart(fig_win, width="stretch")

    # Price stats per horizon day
    st.subheader("Statistiques par jour d'horizon")
    if "days_ahead" in lear.columns:
        day_stats = lear.groupby("days_ahead")["price_lear"].agg(
            ["mean", "std", "min", "max"]
        ).round(1)
        day_stats.columns = ["Moyenne", "Ecart-type", "Min", "Max"]
        day_stats.index.name = "D+"
        st.dataframe(day_stats, use_container_width=True)

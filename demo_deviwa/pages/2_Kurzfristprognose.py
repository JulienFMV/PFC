"""Seite 1 – Kurzfristprognose (LEAR 10 Tage)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    FMV_ACCENT,
    FMV_BLUE,
    FMV_GREEN,
    FMV_GREY,
    FMV_NAVY,
    FMV_RED,
    fmt_eur_mwh,
    freshness_badge,
    kpi_card,
    load_epex_ch,
    load_lear_backtest,
    load_lear_forecast,
    render_header,
)

st.set_page_config(
    page_title="FMV · Kurzfristprognose",
    page_icon="📈",
    layout="wide",
)

render_header("Kurzfristprognose · LEAR-Modell FMV")

forecast = load_lear_forecast()
backtest = load_lear_backtest()
spot = load_epex_ch()

if forecast.empty:
    st.error("Keine Prognose verfügbar. Bitte LEAR-Lauf ausführen.")
    st.stop()

# ---------------------------------------------------------------------------
# Accuracy KPIs (letzte 30 Tage)
# ---------------------------------------------------------------------------
mae_30d = np.nan
rmse_30d = np.nan
n_obs_30d = 0
if not backtest.empty:
    recent = backtest[backtest["forecast_ts"] >= backtest["forecast_ts"].max() - pd.Timedelta(days=30)]
    recent = recent[recent.get("horizon", 1) == 1]
    if not recent.empty:
        err = (recent["forecast"] - recent["actual"]).dropna()
        mae_30d = float(err.abs().mean())
        rmse_30d = float(np.sqrt((err ** 2).mean()))
        n_obs_30d = int(len(err))

# Forecast-KPIs
avg_d1 = float(forecast[forecast["days_ahead"] == 1]["price_lear"].mean()) if not forecast.empty else np.nan
avg_d10 = float(forecast[forecast["days_ahead"] <= 10]["price_lear"].mean()) if not forecast.empty else np.nan
max_10d = float(forecast[forecast["days_ahead"] <= 10]["price_lear"].max()) if not forecast.empty else np.nan
min_10d = float(forecast[forecast["days_ahead"] <= 10]["price_lear"].min()) if not forecast.empty else np.nan

c1, c2, c3, c4 = st.columns(4)
c1.markdown(
    kpi_card("Prognose D+1 (Ø)", fmt_eur_mwh(avg_d1),
             delta=f"Horizont: {forecast['days_ahead'].min():.0f}–{forecast['days_ahead'].max():.0f} Tage",
             delta_color=FMV_BLUE),
    unsafe_allow_html=True,
)
c2.markdown(
    kpi_card("Prognose 10 Tage (Ø)", fmt_eur_mwh(avg_d10),
             delta=f"Spitze: {fmt_eur_mwh(max_10d)} · Tief: {fmt_eur_mwh(min_10d)}",
             delta_color=FMV_NAVY),
    unsafe_allow_html=True,
)
c3.markdown(
    kpi_card(
        "MAE letzte 30 Tage",
        f"{mae_30d:.2f} EUR/MWh" if np.isfinite(mae_30d) else "—",
        delta=f"RMSE: {rmse_30d:.2f} · n={n_obs_30d}" if np.isfinite(rmse_30d) else None,
        delta_color=FMV_GREEN,
        help_text="Mittlerer absoluter Fehler der D+1-Prognose",
    ),
    unsafe_allow_html=True,
)
c4.markdown(
    kpi_card("Modell-Stack",
             "LEAR + GBM + Chronos-2",
             delta="Walk-Forward, tägliches Neulernen",
             delta_color=FMV_GREY),
    unsafe_allow_html=True,
)

st.markdown("")

# ---------------------------------------------------------------------------
# Haupt-Chart: 10-Tage-Prognose mit Unsicherheitsband
# ---------------------------------------------------------------------------
st.subheader("10-Tage-Prognose mit P10–P90-Band")

fig = go.Figure()

# Realisierter Spot der letzten 7 Tage als Referenz
if not spot.empty:
    hist_cutoff = spot.index.max() - pd.Timedelta(days=7)
    spot_recent = spot.loc[spot.index >= hist_cutoff, "price"].resample("h").mean()
    fig.add_trace(go.Scatter(
        x=spot_recent.index, y=spot_recent.values,
        mode="lines", name="Spot realisiert",
        line=dict(color=FMV_NAVY, width=1.5),
    ))

fcst = forecast.sort_values("timestamp").copy()
fig.add_trace(go.Scatter(
    x=fcst["timestamp"], y=fcst["price_p90"],
    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=fcst["timestamp"], y=fcst["price_p10"],
    mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(15,82,204,0.18)", name="P10–P90 Band",
))
fig.add_trace(go.Scatter(
    x=fcst["timestamp"], y=fcst["price_lear"],
    mode="lines", name="LEAR Prognose",
    line=dict(color=FMV_BLUE, width=3),
))

# Foundation-Model Roh-Output als Vergleich
if "fm_raw_price" in fcst.columns and fcst["fm_raw_price"].notna().any():
    fig.add_trace(go.Scatter(
        x=fcst["timestamp"], y=fcst["fm_raw_price"],
        mode="lines", name="Chronos-2 (Rohwert)",
        line=dict(color=FMV_ACCENT, width=1.4, dash="dot"),
    ))

fig.update_layout(
    height=480, margin=dict(l=20, r=20, t=10, b=20),
    hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(title="EUR/MWh", gridcolor="#EEF1F6"),
    xaxis=dict(gridcolor="#EEF1F6"),
    legend=dict(orientation="h", y=-0.12),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tabelle: Stundenprognose morgen (D+1)
# ---------------------------------------------------------------------------
st.subheader("Stündliche Prognose für morgen (D+1)")
d1 = forecast[forecast["days_ahead"] == 1].copy()
if d1.empty:
    st.info("Keine D+1-Daten verfügbar.")
else:
    d1 = d1.sort_values("timestamp")
    table = pd.DataFrame({
        "Stunde": d1["timestamp"].dt.strftime("%H:00"),
        "Prognose EUR/MWh": d1["price_lear"].round(2),
        "P10": d1["price_p10"].round(2),
        "P90": d1["price_p90"].round(2),
        "Unsicherheit (P90–P10)": (d1["price_p90"] - d1["price_p10"]).round(2),
    })
    st.dataframe(
        table, use_container_width=True, hide_index=True,
        column_config={
            "Unsicherheit (P90–P10)": st.column_config.ProgressColumn(
                "Unsicherheit (P90–P10)",
                format="%.2f EUR/MWh",
                min_value=0.0,
                max_value=float(table["Unsicherheit (P90–P10)"].max() or 1.0),
            ),
        },
    )

# ---------------------------------------------------------------------------
# Accuracy-Detail: letzte 30 Tage
# ---------------------------------------------------------------------------
st.subheader("Modell-Genauigkeit · letzte 30 Tage (D+1)")
if backtest.empty:
    st.info("Backtest-Daten nicht verfügbar.")
else:
    recent = backtest[backtest["forecast_ts"] >= backtest["forecast_ts"].max() - pd.Timedelta(days=30)].copy()
    recent = recent[recent.get("horizon", 1) == 1].sort_values("forecast_ts")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=recent["forecast_ts"], y=recent["actual"],
        mode="lines", name="Realisiert",
        line=dict(color=FMV_NAVY, width=2),
    ))
    fig2.add_trace(go.Scatter(
        x=recent["forecast_ts"], y=recent["forecast"],
        mode="lines", name="LEAR Prognose",
        line=dict(color=FMV_BLUE, width=2, dash="dash"),
    ))
    fig2.update_layout(
        height=360, margin=dict(l=20, r=20, t=10, b=20),
        hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="EUR/MWh", gridcolor="#EEF1F6"),
        xaxis=dict(gridcolor="#EEF1F6"),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
cols = st.columns(2)
cols[0].markdown(freshness_badge(forecast.set_index("timestamp"), "Prognose"),
                 unsafe_allow_html=True)
if not backtest.empty:
    cols[1].markdown(freshness_badge(backtest.set_index("forecast_ts"), "Backtest"),
                     unsafe_allow_html=True)

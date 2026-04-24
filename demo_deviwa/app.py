"""FMV Deviwa Demo – Startseite: Marktüberblick."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    FMV_ACCENT,
    FMV_BLUE,
    FMV_GREEN,
    FMV_GREY,
    FMV_NAVY,
    FMV_RED,
    fmt_delta,
    fmt_eur_mwh,
    freshness_badge,
    kpi_card,
    load_commodities,
    load_eex_forwards,
    load_epex_ch,
    load_lear_forecast,
    render_header,
)

st.set_page_config(
    page_title="FMV Deviwa Demo · Marktüberblick",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_header("Marktüberblick heute")

with st.sidebar:
    st.markdown(f"<div style='color:{FMV_NAVY}; font-weight:600; font-size:1rem;'>"
                "FMV · Deviwa Energiepool</div>", unsafe_allow_html=True)
    st.caption("Interne Demo. Daten lokal geladen.")
    st.divider()
    st.markdown("**Inhalt**")
    st.markdown(
        "- Marktüberblick\n"
        "- Kurzfristprognose (LEAR)\n"
        "- Lastprofil-Pricing\n"
        "- Ihre Transaktionen\n"
        "- Marktdaten CH/DE"
    )

# ---------------------------------------------------------------------------
# Daten laden
# ---------------------------------------------------------------------------
epex = load_epex_ch()
forecast = load_lear_forecast()
forwards = load_eex_forwards()
commodities = load_commodities()

if epex.empty:
    st.error("Spot-Daten nicht verfügbar. Bitte Datencache aktualisieren.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI-Zeile
# ---------------------------------------------------------------------------
last_hour_price = float(epex["price"].dropna().iloc[-1])
last_ts = epex.dropna().index[-1]
prev_day_price = float(epex["price"].dropna().iloc[-25]) if len(epex) > 25 else np.nan
delta_vs_yesterday = last_hour_price - prev_day_price

# Forecast D+1 (Durchschnitt 1. Tag)
forecast_d1 = np.nan
if not forecast.empty and "days_ahead" in forecast.columns:
    f_d1 = forecast[forecast["days_ahead"] == 1]
    if not f_d1.empty:
        forecast_d1 = float(f_d1["price_lear"].mean())

# CH Base Cal-27
cal27_price = np.nan
cal27_delta = np.nan
if not forwards.empty:
    ch_base = forwards[
        (forwards["market"].str.upper() == "CH")
        & (forwards["load_type"].str.lower() == "base")
        & (forwards["product"].astype(str).str.contains("2027"))
    ].copy()
    if not ch_base.empty:
        ch_base = ch_base.sort_values("date")
        cal27_price = float(ch_base["price"].iloc[-1])
        recent = ch_base.tail(8)
        if len(recent) >= 2:
            cal27_delta = cal27_price - float(recent["price"].iloc[0])

# Gas / CO2
ttf_price = np.nan
ttf_delta = np.nan
eua_price = np.nan
eua_delta = np.nan
if not commodities.empty:
    if "TTF Gas" in commodities.columns:
        s = commodities["TTF Gas"].dropna()
        if len(s) >= 8:
            ttf_price = float(s.iloc[-1])
            ttf_delta = ttf_price - float(s.iloc[-8])
    if "CO2 EUA (KRBN)" in commodities.columns:
        s = commodities["CO2 EUA (KRBN)"].dropna()
        if len(s) >= 8:
            eua_price = float(s.iloc[-1])
            eua_delta = eua_price - float(s.iloc[-8])

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(
    kpi_card(
        "CH Spot letzte Stunde",
        fmt_eur_mwh(last_hour_price),
        delta=f"{fmt_delta(delta_vs_yesterday, ' EUR/MWh')} ggü. Vortag",
        delta_color=FMV_GREEN if delta_vs_yesterday < 0 else FMV_RED,
        help_text=str(last_ts),
    ),
    unsafe_allow_html=True,
)
c2.markdown(
    kpi_card(
        "Prognose D+1 (Ø)",
        fmt_eur_mwh(forecast_d1) if np.isfinite(forecast_d1) else "—",
        delta="LEAR-Modell FMV",
        delta_color=FMV_BLUE,
    ),
    unsafe_allow_html=True,
)
c3.markdown(
    kpi_card(
        "EEX CH Base Cal-27",
        fmt_eur_mwh(cal27_price) if np.isfinite(cal27_price) else "—",
        delta=f"{fmt_delta(cal27_delta, ' EUR/MWh')} 7-Tage" if np.isfinite(cal27_delta) else None,
        delta_color=FMV_GREEN if (cal27_delta or 0) < 0 else FMV_RED,
    ),
    unsafe_allow_html=True,
)
c4.markdown(
    kpi_card(
        "TTF Gas (Front)",
        f"{fmt_eur_mwh(ttf_price)}" if np.isfinite(ttf_price) else "—",
        delta=f"{fmt_delta(ttf_delta, ' EUR/MWh')} 7-Tage" if np.isfinite(ttf_delta) else None,
        delta_color=FMV_GREEN if (ttf_delta or 0) < 0 else FMV_RED,
    ),
    unsafe_allow_html=True,
)
c5.markdown(
    kpi_card(
        "CO₂ EUA",
        f"{fmt_eur_mwh(eua_price).replace('EUR/MWh','EUR/t')}" if np.isfinite(eua_price) else "—",
        delta=f"{fmt_delta(eua_delta, ' EUR/t')} 7-Tage" if np.isfinite(eua_delta) else None,
        delta_color=FMV_GREEN if (eua_delta or 0) < 0 else FMV_RED,
    ),
    unsafe_allow_html=True,
)

st.markdown("")

# ---------------------------------------------------------------------------
# Chart: Spot 14 Tage + Forecast 10 Tage
# ---------------------------------------------------------------------------
st.subheader("Spotpreis CH · letzte 14 Tage + Prognose 10 Tage")

hist_cutoff = epex.index.max() - pd.Timedelta(days=14)
spot_recent = epex.loc[epex.index >= hist_cutoff, "price"].resample("h").mean()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=spot_recent.index, y=spot_recent.values,
    mode="lines", name="Spot realisiert",
    line=dict(color=FMV_NAVY, width=2),
))

if not forecast.empty:
    fcst = forecast.copy()
    fcst = fcst.sort_values("timestamp")
    fig.add_trace(go.Scatter(
        x=fcst["timestamp"], y=fcst["price_p90"],
        mode="lines", line=dict(width=0), showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fcst["timestamp"], y=fcst["price_p10"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(15,82,204,0.15)",
        name="P10–P90 Band",
    ))
    fig.add_trace(go.Scatter(
        x=fcst["timestamp"], y=fcst["price_lear"],
        mode="lines", name="LEAR Prognose",
        line=dict(color=FMV_BLUE, width=2.5, dash="solid"),
    ))

fig.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=10, b=20),
    hovermode="x unified",
    plot_bgcolor="white",
    paper_bgcolor="white",
    yaxis=dict(title="EUR/MWh", gridcolor="#EEF1F6"),
    xaxis=dict(gridcolor="#EEF1F6"),
    legend=dict(orientation="h", y=-0.15),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Zweite Reihe: Forward Kurve + CO2/Gas
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.subheader("EEX Forward Kurve · CH Base")
    if not forwards.empty:
        last_date = forwards["date"].max()
        snap = forwards[
            (forwards["date"] == last_date)
            & (forwards["market"].str.upper() == "CH")
            & (forwards["load_type"].str.lower() == "base")
        ].copy()
        if not snap.empty:
            snap = snap.sort_values("product")
            fig_fwd = go.Figure()
            fig_fwd.add_trace(go.Bar(
                x=snap["product"], y=snap["price"],
                marker_color=FMV_BLUE,
                name=f"Schlusskurs {pd.Timestamp(last_date).strftime('%d.%m.%Y')}",
            ))
            fig_fwd.update_layout(
                height=320, margin=dict(l=20, r=20, t=10, b=20),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="EUR/MWh", gridcolor="#EEF1F6"),
                xaxis=dict(tickangle=-45),
            )
            st.plotly_chart(fig_fwd, use_container_width=True)
        else:
            st.info("Keine EEX CH Base Daten verfügbar.")
    else:
        st.info("Forward-Daten nicht geladen.")

with col_right:
    st.subheader("Commodities · letzte 90 Tage")
    if not commodities.empty:
        recent_com = commodities.tail(90)
        fig_com = go.Figure()
        for col, color in [("TTF Gas", FMV_BLUE),
                           ("CO2 EUA (KRBN)", FMV_ACCENT),
                           ("Brent", FMV_GREY)]:
            if col in recent_com.columns:
                series = recent_com[col].dropna()
                if not series.empty:
                    # Normalisieren auf 100 am ersten Punkt
                    normalized = series / series.iloc[0] * 100
                    fig_com.add_trace(go.Scatter(
                        x=normalized.index, y=normalized.values,
                        mode="lines", name=col,
                        line=dict(color=color, width=2),
                    ))
        fig_com.update_layout(
            height=320, margin=dict(l=20, r=20, t=10, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="Index (Start = 100)", gridcolor="#EEF1F6"),
            xaxis=dict(gridcolor="#EEF1F6"),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_com, use_container_width=True)
    else:
        st.info("Commodities nicht verfügbar.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
cols = st.columns(4)
cols[0].markdown(freshness_badge(epex, "Spot CH"), unsafe_allow_html=True)
cols[1].markdown(freshness_badge(forecast.set_index("timestamp") if not forecast.empty else forecast, "Prognose"),
                 unsafe_allow_html=True)
cols[2].markdown(
    freshness_badge(
        forwards.set_index("date") if not forwards.empty else forwards, "EEX Forwards"
    ),
    unsafe_allow_html=True,
)
cols[3].markdown(freshness_badge(commodities, "Commodities"), unsafe_allow_html=True)

st.caption(
    "© FMV – Forces Motrices Valaisannes · Demo-Version für den Deviwa-Energiepool. "
    "Alle Preise in EUR/MWh sofern nicht anders angegeben."
)

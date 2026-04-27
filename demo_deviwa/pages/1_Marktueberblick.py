"""Seite 1 — Marktüberblick.

Phase 5 redesign : style "FMV settlement price" screen — top filter bar,
Settlement Price chart with SMA 20 / 50 / 200 + Bollinger Bands, side panel
with High / Low / Range / Position-%, Range-Zone interpretation table.

The dashboard adds a portfolio-aware overlay : when the user selects an
actor, this page exposes that actor's volume-weighted average hedge price
for the chosen product as a horizontal line on the forward chart, so the
GRD can see *where they hedged* relative to the historical market.

Spot + LEAR forecast block stays underneath for the daily view ; commodity
ribbon (TTF Gas, CO₂ EUA, Brent) keeps the macro context.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio_yearly import build_qstrip_year_series, compute_hedge_by_year  # noqa: E402
from utils import (  # noqa: E402
    FMV_ACCENT,
    FMV_BLUE,
    FMV_GREEN,
    FMV_GREY,
    FMV_NAVY,
    FMV_RED,
    load_commodities,
    load_deviwa_deals_cached,
    load_deviwa_programme_cached,
    load_eex_forwards,
    load_epex_ch,
    load_lear_forecast,
    render_cache_status,
    render_header,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FMV Deviwa · Marktüberblick",
    page_icon="📊",
    layout="wide",
)

render_header("Marktüberblick — EEX Forwards & Spot")

with st.sidebar:
    render_cache_status()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

forwards = load_eex_forwards()
spot = load_epex_ch()
forecast = load_lear_forecast()
commodities = load_commodities()
deals = load_deviwa_deals_cached()
programme = load_deviwa_programme_cached()

if forwards.empty:
    st.error("Keine EEX-Forward-Daten verfügbar. Bitte Cache aktualisieren.")
    st.stop()


# ---------------------------------------------------------------------------
# Top filter bar (style screenshot FMV)
# ---------------------------------------------------------------------------

st.subheader("EEX Forward · Settlement Price")

c_market, c_profile, c_product, c_lookback, c_actor = st.columns([0.9, 0.9, 1.2, 1.0, 1.2])

with c_market:
    market_options = sorted(
        forwards["market"].dropna().astype(str).str.upper().unique().tolist()
    )
    market_default = "CH" if "CH" in market_options else market_options[0]
    market = st.selectbox(
        "Markt",
        options=market_options,
        index=market_options.index(market_default),
        key="mu_market",
    )

with c_profile:
    profile = st.selectbox(
        "Profil",
        options=["base", "peak"],
        format_func=lambda p: p.upper(),
        key="mu_profile",
    )

with c_product:
    candidates = forwards[
        (forwards["market"].astype(str).str.upper() == market)
        & (forwards["load_type"].astype(str).str.lower() == profile)
        & (forwards["product"].astype(str).str.contains("Y01_", na=False, regex=False))
    ].copy()

    def _year_from_product(p: str) -> int:
        digits = "".join(c for c in str(p) if c.isdigit())
        return int(digits[-4:]) if len(digits) >= 4 else 0

    products = sorted(
        candidates["product"].dropna().astype(str).unique().tolist(),
        key=_year_from_product,
    )

    # Inject a synthetic Q-Strip pseudo-product for the current year if its
    # Y01 contract has retired but quarterlies are still trading. Without
    # this fallback the current delivery year vanishes from the dropdown
    # — the user can't see the live price the budget projection is using.
    current_year = pd.Timestamp.now().year
    qstrip_label = f"Q-Strip {current_year} ({profile.upper()})"
    has_y01_current = any(_year_from_product(p) == current_year for p in products)
    qstrip_series = pd.DataFrame()
    if not has_y01_current:
        qstrip_series = build_qstrip_year_series(
            forwards, current_year, market=market, load_type=profile
        )
        if not qstrip_series.empty:
            products = [qstrip_label] + products

    if not products:
        st.warning(f"Keine Cal-Produkte für {market} {profile.upper()}.")
        st.stop()

    default_year = current_year + 1
    default_idx = next(
        (i for i, p in enumerate(products) if str(default_year) in p),
        len(products) - 1,
    )
    product = st.selectbox(
        "Produkt",
        options=products,
        index=default_idx,
        format_func=lambda p: (
            p if p.startswith("Q-Strip ")
            else p.replace("_BASE", "").replace("_PEAK", "").split(" ", 1)[-1]
        ),
        key="mu_product",
    )
    delivery_year = (
        current_year if product.startswith("Q-Strip ")
        else (_year_from_product(product) or default_year)
    )

with c_lookback:
    lookback = st.selectbox(
        "Lookback",
        options=["6M", "1J", "2J", "5J", "Alle"],
        index=2,
        key="mu_lookback",
    )

with c_actor:
    actors_in_data = sorted(deals["actor"].dropna().unique().tolist()) if not deals.empty else []
    actor_options = ["—"] + actors_in_data
    actor_choice = st.selectbox(
        "Akteur (Hedge-Overlay)",
        options=actor_options,
        index=0,
        key="mu_actor",
        help=(
            "Wenn ein Akteur gewählt ist, wird sein Ø Hedge-Preis "
            "als horizontale Linie eingeblendet."
        ),
    )
    selected_actor: str | None = None if actor_choice == "—" else actor_choice


# ---------------------------------------------------------------------------
# Filter forwards series + apply lookback
# ---------------------------------------------------------------------------

if product.startswith("Q-Strip ") and not qstrip_series.empty:
    ser = qstrip_series.copy()
    st.caption(
        f"ℹ️ **{product}** ist ein synthetisches Cal-{current_year}-Pendant : "
        f"tagesweise Volumen-gewichteter Mittelwert der verfügbaren "
        f"``Q0x_{current_year}_{profile.upper()}``-Settlements (gewichtet mit "
        f"Quartalsstunden). Wird verwendet, weil das ``Y01_{current_year}_"
        f"{profile.upper()}`` bereits in Lieferung ist und nicht mehr handelbar."
    )
else:
    ser = forwards[
        (forwards["market"].astype(str).str.upper() == market)
        & (forwards["load_type"].astype(str).str.lower() == profile)
        & (forwards["product"] == product)
    ].copy()
ser = ser.dropna(subset=["price", "date"]).sort_values("date").reset_index(drop=True)

if ser.empty:
    st.info("Keine Daten für die aktuelle Auswahl.")
    st.stop()

last_date = ser["date"].max()
lookback_days = {"6M": 183, "1J": 365, "2J": 730, "5J": 1825, "Alle": None}.get(lookback)
if lookback_days is not None:
    cutoff = last_date - pd.Timedelta(days=lookback_days)
    ser = ser[ser["date"] >= cutoff].reset_index(drop=True)

if len(ser) < 5:
    st.warning("Zu wenig Datenpunkte im gewählten Zeitraum.")
    st.stop()


# ---------------------------------------------------------------------------
# Technical overlays
# ---------------------------------------------------------------------------

ser["sma_20"] = ser["price"].rolling(window=20, min_periods=1).mean()
ser["sma_50"] = ser["price"].rolling(window=50, min_periods=1).mean()
ser["sma_200"] = ser["price"].rolling(window=200, min_periods=1).mean()
sma20_std = ser["price"].rolling(window=20, min_periods=1).std(ddof=0)
ser["bb_upper"] = ser["sma_20"] + 2.0 * sma20_std
ser["bb_lower"] = ser["sma_20"] - 2.0 * sma20_std


# ---------------------------------------------------------------------------
# Side-panel statistics
# ---------------------------------------------------------------------------

last_price = float(ser["price"].iloc[-1])
prev_price = float(ser["price"].iloc[-2]) if len(ser) > 1 else last_price
delta_1d = last_price - prev_price
delta_pct = 100.0 * delta_1d / prev_price if prev_price else 0.0

high_price = float(ser["price"].max())
low_price = float(ser["price"].min())
high_date = ser.loc[ser["price"].idxmax(), "date"]
low_date = ser.loc[ser["price"].idxmin(), "date"]
price_range = high_price - low_price
position_in_range = (
    (last_price - low_price) / price_range * 100.0 if price_range > 0 else 50.0
)

if position_in_range >= 80:
    range_zone = "Upper range — close to recent highs"
    zone_color = FMV_RED
elif position_in_range >= 60:
    range_zone = "Upper part of the range"
    zone_color = FMV_ACCENT
elif position_in_range >= 40:
    range_zone = "Mid-range — neutral market"
    zone_color = FMV_GREY
elif position_in_range >= 20:
    range_zone = "Lower part of the range"
    zone_color = FMV_GREEN
else:
    range_zone = "Lower range — potential buying opportunity"
    zone_color = FMV_GREEN


# ---------------------------------------------------------------------------
# Portfolio overlay : actor's avg hedge price for delivery_year
# ---------------------------------------------------------------------------

avg_hedge_price: float | None = None
hedged_volume_mwh: float = 0.0
if selected_actor is not None and not deals.empty:
    hm = compute_hedge_by_year(
        deals, programme, actor=selected_actor,
        today=pd.Timestamp.now(tz="Europe/Zurich"),
    )
    yhm = hm.get(delivery_year)
    if yhm is not None and np.isfinite(yhm.avg_hedge_price_eur_mwh):
        avg_hedge_price = float(yhm.avg_hedge_price_eur_mwh)
        hedged_volume_mwh = float(abs(yhm.hedged_mwh))


# ---------------------------------------------------------------------------
# Main chart + side panel layout
# ---------------------------------------------------------------------------

c_chart, c_side = st.columns([2.6, 1.0], gap="medium")

with c_chart:
    fig = go.Figure()

    # Bollinger band fill
    fig.add_trace(go.Scatter(
        x=ser["date"], y=ser["bb_upper"],
        mode="lines", name="Bollinger Upper",
        line=dict(width=0.6, dash="dot", color="#A0A8B5"),
        hovertemplate="%{x|%d.%m.%Y}<br>BB Upper : %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ser["date"], y=ser["bb_lower"],
        mode="lines", name="Bollinger Lower",
        line=dict(width=0.6, dash="dot", color="#A0A8B5"),
        fill="tonexty", fillcolor="rgba(160, 168, 181, 0.12)",
        hovertemplate="%{x|%d.%m.%Y}<br>BB Lower : %{y:.2f}<extra></extra>",
    ))

    # Moving averages (long → short)
    fig.add_trace(go.Scatter(
        x=ser["date"], y=ser["sma_200"],
        mode="lines", name="SMA 200",
        line=dict(width=1.4, color="#2B2F3A"),
        hovertemplate="%{x|%d.%m.%Y}<br>SMA 200 : %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ser["date"], y=ser["sma_50"],
        mode="lines", name="SMA 50",
        line=dict(width=1.4, color=FMV_ACCENT),
        hovertemplate="%{x|%d.%m.%Y}<br>SMA 50 : %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ser["date"], y=ser["sma_20"],
        mode="lines", name="SMA 20",
        line=dict(width=1.4, color=FMV_RED),
        hovertemplate="%{x|%d.%m.%Y}<br>SMA 20 : %{y:.2f}<extra></extra>",
    ))

    # Settlement price (bold blue, on top)
    fig.add_trace(go.Scatter(
        x=ser["date"], y=ser["price"],
        mode="lines", name="Settlement Price",
        line=dict(color=FMV_BLUE, width=2.5),
        hovertemplate="%{x|%d.%m.%Y}<br>Settle : %{y:.2f} EUR/MWh<extra></extra>",
    ))

    # Portfolio overlay
    if avg_hedge_price is not None:
        fig.add_hline(
            y=avg_hedge_price,
            line_color=FMV_GREEN, line_width=2, line_dash="dash",
            annotation_text=(
                f"Ihr Ø Hedge-Preis ({selected_actor}) : "
                f"{avg_hedge_price:.2f} EUR/MWh"
            ),
            annotation_position="top right",
            annotation_font_color=FMV_GREEN,
        )

    # High / Low markers
    fig.add_trace(go.Scatter(
        x=[high_date, low_date], y=[high_price, low_price],
        mode="markers+text",
        marker=dict(size=10, color=[FMV_RED, FMV_GREEN], symbol="diamond"),
        text=[f" High {high_price:.1f}", f" Low {low_price:.1f}"],
        textposition="middle right",
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.12),
        hovermode="x unified",
        yaxis=dict(title="EUR/MWh", gridcolor="#EEF1F6"),
        xaxis=dict(gridcolor="#EEF1F6"),
    )
    st.plotly_chart(fig, use_container_width=True)


with c_side:
    delta_color = FMV_GREEN if delta_1d < 0 else FMV_RED
    sign = "+" if delta_1d > 0 else ""
    st.markdown(
        (
            f"<div style='background:#FFFFFF; border:1px solid #E5EBF4;"
            f" border-radius:10px; padding:0.9rem 1.1rem;"
            f" box-shadow: 0 1px 2px rgba(14,31,61,0.04);'>"
            f"<div style='font-size:0.78rem; color:{FMV_GREY};"
            f" text-transform:uppercase; letter-spacing:0.08em;'>Last price</div>"
            f"<div style='font-size:1.8rem; font-weight:700; color:{FMV_NAVY};"
            f" margin-top:0.25rem;'>{last_price:.2f}"
            f" <span style='font-size:0.8rem; color:{FMV_GREY};'>EUR/MWh</span></div>"
            f"<div style='font-size:0.95rem; color:{delta_color};"
            f" margin-top:0.25rem;'>{sign}{delta_1d:.2f}"
            f" ({sign}{delta_pct:.2f}%) ggü. Vortag</div>"
            f"<div style='font-size:0.75rem; color:{FMV_GREY};"
            f" margin-top:0.45rem;'>Stand : "
            f"{pd.Timestamp(last_date).strftime('%d.%m.%Y')}</div>"
            f"</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown("")

    stats_df = pd.DataFrame(
        [
            {"Label": "High", "Date": pd.Timestamp(high_date).strftime("%d.%m.%Y"),
             "Wert": f"{high_price:.2f}"},
            {"Label": "Low", "Date": pd.Timestamp(low_date).strftime("%d.%m.%Y"),
             "Wert": f"{low_price:.2f}"},
            {"Label": "Range", "Date": "—", "Wert": f"{price_range:.2f}"},
            {"Label": "% in Range", "Date": "—", "Wert": f"{position_in_range:.0f}%"},
            {"Label": "Last", "Date": pd.Timestamp(last_date).strftime("%d.%m.%Y"),
             "Wert": f"{last_price:.2f}"},
        ]
    )
    st.dataframe(stats_df, use_container_width=True, hide_index=True, height=220)

    st.markdown(
        (
            f"<div style='background:#F7FAFF; border:1px solid #D8E5FF;"
            f" border-left:6px solid {zone_color}; border-radius:8px;"
            f" padding:0.7rem 0.9rem; margin-top:0.6rem;'>"
            f"<div style='font-size:0.78rem; color:{FMV_GREY};"
            f" text-transform:uppercase; letter-spacing:0.08em;'>Range Zone</div>"
            f"<div style='font-size:1.0rem; font-weight:600; color:{zone_color};"
            f" margin-top:0.2rem;'>{range_zone}</div>"
            f"</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown("")

    interp_df = pd.DataFrame(
        [
            {"Range Zone": "0-20 %", "Interpretation": "Lower range — potential buying opportunity"},
            {"Range Zone": "20-40 %", "Interpretation": "Lower part of the range"},
            {"Range Zone": "40-60 %", "Interpretation": "Mid-range — neutral market"},
            {"Range Zone": "60-80 %", "Interpretation": "Upper part of the range"},
            {"Range Zone": ">80 %", "Interpretation": "Upper range — close to recent highs"},
        ]
    )
    st.caption("Interpretation der Range-Zone")
    st.dataframe(interp_df, use_container_width=True, hide_index=True, height=212)


# ---------------------------------------------------------------------------
# Hedge overlay summary
# ---------------------------------------------------------------------------

if avg_hedge_price is not None:
    delta_vs_market = avg_hedge_price - last_price
    is_below_market = delta_vs_market < 0
    delta_color = FMV_GREEN if is_below_market else FMV_RED
    delta_label = "günstiger als Markt" if is_below_market else "teurer als Markt"
    border_color = FMV_GREEN if is_below_market else FMV_ACCENT

    st.markdown(
        (
            f"<div style='background:#F7FAFF; border:1px solid #D8E5FF;"
            f" border-left:6px solid {border_color};"
            f" border-radius:10px; padding:0.9rem 1.15rem;"
            f" margin-top:0.8rem; line-height:1.6;'>"
            f"<strong style='color:{FMV_NAVY};'>Ihre Hedge-Position "
            f"{selected_actor} · Cal-{delivery_year}</strong><br/>"
            f"<span style='color:{FMV_GREY};'>Ø Preis Ihrer Deals : </span>"
            f"<strong style='color:{FMV_NAVY};'>{avg_hedge_price:.2f} EUR/MWh</strong>"
            f"<span style='color:{FMV_GREY};'> · Markt : </span>"
            f"<strong style='color:{FMV_NAVY};'>{last_price:.2f} EUR/MWh</strong>"
            f"<span style='color:{FMV_GREY};'> · Differenz : </span>"
            f"<strong style='color:{delta_color};'>{delta_vs_market:+.2f} EUR/MWh</strong>"
            f"<span style='color:{FMV_GREY};'> ({delta_label})</span>"
            f"<div style='color:{FMV_GREY}; font-size:0.85rem; margin-top:0.3rem;'>"
            f"Hedged Volumen : {hedged_volume_mwh:,.0f} MWh".replace(",", "'") +
            f"</div></div>"
        ),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Spot + LEAR forecast block
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Spot CH · letzte 14 Tage + LEAR-Prognose 10 Tage")

if not spot.empty:
    spot_last = spot.index.max()
    today_ch = pd.Timestamp.now(tz="Europe/Zurich")
    lag_days = (today_ch - spot_last).days
    if lag_days > 3:
        st.warning(
            f"⚠️ Spot-Daten sind {lag_days} Tage alt — letzte beobachtete Stunde "
            f"**{spot_last.strftime('%d.%m.%Y %H:%M')}**. Die Grafik zeigt die "
            f"14 Tage **vor** dem letzten Datenpunkt (nicht vor heute) und 10 Tage "
            f"LEAR-Prognose ab dort. Cache aktualisieren über die Seitenleiste."
        )
    hist_cutoff = spot_last - pd.Timedelta(days=14)
    spot_recent = spot.loc[spot.index >= hist_cutoff, "price"].resample("h").mean()

    fig_spot = go.Figure()
    fig_spot.add_trace(go.Scatter(
        x=spot_recent.index, y=spot_recent.values,
        mode="lines", name="Spot realisiert",
        line=dict(color=FMV_NAVY, width=2),
    ))

    if not forecast.empty:
        fcst = forecast.sort_values("timestamp").copy()
        if {"price_p10", "price_p90"}.issubset(fcst.columns):
            fig_spot.add_trace(go.Scatter(
                x=fcst["timestamp"], y=fcst["price_p90"],
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            fig_spot.add_trace(go.Scatter(
                x=fcst["timestamp"], y=fcst["price_p10"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(15,82,204,0.15)",
                name="P10–P90 Band",
            ))
        if "price_lear" in fcst.columns:
            fig_spot.add_trace(go.Scatter(
                x=fcst["timestamp"], y=fcst["price_lear"],
                mode="lines", name="LEAR Prognose",
                line=dict(color=FMV_BLUE, width=2.5),
            ))

    fig_spot.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
        yaxis=dict(title="EUR/MWh", gridcolor="#EEF1F6"),
        xaxis=dict(gridcolor="#EEF1F6"),
        legend=dict(orientation="h", y=-0.12),
    )
    st.plotly_chart(fig_spot, use_container_width=True)
else:
    st.info("Keine Spot-Daten verfügbar.")


# ---------------------------------------------------------------------------
# Commodities ribbon
# ---------------------------------------------------------------------------

if not commodities.empty:
    st.subheader("Commodities · letzte 90 Tage (Index, Start = 100)")
    recent = commodities.tail(90)
    fig_com = go.Figure()
    palette = {"TTF Gas": FMV_BLUE, "CO2 EUA (KRBN)": FMV_ACCENT, "Brent": FMV_GREY}
    for col, color in palette.items():
        if col in recent.columns:
            s = recent[col].dropna()
            if not s.empty:
                normalized = s / s.iloc[0] * 100
                fig_com.add_trace(go.Scatter(
                    x=normalized.index, y=normalized.values,
                    mode="lines", name=col,
                    line=dict(color=color, width=2),
                ))
    fig_com.update_layout(
        height=280, margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="Index", gridcolor="#EEF1F6"),
        xaxis=dict(gridcolor="#EEF1F6"),
        legend=dict(orientation="h", y=-0.18),
    )
    st.plotly_chart(fig_com, use_container_width=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Methodik : SMA 20 / 50 / 200 = einfache gleitende Mittelwerte. "
    "Bollinger Bands = SMA 20 ± 2 × σ rollend (20 Tage). "
    "Range-Zone = Position des aktuellen Preises im (High − Low)-Intervall des "
    "gewählten Lookback-Fensters. Ø Hedge-Preis = volumen-gewichteter Durchschnitt "
    "der Deals des gewählten Akteurs für das Lieferjahr des gewählten Cal-Produkts."
)

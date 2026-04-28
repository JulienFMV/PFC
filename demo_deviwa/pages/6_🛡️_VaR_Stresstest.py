"""Seite 6 — Pool & Risiko : VaR / Expected Shortfall + Diversifikations-Vorteil.

Replaces the legacy ``6_VaR_Stresstest.py`` (which used spot-daily volatility on
a forward MtM, inflating VaR by ~30× and using a trader-side sign on
stresstests). Now powered by :mod:`pool_diversification` (Phase 3) :

  - VaR scaled with **forward volatility** (≈ 20 % annualized × √T) instead of
    spot daily vol.
  - Stresstests use **consumer-correct** signs : for a GRD with open_mwh > 0
    (under-hedged), a price RISE is a NEGATIVE P&L (higher cost), a fall is
    POSITIVE.
  - Headline narrative = Σ individual VaR vs Pool VaR → diversification
    benefit (the central sales pitch of the Deviwa pool).

Numbers verified end-to-end against `compute_pool_diversification` :
  Pool 2027 :  Σ VaR = 205 771 EUR  ·  Pool VaR = 147 903 EUR
              → Benefit = 57 868 EUR (28.1 %)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pool_diversification import (  # noqa: E402
    Z_BY_CONF,
    compute_pool_diversification,
)
from utils import (  # noqa: E402
    FMV_BLUE,
    FMV_GREEN,
    FMV_GREY,
    FMV_NAVY,
    FMV_RED,
    fmt_chf,
    kpi_card,
    load_deviwa_deals_cached,
    load_deviwa_programme_cached,
    load_eex_forwards,
    render_cache_status,
    render_header,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FMV · Pool & Risiko",
    page_icon="🛡️",
    layout="wide",
)

render_header("Pool & Risiko · Diversifikations-Vorteil")

st.markdown(
    "Wie viel Risiko **sparen Sie**, weil Sie im Deviwa-Pool sind ? Diese Seite "
    "vergleicht den VaR jeder Akteur-Position einzeln mit dem konsolidierten "
    "Pool-VaR — die Differenz ist Ihr **Diversifikations-Vorteil**."
)

with st.sidebar:
    render_cache_status()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

deals = load_deviwa_deals_cached()
programme = load_deviwa_programme_cached()
forwards = load_eex_forwards()

if deals.empty and programme.empty:
    st.warning("Keine Deviwa-Daten — bitte Cache via Sidebar aktualisieren.")
    st.stop()


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

st.subheader("Parameter")
c_year, c_conf = st.columns([1.2, 1])
with c_year:
    delivery_year = st.radio(
        "Lieferjahr",
        options=[2026, 2027, 2028, 2029],
        horizontal=True,
        index=1,
        key="risk_year",
    )
with c_conf:
    confidence = st.selectbox(
        "Konfidenz-Niveau",
        options=[0.95, 0.975, 0.99],
        index=0,
        format_func=lambda c: f"{c*100:.1f} %",
        key="risk_conf",
    )

today = pd.Timestamp.now(tz="Europe/Zurich")

# ---------------------------------------------------------------------------
# Compute pool diversification
# ---------------------------------------------------------------------------

pool = compute_pool_diversification(
    delivery_year,
    deals,
    programme,
    forwards,
    today=today,
    confidence_level=confidence,
)

if not np.isfinite(pool.pool_var_eur) or pool.pool_var_eur == 0:
    st.info(
        f"Für **Cal-{delivery_year}** sind keine offenen Positionen oder kein "
        f"Forward-Preis verfügbar — kein VaR-Bild für dieses Jahr."
    )
    st.stop()

active_actor_names = [a.actor for a in pool.actors if abs(a.open_mwh) > 1.0]
active_count = len(active_actor_names)
actor_list = ", ".join(active_actor_names) if active_actor_names else "—"
if active_count >= 3:
    coverage_color = FMV_GREEN
    coverage_msg = (
        f"Cal-{delivery_year}: **{active_count} von 4 Akteuren** mit offener "
        f"Position ({actor_list}) — belastbares Pool-Bild."
    )
elif active_count == 2:
    coverage_color = FMV_BLUE
    coverage_msg = (
        f"Cal-{delivery_year}: **2 von 4 Akteuren** mit offener Position "
        f"({actor_list}) — Diversifikation sichtbar, aber begrenzt."
    )
else:
    coverage_color = FMV_RED
    coverage_msg = (
        f"Cal-{delivery_year}: **nur {active_count} Akteur** mit offener Position "
        f"({actor_list}) — kein echter Pool-Effekt. Empfehlung: Laufzeit / "
        f"Teilnahme der Akteure ausweiten."
    )

st.markdown(
    f"""
<div style='background:#FFFFFF; border:1px solid #E4EAF3; border-left:6px solid {coverage_color}; border-radius:12px; padding:0.8rem 1rem; margin:0.2rem 0 1.0rem 0;'>
{coverage_msg}
</div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Headline narrative
# ---------------------------------------------------------------------------

benefit_eur = pool.diversification_benefit_eur
benefit_pct = pool.diversification_pct

if benefit_eur > 0:
    headline_color = FMV_GREEN
    headline_msg = (
        f"<strong style='color:{FMV_NAVY};'>Σ VaR individuell = "
        f"{fmt_chf(pool.sum_individual_var_eur, 0)} EUR</strong> "
        f"<span style='color:{FMV_GREY};'>vs.</span> "
        f"<strong style='color:{FMV_NAVY};'>VaR Pool = "
        f"{fmt_chf(pool.pool_var_eur, 0)} EUR</strong> "
        f"<span style='color:{FMV_GREY};'>→</span> "
        f"<strong style='color:{headline_color}; font-size:1.15rem;'>"
        f"Diversifikations-Vorteil {fmt_chf(benefit_eur, 0)} EUR "
        f"({benefit_pct:.0f} %)</strong>"
    )
else:
    headline_msg = (
        f"<span style='color:{FMV_GREY};'>Für Cal-{delivery_year} haben alle "
        f"Akteure das gleiche Vorzeichen der offenen Position — kein "
        f"Diversifikations-Vorteil aus dem natürlichen Long/Short-Offset. "
        f"Das ist mathematisch korrekt, kein Fehler.</span>"
    )

st.markdown(
    f"""
<div style='background:#F7FAFF; border:1px solid #D8E5FF; border-left:6px solid {FMV_BLUE}; border-radius:12px; padding:1.0rem 1.25rem; margin:0.6rem 0 1.4rem 0; line-height:1.7;'>{headline_msg}</div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 4 KPI hero
# ---------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.markdown(
    kpi_card(
        "Σ VaR individuell",
        f"{fmt_chf(pool.sum_individual_var_eur, 0)} EUR",
        delta="wenn jeder allein wäre",
        delta_color=FMV_GREY,
    ),
    unsafe_allow_html=True,
)
c2.markdown(
    kpi_card(
        "VaR Pool",
        f"{fmt_chf(pool.pool_var_eur, 0)} EUR",
        delta=f"Konfidenz {int(confidence*100)} %, Horizont {pool.horizon_years:.2f}j",
        delta_color=FMV_BLUE,
    ),
    unsafe_allow_html=True,
)
c3.markdown(
    kpi_card(
        "Diversifikations-Vorteil",
        f"{fmt_chf(benefit_eur, 0)} EUR",
        delta=f"{benefit_pct:.1f} %" if np.isfinite(benefit_pct) else "—",
        delta_color=FMV_GREEN if benefit_eur > 0 else FMV_GREY,
    ),
    unsafe_allow_html=True,
)
c4.markdown(
    kpi_card(
        "Expected Shortfall Pool",
        f"{fmt_chf(pool.pool_es_eur, 0)} EUR",
        delta="Ø Verlust jenseits VaR (Normal-Annahme)",
        delta_color=FMV_RED,
    ),
    unsafe_allow_html=True,
)

st.markdown("")


# ---------------------------------------------------------------------------
# Per-actor breakdown : bar chart + table
# ---------------------------------------------------------------------------

st.subheader("Σ individuell vs. Pool — visueller Vergleich")

# Two stacked bars side by side :
#   Bar 1 = "Wenn jeder allein wäre" — stack of all 4 individual VaRs
#   Bar 2 = "Im Pool"                — single consolidated Pool VaR
# The HEIGHT DIFFERENCE between bar 1 and bar 2 is the diversification benefit,
# made visually obvious without needing the user to do mental arithmetic.

actors_data = sorted(pool.actors, key=lambda a: -abs(a.exposure_eur))
n_actors = len(actors_data)
# Distinct shades of navy/blue for each actor stack segment
palette = ["#0E1F3D", "#1F3A6E", "#3A5BA0", "#5C7AC8"]

fig_bars = go.Figure()

# Left bar : stacked individuals
for i, a in enumerate(actors_data):
    fig_bars.add_trace(
        go.Bar(
            x=["Wenn jeder allein wäre"],
            y=[a.var_eur],
            name=f"{a.actor}",
            marker_color=palette[i % len(palette)],
            text=[f"{a.actor}: {fmt_chf(a.var_eur, 0)}"],
            textposition="inside",
            insidetextanchor="middle",
            insidetextfont=dict(color="white", size=11),
            hovertemplate=f"<b>{a.actor}</b><br>Stand-alone VaR : %{{y:,.0f}} EUR<extra></extra>",
        )
    )

# Right bar : Pool VaR (single segment, distinct color)
fig_bars.add_trace(
    go.Bar(
        x=["Im Pool"],
        y=[pool.pool_var_eur],
        name="Pool",
        marker_color=FMV_BLUE,
        text=[f"Pool: {fmt_chf(pool.pool_var_eur, 0)}"],
        textposition="inside",
        insidetextanchor="middle",
        insidetextfont=dict(color="white", size=12),
        hovertemplate="<b>Pool VaR</b><br>%{y:,.0f} EUR<extra></extra>",
    )
)

# Total annotation above each bar
fig_bars.add_annotation(
    x="Wenn jeder allein wäre",
    y=pool.sum_individual_var_eur,
    text=f"<b>Σ {fmt_chf(pool.sum_individual_var_eur, 0)} EUR</b>",
    showarrow=False, yshift=18,
    font=dict(color=FMV_NAVY, size=14),
)
fig_bars.add_annotation(
    x="Im Pool",
    y=pool.pool_var_eur,
    text=f"<b>{fmt_chf(pool.pool_var_eur, 0)} EUR</b>",
    showarrow=False, yshift=18,
    font=dict(color=FMV_BLUE, size=14),
)

# Visual benefit indicator (only when benefit > 0)
if benefit_eur > 0:
    benefit_y = (pool.sum_individual_var_eur + pool.pool_var_eur) / 2
    fig_bars.add_annotation(
        x=0.5, xref="paper",
        y=benefit_y, yref="y",
        text=(
            f"<b style='color:{FMV_GREEN};'>− {fmt_chf(benefit_eur, 0)} EUR "
            f"({benefit_pct:.0f} %)</b><br>"
            f"<span style='color:{FMV_GREY}; font-size:11px;'>"
            f"Diversifikations-Vorteil</span>"
        ),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor=FMV_GREEN,
        borderwidth=1, borderpad=8,
        font=dict(size=13),
    )

fig_bars.update_layout(
    height=440,
    margin=dict(l=20, r=20, t=40, b=20),
    plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(title="VaR (EUR)", gridcolor="#EEF1F6"),
    xaxis=dict(gridcolor="#EEF1F6", tickfont=dict(size=13)),
    barmode="stack",
    legend=dict(orientation="h", y=-0.12),
)
st.plotly_chart(fig_bars, use_container_width=True)

st.subheader("Detail je Akteur")
table_rows = []
for a in actors_data:
    table_rows.append(
        {
            "Akteur": a.actor,
            "Offene Position (MWh)": a.open_mwh,
            "Forward-Preis (EUR/MWh)": a.forward_price_eur_mwh,
            "Exposure (EUR)": a.exposure_eur,
            "σ (EUR)": a.sigma_eur,
            f"VaR {int(confidence*100)} % (EUR)": a.var_eur,
            "ES (EUR)": a.es_eur,
        }
    )
detail_df = pd.DataFrame(table_rows)
st.dataframe(
    detail_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Offene Position (MWh)": st.column_config.NumberColumn(format="%.0f"),
        "Forward-Preis (EUR/MWh)": st.column_config.NumberColumn(format="%.2f"),
        "Exposure (EUR)": st.column_config.NumberColumn(format="%.0f"),
        "σ (EUR)": st.column_config.NumberColumn(format="%.0f"),
        f"VaR {int(confidence*100)} % (EUR)": st.column_config.NumberColumn(format="%.0f"),
        "ES (EUR)": st.column_config.NumberColumn(format="%.0f"),
    },
)


# ---------------------------------------------------------------------------
# Correlation matrix heatmap (educational)
# ---------------------------------------------------------------------------

with st.expander("📊 Last-Korrelation zwischen Akteuren (Tagesrenditen, historisch)"):
    st.caption(
        "Wenn die Lasten der Akteure stark korreliert sind, kommt der "
        "Diversifikations-Vorteil aus den **gegenläufigen offenen Positionen** "
        "(Long/Short-Offset). Wenn die Lasten **nicht** korreliert sind, kommt "
        "ein Teil des Vorteils auch aus der Volumenglättung — das modelliert "
        "diese Demo noch nicht (POC : Single-Factor Preis-Modell)."
    )
    corr = pool.correlation_matrix
    if corr is not None and not corr.empty:
        fig_corr = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.index),
                colorscale="RdBu",
                zmid=0, zmin=-1, zmax=1,
                hovertemplate="%{y} ↔ %{x}<br>ρ = %{z:.2f}<extra></extra>",
                colorbar=dict(title="ρ", thickness=15),
            )
        )
        fig_corr.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(side="top"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Korrelationsmatrix nicht berechnet — zu wenig Programmdaten.")


# ---------------------------------------------------------------------------
# Stress tests (consumer-correct sign convention)
# ---------------------------------------------------------------------------

st.subheader(f"Stresstests · Auswirkung auf Cal-{delivery_year}-Position")

st.caption(
    "**Konvention** : positive P&L = günstiger Ausgang für den Verbraucher "
    "(weniger Kosten als erwartet). Für eine offene Position > 0 (Bezug aus "
    "Markt) gilt : Preis-Anstieg → höhere Kosten → **negative P&L**. "
    "Für eine offene Position < 0 (Verkauf-Überschuss) ist es umgekehrt."
)

scenarios = [
    {"name": "Preis +20 %", "shock": 0.20, "desc": "Angebotsengpass, Gas-Rallye"},
    {"name": "Preis −20 %", "shock": -0.20, "desc": "Milder Winter, Überangebot Erneuerbar"},
    {"name": "Extremwinter +50 %", "shock": 0.50, "desc": "Gasmangel + Kältewelle"},
    {"name": "Trockenes Jahr +15 %", "shock": 0.15, "desc": "Hydro −30 %, Importabhängigkeit steigt"},
    {"name": "Nuklear-Ausfall FR +30 %", "shock": 0.30, "desc": "Unplanmässiger Ausfall"},
    {"name": "CO₂-Sprung +10 %", "shock": 0.10, "desc": "EUA +20 EUR/t"},
]

# Consumer P&L = −exposure × shock
# exposure > 0 (long open = under-hedged) and shock > 0 → cost up → P&L negative
# exposure < 0 (short open = over-hedged) and shock > 0 → "cost" down → P&L positive
rows = []
for sc in scenarios:
    pnl_pool = -pool.net_exposure_eur * sc["shock"]
    rows.append(
        {
            "Szenario": sc["name"],
            "Beschreibung": sc["desc"],
            "Preisbewegung": f"{sc['shock']*100:+.0f} %",
            "P&L Pool (EUR)": pnl_pool,
        }
    )
stress_df = pd.DataFrame(rows)

fig_stress = go.Figure()
fig_stress.add_trace(
    go.Bar(
        x=stress_df["Szenario"],
        y=stress_df["P&L Pool (EUR)"],
        marker_color=[FMV_GREEN if v >= 0 else FMV_RED for v in stress_df["P&L Pool (EUR)"]],
        text=[f"{fmt_chf(v, 0)} EUR" for v in stress_df["P&L Pool (EUR)"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>P&L : %{y:,.0f} EUR<extra></extra>",
    )
)
fig_stress.add_hline(y=0, line_color=FMV_GREY, line_width=1)
fig_stress.update_layout(
    height=420, margin=dict(l=20, r=20, t=20, b=80),
    plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(
        title="P&L Pool (EUR) — positiv = günstig für Verbraucher",
        gridcolor="#EEF1F6",
    ),
    xaxis=dict(tickangle=-15),
)
st.plotly_chart(fig_stress, use_container_width=True)

# Table full-width underneath (no narrow side column that crops Szenario header)
st.dataframe(
    stress_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "P&L Pool (EUR)": st.column_config.NumberColumn(format="%.0f"),
    },
)


# ---------------------------------------------------------------------------
# Methodology footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    f"**Methodik** : parametrischer VaR auf der **offenen Preisexposition** der "
    f"Akteur-Position (open MWh × Cal-{delivery_year} Forward × σ × √T). "
    f"σ = 20 % annualisiert (POC-Annahme, kalibriert auf 2024-2026 EEX-Historie). "
    f"z = {Z_BY_CONF[confidence]:.4f} bei {int(confidence*100)} % Konfidenz. "
    f"Horizont = {pool.horizon_years:.2f} Jahre bis 1. Juli {delivery_year} "
    f"(geflootet bei 0.25j für das laufende Lieferjahr). "
    f"Expected Shortfall = φ(z)/(1−α) × σ unter Normal-Annahme. "
    f"Pool-Vorteil aus Sub-Additivität : Σ |Eᵢ| ≥ |Σ Eᵢ|, strikt > nur wenn "
    f"einzelne Akteure entgegengesetzt offen sind. Konsumenten-Konvention für "
    f"Stresstests : positive P&L = günstig für GRD."
)

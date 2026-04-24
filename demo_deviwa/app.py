"""FMV Deviwa Demo – Startseite: Portfolio-Cockpit.

Portfolio-Übersicht auf Energiehandels-Niveau: Hedge-Quote, Notional, MtM,
realisiertes und unrealisiertes P&L, Ø Hedge-Preis, Tenor-Struktur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from portfolio_analytics import (
    compute_portfolio_metrics,
    monthly_position,
)
from utils import (
    FMV_ACCENT,
    FMV_BLUE,
    FMV_GREEN,
    FMV_GREY,
    FMV_NAVY,
    FMV_RED,
    find_deviwa_file,
    fmt_chf,
    fmt_eur_mwh,
    fmt_mwh,
    freshness_badge,
    kpi_card,
    load_deviwa_auto,
    load_eex_forwards,
    load_pfc,
    render_actor_selector,
    render_header,
)

st.set_page_config(
    page_title="FMV Deviwa · Portfolio-Cockpit",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_header("Portfolio-Cockpit")

with st.sidebar:
    st.markdown(
        f"<div style='color:{FMV_NAVY}; font-weight:700; font-size:1.05rem;'>"
        "FMV · Deviwa Energiepool</div>", unsafe_allow_html=True,
    )
    st.caption("Demo · alle Daten lokal")
    st.divider()
    st.markdown("**Navigation**")
    st.markdown(
        "- 🎯 Portfolio-Cockpit\n- 📊 Marktüberblick\n- 📈 Kurzfristprognose\n"
        "- 💡 Lastprofil-Pricing\n- 📒 Ihre Transaktionen\n- 🎚️ Programmqualität\n"
        "- 🛡️ VaR & Stresstest\n- 🌍 Marktdaten"
    )

# ---------------------------------------------------------------------------
# Daten
# ---------------------------------------------------------------------------
data = load_deviwa_auto()
pfc = load_pfc()
forwards = load_eex_forwards()

if not data:
    st.warning("Keine Deviwa-Datei in `/data` gefunden. Bitte `Deviwa.xlsx` ablegen.")
    st.stop()

choice, df = render_actor_selector(data, key="cockpit_actor")
if df.empty:
    st.stop()

st.markdown(
    f"<div style='color:{FMV_GREY}; font-size:0.95rem; margin-top:-0.6rem; margin-bottom:1rem;'>"
    f"Ansicht: <strong style='color:{FMV_NAVY};'>{choice}</strong>"
    f" · Zeitstempel: {pd.Timestamp.now(tz='Europe/Zurich').strftime('%d.%m.%Y %H:%M')}</div>",
    unsafe_allow_html=True,
)

m = compute_portfolio_metrics(df, pfc=pfc)
pfc_avg_price = float(pfc["price_shape"].mean()) if not pfc.empty and "price_shape" in pfc.columns else float("nan")

# ---------------------------------------------------------------------------
# KPI-Reihe 1: Position und Notional
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.markdown(
    kpi_card(
        "Brutto-Volumen",
        fmt_mwh(m.total_volume_mwh, 0),
        delta=f"{m.n_deals} Deals · IN {fmt_mwh(m.intake_mwh,0)} · OUT {fmt_mwh(m.withdrawal_mwh,0)}",
        delta_color=FMV_NAVY,
    ),
    unsafe_allow_html=True,
)
c2.markdown(
    kpi_card(
        "Netto-Position",
        fmt_mwh(m.net_volume_mwh, 0),
        delta="Einspeisung − Bezug",
        delta_color=FMV_GREEN if m.net_volume_mwh >= 0 else FMV_RED,
    ),
    unsafe_allow_html=True,
)
c3.markdown(
    kpi_card(
        "Notional-Volumen",
        f"{fmt_chf(m.notional_total_eur, 0)} EUR",
        delta=f"Ø Deal-Preis {fmt_eur_mwh(m.avg_deal_price_eur_mwh)}",
        delta_color=FMV_BLUE,
    ),
    unsafe_allow_html=True,
)
c4.markdown(
    kpi_card(
        "Mark-to-Market",
        f"{fmt_chf(m.mtm_eur, 0)} EUR" if np.isfinite(m.mtm_eur) else "—",
        delta=f"PFC Ø {fmt_eur_mwh(pfc_avg_price)}" if np.isfinite(pfc_avg_price) else None,
        delta_color=FMV_NAVY,
        help_text="Summe Netto-Volumen × aktueller PFC je Lieferperiode",
    ),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# KPI-Reihe 2: Hedge und P&L
# ---------------------------------------------------------------------------
c5, c6, c7, c8 = st.columns(4)
c5.markdown(
    kpi_card(
        "Hedge-Quote",
        f"{m.hedge_ratio_pct:.0f} %" if np.isfinite(m.hedge_ratio_pct) else "—",
        delta=f"{fmt_mwh(m.hedged_volume_mwh,0)} zukünftig gesichert",
        delta_color=(FMV_GREEN if (m.hedge_ratio_pct or 0) >= 70 else
                     FMV_ACCENT if (m.hedge_ratio_pct or 0) >= 40 else FMV_RED),
    ),
    unsafe_allow_html=True,
)
c6.markdown(
    kpi_card(
        "Ø Hedge-Preis",
        fmt_eur_mwh(m.avg_hedge_price_eur_mwh) if np.isfinite(m.avg_hedge_price_eur_mwh) else "—",
        delta=(f"Markt Ø {fmt_eur_mwh(pfc_avg_price)} · Δ {fmt_eur_mwh(m.avg_hedge_price_eur_mwh - pfc_avg_price)}"
               if np.isfinite(m.avg_hedge_price_eur_mwh) and np.isfinite(pfc_avg_price) else None),
        delta_color=FMV_BLUE,
    ),
    unsafe_allow_html=True,
)
c7.markdown(
    kpi_card(
        "Realisierter P&L",
        f"{fmt_chf(m.realized_pnl_eur, 0)} EUR" if np.isfinite(m.realized_pnl_eur) else "—",
        delta="abgeschlossene Lieferungen",
        delta_color=FMV_GREEN if (m.realized_pnl_eur or 0) >= 0 else FMV_RED,
    ),
    unsafe_allow_html=True,
)
c8.markdown(
    kpi_card(
        "Unrealisierter P&L",
        f"{fmt_chf(m.unrealized_pnl_eur, 0)} EUR" if np.isfinite(m.unrealized_pnl_eur) else "—",
        delta="offene Positionen (MtM)",
        delta_color=FMV_GREEN if (m.unrealized_pnl_eur or 0) >= 0 else FMV_RED,
    ),
    unsafe_allow_html=True,
)

st.markdown("")

# ---------------------------------------------------------------------------
# Hedge-Gauge + Tenor-Struktur
# ---------------------------------------------------------------------------
col_gauge, col_tenor = st.columns([0.9, 1.4])

with col_gauge:
    st.subheader("Hedge-Quote")
    hedge_pct = float(m.hedge_ratio_pct) if np.isfinite(m.hedge_ratio_pct) else 0.0
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=hedge_pct,
        number={"suffix": " %", "font": {"size": 42, "color": FMV_NAVY}},
        delta={"reference": 70, "position": "top",
               "increasing": {"color": FMV_GREEN},
               "decreasing": {"color": FMV_RED}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": FMV_GREY},
            "bar": {"color": FMV_BLUE, "thickness": 0.75},
            "steps": [
                {"range": [0, 40], "color": "#FCE8E8"},
                {"range": [40, 70], "color": "#FFF4D6"},
                {"range": [70, 100], "color": "#E6F4EA"},
            ],
            "threshold": {
                "line": {"color": FMV_NAVY, "width": 3},
                "thickness": 0.8, "value": 70,
            },
        },
    ))
    gauge.update_layout(
        height=320, margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="white",
    )
    st.plotly_chart(gauge, use_container_width=True)
    st.caption("Zielwert 70 % · unter 40 % = rot · über 70 % = grün")

with col_tenor:
    st.subheader("Tenor-Struktur · Volumen und Ø Preis je Lieferperiode")
    tb = m.tenor_buckets
    if tb is None or tb.empty:
        st.info("Keine Lieferdaten verfügbar.")
    else:
        fig_tb = make_subplots(specs=[[{"secondary_y": True}]])
        fig_tb.add_trace(
            go.Bar(
                x=tb["bucket"].astype(str), y=tb["volume_mwh"],
                marker_color=FMV_BLUE, opacity=0.85,
                name="Volumen (MWh)",
                text=[fmt_mwh(v, 0) for v in tb["volume_mwh"]],
                textposition="outside",
            ),
            secondary_y=False,
        )
        fig_tb.add_trace(
            go.Scatter(
                x=tb["bucket"].astype(str), y=tb["avg_price_eur_mwh"],
                mode="lines+markers+text",
                name="Ø Preis (EUR/MWh)",
                line=dict(color=FMV_ACCENT, width=3),
                marker=dict(size=10),
                text=[fmt_eur_mwh(p) if np.isfinite(p) else "" for p in tb["avg_price_eur_mwh"]],
                textposition="top center",
            ),
            secondary_y=True,
        )
        fig_tb.update_layout(
            height=320, margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.18),
        )
        fig_tb.update_yaxes(title_text="Volumen (MWh)", gridcolor="#EEF1F6", secondary_y=False)
        fig_tb.update_yaxes(title_text="Ø Preis (EUR/MWh)", showgrid=False, secondary_y=True)
        st.plotly_chart(fig_tb, use_container_width=True)

# ---------------------------------------------------------------------------
# Monatsverlauf mit Ø Hedge-Preis vs Forward-Kurve
# ---------------------------------------------------------------------------
st.subheader("Monatliche Position und Preisniveau")

mp = monthly_position(df)
if not mp.empty:
    fig_m = make_subplots(specs=[[{"secondary_y": True}]])
    fig_m.add_trace(
        go.Bar(x=mp["month"], y=mp["intake_mwh"],
               name="Einspeisung", marker_color=FMV_GREEN, opacity=0.8),
        secondary_y=False,
    )
    fig_m.add_trace(
        go.Bar(x=mp["month"], y=-mp["withdrawal_mwh"],
               name="Bezug", marker_color=FMV_BLUE, opacity=0.8),
        secondary_y=False,
    )
    fig_m.add_trace(
        go.Scatter(x=mp["month"], y=mp["net_mwh"],
                   mode="lines+markers", name="Netto-Position",
                   line=dict(color=FMV_NAVY, width=2.5)),
        secondary_y=False,
    )

    # Ø Hedge-Preis (Intake = Kaufpreis)
    if mp["avg_intake_price"].notna().any():
        fig_m.add_trace(
            go.Scatter(
                x=mp["month"], y=mp["avg_intake_price"],
                mode="lines+markers", name="Ø Einspeise-Preis",
                line=dict(color=FMV_ACCENT, width=2, dash="dash"),
                marker=dict(size=6),
            ),
            secondary_y=True,
        )
    # PFC-Kurve je Monat als Markt-Referenz
    if not pfc.empty:
        pfc_m = pfc["price_shape"].resample("MS").mean()
        pfc_m.index = pfc_m.index.tz_convert("Europe/Zurich").tz_localize(None)
        pfc_plot = pfc_m.reindex(pd.to_datetime(mp["month"]))
        fig_m.add_trace(
            go.Scatter(
                x=pfc_plot.index, y=pfc_plot.values,
                mode="lines", name="Markt (PFC)",
                line=dict(color=FMV_RED, width=2, dash="dot"),
            ),
            secondary_y=True,
        )

    fig_m.add_hline(y=0, line_width=1, line_dash="dot", line_color=FMV_GREY)
    fig_m.update_layout(
        barmode="relative",
        height=460, margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
    )
    fig_m.update_yaxes(title_text="MWh", gridcolor="#EEF1F6", secondary_y=False)
    fig_m.update_yaxes(title_text="Preis (EUR/MWh)", showgrid=False, secondary_y=True)
    st.plotly_chart(fig_m, use_container_width=True)

# ---------------------------------------------------------------------------
# P&L Wasserfall und kumulierter Verlauf
# ---------------------------------------------------------------------------
col_wf, col_cum = st.columns([1, 1.3])

with col_wf:
    st.subheader("P&L-Aufschlüsselung")
    components = []
    if np.isfinite(m.realized_pnl_eur):
        components.append(("Realisiert", m.realized_pnl_eur))
    if np.isfinite(m.unrealized_pnl_eur):
        components.append(("Unrealisiert (MtM)", m.unrealized_pnl_eur))
    if components:
        total = sum(v for _, v in components)
        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative"] * len(components) + ["total"],
            x=[c[0] for c in components] + ["Gesamt"],
            y=[c[1] for c in components] + [total],
            text=[f"{fmt_chf(c[1],0)} EUR" for c in components] + [f"{fmt_chf(total,0)} EUR"],
            textposition="outside",
            connector={"line": {"color": FMV_GREY}},
            increasing={"marker": {"color": FMV_GREEN}},
            decreasing={"marker": {"color": FMV_RED}},
            totals={"marker": {"color": FMV_NAVY}},
        ))
        fig_wf.update_layout(
            height=360, margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="EUR", gridcolor="#EEF1F6"),
        )
        st.plotly_chart(fig_wf, use_container_width=True)

with col_cum:
    st.subheader("Kumulierter P&L im Zeitverlauf")
    if "pnl_sum" in df.columns and "month" in df.columns:
        g = df.copy()
        g["month"] = pd.to_datetime(g["month"], errors="coerce")
        pnl_m = g.groupby("month")["pnl_sum"].apply(
            lambda s: float(pd.to_numeric(s, errors="coerce").sum())
        ).sort_index()
        cum = pnl_m.cumsum()
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Bar(
            x=pnl_m.index, y=pnl_m.values,
            name="Monats-P&L",
            marker_color=[FMV_GREEN if v >= 0 else FMV_RED for v in pnl_m.values],
            opacity=0.7,
        ))
        fig_cum.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            mode="lines+markers", name="Kumuliert",
            line=dict(color=FMV_NAVY, width=3),
        ))
        fig_cum.add_hline(y=0, line_width=1, line_dash="dot", line_color=FMV_GREY)
        fig_cum.update_layout(
            height=360, margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="EUR", gridcolor="#EEF1F6"),
            xaxis=dict(gridcolor="#EEF1F6"),
            legend=dict(orientation="h", y=-0.18),
            hovermode="x unified",
        )
        st.plotly_chart(fig_cum, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
cols = st.columns(3)
cols[0].markdown(freshness_badge(pfc, "PFC"), unsafe_allow_html=True)
cols[1].markdown(
    freshness_badge(forwards.set_index("date") if not forwards.empty else forwards, "EEX Forwards"),
    unsafe_allow_html=True,
)
path = find_deviwa_file()
if path is not None:
    mtime = pd.Timestamp(path.stat().st_mtime, unit="s", tz="Europe/Zurich")
    cols[2].markdown(
        f"<span style='color:{FMV_GREEN};'>● Deviwa-Datei: "
        f"{mtime.strftime('%d.%m.%Y %H:%M')}</span>",
        unsafe_allow_html=True,
    )

st.caption(
    "Ø Hedge-Preis: volumengewichteter Durchschnittspreis aller Deals mit Lieferung in der Zukunft. "
    "Hedge-Quote: Anteil des Gesamtvolumens, das bereits in Zukunft festgelegt ist. "
    "Unrealisierter P&L: Differenz zwischen Marktbewertung (PFC) und Notional der offenen Positionen."
)

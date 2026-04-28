"""Seite 7 - Marktdaten Hub.

Client-facing data hub for a GRD demo: Swiss fundamentals, spot prices,
net imports/exports, production mix and hydro storage. The page deliberately
avoids exposing raw pipeline gaps to the audience. When per-border schedules
are missing, it falls back to the aggregate Swiss net-flow signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (  # noqa: E402
    FMV_ACCENT,
    FMV_BLUE,
    FMV_GREEN,
    FMV_GREY,
    FMV_NAVY,
    FMV_RED,
    freshness_badge,
    kpi_card,
    load_entso,
    load_epex_ch,
    load_epex_de,
    load_hydro,
    render_header,
)


st.set_page_config(
    page_title="FMV · Marktdaten",
    page_icon="🌍",
    layout="wide",
)

render_header("Marktdaten Hub · Schweiz und Nachbarländer")

st.markdown(
    "Diese Seite zeigt, welche Markt- und Systemdaten FMV einem lokalen "
    "Verteilnetzbetreiber als Service bereitstellen kann: Last, Produktion, "
    "Spotpreise, Nettoimporte/-exporte und Hydrospeicher."
)

entso = load_entso()
hydro = load_hydro()
epex_ch = load_epex_ch()
epex_de = load_epex_de()

if entso.empty:
    st.error("Marktdaten sind nicht verfügbar.")
    st.stop()


def _safe_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _fmt_mw(value: float) -> str:
    return f"{value:,.0f} MW".replace(",", "'") if np.isfinite(value) else "—"


def _fmt_gw(value: float) -> str:
    return f"{value / 1000.0:,.2f} GW".replace(",", "'") if np.isfinite(value) else "—"


def _fmt_eur(value: float) -> str:
    return f"{value:.1f} EUR/MWh" if np.isfinite(value) else "—"


def _latest_numeric(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return float("nan")
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return _safe_float(s.iloc[-1]) if not s.empty else float("nan")


def _last_ts(df: pd.DataFrame) -> str:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return "—"
    return df.index.max().strftime("%d.%m.%Y %H:%M")


def _slice(df: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[df.index >= start].copy()


max_candidates = [entso.index.max()]
if not epex_ch.empty:
    max_candidates.append(epex_ch.index.max())
max_ts = max(max_candidates)

with st.sidebar:
    st.markdown("### Marktdaten")
    range_choice = st.radio(
        "Zeitraum",
        options=["24 h", "7 Tage", "30 Tage", "1 Jahr"],
        index=1,
    )
    granularity = st.radio(
        "Auflösung",
        options=["Stündlich", "Täglich"],
        index=0 if range_choice in {"24 h", "7 Tage"} else 1,
    )
    st.divider()
    st.markdown(freshness_badge(entso, "Systemdaten"), unsafe_allow_html=True)
    st.markdown(freshness_badge(epex_ch, "EPEX CH"), unsafe_allow_html=True)
    st.markdown(freshness_badge(hydro, "Hydro"), unsafe_allow_html=True)

lookback = {
    "24 h": pd.Timedelta(hours=24),
    "7 Tage": pd.Timedelta(days=7),
    "30 Tage": pd.Timedelta(days=30),
    "1 Jahr": pd.Timedelta(days=365),
}[range_choice]
start = max_ts - lookback

entso_sl = _slice(entso, start)
hydro_sl = _slice(hydro, start)
epex_ch_sl = _slice(epex_ch, start)
epex_de_sl = _slice(epex_de, start)

freq = "h" if granularity == "Stündlich" else "D"

load_now = _latest_numeric(entso, "load_mw")
solar_now = _latest_numeric(entso, "solar_mw")
wind_now = _latest_numeric(entso, "wind_mw")
nuclear_now = _latest_numeric(entso, "nuclear_mw")
hydro_now = sum(
    _latest_numeric(entso, c)
    for c in ["hydro_ror_mw", "hydro_reservoir_mw", "hydro_pumped_mw"]
)
renew_share = 100.0 * (solar_now + wind_now) / load_now if load_now > 0 else float("nan")
net_flow_now = _latest_numeric(entso, "cross_border_mw")
flow_label = "Nettoexport" if net_flow_now > 0 else "Nettoimport"
ch_price_now = _latest_numeric(epex_ch, "price")
de_price_col = "price_eur_mwh" if "price_eur_mwh" in epex_de.columns else "price"
de_price_now = _latest_numeric(epex_de, de_price_col)
spread_now = ch_price_now - de_price_now if np.isfinite(ch_price_now) and np.isfinite(de_price_now) else float("nan")
fill_pct = _latest_numeric(hydro, "fill_pct")

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(
    kpi_card("Aktuelle Last CH", _fmt_gw(load_now), delta=f"Stand {_last_ts(entso)}", delta_color=FMV_NAVY),
    unsafe_allow_html=True,
)
c2.markdown(
    kpi_card("Solar + Wind", f"{renew_share:.1f} %" if np.isfinite(renew_share) else "—", delta=f"{_fmt_gw(solar_now + wind_now)} aktuell", delta_color=FMV_ACCENT),
    unsafe_allow_html=True,
)
c3.markdown(
    kpi_card(flow_label, _fmt_gw(abs(net_flow_now)), delta="aggregierter CH-Nettofluss", delta_color=FMV_GREEN if net_flow_now > 0 else FMV_RED),
    unsafe_allow_html=True,
)
c4.markdown(
    kpi_card("CH Spot", _fmt_eur(ch_price_now), delta=f"Spread zu DE: {spread_now:+.1f}" if np.isfinite(spread_now) else "Spread —", delta_color=FMV_BLUE),
    unsafe_allow_html=True,
)
c5.markdown(
    kpi_card("Hydrospeicher", f"{fill_pct:.1f} %" if np.isfinite(fill_pct) else "—", delta="BFE / Schweizer Speicher", delta_color=FMV_GREEN),
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div style='background:#F7FAFF; border:1px solid #D8E5FF; border-left:6px solid {FMV_BLUE}; border-radius:12px; padding:0.85rem 1rem; margin:0.8rem 0 1.2rem 0;'>
<b>Client message:</b> FMV kann diese Daten als laufenden Marktmonitor bereitstellen:
Systemlage Schweiz, Spotpreis-Signale, Nettoimport/-export, Produktionsmix und
Hydrospeicher. Für einen GRD wird daraus ein gemeinsames Lagebild für Beschaffung,
Budget und Risiko.
</div>
    """,
    unsafe_allow_html=True,
)

tab_system, tab_price, tab_mix, tab_hydro, tab_quality = st.tabs(
    ["Systemlage", "Preise & Spread", "Produktionsmix", "Hydro", "Datenqualität"]
)

with tab_system:
    left, right = st.columns([1.25, 1.0])

    sys_cols = [c for c in ["load_mw", "solar_mw", "wind_mw", "cross_border_mw"] if c in entso_sl.columns]
    sys_df = entso_sl[sys_cols].resample(freq).mean()
    if {"load_mw", "solar_mw", "wind_mw"}.issubset(sys_df.columns):
        sys_df["residual_load_mw"] = sys_df["load_mw"] - sys_df["solar_mw"] - sys_df["wind_mw"]

    with left:
        st.subheader("Last, Residual Load und Nettofluss")
        fig = go.Figure()
        if "load_mw" in sys_df:
            fig.add_trace(go.Scatter(x=sys_df.index, y=sys_df["load_mw"], name="Last", line=dict(color=FMV_NAVY, width=2.5)))
        if "residual_load_mw" in sys_df:
            fig.add_trace(go.Scatter(x=sys_df.index, y=sys_df["residual_load_mw"], name="Residual Load", line=dict(color=FMV_RED, width=2.0)))
        if "cross_border_mw" in sys_df:
            fig.add_trace(go.Bar(x=sys_df.index, y=sys_df["cross_border_mw"], name="Nettoexport (+) / Import (-)", marker_color="rgba(15,82,204,0.28)", yaxis="y2"))
            fig.add_hline(y=0, line_color=FMV_GREY, line_width=1)
        fig.update_layout(
            height=430,
            margin=dict(l=20, r=20, t=10, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(title="Last / Residual Load (MW)", gridcolor="#EEF1F6"),
            yaxis2=dict(title="Nettofluss (MW)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=-0.18),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Aktueller System-Snapshot")
        rows = [
            ("Last", load_now),
            ("Kernkraft", nuclear_now),
            ("Hydro gesamt", hydro_now),
            ("Solar", solar_now),
            ("Wind", wind_now),
            (flow_label, net_flow_now),
        ]
        snap = pd.DataFrame(rows, columns=["Signal", "MW"])
        st.dataframe(
            snap,
            use_container_width=True,
            hide_index=True,
            column_config={"MW": st.column_config.NumberColumn(format="%.0f")},
        )

        flow_cols = [c for c in entso.columns if c.startswith("scheduled_net_export_ch_")]
        has_border_detail = bool(flow_cols) and not entso_sl[flow_cols].dropna(how="all").empty
        if not has_border_detail and "cross_border_mw" in entso.columns:
            st.info(
                "Detail pro Grenze ist im POC noch nicht angebunden. "
                "Die aggregierte Schweizer Nettoimport/-export-Sicht ist verfügbar "
                "und wird oben angezeigt."
            )

with tab_price:
    st.subheader("EPEX Spot CH vs. DE-LU")
    price_df = pd.DataFrame(index=epex_ch_sl.index)
    if not epex_ch_sl.empty and "price" in epex_ch_sl:
        price_df["CH"] = epex_ch_sl["price"]
    if not epex_de_sl.empty and de_price_col in epex_de_sl:
        price_df["DE-LU"] = epex_de_sl[de_price_col]
    price_df = price_df.resample(freq).mean()
    if {"CH", "DE-LU"}.issubset(price_df.columns):
        price_df["CH-DE Spread"] = price_df["CH"] - price_df["DE-LU"]

    fig_p = go.Figure()
    if "CH" in price_df:
        fig_p.add_trace(go.Scatter(x=price_df.index, y=price_df["CH"], name="CH Spot", line=dict(color=FMV_BLUE, width=2.3)))
    if "DE-LU" in price_df:
        fig_p.add_trace(go.Scatter(x=price_df.index, y=price_df["DE-LU"], name="DE-LU Spot", line=dict(color=FMV_ACCENT, width=2.0)))
    if "CH-DE Spread" in price_df:
        fig_p.add_trace(go.Bar(x=price_df.index, y=price_df["CH-DE Spread"], name="CH-DE Spread", marker_color="rgba(14,31,61,0.22)", yaxis="y2"))
    fig_p.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(title="Spot (EUR/MWh)", gridcolor="#EEF1F6"),
        yaxis2=dict(title="Spread", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=-0.18),
        hovermode="x unified",
    )
    st.plotly_chart(fig_p, use_container_width=True)

with tab_mix:
    st.subheader("Produktionsmix Schweiz")
    gen_cols = [
        ("nuclear_mw", "Kernkraft", "#7B4EA8"),
        ("hydro_ror_mw", "Laufwasser", FMV_BLUE),
        ("hydro_reservoir_mw", "Speicherhydro", "#2E86C1"),
        ("hydro_pumped_mw", "Pumpspeicher", "#85C1E9"),
        ("solar_mw", "Solar", FMV_ACCENT),
        ("wind_mw", "Wind", FMV_GREEN),
    ]
    available = [c for c, _, _ in gen_cols if c in entso_sl.columns]
    gen_df = entso_sl[available].resample(freq).mean()
    fig_mix = go.Figure()
    for col, label, color in gen_cols:
        if col in gen_df:
            fig_mix.add_trace(go.Scatter(
                x=gen_df.index,
                y=gen_df[col],
                name=label,
                stackgroup="gen",
                line=dict(color=color, width=0.7),
            ))
    fig_mix.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(title="MW", gridcolor="#EEF1F6"),
        legend=dict(orientation="h", y=-0.18),
        hovermode="x unified",
    )
    st.plotly_chart(fig_mix, use_container_width=True)

    totals = gen_df.mean(numeric_only=True).rename("Ø MW").reset_index()
    totals["Technologie"] = totals["index"].map({c: label for c, label, _ in gen_cols})
    totals = totals[["Technologie", "Ø MW"]].dropna()
    st.dataframe(
        totals,
        use_container_width=True,
        hide_index=True,
        column_config={"Ø MW": st.column_config.NumberColumn(format="%.0f")},
    )

with tab_hydro:
    st.subheader("Hydrospeicher Schweiz")
    if hydro.empty:
        st.info("Hydrodaten sind nicht verfügbar.")
    else:
        h_left, h_right = st.columns([1.2, 1.0])
        with h_left:
            h = hydro_sl if not hydro_sl.empty else hydro.tail(365)
            fig_h = go.Figure()
            if "fill_pct" in h:
                fig_h.add_trace(go.Scatter(
                    x=h.index,
                    y=h["fill_pct"],
                    mode="lines",
                    name="Füllstand %",
                    line=dict(color=FMV_BLUE, width=2.4),
                    fill="tozeroy",
                    fillcolor="rgba(15,82,204,0.14)",
                ))
            if "fill_deviation" in h:
                fig_h.add_trace(go.Scatter(
                    x=h.index,
                    y=50 + 10 * h["fill_deviation"],
                    mode="lines",
                    name="Abweichung Proxy",
                    line=dict(color=FMV_RED, width=1.6, dash="dash"),
                ))
            fig_h.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=10, b=20),
                plot_bgcolor="white",
                paper_bgcolor="white",
                yaxis=dict(title="Füllstand (%)", range=[0, 100], gridcolor="#EEF1F6"),
                legend=dict(orientation="h", y=-0.18),
            )
            st.plotly_chart(fig_h, use_container_width=True)
        with h_right:
            regional = [c for c in ["wallis_gwh", "graubuenden_gwh", "tessin_gwh"] if c in hydro.columns]
            if regional:
                last = hydro[regional].dropna(how="all").iloc[-1]
                labels = {"wallis_gwh": "Wallis", "graubuenden_gwh": "Graubünden", "tessin_gwh": "Tessin"}
                fig_reg = go.Figure(go.Bar(
                    x=[labels[c] for c in regional],
                    y=[float(last[c]) for c in regional],
                    marker_color=[FMV_BLUE, FMV_ACCENT, FMV_GREEN][: len(regional)],
                ))
                fig_reg.update_layout(
                    height=320,
                    margin=dict(l=20, r=20, t=10, b=20),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    yaxis=dict(title="GWh", gridcolor="#EEF1F6"),
                )
                st.plotly_chart(fig_reg, use_container_width=True)
            st.markdown(
                kpi_card(
                    "Wallis Speicher",
                    f"{_latest_numeric(hydro, 'wallis_gwh'):,.0f} GWh".replace(",", "'"),
                    delta=f"CH gesamt {_latest_numeric(hydro, 'fill_gwh'):,.0f} GWh".replace(",", "'"),
                    delta_color=FMV_GREEN,
                ),
                unsafe_allow_html=True,
            )

with tab_quality:
    st.subheader("Datenqualität und Abdeckung")
    quality = pd.DataFrame(
        [
            {"Quelle": "Systemdaten CH", "Letzter Punkt": _last_ts(entso), "Zeilen": len(entso), "Status": "OK" if not entso.empty else "Fehlt"},
            {"Quelle": "EPEX CH Spot", "Letzter Punkt": _last_ts(epex_ch), "Zeilen": len(epex_ch), "Status": "OK" if not epex_ch.empty else "Fehlt"},
            {"Quelle": "EPEX DE-LU Spot", "Letzter Punkt": _last_ts(epex_de), "Zeilen": len(epex_de), "Status": "OK" if not epex_de.empty else "Fehlt"},
            {"Quelle": "Hydrospeicher", "Letzter Punkt": _last_ts(hydro), "Zeilen": len(hydro), "Status": "OK" if not hydro.empty else "Fehlt"},
        ]
    )
    st.dataframe(quality, use_container_width=True, hide_index=True)

    border_cols = [c for c in entso.columns if c.startswith(("scheduled_net_export_ch_", "flow_net_export_ch_", "ntc_total_ch_"))]
    border_non_empty = [c for c in border_cols if c in entso.columns and entso[c].notna().any()]
    if not border_non_empty:
        st.warning(
            "POC-Hinweis: detaillierte Grenzdaten pro Grenze (CH-DE, CH-FR, "
            "CH-AT, CH-IT) sind im aktuellen Cache noch nicht befüllt. "
            "Für die Demo wird der verfügbare aggregierte Nettofluss CH genutzt."
        )

st.divider()
st.caption(
    "Quellen: energy-charts / ENTSO-E kompatible Systemdaten, EPEX Spot, "
    "BFE Hydrospeicher. Aggregation und Visualisierung durch FMV."
)

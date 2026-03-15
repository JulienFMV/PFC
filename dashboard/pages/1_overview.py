"""
Page 1 — Overview
"Vue d'ensemble marche + PFC"
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from utils import (
    COLORS, add_range_slider, export_csv_button, format_eur, format_gwh,
    format_pct, latest_run_summary, load_benchmarks, load_epex, load_hydro,
    load_model_quality, load_pfc, no_data_warning, show_freshness_sidebar,
    PROJECT_ROOT,
)

st.header("Overview")
st.caption("Vue d'ensemble du marche et de la PFC")

show_freshness_sidebar()

# ── Load data ─────────────────────────────────────────────────────────────
epex = load_epex()
hydro = load_hydro()
pfc = load_pfc()

has_epex = epex is not None and "price_eur_mwh" in epex.columns
has_hydro = hydro is not None and "fill_pct" in hydro.columns
has_pfc = pfc is not None and "price_shape" in (pfc.columns if pfc is not None else [])

# Load latest EEX forward prices
fwd_path = PROJECT_ROOT / "data" / "eex_forwards_history.parquet"
has_fwd = fwd_path.exists()
fwd_front = None
if has_fwd:
    fwd_all = pd.read_parquet(fwd_path)
    fwd_ch = fwd_all[(fwd_all["market"] == "CH") & (fwd_all["load_type"] == "BASE")]
    if not fwd_ch.empty:
        latest_date = fwd_ch["date"].max()
        fwd_front = fwd_ch[fwd_ch["date"] == latest_date]

# ── KPI Row ───────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    if has_epex:
        last_price = epex["price_eur_mwh"].iloc[-1]
        prev_price = epex["price_eur_mwh"].iloc[-96] if len(epex) > 96 else last_price
        st.metric("Spot EPEX", f"{last_price:.1f}",
                  delta=f"{last_price - prev_price:+.1f}", delta_color="inverse")
    else:
        st.metric("Spot EPEX", "—")

with k2:
    if has_epex and len(epex) > 96 * 7:
        avg_7d = epex["price_eur_mwh"].iloc[-96*7:].mean()
        avg_prev_7d = epex["price_eur_mwh"].iloc[-96*14:-96*7].mean() if len(epex) > 96*14 else avg_7d
        st.metric("Moy. 7j", f"{avg_7d:.1f}",
                  delta=f"{avg_7d - avg_prev_7d:+.1f} vs prec.")
    else:
        st.metric("Moy. 7j", "—")

with k3:
    if has_pfc:
        front_month = pfc["price_shape"].iloc[:96*30].mean()
        st.metric("PFC M+1", f"{front_month:.1f}")
    else:
        st.metric("PFC M+1", "—")

with k4:
    if fwd_front is not None:
        cal_row = fwd_front[fwd_front["product_type"] == "Cal"].sort_values("product")
        if not cal_row.empty:
            cal_price = float(cal_row.iloc[0]["price"])
            cal_name = cal_row.iloc[0]["product"]
            st.metric(f"Cal {cal_name}", f"{cal_price:.1f}")
        else:
            st.metric("EEX Cal", "—")
    else:
        st.metric("EEX Cal", "—")

with k5:
    if has_hydro:
        fill = hydro["fill_pct"].iloc[-1]
        fill_prev = hydro["fill_pct"].iloc[-2] if len(hydro) > 1 else fill
        st.metric("Hydro CH", format_pct(fill),
                  delta=f"{fill - fill_prev:+.1f}pp", delta_color="normal")
    else:
        st.metric("Hydro CH", "—")

st.divider()

# ── Main chart: EPEX spot + PFC overlay ───────────────────────────────────
st.subheader("Prix spot EPEX + PFC previsionnelle")

if has_epex:
    epex_h = epex["price_eur_mwh"].resample("h").mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epex_h.index, y=epex_h.values,
        name="EPEX Spot",
        line=dict(color=COLORS["blue"], width=1.5),
        hovertemplate="%{y:.1f} EUR/MWh<extra>Spot</extra>",
    ))

    if has_pfc:
        pfc_h = pfc["price_shape"].resample("h").mean()
        fig.add_trace(go.Scatter(
            x=pfc_h.index, y=pfc_h.values,
            name="PFC Shape",
            line=dict(color=COLORS["amber"], width=2),
            hovertemplate="%{y:.1f} EUR/MWh<extra>PFC</extra>",
        ))

        if "p10" in pfc.columns and "p90" in pfc.columns:
            p10_h = pfc["p10"].resample("h").mean()
            p90_h = pfc["p90"].resample("h").mean()
            fig.add_trace(go.Scatter(
                x=p90_h.index, y=p90_h.values,
                name="p90", line=dict(width=0), showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=p10_h.index, y=p10_h.values,
                name="IC 80%", line=dict(width=0),
                fill="tonexty", fillcolor=COLORS["band"],
                hoverinfo="skip",
            ))

    fig.update_layout(
        yaxis_title="EUR/MWh", height=450,
        legend=dict(orientation="h", y=1.02, x=0),
    )
    fig = add_range_slider(fig)
    st.plotly_chart(fig, use_container_width=True)
else:
    no_data_warning("prix EPEX")

# ── Qualite du modele ─────────────────────────────────────────────────────
mq = load_model_quality()
if mq and mq.get("status") == "ok":
    st.subheader("Qualite du modele")
    q1, q2, q3, q4 = st.columns(4)

    with q1:
        st.metric("RMSE shape", f"{mq.get('rmse_shape', 0):.1f} EUR",
                  delta=f"bias {mq.get('bias', 0):+.1f}", delta_color="inverse")
    with q2:
        ic80 = mq.get("ic80_coverage", 0)
        st.metric("IC 80%", f"{ic80:.1%}",
                  delta="OK" if 0.75 <= ic80 <= 0.85 else "hors cible",
                  delta_color="normal" if 0.75 <= ic80 <= 0.85 else "inverse")
    with q3:
        st.metric("Corr profil", f"{mq.get('corr_f', 0):.3f}")
    with q4:
        n_days = mq.get("n_days", 0)
        st.metric("Test", f"{int(n_days)}j", delta=str(mq.get("test_period", "")))

    st.divider()

# ── Stats recentes ────────────────────────────────────────────────────────
st.subheader("Statistiques 30 jours")
if has_epex:
    n_30d = min(96 * 30, len(epex))
    last_30d = epex["price_eur_mwh"].iloc[-n_30d:]

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.metric("Moyenne", f"{last_30d.mean():.1f}")
    with s2:
        st.metric("Mediane", f"{last_30d.median():.1f}")
    with s3:
        st.metric("Min", f"{last_30d.min():.1f}")
    with s4:
        st.metric("Max", f"{last_30d.max():.1f}")
    with s5:
        st.metric("Volatilite", f"{last_30d.std():.1f}")

    export_csv_button(last_30d.to_frame(), "epex_30d.csv", "Export 30j")
else:
    no_data_warning("statistiques")


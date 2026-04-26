"""Seite 4 — Ihre Transaktionen.

Refonte Phase 6 :
  - Source = cache parquet (`load_deviwa_deals_cached`) au lieu du legacy
    `load_deviwa_file` qui re-lisait l'xlsx à chaque clic.
  - Filtre Lieferjahr (2026/2027/2028/2029/Alle).
  - Tableau pivoté : **1 ligne par deal**, 12 colonnes mensuelles + totaux.
    Avant : 1 ligne par (deal × mois) → 12 lignes pour un contrat annuel
    et des lignes "None" résiduelles. Après : 1 deal = 1 ligne lisible.
  - Colonne ``Ø Preis`` ajoutée (volume-weighted), demandée pour la démo.
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
    FMV_BLUE,
    FMV_GREEN,
    FMV_GREY,
    FMV_NAVY,
    FMV_RED,
    fmt_chf,
    fmt_mwh,
    kpi_card,
    load_deviwa_deals_cached,
    render_actor_selector,
    render_cache_status,
    render_header,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FMV · Ihre Transaktionen",
    page_icon="📒",
    layout="wide",
)

render_header("Ihre Transaktionen · Deviwa-Energiepool")

with st.sidebar:
    render_cache_status()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

deals = load_deviwa_deals_cached()
if deals.empty:
    st.warning(
        "Keine Transaktionsdaten im Cache. Bitte `data/Deviwa.xlsx` ablegen "
        "und den Cache via Sidebar-Button aktualisieren."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Actor selector
# ---------------------------------------------------------------------------

actors_in_data = sorted(deals["actor"].dropna().unique().tolist())
data_dict = {a: deals[deals["actor"] == a].reset_index(drop=True) for a in actors_in_data}

choice, df = render_actor_selector(data_dict, key="trans_actor")
if df is None or df.empty:
    st.info("Keine Daten für die Auswahl.")
    st.stop()

# Year filter
years_in_df = sorted(pd.to_datetime(df["month"], errors="coerce").dt.year.dropna().astype(int).unique().tolist())
year_options = ["Alle"] + [str(y) for y in years_in_df]
year_choice = st.radio(
    "Lieferjahr",
    options=year_options,
    horizontal=True,
    index=0,
    key="trans_year",
)
if year_choice != "Alle":
    df = df[pd.to_datetime(df["month"], errors="coerce").dt.year == int(year_choice)].copy()
    if df.empty:
        st.info(f"Keine Deals für {year_choice}.")
        st.stop()


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

vol_total = float(pd.to_numeric(df["volume_sum"], errors="coerce").fillna(0.0).abs().sum())
notional_total = float(pd.to_numeric(df.get("notional_sum"), errors="coerce").fillna(0.0).abs().sum())
pnl_total = float(pd.to_numeric(df.get("pnl_sum"), errors="coerce").fillna(0.0).sum())
n_deals = int(df["deal"].nunique()) if "deal" in df.columns else len(df)

scope_lower = df["scope"].astype(str).str.lower()
intake_vol = float(pd.to_numeric(df.loc[scope_lower.str.contains("intake", na=False), "volume_sum"], errors="coerce").fillna(0.0).abs().sum())
withdraw_vol = float(pd.to_numeric(df.loc[scope_lower.str.contains("withdrawal", na=False), "volume_sum"], errors="coerce").fillna(0.0).abs().sum())
net_position = intake_vol - withdraw_vol
avg_price = (notional_total / vol_total) if vol_total > 0 else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.markdown(
    kpi_card(
        "Anzahl Deals",
        f"{n_deals}",
        delta=f"Akteur: {choice}" + (f" · Jahr {year_choice}" if year_choice != "Alle" else ""),
        delta_color=FMV_GREY,
    ),
    unsafe_allow_html=True,
)
c2.markdown(
    kpi_card(
        "Volumen gesamt",
        fmt_mwh(vol_total, 0),
        delta=f"Intake (Kauf): {fmt_mwh(intake_vol, 0)} · Withdrawal (Verkauf): {fmt_mwh(withdraw_vol, 0)}",
        delta_color=FMV_NAVY,
    ),
    unsafe_allow_html=True,
)
c3.markdown(
    kpi_card(
        "Ø Preis (volumen-gewichtet)",
        f"{avg_price:.2f} EUR/MWh" if np.isfinite(avg_price) else "—",
        delta=f"Notional gesamt: {fmt_chf(notional_total, 0)} EUR",
        delta_color=FMV_BLUE,
    ),
    unsafe_allow_html=True,
)
c4.markdown(
    kpi_card(
        "P&L gesamt",
        f"{fmt_chf(pnl_total, 0)} EUR",
        delta=f"Netto-Position: {fmt_mwh(net_position, 0)}",
        delta_color=FMV_GREEN if pnl_total >= 0 else FMV_RED,
    ),
    unsafe_allow_html=True,
)

st.markdown("")


# ---------------------------------------------------------------------------
# Monthly bar charts (volume + P&L)
# ---------------------------------------------------------------------------

df = df.copy()
df["_month_ts"] = pd.to_datetime(df["month"], errors="coerce")
df["_year"] = df["_month_ts"].dt.year

# Aggregate per (month, scope) for the volume bar chart
monthly_vol = df.groupby([df["_month_ts"], scope_lower.where(scope_lower.str.contains("intake|withdrawal", na=False), "other")]).agg(
    volume_mwh=("volume_sum", lambda v: float(pd.to_numeric(v, errors="coerce").abs().sum())),
).reset_index().rename(columns={"_month_ts": "month", "scope": "scope_kind"})

# Aggregate per month for P&L
monthly_pnl = df.groupby("_month_ts").agg(
    pnl_eur=("pnl_sum", lambda v: float(pd.to_numeric(v, errors="coerce").fillna(0).sum())),
).reset_index().rename(columns={"_month_ts": "month"})

if not monthly_vol.empty or not monthly_pnl.empty:
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Volumen pro Monat (Intake vs Withdrawal)")
        fig_vol = go.Figure()
        for kind, color, label in [
            ("intake", FMV_GREEN, "Intake (Kauf)"),
            ("withdrawal", FMV_BLUE, "Withdrawal (Verkauf)"),
        ]:
            sub = monthly_vol[monthly_vol[monthly_vol.columns[1]].astype(str).str.contains(kind)]
            if not sub.empty:
                fig_vol.add_trace(go.Bar(
                    x=sub["month"], y=sub["volume_mwh"],
                    name=label, marker_color=color, opacity=0.85,
                    hovertemplate="%{x|%b %Y}<br>" + label + ": %{y:,.0f} MWh<extra></extra>",
                ))
        fig_vol.update_layout(
            barmode="group",
            height=320, margin=dict(l=20, r=20, t=10, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="MWh", gridcolor="#EEF1F6"),
            xaxis=dict(gridcolor="#EEF1F6"),
            legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_r:
        st.subheader("P&L pro Monat")
        if not monthly_pnl.empty:
            colors = [FMV_GREEN if v >= 0 else FMV_RED for v in monthly_pnl["pnl_eur"]]
            fig_pnl = go.Figure(go.Bar(
                x=monthly_pnl["month"], y=monthly_pnl["pnl_eur"],
                marker_color=colors, opacity=0.85,
                hovertemplate="%{x|%b %Y}<br>P&L: %{y:,.0f} EUR<extra></extra>",
            ))
            fig_pnl.add_hline(y=0, line_color=FMV_GREY, line_width=1)
            fig_pnl.update_layout(
                height=320, margin=dict(l=20, r=20, t=10, b=20),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="EUR", gridcolor="#EEF1F6"),
                xaxis=dict(gridcolor="#EEF1F6"),
            )
            st.plotly_chart(fig_pnl, use_container_width=True)


# ---------------------------------------------------------------------------
# Pivot table : 1 row per deal × 12 monthly columns
# ---------------------------------------------------------------------------

st.subheader("Deal-Übersicht (1 Zeile pro Deal · 12 Monatsspalten)")

# Filter dropdowns
with st.expander("Filter", expanded=False):
    fc1, fc2 = st.columns(2)
    prod_opts = ["Alle"] + sorted(df["product"].dropna().unique().tolist()) if "product" in df.columns else ["Alle"]
    scope_opts = ["Alle", "Intake (Kauf)", "Withdrawal (Verkauf)"]
    sel_prod = fc1.selectbox("Produkt", prod_opts, key="trans_prod_filter")
    sel_scope = fc2.selectbox("Richtung", scope_opts, key="trans_scope_filter")

work = df.copy()
if sel_prod != "Alle":
    work = work[work["product"] == sel_prod]
if sel_scope == "Intake (Kauf)":
    work = work[work["scope"].astype(str).str.lower().str.contains("intake", na=False)]
elif sel_scope == "Withdrawal (Verkauf)":
    work = work[work["scope"].astype(str).str.lower().str.contains("withdrawal", na=False)]


def _build_deal_pivot(work_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per (deal, year) with 12 monthly columns Jan..Dez."""
    if work_df.empty:
        return pd.DataFrame()
    w = work_df.copy()
    w["_year"] = pd.to_datetime(w["month"], errors="coerce").dt.year
    w["_month_num"] = pd.to_datetime(w["month"], errors="coerce").dt.month
    w["_volume"] = pd.to_numeric(w["volume_sum"], errors="coerce").fillna(0.0).abs()
    w["_notional"] = pd.to_numeric(w.get("notional_sum"), errors="coerce").fillna(0.0).abs()
    w["_pnl"] = pd.to_numeric(w.get("pnl_sum"), errors="coerce").fillna(0.0)

    # Monthly volumes pivoted
    vol_wide = w.pivot_table(
        index=["deal", "_year"],
        columns="_month_num",
        values="_volume",
        aggfunc="sum",
        fill_value=0.0,
    )
    # Ensure all 12 columns present
    for m in range(1, 13):
        if m not in vol_wide.columns:
            vol_wide[m] = 0.0
    vol_wide = vol_wide.reindex(columns=range(1, 13))

    # Per-deal aggregates (metadata stays the same across months)
    aggs = w.groupby(["deal", "_year"]).agg(
        trade_date=("trade_date", "first"),
        delivery_from=("delivery_from", "first"),
        delivery_to=("delivery_to", "first"),
        product=("product", "first"),
        scope=("scope", "first"),
        counterparty=("counterparty", "first"),
        total_volume=("_volume", "sum"),
        total_notional=("_notional", "sum"),
        total_pnl=("_pnl", "sum"),
    )
    aggs["avg_price"] = np.where(
        aggs["total_volume"] > 0,
        aggs["total_notional"] / aggs["total_volume"].replace(0, np.nan),
        np.nan,
    )

    out = aggs.join(vol_wide).reset_index()
    out = out.sort_values(["_year", "trade_date", "deal"]).reset_index(drop=True)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
    rename = {i + 1: month_labels[i] for i in range(12)}
    out = out.rename(columns={
        "deal": "Deal",
        "_year": "Jahr",
        "trade_date": "Trade",
        "delivery_from": "Lieferung ab",
        "delivery_to": "Lieferung bis",
        "product": "Produkt",
        "scope": "Richtung",
        "counterparty": "Counterparty",
        "total_volume": "Volumen total (MWh)",
        "avg_price": "Ø Preis (EUR/MWh)",
        "total_notional": "Notional (EUR)",
        "total_pnl": "P&L (EUR)",
        **rename,
    })
    return out


pivot_df = _build_deal_pivot(work)

if pivot_df.empty:
    st.info("Keine Deals für die aktuelle Filter-Auswahl.")
else:
    # Format date columns for display
    for date_col in ["Trade", "Lieferung ab", "Lieferung bis"]:
        if date_col in pivot_df.columns:
            pivot_df[date_col] = pd.to_datetime(pivot_df[date_col], errors="coerce").dt.strftime("%d.%m.%Y")

    # Reorder columns for readability
    front_cols = ["Deal", "Jahr", "Richtung", "Produkt", "Trade", "Lieferung ab", "Lieferung bis"]
    month_cols = ["Jan", "Feb", "Mar", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
    summary_cols = ["Volumen total (MWh)", "Ø Preis (EUR/MWh)", "Notional (EUR)", "P&L (EUR)"]
    ordered = [c for c in front_cols if c in pivot_df.columns]
    ordered += [c for c in month_cols if c in pivot_df.columns]
    ordered += [c for c in summary_cols if c in pivot_df.columns]
    pivot_df = pivot_df[ordered]

    column_config = {}
    for m in month_cols:
        if m in pivot_df.columns:
            column_config[m] = st.column_config.NumberColumn(format="%.1f", help="Volumen MWh")
    column_config["Volumen total (MWh)"] = st.column_config.NumberColumn(format="%.1f")
    column_config["Ø Preis (EUR/MWh)"] = st.column_config.NumberColumn(format="%.2f")
    column_config["Notional (EUR)"] = st.column_config.NumberColumn(format="%.0f")
    column_config["P&L (EUR)"] = st.column_config.NumberColumn(format="%.0f")

    st.dataframe(
        pivot_df,
        use_container_width=True,
        hide_index=True,
        height=min(420, 50 + 38 * len(pivot_df)),
        column_config=column_config,
    )

    # Export
    csv = pivot_df.to_csv(index=False, sep=";").encode("utf-8")
    st.download_button(
        "📥 Tabelle als CSV exportieren",
        csv,
        file_name=f"deviwa_{choice.replace(' ', '_').lower()}_deals_pivot.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Pivot: eine Zeile = ein Deal × ein Lieferjahr · 12 Monatsspalten = "
    "Volumen pro Liefermonat (MWh, immer absolut). Ein 4-Jahres-Vertrag "
    "(z.B. Cal-2026 → Cal-2029) erscheint als 4 Zeilen, eine pro Lieferjahr. "
    "Ø Preis = volumen-gewichteter Durchschnitt (Notional / Volumen)."
)

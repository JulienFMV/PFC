"""Seite 3 — Lastprofil-Pricing gegen PFC.

Phase 10 redesign : state-of-the-art utility-pricing view aligned with
Axpo / Volue / KYOS load-pricing tooling.

Workflow :
  1. GRD lädt eine Lastkurve hoch (CSV / XLSX, viertel-stündlich oder stündlich)
     OR uses the synthetic example profile.
  2. Auto-detect timestamp + load columns, normalize to hourly MWh.
  3. Price against the current PFC.
  4. Show breakdown at three granularities : Jahr / Quartal / Monat.
  5. Hourly heatmap (24 h × 12 Monate) — the canonical "load shape" view.
  6. Cal-Y baseload comparison : "if you bought a flat baseload at Cal-Y
     forward instead of profile-pricing, you'd pay X EUR more / less".
  7. Sensitivity : ±10 % PFC shift impact on total cost.
  8. CSV export of the priced curve.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio_yearly import get_forward_year_proxy  # noqa: E402
from pricing import parse_load_file, price_load_against_pfc  # noqa: E402
from utils import (  # noqa: E402
    FMV_ACCENT,
    FMV_BLUE,
    FMV_GREEN,
    FMV_GREY,
    FMV_NAVY,
    FMV_RED,
    fmt_chf,
    fmt_eur_mwh,
    fmt_mwh,
    kpi_card,
    load_eex_forwards,
    load_pfc,
    render_cache_status,
    render_header,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FMV · Lastprofil-Pricing",
    page_icon="💡",
    layout="wide",
)

render_header("Lastprofil-Pricing — Profil gegen PFC bewerten")

with st.sidebar:
    render_cache_status()


st.markdown(
    "Laden Sie eine Lastkurve hoch (CSV / XLSX, viertel-stündlich oder stündlich, "
    "Spalten *Datum* + *Last in MW oder MWh*) und wir bepreisen sie gegen die aktuelle "
    "**FMV Price Forward Curve**. Drei Granularitäten verfügbar : Jahr, Quartal, Monat."
)


# ---------------------------------------------------------------------------
# Upload + example profile
# ---------------------------------------------------------------------------

c_upload, c_example = st.columns([2.0, 1.0])

with c_upload:
    uploaded = st.file_uploader(
        "Lastprofil hochladen (.csv / .xlsx)",
        type=["csv", "xlsx", "xls"],
        key="load_upload",
    )

with c_example:
    use_example = st.checkbox(
        "Beispielprofil verwenden",
        value=uploaded is None,
        help="Synthetisches Profil 20 GWh/Jahr — typischer Industrie-GRD Haut-Valais.",
    )


pfc = load_pfc()
forwards = load_eex_forwards()

if pfc.empty:
    st.error("PFC nicht verfügbar. Bitte PFC-Lauf ausführen oder Cache aktualisieren.")
    st.stop()


# ---------------------------------------------------------------------------
# Profile preparation
# ---------------------------------------------------------------------------

load_hourly: pd.Series | None = None
meta: dict = {}

if uploaded is not None:
    try:
        load_hourly, meta = parse_load_file(uploaded)
        st.success(
            f"📁 **{uploaded.name}** · {meta['rows']:,} Zeilen · "
            f"{meta['start'].strftime('%d.%m.%Y')} → "
            f"{meta['end'].strftime('%d.%m.%Y')} · "
            f"erkannte Frequenz : **{meta.get('inferred_freq', '?')}**".replace(",", "'")
        )
    except Exception as exc:
        st.error(f"Fehler beim Parsen : {exc}")
        st.stop()
elif use_example:
    pfc_start = pfc.index.min()
    idx = pd.date_range(start=pfc_start, periods=8760, freq="h", tz="Europe/Zurich")
    base_mw = 2.0
    hour_shape = np.array([
        0.65, 0.60, 0.58, 0.58, 0.60, 0.75, 0.95, 1.15, 1.25, 1.30,
        1.30, 1.28, 1.25, 1.22, 1.22, 1.22, 1.25, 1.20, 1.10, 1.00,
        0.90, 0.80, 0.72, 0.68,
    ])
    values = np.array([
        base_mw * hour_shape[t.hour] * (0.75 if t.dayofweek >= 5 else 1.0)
        for t in idx
    ])
    seasonal = 1.0 + 0.18 * np.cos(2 * np.pi * (idx.dayofyear - 15) / 365)
    values = values * seasonal
    load_hourly = pd.Series(values, index=idx, name="load_mwh")
    meta = {
        "rows": len(load_hourly),
        "start": load_hourly.index.min(),
        "end": load_hourly.index.max(),
        "annual_mwh": float(load_hourly.sum()),
        "unit": "MWh (synthetisch)",
        "inferred_freq": "h",
    }
    st.info(
        "📊 Beispielprofil aktiv (20 GWh / Jahr · Tagesgang Industrie · "
        "Wochenende −25 % · Saisonalität ±18 %)."
    )
else:
    st.info("Bitte laden Sie ein Lastprofil hoch oder nutzen Sie das Beispielprofil.")
    st.stop()


# ---------------------------------------------------------------------------
# Pricing (full-period)
# ---------------------------------------------------------------------------

result = price_load_against_pfc(load_hourly, pfc)


# ---------------------------------------------------------------------------
# KPI hero (5 cards)
# ---------------------------------------------------------------------------

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(
    kpi_card(
        "Jahresvolumen",
        fmt_mwh(result["total_mwh"], decimals=0),
        delta=f"{result['coverage_hours']:,} Stunden".replace(",", "'"),
        delta_color=FMV_GREY,
    ),
    unsafe_allow_html=True,
)
c2.markdown(
    kpi_card(
        "Profilpreis",
        fmt_eur_mwh(result["profile_price_eur_mwh"]),
        delta=f"Baseload Ø : {fmt_eur_mwh(result['baseload_price_eur_mwh'])}",
        delta_color=FMV_BLUE,
    ),
    unsafe_allow_html=True,
)
c3.markdown(
    kpi_card(
        "Profilaufschlag",
        fmt_eur_mwh(result["profile_vs_baseload_eur_mwh"]),
        delta="ggü. Baseload",
        delta_color=FMV_RED if result["profile_vs_baseload_eur_mwh"] > 0 else FMV_GREEN,
    ),
    unsafe_allow_html=True,
)
c4.markdown(
    kpi_card(
        "Gesamtkosten",
        f"{fmt_chf(result['cost_total_eur'], decimals=0)} EUR",
        delta=f"Ø {fmt_eur_mwh(result['profile_price_eur_mwh'])}",
        delta_color=FMV_NAVY,
    ),
    unsafe_allow_html=True,
)
c5.markdown(
    kpi_card(
        "Peak-Anteil",
        f"{result['peak_share_pct']:.1f} %",
        delta=(
            f"Peak {fmt_eur_mwh(result['peak_price_eur_mwh'])}"
            f" / Off {fmt_eur_mwh(result['offpeak_price_eur_mwh'])}"
        ),
        delta_color=FMV_GREY,
    ),
    unsafe_allow_html=True,
)

if result.get("projection_used"):
    st.warning(
        "ℹ️ Das Lastprofil liegt nicht im PFC-Zeitraum. Das typische Wochenprofil "
        "wurde auf das erste PFC-Jahr projiziert und auf das geschätzte Jahresvolumen "
        "skaliert. Verwenden Sie ein zukünftiges Profil für eine direkte Bepreisung."
    )


# ---------------------------------------------------------------------------
# Granularité selector + breakdown
# ---------------------------------------------------------------------------

st.markdown("")
st.subheader("Aufschlüsselung nach Periode")

granularity = st.radio(
    "Granularität",
    options=["Jahr", "Quartal", "Monat"],
    horizontal=True,
    index=2,
    key="lastprofil_gran",
)

# Build a priced hourly DataFrame from scratch (not just monthly aggregate),
# so we can re-aggregate at any level.
pfc_h_full = pfc["price_shape"].resample("h").mean()
pfc_h_full.index = pfc_h_full.index.tz_convert("Europe/Zurich")

# Use the projection if necessary by aligning to the price_load_against_pfc convention
common_idx = load_hourly.index.intersection(pfc_h_full.index)
if len(common_idx) < 24:
    st.info("Profil und PFC haben keine direkte Zeitüberlappung — Aggregation nutzt das projizierte Profil.")
    # The result["monthly"] frame already used projection — re-derive the full hourly
    # series from it would require reproducing the projection here. We keep the simpler
    # path : aggregate at month level only when projection_used.
    if granularity != "Monat":
        st.warning("Für projizierte Profile zeigen wir nur Monats-Granularität.")
        granularity = "Monat"

if not common_idx.empty:
    priced = pd.DataFrame({
        "load_mwh": load_hourly.reindex(common_idx).fillna(0.0),
        "pfc_eur_mwh": pfc_h_full.reindex(common_idx),
    })
    priced["cost_eur"] = priced["load_mwh"] * priced["pfc_eur_mwh"]

    if granularity == "Jahr":
        period = priced.index.to_series().dt.year
        period_label = "Jahr"
    elif granularity == "Quartal":
        period = priced.index.to_series().dt.to_period("Q").astype(str)
        period_label = "Quartal"
    else:
        period = priced.index.to_series().dt.to_period("M").dt.to_timestamp()
        period_label = "Monat"

    agg = priced.groupby(period).agg(
        load_mwh=("load_mwh", "sum"),
        cost_eur=("cost_eur", "sum"),
        avg_pfc=("pfc_eur_mwh", "mean"),
    )
    agg["profile_price"] = agg["cost_eur"] / agg["load_mwh"].replace(0, np.nan)
    agg["premium_eur_mwh"] = agg["profile_price"] - agg["avg_pfc"]

    col_chart, col_table = st.columns([1.4, 1.0])

    with col_chart:
        # X labels
        if granularity == "Monat":
            x_labels = pd.to_datetime(agg.index).strftime("%b %Y")
        else:
            x_labels = [str(x) for x in agg.index]

        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(
            x=x_labels, y=agg["load_mwh"],
            name="Volumen (MWh)", marker_color=FMV_BLUE, opacity=0.85,
            yaxis="y1",
            hovertemplate="%{x}<br>Volumen : %{y:,.0f} MWh<extra></extra>",
        ))
        fig_p.add_trace(go.Scatter(
            x=x_labels, y=agg["profile_price"],
            name="Profilpreis (EUR/MWh)",
            mode="lines+markers",
            line=dict(color=FMV_ACCENT, width=2.5),
            marker=dict(size=7, symbol="diamond"),
            yaxis="y2",
            hovertemplate="%{x}<br>Ø Preis : %{y:.2f} EUR/MWh<extra></extra>",
        ))
        fig_p.add_trace(go.Scatter(
            x=x_labels, y=agg["avg_pfc"],
            name="Baseload Ø PFC (EUR/MWh)",
            mode="lines",
            line=dict(color=FMV_NAVY, width=1.8, dash="dash"),
            yaxis="y2",
            hovertemplate="%{x}<br>Ø PFC : %{y:.2f} EUR/MWh<extra></extra>",
        ))
        fig_p.update_layout(
            height=420, margin=dict(l=20, r=20, t=10, b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#EEF1F6", title=period_label),
            yaxis=dict(title="Volumen (MWh)", gridcolor="#EEF1F6", side="left"),
            yaxis2=dict(title="Preis (EUR/MWh)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=-0.2),
            hovermode="x unified",
        )
        st.plotly_chart(fig_p, use_container_width=True)

    with col_table:
        if granularity == "Monat":
            agg["_idx"] = pd.to_datetime(agg.index).strftime("%b %Y")
        else:
            agg["_idx"] = [str(x) for x in agg.index]
        show_df = agg.reset_index(drop=True)[
            ["_idx", "load_mwh", "profile_price", "avg_pfc", "premium_eur_mwh", "cost_eur"]
        ].rename(columns={
            "_idx": period_label,
            "load_mwh": "Volumen (MWh)",
            "profile_price": "Profilpreis (EUR/MWh)",
            "avg_pfc": "Baseload Ø (EUR/MWh)",
            "premium_eur_mwh": "Aufschlag (EUR/MWh)",
            "cost_eur": "Kosten (EUR)",
        })
        st.dataframe(
            show_df, use_container_width=True, hide_index=True,
            height=420,
            column_config={
                "Volumen (MWh)": st.column_config.NumberColumn(format="%.0f"),
                "Profilpreis (EUR/MWh)": st.column_config.NumberColumn(format="%.2f"),
                "Baseload Ø (EUR/MWh)": st.column_config.NumberColumn(format="%.2f"),
                "Aufschlag (EUR/MWh)": st.column_config.NumberColumn(format="%.2f"),
                "Kosten (EUR)": st.column_config.NumberColumn(format="%.0f"),
            },
        )


# ---------------------------------------------------------------------------
# Hourly heatmap : 12 months × 24 hours (Axpo / Volue canonical view)
# ---------------------------------------------------------------------------

st.markdown("")
st.subheader("Lastform · Stunde × Monat (durchschnittliche Leistung)")
st.caption(
    "Die kanonische **Load-Shape**-Sicht (Axpo / Volue) : durchschnittliche Leistung "
    "in MW pro Stunde des Tages und Monat des Jahres. Hot zones zeigen wo das Profil "
    "Spitze macht, kalte Zonen wo Bedarf niedrig ist."
)

hm = pd.DataFrame({"load_mw": load_hourly.values, "ts": load_hourly.index})
hm["month"] = pd.to_datetime(hm["ts"]).dt.month
hm["hour"] = pd.to_datetime(hm["ts"]).dt.hour
hm_pivot = hm.pivot_table(index="month", columns="hour", values="load_mw", aggfunc="mean")
hm_pivot = hm_pivot.reindex(index=range(1, 13), columns=range(0, 24))

month_labels = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]

fig_hm = go.Figure(go.Heatmap(
    z=hm_pivot.values,
    x=[f"{h:02d}h" for h in hm_pivot.columns],
    y=[month_labels[m - 1] for m in hm_pivot.index],
    colorscale="YlOrRd",
    hovertemplate="%{y} · %{x}<br>Ø Leistung : %{z:.2f} MW<extra></extra>",
    colorbar=dict(title="MW"),
))
fig_hm.update_layout(
    height=420, margin=dict(l=20, r=20, t=10, b=20),
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(title="Stunde des Tages", tickfont=dict(size=10)),
    yaxis=dict(title="Monat", autorange="reversed"),
)
st.plotly_chart(fig_hm, use_container_width=True)


# ---------------------------------------------------------------------------
# Cal-Y baseload comparison : "if you bought flat at Cal-Y forward"
# ---------------------------------------------------------------------------

st.markdown("")
st.subheader("Vergleich gegen Cal-Y Baseload-Strip")
st.caption(
    "Was wäre der Preis, wenn Sie statt einer Profil-Bepreisung einfach eine "
    "**flache Baseload-Tranche** zum aktuellen Cal-Y-Forward kaufen würden ? "
    "Die Differenz ist Ihre **Profile-Premium** — der Aufpreis, den Sie zahlen "
    "um Ihre stündliche Last 1:1 zu decken."
)

# Identify the principal year of the load
load_year = int(pd.Timestamp(load_hourly.index.min()).year)
fwd_y_price = get_forward_year_proxy(forwards, load_year)

if np.isfinite(fwd_y_price):
    flat_baseload_cost = result["total_mwh"] * fwd_y_price
    profile_cost = result["cost_total_eur"]
    delta_eur = profile_cost - flat_baseload_cost
    premium_pct = (delta_eur / flat_baseload_cost * 100.0) if flat_baseload_cost > 0 else 0.0

    delta_color = FMV_RED if delta_eur > 0 else FMV_GREEN
    sign = "+" if delta_eur >= 0 else ""

    cb1, cb2, cb3 = st.columns(3)
    cb1.markdown(
        kpi_card(
            f"Profile-Pricing (Cal-{load_year})",
            f"{fmt_chf(profile_cost, 0)} EUR",
            delta=f"@ {fmt_eur_mwh(result['profile_price_eur_mwh'])}",
            delta_color=FMV_NAVY,
        ),
        unsafe_allow_html=True,
    )
    cb2.markdown(
        kpi_card(
            f"Flat Baseload (Cal-{load_year} Fwd)",
            f"{fmt_chf(flat_baseload_cost, 0)} EUR",
            delta=f"@ {fwd_y_price:.2f} EUR/MWh × {fmt_mwh(result['total_mwh'], 0)}",
            delta_color=FMV_BLUE,
        ),
        unsafe_allow_html=True,
    )
    cb3.markdown(
        kpi_card(
            "Profile-Premium",
            f"{sign}{fmt_chf(delta_eur, 0)} EUR",
            delta=f"{sign}{premium_pct:.2f} %",
            delta_color=delta_color,
        ),
        unsafe_allow_html=True,
    )

    if delta_eur > 0:
        st.info(
            f"💡 **Lecture commerciale** : Ihr Profil ist {premium_pct:.1f} % teurer "
            f"als ein flacher Baseload-Strip — typisch für Industrie- / Heizlasten. "
            f"Wenn Sie ein flacheres Profil hätten (z. B. mit PV-Eigenverbrauch), "
            f"könnten Sie sich diesem Baseload-Preis nähern."
        )
    else:
        st.success(
            f"✅ Ihr Profil ist günstiger als ein flacher Baseload-Strip — Ihre Last "
            f"liegt überdurchschnittlich in günstigen Stunden (z. B. PV-Mittag)."
        )
else:
    st.info(
        f"Cal-{load_year} Forward nicht verfügbar (Cal retired oder zu weit in der Zukunft)."
    )


# ---------------------------------------------------------------------------
# Forward sensitivity (±10%)
# ---------------------------------------------------------------------------

st.markdown("")
st.subheader("Sensitivität · was bei einer Forward-Bewegung passiert")

shift_pcts = [-15, -10, -5, 0, 5, 10, 15]
sens_rows = []
for s in shift_pcts:
    new_cost = result["cost_total_eur"] * (1.0 + s / 100.0)
    delta = new_cost - result["cost_total_eur"]
    sens_rows.append({
        "Forward-Shift": f"{s:+d} %",
        "Gesamtkosten (EUR)": new_cost,
        "Differenz vs. heute (EUR)": delta,
    })
sens_df = pd.DataFrame(sens_rows)

fig_sens = go.Figure()
fig_sens.add_trace(go.Bar(
    x=sens_df["Forward-Shift"],
    y=sens_df["Differenz vs. heute (EUR)"],
    marker_color=[FMV_GREEN if v <= 0 else FMV_RED for v in sens_df["Differenz vs. heute (EUR)"]],
    text=[f"{fmt_chf(v, 0)} EUR" for v in sens_df["Differenz vs. heute (EUR)"]],
    textposition="outside",
    hovertemplate="%{x}<br>Δ Kosten : %{y:,.0f} EUR<extra></extra>",
))
fig_sens.add_hline(y=0, line_color=FMV_GREY, line_width=1)
fig_sens.update_layout(
    height=300, margin=dict(l=20, r=20, t=10, b=20),
    plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(title="Δ Gesamtkosten (EUR)", gridcolor="#EEF1F6"),
    xaxis=dict(gridcolor="#EEF1F6"),
)
st.plotly_chart(fig_sens, use_container_width=True)


# ---------------------------------------------------------------------------
# Export priced hourly curve
# ---------------------------------------------------------------------------

st.markdown("")
if not common_idx.empty:
    st.subheader("Export")
    export_df = priced.reset_index()
    export_df.columns = ["timestamp"] + list(export_df.columns[1:])
    export_df["timestamp"] = pd.to_datetime(export_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S%z")
    csv_buf = io.StringIO()
    export_df.to_csv(csv_buf, sep=";", index=False)
    st.download_button(
        "📥 Bepreiste Lastkurve als CSV exportieren",
        csv_buf.getvalue(),
        file_name=f"lastprofil_priced_{load_year}.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Methodik : jede Stunde der Last wird mit dem entsprechenden PFC-Stundenpreis "
    "multipliziert. Peak = 08:00–20:00 Mo–Fr (Schweizer Konvention). "
    "**Profilaufschlag** (KPI-Karte) = profil-gewichteter PFC-Preis − einfacher PFC-"
    "Stundendurchschnitt (reine Profilform). **Profile-Premium** (Cal-Y-Vergleich) "
    "= Profil-Pricing − flacher Cal-Y-Forward-Strip ; enthält damit zusätzlich die "
    "Differenz zwischen PFC-Mittel und Forward-Settlement und ist daher meist grösser "
    "als der reine Profilaufschlag. "
    "Cal-Y Forward = Y01_{Y}_BASE Settlement vom letzten Handelstag (oder Q-Strip-"
    "Average für das laufende Lieferjahr)."
)

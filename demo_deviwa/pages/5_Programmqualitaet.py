"""Seite 5 – Programmqualität: Plan (Programm) vs. Realität (Ist) vs. Budget.

Trader-grade Übersicht mit Kalender-Heatmap, Stunde×Wochentag-Bias-Heatmap,
Top-10 Tage, Verteilung, Pool-Vergleich und kumulierter Ausgleichskosten-
schätzung.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deviwa_parser import extract_programme_data
from utils import (
    FMV_ACCENT,
    FMV_BLUE,
    FMV_GREEN,
    FMV_GREY,
    FMV_NAVY,
    FMV_RED,
    find_deviwa_file,
    fmt_chf,
    fmt_mwh,
    kpi_card,
    render_header,
)

st.set_page_config(
    page_title="FMV · Programmqualität",
    page_icon="🎚️",
    layout="wide",
)

render_header("Programmqualität · Plan gegen Realität")

st.markdown(
    "Wie genau entspricht das **Programm** der **Realität**? "
    "Abweichungen erzeugen Ausgleichsenergiekosten — wir zeigen wo, wann und wieviel."
)

WEEKDAY_LABELS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
ASSUMED_IMBALANCE_PRICE = 150.0  # EUR/MWh (Mittelwert CH 2024-2025)

# ---------------------------------------------------------------------------
# Daten laden
# ---------------------------------------------------------------------------
path = find_deviwa_file()
if path is None:
    st.warning("Keine Deviwa-Datei in `/data` gefunden.")
    st.stop()

prog_data = extract_programme_data(path)
prog_data = {
    k: v for k, v in prog_data.items()
    if v is not None and not v.empty
    and "programme_mw" in v.columns and "actual_mw" in v.columns
    and v["programme_mw"].notna().any() and v["actual_mw"].notna().any()
}

if not prog_data:
    st.info(
        "Keine ProgrammeReal-Sheets gefunden. Erwartet werden Spalten "
        "`<Akteur>_Programm`, `<Akteur>_Real` und optional `<Akteur>_Budget`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Akteur-Auswahl + Zeitraum
# ---------------------------------------------------------------------------
actors = list(prog_data.keys())
options = (["Gesamter Pool"] if len(actors) > 1 else []) + actors
choice = st.radio("Akteur auswählen", options=options, horizontal=True, key="prog_actor")

if choice == "Gesamter Pool":
    frames = [v.assign(_actor=k) for k, v in prog_data.items()]
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    title_suffix = "Gesamter Pool"
else:
    df = prog_data[choice].copy()
    df["_actor"] = choice
    title_suffix = choice

max_ts = df["timestamp"].max()
min_ts = df["timestamp"].min()
period = st.radio(
    "Zeitraum",
    options=["Letzte 30 Tage", "Letzte 90 Tage", "Letzte 365 Tage", "Gesamter Zeitraum"],
    horizontal=True, index=2, key="prog_period",
)
mapping = {
    "Letzte 30 Tage": pd.Timedelta(days=30),
    "Letzte 90 Tage": pd.Timedelta(days=90),
    "Letzte 365 Tage": pd.Timedelta(days=365),
}
start = max_ts - mapping[period] if period in mapping else min_ts
df = df[df["timestamp"] >= start].copy()
df = df.dropna(subset=["programme_mw", "actual_mw"])

if df.empty:
    st.info("Keine Datenpunkte im gewählten Zeitraum.")
    st.stop()

# ---------------------------------------------------------------------------
# Kennzahlen
# ---------------------------------------------------------------------------
err = df["actual_mw"] - df["programme_mw"]
abs_err = err.abs()
mae = float(abs_err.mean())
rmse = float(np.sqrt((err ** 2).mean()))
bias = float(err.mean())
ape = abs_err / df["programme_mw"].abs().clip(lower=1e-3) * 100
accuracy = float(max(0.0, 100.0 - ape.mean()))
total_imbalance_mwh = float(abs_err.sum())  # |MW| × 1h = MWh
imbalance_cost = total_imbalance_mwh * ASSUMED_IMBALANCE_PRICE

# Budget-Vergleich (falls vorhanden)
has_budget = df["budget_mw"].notna().any()
budget_mae = float((df["actual_mw"] - df["budget_mw"]).abs().mean()) if has_budget else float("nan")
budget_bias = float((df["actual_mw"] - df["budget_mw"]).mean()) if has_budget else float("nan")

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(
    kpi_card(
        "Genauigkeit",
        f"{accuracy:.1f} %",
        delta=f"{len(df):,} Stunden".replace(",", "'"),
        delta_color=(FMV_GREEN if accuracy >= 90 else FMV_ACCENT if accuracy >= 80 else FMV_RED),
    ),
    unsafe_allow_html=True,
)
c2.markdown(
    kpi_card(
        "MAE",
        f"{mae:.3f} MW",
        delta=f"RMSE: {rmse:.3f} MW",
        delta_color=FMV_NAVY,
    ),
    unsafe_allow_html=True,
)
c3.markdown(
    kpi_card(
        "Bias (Real − Plan)",
        f"{bias:+.3f} MW",
        delta="positiv = mehr verbraucht als geplant",
        delta_color=(FMV_BLUE if abs(bias) < mae * 0.3 else FMV_RED),
    ),
    unsafe_allow_html=True,
)
c4.markdown(
    kpi_card(
        "Summe |Abweichung|",
        fmt_mwh(total_imbalance_mwh, 0),
        delta=period.lower(),
        delta_color=FMV_GREY,
    ),
    unsafe_allow_html=True,
)
c5.markdown(
    kpi_card(
        "Geschätzte Ausgleichskosten",
        f"{fmt_chf(imbalance_cost, 0)} EUR",
        delta=f"bei Ø {ASSUMED_IMBALANCE_PRICE:.0f} EUR/MWh",
        delta_color=FMV_RED,
    ),
    unsafe_allow_html=True,
)

if has_budget:
    st.caption(
        f"📅 Budget-Vergleich: MAE Budget ↔ Real = **{budget_mae:.3f} MW** "
        f"(Bias {budget_bias:+.3f} MW). MAE Programm ↔ Real = **{mae:.3f} MW**."
    )

st.markdown("")

# ---------------------------------------------------------------------------
# Pool-Vergleich (nur wenn alle Akteure gewählt)
# ---------------------------------------------------------------------------
if choice == "Gesamter Pool" and len(actors) > 1:
    st.subheader("Akteur-Vergleich")
    cmp_rows = []
    for actor in actors:
        sub = prog_data[actor]
        sub = sub[sub["timestamp"] >= start].dropna(subset=["programme_mw", "actual_mw"])
        if sub.empty:
            continue
        e = sub["actual_mw"] - sub["programme_mw"]
        ape_a = e.abs() / sub["programme_mw"].abs().clip(lower=1e-3) * 100
        cmp_rows.append({
            "Akteur": actor,
            "Genauigkeit (%)": max(0.0, 100.0 - float(ape_a.mean())),
            "MAE (MW)": float(e.abs().mean()),
            "Bias (MW)": float(e.mean()),
            "Stunden": int(len(sub)),
            "Σ |Δ| (MWh)": float(e.abs().sum()),
            "Geschätzte Kosten (EUR)": float(e.abs().sum()) * ASSUMED_IMBALANCE_PRICE,
        })
    cmp_df = pd.DataFrame(cmp_rows).sort_values("Genauigkeit (%)", ascending=False)
    if not cmp_df.empty:
        cols_actor = st.columns(len(cmp_df))
        for i, (_, row) in enumerate(cmp_df.iterrows()):
            color = (FMV_GREEN if row["Genauigkeit (%)"] >= 90 else
                     FMV_ACCENT if row["Genauigkeit (%)"] >= 80 else FMV_RED)
            cols_actor[i].markdown(
                kpi_card(
                    row["Akteur"],
                    f"{row['Genauigkeit (%)']:.1f} %",
                    delta=f"MAE {row['MAE (MW)']:.3f} MW · Kosten {fmt_chf(row['Geschätzte Kosten (EUR)'],0)} EUR",
                    delta_color=color,
                ),
                unsafe_allow_html=True,
            )
        st.markdown("")

# ---------------------------------------------------------------------------
# HERO 1: Kalender-Heatmap MAE pro Tag (Wochentag × KW)
# ---------------------------------------------------------------------------
st.subheader("Kalender-Heatmap · tägliche Abweichung")

daily = df.copy()
daily["abs_err"] = (daily["actual_mw"] - daily["programme_mw"]).abs()
daily_agg = daily.groupby("date").agg(
    mae=("abs_err", "mean"),
    n=("abs_err", "count"),
).reset_index()
daily_agg["date"] = pd.to_datetime(daily_agg["date"])
daily_agg["dow"] = daily_agg["date"].dt.dayofweek
daily_agg["iso_week"] = daily_agg["date"].dt.isocalendar().week.astype(int)
daily_agg["year"] = daily_agg["date"].dt.year
daily_agg["yw"] = daily_agg["year"].astype(str) + "-W" + daily_agg["iso_week"].astype(str).str.zfill(2)

# Pivot
pivot = daily_agg.pivot_table(index="dow", columns="yw", values="mae")
pivot = pivot.reindex(range(7))
hover_dates = daily_agg.pivot_table(
    index="dow", columns="yw", values="date", aggfunc="first"
).reindex(range(7))

hover_text = np.where(
    hover_dates.notna(),
    pd.DataFrame(hover_dates).map(
        lambda d: d.strftime("%d.%m.%Y") if pd.notna(d) else ""
    ).values,
    "",
)
mae_text = np.where(
    pivot.notna(),
    np.array([[f"{v:.3f} MW" if pd.notna(v) else "" for v in row] for row in pivot.values]),
    "",
)
custom = np.dstack([hover_text, mae_text])

fig_cal = go.Figure(go.Heatmap(
    z=pivot.values,
    x=pivot.columns,
    y=[WEEKDAY_LABELS[i] for i in pivot.index],
    colorscale=[
        [0.0, "#E6F4EA"],
        [0.3, "#FFF4D6"],
        [0.7, "#FAB1A0"],
        [1.0, "#C0392B"],
    ],
    hoverongaps=False,
    customdata=custom,
    hovertemplate="<b>%{customdata[0]}</b><br>%{y}<br>MAE: %{customdata[1]}<extra></extra>",
    colorbar=dict(title="MAE<br>(MW)", thickness=15),
))
fig_cal.update_layout(
    height=260, margin=dict(l=20, r=20, t=10, b=20),
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(title="ISO-Kalenderwoche", tickangle=-90, tickfont=dict(size=9)),
    yaxis=dict(autorange="reversed"),
)
st.plotly_chart(fig_cal, use_container_width=True)

# ---------------------------------------------------------------------------
# Heatmap Stunde × Wochentag (Bias) + Verteilung
# ---------------------------------------------------------------------------
col_hw, col_dist = st.columns([1.3, 1])

with col_hw:
    st.subheader("Bias-Heatmap · Stunde × Wochentag")
    df["abs_err"] = (df["actual_mw"] - df["programme_mw"]).abs()
    df["err"] = df["actual_mw"] - df["programme_mw"]
    hw_pivot = df.pivot_table(index="dow", columns="hour", values="err", aggfunc="mean").reindex(range(7))
    hw_count = df.pivot_table(index="dow", columns="hour", values="err", aggfunc="count").reindex(range(7))

    abs_max = float(hw_pivot.abs().max().max() or 1.0)
    fig_hw = go.Figure(go.Heatmap(
        z=hw_pivot.values,
        x=[f"{h:02d}h" for h in hw_pivot.columns],
        y=[WEEKDAY_LABELS[i] for i in hw_pivot.index],
        colorscale=[
            [0.0, "#0F52CC"], [0.5, "#FFFFFF"], [1.0, "#C0392B"],
        ],
        zmid=0,
        zmin=-abs_max, zmax=abs_max,
        hovertemplate="%{y} %{x}<br>Bias: %{z:.3f} MW<extra></extra>",
        colorbar=dict(title="Bias<br>(MW)", thickness=15),
    ))
    fig_hw.update_layout(
        height=340, margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Stunde", tickfont=dict(size=9)),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_hw, use_container_width=True)
    st.caption("Blau = systematisch zu viel geplant · Rot = systematisch zu wenig geplant.")

with col_dist:
    st.subheader("Abweichungsverteilung")
    err_arr = (df["actual_mw"] - df["programme_mw"]).values
    fig_h = go.Figure()
    fig_h.add_trace(go.Histogram(
        x=err_arr, nbinsx=50,
        marker_color=FMV_BLUE, opacity=0.8,
        name="Verteilung",
    ))
    fig_h.add_vline(x=0, line_width=2, line_color=FMV_NAVY)
    fig_h.add_vline(
        x=float(np.mean(err_arr)),
        line_width=2, line_dash="dash", line_color=FMV_RED,
        annotation_text=f"Bias {np.mean(err_arr):+.3f} MW",
        annotation_position="top right",
    )
    fig_h.update_layout(
        height=340, margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Real − Programm (MW)", gridcolor="#EEF1F6"),
        yaxis=dict(title="Häufigkeit", gridcolor="#EEF1F6"),
        showlegend=False,
    )
    st.plotly_chart(fig_h, use_container_width=True)

# ---------------------------------------------------------------------------
# Triple-Time-Series: Programme vs Real vs Budget
# ---------------------------------------------------------------------------
st.subheader("Zeitreihe · Programm, Real und Budget")

# Aggregate by hour (already hourly) — or by day for longer ranges
window_days = (df["timestamp"].max() - df["timestamp"].min()).days
agg_freq = "h" if window_days <= 35 else "D"
agg_label = "Stunde" if agg_freq == "h" else "Tag"
ts_df = df.set_index("timestamp")[["programme_mw", "actual_mw", "budget_mw"]].resample(agg_freq).mean()

show_budget = st.checkbox("Budget anzeigen", value=has_budget, disabled=not has_budget)

fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(
    x=ts_df.index, y=ts_df["programme_mw"],
    mode="lines", name="Programm",
    line=dict(color=FMV_BLUE, width=2),
))
fig_ts.add_trace(go.Scatter(
    x=ts_df.index, y=ts_df["actual_mw"],
    mode="lines", name="Real",
    line=dict(color=FMV_NAVY, width=2),
))
if show_budget and has_budget:
    fig_ts.add_trace(go.Scatter(
        x=ts_df.index, y=ts_df["budget_mw"],
        mode="lines", name="Budget",
        line=dict(color=FMV_ACCENT, width=2, dash="dash"),
    ))
fig_ts.update_layout(
    height=380, margin=dict(l=20, r=20, t=10, b=20),
    plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(title=f"MW (Mittelwert je {agg_label})", gridcolor="#EEF1F6"),
    xaxis=dict(gridcolor="#EEF1F6"),
    legend=dict(orientation="h", y=-0.15),
    hovermode="x unified",
)
st.plotly_chart(fig_ts, use_container_width=True)

# ---------------------------------------------------------------------------
# Kumulierte Ausgleichskosten + Top-10 schlechteste Tage
# ---------------------------------------------------------------------------
col_cum, col_worst = st.columns([1.2, 1])

with col_cum:
    st.subheader("Kumulierte Ausgleichskosten")
    cost_h = (df["actual_mw"] - df["programme_mw"]).abs() * ASSUMED_IMBALANCE_PRICE
    cost_h.index = df["timestamp"]
    cost_d = cost_h.resample("D").sum()
    cum = cost_d.cumsum()

    fig_cum = make_subplots(specs=[[{"secondary_y": True}]])
    fig_cum.add_trace(
        go.Bar(x=cost_d.index, y=cost_d.values, name="Tageskosten",
               marker_color=FMV_RED, opacity=0.45),
        secondary_y=False,
    )
    fig_cum.add_trace(
        go.Scatter(x=cum.index, y=cum.values, name="Kumuliert",
                   mode="lines", line=dict(color=FMV_NAVY, width=3),
                   fill="tozeroy", fillcolor="rgba(14,31,61,0.10)"),
        secondary_y=True,
    )
    fig_cum.update_layout(
        height=360, margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.18),
        hovermode="x unified",
    )
    fig_cum.update_yaxes(title_text="Tageskosten (EUR)", gridcolor="#EEF1F6", secondary_y=False)
    fig_cum.update_yaxes(title_text="Kumuliert (EUR)", showgrid=False, secondary_y=True)
    st.plotly_chart(fig_cum, use_container_width=True)

with col_worst:
    st.subheader("Top 10 Problem-Tage")
    daily_full = df.copy()
    daily_full["abs_err"] = (daily_full["actual_mw"] - daily_full["programme_mw"]).abs()
    daily_full["signed_err"] = daily_full["actual_mw"] - daily_full["programme_mw"]
    g = daily_full.groupby("date").agg(
        mae=("abs_err", "mean"),
        max_err=("abs_err", "max"),
        bias=("signed_err", "mean"),
    ).reset_index()
    g = g.sort_values("mae", ascending=False).head(10)
    g["Datum"] = pd.to_datetime(g["date"]).dt.strftime("%d.%m.%Y")
    g["MAE (MW)"] = g["mae"].round(3)
    g["max |Δ| (MW)"] = g["max_err"].round(3)
    g["Bias (MW)"] = g["bias"].round(3).map(lambda v: f"{v:+.3f}")
    g["geschätzte Kosten (EUR)"] = (g["mae"] * 24 * ASSUMED_IMBALANCE_PRICE).round(0)
    show = g[["Datum", "MAE (MW)", "max |Δ| (MW)", "Bias (MW)", "geschätzte Kosten (EUR)"]]
    st.dataframe(show, use_container_width=True, hide_index=True, height=360)

# ---------------------------------------------------------------------------
# Monatliche Genauigkeit
# ---------------------------------------------------------------------------
st.subheader("Monatliche Genauigkeit · Ziel 90 %")

g_mon = df.copy()
g_mon["abs_err"] = (g_mon["actual_mw"] - g_mon["programme_mw"]).abs()
g_mon["ape"] = g_mon["abs_err"] / g_mon["programme_mw"].abs().clip(lower=1e-3) * 100
monthly_acc = g_mon.groupby("month").agg(
    mae=("abs_err", "mean"),
    ape_pct=("ape", "mean"),
    n=("abs_err", "count"),
).reset_index()
monthly_acc["accuracy_pct"] = (100.0 - monthly_acc["ape_pct"]).clip(lower=0, upper=100)

bar_colors = [
    FMV_GREEN if v >= 90 else (FMV_ACCENT if v >= 80 else FMV_RED)
    for v in monthly_acc["accuracy_pct"]
]
fig_m = go.Figure()
fig_m.add_trace(go.Bar(
    x=monthly_acc["month"], y=monthly_acc["accuracy_pct"],
    marker_color=bar_colors,
    text=[f"{v:.1f}%" for v in monthly_acc["accuracy_pct"]],
    textposition="outside",
))
fig_m.add_hline(y=90, line_width=1.5, line_dash="dash", line_color=FMV_GREEN,
                annotation_text="Ziel 90 %", annotation_position="top left")
fig_m.add_hline(y=80, line_width=1, line_dash="dot", line_color=FMV_ACCENT)
fig_m.update_layout(
    height=320, margin=dict(l=20, r=20, t=30, b=20),
    plot_bgcolor="white", paper_bgcolor="white",
    yaxis=dict(title="Genauigkeit (%)", gridcolor="#EEF1F6", range=[0, 105]),
    xaxis=dict(gridcolor="#EEF1F6"),
)
st.plotly_chart(fig_m, use_container_width=True)

st.divider()
st.caption(
    f"Methodik: Genauigkeit = 100 % − Ø |Real − Programm| / |Programm|. "
    f"Geschätzte Ausgleichskosten = Σ |Δ MWh| × {ASSUMED_IMBALANCE_PRICE:.0f} EUR/MWh "
    f"(typischer CH-Mittelwert 2024-25). Tatsächliche Kosten je nach Richtungsfaktor "
    f"höher oder niedriger."
)

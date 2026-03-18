"""
Page 12 - PFC vs HFC
Direct visual comparison between latest CH PFC and HFC OMPEX files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    COLORS,
    align_pfc_hfc,
    export_csv_button,
    hfc_benchmark_dir,
    list_hfc_files,
    load_config,
    load_hfc_series,
    load_hfc_series_from_upload,
    load_pfc_market,
    no_data_warning,
    pfc_hfc_metrics,
    show_freshness_sidebar,
)

st.header("PFC vs HFC (OMPEX)")
st.caption("Comparaison directe de la courbe PFC CH avec les fichiers HFC du dossier benchmark")

show_freshness_sidebar()

pfc = load_pfc_market("CH")
if pfc is None or pfc.empty:
    no_data_warning("PFC CH")
    st.stop()

hfc_dir = hfc_benchmark_dir()
files = list_hfc_files(limit=250)

with st.sidebar:
    st.subheader("Parametres comparaison")
    source_options = ["Upload manuel (.xlsx)"] if not files else ["Dossier benchmark", "Upload manuel (.xlsx)"]
    source_mode = st.radio("Source HFC", source_options, index=0)
    selected_file = None
    if source_mode == "Dossier benchmark":
        file_labels = [f.name for f in files]
        selected_label = st.selectbox("Fichier HFC", file_labels, index=0)
        selected_file = files[file_labels.index(selected_label)]
    else:
        if not files:
            st.info(
                "Sur Streamlit Cloud, le lecteur H: n'est pas accessible. "
                "Upload un fichier HFC pour comparer."
            )
    zoom_days = st.slider("Fenetre recente (jours)", min_value=7, max_value=180, value=45, step=1)

# Main-area uploader (always visible even when sidebar is collapsed)
uploaded_hfc = None
if source_mode == "Upload manuel (.xlsx)":
    uploaded_hfc = st.file_uploader("Uploader un fichier HFC (.xlsx)", type=["xlsx"], key="hfc_upload_main")

if source_mode == "Dossier benchmark" and selected_file is not None:
    hfc = load_hfc_series(selected_file)
    hfc_source_label = selected_file.name
else:
    hfc = load_hfc_series_from_upload(uploaded_hfc)
    hfc_source_label = uploaded_hfc.name if uploaded_hfc is not None else "-"

if hfc is None or hfc.empty:
    if source_mode == "Dossier benchmark" and selected_file is not None:
        st.error(f"Impossible de lire {selected_file.name}. Colonnes date/prix non detectees.")
    else:
        if uploaded_hfc is None:
            st.warning("Upload requis: ajoute un fichier HFC .xlsx pour afficher la comparaison.")
        else:
            st.error(
                "Fichier charge mais non lisible. Verifie le format Excel (colonnes Date + EUR/MWh) "
                "et re-uploade le fichier."
            )
    st.stop()

cmp_df = align_pfc_hfc(pfc, hfc)
if cmp_df.empty:
    st.error("Aucun timestamp commun entre PFC et HFC pour ce fichier.")
    st.stop()

metrics = pfc_hfc_metrics(cmp_df)
window_start = cmp_df.index.min()
window_end = cmp_df.index.max()
quality_cfg = (load_config() or {}).get("quality", {})
mae_limit = float(quality_cfg.get("max_mae_eur_mwh", 20.0))
rmse_limit = float(quality_cfg.get("max_rmse_eur_mwh", 26.0))
bias_limit = float(quality_cfg.get("max_abs_bias_eur_mwh", 5.0))
bias_abs = abs(float(metrics.get("bias", float("nan"))))
gate_ok = (
    float(metrics.get("mae", float("inf"))) <= mae_limit
    and float(metrics.get("rmse", float("inf"))) <= rmse_limit
    and bias_abs <= bias_limit
)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Points", f"{int(metrics.get('n_points', 0)):,}")
with k2:
    st.metric("MAE", f"{metrics.get('mae', float('nan')):.2f}")
with k3:
    st.metric("RMSE", f"{metrics.get('rmse', float('nan')):.2f}")
with k4:
    st.metric("Bias", f"{metrics.get('bias', float('nan')):+.2f}")
with k5:
    st.metric("P95 |err|", f"{metrics.get('p95_abs_error', float('nan')):.2f}")

if gate_ok:
    st.success(
        f"Benchmark gate PASS | MAE {metrics['mae']:.2f}/{mae_limit:.2f} | "
        f"RMSE {metrics['rmse']:.2f}/{rmse_limit:.2f} | |Bias| {bias_abs:.2f}/{bias_limit:.2f}"
    )
else:
    st.error(
        f"Benchmark gate FAIL | MAE {metrics['mae']:.2f}/{mae_limit:.2f} | "
        f"RMSE {metrics['rmse']:.2f}/{rmse_limit:.2f} | |Bias| {bias_abs:.2f}/{bias_limit:.2f}"
    )

st.caption(
    f"Fichier HFC: `{hfc_source_label}` | Dossier config: `{hfc_dir}` | "
    f"Fenetre comparee: {window_start} -> {window_end}"
)

recent_cutoff = window_end - pd.Timedelta(days=zoom_days)
cmp_recent = cmp_df[cmp_df.index >= recent_cutoff]
if cmp_recent.empty:
    cmp_recent = cmp_df

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=cmp_recent.index,
        y=cmp_recent["pfc"],
        name="PFC CH",
        mode="lines",
        line=dict(color=COLORS["blue"], width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=cmp_recent.index,
        y=cmp_recent["hfc"],
        name="HFC OMPEX",
        mode="lines",
        line=dict(color=COLORS["amber"], width=2),
    )
)
fig.update_layout(
    title=f"PFC vs HFC - {zoom_days} derniers jours",
    xaxis_title="Timestamp",
    yaxis_title="EUR/MWh",
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    fig_err = go.Figure()
    fig_err.add_trace(
        go.Scatter(
            x=cmp_recent.index,
            y=cmp_recent["err"],
            name="Erreur (PFC-HFC)",
            mode="lines",
            line=dict(color=COLORS["red"], width=1.6),
        )
    )
    fig_err.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"])
    fig_err.update_layout(title="Erreur signee", yaxis_title="EUR/MWh", height=320)
    st.plotly_chart(fig_err, use_container_width=True)

with col_b:
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=cmp_df["err"],
            nbinsx=50,
            name="Distribution erreur",
            marker_color=COLORS["blue"],
            opacity=0.85,
        )
    )
    fig_hist.add_vline(x=0, line_dash="dot", line_color=COLORS["muted"])
    fig_hist.update_layout(title="Distribution des erreurs", xaxis_title="EUR/MWh", height=320)
    st.plotly_chart(fig_hist, use_container_width=True)

hm = cmp_df.copy()
hm["hour"] = hm.index.hour
hm["weekday"] = hm.index.dayofweek
pivot = hm.pivot_table(index="hour", columns="weekday", values="abs_err", aggfunc="mean")
pivot = pivot.reindex(index=range(24), columns=range(7))
day_labels = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

fig_heat = go.Figure(
    data=go.Heatmap(
        z=pivot.values,
        x=day_labels,
        y=list(range(24)),
        colorscale="YlOrRd",
        colorbar=dict(title="MAE"),
    )
)
fig_heat.update_layout(
    title="Heatmap |erreur| moyenne par heure x jour",
    xaxis_title="Jour",
    yaxis_title="Heure",
    height=420,
)
st.plotly_chart(fig_heat, use_container_width=True)

with st.expander("Table detaillee"):
    out = cmp_df.copy()
    out = out.reset_index().rename(columns={"index": "timestamp"})
    st.dataframe(out.tail(500), hide_index=True, use_container_width=True)

export_csv_button(
    cmp_df.reset_index().rename(columns={"index": "timestamp"}),
    filename=f"pfc_vs_hfc_{Path(hfc_source_label).stem}.csv",
    label="Exporter comparaison (CSV)",
)

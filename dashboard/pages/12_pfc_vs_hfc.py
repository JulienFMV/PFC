"""
Page 12 - PFC vs HFC
Direct visual comparison between latest CH PFC and HFC OMPEX files.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

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
from pfc_shaping.data.ingest_forwards import load_base_prices_from_eex_report

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

# Main-area uploader (always visible). If provided, it has priority over folder selection.
uploaded_hfc = st.file_uploader("Uploader un fichier HFC (.xlsx)", type=["xlsx"], key="hfc_upload_main")

if uploaded_hfc is not None:
    hfc, hfc_upload_error = load_hfc_series_from_upload(uploaded_hfc, return_error=True)
    hfc_source_label = uploaded_hfc.name
elif source_mode == "Dossier benchmark" and selected_file is not None:
    hfc = load_hfc_series(selected_file)
    hfc_upload_error = ""
    hfc_source_label = selected_file.name
else:
    hfc = None
    hfc_upload_error = ""
    hfc_source_label = "-"

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
            if hfc_upload_error:
                st.caption("Detail technique:")
                st.code(hfc_upload_error)
    st.stop()

cmp_df = align_pfc_hfc(pfc, hfc)
if cmp_df.empty:
    st.error("Aucun timestamp commun entre PFC et HFC pour ce fichier.")
    st.stop()

cmp_df = cmp_df.sort_index()


def _delivery_averages(series: pd.Series, label: str) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame(columns=["product", "product_type", label])

    idx = pd.DatetimeIndex(s.index)
    df = pd.DataFrame({label: s.values}, index=idx)
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["is_peak"] = (df.index.dayofweek < 5) & (df.index.hour >= 8) & (df.index.hour < 20)

    rows: list[dict] = []
    for (y, m), grp in df.groupby(["year", "month"]):
        rows.append({"product": f"{y}-{m:02d}", "product_type": "Month", label: float(grp[label].mean())})
    for y in sorted(df["year"].unique()):
        for q, months in {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}.items():
            grp = df[(df["year"] == y) & (df["month"].isin(months))]
            if not grp.empty:
                rows.append({"product": f"{y}-Q{q}", "product_type": "Quarter", label: float(grp[label].mean())})
        grp_y = df[df["year"] == y]
        if not grp_y.empty:
            rows.append({"product": f"{y}", "product_type": "Cal", label: float(grp_y[label].mean())})
    return pd.DataFrame(rows)


def _eex_dict_to_df(base_prices: dict[str, float]) -> pd.DataFrame:
    rows = []
    for key, val in base_prices.items():
        k = str(key)
        if k.endswith("-Peak"):
            rows.append({"product": k[:-5], "eex_peak": float(val)})
        else:
            rows.append({"product": k, "eex_base": float(val)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.groupby("product", as_index=False).first()


def _resolve_eex_report_path(cfg: dict) -> Path | None:
    fwd = cfg.get("forwards", {})
    candidates = [
        fwd.get("eex_report_path_unc"),
        fwd.get("eex_report_path"),
        str(Path(__file__).resolve().parents[2] / "Price_Report_EEX.xlsx"),
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.exists():
            return p
    return None

metrics = pfc_hfc_metrics(cmp_df)
full_window_start = cmp_df.index.min()
full_window_end = cmp_df.index.max()

tomorrow = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
default_start = max(full_window_start.normalize(), tomorrow)
default_end = min(full_window_end.normalize(), pd.Timestamp(year=2029, month=12, day=31))
if default_start > default_end:
    default_start = full_window_start.normalize()
    default_end = full_window_end.normalize()

fcol1, fcol2 = st.columns(2)
with fcol1:
    scope_start_date = st.date_input("Debut analyse", value=default_start.date())
with fcol2:
    scope_end_date = st.date_input("Fin analyse", value=default_end.date())

scope_start = pd.Timestamp(scope_start_date)
scope_end = pd.Timestamp(scope_end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
if scope_end < scope_start:
    st.error("Fenetre invalide: la date de fin doit etre >= date de debut.")
    st.stop()

cmp_scope = cmp_df[(cmp_df.index >= scope_start) & (cmp_df.index <= scope_end)]
if cmp_scope.empty:
    st.error("Aucune donnee PFC/HFC dans la fenetre selectionnee.")
    st.stop()

metrics = pfc_hfc_metrics(cmp_scope)
window_start = cmp_scope.index.min()
window_end = cmp_scope.index.max()
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
    st.metric("P95 |diff|", f"{metrics.get('p95_abs_error', float('nan')):.2f}")

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
    f"Fenetre comparee: {window_start} -> {window_end} "
    f"(disponible: {full_window_start} -> {full_window_end})"
)

recent_cutoff = window_end - pd.Timedelta(days=zoom_days)
cmp_recent = cmp_scope[cmp_scope.index >= recent_cutoff]
if cmp_recent.empty:
    cmp_recent = cmp_scope

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
            name="Difference (PFC-HFC)",
            mode="lines",
            line=dict(color=COLORS["red"], width=1.6),
        )
    )
    fig_err.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"])
    fig_err.update_layout(title="Difference signee", yaxis_title="EUR/MWh", height=320)
    st.plotly_chart(fig_err, use_container_width=True)

with col_b:
    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=cmp_scope["err"],
            nbinsx=50,
            name="Distribution difference",
            marker_color=COLORS["blue"],
            opacity=0.85,
        )
    )
    fig_hist.add_vline(x=0, line_dash="dot", line_color=COLORS["muted"])
    fig_hist.update_layout(title="Distribution des differences", xaxis_title="EUR/MWh", height=320)
    st.plotly_chart(fig_hist, use_container_width=True)

hm = cmp_scope.copy()
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
    title="Heatmap |difference| moyenne par heure x jour",
    xaxis_title="Jour",
    yaxis_title="Heure",
    height=420,
)
st.plotly_chart(fig_heat, use_container_width=True)

st.divider()
st.subheader("Calibration EEX du jour")
st.caption("Comparaison des niveaux moyens PFC et HFC contre les forwards EEX (BASE)")

cfg = load_config() or {}
eex_report_path = _resolve_eex_report_path(cfg)
eex_uploaded = st.file_uploader("Uploader Price_Report_EEX.xlsx (optionnel)", type=["xlsx"], key="eex_upload_main")
eex_base: dict[str, float] | None = None
eex_source = "-"

try:
    if eex_uploaded is not None:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(eex_uploaded.getbuffer())
            tmp_path = Path(tmp.name)
        eex_base = load_base_prices_from_eex_report(tmp_path, market="CH")
        eex_source = eex_uploaded.name
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
    elif eex_report_path is not None:
        eex_base = load_base_prices_from_eex_report(eex_report_path, market="CH")
        eex_source = str(eex_report_path)
except Exception as exc:
    st.warning(f"Lecture EEX impossible: {exc}")
    eex_base = None

if not eex_base:
    st.info("Aucun prix EEX disponible. Upload le fichier Price_Report_EEX.xlsx pour comparer la calibration.")
else:
    pfc_avg = _delivery_averages(cmp_scope["pfc"], "pfc_base")
    hfc_avg = _delivery_averages(cmp_scope["hfc"], "hfc_base")
    eex_df = _eex_dict_to_df(eex_base)

    calib = (
        pfc_avg.merge(hfc_avg, on=["product", "product_type"], how="inner")
        .merge(eex_df[["product", "eex_base"]], on="product", how="inner")
    )
    if calib.empty:
        st.warning("Aucun produit commun entre l'horizon compare PFC/HFC et les produits EEX.")
    else:
        calib["diff_pfc_eex"] = calib["pfc_base"] - calib["eex_base"]
        calib["diff_hfc_eex"] = calib["hfc_base"] - calib["eex_base"]
        calib["abs_pfc_eex"] = calib["diff_pfc_eex"].abs()
        calib["abs_hfc_eex"] = calib["diff_hfc_eex"].abs()
        calib["best_calibrated"] = calib.apply(
            lambda r: "PFC" if r["abs_pfc_eex"] <= r["abs_hfc_eex"] else "HFC", axis=1
        )

        mae_pfc = float(calib["abs_pfc_eex"].mean())
        mae_hfc = float(calib["abs_hfc_eex"].mean())
        pfc_win = float((calib["best_calibrated"] == "PFC").mean() * 100.0)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Produits compares", int(len(calib)))
        with c2:
            st.metric("MAE PFC vs EEX", f"{mae_pfc:.2f}")
        with c3:
            st.metric("MAE HFC vs EEX", f"{mae_hfc:.2f}")
        with c4:
            st.metric("PFC mieux calibree", f"{pfc_win:.1f}%")

        st.caption(f"Source EEX: `{eex_source}`")

        fig_cal = go.Figure()
        fig_cal.add_trace(
            go.Bar(x=calib["product"], y=calib["abs_pfc_eex"], name="|PFC - EEX|", marker_color=COLORS["blue"])
        )
        fig_cal.add_trace(
            go.Bar(x=calib["product"], y=calib["abs_hfc_eex"], name="|HFC - EEX|", marker_color=COLORS["amber"])
        )
        fig_cal.update_layout(
            barmode="group",
            height=360,
            xaxis_title="Produit de livraison",
            yaxis_title="Difference absolue (EUR/MWh)",
        )
        st.plotly_chart(fig_cal, use_container_width=True)

        view = calib[
            [
                "product",
                "product_type",
                "eex_base",
                "pfc_base",
                "hfc_base",
                "diff_pfc_eex",
                "diff_hfc_eex",
                "best_calibrated",
            ]
        ].sort_values(["product_type", "product"])
        st.dataframe(view, hide_index=True, use_container_width=True)

        st.markdown("**Focus forwards Cal 2027-2029**")
        focus = view[(view["product_type"] == "Cal") & (view["product"].isin(["2027", "2028", "2029"]))]
        if focus.empty:
            st.info("Aucun produit Cal 2027-2029 commun dans la fenetre et les prix EEX charges.")
        else:
            st.dataframe(focus, hide_index=True, use_container_width=True)

with st.expander("Table detaillee"):
    out = cmp_scope.copy()
    out = out.reset_index().rename(columns={"index": "timestamp"})
    st.dataframe(out.tail(500), hide_index=True, use_container_width=True)

export_csv_button(
    cmp_scope.reset_index().rename(columns={"index": "timestamp"}),
    filename=f"pfc_vs_hfc_{Path(hfc_source_label).stem}.csv",
    label="Exporter comparaison (CSV)",
)

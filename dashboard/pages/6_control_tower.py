"""
Page 6 - Control Tower
Production health and benchmark monitoring.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    COLORS,
    load_benchmarks,
    load_forecasts_hourly,
    load_pfc_metadata,
    load_runs,
    show_freshness_sidebar,
)

st.header("Control Tower")
st.caption("Pilotage production PFC: runs, qualite benchmark HFC, et controle calibration")

show_freshness_sidebar()

runs = load_runs(limit=120)
bench = load_benchmarks(limit=120)
pfc_meta = load_pfc_metadata()

if runs.empty:
    st.warning("Aucun run disponible dans DuckDB. Lance le pipeline de mise a jour.")
    st.stop()

latest = runs.iloc[0]
latest_run_id = str(latest["run_id"])
latest_row_count = int(latest["row_count"]) if pd.notna(latest["row_count"]) else 0
latest_source = str(latest.get("source_forwards", "-"))
latest_calibrated = bool(latest.get("calibrated"))

with st.sidebar:
    st.subheader("Filtres")
    selected_run = st.selectbox("Run", runs["run_id"].astype(str).tolist(), index=0)
    points_limit = st.slider("Points horaires max", min_value=48, max_value=3000, value=720, step=24)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Dernier run", latest_run_id)
with k2:
    st.metric("Rows PFC", f"{latest_row_count:,}")
with k3:
    st.metric("Forwards source", latest_source)
with k4:
    st.metric("Calibration", "OK" if latest_calibrated else "Fallback")

st.caption(f"Dernier export detecte: `{pfc_meta['file']}` ({pfc_meta['updated_at']})")
st.divider()

tab1, tab2, tab3 = st.tabs(["Runs", "Benchmark HFC", "Forecast Snapshot"])

with tab1:
    st.subheader("Historique des runs")
    runs_view = runs.copy()
    if "run_ts_utc" in runs_view.columns:
        runs_view["run_ts_utc"] = pd.to_datetime(runs_view["run_ts_utc"], errors="coerce")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=runs_view["run_id"].astype(str),
            y=runs_view["row_count"],
            name="Row count",
            marker_color=COLORS["blue"],
        )
    )
    if "calibrated" in runs_view.columns:
        cal_series = runs_view["calibrated"].fillna(False).astype(int) * runs_view["row_count"].fillna(0)
        fig.add_trace(
            go.Scatter(
                x=runs_view["run_id"].astype(str),
                y=cal_series,
                name="Calibrated rows",
                mode="lines+markers",
                line=dict(color=COLORS["amber"], width=2),
            )
        )
    fig.update_layout(
        xaxis_title="Run ID",
        yaxis_title="Rows",
        height=380,
        legend=dict(orientation="h", y=1.05, x=0),
    )
    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        runs_view[["run_id", "run_ts_utc", "source_forwards", "row_count", "calibrated"]],
        hide_index=True,
        width="stretch",
    )

with tab2:
    st.subheader("Benchmark PFC vs HFC (OMPEX)")
    if bench.empty:
        st.info("Aucun benchmark stocke pour le moment.")
    else:
        bench_view = bench.copy()
        bench_view["run_id"] = bench_view["run_id"].astype(str)
        bench_view = bench_view.sort_values("run_id")

        col_a, col_b = st.columns(2)
        with col_a:
            fig_mae = go.Figure()
            fig_mae.add_trace(
                go.Scatter(
                    x=bench_view["run_id"],
                    y=bench_view["mae"],
                    name="MAE",
                    mode="lines+markers",
                    line=dict(color=COLORS["blue"], width=2),
                )
            )
            fig_mae.add_trace(
                go.Scatter(
                    x=bench_view["run_id"],
                    y=bench_view["rmse"],
                    name="RMSE",
                    mode="lines+markers",
                    line=dict(color=COLORS["amber"], width=2),
                )
            )
            fig_mae.update_layout(xaxis_title="Run ID", yaxis_title="EUR/MWh", height=320)
            st.plotly_chart(fig_mae, width="stretch")

        with col_b:
            colors = [COLORS["green"] if v <= 0 else COLORS["red"] for v in bench_view["bias"].fillna(0)]
            fig_bias = go.Figure()
            fig_bias.add_trace(
                go.Bar(
                    x=bench_view["run_id"],
                    y=bench_view["bias"],
                    name="Bias",
                    marker_color=colors,
                )
            )
            fig_bias.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"])
            fig_bias.update_layout(xaxis_title="Run ID", yaxis_title="Bias EUR/MWh", height=320)
            st.plotly_chart(fig_bias, width="stretch")

        st.dataframe(
            bench_view[
                ["run_id", "n_points", "mae", "rmse", "bias", "p95_abs_error", "window_start", "window_end"]
            ],
            hide_index=True,
            width="stretch",
        )

with tab3:
    st.subheader("Snapshot forecast horaire")
    fc = load_forecasts_hourly(run_id=selected_run, limit=points_limit)
    if fc.empty:
        st.info("Aucune prevision horaire disponible pour ce run.")
    else:
        fc["ts_local"] = pd.to_datetime(fc["ts_local"], errors="coerce")
        fc = fc.dropna(subset=["ts_local"]).sort_values("ts_local")

        fig_fc = go.Figure()
        if "p90" in fc.columns and "p10" in fc.columns:
            fig_fc.add_trace(
                go.Scatter(x=fc["ts_local"], y=fc["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip")
            )
            fig_fc.add_trace(
                go.Scatter(
                    x=fc["ts_local"],
                    y=fc["p10"],
                    name="IC 80%",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=COLORS["band"],
                    hoverinfo="skip",
                )
            )
        fig_fc.add_trace(
            go.Scatter(
                x=fc["ts_local"],
                y=fc["price_shape"],
                name="PFC horaire",
                mode="lines",
                line=dict(color=COLORS["blue"], width=2),
            )
        )
        fig_fc.update_layout(xaxis_title="Timestamp local", yaxis_title="EUR/MWh", height=420)
        st.plotly_chart(fig_fc, width="stretch")

        st.dataframe(fc.tail(200), hide_index=True, width="stretch")

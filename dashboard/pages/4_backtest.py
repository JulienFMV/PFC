"""
Page 4 â€” Backtest & Diagnostics
"Peut-on faire confiance au modÃ¨le ?"
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    COLORS, export_csv_button, load_entso, load_epex, load_pfc,
    no_data_warning, show_freshness_sidebar,
)

st.header("Backtest & Diagnostics")
st.caption("Walk-forward mensuel â€” calibration sur 24 mois, test out-of-sample 1 mois")

show_freshness_sidebar()

epex = load_epex()

if epex is None or "price_eur_mwh" not in (epex.columns if epex is not None else []):
    no_data_warning("prix EPEX (nÃ©cessaire pour le backtest)")
    st.stop()

# â”€â”€ Run backtest on demand â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    st.subheader("ParamÃ¨tres backtest")
    bt_start = st.date_input("DÃ©but test", value=pd.Timestamp("2024-01-01"))
    bt_end = st.date_input("Fin test", value=pd.Timestamp("2026-02-28"))
    base_price = st.number_input("Base price (EUR/MWh)", value=75.0, step=5.0)
    run_bt = st.button("Lancer le backtest", type="primary", use_container_width=True)

if run_bt:
    with st.spinner("Backtest en cours... (recalibration mensuelle)"):
        try:
            from pfc_shaping.validation.backtest import WalkForwardBacktest

            entso = load_entso()
            bt = WalkForwardBacktest(
                start=str(bt_start), end=str(bt_end),
                base_price=base_price,
            )
            report = bt.run(epex, entso)

            if not report.results:
                st.error("Aucune pÃ©riode n'a pu Ãªtre Ã©valuÃ©e. VÃ©rifiez l'historique disponible.")
                st.stop()

            st.session_state["bt_report"] = report
            st.session_state["bt_df"] = report.to_dataframe()
            st.session_state["bt_summary"] = report.summary()
            st.success("Backtest terminÃ©.")
        except Exception as e:
            st.error(f"Erreur backtest : {e}")
            st.stop()

# â”€â”€ Display results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if "bt_summary" not in st.session_state:
    st.info(
        "Configure les paramÃ¨tres dans la sidebar et clique sur "
        "**Lancer le backtest** pour dÃ©marrer."
    )
    st.stop()

summary = st.session_state["bt_summary"]
bt_df = st.session_state["bt_df"]

# â”€â”€ KPI Row â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    rmse = summary["RMSE_shape_mean"]
    st.metric("RMSE shape", f"{rmse:.4f}",
              delta="OK" if rmse < 0.15 else "Ã‰levÃ©",
              delta_color="normal" if rmse < 0.15 else "inverse")
with k2:
    mae = summary["MAE_shape_mean"]
    st.metric("MAE shape", f"{mae:.4f}")
with k3:
    bias = summary["Bias_mean"]
    st.metric("Biais moyen", f"{bias:+.4f}",
              delta_color="normal" if abs(bias) < 0.02 else "inverse")
with k4:
    ic80 = summary["IC80_coverage"]
    st.metric("Couverture IC80%", f"{ic80:.1%}",
              delta="OK" if 0.75 <= ic80 <= 0.85 else "Hors cible",
              delta_color="normal" if 0.75 <= ic80 <= 0.85 else "inverse")
with k5:
    skill = summary["Skill_score_mean"]
    st.metric("Skill score", f"{skill:.3f}",
              delta="Mieux que flat" if skill > 0 else "Pire que flat",
              delta_color="normal" if skill > 0 else "inverse")

st.divider()

# â”€â”€ Monthly KPIs chart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
tab1, tab2, tab3 = st.tabs(["KPIs mensuels", "DÃ©tail par pÃ©riode", "Cibles"])

with tab1:
    fig_kpi = go.Figure()

    fig_kpi.add_trace(go.Bar(
        x=bt_df.index, y=bt_df["rmse_shape"],
        name="RMSE", marker_color=COLORS["blue"], opacity=0.8,
    ))
    fig_kpi.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df["skill_score"],
        name="Skill Score", yaxis="y2",
        line=dict(color=COLORS["amber"], width=2),
        mode="lines+markers", marker=dict(size=6),
    ))
    fig_kpi.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"],
                      annotation_text="Skill=0 (flat)")

    fig_kpi.update_layout(
        yaxis=dict(title="RMSE (ratio)"),
        yaxis2=dict(title="Skill Score", overlaying="y", side="right",
                    showgrid=False),
        height=400,
        legend=dict(orientation="h", y=1.05, x=0),
        barmode="group",
    )
    st.plotly_chart(fig_kpi, width="stretch")

with tab2:
    st.dataframe(
        bt_df.style.format({
            "rmse_shape": "{:.4f}",
            "mae_shape": "{:.4f}",
            "bias_mean": "{:+.4f}",
            "ic80_coverage": "{:.1%}",
            "skill_score": "{:.3f}",
        }).background_gradient(subset=["skill_score"], cmap="RdYlGn"),
        use_container_width=True,
    )
    export_csv_button(bt_df, "backtest_results.csv", "Export rÃ©sultats backtest")

with tab3:
    st.markdown("""
    | KPI | Cible | Description |
    |-----|-------|-------------|
    | RMSE shape | < 0.15 | Erreur sur les ratios intra-horaires f_Q |
    | Biais moyen | \\|bias\\| < 0.02 | Biais systÃ©matique du modÃ¨le |
    | Couverture IC80% | 78-82% | Fraction des rÃ©els dans [p10, p90] |
    | Skill score | > 0 | Mieux que profil plat (ratio=1) |
    """)

# â”€â”€ Residual analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.subheader("Analyse des rÃ©sidus")

col1, col2 = st.columns(2)

with col1:
    fig_rmse = go.Figure()
    fig_rmse.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df["rmse_shape"],
        fill="tozeroy", fillcolor=COLORS["band"],
        line=dict(color=COLORS["blue"], width=2),
        name="RMSE",
    ))
    fig_rmse.update_layout(title="RMSE par pÃ©riode", yaxis_title="RMSE", height=300)
    st.plotly_chart(fig_rmse, width="stretch")

with col2:
    fig_ic = go.Figure()
    fig_ic.add_trace(go.Bar(
        x=bt_df.index, y=bt_df["ic80_coverage"] * 100,
        marker_color=[COLORS["green"] if 0.75 <= v <= 0.85 else COLORS["red"]
                      for v in bt_df["ic80_coverage"]],
    ))
    fig_ic.add_hline(y=80, line_dash="dot", line_color=COLORS["amber"],
                     annotation_text="Cible 80%")
    fig_ic.update_layout(
        title="Couverture IC 80%", yaxis_title="%",
        yaxis_range=[0, 100], height=300,
    )
    st.plotly_chart(fig_ic, width="stretch")

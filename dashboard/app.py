"""
PFC Dashboard — Price Forward Curve Monitoring
===============================================
Streamlit multi-page app for PFC 15min CH modeling.

Launch:
    cd dashboard
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="PFC Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Pages ─────────────────────────────────────────────────────────────────
overview = st.Page("pages/1_overview.py", title="Overview", icon="📊", default=True)
pfc_vs_fwd = st.Page("pages/2_pfc_vs_forwards.py", title="PFC vs Forwards", icon="📐")
pfc_curve = st.Page("pages/2_pfc_curve.py", title="Courbe PFC", icon="📈")
ch_de = st.Page("pages/7_ch_de_spread.py", title="CH vs DE", icon="↔️")

shape = st.Page("pages/3_shape_factors.py", title="Shape Factors", icon="🔬")
backtest = st.Page("pages/4_backtest.py", title="Backtest", icon="🎯")
control_tower = st.Page("pages/6_control_tower.py", title="Control Tower", icon="🧭")

hydro = st.Page("pages/5_hydro.py", title="Hydro & Production", icon="💧")
outages = st.Page("pages/10_outages.py", title="Indisponibilites", icon="🔴")
flows_map = st.Page("pages/9_flows_map.py", title="Flux transfrontaliers", icon="🗺️")
commodities = st.Page("pages/8_commodities.py", title="Commodites", icon="🛢️")

nav = st.navigation(
    {
        "Marche": [overview, pfc_vs_fwd, pfc_curve, ch_de],
        "Modele": [shape, backtest, control_tower],
        "Fondamentaux": [hydro, outages, flows_map, commodities],
    }
)

# ── Sidebar branding ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
            <span style="font-size:2rem;">⚡</span><br>
            <span style="font-size:1.1rem; font-weight:700; color:#0F52CC;">
                PFC Monitor
            </span><br>
            <span style="font-size:0.75rem; color:#5A6B8A;">
                Price Forward Curve CH+DE &bull; 15min
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

nav.run()

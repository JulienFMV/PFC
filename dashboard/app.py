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
pfc_curve = st.Page("pages/2_pfc_curve.py", title="Courbe PFC", icon="📈")
shape = st.Page("pages/3_shape_factors.py", title="Shape Factors", icon="🔬")
backtest = st.Page("pages/4_backtest.py", title="Backtest & Diagnostics", icon="🎯")
hydro = st.Page("pages/5_hydro.py", title="Hydro & Fondamentaux", icon="💧")

nav = st.navigation(
    {
        "Monitoring": [overview, pfc_curve],
        "Modèle": [shape, backtest],
        "Fondamentaux": [hydro],
    }
)

# ── Sidebar branding ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
            <span style="font-size:2rem;">⚡</span><br>
            <span style="font-size:1.1rem; font-weight:700; color:#FF9F1C;">
                PFC Monitor
            </span><br>
            <span style="font-size:0.75rem; color:#8B949E;">
                Price Forward Curve CH &bull; 15min
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

nav.run()

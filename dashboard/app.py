"""
PFC Dashboard - Price Forward Curve Monitoring
==============================================
Streamlit multi-page app for PFC 15min CH modeling.

Launch:
    cd dashboard
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="PFC Monitor",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: #F3F6FB; }
    [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #D9E2F1; }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #D9E2F1;
        border-radius: 10px;
        padding: 0.45rem 0.55rem;
    }
    button[kind="primary"] {
        background-color: #0F52CC !important;
        border-color: #0F52CC !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

overview = st.Page("pages/1_overview.py", title="Overview", icon=":material/monitoring:", default=True)
pfc_curve = st.Page("pages/2_pfc_curve.py", title="Courbe PFC", icon=":material/show_chart:")
shape = st.Page("pages/3_shape_factors.py", title="Shape Factors", icon=":material/tune:")
backtest = st.Page("pages/4_backtest.py", title="Backtest & Diagnostics", icon=":material/analytics:")
hydro = st.Page("pages/5_hydro.py", title="Hydro & Fondamentaux", icon=":material/water_drop:")

nav = st.navigation(
    {
        "Monitoring": [overview, pfc_curve],
        "Modele": [shape, backtest],
        "Fondamentaux": [hydro],
    }
)

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.25rem 0 1rem 0;">
            <span style="font-size:1.1rem; letter-spacing:0.08em; font-weight:800; color:#0B2E6F;">
                PFC Monitor
            </span><br>
            <span style="font-size:0.78rem; color:#5A6B8A;">
                Price Forward Curve CH | 15min
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

nav.run()

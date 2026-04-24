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
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 100% !important;
        padding-top: 1.0rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        padding-bottom: 2rem !important;
    }
    .main .block-container {
        width: 100%;
    }
    [data-testid="stSidebar"] {
        min-width: 260px;
        max-width: 260px;
    }
    [data-testid="stDataFrame"] {
        width: 100%;
    }
    [data-testid="stPlotlyChart"] {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

if hasattr(st, "Page") and hasattr(st, "navigation"):
    overview = st.Page("pages/1_overview.py", title="Overview", icon="📊", default=True)
    pfc_vs_fwd = st.Page("pages/2_pfc_vs_forwards.py", title="PFC vs Forwards", icon="📐")
    pfc_curve = st.Page("pages/2_pfc_curve.py", title="Courbe PFC", icon="📈")
    short_term = st.Page("pages/11_lear_forecast.py", title="Prevision Court Terme", icon="🎯")
    ch_de = st.Page("pages/7_ch_de_spread.py", title="CH vs DE", icon="↔️")

    shape = st.Page("pages/3_shape_factors.py", title="Shape Factors", icon="🔬")
    backtest = st.Page("pages/4_backtest.py", title="Backtest", icon="🔍")
    control_tower = st.Page("pages/6_control_tower.py", title="Control Tower", icon="🧭")
    hfc_compare = st.Page("pages/12_pfc_vs_hfc.py", title="PFC vs HFC", icon="🆚")

    doc_ct = st.Page("pages/13_doc_court_terme.py", title="Doc Court Terme", icon="📋")
    doc_lt = st.Page("pages/14_doc_long_terme.py", title="Doc Long Terme", icon="📋")

    hydro = st.Page("pages/5_hydro.py", title="Hydro & Production", icon="💧")
    outages = st.Page("pages/10_outages.py", title="Indisponibilites", icon="🔴")
    flows_map = st.Page("pages/9_flows_map.py", title="Flux transfrontaliers", icon="🗺️")
    commodities = st.Page("pages/8_commodities.py", title="Commodites", icon="🛢️")
    entsoe_data = st.Page("pages/15_entsoe.py", title="ENTSO-E Data", icon="📡")
    weather = st.Page("pages/16_weather.py", title="Meteo", icon="🌦️")

    nav = st.navigation(
        {
            "Marche": [overview, pfc_vs_fwd, pfc_curve, short_term, ch_de],
            "Modele": [shape, backtest, control_tower, hfc_compare],
            "Fondamentaux": [hydro, outages, flows_map, commodities, entsoe_data, weather],
            "Documentation": [doc_ct, doc_lt],
        }
    )
    nav.run()
else:
    st.title("PFC Monitor")
    st.caption(
        "Version locale Streamlit legacy detectee. Utilise la navigation automatique dans la sidebar pour ouvrir les pages."
    )

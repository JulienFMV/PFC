"""
Page 9 — Carte & Flux transfrontaliers
"Carte de la Suisse et flux avec les pays voisins"
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    COLORS, add_range_slider, export_csv_button, load_entso, load_epex,
    no_data_warning, show_freshness_sidebar,
)

st.header("Carte & Flux transfrontaliers")
st.caption("Position de la Suisse dans le marche electrique europeen")

show_freshness_sidebar()

entso = load_entso()
epex = load_epex()

has_entso = entso is not None and not entso.empty
has_epex = epex is not None and "price_eur_mwh" in (epex.columns if epex is not None else [])

if not has_entso:
    no_data_warning("donnees ENTSO-E")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Options carte")
    lookback_days = st.slider("Lookback (jours)", 7, 365, 30)

n_periods = 96 * lookback_days
entso_view = entso.iloc[-min(n_periods, len(entso)):]

# ── KPI Row ───────────────────────────────────────────────────────────────
avg_xb = entso_view["cross_border_mw"].mean()
max_import = entso_view["cross_border_mw"].max()
max_export = entso_view["cross_border_mw"].min()
pct_import = (entso_view["cross_border_mw"] > 0).mean() * 100

k1, k2, k3, k4 = st.columns(4)
with k1:
    direction = "Import" if avg_xb > 0 else "Export"
    st.metric(f"Flux moyen ({direction})", f"{abs(avg_xb):.0f} MW")
with k2:
    st.metric("Max import", f"{max_import:.0f} MW")
with k3:
    st.metric("Max export", f"{abs(max_export):.0f} MW")
with k4:
    st.metric("% temps en import", f"{pct_import:.0f}%")

st.divider()

# ── Carte schematique — Suisse & voisins ─────────────────────────────────
st.subheader("Carte des flux transfrontaliers")

xb = entso_view["cross_border_mw"]
net_flow = float(xb.mean())

# NTC-based split of aggregate cross-border flow to neighbours
# Approximate NTC shares (typical Swissgrid values)
NTC_SHARES = {"DE": 0.30, "FR": 0.35, "IT": 0.25, "AT": 0.10}

# Country positions (lon, lat) for map
COUNTRIES = {
    "CH": {"lon": 8.23, "lat": 46.82, "name": "Suisse"},
    "DE": {"lon": 10.45, "lat": 51.16, "name": "Allemagne"},
    "FR": {"lon": 2.21, "lat": 46.60, "name": "France"},
    "IT": {"lon": 12.57, "lat": 41.87, "name": "Italie"},
    "AT": {"lon": 14.55, "lat": 47.52, "name": "Autriche"},
}

# Build estimated per-border flows
border_flows = {}
for country, share in NTC_SHARES.items():
    border_flows[country] = net_flow * share

fig_map = go.Figure()

# Country markers
for code, info in COUNTRIES.items():
    is_ch = code == "CH"
    fig_map.add_trace(go.Scattergeo(
        lon=[info["lon"]], lat=[info["lat"]],
        text=[info["name"]],
        textposition="top center" if code != "IT" else "bottom center",
        mode="markers+text",
        marker=dict(
            size=28 if is_ch else 18,
            color="#0F52CC" if is_ch else "#D4DEEE",
            line=dict(width=2, color="#0F52CC" if is_ch else "#8899AA"),
            symbol="circle",
        ),
        textfont=dict(
            size=14 if is_ch else 11,
            color="#0F52CC" if is_ch else "#333",
            family="Arial Black" if is_ch else "Arial",
        ),
        showlegend=False,
        hoverinfo="skip",
    ))

# Flow arrows from/to CH using geo annotations (real arrowheads)
ch_lon, ch_lat = COUNTRIES["CH"]["lon"], COUNTRIES["CH"]["lat"]
flow_annotations = []

for country, flow_mw in border_flows.items():
    c = COUNTRIES[country]
    abs_flow = abs(flow_mw)
    is_import = flow_mw > 0  # positive = import to CH

    # Arrow: tip points in direction of flow
    if is_import:
        ax_lon, ax_lat = c["lon"], c["lat"]       # tail (neighbour)
        x_lon, x_lat = ch_lon, ch_lat              # head (CH)
        color = "#C63D3D"
        label = f"{c['name']} → CH"
    else:
        ax_lon, ax_lat = ch_lon, ch_lat             # tail (CH)
        x_lon, x_lat = c["lon"], c["lat"]           # head (neighbour)
        color = "#1F9D55"
        label = f"CH → {c['name']}"

    # Line width proportional to flow
    width = max(2, min(6, abs_flow / 150))

    # Plotly geo annotation with arrowhead
    flow_annotations.append(dict(
        x=x_lon, y=x_lat,
        ax=ax_lon, ay=ax_lat,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.8,
        arrowwidth=width,
        arrowcolor=color,
        text="",
    ))

    # Flow line (thicker, for visual weight + hover)
    fig_map.add_trace(go.Scattergeo(
        lon=[ax_lon, x_lon],
        lat=[ax_lat, x_lat],
        mode="lines",
        line=dict(width=width, color=color),
        showlegend=False,
        hoverinfo="text",
        hovertext=f"{label}: {abs_flow:.0f} MW",
    ))

    # Label at 40% from tail toward head (offset from center to avoid overlap)
    label_lon = ax_lon + 0.45 * (x_lon - ax_lon)
    label_lat = ax_lat + 0.45 * (x_lat - ax_lat)
    # Slight perpendicular offset to avoid line overlap
    dx = x_lon - ax_lon
    dy = x_lat - ax_lat
    length = (dx**2 + dy**2) ** 0.5
    if length > 0:
        perp_x = -dy / length * 0.4
        perp_y = dx / length * 0.4
    else:
        perp_x, perp_y = 0, 0
    label_lon += perp_x
    label_lat += perp_y

    arrow_symbol = "▶" if is_import else "◀"
    fig_map.add_trace(go.Scattergeo(
        lon=[label_lon], lat=[label_lat],
        mode="text",
        text=[f"{'→' if is_import else '←'} {abs_flow:.0f} MW"],
        textfont=dict(size=13, color=color, family="Arial Black"),
        showlegend=False,
        hoverinfo="skip",
    ))

# Legend traces
fig_map.add_trace(go.Scattergeo(
    lon=[None], lat=[None], mode="lines",
    line=dict(width=3, color="#C63D3D"),
    name="Import → CH",
))
fig_map.add_trace(go.Scattergeo(
    lon=[None], lat=[None], mode="lines",
    line=dict(width=3, color="#1F9D55"),
    name="Export ← CH",
))

fig_map.update_layout(
    geo=dict(
        scope="europe",
        projection_type="natural earth",
        center=dict(lon=8.5, lat=47),
        lonaxis=dict(range=[-2, 20]),
        lataxis=dict(range=[40, 54]),
        showland=True,
        landcolor="#F0F2F6",
        showocean=True,
        oceancolor="#E8EEF8",
        showcountries=True,
        countrycolor="#BCC6D4",
        countrywidth=1.5,
        showlakes=True,
        lakecolor="#D4E4F7",
        showrivers=False,
    ),
    height=520,
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(
        orientation="h", y=-0.02, x=0.5, xanchor="center",
        font=dict(size=12),
    ),
)

st.plotly_chart(fig_map, use_container_width=True)

st.caption(
    "NB : Les flux par frontiere sont estimes a partir du flux net agrege CH "
    "et des parts NTC typiques (DE 30%, FR 35%, IT 25%, AT 10%). "
    "Fleches : → = import vers CH, ← = export depuis CH."
)

# ── Import / Export breakdown ─────────────────────────────────────────────
st.subheader("Import vs Export — statistiques")

importing = xb[xb > 0]
exporting = xb[xb < 0]
pct_time_import = len(importing) / len(xb) * 100
pct_time_export = len(exporting) / len(xb) * 100
avg_when_import = float(importing.mean()) if len(importing) > 0 else 0
avg_when_export = float(exporting.mean()) if len(exporting) > 0 else 0

col_imp, col_exp = st.columns(2)

with col_imp:
    st.markdown(
        f'<div style="padding:1.2rem; border-radius:8px; '
        f'border-left:4px solid #C63D3D; background:white;">'
        f'<div style="font-size:0.9rem; color:#C63D3D; font-weight:600;">'
        f'IMPORT (vers CH)</div>'
        f'<div style="font-size:2rem; font-weight:700; color:#C63D3D;">'
        f'{avg_when_import:.0f} MW</div>'
        f'<div style="font-size:0.8rem; color:#5A6B8A;">moy. quand import</div>'
        f'<hr style="margin:0.5rem 0; border-color:#eee;">'
        f'<div style="font-size:0.85rem;">'
        f'<b>{pct_time_import:.0f}%</b> du temps en import<br>'
        f'Max : <b>{float(xb.max()):.0f} MW</b></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with col_exp:
    st.markdown(
        f'<div style="padding:1.2rem; border-radius:8px; '
        f'border-left:4px solid #1F9D55; background:white;">'
        f'<div style="font-size:0.9rem; color:#1F9D55; font-weight:600;">'
        f'EXPORT (depuis CH)</div>'
        f'<div style="font-size:2rem; font-weight:700; color:#1F9D55;">'
        f'{abs(avg_when_export):.0f} MW</div>'
        f'<div style="font-size:0.8rem; color:#5A6B8A;">moy. quand export</div>'
        f'<hr style="margin:0.5rem 0; border-color:#eee;">'
        f'<div style="font-size:0.85rem;">'
        f'<b>{pct_time_export:.0f}%</b> du temps en export<br>'
        f'Max : <b>{abs(float(xb.min())):.0f} MW</b></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Balance bar
net = float(xb.mean())
net_dir = "Import net" if net > 0 else "Export net"
net_color = "#C63D3D" if net > 0 else "#1F9D55"
imp_bar_pct = pct_time_import
st.markdown(
    f'<div style="margin:1rem 0; padding:0.6rem; border-radius:8px; background:white; '
    f'border:1px solid #D4DEEE;">'
    f'<div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">'
    f'<span style="font-size:0.8rem; color:#C63D3D;">Import {pct_time_import:.0f}%</span>'
    f'<span style="font-size:0.85rem; font-weight:700; color:{net_color};">'
    f'{net_dir} : {abs(net):.0f} MW</span>'
    f'<span style="font-size:0.8rem; color:#1F9D55;">Export {pct_time_export:.0f}%</span>'
    f'</div>'
    f'<div style="height:12px; border-radius:6px; overflow:hidden; display:flex;">'
    f'<div style="width:{imp_bar_pct:.0f}%; background:#C63D3D;"></div>'
    f'<div style="width:{100-imp_bar_pct:.0f}%; background:#1F9D55;"></div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Cross-border flow time series ─────────────────────────────────────────
st.subheader("Flux net transfrontalier — serie temporelle")

xb_daily = entso_view["cross_border_mw"].resample("D").mean()

fig_ts = go.Figure()

# Colored bars
pos = xb_daily.clip(lower=0)
neg = xb_daily.clip(upper=0)
fig_ts.add_trace(go.Bar(
    x=pos.index, y=pos.values,
    name="Import", marker_color=COLORS["red"], opacity=0.6,
    hovertemplate="%{x|%d/%m}: +%{y:.0f} MW<extra>Import</extra>",
))
fig_ts.add_trace(go.Bar(
    x=neg.index, y=neg.values,
    name="Export", marker_color=COLORS["green"], opacity=0.6,
    hovertemplate="%{x|%d/%m}: %{y:.0f} MW<extra>Export</extra>",
))

# Rolling average
rolling_avg = xb_daily.rolling(7).mean()
fig_ts.add_trace(go.Scatter(
    x=rolling_avg.index, y=rolling_avg.values,
    name="Moy. mobile 7j",
    line=dict(color=COLORS["amber"], width=2),
    hovertemplate="%{x|%d/%m}: %{y:+.0f} MW<extra>Moy 7j</extra>",
))

fig_ts.add_hline(y=0, line_color=COLORS["muted"], line_width=1)
fig_ts.update_layout(
    yaxis_title="MW (positif = import)", height=450,
    barmode="relative",
    legend=dict(orientation="h", y=1.05, x=0),
)
fig_ts = add_range_slider(fig_ts)
st.plotly_chart(fig_ts, use_container_width=True)

# ── Hourly flow pattern ──────────────────────────────────────────────────
st.subheader("Profil horaire moyen des flux")

xb_hourly = entso_view[["cross_border_mw"]].copy()
xb_hourly["hour"] = xb_hourly.index.hour
hourly_profile = xb_hourly.groupby("hour")["cross_border_mw"].agg(["mean", "std"])

fig_hp = go.Figure()
fig_hp.add_trace(go.Scatter(
    x=hourly_profile.index,
    y=(hourly_profile["mean"] + hourly_profile["std"]).values,
    line=dict(width=0), showlegend=False, hoverinfo="skip",
))
fig_hp.add_trace(go.Scatter(
    x=hourly_profile.index,
    y=(hourly_profile["mean"] - hourly_profile["std"]).values,
    name="±1σ", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(15,82,204,0.1)",
    hoverinfo="skip",
))
fig_hp.add_trace(go.Scatter(
    x=hourly_profile.index, y=hourly_profile["mean"].values,
    name="Flux moyen",
    line=dict(color=COLORS["blue"], width=2),
    hovertemplate="%{x}h: %{y:+.0f} MW<extra></extra>",
))
fig_hp.add_hline(y=0, line_color=COLORS["muted"], line_width=1)
fig_hp.update_layout(
    xaxis_title="Heure", yaxis_title="MW (positif = import)",
    height=350, xaxis=dict(dtick=2),
    legend=dict(orientation="h", y=1.05, x=0),
)
st.plotly_chart(fig_hp, use_container_width=True)

# ── Weekly pattern ────────────────────────────────────────────────────────
st.subheader("Profil hebdomadaire des flux")

xb_weekly = entso_view[["cross_border_mw"]].copy()
xb_weekly["dow"] = xb_weekly.index.dayofweek
dow_names = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
weekly_profile = xb_weekly.groupby("dow")["cross_border_mw"].mean()

fig_wp = go.Figure(go.Bar(
    x=[dow_names[i] for i in weekly_profile.index],
    y=weekly_profile.values,
    marker_color=[COLORS["red"] if v > 0 else COLORS["green"] for v in weekly_profile.values],
    opacity=0.7,
    hovertemplate="%{x}: %{y:+.0f} MW<extra></extra>",
))
fig_wp.add_hline(y=0, line_color=COLORS["muted"], line_width=1)
fig_wp.update_layout(yaxis_title="MW moyen", height=300)
st.plotly_chart(fig_wp, use_container_width=True)

# ── Correlation with price ────────────────────────────────────────────────
if has_epex:
    st.subheader("Flux vs Prix spot")

    common = epex[["price_eur_mwh"]].join(entso[["cross_border_mw"]], how="inner")
    common = common.iloc[-min(n_periods, len(common)):]
    hourly = common.resample("h").mean().dropna()

    if len(hourly) > 50:
        corr = hourly["price_eur_mwh"].corr(hourly["cross_border_mw"])

        col_scatter, col_info = st.columns([2, 1])
        with col_scatter:
            sample = hourly.sample(min(2000, len(hourly)), random_state=42)
            # Color by import/export direction
            colors = ["#C63D3D" if v > 0 else "#1F9D55" for v in sample["cross_border_mw"]]
            fig_sc = go.Figure()
            # Import points
            mask_imp = sample["cross_border_mw"] > 0
            fig_sc.add_trace(go.Scatter(
                x=sample.loc[mask_imp, "cross_border_mw"],
                y=sample.loc[mask_imp, "price_eur_mwh"],
                mode="markers", name="Import",
                marker=dict(size=6, opacity=0.5, color="#C63D3D"),
                hovertemplate="Flux: %{x:+.0f} MW<br>Prix: %{y:.1f} EUR/MWh<extra>Import</extra>",
            ))
            # Export points
            fig_sc.add_trace(go.Scatter(
                x=sample.loc[~mask_imp, "cross_border_mw"],
                y=sample.loc[~mask_imp, "price_eur_mwh"],
                mode="markers", name="Export",
                marker=dict(size=6, opacity=0.5, color="#1F9D55"),
                hovertemplate="Flux: %{x:+.0f} MW<br>Prix: %{y:.1f} EUR/MWh<extra>Export</extra>",
            ))
            fig_sc.add_vline(x=0, line_color=COLORS["muted"], line_width=1, line_dash="dash")
            fig_sc.update_layout(
                xaxis_title="Flux net (MW, + = import)",
                yaxis_title="Prix spot (EUR/MWh)",
                height=400,
                legend=dict(orientation="h", y=1.05, x=0),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        with col_info:
            st.markdown(f"**Correlation** : {corr:.3f}")
            st.markdown(
                "- **Import** (flux > 0) : la Suisse consomme plus "
                "qu'elle ne produit → generalement prix plus eleves\n"
                "- **Export** (flux < 0) : surplus hydro/nucleaire → "
                "prix plus bas en CH que chez les voisins"
            )

# ── Export ────────────────────────────────────────────────────────────────
with st.expander("Export"):
    export_csv_button(
        entso_view[["cross_border_mw"]],
        "cross_border_flows.csv",
        "Export flux transfrontaliers",
    )

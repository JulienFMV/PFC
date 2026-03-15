"""
Page 8 — Commodites & Macro
"TTF, CO2, Brent — Analyse technique"
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    COLORS, COMMODITY_TICKERS, add_range_slider, compute_bollinger,
    export_csv_button, load_commodities, show_freshness_sidebar,
)

st.header("Commodites & Macro")
st.caption("Fondamentaux energie — TTF, CO2, Brent — donnees Yahoo Finance")

show_freshness_sidebar()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Analyse technique")
    sma_short = st.slider("SMA courte (jours)", 5, 50, 20)
    sma_long = st.slider("SMA longue (jours)", 20, 200, 50)
    bb_window = st.slider("Bollinger (jours)", 10, 50, 20)
    bb_std = st.selectbox("Bollinger ecart-type", [1.5, 2.0, 2.5], index=1)
    show_volume = st.checkbox("Afficher volume", value=False)

# ── Load data ─────────────────────────────────────────────────────────────
commodities = load_commodities(period="2y")

if not commodities:
    from pathlib import Path
    import os
    cache = Path(__file__).resolve().parent.parent.parent / "data" / "commodities_cache.parquet"
    alt_cache = Path("data") / "commodities_cache.parquet"
    st.warning(
        f"Aucune donnee commodite disponible.\n\n"
        f"Cache (abs): `{cache}` → {'existe' if cache.exists() else 'absent'}\n\n"
        f"Cache (rel): `{alt_cache.resolve()}` → {'existe' if alt_cache.exists() else 'absent'}\n\n"
        f"CWD: `{os.getcwd()}`\n\n"
        f"PROJECT_ROOT: `{Path(__file__).resolve().parent.parent}`",
        icon="⚠️",
    )
    st.stop()

# ── KPI Row ───────────────────────────────────────────────────────────────
kpi_cols = st.columns(len(commodities))
for i, (name, df) in enumerate(commodities.items()):
    with kpi_cols[i]:
        cfg = COMMODITY_TICKERS[name]
        last = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2]) if len(df) > 1 else last
        chg = last - prev
        chg_pct = (chg / prev * 100) if prev != 0 else 0
        st.metric(
            name,
            f"{last:.2f} {cfg['unit']}",
            delta=f"{chg:+.2f} ({chg_pct:+.1f}%)",
            delta_color="inverse",
        )

st.divider()

# ── Tabs per commodity ────────────────────────────────────────────────────
tabs = st.tabs(list(commodities.keys()) + ["Correlations"])

for idx, (name, df) in enumerate(commodities.items()):
    with tabs[idx]:
        cfg = COMMODITY_TICKERS[name]
        close = df["close"].dropna()

        if close.empty:
            st.info(f"Pas de donnees pour {name}")
            continue

        # Compute technicals
        sma_s = close.rolling(sma_short).mean()
        sma_l = close.rolling(sma_long).mean()
        bb = compute_bollinger(close, window=bb_window, num_std=bb_std)

        # ── Main chart with Bollinger + SMA ───────────────────────────
        st.subheader(f"{name} — Analyse technique")

        fig = go.Figure()

        # Bollinger bands
        fig.add_trace(go.Scatter(
            x=bb.index, y=bb["upper"],
            name="Bollinger sup.", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=bb.index, y=bb["lower"],
            name=f"Bollinger {bb_window}j ±{bb_std}σ",
            line=dict(width=0),
            fill="tonexty", fillcolor="rgba(192,138,43,0.08)",
            hoverinfo="skip",
        ))

        # Price
        fig.add_trace(go.Scatter(
            x=close.index, y=close.values,
            name=name, line=dict(color=cfg["color"], width=2),
            hovertemplate=f"%{{x|%d/%m/%Y}}: %{{y:.2f}} {cfg['unit']}<extra>{name}</extra>",
        ))

        # SMAs
        fig.add_trace(go.Scatter(
            x=sma_s.index, y=sma_s.values,
            name=f"SMA {sma_short}j",
            line=dict(color=COLORS["blue"], width=1.5, dash="dash"),
            hovertemplate=f"%{{y:.2f}}<extra>SMA {sma_short}</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=sma_l.index, y=sma_l.values,
            name=f"SMA {sma_long}j",
            line=dict(color=COLORS["red"], width=1.5, dash="dot"),
            hovertemplate=f"%{{y:.2f}}<extra>SMA {sma_long}</extra>",
        ))

        fig.update_layout(
            yaxis_title=cfg["unit"], height=500,
            legend=dict(orientation="h", y=1.05, x=0),
        )
        fig = add_range_slider(fig)
        st.plotly_chart(fig, use_container_width=True)

        # ── Signal summary ────────────────────────────────────────────
        last_close = float(close.iloc[-1])
        last_sma_s = float(sma_s.dropna().iloc[-1]) if not sma_s.dropna().empty else last_close
        last_sma_l = float(sma_l.dropna().iloc[-1]) if not sma_l.dropna().empty else last_close
        last_bb_upper = float(bb["upper"].dropna().iloc[-1]) if not bb["upper"].dropna().empty else last_close
        last_bb_lower = float(bb["lower"].dropna().iloc[-1]) if not bb["lower"].dropna().empty else last_close

        sig1, sig2, sig3 = st.columns(3)
        with sig1:
            trend = "Haussier" if last_sma_s > last_sma_l else "Baissier"
            color = COLORS["green"] if trend == "Haussier" else COLORS["red"]
            st.markdown(f'**Tendance SMA** : <span style="color:{color};">{trend}</span>',
                        unsafe_allow_html=True)
            st.caption(f"SMA{sma_short} ({last_sma_s:.2f}) vs SMA{sma_long} ({last_sma_l:.2f})")
        with sig2:
            if last_close > last_bb_upper:
                bb_sig, bb_col = "Surachat", COLORS["red"]
            elif last_close < last_bb_lower:
                bb_sig, bb_col = "Survente", COLORS["green"]
            else:
                bb_sig, bb_col = "Neutre", COLORS["muted"]
            st.markdown(f'**Bollinger** : <span style="color:{bb_col};">{bb_sig}</span>',
                        unsafe_allow_html=True)
            st.caption(f"Prix {last_close:.2f} | Bande [{last_bb_lower:.2f} — {last_bb_upper:.2f}]")
        with sig3:
            pct_30d = (last_close / float(close.iloc[-min(30, len(close))]) - 1) * 100
            pct_90d = (last_close / float(close.iloc[-min(90, len(close))]) - 1) * 100
            st.markdown(f"**Perf 30j** : {pct_30d:+.1f}% | **90j** : {pct_90d:+.1f}%")

        # ── Returns distribution ──────────────────────────────────────
        returns = close.pct_change().dropna()
        st.subheader("Distribution des rendements journaliers")
        fig_ret = go.Figure(go.Histogram(
            x=returns.values * 100, nbinsx=60,
            marker_color=cfg["color"], opacity=0.7,
            hovertemplate="%{x:.1f}%<br>%{y} obs<extra></extra>",
        ))
        fig_ret.update_layout(
            xaxis_title="Rendement (%)", yaxis_title="Frequence",
            height=250, margin=dict(l=40, r=10, t=10, b=40),
        )
        st.plotly_chart(fig_ret, use_container_width=True)

        export_csv_button(df, f"{name.lower().replace(' ', '_')}.csv", f"Export {name}")

# ── Correlation tab ───────────────────────────────────────────────────────
with tabs[-1]:
    st.subheader("Correlations entre commodites")

    if len(commodities) < 2:
        st.info("Minimum 2 commodites pour calculer les correlations.")
    else:
        # Align all series on common dates
        aligned = pd.DataFrame()
        for name, df in commodities.items():
            aligned[name] = df["close"]
        aligned = aligned.dropna()

        if len(aligned) < 30:
            st.info("Pas assez de donnees communes pour les correlations.")
        else:
            # Correlation matrix
            corr_matrix = aligned.corr()
            fig_corr = go.Figure(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.index.tolist(),
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr_matrix.values],
                texttemplate="%{text}",
                hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
            ))
            fig_corr.update_layout(height=350)
            st.plotly_chart(fig_corr, use_container_width=True)

            # Rolling correlation
            st.subheader("Correlation glissante (60j)")
            pairs = []
            names_list = list(commodities.keys())
            for i in range(len(names_list)):
                for j in range(i + 1, len(names_list)):
                    pairs.append((names_list[i], names_list[j]))

            fig_rcorr = go.Figure()
            pair_colors = [COLORS["blue"], COLORS["amber"], COLORS["green"]]
            for k, (n1, n2) in enumerate(pairs):
                rc = aligned[n1].rolling(60).corr(aligned[n2])
                fig_rcorr.add_trace(go.Scatter(
                    x=rc.index, y=rc.values,
                    name=f"{n1} / {n2}",
                    line=dict(color=pair_colors[k % len(pair_colors)], width=1.5),
                    hovertemplate=f"%{{x|%d/%m/%Y}}: %{{y:.3f}}<extra>{n1}/{n2}</extra>",
                ))
            fig_rcorr.add_hline(y=0, line_color=COLORS["muted"], line_width=1)
            fig_rcorr.update_layout(
                yaxis_title="Correlation", yaxis_range=[-1, 1], height=350,
                legend=dict(orientation="h", y=1.05, x=0),
            )
            st.plotly_chart(fig_rcorr, use_container_width=True)

            # Normalized price overlay
            st.subheader("Prix normalises (base 100)")
            fig_norm = go.Figure()
            for k, (name, df) in enumerate(commodities.items()):
                s = df["close"].dropna()
                norm = s / float(s.iloc[0]) * 100
                cfg = COMMODITY_TICKERS[name]
                fig_norm.add_trace(go.Scatter(
                    x=norm.index, y=norm.values,
                    name=name,
                    line=dict(color=cfg["color"], width=2),
                    hovertemplate=f"%{{x|%d/%m/%Y}}: %{{y:.1f}}<extra>{name}</extra>",
                ))
            fig_norm.add_hline(y=100, line_color=COLORS["muted"], line_width=1, line_dash="dot")
            fig_norm.update_layout(
                yaxis_title="Base 100", height=400,
                legend=dict(orientation="h", y=1.05, x=0),
            )
            fig_norm = add_range_slider(fig_norm)
            st.plotly_chart(fig_norm, use_container_width=True)

"""
Page 2 — PFC vs Forwards EEX
"Matrice de comparaison PFC vs prix forwards EEX"
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from utils import COLORS, add_range_slider, export_csv_button, load_pfc, show_freshness_sidebar, PROJECT_ROOT

st.header("PFC vs Forwards EEX")
st.caption("Comparaison des moyennes PFC avec les prix forwards du marche")

show_freshness_sidebar()

# ── Load data ──────────────────────────────────────────────────────────────
pfc = load_pfc()
fwd_path = PROJECT_ROOT / "data" / "eex_forwards_history.parquet"

if pfc is None or pfc.empty:
    st.warning("Pas de donnees PFC disponibles.", icon="⚠️")
    st.stop()

has_forwards = fwd_path.exists()
if has_forwards:
    fwd_all = pd.read_parquet(fwd_path)
    fwd_all["date"] = pd.to_datetime(fwd_all["date"])
else:
    fwd_all = pd.DataFrame()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    market = "CH"
    if has_forwards and len(fwd_all["market"].unique()) > 1:
        market = st.radio("Marche forwards", sorted(fwd_all["market"].unique()), horizontal=True)

# ── Compute PFC averages by delivery period ────────────────────────────────
pfc_local = pfc.copy()
if pfc_local.index.tz is not None:
    pfc_local.index = pfc_local.index.tz_convert("Europe/Zurich")
else:
    pfc_local.index = pfc_local.index.tz_localize("UTC").tz_convert("Europe/Zurich")

# Peak definition: weekday 08-20h Zurich
is_weekday = pfc_local.index.weekday < 5
is_peak = is_weekday & (pfc_local.index.hour >= 8) & (pfc_local.index.hour < 20)

pfc_local["year"] = pfc_local.index.year
pfc_local["month"] = pfc_local.index.month
pfc_local["is_peak"] = is_peak


def compute_pfc_averages(df, peak_mask):
    """Compute BASE (all hours), PEAK, and OFFPEAK averages by delivery period."""
    rows = []

    # Monthly
    for (y, m), grp in df.groupby(["year", "month"]):
        key = f"{y}-{m:02d}"
        base_avg = grp["price_shape"].mean()
        peak_avg = grp.loc[grp["is_peak"], "price_shape"].mean() if grp["is_peak"].any() else base_avg
        offpeak_avg = grp.loc[~grp["is_peak"], "price_shape"].mean() if (~grp["is_peak"]).any() else base_avg
        p10_avg = grp["p10"].mean() if "p10" in grp.columns else None
        p90_avg = grp["p90"].mean() if "p90" in grp.columns else None
        rows.append({
            "product": key, "product_type": "Month", "pfc_base": base_avg,
            "pfc_peak": peak_avg, "pfc_offpeak": offpeak_avg,
            "pfc_p10": p10_avg, "pfc_p90": p90_avg, "n_obs": len(grp),
        })

    # Quarterly
    for y in df["year"].unique():
        for q, months in {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}.items():
            mask = (df["year"] == y) & (df["month"].isin(months))
            grp = df[mask]
            if grp.empty:
                continue
            key = f"{y}-Q{q}"
            rows.append({
                "product": key, "product_type": "Quarter", "pfc_base": grp["price_shape"].mean(),
                "pfc_peak": grp.loc[grp["is_peak"], "price_shape"].mean() if grp["is_peak"].any() else None,
                "pfc_offpeak": grp.loc[~grp["is_peak"], "price_shape"].mean() if (~grp["is_peak"]).any() else None,
                "pfc_p10": grp["p10"].mean() if "p10" in grp.columns else None,
                "pfc_p90": grp["p90"].mean() if "p90" in grp.columns else None,
                "n_obs": len(grp),
            })

    # Yearly
    for y in df["year"].unique():
        grp = df[df["year"] == y]
        if grp.empty:
            continue
        rows.append({
            "product": str(y), "product_type": "Cal", "pfc_base": grp["price_shape"].mean(),
            "pfc_peak": grp.loc[grp["is_peak"], "price_shape"].mean() if grp["is_peak"].any() else None,
            "pfc_offpeak": grp.loc[~grp["is_peak"], "price_shape"].mean() if (~grp["is_peak"]).any() else None,
            "pfc_p10": grp["p10"].mean() if "p10" in grp.columns else None,
            "pfc_p90": grp["p90"].mean() if "p90" in grp.columns else None,
            "n_obs": len(grp),
        })

    return pd.DataFrame(rows)


pfc_avg = compute_pfc_averages(pfc_local, is_peak)

# Mark partial coverage products (PFC doesn't cover the full delivery period)
# Expected obs: 96 per day × ~30 days per month
for i, row in pfc_avg.iterrows():
    expected = 96 * 30  # approx 1 month
    if row["product_type"] == "Quarter":
        expected = 96 * 91
    elif row["product_type"] == "Cal":
        expected = 96 * 365
    coverage = row["n_obs"] / expected if expected > 0 else 1.0
    pfc_avg.at[i, "coverage"] = min(coverage, 1.0)

# ── Merge with EEX forwards ───────────────────────────────────────────────
if has_forwards and not fwd_all.empty:
    fwd_mkt = fwd_all[fwd_all["market"] == market].copy()
    latest_fwd_date = fwd_mkt["date"].max()
    fwd_latest = fwd_mkt[fwd_mkt["date"] == latest_fwd_date]

    # Pivot: one row per product, columns for BASE and PEAK
    fwd_base = fwd_latest[fwd_latest["load_type"] == "BASE"][["product", "price"]].rename(
        columns={"price": "eex_base"}
    )
    fwd_peak = fwd_latest[fwd_latest["load_type"] == "PEAK"][["product", "price"]].rename(
        columns={"price": "eex_peak"}
    )
    fwd_pivot = fwd_base.merge(fwd_peak, on="product", how="outer")

    # Merge PFC averages with EEX
    matrix = pfc_avg.merge(fwd_pivot, on="product", how="outer")
    matrix["diff_base"] = matrix["pfc_base"] - matrix["eex_base"]
    matrix["diff_peak"] = matrix["pfc_peak"] - matrix["eex_peak"]
    matrix["diff_base_pct"] = (matrix["diff_base"] / matrix["eex_base"] * 100).round(1)
    matrix["diff_peak_pct"] = (matrix["diff_peak"] / matrix["eex_peak"] * 100).round(1)

    # Nullify diffs for partial coverage products (< 90%)
    partial = matrix["coverage"].fillna(0) < 0.90
    matrix.loc[partial, ["diff_base", "diff_peak", "diff_base_pct", "diff_peak_pct"]] = None

    fwd_date_str = latest_fwd_date.strftime("%d/%m/%Y")
else:
    matrix = pfc_avg.copy()
    matrix["eex_base"] = None
    matrix["eex_peak"] = None
    matrix["diff_base"] = None
    matrix["diff_peak"] = None
    matrix["diff_base_pct"] = None
    matrix["diff_peak_pct"] = None
    fwd_date_str = "N/A"


# ── Sort products by delivery order ────────────────────────────────────────
def delivery_order(p):
    try:
        if len(p) == 4:
            return (int(p), 0, 0)
        if "-Q" in p:
            parts = p.split("-Q")
            return (int(parts[0]), 1, int(parts[1]))
        parts = p.split("-")
        return (int(parts[0]), 2, int(parts[1]))
    except (ValueError, IndexError):
        return (9999, 9, 9)


matrix = matrix.sort_values("product", key=lambda s: s.map(delivery_order))

# ═══════════════════════════════════════════════════════════════════════════
# MATRICE PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════
st.subheader("Matrice de prix")
st.caption(f"Forwards EEX: {market} au {fwd_date_str} | PFC: moyennes par periode de livraison")

# ── Tabs by product type ──────────────────────────────────────────────────
tab_all, tab_cal, tab_quarter, tab_month = st.tabs(["Vue complete", "Cal (annuel)", "Quarters", "Months"])


def format_matrix_table(df, show_cols):
    """Format and display matrix as styled dataframe."""
    display = df[["product"] + show_cols].copy()
    display = display.set_index("product")

    # Rename columns for clarity
    col_map = {
        "eex_base": "EEX Base", "eex_peak": "EEX Peak",
        "pfc_base": "PFC Base", "pfc_peak": "PFC Peak", "pfc_offpeak": "PFC Offpeak",
        "diff_base": "Ecart Base", "diff_peak": "Ecart Peak",
        "diff_base_pct": "Ecart %", "diff_peak_pct": "Ecart Peak %",
        "pfc_p10": "PFC p10", "pfc_p90": "PFC p90",
    }
    display = display.rename(columns=col_map)

    # Style: color ecarts
    def color_diff(val):
        if pd.isna(val):
            return ""
        if isinstance(val, (int, float)):
            if val > 5:
                return "color: #C63D3D; font-weight: bold"
            elif val < -5:
                return "color: #1F9D55; font-weight: bold"
            elif val > 0:
                return "color: #C63D3D"
            elif val < 0:
                return "color: #1F9D55"
        return ""

    diff_cols = [c for c in display.columns if "Ecart" in c]
    styled = display.style.format("{:.1f}", na_rep="—")
    if diff_cols:
        styled = styled.map(color_diff, subset=diff_cols)

    st.dataframe(styled, use_container_width=True, height=min(40 + len(display) * 35, 600))


show_cols = ["eex_base", "pfc_base", "diff_base", "eex_peak", "pfc_peak", "diff_peak"]
if not has_forwards:
    show_cols = ["pfc_base", "pfc_peak", "pfc_offpeak", "pfc_p10", "pfc_p90"]

with tab_all:
    format_matrix_table(matrix, show_cols)

with tab_cal:
    cal_df = matrix[matrix["product_type"] == "Cal"]
    if cal_df.empty:
        st.info("Pas de produits Cal dans les donnees.")
    else:
        format_matrix_table(cal_df, show_cols)

with tab_quarter:
    q_df = matrix[matrix["product_type"] == "Quarter"]
    if q_df.empty:
        st.info("Pas de produits Quarter dans les donnees.")
    else:
        format_matrix_table(q_df, show_cols)

with tab_month:
    m_df = matrix[matrix["product_type"] == "Month"]
    if m_df.empty:
        st.info("Pas de produits Month dans les donnees.")
    else:
        format_matrix_table(m_df, show_cols)

# ═══════════════════════════════════════════════════════════════════════════
# GRAPHIQUE COMPARATIF
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Comparaison visuelle")

# Filter to products that have both PFC and EEX data
has_both = matrix.dropna(subset=["pfc_base", "eex_base"])

if has_both.empty:
    st.info("Pas assez de donnees pour la comparaison visuelle.")
else:
    fig = go.Figure()

    # EEX bars
    fig.add_trace(go.Bar(
        x=has_both["product"], y=has_both["eex_base"],
        name="EEX Forward (Base)",
        marker_color=COLORS["amber"], opacity=0.8,
        hovertemplate="%{x}: %{y:.1f} EUR/MWh<extra>EEX Base</extra>",
    ))

    # PFC bars
    fig.add_trace(go.Bar(
        x=has_both["product"], y=has_both["pfc_base"],
        name="PFC Moyenne (Base)",
        marker_color=COLORS["blue"], opacity=0.8,
        hovertemplate="%{x}: %{y:.1f} EUR/MWh<extra>PFC Base</extra>",
    ))

    # Difference markers
    fig.add_trace(go.Scatter(
        x=has_both["product"], y=has_both["diff_base"],
        name="Ecart (PFC - EEX)",
        mode="markers+text",
        marker=dict(
            size=12, symbol="diamond",
            color=[COLORS["red"] if d > 0 else COLORS["green"] for d in has_both["diff_base"]],
        ),
        text=[f"{d:+.1f}" for d in has_both["diff_base"]],
        textposition="top center", textfont=dict(size=10),
        yaxis="y2",
        hovertemplate="%{x}: %{y:+.1f} EUR/MWh<extra>Ecart</extra>",
    ))

    fig.update_layout(
        barmode="group",
        yaxis_title="Prix EUR/MWh",
        yaxis2=dict(
            title="Ecart EUR/MWh", overlaying="y", side="right",
            zeroline=True, zerolinecolor=COLORS["muted"],
        ),
        height=450,
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Peak comparison
    has_peak = matrix.dropna(subset=["pfc_peak", "eex_peak"])
    if not has_peak.empty:
        fig_pk = go.Figure()
        fig_pk.add_trace(go.Bar(
            x=has_peak["product"], y=has_peak["eex_peak"],
            name="EEX Forward (Peak)", marker_color="#D4A03C", opacity=0.8,
        ))
        fig_pk.add_trace(go.Bar(
            x=has_peak["product"], y=has_peak["pfc_peak"],
            name="PFC Moyenne (Peak)", marker_color="#1A6DFF", opacity=0.8,
        ))
        fig_pk.update_layout(
            barmode="group", yaxis_title="Prix EUR/MWh", height=350,
            legend=dict(orientation="h", y=1.08, x=0), xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_pk, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURE PAR TERME PFC
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Structure par terme PFC")

monthly = matrix[matrix["product_type"] == "Month"].copy()
if not monthly.empty:
    fig_term = go.Figure()

    fig_term.add_trace(go.Scatter(
        x=monthly["product"], y=monthly["pfc_base"],
        name="PFC Base", mode="lines+markers",
        line=dict(color=COLORS["blue"], width=2.5),
        marker=dict(size=8),
    ))
    fig_term.add_trace(go.Scatter(
        x=monthly["product"], y=monthly["pfc_peak"],
        name="PFC Peak", mode="lines+markers",
        line=dict(color=COLORS["red"], width=2),
        marker=dict(size=6),
    ))
    fig_term.add_trace(go.Scatter(
        x=monthly["product"], y=monthly["pfc_offpeak"],
        name="PFC Offpeak", mode="lines+markers",
        line=dict(color=COLORS["green"], width=2),
        marker=dict(size=6),
    ))

    if "pfc_p10" in monthly.columns:
        fig_term.add_trace(go.Scatter(
            x=monthly["product"], y=monthly["pfc_p90"],
            name="IC p90", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_term.add_trace(go.Scatter(
            x=monthly["product"], y=monthly["pfc_p10"],
            name="IC p10-p90", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(15,82,204,0.08)", hoverinfo="skip",
        ))

    # Overlay EEX if available
    if has_forwards:
        eex_monthly = monthly.dropna(subset=["eex_base"])
        if not eex_monthly.empty:
            fig_term.add_trace(go.Scatter(
                x=eex_monthly["product"], y=eex_monthly["eex_base"],
                name="EEX Base", mode="markers",
                marker=dict(size=12, symbol="diamond", color=COLORS["amber"],
                            line=dict(width=2, color="#8B6914")),
            ))

    fig_term.update_layout(
        yaxis_title="EUR/MWh", height=400, xaxis_tickangle=-45,
        legend=dict(orientation="h", y=1.08, x=0),
    )
    st.plotly_chart(fig_term, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# FORWARDS TIMESERIES (if available)
# ═══════════════════════════════════════════════════════════════════════════
if has_forwards and not fwd_all.empty:
    st.divider()
    st.subheader("Evolution des forwards EEX")

    fwd_mkt = fwd_all[fwd_all["market"] == market].copy()

    with st.sidebar:
        st.divider()
        fwd_load_type = st.radio("Type forward", ["BASE", "PEAK"], horizontal=True)
        fwd_product_types = st.multiselect(
            "Produits forwards", ["Cal", "Quarter", "Month"],
            default=["Cal", "Quarter"],
        )

    fwd_filtered = fwd_mkt[
        (fwd_mkt["load_type"] == fwd_load_type) &
        (fwd_mkt["product_type"].isin(fwd_product_types))
    ]

    if fwd_filtered.empty:
        st.info("Aucun forward avec ces filtres.")
    else:
        fig_ts = go.Figure()

        product_colors_cal = ["#0F52CC", "#1A6DFF", "#4D8FFF", "#80B1FF", "#B3D4FF", "#002266"]
        product_colors_q = ["#C08A2B", "#D4A03C", "#E0B85E", "#EDD080", "#1F9D55", "#2BC480", "#0A7E3E", "#5AE0A0"]
        product_colors_m = [
            "#C63D3D", "#D95555", "#E07070", "#E88A8A", "#F0A5A5",
            "#5A6B8A", "#7080A0", "#8898B0", "#A0B0C0", "#1F9D55", "#2BC480", "#40D090",
        ]

        for ptype, colors, width, dash in [
            ("Cal", product_colors_cal, 3, None),
            ("Quarter", product_colors_q, 2, "dash"),
            ("Month", product_colors_m, 1.5, "dot"),
        ]:
            prods = sorted(fwd_filtered[fwd_filtered["product_type"] == ptype]["product"].unique(),
                           key=delivery_order)
            for j, prod in enumerate(prods):
                ps = fwd_filtered[fwd_filtered["product"] == prod].sort_values("date")
                fig_ts.add_trace(go.Scatter(
                    x=ps["date"], y=ps["price"],
                    name=f"{prod}",
                    line=dict(color=colors[j % len(colors)], width=width, dash=dash),
                    hovertemplate=f"%{{x|%d/%m/%Y}}: %{{y:.2f}}<extra>{prod}</extra>",
                ))

        fig_ts.update_layout(
            yaxis_title="EUR/MWh", height=450,
            legend=dict(orientation="h", y=-0.15, x=0, font_size=11),
            hovermode="x unified",
        )
        fig_ts = add_range_slider(fig_ts)
        st.plotly_chart(fig_ts, use_container_width=True)

# ── Export ─────────────────────────────────────────────────────────────────
export_csv_button(matrix, f"pfc_vs_eex_{market}.csv", "Export matrice CSV")

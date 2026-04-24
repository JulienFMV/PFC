"""
Page 15 - ENTSO-E Data Hub
Client-facing showcase page for DSO clients.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    COLORS,
    add_range_slider,
    export_csv_button,
    load_entso,
    load_epex,
    no_data_warning,
    show_freshness_sidebar,
)

st.set_page_config(layout="wide")

st.header("ENTSO-E Data Hub")
st.caption(
    "Vue client des fondamentaux systeme disponibles: charge, production, echanges, profils intrajournaliers et lien avec le prix."
)

show_freshness_sidebar()

entso = load_entso()
epex = load_epex()

if entso is None or entso.empty:
    no_data_warning("donnees ENTSO-E")
    st.stop()

available_cols = set(entso.columns)

ZONE_CONFIG = {
    "CH": {
        "label": "Suisse",
        "load": "load_mw",
        "solar": "solar_mw",
        "wind": "wind_mw",
        "cross_border": "cross_border_mw",
        "mix": ["nuclear_mw", "hydro_ror_mw", "hydro_reservoir_mw", "hydro_pumped_mw", "solar_mw", "wind_mw"],
    },
    "DE": {
        "label": "Allemagne",
        "load": "load_de_mw",
        "solar": "solar_de_mw",
        "wind": "wind_de_mw",
        "cross_border": "scheduled_net_export_ch_de_mw",
        "mix": ["solar_de_mw", "wind_de_mw", "residual_load_de_mw"],
    },
    "FR": {
        "label": "France",
        "load": "load_fr_mw",
        "solar": "solar_fr_mw",
        "wind": "wind_fr_mw",
        "cross_border": "scheduled_net_export_ch_fr_mw",
        "mix": ["solar_fr_mw", "wind_fr_mw"],
    },
    "IT": {
        "label": "Italie",
        "load": "load_it_mw",
        "solar": "solar_it_mw",
        "wind": "wind_it_mw",
        "cross_border": "scheduled_net_export_ch_it_mw",
        "mix": ["solar_it_mw", "wind_it_mw"],
    },
    "AT": {
        "label": "Autriche",
        "load": "load_at_mw",
        "solar": "solar_at_mw",
        "wind": "wind_at_mw",
        "cross_border": "scheduled_net_export_ch_at_mw",
        "mix": ["solar_at_mw", "wind_at_mw"],
    },
}

SERIES_META = {
    "load_mw": ("Charge CH", COLORS["blue"]),
    "solar_mw": ("Solaire CH", COLORS["amber"]),
    "wind_mw": ("Eolien CH", COLORS["green"]),
    "cross_border_mw": ("Echange net CH", COLORS["red"]),
    "nuclear_mw": ("Nucleaire CH", "#6366F1"),
    "hydro_ror_mw": ("Hydro fil de l'eau CH", "#0EA5E9"),
    "hydro_reservoir_mw": ("Hydro reservoir CH", "#2563EB"),
    "hydro_pumped_mw": ("Pompage-turbinage CH", "#22D3EE"),
    "load_de_mw": ("Charge DE", COLORS["blue"]),
    "solar_de_mw": ("Solaire DE", COLORS["amber"]),
    "wind_de_mw": ("Eolien DE", COLORS["green"]),
    "residual_load_de_mw": ("Charge residuelle DE", COLORS["red"]),
    "scheduled_net_export_ch_de_mw": ("Flux CH-DE", COLORS["red"]),
    "load_fr_mw": ("Charge FR", COLORS["blue"]),
    "solar_fr_mw": ("Solaire FR", COLORS["amber"]),
    "wind_fr_mw": ("Eolien FR", COLORS["green"]),
    "scheduled_net_export_ch_fr_mw": ("Flux CH-FR", COLORS["red"]),
    "load_it_mw": ("Charge IT", COLORS["blue"]),
    "solar_it_mw": ("Solaire IT", COLORS["amber"]),
    "wind_it_mw": ("Eolien IT", COLORS["green"]),
    "scheduled_net_export_ch_it_mw": ("Flux CH-IT", COLORS["red"]),
    "load_at_mw": ("Charge AT", COLORS["blue"]),
    "solar_at_mw": ("Solaire AT", COLORS["amber"]),
    "wind_at_mw": ("Eolien AT", COLORS["green"]),
    "scheduled_net_export_ch_at_mw": ("Flux CH-AT", COLORS["red"]),
}


def _has_data(df: pd.DataFrame, col: str | None) -> bool:
    return bool(col and col in df.columns and df[col].notna().any())


def _available_series(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c and _has_data(df, c)]


def _missing_series(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c and c in df.columns and not df[c].notna().any()]


def _series_label(col: str) -> str:
    return SERIES_META.get(col, (col, COLORS["muted"]))[0]


def _series_color(col: str) -> str:
    return SERIES_META.get(col, (col, COLORS["muted"]))[1]


def _existing_zone_columns(zone_code: str) -> dict[str, list[str] | str]:
    cfg = ZONE_CONFIG[zone_code]
    return {
        "load": cfg["load"] if cfg["load"] in available_cols else None,
        "solar": cfg["solar"] if cfg["solar"] in available_cols else None,
        "wind": cfg["wind"] if cfg["wind"] in available_cols else None,
        "cross_border": cfg["cross_border"] if cfg["cross_border"] in available_cols else None,
        "mix": [c for c in cfg["mix"] if c in available_cols],
    }


with st.sidebar:
    st.subheader("Perimetre")
    zone = st.selectbox(
        "Pays / zone",
        list(ZONE_CONFIG.keys()),
        index=0,
        format_func=lambda z: f"{z} · {ZONE_CONFIG[z]['label']}",
    )
    window = st.selectbox("Fenetre", ["14j", "30j", "90j", "1an", "Tout"], index=2)

zone_cols = _existing_zone_columns(zone)
load_col = zone_cols["load"]
solar_col = zone_cols["solar"]
wind_col = zone_cols["wind"]
cross_col = zone_cols["cross_border"]
mix_cols = zone_cols["mix"]

driver_options = [c for c in [load_col, solar_col, wind_col, cross_col] if c]

with st.sidebar:
    selected_drivers = st.multiselect(
        "Series comparables",
        driver_options,
        default=driver_options,
        format_func=_series_label,
    )
    heatmap_candidates = driver_options.copy()
    if zone == "CH" and "load_deviation" in available_cols:
        heatmap_candidates.append("load_deviation")
    heatmap_var = st.selectbox(
        "Variable heatmap",
        heatmap_candidates,
        index=0,
        format_func=lambda c: "Deviation de charge" if c == "load_deviation" else _series_label(c),
    )

lookback = {
    "14j": 96 * 14,
    "30j": 96 * 30,
    "90j": 96 * 90,
    "1an": 96 * 365,
    "Tout": len(entso),
}[window]

view = entso.iloc[-min(lookback, len(entso)) :].copy()
view_h = view.resample("h").mean(numeric_only=True)

available_mix_cols = _available_series(view_h, mix_cols)
missing_mix_cols = _missing_series(view_h, mix_cols)
available_drivers = _available_series(view_h, driver_options)
latest_h = view_h.iloc[-1]
last_week_h = view_h.tail(min(len(view_h), 24 * 7))

renewable_cols = _available_series(
    view_h,
    [solar_col, wind_col, "hydro_ror_mw" if zone == "CH" else None, "hydro_reservoir_mw" if zone == "CH" else None],
)

row = st.columns(5)
with row[0]:
    if _has_data(view_h, load_col):
        st.metric("Charge actuelle", f"{latest_h[load_col] / 1000:.2f} GW")
    else:
        st.metric("Charge actuelle", "n/a")
with row[1]:
    if renewable_cols and _has_data(view_h, load_col):
        share = float(latest_h[renewable_cols].fillna(0.0).sum() / max(latest_h[load_col], 1.0) * 100)
        st.metric("Part renouvelable", f"{share:.1f}%")
    else:
        st.metric("Part renouvelable", "n/a")
with row[2]:
    if _has_data(last_week_h, cross_col):
        st.metric("Echange net 7j", f"{last_week_h[cross_col].mean():+.0f} MW")
    else:
        st.metric("Echange net 7j", "n/a")
with row[3]:
    if _has_data(last_week_h, solar_col):
        st.metric("Pic solaire 7j", f"{last_week_h[solar_col].max():.0f} MW")
    else:
        st.metric("Pic solaire 7j", "n/a")
with row[4]:
    st.metric("Series disponibles", f"{len(available_mix_cols)}/{len(mix_cols)}")

st.markdown(
    """
    **Message client**  
    Cette page montre le type de data-service que l'on peut exposer a un GRD local:
    monitoring systeme, comparaison entre zones, profils intrajournaliers, export et lecture de marche.
    """
)
st.caption(f"Zone affichee: **{zone} - {ZONE_CONFIG[zone]['label']}**. Les series sont filtrees pour eviter les melanges entre pays.")
if missing_mix_cols:
    st.info("Technologies absentes du cache actuel: " + ", ".join(_series_label(c) for c in missing_mix_cols))

tab1, tab2, tab3, tab4 = st.tabs(["Vue systeme", "Heatmaps", "Profils", "Prix & exports"])

with tab1:
    top_left, top_mid, top_right = st.columns([1.3, 1.1, 0.85], gap="large")

    with top_left:
        st.subheader("Charge et echange")
        fig_load = go.Figure()
        if _has_data(view_h, load_col):
            fig_load.add_trace(
                go.Scatter(
                    x=view_h.index,
                    y=view_h[load_col],
                    name=_series_label(load_col),
                    line=dict(color=_series_color(load_col), width=2.8),
                    hovertemplate="%{x|%d/%m %H:%M}<br>%{y:.0f} MW<extra>" + _series_label(load_col) + "</extra>",
                )
            )
        if _has_data(view_h, cross_col):
            fig_load.add_trace(
                go.Scatter(
                    x=view_h.index,
                    y=view_h[cross_col],
                    name=_series_label(cross_col),
                    line=dict(color=_series_color(cross_col), width=1.8),
                    hovertemplate="%{x|%d/%m %H:%M}<br>%{y:.0f} MW<extra>" + _series_label(cross_col) + "</extra>",
                )
            )
            fig_load.add_hline(y=0, line_color="#94A3B8", line_width=1)
        fig_load.update_layout(height=430, yaxis_title="MW", legend=dict(orientation="h", y=1.08, x=0))
        st.plotly_chart(add_range_slider(fig_load), use_container_width=True)

    with top_mid:
        st.subheader("Production visible")
        prod_cols = _available_series(
            view_h,
            [
                solar_col,
                wind_col,
                "nuclear_mw" if zone == "CH" else None,
                "hydro_ror_mw" if zone == "CH" else None,
                "hydro_reservoir_mw" if zone == "CH" else None,
                "hydro_pumped_mw" if zone == "CH" else None,
            ],
        )
        fig_prod = go.Figure()
        for col in prod_cols:
            fig_prod.add_trace(
                go.Scatter(
                    x=view_h.index,
                    y=view_h[col],
                    name=_series_label(col),
                    line=dict(color=_series_color(col), width=2.2),
                    hovertemplate="%{x|%d/%m %H:%M}<br>%{y:.0f} MW<extra>" + _series_label(col) + "</extra>",
                )
            )
        fig_prod.update_layout(height=430, yaxis_title="MW", legend=dict(orientation="h", y=1.12, x=0))
        st.plotly_chart(add_range_slider(fig_prod), use_container_width=True)

    with top_right:
        st.subheader("Lecture rapide")
        quick = pd.DataFrame(index=["Dernier point", "Moyenne 7j", "Min 30j", "Max 30j"])
        for col in available_drivers:
            quick[_series_label(col)] = [
                view_h[col].iloc[-1],
                view_h[col].tail(min(len(view_h), 24 * 7)).mean(),
                view_h[col].tail(min(len(view_h), 24 * 30)).min(),
                view_h[col].tail(min(len(view_h), 24 * 30)).max(),
            ]
        st.dataframe(quick.style.format("{:.0f}"), use_container_width=True, height=210)

        coverage = pd.DataFrame(
            [
                {"Serie": _series_label(col), "Disponibilite %": 100.0 * float(view_h[col].notna().mean())}
                for col in [*driver_options, *mix_cols]
                if col and col in view_h.columns
            ]
        )
        if not coverage.empty:
            coverage = coverage.sort_values("Disponibilite %", ascending=True)
            fig_cov = go.Figure(
                go.Bar(
                    x=coverage["Disponibilite %"],
                    y=coverage["Serie"],
                    orientation="h",
                    marker_color=COLORS["blue"],
                    hovertemplate="%{y}<br>%{x:.1f}%<extra>Disponibilite</extra>",
                )
            )
            fig_cov.update_layout(height=210, xaxis_title="Disponibilite %", yaxis_title="")
            st.plotly_chart(fig_cov, use_container_width=True)

    bottom_left, bottom_right = st.columns([1.3, 1.0], gap="large")
    with bottom_left:
        st.subheader("Series comparables")
        fig_cmp = go.Figure()
        for col in [c for c in selected_drivers if _has_data(view_h, c)]:
            fig_cmp.add_trace(
                go.Scatter(
                    x=view_h.index,
                    y=view_h[col],
                    name=_series_label(col),
                    line=dict(color=_series_color(col), width=2.0),
                    hovertemplate="%{x|%d/%m %H:%M}<br>%{y:.0f} MW<extra>" + _series_label(col) + "</extra>",
                )
            )
        fig_cmp.update_layout(height=400, yaxis_title="MW", legend=dict(orientation="h", y=1.08, x=0))
        st.plotly_chart(add_range_slider(fig_cmp), use_container_width=True)

    with bottom_right:
        st.subheader("Profil moyen de charge")
        if _has_data(view_h, load_col):
            hour_profile = view_h.groupby(view_h.index.hour)[load_col].mean()
            fig_hour = go.Figure(
                go.Bar(
                    x=hour_profile.index,
                    y=hour_profile.values,
                    marker_color=COLORS["blue"],
                    hovertemplate="Heure %{x}:00<br>%{y:.0f} MW<extra>Charge moyenne</extra>",
                )
            )
            fig_hour.update_layout(height=190, xaxis_title="Heure", yaxis_title="MW")
            st.plotly_chart(fig_hour, use_container_width=True)

        st.subheader("Composition moyenne 7 jours")
        comp_cols = _available_series(view_h, mix_cols)
        if comp_cols:
            comp_values = view_h[comp_cols].tail(min(len(view_h), 24 * 7)).mean().sort_values(ascending=False)
            fig_pie = go.Figure(
                go.Pie(
                    labels=[_series_label(c) for c in comp_values.index],
                    values=comp_values.values,
                    hole=0.45,
                )
            )
            fig_pie.update_layout(height=240)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("Aucune technologie de production exploitable dans le cache actuel.")

    if available_mix_cols:
        st.subheader("Drivers journaliers visibles")
        gen_daily = view[available_mix_cols].resample("D").mean(numeric_only=True)
        fig_daily = go.Figure()
        for col in available_mix_cols:
            fig_daily.add_trace(
                go.Scatter(
                    x=gen_daily.index,
                    y=gen_daily[col],
                    name=_series_label(col),
                    line=dict(width=2.3, color=_series_color(col)),
                    hovertemplate="%{x|%d/%m}<br>%{y:.0f} MW<extra>" + _series_label(col) + "</extra>",
                )
            )
        fig_daily.update_layout(height=400, yaxis_title="MW", legend=dict(orientation="h", y=1.08, x=0))
        st.plotly_chart(add_range_slider(fig_daily), use_container_width=True)

with tab2:
    left, right = st.columns([1.4, 1.0], gap="large")
    with left:
        st.subheader("Signature intrajournaliere")
        hm = view_h[[heatmap_var]].copy()
        hm["date"] = hm.index.date
        hm["hour"] = hm.index.hour
        pivot = hm.pivot_table(values=heatmap_var, index="hour", columns="date", aggfunc="mean")
        fig_hm = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns.astype(str),
                y=pivot.index,
                colorscale="RdBu_r" if "deviation" in heatmap_var or "cross_border" in heatmap_var else "YlOrRd",
                colorbar_title=_series_label(heatmap_var) if heatmap_var != "load_deviation" else "Deviation charge",
                hovertemplate="Date %{x}<br>Heure %{y}:00<br>Valeur %{z:.1f}<extra></extra>",
            )
        )
        fig_hm.update_layout(height=560, xaxis_title="Date", yaxis_title="Heure locale")
        st.plotly_chart(fig_hm, use_container_width=True)

    with right:
        st.subheader("Profil horaire moyen")
        hourly_box = view_h.copy()
        hourly_box["hour"] = hourly_box.index.hour
        avg_by_hour = hourly_box.groupby("hour")[heatmap_var].mean()
        fig_hour = go.Figure(
            go.Bar(
                x=avg_by_hour.index,
                y=avg_by_hour.values,
                marker_color=COLORS["amber"],
                hovertemplate="Heure %{x}:00<br>%{y:.1f}<extra>Moyenne</extra>",
            )
        )
        fig_hour.update_layout(height=250, xaxis_title="Heure", yaxis_title=_series_label(heatmap_var))
        st.plotly_chart(fig_hour, use_container_width=True)

        if zone == "CH" and _has_data(view_h, "load_deviation") and _has_data(view_h, solar_col):
            st.subheader("Regimes systeme")
            regime = pd.DataFrame(index=view_h.index)
            regime["solar_bucket"] = pd.qcut(view_h[solar_col].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
            regime["load_deviation"] = view_h["load_deviation"]
            summary = regime.groupby("solar_bucket")["load_deviation"].agg(["mean", "std", "count"])
            st.dataframe(summary.style.format({"mean": "{:.2f}", "std": "{:.2f}"}), use_container_width=True, height=250)

with tab3:
    left, right = st.columns([1.2, 1.0], gap="large")
    profiles = view_h.copy()
    profiles["hour"] = profiles.index.hour
    profiles["weekday"] = profiles.index.dayofweek
    weekday_mask = profiles["weekday"] < 5
    weekend_mask = ~weekday_mask

    with left:
        st.subheader("Profils horaires moyens")
        fig_prof = go.Figure()
        for col in _available_series(profiles, [load_col, solar_col, wind_col]):
            name = _series_label(col)
            color = _series_color(col)
            wd = profiles.loc[weekday_mask].groupby("hour")[col].mean()
            we = profiles.loc[weekend_mask].groupby("hour")[col].mean()
            fig_prof.add_trace(go.Scatter(x=wd.index, y=wd.values, name=f"{name} ouvrable", line=dict(color=color, width=3)))
            fig_prof.add_trace(go.Scatter(x=we.index, y=we.values, name=f"{name} weekend", line=dict(color=color, width=2, dash="dot")))
        fig_prof.update_layout(height=470, xaxis_title="Heure", yaxis_title="MW", legend=dict(orientation="h", y=1.08, x=0))
        st.plotly_chart(fig_prof, use_container_width=True)

    with right:
        if _has_data(profiles, cross_col):
            st.subheader("Distribution des echanges")
            fig_dist = go.Figure(
                go.Histogram(
                    x=profiles[cross_col].dropna().values,
                    nbinsx=60,
                    marker_color=COLORS["red"],
                    opacity=0.75,
                    hovertemplate="%{x:.0f} MW<br>%{y} obs<extra></extra>",
                )
            )
            fig_dist.update_layout(height=220, xaxis_title="MW", yaxis_title="Frequence")
            st.plotly_chart(fig_dist, use_container_width=True)

        if _has_data(profiles, load_col) and _has_data(profiles, solar_col):
            st.subheader("Solaire vs charge")
            sample = profiles[[load_col, solar_col]].dropna().tail(min(len(profiles), 24 * 60))
            fig_sc = go.Figure(
                go.Scatter(
                    x=sample[solar_col],
                    y=sample[load_col],
                    mode="markers",
                    marker=dict(size=6, color=COLORS["green"], opacity=0.35),
                    hovertemplate="Solaire %{x:.0f} MW<br>Charge %{y:.0f} MW<extra></extra>",
                )
            )
            fig_sc.update_layout(height=220, xaxis_title="Solaire MW", yaxis_title="Charge MW")
            st.plotly_chart(fig_sc, use_container_width=True)

with tab4:
    st.subheader("Lien avec le prix")
    if epex is None or epex.empty or "price_eur_mwh" not in epex.columns:
        st.info("Les donnees EPEX sont necessaires pour la lecture prix.")
    else:
        common = epex[["price_eur_mwh"]].join(view_h, how="inner")
        price_candidates = _available_series(common, [load_col, solar_col, wind_col, cross_col])
        if price_candidates:
            driver = st.selectbox("Driver a comparer au prix", price_candidates, index=0, format_func=_series_label)
            sample = common[[driver, "price_eur_mwh"]].dropna().iloc[-min(len(common), 1000) :]

            c1, c2 = st.columns(2)
            with c1:
                fig_scatter = go.Figure(
                    go.Scatter(
                        x=sample[driver],
                        y=sample["price_eur_mwh"],
                        mode="markers",
                        marker=dict(size=6, color=COLORS["amber"], opacity=0.45),
                        hovertemplate="%{x:.1f}<br>Prix %{y:.1f} EUR/MWh<extra>" + _series_label(driver) + "</extra>",
                    )
                )
                fig_scatter.update_layout(height=360, xaxis_title=_series_label(driver), yaxis_title="Prix EUR/MWh")
                st.plotly_chart(fig_scatter, use_container_width=True)
            with c2:
                rc = common["price_eur_mwh"].rolling(24 * 14).corr(common[driver])
                fig_corr = go.Figure(
                    go.Scatter(
                        x=rc.index,
                        y=rc.values,
                        line=dict(color=COLORS["blue"], width=2),
                        hovertemplate="%{x|%d/%m %H:%M}<br>Corr %{y:.2f}<extra></extra>",
                    )
                )
                fig_corr.add_hline(y=0, line_color=COLORS["muted"], line_width=1)
                fig_corr.update_layout(height=360, yaxis_title="Corr 14j glissante")
                st.plotly_chart(add_range_slider(fig_corr), use_container_width=True)

    st.subheader("Export client")
    export_csv_button(view_h.reset_index(), f"entsoe_data_hub_{zone.lower()}.csv", "Export ENTSO-E")

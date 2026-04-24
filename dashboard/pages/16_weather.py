"""
Page 16 — Meteo Haut-Valais
Weather showcase for local DSO clients.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    COLORS,
    WEATHER_LOCATIONS,
    add_range_slider,
    export_csv_button,
    load_weather_data,
    show_freshness_sidebar,
)

st.set_page_config(layout="wide")

st.header("Météo Haut-Valais")
st.caption(
    "Démonstration de données météo exploitables pour un GRD local: température, vent, précipitations, nuages et rayonnement."
)

show_freshness_sidebar()

with st.sidebar:
    st.subheader("Périmètre météo")
    locations = st.multiselect(
        "Localités",
        list(WEATHER_LOCATIONS.keys()),
        default=["Brig-Glis", "Visp", "Leuk", "Zermatt"],
    )
    forecast_days = st.slider("Horizon forecast (jours)", 3, 10, 7)
    primary_var = st.selectbox(
        "Variable principale",
        [
            "temperature_2m",
            "wind_speed_10m",
            "precipitation",
            "cloud_cover",
            "shortwave_radiation",
            "relative_humidity_2m",
        ],
        index=0,
    )

if not locations:
    st.info("Sélectionne au moins une localité.")
    st.stop()

bundle = load_weather_data(tuple(locations), forecast_days=forecast_days, past_days=2)
hourly = bundle.get("hourly", pd.DataFrame())
daily = bundle.get("daily", pd.DataFrame())

if hourly.empty:
    st.warning("Aucune donnée météo chargée depuis Open-Meteo.")
    st.stop()

latest_rows = (
    hourly.sort_index()
    .groupby("location", as_index=False)
    .tail(1)
    .set_index("location")
    .loc[locations]
)

st.markdown(
    """
    **Message client**  
    Cette page matérialise le type de service data que l’on peut exposer à un distributeur local:
    météo multi-sites, forecast horaire, indicateurs opérationnels et export immédiat.
    """
)

metric_cols = st.columns(min(4, len(locations)))
for i, loc in enumerate(locations[:4]):
    row = latest_rows.loc[loc]
    with metric_cols[i]:
        st.metric(
            loc,
            f"{row['temperature_2m']:.1f} °C",
            delta=f"Vent {row['wind_speed_10m']:.0f} km/h | Pluie {row['precipitation']:.1f} mm",
        )

st.subheader("Snapshot actuel")
snapshot = latest_rows[[
    "elevation_m", "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_gusts_10m", "precipitation", "cloud_cover", "shortwave_radiation",
]].rename(columns={
    "elevation_m": "Altitude m",
    "temperature_2m": "Temp °C",
    "relative_humidity_2m": "Humidité %",
    "wind_speed_10m": "Vent km/h",
    "wind_gusts_10m": "Rafales km/h",
    "precipitation": "Pluie mm/h",
    "cloud_cover": "Nuages %",
    "shortwave_radiation": "Rayonnement W/m²",
})
st.dataframe(snapshot.style.format("{:.1f}"), use_container_width=True)

tab1, tab2, tab3 = st.tabs(["Météogrammes", "Synthèse journalière", "Comparaison sites"])

with tab1:
    forecast_only = hourly[hourly.index >= pd.Timestamp.now(tz="Europe/Zurich").floor("h")].copy()
    label_map = {
        "temperature_2m": "Température (°C)",
        "wind_speed_10m": "Vent (km/h)",
        "precipitation": "Précipitation (mm/h)",
        "cloud_cover": "Couverture nuageuse (%)",
        "shortwave_radiation": "Rayonnement (W/m²)",
        "relative_humidity_2m": "Humidité (%)",
    }

    color_cycle = [COLORS["blue"], COLORS["amber"], COLORS["green"], COLORS["red"], COLORS["muted"]]
    left, right = st.columns([1.5, 1.0], gap="large")

    with left:
        st.subheader("Forecast horaire multi-sites")
        fig = go.Figure()
        for i, loc in enumerate(locations):
            loc_df = forecast_only[forecast_only["location"] == loc]
            fig.add_trace(go.Scatter(
                x=loc_df.index, y=loc_df[primary_var], name=loc,
                line=dict(color=color_cycle[i % len(color_cycle)], width=3),
                hovertemplate="%{x|%d/%m %H:%M}<br>%{y:.1f}<extra>" + loc + "</extra>",
            ))
        fig.update_layout(
            height=540,
            yaxis_title=label_map.get(primary_var, primary_var),
            legend=dict(orientation="h", y=1.08, x=0),
        )
        st.plotly_chart(add_range_slider(fig), use_container_width=True)

    with right:
        st.subheader("Lecture opérationnelle")
        ops = forecast_only.groupby("location").agg(
            temp_min=("temperature_2m", "min"),
            temp_max=("temperature_2m", "max"),
            rain_sum=("precipitation", "sum"),
            wind_max=("wind_gusts_10m", "max"),
            cloud_avg=("cloud_cover", "mean"),
        )
        ops = ops.rename(columns={
            "temp_min": "Tmin °C",
            "temp_max": "Tmax °C",
            "rain_sum": "Pluie cumulée mm",
            "wind_max": "Rafale max km/h",
            "cloud_avg": "Nuages moyens %",
        })
        st.dataframe(ops.style.format("{:.1f}"), use_container_width=True, height=240)

        if "precipitation" in forecast_only.columns:
            fig_rain = go.Figure()
            for i, loc in enumerate(locations):
                loc_df = forecast_only[forecast_only["location"] == loc]
                fig_rain.add_trace(go.Bar(
                    x=loc_df.index, y=loc_df["precipitation"], name=loc,
                    marker_color=color_cycle[i % len(color_cycle)],
                    opacity=0.6,
                ))
            fig_rain.update_layout(height=290, yaxis_title="mm/h", barmode="group")
            st.plotly_chart(add_range_slider(fig_rain), use_container_width=True)

with tab2:
    daily_view = daily[daily["location"].isin(locations)].copy()
    if daily_view.empty:
        st.info("Pas de synthèse journalière disponible.")
    else:
        daily_view["sunshine_h"] = daily_view["sunshine_duration"] / 3600.0

        left, right = st.columns([1.25, 1.0], gap="large")
        with left:
            st.subheader("Synthèse journalière")
            fig_tmax = go.Figure()
            for i, loc in enumerate(locations):
                loc_df = daily_view[daily_view["location"] == loc]
                fig_tmax.add_trace(go.Bar(
                    x=loc_df.index, y=loc_df["temperature_2m_max"],
                    name=f"{loc} Tmax", marker_color=color_cycle[i % len(color_cycle)],
                    opacity=0.8,
                ))
            fig_tmax.update_layout(height=420, yaxis_title="°C", barmode="group")
            st.plotly_chart(fig_tmax, use_container_width=True)

        with right:
            st.subheader("Carte pluie")
            pivot_rain = daily_view.pivot_table(values="precipitation_sum", index="location", columns=daily_view.index.date, aggfunc="mean")
            fig_heat = go.Figure(go.Heatmap(
                z=pivot_rain.values,
                x=[str(c) for c in pivot_rain.columns],
                y=pivot_rain.index,
                colorscale="Blues",
                colorbar_title="mm/j",
                hovertemplate="%{y}<br>%{x}<br>%{z:.1f} mm<extra></extra>",
            ))
            fig_heat.update_layout(height=420)
            st.plotly_chart(fig_heat, use_container_width=True)

        summary = daily_view[[
            "location", "temperature_2m_min", "temperature_2m_max",
            "precipitation_sum", "precipitation_hours", "sunshine_h", "wind_speed_10m_max",
        ]].rename(columns={
            "location": "Localité",
            "temperature_2m_min": "Tmin °C",
            "temperature_2m_max": "Tmax °C",
            "precipitation_sum": "Pluie mm/j",
            "precipitation_hours": "Heures pluie",
            "sunshine_h": "Ensoleillement h",
            "wind_speed_10m_max": "Vent max km/h",
        })
        st.dataframe(summary.style.format({
            "Tmin °C": "{:.1f}", "Tmax °C": "{:.1f}", "Pluie mm/j": "{:.1f}",
            "Heures pluie": "{:.1f}", "Ensoleillement h": "{:.1f}", "Vent max km/h": "{:.1f}",
        }), use_container_width=True)

with tab3:
    sample = forecast_only.copy()
    sample["heating_degree"] = np.maximum(0.0, 18.0 - sample["temperature_2m"])
    sample["cooling_degree"] = np.maximum(0.0, sample["temperature_2m"] - 22.0)

    comp = sample.groupby("location").agg(
        temp_avg=("temperature_2m", "mean"),
        rain_sum=("precipitation", "sum"),
        wind_avg=("wind_speed_10m", "mean"),
        cloud_avg=("cloud_cover", "mean"),
        radiation_avg=("shortwave_radiation", "mean"),
        hdh=("heating_degree", "sum"),
        cdh=("cooling_degree", "sum"),
    )
    comp = comp.rename(columns={
        "temp_avg": "Temp moyenne °C",
        "rain_sum": "Pluie cumulée mm",
        "wind_avg": "Vent moyen km/h",
        "cloud_avg": "Nuages moyens %",
        "radiation_avg": "Rayonnement moyen W/m²",
        "hdh": "Heating degree hours",
        "cdh": "Cooling degree hours",
    })
    left, right = st.columns([1.0, 1.2], gap="large")
    with left:
        st.subheader("Comparaison opérationnelle")
        st.dataframe(comp.style.format("{:.1f}"), use_container_width=True, height=320)

    with right:
        fig_compare = go.Figure()
        for i, metric in enumerate(["Temp moyenne °C", "Vent moyen km/h", "Pluie cumulée mm"]):
            fig_compare.add_trace(go.Bar(
                x=comp.index, y=comp[metric], name=metric,
                opacity=0.8,
            ))
        fig_compare.update_layout(height=360, barmode="group")
        st.plotly_chart(fig_compare, use_container_width=True)

    export_csv_button(hourly.reset_index(), "weather_hourly_haut_valais.csv", "Export météo horaire")

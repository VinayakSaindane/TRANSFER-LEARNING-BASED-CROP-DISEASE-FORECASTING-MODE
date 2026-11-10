"""Page 3: Real-Time Weather Intelligence via Open-Meteo."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from components.cards import render_callout, render_kpi_card
from components.charts import plot_humidity_dew_trend, plot_rainfall_chart, plot_temperature_trend
from components.styles import render_page_header
from services.weather_service import search_locations

PRESET_LOCATIONS = [
    {"name": "Nashik", "admin1": "Maharashtra", "country": "India", "latitude": 19.9975, "longitude": 73.7898},
    {"name": "Pune", "admin1": "Maharashtra", "country": "India", "latitude": 18.5204, "longitude": 73.8567},
    {"name": "Kolar", "admin1": "Karnataka", "country": "India", "latitude": 13.1367, "longitude": 78.1292},
    {"name": "Madanapalle", "admin1": "Andhra Pradesh", "country": "India", "latitude": 13.5503, "longitude": 78.5029},
    {"name": "Fresno", "admin1": "California", "country": "United States", "latitude": 36.7378, "longitude": -119.7871},
]


def render_weather_page(location: dict[str, Any], weather: dict[str, Any]) -> None:
    status = weather.get("status", "unavailable")
    badge_cls = "ag-pill-green" if "live" in status else ("ag-pill-amber" if "cache" in status else "ag-pill-red")
    loc_label = ", ".join(p for p in [location.get("name"), location.get("admin1"), location.get("country")] if p) or "Selected Location"
    render_page_header(
        "Weather Intelligence",
        "Location-level modelled weather conditions and 7-day NWP forecast powered by Open-Meteo Forecast & Geocoding APIs (not in-situ farm sensors).",
        badges=[
            (f"Status: {status.replace('_', ' ').title()}", badge_cls),
            (f"Location: {location.get('name', 'Plot')} ({location['latitude']:.3f}, {location['longitude']:.3f})", "ag-pill-slate"),
        ],
    )

    # Location Selector Card
    with st.container(border=True):
        st.markdown("**Select Farm Location (City Geocoding, Preset Belt, or Coordinates)**")
        s1, s2 = st.columns([1.4, 1.0])

        with s1:
            q = st.text_input("Search City / Town", placeholder="e.g., Nashik, Bengaluru, Salinas...")
            if st.button("🔎 Search Location", use_container_width=False):
                st.session_state["geo_search_result"] = search_locations(q)

            geo_res = st.session_state.get("geo_search_result")
            if geo_res:
                if geo_res["status"] == "ok" and geo_res["results"]:
                    options = {r["display_name"]: r for r in geo_res["results"]}
                    picked = st.selectbox("Select matching location", list(options.keys()))
                    if st.button("Apply Selected City", type="primary"):
                        st.session_state["selected_location"] = options[picked]
                        st.session_state["force_api_refresh"] = True
                        st.rerun()
                else:
                    st.warning(geo_res.get("error") or "No matching location found.")

        with s2:
            preset_labels = [f"{p['name']}, {p['admin1']}" for p in PRESET_LOCATIONS]
            chosen_preset = st.selectbox("Quick Tomato-Belt Presets", preset_labels)
            if st.button("Use Preset Location", use_container_width=True):
                idx = preset_labels.index(chosen_preset)
                st.session_state["selected_location"] = PRESET_LOCATIONS[idx]
                st.session_state["force_api_refresh"] = True
                st.rerun()

            with st.expander("Or enter custom Latitude / Longitude"):
                lat_in = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=float(location["latitude"]), format="%.4f")
                lon_in = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=float(location["longitude"]), format="%.4f")
                label_in = st.text_input("Custom Farm Label", value=location.get("name", "Custom Plot"))
                if st.button("Apply Coordinates"):
                    st.session_state["selected_location"] = {
                        "name": label_in or "Custom Plot",
                        "admin1": "",
                        "country": f"({lat_in:.2f}, {lon_in:.2f})",
                        "latitude": lat_in,
                        "longitude": lon_in,
                    }
                    st.session_state["force_api_refresh"] = True
                    st.rerun()

    if weather.get("error"):
        render_callout("Weather Data Notice", weather["error"], variant="warn" if weather.get("current") else "danger")

    cur = weather.get("current")
    if not cur or not weather.get("daily"):
        render_callout(
            "Weather Data Unavailable",
            "Live weather and forecast metrics could not be retrieved from Open-Meteo. No synthetic fallback values are displayed.",
            variant="danger",
        )
        return

    # Current Conditions KPIs
    st.markdown('<div class="ag-section-title">Current Location-Level Modelled Weather (Open-Meteo NWP Grid)</div>', unsafe_allow_html=True)
    render_callout(
        "Modelled Forecast vs. Farm Micro-Climate Distinction",
        "Open-Meteo delivers gridded numerical weather prediction (NWP) estimates for the selected coordinates — NOT direct on-farm IoT weather station or canopy sensor readings. Open-Meteo does NOT measure actual leaf wetness; surface soil temperature (0 cm) and moisture (0–1 cm) are modelled grid outputs, and canopy wetness favourability is approximated from dew-point depression and hours with RH >= 85%.",
        variant="info",
    )
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        app_t = cur.get("apparent_temperature")
        t2m = cur.get("temperature_2m")
        render_kpi_card(
            "Air Temperature",
            f"{t2m:.1f}°C" if t2m is not None else "N/A",
            f"Feels like {app_t:.1f}°C • {cur.get('weather_label', '')}" if app_t is not None else str(cur.get("weather_label", "")),
        )
    with k2:
        render_kpi_card(
            "Relative Humidity",
            f"{cur.get('relative_humidity_2m', 'N/A')}%",
            f"Cloud cover: {cur.get('cloud_cover', 'N/A')}%",
        )
    with k3:
        dp = cur.get("dew_point_2m")
        dep = cur.get("dew_point_depression")
        render_kpi_card(
            "Dew Point & Spread",
            f"{dp:.1f}°C" if dp is not None else "N/A",
            f"Dew-point depression: {dep:.1f}°C" if dep is not None else "Spread unavailable",
        )
    with k4:
        pr = cur.get("precipitation")
        render_kpi_card(
            "Current Precipitation",
            f"{pr:.2f} mm" if pr is not None else "N/A",
            f"Wind speed: {cur.get('wind_speed_10m', 'N/A')} km/h",
        )

    st.caption(
        f"Data Source: {weather.get('source', 'Open-Meteo Forecast API (Gridded Model Output)')} | "
        f"Location: {loc_label} ({float(location['latitude']):.4f}°, {float(location['longitude']):.4f}°) | "
        f"Timezone: {weather.get('timezone')} | Elevation: {weather.get('elevation')} m | "
        f"Retrieved (UTC): {weather.get('retrieved_at')}"
    )

    # Trend Charts
    daily = weather["daily"]
    c_a, c_b = st.columns(2)
    with c_a:
        st.plotly_chart(plot_temperature_trend(daily), use_container_width=True)
    with c_b:
        st.plotly_chart(plot_humidity_dew_trend(daily), use_container_width=True)

    st.plotly_chart(plot_rainfall_chart(daily), use_container_width=True)

    # Complete 7-Day Table including modelled Open-Meteo surface soil outputs
    st.markdown('<div class="ag-section-title">7-Day Modelled Daily Forecast &amp; Grid Surface Variables</div>', unsafe_allow_html=True)
    df_w = pd.DataFrame(
        [
            {
                "Date": d["date"],
                "Conditions": d["weather_label"],
                "Min °C": d["temp_min"],
                "Mean °C": d["temp_mean"],
                "Max °C": d["temp_max"],
                "Mean RH %": d["rh_mean"],
                "Hrs RH ≥ 85%": d["hours_rh_ge_85"],
                "Dew Pt Mean °C": d["dew_point_mean"],
                "Rain (mm)": d["precipitation_sum"],
                "Soil Temp 0cm (°C)": d["soil_temp_0cm_mean"],
                "Soil Moisture (m³/m³)": d["soil_moisture_0_1cm_mean"],
            }
            for d in daily
        ]
    )
    st.dataframe(df_w, use_container_width=True, hide_index=True)

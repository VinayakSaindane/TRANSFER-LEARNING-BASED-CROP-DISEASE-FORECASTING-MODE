"""AgriGuard AI — Main Streamlit Application Entry Point."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="AgriGuard AI — Crop Disease & Micro-Climate DSS",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.sidebar import render_sidebar
from components.styles import inject_custom_css
from config.settings import DEFAULT_LOCATION
from pages.advisory import render_advisory_page
from pages.climate import render_climate_page
from pages.disease_detection import render_disease_detection_page
from pages.model_performance import render_model_performance_page
from pages.overview import render_overview_page
from pages.research import render_research_page
from pages.risk_forecast import render_risk_forecast_page
from pages.weather import render_weather_page
from services.climate_service import fetch_climate_status
from services.risk_engine import assess_seven_day_forecast
from services.weather_service import fetch_weather_data


def _load_platform_feeds(location: dict, force_refresh: bool) -> tuple[dict, dict, dict]:
    """Fetch live/cached weather & NOAA ENSO feeds and evaluate 7-day disease risk."""
    weather = fetch_weather_data(
        latitude=float(location["latitude"]),
        longitude=float(location["longitude"]),
        location_meta=location,
        force_refresh=force_refresh,
    )
    climate = fetch_climate_status(force_refresh=force_refresh)
    risk = assess_seven_day_forecast(weather.get("daily") or [])
    return weather, climate, risk


def main() -> None:
    inject_custom_css()

    if "selected_location" not in st.session_state:
        st.session_state["selected_location"] = dict(DEFAULT_LOCATION)

    location = st.session_state["selected_location"]
    force_refresh = bool(st.session_state.pop("force_api_refresh", False))

    weather, climate, risk_assessment = _load_platform_feeds(location, force_refresh=force_refresh)

    active_page = render_sidebar(
        active_location=location,
        weather_status=str(weather.get("status", "unavailable")).replace("_", " ").title(),
        climate_status=str(climate.get("phase", "Unavailable")),
    )

    if active_page == "Overview":
        render_overview_page(location=location, weather=weather, risk_assessment=risk_assessment, climate=climate)
    elif active_page == "AI Disease Detection":
        render_disease_detection_page()
    elif active_page == "Weather Intelligence":
        render_weather_page(location=location, weather=weather)
    elif active_page == "Disease Risk Forecast":
        render_risk_forecast_page(weather=weather, risk_assessment=risk_assessment)
    elif active_page == "Climate Intelligence":
        render_climate_page(location=location, weather=weather, climate=climate)
    elif active_page == "Farmer Advisory":
        render_advisory_page(location=location, risk_assessment=risk_assessment)
    elif active_page == "Model Performance":
        render_model_performance_page()
    elif active_page == "About the Research":
        render_research_page()


if __name__ == "__main__":
    main()


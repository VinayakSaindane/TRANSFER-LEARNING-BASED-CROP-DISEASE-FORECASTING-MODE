"""Page 1: Overview Dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from components.cards import render_advisory_card, render_callout, render_kpi_card, risk_badge_class
from components.charts import plot_seven_day_risk_chart, plot_temperature_trend
from components.styles import render_page_header
from services.advisory_service import generate_farmer_advisories


def render_overview_page(
    location: dict[str, Any],
    weather: dict[str, Any],
    risk_assessment: dict[str, Any],
    climate: dict[str, Any],
) -> None:
    loc_label = ", ".join(p for p in [location.get("name"), location.get("admin1"), location.get("country")] if p)
    w_status = weather.get("status", "unavailable")
    c_status = climate.get("status", "unavailable")

    badges = [
        (f"📍 {location.get('name', 'Farm')}", "ag-pill-green"),
        (
            f"🌤️ Weather: {w_status.replace('_', ' ').title()}",
            "ag-pill-green" if "live" in w_status else ("ag-pill-amber" if "cache" in w_status else "ag-pill-red"),
        ),
        (
            f"🌊 ENSO: {climate.get('phase', 'Unavailable')}",
            "ag-pill-amber" if climate.get("phase") == "El Niño" else ("ag-pill-green" if c_status != "unavailable" else "ag-pill-slate"),
        ),
    ]
    render_page_header(
        "AgriGuard AI — Decision-Support Overview",
        f"Visual screening prototype and modelled location-level weather & disease-favourability monitoring for {loc_label}.",
        badges=badges,
    )

    # Top 4 KPI Cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        cur = weather.get("current")
        if cur and cur.get("temperature_2m") is not None:
            dp_val = cur.get("dew_point_2m")
            render_kpi_card(
                label="Current Weather",
                value=f"{cur['temperature_2m']:.1f}°C",
                subtitle=f"RH {cur.get('relative_humidity_2m', 'N/A')}% • Precip {cur.get('precipitation', 'N/A')} mm • {cur.get('weather_label', '')}",
                badge_text=f"Dew Pt {dp_val:.1f}°C" if dp_val is not None else "Dew Pt N/A",
                badge_class="ag-pill-green",
            )
        else:
            render_kpi_card(
                label="Current Weather",
                value="Unavailable",
                subtitle=weather.get("error") or "Open-Meteo feed could not be reached.",
                badge_text="Offline / Error",
                badge_class="ag-pill-red",
            )

    with c2:
        days = risk_assessment.get("days") or []
        if days and days[0].get("score") is not None:
            today = days[0]
            render_kpi_card(
                label="Today's Disease-Risk Index",
                value=f"{today['score']:.1f} / 100",
                subtitle=f"7-Day Mean: {risk_assessment['mean_score']}/100 • Peak: {risk_assessment['peak_day']['date']}",
                badge_text=f"{today['category']} Favourability",
                badge_class=risk_badge_class(today["category"]),
            )
        else:
            render_kpi_card(
                label="Today's Disease-Risk Index",
                value="Unavailable",
                subtitle="Requires valid local temperature, humidity, and rainfall inputs.",
                badge_text="No Weather Data",
                badge_class="ag-pill-slate",
            )

    with c3:
        if climate.get("latest_roni") is not None:
            render_kpi_card(
                label="NOAA ENSO / RONI Status",
                value=f"{climate['latest_roni']:+.2f}°C",
                subtitle=f"Season: {climate['latest_season_readable']} • {climate['strength']} tier",
                badge_text=climate["phase"],
                badge_class="ag-pill-amber" if climate["phase"] == "El Niño" else "ag-pill-green",
            )
        else:
            render_kpi_card(
                label="NOAA ENSO / RONI Status",
                value="Unavailable",
                subtitle=climate.get("error") or "Official NOAA CPC feed unreachable.",
                badge_text="Unavailable",
                badge_class="ag-pill-slate",
            )

    with c4:
        pred = st.session_state.get("latest_prediction")
        if pred and pred.get("status") == "ok":
            p_cls = "ag-pill-red" if pred["predicted_class"] == "Tomato Early Blight" else "ag-pill-green"
            render_kpi_card(
                label="Latest AI Leaf Screening",
                value=pred["predicted_class"],
                subtitle=f"Softmax: {pred['confidence']:.1%} (not diagnostic certainty) • {pred['certainty_label']}",
                badge_text="Session Scan Active",
                badge_class=p_cls,
            )
        else:
            render_kpi_card(
                label="Crop Leaf Monitoring",
                value="No Scan Yet",
                subtitle="Upload a tomato leaf image in AI Disease Detection to evaluate symptoms.",
                badge_text="Awaiting Leaf Image",
                badge_class="ag-pill-slate",
            )

    st.markdown("")
    # Quick Action Strip
    qa1, qa2, qa3 = st.columns([1.2, 1.2, 2.6])
    with qa1:
        if st.button("🍃 Scan Tomato Leaf", type="primary", use_container_width=True):
            st.session_state["active_page"] = "AI Disease Detection"
            st.rerun()
    with qa2:
        if st.button("📍 Change Farm Location", use_container_width=True):
            st.session_state["active_page"] = "Weather Intelligence"
            st.rerun()
    with qa3:
        if weather.get("retrieved_at"):
            st.caption(
                f"Source: {weather.get('source', 'Open-Meteo Forecast API')} | "
                f"Location: {loc_label} ({float(location.get('latitude', 0)):.3f}°, {float(location.get('longitude', 0)):.3f}°) | "
                f"Retrieved (UTC): {weather['retrieved_at']} | "
                "This is an experimental environmental disease-favourability index, not a validated probability of future infection."
            )

    # Charts Row
    if risk_assessment.get("status") == "ok" and weather.get("daily"):
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(plot_seven_day_risk_chart(risk_assessment["days"]), use_container_width=True)
        with ch2:
            st.plotly_chart(plot_temperature_trend(weather["daily"]), use_container_width=True)

        # Compact 7-day overview table
        st.markdown('<div class="ag-section-title">7-Day Modelled Weather &amp; Environmental Favourability Snapshot</div>', unsafe_allow_html=True)
        df_ov = pd.DataFrame(
            [
                {
                    "Date": d["date"],
                    "Condition": d["weather_label"],
                    "Mean Temp (°C)": d["temp_mean"],
                    "Mean RH (%)": d["rh_mean"],
                    "Rain (mm)": d["precipitation_sum"],
                    "Risk Index (/100)": d["score"],
                    "Category": d["category"],
                    "Primary Driver": d["primary_driver"],
                }
                for d in risk_assessment["days"]
            ]
        )
        st.dataframe(df_ov, use_container_width=True, hide_index=True)
    else:
        render_callout(
            "Weather & Risk Charts Unavailable",
            weather.get("error") or "Live weather data could not be loaded for this location. Use Refresh Live APIs in the sidebar or verify network connectivity.",
            variant="warn",
        )

    # Top Alerts & Advisories
    st.markdown('<div class="ag-section-title">Priority Preventive Alerts &amp; Advisories</div>', unsafe_allow_html=True)
    advisories = generate_farmer_advisories(
        risk_assessment=risk_assessment,
        latest_prediction=st.session_state.get("latest_prediction"),
    )
    for adv in advisories[:3]:
        render_advisory_card(adv)

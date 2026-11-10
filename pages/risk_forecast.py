"""Page 4: Explainable Disease Risk Forecast & What-If Climate Simulator."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from components.cards import render_callout, render_kpi_card, risk_badge_class
from components.charts import (
    plot_risk_factor_contributions,
    plot_seven_day_risk_chart,
    plot_what_if_comparison,
)
from components.styles import render_page_header
from services.risk_engine import simulate_what_if_scenario


def render_risk_forecast_page(weather: dict[str, Any], risk_assessment: dict[str, Any]) -> None:
    render_page_header(
        "Disease Risk Forecast & What-If Simulator",
        "Weighted environmental favourability model for Tomato Early Blight (Alternaria solani).",
        badges=[("Prototype Weights: RH 30% | Temp 25% | Rain 25% | Dew 20%", "ag-pill-green")],
    )

    render_callout(
        "Scientific Scope Clarification",
        "This is an experimental environmental disease-favourability index, not a validated probability of future infection. Prototype factor weights (30% RH / 25% Temp / 25% Rain / 20% Dew-Wetness Proxy) are literature-informed rather than locally field-calibrated, and inputs rely on location-level Open-Meteo NWP forecasts rather than in-situ leaf wetness sensors.",
        variant="warn",
    )

    if risk_assessment.get("status") != "ok" or not risk_assessment.get("days"):
        render_callout(
            "Risk Forecast Unavailable",
            "Forecast weather variables are unavailable for the selected location. Check Weather Intelligence or refresh live APIs.",
            variant="danger",
        )
        return

    days = risk_assessment["days"]
    today = days[0]
    peak = risk_assessment["peak_day"]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_kpi_card(
            "Day 1 Risk Index",
            f"{today['score']:.1f} / 100",
            f"Primary driver: {today['primary_driver']}",
            badge_text=today["category"],
            badge_class=risk_badge_class(today["category"]),
        )
    with m2:
        render_kpi_card(
            "7-Day Mean Index",
            f"{risk_assessment['mean_score']:.1f} / 100",
            "Average environmental favourability across 7 days",
        )
    with m3:
        render_kpi_card(
            "Peak Forecast Day",
            f"{peak['score']:.1f} / 100",
            f"Occurs on {peak['date']} (Day {peak['day_index']})",
            badge_text=peak["category"],
            badge_class=risk_badge_class(peak["category"]),
        )
    with m4:
        render_kpi_card(
            "High / Very High Days",
            f"{risk_assessment['high_risk_days']} / 7",
            "Days scoring ≥ 61/100 in 7-day window",
            badge_text="Early Warning Active" if risk_assessment["high_risk_days"] > 0 else "Within Low/Mod Band",
            badge_class="ag-pill-red" if risk_assessment["high_risk_days"] > 0 else "ag-pill-green",
        )

    st.markdown("")
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(plot_seven_day_risk_chart(days), use_container_width=True)
    with g2:
        st.plotly_chart(plot_risk_factor_contributions(days), use_container_width=True)

    st.markdown('<div class="ag-section-title">Daily Score Explanations &amp; Driver Changes</div>', unsafe_allow_html=True)
    df_risk = pd.DataFrame(
        [
            {
                "Day": f"Day {d['day_index']}",
                "Date": d["date"],
                "Score (/100)": d["score"],
                "Category": d["category"],
                "Change vs Prev": f"{d['delta_from_prev']:+.1f}" if d["delta_from_prev"] is not None else "—",
                "Explanation": d["explanation"],
                "Shift Analysis": d["change_explanation"],
            }
            for d in days
        ]
    )
    st.dataframe(df_risk, use_container_width=True, hide_index=True)

    with st.expander("View Scoring Formula & Agronomic Thresholds"):
        st.markdown(
            """
| Factor | Prototype Weight | Favourability Basis |
|---|---:|---|
| **Relative Humidity** | 30% | Steep ramp above 65% RH; peak suitability at ≥ 92% RH supporting sporulation & germination. |
| **Temperature Suitability** | 25% | Peak suitability (`1.0`) in the 24–29°C *Alternaria solani* optimum; drops below 12°C or above 35°C. |
| **Rainfall (Splash)** | 25% | Increases splash dispersal from 0 to 25+ mm; does **not** alone confirm persistent leaf wetness. |
| **Dew / Wetness Proxy** | 20% | Uses dew-point depression (T − Td ≤ 2.5°C) and daily hours with RH ≥ 85%. |

**Categories:** `0–30 Low` | `31–60 Moderate` | `61–80 High` | `81–100 Very High`
"""
        )

    # Section 12: What-If Climate Simulator
    st.markdown("---")
    st.markdown('<div class="ag-section-title">What-If Micro-Climate Simulator</div>', unsafe_allow_html=True)
    render_callout(
        "Hypothetical Scenario Tool (Same 30/25/25/20 Scoring Engine)",
        "Adjust hypothetical temperature, relative humidity, and rainfall below to test sensitivity against a baseline forecast day. Both baseline and hypothetical scores execute the identical compute_environmental_risk() weighted equations (with dew point estimated via Magnus-Tetens from the adjusted Temperature and RH sliders). Results represent a hypothetical scenario, NOT an actual weather forecast or climate projection.",
        variant="warn",
    )

    daily_raw = weather.get("daily") or []
    day_options = [f"Day {i+1} ({d['date']})" for i, d in enumerate(daily_raw)]
    selected_idx = st.selectbox("Select Baseline Forecast Day", range(len(day_options)), format_func=lambda i: day_options[i])
    base_day = daily_raw[selected_idx]

    sl1, sl2, sl3 = st.columns(3)
    with sl1:
        sim_temp = st.slider(
            "Hypothetical Mean Temperature (°C)",
            min_value=5.0,
            max_value=42.0,
            value=float(base_day.get("temp_mean") or 25.0),
            step=0.5,
        )
    with sl2:
        sim_rh = st.slider(
            "Hypothetical Relative Humidity (%)",
            min_value=15.0,
            max_value=100.0,
            value=float(base_day.get("rh_mean") or 70.0),
            step=1.0,
        )
    with sl3:
        sim_rain = st.slider(
            "Hypothetical Daily Rainfall (mm)",
            min_value=0.0,
            max_value=50.0,
            value=min(50.0, float(base_day.get("precipitation_sum") or 0.0)),
            step=0.5,
        )

    sim = simulate_what_if_scenario(base_day, sim_temp_c=sim_temp, sim_rh_pct=sim_rh, sim_rain_mm=sim_rain)
    base_res = sim["baseline"]
    hyp_res = sim["hypothetical"]

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        b_inputs = base_res.get("inputs") or {}
        render_kpi_card(
            "Baseline Forecast Score",
            f"{base_res['score']:.1f} / 100" if base_res.get("score") is not None else "Unavailable",
            (
                f"T={b_inputs.get('temperature_c', 'N/A')}°C, "
                f"RH={b_inputs.get('relative_humidity_pct', 'N/A')}%, "
                f"Rain={b_inputs.get('precipitation_mm', 'N/A')} mm"
            ),
            badge_text=base_res["category"],
            badge_class=risk_badge_class(base_res["category"]),
        )
    with sc2:
        render_kpi_card(
            "Hypothetical Scenario Score",
            f"{hyp_res['score']:.1f} / 100" if hyp_res.get("score") is not None else "Unavailable",
            f"T={sim_temp:.1f}°C, RH={sim_rh:.0f}%, Rain={sim_rain:.1f} mm",
            badge_text=hyp_res["category"],
            badge_class=risk_badge_class(hyp_res["category"]),
        )
    with sc3:
        delta = sim["score_delta"] if sim["score_delta"] is not None else 0.0
        delta_cls = "ag-pill-red" if delta > 5 else ("ag-pill-green" if delta < -5 else "ag-pill-slate")
        render_kpi_card(
            "Simulated Difference",
            f"{delta:+.1f} pts",
            sim["label"],
            badge_text="Risk Amplified" if delta > 0 else ("Risk Reduced" if delta < 0 else "Unchanged"),
            badge_class=delta_cls,
        )

    st.plotly_chart(plot_what_if_comparison(base_res, hyp_res), use_container_width=True)

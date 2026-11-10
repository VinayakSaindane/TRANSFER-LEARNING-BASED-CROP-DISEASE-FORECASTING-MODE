"""Page 6: Preventive Farmer Advisory Engine."""

from __future__ import annotations

from typing import Any

import streamlit as st

from components.cards import render_advisory_card, render_callout, render_kpi_card, risk_badge_class
from components.styles import render_page_header
from services.advisory_service import generate_farmer_advisories


def render_advisory_page(
    location: dict[str, Any],
    risk_assessment: dict[str, Any],
) -> None:
    pred = st.session_state.get("latest_prediction")
    has_scan = bool(pred and pred.get("status") == "ok")

    render_page_header(
        "Farmer Advisory & Preventive Guidance",
        f"Actionable cultural and preventive recommendations derived from {location.get('name', 'farm')} weather & AI screening.",
        badges=[
            ("Non-Chemical Preventive Focus", "ag-pill-green"),
            ("AI Scan Linked" if has_scan else "Weather-Only Mode", "ag-pill-green" if has_scan else "ag-pill-slate"),
        ],
    )

    c1, c2 = st.columns(2)
    with c1:
        days = risk_assessment.get("days") or []
        if days and days[0].get("score") is not None:
            peak = risk_assessment["peak_day"]
            render_kpi_card(
                "Current / Peak 7-Day Risk",
                f"{days[0]['score']:.1f} → {peak['score']:.1f}",
                f"Today: {days[0]['category']} | Peak on {peak['date']} ({peak['category']})",
                badge_text=peak["category"],
                badge_class=risk_badge_class(peak["category"]),
            )
        else:
            render_kpi_card("Environmental Risk", "Unavailable", "Forecast weather feed unavailable.", badge_text="Unavailable", badge_class="ag-pill-slate")

    with c2:
        if has_scan:
            render_kpi_card(
                "Linked AI Leaf Screening",
                pred["predicted_class"],
                f"Softmax: {pred['confidence']:.1%} (not diagnostic certainty) — {pred['certainty_label']}",
                badge_text=pred["predicted_class"],
                badge_class="ag-pill-red" if pred["predicted_class"] == "Tomato Early Blight" else "ag-pill-green",
            )
        else:
            render_kpi_card(
                "Linked AI Leaf Screening",
                "No Leaf Scan Yet",
                "Scan a tomato leaf in AI Disease Detection to tailor advisories with symptom status.",
                badge_text="Optional Enhancement",
                badge_class="ag-pill-slate",
            )

    st.caption(
        "This is an experimental environmental disease-favourability index, not a validated probability of future infection. "
        "Modelled forecast inputs from Open-Meteo do not directly measure farm-level leaf wetness."
    )

    st.markdown('<div class="ag-section-title">Active Preventive Recommendations</div>', unsafe_allow_html=True)
    advisories = generate_farmer_advisories(risk_assessment=risk_assessment, latest_prediction=pred)
    for adv in advisories:
        render_advisory_card(adv)

    with st.container(border=True):
        st.markdown("**Field Scouting Checklist for Tomato Early Blight (*Alternaria solani*)**")
        st.markdown(
            """
1. **Start at the Lower Canopy:** Early blight almost always begins on older, shaded lower leaves closest to soil splash.
2. **Look for Concentric Target Rings:** Inspect dark brown necrotic spots for characteristic concentric ridges ("bull's-eye" pattern) surrounded by a yellow chlorotic halo.
3. **Time Irrigation Carefully:** Water at the root zone (drip/furrow) in early morning hours rather than overhead sprinklers in late afternoon.
4. **Scout 24–48 Hours After Rain:** Check plots following rain or dense morning dew, particularly when temperatures stay between 24°C and 29°C.
"""
        )

    render_callout(
        "Agronomic Safety & Non-Prescription Policy",
        "AgriGuard AI provides educational and preventive crop management guidance only. It does NOT prescribe specific fungicides, chemical concentrations, or application rates. Always consult your local Krishi Vigyan Kendra (KVK), state agricultural university, or certified agronomist for locally approved disease management protocols.",
        variant="warn",
    )


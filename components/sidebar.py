"""Clean SaaS sidebar navigation and active farm status panel."""

from __future__ import annotations

import html
import streamlit as st

from components.styles import get_logo_base64

NAV_ITEMS = [
    ("Overview", "📊"),
    ("AI Disease Detection", "🍃"),
    ("Weather Intelligence", "🌤️"),
    ("Disease Risk Forecast", "📈"),
    ("Climate Intelligence", "🌊"),
    ("Farmer Advisory", "🌾"),
    ("Model Performance", "🧠"),
    ("About the Research", "📚"),
]


def render_sidebar(active_location: dict, weather_status: str, climate_status: str) -> str:
    """Render branded navigation sidebar and return the active page name."""
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Overview"

    logo_b64 = get_logo_base64()
    with st.sidebar:
        if logo_b64:
            st.markdown(
                f"""
<div style="display:flex;align-items:center;gap:0.75rem;padding:0.25rem 0.2rem 0.9rem 0.2rem;border-bottom:1px solid #E2EBE4;margin-bottom:0.85rem;">
  <img src="data:image/svg+xml;base64,{logo_b64}" width="40" height="40" alt="AgriGuard AI"/>
  <div>
    <div style="font-size:1.08rem;font-weight:800;color:#1B5E20;line-height:1.1;">AgriGuard AI</div>
    <div style="font-size:0.74rem;color:#526558;">Micro-Climate &amp; Blight DSS</div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#64748B;margin-bottom:0.4rem;padding-left:0.25rem;">Navigation</div>',
            unsafe_allow_html=True,
        )

        current_page = st.session_state["active_page"]
        for page_name, icon in NAV_ITEMS:
            btn_type = "primary" if current_page == page_name else "secondary"
            if st.button(f"{icon}  {page_name}", key=f"nav_{page_name}", type=btn_type, use_container_width=True):
                st.session_state["active_page"] = page_name
                st.rerun()

        st.markdown("---")
        loc_display = html.escape(
            ", ".join(p for p in [active_location.get("name"), active_location.get("country")] if p)
        )
        safe_weather_status = html.escape(weather_status)
        safe_climate_status = html.escape(climate_status)
        st.markdown(
            f"""
<div style="background:#F6FAF6;border:1px solid #DCE6DE;border-radius:12px;padding:0.8rem 0.9rem;font-size:0.82rem;">
  <div style="font-weight:700;color:#1B5E20;margin-bottom:0.25rem;">Active Farm Location</div>
  <div style="color:#142419;font-weight:600;">{loc_display}</div>
  <div style="color:#64748B;font-size:0.75rem;margin-bottom:0.55rem;">
    {float(active_location.get('latitude', 0)):.3f}°, {float(active_location.get('longitude', 0)):.3f}°
  </div>
  <div style="display:flex;flex-direction:column;gap:0.25rem;font-size:0.75rem;color:#475569;">
    <span>Weather Feed: <strong>{safe_weather_status}</strong></span>
    <span>NOAA ENSO: <strong>{safe_climate_status}</strong></span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        if st.button("🔄 Refresh Live APIs", use_container_width=True):
            st.session_state["force_api_refresh"] = True
            st.rerun()

    return st.session_state["active_page"]

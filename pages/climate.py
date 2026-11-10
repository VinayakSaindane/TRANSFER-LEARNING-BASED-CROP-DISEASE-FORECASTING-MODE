"""Page 5: El Niño / ENSO Climate Intelligence via Official NOAA CPC Indices."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from components.cards import render_callout, render_kpi_card
from components.charts import plot_enso_history_chart
from components.styles import render_page_header
from services.climate_service import get_enso_regional_context
from services.weather_service import fetch_historical_seasonal_baseline


def render_climate_page(location: dict[str, Any], weather: dict[str, Any], climate: dict[str, Any]) -> None:
    status = climate.get("status", "unavailable")
    phase = climate.get("phase", "Unavailable")
    phase_badge = (
        "ag-pill-amber" if phase == "El Niño" else ("ag-pill-green" if status != "unavailable" else "ag-pill-slate")
    )

    render_page_header(
        "El Niño / ENSO Climate Intelligence",
        "Official NOAA Climate Prediction Center (CPC) RONI & ONI monitoring with seasonal regional context.",
        badges=[
            (f"ENSO Phase: {phase}", phase_badge),
            (f"Source Status: {status.replace('_', ' ').title()}", "ag-pill-slate"),
        ],
    )

    if climate.get("error"):
        render_callout(
            "NOAA Data Retrieval Notice",
            climate["error"],
            variant="warn" if climate.get("latest_roni") is not None else "danger",
        )

    if climate.get("latest_roni") is None:
        render_callout(
            "Official NOAA ENSO Data Unavailable",
            "AgriGuard AI could not fetch official RONI/ONI observations from NOAA CPC and no verified cache is available. ENSO values are never fabricated.",
            variant="danger",
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            "NOAA RONI (Primary Index)",
            f"{climate['latest_roni']:+.2f}°C",
            f"Relative Niño 3.4 anomaly (1991–2020 base; tropical mean subtracted)",
            badge_text=f"{climate['strength']} {phase}",
            badge_class=phase_badge,
        )
    with c2:
        oni_val = climate.get("latest_oni")
        oni_streak = climate.get("oni_consecutive_seasons")
        oni_sub = f"Season: {climate.get('latest_oni_season') or climate['latest_season']}"
        if oni_streak:
            oni_sub += f" • {oni_streak} consec. season(s) >= |0.5|°C"
        render_kpi_card(
            "Traditional ONI (Reference)",
            f"{oni_val:+.2f}°C" if oni_val is not None else "Unavailable",
            oni_sub,
        )
    with c3:
        render_kpi_card(
            "Latest Observation Season",
            climate["latest_season_readable"],
            f"Code: {climate['latest_season']} • RONI streak: {climate.get('consecutive_seasons', 0)}/5 seasons",
            badge_text="Sustained 5+ Seasons" if climate.get("sustained_roni_episode") else "Developing / Context-Confirmed",
            badge_class="ag-pill-green" if climate.get("sustained_roni_episode") else "ag-pill-amber",
        )
    with c4:
        src_upd = str(climate.get("source_updated_at") or "")[:19].replace("T", " ")
        ret = str(climate.get("retrieved_at") or "")[:19].replace("T", " ")
        cpc_alert = climate.get("cpc_alert_status") or "N/A"
        cpc_dt = climate.get("cpc_issued_date") or "N/A"
        render_kpi_card(
            "NOAA Source Update (UTC)",
            src_upd or ret or "N/A",
            f"CPC Alert: {cpc_alert} ({cpc_dt}) • Retrieved: {ret}",
            badge_text="Live Verified" if "live" in status else "Cached Observation",
            badge_class="ag-pill-green" if "live" in status else "ag-pill-amber",
        )

    st.markdown("")
    render_callout(
        "Active Classification Basis & Episode Status",
        f"{climate.get('episode_status', '')} — {climate.get('classification_basis', '')}",
        variant="info",
    )
    st.plotly_chart(
        plot_enso_history_chart(climate.get("roni_series", []), climate.get("oni_series", [])),
        use_container_width=True,
    )

    # Critical scientific separation disclaimer
    context = get_enso_regional_context(phase, country=location.get("country", ""))
    render_callout("Critical Scientific Boundary: Climate Context vs. Local Disease Risk", context["scientific_caveat"], variant="warn")

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        with st.container(border=True):
            st.markdown("**NOAA CPC Sustained-Threshold Classification Methodology**")
            st.markdown(
                f"""
- **RONI (Relative Oceanic Niño Index):** Operational NOAA CPC index (ERSSTv6, 1991–2020 base) subtracting the global tropical mean (20°N–20°S) SST anomaly from Niño 3.4 (5°N–5°S, 120°–170°W), variance-scaled to Niño 3.4 (L'Heureux et al., 2024).
- **Sustained Episode Rule (5+ Seasons):** Per [NOAA CPC RONI]({climate.get('methodology_url', 'https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/')}), El Niño (`>= +0.5°C`) and La Niña (`<= -0.5°C`) historical episodes require **a minimum of 5 consecutive overlapping 3-month seasons**. A **single overlapping season** crossing `±0.5°C` does NOT alone classify an episode.
- **Official Climate Context Check:** When 2–4 consecutive RONI seasons exceed `±0.5°C` in real time, AgriGuard AI cross-checks NOAA CPC's official ENSO Alert System status (`{climate.get('cpc_alert_status') or 'N/A'}`, issued `{climate.get('cpc_issued_date') or 'N/A'}`) and 5-season reference ONI persistence (`{climate.get('oni_consecutive_seasons', 0)}` consecutive seasons).
- **Update & Revision Notice:** NOAA updates RONI by the 5th of each month; recent ERSSTv6 values remain real-time estimates subject to revision for up to two months.
- **Source Attribution:** NOAA Climate Prediction Center (`RONI.ascii.txt`, `oni.ascii.txt`, `ensodisc.shtml`).
"""
            )
    with col_exp2:
        with st.container(border=True):
            st.markdown(f"**Climate-to-Weather Connection for {location.get('name', 'Selected Location')}**")
            st.write(context["global_mechanism"])
            st.write(context["regional_context"])

    # Local 5-Year Historical Comparison via Open-Meteo Archive
    loc_label = ", ".join(p for p in [location.get("name"), location.get("admin1"), location.get("country")] if p) or "Selected Location"
    st.markdown('<div class="ag-section-title">Local Forecast vs. Recent 5-Year Historical Comparison (Not a 30-Year Climatological Normal)</div>', unsafe_allow_html=True)
    baseline = fetch_historical_seasonal_baseline(location["latitude"], location["longitude"])
    daily = weather.get("daily") or []

    if baseline.get("status") == "ok" and daily:
        fc_temps = [d["temp_mean"] for d in daily if d.get("temp_mean") is not None]
        fc_rain = sum(d["precipitation_sum"] or 0.0 for d in daily)
        fc_rhs = [d["rh_mean"] for d in daily if d.get("rh_mean") is not None]

        fc_temp_mean = round(sum(fc_temps) / len(fc_temps), 2) if fc_temps else None
        fc_rh_mean = round(sum(fc_rhs) / len(fc_rhs), 2) if fc_rhs else None
        yrs_lbl = f"{baseline['years_sampled'][0]}–{baseline['years_sampled'][-1]}" if baseline.get("years_sampled") else "5 yrs"

        b1, b2, b3 = st.columns(3)
        with b1:
            t_diff = round(fc_temp_mean - baseline["mean_temp_c"], 2) if (fc_temp_mean is not None and baseline["mean_temp_c"] is not None) else None
            render_kpi_card(
                "7-Day Mean Temp vs. Recent 5-Yr Avg",
                f"{fc_temp_mean}°C" if fc_temp_mean is not None else "N/A",
                f"Recent {yrs_lbl} comparison ({baseline['window_label']}): {baseline['mean_temp_c']}°C ({t_diff:+.1f}°C vs. 5-yr avg)" if t_diff is not None else "No comparison",
            )
        with b2:
            r_diff = round(fc_rain - baseline["mean_7day_precip_mm"], 2) if baseline["mean_7day_precip_mm"] is not None else None
            render_kpi_card(
                "7-Day Total Rain vs. Recent 5-Yr Avg",
                f"{fc_rain:.1f} mm",
                f"Recent {yrs_lbl} 7-day avg: {baseline['mean_7day_precip_mm']} mm ({r_diff:+.1f} mm vs. 5-yr avg)" if r_diff is not None else "No comparison",
            )
        with b3:
            rh_diff = round(fc_rh_mean - baseline["mean_rh_pct"], 1) if (fc_rh_mean is not None and baseline["mean_rh_pct"] is not None) else None
            render_kpi_card(
                "7-Day Mean RH vs. Recent 5-Yr Avg",
                f"{fc_rh_mean:.1f}%" if fc_rh_mean is not None else "N/A",
                f"Recent {yrs_lbl} RH avg: {baseline['mean_rh_pct']}% ({rh_diff:+.1f}% vs. 5-yr avg)" if rh_diff is not None else "No comparison",
            )

        st.caption(
            f"Source: {baseline.get('source', 'Open-Meteo Historical Archive API')} (gridded reanalysis/model archive; "
            "recent 5-year historical comparison, NOT a 30-year WMO climatological normal) | "
            f"Location: {loc_label} ({float(location['latitude']):.4f}°, {float(location['longitude']):.4f}°) | "
            f"Years: {baseline.get('years_sampled')} | Retrieved (UTC): {baseline.get('retrieved_at')}"
        )

        with st.expander("View Year-by-Year Recent Historical Comparison Window (Open-Meteo Archive)"):
            st.dataframe(pd.DataFrame(baseline["yearly_records"]), use_container_width=True, hide_index=True)
    else:
        render_callout(
            "Historical Local Baseline Unavailable",
            baseline.get("error") or "Historical comparison requires Open-Meteo Archive access and current forecast records.",
            variant="info",
        )

"""Reusable SaaS card, badge, and alert components."""

from __future__ import annotations

import html
import streamlit as st


def risk_badge_class(category: str) -> str:
    cat = (category or "").lower()
    if cat == "low" or cat == "healthy scan":
        return "ag-pill-green"
    if cat == "moderate" or "uncertain" in cat:
        return "ag-pill-amber"
    if cat in {"high", "very high"} or "alert" in cat:
        return "ag-pill-red"
    return "ag-pill-slate"


def render_kpi_card(
    label: str,
    value: str,
    subtitle: str = "",
    badge_text: str | None = None,
    badge_class: str = "ag-pill-green",
) -> None:
    """Render a rounded white KPI card with optional status badge."""
    safe_label = html.escape(label)
    safe_val = html.escape(value)
    safe_sub = html.escape(subtitle)
    badge_html = (
        f'<span class="ag-pill {badge_class}" style="margin-bottom:0.4rem;">{html.escape(badge_text)}</span>'
        if badge_text
        else ""
    )
    st.markdown(
        f"""
<div class="ag-card">
  <div class="ag-metric-label">{safe_label}</div>
  <div class="ag-metric-value">{safe_val}</div>
  {badge_html}
  <div class="ag-metric-sub">{safe_sub}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_callout(title: str, body: str, variant: str = "info") -> None:
    """Render a clean alert / informational callout box."""
    cls = "ag-callout"
    if variant == "warn":
        cls = "ag-callout-warn"
    elif variant == "danger":
        cls = "ag-callout-danger"

    st.markdown(
        f"""
<div class="{cls}">
  <div style="font-weight:700;margin-bottom:0.25rem;color:#142419;">{html.escape(title)}</div>
  <div style="font-size:0.9rem;color:#334155;line-height:1.45;">{html.escape(body)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_advisory_card(advisory: dict[str, str]) -> None:
    """Render a structured farmer advisory card showing trigger reason, forecast day, and action."""
    badge_cls = risk_badge_class(advisory.get("risk_level", ""))
    prio_cls = (
        "ag-pill-red"
        if advisory.get("priority") == "Urgent"
        else ("ag-pill-amber" if "Action" in advisory.get("priority", "") else "ag-pill-green")
    )
    st.markdown(
        f"""
<div class="ag-advisory-card">
  <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:0.5rem;margin-bottom:0.55rem;">
    <div style="font-size:1.02rem;font-weight:700;color:#134417;">{html.escape(advisory['title'])}</div>
    <div style="display:flex;gap:0.4rem;flex-wrap:wrap;">
      <span class="ag-pill {prio_cls}">{html.escape(advisory['priority'])}</span>
      <span class="ag-pill {badge_cls}">{html.escape(advisory['risk_level'])}</span>
      <span class="ag-pill ag-pill-slate">{html.escape(advisory['relevant_forecast_day'])}</span>
    </div>
  </div>
  <div style="font-size:0.88rem;color:#475569;margin-bottom:0.45rem;">
    <strong>Trigger Reason:</strong> {html.escape(advisory['reason'])}
  </div>
  <div style="font-size:0.92rem;color:#142419;background:#F6FAF6;border:1px solid #DCE6DE;border-radius:10px;padding:0.7rem 0.85rem;">
    <strong style="color:#1B5E20;">Recommended Preventive Action:</strong> {html.escape(advisory['recommended_action'])}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


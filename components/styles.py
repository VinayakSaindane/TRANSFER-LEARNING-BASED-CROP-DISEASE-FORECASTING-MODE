"""Custom CSS design system for the AgriGuard AI green-and-white SaaS dashboard."""

from __future__ import annotations

import base64
import html
from pathlib import Path
import streamlit as st

from config.settings import ASSETS_DIR


def get_logo_base64() -> str:
    logo_path = ASSETS_DIR / "logo" / "agriguard_logo.svg"
    if logo_path.exists():
        return base64.b64encode(logo_path.read_bytes()).decode("ascii")
    return ""


def inject_custom_css() -> None:
    """Inject modern agricultural SaaS styling and remove default Streamlit clutter."""
    st.markdown(
        """
<style>
:root {
    --ag-bg: #F6F8F5;
    --ag-surface: #FFFFFF;
    --ag-surface-alt: #F0F5F0;
    --ag-border: #DCE6DE;
    --ag-primary: #1B5E20;
    --ag-primary-dark: #134417;
    --ag-accent: #2E7D32;
    --ag-soft-green: #E8F5E9;
    --ag-text: #142419;
    --ag-muted: #526558;
    --ag-shadow: 0 2px 12px rgba(20, 36, 25, 0.05);
}

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    color: var(--ag-text);
}

.stApp {
    background-color: var(--ag-bg);
}

/* Hide default footer and deploy button; keep header compact */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton,
[data-testid="stAppDeployButton"] {display: none !important;}
header[data-testid="stHeader"] {
    background: rgba(246, 248, 245, 0.85);
    backdrop-filter: blur(6px);
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1320px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid var(--ag-border);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.1rem !important;
}

/* Navigation buttons inside sidebar */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] button[kind="secondary"] {
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    background-color: transparent;
    color: var(--ag-text);
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 0.55rem 0.85rem;
    font-weight: 500;
    font-size: 0.92rem;
    transition: background-color 0.15s ease, color 0.15s ease;
}

section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] button[kind="secondary"]:hover {
    background-color: var(--ag-soft-green);
    color: var(--ag-primary);
    border-color: #C8E6C9;
}

section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] button[kind="primary"] {
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    background-color: var(--ag-primary);
    color: #FFFFFF;
    border: 1px solid var(--ag-primary);
    border-radius: 10px;
    padding: 0.55rem 0.85rem;
    font-weight: 600;
    font-size: 0.92rem;
    box-shadow: 0 2px 8px rgba(27, 94, 32, 0.20);
}

/* Top Hero / Header Card */
.ag-page-header {
    background: linear-gradient(135deg, #FFFFFF 0%, #F1F8F2 100%);
    border: 1px solid var(--ag-border);
    border-left: 5px solid var(--ag-primary);
    border-radius: 14px;
    padding: 1.15rem 1.4rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--ag-shadow);
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.85rem;
}

.ag-page-title {
    margin: 0;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--ag-primary-dark);
    letter-spacing: -0.015em;
}

.ag-page-subtitle {
    margin: 0.2rem 0 0 0;
    font-size: 0.90rem;
    color: var(--ag-muted);
}

.ag-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.28rem 0.72rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid transparent;
    white-space: normal;
    max-width: 100%;
    line-height: 1.25;
}

.ag-pill-green {
    background-color: #E8F5E9;
    color: #1B5E20;
    border-color: #C8E6C9;
}

.ag-pill-amber {
    background-color: #FEF3C7;
    color: #92400E;
    border-color: #FDE68A;
}

.ag-pill-red {
    background-color: #FEE2E2;
    color: #991B1B;
    border-color: #FECACA;
}

.ag-pill-slate {
    background-color: #F1F5F9;
    color: #334155;
    border-color: #CBD5E1;
}

/* Metric Cards */
.ag-card {
    background: var(--ag-surface);
    border: 1px solid var(--ag-border);
    border-radius: 14px;
    padding: 1.05rem 1.15rem;
    box-shadow: var(--ag-shadow);
    height: 100%;
}

.ag-metric-label {
    font-size: 0.76rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--ag-muted);
    margin-bottom: 0.35rem;
}

.ag-metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--ag-text);
    line-height: 1.15;
    margin-bottom: 0.35rem;
}

.ag-metric-sub {
    font-size: 0.82rem;
    color: var(--ag-muted);
    line-height: 1.35;
}

/* Section containers */
.ag-section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--ag-primary-dark);
    margin: 0.6rem 0 0.75rem 0;
}

.ag-callout {
    background: #FFFFFF;
    border: 1px solid var(--ag-border);
    border-left: 4px solid var(--ag-accent);
    border-radius: 12px;
    padding: 0.95rem 1.1rem;
    margin: 0.65rem 0;
    box-shadow: var(--ag-shadow);
}

.ag-callout-warn {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #D97706;
    border-radius: 12px;
    padding: 0.95rem 1.1rem;
    margin: 0.65rem 0;
}

.ag-callout-danger {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-left: 4px solid #DC2626;
    border-radius: 12px;
    padding: 0.95rem 1.1rem;
    margin: 0.65rem 0;
}

.ag-advisory-card {
    background: #FFFFFF;
    border: 1px solid var(--ag-border);
    border-radius: 14px;
    padding: 1.05rem 1.2rem;
    margin-bottom: 0.85rem;
    box-shadow: var(--ag-shadow);
}

/* Make primary action buttons look crisp */
div.stButton > button[kind="primary"] {
    background-color: var(--ag-primary);
    border-color: var(--ag-primary);
    border-radius: 10px;
    font-weight: 600;
}
div.stButton > button[kind="primary"]:hover {
    background-color: var(--ag-primary-dark);
    border-color: var(--ag-primary-dark);
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str, badges: list[tuple[str, str]] | None = None) -> None:
    """Render a consistent SaaS page header banner with optional status pills."""
    badge_html = ""
    if badges:
        pills = []
        for label, style_cls in badges:
            pills.append(f'<span class="ag-pill {html.escape(style_cls)}">{html.escape(label)}</span>')
        badge_html = f'<div style="display:flex;flex-wrap:wrap;gap:0.45rem;">{"".join(pills)}</div>'

    st.markdown(
        f"""
<div class="ag-page-header">
  <div>
    <h1 class="ag-page-title">{html.escape(title)}</h1>
    <p class="ag-page-subtitle">{html.escape(subtitle)}</p>
  </div>
  {badge_html}
</div>
""",
        unsafe_allow_html=True,
    )

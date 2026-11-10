"""End-to-end Streamlit UI smoke and interaction tests across all 8 application pages."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


ALL_PAGES = [
    "Overview",
    "AI Disease Detection",
    "Weather Intelligence",
    "Disease Risk Forecast",
    "Climate Intelligence",
    "Farmer Advisory",
    "Model Performance",
    "About the Research",
]


def test_all_eight_pages_render_without_exceptions() -> None:
    at = AppTest.from_file("app.py", default_timeout=25)
    at.run()
    assert not at.exception, f"Initial app load failed: {at.exception}"

    for page_name in ALL_PAGES:
        nav_btn = next(b for b in at.sidebar.button if b.key == f"nav_{page_name}")
        nav_btn.click().run()
        assert not at.exception, f"Page '{page_name}' raised exception: {at.exception}"

        if page_name == "Disease Risk Forecast":
            rendered = " ".join(m.value for m in at.markdown)
            assert "experimental environmental disease-favourability index" in rendered.lower()
        elif page_name == "Model Performance":
            rendered = " ".join(m.value for m in at.markdown)
            assert "52.38%" in rendered and "4.76%" in rendered
        elif page_name == "AI Disease Detection":
            rendered = " ".join(m.value for m in at.markdown)
            assert "not reliable for unverified field diagnosis" in rendered.lower()
        elif page_name == "Climate Intelligence":
            rendered = " ".join(m.value for m in at.markdown)
            assert "5 consecutive overlapping" in rendered.lower() or "roni" in rendered.lower()
            assert "not a 30-year climatological normal" in rendered.lower()


def test_what_if_simulator_and_advisory_flow() -> None:
    at = AppTest.from_file("app.py", default_timeout=25)
    at.run()
    assert not at.exception

    # Navigate to Disease Risk Forecast and adjust What-If simulator sliders
    next(b for b in at.sidebar.button if b.key == "nav_Disease Risk Forecast").click().run()
    assert not at.exception
    if len(at.slider) >= 3:
        at.slider[0].set_value(26.5)  # Optimal Alternaria temperature
        at.slider[1].set_value(92.0)  # High RH
        at.slider[2].set_value(15.0)  # Measurable rainfall
        at.run()
        assert not at.exception

    # Navigate to AI Disease Detection and pick a held-out sample if available
    next(b for b in at.sidebar.button if b.key == "nav_AI Disease Detection").click().run()
    assert not at.exception
    if at.selectbox:
        opts = list(at.selectbox[0].options)
        if len(opts) > 1:
            at.selectbox[0].select(opts[1]).run()
            assert not at.exception
            analyze_btn = next((b for b in at.button if "Diagnosis" in b.label), None)
            if analyze_btn is not None:
                analyze_btn.click().run()
                assert not at.exception
                assert "latest_prediction" in at.session_state
                assert at.session_state["latest_prediction"]["status"] == "ok"

                # Verify Farmer Advisory reflects the active AI scan without error
                next(b for b in at.sidebar.button if b.key == "nav_Farmer Advisory").click().run()
                assert not at.exception

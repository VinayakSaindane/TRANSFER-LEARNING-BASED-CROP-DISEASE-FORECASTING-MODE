"""Page 2: AI Disease Detection using MobileNetV2."""

from __future__ import annotations

import streamlit as st

from components.cards import render_callout, render_kpi_card
from components.styles import render_page_header
from components.upload import render_image_input_panel
from models.predict import get_loaded_model, predict_leaf_image


def render_disease_detection_page() -> None:
    _, model_info = get_loaded_model()
    model_ready = model_info.get("status") == "ready"
    badge = ("Model Ready (MobileNetV2)", "ag-pill-green") if model_ready else ("Model Missing", "ag-pill-red")

    render_page_header(
        "AI Disease Detection",
        "Prototype transfer-learning screening of tomato leaf images for Healthy vs. Tomato Early Blight (Alternaria solani).",
        badges=[badge, ("Input: 224×224 RGB", "ag-pill-slate"), ("Experimental Prototype — Not Field-Certified", "ag-pill-amber")],
    )

    render_callout(
        "Diagnostic Safety, Domain-Shift & Scope Warning",
        "This model is NOT reliable for unverified field diagnosis. Trained on PlantVillage studio images (99.23% held-out test accuracy), zero-shot evaluation on cluttered PlantDoc field images achieved 52.38% balanced accuracy and 4.76% Healthy specificity (60/63 healthy field leaves misclassified as Tomato Early Blight). High softmax confidence is NOT guaranteed diagnostic certainty, and this binary prototype cannot detect late blight, Septoria leaf spot, bacterial spot, viral infections, or nutrient deficiencies.",
        variant="warn",
    )

    if not model_ready:
        render_callout(
            "Trained Model Unavailable",
            model_info.get("error", "Run 'python3 models/train.py' to train and save the MobileNetV2 model."),
            variant="danger",
        )
        return

    col_left, col_right = st.columns([1.05, 1.15], gap="large")

    with col_left:
        st.markdown('<div class="ag-section-title">1. Leaf Image Input</div>', unsafe_allow_html=True)
        file_bytes, filename = render_image_input_panel()

        if file_bytes is not None:
            st.image(file_bytes, caption=f"Preview: {filename}", use_container_width=True)
            analyze_clicked = st.button("🔍 Run MobileNetV2 Diagnosis", type="primary", use_container_width=True)
            if analyze_clicked:
                with st.spinner("Validating image and running MobileNetV2 inference..."):
                    result = predict_leaf_image(file_bytes, filename=filename or "upload.jpg")
                st.session_state["latest_prediction"] = result
        else:
            st.info("Upload a tomato leaf image above or pick a held-out evaluation sample to run inference.")

    with col_right:
        st.markdown('<div class="ag-section-title">2. Screening Result &amp; Uncertainty Guidance</div>', unsafe_allow_html=True)
        result = st.session_state.get("latest_prediction")

        if not result:
            render_callout(
                "Awaiting Image Analysis",
                "Select or upload a tomato leaf photograph on the left and click 'Run MobileNetV2 Diagnosis' to inspect class probabilities and preventive guidance.",
            )
            return

        status = result.get("status")
        if status == "rejected":
            render_callout(
                "Input Rejected / Retake Recommended",
                result.get("error", "The uploaded image did not pass validation."),
                variant="warn",
            )
            st.markdown(
                "**Photography Guidance:** Capture a single tomato leaf centered in clear natural light against a clean, non-cluttered background."
            )
            return

        if status != "ok":
            render_callout("Inference Error", result.get("error", "Prediction failed."), variant="danger")
            return

        pred_cls = result["predicted_class"]
        conf = result["confidence"]
        probs = result["class_probabilities"]

        r1, r2 = st.columns(2)
        with r1:
            pill_cls = "ag-pill-red" if pred_cls == "Tomato Early Blight" else "ag-pill-green"
            render_kpi_card(
                label="Predicted Class",
                value=pred_cls,
                subtitle=f"Original size: {result['original_size'][0]}×{result['original_size'][1]} px • Not diagnostic certainty",
                badge_text="Verify Visually" if pred_cls == "Tomato Early Blight" else "Healthy Pattern",
                badge_class=pill_cls,
            )
        with r2:
            render_kpi_card(
                label="Softmax Confidence Score",
                value=f"{conf:.1%}",
                subtitle="Two-class softmax output only — NOT guaranteed diagnostic certainty.",
                badge_text="Low Confidence (<65%)" if result["low_confidence"] else "Softmax ≥ 65% (Verify Visually)",
                badge_class="ag-pill-amber" if (result["low_confidence"] or result["moderate_confidence"] or pred_cls == "Tomato Early Blight") else "ag-pill-green",
            )

        st.markdown("")
        st.markdown("**Class Probability Breakdown (Softmax Distribution)**")
        for cls_name, p_val in probs.items():
            st.write(f"**{cls_name}** — {p_val:.1%}")
            st.progress(min(max(float(p_val), 0.0), 1.0))

        variant = "warn" if (result["low_confidence"] or result["moderate_confidence"] or pred_cls == "Tomato Early Blight") else "info"
        render_callout("Uncertainty, Domain-Shift & Scope Notice", result["uncertainty_warning"], variant=variant)
        render_callout("Classification Explanation", result["explanation"], variant="info")
        render_callout("Recommended Next Step", result["next_step"], variant="info")


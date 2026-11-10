"""Inference pipeline for AgriGuard AI MobileNetV2 disease detection."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import (
    ACTIVE_KERAS_BACKEND,
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    STRONG_CONFIDENCE_THRESHOLD,
)
from models.preprocessing import validate_uploaded_image
import keras


@lru_cache(maxsize=1)
def get_loaded_model() -> tuple[keras.Model | None, dict[str, Any]]:
    """Load the saved MobileNetV2 model once and cache it in memory."""
    if not MODEL_PATH.exists():
        return None, {
            "status": "missing_model",
            "error": (
                f"Trained model file not found at '{MODEL_PATH}'. "
                "Train the model using 'python3 models/train.py' before running predictions."
            ),
        }

    try:
        model = keras.saving.load_model(MODEL_PATH)
        meta: dict[str, Any] = {}
        if MODEL_METADATA_PATH.exists():
            meta = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        return model, {"status": "ready", "backend": ACTIVE_KERAS_BACKEND, "metadata": meta}
    except Exception as exc:  # noqa: BLE001
        return None, {"status": "load_error", "error": f"Failed to load saved model: {exc}"}


def predict_leaf_image(file_bytes: bytes, filename: str = "upload.jpg") -> dict[str, Any]:
    """Validate an uploaded image and run MobileNetV2 inference with uncertainty guardrails."""
    validation = validate_uploaded_image(file_bytes, filename=filename)
    if not validation["valid"]:
        return {
            "status": "rejected",
            "unsuitable": validation.get("unsuitable", False),
            "error": validation["error"],
            "image": validation.get("image"),
            "foliar_check": validation.get("foliar_check"),
        }

    model, model_info = get_loaded_model()
    if model is None:
        return {
            "status": "model_unavailable",
            "error": model_info.get("error", "Trained model is unavailable."),
            "image": validation["image"],
        }

    meta = model_info.get("metadata") or {}
    classes = meta.get("classes") or CLASS_NAMES

    probs = model.predict(validation["tensor"], verbose=0)[0]
    probs = np.asarray(probs, dtype=np.float64)
    pred_idx = int(np.argmax(probs))
    predicted_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    class_probs = {cls: round(float(probs[i]), 4) for i, cls in enumerate(classes)}
    low_confidence = confidence < CONFIDENCE_THRESHOLD
    moderate_confidence = CONFIDENCE_THRESHOLD <= confidence < STRONG_CONFIDENCE_THRESHOLD

    if low_confidence:
        certainty_label = "Low Softmax Score — Uncertain"
        uncertainty_warning = (
            f"Top softmax probability ({confidence:.1%}) is below the {CONFIDENCE_THRESHOLD:.0%} threshold. "
            "Model confidence is NOT diagnostic certainty. Retake the photograph with a single tomato leaf "
            "centered against a clean neutral background in diffused daylight, and verify visually in the field."
        )
    elif moderate_confidence:
        certainty_label = "Moderate Softmax Score — Verify Visually"
        uncertainty_warning = (
            f"Softmax score ({confidence:.1%}) is moderate. Model confidence reflects two-class separation "
            "between studio-trained 'Healthy' and 'Tomato Early Blight' only — NOT guaranteed diagnostic certainty. "
            "Cluttered field backgrounds can bias predictions toward Tomato Early Blight (4.76% specificity on PlantDoc)."
        )
    else:
        certainty_label = "High Softmax Score — Not Diagnostic Certainty"
        uncertainty_warning = (
            "Important: High softmax confidence is NOT guaranteed diagnostic certainty. In external PlantDoc field "
            "testing, cluttered soil/canopy backgrounds caused 60/63 healthy field leaves to be predicted as Early Blight "
            "(4.76% specificity; 52.38% balanced accuracy). Visually confirm concentric target rings and remember this "
            "binary prototype does not classify other tomato diseases."
        )

    if predicted_class == "Tomato Early Blight":
        explanation = (
            f"MobileNetV2 extracted foliar visual features (lesion texture, necrotic/chlorotic contrast) "
            f"that scored {confidence:.1%} toward 'Tomato Early Blight' vs {class_probs.get('Healthy', 0.0):.1%} "
            "toward 'Healthy'."
        )
        next_step = (
            "Inspect lower canopy leaves for characteristic dark brown concentric rings with yellow halos; "
            "remove severely affected lower foliage away from healthy rows, avoid overhead irrigation, and "
            "consult local agricultural extension guidance."
        )
    else:
        explanation = (
            f"MobileNetV2 extracted uniform green foliar features that scored {confidence:.1%} toward "
            f"'Healthy' vs {class_probs.get('Tomato Early Blight', 0.0):.1%} toward 'Tomato Early Blight'."
        )
        next_step = (
            "Continue routine twice-weekly scouting of lower tomato leaves—especially if the 7-day "
            "weather risk forecast shows Moderate, High, or Very High environmental favourability."
        )

    return {
        "status": "ok",
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "class_probabilities": class_probs,
        "low_confidence": low_confidence,
        "moderate_confidence": moderate_confidence,
        "certainty_label": certainty_label,
        "uncertainty_warning": uncertainty_warning,
        "explanation": explanation,
        "next_step": next_step,
        "original_size": validation["original_size"],
        "foliar_check": validation["foliar_check"],
        "image": validation["image"],
        "backend": model_info.get("backend"),
    }


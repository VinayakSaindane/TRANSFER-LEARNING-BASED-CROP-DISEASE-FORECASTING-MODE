"""Reusable leaf image upload and held-out test sample selector component."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import streamlit as st

from config.settings import BASE_DIR, PLANTDOC_MANIFEST_PATH, SPLIT_MANIFEST_PATH


def _load_demo_samples() -> list[dict[str, str]]:
    """Return a few genuine held-out test images (never from training split) for quick verification."""
    samples: list[dict[str, str]] = []
    if SPLIT_MANIFEST_PATH.exists():
        with SPLIT_MANIFEST_PATH.open("r", encoding="utf-8") as f:
            test_rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
        for label in ("Healthy", "Tomato Early Blight"):
            match = next((r for r in test_rows if r["label"] == label), None)
            if match:
                samples.append(
                    {
                        "label": f"PlantVillage Test — {label}",
                        "path": str(BASE_DIR / match["processed_path"]),
                    }
                )
    if PLANTDOC_MANIFEST_PATH.exists():
        with PLANTDOC_MANIFEST_PATH.open("r", encoding="utf-8") as f:
            pd_rows = list(csv.DictReader(f))
        for label in ("Healthy", "Tomato Early Blight"):
            match = next((r for r in pd_rows if r["label"] == label), None)
            if match:
                samples.append(
                    {
                        "label": f"PlantDoc Field Eval — {label}",
                        "path": str(BASE_DIR / match["processed_path"]),
                    }
                )
    return samples


def render_image_input_panel() -> tuple[bytes | None, str | None]:
    """Render file upload dropzone + optional held-out sample picker."""
    uploaded = st.file_uploader(
        "Upload a tomato leaf image (JPG, PNG, WEBP, BMP — max 15 MB)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="For best accuracy, capture a single tomato leaf in natural daylight against a simple background.",
    )
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name

    samples = _load_demo_samples()
    if samples:
        with st.expander("Optional: Load a held-out evaluation sample (never used in training)"):
            choice = st.selectbox(
                "Choose held-out sample",
                ["— None —"] + [s["label"] for s in samples],
                index=0,
            )
            if choice != "— None —":
                selected = next(s for s in samples if s["label"] == choice)
                p = Path(selected["path"])
                if p.exists():
                    return p.read_bytes(), p.name

    return None, None


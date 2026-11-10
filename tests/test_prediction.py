"""Tests for MobileNetV2 model loading, preprocessing, valid inference, and rejection guardrails."""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from config.settings import BASE_DIR, CLASS_NAMES, EVAL_RESULTS_PATH, MODEL_PATH, SPLIT_MANIFEST_PATH
from models.predict import get_loaded_model, predict_leaf_image


def test_model_file_and_eval_metrics_exist() -> None:
    assert MODEL_PATH.exists(), "Saved MobileNetV2 model must exist"
    assert EVAL_RESULTS_PATH.exists(), "evaluation_metrics.json must exist"
    model, info = get_loaded_model()
    assert model is not None
    assert info["status"] == "ready"

    eval_data = json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    assert "plantvillage_test" in eval_data
    assert 0.0 <= eval_data["plantvillage_test"]["accuracy"] <= 1.0

    pd_eval = eval_data["plantdoc_external_eval"]
    assert eval_data["classes"] == ["Healthy", "Tomato Early Blight"]
    assert pd_eval["confusion_matrix"] == [[3, 60], [0, 88]]
    assert pd_eval["specificity"] == round(3 / 63, 4)
    assert pd_eval["sensitivity"] == 1.0
    assert pd_eval["balanced_accuracy"] == round((1.0 + 3 / 63) / 2, 4)
    assert pd_eval["prediction_distribution"]["predicted_counts"] == {
        "Healthy": 3,
        "Tomato Early Blight": 148,
    }


def test_keras_backend_configured_before_importing_keras() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; os.environ.pop('KERAS_BACKEND', None); "
                "import models.predict; "
                "from config.settings import ACTIVE_KERAS_BACKEND; "
                "import keras; "
                "assert os.environ.get('KERAS_BACKEND') == ACTIVE_KERAS_BACKEND; "
                "assert keras.backend.backend() == ACTIVE_KERAS_BACKEND"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_valid_held_out_image_prediction() -> None:
    with SPLIT_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        test_row = next(r for r in csv.DictReader(f) if r["split"] == "test")
    img_bytes = (BASE_DIR / test_row["processed_path"]).read_bytes()
    res = predict_leaf_image(img_bytes, filename=Path(test_row["processed_path"]).name)
    assert res["status"] == "ok"
    assert res["predicted_class"] in CLASS_NAMES
    assert 0.0 <= res["confidence"] <= 1.0
    assert abs(sum(res["class_probabilities"].values()) - 1.0) < 0.01
    assert res["uncertainty_warning"]
    assert "diagnostic certainty" in res["uncertainty_warning"].lower()
    assert "not" in res["uncertainty_warning"].lower()
    assert res["next_step"]


def test_invalid_and_unsuitable_inputs_rejected_safely() -> None:
    # 1. Corrupted bytes
    bad = predict_leaf_image(b"not an image", filename="broken.jpg")
    assert bad["status"] == "rejected"

    # 2. Too small image (<64x64)
    tiny_buf = io.BytesIO()
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(tiny_buf, format="PNG")
    tiny = predict_leaf_image(tiny_buf.getvalue(), filename="tiny.png")
    assert tiny["status"] == "rejected"

    # 3. Solid blank image
    blank_buf = io.BytesIO()
    Image.fromarray(np.full((224, 224, 3), 240, dtype=np.uint8)).save(blank_buf, format="PNG")
    blank = predict_leaf_image(blank_buf.getvalue(), filename="blank.png")
    assert blank["status"] == "rejected"

    # 4. Non-foliar blue image
    blue_arr = np.random.randint(0, 30, (224, 224, 3), dtype=np.uint8) + np.array([20, 60, 180], dtype=np.uint8)
    blue_buf = io.BytesIO()
    Image.fromarray(blue_arr).save(blue_buf, format="PNG")
    non_leaf = predict_leaf_image(blue_buf.getvalue(), filename="sky.png")
    assert non_leaf["status"] == "rejected"
    assert non_leaf["unsuitable"] is True


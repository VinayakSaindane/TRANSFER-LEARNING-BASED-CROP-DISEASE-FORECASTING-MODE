"""Shared image preprocessing, validation, and botanical screening utilities."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from config.settings import (
    ACTIVE_KERAS_BACKEND,  # noqa: F401 - ensures KERAS_BACKEND env var is configured before importing keras
    ALLOWED_IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    MAX_UPLOAD_SIZE_MB,
    MIN_IMAGE_DIMENSION,
)
import keras


def preprocess_rgb_array(rgb_array: np.ndarray) -> np.ndarray:
    """Apply MobileNetV2 preprocessing identically across training, evaluation, and inference."""
    arr = rgb_array.astype(np.float32)
    return keras.applications.mobilenet_v2.preprocess_input(arr)


def load_and_preprocess_path(path: str | Path, target_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load an image file from disk, resize cleanly to 224x224 RGB, and preprocess for MobileNetV2."""
    with Image.open(path) as img:
        rgb = ImageOps.exif_transpose(img).convert("RGB")
        if rgb.size != target_size:
            rgb = rgb.resize(target_size, Image.Resampling.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32)
    return preprocess_rgb_array(arr)


def check_foliar_characteristics(rgb_image: Image.Image) -> dict[str, Any]:
    """Inspect basic image statistics and foliar color presence to screen unsuitable inputs."""
    arr = np.asarray(rgb_image, dtype=np.uint8)
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray_std = float(np.std(gray))
    gray_mean = float(np.mean(gray))

    if gray_std < 6.0:
        return {
            "suitable": False,
            "reason": "Image appears blank, solid-colored, or extremely low contrast.",
            "foliar_ratio": 0.0,
            "gray_std": round(gray_std, 2),
        }
    if gray_mean < 12.0 or gray_mean > 248.0:
        return {
            "suitable": False,
            "reason": "Image is severely underexposed or overexposed (nearly pure black or pure white).",
            "foliar_ratio": 0.0,
            "gray_std": round(gray_std, 2),
        }

    aspect = max(w / max(h, 1), h / max(w, 1))
    if aspect > 4.5:
        return {
            "suitable": False,
            "reason": f"Extreme aspect ratio ({aspect:.1f}:1). Upload a standard leaf photograph.",
            "foliar_ratio": 0.0,
            "gray_std": round(gray_std, 2),
        }

    # Check presence of organic plant hues (green chlorophyll, yellow chlorosis, olive/brown lesions)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]   # OpenCV H in [0, 179]
    sat = hsv[:, :, 1]   # S in [0, 255]
    val = hsv[:, :, 2]   # V in [0, 255]

    plant_mask = (
        (sat >= 22)
        & (val >= 20)
        & (val <= 245)
        & ((hue >= 10) & (hue <= 95))
    )
    foliar_ratio = float(np.mean(plant_mask))

    if foliar_ratio < 0.05:
        return {
            "suitable": False,
            "reason": (
                f"Only {foliar_ratio:.1%} plant-like leaf coloration detected. "
                "This image does not appear to show a crop leaf; please retake or upload a clear tomato leaf photo."
            ),
            "foliar_ratio": round(foliar_ratio, 4),
            "gray_std": round(gray_std, 2),
        }

    return {
        "suitable": True,
        "reason": "Image passed quality and foliar screening.",
        "foliar_ratio": round(foliar_ratio, 4),
        "gray_std": round(gray_std, 2),
    }


def validate_uploaded_image(
    file_bytes: bytes,
    filename: str = "upload.jpg",
) -> dict[str, Any]:
    """Validate uploaded bytes, dimensions, format, and foliar suitability for inference."""
    if not file_bytes:
        return {"valid": False, "error": "Uploaded file is empty.", "image": None}

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        return {
            "valid": False,
            "error": f"File size ({size_mb:.1f} MB) exceeds maximum limit of {MAX_UPLOAD_SIZE_MB} MB.",
            "image": None,
        }

    ext = Path(filename).suffix.lower().lstrip(".")
    if ext and ext not in ALLOWED_IMAGE_EXTENSIONS:
        return {
            "valid": False,
            "error": (
                f"Unsupported file extension '.{ext}'. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}."
            ),
            "image": None,
        }

    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()
        with Image.open(io.BytesIO(file_bytes)) as img2:
            rgb = ImageOps.exif_transpose(img2).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return {
            "valid": False,
            "error": f"Unable to decode image safely ({exc}). Please upload a valid JPEG or PNG leaf photo.",
            "image": None,
        }

    w, h = rgb.size
    if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
        return {
            "valid": False,
            "error": (
                f"Image resolution ({w}x{h}) is too small; minimum required is "
                f"{MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels."
            ),
            "image": None,
        }

    foliar_check = check_foliar_characteristics(rgb)
    if not foliar_check["suitable"]:
        return {
            "valid": False,
            "unsuitable": True,
            "error": foliar_check["reason"],
            "foliar_check": foliar_check,
            "image": rgb,
        }

    resized = rgb.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    batch_tensor = np.expand_dims(preprocess_rgb_array(np.asarray(resized, dtype=np.float32)), axis=0)

    return {
        "valid": True,
        "unsuitable": False,
        "error": None,
        "original_size": (w, h),
        "foliar_check": foliar_check,
        "image": rgb,
        "tensor": batch_tensor,
    }

"""Centralized configuration and backend bootstrap for AgriGuard AI."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def configure_keras_backend() -> str:
    """Select the Keras 3 backend without hard-failing if TensorFlow is unavailable."""
    configured = os.environ.get("KERAS_BACKEND", "").strip().lower()
    if configured in {"tensorflow", "torch", "jax"}:
        return configured
    # Prefer TensorFlow when installed cleanly on the host; otherwise use PyTorch.
    if importlib.util.find_spec("tensorflow") is not None:
        backend = "tensorflow"
    elif importlib.util.find_spec("torch") is not None:
        backend = "torch"
    else:
        backend = "tensorflow"
    os.environ["KERAS_BACKEND"] = backend
    return backend


ACTIVE_KERAS_BACKEND = configure_keras_backend()

# Directory Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
CACHE_DIR = DATA_DIR / "cache"

MODELS_DIR = BASE_DIR / "models"
MODEL_SAVED_DIR = MODELS_DIR / "saved"
DOCS_DIR = BASE_DIR / "docs"
ASSETS_DIR = BASE_DIR / "assets"

for _dir in (
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    CACHE_DIR,
    MODEL_SAVED_DIR,
    DOCS_DIR,
    ASSETS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)

# Saved ML Artifacts
MODEL_PATH = MODEL_SAVED_DIR / "agriguard_mobilenetv2.keras"
CHECKPOINT_PATH = MODEL_SAVED_DIR / "agriguard_checkpoint.keras"
MODEL_METADATA_PATH = MODEL_SAVED_DIR / "model_metadata.json"

# Report Paths
DATASET_STATS_PATH = REPORTS_DIR / "dataset_statistics.json"
SPLIT_MANIFEST_PATH = PROCESSED_DATA_DIR / "splits.csv"
PLANTDOC_MANIFEST_PATH = PROCESSED_DATA_DIR / "plantdoc_eval.csv"
EVAL_RESULTS_PATH = REPORTS_DIR / "evaluation_metrics.json"
TRAINING_HISTORY_PATH = REPORTS_DIR / "training_history.json"
CONFUSION_MATRIX_IMG_PATH = REPORTS_DIR / "confusion_matrix.png"
TRAINING_CURVES_IMG_PATH = REPORTS_DIR / "training_curves.png"

# Model Hyperparameters & Preprocessing
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Healthy", "Tomato Early Blight"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

RANDOM_SEED = 42
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

DEFAULT_BATCH_SIZE = 32
DEFAULT_HEAD_EPOCHS = 6
DEFAULT_FINETUNE_EPOCHS = 6
DEFAULT_HEAD_LR = 1e-3
DEFAULT_FINETUNE_LR = 5e-5
FINETUNE_UNFREEZE_LAYERS = 25
DROPOUT_RATE = 0.30

# Inference Guardrails
CONFIDENCE_THRESHOLD = 0.65
STRONG_CONFIDENCE_THRESHOLD = 0.85
MIN_IMAGE_DIMENSION = 64
MAX_UPLOAD_SIZE_MB = 15
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

# Weather & Climate Endpoints
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

NOAA_RONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt"
NOAA_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

API_TIMEOUT_SECONDS = int(os.environ.get("AGRIGUARD_API_TIMEOUT", "12"))
WEATHER_CACHE_TTL_MINUTES = int(os.environ.get("AGRIGUARD_WEATHER_CACHE_MINUTES", "30"))
CLIMATE_CACHE_TTL_HOURS = int(os.environ.get("AGRIGUARD_CLIMATE_CACHE_HOURS", "12"))

DEFAULT_LOCATION = {
    "name": os.environ.get("AGRIGUARD_DEFAULT_CITY", "Nashik"),
    "admin1": "Maharashtra",
    "country": "India",
    "latitude": float(os.environ.get("AGRIGUARD_DEFAULT_LAT", "19.9975")),
    "longitude": float(os.environ.get("AGRIGUARD_DEFAULT_LON", "73.7898")),
    "timezone": "Asia/Kolkata",
}

# Disease Risk Engine (Environmental Favourability Weights — Initial Prototype)
RISK_WEIGHTS = {
    "relative_humidity": 0.30,
    "temperature": 0.25,
    "rainfall": 0.25,
    "dew_wetness": 0.20,
}

RISK_THRESHOLDS = [
    (0, 30, "Low", "#2E7D32", "#E8F5E9"),
    (31, 60, "Moderate", "#D97706", "#FEF3C7"),
    (61, 80, "High", "#DC2626", "#FEE2E2"),
    (81, 100, "Very High", "#7F1D1D", "#FECACA"),
]


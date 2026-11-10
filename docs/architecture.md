# AgriGuard AI System Architecture

## 1. Modular Software Architecture

AgriGuard AI follows a clean four-layer Python architecture separated across configuration, data/ML, domain services, and Streamlit UI components:

```text
AgriGuard-AI/
├── app.py                     # Main Streamlit entry point & page router
├── config/settings.py         # Paths, API URLs, model constants, risk weights, Keras backend selection
├── data/
│   ├── prepare_datasets.py    # Reproducible acquisition, verification, deduplication & leaf-grouped splitting
│   ├── raw/                   # Sparse checkouts of PlantVillage & PlantDoc
│   ├── processed/             # Resized 224x224 train/val/test/plantdoc splits + CSV manifests
│   ├── cache/                 # Disk TTL caches for Open-Meteo & NOAA CPC responses
│   └── reports/               # Verified JSON statistics, evaluation metrics & PNG figures
├── models/
│   ├── preprocessing.py       # Shared MobileNetV2 normalization & foliar input validation
│   ├── train.py               # 2-stage transfer-learning trainer (head + fine-tuning + resume)
│   ├── evaluate.py            # PlantVillage & PlantDoc zero-shot evaluation + charts
│   ├── predict.py             # Cached inference loader + uncertainty & suitability gating
│   └── saved/                 # Saved .keras model, checkpoint & model_metadata.json
├── services/
│   ├── weather_service.py     # Open-Meteo Forecast, Geocoding & Archive baseline client
│   ├── climate_service.py     # NOAA CPC RONI & ONI ASCII parser and phase classifier
│   ├── risk_engine.py         # Explainable 0-100 risk scoring & What-If simulator
│   └── advisory_service.py    # Non-chemical preventive farmer advisory generator
├── components/                # Custom CSS, KPI cards, Plotly charts, sidebar & upload widget
├── pages/                     # 8 dedicated modular dashboard views
├── tests/                     # Pytest suite covering risk, weather, climate, data, ML & UI
└── docs/                      # Research references, methodology, datasets & implementation docs
```

## 2. Data & Control Flow

1. **Startup & API Sync (`app.py` → `services/`):**
   - `selected_location` defaults to Nashik, Maharashtra, India (`19.9975°N, 73.7898°E`) and updates dynamically via Open-Meteo Geocoding or coordinates.
   - `weather_service.fetch_weather_data()` retrieves current conditions, 168 hourly records, and 7 daily summaries with disk caching (`30 min` TTL) and graceful `stale_cache` / `unavailable` states.
   - `climate_service.fetch_climate_status()` retrieves NOAA CPC `RONI.ascii.txt` and `oni.ascii.txt` (`12 hr` TTL cache).
   - `risk_engine.assess_seven_day_forecast()` transforms the 7 daily weather objects into explainable risk scores and day-over-day shift analyses.
2. **Visual Screening (`pages/disease_detection.py` → `models/predict.py`):**
   - Uploaded image bytes undergo format, dimension, exposure, contrast, and foliar HSV validation before running cached `MobileNetV2` inference.
   - Results populate `st.session_state["latest_prediction"]`, automatically enriching `Overview` and `Farmer Advisory`.
3. **Deep Learning Backend Abstraction (`config/settings.py`):**
   - Uses Keras 3 (`keras>=3.10`). Automatically selects `"tensorflow"` if TensorFlow is installed in the Python environment, or `"torch"` on space-constrained / Apple Silicon hosts where PyTorch is available.


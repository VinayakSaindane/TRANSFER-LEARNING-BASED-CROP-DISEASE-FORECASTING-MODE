# AgriGuard AI — Technical Implementation Report

## 1. Executive Summary

AgriGuard AI has been implemented as a complete, software-only agricultural decision-support platform built with Python, Streamlit, Keras 3 (`MobileNetV2`), OpenCV/Pillow, Plotly, Scikit-learn, Open-Meteo APIs, and NOAA Climate Prediction Center (CPC) ENSO indices (`RONI` and `ONI`).

---

## 2. Environment & Deep Learning Backend Configuration

- **Host Platform:** macOS (Apple M4, 16 GB RAM, Python 3.13.5).
- **Backend Constraint & Resolution:** At setup time, the host filesystem `/System/Volumes/Data` had limited free disk space (`~0.7–1.2 GB` free), whereas `tensorflow` unpacks to `~1.05 GB` (`~1.25 GB` peak). AgriGuard AI uses **Keras 3 (`keras==3.15.1`) with the PyTorch backend (`KERAS_BACKEND=torch`, `torch==2.11.0`)**. `config/settings.py` and `models/__init__.py` configure `KERAS_BACKEND` **before importing `keras`** (verified via subprocess test), selecting `"tensorflow"` when present and `"torch"` otherwise.

---

## 3. Dataset Acquisition & Verification Summary

1. **PlantVillage (`Tomato___healthy`, `Tomato___Early_blight`):**
   - Raw images retrieved: **2,591** (`1,591 Healthy`, `1,000 Tomato Early Blight`).
   - Corrupted images removed: **0**.
   - Exact SHA-256 duplicates removed: **6** (all inside `Tomato___healthy`).
   - Valid unique images retained: **2,585** (`256×256` RGB → resized to `224×224` RGB).
   - Leaf-grouped stratified splits (`seed=42`, `863` unique physical leaf groups):
     - **Train:** `1,805` (`1,109 Healthy`, `696 Tomato Early Blight` — `69.83%`)
     - **Validation:** `390` (`238 Healthy`, `152 Tomato Early Blight` — `15.09%`)
     - **Held-Out Test:** `390` (`238 Healthy`, `152 Tomato Early Blight` — `15.09%`)
   - Verified zero leaf-group overlap and zero SHA-256 hash overlap across all splits.
2. **PlantDoc (External Field Evaluation Only):**
   - Total valid images: **151** (`63 Healthy`, `88 Tomato Early Blight`).
   - Isolated strictly from training (`0` overlap with PlantVillage hashes).

---

## 4. Model Architecture, Training & Evaluation Results

- **Model:** ImageNet-pretrained `MobileNetV2` (`include_top=False`) + `GlobalAveragePooling2D` + `Dropout(0.30)` + `Dense(2, softmax)` (`2,260,546` total parameters; `1,351,042` trainable in Stage 2).
- **Training:**
  - Stage 1 (6 epochs, frozen backbone, balanced class weights `{0: 0.8138, 1: 1.2967}`, `Adam lr=1e-3`): reached `val_accuracy = 0.9923`, `val_loss = 0.0431`.
  - Stage 2 (6 epochs, top 25 non-BN layers unfrozen, `Adam lr=5e-5`, `ReduceLROnPlateau`, `EarlyStopping` restoring Epoch 4 of Stage 2): reached best `val_loss = 0.00557` (`val_accuracy = 1.0000`).
- **Actual Evaluation Metrics (`data/reports/evaluation_metrics.json`):**
  - **PlantVillage Train (`n=1,805`):** Accuracy **99.89%**, Macro F1 **0.9988** (`[[1107, 2], [0, 696]]`)
  - **PlantVillage Validation (`n=390`):** Accuracy **100.00%**, Macro F1 **1.0000** (`[[238, 0], [0, 152]]`)
  - **PlantVillage Held-Out Test (`n=390`):** Accuracy **99.23%**, Balanced Accuracy **99.01%**, Specificity **100.00%**, Sensitivity **98.03%**, Macro Precision **0.9938**, Macro Recall **0.9901**, Macro F1 **0.9919** (`[[238, 0], [3, 149]]`)
  - **PlantDoc External Zero-Shot Field Test (`n=151`):** Raw Accuracy **60.26%**, **Balanced Accuracy `52.38%`**, **Specificity (`Healthy` recall) `4.76%` (`3/63`)**, Sensitivity (`Early Blight` recall) `100.00%` (`88/88`), Macro Precision `0.7973`, Macro F1 `0.4183` (`[[3, 60], [0, 88]]`, ordering `[Healthy, Tomato Early Blight]`), with `148/151` (`98.01%`) predicted as `Tomato Early Blight` vs `3/151` (`1.99%`) predicted as `Healthy`. This demonstrates a `-46.63` percentage-point balanced accuracy drop (`-38.97` raw) and confirms the model is **not reliable for unverified field diagnosis**.
- **Inference Guardrails:** Rejects unreadable/corrupted files, images below `64×64`, blank/clipped images, and non-foliar inputs (`< 5%` plant HSV hue coverage), issues uncertainty warnings below `65%` confidence, and explicitly clarifies across all confidence tiers that softmax confidence is not guaranteed diagnostic certainty.
- **Resume Verification:** `models/train.py --resume` verified into an isolated `--output-dir` (`initial_value_threshold=0.00557`), completing 1 fine-tuning epoch while preserving identical SHA-256 checksums for `agriguard_mobilenetv2.keras`, `agriguard_checkpoint.keras`, `model_metadata.json`, and `training_history.json`.

---

## 5. Micro-Climate Weather, Risk Engine & NOAA ENSO Integrations

- **Open-Meteo Integration:** Geocoding search, location-level **modelled** NWP current weather, 168-hour hourly telemetry (including modelled `soil_temperature_0cm` and `soil_moisture_0_to_1cm`), 7-day daily summary, and **recent 5-year historical comparison** (`2021–2025` Open-Meteo Archive, explicitly labeled as a recent historical comparison rather than a 30-year WMO climatological normal). Explicitly clarifies that Open-Meteo does not measure farm-level sensors or physical leaf wetness.
- **Risk Engine (`0–100` Experimental Environmental Favourability Index):** Weighted prototype (`RH 30%`, `Temp 25%`, `Rainfall 25%`, `Dew/Wetness Proxy 20%`), explicit day-over-day shift explanation, missing/`NaN` guards, consistent `high_risk_days` threshold alignment, and interactive What-If Sensitivity Simulator using the identical `compute_environmental_risk()` functions. Prominently displays: *"This is an experimental environmental disease-favourability index, not a validated probability of future infection."*
- **NOAA CPC Climate Intelligence:** Live parsing of `RONI.ascii.txt` (primary operational index), `oni.ascii.txt` (reference index), and CPC ENSO Diagnostic Discussion (`ensodisc.shtml`) plus HTTP `Last-Modified` update timestamp (`2026-11-03T19:31:48Z`), enforcing NOAA's `±0.5°C` sustained 5-season threshold and official climate context cross-checks (a single overlapping season crossing `±0.5°C` never triggers an episode). Latest verified observation (`ASO 2026`): `RONI = +1.97°C` (4 consecutive seasons `>= +0.50°C`, following `AMJ 2026 = +0.49°C`), `ONI = +2.46°C` (5 consecutive seasons), CPC Alert System `El Niño Advisory` (`8 October 2026`) → `El Niño` context only (never modifies local disease risk scores).
- **Farmer Advisory:** Dynamic, non-chemical cultural advisories tied to actual 7-day forecast risks and session AI scan state.
- **Automated & Browser Verification:** `22/22` Pytest tests passing in `11.81s`; all 8 pages visually inspected via headless Google Chrome (`1440×900` desktop + `390×844` mobile) on live Streamlit (`/_stcore/health` HTTP `200`) with zero exceptions and zero horizontal overflow.


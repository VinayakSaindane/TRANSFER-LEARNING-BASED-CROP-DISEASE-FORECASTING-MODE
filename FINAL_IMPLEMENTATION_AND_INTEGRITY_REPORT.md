# AgriGuard AI — Final Implementation and Integrity Report

**Project Title:** Transfer Learning Based Crop Disease Forecasting Model Linked to El Niño-Induced Micro-climate Variations  
**Date Completed:** 2026-11-10  
**Workspace Root:** `/Users/vinayaksaindane/Documents/saish-project`  
**Execution Status:** Complete software-only prototype built, trained, evaluated, QA-verified in headless Google Chrome across all 8 pages, and re-tested (`22/22` Pytest passing).

---

## A. Implementation Summary

AgriGuard AI was built from scratch as a modular Python and Streamlit agricultural decision-support platform with zero hardware or IoT sensor dependencies. Implemented capabilities include:
1. **Reproducible Dataset Pipeline (`data/prepare_datasets.py`):** Sparse acquisition, integrity decoding, SHA-256 deduplication, and physical leaf-grouped stratified splitting (`70% / 15% / 15%`) of PlantVillage tomato images (`Healthy` vs. `Tomato Early Blight`), plus isolated preparation of PlantDoc field imagery (`Tomato leaf` → `Healthy` [63], `Tomato Early blight leaf` → `Tomato Early Blight` [88]) for external evaluation.
2. **Transfer Learning Vision Model (`models/`):** ImageNet-pretrained `MobileNetV2` trained in two stages (head training followed by fine-tuning of upper non-BatchNormalization layers) using Keras 3 (`torch` backend configured in `config/settings.py` and `models/__init__.py` before importing `keras`), with balanced class weights, early stopping, non-destructive `--resume` checkpointing, shared preprocessing, and inference suitability/uncertainty guardrails.
3. **Modelled Weather Feeds (`services/weather_service.py`):** Open-Meteo Geocoding, Forecast (current + 168-hour + 7-day daily + modelled surface soil temperature/moisture), and **recent 5-year historical comparison** (`2021–2025` Open-Meteo Archive, explicitly distinguished from a 30-year WMO climatological normal) with disk TTL caching and graceful offline/unavailable fallbacks. Explicitly distinguishes gridded location-level NWP outputs from direct farm micro-climate sensors or physical leaf wetness.
4. **Experimental Disease Risk Engine & What-If Simulator (`services/risk_engine.py`):** Weighted `0–100` Environmental Disease-Favourability Index (`30%` RH, `25%` Temperature, `25%` Rainfall splash, `20%` Dew/Wetness proxy) with missing/`NaN` guards, day-over-day driver delta explanations, consistent boundary classification, and an interactive What-If sensitivity simulator using the identical scoring functions. Prominently states: *"This is an experimental environmental disease-favourability index, not a validated probability of future infection."*
5. **NOAA ENSO / RONI Climate Intelligence (`services/climate_service.py`):** Live parsing of NOAA CPC's operational `RONI.ascii.txt`, reference `oni.ascii.txt`, HTTP `Last-Modified` timestamps, and CPC ENSO Diagnostic Discussion (`ensodisc.shtml`), enforcing NOAA's sustained `±0.5°C` threshold across **≥ 5 consecutive overlapping 3-month seasons** and official climate context cross-checks (never triggered by a single season crossing `±0.5°C`, and strictly separated from local disease-risk scoring).
6. **Preventive Farmer Advisory Engine (`services/advisory_service.py`):** Non-chemical educational and cultural recommendations tied directly to forecast risk drivers and any active AI leaf scan in session.
7. **Green-and-White SaaS Dashboard (`app.py`, `components/`, `pages/`):** Custom-styled 8-page Streamlit application with custom SVG branding, rounded white KPI cards, interactive Plotly charts, and consistent sidebar navigation—visually inspected and verified across all 8 pages in headless Google Chrome.
8. **Automated Verification & Documentation (`tests/`, `docs/`, `README.md`):** `22` Pytest unit, integration, backend-order, and Streamlit `AppTest` cases plus complete literature, dataset, methodology, architecture, and QA verification documentation.

---

## B. File Inventory

| File / Directory | Purpose |
|---|---|
| `app.py` | Main Streamlit entry point (`streamlit run app.py`), feed loader, and 8-page router |
| `.streamlit/config.toml` | Green-and-white Streamlit theme & hides native multipage sidebar menu (`showSidebarNavigation = false`) |
| `config/settings.py` | Centralized paths, API URLs, model constants, risk weights, and Keras backend selector |
| `requirements.txt` | Python package requirements |
| `.env.example` / `.gitignore` | Optional configuration template and repository exclusions |
| `pytest.ini` | Configures `pythonpath = .` and `testpaths = tests` |
| `data/prepare_datasets.py` | Dataset download, validation, SHA-256 deduplication, leaf-grouped splits & statistics JSON |
| `data/reports/dataset_statistics.json` | Verified dataset counts, duplicates removed, and split summaries |
| `data/reports/evaluation_metrics.json` | Computed Scikit-learn metrics on PlantVillage splits and PlantDoc external set |
| `data/reports/training_history.json` | Per-epoch training/validation accuracy and loss logs |
| `data/reports/training_curves.png` | Saved Matplotlib accuracy and loss progression plot |
| `data/reports/confusion_matrix.png` | Saved Matplotlib PlantVillage Test and PlantDoc confusion matrices |
| `models/__init__.py` | Ensures `KERAS_BACKEND` is configured via `config/settings.py` before any `keras` import |
| `models/preprocessing.py` | Shared MobileNetV2 `[-1, 1]` normalization, file/size validation, and HSV foliar screening |
| `models/train.py` | Two-stage MobileNetV2 trainer (`TomatoLeafSequence`, class weights, checkpoints, safe non-overwriting `--resume`) |
| `models/evaluate.py` | Full evaluation across PlantVillage train/val/test and PlantDoc external eval |
| `models/predict.py` | Cached model loader (`lru_cache`) and inference pipeline with uncertainty warnings |
| `models/saved/agriguard_mobilenetv2.keras` | Saved best MobileNetV2 model weights & architecture (`20 MB`) |
| `models/saved/agriguard_checkpoint.keras` | Saved best validation-loss checkpoint (`20 MB`) |
| `models/saved/model_metadata.json` | Model parameters, class indices, backend metadata, and hyperparameters |
| `services/weather_service.py` | Open-Meteo Forecast, Geocoding & Archive API client with TTL disk cache |
| `services/climate_service.py` | NOAA CPC `RONI.ascii.txt` & `oni.ascii.txt` parser, phase classifier & regional context |
| `services/risk_engine.py` | Transparent 0–100 environmental favourability scoring & What-If simulator |
| `services/advisory_service.py` | Weather- and scan-linked non-chemical preventive farmer advisory generator |
| `assets/logo/agriguard_logo.svg` | Custom SVG green shield/leaf/pulse brand emblem |
| `components/styles.py` | Custom CSS design system, typography, pills, callouts, and `render_page_header()` |
| `components/cards.py` | Reusable SaaS KPI cards, alert callouts, and advisory cards |
| `components/charts.py` | Plotly risk, temperature, humidity, rainfall, ENSO, What-If, and training/CM charts |
| `components/sidebar.py` | Branded 8-section sidebar navigation, farm status card, and manual API sync button |
| `components/upload.py` | Leaf image file uploader and optional held-out sample selector |
| `pages/overview.py` | Page 1: Overview Dashboard |
| `pages/disease_detection.py` | Page 2: AI Disease Detection |
| `pages/weather.py` | Page 3: Weather Intelligence |
| `pages/risk_forecast.py` | Page 4: Disease Risk Forecast & What-If Climate Simulator |
| `pages/climate.py` | Page 5: Climate Intelligence (NOAA ENSO/RONI + 5-year local baseline) |
| `pages/advisory.py` | Page 6: Farmer Advisory |
| `pages/model_performance.py` | Page 7: Model Performance |
| `pages/research.py` | Page 8: About the Research |
| `tests/conftest.py` & `tests/test_*.py` | 6 test modules (19 test functions total) covering all backend logic & UI rendering |
| `docs/*.md` & `README.md` | Literature references, methodology, architecture, dataset docs, implementation report, setup guide |

---

## C. Dataset Report

### 1. PlantVillage (Primary Training Dataset)
- **Origin:** `https://github.com/spMohanty/PlantVillage-Dataset` (sparse checkout of `raw/color/Tomato___healthy`, `raw/color/Tomato___Early_blight`, and `leaf_grouping` metadata). License: `CC-BY-SA-3.0`.
- **Raw Counts:** `1,591` Healthy + `1,000` Tomato Early Blight = **`2,591` images** (`256×256` RGB).
- **Data Cleaning:** `0` corrupted/unreadable files; **`6` byte-identical SHA-256 duplicates removed** inside `Tomato___healthy` → **`2,585` unique valid images**.
- **Anti-Leakage Grouping:** `1,999` images matched directly to PlantVillage `leaf_grouping/filtered_leafmaps/Tomato___*.csv` + `leaf-map.json` after stripping UUID prefixes; remaining `GH_HL Leaf <N>` multi-angle images grouped by leaf identifier (**`863` unique physical leaf groups**).
- **Stratified Leaf-Grouped Splits (`seed=42`):**
  - **Train:** `1,805` (`69.83%`: `1,109` Healthy, `696` Tomato Early Blight)
  - **Validation:** `390` (`15.09%`: `238` Healthy, `152` Tomato Early Blight)
  - **Held-Out Test:** `390` (`15.09%`: `238` Healthy, `152` Tomato Early Blight)
  - Verified `0` leaf-group overlap and `0` SHA-256 overlap across splits.
- **Preprocessing:** Resized via bilinear interpolation to `224×224` RGB (`data/processed/{train,val,test}/`) and normalized to `[-1.0, +1.0]` using `keras.applications.mobilenet_v2.preprocess_input`.

### 2. PlantDoc (Isolated External Field Evaluation Dataset)
- **Origin:** `https://github.com/pratikkayal/PlantDoc-Dataset` (`train/` and `test/` folders for `Tomato leaf` and `Tomato Early blight leaf`). License: `CC-BY-4.0`.
- **Counts:** `63` Healthy + `88` Tomato Early Blight = **`151` valid images** (`0` corrupted, `0` internal duplicates, `0` SHA-256 overlap with PlantVillage).
- **Isolation:** Kept strictly in `data/processed/plantdoc_eval/`; never used during training or early stopping.

---

## D. AI Model Report

- **Architecture:** `MobileNetV2` (ImageNet weights, `include_top=False`, `224×224×3`) → `GlobalAveragePooling2D` → `Dropout(0.30)` → `Dense(2, activation="softmax")`.
- **Parameters:** `2,260,546` total (`1,351,042` trainable during Stage 2 fine-tuning).
- **Deep Learning Stack Note:** Host `/System/Volumes/Data` had only `~1.1–1.2 GB` free disk space at setup time, while `tensorflow==2.20.0` requires `~1.05 GB` unpacked (`~1.25 GB` peak during extraction). Pursuant to Section 4 (*genuine compatibility/resource constraint*), `keras==3.15.1` runs on the **`torch` backend (`torch==2.11.0`)**, while `config/settings.py` remains compatible with TensorFlow when installed.
- **Training Configuration:** `batch_size=32`, `seed=42`, balanced class weights (`0: 0.8138`, `1: 1.2967`), training augmentation (flips, ±18° rotation, brightness/contrast jitter).
  - Stage 1 (6 epochs, frozen backbone, `Adam lr=1e-3`): converged to `val_accuracy = 0.9923`, `val_loss = 0.0431`.
  - Stage 2 (6 epochs, top 25 non-BatchNorm layers unfrozen, `Adam lr=5e-5`, `ReduceLROnPlateau`, `EarlyStopping`): best `val_loss = 0.00557` (`val_accuracy = 1.0000`) at Epoch 4, weights restored automatically.
- **Saved Model:** `models/saved/agriguard_mobilenetv2.keras` (`20 MB`).
- **Actual Observed Evaluation Results (`data/reports/evaluation_metrics.json`):**

| Evaluation Split | Samples | Accuracy | Balanced Acc | Specificity (`Healthy` Recall) | Sensitivity (`Early Blight` Recall) | Macro F1 | Confusion Matrix `[[TN, FP], [FN, TP]]` |
|---|---:|---:|---:|---:|---:|---:|---|
| PlantVillage Train | 1,805 | **99.89%** | 99.91% | 99.82% | 100.00% | 0.9988 | `[[1107, 2], [0, 696]]` |
| PlantVillage Validation | 390 | **100.00%** | 100.00% | 100.00% | 100.00% | 1.0000 | `[[238, 0], [0, 152]]` |
| PlantVillage Held-Out Test | 390 | **99.23%** | 99.01% | 100.00% | 98.03% | 0.9919 | `[[238, 0], [3, 149]]` |
| PlantDoc External Field (Zero-Shot) | 151 | **60.26%** | **52.38%** | **4.76%** (`3/63`) | 100.00% (`88/88`) | 0.4183 | `[[3, 60], [0, 88]]` |

- **Class Ordering & PlantDoc Mapping Verification:** Verified `CLASS_TO_INDEX = {"Healthy": 0, "Tomato Early Blight": 1}` (confusion matrix rows = True Label, columns = Predicted Label) and `plantdoc_eval.csv` mapping (`Tomato leaf` → `Healthy` [`63`], `Tomato Early blight leaf` → `Tomato Early Blight` [`88`]).
- **PlantDoc Prediction Distribution & Domain-Shift Finding:** The model predicted `Tomato Early Blight` on **`148/151` (`98.01%`)** of PlantDoc images and `Healthy` on only **`3/151` (`1.99%`)** (`mean P(EB) = 0.9440`, `median P(EB) = 0.9993` on true Healthy field leaves). Thus `60.26%` raw accuracy is artificially sustained by class prevalence (`58.28%` diseased), while **Balanced Accuracy (`52.38%`)** and **Specificity (`4.76%`)** show near-random discrimination on outdoor images. **Current weights are NOT reliable for unverified field diagnosis**; this finding and diagnostic safety warnings are surfaced prominently on both Model Performance and AI Disease Detection pages.

---

## E. API Integration Report

1. **Open-Meteo Geocoding API (`https://geocoding-api.open-meteo.com/v1/search`):**
   - Verified live request (`Nashik` → 3 matches returned, `status="ok"`).
2. **Open-Meteo Forecast API (`https://api.open-meteo.com/v1/forecast`):**
   - Variables fetched: `current` (`temperature_2m`, `relative_humidity_2m`, `apparent_temperature`, `precipitation`, `dew_point_2m`, `weather_code`, `wind_speed_10m`, `cloud_cover`), `hourly` (`temperature_2m`, `relative_humidity_2m`, `dew_point_2m`, `precipitation`, `soil_temperature_0cm`, `soil_moisture_0_to_1cm`), `daily` (`weather_code`, `temperature_2m_max/min/mean`, `precipitation_sum`, `precipitation_probability_max`, `et0_fao_evapotranspiration`).
   - Verified live response for Nashik (`19.9975, 73.7898`): `status="live"`, current temperature `29.9°C`, `7` daily records, `retrieved_at: 2026-11-10T07:44:35.902396+00:00`. Explicitly labeled as gridded location-level modelled output (not direct farm sensors or measured leaf wetness).
   - Disk cache (`30 min` TTL), timeout (`12 s`), and tested offline `stale_cache` / `unavailable` states.
3. **Open-Meteo Historical Archive API (`https://archive-api.open-meteo.com/v1/archive`):**
   - Verified live **recent 5-year historical comparison** (`[2021, 2022, 2023, 2024, 2025]`) for Nashik Nov 10–16 window: mean temperature `22.47°C`, mean 7-day rain `0.68 mm`, mean RH `56.43%` (explicitly labeled as a recent historical comparison, NOT a 30-year WMO climatological normal).
4. **NOAA CPC ENSO Indices & Advisory (`RONI.ascii.txt`, `oni.ascii.txt`, `ensodisc.shtml`):**
   - Verified live request (`status="live"`): latest observation season **`ASO 2026` (`Aug–Oct 2026`)**, **`RONI = +1.97°C`**, **`ONI = +2.46°C`**, NOAA `Last-Modified: 2026-11-03T19:31:48+00:00`, CPC ENSO Diagnostic Discussion issued **`8 October 2026`** with **`ENSO Alert System Status: El Niño Advisory`**.
   - **Classification Logic:** Requires `±0.5°C` met across **≥ 5 consecutive overlapping 3-month seasons** for standalone RONI episode confirmation (`AMJ 2026 = +0.49°C` leaves 2-decimal RONI at `4` consecutive seasons `>= +0.50°C`, though NOAA's 1-decimal table rounds `AMJ 2026` to `+0.5°C`), corroborated for `2–4` season developing runs by official CPC `El Niño Advisory` and `5`-season sustained `ONI >= +0.50°C` (`12 hr` disk cache). Single overlapping seasons crossing `±0.5°C` never trigger El Niño/La Niña classification.

---

## F. Frontend & Browser Visual Verification Report

All 8 pages were implemented in `pages/` and verified **both** programmatically (`AppTest`) and visually in headless Google Chrome (`157.0.8103.38`) against live Streamlit (`http://127.0.0.1:8599`, `/_stcore/health` HTTP `200`) at desktop (`1440×900`) and mobile (`390×844`) viewports:
1. **Overview (`pages/overview.py`):** Location summary, weather/risk/ENSO/AI KPIs, source/location/timestamp caption, experimental favourability notice, 7-day risk & temperature charts, forecast snapshot table, and top preventive advisories.
2. **AI Disease Detection (`pages/disease_detection.py`):** Prominent domain-shift & diagnostic safety warning banner, file upload + held-out sample picker, tested live image upload + MobileNetV2 inference (`Tomato Early Blight`, `100.0%` softmax), probability bars, and non-certainty notices.
3. **Weather Intelligence (`pages/weather.py`):** City geocoding, tomato-belt presets, custom lat/lon input, explicit distinction between gridded NWP model output and farm-level micro-climate sensors/leaf wetness, source/location/UTC timestamp, 3 Plotly weather charts, and 7-day table.
4. **Disease Risk Forecast (`pages/risk_forecast.py`):** Prominent `"This is an experimental environmental disease-favourability index, not a validated probability of future infection."` banner, 7-day risk & stacked contribution charts, daily shift table, and verified What-If Simulator sharing `compute_environmental_risk()`.
5. **Climate Intelligence (`pages/climate.py`):** Latest season (`Aug–Oct 2026`), RONI (`+1.97°C`), ONI (`+2.46°C`), NOAA source update timestamp (`2026-11-03 19:31:48 UTC`) + CPC Advisory (`El Niño Advisory`), 36-season ENSO trajectory chart, sustained 5-season methodology & attribution, recent 5-year historical comparison (not a 30-year climatological normal), and non-causality warning.
6. **Farmer Advisory (`pages/advisory.py`):** Risk & linked AI scan summary cards, dynamic non-chemical preventive advisory cards, early blight scouting checklist, and non-prescription policy.
7. **Model Performance (`pages/model_performance.py`):** Prominent red domain-shift unreliability warning, PlantVillage & PlantDoc confusion matrices with ordering & mapping documentation, per-class precision/recall/F1, Specificity (`4.76%`), Balanced Accuracy (`52.38%`), PlantDoc Prediction Distribution chart & table (`98.01%` predicted Early Blight), and architecture/dataset summaries.
8. **About the Research (`pages/research.py`):** Abstract, problem statement, objectives, architecture, 10 verified citations, contributions, and realistic limitations.

---

## G. Risk Engine Report

- **Formula:** `Score = 100 * (0.30 * f_RH + 0.25 * f_Temp + 0.25 * f_Rain + 0.20 * f_DewWetness)`, clamped to `[0.0, 100.0]`.
- **Factor Curves (`services/risk_engine.py`):**
  - `score_relative_humidity`: piecewise ramp accelerating above `65%` RH toward saturation (`≥92%`).
  - `score_temperature_suitability`: peaks (`1.0`) in the `24–29°C` *Alternaria solani* optimum; drops to `0.0` at `≤10°C` or `≥38°C`.
  - `score_rainfall`: saturates at `25 mm/day` for conidial splash dispersal; rainfall alone never forces leaf wetness to `1.0`.
  - `score_dew_wetness_proxy`: blends `0.60` dew-point depression ($T - T_d$, high below `2.5°C`) and `0.40` daily hours with `RH ≥ 85%` (normalized to `12 hrs`); falls back to Magnus-Tetens dew-point estimation when only `T` and `RH` are provided (e.g., What-If simulator).
- **Risk Bands:** `0–30 Low` | `31–60 Moderate` | `61–80 High` | `81–100 Very High`.
- **Assumptions & Limitations:** Prototype weights (`30/25/25/20`) are literature-informed rather than locally fitted against field disease-severity trials; the score represents **environmental favourability**, NOT confirmed probability of infection.

---

## H. Testing Report

### Commands Executed & Observed Results (After Final Code Edits)

1. **Dataset Preparation & Leakage Verification:**
   - Command: `python3 data/prepare_datasets.py`
   - Result: Exit `0`; verified `2,585` unique PlantVillage images across `863` physical leaf groups (`1,805` train / `390` val / `390` test) and `151` PlantDoc images with `0` overlap.
2. **Model Training:**
   - Command: `python3 models/train.py --head-epochs 6 --finetune-epochs 6`
   - Result: Exit `0`; saved `agriguard_mobilenetv2.keras`, `agriguard_checkpoint.keras`, `model_metadata.json`, `training_history.json`.
3. **Model Evaluation (`models/evaluate.py`):**
   - Command: `python3 models/evaluate.py`
   - Result: Exit `0`; regenerated `evaluation_metrics.json` with `specificity`, `sensitivity`, `balanced_accuracy`, `prediction_distribution`, `confusion_matrix_ordering`, and `plantdoc_label_mapping` on the unmodified saved model.
4. **Model `--resume` Non-Destructive Verification:**
   - Command: `python3 models/train.py --resume --head-epochs 0 --finetune-epochs 1 --output-dir /tmp/ag_resume_verify` + `shasum -a 256` verification.
   - Result: Exit `0`; loaded `agriguard_checkpoint.keras` (`baseline val_loss = 0.00557`), refused to overwrite checkpoint when `val_loss` did not improve (`0.1520`), saved resumed artifact to isolated `/tmp/ag_resume_verify`, and preserved identical SHA-256 hashes for `agriguard_mobilenetv2.keras`, `agriguard_checkpoint.keras`, `model_metadata.json`, and `training_history.json`.
5. **Live API Integration Verification:**
   - Command: Python check calling `search_locations`, `fetch_weather_data`, `fetch_historical_seasonal_baseline`, `fetch_climate_status`, `assess_seven_day_forecast`, `simulate_what_if_scenario`, `generate_farmer_advisories`.
   - Result: Exit `0`; live HTTP 200 responses parsed cleanly from Open-Meteo and NOAA CPC (`ASO 2026`, `RONI: +1.97°C`, `ONI: +2.46°C`, `El Niño Advisory` issued `8 October 2026`, `Last-Modified: 2026-11-03T19:31:48Z`).
6. **Automated Test Suite (`pytest -v`):**
   - Command: `pytest -v`
   - Result: **`22 passed` in `11.81s` (`0` failed, `0` skipped):**
     - `tests/test_app_smoke.py` (2 tests): All 8 pages render cleanly with `AppTest` including safety/methodology assertions; What-If sliders + AI sample inference + Farmer Advisory workflow succeed without exceptions.
     - `tests/test_risk_engine.py` (5 tests): Bounds `[0, 100]`, category thresholds, `None`/`NaN` isolation (required & optional fields), extreme-temperature Magnus-Tetens guard, rain-only leaf-wetness guardrail, exact disclaimer text, 7-day `high_risk_days` boundary consistency & identical What-If scoring.
     - `tests/test_weather_service.py` (6 tests): Coordinate bounds, Open-Meteo schema parsing, gridded NWP vs farm micro-climate & 5-year comparison vs 30-year normal labels, mocked offline `unavailable`, mocked `stale_cache` timestamp preservation, short-query rejection.
     - `tests/test_climate_and_advisory.py` (4 tests): Single-season crossing rejection, 5-season sustained episode logic, developing RONI + CPC Advisory/ONI context confirmation, mocked NOAA offline `unavailable`, non-chemical advisory check, and zero-coupling test confirming RONI/ONI never modify disease risk.
     - `tests/test_data_validation.py` (1 test): Zero leaf-group overlap, zero SHA-256 leakage, zero PlantVillage/PlantDoc overlap.
     - `tests/test_prediction.py` (4 tests): Model load & evaluation metrics/specificity/balanced accuracy/prediction distribution checks, subprocess verification that `KERAS_BACKEND` is set before `keras` import, held-out image inference with non-diagnostic-certainty notice, rejection of corrupted/tiny/blank/non-foliar inputs.
7. **Live Streamlit Server & Headless Chrome Visual Verification (All 8 Pages):**
   - Command: `streamlit run app.py --server.port 8599 --server.headless true` + `curl http://127.0.0.1:8599/_stcore/health` + headless Google Chrome `157.0.8103.38` CDP sweep across all 8 pages (`1440×900` desktop + `390×844` mobile) + live file upload & inference test.
   - Result: `ok 200`; all 8 pages rendered with zero exceptions, zero horizontal overflow on desktop or mobile, interactive file upload & diagnosis confirmed, and 8 full-page PNG screenshots visually inspected.

*Additional defects discovered and fixed during Final Independent QA Pass:*
- Replaced single-season `latest >= ±0.5` ENSO trigger in `services/climate_service.py` with NOAA's sustained 5-consecutive-overlapping-season criterion + HTML-unescaped `ensodisc.shtml` CPC Alert System & HTTP `Last-Modified` parsing.
- Added `specificity`, `sensitivity`, `balanced_accuracy`, `prediction_distribution`, `confusion_matrix_ordering`, and `plantdoc_label_mapping` to `models/evaluate.py` and surfaced prominent domain-shift/field-unreliability warnings across `Model Performance` and `AI Disease Detection`.
- Clarified gridded Open-Meteo NWP forecast outputs vs. farm micro-climate/leaf wetness and relabeled the 5-year Open-Meteo Archive comparison across `weather_service.py`, `weather.py`, `climate.py`, and `overview.py`.
- Aligned `high_risk_days` category boundary consistency in `services/risk_engine.py` & `services/advisory_service.py`, guarded optional `NaN` propagation and `-243.04°C` Magnus-Tetens edge cases, and added the exact required experimental index notice.
- Made `models/train.py --resume` non-destructive by default (`--output-dir` / `resumed/` + `initial_value_threshold=baseline_val_loss`) and ensured `models/__init__.py` configures `KERAS_BACKEND` before importing `keras`.
- Fixed 3 real visual defects discovered in Chrome screenshots: Plotly horizontal legend overlapping chart titles (`components/charts.py`), `_apply_clean_axes` overwriting secondary `yaxis2` titles (`Loss` & `Probability (%)`), `.ag-pill` `white-space: nowrap` overflowing narrow KPI cards (`components/styles.py`), negative rainfall y-axis range (`[-1, 1]` when `0 mm`), and clipped PlantDoc prediction distribution table header (`pages/model_performance.py`).

---

## I. Remaining Limitations

1. **TensorFlow Runtime Substituted with Keras 3 + PyTorch Backend:** Host filesystem free space (`~0.75–1.2 GB`) could not safely accommodate the `~1.05 GB` unpacked `tensorflow` wheel; Keras 3 (`torch` backend) is used cleanly instead.
2. **Binary Visual Scope:** Model classifies only `Healthy` vs. `Tomato Early Blight` and cannot identify late blight, Septoria leaf spot, bacterial spot, mosaic virus, or nutrient stress.
3. **Severe Field Domain Shift (Not Field-Reliable):** Zero-shot PlantDoc accuracy (`60.26%` raw, **`52.38%` balanced accuracy**, **`4.76%` Healthy specificity**) confirms that studio-trained weights misclassify `60/63` healthy natural-background leaves as Early Blight and must not be used for unverified field diagnosis without field-domain fine-tuning.
4. **Uncalibrated Prototype Risk Weights & Gridded Weather Inputs:** The `30/25/25/20` risk weights are experimental literature-derived prototypes requiring longitudinal multi-season field incidence trials before operational deployment, and Open-Meteo NWP inputs approximate rather than directly measure farm-canopy leaf wetness.
5. **Real-Time NOAA RONI Estimate Revision Window:** Per NOAA CPC, recent RONI values are real-time estimates subject to revision for up to two months as ERSSTv6 filtering settles.

---

## J. Execution Guide

```bash
# 1. Navigate to project root
cd /Users/vinayaksaindane/Documents/saish-project

# 2. (Optional) Re-prepare datasets from scratch
python3 data/prepare_datasets.py

# 3. (Optional) Train MobileNetV2 from scratch
python3 models/train.py --head-epochs 6 --finetune-epochs 6 --batch-size 32
# Or resume safely into an isolated directory without overwriting best saved weights:
# python3 models/train.py --resume --finetune-epochs 2 --output-dir /tmp/agriguard_resumed

# 4. (Optional) Re-run evaluation & regenerate metrics/plots
python3 models/evaluate.py

# 5. Run the automated test suite
pytest -v

# 6. Launch AgriGuard AI locally
streamlit run app.py
```

---

## K. Final Integrity Checklist

| Feature / Requirement | Classification | Notes |
|---|---|---|
| Software-only operation (no IoT/hardware) | **IMPLEMENTED AND TESTED** | Zero hardware dependencies anywhere in codebase |
| PlantVillage acquisition, cleaning & zero-leakage splits | **IMPLEMENTED AND TESTED** | 2,585 images across 863 leaf groups; verified by `test_data_validation.py` |
| PlantDoc isolated external evaluation & corrected metrics | **IMPLEMENTED AND TESTED** | `[[3, 60], [0, 88]]`; 52.38% balanced acc, 4.76% specificity, 98.01% positive prediction bias documented |
| MobileNetV2 2-stage transfer learning & Keras 3 `torch` backend order | **IMPLEMENTED AND TESTED** | `KERAS_BACKEND` verified set before `keras` import in `test_prediction.py` |
| TensorFlow runtime installation on host | **BLOCKED BY EXTERNAL DEPENDENCY** | Host disk had `~0.75–1.2 GB` free vs `~1.25 GB` peak TF install; resolved via Keras 3 `torch` backend |
| Training `--resume` CLI continuation (non-destructive) | **IMPLEMENTED AND TESTED** | Verified via 1-epoch `--resume --output-dir /tmp/ag_resume_verify` with SHA-256 preservation of production model |
| Inference validation, uncertainty & non-certainty safety notices | **IMPLEMENTED AND TESTED** | Verified in `test_prediction.py`, `test_app_smoke.py`, and Chrome upload test |
| Open-Meteo live weather, geocoding, cache, offline & NWP scope labels | **IMPLEMENTED AND TESTED** | Live API + 6 unit tests distinguishing modelled forecasts from farm sensors/leaf wetness |
| Open-Meteo Archive recent 5-year historical comparison | **IMPLEMENTED AND TESTED** | Explicitly labeled as recent 5-year comparison (`2021–2025`), NOT a 30-year climatological normal |
| Experimental 0–100 Disease Risk Engine & 7-day forecast | **IMPLEMENTED AND TESTED** | Verified bounds, `NaN` isolation, boundary alignment, exact experimental notice & 5 unit tests |
| What-If Climate Simulator | **IMPLEMENTED AND TESTED** | Verified using identical `compute_environmental_risk()` functions + `AppTest` & Chrome check |
| NOAA CPC RONI / ONI ENSO sustained-threshold integration | **IMPLEMENTED AND TESTED** | Verified 5-season threshold, single-season non-trigger, CPC `El Niño Advisory` context & zero risk coupling |
| Non-chemical preventive Farmer Advisory Engine | **IMPLEMENTED AND TESTED** | Linked to risk + AI scan; verified free of pesticide dosage prescriptions |
| Green-and-white 8-page SaaS Streamlit UI (Browser Verified) | **IMPLEMENTED AND TESTED** | Visually inspected all 8 pages in headless Chrome (`1440×900` & `390×844` mobile) with zero errors/overflow |
| Research citations & technical documentation | **IMPLEMENTED AND TESTED** | Verified DOIs/URLs and consistent metrics across `docs/`, `README.md`, and this report |

---

## L. Final Declaration

**Readiness Assessment:**
- **Ready for controlled academic demonstration:** **YES** — runs cleanly via `streamlit run app.py` with live Open-Meteo/NOAA feeds, trained MobileNetV2 inference, interactive charts, `22/22` passing automated tests, and full visual verification across all 8 pages.
- **Ready for autonomous field/production agronomic deployment:** **NO** — studio-trained vision weights exhibit severe domain shift (`52.38%` balanced accuracy, `4.76%` specificity on PlantDoc), risk weights (`30/25/25/20`) are experimental uncalibrated prototypes, and Open-Meteo provides location-level NWP estimates rather than farm-canopy leaf wetness sensors. Appropriate disclaimers are surfaced prominently throughout the interface.


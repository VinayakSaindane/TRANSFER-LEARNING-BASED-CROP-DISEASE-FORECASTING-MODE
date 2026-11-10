# AgriGuard AI

**Transfer Learning Based Crop Disease Forecasting Model Linked to El Niño-Induced Micro-climate Variations**

AgriGuard AI is a software-only agricultural decision-support prototype combining visual tomato leaf screening (MobileNetV2 transfer learning), live 7-day micro-climate monitoring (Open-Meteo), an explainable weather-driven disease-risk engine for tomato early blight (*Alternaria solani*), official NOAA ENSO/RONI climate intelligence, and non-chemical preventive farmer advisories.

> **No Hardware / IoT Dependencies:** AgriGuard AI runs entirely through Python software, publicly available datasets, open meteorological/climate APIs, and user-uploaded leaf images.

---

## 1. Problem Statement & Objectives

Tomato early blight (*Alternaria solani*) causes rapid foliar necrosis when warm temperatures (24–29 °C) coincide with high humidity, rainfall splash, and prolonged leaf wetness. Smallholder farmers often lack expensive IoT weather stations and laboratory diagnostic tools, while studio-trained visual classifiers suffer severe accuracy degradation on natural field backgrounds.

**Core Objectives:**
1. Train and evaluate a lightweight **MobileNetV2** classifier (`Healthy` vs. `Tomato Early Blight`) using deduplicated, **leaf-grouped** stratified PlantVillage splits (zero same-leaf leakage).
2. Measure real-world background **domain shift** objectively against the isolated **PlantDoc** field dataset.
3. Integrate live **Open-Meteo** weather forecasts, geocoding, and historical local seasonal baselines.
4. Score daily micro-climate favourability (`0–100` Environmental Disease-Risk Index) with transparent factor weights and a **What-If Climate Simulator**.
5. Surface official **NOAA CPC RONI & ONI** ENSO climate observations strictly as seasonal context (never artificially inflating local disease risk).
6. Generate grounded, educational, non-chemical farmer preventive recommendations.

---

## 2. Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python 3.10+ |
| Deep Learning | Keras 3 (TensorFlow or PyTorch backend) |
| Model | MobileNetV2 (ImageNet pretrained) |
| Image Processing | Pillow / OpenCV |
| Charts | Plotly & Matplotlib |
| Data Processing | Pandas / NumPy |
| Weather Telemetry | Open-Meteo Forecast, Geocoding & Archive APIs |
| Climate Intelligence | NOAA CPC RONI & ONI ASCII Feeds |
| Metrics & Testing | Scikit-learn & Pytest |
| Configuration | python-dotenv |

> **Backend Note:** AgriGuard AI uses Keras 3 (`keras>=3.10`) configured via [`config/settings.py`](config/settings.py) and [`models/__init__.py`](models/__init__.py) **before importing `keras`**. It auto-selects TensorFlow when installed, or configures the PyTorch backend (`KERAS_BACKEND=torch`) on hosts without TensorFlow (such as disk-constrained macOS environments).

---

## 3. System Architecture

```text
AgriGuard-AI/
├── app.py                     # Main Streamlit application & page router
├── requirements.txt
├── .env.example
├── config/settings.py         # Paths, API endpoints, thresholds & Keras backend setup
├── data/
│   ├── prepare_datasets.py    # Sparse dataset acquisition, verification, leaf-grouped split
│   └── reports/               # dataset_statistics.json, evaluation_metrics.json, plots
├── models/
│   ├── preprocessing.py       # Shared [-1, 1] MobileNetV2 preprocessing & foliar check
│   ├── train.py               # 2-stage transfer learning + checkpointing + resume
│   ├── evaluate.py            # PlantVillage & PlantDoc evaluation + confusion matrices
│   └── predict.py             # Cached model inference + uncertainty gating
├── services/
│   ├── weather_service.py     # Open-Meteo forecast, geocoding, archive & TTL cache
│   ├── climate_service.py     # NOAA CPC RONI/ONI parser & ENSO phase classifier
│   ├── risk_engine.py         # Explainable 0-100 risk engine & What-If simulator
│   └── advisory_service.py    # Non-chemical preventive farmer advisory generator
├── components/                # Custom CSS, KPI cards, Plotly charts, sidebar & uploaders
├── pages/                     # 8 modular dashboard pages
├── tests/                     # Pytest test suite (unit, integration & Streamlit AppTest)
└── docs/                      # Research references, methodology & reports
```

---

## 4. Environment Setup & Quick Run

> **Full Execution Guide:** See [`RUN.md`](RUN.md) or run the one-command launcher [`run.sh`](run.sh) (`./run.sh --setup` for first-time installation, `./run.sh --verify-only` for terminal verification, or `./run.sh` to verify & launch Streamlit).

```bash
# Quick one-command run (macOS / Linux)
chmod +x run.sh
./run.sh --setup

# Or manual setup:
# 1. Install dependencies
pip install -r requirements.txt

# 2. Optional: copy environment configuration template (no API keys required)
cp .env.example .env
```

---

## 5. Dataset Download & Preparation

Download, validate, deduplicate, and create leakage-free `70% / 15% / 15%` leaf-grouped stratified splits for PlantVillage plus the isolated PlantDoc external evaluation set:

```bash
python3 data/prepare_datasets.py
```

Outputs:
- Processed images in `data/processed/{train,val,test}/` and `data/processed/plantdoc_eval/`
- Manifests: `data/processed/splits.csv` and `data/processed/plantdoc_eval.csv`
- Verified counts & deduplication summary: `data/reports/dataset_statistics.json`

---

## 6. Model Training & Evaluation

Train MobileNetV2 (Stage 1 classification head → Stage 2 fine-tuning of top 25 non-BN layers):

```bash
python3 models/train.py --head-epochs 6 --finetune-epochs 6 --batch-size 32
```

To resume training from the saved checkpoint without overwriting the best saved production model (`models/saved/agriguard_mobilenetv2.keras`):

```bash
python3 models/train.py --resume --finetune-epochs 2
# Optional: specify an isolated output directory or explicitly allow overwriting:
# python3 models/train.py --resume --output-dir /tmp/resumed_run
```

Evaluate on PlantVillage train/val/test and isolated PlantDoc external field images:

```bash
python3 models/evaluate.py
```

Outputs:
- Best model: `models/saved/agriguard_mobilenetv2.keras`
- Metadata & curves: `models/saved/model_metadata.json`, `data/reports/training_history.json`
- Evaluation report & figures: `data/reports/evaluation_metrics.json`, `data/reports/training_curves.png`, `data/reports/confusion_matrix.png`
- **External Field Evaluation Finding (PlantDoc, `n=151`):** Raw accuracy `60.26%`, **Balanced Accuracy `52.38%`**, **Specificity (`Healthy` recall) `4.76%` (`3/63`)**, Sensitivity (`Tomato Early Blight` recall) `100.00%` (`88/88`), confusion matrix `[[3, 60], [0, 88]]`, with `148/151` (`98.01%`) predicted as `Tomato Early Blight`. Studio-trained weights are **not reliable for unverified field diagnosis**.

---

## 7. Running the Application

Run locally from the project root:

```bash
streamlit run app.py
```

Pages included:
1. **Overview** — Farm location summary, live weather, risk index, ENSO status, recent AI scan, and priority alerts
2. **AI Disease Detection** — Leaf upload, validation, MobileNetV2 prediction, confidence & uncertainty warnings
3. **Weather Intelligence** — City search/geocoding, coordinates, current weather & 7-day charts
4. **Disease Risk Forecast** — 7-day weighted risk index, factor attribution, explanations & **What-If Climate Simulator**
5. **Climate Intelligence** — Official NOAA CPC RONI/ONI trends, ENSO context & 5-year historical comparison
6. **Farmer Advisory** — Weather- and scan-linked preventive agronomic recommendations
7. **Model Performance** — Actual metrics, accuracy/loss curves, PlantVillage & PlantDoc confusion matrices
8. **About the Research** — Abstract, methodology, verified citations, limitations & future scope

---

## 8. API Integrations

- **Open-Meteo Forecast API** (`https://api.open-meteo.com/v1/forecast`): Location-level **modelled** gridded NWP forecast (current, 7-day hourly/daily, modelled surface soil outputs; `30 min` disk cache). Does **not** measure actual farm-level micro-climate sensors or physical leaf wetness.
- **Open-Meteo Geocoding API** (`https://geocoding-api.open-meteo.com/v1/search`): Global city/town search.
- **Open-Meteo Historical Archive API** (`https://archive-api.open-meteo.com/v1/archive`): **Recent 5-year historical comparison** (`2021–2025`) for the same 7-day calendar window (explicitly labeled as a recent historical comparison, **not** a 30-year WMO climatological normal).
- **NOAA CPC Climate Indices & Advisory** (`https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt`, `oni.ascii.txt`, `ensodisc.shtml`): Official 3-month overlapping seasonal observations classified using NOAA's `±0.5°C` threshold across **≥ 5 consecutive overlapping seasons** and official CPC climate context (`12 hr` disk cache). Displayed strictly as seasonal climate context—never modifies local disease risk.

If external APIs fail, AgriGuard AI displays either the last verified cached observation (with its exact UTC retrieval timestamp) or a clear unavailable state—never fabricated weather or ENSO numbers.

---

## 9. Running Tests

```bash
pytest -v
```

---

## 10. Limitations & Future Scope

- **Binary Classifier Scope:** Trained only on `Healthy` and `Tomato Early Blight`; cannot diagnose late blight, Septoria leaf spot, bacterial spot, viral curls, or nutrient deficiencies. Softmax confidence is never diagnostic certainty.
- **Severe Field Domain Shift:** Controlled PlantVillage studio performance (`99.23%` accuracy) drops to `52.38%` balanced accuracy and `4.76%` Healthy specificity (`[[3, 60], [0, 88]]`, `98.01%` predicted Early Blight) on cluttered PlantDoc outdoor photos. Reliable field diagnosis cannot be claimed from current studio-trained weights.
- **Experimental Environmental Index:** *"This is an experimental environmental disease-favourability index, not a validated probability of future infection."* Prototype weights (`30/25/25/20`) are literature-informed rather than locally calibrated against multi-year field disease incidence trials, and Open-Meteo inputs are location-level NWP estimates rather than physical leaf wetness sensors.
- **Future Scope:** Expand to multi-class field-augmented datasets, integrate sub-seasonal climate ensembles, and calibrate risk curves against empirical regional plant pathology surveys.


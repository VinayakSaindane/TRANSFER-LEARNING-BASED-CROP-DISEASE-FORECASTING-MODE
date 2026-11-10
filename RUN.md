# How to Run AgriGuard AI and Get Real Working Output

This guide gives exact step-by-step commands to clone, set up, verify, and run **AgriGuard AI** on a fresh machine.

---

## 1. Quick Start (macOS / Linux — Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/VinayakSaindane/TRANSFER-LEARNING-BASED-CROP-DISEASE-FORECASTING-MODE.git
cd TRANSFER-LEARNING-BASED-CROP-DISEASE-FORECASTING-MODE

# 2. First-time setup (creates .venv, installs requirements, downloads/prepares PlantVillage + PlantDoc, runs pytest & live API/AI verification)
chmod +x run.sh
./run.sh --setup
```

After setup completes, the Streamlit web application launches automatically at **`http://localhost:8501`**.

On subsequent runs:

```bash
# Verify everything + launch dashboard
./run.sh

# Or launch immediately without re-running pytest
./run.sh --fast

# Or verify tests + live APIs + model inference in terminal only (no browser server)
./run.sh --verify-only
```

---

## 2. Manual Setup & Execution (All Platforms / Windows PowerShell)

If you prefer running commands manually instead of `./run.sh`:

```bash
# 1. Create and activate a Python 3.10+ virtual environment
python3 -m venv .venv
source .venv/bin/activate        # On Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. Install dependencies (Keras 3 + PyTorch backend; no API keys required)
pip install --upgrade pip
pip install -r requirements.txt

# 3. Prepare datasets (downloads PlantVillage tomato subset & PlantDoc external evaluation set cleanly)
python3 data/prepare_datasets.py

# 4. Run automated verification suite (unit tests, leakage tests, model inference & all 8 UI pages)
pytest -v

# 5. Launch the Streamlit application
streamlit run app.py
```

*(Note: Pre-trained weights `models/saved/agriguard_mobilenetv2.keras` and evaluation artifacts under `data/reports/` are already included in the repository. Step 3 is only needed to enable the held-out sample selector and dataset/inference tests; retraining from scratch is optional via `python3 models/train.py`.)*

---

## 3. What Real Working Output Looks Like

Running `./run.sh --verify-only` (or `pytest -v` + `streamlit run app.py`) produces genuine live outputs:

1. **Automated Test Suite (`pytest -v`):**
   - `22 passed` covering dataset zero-leakage assertions, `MobileNetV2` loading & inference guardrails, Open-Meteo forecast/archive/geocoding parsing and offline fallbacks, NOAA RONI/ONI sustained-threshold classification, risk engine bounds/`NaN` guards, and end-to-end rendering of all 8 Streamlit pages.
2. **Terminal End-to-End Verification Output:**
   - **Live Open-Meteo Weather:** Real-time current temperature, relative humidity, dew point, 7-day forecast, and recent 5-year historical comparison for Nashik (`19.9975, 73.7898`) or any searched global location.
   - **Live NOAA CPC ENSO Status:** Real-time `RONI` and `ONI` values, latest 3-month overlapping season (e.g., `ASO 2026`), NOAA `Last-Modified` date, and CPC Alert System status.
   - **Experimental Disease-Risk Index & What-If Simulator:** Daily `0–100` environmental favourability scores plus hypothetical What-If sensitivity recalculation.
   - **MobileNetV2 Leaf Inference:** Real softmax probabilities, foliar validation, and domain-shift uncertainty warnings on a held-out PlantVillage test leaf.
3. **Interactive 8-Page Web Dashboard (`http://localhost:8501`):**
   - **Overview:** KPI summary cards, 7-day risk and temperature charts, and top preventive advisories.
   - **AI Disease Detection:** Upload your own leaf image (`.jpg`, `.png`, `.webp`, `.bmp`) or pick a held-out sample from the optional expander, then click **Run MobileNetV2 Diagnosis**.
   - **Weather Intelligence:** Search any city/town worldwide (e.g., `Pune`, `Kolar`, `Fresno`) or enter custom coordinates.
   - **Disease Risk Forecast:** Inspect daily factor contributions and test temperature/humidity/rainfall sliders in the **What-If Micro-Climate Simulator**.
   - **Climate Intelligence:** Inspect NOAA RONI/ONI 36-season trends and recent 5-year historical comparisons.
   - **Farmer Advisory:** Non-chemical preventive recommendations linked to weather risk and any active leaf scan.
   - **Model Performance:** Real PlantVillage (`99.23%`) vs. PlantDoc (`60.26%` raw, `52.38%` balanced, `4.76%` specificity) metrics, confusion matrices, and domain-shift diagnostics.
   - **About the Research:** Architecture, methodology, verified citations, and limitations.

---

## 4. Optional Training & Re-Evaluation Commands

```bash
# Re-run model evaluation without retraining
python3 models/evaluate.py

# Retrain MobileNetV2 from scratch (2 stages: head + top-25 non-BN fine-tuning)
python3 models/train.py --head-epochs 6 --finetune-epochs 6 --batch-size 32

# Resume training safely into an isolated directory without overwriting best saved weights
python3 models/train.py --resume --finetune-epochs 2 --output-dir /tmp/agriguard_resumed
```


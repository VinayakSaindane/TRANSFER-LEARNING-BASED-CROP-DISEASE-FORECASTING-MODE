#!/usr/bin/env bash
# AgriGuard AI — One-Command Setup, Verification & Launch Script
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="launch"
SKIP_TESTS=false
PORT="${PORT:-8501}"

for arg in "$@"; do
  case "$arg" in
    --setup)
      MODE="setup"
      ;;
    --verify-only)
      MODE="verify"
      ;;
    --fast)
      SKIP_TESTS=true
      ;;
    --help|-h)
      cat <<'EOF'
Usage: ./run.sh [OPTIONS]

Options:
  (no flags)      Ensure dependencies/datasets/model exist, run verification + pytest, and start Streamlit UI
  --setup         Create .venv (if outside venv), install requirements.txt, prepare datasets if missing, and verify
  --verify-only   Run pytest and print live Open-Meteo, NOAA RONI, Risk Engine, and MobileNetV2 inference output (no server)
  --fast          Skip pytest and launch Streamlit directly (after checking datasets/model)
  --help, -h      Show this help message
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $arg (use --help for usage)" >&2
      exit 1
      ;;
  esac
done

echo "============================================================"
echo " AgriGuard AI — Crop Disease & Micro-Climate DSS"
echo "============================================================"

# 1. Select Python / virtual environment
PYTHON_BIN=""
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: Python 3.10+ is required but was not found on PATH." >&2
  exit 1
fi

if [[ "$MODE" == "setup" ]] && [[ -z "${VIRTUAL_ENV:-}" ]] && [[ ! -d "$ROOT_DIR/.venv" ]]; then
  echo "[1/5] Creating virtual environment at .venv ..."
  "$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi

echo "[1/5] Using Python: $PYTHON_BIN ($("$PYTHON_BIN" --version))"

# 2. Check / install dependencies
if [[ "$MODE" == "setup" ]] || ! "$PYTHON_BIN" -c "from config.settings import ACTIVE_KERAS_BACKEND; import streamlit, keras, torch, plotly, cv2, sklearn, pytest" >/dev/null 2>&1; then
  echo "[2/5] Installing Python dependencies from requirements.txt ..."
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"
else
  echo "[2/5] Python dependencies already installed."
fi

# Ensure .env exists if user wants local overrides (optional; no API keys required)
if [[ ! -f "$ROOT_DIR/.env" ]] && [[ -f "$ROOT_DIR/.env.example" ]]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
  echo "      Created default .env from .env.example (no API keys required)."
fi

# 3. Prepare datasets if missing (needed for held-out test sample selector and dataset/prediction tests)
if [[ ! -f "$ROOT_DIR/data/processed/splits.csv" ]] || [[ ! -f "$ROOT_DIR/data/processed/plantdoc_eval.csv" ]]; then
  echo "[3/5] Dataset splits not found locally; downloading & preparing PlantVillage + PlantDoc..."
  "$PYTHON_BIN" "$ROOT_DIR/data/prepare_datasets.py"
else
  echo "[3/5] Verified dataset splits already present (data/processed/splits.csv)."
fi

# 4. Ensure trained model and evaluation reports exist
if [[ ! -f "$ROOT_DIR/models/saved/agriguard_mobilenetv2.keras" ]]; then
  echo "[4/5] Saved model not found; training MobileNetV2 and running evaluation..."
  "$PYTHON_BIN" "$ROOT_DIR/models/train.py" --head-epochs 6 --finetune-epochs 6 --batch-size 32
  "$PYTHON_BIN" "$ROOT_DIR/models/evaluate.py"
elif [[ ! -f "$ROOT_DIR/data/reports/evaluation_metrics.json" ]]; then
  echo "[4/5] Evaluation report missing; running models/evaluate.py..."
  "$PYTHON_BIN" "$ROOT_DIR/models/evaluate.py"
else
  echo "[4/5] Pre-trained MobileNetV2 model and evaluation metrics verified."
fi

# 5. Run live end-to-end verification + pytest
if [[ "$SKIP_TESTS" == "false" ]]; then
  echo "[5/5] Running automated test suite (pytest -v)..."
  "$PYTHON_BIN" -m pytest -v

  echo ""
  echo "----- Live End-to-End Output Verification -----"
  "$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path
from config.settings import ACTIVE_KERAS_BACKEND, BASE_DIR, DEFAULT_LOCATION, SPLIT_MANIFEST_PATH
from models.predict import predict_leaf_image
from services.advisory_service import generate_farmer_advisories
from services.climate_service import fetch_climate_status
from services.risk_engine import assess_seven_day_forecast, simulate_what_if_scenario
from services.weather_service import fetch_historical_seasonal_baseline, fetch_weather_data

weather = fetch_weather_data(DEFAULT_LOCATION["latitude"], DEFAULT_LOCATION["longitude"], DEFAULT_LOCATION)
baseline = fetch_historical_seasonal_baseline(DEFAULT_LOCATION["latitude"], DEFAULT_LOCATION["longitude"])
climate = fetch_climate_status()
risk = assess_seven_day_forecast(weather.get("daily") or [])
what_if = simulate_what_if_scenario(weather["daily"][0], 26.5, 92.0, 15.0) if weather.get("daily") else {}

with SPLIT_MANIFEST_PATH.open("r", encoding="utf-8") as f:
    test_row = next(r for r in csv.DictReader(f) if r["split"] == "test" and r["label"] == "Tomato Early Blight")
img_path = BASE_DIR / test_row["processed_path"]
pred = predict_leaf_image(img_path.read_bytes(), filename=img_path.name)
advisories = generate_farmer_advisories(risk, latest_prediction=pred)

print(f"Keras Backend : {ACTIVE_KERAS_BACKEND}")
cur = weather.get("current") or {}
print(
    f"Weather       : [{weather.get('status')}] {DEFAULT_LOCATION['name']} "
    f"Temp={cur.get('temperature_2m')} C, RH={cur.get('relative_humidity_2m')}%, "
    f"Retrieved={weather.get('retrieved_at')}"
)
print(
    f"5-Yr Archive  : [{baseline.get('status')}] Mean Temp={baseline.get('mean_temp_c')} C "
    f"({baseline.get('years_sampled')})"
)
print(
    f"NOAA ENSO     : [{climate.get('status')}] {climate.get('latest_season')} | "
    f"RONI={climate.get('latest_roni')} C | ONI={climate.get('latest_oni')} C | "
    f"Phase={climate.get('phase')} | CPC={climate.get('cpc_alert_status')}"
)
if risk.get("days"):
    d0 = risk["days"][0]
    print(
        f"Risk Engine   : Day 1={d0['score']}/100 ({d0['category']}), "
        f"7-Day Mean={risk['mean_score']}/100 | What-If (26.5C/92%/15mm)={what_if['hypothetical']['score']}/100"
    )
print(
    f"AI Screening  : Sample={img_path.name} -> {pred['predicted_class']} "
    f"({pred['confidence']:.1%} softmax; {pred['certainty_label']})"
)
print(f"Advisories    : {len(advisories)} non-chemical preventive recommendations generated")
PY
  echo "-----------------------------------------------"
else
  echo "[5/5] Skipping pytest (--fast specified)."
fi

if [[ "$MODE" == "verify" ]]; then
  echo "Verification complete (--verify-only). No web server started."
  exit 0
fi

echo ""
echo "Launching AgriGuard AI Streamlit Dashboard at http://localhost:$PORT ..."
echo "Press Ctrl+C to stop."
exec "$PYTHON_BIN" -m streamlit run "$ROOT_DIR/app.py" --server.port "$PORT"


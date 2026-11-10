# AgriGuard AI Methodology

## 1. Research Design Overview

AgriGuard AI connects three software-only intelligence streams into a transparent agricultural decision-support pipeline:
1. **Visual Foliar Screening (Experimental Studio Prototype):** Transfer learning with ImageNet-pretrained `MobileNetV2` distinguishing `Healthy` (index `0`) vs. `Tomato Early Blight` (index `1`) leaves. Evaluated both on held-out leaf-grouped PlantVillage images (`99.23%` accuracy) and isolated zero-shot PlantDoc field images (`60.26%` raw accuracy, `52.38%` balanced accuracy, `4.76%` specificity, demonstrating severe domain shift).
2. **Location-Level Modelled Environmental Favourability Scoring:** Experimental weighted scoring (0–100) using location-level NWP forecasts from Open-Meteo (explicitly distinguished from farm-level sensor or measured leaf wetness).
3. **Macro-Climate ENSO Context:** Live monitoring of NOAA CPC's Relative Oceanic Niño Index (RONI), reference ONI, and CPC ENSO Alert System status using NOAA's `±0.5°C` threshold over **≥ 5 consecutive overlapping 3-month seasons** (and official climate context when developing), alongside a **recent 5-year historical comparison** (`2021–2025`, not a 30-year climatological normal).

Crucially, macro-climate ENSO phase is treated strictly as seasonal context and never alters the daily disease-risk score directly; local modelled weather variables drive disease favourability.

---

## 2. Dataset Integrity & Leakage Prevention

- **Target Subset:** Only `Tomato___healthy` and `Tomato___Early_blight` from PlantVillage are included.
- **Validation & Deduplication:** Every image is decoded and verified with Pillow; exact SHA-256 hashing removed **6 byte-identical duplicates** inside `Tomato___healthy`.
- **Leaf-Grouped Stratified Splitting:** PlantVillage photographs often include multiple images of the same physical leaf (`leaf_grouping/filtered_leafmaps/`). Images are grouped by physical leaf ID (`863` unique leaf groups across `2,585` unique valid images) before performing a `70% / 15% / 15%` stratified split (`seed=42`). Zero leaf-group overlap and zero hash overlap across train/val/test is enforced via assertions.
- **External Field Benchmark & Domain-Shift Analysis:** PlantDoc tomato classes (`151` images: `Tomato leaf` → `Healthy` [`63`], `Tomato Early blight leaf` → `Tomato Early Blight` [`88`]) are isolated in `data/processed/plantdoc_eval/` and excluded from all training/validation/tuning. On PlantDoc (`[[3, 60], [0, 88]]`), the classifier predicts `Tomato Early Blight` on `148/151` (`98.01%`) of images, yielding `4.76%` specificity and `52.38%` balanced accuracy (`60.26%` raw accuracy). Thus, current weights must not be claimed as reliable for unverified field diagnosis.

---

## 3. Transfer Learning & Preprocessing Pipeline

- **Identical Preprocessing:** Both training and inference resize RGB inputs to `224×224` and apply `keras.applications.mobilenet_v2.preprocess_input` (`[-1.0, +1.0]` scaling).
- **Architecture:**
  - `MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))`
  - `GlobalAveragePooling2D()`
  - `Dropout(0.30)`
  - `Dense(2, activation="softmax")`
- **Two-Stage Training:**
  - *Stage 1 (Head):* Backbone frozen (`training=False`), balanced class weights, `Adam(lr=1e-3)`, `EarlyStopping`, `ModelCheckpoint`.
  - *Stage 2 (Fine-Tuning):* Top `25` non-BatchNormalization layers unfrozen, `Adam(lr=5e-5)`, `ReduceLROnPlateau`, `EarlyStopping(restore_best_weights=True)`.
- **Inference Guardrails:** Uploads pass extension/size checks, Pillow decoding, minimum resolution (`64×64`), contrast/exposure screening, HSV foliar coloration screening (`>= 5%`), and confidence uncertainty gating (`< 65%` triggers low-confidence retake guidance).

---

## 4. Explainable Disease Risk Engine

Daily environmental favourability score $S \in [0, 100]$ is computed as:

$$S = 100 \times \left(0.30 f_{\text{RH}} + 0.25 f_{T} + 0.25 f_{\text{Rain}} + 0.20 f_{\text{Wet}}\right)$$

Where:
- $f_{\text{RH}} \in [0,1]$: Piecewise moisture curve ramping steeply above $65\%$ RH toward saturation ($\ge 92\%$).
- $f_{T} \in [0,1]$: Thermal suitability curve peaking at $24^\circ\text{C}\text{–}29^\circ\text{C}$ (*Alternaria solani* optimum) and tapering to $0$ at $\le 10^\circ\text{C}$ or $\ge 38^\circ\text{C}$.
- $f_{\text{Rain}} \in [0,1]$: Precipitation splash-dispersal curve saturating at $25\text{ mm/day}$. Rainfall alone never forces leaf wetness to $1.0$.
- $f_{\text{Wet}} \in [0,1]$: Dew/wetness proxy combining dew-point depression ($T - T_d$, weighted $0.60$) and daily hours with $\text{RH} \ge 85\%$ (weighted $0.40$). When hourly duration is absent (e.g., What-If simulation), dew-point depression estimated via Magnus-Tetens is used directly.

Prototype bands: `0–30 Low` | `31–60 Moderate` | `61–80 High` | `81–100 Very High`.

> **Scientific Boundary:** This is an experimental environmental disease-favourability index, not a validated probability of future infection. Open-Meteo supplies gridded location-level modelled NWP estimates and does not measure actual farm-level canopy micro-climate or physical leaf wetness. Both baseline forecast scoring and the What-If sensitivity simulator execute the identical `compute_environmental_risk()` scoring functions.

---

## 5. NOAA ENSO / RONI Classification Methodology

- **Operational Index:** NOAA CPC Relative Oceanic Niño Index (`RONI.ascii.txt`, ERSSTv6, 1991–2020 base period), which subtracts the 20°N–20°S tropical mean SST anomaly from Niño 3.4 and scales variance to match Niño 3.4 ([NOAA CPC RONI](https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/)).
- **Sustained Episode Rule:** Historical El Niño (`≥ +0.5°C`) and La Niña (`≤ -0.5°C`) episodes require at least **5 consecutive overlapping 3-month seasons**. A single overlapping seasonal observation crossing `±0.5°C` does **not** classify an episode.
- **Official Climate Context Cross-Check:** Developing multi-season RONI runs (`2–4` consecutive seasons) are only confirmed active when corroborated by official NOAA CPC ENSO Alert System Advisories (`ensodisc.shtml`) and/or 5-season sustained reference ONI (`oni.ascii.txt`). RONI and ONI never alter the local disease-risk score.

---

## 6. Farmer Advisory Linking

Advisories combine actual 7-day risk drivers (humidity/dew duration, rainfall splash, optimal 24–29°C windows) with any active session AI leaf scan to recommend non-chemical cultural actions (basal morning irrigation, mulching, pruning for airflow, lower-canopy scouting, and consulting local agricultural extension services).


# Dataset Preparation and Integrity Report

Generated directly from `data/prepare_datasets.py` and verified against `data/reports/dataset_statistics.json`.

## 1. PlantVillage Acquisition & Cleaning Summary

| Metric | Healthy | Tomato Early Blight | Total |
|---|---:|---:|---:|
| Raw downloaded files | 1,591 | 1,000 | **2,591** |
| Corrupted / unreadable removed | 0 | 0 | **0** |
| Exact SHA-256 duplicates removed | 6 | 0 | **6** |
| Valid unique images retained | 1,585 | 1,000 | **2,585** |

### Removed Exact SHA-256 Duplicates
Six files in `Tomato___healthy` were byte-for-byte identical copies stored under different UUID prefixes (`GH_HL Leaf 466.1`, `389`, `434`, `482.2`, `220`, and `342.1`). Removing them before splitting eliminates exact-copy train/test contamination.

### Physical Leaf Grouping (Anti-Leakage Control)
- PlantVillage contains multiple angles/crops of the same physical detached leaf.
- `1,999` images matched directly against PlantVillage's `leaf_grouping/filtered_leafmaps/` (`Tomato___healthy.csv`, `Tomato___Early_blight.csv`) and `leaf-map.json` after stripping UUID prefixes.
- Remaining `GH_HL Leaf <N>` multi-angle variants were grouped by physical leaf identifier.
- Total distinct physical leaf groups across all `2,585` images: **863**.

## 2. Reproducible Stratified Grouped Splits (`seed=42`)

| Split | Total Images | Share | Healthy | Tomato Early Blight | Leaf Group Overlap | SHA-256 Overlap |
|---|---:|---:|---:|---:|---:|---:|
| **Train** | 1,805 | 69.83% | 1,109 | 696 | 0 | 0 |
| **Validation** | 390 | 15.09% | 238 | 152 | 0 | 0 |
| **Held-Out Test** | 390 | 15.09% | 238 | 152 | 0 | 0 |

## 3. Preprocessing Specification

- **Original resolution:** All 2,585 valid PlantVillage images were `256×256` RGB.
- **Target resolution:** Resized via bilinear interpolation to **`224×224` RGB** (stored as clean JPEG quality 95 in `data/processed/{train,val,test}/`).
- **Model input normalization:** `keras.applications.mobilenet_v2.preprocess_input` scaling pixel values from `[0, 255]` to `[-1.0, +1.0]`, shared identically by training, evaluation, and inference.

## 4. PlantDoc External Evaluation Set

| Class | Raw Count | Corrupted | Internal Duplicates | Overlap w/ PlantVillage | Valid Eval Count |
|---|---:|---:|---:|---:|---:|
| **Healthy** | 63 | 0 | 0 | 0 | **63** |
| **Tomato Early Blight** | 88 | 0 | 0 | 0 | **88** |
| **Total** | **151** | **0** | **0** | **0** | **151** |

All PlantDoc images were resized to `224×224` RGB in `data/processed/plantdoc_eval/` strictly for post-training domain-shift evaluation.


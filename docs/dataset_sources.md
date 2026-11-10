# Dataset Sources, Scope, Licenses, and Limitations

## 1. Primary Training Dataset — PlantVillage

- **Source Repository:** [https://github.com/spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- **Foundational Reference:** Hughes & Salathé (2015), *arXiv:1511.08060*; Mohanty et al. (2016), *Frontiers in Plant Science*.
- **License:** Creative Commons Attribution-ShareAlike 3.0 (`CC-BY-SA-3.0`, per Hugging Face `mohanty/PlantVillage` dataset card).
- **Selected Subset:** Strictly restricted to the two target tomato classes:
  - `raw/color/Tomato___healthy` (**1,591** raw images)
  - `raw/color/Tomato___Early_blight` (**1,000** raw images)
- **Role in AgriGuard AI:** Deduplicated and leaf-grouped into 70% Training / 15% Validation / 15% Held-Out Testing splits.
- **Known Limitations:**
  - Leaves were detached and photographed in controlled laboratory lighting over uniform grey/neutral backgrounds.
  - Because studio backgrounds lack weeds, overlapping canopies, shadows, and soil clutter, models trained solely on PlantVillage experience significant domain shift on raw smartphone field photographs.
  - Class imbalance exists (`1,591` Healthy vs. `1,000` Early Blight), addressed via balanced class weighting during training.

---

## 2. External Field Evaluation Dataset — PlantDoc

- **Source Repository:** [https://github.com/pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)
- **Foundational Reference:** Singh et al. (2020), *Proceedings of ACM CoDS-COMAD 2020*, DOI: `10.1145/3371158.3371196`.
- **License:** Creative Commons Attribution 4.0 International (`CC-BY-4.0`, verified from `LICENSE.txt` in repository).
- **Selected Subset:** Cropped tomato classification folders corresponding to the two target classes:
  - `train/Tomato leaf` (55) + `test/Tomato leaf` (8) → **63** `Healthy` images
  - `train/Tomato Early blight leaf` (79) + `test/Tomato Early blight leaf` (9) → **88** `Tomato Early Blight` images
- **Role in AgriGuard AI:** **Isolated external evaluation only.** Never mixed into training or validation splits. Used strictly to measure zero-shot generalization under real-world backgrounds and variable lighting.
- **Known Limitations:**
  - Small sample size (`151` total images across both classes).
  - Internet-scraped field photographs exhibit variable focus, multiple leaves per frame, lighting glare, and occasional label noise noted in the original PlantDoc study.


"""Page 8: About the Research, Methodology, Citations & Limitations."""

from __future__ import annotations

import streamlit as st

from components.cards import render_callout
from components.styles import render_page_header


def render_research_page() -> None:
    render_page_header(
        "About the Research",
        "Transfer Learning Based Crop Disease Forecasting Model Linked to El Niño-Induced Micro-climate Variations.",
        badges=[("Software-Only AI/ML Prototype", "ag-pill-green"), ("Zero Hardware / IoT Dependencies", "ag-pill-slate")],
    )

    with st.container(border=True):
        st.markdown("### Abstract")
        st.write(
            "AgriGuard AI is a software-only agricultural decision-support research prototype integrating "
            "(1) MobileNetV2 transfer learning for visual screening of tomato early blight (Alternaria solani), "
            "(2) location-level modelled 7-day weather forecasts via Open-Meteo, (3) an experimental weighted environmental "
            "favourability risk engine, and (4) macro-climate ENSO monitoring via NOAA's Relative Oceanic Niño Index (RONI). "
            "Rather than falsely conflating large-scale Pacific climate indices with direct disease causation, the platform "
            "uses ENSO for seasonal context while driving daily disease-risk calculations strictly from local temperature, "
            "relative humidity, precipitation, and dew-point wetness proxies."
        )

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("### Problem Statement")
            st.write(
                "Smallholder tomato growers face severe yield losses from foliar fungal pathogens such as Alternaria solani, "
                "which rapidly escalate when warm temperatures (24–29°C) overlap with high humidity and persistent leaf wetness. "
                "Physical IoT weather stations are costly to deploy at farm scale, while purely image-based classifiers are reactive "
                "(detecting disease only after necrotic lesions appear) and often fail under field domain shift."
            )
    with c2:
        with st.container(border=True):
            st.markdown("### Research Objectives")
            st.markdown(
                """
1. Train a lightweight MobileNetV2 transfer-learning classifier on strictly deduplicated, leaf-grouped PlantVillage tomato images.
2. Quantify real-world domain shift transparently using an isolated external field dataset (PlantDoc).
3. Build an explainable 0–100 environmental favourability model driven by live 7-day weather forecasts.
4. Contextualize local weather anomalies against official NOAA CPC RONI / ONI climate phases without inventing artificial multipliers.
5. Deliver preventive, non-chemical farmer advisories tied directly to forecast risk and visual scans.
"""
            )

    with st.container(border=True):
        st.markdown("### End-to-End System Architecture & Methodology")
        st.markdown(
            """
- **Visual Screening Layer:** Input image validation (format, resolution, contrast, foliar color screening) → `224×224` normalization (`[-1, 1]`) → ImageNet-pretrained `MobileNetV2` + `GlobalAveragePooling2D` + `Dropout(0.30)` + 2-class Softmax + uncertainty threshold (`< 65%`; softmax confidence is not diagnostic certainty).
- **Weather Telemetry Layer:** Open-Meteo Forecast, Geocoding, and Historical Archive APIs (gridded NWP location-level estimates, not direct farm sensors or measured leaf wetness) with schema validation, local JSON caching, and transparent unavailable states.
- **Disease Risk Engine:** Experimental weighted prototype (`30%` Relative Humidity, `25%` Temperature Suitability, `25%` Rainfall Splash, `20%` Dew-Point Depression / High-RH Wetness Proxy) + interactive What-If Sensitivity Simulator (not a validated probability of future infection).
- **Macro-Climate Layer:** Live parsing of NOAA CPC `RONI.ascii.txt`, `oni.ascii.txt`, and CPC ENSO Diagnostic Discussion status under NOAA's `±0.5°C` threshold across `>= 5` consecutive overlapping 3-month seasons.
"""
        )

    with st.container(border=True):
        st.markdown("### Key Peer-Reviewed Literature & Official Sources")
        st.markdown(
            """
1. **Hughes, D. P., & Salathé, M. (2015).** *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060. DOI: `10.48550/arXiv.1511.08060`
2. **Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016).** *Using Deep Learning for Image-Based Plant Disease Detection.* Frontiers in Plant Science, 7:1419. DOI: `10.3389/fpls.2016.01419`
3. **Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018).** *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* IEEE/CVF CVPR, pp. 4510–4520. DOI: `10.1109/CVPR.2018.00474`
4. **Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020).** *PlantDoc: A Dataset for Visual Plant Disease Detection.* ACM CoDS-COMAD 2020, pp. 249–253. DOI: `10.1145/3371158.3371196`
5. **Madden, L., Pennypacker, S. P., & MacNab, A. A. (1978).** *FAST, a forecast system for Alternaria solani on tomato.* Phytopathology, 68(9), 1354–1358. DOI: `10.1094/Phyto-68-1354`
6. **Attri, K., Gupta, P., Kansal, S., Gupta, M., Goswami, M., & Thakur, S. (2024).** *Influence of Temperature and Relative Humidity on Spore Germination and Early Blight Disease Development of Tomato Caused by Alternaria solani.* Int. J. Econ. Plants. DOI: `10.23910/2/2024.5501a`
7. **L'Heureux, M. L., et al. (2024).** *A Relative Sea Surface Temperature Index for Classifying ENSO Events in a Changing Climate.* Journal of Climate. DOI: `10.1175/JCLI-D-23-0406.1`
8. **Rasmusson, E. M., & Carpenter, T. H. (1983).** *The Relationship Between Eastern Equatorial Pacific Sea Surface Temperatures and Rainfall over India and Sri Lanka.* Monthly Weather Review, 111(3), 517–528. DOI: `10.1175/1520-0493(1983)111<0517:TRBEEP>2.0.CO;2`
9. **Krishna Kumar, K., Rajagopalan, B., Hoerling, M., Bates, G., & Cane, M. (2006).** *Unraveling the Mystery of Indian Monsoon Failure During El Niño.* Science, 314(5796), 115–119. DOI: `10.1126/science.1131152`
10. **Ahmad, A., Saraswat, D., & El Gamal, A. (2023).** *A Survey on Using Deep Learning Techniques for Plant Disease Diagnosis and Recommendations for Development of Appropriate Tools.* Smart Agricultural Technology, 3:100083. DOI: `10.1016/j.atech.2022.100083`
"""
        )

    l1, l2 = st.columns(2)
    with l1:
        render_callout(
            "Key Research Contributions",
            "Demonstrates zero-leakage leaf-grouped PlantVillage evaluation, transparent quantification of PlantDoc real-world domain shift, and integration of NOAA's 2026 operational RONI standard with explainable micro-climate risk scoring.",
            variant="info",
        )
    with l2:
        render_callout(
            "Scope Limitations & Future Work",
            "Binary classification only; zero-shot PlantDoc field testing (52.38% balanced accuracy, 4.76% specificity) shows current studio-trained weights are not reliable for unverified field diagnosis without field-domain adaptation. Prototype risk weights require multi-season empirical calibration with field disease-incidence trials before operational deployment.",
            variant="warn",
        )


# Literature Review and Verified Research References

All references listed below have been verified against published journals, conference proceedings, official arXiv preprints, and government climate bulletins. No citations, authors, DOIs, or findings have been fabricated.

---

## 1. Transfer Learning & MobileNetV2 for Crop Disease Identification

1. **Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018).**
   - **Title:** *MobileNetV2: Inverted Residuals and Linear Bottlenecks*
   - **Venue:** 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Salt Lake City, UT, USA, pp. 4510–4520.
   - **DOI:** [10.1109/CVPR.2018.00474](https://doi.org/10.1109/CVPR.2018.00474)
   - **Relevance:** Establishes the lightweight inverted residual / linear bottleneck CNN backbone used in AgriGuard AI for CPU- and edge-efficient transfer learning.

2. **Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016).**
   - **Title:** *Using Deep Learning for Image-Based Plant Disease Detection*
   - **Journal:** *Frontiers in Plant Science*, 7:1419.
   - **DOI:** [10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419)
   - **Relevance:** Landmark benchmark demonstrating transfer learning effectiveness on PlantVillage images while explicitly noting severe accuracy drops when evaluating on images outside controlled laboratory conditions.

3. **Ahmad, A., Saraswat, D., & El Gamal, A. (2023).**
   - **Title:** *A Survey on Using Deep Learning Techniques for Plant Disease Diagnosis and Recommendations for Development of Appropriate Tools*
   - **Journal:** *Smart Agricultural Technology*, Vol. 3, Article 100083.
   - **DOI:** [10.1016/j.atech.2022.100083](https://doi.org/10.1016/j.atech.2022.100083)
   - **Relevance:** Reviews architectural tradeoffs, domain-shift pitfalls, and design requirements for farmer-facing AI agricultural decision-support tools.

4. **Hybrid MobileNetV2 & Tomato Leaf Classification Studies (2023):**
   - **Title:** *An Improved MobileNet for Disease Detection on Tomato Leaves*
   - **Journal:** *Applied Technologies and Innovations* (2023).
   - **DOI:** [10.46604/aiti.2023.11568](https://doi.org/10.46604/aiti.2023.11568)
   - **Relevance:** Evaluates lightweight MobileNet variants specifically on tomato foliar diseases including early blight.

---

## 2. Primary & External Evaluation Datasets

5. **Hughes, D. P., & Salathé, M. (2015).**
   - **Title:** *An open access repository of images on plant health to enable the development of mobile disease diagnostics*
   - **Venue:** arXiv Preprint (`arXiv:1511.08060 [cs.CY]`).
   - **DOI:** [10.48550/arXiv.1511.08060](https://doi.org/10.48550/arXiv.1511.08060)
   - **Relevance:** Primary training dataset release (PlantVillage), documenting controlled background acquisition and physical leaf grouping considerations.

6. **Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020).**
   - **Title:** *PlantDoc: A Dataset for Visual Plant Disease Detection*
   - **Venue:** Proceedings of the 7th ACM IKDD CoDS and 25th COMAD (CoDS-COMAD 2020), Hyderabad, India, pp. 249–253.
   - **DOI:** [10.1145/3371158.3371196](https://doi.org/10.1145/3371158.3371196)
   - **Relevance:** Used strictly as an unmixed external evaluation benchmark in AgriGuard AI to quantify domain shift on natural-background photographs.

---

## 3. Weather-Driven Tomato Early Blight (*Alternaria solani*) Epidemiology

7. **Madden, L. V., Pennypacker, S. P., & MacNab, A. A. (1978).**
   - **Title:** *FAST, a forecast system for Alternaria solani on tomato*
   - **Journal:** *Phytopathology*, 68(9), 1354–1358.
   - **DOI:** [10.1094/Phyto-68-1354](https://doi.org/10.1094/Phyto-68-1354)
   - **Relevance:** Foundational weather-driven early blight forecasting system establishing how combinations of temperature, relative humidity, rainfall, and leaf wetness govern *A. solani* daily severity values.

8. **Rotem, J. (1994).**
   - **Title:** *The Genus Alternaria: Biology, Epidemiology, and Pathogenicity*
   - **Publisher:** APS Press, St. Paul, MN, USA.
   - **URL:** [https://www.apsnet.org/](https://www.apsnet.org/)
   - **Relevance:** Comprehensive epidemiological reference defining moisture regimes (dew, rainfall, RH) and temperature curves for *Alternaria* conidial germination and sporulation.

9. **Attri, K., Gupta, P., Kansal, S., Gupta, M., Goswami, M., & Thakur, S. (2024).**
   - **Title:** *Influence of Temperature and Relative Humidity on Spore Germination and Early Blight Disease Development of Tomato Caused by Alternaria solani*
   - **Journal:** *International Journal of Economic Plants* (2024).
   - **DOI:** [10.23910/2/2024.5501a](https://doi.org/10.23910/2/2024.5501a)
   - **Relevance:** Confirms optimal *A. solani* temperature regimes (24–30°C) and elevated humidity requirements (80–100% RH) informing AgriGuard AI's risk engine thresholds.

---

## 4. ENSO / RONI Climate Monitoring & Monsoon Variability

10. **L'Heureux, M. L., Tippett, M. K., Wheeler, M. C., Nguyen, H., Narsey, S., Johnson, N., Hu, Z.-Z., Watkins, A. B., Lucas, C., Ganter, C., Becker, E., Wang, W., & Di Liberto, T. (2024).**
    - **Title:** *A Relative Sea Surface Temperature Index for Classifying ENSO Events in a Changing Climate*
    - **Journal:** *Journal of Climate* (American Meteorological Society).
    - **DOI:** [10.1175/JCLI-D-23-0406.1](https://doi.org/10.1175/JCLI-D-23-0406.1)
    - **Relevance:** Defines the Relative Oceanic Niño Index (RONI), adopted operationally by NOAA CPC in February 2026 and fetched directly from NOAA CPC's `RONI.ascii.txt` in AgriGuard AI.

11. **Rasmusson, E. M., & Carpenter, T. H. (1983).**
    - **Title:** *The Relationship Between Eastern Equatorial Pacific Sea Surface Temperatures and Rainfall over India and Sri Lanka*
    - **Journal:** *Monthly Weather Review*, 111(3), 517–528.
    - **DOI:** [10.1175/1520-0493(1983)111<0517:TRBEEP>2.0.CO;2](https://doi.org/10.1175/1520-0493(1983)111<0517:TRBEEP>2.0.CO;2)
    - **Relevance:** Establishes the classical teleconnection linking equatorial Pacific SST anomalies (El Niño) to suppressed Indian summer monsoon rainfall.

12. **Krishna Kumar, K., Rajagopalan, B., & Cane, M. A. (1999).**
    - **Title:** *On the Weakening Relationship between the Indian Monsoon and ENSO*
    - **Journal:** *Science*, 284(5423), 2156–2159.
    - **DOI:** [10.1126/science.284.5423.2156](https://doi.org/10.1126/science.284.5423.2156)
    - **Relevance:** Explains non-stationary ENSO–monsoon coupling, reinforcing why AgriGuard AI uses ENSO strictly as macro-climate context rather than a hardcoded disease-risk multiplier.

13. **Krishna Kumar, K., Rajagopalan, B., Hoerling, M., Bates, G., & Cane, M. (2006).**
    - **Title:** *Unraveling the Mystery of Indian Monsoon Failure During El Niño*
    - **Journal:** *Science*, 314(5796), 115–119.
    - **DOI:** [10.1126/science.1131152](https://doi.org/10.1126/science.1131152)
    - **Relevance:** Demonstrates why Central Pacific vs. Eastern Pacific warming shapes Indian regional rainfall anomalies differently.

14. **Official Data & APIs:**
    - **NOAA Climate Prediction Center (CPC) Indices:** [https://www.cpc.ncep.noaa.gov/data/indices/](https://www.cpc.ncep.noaa.gov/data/indices/) (`RONI.ascii.txt`, `oni.ascii.txt`)
    - **NOAA PSL ENSO Resource:** [https://psl.noaa.gov/enso/](https://psl.noaa.gov/enso/)
    - **Open-Meteo Weather, Geocoding & Historical Archive APIs:** [https://open-meteo.com/en/docs](https://open-meteo.com/en/docs)


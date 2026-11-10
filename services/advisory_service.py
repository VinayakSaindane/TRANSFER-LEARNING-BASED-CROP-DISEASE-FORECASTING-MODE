"""Preventive, weather- and AI-linked Farmer Advisory Engine."""

from __future__ import annotations

from typing import Any


def generate_farmer_advisories(
    risk_assessment: dict[str, Any] | None,
    latest_prediction: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Generate transparent, non-chemical preventive advisories from real forecast & AI outputs."""
    advisories: list[dict[str, str]] = []

    # 1. Link directly to latest AI leaf scan if available
    if latest_prediction and latest_prediction.get("status") == "ok":
        pred_class = latest_prediction.get("predicted_class", "")
        conf = float(latest_prediction.get("confidence", 0.0))
        uncertain = bool(latest_prediction.get("low_confidence", False))

        if pred_class == "Tomato Early Blight":
            if uncertain:
                advisories.append(
                    {
                        "title": "Follow-up Field Verification Required (Low-Confidence Scan)",
                        "priority": "Action Recommended",
                        "risk_level": "Uncertain AI Flag",
                        "relevant_forecast_day": "Immediate",
                        "reason": (
                            f"Latest AI scan returned 'Tomato Early Blight' with low confidence ({conf:.1%}). "
                            "Visual overlap with nutrient deficiency or natural background clutter is possible."
                        ),
                        "recommended_action": (
                            "Retake a clear, well-lit close-up photo of a single detached or flat leaf against a plain "
                            "background, and inspect lower mature leaves in the field for concentric dark ring lesions. "
                            "Consult a local agricultural officer before taking management action."
                        ),
                    }
                )
            else:
                advisories.append(
                    {
                        "title": "Isolate Symptomatic Foliage & Consult Local Agronomist",
                        "priority": "Urgent",
                        "risk_level": "AI Detection Alert",
                        "relevant_forecast_day": "Immediate (Today)",
                        "reason": (
                            f"Latest AI image analysis identified 'Tomato Early Blight' patterns at {conf:.1%} "
                            "model confidence (note: model confidence is not diagnostic certainty)."
                        ),
                        "recommended_action": (
                            "Inspect older lower leaves across adjacent tomato rows for target-like necrotic spots "
                            "with yellow halos. Carefully prune and dispose of heavily infected lower leaves away from "
                            "the field, sanitize pruning tools, avoid overhead watering, and consult your local Krishi "
                            "Vigyan Kendra (KVK) or state agricultural extension specialist for approved treatment options."
                        ),
                    }
                )
        elif pred_class == "Healthy":
            advisories.append(
                {
                    "title": "Maintain Preventive Canopy Hygiene",
                    "priority": "Routine",
                    "risk_level": "Healthy Scan",
                    "relevant_forecast_day": "Ongoing",
                    "reason": f"Latest AI leaf scan was classified as 'Healthy' ({conf:.1%} confidence).",
                    "recommended_action": (
                        "Continue periodic scouting of lower canopy foliage twice weekly, keep lower leaves clear "
                        "of soil splash using clean organic mulch, and verify irrigation runs early in the morning."
                    ),
                }
            )

    # 2. Link directly to 7-day weather-driven risk forecast
    days = (risk_assessment or {}).get("days") or []
    if not days or (risk_assessment or {}).get("status") != "ok":
        advisories.append(
            {
                "title": "Weather Feed Unavailable — Use Direct Field Scouting",
                "priority": "Preventive Monitoring",
                "risk_level": "Unavailable",
                "relevant_forecast_day": "7-Day Window",
                "reason": "Live weather forecast data could not be retrieved for environmental risk scoring.",
                "recommended_action": (
                    "Inspect lower tomato leaves after any overnight dew or rain event, ensure rows remain well "
                    "ventilated, and retry live weather sync when connectivity returns."
                ),
            }
        )
        return advisories

    high_days = [d for d in days if d.get("category") in {"High", "Very High"} or (d.get("score") or 0) >= 60.5]
    mod_days = [d for d in days if d.get("category") == "Moderate" or (30.5 <= (d.get("score") or 0) < 60.5)]
    rainy_days = [d for d in days if (d.get("precipitation_sum") or 0) >= 2.5]
    dew_days = [
        d for d in days
        if (d.get("dew_point_depression_mean") is not None and d["dew_point_depression_mean"] <= 2.5)
        or (d.get("rh_mean") is not None and d["rh_mean"] >= 80.0)
    ]
    warm_days = [d for d in days if d.get("temp_mean") is not None and 24.0 <= d["temp_mean"] <= 29.0]

    if high_days:
        peak = max(high_days, key=lambda d: d["score"])
        advisories.append(
            {
                "title": "Increase Canopy Scouting During High Environmental Favourability",
                "priority": "Urgent",
                "risk_level": f"{peak['category']} ({peak['score']:.1f}/100)",
                "relevant_forecast_day": f"Day {peak['day_index']} ({peak['date']}) & {len(high_days)} high-risk day(s)",
                "reason": peak["explanation"],
                "recommended_action": (
                    "Inspect lower and shaded interior leaves daily for early dark concentric lesions. "
                    "Avoid working in wet foliage to prevent mechanical spore spread, and seek locally approved "
                    "preventive extension guidance if symptoms emerge."
                ),
            }
        )

    if dew_days:
        first_dew = dew_days[0]
        advisories.append(
            {
                "title": "Avoid Overhead Irrigation & Shift Watering to Early Morning",
                "priority": "Action Recommended" if high_days else "Preventive Monitoring",
                "risk_level": f"{first_dew['category']} ({first_dew['score']:.1f}/100)",
                "relevant_forecast_day": ", ".join(f"Day {d['day_index']} ({d['date']})" for d in dew_days[:3]),
                "reason": (
                    f"Elevated humidity (up to {max(d['rh_mean'] or 0 for d in dew_days):.0f}% RH) and/or "
                    f"narrow dew-point depression indicate prolonged leaf surface moisture."
                ),
                "recommended_action": (
                    "Use basal drip or furrow irrigation strictly in early morning hours so foliage dries "
                    "rapidly after sunrise. Do not run sprinklers in late afternoon or evening."
                ),
            }
        )

    if rainy_days:
        heaviest = max(rainy_days, key=lambda d: d["precipitation_sum"] or 0)
        advisories.append(
            {
                "title": "Mitigate Soil Splash & Inspect Plots After Rainfall",
                "priority": "Action Recommended",
                "risk_level": f"{heaviest['category']} ({heaviest['score']:.1f}/100)",
                "relevant_forecast_day": f"Day {heaviest['day_index']} ({heaviest['date']})",
                "reason": (
                    f"Forecast precipitation of {heaviest['precipitation_sum']:.1f} mm on {heaviest['date']} "
                    "can splash soil-borne Alternaria spores onto lower tomato foliage."
                ),
                "recommended_action": (
                    "Ensure raised beds and field drainage channels are clear before rainfall, apply clean straw "
                    "or plastic mulch beneath plants to block soil splash, and scout lower leaves 24–48 hours "
                    "after rain stops."
                ),
            }
        )

    if warm_days and not high_days:
        sample = warm_days[0]
        advisories.append(
            {
                "title": "Improve Canopy Airflow During Optimal Temperature Window",
                "priority": "Preventive Monitoring",
                "risk_level": f"{sample['category']} ({sample['score']:.1f}/100)",
                "relevant_forecast_day": f"{len(warm_days)} forecast day(s) in 24–29°C range",
                "reason": (
                    f"Mean temperatures between 24°C and 29°C ({sample['temp_mean']:.1f}°C on {sample['date']}) "
                    "lie within the optimal thermal window for Alternaria solani if moisture rises."
                ),
                "recommended_action": (
                    "Maintain proper plant spacing, stake indeterminate varieties cleanly, remove non-productive "
                    "basal suckers to enhance airflow, and eliminate solanaceous weeds."
                ),
            }
        )

    if not high_days and not mod_days and not rainy_days and not dew_days:
        first = days[0]
        advisories.append(
            {
                "title": "Low Environmental Risk — Standard Preventive Field Hygiene",
                "priority": "Routine",
                "risk_level": f"Low ({first['score']:.1f}/100)",
                "relevant_forecast_day": "Days 1–7",
                "reason": (
                    "All 7 forecast days remain in the Low environmental favourability band (0–30/100) "
                    "due to low ambient moisture and/or lack of rainfall."
                ),
                "recommended_action": (
                    "Maintain balanced crop nutrition (adequate potassium and nitrogen help reduce early blight "
                    "susceptibility in stressed plants), keep drip lines calibrated, and continue weekly visual checks."
                ),
            }
        )

    return advisories


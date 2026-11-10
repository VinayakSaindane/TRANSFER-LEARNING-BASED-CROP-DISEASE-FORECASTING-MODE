"""Explainable environmental disease-risk scoring engine for Tomato Early Blight."""

from __future__ import annotations

import math
from typing import Any

from config.settings import RISK_THRESHOLDS, RISK_WEIGHTS


def _is_valid_num(val: Any) -> bool:
    if val is None or isinstance(val, (bool, str, bytes)):
        return False
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def estimate_dew_point_celsius(temp_c: float, rel_humidity_pct: float) -> float:
    """Estimate dew point (°C) using the Magnus-Tetens approximation (used in What-If simulation)."""
    rh = max(1.0, min(100.0, float(rel_humidity_pct)))
    t = max(-60.0, min(65.0, float(temp_c)))
    a, b = 17.625, 243.04
    gamma = math.log(rh / 100.0) + (a * t) / (b + t)
    denom = a - gamma
    if abs(denom) < 1e-9:
        return round(t, 2)
    return round((b * gamma) / denom, 2)


def score_temperature_suitability(temp_c: float) -> float:
    """Return normalized [0, 1] thermal suitability for Alternaria solani (optimal 24–29°C)."""
    t = float(temp_c)
    if t <= 10.0 or t >= 38.0:
        return 0.0
    if t < 18.0:
        return 0.05 + 0.40 * ((t - 10.0) / 8.0)
    if t < 24.0:
        return 0.45 + 0.55 * ((t - 18.0) / 6.0)
    if t <= 29.0:
        return 1.0
    if t <= 33.0:
        return 1.0 - 0.60 * ((t - 29.0) / 4.0)
    return max(0.0, 0.40 * (1.0 - (t - 33.0) / 5.0))


def score_relative_humidity(rh_pct: float) -> float:
    """Return normalized [0, 1] moisture suitability from mean/ambient relative humidity."""
    rh = max(0.0, min(100.0, float(rh_pct)))
    if rh < 40.0:
        return 0.05 * (rh / 40.0)
    if rh < 65.0:
        return 0.05 + 0.25 * ((rh - 40.0) / 25.0)
    if rh < 80.0:
        return 0.30 + 0.35 * ((rh - 65.0) / 15.0)
    if rh < 92.0:
        return 0.65 + 0.30 * ((rh - 80.0) / 12.0)
    return min(1.0, 0.95 + 0.05 * ((rh - 92.0) / 8.0))


def score_rainfall(precip_mm: float) -> float:
    """Return normalized [0, 1] splash-dispersal suitability from daily precipitation (mm).

    Note: Rainfall aids conidial splash dispersal, but does NOT alone confirm persistent leaf wetness.
    """
    mm = max(0.0, float(precip_mm))
    if mm == 0.0:
        return 0.0
    if mm <= 2.5:
        return 0.25 * (mm / 2.5)
    if mm <= 10.0:
        return 0.25 + 0.45 * ((mm - 2.5) / 7.5)
    if mm <= 25.0:
        return 0.70 + 0.30 * ((mm - 10.0) / 15.0)
    return 1.0


def score_dew_wetness_proxy(
    dew_point_depression_c: float,
    hours_rh_ge_85: float | None = None,
) -> float:
    """Return normalized [0, 1] leaf-wetness proxy from dew point depression and high-RH duration."""
    diff = max(0.0, float(dew_point_depression_c))
    if diff <= 1.5:
        dew_score = 0.95 + 0.05 * (1.0 - diff / 1.5)
    elif diff <= 3.0:
        dew_score = 0.65 + 0.30 * (1.0 - (diff - 1.5) / 1.5)
    elif diff <= 6.0:
        dew_score = 0.25 + 0.40 * (1.0 - (diff - 3.0) / 3.0)
    elif diff <= 10.0:
        dew_score = 0.25 * (1.0 - (diff - 6.0) / 4.0)
    else:
        dew_score = 0.0

    if _is_valid_num(hours_rh_ge_85):
        duration_score = min(1.0, max(0.0, float(hours_rh_ge_85)) / 12.0)
        return min(1.0, max(0.0, 0.60 * dew_score + 0.40 * duration_score))
    return min(1.0, max(0.0, dew_score))


def classify_risk_score(score: float) -> dict[str, str]:
    """Map an environmental risk score [0, 100] to its category and badge colors."""
    rounded_int = int(round(max(0.0, min(100.0, float(score)))))
    for low, high, label, color, bg in RISK_THRESHOLDS:
        if low <= rounded_int <= high:
            return {"category": label, "color": color, "bg_color": bg}
    return {"category": "Very High", "color": "#7F1D1D", "bg_color": "#FECACA"}


def compute_environmental_risk(
    temperature_c: float | None,
    relative_humidity_pct: float | None,
    precipitation_mm: float | None,
    dew_point_c: float | None = None,
    dew_point_depression_c: float | None = None,
    hours_rh_ge_85: float | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute the 0–100 Tomato Early Blight Environmental Favourability Index."""
    w = weights or RISK_WEIGHTS
    missing: list[str] = []
    if not _is_valid_num(temperature_c):
        missing.append("temperature")
    if not _is_valid_num(relative_humidity_pct):
        missing.append("relative_humidity")
    if not _is_valid_num(precipitation_mm):
        missing.append("precipitation")

    if missing:
        return {
            "status": "incomplete",
            "score": None,
            "category": "Unavailable",
            "color": "#64748B",
            "bg_color": "#F1F5F9",
            "missing_fields": missing,
            "factors": {},
            "contributions": {},
            "primary_driver": None,
            "explanation": f"Risk score could not be computed; missing variables: {', '.join(missing)}.",
            "disclaimer": "This is an experimental environmental disease-favourability index, not a validated probability of future infection.",
        }

    temp_val = float(temperature_c)
    rh_val = max(0.0, min(100.0, float(relative_humidity_pct)))
    rain_val = max(0.0, float(precipitation_mm))

    if _is_valid_num(dew_point_depression_c):
        dep_val = max(0.0, float(dew_point_depression_c))
        dew_source = "observed/forecast dew point"
    elif _is_valid_num(dew_point_c):
        dep_val = max(0.0, temp_val - float(dew_point_c))
        dew_source = "derived from temperature and dew point"
    else:
        est_dp = estimate_dew_point_celsius(temp_val, rh_val)
        dep_val = max(0.0, temp_val - est_dp)
        dew_source = "estimated via Magnus-Tetens from temperature & RH"

    f_rh = score_relative_humidity(rh_val)
    f_temp = score_temperature_suitability(temp_val)
    f_rain = score_rainfall(rain_val)
    f_wet = score_dew_wetness_proxy(dep_val, hours_rh_ge_85=hours_rh_ge_85)

    contrib_rh = round(100.0 * w["relative_humidity"] * f_rh, 1)
    contrib_temp = round(100.0 * w["temperature"] * f_temp, 1)
    contrib_rain = round(100.0 * w["rainfall"] * f_rain, 1)
    contrib_wet = round(100.0 * w["dew_wetness"] * f_wet, 1)

    total_score = round(min(100.0, max(0.0, contrib_rh + contrib_temp + contrib_rain + contrib_wet)), 1)
    cat_info = classify_risk_score(total_score)

    contributions = {
        "Relative Humidity (30%)": contrib_rh,
        "Temperature Suitability (25%)": contrib_temp,
        "Rainfall Splash (25%)": contrib_rain,
        "Dew / Wetness Proxy (20%)": contrib_wet,
    }
    sorted_drivers = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    primary_driver = sorted_drivers[0][0]

    # Build a grounded, non-generic explanation tied directly to actual numeric values.
    reasons: list[str] = []
    if rh_val >= 80.0:
        reasons.append(f"relative humidity is elevated ({rh_val:.0f}%)")
    elif rh_val < 55.0:
        reasons.append(f"ambient air is relatively dry ({rh_val:.0f}% RH)")
    else:
        reasons.append(f"relative humidity is moderate ({rh_val:.0f}%)")

    if 24.0 <= temp_val <= 29.0:
        reasons.append(f"mean temperature ({temp_val:.1f}°C) falls inside the optimal 24–29°C Alternaria window")
    elif 18.0 <= temp_val < 24.0 or 29.0 < temp_val <= 33.0:
        reasons.append(f"temperature ({temp_val:.1f}°C) is moderately favourable")
    else:
        reasons.append(f"temperature ({temp_val:.1f}°C) is outside peak fungal growth thresholds")

    if rain_val >= 5.0:
        reasons.append(f"measurable rainfall ({rain_val:.1f} mm) supports conidial splash dispersal")
    elif rain_val > 0.0:
        reasons.append(f"light precipitation ({rain_val:.1f} mm) is forecast")
    else:
        reasons.append("no rainfall (0.0 mm) is expected")

    if dep_val <= 2.5:
        reasons.append(f"narrow dew-point depression ({dep_val:.1f}°C) favors nocturnal leaf moisture")

    explanation = (
        f"Environmental favourability is {cat_info['category']} ({total_score:.1f}/100) because "
        + "; ".join(reasons)
        + "."
    )

    return {
        "status": "ok",
        "score": total_score,
        "category": cat_info["category"],
        "color": cat_info["color"],
        "bg_color": cat_info["bg_color"],
        "missing_fields": [],
        "inputs": {
            "temperature_c": round(temp_val, 2),
            "relative_humidity_pct": round(rh_val, 1),
            "precipitation_mm": round(rain_val, 2),
            "dew_point_depression_c": round(dep_val, 2),
            "hours_rh_ge_85": hours_rh_ge_85,
            "dew_source": dew_source,
        },
        "factors": {
            "relative_humidity": round(f_rh, 3),
            "temperature": round(f_temp, 3),
            "rainfall": round(f_rain, 3),
            "dew_wetness": round(f_wet, 3),
        },
        "contributions": contributions,
        "primary_driver": primary_driver,
        "explanation": explanation,
        "disclaimer": (
            "This is an experimental environmental disease-favourability index, not a validated probability of future infection."
        ),
    }


def assess_seven_day_forecast(daily_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate 7-day weather records and explain day-to-day risk shifts."""
    if not daily_records:
        return {
            "status": "unavailable",
            "days": [],
            "peak_day": None,
            "mean_score": None,
            "high_risk_days": 0,
            "summary": "No forecast records available to calculate environmental risk.",
        }

    assessed_days: list[dict[str, Any]] = []
    prev_day: dict[str, Any] | None = None

    for idx, day in enumerate(daily_records):
        res = compute_environmental_risk(
            temperature_c=day.get("temp_mean"),
            relative_humidity_pct=day.get("rh_mean"),
            precipitation_mm=day.get("precipitation_sum"),
            dew_point_c=day.get("dew_point_mean"),
            dew_point_depression_c=day.get("dew_point_depression_mean"),
            hours_rh_ge_85=day.get("hours_rh_ge_85"),
        )

        delta_score: float | None = None
        change_note = "Baseline forecast day (Day 1)."
        if prev_day is not None and res["score"] is not None and prev_day["score"] is not None:
            delta_score = round(res["score"] - prev_day["score"], 1)
            prev_c = prev_day["contributions"]
            curr_c = res["contributions"]
            diffs = {k: round(curr_c[k] - prev_c[k], 1) for k in curr_c}
            top_shift_key, top_shift_val = max(diffs.items(), key=lambda kv: abs(kv[1]))
            if abs(delta_score) < 2.0:
                change_note = f"Nearly unchanged ({delta_score:+.1f} pts vs prior day)."
            elif delta_score > 0:
                change_note = (
                    f"Risk increased by {delta_score:+.1f} pts vs prior day, driven primarily by "
                    f"{top_shift_key} ({top_shift_val:+.1f} pts)."
                )
            else:
                change_note = (
                    f"Risk decreased by {delta_score:.1f} pts vs prior day, driven primarily by "
                    f"lower {top_shift_key} ({top_shift_val:+.1f} pts)."
                )

        merged = {
            "day_index": idx + 1,
            "date": day.get("date"),
            "weather_label": day.get("weather_label"),
            "temp_min": day.get("temp_min"),
            "temp_max": day.get("temp_max"),
            "temp_mean": day.get("temp_mean"),
            "rh_mean": day.get("rh_mean"),
            "precipitation_sum": day.get("precipitation_sum"),
            "precip_prob_max": day.get("precip_prob_max"),
            "dew_point_depression_mean": day.get("dew_point_depression_mean"),
            "hours_rh_ge_85": day.get("hours_rh_ge_85"),
            "delta_from_prev": delta_score,
            "change_explanation": change_note,
            **res,
        }
        assessed_days.append(merged)
        prev_day = merged

    valid_days = [d for d in assessed_days if d.get("score") is not None]
    if not valid_days:
        return {
            "status": "unavailable",
            "days": assessed_days,
            "peak_day": None,
            "mean_score": None,
            "high_risk_days": 0,
            "summary": "Weather inputs were incomplete across the forecast window.",
        }

    peak_day = max(valid_days, key=lambda d: d["score"])
    mean_score = round(sum(d["score"] for d in valid_days) / len(valid_days), 1)
    high_risk_days = sum(1 for d in valid_days if d.get("category") in {"High", "Very High"})

    return {
        "status": "ok",
        "days": assessed_days,
        "peak_day": peak_day,
        "mean_score": mean_score,
        "high_risk_days": high_risk_days,
        "summary": (
            f"7-day mean environmental favourability is {mean_score}/100; peak conditions occur on "
            f"{peak_day['date']} ({peak_day['score']}/100 — {peak_day['category']})."
        ),
    }


def simulate_what_if_scenario(
    baseline_day: dict[str, Any],
    sim_temp_c: float | None,
    sim_rh_pct: float | None,
    sim_rain_mm: float | None,
) -> dict[str, Any]:
    """Compare baseline day risk against a hypothetical user-adjusted weather scenario."""
    baseline = compute_environmental_risk(
        temperature_c=baseline_day.get("temp_mean"),
        relative_humidity_pct=baseline_day.get("rh_mean"),
        precipitation_mm=baseline_day.get("precipitation_sum"),
        dew_point_c=baseline_day.get("dew_point_mean"),
        dew_point_depression_c=baseline_day.get("dew_point_depression_mean"),
        hours_rh_ge_85=baseline_day.get("hours_rh_ge_85"),
    )
    est_dew = (
        estimate_dew_point_celsius(float(sim_temp_c), float(sim_rh_pct))
        if _is_valid_num(sim_temp_c) and _is_valid_num(sim_rh_pct)
        else None
    )
    hypothetical = compute_environmental_risk(
        temperature_c=sim_temp_c,
        relative_humidity_pct=sim_rh_pct,
        precipitation_mm=sim_rain_mm,
        dew_point_c=est_dew,
    )

    delta = (
        round(hypothetical["score"] - baseline["score"], 1)
        if baseline.get("score") is not None and hypothetical.get("score") is not None
        else None
    )

    return {
        "baseline": baseline,
        "hypothetical": hypothetical,
        "score_delta": delta,
        "label": "Hypothetical Scenario Simulation (NOT an actual weather forecast or climate projection)",
    }


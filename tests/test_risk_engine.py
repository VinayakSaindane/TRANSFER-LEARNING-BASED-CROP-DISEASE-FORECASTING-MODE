"""Unit tests for the Tomato Early Blight environmental disease-risk engine."""

from __future__ import annotations

import math

from services.risk_engine import (
    assess_seven_day_forecast,
    classify_risk_score,
    compute_environmental_risk,
    estimate_dew_point_celsius,
    simulate_what_if_scenario,
)


def test_risk_scores_always_bounded_0_to_100() -> None:
    extremes = [
        (-10.0, 0.0, 0.0, -20.0),
        (26.5, 98.0, 45.0, 26.0),
        (50.0, 100.0, 200.0, 49.0),
        (12.0, 45.0, 2.0, 2.0),
    ]
    for temp, rh, rain, dew in extremes:
        res = compute_environmental_risk(temp, rh, rain, dew_point_c=dew)
        assert res["status"] == "ok"
        assert 0.0 <= res["score"] <= 100.0
        assert res["explanation"]
        assert (
            res["disclaimer"]
            == "This is an experimental environmental disease-favourability index, not a validated probability of future infection."
        )


def test_risk_category_boundaries() -> None:
    assert classify_risk_score(0.0)["category"] == "Low"
    assert classify_risk_score(30.0)["category"] == "Low"
    assert classify_risk_score(31.0)["category"] == "Moderate"
    assert classify_risk_score(60.0)["category"] == "Moderate"
    assert classify_risk_score(61.0)["category"] == "High"
    assert classify_risk_score(80.0)["category"] == "High"
    assert classify_risk_score(81.0)["category"] == "Very High"
    assert classify_risk_score(100.0)["category"] == "Very High"


def test_missing_or_nan_inputs_handled_safely() -> None:
    res_none = compute_environmental_risk(None, 80.0, 5.0)
    assert res_none["status"] == "incomplete"
    assert res_none["score"] is None
    assert "temperature" in res_none["missing_fields"]

    res_nan = compute_environmental_risk(25.0, float("nan"), 0.0)
    assert res_nan["status"] == "incomplete"
    assert "relative_humidity" in res_nan["missing_fields"]

    # NaN in optional fields must NOT propagate into the final score
    res_opt_nan = compute_environmental_risk(
        25.0,
        80.0,
        5.0,
        dew_point_c=float("nan"),
        dew_point_depression_c=float("nan"),
        hours_rh_ge_85=float("nan"),
    )
    assert res_opt_nan["status"] == "ok"
    assert math.isfinite(res_opt_nan["score"])

    # Extreme temperature guard in Magnus-Tetens helper
    assert math.isfinite(estimate_dew_point_celsius(-243.04, 50.0))


def test_rainfall_alone_does_not_confirm_leaf_wetness_or_high_risk() -> None:
    # Cold dry ambient air with rain should not produce high dew-wetness factor or Very High risk
    res = compute_environmental_risk(
        temperature_c=9.0,
        relative_humidity_pct=30.0,
        precipitation_mm=20.0,
        dew_point_depression_c=11.0,
    )
    assert res["status"] == "ok"
    assert res["factors"]["dew_wetness"] == 0.0
    assert res["score"] < 40.0


def test_seven_day_forecast_and_what_if_simulator() -> None:
    sample_days = [
        {
            "date": f"2026-11-{10 + i:02d}",
            "weather_label": "Partly cloudy",
            "temp_min": 18.0,
            "temp_max": 30.0,
            "temp_mean": 24.0 + (i % 3),
            "rh_mean": 45.0 + i * 7.0,
            "precipitation_sum": 0.0 if i < 3 else 8.5,
            "precip_prob_max": 10 if i < 3 else 75,
            "dew_point_mean": 14.0 + i * 1.5,
            "dew_point_depression_mean": max(1.0, 8.0 - i * 1.1),
            "hours_rh_ge_85": i * 2,
        }
        for i in range(7)
    ]
    out = assess_seven_day_forecast(sample_days)
    assert out["status"] == "ok"
    assert len(out["days"]) == 7
    assert out["peak_day"] is not None
    assert all(d["change_explanation"] for d in out["days"])
    assert math.isfinite(out["mean_score"])

    sim = simulate_what_if_scenario(sample_days[0], sim_temp_c=26.5, sim_rh_pct=92.0, sim_rain_mm=15.0)
    direct = compute_environmental_risk(
        26.5,
        92.0,
        15.0,
        dew_point_c=estimate_dew_point_celsius(26.5, 92.0),
    )
    assert sim["hypothetical"]["score"] == direct["score"]
    assert sim["hypothetical"]["contributions"] == direct["contributions"]
    assert sim["hypothetical"]["score"] > sim["baseline"]["score"]
    assert sim["score_delta"] > 0
    assert "NOT an actual weather forecast" in sim["label"]

    # Boundary consistency check: score rounding to 61 -> High must count in high_risk_days
    boundary_days = [
        {
            "date": "2026-11-10",
            "weather_label": "Overcast",
            "temp_min": 20.0,
            "temp_max": 28.0,
            "temp_mean": 25.0,
            "rh_mean": 82.0,
            "precipitation_sum": 5.0,
            "dew_point_mean": 21.5,
            "dew_point_depression_mean": 3.5,
            "hours_rh_ge_85": 3,
        }
    ]
    b_out = assess_seven_day_forecast(boundary_days)
    d0 = b_out["days"][0]
    expected_high = 1 if d0["category"] in {"High", "Very High"} else 0
    assert b_out["high_risk_days"] == expected_high


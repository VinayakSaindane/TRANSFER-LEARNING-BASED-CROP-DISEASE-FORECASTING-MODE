"""Tests for NOAA climate parsing/classification and preventive farmer advisories."""

from __future__ import annotations

from unittest.mock import patch

import requests

from services import climate_service
from services.advisory_service import generate_farmer_advisories
from services.risk_engine import assess_seven_day_forecast, compute_environmental_risk


SAMPLE_RONI_TXT = """SEAS   YR  ANOM
DJF  2026 -0.91
JFM  2026 -0.76
FMA  2026 -0.44
MAM  2026 -0.04
AMJ  2026  0.49
MJJ  2026  0.97
JJA  2026  1.36
JAS  2026  1.69
ASO  2026  1.97
"""


def test_parse_noaa_roni_and_sustained_phase_rules() -> None:
    records = climate_service.parse_noaa_ascii_table(SAMPLE_RONI_TXT, has_total_column=False)
    assert len(records) == 9
    assert records[-1]["label"] == "ASO 2026"
    assert records[-1]["anomaly"] == 1.97

    # 1. Single overlapping season crossing +/-0.5 C must NOT classify El Nino / La Nina
    single_spike = records[:4] + [
        {
            "season": "AMJ",
            "season_readable": "Apr–Jun 2026",
            "year": 2026,
            "season_index": 5,
            "sort_key": 202605,
            "label": "AMJ 2026",
            "sst_total": None,
            "anomaly": 0.65,
        }
    ]
    cls_single = climate_service.classify_enso_phase(single_spike)
    assert cls_single["phase"] == "ENSO-Neutral"
    assert cls_single["sustained_roni_episode"] is False
    assert cls_single["consecutive_seasons"] == 1

    # 2. Fewer than 5 consecutive RONI seasons without official ONI/CPC confirmation stays Neutral
    cls_unconfirmed = climate_service.classify_enso_phase(records)
    assert cls_unconfirmed["phase"] == "ENSO-Neutral"
    assert cls_unconfirmed["sustained_roni_episode"] is False
    assert cls_unconfirmed["consecutive_seasons"] == 4

    # 3. 5+ consecutive overlapping seasons in RONI directly confirms sustained episode
    five_seasons = records + [
        {
            "season": "SON",
            "season_readable": "Sep–Nov 2026",
            "year": 2026,
            "season_index": 10,
            "sort_key": 202610,
            "label": "SON 2026",
            "sst_total": None,
            "anomaly": 2.10,
        }
    ]
    cls_sustained = climate_service.classify_enso_phase(five_seasons)
    assert cls_sustained["phase"] == "El Niño"
    assert cls_sustained["sustained_roni_episode"] is True
    assert cls_sustained["consecutive_seasons"] == 5
    assert cls_sustained["strength"] == "Very Strong"

    # 4. Developing multi-season RONI (4 seasons) + official CPC Advisory / 5-season ONI confirms active phase
    oni_five = [
        {"season": s, "year": 2026, "label": f"{s} 2026", "anomaly": v}
        for s, v in [("AMJ", 0.95), ("MJJ", 1.39), ("JJA", 1.80), ("JAS", 2.16), ("ASO", 2.46)]
    ]
    cpc_adv = climate_service.parse_cpc_advisory_html(
        "ENSO Diagnostic Discussion issued by CLIMATE PREDICTION CENTER/NCEP/NWS 8 October 2026 "
        "ENSO Alert System Status: El Niño Advisory Synopsis: El Niño continues to strengthen."
    )
    assert cpc_adv is not None
    assert cpc_adv["alert_status"] == "El Niño Advisory"
    assert cpc_adv["issued_date"] == "8 October 2026"

    cls_ctx = climate_service.classify_enso_phase(records, oni_records=oni_five, cpc_advisory=cpc_adv)
    assert cls_ctx["phase"] == "El Niño"
    assert cls_ctx["strength"] == "Strong"
    assert cls_ctx["sustained_roni_episode"] is False
    assert cls_ctx["oni_consecutive_seasons"] == 5
    assert cls_ctx["cpc_alert_status"] == "El Niño Advisory"


def test_climate_unavailable_never_invents_enso(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(climate_service, "CACHE_DIR", tmp_path)
    with patch("services.climate_service.requests.get", side_effect=requests.Timeout("NOAA timeout")):
        res = climate_service.fetch_climate_status(force_refresh=True)
    assert res["status"] == "unavailable"
    assert res["phase"] == "Unavailable"
    assert res["latest_roni"] is None


def test_advisory_engine_links_risk_and_avoids_chemical_prescriptions() -> None:
    days = [
        {
            "date": f"2026-11-{10 + i:02d}",
            "weather_label": "Moderate rain",
            "temp_min": 20.0,
            "temp_max": 28.0,
            "temp_mean": 26.0,
            "rh_mean": 88.0,
            "precipitation_sum": 12.0,
            "precip_prob_max": 85,
            "dew_point_mean": 24.5,
            "dew_point_depression_mean": 1.5,
            "hours_rh_ge_85": 14,
        }
        for i in range(7)
    ]
    risk = assess_seven_day_forecast(days)
    fake_pred = {
        "status": "ok",
        "predicted_class": "Tomato Early Blight",
        "confidence": 0.91,
        "low_confidence": False,
    }
    advs = generate_farmer_advisories(risk, latest_prediction=fake_pred)
    assert len(advs) >= 3
    joined = " ".join(a["recommended_action"].lower() + " " + a["reason"].lower() for a in advs)
    # Must NOT prescribe specific pesticide chemicals or dosage rates
    for banned in ("mancozeb", "chlorothalonil", "azoxystrobin", "ml/l", "g/l", "ppm"):
        assert banned not in joined
    assert all(a["reason"] and a["recommended_action"] and a["relevant_forecast_day"] for a in advs)


def test_enso_indices_never_modify_local_disease_risk_score() -> None:
    import inspect

    sig = inspect.signature(compute_environmental_risk)
    param_names = set(sig.parameters.keys())
    assert not (param_names & {"roni", "oni", "enso", "enso_phase"})
    score_a = compute_environmental_risk(26.0, 85.0, 5.0, dew_point_c=23.5)["score"]
    score_b = compute_environmental_risk(26.0, 85.0, 5.0, dew_point_c=23.5)["score"]
    assert score_a == score_b


"""Tests for Open-Meteo weather service, response validation, caching, and offline safety."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import requests

from services import weather_service


def _sample_open_meteo_payload() -> dict:
    dates = [f"2026-11-{10 + i:02d}" for i in range(7)]
    hourly_times = []
    hourly_temp = []
    hourly_rh = []
    hourly_dew = []
    for d in dates:
        for h in range(24):
            hourly_times.append(f"{d}T{h:02d}:00")
            hourly_temp.append(25.0)
            hourly_rh.append(88.0 if h < 8 else 62.0)
            hourly_dew.append(23.0 if h < 8 else 16.0)

    return {
        "latitude": 19.9975,
        "longitude": 73.7898,
        "elevation": 584.0,
        "timezone": "Asia/Kolkata",
        "timezone_abbreviation": "IST",
        "current": {
            "time": "2026-11-10T12:00",
            "temperature_2m": 28.4,
            "apparent_temperature": 29.1,
            "relative_humidity_2m": 58,
            "precipitation": 0.0,
            "dew_point_2m": 19.2,
            "wind_speed_10m": 8.2,
            "cloud_cover": 20,
            "weather_code": 1,
        },
        "hourly": {
            "time": hourly_times,
            "temperature_2m": hourly_temp,
            "relative_humidity_2m": hourly_rh,
            "dew_point_2m": hourly_dew,
            "precipitation": [0.0] * len(hourly_times),
            "soil_temperature_0cm": [26.0] * len(hourly_times),
            "soil_moisture_0_to_1cm": [0.22] * len(hourly_times),
        },
        "daily": {
            "time": dates,
            "weather_code": [1, 2, 3, 61, 63, 2, 0],
            "temperature_2m_min": [18.0] * 7,
            "temperature_2m_max": [31.0] * 7,
            "temperature_2m_mean": [24.5] * 7,
            "precipitation_sum": [0.0, 0.0, 1.2, 6.4, 11.0, 0.5, 0.0],
            "precipitation_probability_max": [5, 10, 35, 70, 85, 20, 5],
            "et0_fao_evapotranspiration": [4.1] * 7,
        },
    }


def test_coordinate_validation_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        weather_service.fetch_weather_data(95.0, 73.0)
    with pytest.raises(ValueError):
        weather_service.fetch_weather_data(19.0, 190.0)


def test_parse_valid_open_meteo_response() -> None:
    parsed = weather_service._parse_open_meteo_response(_sample_open_meteo_payload(), {"name": "Nashik"})
    assert parsed["current"]["temperature_2m"] == 28.4
    assert parsed["current"]["weather_label"] == "Mainly clear"
    assert len(parsed["daily"]) == 7
    assert parsed["daily"][0]["hours_rh_ge_85"] == 8
    assert parsed["daily"][0]["soil_moisture_0_1cm_mean"] == 0.22


def test_weather_and_baseline_labels_distinguish_modelled_data_and_5yr_window(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(weather_service, "CACHE_DIR", tmp_path)
    sample = _sample_open_meteo_payload()

    class _Resp:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    with patch("services.weather_service.requests.get", return_value=_Resp(sample)):
        w = weather_service.fetch_weather_data(19.9975, 73.7898, force_refresh=True)
    assert "gridded" in w["source"].lower() or "model" in w["source"].lower()
    assert "not measure actual" in w["data_scope"].lower()

    archive_payload = {
        "daily": {
            "temperature_2m_mean": [22.0] * 7,
            "precipitation_sum": [0.0] * 7,
            "relative_humidity_2m_mean": [55.0] * 7,
        }
    }
    with patch("services.weather_service.requests.get", return_value=_Resp(archive_payload)):
        b = weather_service.fetch_historical_seasonal_baseline(19.9975, 73.7898)
    assert b["status"] == "ok"
    assert "not a 30-year" in b["baseline_type"].lower()


def test_api_failure_without_cache_shows_unavailable_no_fake_data(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(weather_service, "CACHE_DIR", tmp_path)
    with patch("services.weather_service.requests.get", side_effect=requests.Timeout("Simulated timeout")):
        res = weather_service.fetch_weather_data(12.345, 67.890, force_refresh=True)
    assert res["status"] == "unavailable"
    assert res["current"] is None
    assert res["daily"] == []
    assert "unavailable" in res["error"].lower()


def test_api_failure_with_stale_cache_returns_transparent_stale_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(weather_service, "CACHE_DIR", tmp_path)
    cache_file = weather_service._weather_cache_path(12.345, 67.890)
    cached_payload = weather_service._parse_open_meteo_response(_sample_open_meteo_payload())
    cached_payload["retrieved_at"] = "2026-11-01T00:00:00+00:00"
    cache_file.write_text(json.dumps(cached_payload), encoding="utf-8")

    with patch("services.weather_service.requests.get", side_effect=requests.ConnectionError("Offline")):
        res = weather_service.fetch_weather_data(12.345, 67.890, force_refresh=True)
    assert res["status"] == "stale_cache"
    assert res["retrieved_at"] == "2026-11-01T00:00:00+00:00"
    assert "cached forecast" in res["error"].lower()


def test_geocoding_rejects_short_query() -> None:
    res = weather_service.search_locations("a")
    assert res["status"] == "invalid"
    assert res["results"] == []


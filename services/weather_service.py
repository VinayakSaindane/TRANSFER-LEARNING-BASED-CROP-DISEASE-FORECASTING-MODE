"""Open-Meteo weather, geocoding, and historical seasonal baseline service."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from config.settings import (
    API_TIMEOUT_SECONDS,
    CACHE_DIR,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_GEOCODING_URL,
    WEATHER_CACHE_TTL_MINUTES,
)

WMO_WEATHER_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def describe_wmo_code(code: Any) -> str:
    """Return a readable weather description for a WMO weather interpretation code."""
    try:
        return WMO_WEATHER_CODES.get(int(code), f"Weather code {code}")
    except (TypeError, ValueError):
        return "Unavailable"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _is_finite(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)


def _mean_or_none(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if _is_finite(v)]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 2)


def _max_or_none(values: list[Any]) -> float | None:
    clean = [float(v) for v in values if _is_finite(v)]
    if not clean:
        return None
    return round(max(clean), 2)


def _validate_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    lat = float(latitude)
    lon = float(longitude)
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Coordinates out of range: lat={lat}, lon={lon}")
    return lat, lon


def _weather_cache_path(latitude: float, longitude: float) -> Path:
    return CACHE_DIR / f"weather_{latitude:.3f}_{longitude:.3f}.json"


def _read_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except OSError:
        pass


def search_locations(query: str, count: int = 8) -> dict[str, Any]:
    """Search global cities/villages using the Open-Meteo Geocoding API."""
    clean_query = (query or "").strip()
    if len(clean_query) < 2:
        return {
            "status": "invalid",
            "results": [],
            "error": "Enter at least 2 characters to search for a location.",
        }

    try:
        resp = requests.get(
            OPEN_METEO_GEOCODING_URL,
            params={"name": clean_query, "count": count, "language": "en", "format": "json"},
            timeout=API_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        raw = resp.json()
    except requests.RequestException as exc:
        return {
            "status": "unavailable",
            "results": [],
            "error": f"Geocoding API request failed: {exc}",
        }
    except ValueError as exc:
        return {
            "status": "unavailable",
            "results": [],
            "error": f"Invalid JSON received from geocoding API: {exc}",
        }

    items = raw.get("results") or []
    formatted = []
    for item in items:
        if "latitude" not in item or "longitude" not in item:
            continue
        parts = [p for p in [item.get("name"), item.get("admin1"), item.get("country")] if p]
        formatted.append(
            {
                "name": item.get("name", clean_query),
                "admin1": item.get("admin1", ""),
                "country": item.get("country", ""),
                "latitude": float(item["latitude"]),
                "longitude": float(item["longitude"]),
                "elevation": item.get("elevation"),
                "timezone": item.get("timezone", "auto"),
                "display_name": ", ".join(parts),
            }
        )

    return {
        "status": "ok",
        "results": formatted,
        "error": None if formatted else f"No matching locations found for '{clean_query}'.",
    }


def _aggregate_daily_records(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Combine Open-Meteo daily & hourly arrays into validated 7-day daily records."""
    daily = raw.get("daily") or {}
    hourly = raw.get("hourly") or {}

    dates = daily.get("time") or []
    if not dates:
        raise ValueError("Missing daily timestamps in Open-Meteo forecast response.")

    hourly_times = hourly.get("time") or []
    hourly_temp = hourly.get("temperature_2m") or []
    hourly_rh = hourly.get("relative_humidity_2m") or []
    hourly_dew = hourly.get("dew_point_2m") or []
    hourly_soil_t = hourly.get("soil_temperature_0cm") or []
    hourly_soil_m = hourly.get("soil_moisture_0_to_1cm") or []

    hourly_by_date: dict[str, list[int]] = {}
    for idx, ts in enumerate(hourly_times):
        day_str = str(ts)[:10]
        hourly_by_date.setdefault(day_str, []).append(idx)

    records: list[dict[str, Any]] = []
    for i, day_str in enumerate(dates):
        idxs = hourly_by_date.get(day_str, [])
        rh_vals = [hourly_rh[j] for j in idxs if j < len(hourly_rh)]
        dew_vals = [hourly_dew[j] for j in idxs if j < len(hourly_dew)]
        temp_vals = [hourly_temp[j] for j in idxs if j < len(hourly_temp)]
        soil_t_vals = [hourly_soil_t[j] for j in idxs if j < len(hourly_soil_t)]
        soil_m_vals = [hourly_soil_m[j] for j in idxs if j < len(hourly_soil_m)]

        depressions = []
        hours_rh_ge_85 = 0
        hours_dew_close = 0
        for t_val, r_val, d_val in zip(temp_vals, rh_vals, dew_vals):
            if _is_finite(r_val) and r_val >= 85.0:
                hours_rh_ge_85 += 1
            if _is_finite(t_val) and _is_finite(d_val):
                diff = float(t_val) - float(d_val)
                depressions.append(diff)
                if diff <= 2.5:
                    hours_dew_close += 1

        def _daily_get(key: str) -> Any:
            arr = daily.get(key) or []
            return arr[i] if i < len(arr) else None

        wcode = _daily_get("weather_code")
        t_mean = _daily_get("temperature_2m_mean")
        if not _is_finite(t_mean):
            t_mean = _mean_or_none(temp_vals)

        records.append(
            {
                "date": day_str,
                "weather_code": wcode,
                "weather_label": describe_wmo_code(wcode),
                "temp_min": _daily_get("temperature_2m_min"),
                "temp_max": _daily_get("temperature_2m_max"),
                "temp_mean": round(float(t_mean), 2) if _is_finite(t_mean) else None,
                "rh_mean": _mean_or_none(rh_vals),
                "rh_max": _max_or_none(rh_vals),
                "dew_point_mean": _mean_or_none(dew_vals),
                "dew_point_depression_mean": _mean_or_none(depressions),
                "hours_rh_ge_85": hours_rh_ge_85 if idxs else None,
                "hours_dew_close": hours_dew_close if idxs else None,
                "precipitation_sum": _daily_get("precipitation_sum"),
                "precip_prob_max": _daily_get("precipitation_probability_max"),
                "et0_fao": _daily_get("et0_fao_evapotranspiration"),
                "soil_temp_0cm_mean": _mean_or_none(soil_t_vals),
                "soil_moisture_0_1cm_mean": _mean_or_none(soil_m_vals),
            }
        )

    return records


def _parse_open_meteo_response(
    raw: dict[str, Any],
    location_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current_raw = raw.get("current") or {}
    if not current_raw or "temperature_2m" not in current_raw:
        raise ValueError("Missing current weather variables in Open-Meteo response.")

    daily_records = _aggregate_daily_records(raw)
    hourly_raw = raw.get("hourly") or {}
    hourly_records = []
    h_times = hourly_raw.get("time") or []
    for idx, ts in enumerate(h_times):
        def _h(key: str) -> Any:
            arr = hourly_raw.get(key) or []
            return arr[idx] if idx < len(arr) else None

        hourly_records.append(
            {
                "time": ts,
                "temperature_2m": _h("temperature_2m"),
                "relative_humidity_2m": _h("relative_humidity_2m"),
                "dew_point_2m": _h("dew_point_2m"),
                "precipitation": _h("precipitation"),
                "soil_temperature_0cm": _h("soil_temperature_0cm"),
                "soil_moisture_0_to_1cm": _h("soil_moisture_0_to_1cm"),
            }
        )

    wcode = current_raw.get("weather_code")
    temp = current_raw.get("temperature_2m")
    dew = current_raw.get("dew_point_2m")
    dew_depression = (
        round(float(temp) - float(dew), 2)
        if _is_finite(temp) and _is_finite(dew)
        else None
    )

    current = {
        "time": current_raw.get("time"),
        "temperature_2m": temp,
        "apparent_temperature": current_raw.get("apparent_temperature"),
        "relative_humidity_2m": current_raw.get("relative_humidity_2m"),
        "precipitation": current_raw.get("precipitation"),
        "dew_point_2m": dew,
        "dew_point_depression": dew_depression,
        "wind_speed_10m": current_raw.get("wind_speed_10m"),
        "cloud_cover": current_raw.get("cloud_cover"),
        "weather_code": wcode,
        "weather_label": describe_wmo_code(wcode),
    }

    return {
        "latitude": raw.get("latitude"),
        "longitude": raw.get("longitude"),
        "elevation": raw.get("elevation"),
        "timezone": raw.get("timezone"),
        "timezone_abbreviation": raw.get("timezone_abbreviation"),
        "location": location_meta or {},
        "current": current,
        "daily": daily_records,
        "hourly": hourly_records,
    }


def fetch_weather_data(
    latitude: float,
    longitude: float,
    location_meta: dict[str, Any] | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Fetch current weather and 7-day forecast from Open-Meteo with disk cache fallback."""
    lat, lon = _validate_coordinates(latitude, longitude)
    cache_path = _weather_cache_path(lat, lon)
    cached = _read_cache(cache_path)

    if not force_refresh and cached is not None:
        try:
            retrieved_dt = datetime.fromisoformat(cached["retrieved_at"])
            age_minutes = (_utc_now() - retrieved_dt).total_seconds() / 60.0
            if age_minutes <= WEATHER_CACHE_TTL_MINUTES and cached.get("current") and cached.get("daily"):
                cached["status"] = "live_cached"
                cached["cache_age_minutes"] = round(age_minutes, 1)
                cached["error"] = None
                return cached
        except (KeyError, ValueError, TypeError):
            pass

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": (
            "temperature_2m,relative_humidity_2m,apparent_temperature,"
            "precipitation,dew_point_2m,weather_code,wind_speed_10m,cloud_cover"
        ),
        "hourly": (
            "temperature_2m,relative_humidity_2m,dew_point_2m,"
            "precipitation,soil_temperature_0cm,soil_moisture_0_to_1cm"
        ),
        "daily": (
            "weather_code,temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
            "precipitation_sum,precipitation_probability_max,et0_fao_evapotranspiration"
        ),
        "forecast_days": 7,
        "timezone": "auto",
    }

    try:
        resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=API_TIMEOUT_SECONDS)
        resp.raise_for_status()
        raw = resp.json()
        parsed = _parse_open_meteo_response(raw, location_meta=location_meta)
        parsed["status"] = "live"
        parsed["retrieved_at"] = _utc_now().isoformat()
        parsed["cache_age_minutes"] = 0.0
        parsed["source"] = "Open-Meteo Forecast API (Gridded Location-Level Model)"
        parsed["data_scope"] = (
            "Modelled location-level weather forecast (NWP grid); does NOT measure actual farm-level "
            "canopy micro-climate or physical leaf wetness."
        )
        parsed["error"] = None
        _write_cache(cache_path, parsed)
        return parsed
    except (requests.RequestException, ValueError, KeyError) as exc:
        if cached is not None and cached.get("current") and cached.get("daily"):
            cached["status"] = "stale_cache"
            cached["error"] = (
                f"Live Open-Meteo request failed ({exc}). "
                f"Showing last verified cached forecast retrieved at {cached.get('retrieved_at')}."
            )
            return cached
        return {
            "status": "unavailable",
            "source": "Open-Meteo Forecast API (Gridded Location-Level Model)",
            "data_scope": (
                "Modelled location-level weather forecast (NWP grid); does NOT measure actual farm-level "
                "canopy micro-climate or physical leaf wetness."
            ),
            "retrieved_at": None,
            "latitude": lat,
            "longitude": lon,
            "location": location_meta or {},
            "current": None,
            "daily": [],
            "hourly": [],
            "error": f"Weather data unavailable: {exc}",
        }


def fetch_historical_seasonal_baseline(
    latitude: float,
    longitude: float,
    reference_date: date | None = None,
    years: int = 5,
) -> dict[str, Any]:
    """Fetch a multi-year historical local baseline for the same 7-day calendar window."""
    lat, lon = _validate_coordinates(latitude, longitude)
    ref = reference_date or date.today()
    cache_path = CACHE_DIR / f"baseline_{lat:.3f}_{lon:.3f}_{ref.strftime('%m%d')}.json"
    cached = _read_cache(cache_path)

    if cached is not None:
        try:
            retrieved_dt = datetime.fromisoformat(cached["retrieved_at"])
            if (_utc_now() - retrieved_dt).days < 7 and cached.get("status") == "ok" and cached.get("baseline_type"):
                return cached
        except (KeyError, ValueError, TypeError):
            pass

    temps: list[float] = []
    rhs: list[float] = []
    precips: list[float] = []
    yearly_records: list[dict[str, Any]] = []

    current_year = date.today().year
    start_year = current_year - years
    for yr in range(start_year, current_year):
        try:
            start_d = ref.replace(year=yr)
        except ValueError:
            start_d = date(yr, ref.month, 28)
        end_d = start_d + timedelta(days=6)
        if end_d >= date.today():
            continue
        try:
            resp = requests.get(
                OPEN_METEO_ARCHIVE_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start_d.isoformat(),
                    "end_date": end_d.isoformat(),
                    "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
                    "timezone": "auto",
                },
                timeout=API_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            data = resp.json().get("daily") or {}
            t_list = [v for v in (data.get("temperature_2m_mean") or []) if _is_finite(v)]
            p_list = [v for v in (data.get("precipitation_sum") or []) if _is_finite(v)]
            rh_list = [v for v in (data.get("relative_humidity_2m_mean") or []) if _is_finite(v)]
            if t_list:
                temps.extend(t_list)
                precips.extend(p_list)
                rhs.extend(rh_list)
                yearly_records.append(
                    {
                        "year": yr,
                        "mean_temp_c": round(sum(t_list) / len(t_list), 2),
                        "total_precip_mm": round(sum(p_list), 2) if p_list else None,
                        "mean_rh_pct": round(sum(rh_list) / len(rh_list), 2) if rh_list else None,
                    }
                )
        except (requests.RequestException, ValueError):
            continue

    if not yearly_records:
        return {
            "status": "unavailable",
            "source": "Open-Meteo Historical Archive API",
            "baseline_type": "Recent 5-Year Historical Comparison (NOT a 30-year WMO climatological normal)",
            "latitude": lat,
            "longitude": lon,
            "retrieved_at": None,
            "years_sampled": [],
            "mean_temp_c": None,
            "mean_daily_precip_mm": None,
            "mean_7day_precip_mm": None,
            "mean_rh_pct": None,
            "yearly_records": [],
            "error": "Unable to retrieve local multi-year historical comparison from Open-Meteo Archive.",
        }

    result = {
        "status": "ok",
        "source": "Open-Meteo Historical Archive API",
        "baseline_type": "Recent 5-Year Historical Comparison (NOT a 30-year WMO climatological normal)",
        "latitude": lat,
        "longitude": lon,
        "retrieved_at": _utc_now().isoformat(),
        "window_label": f"{ref.strftime('%b %d')} to {(ref + timedelta(days=6)).strftime('%b %d')}",
        "years_sampled": [r["year"] for r in yearly_records],
        "mean_temp_c": round(sum(temps) / len(temps), 2) if temps else None,
        "mean_daily_precip_mm": round(sum(precips) / len(precips), 2) if precips else None,
        "mean_7day_precip_mm": round(
            sum(r["total_precip_mm"] for r in yearly_records if r["total_precip_mm"] is not None)
            / len([r for r in yearly_records if r["total_precip_mm"] is not None]),
            2,
        )
        if any(r["total_precip_mm"] is not None for r in yearly_records)
        else None,
        "mean_rh_pct": round(sum(rhs) / len(rhs), 2) if rhs else None,
        "yearly_records": yearly_records,
        "error": None,
    }
    _write_cache(cache_path, result)
    return result


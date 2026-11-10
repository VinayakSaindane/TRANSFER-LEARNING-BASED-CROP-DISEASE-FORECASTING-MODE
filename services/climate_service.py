"""NOAA CPC ENSO / RONI and ONI climate intelligence service."""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import requests

from config.settings import (
    API_TIMEOUT_SECONDS,
    CACHE_DIR,
    CLIMATE_CACHE_TTL_HOURS,
    NOAA_ONI_URL,
    NOAA_RONI_URL,
)

NOAA_RONI_METHODOLOGY_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/"
NOAA_CPC_ADVISORY_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml"
MIN_CONSECUTIVE_EPISODE_SEASONS = 5

SEASON_LABELS: dict[str, str] = {
    "DJF": "Dec–Feb",
    "JFM": "Jan–Mar",
    "FMA": "Feb–Apr",
    "MAM": "Mar–May",
    "AMJ": "Apr–Jun",
    "MJJ": "May–Jul",
    "JJA": "Jun–Aug",
    "JAS": "Jul–Sep",
    "ASO": "Aug–Oct",
    "SON": "Sep–Nov",
    "OND": "Oct–Dec",
    "NDJ": "Nov–Jan",
}

SEASON_ORDER = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _climate_cache_path() -> Path:
    return CACHE_DIR / "noaa_enso_cache.json"


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


def parse_noaa_ascii_table(text: str, has_total_column: bool = False) -> list[dict[str, Any]]:
    """Parse NOAA CPC ASCII tables (RONI.ascii.txt or oni.ascii.txt)."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        raise ValueError("NOAA ASCII response is empty.")

    header_tokens = lines[0].upper().split()
    if "SEAS" not in header_tokens or "ANOM" not in header_tokens:
        raise ValueError(f"Unexpected NOAA table header: {lines[0]}")

    records: list[dict[str, Any]] = []
    for line in lines[1:]:
        parts = line.split()
        expected_len = 4 if has_total_column else 3
        if len(parts) < expected_len:
            continue
        seas = parts[0].upper()
        if seas not in SEASON_LABELS:
            continue
        try:
            year = int(parts[1])
            if has_total_column:
                sst_total = float(parts[2])
                anom = float(parts[3])
            else:
                sst_total = None
                anom = float(parts[2])
        except ValueError:
            continue

        seas_idx = SEASON_ORDER.index(seas) + 1
        records.append(
            {
                "season": seas,
                "season_readable": f"{SEASON_LABELS[seas]} {year}",
                "year": year,
                "season_index": seas_idx,
                "sort_key": year * 100 + seas_idx,
                "label": f"{seas} {year}",
                "sst_total": sst_total,
                "anomaly": round(anom, 2),
            }
        )

    if not records:
        raise ValueError("No valid seasonal observations parsed from NOAA ASCII data.")

    records.sort(key=lambda r: r["sort_key"])
    return records


def _parse_http_last_modified(header_val: str | None) -> str | None:
    if not header_val:
        return None
    try:
        dt = parsedate_to_datetime(header_val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def parse_cpc_advisory_html(html_text: str) -> dict[str, Any] | None:
    """Extract NOAA CPC ENSO Alert System Status and issue date from ensodisc.shtml."""
    if not html_text:
        return None
    clean = re.sub(r"<style.*?</style>|<script.*?</script>", " ", html_text, flags=re.S | re.I)
    clean = html.unescape(re.sub(r"<[^>]+>", " ", clean))
    clean = re.sub(r"\s+", " ", clean).strip()

    status_match = re.search(
        r"ENSO\s+Alert\s+System\s+Status\s*:\s*([A-Za-z0-9\s\-/\u00f1\u00d1]+?)(?:\s+Synopsis|\s+issued|\s{2,}|$)",
        clean,
        flags=re.I,
    )
    date_match = re.search(
        r"issued\s+by\s+CLIMATE\s+PREDICTION\s+CENTER[^\d]*(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        clean,
        flags=re.I,
    )
    alert_status = status_match.group(1).strip(" .:-") if status_match else None
    issued_date = date_match.group(1).strip() if date_match else None
    if not alert_status and not issued_date:
        return None
    return {
        "alert_status": alert_status,
        "issued_date": issued_date,
        "source_url": NOAA_CPC_ADVISORY_URL,
    }


def _consecutive_tail_run(records: list[dict[str, Any]], predicate: Any) -> int:
    streak = 0
    for rec in reversed(records):
        if predicate(rec["anomaly"]):
            streak += 1
        else:
            break
    return streak


def _anomaly_magnitude_tier(anomaly: float) -> str:
    abs_val = abs(anomaly)
    if abs_val < 0.5:
        return "Neutral"
    if abs_val < 1.0:
        return "Weak"
    if abs_val < 1.5:
        return "Moderate"
    if abs_val < 2.0:
        return "Strong"
    return "Very Strong"


def classify_enso_phase(
    records: list[dict[str, Any]],
    oni_records: list[dict[str, Any]] | None = None,
    cpc_advisory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify current ENSO phase following NOAA CPC's sustained 5-season threshold and official context.

    Per NOAA CPC RONI documentation (https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/):
    - El Niño / La Niña episodes require +/-0.5°C met across a minimum of 5 consecutive overlapping 3-month seasons.
    - A single overlapping season crossing +/-0.5°C must NOT alone classify an El Niño or La Niña episode.
    - When 2–4 consecutive RONI seasons meet +/-0.5°C in real time, official NOAA climate context
      (CPC ENSO Alert System Advisory and/or 5-season sustained ONI) is evaluated before confirming an active phase.
    """
    methodology_summary = (
        "NOAA CPC defines El Niño (RONI >= +0.5°C) and La Niña (RONI <= -0.5°C) historical episodes when the "
        "threshold is met for at least 5 consecutive overlapping 3-month seasons (1991–2020 base, ERSSTv6). "
        "Single-season threshold crossings do not establish an episode; developing multi-season runs (<5 seasons) "
        "are cross-checked with NOAA CPC ENSO Alert System status and reference ONI persistence."
    )

    if not records:
        return {
            "phase": "Unavailable",
            "strength": "Unknown",
            "latest_anomaly_tier": "Unknown",
            "consecutive_seasons": 0,
            "sustained_roni_episode": False,
            "oni_consecutive_seasons": 0,
            "cpc_alert_status": None,
            "cpc_issued_date": None,
            "episode_status": "No data available",
            "classification_basis": "No NOAA RONI seasonal observations available.",
            "methodology_summary": methodology_summary,
            "methodology_url": NOAA_RONI_METHODOLOGY_URL,
        }

    latest = float(records[-1]["anomaly"])
    anomaly_tier = _anomaly_magnitude_tier(latest)

    warm_streak = _consecutive_tail_run(records, lambda a: a >= 0.5)
    cold_streak = _consecutive_tail_run(records, lambda a: a <= -0.5)
    neutral_streak = _consecutive_tail_run(records, lambda a: -0.5 < a < 0.5)

    oni_warm_streak = _consecutive_tail_run(oni_records, lambda a: a >= 0.5) if oni_records else 0
    oni_cold_streak = _consecutive_tail_run(oni_records, lambda a: a <= -0.5) if oni_records else 0

    alert_status = (cpc_advisory or {}).get("alert_status")
    issued_date = (cpc_advisory or {}).get("issued_date")
    alert_norm = (alert_status or "").lower().replace("ñ", "n")

    if warm_streak >= MIN_CONSECUTIVE_EPISODE_SEASONS:
        phase = "El Niño"
        streak = warm_streak
        sustained_roni = True
        strength = anomaly_tier
        episode_status = (
            f"Sustained El Niño episode ({streak} consecutive overlapping seasons >= +0.5°C; "
            f"meets NOAA's {MIN_CONSECUTIVE_EPISODE_SEASONS}-season criterion)"
        )
        classification_basis = (
            f"NOAA RONI sustained threshold met: {streak} consecutive overlapping 3-month seasons >= +0.50°C."
        )
    elif cold_streak >= MIN_CONSECUTIVE_EPISODE_SEASONS:
        phase = "La Niña"
        streak = cold_streak
        sustained_roni = True
        strength = anomaly_tier
        episode_status = (
            f"Sustained La Niña episode ({streak} consecutive overlapping seasons <= -0.5°C; "
            f"meets NOAA's {MIN_CONSECUTIVE_EPISODE_SEASONS}-season criterion)"
        )
        classification_basis = (
            f"NOAA RONI sustained threshold met: {streak} consecutive overlapping 3-month seasons <= -0.50°C."
        )
    elif 2 <= warm_streak < MIN_CONSECUTIVE_EPISODE_SEASONS and (
        "el nino advisory" in alert_norm or oni_warm_streak >= MIN_CONSECUTIVE_EPISODE_SEASONS
    ):
        phase = "El Niño"
        streak = warm_streak
        sustained_roni = False
        strength = anomaly_tier
        ctx_parts: list[str] = []
        if "el nino advisory" in alert_norm:
            ctx_parts.append(f"NOAA CPC Alert System: {alert_status}")
        if oni_warm_streak >= MIN_CONSECUTIVE_EPISODE_SEASONS:
            ctx_parts.append(f"ONI >= +0.50°C for {oni_warm_streak} consecutive seasons")
        ctx_str = " & ".join(ctx_parts)
        episode_status = (
            f"Active El Niño confirmed via official climate context ({ctx_str}); "
            f"RONI 2-decimal run is {streak}/{MIN_CONSECUTIVE_EPISODE_SEASONS} consecutive seasons >= +0.50°C"
        )
        prior_rec = records[-streak - 1] if len(records) > streak else None
        classification_basis = (
            f"Multi-season developing RONI warm run ({streak} consecutive seasons >= +0.50°C, following "
            f"{prior_rec['label']} at {prior_rec['anomaly']:+.2f}°C) corroborated by {ctx_str}."
            if prior_rec is not None
            else f"Multi-season developing RONI warm run ({streak} consecutive seasons >= +0.50°C) corroborated by {ctx_str}."
        )
    elif 2 <= cold_streak < MIN_CONSECUTIVE_EPISODE_SEASONS and (
        "la nina advisory" in alert_norm or oni_cold_streak >= MIN_CONSECUTIVE_EPISODE_SEASONS
    ):
        phase = "La Niña"
        streak = cold_streak
        sustained_roni = False
        strength = anomaly_tier
        ctx_parts = []
        if "la nina advisory" in alert_norm:
            ctx_parts.append(f"NOAA CPC Alert System: {alert_status}")
        if oni_cold_streak >= MIN_CONSECUTIVE_EPISODE_SEASONS:
            ctx_parts.append(f"ONI <= -0.50°C for {oni_cold_streak} consecutive seasons")
        ctx_str = " & ".join(ctx_parts)
        episode_status = (
            f"Active La Niña confirmed via official climate context ({ctx_str}); "
            f"RONI 2-decimal run is {streak}/{MIN_CONSECUTIVE_EPISODE_SEASONS} consecutive seasons <= -0.50°C"
        )
        classification_basis = (
            f"Multi-season developing RONI cold run ({streak} consecutive seasons <= -0.50°C) corroborated by {ctx_str}."
        )
    else:
        phase = "ENSO-Neutral"
        sustained_roni = False
        strength = "Neutral"
        if warm_streak >= 1:
            streak = warm_streak
            episode_status = (
                f"ENSO-Neutral (unconfirmed warm threshold crossing: {warm_streak} of "
                f"{MIN_CONSECUTIVE_EPISODE_SEASONS} required consecutive overlapping seasons >= +0.5°C)"
            )
            classification_basis = (
                f"Only {warm_streak} consecutive overlapping RONI season(s) >= +0.50°C without official 5-season "
                "persistence or CPC Advisory confirmation; per NOAA rules, one or unconfirmed seasons do not constitute El Niño."
            )
        elif cold_streak >= 1:
            streak = cold_streak
            episode_status = (
                f"ENSO-Neutral (unconfirmed cold threshold crossing: {cold_streak} of "
                f"{MIN_CONSECUTIVE_EPISODE_SEASONS} required consecutive overlapping seasons <= -0.5°C)"
            )
            classification_basis = (
                f"Only {cold_streak} consecutive overlapping RONI season(s) <= -0.50°C without official 5-season "
                "persistence or CPC Advisory confirmation; per NOAA rules, one or unconfirmed seasons do not constitute La Niña."
            )
        else:
            streak = neutral_streak
            episode_status = f"Neutral conditions ({streak} consecutive overlapping 3-month season(s) within +/-0.5°C)"
            classification_basis = (
                "Latest overlapping 3-month RONI anomaly lies strictly within the -0.50°C to +0.50°C neutral band."
            )

    return {
        "phase": phase,
        "strength": strength,
        "latest_anomaly_tier": anomaly_tier,
        "consecutive_seasons": streak,
        "sustained_roni_episode": sustained_roni,
        "oni_consecutive_seasons": max(oni_warm_streak, oni_cold_streak),
        "cpc_alert_status": alert_status,
        "cpc_issued_date": issued_date,
        "episode_status": episode_status,
        "classification_basis": classification_basis,
        "methodology_summary": methodology_summary,
        "methodology_url": NOAA_RONI_METHODOLOGY_URL,
    }


def fetch_climate_status(force_refresh: bool = False) -> dict[str, Any]:
    """Fetch official NOAA CPC RONI (primary), ONI (reference), and CPC Alert System context."""
    cache_path = _climate_cache_path()
    cached = _read_cache(cache_path)

    if not force_refresh and cached is not None:
        try:
            retrieved_dt = datetime.fromisoformat(cached["retrieved_at"])
            age_hours = (_utc_now() - retrieved_dt).total_seconds() / 3600.0
            has_required_fields = bool(
                cached.get("roni_series")
                and cached.get("classification_basis")
                and cached.get("methodology_summary")
            )
            if age_hours <= CLIMATE_CACHE_TTL_HOURS and has_required_fields:
                cached["status"] = "live_cached"
                cached["cache_age_hours"] = round(age_hours, 2)
                cached["error"] = None
                return cached
        except (KeyError, ValueError, TypeError):
            pass

    try:
        roni_resp = requests.get(NOAA_RONI_URL, timeout=API_TIMEOUT_SECONDS)
        roni_resp.raise_for_status()
        roni_records = parse_noaa_ascii_table(roni_resp.text, has_total_column=False)
        roni_last_modified = _parse_http_last_modified(roni_resp.headers.get("Last-Modified"))

        oni_records: list[dict[str, Any]] = []
        oni_last_modified: str | None = None
        try:
            oni_resp = requests.get(NOAA_ONI_URL, timeout=API_TIMEOUT_SECONDS)
            oni_resp.raise_for_status()
            oni_records = parse_noaa_ascii_table(oni_resp.text, has_total_column=True)
            oni_last_modified = _parse_http_last_modified(oni_resp.headers.get("Last-Modified"))
        except (requests.RequestException, ValueError):
            oni_records = []

        cpc_advisory: dict[str, Any] | None = None
        try:
            adv_resp = requests.get(NOAA_CPC_ADVISORY_URL, timeout=API_TIMEOUT_SECONDS)
            adv_resp.raise_for_status()
            cpc_advisory = parse_cpc_advisory_html(adv_resp.text)
        except requests.RequestException:
            cpc_advisory = None

        classification = classify_enso_phase(
            roni_records,
            oni_records=oni_records or None,
            cpc_advisory=cpc_advisory,
        )
        latest_roni = roni_records[-1]
        latest_oni = oni_records[-1] if oni_records else None

        payload = {
            "status": "live",
            "retrieved_at": _utc_now().isoformat(),
            "source_updated_at": roni_last_modified,
            "oni_source_updated_at": oni_last_modified,
            "cache_age_hours": 0.0,
            "primary_index_name": "Relative Oceanic Niño Index (RONI)",
            "primary_source_url": NOAA_RONI_URL,
            "reference_source_url": NOAA_ONI_URL,
            "methodology_url": NOAA_RONI_METHODOLOGY_URL,
            "advisory_source_url": NOAA_CPC_ADVISORY_URL,
            "latest_season": latest_roni["label"],
            "latest_season_readable": latest_roni["season_readable"],
            "latest_roni": latest_roni["anomaly"],
            "latest_oni": latest_oni["anomaly"] if latest_oni else None,
            "latest_oni_season": latest_oni["label"] if latest_oni else None,
            "phase": classification["phase"],
            "strength": classification["strength"],
            "latest_anomaly_tier": classification["latest_anomaly_tier"],
            "consecutive_seasons": classification["consecutive_seasons"],
            "sustained_roni_episode": classification["sustained_roni_episode"],
            "oni_consecutive_seasons": classification["oni_consecutive_seasons"],
            "cpc_alert_status": classification["cpc_alert_status"],
            "cpc_issued_date": classification["cpc_issued_date"],
            "episode_status": classification["episode_status"],
            "classification_basis": classification["classification_basis"],
            "methodology_summary": classification["methodology_summary"],
            "roni_series": roni_records[-36:],
            "oni_series": oni_records[-36:] if oni_records else [],
            "error": None,
        }
        _write_cache(cache_path, payload)
        return payload
    except (requests.RequestException, ValueError) as exc:
        if cached is not None and cached.get("roni_series"):
            cached["status"] = "stale_cache"
            cached["error"] = (
                f"Live NOAA CPC request failed ({exc}). "
                f"Showing last verified cached observation retrieved at {cached.get('retrieved_at')}."
            )
            return cached
        return {
            "status": "unavailable",
            "retrieved_at": None,
            "source_updated_at": None,
            "oni_source_updated_at": None,
            "primary_index_name": "Relative Oceanic Niño Index (RONI)",
            "primary_source_url": NOAA_RONI_URL,
            "reference_source_url": NOAA_ONI_URL,
            "methodology_url": NOAA_RONI_METHODOLOGY_URL,
            "advisory_source_url": NOAA_CPC_ADVISORY_URL,
            "latest_season": None,
            "latest_season_readable": None,
            "latest_roni": None,
            "latest_oni": None,
            "latest_oni_season": None,
            "phase": "Unavailable",
            "strength": "Unknown",
            "latest_anomaly_tier": "Unknown",
            "consecutive_seasons": 0,
            "sustained_roni_episode": False,
            "oni_consecutive_seasons": 0,
            "cpc_alert_status": None,
            "cpc_issued_date": None,
            "episode_status": "NOAA climate data unavailable",
            "classification_basis": "NOAA CPC observations could not be retrieved.",
            "methodology_summary": (
                "NOAA CPC requires >= 5 consecutive overlapping 3-month seasons meeting +/-0.5°C for historical episodes."
            ),
            "roni_series": [],
            "oni_series": [],
            "error": f"Unable to retrieve official NOAA ENSO data: {exc}",
        }


def get_enso_regional_context(phase: str, country: str = "") -> dict[str, Any]:
    """Provide scientific climate-to-weather context without altering local disease risk scores."""
    is_india = "india" in (country or "").lower()
    if phase == "El Niño":
        global_note = (
            "El Niño (warm east-central tropical Pacific SST anomalies relative to the tropical mean) "
            "typically weakens Walker Circulation and shifts large-scale convective rainfall belts."
        )
        regional_note = (
            "In South Asia / India, El Niño episodes are historically associated with an elevated probability "
            "of suppressed or erratic southwest monsoon rainfall and warmer daytime anomalies, though spatial "
            "SST patterns and Indian Ocean Dipole (IOD) interactions strongly modulate local outcomes "
            "(Rasmusson & Carpenter, 1983; Krishna Kumar et al., 1999, 2006)."
            if is_india
            else "Regional impacts depend strongly on season and geography; some regions experience warmer/drier "
            "conditions while others experience enhanced storm tracks."
        )
    elif phase == "La Niña":
        global_note = (
            "La Niña (cool east-central tropical Pacific SST anomalies relative to the tropical mean) "
            "strengthens tropical Pacific trade winds and Walker Circulation."
        )
        regional_note = (
            "In India, La Niña often favors normal-to-above-normal monsoon rainfall and prolonged cloudy or "
            "humid spells, which can indirectly increase canopy moisture duration when combined with warm temperatures."
            if is_india
            else "Regional anomalies vary by latitude and season, often bringing cooler/wetter patterns in monsoon belts."
        )
    elif phase == "ENSO-Neutral":
        global_note = (
            "ENSO-Neutral conditions indicate tropical Pacific SST anomalies remain within +/-0.5°C of the "
            "tropical-mean baseline, reducing large-scale Pacific climate forcing."
        )
        regional_note = (
            "Local synoptic weather systems, regional sea surface temperatures, and seasonal monsoon onset "
            "dominate day-to-day farm microclimates."
        )
    else:
        global_note = "Official NOAA ENSO phase is currently unavailable."
        regional_note = "Rely strictly on local weather forecasts and field scouting for microclimate assessment."

    return {
        "global_mechanism": global_note,
        "regional_context": regional_note,
        "scientific_caveat": (
            "Important: El Niño / ENSO does NOT directly cause tomato early blight and is NEVER added as a "
            "multiplier to the local disease-risk score. Local temperature, relative humidity, rainfall, and "
            "dew-point wetness indicators directly govern pathogen favourability."
        ),
    }


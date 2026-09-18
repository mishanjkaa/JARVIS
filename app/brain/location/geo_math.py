from __future__ import annotations

import math

_EARTH_RADIUS_METERS = 6_371_000.0


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return _EARTH_RADIUS_METERS * c


def parse_raw_coordinates(text: str) -> tuple[float, float] | None:
    parts = text.strip().split(",")
    if len(parts) != 2:
        return None
    try:
        latitude = float(parts[0].strip())
        longitude = float(parts[1].strip())
    except ValueError:
        return None
    if not (-90.0 <= latitude <= 90.0) or not (-180.0 <= longitude <= 180.0):
        return None
    return latitude, longitude


def format_distance(meters: float) -> str:
    if meters >= 1000:
        return f"{meters / 1000:.1f} km"
    return f"{meters:.0f} meters"


def format_duration(seconds: float) -> str:
    minutes = seconds / 60
    if minutes >= 60:
        hours = int(minutes // 60)
        remaining_minutes = int(minutes % 60)
        return f"{hours} h {remaining_minutes} min"
    if minutes < 1:
        return "under a minute"
    return f"{minutes:.0f} min"


def format_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)} seconds"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f} minutes"
    hours = minutes / 60
    return f"{hours:.1f} hours"

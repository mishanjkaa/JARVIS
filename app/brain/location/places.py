from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app.brain.location.errors import LocationError

# Owner-managed saved places (RFC-010): "home" is the motivating example, but there is no
# preset default — a place only exists once the owner explicitly calls location.save_place.
# Never inferred from location history, matching RFC-010's "Not supported" list.
MAX_SAVED_PLACES = 100


def get_places_file_path(places_file: Optional[Path] = None) -> Path:
    if places_file is not None:
        return places_file
    base_dir = Path(__file__).resolve().parents[3]
    return base_dir / "data" / "places.json"


def _load_places(places_file: Path) -> dict[str, dict[str, float]]:
    if not places_file.exists():
        return {}
    try:
        content = places_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {}
    if not content.strip():
        return {}
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    normalized: dict[str, dict[str, float]] = {}
    for name, value in parsed.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            continue
        latitude = value.get("latitude")
        longitude = value.get("longitude")
        if isinstance(latitude, (int, float)) and isinstance(longitude, (int, float)):
            normalized[name] = {"latitude": float(latitude), "longitude": float(longitude)}
    return normalized


def _save_places(places_file: Path, data: dict[str, dict[str, float]]) -> None:
    places_file.parent.mkdir(parents=True, exist_ok=True)
    places_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def list_places(places_file: Optional[Path] = None) -> dict[str, dict[str, float]]:
    return _load_places(get_places_file_path(places_file))


def get_place(name: str, places_file: Optional[Path] = None) -> Optional[tuple[float, float]]:
    normalized_name = name.strip().lower()
    places = _load_places(get_places_file_path(places_file))
    entry = places.get(normalized_name)
    if entry is None:
        return None
    return entry["latitude"], entry["longitude"]


def save_place(name: str, latitude: float, longitude: float, places_file: Optional[Path] = None) -> None:
    normalized_name = name.strip().lower()
    path = get_places_file_path(places_file)
    places = _load_places(path)
    if normalized_name not in places and len(places) >= MAX_SAVED_PLACES:
        raise LocationError("Saved place limit reached.")
    places[normalized_name] = {"latitude": float(latitude), "longitude": float(longitude)}
    _save_places(path, places)

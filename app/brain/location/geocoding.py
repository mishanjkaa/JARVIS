from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from app.brain.location.errors import LocationGeocodeError

# The public Nominatim instance is a standard, unauthenticated OSM service (no API key),
# consistent with this project's "no new dependency, no per-request cost" preference for
# maps data. RFC-010 only calls out OSRM as a config-driven swap point (the routing
# provider), so this base URL is a plain constant rather than a config key.
DEFAULT_NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org"
_USER_AGENT = "JARVIS-Location/1.0 (RFC-010; local personal assistant)"


class Geocoder:
    def reverse(self, latitude: float, longitude: float) -> str:
        raise NotImplementedError

    def search(self, query: str) -> tuple[float, float]:
        raise NotImplementedError


class NominatimGeocoder(Geocoder):
    def __init__(self, *, base_url: str = DEFAULT_NOMINATIM_BASE_URL, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def reverse(self, latitude: float, longitude: float) -> str:
        query = urllib.parse.urlencode({"lat": latitude, "lon": longitude, "format": "jsonv2"})
        payload = self._get(f"{self.base_url}/reverse?{query}")
        display_name = payload.get("display_name") if isinstance(payload, dict) else None
        if not isinstance(display_name, str) or not display_name.strip():
            raise LocationGeocodeError("Could not resolve a place name for that location.")
        return display_name

    def search(self, query: str) -> tuple[float, float]:
        encoded = urllib.parse.urlencode({"q": query, "format": "jsonv2", "limit": 1})
        payload = self._get(f"{self.base_url}/search?{encoded}")
        if not isinstance(payload, list) or not payload:
            raise LocationGeocodeError(f"Could not find a location matching '{query}'.")
        first = payload[0]
        try:
            return float(first["lat"]), float(first["lon"])
        except (KeyError, TypeError, ValueError) as error:
            raise LocationGeocodeError(f"Could not find a location matching '{query}'.") from error

    def _get(self, url: str):
        request = urllib.request.Request(url, method="GET", headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            raise LocationGeocodeError("The geocoding service is unavailable.") from error
        try:
            return json.loads(body)
        except json.JSONDecodeError as error:
            raise LocationGeocodeError("The geocoding service returned an invalid response.") from error

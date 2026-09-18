from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from app.brain.location.errors import LocationRoutingError

# RFC-010 locks in OSRM as the routing provider but deliberately keeps it behind this small
# interface: a self-hosted OSRM instance, or an entirely different provider (e.g. one with
# sharper ETA data), can be swapped in later by adding another RoutingProvider subclass and
# changing what LocationController._routing_provider() builds, without touching the
# navigation.* tool surface at all.
DEFAULT_OSRM_BASE_URL = "http://router.project-osrm.org"

_MODIFIER_PHRASES = {
    "uturn": "make a U-turn",
    "sharp right": "turn sharp right",
    "right": "turn right",
    "slight right": "bear right",
    "straight": "continue straight",
    "slight left": "bear left",
    "left": "turn left",
    "sharp left": "turn sharp left",
}


@dataclass
class RouteStep:
    instruction: str
    distance_meters: float


@dataclass
class RouteResult:
    distance_meters: float
    duration_seconds: float
    steps: list[RouteStep] = field(default_factory=list)


class RoutingProvider:
    def route(self, origin: tuple[float, float], destination: tuple[float, float]) -> RouteResult:
        raise NotImplementedError


class OSRMRoutingProvider(RoutingProvider):
    def __init__(self, *, base_url: str = DEFAULT_OSRM_BASE_URL, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def route(self, origin: tuple[float, float], destination: tuple[float, float]) -> RouteResult:
        origin_lat, origin_lon = origin
        dest_lat, dest_lon = destination
        coordinates = f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
        url = f"{self.base_url}/route/v1/driving/{coordinates}?overview=false&steps=true"
        payload = self._get(url)
        if not isinstance(payload, dict) or payload.get("code") != "Ok":
            raise LocationRoutingError("The routing service could not find a route to that destination.")
        routes = payload.get("routes")
        if not isinstance(routes, list) or not routes:
            raise LocationRoutingError("The routing service could not find a route to that destination.")
        route = routes[0]
        steps: list[RouteStep] = []
        for leg in route.get("legs", []):
            for raw_step in leg.get("steps", []):
                instruction = _instruction_from_step(raw_step)
                distance = float(raw_step.get("distance", 0.0))
                steps.append(RouteStep(instruction=instruction, distance_meters=distance))
        return RouteResult(
            distance_meters=float(route.get("distance", 0.0)),
            duration_seconds=float(route.get("duration", 0.0)),
            steps=steps,
        )

    def _get(self, url: str):
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            raise LocationRoutingError("The routing service is unavailable.") from error
        try:
            return json.loads(body)
        except json.JSONDecodeError as error:
            raise LocationRoutingError("The routing service returned an invalid response.") from error


def _instruction_from_step(raw_step: dict) -> str:
    maneuver = raw_step.get("maneuver", {}) if isinstance(raw_step, dict) else {}
    maneuver_type = str(maneuver.get("type", ""))
    modifier = str(maneuver.get("modifier", ""))
    street_name = str(raw_step.get("name") or "").strip()

    if maneuver_type == "arrive":
        return "You have arrived at your destination."
    if maneuver_type == "depart":
        return f"Head toward {street_name}." if street_name else "Head toward your destination."
    if maneuver_type in {"turn", "end of road", "fork", "merge", "ramp"}:
        phrase = _MODIFIER_PHRASES.get(modifier, "continue")
        if street_name:
            return f"{phrase.capitalize()} onto {street_name}."
        return f"{phrase.capitalize()}."
    if maneuver_type in {"continue", "new name"}:
        if street_name:
            return f"Continue straight on {street_name}."
        return "Continue straight."
    if maneuver_type == "roundabout":
        return "Enter the roundabout."
    return f"Continue on {street_name}." if street_name else "Continue."

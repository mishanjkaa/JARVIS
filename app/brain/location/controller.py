from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.brain.agent.state import get_agent_runtime_state
from app.brain.audit.audit_log import record_audit_event
from app.brain.configuration.runtime_config import get_effective_runtime_config
from app.brain.location import places
from app.brain.location.errors import (
    LocationDestinationError,
    LocationDisabledError,
    LocationPlaceNotFoundError,
    LocationSessionNotFoundError,
    LocationUnavailableError,
)
from app.brain.location.geo_math import format_age, format_distance, format_duration, haversine_distance_meters, parse_raw_coordinates
from app.brain.location.geocoding import Geocoder, NominatimGeocoder
from app.brain.location.models import LocationPoint, NavigationSession
from app.brain.location.routing import DEFAULT_OSRM_BASE_URL, OSRMRoutingProvider, RoutingProvider
from app.brain.location.state import get_location_state

# A step's remaining distance below this is treated as "arrived", regardless of what OSRM's
# own "arrive" maneuver reports for the final leg. Not exposed as a config key: it is an
# internal navigation-quality constant, the same way MAX_DESKTOP_CAPTURES_PER_PLAN is a
# plain module constant rather than a runtime setting.
ARRIVAL_THRESHOLD_METERS = 15.0


class LocationController:
    def __init__(self, *, geocoder: Geocoder | None = None, routing_provider: RoutingProvider | None = None) -> None:
        self._geocoder_override = geocoder
        self._routing_provider_override = routing_provider

    def effective_config(self) -> dict[str, Any]:
        return get_effective_runtime_config()

    def geocoder(self) -> Geocoder:
        if self._geocoder_override is not None:
            return self._geocoder_override
        return NominatimGeocoder()

    def routing_provider(self) -> RoutingProvider:
        if self._routing_provider_override is not None:
            return self._routing_provider_override
        config = self.effective_config()
        return OSRMRoutingProvider(base_url=str(config.get("osrm_base_url", DEFAULT_OSRM_BASE_URL)))

    def _require_enabled(self) -> None:
        if not bool(self.effective_config().get("location_enabled", False)):
            raise LocationDisabledError("Location and navigation are disabled.")

    # --- Overland ingestion (called only by app.brain.location.server) -----

    def ingest_overland_point(self, *, device_id: str, latitude: float, longitude: float, accuracy_meters: float | None, captured_at: str) -> None:
        state = get_location_state()
        point = LocationPoint(
            device_id=device_id,
            latitude=latitude,
            longitude=longitude,
            accuracy_meters=accuracy_meters,
            captured_at=captured_at,
            received_at=datetime.now(timezone.utc).isoformat(),
        )
        with state.lock:
            state.points[device_id] = point

    def get_freshest_point(self) -> LocationPoint | None:
        state = get_location_state()
        with state.lock:
            if not state.points:
                return None
            return max(state.points.values(), key=lambda point: point.received_at)

    # --- location.* tools ---------------------------------------------------

    def where_am_i(self) -> str:
        self._require_enabled()
        point = self._require_freshest_point()
        display_name = self.geocoder().reverse(point.latitude, point.longitude)
        message = f"You are near {display_name}."
        age_seconds = self._point_age_seconds(point)
        stale_after = int(self.effective_config().get("location_stale_after_seconds", 120))
        if age_seconds > stale_after:
            message += f" This location is {format_age(age_seconds)} old."
        return message

    def distance_to(self, place_name: str) -> str:
        self._require_enabled()
        point = self._require_freshest_point()
        place = places.get_place(place_name)
        if place is None:
            raise LocationPlaceNotFoundError(f"No saved place named '{place_name}'. Save it first with location.save_place.")
        place_latitude, place_longitude = place
        distance_meters = haversine_distance_meters(point.latitude, point.longitude, place_latitude, place_longitude)
        return f"{place_name.strip()} is {format_distance(distance_meters)} away."

    def save_place(self, name: str, latitude: float, longitude: float) -> str:
        self._require_enabled()
        if not (-90.0 <= latitude <= 90.0) or not (-180.0 <= longitude <= 180.0):
            raise LocationDestinationError("Invalid latitude/longitude.")
        places.save_place(name, latitude, longitude)
        return f"Saved place '{name.strip().lower()}'."

    # --- navigation.* tools --------------------------------------------------

    def start_navigation(self, destination: str) -> str:
        self._require_enabled()
        origin = self._require_freshest_point()
        destination_latitude, destination_longitude, destination_name = self._resolve_destination(destination)
        route = self.routing_provider().route((origin.latitude, origin.longitude), (destination_latitude, destination_longitude))
        owner_request_id, owner_agent_task_id = self._capture_owner_ids()
        session = NavigationSession(
            session_id=self._next_session_id(),
            owner_request_id=owner_request_id,
            owner_agent_task_id=owner_agent_task_id,
            destination_name=destination_name,
            destination_latitude=destination_latitude,
            destination_longitude=destination_longitude,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        first_instruction = route.steps[0].instruction if route.steps else "Head toward your destination."
        session.last_instruction = first_instruction
        state = get_location_state()
        with state.lock:
            state.sessions[session.session_id] = session
        record_audit_event("navigation_started", message=f"session {session.session_id} to {destination_name}")
        return (
            f"Navigating to {destination_name} ({format_distance(route.distance_meters)}, "
            f"ETA {format_duration(route.duration_seconds)}). Session {session.session_id}: {first_instruction}"
        )

    def get_next_instruction(self, session_id: str) -> str:
        self._require_enabled()
        session = self._require_session(session_id)
        origin = self._require_freshest_point()
        route = self.routing_provider().route(
            (origin.latitude, origin.longitude),
            (session.destination_latitude, session.destination_longitude),
        )
        if route.distance_meters <= ARRIVAL_THRESHOLD_METERS:
            session.last_instruction = "You have arrived at your destination."
            return f"You have arrived at {session.destination_name}."
        instruction = route.steps[0].instruction if route.steps else "Continue toward your destination."
        session.last_instruction = instruction
        return f"{instruction} ({format_distance(route.distance_meters)} remaining, ETA {format_duration(route.duration_seconds)})."

    def stop_navigation(self, session_id: str) -> str:
        self._require_enabled()
        state = get_location_state()
        with state.lock:
            session = state.sessions.pop(session_id, None)
        if session is None:
            raise LocationSessionNotFoundError(f"No active navigation session '{session_id}'.")
        record_audit_event("navigation_stopped", message=f"session {session_id}")
        return f"Navigation session {session_id} ended."

    def cleanup_sessions_for_task(self, *, owner_request_id: int | None = None, owner_agent_task_id: int | None = None) -> int:
        state = get_location_state()
        removed = 0
        with state.lock:
            for session_id in list(state.sessions):
                session = state.sessions[session_id]
                if (owner_agent_task_id is not None and session.owner_agent_task_id == owner_agent_task_id) or (
                    owner_request_id is not None and session.owner_request_id == owner_request_id
                ):
                    del state.sessions[session_id]
                    removed += 1
        return removed

    # --- status commands -------------------------------------------------------

    def location_status_message(self) -> str:
        config = self.effective_config()
        point = self.get_freshest_point()
        lines = [f"Location enabled: {'yes' if config.get('location_enabled', False) else 'no'}"]
        if point is None:
            lines.append("Freshest point: none received yet")
        else:
            age_seconds = self._point_age_seconds(point)
            lines.append(f"Freshest point age: {format_age(age_seconds)}")
        lines.append(f"Tailscale bind interface present: {'yes' if self._bind_interface_present(config) else 'no'}")
        return "\n".join(lines)

    def navigation_status_message(self) -> str:
        state = get_location_state()
        with state.lock:
            session_count = len(state.sessions)
            sessions = list(state.sessions.values())
        lines = [f"Active navigation sessions: {session_count}"]
        for session in sessions:
            lines.append(f"- {session.session_id}: to {session.destination_name} ({session.last_instruction or 'no instruction yet'})")
        return "\n".join(lines)

    # --- internal helpers --------------------------------------------------

    def _require_freshest_point(self) -> LocationPoint:
        point = self.get_freshest_point()
        if point is None:
            raise LocationUnavailableError("No location has been received from the phone yet.")
        return point

    def _require_session(self, session_id: str) -> NavigationSession:
        state = get_location_state()
        with state.lock:
            session = state.sessions.get(session_id)
        if session is None:
            raise LocationSessionNotFoundError(f"No active navigation session '{session_id}'.")
        return session

    def _resolve_destination(self, destination: str) -> tuple[float, float, str]:
        raw = destination.strip()
        saved = places.get_place(raw)
        if saved is not None:
            return saved[0], saved[1], raw
        coordinates = parse_raw_coordinates(raw)
        if coordinates is not None:
            return coordinates[0], coordinates[1], raw
        latitude, longitude = self.geocoder().search(raw)
        return latitude, longitude, raw

    def _point_age_seconds(self, point: LocationPoint) -> float:
        reference = point.captured_at or point.received_at
        try:
            parsed = datetime.fromisoformat(reference.replace("Z", "+00:00"))
        except ValueError:
            return 0.0
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())

    def _bind_interface_present(self, config: dict[str, Any]) -> bool:
        try:
            from app.brain.location.server import is_bind_host_present

            return is_bind_host_present(str(config.get("location_bind_host", "")))
        except Exception:
            return False

    def _next_session_id(self) -> str:
        state = get_location_state()
        with state.lock:
            session_id = f"nav-{state.next_session_id}"
            state.next_session_id += 1
        return session_id

    def _capture_owner_ids(self) -> tuple[int | None, int | None]:
        owner_request_id: int | None = None
        try:
            from app.brain.intelligence.controller import get_intelligence_controller

            request = get_intelligence_controller().state.current_request
            if request is not None:
                owner_request_id = request.request_id
        except Exception:
            owner_request_id = None
        runtime_state = get_agent_runtime_state()
        owner_agent_task_id = runtime_state.current_task.task_id if runtime_state.current_task is not None else None
        return owner_request_id, owner_agent_task_id


_CONTROLLER = LocationController()


def get_location_controller() -> LocationController:
    return _CONTROLLER


def reset_location_controller(*, geocoder: Geocoder | None = None, routing_provider: RoutingProvider | None = None) -> LocationController:
    global _CONTROLLER
    _CONTROLLER = LocationController(geocoder=geocoder, routing_provider=routing_provider)
    return _CONTROLLER

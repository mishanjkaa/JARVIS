from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

from app.brain.location.models import LocationPoint, NavigationSession


@dataclass
class LocationRuntimeState:
    # Freshest point per device, keyed by Overland's device_id. RFC-010 keeps no history:
    # each new Overland post overwrites the previous point for that device_id.
    points: dict[str, LocationPoint] = field(default_factory=dict)
    sessions: dict[str, NavigationSession] = field(default_factory=dict)
    next_session_id: int = 1
    last_safe_status: str = "idle"
    lock: RLock = field(default_factory=RLock)


_STATE = LocationRuntimeState()


def get_location_state() -> LocationRuntimeState:
    return _STATE


def reset_location_state() -> None:
    with _STATE.lock:
        _STATE.points.clear()
        _STATE.sessions.clear()
        _STATE.next_session_id = 1
        _STATE.last_safe_status = "idle"

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LocationPoint:
    device_id: str
    latitude: float
    longitude: float
    accuracy_meters: float | None
    captured_at: str
    received_at: str


@dataclass
class NavigationSession:
    session_id: str
    owner_request_id: int | None
    owner_agent_task_id: int | None
    destination_name: str
    destination_latitude: float
    destination_longitude: float
    created_at: str
    last_instruction: str = ""

    def public_metadata(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "destination_name": self.destination_name,
            "created_at": self.created_at,
            "last_instruction": self.last_instruction,
        }

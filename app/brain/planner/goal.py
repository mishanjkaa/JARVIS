from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass(frozen=True)
class Goal:
    goal_id: str = field(default_factory=lambda: uuid4().hex)
    user_request_category: str = "unsupported"
    requested_outcome: str = ""
    required_capabilities: tuple[str, ...] = ()
    contains_persistent_write: bool = False
    contains_external_navigation: bool = False
    contains_power_request: bool = False
    maximum_steps: int = 5
    creation_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Goal:
    goal_id: str
    user_request_category: str
    requested_outcome: str
    required_capabilities: list[str] = field(default_factory=list)
    contains_persistent_write: bool = False
    contains_external_navigation: bool = False
    contains_power_request: bool = False
    maximum_steps: int = 5
    creation_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "user_request_category": self.user_request_category,
            "requested_outcome": self.requested_outcome,
            "required_capabilities": self.required_capabilities,
            "contains_persistent_write": self.contains_persistent_write,
            "contains_external_navigation": self.contains_external_navigation,
            "contains_power_request": self.contains_power_request,
            "maximum_steps": self.maximum_steps,
            "creation_timestamp": self.creation_timestamp,
        }

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentStep:
    step_id: int
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)
    risk_level: str = "read_only"
    user_visible_description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentPlan:
    steps: list[AgentStep] = field(default_factory=list)
    goal: object | None = None
    original_request: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "goal": self.goal.to_dict() if hasattr(self.goal, "to_dict") else None,
            "original_request": self.original_request,
        }

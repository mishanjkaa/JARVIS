from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class AIIntent(str, Enum):
    CONVERSATION = "conversation"
    TOOL_CALL = "tool_call"
    PLAN = "plan"
    FALLBACK = "fallback"


class ProviderStatusCategory(str, Enum):
    UNAVAILABLE = "unavailable"
    READY = "ready"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ProviderStatus:
    category: ProviderStatusCategory
    detail: str = ""
    provider: str = "ollama"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanStep:
    id: int
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reference: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionPlan:
    steps: list[PlanStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps]}


@dataclass
class AIResponse:
    category: str
    message: str = ""
    intent: Optional[AIIntent] = None
    tool_call: Optional[ToolCall] = None
    plan: Optional[ExecutionPlan] = None
    status: Optional[ProviderStatus] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "message": self.message,
            "intent": self.intent.value if self.intent else None,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "status": self.status.to_dict() if self.status else None,
        }

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ToolResult:
    success: bool
    status_category: str
    message: str
    display_value: str = ""
    reference_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolDefinition:
    name: str
    description: str
    argument_schema: dict[str, Any]
    risk_level: str
    requires_confirmation: bool
    handler: Callable[[dict[str, Any]], ToolResult]
    formatter: Callable[[ToolResult], str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

from __future__ import annotations

from app.brain.tools.builtin_tools import build_builtin_tools
from app.brain.tools.models import ToolDefinition


class ToolRegistry:
    def __init__(self, tools: list[ToolDefinition] | None = None) -> None:
        self._tools = {tool.name: tool for tool in (tools or build_builtin_tools())}

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def list_names(self) -> list[str]:
        return sorted(self._tools)

    def list_tools(self) -> list[ToolDefinition]:
        return [self._tools[name] for name in sorted(self._tools)]

    def count(self) -> int:
        return len(self._tools)

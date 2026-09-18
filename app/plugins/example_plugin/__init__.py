from __future__ import annotations

from app.brain.tools.models import ToolDefinition, ToolResult
from app.brain.tools.validators import validate_text


def build_tools() -> list[ToolDefinition]:
    return [ToolDefinition("example_plugin.echo", "read_only", {"text": {"type": "text", "max_length": 80}}, _echo)]


def _echo(arguments: dict) -> ToolResult:
    text = validate_text(arguments["text"], field_name="text", max_length=80)
    return ToolResult(True, "success", text, display_value=text)
from app.plugins.example_plugin.tools import register

__all__ = ["register"]

from __future__ import annotations

from app.brain.tools.models import ToolDefinition, ToolResult
from app.brain.tools.validators import validate_text


def _handler(args: dict) -> ToolResult:
    text = validate_text(args.get("text"), field_name="text", max_length=60)
    return ToolResult(True, "success", text, text)


def register() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="example.echo",
            description="Echo a short validated text string.",
            argument_schema={"text": {"type": "text", "max_length": 60}},
            risk_level="read_only",
            requires_confirmation=False,
            handler=_handler,
            formatter=lambda result: result.display_value or result.message,
        )
    ]

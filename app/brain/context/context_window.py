from __future__ import annotations

from app.brain.context.sanitizer import sanitize_summary


def build_context_window(entries: list[str], limit: int = 8) -> str:
    return "\n".join(f"- {item}" for item in sanitize_summary(entries, limit))
from __future__ import annotations

from app.brain.context.conversation import get_conversation_summaries
from app.brain.context.sanitizer import sanitize_for_prompt


def build_context_window(limit: int = 8) -> str:
    summaries = get_conversation_summaries()[-limit:]
    if not summaries:
        return ""
    return "\n".join(sanitize_for_prompt(item) for item in summaries)

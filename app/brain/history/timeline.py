from __future__ import annotations

from app.brain.history.timeline_models import TimelineEntry
from app.brain.history.timeline_state import get_timeline_state, reset_timeline_state
from config.config_loader import load_config


def add_timeline_entry(display_label: str, *, success: bool, tool_name: str = "") -> None:
    if not isinstance(display_label, str) or not display_label.strip():
        return
    state = get_timeline_state()
    state.entries.append(TimelineEntry(display_label=display_label.strip(), success=success, tool_name=tool_name))
    maximum = int(load_config().get("max_timeline_entries", 100))
    if len(state.entries) > maximum:
        state.entries = state.entries[-maximum:]


def get_timeline() -> list[TimelineEntry]:
    return list(reversed(get_timeline_state().entries))


def format_timeline() -> str:
    entries = get_timeline()
    if not entries:
        return "No timeline entries yet."
    return "\n".join(entry.display_label for entry in entries)


def clear_timeline() -> None:
    get_timeline_state().entries.clear()


def reset_timeline() -> None:
    reset_timeline_state()

from __future__ import annotations

from dataclasses import dataclass, field

from app.brain.history.timeline_models import TimelineEntry


@dataclass
class TimelineState:
    entries: list[TimelineEntry] = field(default_factory=list)


_STATE = TimelineState()


def get_timeline_state() -> TimelineState:
    return _STATE


def reset_timeline_state() -> None:
    _STATE.entries = []

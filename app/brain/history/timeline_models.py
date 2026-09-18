from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TimelineEntry:
    display_label: str
    success: bool
    tool_name: str = ""

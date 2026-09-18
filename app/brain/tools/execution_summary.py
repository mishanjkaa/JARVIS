from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExecutionSummary:
    success: bool
    message: str
    results: list[dict] = field(default_factory=list)
    skipped_steps: list[int] = field(default_factory=list)
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionSummary:
    success: bool
    message: str
    summaries: list[str] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    skipped_steps: list[int] = field(default_factory=list)

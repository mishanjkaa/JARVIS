from __future__ import annotations

class ExecutionContext:
    """Per-plan results; never shared as global runtime state."""
    def __init__(self) -> None:
        self.results: dict[int, dict] = {}
        self.executed_steps: list[int] = []
        self.skipped_steps: list[int] = []
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionContext:
    step_results: dict[int, dict[str, Any]] = field(default_factory=dict)
    executed_steps: list[int] = field(default_factory=list)
    skipped_steps: list[int] = field(default_factory=list)
    failure_step: int | None = None

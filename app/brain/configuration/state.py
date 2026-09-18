from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeConfigState:
    snapshot: dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    last_safe_status: str = "default"


_STATE = RuntimeConfigState()


def get_runtime_config_state() -> RuntimeConfigState:
    return _STATE


def reset_runtime_config_state() -> None:
    _STATE.snapshot = {}
    _STATE.generation = 0
    _STATE.last_safe_status = "default"

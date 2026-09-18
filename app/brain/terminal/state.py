from __future__ import annotations

import threading
from dataclasses import dataclass, field
from subprocess import Popen
from typing import Any

from app.brain.terminal.models import TerminalCommandRequest, TerminalExecutionRecord, TerminalExecutionResult


@dataclass
class TerminalRuntimeState:
    active_request: TerminalCommandRequest | None = None
    active_process: Popen[str] | None = None
    active_result: TerminalExecutionResult | None = None
    recent_history: list[TerminalExecutionRecord] = field(default_factory=list)
    cancellation_requested: bool = False
    last_safe_status: str = "Terminal enabled. No command currently running."
    lock: threading.RLock = field(default_factory=threading.RLock)


_STATE = TerminalRuntimeState()


def get_terminal_state() -> TerminalRuntimeState:
    return _STATE


def reset_terminal_state() -> None:
    with _STATE.lock:
        _STATE.active_request = None
        _STATE.active_process = None
        _STATE.active_result = None
        _STATE.recent_history = []
        _STATE.cancellation_requested = False
        _STATE.last_safe_status = "Terminal enabled. No command currently running."


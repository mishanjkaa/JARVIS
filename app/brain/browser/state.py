from __future__ import annotations

import threading
from dataclasses import dataclass, field

from app.brain.browser.models import BrowserSessionRecord


@dataclass
class BrowserRuntimeState:
    sessions: dict[str, BrowserSessionRecord] = field(default_factory=dict)
    next_session_id: int = 1
    focused_session_id: str = ""
    active_session_id: str = ""
    active_operation: str = ""
    cancellation_requested: bool = False
    last_safe_status: str = "Browser runtime idle."
    lock: threading.RLock = field(default_factory=threading.RLock)


_STATE = BrowserRuntimeState()


def get_browser_state() -> BrowserRuntimeState:
    return _STATE


def reset_browser_state() -> None:
    with _STATE.lock:
        _STATE.sessions = {}
        _STATE.next_session_id = 1
        _STATE.focused_session_id = ""
        _STATE.active_session_id = ""
        _STATE.active_operation = ""
        _STATE.cancellation_requested = False
        _STATE.last_safe_status = "Browser runtime idle."

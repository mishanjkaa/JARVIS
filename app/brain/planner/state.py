from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable


@dataclass
class PlannerState:
    pending_plan: object | None = None
    pending_summary: str | None = None
    created_at: datetime | None = None
    approved: bool = False

    @property
    def plan_created_at(self):
        return self.created_at

    @property
    def approval_state(self) -> str:
        return "approved" if self.approved else "preview_required" if self.pending_plan is not None else "no_approval"

    def clear(self) -> None:
        reset_planner_state()

    def set_pending_plan(self, plan: object, summary: str | None = None) -> None:
        self.pending_plan = plan
        self.pending_summary = summary
        self.created_at = planner_now()
        self.approved = False

    def is_expired(self) -> bool:
        return self.pending_plan is None or pending_plan_expired()


_STATE = PlannerState()
_NOW_PROVIDER: Callable[[], datetime] = lambda: datetime.now(timezone.utc)


def get_planner_state() -> PlannerState:
    return _STATE


def reset_planner_state() -> None:
    _STATE.pending_plan = None
    _STATE.pending_summary = None
    _STATE.created_at = None
    _STATE.approved = False


def pending_plan_expired(timeout_seconds: int = 60) -> bool:
    return _STATE.created_at is not None and planner_now() - _STATE.created_at > timedelta(seconds=timeout_seconds)


def planner_now() -> datetime:
    return _NOW_PROVIDER()


def set_planner_now_provider(provider: Callable[[], datetime]) -> None:
    global _NOW_PROVIDER
    _NOW_PROVIDER = provider


def reset_planner_now_provider() -> None:
    global _NOW_PROVIDER
    _NOW_PROVIDER = lambda: datetime.now(timezone.utc)

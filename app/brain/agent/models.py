from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from app.brain.agent.errors import AgentStateTransitionError
from app.brain.planner.plan_models import AgentPlan


class AgentLifecycleState(str, Enum):
    IDLE = "idle"
    PENDING_APPROVAL = "pending_approval"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EMERGENCY_STOPPED = "emergency_stopped"


ACTIVE_AGENT_STATES = {
    AgentLifecycleState.PENDING_APPROVAL,
    AgentLifecycleState.RUNNING,
}

TERMINAL_AGENT_STATES = {
    AgentLifecycleState.COMPLETED,
    AgentLifecycleState.FAILED,
    AgentLifecycleState.CANCELLED,
    AgentLifecycleState.EMERGENCY_STOPPED,
}

VALID_AGENT_TRANSITIONS: dict[AgentLifecycleState, set[AgentLifecycleState]] = {
    AgentLifecycleState.IDLE: {AgentLifecycleState.PENDING_APPROVAL},
    AgentLifecycleState.PENDING_APPROVAL: {
        AgentLifecycleState.RUNNING,
        AgentLifecycleState.CANCELLED,
        AgentLifecycleState.EMERGENCY_STOPPED,
    },
    AgentLifecycleState.RUNNING: {
        AgentLifecycleState.COMPLETED,
        AgentLifecycleState.FAILED,
        AgentLifecycleState.CANCELLED,
        AgentLifecycleState.EMERGENCY_STOPPED,
    },
    AgentLifecycleState.COMPLETED: set(),
    AgentLifecycleState.FAILED: set(),
    AgentLifecycleState.CANCELLED: set(),
    AgentLifecycleState.EMERGENCY_STOPPED: set(),
}


def validate_agent_transition(current: AgentLifecycleState, new: AgentLifecycleState) -> None:
    allowed = VALID_AGENT_TRANSITIONS.get(current, set())
    if new not in allowed:
        raise AgentStateTransitionError(f"invalid agent transition: {current.value} -> {new.value}")


@dataclass
class AgentTaskRecord:
    task_id: int
    plan: AgentPlan
    original_user_goal: str
    total_steps: int
    state: AgentLifecycleState = AgentLifecycleState.IDLE
    current_step_index: int = 0
    start_time: datetime | None = None
    finish_time: datetime | None = None
    latest_safe_status_message: str = ""
    final_result: str = ""
    failure_reason: str = ""
    failed_step_index: int = 0
    failed_tool_name: str = ""
    failed_error_category: str = ""
    cancellation_requested: bool = False
    step_results: list[dict[str, Any]] = field(default_factory=list)

    def transition_to(self, new_state: AgentLifecycleState) -> None:
        validate_agent_transition(self.state, new_state)
        self.state = new_state

    def is_active(self) -> bool:
        return self.state in ACTIVE_AGENT_STATES

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_AGENT_STATES

    def progress_label(self) -> str:
        return f"{self.current_step_index}/{self.total_steps}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "state": self.state.value,
            "current_step_index": self.current_step_index,
            "total_steps": self.total_steps,
            "original_user_goal": self.original_user_goal,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "finish_time": self.finish_time.isoformat() if self.finish_time else None,
            "latest_safe_status_message": self.latest_safe_status_message,
            "final_result": self.final_result,
            "failure_reason": self.failure_reason,
            "failed_step_index": self.failed_step_index,
            "failed_tool_name": self.failed_tool_name,
            "failed_error_category": self.failed_error_category,
        }


@dataclass
class AgentRuntimeStatus:
    session_state: AgentLifecycleState
    task: AgentTaskRecord | None = None
    emergency_stop_active: bool = False

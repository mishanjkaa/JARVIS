from __future__ import annotations

from dataclasses import dataclass, field

from app.brain.agent.models import AgentLifecycleState, AgentTaskRecord


@dataclass
class AgentRuntimeState:
    current_task: AgentTaskRecord | None = None
    archived_tasks: list[AgentTaskRecord] = field(default_factory=list)
    next_task_id: int = 1
    emergency_stop_active: bool = False
    last_safe_status_message: str = "Agent state: idle."

    def archive_current_task(self) -> None:
        if self.current_task is not None:
            self.archived_tasks.append(self.current_task)
        self.current_task = None

    def get_effective_state(self) -> AgentLifecycleState:
        if self.current_task is not None:
            return self.current_task.state
        if self.emergency_stop_active:
            return AgentLifecycleState.EMERGENCY_STOPPED
        return AgentLifecycleState.IDLE


_STATE = AgentRuntimeState()


def get_agent_runtime_state() -> AgentRuntimeState:
    return _STATE


def reset_agent_runtime_state() -> None:
    _STATE.current_task = None
    _STATE.archived_tasks = []
    _STATE.next_task_id = 1
    _STATE.emergency_stop_active = False
    _STATE.last_safe_status_message = "Agent state: idle."

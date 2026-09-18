from app.brain.agent.models import AgentLifecycleState, AgentTaskRecord, validate_agent_transition
from app.brain.agent.state import get_agent_runtime_state, reset_agent_runtime_state


def get_agent_controller():
    from app.brain.agent.controller import get_agent_controller as _get_agent_controller

    return _get_agent_controller()


class AgentController:  # pragma: no cover - compatibility export
    def __new__(cls, *args, **kwargs):
        from app.brain.agent.controller import AgentController as _AgentController

        return _AgentController(*args, **kwargs)

__all__ = [
    "AgentController",
    "AgentLifecycleState",
    "AgentTaskRecord",
    "get_agent_controller",
    "get_agent_runtime_state",
    "reset_agent_runtime_state",
    "validate_agent_transition",
]

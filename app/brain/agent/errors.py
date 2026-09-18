from __future__ import annotations


class AgentError(ValueError):
    """Base class for safe agent runtime failures."""


class AgentStateTransitionError(AgentError):
    """Raised when a task attempts an invalid lifecycle transition."""


class AgentDisabledError(AgentError):
    """Raised when agent runtime is disabled by configuration."""


class ActiveAgentConflictError(AgentError):
    """Raised when a new task is requested while another is still active."""


class AgentEmergencyStopError(AgentError):
    """Raised when the runtime is blocked by an emergency stop."""


class AgentPlanError(AgentError):
    """Raised when a plan cannot be executed safely."""

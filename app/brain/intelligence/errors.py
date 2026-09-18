from __future__ import annotations


class IntelligenceError(ValueError):
    """Base class for safe intelligence-runtime failures."""


class IntelligenceProviderUnavailableError(IntelligenceError):
    """Raised when the configured intelligence provider is unavailable."""


class IntelligenceProviderRequestError(IntelligenceError):
    """Raised when the configured intelligence provider rejects or fails a request."""

    def __init__(self, category: str, safe_detail: str, *, status_code: int | None = None) -> None:
        super().__init__(safe_detail)
        self.category = category
        self.safe_detail = safe_detail
        self.status_code = status_code


class MalformedModelOutputError(IntelligenceError):
    """Raised when the provider returns malformed structured output."""


class DynamicPlanError(IntelligenceError):
    """Raised when the provider or planner cannot build a valid plan."""


class ClarificationRequiredError(IntelligenceError):
    """Raised when an actionable task still needs clarification."""


class UnsupportedTaskError(IntelligenceError):
    """Raised when a request targets an unsupported capability."""

from __future__ import annotations


class TerminalError(ValueError):
    """Base class for safe terminal runtime failures."""


class TerminalPolicyError(TerminalError):
    """Raised when terminal policy rejects a request."""


class TerminalTimeoutError(TerminalError):
    """Raised when terminal execution times out."""


class TerminalCancellationError(TerminalError):
    """Raised when terminal execution is cancelled."""


class TerminalWorkingDirectoryError(TerminalError):
    """Raised when the working directory is invalid."""


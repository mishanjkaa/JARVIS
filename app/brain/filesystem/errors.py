from __future__ import annotations


class FilesystemError(ValueError):
    """Base class for safe filesystem runtime failures."""


class FilesystemDisabledError(FilesystemError):
    """Raised when filesystem runtime is disabled."""


class FilesystemPathError(FilesystemError):
    """Raised when a path is unsafe or escapes trusted roots."""


class FilesystemLimitError(FilesystemError):
    """Raised when a configured size or depth limit is exceeded."""


class FilesystemOperationError(FilesystemError):
    """Raised when a filesystem operation cannot complete safely."""

from __future__ import annotations


class VisionError(ValueError):
    """Base class for safe Vision-runtime failures."""


class VisionDisabledError(VisionError):
    """Raised when the Vision runtime is disabled."""


class VisionPolicyError(VisionError):
    """Raised when an image request violates local Vision policy."""


class VisionImageError(VisionError):
    """Raised when a local image cannot be loaded safely."""


class VisionProviderUnavailableError(VisionError):
    """Raised when the local Vision provider is unavailable."""


class VisionProviderError(VisionError):
    """Raised when the Vision provider rejects or malforms a request."""


class VisionEvidenceExpiredError(VisionError):
    """Raised when temporary Vision evidence is no longer current."""


class VisionCaptureUnsupportedError(VisionError):
    """Raised when a capture operation is not supported on the current platform."""

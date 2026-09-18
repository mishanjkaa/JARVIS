from __future__ import annotations


class VoiceError(ValueError):
    """Base class for safe Voice-runtime failures."""


class VoiceDisabledError(VoiceError):
    """Raised when the Voice runtime is disabled."""


class VoiceNotEnrolledError(VoiceError):
    """Raised when a voice turn is attempted before the owner has enrolled."""


class VoiceVerificationFailedError(VoiceError):
    """Raised when captured speech does not match the enrolled owner's voice. No transcript
    is produced or kept for a turn that fails verification."""


class VoiceCaptureError(VoiceError):
    """Raised when local microphone capture or playback fails."""


class VoiceProviderError(VoiceError):
    """Raised when a speech-to-text or text-to-speech provider fails or is unavailable."""


class VoiceEnrollmentError(VoiceError):
    """Raised for an invalid enrollment operation (e.g. finalizing with no samples banked)."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock


@dataclass
class VoiceRuntimeState:
    # Visible mic-active indicator (RFC-009's "Visible state" requirement, mirroring the
    # camera-state visibility already promised for RFC-007D in VISION_ROADMAP.md): true only
    # for the duration audio is actively being captured for one voice turn.
    mic_active: bool = False
    # In-progress `voice enroll` samples, banked one at a time across separate command
    # invocations and not yet averaged into a saved VoiceProfile. Cleared on finalize,
    # cancellation, or reset.
    pending_enrollment_embeddings: list[list[float]] = field(default_factory=list)
    last_safe_status: str = "idle"
    lock: RLock = field(default_factory=RLock)


_STATE = VoiceRuntimeState()


def get_voice_state() -> VoiceRuntimeState:
    return _STATE


def reset_voice_state() -> None:
    with _STATE.lock:
        _STATE.mic_active = False
        _STATE.pending_enrollment_embeddings.clear()
        _STATE.last_safe_status = "idle"

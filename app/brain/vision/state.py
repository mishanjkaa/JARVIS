from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

from app.brain.vision.models import BrowserCaptureRecord, VisionEvidence


@dataclass
class VisionRuntimeState:
    evidences: dict[str, VisionEvidence] = field(default_factory=dict)
    captures: dict[str, BrowserCaptureRecord] = field(default_factory=dict)
    next_evidence_id: int = 1
    next_capture_id: int = 1
    last_safe_status: str = "idle"
    readiness_generation: int = -1
    readiness_signature: str = ""
    generation_ready: bool | None = None
    generation_detail: str = ""
    generation_checked_at: str = ""
    generation_elapsed_ms: int | None = None
    lock: RLock = field(default_factory=RLock)


_STATE = VisionRuntimeState()


def get_vision_state() -> VisionRuntimeState:
    return _STATE


def reset_vision_state() -> None:
    _STATE.evidences.clear()
    _STATE.captures.clear()
    _STATE.next_evidence_id = 1
    _STATE.next_capture_id = 1
    _STATE.last_safe_status = "idle"
    invalidate_vision_readiness_cache()


def invalidate_vision_readiness_cache() -> None:
    _STATE.readiness_generation = -1
    _STATE.readiness_signature = ""
    _STATE.generation_ready = None
    _STATE.generation_detail = ""
    _STATE.generation_checked_at = ""
    _STATE.generation_elapsed_ms = None

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VoiceProfile:
    """The enrolled owner's voice, as a single averaged speaker embedding. RFC-009 keeps
    this closed-set and opt-in, mirroring the face-enrollment shape already described as a
    permanent decision in docs/VISION_ROADMAP.md: one local template, no raw audio kept,
    explicit and confirmed deletion."""

    embedding: list[float]
    sample_count: int
    enrolled_at: str
    updated_at: str


@dataclass
class VoiceTurnResult:
    transcript: str
    reply_text: str
    reply_audio_wav: bytes = field(repr=False, default=b"")

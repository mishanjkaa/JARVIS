from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.brain.voice.errors import VoiceEnrollmentError
from app.brain.voice.models import VoiceProfile
from app.brain.voice.state import get_voice_state
from app.brain.voice.verification import average_embeddings

# Closed-set, opt-in enrollment, mirroring the face-enrollment shape already established as
# a permanent decision in docs/VISION_ROADMAP.md: one local template for the owner only, no
# raw audio retained (only the derived embedding), and confirmed deletion.
MIN_ENROLLMENT_SAMPLES = 3


def get_profile_file_path(profile_file: Optional[Path] = None) -> Path:
    if profile_file is not None:
        return profile_file
    base_dir = Path(__file__).resolve().parents[3]
    return base_dir / "data" / "voice_profile.json"


def _load_profile(path: Path) -> Optional[VoiceProfile]:
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    if not content.strip():
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    embedding = parsed.get("embedding")
    if not isinstance(embedding, list) or not all(isinstance(value, (int, float)) for value in embedding):
        return None
    return VoiceProfile(
        embedding=[float(value) for value in embedding],
        sample_count=int(parsed.get("sample_count", 0)),
        enrolled_at=str(parsed.get("enrolled_at", "")),
        updated_at=str(parsed.get("updated_at", "")),
    )


def _save_profile(path: Path, profile: VoiceProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embedding": profile.embedding,
        "sample_count": profile.sample_count,
        "enrolled_at": profile.enrolled_at,
        "updated_at": profile.updated_at,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_profile(profile_file: Optional[Path] = None) -> Optional[VoiceProfile]:
    return _load_profile(get_profile_file_path(profile_file))


def is_enrolled(profile_file: Optional[Path] = None) -> bool:
    return load_profile(profile_file) is not None


def delete_profile(profile_file: Optional[Path] = None) -> bool:
    path = get_profile_file_path(profile_file)
    if not path.exists():
        return False
    path.unlink()
    return True


def bank_enrollment_sample(embedding: list[float]) -> int:
    """Add one sample to the in-progress enrollment buffer (not yet persisted) and return
    how many samples are banked so far."""
    state = get_voice_state()
    with state.lock:
        state.pending_enrollment_embeddings.append(list(embedding))
        return len(state.pending_enrollment_embeddings)


def pending_sample_count() -> int:
    state = get_voice_state()
    with state.lock:
        return len(state.pending_enrollment_embeddings)


def cancel_enrollment() -> None:
    state = get_voice_state()
    with state.lock:
        state.pending_enrollment_embeddings.clear()


def finalize_enrollment(profile_file: Optional[Path] = None) -> VoiceProfile:
    state = get_voice_state()
    with state.lock:
        embeddings = list(state.pending_enrollment_embeddings)
        if len(embeddings) < MIN_ENROLLMENT_SAMPLES:
            raise VoiceEnrollmentError(
                f"Need at least {MIN_ENROLLMENT_SAMPLES} voice samples to enroll; have {len(embeddings)} so far. Run 'voice enroll' again."
            )
        now = datetime.now(timezone.utc).isoformat()
        existing = load_profile(profile_file)
        profile = VoiceProfile(
            embedding=average_embeddings(embeddings),
            sample_count=len(embeddings),
            enrolled_at=existing.enrolled_at if existing is not None else now,
            updated_at=now,
        )
        _save_profile(get_profile_file_path(profile_file), profile)
        state.pending_enrollment_embeddings.clear()
        return profile

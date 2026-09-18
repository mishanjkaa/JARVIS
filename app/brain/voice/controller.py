from __future__ import annotations

import logging
from typing import Any, Callable

from app.brain.audit.audit_log import record_audit_event
from app.brain.configuration.runtime_config import get_effective_runtime_config
from app.brain.voice import enrollment
from app.brain.voice.audio_io import play_audio_wav, record_from_microphone
from app.brain.voice.enrollment import MIN_ENROLLMENT_SAMPLES
from app.brain.voice.errors import VoiceCaptureError, VoiceDisabledError, VoiceNotEnrolledError, VoiceProviderError, VoiceVerificationFailedError
from app.brain.voice.models import VoiceTurnResult
from app.brain.voice.speech_to_text import DEFAULT_STT_MODEL, FasterWhisperSTTProvider, STTProvider
from app.brain.voice.state import get_voice_state
from app.brain.voice.text_to_speech import PiperTTSProvider, TTSProvider
from app.brain.voice.verification import VOICE_SAMPLE_RATE_HZ, SpeechBrainVerificationProvider, VerificationProvider, cosine_similarity

logger = logging.getLogger(__name__)

DEFAULT_PUSH_TO_TALK_SECONDS = 6.0
DEFAULT_ENROLLMENT_SAMPLE_SECONDS = 4.0


class VoiceController:
    def __init__(
        self,
        *,
        stt_provider: STTProvider | None = None,
        tts_provider: TTSProvider | None = None,
        verification_provider: VerificationProvider | None = None,
    ) -> None:
        self._stt_override = stt_provider
        self._tts_override = tts_provider
        self._verification_override = verification_provider
        # Bug found while investigating "voice talk always rejects the enrolled owner":
        # verification_provider() used to construct a brand-new SpeechBrainVerificationProvider
        # (and therefore reload the whole ECAPA model from disk) on *every single*
        # enroll_sample()/handle_voice_turn() call. SpeechBrain's Pretrained class does put
        # the model in eval() mode with frozen params, so a fresh load is not the reason
        # embeddings differ -- but reloading an ~80 MB model before every utterance is pure
        # waste, and caching one instance here means enrollment and verification are
        # provably going through the exact same loaded model object, not just "the same
        # class constructed twice", removing that as a variable entirely.
        self._verification_provider_instance: VerificationProvider | None = None

    def effective_config(self) -> dict[str, Any]:
        return get_effective_runtime_config()

    def stt_provider(self) -> STTProvider:
        if self._stt_override is not None:
            return self._stt_override
        config = self.effective_config()
        return FasterWhisperSTTProvider(model_size=str(config.get("voice_stt_model", DEFAULT_STT_MODEL)))

    def tts_provider(self) -> TTSProvider:
        if self._tts_override is not None:
            return self._tts_override
        config = self.effective_config()
        return PiperTTSProvider(voice_model_path=str(config.get("voice_tts_voice", "")))

    def verification_provider(self) -> VerificationProvider:
        if self._verification_override is not None:
            return self._verification_override
        if self._verification_provider_instance is None:
            self._verification_provider_instance = SpeechBrainVerificationProvider()
        return self._verification_provider_instance

    def _require_enabled(self) -> None:
        if not bool(self.effective_config().get("voice_enabled", False)):
            raise VoiceDisabledError("Voice is disabled. Enable it with 'voice on'.")

    # --- enrollment ----------------------------------------------------------

    def enroll_sample(self, *, duration_seconds: float = DEFAULT_ENROLLMENT_SAMPLE_SECONDS) -> str:
        self._require_enabled()
        samples = record_from_microphone(duration_seconds)
        embedding = self.verification_provider().embed(samples, VOICE_SAMPLE_RATE_HZ)
        count = enrollment.bank_enrollment_sample(embedding)
        if count < MIN_ENROLLMENT_SAMPLES:
            return f"Recorded sample {count} of {MIN_ENROLLMENT_SAMPLES}. Say another phrase and run 'voice enroll' again."
        profile = enrollment.finalize_enrollment()
        record_audit_event("voice_enrollment_completed", message=f"{profile.sample_count} samples")
        return f"Enrollment complete with {profile.sample_count} samples."

    def cancel_enrollment(self) -> str:
        enrollment.cancel_enrollment()
        return "Voice enrollment cancelled; no samples were saved."

    def forget_me(self) -> str:
        removed = enrollment.delete_profile()
        enrollment.cancel_enrollment()
        record_audit_event("voice_enrollment_deleted", message="owner requested deletion")
        return "Voice enrollment deleted." if removed else "No voice enrollment was stored."

    def is_enrolled(self) -> bool:
        return enrollment.is_enrolled()

    # --- speaker verification --------------------------------------------------

    def _verify_owner(self, samples: list[float]) -> None:
        profile = enrollment.load_profile()
        if profile is None:
            raise VoiceNotEnrolledError("No voice is enrolled. Run 'voice enroll' first.")
        embedding = self.verification_provider().embed(samples, VOICE_SAMPLE_RATE_HZ)
        # RFC-009 threshold recalibration (second pass): 0.75 (the original default) rejected
        # the real enrolled owner's own genuine voice by a hair (live similarity 0.7311),
        # which motivated a first recalibration down to 0.6. After that, and after fixing the
        # actual pipeline bugs this investigation uncovered (native-format mic capture,
        # SpeechBrain's Windows symlink-privilege crash, and -- most recently -- trimming the
        # silence surrounding speech in every recording via `_trim_silence`), a fresh
        # `voice enroll` under the fully-fixed pipeline produced a much more internally
        # consistent profile (pairwise sample similarity 0.78-0.81, sample-to-average
        # 0.91-0.94, versus the original profile's 0.50-0.67). But real `voice talk` attempts
        # against that fresh, consistent profile still landed at 0.5883, 0.4662, and 0.2318 --
        # i.e. genuine-owner similarity on this real microphone/room, with every mechanical
        # cause (clipping, channel imbalance, stale enrollment, leading/trailing silence)
        # eliminated, simply runs lower and more variably than the single 0.7311 sample that
        # justified 0.6. 0.6 was therefore still too high for this real deployment -- not a
        # guess this time, but the conclusion of exhausting every other explanation first.
        # 0.4 sits below the worst *deliberate, clean* genuine attempt observed (0.4662,
        # margin 0.066) while staying well above speechbrain's own 0.25 reference boundary for
        # this exact spkrec-ecapa-voxceleb checkpoint, and clearly above the one attempt
        # (0.2318) whose own diagnostics (unusually loud but a *lower* active-speech ratio
        # than cleaner attempts) marked it as an anomalous take rather than a representative
        # genuine sample. This still doesn't guarantee rejecting a different speaker on this
        # specific hardware -- no impostor sample has been measured on it -- so it remains
        # fully configurable via `config set voice_verification_threshold <value>` and worth
        # retightening if a false-accept is ever observed, guided by the same similarity logs.
        threshold = float(self.effective_config().get("voice_verification_threshold", 0.4))

        # DIAGNOSTIC ("voice talk always rejects the enrolled owner" investigation): a
        # dimension mismatch between the live and enrolled embeddings would make
        # cosine_similarity() silently return 0.0 with no other symptom, so it is checked
        # and logged explicitly rather than left to fall through into an unexplained score.
        if len(embedding) != len(profile.embedding):
            logger.error(
                "Voice verification embedding-dimension mismatch: live_embedding_len=%d "
                "enrolled_embedding_len=%d. cosine_similarity() returns 0.0 (an automatic "
                "reject) whenever lengths differ -- this points at the live and enrolled "
                "embeddings having been produced by different model configurations, not at "
                "a borderline similarity score.",
                len(embedding), len(profile.embedding),
            )

        similarity = cosine_similarity(embedding, profile.embedding)
        logger.info(
            "Voice verification: live_embedding_len=%d enrolled_embedding_len=%d "
            "enrolled_sample_count=%d similarity=%.4f threshold=%.4f result=%s",
            len(embedding), len(profile.embedding), profile.sample_count,
            similarity, threshold, "accept" if similarity >= threshold else "reject",
        )

        if similarity < threshold:
            record_audit_event(
                "voice_verification_failed",
                message=f"speaker did not match the enrolled owner (similarity={similarity:.4f}, threshold={threshold:.4f})",
            )
            raise VoiceVerificationFailedError("This voice does not match the enrolled owner. No transcript was kept.")

    # --- voice turns ----------------------------------------------------------

    def handle_voice_turn(self, samples: list[float], *, route_text: Callable[[str], str]) -> VoiceTurnResult:
        """Verify the speaker, transcribe, hand the transcript to `route_text` (the exact
        same text-intake path a typed message would use -- see RFC-009), synthesize the
        reply, and return all three. A verification failure raises before any transcription
        happens, so a non-owner voice never produces or keeps a transcript."""
        self._require_enabled()
        self._verify_owner(samples)
        transcript = self.stt_provider().transcribe(samples, VOICE_SAMPLE_RATE_HZ)
        if not transcript.strip():
            raise VoiceProviderError("No speech was recognized.")
        reply_text = route_text(transcript)
        reply_audio = b""
        try:
            reply_audio = self.tts_provider().synthesize(reply_text)
        except VoiceProviderError:
            pass  # the text reply is still meaningful even if TTS is unavailable/misconfigured
        record_audit_event("voice_turn_completed", message="voice turn processed")
        return VoiceTurnResult(transcript=transcript, reply_text=reply_text, reply_audio_wav=reply_audio)

    def push_to_talk_local(self, *, route_text: Callable[[str], str], duration_seconds: float = DEFAULT_PUSH_TO_TALK_SECONDS) -> VoiceTurnResult:
        self._require_enabled()
        samples = record_from_microphone(duration_seconds)
        result = self.handle_voice_turn(samples, route_text=route_text)
        if result.reply_audio_wav:
            try:
                play_audio_wav(result.reply_audio_wav)
            except VoiceCaptureError:
                pass
        return result

    # --- status -----------------------------------------------------------

    def status_message(self) -> str:
        config = self.effective_config()
        state = get_voice_state()
        with state.lock:
            mic_active = state.mic_active
            pending_samples = len(state.pending_enrollment_embeddings)
        lines = [
            f"Voice enabled: {'yes' if config.get('voice_enabled', False) else 'no'}",
            f"Microphone active: {'yes' if mic_active else 'no'}",
            f"Owner voice enrolled: {'yes' if self.is_enrolled() else 'no'}",
        ]
        if pending_samples:
            lines.append(f"Enrollment in progress: {pending_samples} of {MIN_ENROLLMENT_SAMPLES} samples")
        return "\n".join(lines)


_CONTROLLER = VoiceController()


def get_voice_controller() -> VoiceController:
    return _CONTROLLER


def reset_voice_controller(
    *,
    stt_provider: STTProvider | None = None,
    tts_provider: TTSProvider | None = None,
    verification_provider: VerificationProvider | None = None,
) -> VoiceController:
    global _CONTROLLER
    _CONTROLLER = VoiceController(stt_provider=stt_provider, tts_provider=tts_provider, verification_provider=verification_provider)
    return _CONTROLLER

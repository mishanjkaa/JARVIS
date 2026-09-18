from __future__ import annotations

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
        return SpeechBrainVerificationProvider()

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
        threshold = float(self.effective_config().get("voice_verification_threshold", 0.75))
        similarity = cosine_similarity(embedding, profile.embedding)
        if similarity < threshold:
            record_audit_event("voice_verification_failed", message="speaker did not match the enrolled owner")
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

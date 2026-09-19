from __future__ import annotations

import base64
import io
import json
import unittest
import wave
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from app.brain.audit.audit_log import reset_audit_log
from app.brain.configuration.runtime_config import get_effective_runtime_config, replace_runtime_config, reset_runtime_config, set_runtime_config_value
from app.brain.voice import enrollment
from app.brain.voice.audio_io import resample_audio, wav_bytes_to_samples
from app.brain.voice.controller import get_voice_controller, reset_voice_controller
from app.brain.voice.errors import VoiceDisabledError, VoiceNotEnrolledError, VoiceProviderError, VoiceVerificationFailedError
from app.brain.voice.models import VoiceTurnResult
from app.brain.voice.server import handle_voice_turn_request
from app.brain.voice.speech_to_text import STTProvider
from app.brain.voice.state import get_voice_state, reset_voice_state
from app.brain.voice.text_to_speech import TTSProvider
from app.brain.voice.verification import VOICE_SAMPLE_RATE_HZ, VerificationProvider, average_embeddings, cosine_similarity


def _wav_bytes(*, sample_rate: int = VOICE_SAMPLE_RATE_HZ, num_samples: int = 1600) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * num_samples)
    return buffer.getvalue()


class _FakeSTTProvider(STTProvider):
    def __init__(self, *, transcript: str = "remember that the sky is blue") -> None:
        self.transcript = transcript
        self.calls = 0

    def transcribe(self, audio, sample_rate) -> str:
        self.calls += 1
        return self.transcript


class _FakeTTSProvider(TTSProvider):
    def __init__(self, *, wav_bytes: bytes | None = None) -> None:
        self.wav_bytes = wav_bytes if wav_bytes is not None else _wav_bytes()
        self.calls: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        return self.wav_bytes


class _FakeVerificationProvider(VerificationProvider):
    """Embeds by returning a fixed vector per caller-labeled "identity"; tests control who
    "sounds like the owner" by choosing which embedding a sample maps to via `embedding_for`."""

    def __init__(self, *, embedding_for=None) -> None:
        self._embedding_for = embedding_for or (lambda audio: [1.0, 0.0, 0.0])

    def embed(self, audio, sample_rate) -> list[float]:
        return self._embedding_for(audio)


def _owner_embedding(_audio) -> list[float]:
    return [1.0, 0.0, 0.0]


def _stranger_embedding(_audio) -> list[float]:
    return [0.0, 1.0, 0.0]


class VoiceRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = Path.cwd() / ".tmp-tests"
        self.temp_root.mkdir(exist_ok=True)
        self.profile_file = (self.temp_root / f"voice-profile-{uuid4().hex}.json").absolute()
        self.profile_patcher = patch("app.brain.voice.enrollment.get_profile_file_path", return_value=self.profile_file)
        self.profile_patcher.start()

        reset_runtime_config()
        reset_audit_log()
        reset_voice_state()
        set_runtime_config_value("voice_enabled", True)
        config = get_effective_runtime_config()
        config["location_shared_secret"] = "test-secret"
        replace_runtime_config(config)

        self.fake_stt = _FakeSTTProvider()
        self.fake_tts = _FakeTTSProvider()
        self.fake_verification = _FakeVerificationProvider(embedding_for=_owner_embedding)
        reset_voice_controller(stt_provider=self.fake_stt, tts_provider=self.fake_tts, verification_provider=self.fake_verification)

    def tearDown(self) -> None:
        self.profile_patcher.stop()
        reset_runtime_config()
        reset_audit_log()
        reset_voice_state()
        reset_voice_controller()
        if self.profile_file.exists():
            self.profile_file.unlink()

    def _enroll_owner(self) -> None:
        with patch("app.brain.voice.controller.record_from_microphone", return_value=[0.0] * 1600):
            for _ in range(enrollment.MIN_ENROLLMENT_SAMPLES):
                get_voice_controller().enroll_sample()

    # --- pure helpers -----------------------------------------------------

    def test_cosine_similarity_identical_vectors_is_one(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_cosine_similarity_orthogonal_vectors_is_zero(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_average_embeddings(self) -> None:
        self.assertEqual(average_embeddings([[1.0, 1.0], [3.0, 3.0]]), [2.0, 2.0])

    def test_wav_round_trip_decodes_expected_sample_count(self) -> None:
        samples, rate = wav_bytes_to_samples(_wav_bytes(num_samples=1600))
        self.assertEqual(rate, VOICE_SAMPLE_RATE_HZ)
        self.assertEqual(len(samples), 1600)

    def test_resample_audio_is_identity_when_rate_matches(self) -> None:
        samples = [0.1, 0.2, 0.3]
        self.assertEqual(resample_audio(samples, VOICE_SAMPLE_RATE_HZ, VOICE_SAMPLE_RATE_HZ), samples)

    # --- enrollment ---------------------------------------------------------

    def test_enrollment_requires_minimum_samples_before_finalizing(self) -> None:
        with patch("app.brain.voice.controller.record_from_microphone", return_value=[0.0] * 1600):
            for expected_count in range(1, enrollment.MIN_ENROLLMENT_SAMPLES):
                message = get_voice_controller().enroll_sample()
                self.assertIn(f"{expected_count} of {enrollment.MIN_ENROLLMENT_SAMPLES}", message)
                self.assertFalse(enrollment.is_enrolled())
            final_message = get_voice_controller().enroll_sample()
            self.assertIn("Enrollment complete", final_message)
            self.assertTrue(enrollment.is_enrolled())

    def test_cancel_enrollment_discards_pending_samples(self) -> None:
        with patch("app.brain.voice.controller.record_from_microphone", return_value=[0.0] * 1600):
            get_voice_controller().enroll_sample()
        self.assertEqual(enrollment.pending_sample_count(), 1)
        get_voice_controller().cancel_enrollment()
        self.assertEqual(enrollment.pending_sample_count(), 0)
        self.assertFalse(enrollment.is_enrolled())

    def test_forget_me_deletes_the_profile(self) -> None:
        self._enroll_owner()
        self.assertTrue(enrollment.is_enrolled())
        message = get_voice_controller().forget_me()
        self.assertIn("deleted", message)
        self.assertFalse(enrollment.is_enrolled())

    def test_forget_me_when_nothing_enrolled_is_safe(self) -> None:
        message = get_voice_controller().forget_me()
        self.assertIn("No voice enrollment", message)

    def test_enrollment_finalization_logs_similarity_diagnostics(self) -> None:
        # Regression coverage for the "voice talk always rejects the enrolled owner"
        # investigation: finalize_enrollment() must log a same-session self-similarity
        # baseline (pairwise among the banked samples, and each sample vs. their average)
        # so a later voice-talk rejection's similarity score has something concrete to be
        # compared against.
        with patch("app.brain.voice.controller.record_from_microphone", return_value=[0.0] * 1600):
            with self.assertLogs("app.brain.voice.enrollment", level="INFO") as logs:
                for _ in range(enrollment.MIN_ENROLLMENT_SAMPLES):
                    get_voice_controller().enroll_sample()
        joined = "\n".join(logs.output)
        self.assertIn("pairwise_similarity", joined)
        self.assertIn("sample_to_average_similarity", joined)

    # --- speaker verification gate -------------------------------------------

    def test_voice_turn_before_enrollment_raises_when_verification_required(self) -> None:
        set_runtime_config_value("voice_require_speaker_verification", True)
        with self.assertRaises(VoiceNotEnrolledError):
            get_voice_controller().handle_voice_turn([0.0] * 1600, route_text=lambda text: "reply")

    def test_voice_turn_rejects_non_owner_voice_and_produces_no_transcript_when_verification_required(self) -> None:
        set_runtime_config_value("voice_require_speaker_verification", True)
        self._enroll_owner()
        self.fake_verification._embedding_for = _stranger_embedding
        route_calls = []
        with self.assertRaises(VoiceVerificationFailedError):
            get_voice_controller().handle_voice_turn([0.0] * 1600, route_text=lambda text: route_calls.append(text) or "reply")
        self.assertEqual(route_calls, [])
        self.assertEqual(self.fake_stt.calls, 0)

    def test_voice_turn_accepts_owner_voice_when_verification_required(self) -> None:
        set_runtime_config_value("voice_require_speaker_verification", True)
        self._enroll_owner()
        route_calls = []
        result = get_voice_controller().handle_voice_turn([0.0] * 1600, route_text=lambda text: route_calls.append(text) or "Remembered.")
        self.assertIsInstance(result, VoiceTurnResult)
        self.assertEqual(route_calls, [self.fake_stt.transcript])
        self.assertEqual(result.reply_text, "Remembered.")
        self.assertEqual(result.reply_audio_wav, self.fake_tts.wav_bytes)

    # --- speaker verification is opt-in (owner's explicit request) ------------
    #
    # RFC-009 originally ran speaker verification unconditionally, as a hard boundary (see
    # docs/RFC-009_ON_DEMAND_VOICE_ASSISTANT.md). After extensive real-hardware testing
    # showed the enrolled owner's own genuine similarity score varying widely (0.23-0.68)
    # even with every mechanical cause (mic capture, stale enrollment, silence padding)
    # fixed, the owner explicitly asked for plain voice control without speaker
    # recognition. `voice_require_speaker_verification` defaults to False: any voice is
    # now transcribed and routed like typed input, and no enrollment is required at all.
    # The feature itself is not removed -- `config set voice_require_speaker_verification
    # true` restores the original hard-boundary behavior above.

    def test_voice_turn_skips_speaker_verification_by_default(self) -> None:
        self.fake_verification._embedding_for = _stranger_embedding  # would reject if checked
        route_calls = []
        result = get_voice_controller().handle_voice_turn([0.0] * 1600, route_text=lambda text: route_calls.append(text) or "ok")
        self.assertEqual(route_calls, [self.fake_stt.transcript])
        self.assertEqual(result.reply_text, "ok")

    def test_voice_turn_works_without_any_enrollment_by_default(self) -> None:
        self.assertFalse(enrollment.is_enrolled())
        result = get_voice_controller().handle_voice_turn([0.0] * 1600, route_text=lambda text: "ok")
        self.assertEqual(result.reply_text, "ok")

    def test_voice_status_reports_speaker_verification_requirement(self) -> None:
        self.assertIn("Speaker verification required: no", get_voice_controller().status_message())
        set_runtime_config_value("voice_require_speaker_verification", True)
        self.assertIn("Speaker verification required: yes", get_voice_controller().status_message())

    def test_voice_turn_when_disabled_raises(self) -> None:
        self._enroll_owner()
        set_runtime_config_value("voice_enabled", False)
        with self.assertRaises(VoiceDisabledError):
            get_voice_controller().handle_voice_turn([0.0] * 1600, route_text=lambda text: "reply")

    def test_voice_turn_with_empty_transcript_raises(self) -> None:
        self._enroll_owner()
        self.fake_stt.transcript = "   "
        with self.assertRaises(VoiceProviderError):
            get_voice_controller().handle_voice_turn([0.0] * 1600, route_text=lambda text: "reply")

    def test_voice_verification_logs_similarity_score_and_threshold(self) -> None:
        # Regression coverage for the "voice talk always rejects the enrolled owner"
        # investigation: every verification attempt must log the live/enrolled embedding
        # lengths, the enrolled sample count, the computed similarity, the configured
        # threshold, and the accept/reject outcome -- not just silently accept or raise.
        set_runtime_config_value("voice_require_speaker_verification", True)
        self._enroll_owner()
        with self.assertLogs("app.brain.voice.controller", level="INFO") as logs:
            get_voice_controller().handle_voice_turn([0.0] * 1600, route_text=lambda text: "reply")
        joined = "\n".join(logs.output)
        self.assertIn("similarity=", joined)
        self.assertIn("threshold=", joined)
        self.assertIn("result=accept", joined)

    def test_voice_verification_logs_reject_outcome_with_score(self) -> None:
        set_runtime_config_value("voice_require_speaker_verification", True)
        self._enroll_owner()
        self.fake_verification._embedding_for = _stranger_embedding
        with self.assertLogs("app.brain.voice.controller", level="INFO") as logs:
            with self.assertRaises(VoiceVerificationFailedError):
                get_voice_controller().handle_voice_turn([0.0] * 1600, route_text=lambda text: "reply")
        joined = "\n".join(logs.output)
        self.assertIn("result=reject", joined)

    def test_voice_talk_accepts_real_field_observed_similarity_score(self) -> None:
        # Regression test encoding an actual field observation from a real voice enroll +
        # voice talk run: the enrolled owner's own genuine voice produced a live embedding
        # with cosine similarity 0.7311 to the enrolled profile (logged as
        # "similarity=0.7311 threshold=0.7500 result=reject"). The original default
        # threshold (0.75) rejected this genuine attempt by a hair; the recalibrated default
        # (0.6) must accept it.
        import math

        set_runtime_config_value("voice_require_speaker_verification", True)
        self._enroll_owner()  # profile embedding is [1.0, 0.0, 0.0] -- see _owner_embedding
        target_similarity = 0.7311
        orthogonal_component = math.sqrt(1.0 - target_similarity**2)
        live_embedding = [target_similarity, orthogonal_component, 0.0]
        self.fake_verification._embedding_for = lambda _audio: live_embedding

        threshold = get_effective_runtime_config()["voice_verification_threshold"]
        self.assertLess(threshold, target_similarity, "default threshold regressed back above a real observed genuine-match score")

        result = get_voice_controller().handle_voice_turn([0.0] * 1600, route_text=lambda text: "ok")
        self.assertEqual(result.reply_text, "ok")

    def test_voice_verification_logs_embedding_dimension_mismatch(self) -> None:
        # A dimension mismatch makes cosine_similarity() silently return 0.0 with no other
        # symptom; this must be surfaced explicitly rather than presented as an ordinary
        # low-similarity rejection.
        set_runtime_config_value("voice_require_speaker_verification", True)
        self._enroll_owner()
        self.fake_verification._embedding_for = lambda _audio: [1.0, 0.0]  # enrolled profile has length 3
        with self.assertLogs("app.brain.voice.controller", level="ERROR") as logs:
            with self.assertRaises(VoiceVerificationFailedError):
                get_voice_controller().handle_voice_turn([0.0] * 1600, route_text=lambda text: "reply")
        self.assertIn("embedding-dimension mismatch", "\n".join(logs.output))

    def test_voice_turn_still_returns_text_reply_if_tts_fails(self) -> None:
        self._enroll_owner()

        class _BrokenTTS(TTSProvider):
            def synthesize(self, text: str) -> bytes:
                raise VoiceProviderError("voice model not configured")

        reset_voice_controller(stt_provider=self.fake_stt, tts_provider=_BrokenTTS(), verification_provider=self.fake_verification)
        result = get_voice_controller().handle_voice_turn([0.0] * 1600, route_text=lambda text: "Reply text.")
        self.assertEqual(result.reply_text, "Reply text.")
        self.assertEqual(result.reply_audio_wav, b"")

    # --- HTTP layer: POST /voice/turn -----------------------------------------

    def test_voice_turn_request_rejects_unauthenticated(self) -> None:
        status, body = handle_voice_turn_request(headers={}, raw_body=_wav_bytes())
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["result"], "unauthorized")

    def test_voice_turn_request_rejects_malformed_body(self) -> None:
        status, body = handle_voice_turn_request(headers={"Authorization": "Bearer test-secret"}, raw_body=b"not a wav file")
        self.assertEqual(status, 400)

    def test_voice_turn_request_end_to_end_with_owner_voice(self) -> None:
        self._enroll_owner()
        status, body = handle_voice_turn_request(headers={"Authorization": "Bearer test-secret"}, raw_body=_wav_bytes())
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["result"], "ok")
        self.assertEqual(payload["transcript"], self.fake_stt.transcript)
        self.assertTrue(payload["reply_audio_wav_base64"])
        self.assertEqual(base64.b64decode(payload["reply_audio_wav_base64"]), self.fake_tts.wav_bytes)

    def test_voice_turn_request_reports_no_action_for_unverified_voice(self) -> None:
        set_runtime_config_value("voice_require_speaker_verification", True)
        self._enroll_owner()
        self.fake_verification._embedding_for = _stranger_embedding
        status, body = handle_voice_turn_request(headers={"Authorization": "Bearer test-secret"}, raw_body=_wav_bytes())
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["result"], "no_action")
        self.assertNotIn("transcript", payload)

    # --- visible mic-active state --------------------------------------------

    def test_mic_active_flag_is_set_only_during_capture(self) -> None:
        import numpy as np
        import sounddevice as sd

        from app.brain.voice import audio_io

        seen_active = []

        def fake_rec(frame_count, samplerate, channels, dtype, device=None):
            seen_active.append(get_voice_state().mic_active)
            return np.zeros((frame_count, channels), dtype=dtype)

        with patch.object(sd, "rec", side_effect=fake_rec), patch.object(sd, "wait", return_value=None):
            self.assertFalse(get_voice_state().mic_active)
            audio_io.record_from_microphone(0.1)
            self.assertFalse(get_voice_state().mic_active)
        self.assertEqual(seen_active, [True])

    # --- native-format capture: multi-channel downmix + resample --------------

    def test_record_from_microphone_uses_configured_native_device_settings(self) -> None:
        import numpy as np
        import sounddevice as sd

        from app.brain.voice import audio_io

        set_runtime_config_value("voice_input_device", 1)
        set_runtime_config_value("voice_input_sample_rate", 44100)
        set_runtime_config_value("voice_input_channels", 4)
        seen_calls = []

        def fake_rec(frame_count, samplerate, channels, dtype, device=None):
            seen_calls.append({"frame_count": frame_count, "samplerate": samplerate, "channels": channels, "device": device})
            return np.zeros((frame_count, channels), dtype=dtype)

        with patch.object(sd, "rec", side_effect=fake_rec), patch.object(sd, "wait", return_value=None):
            audio_io.record_from_microphone(1.0)
        self.assertEqual(len(seen_calls), 1)
        call = seen_calls[0]
        self.assertEqual(call["device"], 1)
        self.assertEqual(call["samplerate"], 44100)
        self.assertEqual(call["channels"], 4)
        self.assertEqual(call["frame_count"], 44100)

    def test_record_from_microphone_downmixes_and_resamples_to_target_rate(self) -> None:
        import numpy as np
        import sounddevice as sd

        from app.brain.voice import audio_io

        set_runtime_config_value("voice_input_device", 1)
        set_runtime_config_value("voice_input_sample_rate", 44100)
        set_runtime_config_value("voice_input_channels", 4)

        def fake_rec(frame_count, samplerate, channels, dtype, device=None):
            # A constant, non-zero signal on every channel -- averaging channels and
            # resampling a constant signal should still be (approximately) that constant,
            # which is enough to prove the downmix and resample both actually ran rather
            # than the raw 4-channel 44.1kHz buffer being handed straight to the caller.
            return np.full((frame_count, channels), 0.25, dtype=dtype)

        with patch.object(sd, "rec", side_effect=fake_rec), patch.object(sd, "wait", return_value=None):
            samples = audio_io.record_from_microphone(1.0, sample_rate=VOICE_SAMPLE_RATE_HZ)
        # ~16000 samples for a 1s capture resampled to 16 kHz, not the native 44100.
        self.assertAlmostEqual(len(samples), VOICE_SAMPLE_RATE_HZ, delta=200)
        interior = samples[50:-50]
        self.assertTrue(all(abs(value - 0.25) < 0.05 for value in interior))

    def test_record_from_microphone_reports_device_on_failure(self) -> None:
        import sounddevice as sd

        from app.brain.voice import audio_io
        from app.brain.voice.errors import VoiceCaptureError

        set_runtime_config_value("voice_input_device", 12)

        def fake_rec(frame_count, samplerate, channels, dtype, device=None):
            raise OSError("Invalid number of channels")

        with patch.object(sd, "rec", side_effect=fake_rec):
            with self.assertRaises(VoiceCaptureError) as context:
                audio_io.record_from_microphone(1.0)
        self.assertIn("device 12", str(context.exception))

    # --- distinguishing "not installed" from other sounddevice import failures --------

    def test_record_from_microphone_reports_not_installed_when_module_missing(self) -> None:
        # Regression test for the same class of bug already found once in this codebase
        # (SpeechBrain's "Speaker embedding failed." masking a Windows symlink-privilege
        # error): a broad `except Exception` around `import sounddevice` used to report a
        # single hardcoded "not installed" message no matter what actually failed. A real
        # ModuleNotFoundError for the package itself must still get the actionable
        # `pip install` message.
        import sys

        from app.brain.voice import audio_io
        from app.brain.voice.errors import VoiceCaptureError

        with patch.dict(sys.modules, {"sounddevice": None}):
            with self.assertRaises(VoiceCaptureError) as context:
                audio_io.record_from_microphone(1.0)
        message = str(context.exception)
        self.assertIn("not installed", message)
        self.assertIn("pip install", message)

    def test_record_from_microphone_reports_real_error_for_non_missing_import_failure(self) -> None:
        # A `sounddevice` import can fail for reasons other than "package not installed" --
        # most commonly a missing PortAudio native library even when the package itself is
        # present. That real exception must be surfaced (and logged), not papered over with
        # the same "not installed" wording, which would send someone straight to a `pip
        # install` that can't fix the actual problem.
        import builtins

        from app.brain.voice import audio_io
        from app.brain.voice.errors import VoiceCaptureError

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "sounddevice":
                raise OSError("PortAudio library not found")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaises(VoiceCaptureError) as context:
                audio_io.record_from_microphone(1.0)
        message = str(context.exception)
        self.assertNotIn("not installed", message)
        self.assertIn("PortAudio library not found", message)
        self.assertIn("jarvis.log", message)


class VoiceRecordingDiagnosticsTests(unittest.TestCase):
    """RFC-009 instability investigation: real-hardware logs showed `voice talk` similarity
    swing 0.684 -> 0.373 -> 0.202 across consecutive attempts against the *same* enrolled
    owner, with `voice_verification_threshold` unchanged at 0.6. Per the explicit
    instruction accompanying that report, these are diagnostics-only tests -- they do not
    touch the threshold or disable/bypass verification -- covering the new per-recording
    logging that lets a real run's logs show which of the suspected causes (silence,
    clipping, per-channel imbalance, insufficient speech activity within the fixed capture
    window) actually explains a given attempt's low similarity. Because
    `record_from_microphone()` is shared by both `voice enroll` and `voice talk`, this
    logging automatically applies identically to both, which is itself the check for
    "do enrollment and verification use the same preprocessing".
    """

    def setUp(self) -> None:
        reset_runtime_config()
        set_runtime_config_value("voice_input_device", 1)
        set_runtime_config_value("voice_input_sample_rate", 44100)
        set_runtime_config_value("voice_input_channels", 4)

    def tearDown(self) -> None:
        reset_runtime_config()

    def test_record_from_microphone_logs_native_and_preprocessed_diagnostics(self) -> None:
        import numpy as np
        import sounddevice as sd

        from app.brain.voice import audio_io

        def fake_rec(frame_count, samplerate, channels, dtype, device=None):
            return np.full((frame_count, channels), 0.3, dtype=dtype)

        with patch.object(sd, "rec", side_effect=fake_rec), patch.object(sd, "wait", return_value=None):
            with self.assertLogs("app.brain.voice.audio_io", level="INFO") as logs:
                audio_io.record_from_microphone(1.0, sample_rate=VOICE_SAMPLE_RATE_HZ)

        joined = "\n".join(logs.output)
        self.assertIn("Voice recording diagnostics (native_capture):", joined)
        self.assertIn("Voice recording diagnostics (preprocessed):", joined)
        for field in ("shape=", "sample_rate=", "channels=", "duration=", "min=", "max=", "peak=", "rms=", "n_samples="):
            self.assertIn(field, joined)

    def test_record_from_microphone_warns_on_very_quiet_signal(self) -> None:
        import numpy as np
        import sounddevice as sd

        from app.brain.voice import audio_io

        def fake_rec(frame_count, samplerate, channels, dtype, device=None):
            return np.full((frame_count, channels), 0.0005, dtype=dtype)

        with patch.object(sd, "rec", side_effect=fake_rec), patch.object(sd, "wait", return_value=None):
            with self.assertLogs("app.brain.voice.audio_io", level="WARNING") as logs:
                audio_io.record_from_microphone(1.0, sample_rate=VOICE_SAMPLE_RATE_HZ)

        self.assertTrue(any("very quiet" in message for message in logs.output))

    def test_record_from_microphone_warns_on_clipping(self) -> None:
        import numpy as np
        import sounddevice as sd

        from app.brain.voice import audio_io

        def fake_rec(frame_count, samplerate, channels, dtype, device=None):
            return np.full((frame_count, channels), 0.999, dtype=dtype)

        with patch.object(sd, "rec", side_effect=fake_rec), patch.object(sd, "wait", return_value=None):
            with self.assertLogs("app.brain.voice.audio_io", level="WARNING") as logs:
                audio_io.record_from_microphone(1.0, sample_rate=VOICE_SAMPLE_RATE_HZ)

        self.assertTrue(any("clipping" in message for message in logs.output))

    def test_record_from_microphone_warns_on_uneven_channel_levels(self) -> None:
        import numpy as np
        import sounddevice as sd

        from app.brain.voice import audio_io

        def fake_rec(frame_count, samplerate, channels, dtype, device=None):
            recording = np.full((frame_count, channels), 0.3, dtype=dtype)
            recording[:, 0] = 0.0  # one channel picking up (near-)nothing this attempt
            return recording

        with patch.object(sd, "rec", side_effect=fake_rec), patch.object(sd, "wait", return_value=None):
            with self.assertLogs("app.brain.voice.audio_io", level="WARNING") as logs:
                audio_io.record_from_microphone(1.0, sample_rate=VOICE_SAMPLE_RATE_HZ)

        self.assertTrue(any("Uneven per-channel levels" in message for message in logs.output))

    def test_estimate_speech_activity_reports_leading_and_trailing_silence(self) -> None:
        from app.brain.voice.audio_io import _estimate_speech_activity

        sample_rate = 16000
        silence = [0.0] * (sample_rate // 2)  # 0.5s
        speech = [0.5] * sample_rate  # 1.0s of "loud" signal
        samples = silence + speech + silence

        active_ratio, leading_silence_s, trailing_silence_s = _estimate_speech_activity(samples, sample_rate)

        self.assertGreater(active_ratio, 0.3)
        self.assertLess(active_ratio, 1.0)
        self.assertAlmostEqual(leading_silence_s, 0.5, delta=0.05)
        self.assertAlmostEqual(trailing_silence_s, 0.5, delta=0.05)

    def test_estimate_speech_activity_handles_empty_and_fully_silent_input(self) -> None:
        from app.brain.voice.audio_io import _estimate_speech_activity

        self.assertEqual(_estimate_speech_activity([], 16000), (0.0, 0.0, 0.0))
        active_ratio, leading_silence_s, _trailing_silence_s = _estimate_speech_activity([0.0] * 16000, 16000)
        self.assertEqual(active_ratio, 0.0)
        self.assertAlmostEqual(leading_silence_s, 1.0, delta=0.01)


class VoiceSilenceTrimFixTests(unittest.TestCase):
    """Root-cause fix, confirmed by real-hardware logs gathered via the diagnostics above:
    every live `voice talk` recording had active_speech_ratio between 0.115 and 0.200 (i.e.
    80-88% silence/room noise inside the fixed 6-second capture window), with leading
    silence 0.87-1.2s and trailing silence 2.79-3.84s that varied unpredictably between
    attempts -- while clipping_ratio was 0.0 and per-channel peaks were even, ruling out a
    mic-level cause. `_trim_silence()` crops each recording down to its speech-bearing
    region (plus a small padding) before it reaches STT/speaker verification, so the
    encoder always sees a consistent, speech-dense clip instead of a different, unpredictable
    silence ratio every attempt. `voice_verification_threshold` is untouched (still 0.6) and
    verification is not disabled anywhere in this fix."""

    def setUp(self) -> None:
        reset_runtime_config()
        set_runtime_config_value("voice_input_device", 1)
        set_runtime_config_value("voice_input_sample_rate", 44100)
        set_runtime_config_value("voice_input_channels", 4)

    def tearDown(self) -> None:
        reset_runtime_config()

    def test_trim_silence_crops_leading_and_trailing_silence(self) -> None:
        from app.brain.voice.audio_io import _trim_silence

        sample_rate = 16000
        silence = [0.0] * sample_rate  # 1.0s
        speech = [0.3] * sample_rate  # 1.0s
        samples = silence + speech + silence  # 3.0s total, 1.0s of it real signal

        trimmed = _trim_silence(samples, sample_rate)

        # Well under the untrimmed 3.0s, but comfortably covers the 1.0s of speech plus
        # the padding on each side.
        self.assertLess(len(trimmed), len(samples))
        self.assertGreater(len(trimmed), sample_rate)
        self.assertLess(len(trimmed), int(1.6 * sample_rate))
        interior = trimmed[len(trimmed) // 4 : -(len(trimmed) // 4)]
        self.assertTrue(all(abs(value - 0.3) < 1e-6 for value in interior))

    def test_trim_silence_leaves_fully_active_signal_untouched(self) -> None:
        from app.brain.voice.audio_io import _trim_silence

        samples = [0.25] * 16000
        self.assertEqual(_trim_silence(samples, 16000), samples)

    def test_trim_silence_falls_back_to_untouched_on_fully_silent_input(self) -> None:
        from app.brain.voice.audio_io import _trim_silence

        samples = [0.0] * 16000
        self.assertEqual(_trim_silence(samples, 16000), samples)

    def test_trim_silence_falls_back_when_result_would_be_too_short(self) -> None:
        from app.brain.voice.audio_io import _trim_silence

        sample_rate = 16000
        # A single very brief blip of signal surrounded by silence: trimming down to just
        # that blip (plus padding) would leave well under MIN_TRIMMED_SECONDS, so the
        # original (untrimmed) audio must be returned instead of an almost-empty clip.
        samples = [0.0] * sample_rate + [0.3] * int(0.05 * sample_rate) + [0.0] * sample_rate
        self.assertEqual(_trim_silence(samples, sample_rate), samples)

    def test_record_from_microphone_trims_silence_surrounding_speech(self) -> None:
        import numpy as np
        import sounddevice as sd

        from app.brain.voice import audio_io

        def fake_rec(frame_count, samplerate, channels, dtype, device=None):
            # 6s native capture: 2s silence, 2s of signal, 2s silence -- matching the shape
            # of the real field recordings that exposed this bug (mostly silence, a real
            # utterance somewhere in the middle, differing start/end silence each attempt).
            segment = frame_count // 3
            recording = np.zeros((frame_count, channels), dtype=dtype)
            recording[segment : 2 * segment, :] = 0.3
            return recording

        with patch.object(sd, "rec", side_effect=fake_rec), patch.object(sd, "wait", return_value=None):
            with self.assertLogs("app.brain.voice.audio_io", level="INFO") as logs:
                samples = audio_io.record_from_microphone(6.0, sample_rate=VOICE_SAMPLE_RATE_HZ)

        full_length = int(6.0 * VOICE_SAMPLE_RATE_HZ)
        # The trimmed recording must be meaningfully shorter than the untrimmed 6s window --
        # this is the actual fix, not just a diagnostic -- while still comfortably covering
        # the ~2s of real signal plus padding.
        self.assertLess(len(samples), full_length // 2)
        self.assertGreater(len(samples), int(1.5 * VOICE_SAMPLE_RATE_HZ))
        joined = "\n".join(logs.output)
        self.assertIn("Trimmed", joined)
        self.assertIn("Voice recording diagnostics (trimmed):", joined)


class VoiceControllerProviderCachingTests(unittest.TestCase):
    """Regression coverage for the "voice talk always rejects the enrolled owner"
    investigation: VoiceController.verification_provider() used to construct a brand-new
    SpeechBrainVerificationProvider on every single call, reloading the whole ECAPA model
    from disk before every enroll_sample()/handle_voice_turn(). It must now be built once
    and reused. Constructing SpeechBrainVerificationProvider() itself does not import
    speechbrain (that happens lazily inside _classifier_instance()), so this needs no real
    or stubbed speechbrain install."""

    def test_verification_provider_is_cached_not_recreated_per_call(self) -> None:
        from app.brain.voice.controller import VoiceController

        controller = VoiceController()  # no override -> exercises the real caching path
        first = controller.verification_provider()
        second = controller.verification_provider()
        third = controller.verification_provider()
        self.assertIs(first, second)
        self.assertIs(second, third)

    def test_verification_override_is_not_shadowed_by_caching(self) -> None:
        from app.brain.voice.controller import VoiceController

        override = _FakeVerificationProvider()
        controller = VoiceController(verification_provider=override)
        self.assertIs(controller.verification_provider(), override)
        self.assertIs(controller.verification_provider(), override)


class SpeechBrainModelLoadingTests(unittest.TestCase):
    """Regression coverage for a real Windows bug found via voice enroll's own traceback
    logging: SpeechBrain's default fetch strategy (LocalStrategy.SYMLINK) symlinks each
    downloaded model file into `savedir`, and `dst.symlink_to(src)` raises `OSError:
    [WinError 1314] A required privilege is not held by the client` on a plain Windows
    account (no Developer Mode, not running as administrator) -- with well-formed mono
    16 kHz float32 audio already confirmed reaching this point, so this was never an audio
    problem. speechbrain/torch are stubbed via sys.modules rather than requiring the real
    (heavy) packages, matching the audio backend already being optional for this suite."""

    def test_classifier_instance_uses_copy_skip_cache_not_symlink(self) -> None:
        import sys
        import types

        from app.brain.voice.verification import SpeechBrainVerificationProvider

        class _FakeLocalStrategy:
            SYMLINK = "SYMLINK"
            COPY = "COPY"
            COPY_SKIP_CACHE = "COPY_SKIP_CACHE"
            NO_LINK = "NO_LINK"

        fake_fetching_module = types.ModuleType("speechbrain.utils.fetching")
        fake_fetching_module.LocalStrategy = _FakeLocalStrategy

        captured_kwargs: dict = {}

        class _FakeEncoderClassifier:
            @classmethod
            def from_hparams(cls, **kwargs):
                captured_kwargs.update(kwargs)
                return object()

        fake_speaker_module = types.ModuleType("speechbrain.inference.speaker")
        fake_speaker_module.EncoderClassifier = _FakeEncoderClassifier

        with patch.dict(sys.modules, {
            "speechbrain.inference.speaker": fake_speaker_module,
            "speechbrain.utils.fetching": fake_fetching_module,
        }):
            provider = SpeechBrainVerificationProvider(download_dir="unused-for-this-test")
            provider._classifier_instance()

        self.assertEqual(captured_kwargs.get("local_strategy"), _FakeLocalStrategy.COPY_SKIP_CACHE)
        self.assertNotEqual(captured_kwargs.get("local_strategy"), _FakeLocalStrategy.SYMLINK)

    def test_clear_broken_symlinks_removes_only_dangling_entries(self) -> None:
        from app.brain.voice.verification import _clear_broken_symlinks

        temp_root = Path.cwd() / ".tmp-tests"
        temp_root.mkdir(exist_ok=True)
        directory = temp_root / f"models-{uuid4().hex}"
        directory.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, ignore_errors=True))

        real_file = directory / "real.txt"
        real_file.write_text("data", encoding="utf-8")
        working_link_target = directory / "real.txt"
        working_link = directory / "working_link"
        working_link.symlink_to(working_link_target)
        broken_link = directory / "broken_link"
        broken_link.symlink_to(directory / "does-not-exist.txt")
        self.assertTrue(broken_link.is_symlink())
        self.assertFalse(broken_link.exists())  # dangling: target doesn't resolve

        _clear_broken_symlinks(str(directory))

        self.assertTrue(real_file.exists())
        self.assertTrue(working_link.is_symlink())
        self.assertFalse(broken_link.is_symlink())
        self.assertFalse(broken_link.exists())

    def test_clear_broken_symlinks_on_missing_directory_is_a_noop(self) -> None:
        from app.brain.voice.verification import _clear_broken_symlinks

        # Must not raise even though the directory doesn't exist yet (e.g. first-ever run,
        # before anything has attempted to download the model).
        _clear_broken_symlinks(str(Path.cwd() / ".tmp-tests" / f"missing-{uuid4().hex}"))

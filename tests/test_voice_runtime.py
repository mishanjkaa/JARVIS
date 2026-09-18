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

    # --- speaker verification gate -------------------------------------------

    def test_voice_turn_before_enrollment_raises(self) -> None:
        with self.assertRaises(VoiceNotEnrolledError):
            get_voice_controller().handle_voice_turn([0.0] * 1600, route_text=lambda text: "reply")

    def test_voice_turn_rejects_non_owner_voice_and_produces_no_transcript(self) -> None:
        self._enroll_owner()
        self.fake_verification._embedding_for = _stranger_embedding
        route_calls = []
        with self.assertRaises(VoiceVerificationFailedError):
            get_voice_controller().handle_voice_turn([0.0] * 1600, route_text=lambda text: route_calls.append(text) or "reply")
        self.assertEqual(route_calls, [])
        self.assertEqual(self.fake_stt.calls, 0)

    def test_voice_turn_accepts_owner_voice(self) -> None:
        self._enroll_owner()
        route_calls = []
        result = get_voice_controller().handle_voice_turn([0.0] * 1600, route_text=lambda text: route_calls.append(text) or "Remembered.")
        self.assertIsInstance(result, VoiceTurnResult)
        self.assertEqual(route_calls, [self.fake_stt.transcript])
        self.assertEqual(result.reply_text, "Remembered.")
        self.assertEqual(result.reply_audio_wav, self.fake_tts.wav_bytes)

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

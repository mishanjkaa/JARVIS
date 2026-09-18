from __future__ import annotations

import io
import wave

from app.brain.voice.errors import VoiceProviderError

# Piper is a real (heavy) dependency once TTS is actually used, but importing it costs real
# startup time -- it is imported lazily inside PiperTTSProvider methods, not at module
# import time. voice_tts_voice must point at a downloaded Piper .onnx voice model; JARVIS
# does not download one automatically (same convention as Ollama models, which the operator
# pulls themselves).


class TTSProvider:
    def synthesize(self, text: str) -> bytes:
        """Return a WAV file's bytes for the spoken reply."""
        raise NotImplementedError


class PiperTTSProvider(TTSProvider):
    def __init__(self, *, voice_model_path: str) -> None:
        self.voice_model_path = voice_model_path
        self._voice = None

    def _voice_instance(self):
        if self._voice is None:
            if not self.voice_model_path:
                raise VoiceProviderError("voice_tts_voice is not configured with a Piper voice model path.")
            try:
                from piper import PiperVoice
            except Exception as error:
                raise VoiceProviderError("The text-to-speech voice could not be loaded.") from error
            try:
                self._voice = PiperVoice.load(self.voice_model_path)
            except (OSError, ValueError) as error:
                raise VoiceProviderError(f"Could not load Piper voice model '{self.voice_model_path}'.") from error
        return self._voice

    def synthesize(self, text: str) -> bytes:
        if not text or not text.strip():
            raise VoiceProviderError("Nothing to say.")
        try:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                self._voice_instance().synthesize_wav(text, wav_file)
            return buffer.getvalue()
        except VoiceProviderError:
            raise
        except Exception as error:
            raise VoiceProviderError("Text-to-speech synthesis failed.") from error

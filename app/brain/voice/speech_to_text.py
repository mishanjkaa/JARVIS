from __future__ import annotations

from app.brain.voice.errors import VoiceProviderError
from app.brain.voice.verification import VOICE_SAMPLE_RATE_HZ

# faster-whisper is a real (heavy) dependency once STT is actually used, but importing it
# costs real startup time -- it is imported lazily inside FasterWhisperSTTProvider methods,
# not at module import time.
DEFAULT_STT_MODEL = "small"


class STTProvider:
    def transcribe(self, audio: "list[float]", sample_rate: int) -> str:
        raise NotImplementedError


class FasterWhisperSTTProvider(STTProvider):
    def __init__(self, *, model_size: str = DEFAULT_STT_MODEL, device: str = "cpu", compute_type: str = "int8") -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _model_instance(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except Exception as error:
                raise VoiceProviderError("The speech-to-text model could not be loaded.") from error
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        return self._model

    def transcribe(self, audio: "list[float]", sample_rate: int) -> str:
        if sample_rate != VOICE_SAMPLE_RATE_HZ:
            raise VoiceProviderError(f"Speech-to-text requires {VOICE_SAMPLE_RATE_HZ} Hz audio.")
        try:
            import numpy as np

            waveform = np.asarray(audio, dtype=np.float32)
            segments, _info = self._model_instance().transcribe(waveform, language=None)
            return " ".join(segment.text.strip() for segment in segments).strip()
        except VoiceProviderError:
            raise
        except Exception as error:
            raise VoiceProviderError("Speech-to-text transcription failed.") from error

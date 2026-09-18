from __future__ import annotations

import math

from app.brain.voice.errors import VoiceProviderError

VOICE_SAMPLE_RATE_HZ = 16000

# speechbrain/torch are real (heavy) dependencies once voice is actually used, but importing
# them costs real startup time -- they are imported lazily inside
# SpeechBrainVerificationProvider methods, not at module import time, so importing this
# module (or constructing a fake VerificationProvider for tests) never pays that cost.
_DEFAULT_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


class VerificationProvider:
    def embed(self, audio: "list[float]", sample_rate: int) -> list[float]:
        raise NotImplementedError


class SpeechBrainVerificationProvider(VerificationProvider):
    def __init__(self, *, model_source: str = _DEFAULT_MODEL_SOURCE, download_dir: str = "data/models/spkrec-ecapa-voxceleb") -> None:
        self.model_source = model_source
        self.download_dir = download_dir
        self._classifier = None

    def _classifier_instance(self):
        if self._classifier is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except Exception as error:
                raise VoiceProviderError("The speaker-verification model could not be loaded.") from error
            self._classifier = EncoderClassifier.from_hparams(source=self.model_source, savedir=self.download_dir)
        return self._classifier

    def embed(self, audio: "list[float]", sample_rate: int) -> list[float]:
        if sample_rate != VOICE_SAMPLE_RATE_HZ:
            raise VoiceProviderError(f"Speaker verification requires {VOICE_SAMPLE_RATE_HZ} Hz audio.")
        try:
            import torch

            tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            embedding = self._classifier_instance().encode_batch(tensor)
            return embedding.squeeze().detach().cpu().tolist()
        except VoiceProviderError:
            raise
        except Exception as error:
            raise VoiceProviderError("Speaker embedding failed.") from error


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []
    length = len(embeddings[0])
    sums = [0.0] * length
    for embedding in embeddings:
        for index, value in enumerate(embedding):
            sums[index] += value
    return [value / len(embeddings) for value in sums]

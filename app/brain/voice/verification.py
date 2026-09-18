from __future__ import annotations

import logging
import math

from app.brain.voice.errors import VoiceProviderError

logger = logging.getLogger(__name__)

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
                logger.exception("Could not import speechbrain.inference.speaker.EncoderClassifier.")
                raise VoiceProviderError("The speaker-verification model could not be loaded.") from error
            # DIAGNOSTIC (RFC-009 mic-pipeline investigation): from_hparams() was previously
            # called *outside* any try/except in this method -- only the import line above
            # was guarded. On first use it downloads the model from Hugging Face Hub, so any
            # failure here (network/proxy block, Hub outage, a Windows symlink/cache-
            # permission issue inside huggingface_hub, a corrupted partial download) escaped
            # this function entirely and was caught by embed()'s own broad `except Exception`
            # below, which reported it as the exact same generic "Speaker embedding failed."
            # message a bad *audio* tensor would produce. Wrapping it here, with its own
            # distinct message and a full logged traceback, is what lets model-loading
            # failures be told apart from an audio-format/content problem in encode_batch.
            try:
                self._classifier = EncoderClassifier.from_hparams(source=self.model_source, savedir=self.download_dir)
            except Exception as error:
                logger.exception(
                    "Could not download/load the speaker-verification model '%s' into '%s'.",
                    self.model_source, self.download_dir,
                )
                raise VoiceProviderError("The speaker-verification model could not be downloaded or loaded.") from error
        return self._classifier

    def embed(self, audio: "list[float]", sample_rate: int) -> list[float]:
        if sample_rate != VOICE_SAMPLE_RATE_HZ:
            raise VoiceProviderError(f"Speaker verification requires {VOICE_SAMPLE_RATE_HZ} Hz audio.")
        try:
            import torch

            tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

            # DIAGNOSTIC (RFC-009 mic-pipeline investigation): log exactly what is about to
            # be handed to SpeechBrain, immediately before the call, so a captured-audio
            # format/content problem (wrong shape, wrong dtype, a near-silent or NaN/Inf
            # signal) is visible in the log rather than indistinguishable from a
            # SpeechBrain/model-loading failure behind the same "Speaker embedding failed."
            # message.
            flat = tensor.reshape(-1)
            n_samples = int(flat.numel())
            if n_samples:
                sample_min = float(flat.min())
                sample_max = float(flat.max())
                sample_rms = float(torch.sqrt(torch.mean(flat * flat)))
                has_nan = bool(torch.isnan(flat).any())
                has_inf = bool(torch.isinf(flat).any())
            else:
                sample_min = sample_max = sample_rms = float("nan")
                has_nan = has_inf = False
            logger.info(
                "Speaker embedding input: shape=%s dtype=%s sample_rate=%d n_samples=%d "
                "min=%.6f max=%.6f rms=%.6f has_nan=%s has_inf=%s",
                tuple(tensor.shape), tensor.dtype, sample_rate, n_samples,
                sample_min, sample_max, sample_rms, has_nan, has_inf,
            )

            classifier = self._classifier_instance()
            embedding = classifier.encode_batch(tensor)
            return embedding.squeeze().detach().cpu().tolist()
        except VoiceProviderError:
            raise
        except Exception as error:
            logger.exception(
                "Speaker embedding failed (sample_rate=%d, n_samples=%d).",
                sample_rate, len(audio) if audio else 0,
            )
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

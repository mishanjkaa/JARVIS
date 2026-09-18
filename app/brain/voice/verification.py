from __future__ import annotations

import logging
import math
from pathlib import Path

from app.brain.voice.errors import VoiceProviderError

logger = logging.getLogger(__name__)

VOICE_SAMPLE_RATE_HZ = 16000

# speechbrain/torch are real (heavy) dependencies once voice is actually used, but importing
# them costs real startup time -- they are imported lazily inside
# SpeechBrainVerificationProvider methods, not at module import time, so importing this
# module (or constructing a fake VerificationProvider for tests) never pays that cost.
_DEFAULT_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


def _clear_broken_symlinks(directory: str) -> None:
    """Remove any dangling symlink left behind in `directory` by an earlier model-loading
    attempt that used SpeechBrain's default `LocalStrategy.SYMLINK` and failed partway
    through -- e.g. Windows raising WinError 1314 ("A required privilege is not held by the
    client") for `dst.symlink_to(src)` on a plain account with neither Developer Mode nor
    administrator rights. `LocalStrategy.COPY_SKIP_CACHE` (used below) never creates a
    symlink itself, but a stale broken one left over from a *previous* run, at the exact
    path a fresh download needs to write to, could otherwise still be sitting there. A real
    file, a working symlink, or a missing directory are all left untouched -- this only ever
    removes an entry that is a symlink AND does not resolve to anything."""
    path = Path(directory)
    if not path.is_dir():
        return
    for entry in path.iterdir():
        if entry.is_symlink() and not entry.exists():
            try:
                entry.unlink()
                logger.info("Removed a dangling symlink left over from a previous model-loading attempt: %s", entry)
            except OSError:
                logger.exception("Could not remove dangling symlink %s", entry)


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
                from speechbrain.utils.fetching import LocalStrategy
            except Exception as error:
                logger.exception("Could not import speechbrain.inference.speaker.EncoderClassifier.")
                raise VoiceProviderError("The speaker-verification model could not be loaded.") from error
            # Root cause found via the diagnostic logging below (RFC-009 mic-pipeline
            # investigation): SpeechBrain's `fetch()` defaults to `LocalStrategy.SYMLINK`,
            # which symlinks each downloaded model file from Hugging Face's local cache into
            # `savedir`. On a plain Windows account (no Developer Mode, not running as
            # administrator) `dst.symlink_to(src)` raises `OSError: [WinError 1314] A
            # required privilege is not held by the client` -- confirmed from a real
            # `voice enroll` traceback, with well-formed mono 16 kHz float32 audio already
            # having reached this point (has_nan=False, has_inf=False, real signal), so the
            # audio pipeline was never the problem. `LocalStrategy.COPY_SKIP_CACHE` downloads
            # straight into `savedir` as a normal file via `huggingface_hub`'s own
            # `local_dir=` download path, which never creates a symlink -- no Developer Mode,
            # no administrator rights, and no change to Windows' security settings needed.
            _clear_broken_symlinks(self.download_dir)
            try:
                self._classifier = EncoderClassifier.from_hparams(
                    source=self.model_source,
                    savedir=self.download_dir,
                    local_strategy=LocalStrategy.COPY_SKIP_CACHE,
                )
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

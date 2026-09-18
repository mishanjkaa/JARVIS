from __future__ import annotations

import io
import wave

from app.brain.voice.errors import VoiceCaptureError
from app.brain.voice.state import get_voice_state
from app.brain.voice.verification import VOICE_SAMPLE_RATE_HZ

# sounddevice/numpy are imported lazily inside these functions, not at module import time,
# so this module can be imported (and its mic_active bookkeeping tested) without the audio
# backend actually being installed/available -- relevant for CI machines with no audio
# hardware at all.
MAX_PUSH_TO_TALK_SECONDS = 15.0


def record_from_microphone(duration_seconds: float, *, sample_rate: int = VOICE_SAMPLE_RATE_HZ) -> list[float]:
    """Local PC push-to-talk capture. Sets the visible mic-active indicator for exactly the
    duration audio is actively being captured, per RFC-009's 'Visible state' requirement --
    never a window where JARVIS is listening without something reflecting it in
    `voice status`."""
    duration_seconds = min(max(duration_seconds, 0.1), MAX_PUSH_TO_TALK_SECONDS)
    state = get_voice_state()
    try:
        import numpy as np
        import sounddevice as sd
    except Exception as error:
        raise VoiceCaptureError("Local microphone capture is unavailable (sounddevice not installed).") from error

    with state.lock:
        state.mic_active = True
    try:
        frame_count = int(duration_seconds * sample_rate)
        recording = sd.rec(frame_count, samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
    except Exception as error:
        raise VoiceCaptureError("Microphone recording failed.") from error
    finally:
        with state.lock:
            state.mic_active = False
    return np.asarray(recording).reshape(-1).tolist()


def play_audio_wav(wav_bytes: bytes) -> None:
    try:
        import sounddevice as sd
    except Exception as error:
        raise VoiceCaptureError("Local speaker playback is unavailable (sounddevice not installed).") from error
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            frame_rate = wav_file.getframerate()
        import numpy as np

        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sample_width, np.int16)
        samples = np.frombuffer(frames, dtype=dtype)
        if channels > 1:
            samples = samples.reshape(-1, channels)
        sd.play(samples, samplerate=frame_rate)
        sd.wait()
    except Exception as error:
        raise VoiceCaptureError("Speaker playback failed.") from error


def wav_bytes_to_samples(wav_bytes: bytes) -> tuple[list[float], int]:
    """Decode a WAV blob (e.g. one uploaded by the phone client) into mono float32 samples
    plus its sample rate, without any transcoding dependency -- callers on the audio-blob
    side (the phone PWA, or a local recording) are expected to produce PCM WAV, not a
    compressed format, precisely to avoid needing an ffmpeg-class dependency here."""
    try:
        import numpy as np

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            frame_rate = wav_file.getframerate()
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sample_width)
        if dtype is None:
            raise VoiceCaptureError("Unsupported WAV sample width.")
        samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        if dtype == np.int16:
            samples /= 32768.0
        elif dtype == np.int32:
            samples /= 2147483648.0
        elif dtype == np.uint8:
            samples = (samples - 128.0) / 128.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return samples.tolist(), frame_rate
    except VoiceCaptureError:
        raise
    except Exception as error:
        raise VoiceCaptureError("Could not decode WAV audio.") from error


def resample_audio(samples: list[float], source_rate: int, target_rate: int = VOICE_SAMPLE_RATE_HZ) -> list[float]:
    """Resample to the rate STT/verification require. A browser or phone recording is not
    guaranteed to already be at that rate. Uses torchaudio (already a dependency for
    speechbrain) rather than adding a dedicated audio-resampling dependency."""
    if source_rate == target_rate:
        return samples
    try:
        import torch
        import torchaudio

        waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
        resampled = torchaudio.functional.resample(waveform, source_rate, target_rate)
        return resampled.squeeze(0).tolist()
    except Exception as error:
        raise VoiceCaptureError("Could not resample audio.") from error

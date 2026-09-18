from __future__ import annotations

import io
import logging
import wave

from app.brain.configuration.runtime_config import get_effective_runtime_config
from app.brain.voice.errors import VoiceCaptureError
from app.brain.voice.state import get_voice_state
from app.brain.voice.verification import VOICE_SAMPLE_RATE_HZ

logger = logging.getLogger(__name__)

# sounddevice/numpy are imported lazily inside these functions, not at module import time,
# so this module can be imported (and its mic_active bookkeeping tested) without the audio
# backend actually being installed/available -- relevant for CI machines with no audio
# hardware at all.
MAX_PUSH_TO_TALK_SECONDS = 15.0

# RFC-009 hardware note (found via real-device diagnostics on a Windows 11 target machine):
# a laptop's built-in "Microphone Array" typically exposes *two* PortAudio devices for the
# same physical hardware -- a plain MME/DirectSound-style index and a separate WASAPI index.
# Asking PortAudio to open that device directly in the 1-channel/16 kHz format STT and
# speaker verification want (the previous behaviour here) silently "succeeds" but returns
# near-silence (amplitude on the order of 1e-4) on the plain index, because the array mic's
# driver only does its real signal processing in its native multi-channel format; the WASAPI
# index for the same hardware instead refuses those parameters outright ("Invalid number of
# channels" / "Incompatible host API specific stream info"). Recording at the device's own
# native rate/channel count (confirmed on that machine to return real signal, e.g. RMS
# ~0.047 instead of ~0.0002) and doing the channel-downmix and 16 kHz resample ourselves in
# software avoids both failure modes. Hence the device/rate/channels below are configurable
# (a different machine's mic may have different native values or device index), but default
# to the values confirmed to work: device 1, 44100 Hz, 4 channels -- never the WASAPI-variant
# index of the same device.
DEFAULT_INPUT_DEVICE = 1
DEFAULT_INPUT_SAMPLE_RATE = 44100
DEFAULT_INPUT_CHANNELS = 4


def _input_device_settings() -> tuple[int, int, int]:
    config = get_effective_runtime_config()
    device = int(config.get("voice_input_device", DEFAULT_INPUT_DEVICE))
    native_sample_rate = int(config.get("voice_input_sample_rate", DEFAULT_INPUT_SAMPLE_RATE))
    channels = int(config.get("voice_input_channels", DEFAULT_INPUT_CHANNELS))
    return device, native_sample_rate, channels


def _downmix_to_mono(recording, channels: int) -> list[float]:
    import numpy as np

    samples = np.asarray(recording, dtype=np.float32).reshape(-1, channels)
    if channels > 1:
        samples = samples.mean(axis=1)
    else:
        samples = samples.reshape(-1)
    return samples.tolist()


def record_from_microphone(duration_seconds: float, *, sample_rate: int = VOICE_SAMPLE_RATE_HZ) -> list[float]:
    """Local PC push-to-talk capture. Sets the visible mic-active indicator for exactly the
    duration audio is actively being captured, per RFC-009's 'Visible state' requirement --
    never a window where JARVIS is listening without something reflecting it in
    `voice status`.

    Always records at the configured input device's own native sample rate/channel count
    (see the hardware note above) and only then downmixes to mono and resamples to
    `sample_rate` (16 kHz by default, what STT/speaker verification require) in software --
    never by asking PortAudio to capture directly in that target format, which silently
    produces near-silent audio on at least one real target device."""
    duration_seconds = min(max(duration_seconds, 0.1), MAX_PUSH_TO_TALK_SECONDS)
    state = get_voice_state()
    try:
        import sounddevice as sd  # noqa: F401  (import-availability probe)
    except Exception as error:
        raise VoiceCaptureError("Local microphone capture is unavailable (sounddevice not installed).") from error

    device, native_sample_rate, channels = _input_device_settings()
    frame_count = int(duration_seconds * native_sample_rate)
    logger.info(
        "Recording %.1fs of voice input from device=%s (native %d Hz, %d channel(s))",
        duration_seconds, device, native_sample_rate, channels,
    )

    with state.lock:
        state.mic_active = True
    try:
        recording = sd.rec(frame_count, samplerate=native_sample_rate, channels=channels, dtype="float32", device=device)
        sd.wait()
    except Exception as error:
        logger.error(
            "Microphone recording failed on device=%s (native %d Hz, %d channel(s)): %s",
            device, native_sample_rate, channels, error,
        )
        raise VoiceCaptureError(
            f"Microphone recording failed on voice input device {device} "
            f"({native_sample_rate} Hz, {channels} channel(s)). If this device only exposes "
            "a WASAPI variant at this index, configure 'voice_input_device' to that "
            "device's plain (non-WASAPI) index instead."
        ) from error
    finally:
        with state.lock:
            state.mic_active = False

    mono_native = _downmix_to_mono(recording, channels)
    return resample_audio(mono_native, native_sample_rate, sample_rate)


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

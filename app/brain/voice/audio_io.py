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


# RFC-009 verification-instability investigation (real-hardware logs showed
# similarity 0.684 -> 0.373 -> 0.202 across consecutive `voice talk` attempts against
# the *same* enrolled owner, with `voice_verification_threshold` unchanged at 0.6).
# These helpers add the diagnostic evidence needed to find the real cause -- per the
# explicit instruction that accompanied that report -- without touching the threshold,
# without disabling verification, and without altering the already-fixed native-format
# capture/downmix/resample pipeline above. Because `record_from_microphone()` is the one
# function used by both `voice enroll` and `voice talk`, wiring these into it produces
# directly comparable log lines for enrollment and live-verification recordings alike.
def _compute_window_activity(
    arr, sample_rate: int, *, window_ms: float = 30.0, threshold_ratio: float = 0.1
) -> tuple[list[bool], int]:
    """Shared sliding-window energy-based activity detector underlying both
    `_estimate_speech_activity()` (diagnostics) and `_trim_silence()` (the fix below) --
    kept as one function so the silence boundaries reported in the logs always match the
    boundaries actually used to trim audio. A window is "active" when its RMS is at least
    `threshold_ratio` of the recording's own peak amplitude. Returns an empty list when
    there is nothing to analyze, and a list of all-`False` (with the recording's actual
    window size) when the recording has no signal at all (peak == 0)."""
    import numpy as np

    if sample_rate <= 0 or arr.size == 0:
        return [], 0

    window_size = max(1, int(sample_rate * window_ms / 1000.0))
    n_windows = int(np.ceil(arr.size / window_size))
    if n_windows == 0:
        return [], window_size

    peak = float(np.abs(arr).max())
    if peak <= 0.0:
        return [False] * n_windows, window_size

    threshold = peak * threshold_ratio
    activity = []
    for i in range(n_windows):
        window = arr[i * window_size : (i + 1) * window_size]
        window_rms = float(np.sqrt(np.mean(window.astype(np.float64) ** 2))) if window.size else 0.0
        activity.append(window_rms >= threshold)
    return activity, window_size


def _estimate_speech_activity(
    samples: list[float], sample_rate: int, *, window_ms: float = 30.0, threshold_ratio: float = 0.1
) -> tuple[float, float, float]:
    """Dependency-free energy-based voice-activity estimate (no webrtcvad -- already
    rejected earlier in this project for its Windows compiler requirement). Splits the
    signal into `window_ms` windows, calls a window "active" when its RMS exceeds
    `threshold_ratio` of the recording's own peak amplitude, and reports the fraction of
    active windows plus how much leading/trailing silence surrounds the speech. This is a
    coarse diagnostic signal, not a production VAD: its purpose here is to show whether a
    fixed-duration recording window is capturing wildly different ratios of real speech to
    silence/room noise between consecutive attempts."""
    import numpy as np

    if sample_rate <= 0 or not samples:
        return 0.0, 0.0, 0.0

    arr = np.asarray(samples, dtype=np.float32)
    activity, window_size = _compute_window_activity(arr, sample_rate, window_ms=window_ms, threshold_ratio=threshold_ratio)
    if not activity:
        return 0.0, 0.0, 0.0
    if not any(activity):
        return 0.0, len(arr) / sample_rate, 0.0

    active_ratio = sum(activity) / len(activity)

    leading_silence_windows = 0
    for is_active in activity:
        if is_active:
            break
        leading_silence_windows += 1

    trailing_silence_windows = 0
    for is_active in reversed(activity):
        if is_active:
            break
        trailing_silence_windows += 1

    window_seconds = window_size / sample_rate
    return active_ratio, leading_silence_windows * window_seconds, trailing_silence_windows * window_seconds


# RFC-009 verification-instability root cause (confirmed on real hardware after the
# diagnostics above were deployed): every live `voice talk` recording showed
# active_speech_ratio between 0.115 and 0.200 -- i.e. 80-88% of the fixed 6-second capture
# window was silence/room noise, not speech -- with 0.87-1.2s of leading silence and
# 2.79-3.84s of trailing silence that varied unpredictably between attempts. No clipping,
# no per-channel imbalance, and the signal was not too quiet (peak 0.15-0.28, well above
# the "very quiet" threshold) -- so this is not a mic-level problem, it's a content
# problem: `voice talk` starts recording for a fixed duration immediately, with no "get
# ready" cue, so how much silence surrounds the actual utterance -- and therefore what
# fraction of the waveform SpeechBrain's ECAPA-TDNN encoder actually has to work with --
# depends purely on when the user happens to start/stop talking relative to that fixed
# window. Feeding the encoder a waveform that is mostly silence, in a different ratio and
# position every time, produces a correspondingly unstable embedding even for the same
# genuine speaker. `voice_verification_threshold` was left at 0.6 throughout this
# investigation, exactly as instructed.
MIN_TRIMMED_SECONDS = 0.5


def _trim_silence(
    samples: list[float],
    sample_rate: int,
    *,
    window_ms: float = 30.0,
    threshold_ratio: float = 0.1,
    padding_seconds: float = 0.2,
) -> list[float]:
    """Crops leading/trailing silence from a recording down to its speech-bearing region
    (plus `padding_seconds` on each side), using the same energy-based window activity as
    `_estimate_speech_activity()` so the trim boundaries always match what the diagnostics
    log. This normalizes what SpeechBrain/faster-whisper actually see across every
    recording -- enrollment and every `voice talk` attempt alike, since both call
    `record_from_microphone()` -- instead of each attempt handing the encoder a different,
    unpredictable ratio of real speech to silence. Falls back to returning the audio
    untouched if no activity is detected at all, or if trimming would leave less than
    `MIN_TRIMMED_SECONDS` of audio, rather than risk handing the encoder an almost-empty
    clip."""
    import numpy as np

    if not samples or sample_rate <= 0:
        return samples

    arr = np.asarray(samples, dtype=np.float32)
    activity, window_size = _compute_window_activity(arr, sample_rate, window_ms=window_ms, threshold_ratio=threshold_ratio)
    if not activity or not any(activity):
        return samples

    first_active = next(i for i, is_active in enumerate(activity) if is_active)
    last_active = len(activity) - 1 - next(i for i, is_active in enumerate(reversed(activity)) if is_active)

    padding_samples = int(padding_seconds * sample_rate)
    start = max(0, first_active * window_size - padding_samples)
    end = min(arr.size, (last_active + 1) * window_size + padding_samples)

    if (end - start) < int(MIN_TRIMMED_SECONDS * sample_rate):
        return samples

    return arr[start:end].tolist()


def _log_channel_diagnostics(recording, *, channels: int) -> None:
    """Logs per-channel peak/RMS from the *native*, pre-downmix multi-channel capture, and
    warns if channel levels are unusually uneven -- a possible source of instability if, for
    example, the user's mouth position relative to a multi-mic array shifts between
    attempts, changing which channel(s) actually pick up voice."""
    import numpy as np

    if channels <= 1:
        return
    try:
        samples = np.asarray(recording, dtype=np.float32).reshape(-1, channels)
        peaks = np.abs(samples).max(axis=0)
        rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2, axis=0))
    except Exception:
        logger.exception("Could not compute per-channel diagnostics for native capture.")
        return

    logger.info("Native capture per-channel levels: peak=%s rms=%s", peaks.tolist(), rms.tolist())
    max_peak = float(peaks.max())
    if max_peak > 0.0:
        min_max_ratio = float(peaks.min()) / max_peak
        if min_max_ratio < 0.2:
            logger.warning(
                "Uneven per-channel levels in native capture (min/max peak ratio=%.3f); "
                "one or more mic channels may be picking up little or no signal this attempt.",
                min_max_ratio,
            )


def _log_recording_diagnostics(samples: list[float], *, sample_rate: int, channels: int, stage: str) -> None:
    """Logs the full set of diagnostics requested for the voice-verification-instability
    investigation: shape, sample rate, channel count, duration, min/max, RMS, peak, and
    sample count, for a given pipeline `stage` ("native_capture" or "preprocessed"). Also
    flags likely-quiet, likely-clipped, and low-speech-activity recordings, since any of
    these could explain wildly different embeddings/similarity scores between consecutive
    `voice talk` attempts against the same enrolled owner."""
    import numpy as np

    arr = np.asarray(samples, dtype=np.float32)
    n_samples = int(arr.size)
    duration_seconds = (n_samples / sample_rate) if sample_rate else 0.0

    if n_samples == 0:
        logger.warning("Voice recording diagnostics (%s): recording is empty (0 samples).", stage)
        return

    min_val = float(arr.min())
    max_val = float(arr.max())
    peak = float(np.abs(arr).max())
    rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
    clipping_ratio = float(np.mean(np.abs(arr) >= 0.99))
    active_ratio, leading_silence_s, trailing_silence_s = _estimate_speech_activity(samples, sample_rate)

    logger.info(
        "Voice recording diagnostics (%s): shape=(%d,) sample_rate=%d channels=%d duration=%.3fs "
        "min=%.6f max=%.6f peak=%.6f rms=%.6f n_samples=%d clipping_ratio=%.4f "
        "active_speech_ratio=%.3f leading_silence=%.3fs trailing_silence=%.3fs",
        stage, n_samples, sample_rate, channels, duration_seconds,
        min_val, max_val, peak, rms, n_samples, clipping_ratio,
        active_ratio, leading_silence_s, trailing_silence_s,
    )

    if peak < 0.01:
        logger.warning(
            "Voice recording diagnostics (%s): very quiet recording (peak=%.6f); this attempt's "
            "embedding/similarity may be unreliable due to insufficient signal.", stage, peak,
        )
    if clipping_ratio > 0.001:
        logger.warning(
            "Voice recording diagnostics (%s): possible clipping (%.2f%% of samples at/near full "
            "scale); this attempt's embedding/similarity may be distorted.", stage, clipping_ratio * 100.0,
        )
    if active_ratio < 0.3:
        logger.warning(
            "Voice recording diagnostics (%s): low active-speech ratio (%.2f); the fixed-duration "
            "recording window may contain mostly silence/room noise instead of speech this attempt.",
            stage, active_ratio,
        )


def record_from_microphone(duration_seconds: float, *, sample_rate: int = VOICE_SAMPLE_RATE_HZ) -> list[float]:
    """Local PC push-to-talk capture. Sets the visible mic-active indicator for exactly the
    duration audio is actively being captured, per RFC-009's 'Visible state' requirement --
    never a window where JARVIS is listening without something reflecting it in
    `voice status`.

    Always records at the configured input device's own native sample rate/channel count
    (see the hardware note above) and only then downmixes to mono and resamples to
    `sample_rate` (16 kHz by default, what STT/speaker verification require) in software --
    never by asking PortAudio to capture directly in that target format, which silently
    produces near-silent audio on at least one real target device. Finally trims leading and
    trailing silence (see `_trim_silence` above) so STT/speaker verification always receive a
    speech-dense clip instead of a fixed-duration window padded with an unpredictable amount
    of silence/room noise -- confirmed on real hardware to be the cause of unstable
    `voice talk` speaker-verification similarity scores between consecutive attempts."""
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

    _log_channel_diagnostics(recording, channels=channels)
    mono_native = _downmix_to_mono(recording, channels)
    _log_recording_diagnostics(mono_native, sample_rate=native_sample_rate, channels=channels, stage="native_capture")

    preprocessed = resample_audio(mono_native, native_sample_rate, sample_rate)
    _log_recording_diagnostics(preprocessed, sample_rate=sample_rate, channels=1, stage="preprocessed")

    trimmed = _trim_silence(preprocessed, sample_rate)
    if len(trimmed) != len(preprocessed):
        logger.info(
            "Trimmed %.3fs of leading/trailing silence from recording (%.3fs -> %.3fs) before "
            "handing audio to STT/speaker verification.",
            (len(preprocessed) - len(trimmed)) / sample_rate if sample_rate else 0.0,
            len(preprocessed) / sample_rate if sample_rate else 0.0,
            len(trimmed) / sample_rate if sample_rate else 0.0,
        )
    _log_recording_diagnostics(trimmed, sample_rate=sample_rate, channels=1, stage="trimmed")
    return trimmed


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

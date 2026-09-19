# Runtime Configuration

Configuration changes are deterministic commands only. The mutable key allowlist excludes paths, logging destinations, provider URLs, security policy, tool registries, plugin directories, secrets, and confirmation settings.

Writes validate the complete candidate, write a temporary file in the target directory, flush and synchronize it when supported, preserve the last valid backup, and replace the target atomically. Failed writes preserve the previous file and runtime snapshot.

## Vision Runtime

RFC-007A and RFC-007B keep Vision configuration separate from the text intelligence model.

- `vision_enabled`
  Enables or disables the Vision Runtime.
- `vision_provider`
  Provider name for local image analysis. RFC-007A supports only `ollama`.
- `vision_model`
  Local multimodal Ollama model name. This is independent from `intelligence_model`.
- `vision_ollama_base_url`
  Ollama base URL for image analysis. RFC-007A accepts only loopback addresses.
- `vision_timeout_seconds`
  Timeout for local Vision provider requests.
- `vision_max_file_size`
  Maximum accepted image file size in bytes before provider invocation.
- `vision_max_width`
  Maximum accepted image width in pixels.
- `vision_max_height`
  Maximum accepted image height in pixels.
- `vision_max_pixels`
  Maximum accepted total pixel count.
- `vision_max_ocr_chars`
  Maximum extracted OCR characters kept in structured evidence and user-visible results.
- `vision_max_regions`
  Maximum visual regions returned by `vision.find_visual_element`.
- `vision_evidence_retention_seconds`
  In-memory retention window for temporary Vision evidence.
- `vision_browser_capture_enabled`
  Enables or disables RFC-007B browser viewport capture analysis.
- `vision_browser_capture_ttl_seconds`
  Time-to-live for opaque browser capture records before deterministic cleanup rejects reuse.
- `vision_browser_capture_max_bytes`
  Maximum temporary browser capture size retained in memory.
- `vision_browser_capture_max_width`
  Maximum allowed captured viewport width.
- `vision_browser_capture_max_height`
  Maximum allowed captured viewport height.
- `vision_browser_capture_max_pixels`
  Maximum allowed captured viewport pixel count.

RFC-007B browser captures remain temporary, loopback-only, and metadata-only outside the private capture store. They do not expose filesystem paths, bytes, base64, or reusable screenshot artifacts to the planner.

RFC-007A and RFC-007B do not support remote image URLs, camera input, external image upload, or persistent Vision evidence export.

## Desktop Capture

RFC-007C keeps desktop and window capture configuration separate from browser capture, in its own `vision_desktop_capture_*` family.

- `vision_desktop_capture_enabled`
  Enables or disables RFC-007C desktop and window capture, including `desktop.list_windows`. All desktop tools additionally require `vision_enabled`. Neither listing nor capturing requires the Vision provider (Ollama) to be reachable; only analyzing a capture (`vision.describe_desktop_capture` and similar) does.
- `vision_desktop_capture_ttl_seconds`
  Time-to-live for opaque desktop capture records before deterministic cleanup rejects reuse.
- `vision_desktop_capture_max_bytes`
  Maximum temporary desktop capture size retained in memory.
- `vision_desktop_capture_max_width`
  Maximum allowed captured width.
- `vision_desktop_capture_max_height`
  Maximum allowed captured height.
- `vision_desktop_capture_max_pixels`
  Maximum allowed captured pixel count.

Desktop and window captures remain temporary and metadata-only outside their private capture store, which is kept separate from the browser capture store. They do not expose filesystem paths, bytes, base64, or reusable screenshot artifacts to the planner. Desktop capture is one-shot only: `desktop.capture_screen` and `desktop.capture_window` are capped, combined, at `MAX_DESKTOP_CAPTURES_PER_PLAN` (currently 1) per agent plan, and both are always MEDIUM risk requiring explicit `approve plan`, never auto-executed. `desktop.capture_window` also revalidates the target window's title immediately before capturing, rejecting the capture if the window has changed since the plan was approved. RFC-007C does not support remote desktop, full-page stitching, camera input, or persistent capture export, and is only supported on Windows.

## Memory

RFC-008 adds a small allowlisted configuration surface for the memory store.

- `memory_max_entries`
  Soft cap on the number of distinct memory keys. New keys are rejected once this limit is reached; updates to existing keys are still allowed. Default `500`.
- `memory_learned_capture_enabled`
  Enables or disables writing memory entries with `category=learned_pattern`. When disabled, a `memory.remember` call requesting that category is rejected and nothing is written. Default `true`.

These keys follow the same deterministic-commands-only, atomic-write path as every other configuration key; they are not written directly to `config/config.json`.

## Location and Navigation

RFC-010 keeps location/navigation configuration in its own `location_*`/`osrm_*` family, separate from Vision.

- `location_enabled`
  Enables or disables the Location/Navigation Runtime, including the `/location/overland` HTTP route and all `location.*`/`navigation.*` tools.
- `location_bind_host`
  The Tailscale interface IP the `/location/overland` route binds to. Never `0.0.0.0`/`::`. Empty (the default) means the location server does not start. Mutable via `config set`, but still independently validated at bind time: the server refuses to start with a clear error if this is empty, a wildcard address, or not an address of any interface actually present on this machine, rather than silently falling back to a public bind.
- `location_port`
  TCP port the `/location/overland` route listens on.
- `location_shared_secret`
  The bearer token Overland's "access token"/`Authorization: Bearer` field must match. Unlike every other key on this page, this one is a credential, not a tunable setting: it is deliberately excluded from the `config set` mutable-key allowlist (set it by editing `config/config.json` directly) and `config show`/`config get` redact it instead of echoing it back in plaintext.
- `location_stale_after_seconds`
  How old the freshest phone location can be before `location.where_am_i` appends an explicit "this location is stale" note instead of presenting it as current.
- `osrm_base_url`
  Base URL of the OSRM routing provider used by `navigation.start`/`navigation.get_next_instruction`. Defaults to the public OSRM demo server; point this at a self-hosted OSRM instance to remove the demo server's rate limits, without any change to the `navigation.*` tool surface.

`/location/overland` is bound only to `location_bind_host`, is not reachable from `0.0.0.0`, and is not registered as an AI-callable tool — see `docs/RFC-010_PHONE_LOCATION_NAVIGATION.md` for the full security rationale.

## Voice

RFC-009 keeps voice configuration in its own `voice_*` family. `voice_enabled` (already present for the pre-existing `voice on`/`voice off` toggle) now actually gates the Voice Runtime, including `/voice/turn` and `/voice/client`, which share RFC-010's Tailscale-bound listener and `location_bind_host`/`location_port`/`location_shared_secret` rather than opening a second one.

- `voice_stt_model`
  The `faster-whisper` model size used for local speech-to-text (e.g. `tiny`, `base`, `small`). Default `small`.
- `voice_tts_voice`
  Filesystem path to a downloaded Piper `.onnx` voice model. Empty (the default) means text-to-speech is unavailable until the operator downloads a voice and sets this — the same "operator provides the model" convention as Ollama.
- `voice_require_speaker_verification`
  Whether `voice talk`/`/voice/turn` require the live utterance's speaker embedding to match the enrolled owner before it's transcribed and acted on. Default `false`, changed from RFC-009's original always-on design at the owner's explicit request: extensive real-hardware testing showed the enrolled owner's own genuine similarity score varying widely (0.23-0.68) even after every mechanical cause (mic capture, stale enrollment, silence padding) was found and fixed, making the gate more friction than protection for a single-user local device. With this `false`, `voice talk` needs no enrollment at all and transcribes/routes any voice like typed input. Setting it to `true` restores speaker verification as a hard requirement (see the "Hard boundary" section of `docs/RFC-009_ON_DEMAND_VOICE_ASSISTANT.md`) -- appropriate again if more than one person's voice might reach the microphone, since with it `false` there is no restriction on whose voice is treated as a command.
- `voice_verification_threshold`
  Minimum cosine similarity (0.0-1.0) between a live utterance's speaker embedding and the enrolled owner's stored embedding before a voice turn is accepted. Default `0.4`, after two rounds of data-driven recalibration on real hardware: `0.75` (original) rejected a genuine live attempt scoring `0.7311`, motivating a drop to `0.6`. After that, fixing the pipeline bugs this investigation uncovered (native-format mic capture, SpeechBrain's Windows symlink-privilege crash, and trimming the silence surrounding speech in every recording) and re-enrolling under the fully-fixed pipeline produced a far more internally consistent profile (pairwise sample similarity `0.78-0.81`, up from `0.50-0.67`) -- yet real `voice talk` attempts against that clean profile still landed at `0.5883`, `0.4662`, and `0.2318`, all below `0.6`. `0.4` sits below the worst clean genuine-match score observed (`0.4662`) while staying above `speechbrain`'s own `SpeakerRecognition.verify_batch()` reference default of `0.25` for this exact `spkrec-ecapa-voxceleb` checkpoint. This has not been validated against an impostor (different-speaker) sample on this specific hardware -- only against the enrolled owner's own genuine attempts -- so it remains fully configurable via `config set voice_verification_threshold <value>` and worth retightening if a false-accept is ever observed, guided by the similarity scores `voice enroll`/`voice talk` log to `logs/jarvis.log`.
- `voice_input_device`
  PortAudio input device index used for local microphone capture (`voice enroll`, `voice talk`). Default `1`. Confirmed on a real Windows 11 target machine: some built-in "Microphone Array" hardware exposes two PortAudio device indices for the same physical mic — a plain index that records real signal and a separate WASAPI-variant index that either refuses the requested channel count outright or, if opened at a non-native format, returns near-silent audio. `voice_input_device` should point at the plain (non-WASAPI) index; run `python -m sounddevice` in the same environment JARVIS runs in to list device indices/names if unsure.
- `voice_input_sample_rate`
  The input device's own native sample rate in Hz (8000-192000). Default `44100`. Recording always happens at this native rate and is resampled in software to the 16 kHz `faster-whisper`/SpeechBrain expect — never captured directly at 16 kHz, which silently produced near-silent audio on the array-mic hardware above.
- `voice_input_channels`
  The input device's own native channel count (1-8). Default `4`, matching a typical laptop array mic. Captured audio is downmixed to mono (channel average) in software before transcription/verification.
- `voice_talk_auto_stop_on_silence`
  Whether local microphone capture (`voice enroll`, `voice talk`) stops as soon as trailing silence follows detected speech, instead of always recording for the full requested duration. Default `true`. Added after the owner reported a roughly 7-second wait after finishing speaking before `voice talk` acted on it — root-caused to the previous fixed-duration blocking recording continuing to capture silence long after speech had ended. Set `false` to restore the old always-record-the-full-window behavior (e.g. if auto-stop is ever cutting off a real utterance too early on different hardware).
- `voice_talk_silence_timeout_seconds`
  How many seconds of trailing silence (below the recording's own calibrated noise floor) must follow detected speech before capture auto-stops. Default `1.0`, range 0.2-10.0. Only used when `voice_talk_auto_stop_on_silence` is `true`.
- `voice_talk_min_duration_seconds`
  Minimum seconds a capture must run before auto-stop-on-silence is allowed to end it early, even if trailing silence is detected sooner — guards against a very short opening word triggering an instant cutoff. Default `1.0`, range 0.2-10.0. Only used when `voice_talk_auto_stop_on_silence` is `true`.
- `voice_listen_on_startup`
  Whether JARVIS enters continuous voice listening (the same loop `voice listen` starts) automatically right after the startup banner, with no command typed first. Default `true`, added at the owner's explicit request alongside `ai_enabled`/`voice_enabled` defaulting to `true` (see "Enabled by default" below): together, these mean a fresh `python main.py` can already hear and act on spoken commands, with none of `ai on`/`voice on`/`voice talk` needed. Only takes effect when `voice_enabled` is also `true`. A stop phrase (e.g. "stop listening") or Ctrl+C ends the startup listening session and falls back to the normal typed `You:` prompt — it does not exit JARVIS, and typing works normally afterwards. Set `false` to keep the previous behavior of starting silent and waiting for typed/voice commands to be requested explicitly.

### Enabled by default

`ai_enabled` and `voice_enabled` both default to `true` as of this change (previously `false`) — another owner request in the same vein as `voice_listen_on_startup` above and the speaker-verification opt-in change: this is a single-user local assistant, and typing `ai on`/`voice on` at the start of every session was pure friction with no safety benefit for that use case. `ai on`/`ai off`/`voice on`/`voice off` remain available and still only apply for the current process (never persisted to `config/config.json`), for anyone who wants to start a given run with either off; `config set ai_enabled false` (or `voice_enabled`) changes the persisted default instead.

Every local-mic recording (`voice enroll` and `voice talk` alike, since both call the same capture function) is also trimmed to its speech-bearing region -- leading/trailing silence cropped off, plus a small padding -- before being handed to STT/speaker verification. This was added after real-hardware logs showed `voice talk`'s fixed-duration recording window capturing mostly silence (80-88% of the clip on the affected machine) in a different ratio each attempt, which produced unstable speaker-verification similarity scores for the same genuine owner even though the threshold was correctly calibrated. This is not configurable via `config set`; it is not a tunable setting but a fixed part of the capture pipeline. `voice enroll`/`voice talk` log a `Voice recording diagnostics (native_capture/preprocessed/trimmed):` line at each stage plus per-channel levels to `logs/jarvis.log`, so a future capture problem on different hardware is diagnosable the same way this one was.

Voice enrollment (`voice enroll`, `voice forget me`) stores a single averaged speaker embedding in `data/voice_profile.json` — never raw audio. See `docs/RFC-009_ON_DEMAND_VOICE_ASSISTANT.md` for the full hard-boundary rationale (only the enrolled owner's voice is ever processed) and the Phase 0 library choices.

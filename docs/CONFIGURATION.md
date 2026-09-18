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
- `voice_verification_threshold`
  Minimum cosine similarity (0.0-1.0) between a live utterance's speaker embedding and the enrolled owner's stored embedding before a voice turn is accepted. Default `0.75`.

Voice enrollment (`voice enroll`, `voice forget me`) stores a single averaged speaker embedding in `data/voice_profile.json` — never raw audio. See `docs/RFC-009_ON_DEMAND_VOICE_ASSISTANT.md` for the full hard-boundary rationale (only the enrolled owner's voice is ever processed) and the Phase 0 library choices.

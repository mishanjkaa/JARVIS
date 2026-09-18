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

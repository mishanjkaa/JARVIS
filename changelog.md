# Changelog

## RFC-007C
- Added one-shot desktop and window capture: `desktop.list_windows`, `desktop.capture_screen`, `desktop.capture_window`, and `vision.describe_desktop_capture`/`extract_text_from_desktop_capture`/`find_visual_element_in_desktop_capture`, mirroring the RFC-007B browser-capture shape (opaque `capture_id`, private TTL-bound store, no verification pipeline since a desktop capture has no DOM to check against).
- `desktop.capture_screen` and `desktop.capture_window` are always MEDIUM risk and are deliberately excluded from every project-scoping exemption in the risk analyzer, so they can never auto-execute even with `developer_mode` and `auto_execute_medium_project` both enabled. Added a regression test proving this explicitly.
- Enforced "one-shot" as a real limit, not just a description: added `MAX_DESKTOP_CAPTURES_PER_PLAN` (currently `1`), mirroring the existing `MAX_MEMORY_READS` plan-validator pattern, capping `desktop.capture_screen`/`desktop.capture_window` combined per plan.
- `desktop.capture_window` revalidates the target window immediately before capturing: existence, visibility, and title. Windows can reuse a closed window's handle for an unrelated window, so the title check specifically catches a plan being executed against a different window than the one the user approved, rejecting the capture with a clear reason instead of silently capturing the wrong thing.
- Added a deterministic `planner_v2.py` trigger for "screenshot/capture the desktop" and "screenshot/capture window `<title>`", resolving the window title against currently open windows at plan-construction time so the approval preview names the real window, not a placeholder.
- Implemented capture with zero new dependencies: Pillow's `ImageGrab` (already a dependency) plus `ctypes` calls into `user32.dll`/`dwmapi.dll` for window enumeration and rectangles, matching this codebase's existing `ctypes`-based Windows API style. Windows-only; fails explicitly and cleanly on other platforms.
- Added `vision_desktop_capture_enabled`, `vision_desktop_capture_ttl_seconds`, `vision_desktop_capture_max_bytes`, `vision_desktop_capture_max_width`, `vision_desktop_capture_max_height`, and `vision_desktop_capture_max_pixels` to the runtime configuration allowlist.

## Bug fix: config set discarding session-only toggles
- Fixed `config_set()` in `app/brain/configuration/config_commands.py` silently disabling any active session-only setting (`ai on`, `voice on`, `developer mode on`, `conversation on`, etc.) whenever an unrelated `config set <key> <value>` command ran. The cause: `config_set()` rebuilt the in-memory runtime snapshot from the just-reloaded config file alone, discarding whatever session-only overrides were layered on top of it.
- `config_set()` now captures the current effective runtime config (file plus active session overrides) before writing, patches in only the newly persisted key, and installs that merged result as the new snapshot. Every other active session-only override now survives an unrelated `config set` call, and a `config set` on a key that itself has a stale session override still ends up with the newly persisted value. `config_reset()` and `config_reload()` are unchanged; they still intentionally wipe the session snapshot.

## RFC-008
- Added `category` (`fact`/`preference`/`learned_pattern`) and `source` (`user`/`ai_proposed`) fields to memory entries, defaulted for backward compatibility.
- Added a `memory list learned` deterministic command and surfaced category/source in `memory list` output.
- Added a "did you mean" key-name hint (substring/prefix match only) to the deterministic `recall <key>` command on a miss; the AI-facing `memory.recall` tool contract is unchanged.
- Wired `MAX_MEMORY_READS` into `app.brain.planner.plan_validator` as a real per-plan cap on `memory.recall` steps, and removed the dead `app/brain/memory/retrieval.py` module it replaced.
- Added `memory.forget` as a registered AI tool with the same `persistent_write` approval path as `memory.remember`.
- Added `memory_max_entries` and `memory_learned_capture_enabled` to the runtime configuration allowlist.

## 2.0.0
- Added a hybrid router with deterministic-first execution and optional local AI fallback.
- Introduced a safe AI provider, parser, orchestrator, tool registry, and executor.
- Added config, privacy, and policy safeguards for AI-driven tool use.

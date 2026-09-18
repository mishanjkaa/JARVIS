# Changelog

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

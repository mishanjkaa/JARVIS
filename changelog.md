# Changelog

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

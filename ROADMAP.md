# Roadmap

## 2.0.0
- Complete the hybrid deterministic + AI routing architecture.
- Keep deterministic commands fully compatible with 1.3.0 behavior.
- Add safe local provider integration and policy-guarded tool execution.

## Vision Stages
- RFC-007A: local file Vision foundation with description, OCR, element search, structured evidence, and local Ollama multimodal integration.
- RFC-007B: browser visual evidence through temporary viewport captures, opaque capture IDs, local-only browser-page analysis, strict capture ownership, and grounded responses.
- RFC-007C and later: consented desktop capture, visible camera sessions, opt-in local identity support, GUI-runtime integration, and local voice runtime.

## RFC-008: Memory Usefulness and a Minimal Self-Learning Trail

Shipped:
- `category` (`fact` default, `preference`, `learned_pattern`) and `source` (`user`, `ai_proposed`) fields on memory entries, both optional and backward compatible with existing stored data.
- Source is set by code path, not by argument: the deterministic `remember <key> = <value>` command always writes `source=user`; the AI-facing `memory.remember` tool call always writes `source=ai_proposed`, regardless of an approved plan's contents.
- A `memory list learned` deterministic command and category/source annotations in `memory list` output, giving a visible audit trail of what the user stated directly versus what the AI wrote after an approved plan.
- A "did you mean" hint on the deterministic `recall <key>` command when the key is not found, using plain substring/prefix matching over stored key names only. It returns no memory values and is not part of the AI-facing `memory.recall` tool contract.
- `MAX_MEMORY_READS`, previously an unused constant, now enforced by the plan validator as a real per-plan cap on `memory.recall` steps, on top of the existing five-step overall plan limit.
- `memory.forget` as a registered AI tool with the same `persistent_write` approval path as `memory.remember`, closing the previous gap where the AI could create and read memory but never delete it.
- `memory_max_entries` (soft cap, default 500) and `memory_learned_capture_enabled` (default on) as allowlisted runtime configuration keys, gating store growth and learned-pattern writes respectively.
- Removed the dead `app/brain/memory/retrieval.py` module and the unused `validate_memory_key`/`can_retrieve_memory` helpers it depended on.

Explicitly deferred, by design, not "not yet done":
- Semantic search or embedding-based recall. Retrieval stays explicit single-key lookup, with only literal substring/prefix matching for the "did you mean" hint.
- Automatic injection of memory content into the AI's context. There is still no code path that surfaces memory to the AI other than an explicit `memory.recall` tool call inside an approved plan.
- A `memory.list`-style tool for the AI. Listing and filtering memory (including `memory list learned`) remain deterministic-command-only, matching the single-key-retrieval invariant.

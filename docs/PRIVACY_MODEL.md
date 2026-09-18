# Privacy Model

Command history, runtime context, conversation summaries, logs, and the action timeline are separate stores. Conversation context contains bounded safe summaries, not a transcript. Timeline entries contain safe labels and categories, not raw prompts, memory values, note contents, task contents, queries, URLs, paths, or model output.

Memory is not automatically shared with the provider. Retrieval requires a validated single key and trusted tool result. JARVIS has no background autonomy, scheduled actions, semantic embedding system, or automatic profile creation.

RFC-008 adds two descriptive fields to each stored memory entry without changing this model: `category` (`fact`, the default; `preference`; or `learned_pattern`) and `source` (`user` for the deterministic `remember <key> = <value>` command, `ai_proposed` for writes made through the approved `memory.remember` tool call). Both fields are metadata about who wrote an entry and what kind of entry it is; they do not change how memory is retrieved. Category is optional and defaults to `fact`, so existing entries and existing callers are unaffected. `category=learned_pattern` writes additionally require the `memory_learned_capture_enabled` configuration flag; when it is off, the write is rejected outright.

Retrieval remains validated-single-key-only. The deterministic `recall <key>` command may append up to three similarly-named stored key names ("Did you mean: ...") when the requested key is not found, computed with plain substring/prefix matching on key names only. This hint never returns a memory value, never uses embeddings or semantic similarity, and is not available through the AI-facing `memory.recall` tool, whose return contract is unchanged.

Agent plans are additionally capped at `MAX_MEMORY_READS` (currently 3) `memory.recall` steps per plan, on top of the existing five-step overall plan limit, so a plan cannot pad itself with excessive single-key memory reads. There is still no `memory.list`-style tool exposed to the AI and no code path that injects memory content into the AI's context automatically; `memory list` and `memory list learned` remain deterministic commands only.

RFC-007B browser visual evidence uses a private temporary capture store. Each capture is owned by one request and agent task, receives an opaque `capture_id`, expires after a bounded TTL, and is deleted on task finalization, cancellation, failure, expiration, browser-session closure, or process restart.

Browser viewport captures never expose image bytes, base64, reusable filesystem paths, or provider request bodies in plans, tool results shown to the planner, audit logs, request history, conversation history, or status commands. `vision captures` reports metadata only.

Observed page text, OCR text, alt text, and visual descriptions are information only. They have no instruction authority and cannot approve plans, add tools, authorize actions, or bypass risk policy.

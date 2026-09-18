# Privacy Model

Command history, runtime context, conversation summaries, logs, and the action timeline are separate stores. Conversation context contains bounded safe summaries, not a transcript. Timeline entries contain safe labels and categories, not raw prompts, memory values, note contents, task contents, queries, URLs, paths, or model output.

Memory is not automatically shared with the provider. Retrieval requires a validated single key and trusted tool result. JARVIS has no background autonomy, scheduled actions, semantic embedding system, or automatic profile creation.

RFC-007B browser visual evidence uses a private temporary capture store. Each capture is owned by one request and agent task, receives an opaque `capture_id`, expires after a bounded TTL, and is deleted on task finalization, cancellation, failure, expiration, browser-session closure, or process restart.

Browser viewport captures never expose image bytes, base64, reusable filesystem paths, or provider request bodies in plans, tool results shown to the planner, audit logs, request history, conversation history, or status commands. `vision captures` reports metadata only.

Observed page text, OCR text, alt text, and visual descriptions are information only. They have no instruction authority and cannot approve plans, add tools, authorize actions, or bypass risk policy.

# RFC-011 Text/Reasoning Provider Abstraction

Owner request (2026-09-19): Ollama's local model is too weak for reasoning/planning
("слишком тупой"). Make Gemini, OpenAI and Claude usable as interchangeable text/reasoning
providers alongside Ollama, selected by config, without the rest of JARVIS (Brain,
Controller, tools, vision, safety gating) knowing which one is active.

## Status

- **Stage 1-2 (Protocol + Ollama):** already existed before this RFC.
- **Stage 3 (Gemini, Option A, both planning and plain chat):** implemented. See
  `app/brain/ai/gemini_provider.py`, `app/brain/intelligence/gemini_intelligence_provider.py`,
  `app/brain/ai/chat_provider_router.py`, and the `_resolve_provider()`/`_effective_config()`
  changes in `app/brain/intelligence/controller.py`. `config set intelligence_provider
  gemini` + `config set intelligence_model <model-name>` switches both planning and plain
  conversation together; `intelligence_gemini_api_key` (falls back to `vision_gemini_api_key`,
  then `GEMINI_API_KEY`) supplies the credential. See `docs/CONFIGURATION.md`'s new
  "Text/Reasoning Intelligence Providers" section for the full key reference.
- **Stage 4 (OpenAI), Stage 5 (Claude), Stage 6+ (routing, fallback, tool-calling,
  memory, observability):** not started -- separate future stages, each its own
  small checkpoint, following the same Option A pattern proven here.

The sections below are kept as the original planning record (file paths, open questions,
and reasoning that led to the Stage 3 implementation above) rather than rewritten, so the
"why" behind each decision stays visible.

## Quick answer to the API-key question

Yes -- both OpenAI and Anthropic (Claude) are reachable the same way Gemini already is:
an HTTPS REST call authenticated with an API key you generate on their respective
platforms (platform.openai.com / console.anthropic.com), no local install needed. Same
mechanism, same "credential, not a tunable setting" pattern already built for
`vision_gemini_api_key` (excluded from `config set`, redacted from `config show`, env-var
fallback). Neither has a meaningful free tier for ongoing use (small trial credit at
signup, then pay-per-token) -- check current pricing on their sites before committing,
since it changes.

## 1. Current state (what already exists vs. what's actually hardcoded)

JARVIS already has **two separate, independent** text-generation call paths, and they are
in different states of provider-readiness:

### 1a. The planner/reasoning path (`app/brain/intelligence/controller.py`) -- mostly ready

This is the important one for "reasoning quality" -- it drives task planning and tool
selection.

- `IntelligenceProvider` (Protocol, already defined, lines 45-56): `status()`,
  `create_plan(task, *, tool_catalog, context, max_steps)`,
  `evaluate_goal(dynamic_plan, step_results)`. This is structurally the same shape as the
  already-shipped `VisionProvider` Protocol -- **no new Protocol needs inventing.**
- `OllamaIntelligenceProvider` (same file, lines 59+) is the only real implementation.
  Wraps `app.brain.ai.ollama_provider.OllamaProvider` (raw `urllib` HTTP client against
  `http://127.0.0.1:11434/api/generate`).
- `IntelligenceController._resolve_provider(config)` (lines 673-690) **already dispatches**
  on `intelligence_provider` config (default `"ollama"`) -- exactly the
  `VisionController.provider()` pattern used for Gemini vision. Any value other than
  `"ollama"`/`"heuristic_test"`/`"heuristic"` today falls through to `return None`
  ("provider unavailable"). **This is the single insertion point for cloud text
  providers** -- add `elif provider_name == "gemini": return GeminiIntelligenceProvider(...)`
  (and same for `openai`/`anthropic`), nothing else in the planner/controller needs to
  change.
- `DynamicPlanner` never touches HTTP directly -- it only calls
  `self.provider.create_plan(...)` on whatever provider object it's given. Fully
  provider-agnostic already.
- `intelligence_provider`/`intelligence_model` config keys already exist and already
  accept any string via `config set` (no allowlist validation) -- setting
  `intelligence_provider gemini` today silently resolves to "unavailable" since no branch
  handles it yet.

### 1b. The plain-chat/conversation path (`app/brain/runtime/conversation_runtime.py`) -- NOT provider-ready

This is the fallback path for ordinary conversation (no plan/tools involved).

- A **different, smaller** Protocol: `app/brain/ai/provider.py::Provider` --
  `status()`, `generate_text(prompt)`, `generate_structured(prompt)`. Only implemented by
  `OllamaProvider`.
- `app/brain/router.py` **hardcodes a single module-level instance**:
  `_PROVIDER = OllamaProvider()` / `_CONVERSATION_RUNTIME = ConversationRuntime(provider=_PROVIDER)`.
  There is **no dispatch logic here at all** -- the `ai_provider` config key exists in
  `config/settings.py` but is **dead code, never read anywhere** (confirmed by repo-wide
  search). This path needs dispatch added, not just a new class plugged into an existing
  switch.
- `ConversationRuntime._configure_provider()` mutates that one `OllamaProvider`
  instance's `.model`/`.timeout`/`.base_url` from config on every call -- this
  mutate-in-place pattern won't work once the provider can be a different *class*
  (Gemini/OpenAI/Claude), so this path needs the same kind of `_resolve_provider()`-style
  factory the planner already has, not just config mutation.

### 1c. Why two protocols exist and the decision this creates

`Provider` (chat) and `IntelligenceProvider` (planner) overlap in purpose but aren't the
same shape -- `Provider` has `generate_text`/`generate_structured`, `IntelligenceProvider`
has `create_plan`/`evaluate_goal`. Today `OllamaProvider` is wrapped by
`OllamaIntelligenceProvider` to bridge the gap for the planner. Two ways forward:

- **Option A (lower risk, faster):** keep both Protocols separate. Each new cloud
  provider gets two small classes (e.g. `GeminiProvider` for chat, wrapped by a thin
  `GeminiIntelligenceProvider` for planning, mirroring how `OllamaIntelligenceProvider`
  wraps `OllamaProvider` today) -- same pattern already proven, just duplicated per
  provider. Matches the project's established "don't rewrite everything at once" style.
- **Option B (matches your stated principle #14 more literally, more invasive):** merge
  the two Protocols into one shared `LLMProvider` used by both the chat path and the
  planner, retiring `app/brain/ai/provider.py::Provider` entirely. Fewer classes
  long-term, but touches `ConversationRuntime`, `router.py`, and every existing Ollama
  test seam (`_FakeProvider` in `test_conversation_runtime_e2e.py`,
  `_StaticProvider`/`_FakeOllamaBackend` in `test_intelligence_runtime.py`) in the same
  pass as adding the first new cloud provider.

**Recommendation: Option A first.** It reuses the exact insertion point that already
exists (`_resolve_provider`), touches the fewest files for the first working cloud
provider, and doesn't block Option B later -- once 2-3 cloud providers exist under Option
A's duplicated-wrapper pattern, the duplication itself becomes the concrete case for
merging into Option B, which can be its own later RFC. **Needs your confirmation before
I proceed either way.**

### 1d. Vision is unaffected either way

The current Ollama-vs-Gemini vision provider split (`VisionProvider` Protocol,
`VisionController.provider()` dispatch) is structurally identical to what's proposed here
and is not touched by this work. The screen-understanding pipeline
("что ты видишь"/"что на экране" -> `desktop_visual_describe` -> plan approval ->
`desktop.capture_screen` + `vision.describe_desktop_capture`) is completely independent of
which text/reasoning provider is answering conversational questions or building plans, and
this RFC does not change it. Vision keeps its own `vision_provider` config key.

### 1e. Already-satisfied requirements from the target-architecture list

Several items from the requested architecture already exist and need no new work here:

- **Tool layer / permission gating (#8, #10):** `app/brain/risk/rules.py` (risk
  classification), plan approval flow (`approve plan`/`cancel plan`), and per-tool
  `ToolCatalogEntry` restrictions already sit between "the LLM asked for a tool" and "the
  tool actually runs" -- this is provider-agnostic today (works identically regardless of
  which `IntelligenceProvider` produced the plan) and doesn't change.
- **Capability awareness (#4):** the vision side already has this shape informally
  (`vision_desktop_capture_enabled`, capability checks before exposing tools) -- Section 3
  below proposes making it an explicit, declared field on each new text provider too.
- **Config keeping API keys out of git (#6):** already solved once for
  `vision_gemini_api_key` (excluded from `MUTABLE_KEYS`/`_ALLOWED_KEYS`, redacted from
  `config show`/`config get`, `config/config.json` is already gitignored per your local
  setup) -- Section 4 replicates this exactly for the new keys.

Not yet present and out of scope for this RFC (flagged for later, per your own
step-ordering): structured cross-provider request logging (#11 observability --
`app/brain/audit/audit_log.py` exists today but records plan-step events only, no
provider/model/latency fields), and a formal `ProviderRouter` that picks providers by task
complexity rather than one static config value (#5's "routing by task type" -- config-based
single-active-provider selection is Phase 1 here; task-aware routing is explicitly your
Stage 6, after at least two cloud providers exist to route between).

## 2. Files that need to change or be created

No file listed here has been modified yet -- this is the change list for your approval.

**New files (one pair per cloud provider, following the existing Gemini-vision pattern):**
- `app/brain/ai/gemini_provider.py` -- `GeminiProvider` implementing `Provider` (chat path)
- `app/brain/ai/openai_provider.py` -- `OpenAIProvider` implementing `Provider`
- `app/brain/ai/anthropic_provider.py` -- `AnthropicProvider` implementing `Provider`
- `app/brain/intelligence/gemini_intelligence_provider.py` -- `GeminiIntelligenceProvider`
  implementing `IntelligenceProvider` (planner path), thin wrapper analogous to
  `OllamaIntelligenceProvider` (or these could live alongside their `Provider` sibling in
  the same file -- open question, see Section 5)
- same pair for OpenAI and Anthropic
- `tests/test_gemini_ai_provider.py`, `tests/test_openai_provider.py`,
  `tests/test_anthropic_provider.py` -- unit tests per provider, mocking HTTP the same way
  `tests/test_ollama_provider.py` and `tests/test_gemini_vision_provider.py` already do
  (no real network calls, no API keys needed to run the suite)
- `docs/RFC-011_TEXT_INTELLIGENCE_PROVIDERS.md` -- this file, to be finalized (not draft)
  once the plan is confirmed and Phase 1-3 are actually implemented

**Existing files that need edits:**
- `app/brain/intelligence/controller.py` -- `_resolve_provider()` gains
  `gemini`/`openai`/`anthropic` branches (mirrors `VisionController.provider()`)
- `app/brain/router.py` / `app/brain/runtime/conversation_runtime.py` -- add a
  `_resolve_provider()`-equivalent factory for the chat path (currently hardcoded, see
  1b); wire the (dead) `ai_provider` config key into it or retire it in favor of reusing
  `intelligence_provider` for both paths (open question, Section 5)
- `config/settings.py` -- add `intelligence_gemini_api_key`,
  `intelligence_openai_api_key`, `intelligence_anthropic_api_key` (credentials, blank
  default, same comment style as `vision_gemini_api_key`); add
  `intelligence_openai_model`/`intelligence_anthropic_model` or reuse the existing
  `intelligence_model` key for whichever provider is active (matches how `vision_model` is
  reused across vision providers today -- recommend reusing `intelligence_model` the same
  way, one fewer config key to maintain)
- `app/brain/configuration/runtime_config.py` -- mirror the inline defaults dict; add the
  three new credential keys to the fallback dict but explicitly **not** to
  `MUTABLE_KEYS`/`_ALLOWED_KEYS`
- `app/brain/configuration/config_commands.py` -- add the three new keys to
  `_REDACTED_CONFIG_KEYS`
- `docs/CONFIGURATION.md` -- document the new keys; also fix the already-stale line 14
  (says "RFC-007A supports only ollama" for `vision_provider`, which stopped being true
  once Gemini vision shipped -- unrelated pre-existing doc drift, noticed during this
  research, harmless to fix in passing)
- `changelog.md` -- new entry once code lands

**Files that do NOT need to change:** `app/brain/intelligence/dynamic_planner.py` (already
fully provider-agnostic), `app/brain/intelligence/task_interpreter.py`/
`app/brain/command_normalizer.py` (deterministic, no LLM calls at all -- this is why the
recent screen-understanding classification fixes were unaffected by which text provider is
active), `app/brain/vision/*` (separate provider axis, untouched), the risk/permission/plan-
approval system (already provider-agnostic).

## 3. Capability declaration (requested architecture point #4)

Propose adding a small, explicit capability marker to each new `IntelligenceProvider`
implementation rather than assuming uniform behavior:

```python
class GeminiIntelligenceProvider:
    name = "gemini"
    capabilities = {"structured_output", "vision"}  # no native tool-calling used here;
                                                       # JARVIS's own JSON-plan-step format
                                                       # is used uniformly across all
                                                       # providers instead, same as Ollama
                                                       # today -- avoids provider-specific
                                                       # tool-calling formats leaking into
                                                       # DynamicPlanner
```

`IntelligenceController._resolve_provider()` can check `intelligence_require_structured_output`
against the selected provider's declared capabilities and fail closed with a clear message
(same "Vision runtime is unavailable right now."-style clear error, not a stack trace) if a
configured provider/model combination can't do what's required -- this already partially
exists as a pattern (`intelligence_fail_closed` config key) and just needs the capability
check wired to it.

## 4. Config surface (concrete, ready to implement once approved)

```
intelligence_provider: "ollama" | "gemini" | "openai" | "anthropic"   (existing key, reused)
intelligence_model: <provider-specific model name>                    (existing key, reused)
intelligence_gemini_api_key: ""     [credential -- config.json edit or GEMINI_API_KEY env]
intelligence_openai_api_key: ""     [credential -- config.json edit or OPENAI_API_KEY env]
intelligence_anthropic_api_key: ""  [credential -- config.json edit or ANTHROPIC_API_KEY env]
```

Same three rules as `vision_gemini_api_key` already established: never in `MUTABLE_KEYS`
(can't be set via voice/`config set`), redacted by `config show`/`config get`, env var
checked automatically when the config value is blank. `GEMINI_API_KEY` env var would be
shared between vision and text use if you don't set `intelligence_gemini_api_key`
separately -- flagging this as intentional reuse, not a bug, unless you'd rather keep them
fully independent (e.g. different Google Cloud projects/quotas for each).

## 5. Open questions needing your confirmation before any code is written

1. **Option A vs Option B** (Section 1c) -- duplicate-wrapper-per-provider now, or merge
   the two Protocols into one shared `LLMProvider` up front? Recommend A.
2. **Which provider first?** All three (Gemini/OpenAI/Claude) in one pass, or one at a
   time so each can be tested on real hardware before the next (matches how Gemini vision
   was done -- one provider, fully verified, before touching anything else)? Recommend one
   at a time; suggest starting with whichever you already have an API key for.
3. **Separate API keys for vision vs. text**, or share one key per provider brand across
   both subsystems (e.g. one `GEMINI_API_KEY` used by both `GeminiVisionProvider` and the
   new `GeminiIntelligenceProvider`)? Recommend sharing by default (simpler), with the
   option to override per-subsystem if you ever need separate quotas/billing.
4. **`intelligence_model` reuse** vs. a separate model-name key per provider (Section 2) --
   recommend reuse, matching the existing `vision_model` convention.
5. Does the plain-chat path (`ConversationRuntime`, Section 1b) need cloud providers in
   this same pass, or is reasoning/planning quality (the thing you actually called
   "тупой") the real target, with plain chat staying on Ollama for now as a cheaper
   default and upgraded later? This changes how much of Phase 1 needs building before you
   see the "smarter" behavior you're after.

## 6. Proposed phased plan (your Stage numbering, adjusted for what already exists)

1. **Stage 1 (mostly done):** `IntelligenceProvider` Protocol already exists; confirm
   Section 5's open questions, no code changes yet.
2. **Stage 2 (skip -- already done):** Ollama is already behind this interface via
   `OllamaIntelligenceProvider`.
3. **Stage 3:** Gemini text provider (`GeminiProvider` + `GeminiIntelligenceProvider`,
   `_resolve_provider()` branch, config keys, tests) -- can reuse
   `app/brain/vision/gemini_provider.py`'s HTTP/error-handling patterns (429/401/403
   handling, urllib-only) almost directly, just against Gemini's text `generateContent`
   call shape instead of the vision one.
4. **Stage 4:** OpenAI text provider, same shape, against `chat/completions` (or
   `responses`) endpoint.
5. **Stage 5:** Anthropic (Claude) text provider, same shape, against the `messages`
   endpoint.
6. **Stage 6:** `ProviderRouter`/task-aware routing (simple task -> Ollama, complex
   reasoning -> cloud, vision -> capable provider, offline -> Ollama) -- meaningful only
   once 2+ cloud providers exist to route between.
7. **Stage 7:** Capability-aware fallback chain (Gemini unavailable -> OpenAI -> Claude ->
   Ollama), configurable, not blind -- builds on Stage 6.
8. **Stage 8:** Tool/function-calling -- deferred; JARVIS's existing JSON-plan-step format
   already serves this role uniformly across providers (see Section 3), so native
   provider tool-calling APIs are not needed unless a specific provider's plan-generation
   quality turns out to depend on using its native tool-calling format specifically --
   revisit only if Stage 3-5 testing shows a real quality gap.
9. **Stage 9:** Memory layer -- separate, pre-existing partial groundwork
   (`memory_max_entries`, `memory_learned_capture_enabled`), not blocked by or blocking
   this RFC.
10. **Stage 10:** Observability -- extend `app/brain/audit/audit_log.py` with
    provider/model/latency/request_id fields once multiple providers exist to compare.

## 7. Testing strategy

Every new provider gets a unit test file that mocks `urllib.request.urlopen` the same way
`tests/test_ollama_provider.py` and `tests/test_gemini_vision_provider.py` already do --
**the full test suite must never require a real API key or network access to pass**, same
rule already in effect for the Gemini vision tests. `tests/test_intelligence_runtime.py`'s
`_resolve_provider()` tests get new cases for each `provider_name` branch, following the
existing `test_desktop_capture_requests_are_no_longer_reported_as_unsupported`-style
before/after pattern. No existing test should need to change unless Option B (Section 1c)
is chosen, in which case `test_conversation_runtime_e2e.py`'s `_FakeProvider` and
`test_intelligence_runtime.py`'s `_StaticProvider` would need to converge on one shape --
flagged as part of that option's cost.

## Acceptance expectations (once implemented, per provider)

1. `config set intelligence_provider gemini` + `config set intelligence_model
   <model-name>` selects it; `config show`/`config get` never reveal the API key.
2. `vision provider check`-equivalent status command reports the text provider as ready
   (mirroring the existing vision command).
3. A planning request ("open browser and search for X", or any existing tested phrase)
   produces a valid plan through the new provider, gated by the exact same approval flow
   as today.
4. Screen-understanding phrases ("что на экране", "что ты видишь") continue to work
   unchanged, regardless of which text provider is active -- proving the two axes
   (vision provider, text/reasoning provider) are genuinely independent.
5. Full test suite passes with no API key configured and no network access.

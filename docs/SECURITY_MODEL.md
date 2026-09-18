# Security Model

JARVIS 2.0 uses a policy-driven architecture.

- The model proposes actions only.
- The validator checks tool names, arguments, and plan structure.
- The registry limits execution to allowlisted tools.
- The executor performs only validated steps.
- Destructive power actions remain behind the existing confirmation flow.

RFC-007B adds a browser-viewport evidence path with these extra boundaries:

- Browser Runtime captures only the visible web-page viewport, not desktop pixels, browser chrome, other windows, camera input, or full-page content outside the viewport.
- Browser captures are referenced only by opaque `capture_id` values. Planner-visible steps cannot use temporary screenshot paths, bytes, or base64 payloads.
- Capture references are scoped to the current request and task, validated before use, and rejected when stale, expired, cross-session, or cross-request.
- Vision analysis for browser captures is loopback-only. Non-loopback Vision provider URLs remain rejected.
- Browser-page text, OCR text, visual descriptions, SVG/canvas content, QR payloads, and prompt-injection text are untrusted data. They may be reported, but they cannot modify the task, change risk, add tools, approve plans, or trigger Terminal, Filesystem, or Browser actions.
- CAPTCHA may be reported as suspected, but JARVIS does not solve, click, bypass, or outsource it in RFC-007B.

RFC-007C adds one-shot desktop and window capture with these extra boundaries:

- `desktop.capture_screen` and `desktop.capture_window` are always MEDIUM risk and always require explicit `approve plan`. They are absent from every project-scoping exemption in the risk analyzer, so they can never become project-scoped and can never auto-execute, even with `developer_mode` and `auto_execute_medium_project` both enabled. This is enforced code, verified by a regression test, not a documentation-only claim.
- A single agent plan may contain at most `MAX_DESKTOP_CAPTURES_PER_PLAN` (currently 1) `desktop.capture_screen`/`desktop.capture_window` steps, combined. The plan validator rejects a plan that exceeds this the same way it rejects a plan exceeding the memory-recall cap or the overall step cap.
- `desktop.capture_window` revalidates the target window immediately before capturing: it must still exist, still be visible, and its current title must still exactly match the title shown to the user in the approved plan's description. Windows can recycle a closed window's handle for an unrelated window; checking existence and visibility alone would not catch that reuse, so the title is checked explicitly and the capture is rejected with a clear reason if it no longer matches.
- Desktop and window captures are referenced only by opaque `capture_id` values, kept in a store separate from browser captures. Planner-visible steps cannot use capture file paths, bytes, or base64 payloads.
- Capture references are scoped to the current request and task, validated before use, and rejected when stale, expired, or cross-task.
- Vision analysis of desktop/window captures uses the same simple direct-provider call as local-file Vision analysis. There is no DOM-grounding or crop-verification pipeline, because a desktop or window capture has no DOM to cross-check against.
- Desktop capture is a single explicit, one-shot action: there is no loop, no repeated automatic capture, and no background or scheduled capture.
- On a non-Windows platform, desktop capture fails explicitly with a clear error message. It never silently no-ops and never pretends a capture happened.

RFC-010 adds phone location and turn-by-turn navigation over Tailscale with these extra boundaries:

- The `/location/overland` route binds only to `location_bind_host`, never `0.0.0.0`/`::`. Startup fails with a clear error, without crashing the rest of JARVIS, if that address is empty, a wildcard, or not present on this machine — it never silently falls back to a public bind.
- Every `/location/overland` POST must carry the correct `Authorization: Bearer <location_shared_secret>` token. A request without it is rejected with 401, logged via `record_audit_event`, and never touches stored state — being on the Tailscale network is necessary but not sufficient.
- `location_shared_secret` is excluded from the `config set` mutable-key allowlist and from `config show`/`config get` output (both redact it), so it cannot be set or read back through the normal live configuration surface — only by editing `config/config.json` directly.
- `location.receive_overland_point` is intentionally not registered as an AI-callable tool. The only way a location point enters JARVIS's state is the authenticated HTTP route above; no agent plan, however constructed, can inject or forge one.
- Only the single freshest point per device is ever stored; there is no history and no tool or command that returns one.
- `location.save_place` is `persistent_write`/MEDIUM risk, the same tier as `memory.remember`, and is absent from every project-scoping exemption in the risk analyzer, so it never auto-executes even with `developer_mode` and `auto_execute_medium_project` both enabled. `location.where_am_i`, `location.distance_to`, and every `navigation.*` tool are LOW risk: read-only or ephemeral-session operations against the owner's own data, with no persistent write.
- A `navigation.start` session is request/task-scoped exactly like a Vision capture, and is deleted on task completion, failure, cancellation, and approval expiration — all four paths, verified by a regression test, after this exact class of gap (a cleanup call wired into `AgentController` but missed in `app.brain.planner.approval`'s separate approval-expiration path) previously affected both browser and desktop Vision captures.
- `navigation.get_next_instruction` is pull-only: it recomputes from whichever point is freshest at the moment it's called. There is no server-side background poller re-routing on a timer.

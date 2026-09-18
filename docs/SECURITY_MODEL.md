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

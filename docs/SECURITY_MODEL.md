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

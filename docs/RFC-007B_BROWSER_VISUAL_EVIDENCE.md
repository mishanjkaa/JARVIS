# RFC-007B Browser Visual Evidence

RFC-007B extends the accepted Browser Runtime and Vision Runtime by letting JARVIS capture the visible viewport of a JARVIS-owned browser tab, analyze that capture locally, and return grounded visual evidence.

## Supported in RFC-007B

- `browser.capture_view`
- `vision.describe_browser_capture`
- `vision.extract_text_from_browser_capture`
- `vision.find_visual_element_in_browser_capture`
- temporary-session workflows:
  - `browser.start_session`
  - `browser.open_url`
  - `browser.capture_view`
  - exactly one browser-capture Vision tool
  - `browser.close_session`
- follow-up active-session workflows:
  - `browser.get_active_session`
  - `browser.capture_view`
  - exactly one browser-capture Vision tool
- `vision browser status`
- `vision captures`
- `vision clear captures`

## Not supported in RFC-007B

- full-page capture
- desktop or window capture
- camera input
- coordinate clicking
- GUI control
- login, password entry, payment, or CAPTCHA solving
- external image upload
- planner-visible capture paths or image bytes

## Capture lifecycle

Each temporary browser capture receives an opaque `capture_id` and private metadata:

- owner request ID
- owner agent task ID
- source session/tab
- URL and origin
- page version
- viewport size
- capture time
- expiration time
- image format
- digest and byte size

Captures are private to the current request/task, expire after a bounded TTL, and are deleted after:

- successful task completion
- execution failure
- cancellation
- approval expiration before reuse
- browser-session closure
- explicit `vision clear captures`
- process restart

## Security and privacy

- Browser captures are analyzed only through the loopback Vision provider.
- Planner-visible tool results never contain capture bytes, base64, or temporary paths.
- `vision captures` exposes metadata only.
- Browser page text and OCR are information only. They have no instruction authority.
- CAPTCHA may be reported as suspected, but JARVIS does not solve or bypass it.

## Owner verification

Owner verification should confirm:

1. `vision browser status` reports the integration state cleanly.
2. `Open https://example.com and visually describe the page.` produces a browser-start/open/capture/analyze/close plan.
3. `Read the visible text on the current browser page.` reuses an active session when appropriate.
4. `Find the search field on the current page.` returns grounded visual-element evidence only.
5. `vision captures` shows metadata only and never reveals image bytes or paths.
6. Temporary captures are cleared after task completion and after `vision clear captures`.

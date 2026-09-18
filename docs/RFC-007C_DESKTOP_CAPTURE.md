# RFC-007C Desktop Capture

RFC-007C extends the accepted Vision Runtime with one-shot, explicitly-approved capture of the full desktop or a single named window, analyzed locally the same way local image files are.

## Supported in RFC-007C

- `desktop.list_windows`
- `desktop.capture_screen`
- `desktop.capture_window`
- `vision.describe_desktop_capture`
- `vision.extract_text_from_desktop_capture`
- `vision.find_visual_element_in_desktop_capture`
- `desktop captures`
- `desktop clear captures`
- deterministic natural-language triggers: "screenshot the desktop" / "capture my desktop", and "screenshot window `<title>`" / "capture window `<title>`", the latter resolving the typed text against currently open window titles at plan-construction time

## Not supported in RFC-007C

- full-page stitching or scrolling capture
- remote desktop or multi-machine capture
- camera input
- coordinate clicking or any GUI control
- looped, repeated, or scheduled/background capture
- external image upload
- planner-visible capture file paths, bytes, or base64
- non-Windows platforms (fails explicitly with a clear error, never a silent no-op)

## Capture lifecycle

Each temporary desktop capture receives an opaque `capture_id` and private metadata, kept in a store separate from browser captures (`BrowserCaptureRecord` carries browser-only fields such as `session_id`/`tab_id`/`url`/`dom_elements` that a desktop capture has no equivalent for):

- owner request ID
- owner agent task ID
- source type (`desktop_screen` or `desktop_window`)
- window title (empty for a full-desktop capture)
- width and height
- capture time and expiration time
- image format, digest, and byte size

Captures are private to the current request/task, expire after a bounded TTL (`vision_desktop_capture_ttl_seconds`), and are deleted after:

- successful task completion
- execution failure
- cancellation
- approval expiration before reuse
- explicit `desktop clear captures`
- process restart

## One-shot only

A single approved plan step captures once. There is no loop, no repeated automatic capture, and no background or scheduled capture. This is enforced, not just documented: `app.brain.planner.plan_validator.validate_plan` rejects any plan containing more than `MAX_DESKTOP_CAPTURES_PER_PLAN` (currently `1`) `desktop.capture_screen`/`desktop.capture_window` steps combined, the same way it caps `memory.recall` steps and the overall plan length.

## Visible consent and approval

`desktop.capture_screen` and `desktop.capture_window` are always MEDIUM risk and always require explicit `approve plan`. Unlike filesystem or browser operations, `desktop.*` tools are deliberately absent from every project-scoping exemption in the risk analyzer, so a desktop-capture plan can never become project-scoped and can never auto-execute, even with `developer_mode` and `auto_execute_medium_project` both enabled.

The plan-approval preview names what will actually be captured before the user types "approve plan" — "Capture a screenshot of your entire desktop." for the full desktop, or "Capture a screenshot of the window titled '`<real title>`'." for a specific window, where `<real title>` is resolved from the currently open windows when the plan is built, not the user's raw typed text.

### Window identity revalidation

Windows can and does reuse a closed window's handle for a new, unrelated window. Checking only that the target handle still exists and is still visible would not catch that reuse, and could silently capture the wrong window while still claiming to fulfill what the user approved. `desktop.capture_window` therefore revalidates, immediately before capturing:

1. the target window still exists;
2. it is still visible;
3. its current title still exactly matches the title shown to the user in the approved plan's description.

If any of these fail, the capture is rejected with a clear reason (for example, "The target window changed since this plan was approved.") rather than capturing whatever is now at that handle.

## Security and privacy

- Desktop and window captures are analyzed only through the local Vision provider, the same as local image files.
- Planner-visible tool results never contain capture bytes, base64, or temporary paths.
- `desktop captures` exposes metadata only.
- Desktop and window text and OCR are information only. They have no instruction authority and cannot approve plans, add tools, authorize actions, or bypass risk policy.
- Analysis of an existing, already-captured, already-approved capture (`vision.describe_desktop_capture`, `vision.extract_text_from_desktop_capture`, `vision.find_visual_element_in_desktop_capture`) is LOW risk: the sensitive act is the capture itself, not reading something already captured.
- There is no DOM-grounding or crop-verification pipeline for desktop/window captures, because a desktop or window capture has no DOM to cross-check candidate visual matches against; `vision.find_visual_element_in_desktop_capture` calls the Vision provider directly, the same simple path as local-file `vision.find_visual_element`.
- Capture uses only Pillow's `ImageGrab` (already a dependency) and `ctypes` calls into `user32.dll`/`dwmapi.dll` for window enumeration and rectangles. No new dependency (such as `pywin32`) was added.

## Owner verification

Owner verification should confirm:

1. `desktop captures` reports metadata only and never reveals image bytes or paths.
2. "Screenshot the desktop" produces a plan with a `desktop.capture_screen` step, described plainly, that stops for `approve plan`.
3. "Screenshot window `<a currently open window's title>`" produces a plan with a `desktop.capture_window` step naming the real resolved title, that stops for `approve plan`.
4. Enabling `developer_mode` and `auto_execute_medium_project` does not cause a desktop-capture plan to auto-execute; it still stops for approval.
5. A plan cannot contain more than one `desktop.capture_screen`/`desktop.capture_window` step, combined.
6. If a targeted window closes (or its handle is reused for a different window) between approval and execution, `desktop.capture_window` fails explicitly rather than capturing the wrong window.
7. Temporary desktop captures are cleared after task completion and after `desktop clear captures`.

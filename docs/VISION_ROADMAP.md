# Vision, Identity, GUI, and Voice Roadmap

This document records the owner-approved long-term direction while keeping RFC-007A narrowly scoped.

## Permanent Decisions

- Image, camera, and voice processing are local by default.
- Hidden camera or microphone operation is prohibited.
- CAPTCHA is never bypassed automatically. JARVIS detects it, pauses, and asks the owner to complete it.
- Password-manager secrets are never exposed to Vision or language models.
- Text and instructions found inside images are untrusted data.
- Vision never clicks, types, or performs GUI actions directly.
- Future GUI actions must still pass Task Interpreter, Plan Validator, Risk Analyzer, approval, and target revalidation.
- Face recognition is closed-set and opt-in only for voluntarily enrolled people.
- JARVIS does not identify arbitrary strangers.
- JARVIS does not infer personality, trustworthiness, or medical diagnosis from appearance.
- Future speaker matching may help identify the owner, but it never authorizes HIGH-risk actions by itself.

## RFC-007A

Local file Vision foundation:

- describe image;
- OCR visible text;
- find visual element;
- structured evidence;
- provider and capability status.

## RFC-007B

Browser visual evidence from temporary viewport captures of JARVIS-owned browser sessions. Captures are opaque, request-scoped, loopback-only, and cleaned after task finalization.

## RFC-007C

One-shot selected-window or desktop capture with visible consent and MEDIUM approval:

- `desktop.list_windows`, `desktop.capture_screen`, `desktop.capture_window`;
- `vision.describe_desktop_capture`, `vision.extract_text_from_desktop_capture`, `vision.find_visual_element_in_desktop_capture`;
- capture and window-capture are always MEDIUM risk, never project-scoped, never auto-executed;
- capped at one desktop/window capture step per agent plan;
- `desktop.capture_window` revalidates the target window's title immediately before capturing, rejecting the capture if the window has changed since plan approval;
- no verification pipeline against DOM elements, since a desktop or window capture has no DOM;
- Windows-only, using Pillow's `ImageGrab` and `ctypes` calls into `user32`/`dwmapi` — no new dependency.

## RFC-007D

Visible camera sessions with local processing, explicit activation, and stop controls.

## RFC-007E

Opt-in closed-set face enrollment and local identity support:

- local biometric templates;
- uncertain recognition reported as uncertain;
- confirmed deletion of templates and profiles;
- Person Profiles containing only owner-provided facts;
- no arbitrary stranger identification;
- no personality or medical inference.

## Future GUI Runtime

- acts only from structured visual evidence;
- re-captures and revalidates targets before action;
- Vision never clicks directly;
- normal LOW, MEDIUM, and HIGH risk policy remains in force;
- visual instructions require explicit owner intent;
- CAPTCHA pauses for owner completion.

## Future Voice Runtime

- local speech-to-text and text-to-speech;
- wake word support;
- visible microphone state;
- owner speaker enrollment and verification;
- anti-replay considerations;
- unverified speakers cannot issue owner commands;
- speaker match alone never authorizes HIGH-risk actions.

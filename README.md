# JARVIS

JARVIS 2.0 is a hybrid assistant that combines deterministic commands with an optional local AI fallback.

## Features

- Offline deterministic mode
- Optional local AI mode through Ollama
- Safe tool registry and policy checks
- Local Browser Runtime for policy-guarded web navigation and interaction
- Local Vision Runtime for trusted image description, OCR, visual-element search, browser-viewport visual evidence, and one-shot desktop/window capture (Windows only)
- Phone location and turn-by-turn navigation over a private Tailscale network, using the owner's own phone (via the Overland app) as the GPS source and OpenStreetMap/OSRM for routing
- On-demand voice input/output (local speech-to-text/text-to-speech, owner-only speaker verification), from the local PC mic or a phone push-to-talk page over the same Tailscale network
- Existing memory, notes, tasks, calculator, and folder features remain available

## Setup

1. Run the project with Python 3.14.
2. Configure AI settings in config/config.json if you want to enable local AI.
3. Use ai on to enable the runtime AI path.

## Test command

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Example commands

- hello
- open desktop
- calculate 25 * 4
- remember project = Python
- recall project
- forget project
- memory list
- memory list learned
- screenshot the desktop
- screenshot window Notepad
- desktop captures
- location status
- navigation status
- voice status
- voice enroll
- voice talk
- ai on

## Safety limitations

- The local AI layer can propose actions, but only the trusted tool registry can execute them.
- Power actions require existing confirmation commands.
- Ollama is optional and not required for deterministic operation.
- Vision in RFC-007A accepts only explicitly named trusted-root image files and never uploads them to external services.
- RFC-007B adds temporary browser viewport captures with opaque `capture_id` handles, local-only analysis, TTL cleanup, and no planner-visible screenshot paths or bytes.
- RFC-007C adds one-shot desktop/window capture (Windows only, no new dependency), always MEDIUM risk, never auto-executed, capped at one capture per plan, with the target window's title revalidated immediately before capture to catch Windows reusing a closed window's handle.
- RFC-010 adds phone location and turn-by-turn navigation. The location HTTP endpoint binds only to a configured Tailscale interface IP, never `0.0.0.0`, and requires a shared-secret bearer token; only the single freshest point per device is ever stored, never a history; `location.receive_overland_point` is not an AI-callable tool.
- RFC-009 adds on-demand voice. Only the enrolled owner's voice is ever processed — speaker verification runs before transcription, so an unverified voice produces no transcript at all, not merely a discarded one. Voice input carries no special authority: a transcript is handed to the exact same command path typed input uses, so risk classification and `approve plan` apply identically. No raw audio is ever persisted; anti-replay/liveness detection is explicitly out of scope for this version, documented rather than silently assumed solved.

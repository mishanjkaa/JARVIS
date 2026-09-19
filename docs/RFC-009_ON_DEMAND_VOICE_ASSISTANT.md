# RFC-009 On-Demand Voice Assistant

RFC-009 gives JARVIS a voice input/output channel: the owner can talk to JARVIS instead of
typing, and get spoken answers back, including free-form "help me think through this"
requests (for example working out compromise options for a disagreement the owner
describes). It reuses the existing Task Interpreter / Planner / Risk Analyzer / Agent
Runtime pipeline exactly as every other input channel does — voice is a new way to say
things to JARVIS, not a new way to bypass how JARVIS decides what to do.

## Hard boundary (read this first) — opt-in as of the second real-hardware round

RFC-009 was originally designed to only ever process the enrolled owner's own speech,
enforced unconditionally by speaker verification before transcription. After extensive
real-hardware testing (see `changelog.md`'s "RFC-009 verification threshold, second
recalibration" and the entries above it) showed the enrolled owner's *own* genuine
similarity score swinging widely (0.23–0.68) even after every mechanical cause was found
and fixed, the owner explicitly asked for plain voice control without speaker
recognition: JARVIS should hear and act on any voice, not gate every utterance behind a
verification step that was rejecting the owner's own genuine speech as often as it
accepted it.

This is now controlled by `voice_require_speaker_verification` (default `false`): with it
off, `voice talk`/`/voice/turn` transcribe and route any voice exactly like typed input,
and no enrollment is required at all. Setting it back to `true` (`config set
voice_require_speaker_verification true`) restores the original behavior below in full —
the enrollment/verification code was not removed, only made optional. Whoever runs this
with the default off should understand what changes: the two reasons below, which
justified making this unconditional, still apply whenever more than one person's voice
might reach the microphone.

1. **Consent.** Processing another identifiable person's speech (their voice is personal
   data) without their knowledge has real legal exposure that varies by jurisdiction —
   two-party consent / wiretapping-style rules exist in many places, GDPR treats voice as
   personal data regardless of whether it's stored, and "we didn't save the recording"
   does not make the initial capture-and-processing step consent-free. This isn't
   something a codebase decision can wave away; it needs the owner's own legal judgment
   for their own jurisdiction, and defaulting to "don't capture third parties" is the only
   choice that doesn't require that judgment call up front.
2. **Consistency with the project's own rules.** `docs/VISION_ROADMAP.md`'s Permanent
   Decisions already say "unverified speakers cannot issue owner commands" and "speaker
   match alone never authorizes HIGH-risk actions" — the project has already committed to
   "only the verified owner's voice matters." RFC-009 extends that same line to "only the
   verified owner's voice is processed at all."

**What this means in practice for the "help me in an argument" use case:** the owner
describes the situation to JARVIS in their own words — "I'm arguing with a friend about X,
he wants A, I want B, what's a fair middle ground" — the same way they'd type it, just
spoken instead. JARVIS never needs to hear the other person to help with this; it needs the
owner's account of the disagreement. If you want JARVIS to literally listen to both people
talking in real time, that is a different, explicitly out-of-scope feature (see "Not
supported") that would need its own RFC, its own mutual-consent flow (both people verbally
agreeing on the record, similar to how meeting-recording tools announce "this call is being
recorded"), and your own legal sign-off for Latvia specifically before it's worth spending
engineering time on.

## Transport: phone as the remote mic/speaker, over Tailscale (owner-confirmed)

The owner wants to talk to JARVIS while out walking, without carrying the PC. JARVIS's
STT/TTS/planner stay on the PC exactly as designed; the phone becomes a thin remote
mic/speaker client, reusing the same Tailscale network as RFC-010's location transport:

- the phone runs a small client (a PWA/web page is enough to start — modern mobile Safari
  and Chrome can request microphone access and play audio from a page over HTTPS; a native
  app is a later upgrade, not a blocker) that opens a connection to JARVIS's Tailscale
  address
- push-to-talk from the phone (a button in the page) captures audio, streams it to the PC
  over that Tailscale connection, the PC runs the same local STT/verification/planner/TTS
  pipeline it would for a local mic, and streams the spoken reply back to the phone
- wake-word detection, to avoid draining the phone's battery and data running a
  continuous stream, runs *on the phone* for the remote case (a small local wake-word
  model, e.g. via the browser's WebAssembly-based options, or deferred to push-to-talk only
  in the first version) rather than streaming raw audio to the PC nonstop just to detect a
  trigger phrase
- exactly the same owner-speaker-verification and risk/approval rules apply regardless of
  whether the mic is local (PC) or remote (phone) — "it came in over the phone client"
  changes nothing about who is allowed to act as the owner or what still needs `approve
  plan`
- the phone-to-PC audio channel is authenticated the same way as RFC-010's location route
  (Tailscale network membership plus a shared secret/token), and — like the location
  route — bound to the Tailscale interface only, never exposed on the public internet

## Supported in RFC-009

- local wake-word detection, always running, matching only against the trigger phrase;
  everything that isn't a wake-word match is discarded immediately and never transcribed
  or stored
- push-to-talk as an alternative to wake-word (a command or hotkey starts a voice turn
  directly)
- local speech-to-text of the owner's utterance for one voice turn (one question/command,
  the same size as one typed message)
- local text-to-speech for JARVIS's spoken reply
- local owner speaker verification, now opt-in via `voice_require_speaker_verification`
  (default `false`, see "Hard boundary" above): when enabled, the enrolled owner's voice
  is required before an utterance is treated as input at all
- free-form "help me think through this" requests, handled exactly like a typed request —
  including disagreement/compromise/decision-support questions where the owner narrates
  the situation
- conversational memory phrasing ("джарвис, запомни это", "запомни, что...", or any other
  natural phrasing, not only the fixed `remember <key> = <value>` deterministic command) —
  this needs no new memory code, since a voice transcript is handed to the exact same
  text-intake path (`app.brain.router.route_command`) a typed message would use
- `voice status` — reports whether voice is enabled, whether the mic is currently active,
  and whether the owner's voice is currently enrolled
- `voice on` / `voice off` — explicit enable/disable, mirroring the existing `ai on`/
  `developer mode on` session-toggle pattern
- `voice enroll` / `cancel voice enrollment` / `voice forget me` — closed-set, opt-in
  enrollment and confirmed deletion, mirroring the face-enrollment shape already described
  as a permanent decision in `docs/VISION_ROADMAP.md`
- `voice talk` — push-to-talk from the local PC microphone
- `voice listen` — continuous push-to-talk: repeats the same turn `voice talk` does, one
  after another, until a stop phrase or Ctrl+C, so the owner doesn't have to type `voice
  talk` before every utterance. Not wake-word detection (see "Not supported" below) — it
  is still push-to-talk under the hood, just automatically re-armed after each turn.
  `voice_listen_on_startup` (default `true`) enters this loop automatically right after
  JARVIS starts, alongside `ai_enabled`/`voice_enabled` now also defaulting to `true`, so a
  fresh launch needs none of `ai on`/`voice on`/`voice talk` typed first — another
  owner-requested friction reduction for a single-user local device, in the same vein as
  the speaker-verification opt-in change above.

## Not supported in RFC-009

- capturing, transcribing, or analyzing any voice other than the enrolled owner's --
  **only while `voice_require_speaker_verification` is enabled**; with it at its default
  `false`, any voice reaching the microphone is transcribed and treated as input, which is
  the owner's explicit, deliberate choice for a single-user local device (see "Hard
  boundary" above)
- treating a second/unverified voice as a command or as input of any kind -- same
  qualification as above
- continuous ambient transcription without a wake-word match or an explicit push-to-talk
  session
- recording or persisting raw audio; only the text of what was actually transcribed and
  acted on is kept, subject to the same memory rules (retention, redaction) as any other
  input
- speaker verification alone authorizing a MEDIUM or HIGH risk action — voice input still
  goes through the same Risk Analyzer and approval flow as typed input; "it was spoken by
  the verified owner" answers *who is asking*, never *whether the action is safe to
  auto-run*
- multi-party conversation transcription or mediation (see "Hard boundary" above)
- anti-replay / liveness detection is out of scope for this first version — flagged here so
  it isn't silently assumed solved; a played recording of the owner's voice may currently
  pass verification, so voice alone should not gate anything the project wouldn't already
  let a LOW-risk action do
- wake-word detection in this first version (deferred; see "Implementation notes" below) —
  both the local PC and the phone client are push-to-talk only for now. `voice listen`
  (above) covers the "don't make me type a command every time" friction without wake-word
  detection: it is a loop of push-to-talk turns, not an always-on model listening for a
  trigger phrase, so it still uses the microphone for its full capture window each turn
  rather than only reacting to a spoken name

## Visible state

- The mic-active state is exposed the same way camera state is under RFC-007D: `voice
  status` is always accurate, and there is a visible (not just logged) indicator while
  audio is actively being captured for a voice turn — never a state where JARVIS is
  listening to content without something on-screen saying so.

## Owner verification (when `voice_require_speaker_verification` is enabled)

1. `voice status` reports enabled/disabled, mic-active state, speaker-verification
   requirement, and enrollment state cleanly.
2. Speaking a request in a voice not matching the enrolled owner produces no action and no
   transcript is kept of it.
3. A described (not overheard) disagreement — "my friend and I disagree about X, he says A,
   I say B" — produces a normal planner response with compromise suggestions, entirely
   through the owner's own narrated speech.
4. A MEDIUM/HIGH-risk action requested by voice still stops for `approve plan` exactly like
   the same request typed would.
5. No code path transcribes or retains audio once a voice turn ends, beyond the same text
   record any other input leaves.

## Implementation notes (Phase 0 outcomes)

These are the concrete choices made when RFC-009 was implemented, recorded here since the
RFC above intentionally didn't name specific libraries:

- **Speech-to-text**: `faster-whisper` (CTranslate2/Whisper weights), CPU, `small` model by
  default (`voice_stt_model`).
- **Text-to-speech**: Piper (`piper-tts`). Its actively maintained fork is licensed
  GPL-3.0-or-later (the original MIT `rhasspy/piper` is archived) — a real license
  consideration the owner explicitly accepted for the quality/Russian-voice-support
  tradeoff. `voice_tts_voice` must point at a downloaded Piper `.onnx` voice model; JARVIS
  does not download one automatically, the same convention as Ollama models.
- **Speaker verification**: SpeechBrain's pretrained ECAPA-TDNN
  (`speechbrain/spkrec-ecapa-voxceleb`), not Resemblyzer as first considered — Resemblyzer's
  hard dependency on `webrtcvad` has no prebuilt Windows wheel and needs a C compiler that
  wasn't available in the target environment; SpeechBrain's dependency chain avoids that
  package entirely. The model downloads from Hugging Face Hub on first use.
- **Wake-word**: deferred entirely for this version — both the local PC and the phone
  client are push-to-talk only. `openWakeWord` (Apache-2.0, no account/key needed, unlike
  Porcupine) is the intended engine for a later phase, but training/tuning a custom
  "джарвис" wake word is real effort not included here.
- **Phone transport**: `/voice/turn` and a static push-to-talk page at `/voice/client` are
  added to RFC-010's existing Tailscale-bound HTTP listener
  (`app/brain/location/server.py`, generalized into a small path-registry dispatcher for
  this), reusing `location_bind_host`/`location_port`/`location_shared_secret` rather than
  opening a second listener. The phone client records raw PCM via the Web Audio API and
  encodes it to WAV in-browser (not `MediaRecorder`'s compressed webm/opus), so the server
  never needs an audio-transcoding dependency; the server resamples to 16kHz itself
  (`torchaudio`, already a dependency for SpeechBrain) since a browser's native sample rate
  varies by device.
- Enrollment requires `MIN_ENROLLMENT_SAMPLES` (3) local-mic samples, banked one at a time
  across separate `voice enroll` calls and averaged into a single stored embedding
  (`data/voice_profile.json`) — no raw audio is kept, only the derived embedding.

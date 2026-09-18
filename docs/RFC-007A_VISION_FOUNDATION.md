# RFC-007A Vision Foundation

RFC-007A implements the first local Vision Runtime for explicitly requested image files inside trusted roots.

## Scope

Supported in RFC-007A:

- describe a trusted local image;
- extract visible text from a trusted local image;
- find a visually described element in a trusted local image;
- return grounded structured Vision evidence;
- report local Vision provider status.

Not supported in RFC-007A:

- browser screenshots as Vision input;
- desktop or window capture;
- camera input;
- video, PDF, SVG, GIF, or remote image URLs;
- face recognition or enrollment;
- Person Profiles;
- voice or speaker recognition;
- GUI actions;
- automatic URL opening, QR execution, or shell execution from image contents;
- saving persistent analysis reports.

## Architecture

Pipeline:

User request
→ Task Interpreter
→ Dynamic Planner
→ Plan Validator
→ Risk Analyzer
→ Agent Runtime
→ Vision Controller
→ local Vision Provider
→ structured Vision Evidence
→ Goal Evaluator

The Vision Runtime integrates only through the existing tool registry and Agent Runtime. It does not bypass planning, risk review, audit, or grounding.

## Safety Rules

- local files only, inside trusted roots;
- supported formats only: PNG, JPEG, WebP;
- signature validation before provider use;
- path traversal and symlink escape rejected;
- file-size and pixel-count limits enforced before provider invocation;
- image contents are untrusted data;
- OCR text, visible instructions, QR payloads, and URLs discovered in images are never treated as commands;
- no external image transmission;
- no base64 image bytes or raw provider prompts in logs or audit;
- sensitive-looking OCR content is redacted from persistence and diagnostics;
- temporary Vision evidence expires from memory.

## Tool Surface

- `vision.describe_image`
- `vision.extract_text`
- `vision.find_visual_element`
- `vision status`
- `vision provider status`
- `vision provider check`

Each tool uses strict argument validation and read-only execution.

## Acceptance Expectations

Owner verification for RFC-007A should confirm:

1. trusted local image analysis works through the Agent Runtime;
2. missing-path requests ask for the exact image path;
3. unsupported camera/desktop/face/voice requests fail closed;
4. OCR output is grounded and sensitive-looking segments are redacted;
5. no Browser, Filesystem, Terminal, Risk, or approval regressions were introduced.

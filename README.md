# JARVIS

JARVIS 2.0 is a hybrid assistant that combines deterministic commands with an optional local AI fallback.

## Features

- Offline deterministic mode
- Optional local AI mode through Ollama
- Safe tool registry and policy checks
- Local Browser Runtime for policy-guarded web navigation and interaction
- Local Vision Runtime for trusted image description, OCR, visual-element search, and browser-viewport visual evidence
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
- ai on

## Safety limitations

- The local AI layer can propose actions, but only the trusted tool registry can execute them.
- Power actions require existing confirmation commands.
- Ollama is optional and not required for deterministic operation.
- Vision in RFC-007A accepts only explicitly named trusted-root image files and never uploads them to external services.
- RFC-007B adds temporary browser viewport captures with opaque `capture_id` handles, local-only analysis, TTL cleanup, and no planner-visible screenshot paths or bytes.

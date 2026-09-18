# JARVIS 2.0 Architecture

JARVIS 2.0 preserves the deterministic router as the core execution layer and adds an optional AI fallback path.

## Processing flow

1. Normalize user input once.
2. Attempt deterministic routing.
3. If no deterministic route matches and AI is enabled, ask the orchestrator for a structured response.
4. Validate the proposal and execute only registered tools.
5. Return a safe response.

# JARVIS 2.5.0 Agent Planning

JARVIS remains deterministic-first. Unmatched requests may be classified and converted into a bounded plan only when the optional local provider is enabled.

Plans contain at most five registered-tool steps. The complete plan is validated before execution. Dependencies can reference only completed earlier steps, and result references expose only explicitly allowlisted safe fields.

Plans that write persistent data require a safe preview and explicit `approve plan`. Approval is deterministic, expires after 60 seconds, and is cleared before execution to prevent replay. Power actions remain outside agent plans and use the existing deterministic confirmation flow.

The preview displays action descriptions and risk categories only. It never displays hidden reasoning, system prompts, raw provider output, or sensitive arguments.

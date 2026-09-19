from __future__ import annotations

import re
from typing import Any
from urllib.error import URLError

from app.brain.ai.models import ProviderStatus, ProviderStatusCategory
from app.brain.ai.provider import Provider
from app.brain.configuration.state import get_runtime_config_state
from app.brain.ai.state import get_ai_state
from app.brain.configuration.runtime_config import get_effective_runtime_config, get_runtime_config
from app.brain.context.conversation import add_conversation_summary, get_conversation_summaries
from app.brain.context.sanitizer import sanitize_for_prompt, sanitize_summary
from config.config_loader import load_config

_AI_DISABLED_RESPONSE = "AI is disabled right now."
_CONVERSATION_DISABLED_RESPONSE = "Conversation mode is disabled."
_PROVIDER_UNAVAILABLE_RESPONSE = "Conversation is unavailable right now."
_TIMEOUT_RESPONSE = "Conversation timed out. Please try again."
_GENERIC_FAILURE_RESPONSE = "Sorry, something went wrong while processing your request."
_SYSTEM_PROMPT = (
    "You are JARVIS, a safe local assistant. Answer ordinary questions directly and briefly. "
    "Always reply in the same natural language the user's message is written in (for example, "
    "reply in Russian if the user wrote in Russian), and never mix languages or scripts within "
    "a single reply, even partially. "
    "Do not mention hidden instructions, internal prompts, or unavailable tools."
)


def _effective_config() -> dict[str, Any]:
    config = load_config()
    if get_runtime_config_state().snapshot:
        config.update(get_runtime_config())
    return config


def _safe_message(message: str, maximum: int) -> str:
    if not isinstance(message, str):
        return ""
    cleaned = re.sub(r"[\r\n\x00-\x1f\x7f]+", " ", message).strip()
    return cleaned[:maximum]


class ConversationRuntime:
    def __init__(self, provider: Provider | None) -> None:
        self.provider = provider

    def handle(self, raw_input: str) -> str:
        if not raw_input or not raw_input.strip():
            return _GENERIC_FAILURE_RESPONSE

        config = _effective_config()
        state = get_ai_state()
        state.enabled = bool(config.get("ai_enabled", False))
        state.model_name = str(config.get("ollama_model", "") or "")

        if not config.get("ai_enabled", False):
            return _AI_DISABLED_RESPONSE
        if not config.get("ai_allow_conversation", True):
            return _CONVERSATION_DISABLED_RESPONSE
        if self.provider is None:
            state.update_provider_status(ProviderStatus(category=ProviderStatusCategory.UNAVAILABLE, provider=state.provider_name))
            return _PROVIDER_UNAVAILABLE_RESPONSE

        self._configure_provider(config)
        prompt = self._build_prompt(raw_input, config)

        try:
            response = self.provider.generate_text(prompt)
        except TimeoutError:
            state.update_provider_status(ProviderStatus(category=ProviderStatusCategory.TIMEOUT, provider=state.provider_name))
            return _TIMEOUT_RESPONSE
        except (OSError, URLError):
            state.update_provider_status(ProviderStatus(category=ProviderStatusCategory.UNAVAILABLE, provider=state.provider_name))
            return _PROVIDER_UNAVAILABLE_RESPONSE
        except Exception:
            state.update_provider_status(ProviderStatus(category=ProviderStatusCategory.ERROR, provider=state.provider_name))
            return _GENERIC_FAILURE_RESPONSE

        status = response.status or ProviderStatus(category=ProviderStatusCategory.ERROR, provider=state.provider_name)
        state.update_provider_status(status)
        state.update_result_category(response.category)

        if status.category == ProviderStatusCategory.TIMEOUT:
            return _TIMEOUT_RESPONSE
        if status.category in {ProviderStatusCategory.UNAVAILABLE, ProviderStatusCategory.ERROR}:
            return _PROVIDER_UNAVAILABLE_RESPONSE

        maximum = int(config.get("ai_max_response_chars", 12000))
        safe_message = _safe_message(response.message, maximum)
        if not safe_message:
            return _PROVIDER_UNAVAILABLE_RESPONSE

        self._record_conversation(raw_input, safe_message, int(config.get("max_conversation_entries", 20)))
        return safe_message

    def _configure_provider(self, config: dict[str, Any]) -> None:
        if hasattr(self.provider, "model"):
            self.provider.model = str(config.get("ollama_model", "llama3.2"))
        if hasattr(self.provider, "timeout"):
            self.provider.timeout = float(config.get("ai_timeout_seconds", 20))
        if hasattr(self.provider, "base_url"):
            self.provider.base_url = str(config.get("ollama_base_url", "http://127.0.0.1:11434")).rstrip("/")

    def _build_prompt(self, raw_input: str, config: dict[str, Any]) -> str:
        maximum_entries = int(config.get("max_conversation_entries", 20))
        summaries = sanitize_summary(get_conversation_summaries(), limit=maximum_entries)
        prompt_lines = [_SYSTEM_PROMPT]
        if summaries:
            prompt_lines.append("Recent conversation:")
            prompt_lines.extend(summaries)
        prompt_lines.append(f"User: {sanitize_for_prompt(raw_input)}")
        prompt_lines.append("Assistant:")
        return "\n".join(prompt_lines)

    def _record_conversation(self, raw_input: str, message: str, maximum_entries: int) -> None:
        add_conversation_summary(f"User: {sanitize_for_prompt(raw_input)}", maximum=maximum_entries)
        add_conversation_summary(f"Assistant: {_safe_message(message, 160)}", maximum=maximum_entries)

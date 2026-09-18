from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.brain.ai.models import ProviderStatus, ProviderStatusCategory


@dataclass
class AIState:
    enabled: bool = False
    provider_name: str = "ollama"
    model_name: str = ""
    last_provider_status_category: str = ProviderStatusCategory.UNAVAILABLE.value
    last_ai_result_category: str = "fallback"

    def reset(self) -> None:
        self.enabled = False
        self.provider_name = "ollama"
        self.model_name = ""
        self.last_provider_status_category = ProviderStatusCategory.UNAVAILABLE.value
        self.last_ai_result_category = "fallback"

    def update_provider_status(self, status: ProviderStatus) -> None:
        self.last_provider_status_category = status.category.value

    def update_result_category(self, category: str) -> None:
        self.last_ai_result_category = category


AI_STATE = AIState()


def get_ai_state() -> AIState:
    return AI_STATE


def reset_ai_state() -> None:
    AI_STATE.reset()

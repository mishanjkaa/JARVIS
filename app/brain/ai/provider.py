from __future__ import annotations

from typing import Protocol

from app.brain.ai.models import AIResponse, ProviderStatus


class Provider(Protocol):
    def status(self) -> ProviderStatus:
        ...

    def generate_text(self, prompt: str) -> AIResponse:
        ...

    def generate_structured(self, prompt: str) -> AIResponse:
        ...

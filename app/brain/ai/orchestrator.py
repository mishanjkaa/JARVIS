from __future__ import annotations

from app.brain.ai.models import AIIntent, AIResponse, ExecutionPlan, PlanStep, ProviderStatus, ProviderStatusCategory, ToolCall
from app.brain.ai.prompts import INTENT_PROMPT, PLANNING_PROMPT
from app.brain.ai.provider import Provider
from app.brain.ai.state import get_ai_state
from app.brain.security.ai_policy import validate_plan
from app.brain.tools.registry import ToolRegistry


class AIOrchestrator:
    def __init__(self, provider: Provider | None = None, registry: ToolRegistry | None = None) -> None:
        self.provider = provider
        self.registry = registry or ToolRegistry()

    def handle(self, raw_input: str) -> AIResponse:
        if not raw_input or not raw_input.strip():
            return AIResponse(category="fallback", message="I need a request to help.")
        if self.provider is None:
            return AIResponse(category="fallback", message="AI is unavailable.")
        state = get_ai_state()
        prompt = f"{INTENT_PROMPT}\nUser input: {raw_input.strip()}"
        result = self.provider.generate_structured(prompt)
        state.update_provider_status(result.status or ProviderStatus(category=ProviderStatusCategory.UNAVAILABLE))
        state.update_result_category(result.category)
        if result.category == "fallback":
            return AIResponse(category="fallback", message="I’m offline right now.")
        if result.intent == AIIntent.CONVERSATION:
            return AIResponse(category="conversation", message=result.message or "I can help with that.")
        if result.intent == AIIntent.TOOL_CALL and result.tool_call:
            self._validate_tool_call(result.tool_call)
            return AIResponse(category="tool_call", intent=result.intent, tool_call=result.tool_call, message="Tool requested")
        if result.intent == AIIntent.PLAN and result.plan:
            try:
                validate_plan(result.plan.to_dict().get("steps", []), max_steps=5)
            except Exception:
                return AIResponse(category="fallback", message="I couldn’t build a safe plan.")
            return AIResponse(category="plan", intent=result.intent, plan=result.plan, message="Plan ready")
        return AIResponse(category="fallback", message="I’m offline right now.")

    def _validate_tool_call(self, tool_call: ToolCall) -> None:
        if not tool_call.name:
            raise ValueError("tool name missing")
        if tool_call.name not in self.registry.list_names():
            raise ValueError("unknown tool")

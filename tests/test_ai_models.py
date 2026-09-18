import unittest

from app.brain.ai.models import AIIntent, AIResponse, ExecutionPlan, PlanStep, ProviderStatus, ProviderStatusCategory, ToolCall


class AIModelsTests(unittest.TestCase):
    def test_model_serialization(self) -> None:
        response = AIResponse(category="plan", intent=AIIntent.PLAN, tool_call=ToolCall(name="calculator.calculate"), plan=ExecutionPlan(steps=[PlanStep(id=1, tool_name="calculator.calculate")]))
        payload = response.to_dict()
        self.assertEqual(payload["category"], "plan")
        self.assertEqual(payload["intent"], AIIntent.PLAN.value)
        self.assertEqual(payload["tool_call"]["name"], "calculator.calculate")

    def test_provider_status_serialization(self) -> None:
        status = ProviderStatus(category=ProviderStatusCategory.READY, provider="ollama")
        self.assertEqual(status.to_dict()["category"], ProviderStatusCategory.READY.value)

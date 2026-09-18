import unittest
from unittest.mock import Mock

from app.brain.ai.models import AIIntent, AIResponse, ExecutionPlan, PlanStep, ToolCall
from app.brain.ai.orchestrator import AIOrchestrator
from app.brain.ai.state import reset_ai_state


class AIOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_ai_state()

    def test_handles_conversation(self) -> None:
        provider = Mock()
        provider.generate_structured.return_value = AIResponse(category="conversation", message="hello", intent=AIIntent.CONVERSATION)
        orchestrator = AIOrchestrator(provider=provider)
        result = orchestrator.handle("hello")
        self.assertEqual(result.category, "conversation")

    def test_handles_plan(self) -> None:
        provider = Mock()
        provider.generate_structured.return_value = AIResponse(category="plan", intent=AIIntent.PLAN, plan=ExecutionPlan(steps=[PlanStep(id=1, tool_name="system.get_time")]))
        orchestrator = AIOrchestrator(provider=provider)
        result = orchestrator.handle("time")
        self.assertEqual(result.category, "plan")

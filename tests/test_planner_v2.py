import unittest

from app.brain.memory.memory_policy import MAX_MEMORY_READS
from app.brain.planner.plan_models import AgentPlan, AgentStep
from app.brain.planner.planner_v2 import create_plan_from_request
from app.brain.planner.plan_validator import validate_plan
from app.brain.vision.desktop_capture_backend import MAX_DESKTOP_CAPTURES_PER_PLAN


class PlannerV2Tests(unittest.TestCase):
    def test_creates_valid_multi_step_plan(self) -> None:
        plan = create_plan_from_request("Calculate 25 times 4 and make a note with the result.")
        self.assertIsInstance(plan, AgentPlan)
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].tool_name, "calculator.calculate")
        self.assertEqual(plan.steps[1].tool_name, "notes.create")
        self.assertEqual(validate_plan(plan).valid, True)

    def test_rejects_plan_with_forward_dependency(self) -> None:
        plan = AgentPlan(steps=[
            AgentStep(step_id=2, tool_name="notes.create", arguments={}, depends_on=[1], risk_level="persistent_write", user_visible_description="Create a note"),
            AgentStep(step_id=1, tool_name="calculator.calculate", arguments={}, depends_on=[], risk_level="read_only", user_visible_description="Calculate"),
        ])
        result = validate_plan(plan)
        self.assertFalse(result.valid)
        self.assertIn("dependency", result.reason.lower())

    def test_accepts_plan_at_the_memory_recall_limit(self) -> None:
        plan = AgentPlan(steps=[
            AgentStep(step_id=index, tool_name="memory.recall", arguments={"key": f"k{index}"}, risk_level="read_only")
            for index in range(1, MAX_MEMORY_READS + 1)
        ])
        self.assertEqual(validate_plan(plan).valid, True)

    def test_rejects_plan_exceeding_max_memory_reads(self) -> None:
        plan = AgentPlan(steps=[
            AgentStep(step_id=index, tool_name="memory.recall", arguments={"key": f"k{index}"}, risk_level="read_only")
            for index in range(1, MAX_MEMORY_READS + 2)
        ])
        result = validate_plan(plan)
        self.assertFalse(result.valid)
        self.assertIn("memory.recall", result.reason)

    def test_accepts_plan_at_the_desktop_capture_limit(self) -> None:
        plan = AgentPlan(steps=[
            AgentStep(step_id=1, tool_name="desktop.capture_screen", arguments={}, risk_level="persistent_write"),
        ])
        self.assertEqual(MAX_DESKTOP_CAPTURES_PER_PLAN, 1)
        self.assertEqual(validate_plan(plan).valid, True)

    def test_rejects_plan_exceeding_max_desktop_captures(self) -> None:
        # RFC-007C is one-shot: desktop.capture_screen and desktop.capture_window are capped
        # combined, not per tool name, so mixing the two still trips the limit.
        plan = AgentPlan(steps=[
            AgentStep(step_id=1, tool_name="desktop.capture_screen", arguments={}, risk_level="persistent_write"),
            AgentStep(step_id=2, tool_name="desktop.capture_window", arguments={"window_id": 1, "window_title": "Notepad"}, risk_level="persistent_write"),
        ])
        result = validate_plan(plan)
        self.assertFalse(result.valid)
        self.assertIn("desktop capture", result.reason.lower())

import unittest

from app.brain.planner.plan_models import AgentPlan, AgentStep
from app.brain.planner.planner_v2 import create_plan_from_request
from app.brain.planner.plan_validator import validate_plan


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

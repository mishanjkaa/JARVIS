import unittest

from app.brain.security.ai_policy import PolicyError, validate_plan


class AIPlanningTests(unittest.TestCase):
    def test_rejects_plan_longer_than_limit(self) -> None:
        plan = [{"id": i, "tool_name": "system.get_time", "arguments": {}} for i in range(1, 7)]
        with self.assertRaises(PolicyError):
            validate_plan(plan, max_steps=5)

    def test_rejects_forward_reference(self) -> None:
        plan = [{"id": 1, "tool_name": "system.get_time", "arguments": {}}, {"id": 2, "tool_name": "system.get_date", "arguments": {}, "reference": {"from_step": 3, "field": "display_value"}}]
        with self.assertRaises(PolicyError):
            validate_plan(plan, max_steps=5)

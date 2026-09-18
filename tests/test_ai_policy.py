import unittest

from app.brain.security.ai_policy import PolicyError, validate_plan, validate_risk_level


class AIPolicyTests(unittest.TestCase):
    def test_rejects_unknown_risk(self) -> None:
        with self.assertRaises(PolicyError):
            validate_risk_level("unknown")

    def test_rejects_power_confirmation_plan(self) -> None:
        with self.assertRaises(PolicyError):
            validate_plan([{"id": 1, "tool_name": "power.request_shutdown", "arguments": {}}])

import unittest
from unittest.mock import Mock

from app.brain.configuration.runtime_config import get_runtime_config, reset_runtime_config, set_runtime_config_value
from app.brain.ai.state import reset_ai_state
from app.brain.router import route_command


class AIRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_ai_state()
        reset_runtime_config()

    def test_ai_disabled_fallback(self) -> None:
        # ai_enabled now defaults to True (owner's explicit request: JARVIS should be
        # usable immediately at startup, no "ai on" needed) -- this test is specifically
        # about the disabled-state behavior, so it turns AI off itself rather than relying
        # on that no longer being the out-of-the-box default.
        set_runtime_config_value("ai_enabled", False)
        self.assertEqual(route_command("give me a random fact"), "AI is disabled right now.")

    def test_ai_status_commands(self) -> None:
        self.assertEqual(route_command("ai on"), "AI enabled for this session.")
        self.assertEqual(route_command("ai status"), "AI is enabled.")
        self.assertEqual(route_command("ai off"), "AI disabled for this session.")

    def test_conversation_runtime_setting_commands(self) -> None:
        self.assertEqual(route_command("conversation status"), "Conversation is enabled.")
        self.assertEqual(route_command("conversation off"), "Conversation disabled.")
        self.assertEqual(route_command("conversation status"), "Conversation is disabled.")
        self.assertFalse(get_runtime_config()["ai_allow_conversation"])
        self.assertEqual(route_command("conversation on"), "Conversation enabled.")
        self.assertTrue(get_runtime_config()["ai_allow_conversation"])

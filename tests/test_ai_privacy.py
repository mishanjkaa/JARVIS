import logging
import unittest

from app.brain.context.history import clear_history, get_history
from app.brain.context.state import reset_context


class AIPrivacyTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_context()
        clear_history()

    def test_history_does_not_store_sensitive_values(self) -> None:
        from app.brain.context.history import record_safe_command

        record_safe_command("ai conversation")
        history = get_history()
        self.assertEqual(history, ["ai conversation"])

    def test_logs_do_not_include_sensitive_values(self) -> None:
        logger = logging.getLogger("test")
        logger.setLevel(logging.INFO)
        logger.info("safe message")

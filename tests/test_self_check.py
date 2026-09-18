import unittest
from unittest.mock import patch

from app.brain.context.history import clear_history
from app.brain.context.state import get_context, reset_context
from app.brain.skills.self_check import run_self_check


class SelfCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_context()
        clear_history()

    @patch("app.brain.skills.self_check.initialize_logging")
    @patch("app.brain.skills.self_check.importlib.import_module")
    def test_self_check_reports_summary(self, import_module_mock, initialize_logging_mock) -> None:
        initialize_logging_mock.side_effect = RuntimeError("boom")
        import_module_mock.side_effect = RuntimeError("boom")
        result = run_self_check()
        self.assertIn("PASS", result)
        self.assertIn("WARNING", result)

    def test_self_check_lists_safe_categories(self) -> None:
        result = run_self_check()
        self.assertIn("settings: PASS", result)
        self.assertIn("config_loader: PASS", result)
        self.assertIn("router: PASS", result)
        self.assertIn("context_state: PASS", result)
        self.assertIn("voice_modules: PASS", result)
        self.assertEqual(get_context().last_calculator_result, None)
        self.assertEqual(get_context().last_opened_folder, None)

import unittest
from unittest.mock import patch

from app.brain.computer.power_actions import execute_power_action


class PowerActionsTests(unittest.TestCase):
    @patch("app.brain.computer.power_actions.os.name", "nt")
    @patch("app.brain.computer.power_actions.subprocess.run")
    def test_shutdown_is_executed(self, run_mock) -> None:
        run_mock.return_value = None
        self.assertEqual(execute_power_action("shutdown"), "Shutdown request sent.")
        run_mock.assert_called_once()

    @patch("app.brain.computer.power_actions.os.name", "nt")
    @patch("app.brain.computer.power_actions.subprocess.run")
    def test_restart_is_executed(self, run_mock) -> None:
        run_mock.return_value = None
        self.assertEqual(execute_power_action("restart"), "Restart request sent.")
        run_mock.assert_called_once()

    @patch("app.brain.computer.power_actions.os.name", "nt")
    @patch("app.brain.computer.power_actions.subprocess.run")
    def test_lock_is_executed(self, run_mock) -> None:
        run_mock.return_value = None
        self.assertEqual(execute_power_action("lock"), "Lock request sent.")
        run_mock.assert_called_once()

    def test_unknown_action_is_rejected(self) -> None:
        self.assertEqual(execute_power_action("delete"), "Unsupported action.")

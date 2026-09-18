import unittest
from unittest.mock import patch

from app.brain.agent.controller import AgentController
from app.brain.agent.state import reset_agent_runtime_state
from app.brain.audit.audit_log import reset_audit_log
from app.brain.configuration.runtime_config import reset_runtime_config
from app.brain.filesystem.state import reset_filesystem_state
from app.brain.planner.approval import store_pending_plan
from app.brain.planner.plan_models import AgentPlan, AgentStep
from app.brain.planner.planner_v2 import create_plan_from_request
from app.brain.planner.state import reset_planner_state
from app.brain.router import route_command
from app.brain.security.confirmation import clear_pending_action, get_pending_action, request_confirmation
from app.brain.tools.executor import ToolExecutor
from app.brain.tools.models import ToolDefinition, ToolResult
from app.brain.tools.registry import ToolRegistry


class AgentRouterRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_planner_state()
        reset_agent_runtime_state()
        reset_audit_log()
        reset_runtime_config()
        reset_filesystem_state()
        clear_pending_action()

    def tearDown(self) -> None:
        reset_planner_state()
        reset_agent_runtime_state()
        reset_audit_log()
        reset_runtime_config()
        reset_filesystem_state()
        clear_pending_action()

    def _controller(self, *tools: ToolDefinition) -> AgentController:
        return AgentController(ToolExecutor(ToolRegistry(list(tools))))

    def test_empty_plan_commands_are_safe(self) -> None:
        self.assertEqual(route_command("show pending plan"), "No pending plan.")
        self.assertEqual(route_command("approve plan"), "No pending plan.")
        self.assertEqual(route_command("cancel plan"), "No plan is currently pending.")

    def test_empty_conversation_summary_is_safe(self) -> None:
        from app.brain.context.conversation import clear_conversation
        clear_conversation()
        self.assertEqual(route_command("conversation summary"), "No conversation entries yet.")

    def test_configuration_get_and_rejection(self) -> None:
        self.assertEqual(route_command("config get ai_enabled"), "False")
        self.assertEqual(route_command("config get unknown_key"), "Configuration key not found.")

    @patch("app.brain.router.config_set")
    def test_configuration_set_allowlist(self, config_set_mock) -> None:
        config_set_mock.return_value = "Configuration updated: ai_enabled."
        self.assertEqual(route_command("config set ai_enabled true"), "Configuration updated: ai_enabled.")
        config_set_mock.assert_called_once_with("ai_enabled", True)

    @patch("app.brain.router.config_set")
    def test_configuration_set_disallowed_key(self, config_set_mock) -> None:
        config_set_mock.return_value = "Configuration change rejected."
        self.assertEqual(route_command("config set security_policy unsafe"), "Configuration change rejected.")
        config_set_mock.assert_called_once_with("security_policy", "unsafe")

    def test_single_calculation_remains_deterministic(self) -> None:
        with patch("app.brain.router._build_ai_response") as ai_mock:
            self.assertEqual(route_command("calculate 5 * 8"), "Result: 40")
            ai_mock.assert_not_called()

    def test_multi_step_calculation_creates_pending_plan(self) -> None:
        with patch("app.brain.router._build_ai_response") as ai_mock:
            response = route_command("calculate 25 times 4 and make a note with the result")
        self.assertIn("This plan is MEDIUM risk.", response)
        ai_mock.assert_not_called()
        plan = create_plan_from_request("calculate 25 times 4 and make a note with the result")
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].tool_name, "calculator.calculate")
        self.assertEqual(plan.steps[1].tool_name, "notes.create")
        self.assertEqual(plan.steps[1].depends_on, [1])
        self.assertEqual(plan.steps[1].arguments["text"], {"from_step": 1, "field": "display_value"})
        self.assertIn("Pending plan", route_command("show pending plan"))

    def test_calculation_plus_save_enters_planner(self) -> None:
        with patch("app.brain.router._build_ai_response") as ai_mock:
            response = route_command("calculate 5*8 and save the result")
        self.assertIn("This plan is MEDIUM risk.", response)
        ai_mock.assert_not_called()
        plan = create_plan_from_request("calculate 5*8 and save the result")
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].tool_name, "calculator.calculate")
        self.assertEqual(plan.steps[0].arguments["expression"], "5*8")
        self.assertEqual(plan.steps[1].tool_name, "notes.create")

    def test_calculation_plus_record_enters_planner(self) -> None:
        with patch("app.brain.router._build_ai_response") as ai_mock:
            response = route_command("work out 100 / 4 and record the answer")
        self.assertIn("This plan is MEDIUM risk.", response)
        ai_mock.assert_not_called()
        plan = create_plan_from_request("work out 100 / 4 and record the answer")
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].tool_name, "calculator.calculate")
        self.assertEqual(plan.steps[0].arguments["expression"], "100 / 4")
        self.assertEqual(plan.steps[1].tool_name, "notes.create")

    def test_combined_calculation_requests_do_not_call_ollama(self) -> None:
        with patch("app.brain.router._build_ai_response") as ai_mock:
            response = route_command("calculate 9 * 9 then save it")
        self.assertIn("This plan is MEDIUM risk.", response)
        ai_mock.assert_not_called()

    def test_approval_executes_once_and_replay_is_rejected(self) -> None:
        controller = self._controller(
            ToolDefinition(
                name="notes.create",
                description="note",
                argument_schema={"text": {"type": "text", "max_length": 40}},
                risk_level="persistent_write",
                requires_confirmation=False,
                handler=lambda args: ToolResult(True, "success", "Plan executed.", "Plan executed."),
                formatter=lambda result: result.display_value or result.message,
            )
        )
        plan = AgentPlan(steps=[AgentStep(1, "notes.create", {"text": "hello"}, risk_level="persistent_write")], original_request="make note")
        with patch("app.brain.agent.controller._CONTROLLER", controller):
            self.assertIn("This plan is MEDIUM risk.", store_pending_plan(plan))
            self.assertEqual(route_command("approve plan"), "Plan executed.")
            self.assertEqual(route_command("approve plan"), "No pending plan.")

    def test_install_package_creates_high_risk_pending_plan(self) -> None:
        response = route_command("Install Python package requests")
        self.assertIn("HIGH RISK operation.", response)
        self.assertIn("Approval required.", response)

    def test_agent_commands_report_runtime_state(self) -> None:
        self.assertEqual(route_command("agent status"), "Agent state: idle.")
        self.assertEqual(route_command("agent task"), "No agent task.")

    def test_emergency_stop_without_task_is_safe(self) -> None:
        self.assertEqual(route_command("emergency stop"), "Emergency stop engaged. No active task was running.")
        self.assertEqual(route_command("agent status"), "Agent state: emergency_stopped.")

    def test_plan_commands_do_not_modify_power_confirmation(self) -> None:
        request_confirmation("shutdown")
        self.assertEqual(route_command("show pending plan"), "No pending plan.")
        self.assertEqual(route_command("cancel plan"), "No plan is currently pending.")
        self.assertEqual(get_pending_action(), "shutdown")


if __name__ == "__main__":
    unittest.main()

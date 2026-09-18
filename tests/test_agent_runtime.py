import unittest
from unittest.mock import patch

from app.brain.agent.controller import AgentController
from app.brain.agent.errors import AgentPlanError, AgentStateTransitionError
from app.brain.agent.models import AgentLifecycleState, AgentTaskRecord
from app.brain.agent.state import get_agent_runtime_state, reset_agent_runtime_state
from app.brain.audit.audit_log import get_audit_entries, reset_audit_log
from app.brain.configuration.runtime_config import reset_runtime_config, set_runtime_config_value
from app.brain.filesystem.state import reset_filesystem_state
from app.brain.planner.plan_models import AgentPlan, AgentStep
from app.brain.router import _CONVERSATION_RUNTIME, route_command
from app.brain.tools.executor import ToolExecutor
from app.brain.tools.models import ToolDefinition, ToolResult
from app.brain.tools.registry import ToolRegistry


def _tool(
    name: str,
    handler,
    *,
    argument_schema: dict | None = None,
    risk_level: str = "read_only",
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=name,
        argument_schema=argument_schema or {},
        risk_level=risk_level,
        requires_confirmation=False,
        handler=handler,
        formatter=lambda result: result.display_value or result.message,
    )


class _FakeConversationProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate_text(self, prompt: str):
        self.prompts.append(prompt)
        from app.brain.ai.models import AIIntent, AIResponse, ProviderStatus, ProviderStatusCategory

        return AIResponse(
            category="conversation",
            message="Conversation still works.",
            intent=AIIntent.CONVERSATION,
            status=ProviderStatus(category=ProviderStatusCategory.READY, provider="ollama"),
        )


class AgentRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_agent_runtime_state()
        reset_audit_log()
        reset_runtime_config()
        reset_filesystem_state()

    def tearDown(self) -> None:
        reset_agent_runtime_state()
        reset_audit_log()
        reset_runtime_config()
        reset_filesystem_state()

    def _controller(self, *tools: ToolDefinition) -> AgentController:
        return AgentController(ToolExecutor(ToolRegistry(list(tools))))

    def test_approved_plan_starts_execution(self) -> None:
        controller = self._controller(
            _tool("system.get_time", lambda args: ToolResult(True, "success", "12:00", "12:00")),
        )
        plan = AgentPlan(steps=[AgentStep(1, "system.get_time", {})], original_request="what time is it")

        controller.create_pending_task(plan)
        result = controller.approve_and_execute_current_task()

        self.assertEqual(result, "12:00")
        self.assertEqual(get_agent_runtime_state().current_task.state, AgentLifecycleState.COMPLETED)

    def test_two_step_plan_executes_in_order(self) -> None:
        order: list[str] = []
        controller = self._controller(
            _tool("step.one", lambda args: order.append("one") or ToolResult(True, "success", "1", "1")),
            _tool("step.two", lambda args: order.append("two") or ToolResult(True, "success", "2", "2")),
        )
        plan = AgentPlan(steps=[AgentStep(1, "step.one", {}), AgentStep(2, "step.two", {})], original_request="run both")

        controller.create_pending_task(plan)
        controller.approve_and_execute_current_task()

        self.assertEqual(order, ["one", "two"])

    def test_prior_step_result_is_passed_safely_to_next_step(self) -> None:
        captured: list[str] = []

        def capture_handler(args):
            captured.append(args["text"])
            return ToolResult(True, "success", f"saved {args['text']}", args["text"], {"display_value": args["text"]})

        controller = self._controller(
            _tool("calculator.calculate", lambda args: ToolResult(True, "success", "Result: 40", "40", {"display_value": "40"}), argument_schema={"expression": {"type": "text", "max_length": 40}}),
            _tool("notes.create", capture_handler, argument_schema={"text": {"type": "text", "max_length": 40}}, risk_level="persistent_write"),
        )
        plan = AgentPlan(
            steps=[
                AgentStep(1, "calculator.calculate", {"expression": "5 * 8"}),
                AgentStep(2, "notes.create", {"text": {"from_step": 1, "field": "display_value"}}, depends_on=[1], risk_level="persistent_write"),
            ],
            original_request="calculate and save",
        )

        controller.create_pending_task(plan)
        controller.approve_and_execute_current_task()

        self.assertEqual(captured, ["40"])

    def test_persistent_action_does_not_execute_before_approval(self) -> None:
        calls: list[str] = []
        controller = self._controller(
            _tool("calculator.calculate", lambda args: ToolResult(True, "success", "Result: 40", "40", {"display_value": "40"}), argument_schema={"expression": {"type": "text", "max_length": 40}}),
            _tool("notes.create", lambda args: calls.append(args["text"]) or ToolResult(True, "success", "Added note 1.", "1"), argument_schema={"text": {"type": "text", "max_length": 40}}, risk_level="persistent_write"),
        )
        plan = AgentPlan(
            steps=[
                AgentStep(1, "calculator.calculate", {"expression": "5 * 8"}),
                AgentStep(2, "notes.create", {"text": {"from_step": 1, "field": "display_value"}}, depends_on=[1], risk_level="persistent_write"),
            ],
            original_request="calculate and save",
        )

        controller.create_pending_task(plan)

        self.assertEqual(calls, [])
        self.assertEqual(get_agent_runtime_state().current_task.state, AgentLifecycleState.PENDING_APPROVAL)

    def test_cancelled_plan_never_starts(self) -> None:
        calls: list[str] = []
        controller = self._controller(
            _tool("notes.create", lambda args: calls.append(args["text"]) or ToolResult(True, "success", "Added note 1.", "1"), argument_schema={"text": {"type": "text", "max_length": 40}}, risk_level="persistent_write"),
        )
        plan = AgentPlan(steps=[AgentStep(1, "notes.create", {"text": "hello"}, risk_level="persistent_write")], original_request="make note")

        controller.create_pending_task(plan)
        self.assertEqual(controller.cancel_pending_or_running_task(), "Pending plan cancelled.")
        self.assertEqual(calls, [])

    def test_active_agent_status(self) -> None:
        controller = self._controller(_tool("system.get_time", lambda args: ToolResult(True, "success", "12:00", "12:00")))
        controller.create_pending_task(AgentPlan(steps=[AgentStep(1, "system.get_time", {})], original_request="time"))

        self.assertEqual(controller.get_status_message(), "Agent state: pending_approval.")

    def test_completed_agent_status(self) -> None:
        controller = self._controller(_tool("system.get_time", lambda args: ToolResult(True, "success", "12:00", "12:00")))
        controller.create_pending_task(AgentPlan(steps=[AgentStep(1, "system.get_time", {})], original_request="time"))
        controller.approve_and_execute_current_task()

        self.assertEqual(controller.get_status_message(), "Agent state: completed.")

    def test_failed_tool_changes_state_to_failed(self) -> None:
        controller = self._controller(_tool("system.get_time", lambda args: ToolResult(False, "failed", "Tool execution failed.")))
        controller.create_pending_task(AgentPlan(steps=[AgentStep(1, "system.get_time", {})], original_request="time"))
        result = controller.approve_and_execute_current_task()

        self.assertEqual(result, "Tool execution failed.")
        self.assertEqual(get_agent_runtime_state().current_task.state, AgentLifecycleState.FAILED)

    def test_later_steps_do_not_execute_after_failure(self) -> None:
        calls: list[str] = []
        controller = self._controller(
            _tool("step.one", lambda args: ToolResult(False, "failed", "first failed")),
            _tool("step.two", lambda args: calls.append("two") or ToolResult(True, "success", "ok", "ok")),
        )
        plan = AgentPlan(steps=[AgentStep(1, "step.one", {}), AgentStep(2, "step.two", {})], original_request="fail early")

        controller.create_pending_task(plan)
        controller.approve_and_execute_current_task()

        self.assertEqual(calls, [])

    def test_agent_cancel_stops_later_steps(self) -> None:
        def cancel_then_succeed(args):
            controller.cancel_pending_or_running_task()
            return ToolResult(True, "success", "done", "done")

        controller = self._controller(
            _tool("step.one", cancel_then_succeed),
            _tool("step.two", lambda args: ToolResult(True, "success", "should not run", "x")),
        )
        plan = AgentPlan(steps=[AgentStep(1, "step.one", {}), AgentStep(2, "step.two", {})], original_request="cancel during run")

        controller.create_pending_task(plan)
        result = controller.approve_and_execute_current_task()

        self.assertEqual(result, "Agent task cancelled.")
        self.assertEqual(get_agent_runtime_state().current_task.state, AgentLifecycleState.CANCELLED)

    def test_repeated_cancel_is_safe(self) -> None:
        controller = self._controller(_tool("system.get_time", lambda args: ToolResult(True, "success", "12:00", "12:00")))
        controller.create_pending_task(AgentPlan(steps=[AgentStep(1, "system.get_time", {})], original_request="time"))

        first = controller.cancel_pending_or_running_task()
        second = controller.cancel_pending_or_running_task()

        self.assertEqual(first, "Pending plan cancelled.")
        self.assertEqual(second, "Agent task is already cancelled.")

    def test_emergency_stop_prevents_further_execution(self) -> None:
        calls: list[str] = []

        def stop_then_succeed(args):
            controller.engage_emergency_stop()
            return ToolResult(True, "success", "done", "done")

        controller = self._controller(
            _tool("step.one", stop_then_succeed),
            _tool("step.two", lambda args: calls.append("two") or ToolResult(True, "success", "ok", "ok")),
        )
        plan = AgentPlan(steps=[AgentStep(1, "step.one", {}), AgentStep(2, "step.two", {})], original_request="stop during run")

        controller.create_pending_task(plan)
        result = controller.approve_and_execute_current_task()

        self.assertEqual(result, "Emergency stop engaged.")
        self.assertEqual(calls, [])
        self.assertEqual(get_agent_runtime_state().current_task.state, AgentLifecycleState.EMERGENCY_STOPPED)

    def test_duplicate_approval_is_safe(self) -> None:
        controller = self._controller(_tool("system.get_time", lambda args: ToolResult(True, "success", "12:00", "12:00")))
        controller.create_pending_task(AgentPlan(steps=[AgentStep(1, "system.get_time", {})], original_request="time"))

        self.assertEqual(controller.approve_and_execute_current_task(), "12:00")
        self.assertEqual(controller.approve_and_execute_current_task(), "No pending plan.")

    def test_missing_result_reference_fails_safely(self) -> None:
        controller = self._controller(
            _tool("system.get_time", lambda args: ToolResult(True, "success", "12:00", "12:00")),
            _tool("notes.create", lambda args: ToolResult(True, "success", "Added note 1.", "1"), argument_schema={"text": {"type": "text", "max_length": 40}}, risk_level="persistent_write"),
        )
        plan = AgentPlan(
            steps=[
                AgentStep(1, "system.get_time", {}),
                AgentStep(2, "notes.create", {"text": {"from_step": 1, "field": "missing_field"}}, depends_on=[1], risk_level="persistent_write"),
            ],
            original_request="bad reference",
        )

        controller.create_pending_task(plan)
        result = controller.approve_and_execute_current_task()

        self.assertEqual(result, "Invalid result reference.")
        self.assertEqual(get_agent_runtime_state().current_task.state, AgentLifecycleState.FAILED)

    def test_invalid_state_transition_is_rejected(self) -> None:
        task = AgentTaskRecord(task_id=1, plan=AgentPlan(), original_user_goal="test", total_steps=0)
        task.transition_to(AgentLifecycleState.PENDING_APPROVAL)
        task.transition_to(AgentLifecycleState.RUNNING)
        task.transition_to(AgentLifecycleState.COMPLETED)

        with self.assertRaises(AgentStateTransitionError):
            task.transition_to(AgentLifecycleState.RUNNING)

    def test_multiple_active_agents_are_prevented(self) -> None:
        controller = self._controller(_tool("system.get_time", lambda args: ToolResult(True, "success", "12:00", "12:00")))
        controller.create_pending_task(AgentPlan(steps=[AgentStep(1, "system.get_time", {})], original_request="time"))

        with self.assertRaises(ValueError):
            controller.create_pending_task(AgentPlan(steps=[AgentStep(1, "system.get_time", {})], original_request="time again"))

    def test_agent_disabled_configuration(self) -> None:
        set_runtime_config_value("agent_enabled", False)
        controller = self._controller(_tool("system.get_time", lambda args: ToolResult(True, "success", "12:00", "12:00")))

        with self.assertRaises(ValueError):
            controller.create_pending_task(AgentPlan(steps=[AgentStep(1, "system.get_time", {})], original_request="time"))

    def test_max_step_limit(self) -> None:
        set_runtime_config_value("agent_max_steps", 1)
        controller = self._controller(_tool("system.get_time", lambda args: ToolResult(True, "success", "12:00", "12:00")))

        with self.assertRaises(AgentPlanError):
            controller.create_pending_task(AgentPlan(steps=[AgentStep(1, "system.get_time", {}), AgentStep(2, "system.get_time", {})], original_request="too long"))

    def test_result_size_limit(self) -> None:
        set_runtime_config_value("agent_result_size_limit", 40)
        controller = self._controller(_tool("system.get_time", lambda args: ToolResult(True, "success", "x" * 100, "x" * 100)))
        controller.create_pending_task(AgentPlan(steps=[AgentStep(1, "system.get_time", {})], original_request="big result"))

        result = controller.approve_and_execute_current_task()

        self.assertEqual(result, "Tool result too large.")
        self.assertEqual(get_agent_runtime_state().current_task.state, AgentLifecycleState.FAILED)

    def test_persistent_action_configuration(self) -> None:
        set_runtime_config_value("agent_allow_persistent_actions", False)
        controller = self._controller(_tool("notes.create", lambda args: ToolResult(True, "success", "Added note 1.", "1"), argument_schema={"text": {"type": "text", "max_length": 40}}, risk_level="persistent_write"))

        with self.assertRaises(AgentPlanError):
            controller.create_pending_task(AgentPlan(steps=[AgentStep(1, "notes.create", {"text": "hello"}, risk_level="persistent_write")], original_request="make note"))

    def test_audit_events_are_generated(self) -> None:
        controller = self._controller(_tool("system.get_time", lambda args: ToolResult(True, "success", "12:00", "12:00")))
        controller.create_pending_task(AgentPlan(steps=[AgentStep(1, "system.get_time", {})], original_request="time"))
        controller.approve_and_execute_current_task()

        event_types = [entry.event_type for entry in get_audit_entries()]
        self.assertIn("plan_created", event_types)
        self.assertIn("plan_approved", event_types)
        self.assertIn("agent_started", event_types)
        self.assertIn("step_started", event_types)
        self.assertIn("step_completed", event_types)
        self.assertIn("agent_completed", event_types)

    def test_manual_acceptance_flow_uses_real_runtime(self) -> None:
        fake_provider = _FakeConversationProvider()

        def calculate_handler(args):
            return ToolResult(True, "success", "Result: 40", "40", {"display_value": "40"})

        created_notes: list[str] = []

        def note_handler(args):
            created_notes.append(args["text"])
            return ToolResult(True, "success", "Added note 1.", "1", {"display_value": "1"})

        controller = self._controller(
            _tool("calculator.calculate", calculate_handler, argument_schema={"expression": {"type": "text", "max_length": 40}}),
            _tool("notes.create", note_handler, argument_schema={"text": {"type": "text", "max_length": 40}}, risk_level="persistent_write"),
        )

        with patch("app.brain.agent.controller._CONTROLLER", controller), patch.object(_CONVERSATION_RUNTIME, "provider", fake_provider):
            pending = route_command("calculate 5*8 and save the result")
            approved = route_command("approve plan")
            status = route_command("agent status")
            task = route_command("agent task")

        self.assertIn("Pending plan", pending)
        self.assertEqual(approved, "Added note 1.")
        self.assertEqual(status, "Agent state: completed.")
        self.assertIn("progress=2/2", task)
        self.assertEqual(created_notes, ["40"])
        self.assertEqual(fake_provider.prompts, [])

    def test_cancel_plan_acceptance_flow_skips_persistent_action(self) -> None:
        created_notes: list[str] = []
        controller = self._controller(
            _tool("notes.create", lambda args: created_notes.append(args["text"]) or ToolResult(True, "success", "Added note 1.", "1"), argument_schema={"text": {"type": "text", "max_length": 80}}, risk_level="persistent_write"),
        )

        with patch("app.brain.agent.controller._CONTROLLER", controller):
            pending = route_command("make a note saying hello")
            cancelled = route_command("cancel plan")

        self.assertIn("Pending plan", pending)
        self.assertEqual(cancelled, "Pending plan cancelled.")
        self.assertEqual(created_notes, [])

    def test_ordinary_conversation_routing_remains_unaffected(self) -> None:
        fake_provider = _FakeConversationProvider()
        set_runtime_config_value("ai_enabled", True)

        with patch.object(_CONVERSATION_RUNTIME, "provider", fake_provider):
            result = route_command("Tell me a short greeting.")

        self.assertEqual(result, "Conversation still works.")

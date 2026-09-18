from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.brain.agent.errors import (
    ActiveAgentConflictError,
    AgentDisabledError,
    AgentEmergencyStopError,
    AgentPlanError,
)
from app.brain.agent.models import ACTIVE_AGENT_STATES, AgentLifecycleState, AgentTaskRecord
from app.brain.agent.state import get_agent_runtime_state
from app.brain.audit.audit_log import record_audit_event
from app.brain.browser.controller import get_browser_controller
from app.brain.configuration.runtime_config import get_effective_runtime_config
from app.brain.location.controller import get_location_controller
from app.brain.planner.plan_models import AgentPlan, AgentStep
from app.brain.planner.plan_validator import validate_plan as validate_agent_plan
from app.brain.planner.state import reset_planner_state
from app.brain.security.ai_policy import PolicyError, PERSISTENT_WRITE_TOOLS, validate_plan_policy
from app.brain.terminal.controller import get_terminal_controller
from app.brain.tools.executor import ToolExecutor
from app.brain.vision.controller import get_vision_controller
from config.config_loader import load_config


class AgentController:
    def __init__(self, executor: ToolExecutor | None = None) -> None:
        self.executor = executor or ToolExecutor()

    def create_pending_task(self, plan: AgentPlan, *, original_user_goal: str | None = None) -> AgentTaskRecord:
        config = self._effective_config()
        self._ensure_agent_available(config)
        self._archive_finished_task_if_needed()
        runtime_state = get_agent_runtime_state()
        if runtime_state.current_task is not None and runtime_state.current_task.state in ACTIVE_AGENT_STATES:
            if runtime_state.current_task.state == AgentLifecycleState.PENDING_APPROVAL:
                raise ActiveAgentConflictError('A plan is already pending. Use "approve plan" or "cancel plan".')
            raise ActiveAgentConflictError('An agent task is already running. Use "agent cancel", "terminal cancel", or "emergency stop".')
        self._validate_plan(plan, config)

        task = AgentTaskRecord(
            task_id=runtime_state.next_task_id,
            plan=plan,
            original_user_goal=(original_user_goal or getattr(plan, "original_request", "") or "Agent task").strip()[:200],
            total_steps=len(plan.steps),
            state=AgentLifecycleState.IDLE,
            latest_safe_status_message="Waiting for approval.",
        )
        task.transition_to(AgentLifecycleState.PENDING_APPROVAL)
        runtime_state.current_task = task
        runtime_state.next_task_id += 1
        runtime_state.last_safe_status_message = task.latest_safe_status_message
        record_audit_event("plan_created", task_id=task.task_id, message=f"{task.total_steps} step plan queued")
        return task

    def approve_and_execute_current_task(self) -> str:
        runtime_state = get_agent_runtime_state()
        task = runtime_state.current_task
        if task is None or task.state != AgentLifecycleState.PENDING_APPROVAL:
            return "No pending plan."

        config = self._effective_config()
        try:
            self._ensure_agent_available(config)
            self._validate_plan(task.plan, config)
        except (AgentDisabledError, AgentEmergencyStopError, AgentPlanError) as error:
            return self._fail_pending_task(task, str(error))

        record_audit_event("plan_approved", task_id=task.task_id, message="Plan approved for execution")
        try:
            return self._run_task(task, config)
        except Exception:
            return self._fail_running_task(task, "Agent execution failed safely.")

    def cancel_pending_or_running_task(self) -> str:
        runtime_state = get_agent_runtime_state()
        task = runtime_state.current_task
        if task is None:
            return "No active agent task."
        if task.state == AgentLifecycleState.CANCELLED:
            return "Agent task is already cancelled."
        if task.state == AgentLifecycleState.PENDING_APPROVAL:
            task.transition_to(AgentLifecycleState.CANCELLED)
            task.finish_time = datetime.now(timezone.utc)
            task.latest_safe_status_message = "Pending plan cancelled."
            runtime_state.last_safe_status_message = task.latest_safe_status_message
            self._cleanup_browser_captures_for_task(task)
            self._cleanup_desktop_captures_for_task(task)
            self._cleanup_navigation_sessions_for_task(task)
            reset_planner_state()
            record_audit_event("plan_cancelled", task_id=task.task_id, message="Plan cancelled before execution")
            record_audit_event("agent_cancelled", task_id=task.task_id, message="Agent task cancelled before execution")
            return "Pending plan cancelled."
        if task.state == AgentLifecycleState.RUNNING:
            if task.cancellation_requested:
                return "Agent cancellation already requested."
            task.cancellation_requested = True
            task.latest_safe_status_message = "Cancellation requested."
            runtime_state.last_safe_status_message = task.latest_safe_status_message
            get_terminal_controller().cancel_active_execution()
            get_browser_controller().cancel_current_operation()
            return "Agent cancellation requested."
        return "No active agent task."

    def engage_emergency_stop(self) -> str:
        runtime_state = get_agent_runtime_state()
        if runtime_state.emergency_stop_active and runtime_state.get_effective_state() == AgentLifecycleState.EMERGENCY_STOPPED:
            return "Emergency stop is already active."
        runtime_state.emergency_stop_active = True
        task = runtime_state.current_task
        if task is not None and task.state in {AgentLifecycleState.PENDING_APPROVAL, AgentLifecycleState.RUNNING}:
            get_terminal_controller().cancel_active_execution(reason="emergency")
            get_browser_controller().cancel_current_operation(reason="emergency")
            get_browser_controller().close_all_sessions(reason="emergency")
            task.transition_to(AgentLifecycleState.EMERGENCY_STOPPED)
            task.finish_time = datetime.now(timezone.utc)
            task.latest_safe_status_message = "Emergency stop engaged."
            runtime_state.last_safe_status_message = task.latest_safe_status_message
            record_audit_event("emergency_stop", task_id=task.task_id, message="Emergency stop engaged")
            return "Emergency stop engaged."
        runtime_state.last_safe_status_message = "Emergency stop engaged."
        get_browser_controller().close_all_sessions(reason="emergency")
        record_audit_event("emergency_stop", message="Emergency stop engaged without an active task")
        return "Emergency stop engaged. No active task was running."

    def get_status_message(self) -> str:
        runtime_state = get_agent_runtime_state()
        state = runtime_state.get_effective_state()
        return f"Agent state: {state.value}."

    def get_task_message(self) -> str:
        runtime_state = get_agent_runtime_state()
        task = runtime_state.current_task
        if task is None:
            if runtime_state.emergency_stop_active:
                return "No agent task. Emergency stop is active."
            return "No agent task."
        parts = [
            f"Task {task.task_id}",
            f"state={task.state.value}",
            f"progress={task.progress_label()}",
            f"goal={task.original_user_goal or 'Agent task'}",
        ]
        if task.latest_safe_status_message:
            parts.append(f"status={task.latest_safe_status_message}")
        if task.final_result:
            parts.append(f"result={task.final_result}")
        if task.failure_reason:
            parts.append(f"failure={task.failure_reason}")
        return " | ".join(parts)

    def _run_task(self, task: AgentTaskRecord, config: dict[str, Any]) -> str:
        runtime_state = get_agent_runtime_state()
        task.transition_to(AgentLifecycleState.RUNNING)
        task.start_time = datetime.now(timezone.utc)
        task.latest_safe_status_message = "Agent execution started."
        runtime_state.last_safe_status_message = task.latest_safe_status_message
        record_audit_event("agent_started", task_id=task.task_id, message="Agent execution started")

        prior_results: dict[int, dict[str, Any]] = {}
        for step in task.plan.steps:
            if self._stop_if_requested(task):
                return self._finalize_stop(task)
            record_audit_event("step_started", task_id=task.task_id, step_index=step.step_id, message=step.tool_name)
            task.latest_safe_status_message = f"Running step {step.step_id} of {task.total_steps}."
            runtime_state.last_safe_status_message = task.latest_safe_status_message
            result = self.executor.execute_step(
                self._step_to_dict(step),
                prior_results=prior_results,
                result_size_limit=int(config["agent_result_size_limit"]),
            )
            if not result.success:
                result_payload = result.to_dict()
                result_payload["tool_name"] = step.tool_name
                task.step_results.append(result_payload)
                prior_results[step.step_id] = result_payload
                max_chars = 700 if step.tool_name == "terminal.execute" else 160
                task.failed_step_index = step.step_id
                task.failed_tool_name = step.tool_name
                task.failed_error_category = str(result.reference_fields.get("error_category") or "")
                task.failure_reason = (result.message or "Tool execution failed safely.")[:max_chars]
                task.latest_safe_status_message = task.failure_reason
                record_audit_event(
                    "step_failed",
                    task_id=task.task_id,
                    step_index=step.step_id,
                    message=self._format_step_failure(step, result.message, max_chars=max_chars),
                )
                self._cleanup_browser_sessions(prior_results)
                return self._fail_running_task(task, task.failure_reason, max_chars=max_chars)
            result_payload = result.to_dict()
            result_payload["tool_name"] = step.tool_name
            task.step_results.append(result_payload)
            prior_results[step.step_id] = result_payload
            task.current_step_index = step.step_id
            task.latest_safe_status_message = result.message[:160] or "Step completed."
            runtime_state.last_safe_status_message = task.latest_safe_status_message
            record_audit_event("step_completed", task_id=task.task_id, step_index=step.step_id, message=step.tool_name)
            if self._stop_if_requested(task):
                self._cleanup_browser_sessions(prior_results)
                return self._finalize_stop(task)

        task.transition_to(AgentLifecycleState.COMPLETED)
        task.finish_time = datetime.now(timezone.utc)
        task.final_result = self._build_final_result(task)
        task.latest_safe_status_message = task.final_result
        runtime_state.last_safe_status_message = task.latest_safe_status_message
        self._cleanup_browser_captures_for_task(task)
        self._cleanup_desktop_captures_for_task(task)
        self._cleanup_navigation_sessions_for_task(task)
        record_audit_event("agent_completed", task_id=task.task_id, message="Agent execution completed")
        return task.final_result

    def _stop_if_requested(self, task: AgentTaskRecord) -> bool:
        runtime_state = get_agent_runtime_state()
        return runtime_state.emergency_stop_active or task.cancellation_requested

    def _finalize_stop(self, task: AgentTaskRecord) -> str:
        runtime_state = get_agent_runtime_state()
        self._cleanup_browser_captures_for_task(task)
        self._cleanup_desktop_captures_for_task(task)
        self._cleanup_navigation_sessions_for_task(task)
        self._cleanup_browser_sessions(self._prior_results_from_task(task))
        if runtime_state.emergency_stop_active:
            if task.state != AgentLifecycleState.EMERGENCY_STOPPED:
                task.transition_to(AgentLifecycleState.EMERGENCY_STOPPED)
                task.finish_time = datetime.now(timezone.utc)
                task.latest_safe_status_message = "Emergency stop engaged."
                runtime_state.last_safe_status_message = task.latest_safe_status_message
                record_audit_event("emergency_stop", task_id=task.task_id, message="Emergency stop engaged during execution")
            return "Emergency stop engaged."
        if task.state != AgentLifecycleState.CANCELLED:
            task.transition_to(AgentLifecycleState.CANCELLED)
            task.finish_time = datetime.now(timezone.utc)
            task.latest_safe_status_message = "Agent task cancelled."
            runtime_state.last_safe_status_message = task.latest_safe_status_message
            record_audit_event("agent_cancelled", task_id=task.task_id, message="Agent execution cancelled")
        return "Agent task cancelled."

    def _fail_pending_task(self, task: AgentTaskRecord, reason: str) -> str:
        runtime_state = get_agent_runtime_state()
        task.transition_to(AgentLifecycleState.FAILED if task.state == AgentLifecycleState.RUNNING else AgentLifecycleState.CANCELLED)
        task.finish_time = datetime.now(timezone.utc)
        task.failure_reason = reason[:160]
        task.latest_safe_status_message = task.failure_reason
        runtime_state.last_safe_status_message = task.latest_safe_status_message
        self._cleanup_browser_captures_for_task(task)
        self._cleanup_desktop_captures_for_task(task)
        self._cleanup_navigation_sessions_for_task(task)
        if task.state == AgentLifecycleState.CANCELLED:
            record_audit_event("plan_cancelled", task_id=task.task_id, message=task.failure_reason)
            record_audit_event("agent_cancelled", task_id=task.task_id, message=task.failure_reason)
        else:
            record_audit_event("agent_failed", task_id=task.task_id, message=task.failure_reason)
        return task.failure_reason

    def _fail_running_task(self, task: AgentTaskRecord, reason: str, *, max_chars: int = 160) -> str:
        runtime_state = get_agent_runtime_state()
        if task.state == AgentLifecycleState.RUNNING:
            task.transition_to(AgentLifecycleState.FAILED)
        task.finish_time = datetime.now(timezone.utc)
        task.failure_reason = reason[:max_chars]
        task.latest_safe_status_message = task.failure_reason
        runtime_state.last_safe_status_message = task.latest_safe_status_message
        self._cleanup_browser_captures_for_task(task)
        self._cleanup_desktop_captures_for_task(task)
        self._cleanup_navigation_sessions_for_task(task)
        record_audit_event("agent_failed", task_id=task.task_id, message=task.failure_reason)
        return task.failure_reason

    def _build_final_result(self, task: AgentTaskRecord) -> str:
        if task.step_results:
            last_result = task.step_results[-1]
            if isinstance(last_result.get("message"), str) and last_result["message"].strip():
                max_chars = 700 if last_result.get("tool_name") == "terminal.execute" else 160
                return last_result["message"][:max_chars]
        return "Plan completed safely."

    def _format_step_failure(self, step: AgentStep, message: str, *, max_chars: int) -> str:
        reason = (message or "Tool execution failed safely.").strip()
        prefix = f"Step {step.step_id} ({step.tool_name}) failed: "
        if reason.lower().startswith(prefix.lower()):
            return reason[:max_chars]
        return f"{prefix}{reason}"[:max_chars]

    def _cleanup_browser_sessions(self, prior_results: dict[int, dict[str, Any]]) -> None:
        controller = get_browser_controller()
        for session_id in self._browser_session_ids(prior_results):
            try:
                evidence = controller.close_session(session_id)
            except Exception:
                record_audit_event("browser_cleanup_failed", message=f"{session_id}:exception")
                continue
            if not getattr(evidence, "success", False):
                record_audit_event(
                    "browser_cleanup_failed",
                    message=f"{session_id}:{getattr(evidence, 'error_reason', '')[:120]}",
                )
                continue
            record_audit_event("browser_cleanup_completed", message=session_id)

    def _prior_results_from_task(self, task: AgentTaskRecord) -> dict[int, dict[str, Any]]:
        prior_results: dict[int, dict[str, Any]] = {}
        for index, result in enumerate(task.step_results, 1):
            if isinstance(result, dict):
                prior_results[index] = result
        return prior_results

    def _browser_session_ids(self, prior_results: dict[int, dict[str, Any]]) -> list[str]:
        session_ids: list[str] = []
        for result in prior_results.values():
            if not isinstance(result, dict):
                continue
            reference_fields = result.get("reference_fields")
            reference_payload = reference_fields if isinstance(reference_fields, dict) else {}
            browser_operation = str(result.get("browser_operation") or reference_payload.get("browser_operation") or "")
            session_created = bool(result.get("session_created") or reference_payload.get("session_created"))
            if browser_operation != "start_session" and not session_created:
                continue
            direct_value = result.get("session_id")
            if isinstance(direct_value, str) and direct_value and direct_value not in session_ids:
                session_ids.append(direct_value)
                continue
            if not isinstance(reference_fields, dict):
                continue
            session_value = reference_fields.get("session_id")
            if isinstance(session_value, str) and session_value and session_value not in session_ids:
                session_ids.append(session_value)
        return session_ids

    def _cleanup_browser_captures_for_task(self, task: AgentTaskRecord) -> None:
        try:
            removed = get_vision_controller().cleanup_captures_for_task(owner_agent_task_id=task.task_id)
        except Exception:
            record_audit_event("vision_capture_cleanup_failed", task_id=task.task_id, message="task capture cleanup failed")
            return
        if removed:
            record_audit_event("vision_capture_cleanup_completed", task_id=task.task_id, message=f"{removed} capture(s)")

    def _cleanup_desktop_captures_for_task(self, task: AgentTaskRecord) -> None:
        try:
            removed = get_vision_controller().cleanup_desktop_captures_for_task(owner_agent_task_id=task.task_id)
        except Exception:
            record_audit_event("vision_desktop_capture_cleanup_failed", task_id=task.task_id, message="task desktop capture cleanup failed")
            return
        if removed:
            record_audit_event("vision_desktop_capture_cleanup_completed", task_id=task.task_id, message=f"{removed} capture(s)")

    def _cleanup_navigation_sessions_for_task(self, task: AgentTaskRecord) -> None:
        try:
            removed = get_location_controller().cleanup_sessions_for_task(owner_agent_task_id=task.task_id)
        except Exception:
            record_audit_event("navigation_session_cleanup_failed", task_id=task.task_id, message="task navigation session cleanup failed")
            return
        if removed:
            record_audit_event("navigation_session_cleanup_completed", task_id=task.task_id, message=f"{removed} session(s)")

    def _validate_plan(self, plan: AgentPlan, config: dict[str, Any]) -> None:
        if not isinstance(plan, AgentPlan):
            raise AgentPlanError("Plan invalid.")
        validation = validate_agent_plan(plan, max_steps=int(config["agent_max_steps"]))
        if not validation.valid:
            raise AgentPlanError("Plan invalid.")
        plan_payload = [self._step_to_dict(step) for step in plan.steps]
        try:
            validate_plan_policy(
                plan_payload,
                max_steps=int(config["agent_max_steps"]),
                max_persistent_writes=int(config.get("max_persistent_writes_per_plan", 2)),
                max_external_actions=int(config.get("max_external_actions_per_plan", 2)),
            )
        except PolicyError:
            raise AgentPlanError("Plan invalid.")
        if not config.get("agent_allow_persistent_actions", True):
            if any(step.tool_name in PERSISTENT_WRITE_TOOLS for step in plan.steps):
                raise AgentPlanError("Persistent actions are disabled.")

    def _ensure_agent_available(self, config: dict[str, Any]) -> None:
        runtime_state = get_agent_runtime_state()
        if runtime_state.emergency_stop_active:
            raise AgentEmergencyStopError("Agent runtime is emergency stopped.")
        if not config.get("agent_enabled", True):
            raise AgentDisabledError("Agent runtime is disabled.")

    def _archive_finished_task_if_needed(self) -> None:
        runtime_state = get_agent_runtime_state()
        if runtime_state.current_task is not None and runtime_state.current_task.is_terminal():
            runtime_state.archive_current_task()

    def _effective_config(self) -> dict[str, Any]:
        return get_effective_runtime_config()

    def _step_to_dict(self, step: AgentStep) -> dict[str, Any]:
        return {
            "id": step.step_id,
            "tool_name": step.tool_name,
            "arguments": dict(step.arguments),
        }


_CONTROLLER = AgentController()


def get_agent_controller() -> AgentController:
    return _CONTROLLER

from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock

from app.brain.agent.models import AgentLifecycleState
from app.brain.agent.controller import get_agent_controller
from app.brain.agent.state import get_agent_runtime_state
from app.brain.audit.audit_log import record_audit_event
from app.brain.configuration.runtime_config import get_effective_runtime_config
from app.brain.planner.plan_summary import summarize_plan
from app.brain.planner.state import get_planner_state, pending_plan_expired, planner_now, reset_planner_state
from app.brain.risk.analyzer import analyze_plan
from app.brain.risk.models import PolicyOutcome
from app.brain.terminal.controller import get_terminal_controller
from config.config_loader import load_config

_APPROVAL_LOCK = RLock()


def store_pending_plan(plan: object) -> str:
    with _APPROVAL_LOCK:
        _reconcile_expired_or_stale_pending_plan_locked()
        state = get_planner_state()
        if state.pending_plan is not None and not pending_plan_expired():
            return 'A plan is already pending. Use "approve plan" or "cancel plan".'
        config = get_effective_runtime_config()
        assessment = analyze_plan(plan, config=config)
        if assessment.policy_outcome == PolicyOutcome.REJECTED:
            return _reject_plan_by_policy(plan, assessment.reasons)
        try:
            get_agent_controller().create_pending_task(plan)
        except ValueError as error:
            return str(error)
        if assessment.auto_execute:
            return get_agent_controller().approve_and_execute_current_task()
        summary = summarize_plan(plan, assessment)
        state.set_pending_plan(plan, summary)
        return summary


def show_pending_plan() -> str:
    with _APPROVAL_LOCK:
        expired_message = _reconcile_expired_or_stale_pending_plan_locked()
        if expired_message:
            return expired_message
        state = get_planner_state()
        if state.pending_plan is None:
            return "No pending plan."
        return state.pending_summary or summarize_plan(state.pending_plan)


def cancel_pending_plan() -> str:
    with _APPROVAL_LOCK:
        _reconcile_expired_or_stale_pending_plan_locked()
        if get_planner_state().pending_plan is None:
            return "No plan is currently pending."
        runtime_state = get_agent_runtime_state()
        current_task = runtime_state.current_task
        if current_task is not None and current_task.state.value == "running":
            return "No plan is currently pending."
        get_agent_controller().cancel_pending_or_running_task()
        reset_planner_state()
        _synchronize_terminal_pending_state_locked(status="cancelled", message="Pending plan cancelled.")
        return "Pending plan cancelled."


def approve_pending_plan(execute=None) -> str:
    """Approve exactly one validated plan; the callback is trusted executor code."""
    with _APPROVAL_LOCK:
        state = get_planner_state()
        if state.pending_plan is None:
            _reconcile_expired_or_stale_pending_plan_locked()
            return "No pending plan."
        if pending_plan_expired():
            return _expire_pending_plan_locked()
        if execute is None:
            state.approved = True
            plan = state.pending_plan
            reset_planner_state()
            _mark_pending_plan_executing_locked()
            result = get_agent_controller().approve_and_execute_current_task()
            try:
                from app.brain.intelligence.controller import get_intelligence_controller

                return get_intelligence_controller().finalize_execution(plan, result)
            except Exception:
                return result
        plan = state.pending_plan
        reset_planner_state()
        _mark_pending_plan_executing_locked()
    try:
        result = execute(plan)
    except Exception:
        return "Plan execution failed."
    return result if isinstance(result, str) else "Plan completed safely."


def has_active_pending_approval() -> bool:
    with _APPROVAL_LOCK:
        return _active_pending_approval_locked()


def _reject_plan_by_policy(plan: object, reasons: list[str]) -> str:
    reason = reasons[0] if reasons else "That command is not allowed."
    if hasattr(plan, "steps"):
        for step in getattr(plan, "steps", []):
            if getattr(step, "tool_name", "") == "terminal.execute":
                get_terminal_controller().record_policy_rejection(getattr(step, "arguments", {}), reason)
                break
    record_audit_event("plan_cancelled", message=reason)
    return f"Command rejected by terminal policy.\nReason: {reason}"


def _active_pending_approval_locked() -> bool:
    expired_message = _reconcile_expired_or_stale_pending_plan_locked()
    if expired_message:
        return False
    state = get_planner_state()
    return state.pending_plan is not None and not pending_plan_expired()


def _reconcile_expired_or_stale_pending_plan_locked() -> str:
    state = get_planner_state()
    if state.pending_plan is not None and pending_plan_expired():
        return _expire_pending_plan_locked()
    if state.pending_plan is None:
        runtime_state = get_agent_runtime_state()
        current_task = runtime_state.current_task
        if current_task is not None and current_task.state == AgentLifecycleState.PENDING_APPROVAL:
            _synchronize_terminal_pending_state_locked(status="approval_expired", message="The pending plan expired.")
    return ""


def _expire_pending_plan_locked() -> str:
    reset_planner_state()
    _synchronize_terminal_pending_state_locked(status="approval_expired", message="The pending plan expired.")
    record_audit_event("plan_expired", message="Pending plan expired before approval")
    return "The pending plan expired."


def _mark_pending_plan_executing_locked() -> None:
    try:
        from app.brain.intelligence.controller import get_intelligence_controller

        get_intelligence_controller().mark_pending_plan_executing()
    except Exception:
        return


def _synchronize_terminal_pending_state_locked(*, status: str, message: str) -> None:
    runtime_state = get_agent_runtime_state()
    current_task = runtime_state.current_task
    if current_task is not None and current_task.state == AgentLifecycleState.PENDING_APPROVAL:
        current_task.transition_to(AgentLifecycleState.CANCELLED)
        finish_time = planner_now()
        if isinstance(finish_time, datetime) and finish_time.tzinfo is None:
            finish_time = finish_time.replace(tzinfo=timezone.utc)
        current_task.finish_time = finish_time
        current_task.latest_safe_status_message = message
        current_task.failure_reason = message[:160]
        runtime_state.last_safe_status_message = "Agent state: idle."
        runtime_state.archive_current_task()
        _cleanup_vision_captures_for_task(current_task)
        _cleanup_navigation_sessions_for_task(current_task)
    elif current_task is not None and current_task.is_terminal():
        runtime_state.last_safe_status_message = "Agent state: idle."
        runtime_state.archive_current_task()
    try:
        from app.brain.intelligence.controller import get_intelligence_controller

        get_intelligence_controller().mark_pending_plan_terminal(status=status, message=message)
    except Exception:
        return


def _cleanup_vision_captures_for_task(task) -> None:
    """RFC-007B/RFC-007C: release any temporary vision captures owned by a task whose
    approval expired or was cancelled before reuse, mirroring
    AgentController._cleanup_browser_captures_for_task and
    AgentController._cleanup_desktop_captures_for_task. Browser and desktop cleanup are
    independent so a failure in one can never suppress the other."""
    try:
        from app.brain.vision.controller import get_vision_controller

        removed = get_vision_controller().cleanup_captures_for_task(owner_agent_task_id=task.task_id)
    except Exception:
        record_audit_event("vision_capture_cleanup_failed", task_id=task.task_id, message="task capture cleanup failed")
    else:
        if removed:
            record_audit_event("vision_capture_cleanup_completed", task_id=task.task_id, message=f"{removed} capture(s)")
    _cleanup_desktop_vision_captures_for_task(task)


def _cleanup_desktop_vision_captures_for_task(task) -> None:
    """RFC-007C: release any temporary desktop/window captures owned by a task whose
    approval expired or was cancelled before reuse, mirroring
    AgentController._cleanup_desktop_captures_for_task."""
    try:
        from app.brain.vision.controller import get_vision_controller

        removed = get_vision_controller().cleanup_desktop_captures_for_task(owner_agent_task_id=task.task_id)
    except Exception:
        record_audit_event("vision_desktop_capture_cleanup_failed", task_id=task.task_id, message="task desktop capture cleanup failed")
        return
    if removed:
        record_audit_event("vision_desktop_capture_cleanup_completed", task_id=task.task_id, message=f"{removed} capture(s)")


def _cleanup_navigation_sessions_for_task(task) -> None:
    """RFC-010: release any navigation sessions owned by a task whose approval expired or
    was cancelled before reuse, mirroring AgentController._cleanup_navigation_sessions_for_task.
    This is the same class of gap that bit browser and desktop capture cleanup above: a
    pending plan that simply times out is a separate code path from
    AgentController.cancel_pending_or_running_task, so cleanup has to be wired in here too,
    not only there."""
    try:
        from app.brain.location.controller import get_location_controller

        removed = get_location_controller().cleanup_sessions_for_task(owner_agent_task_id=task.task_id)
    except Exception:
        record_audit_event("navigation_session_cleanup_failed", task_id=task.task_id, message="task navigation session cleanup failed")
        return
    if removed:
        record_audit_event("navigation_session_cleanup_completed", task_id=task.task_id, message=f"{removed} session(s)")

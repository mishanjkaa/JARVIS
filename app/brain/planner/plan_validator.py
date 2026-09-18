from __future__ import annotations

from dataclasses import dataclass

from app.brain.planner.plan_models import AgentPlan
from app.brain.planner.result_references import is_result_reference, validate_reference

_BROWSER_SESSION_PRODUCERS = {"browser.start_session", "browser.get_active_session"}


@dataclass
class PlanValidation:
    valid: bool
    reason: str = ""


def validate_plan(plan: AgentPlan, max_steps: int = 5) -> PlanValidation:
    if not isinstance(plan, AgentPlan):
        return PlanValidation(False, "Plan is invalid.")
    if len(plan.steps) > max_steps:
        return PlanValidation(False, "Plan exceeds the maximum number of steps.")

    seen_ids: set[int] = set()
    seen_tools: dict[int, str] = {}
    browser_steps: list[tuple[int, str, dict, list[int]]] = []
    for step in plan.steps:
        if step.step_id in seen_ids or step.step_id <= 0:
            return PlanValidation(False, "Plan contains an invalid step id.")
        if not step.tool_name:
            return PlanValidation(False, "Plan contains an invalid tool.")
        for dependency in step.depends_on:
            if dependency not in seen_ids:
                return PlanValidation(False, "Plan dependency must reference an earlier step.")
        for name, value in step.arguments.items():
            if not is_result_reference(value):
                if isinstance(value, dict):
                    return PlanValidation(False, "Plan contains an invalid result reference.")
                continue
            try:
                from_step, field = validate_reference(
                    value,
                    current_step_id=step.step_id,
                    available_steps=seen_ids,
                    dependency_steps=set(step.depends_on),
                )
            except ValueError as error:
                reason = str(error)
                if reason == "reference dependency is missing":
                    return PlanValidation(False, "Plan result reference must appear in depends_on.")
                if reason == "forward references are not allowed":
                    return PlanValidation(False, "Plan dependency must reference an earlier step.")
                return PlanValidation(False, "Plan contains an invalid result reference.")
            if name == "session_id":
                if seen_tools.get(from_step) not in _BROWSER_SESSION_PRODUCERS:
                    return PlanValidation(False, "Plan contains an invalid browser session reference.")
                if field not in {"session_id", "display_value"}:
                    return PlanValidation(False, "Plan contains an invalid browser session reference.")
        seen_ids.add(step.step_id)
        seen_tools[step.step_id] = step.tool_name
        if step.tool_name.startswith("browser."):
            browser_steps.append((step.step_id, step.tool_name, step.arguments, step.depends_on))
    browser_validation = _validate_browser_lifecycle(browser_steps)
    if browser_validation is not None:
        return PlanValidation(False, browser_validation)
    return PlanValidation(True, "")


def _validate_browser_lifecycle(browser_steps: list[tuple[int, str, dict, list[int]]]) -> str | None:
    if not browser_steps:
        return None
    start_steps = [step for step in browser_steps if step[1] in _BROWSER_SESSION_PRODUCERS]
    close_steps = [step for step in browser_steps if step[1] == "browser.close_session"]
    if len(start_steps) != 1:
        return "Plan contains an invalid browser lifecycle."
    if len(close_steps) > 1:
        return "Plan contains an invalid browser lifecycle."
    if browser_steps[0][1] not in _BROWSER_SESSION_PRODUCERS:
        return "Plan contains an invalid browser lifecycle."
    start_step_id = start_steps[0][0]
    producer_tool = start_steps[0][1]
    if producer_tool != "browser.start_session":
        if browser_steps[-1][1] != "browser.close_session":
            return None if not close_steps else "Plan contains an invalid browser lifecycle."
        return "Plan contains an invalid browser lifecycle."
    if len(close_steps) != 1:
        return "Plan contains an invalid browser lifecycle."
    if browser_steps[-1][1] != "browser.close_session":
        return "Plan contains an invalid browser lifecycle."
    close_step_id, _, close_arguments, close_depends_on = close_steps[0]
    session_user_steps = [step for step in browser_steps if step[1] not in _BROWSER_SESSION_PRODUCERS | {"browser.close_session"}]
    if not session_user_steps:
        return "Plan contains an invalid browser lifecycle."
    final_session_user_step_id = session_user_steps[-1][0]
    session_reference = close_arguments.get("session_id")
    if not is_result_reference(session_reference):
        return "Plan contains an invalid browser lifecycle."
    if session_reference.get("from_step") != start_step_id:
        return "Plan contains an invalid browser lifecycle."
    if session_reference.get("field") not in {"session_id", "display_value"}:
        return "Plan contains an invalid browser lifecycle."
    if not {start_step_id, final_session_user_step_id}.issubset(set(close_depends_on)):
        return "Plan contains an invalid browser lifecycle."
    if close_step_id <= final_session_user_step_id:
        return "Plan contains an invalid browser lifecycle."
    return None

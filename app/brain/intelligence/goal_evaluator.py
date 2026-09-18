from __future__ import annotations

from typing import Any

from app.brain.agent.models import AgentLifecycleState, AgentTaskRecord
from app.brain.intelligence.models import DynamicPlan, GoalEvaluation, GoalEvaluationStatus, NaturalLanguageTask

def evaluate_goal(dynamic_plan: DynamicPlan, task_record: AgentTaskRecord | None, *, task: NaturalLanguageTask | None = None) -> GoalEvaluation:
    if task_record is None:
        return GoalEvaluation(GoalEvaluationStatus.FAILED, "Task did not start.", [])
    if task_record.state == AgentLifecycleState.CANCELLED:
        return GoalEvaluation(GoalEvaluationStatus.CANCELLED, "Task was cancelled.", [])
    if task_record.state == AgentLifecycleState.EMERGENCY_STOPPED:
        return GoalEvaluation(GoalEvaluationStatus.CANCELLED, "Task stopped during emergency stop.", [])
    if task_record.state == AgentLifecycleState.FAILED:
        evidence = _collect_evidence(task_record.step_results)
        details = _collect_evaluation_details(task_record.step_results, task)
        _merge_failed_step_details(details, task_record)
        if task_record.failure_reason:
            evidence.insert(0, task_record.failure_reason[:160])
            if task is not None and task.requested_operation == "git_status":
                details.setdefault("git_error_reason", _git_failure_reason(task_record.failure_reason))
            if task is not None and task.requested_operation.startswith("browser_"):
                details.setdefault("browser_error_reason", task_record.failure_reason[:160])
        return GoalEvaluation(GoalEvaluationStatus.FAILED, "Task failed during execution.", evidence[:6], details)

    details = _collect_evaluation_details(task_record.step_results, task)
    evidence, missing = _evaluate_requirements(dynamic_plan, task_record.step_results, task, details)
    if task_record.state != AgentLifecycleState.COMPLETED:
        summary = "Task partially completed." if evidence else "Task did not reach a final completed state."
        return GoalEvaluation(GoalEvaluationStatus.PARTIALLY_COMPLETED, summary, evidence + missing, details)
    if missing:
        return GoalEvaluation(GoalEvaluationStatus.PARTIALLY_COMPLETED, "Task partially completed.", evidence + missing, details)
    return GoalEvaluation(GoalEvaluationStatus.COMPLETED, "Task completed.", evidence, details)


def render_goal_evaluation(dynamic_plan: DynamicPlan, evaluation: GoalEvaluation, *, task: NaturalLanguageTask | None = None) -> str:
    if task is not None and task.requested_operation == "git_status":
        return _render_git_status_evaluation(evaluation)
    if task is not None and task.requested_operation.startswith("browser_visual_"):
        return _render_browser_visual_evaluation(evaluation, task)
    if task is not None and task.requested_operation.startswith("browser_"):
        return _render_browser_evaluation(evaluation, task)
    if task is not None and task.requested_operation.startswith("vision_"):
        return _render_vision_evaluation(evaluation, task)
    lines = [evaluation.summary, "", f"Goal: {dynamic_plan.goal}"]
    if evaluation.evidence:
        completed: list[str] = []
        missing: list[str] = []
        for item in evaluation.evidence[:8]:
            if item.startswith("Missing:"):
                missing.append(item[len("Missing:"):].strip())
            else:
                completed.append(item)
        if completed:
            lines.append("Completed:")
            for item in completed[:6]:
                lines.append(f"- {item}")
        if missing:
            lines.append("Missing:")
            for item in missing[:6]:
                lines.append(f"- {item}")
    elif dynamic_plan.success_criteria:
        lines.append("Expected:")
        for criterion in dynamic_plan.success_criteria[:6]:
            lines.append(f"- {criterion}")
    return "\n".join(lines)


def _evaluate_requirements(
    dynamic_plan: DynamicPlan,
    step_results: list[dict[str, Any]],
    task: NaturalLanguageTask | None,
    details: dict[str, Any],
) -> tuple[list[str], list[str]]:
    evidence: list[str] = []
    missing: list[str] = []
    if task is None:
        return _collect_evidence(step_results), []

    file_messages = [result.get("message", "") for result in step_results if isinstance(result.get("message"), str)]
    terminal_results = [result for result in step_results if result.get("tool_name") == "terminal.execute"]

    if not task.requested_operation.startswith("vision_"):
        for artifact in task.requested_artifacts:
            if any(artifact in message for message in file_messages):
                evidence.append(f"Created or updated {artifact}.")
            else:
                missing.append(f"Missing: {artifact} was not created or updated.")

    for content in task.requested_contents:
        if _plan_contains_written_content(dynamic_plan, content):
            evidence.append(f"Captured requested content: {content}.")
        else:
            missing.append(f"Missing: requested content {content!r}.")

    if task.requires_execution:
        if terminal_results:
            result = terminal_results[-1]
            message = str(result.get("message") or "").strip()
            if "Exit code: 0" in message:
                evidence.append("Executed the requested command successfully.")
            else:
                missing.append("Missing: successful execution evidence.")
        else:
            missing.append("Missing: execution step did not run.")

    if task.requires_stdout_match and task.requested_output_texts:
        matched = False
        for result in terminal_results:
            message = str(result.get("message") or "")
            if all(output in message for output in task.requested_output_texts):
                matched = True
                break
        if matched:
            for output in task.requested_output_texts:
                evidence.append(f"Verified stdout contained {output}.")
        else:
            missing.append("Missing: requested stdout was not verified.")

    if task.requires_tests:
        test_message = next((str(result.get("message") or "") for result in terminal_results if "test" in str(result.get("message") or "").lower() or "unittest" in str(result.get("message") or "").lower()), "")
        if test_message and "Exit code: 0" in test_message:
            evidence.append("Executed the requested test verification.")
        elif test_message:
            missing.append("Missing: test execution did not succeed.")
        else:
            missing.append("Missing: test verification did not run.")

    if task.read_only_task:
        write_detected = any("Created " in str(message) or "Wrote " in str(message) or "Appended " in str(message) for message in file_messages)
        if write_detected:
            missing.append("Missing: read-only task introduced unrelated write actions.")
        elif task.requested_summary:
            evidence.append("Used read-only operations only.")

    if task.requested_operation == "git_status" and task.requested_summary:
        git_files = _git_untracked_files(details)
        repository_detected = bool(details.get("repository_detected"))
        git_success = bool(details.get("git_success"))
        if git_success and repository_detected:
            evidence.append("Inspected Git status and captured untracked-file evidence.")
            if git_files:
                evidence.append("Reported exact untracked paths from porcelain Git output.")
            else:
                evidence.append("Reported that there are no untracked files.")
        else:
            missing.append("Missing: Git status summary evidence.")

    if task.requested_operation.startswith("browser_visual_"):
        if details.get("browser_session_started"):
            evidence.append("Started an isolated browser session.")
        elif details.get("browser_session_reused"):
            evidence.append("Reused the active isolated browser session.")
        else:
            missing.append("Missing: browser session evidence.")
        if details.get("browser_navigation_success"):
            evidence.append("Opened the requested page successfully.")
        elif task.raw_input.lower().startswith(("open ", "открой ")):
            missing.append("Missing: browser navigation evidence.")
        if details.get("browser_capture_id"):
            evidence.append("Captured the current browser viewport.")
        else:
            missing.append("Missing: browser viewport capture evidence.")
        if details.get("vision_grounded"):
            evidence.append("Captured grounded browser visual evidence.")
        else:
            missing.append("Missing: grounded browser visual evidence.")
        if task.requested_operation == "browser_visual_describe":
            if details.get("vision_description"):
                evidence.append("Produced a visual description grounded in the captured viewport.")
            else:
                missing.append("Missing: browser visual description evidence.")
        if task.requested_operation == "browser_visual_extract_text":
            if details.get("vision_extracted_text"):
                evidence.append("Extracted visible text from the captured viewport.")
            else:
                missing.append("Missing: browser visual OCR evidence.")
        if task.requested_operation == "browser_visual_find_element":
            match_outcome = _effective_browser_visual_match_outcome(details)
            if match_outcome == "found":
                evidence.append("Completed a grounded browser visual search with a verified match.")
            elif match_outcome == "not_found":
                evidence.append("Completed a grounded browser visual search with no verified match detected.")
            elif match_outcome == "uncertain":
                evidence.append("Completed a grounded browser visual search with an uncertain result.")
            else:
                missing.append("Missing: grounded browser visual search outcome.")
        if details.get("browser_session_started"):
            if details.get("browser_session_closed"):
                evidence.append("Closed the browser session safely.")
            else:
                missing.append("Missing: browser session close evidence.")
    elif task.requested_operation.startswith("browser_"):
        if details.get("browser_session_started"):
            evidence.append("Started an isolated browser session.")
        elif details.get("browser_session_reused"):
            evidence.append("Reused the active isolated browser session.")
        else:
            missing.append("Missing: browser session evidence.")
        if task.requested_operation in {"browser_title", "browser_summary", "browser_screenshot", "browser_navigation"}:
            if details.get("browser_navigation_success"):
                evidence.append("Opened the requested page successfully.")
            else:
                missing.append("Missing: browser navigation evidence.")
        if task.requested_operation == "browser_title":
            if details.get("browser_title"):
                evidence.append("Captured the actual page title.")
            else:
                missing.append("Missing: page title evidence.")
        if task.requested_operation == "browser_summary":
            if details.get("browser_text"):
                evidence.append("Captured visible page text for summary grounding.")
            else:
                missing.append("Missing: visible page text evidence.")
        if task.requested_operation == "browser_screenshot":
            if details.get("browser_screenshot_path"):
                evidence.append("Saved the requested screenshot.")
            else:
                missing.append("Missing: screenshot evidence.")
        if task.requested_operation == "browser_scroll":
            if details.get("browser_scroll_y") is not None:
                evidence.append("Scrolled the current page.")
            else:
                missing.append("Missing: browser scroll evidence.")
        if task.requested_operation == "browser_scroll_to_element":
            if details.get("browser_target_description") or details.get("browser_scroll_y") is not None:
                evidence.append("Scrolled to the requested page element.")
            else:
                missing.append("Missing: browser element scroll evidence.")
        if task.requested_operation == "browser_click":
            if details.get("browser_click_success"):
                evidence.append("Clicked the requested browser element.")
            else:
                missing.append("Missing: browser click evidence.")
        if task.requested_operation == "browser_reload":
            if details.get("browser_reload_success"):
                evidence.append("Reloaded the current page.")
            else:
                missing.append("Missing: browser reload evidence.")
        if task.requested_operation == "browser_new_tab":
            if details.get("browser_new_tab_success"):
                evidence.append("Opened the requested page in a new tab.")
            else:
                missing.append("Missing: browser new-tab evidence.")
        if task.requested_operation == "browser_switch_tab":
            if details.get("browser_tab_switch_success"):
                evidence.append("Switched to the requested browser tab.")
            else:
                missing.append("Missing: browser tab-switch evidence.")
        if task.requested_operation == "browser_list_tabs":
            if isinstance(details.get("browser_tabs"), list):
                evidence.append("Captured the current browser tab list.")
            else:
                missing.append("Missing: browser tab-list evidence.")
        if task.requested_operation == "browser_close_tab":
            if details.get("browser_tab_close_success"):
                evidence.append("Closed the requested browser tab.")
            else:
                missing.append("Missing: browser tab-close evidence.")
        if task.requested_operation == "browser_clickable_inspection":
            if details.get("browser_clickable_count") is not None:
                evidence.append("Captured clickable element metadata.")
            else:
                missing.append("Missing: clickable element inspection evidence.")
        if details.get("browser_session_started"):
            if details.get("browser_session_closed"):
                evidence.append("Closed the browser session safely.")
            else:
                missing.append("Missing: browser session close evidence.")
        if task.requested_operation == "browser_form_inspection":
            if details.get("browser_control_count") is not None:
                evidence.append("Inspected visible form controls.")
            else:
                missing.append("Missing: form-control inspection evidence.")
        if task.requested_operation == "browser_form_fill":
            if details.get("browser_input_changed"):
                evidence.append("Entered text into the requested form control.")
            else:
                missing.append("Missing: browser text-entry evidence.")
            if details.get("browser_submitted"):
                missing.append("Missing: fill-only task must not submit a form.")
        if task.requested_operation == "browser_form_clear":
            if details.get("browser_input_cleared"):
                evidence.append("Cleared the requested form control.")
            else:
                missing.append("Missing: browser clear-input evidence.")
            if details.get("browser_submitted"):
                missing.append("Missing: clear-only task must not submit a form.")
        if task.requested_operation in {"browser_form_submit", "browser_form_fill_submit"}:
            if task.requested_operation == "browser_form_fill_submit":
                if details.get("browser_input_changed"):
                    evidence.append("Entered text into the requested form control before submission.")
                else:
                    missing.append("Missing: browser text-entry evidence before submission.")
            if details.get("browser_submitted"):
                evidence.append("Submitted the requested browser form.")
            else:
                missing.append("Missing: browser form submission evidence.")
            if details.get("browser_post_submit_evidence"):
                evidence.append("Captured grounded post-submit browser evidence.")
            else:
                missing.append("Missing: post-submit browser evidence.")
    if task.requested_operation.startswith("vision_"):
        if details.get("vision_grounded"):
            evidence.append("Captured grounded Vision evidence from the requested local image.")
        else:
            missing.append("Missing: grounded Vision evidence.")
        if task.requested_operation == "vision_describe_image":
            if details.get("vision_description"):
                evidence.append("Produced an image description grounded in the local image.")
            else:
                missing.append("Missing: image description evidence.")
        if task.requested_operation == "vision_extract_text":
            if details.get("vision_extracted_text"):
                evidence.append("Extracted visible text from the local image.")
            else:
                missing.append("Missing: image OCR evidence.")
        if task.requested_operation == "vision_find_visual_element":
            match_outcome = str(details.get("vision_match_outcome") or "").strip()
            if not match_outcome:
                if details.get("vision_no_match"):
                    match_outcome = "not_found"
                elif details.get("vision_region_count"):
                    match_outcome = "found"
            if match_outcome == "found":
                evidence.append("Completed a grounded visual search with a verified match.")
            elif match_outcome == "not_found":
                evidence.append("Completed a grounded visual search with no verified match detected.")
            elif match_outcome == "uncertain":
                evidence.append("Completed a grounded visual search with an uncertain result.")
            else:
                missing.append("Missing: visual search outcome.")

    return evidence[:6], missing[:6]


def _collect_evidence(step_results: list[dict[str, Any]]) -> list[str]:
    evidence: list[str] = []
    for result in step_results:
        message = result.get("message")
        if isinstance(message, str) and message.strip():
            evidence.append(message.strip()[:160])
    return evidence[:6]


def _plan_contains_written_content(dynamic_plan: DynamicPlan, content: str) -> bool:
    lowered = content.lower()
    for step in dynamic_plan.steps:
        value = step.arguments.get("text")
        if isinstance(value, str) and lowered in value.lower():
            return True
    return False


def _collect_evaluation_details(step_results: list[dict[str, Any]], task: NaturalLanguageTask | None) -> dict[str, Any]:
    if task is None or task.requested_operation != "git_status":
        if task is not None and task.requested_operation.startswith("browser_visual_"):
            details = _collect_browser_details(step_results)
            details.update(_collect_vision_details(step_results))
            return details
        if task is not None and task.requested_operation.startswith("browser_"):
            return _collect_browser_details(step_results)
        if task is not None and task.requested_operation.startswith("vision_"):
            return _collect_vision_details(step_results)
        return {}
    for result in reversed(step_results):
        reference_fields = result.get("reference_fields")
        if not isinstance(reference_fields, dict):
            continue
        if reference_fields.get("terminal_operation") != "git_untracked_files":
            continue
        details: dict[str, Any] = {
            "git_success": bool(reference_fields.get("success")),
            "repository_detected": bool(reference_fields.get("repository_detected")),
            "git_untracked_files": _git_untracked_files(reference_fields),
            "exit_code": reference_fields.get("exit_code"),
            "output_truncated": bool(reference_fields.get("output_truncated")),
            "git_error_category": str(reference_fields.get("git_error_category") or "").strip(),
            "git_error_reason": str(reference_fields.get("git_error_reason") or "").strip(),
        }
        return details
    return {}


def _git_untracked_files(details: dict[str, Any]) -> list[str]:
    raw_value = details.get("git_untracked_files")
    if not isinstance(raw_value, list):
        return []
    return [str(item).strip() for item in raw_value if str(item).strip()]


def _collect_browser_details(step_results: list[dict[str, Any]]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for result in step_results:
        if not bool(result.get("success", True)) and str(result.get("tool_name") or "").startswith("browser."):
            details.setdefault("browser_failed_tool", str(result.get("tool_name") or "").strip())
            details.setdefault("browser_failed_step_reason", str(result.get("message") or "").strip())
        reference_fields = result.get("reference_fields")
        if not isinstance(reference_fields, dict):
            continue
        operation = str(reference_fields.get("browser_operation") or "").strip()
        if operation == "start_session":
            details["browser_session_started"] = bool(reference_fields.get("success"))
            details["browser_session_id"] = str(reference_fields.get("session_id") or "").strip()
        elif operation == "get_active_session":
            details["browser_session_reused"] = bool(reference_fields.get("success"))
            details["browser_session_id"] = str(reference_fields.get("session_id") or details.get("browser_session_id") or "").strip()
            details["browser_final_url"] = str(reference_fields.get("url") or details.get("browser_final_url") or "").strip()
            details["browser_title"] = str(reference_fields.get("title") or details.get("browser_title") or "").strip()
        elif operation == "open_url":
            details["browser_navigation_success"] = bool(reference_fields.get("success"))
            details["browser_requested_url"] = str(reference_fields.get("requested_url") or "").strip()
            details["browser_final_url"] = str(reference_fields.get("final_url") or "").strip()
            details["browser_title"] = str(reference_fields.get("title") or "").strip()
            details["browser_error_reason"] = str(reference_fields.get("error_reason") or details.get("browser_error_reason") or "").strip()
        elif operation == "capture_view":
            details["browser_capture_id"] = str(reference_fields.get("capture_id") or "").strip()
            details["browser_capture_url"] = str(reference_fields.get("url") or "").strip()
            details["browser_capture_origin"] = str(reference_fields.get("origin") or "").strip()
            details["browser_capture_page_version"] = reference_fields.get("page_version")
            details["browser_capture_width"] = reference_fields.get("viewport_width")
            details["browser_capture_height"] = reference_fields.get("viewport_height")
        elif operation == "open_new_tab":
            details["browser_new_tab_success"] = bool(reference_fields.get("success"))
            details["browser_navigation_success"] = bool(reference_fields.get("success"))
            details["browser_final_url"] = str(reference_fields.get("final_url") or details.get("browser_final_url") or "").strip()
            details["browser_title"] = str(reference_fields.get("title") or details.get("browser_title") or "").strip()
            details["browser_tab_count"] = reference_fields.get("tab_count")
            details["browser_active_tab_id"] = str(reference_fields.get("active_tab_id") or details.get("browser_active_tab_id") or "").strip()
        elif operation == "get_page_info":
            details["browser_title"] = str(reference_fields.get("title") or details.get("browser_title") or "").strip()
            details["browser_final_url"] = str(reference_fields.get("url") or details.get("browser_final_url") or "").strip()
        elif operation == "extract_visible_text":
            details["browser_text"] = str(reference_fields.get("text") or "").strip()
            details["browser_title"] = str(reference_fields.get("title") or details.get("browser_title") or "").strip()
        elif operation == "inspect_elements":
            details["browser_clickable_count"] = reference_fields.get("element_count")
            details["browser_elements"] = reference_fields.get("elements")
        elif operation == "inspect_clickable_elements":
            details["browser_clickable_count"] = reference_fields.get("element_count")
            details["browser_elements"] = reference_fields.get("elements")
        elif operation == "inspect_form_controls":
            details["browser_control_count"] = reference_fields.get("control_count")
            details["browser_controls"] = reference_fields.get("controls")
            details["browser_title"] = str(reference_fields.get("title") or details.get("browser_title") or "").strip()
            details["browser_final_url"] = str(reference_fields.get("url") or details.get("browser_final_url") or "").strip()
        elif operation == "scroll_page":
            details["browser_scroll_y"] = reference_fields.get("scroll_y")
            details["browser_final_url"] = str(reference_fields.get("url") or details.get("browser_final_url") or "").strip()
            details["browser_title"] = str(reference_fields.get("title") or details.get("browser_title") or "").strip()
        elif operation == "scroll_to_element":
            details["browser_scroll_y"] = reference_fields.get("scroll_y")
            details["browser_target_description"] = str(reference_fields.get("target_description") or "").strip()
            details["browser_final_url"] = str(reference_fields.get("url") or details.get("browser_final_url") or "").strip()
            details["browser_title"] = str(reference_fields.get("title") or details.get("browser_title") or "").strip()
        elif operation == "click_element":
            details["browser_click_success"] = bool(reference_fields.get("success"))
            details["browser_target_description"] = str(reference_fields.get("target_description") or "").strip()
            details["browser_target_type"] = str(reference_fields.get("target_type") or "").strip()
            details["browser_navigated"] = bool(reference_fields.get("navigated"))
            details["browser_final_url"] = str(reference_fields.get("final_url") or details.get("browser_final_url") or "").strip()
            details["browser_title"] = str(reference_fields.get("title") or details.get("browser_title") or "").strip()
        elif operation == "input_text":
            details["browser_input_changed"] = bool(reference_fields.get("field_changed"))
            details["browser_target_description"] = str(reference_fields.get("target_description") or details.get("browser_target_description") or "").strip()
            details["browser_target_type"] = str(reference_fields.get("target_type") or details.get("browser_target_type") or "").strip()
            details["browser_input_text_length"] = reference_fields.get("text_length")
            details["browser_final_url"] = str(reference_fields.get("url") or details.get("browser_final_url") or "").strip()
            details["browser_title"] = str(reference_fields.get("title") or details.get("browser_title") or "").strip()
        elif operation == "clear_input":
            details["browser_input_cleared"] = bool(reference_fields.get("field_changed"))
            details["browser_target_description"] = str(reference_fields.get("target_description") or details.get("browser_target_description") or "").strip()
            details["browser_target_type"] = str(reference_fields.get("target_type") or details.get("browser_target_type") or "").strip()
            details["browser_final_url"] = str(reference_fields.get("url") or details.get("browser_final_url") or "").strip()
            details["browser_title"] = str(reference_fields.get("title") or details.get("browser_title") or "").strip()
        elif operation == "submit_form":
            details["browser_submitted"] = bool(reference_fields.get("submitted"))
            details["browser_target_description"] = str(reference_fields.get("target_description") or details.get("browser_target_description") or "").strip()
            details["browser_target_type"] = str(reference_fields.get("target_type") or details.get("browser_target_type") or "").strip()
            details["browser_form_action"] = str(reference_fields.get("form_action") or "").strip()
            details["browser_form_method"] = str(reference_fields.get("form_method") or "").strip()
            details["browser_error_category"] = str(reference_fields.get("error_category") or details.get("browser_error_category") or "").strip()
            details["browser_error_reason"] = str(reference_fields.get("error_reason") or details.get("browser_error_reason") or "").strip()
            details["browser_confirmation_text"] = str(reference_fields.get("confirmation_text") or "").strip()
            details["browser_confirmation_text_truncated"] = bool(reference_fields.get("confirmation_text_truncated"))
            details["browser_navigated"] = bool(reference_fields.get("navigated"))
            details["browser_final_url"] = str(reference_fields.get("final_url") or details.get("browser_final_url") or "").strip()
            details["browser_title"] = str(reference_fields.get("title") or details.get("browser_title") or "").strip()
            details["browser_post_submit_evidence"] = bool(
                details.get("browser_final_url") or details.get("browser_title") or details.get("browser_confirmation_text")
            )
        elif operation and not bool(reference_fields.get("success", True)):
            details["browser_error_category"] = str(reference_fields.get("error_category") or details.get("browser_error_category") or "").strip()
            details["browser_error_reason"] = str(reference_fields.get("error_reason") or details.get("browser_error_reason") or "").strip()
        elif operation == "reload_page":
            details["browser_reload_success"] = bool(reference_fields.get("success"))
            details["browser_final_url"] = str(reference_fields.get("final_url") or details.get("browser_final_url") or "").strip()
            details["browser_title"] = str(reference_fields.get("title") or details.get("browser_title") or "").strip()
        elif operation == "switch_tab":
            details["browser_tab_switch_success"] = bool(reference_fields.get("success"))
            details["browser_active_tab_id"] = str(reference_fields.get("active_tab_id") or "").strip()
            details["browser_tab_count"] = reference_fields.get("tab_count")
            details["browser_tabs"] = reference_fields.get("tabs")
            details["browser_final_url"] = str(reference_fields.get("url") or details.get("browser_final_url") or "").strip()
            details["browser_title"] = str(reference_fields.get("title") or details.get("browser_title") or "").strip()
        elif operation == "list_tabs":
            details["browser_tab_count"] = reference_fields.get("tab_count")
            details["browser_active_tab_id"] = str(reference_fields.get("active_tab_id") or "").strip()
            details["browser_tabs"] = reference_fields.get("tabs")
        elif operation == "close_tab":
            details["browser_tab_close_success"] = bool(reference_fields.get("success"))
            details["browser_tab_count"] = reference_fields.get("tab_count")
            details["browser_active_tab_id"] = str(reference_fields.get("active_tab_id") or "").strip()
            details["browser_tabs"] = reference_fields.get("tabs")
            details["browser_final_url"] = str(reference_fields.get("url") or details.get("browser_final_url") or "").strip()
            details["browser_title"] = str(reference_fields.get("title") or details.get("browser_title") or "").strip()
        elif operation == "take_screenshot":
            details["browser_screenshot_path"] = str(reference_fields.get("screenshot_path") or "").strip()
        elif operation == "close_session":
            details["browser_session_closed"] = bool(reference_fields.get("success"))
    return details


def _render_git_status_evaluation(evaluation: GoalEvaluation) -> str:
    files = _git_untracked_files(evaluation.details)
    if evaluation.status == GoalEvaluationStatus.COMPLETED and bool(evaluation.details.get("repository_detected")):
        if files:
            return "\n".join(["Untracked files:", *[f"- {path}" for path in files]])
        return "There are no untracked files."
    reason = str(evaluation.details.get("git_error_reason") or "").strip()
    if reason:
        return f"Git status could not be completed.\nReason: {reason}"
    return "Git status could not be completed."


def _render_browser_evaluation(evaluation: GoalEvaluation, task: NaturalLanguageTask) -> str:
    reason = str(evaluation.details.get("browser_error_reason") or "").strip()
    if evaluation.status != GoalEvaluationStatus.COMPLETED:
        failed_step = evaluation.details.get("browser_failed_step_index")
        failed_tool = str(evaluation.details.get("browser_failed_tool") or "").strip()
        safe_reason = _browser_safe_failure_reason(evaluation.details)
        if isinstance(failed_step, int) and failed_step > 0 and failed_tool and safe_reason:
            return f"Browser task failed at step {failed_step} ({failed_tool}): {safe_reason}"
        if safe_reason:
            return f"Browser task could not be completed.\nReason: {safe_reason}"
        return "Browser task could not be completed."
    if task.requested_operation == "browser_title":
        title = str(evaluation.details.get("browser_title") or "").strip()
        final_url = str(evaluation.details.get("browser_final_url") or "").strip()
        lines = [f"Page title: {title or '(empty title)'}"]
        if final_url:
            lines.append(f"URL: {final_url}")
        return "\n".join(lines)
    if task.requested_operation == "browser_summary":
        title = str(evaluation.details.get("browser_title") or "").strip()
        text = str(evaluation.details.get("browser_text") or "").strip()
        summary = _browser_text_summary(text)
        if title:
            return f"Page title: {title}\nSummary: {summary}"
        return f"Summary: {summary}"
    if task.requested_operation == "browser_screenshot":
        path = str(evaluation.details.get("browser_screenshot_path") or "").strip()
        return f"Saved screenshot to {path}."
    if task.requested_operation == "browser_scroll":
        title = str(evaluation.details.get("browser_title") or "").strip()
        scroll_y = evaluation.details.get("browser_scroll_y")
        suffix = f" Current title: {title}." if title else ""
        return f"Scrolled the page to position {scroll_y}.{suffix}".strip()
    if task.requested_operation == "browser_scroll_to_element":
        target = str(evaluation.details.get("browser_target_description") or "the requested element").strip()
        return f"Scrolled to {target}."
    if task.requested_operation == "browser_click":
        target = str(evaluation.details.get("browser_target_description") or "the requested element").strip()
        final_url = str(evaluation.details.get("browser_final_url") or "").strip()
        if final_url:
            return f"Clicked {target}.\nURL: {final_url}"
        return f"Clicked {target}."
    if task.requested_operation == "browser_reload":
        final_url = str(evaluation.details.get("browser_final_url") or "").strip()
        return f"Reloaded {final_url}." if final_url else "Reloaded the current page."
    if task.requested_operation == "browser_new_tab":
        final_url = str(evaluation.details.get("browser_final_url") or "").strip()
        active_tab_id = str(evaluation.details.get("browser_active_tab_id") or "").strip()
        return f"Opened a new tab: {final_url}\nActive tab: {active_tab_id}" if active_tab_id else f"Opened a new tab: {final_url}"
    if task.requested_operation == "browser_switch_tab":
        active_tab_id = str(evaluation.details.get("browser_active_tab_id") or "").strip()
        title = str(evaluation.details.get("browser_title") or "").strip()
        lines = [f"Switched to tab {active_tab_id or '(unknown tab)'}."]
        if title:
            lines.append(f"Page title: {title}")
        return "\n".join(lines)
    if task.requested_operation == "browser_list_tabs":
        tabs = evaluation.details.get("browser_tabs")
        if isinstance(tabs, list) and tabs:
            lines = ["Open tabs:"]
            for item in tabs[:8]:
                if not isinstance(item, dict):
                    continue
                marker = "*" if item.get("active") else "-"
                lines.append(f"{marker} {item.get('tab_id', '(unknown)')} | {item.get('title') or '(untitled)'} | {item.get('url') or '(no page loaded)'}")
            return "\n".join(lines)
        return "There are no open browser tabs."
    if task.requested_operation == "browser_close_tab":
        remaining = evaluation.details.get("browser_tab_count")
        return f"Closed the requested tab.\nRemaining tabs: {remaining}" if remaining is not None else "Closed the requested tab."
    if task.requested_operation == "browser_clickable_inspection":
        count = evaluation.details.get("browser_clickable_count")
        return f"Inspected {count} clickable elements." if count is not None else "Inspected clickable elements."
    if task.requested_operation == "browser_form_inspection":
        count = evaluation.details.get("browser_control_count")
        return f"Inspected {count} visible form controls." if count is not None else "Inspected visible form controls."
    if task.requested_operation == "browser_form_fill":
        target = str(evaluation.details.get("browser_target_description") or "the requested form control").strip()
        text_length = evaluation.details.get("browser_input_text_length")
        if isinstance(text_length, int):
            return f"Entered text into {target}.\nText length: {text_length}"
        return f"Entered text into {target}."
    if task.requested_operation == "browser_form_clear":
        target = str(evaluation.details.get("browser_target_description") or "the requested form control").strip()
        return f"Cleared {target}."
    if task.requested_operation in {"browser_form_submit", "browser_form_fill_submit"}:
        final_url = str(evaluation.details.get("browser_final_url") or "").strip()
        title = str(evaluation.details.get("browser_title") or "").strip()
        lines = ["Submitted the requested form."]
        if title:
            lines.append(f"Page title: {title}")
        if final_url:
            lines.append(f"URL: {final_url}")
        return "\n".join(lines)
    final_url = str(evaluation.details.get("browser_final_url") or "").strip()
    title = str(evaluation.details.get("browser_title") or "").strip()
    if title and final_url:
        return f"Opened {final_url}.\nPage title: {title}"
    if final_url:
        return f"Opened {final_url}."
    return "Browser task completed."


def _render_browser_visual_evaluation(evaluation: GoalEvaluation, task: NaturalLanguageTask) -> str:
    if evaluation.status != GoalEvaluationStatus.COMPLETED:
        reason = str(evaluation.details.get("vision_error_reason") or evaluation.details.get("browser_error_reason") or "").strip()
        if reason:
            return f"Browser visual task could not be completed.\nReason: {reason}"
        return "Browser visual task could not be completed."
    source_url = str(
        evaluation.details.get("vision_source_url")
        or evaluation.details.get("browser_final_url")
        or evaluation.details.get("browser_capture_url")
        or ""
    ).strip()
    captcha_reason = str(evaluation.details.get("vision_captcha_reason") or "").strip()
    lines: list[str] = []
    if task.requested_operation == "browser_visual_describe":
        description = str(evaluation.details.get("vision_description") or "").strip()
        lines.append(description or "The captured browser viewport was described successfully.")
    elif task.requested_operation == "browser_visual_extract_text":
        text = str(evaluation.details.get("vision_extracted_text") or "").strip()
        lines.append(text or "No visible text was extracted from the captured browser viewport.")
    elif task.requested_operation == "browser_visual_find_element":
        lines.extend(_render_visual_find_lines(evaluation, task))
    if source_url:
        lines.append(f"Source URL: {source_url}")
    if captcha_reason:
        lines.append(captcha_reason)
    return "\n".join(lines)


def _collect_vision_details(step_results: list[dict[str, Any]]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for result in step_results:
        reference_fields = result.get("reference_fields")
        if not isinstance(reference_fields, dict):
            continue
        operation = str(reference_fields.get("vision_operation") or "").strip()
        if not operation:
            continue
        details["vision_operation"] = operation
        details["vision_grounded"] = bool(reference_fields.get("grounded"))
        details["vision_safe_display_name"] = str(reference_fields.get("safe_display_name") or "").strip()
        details["vision_description"] = str(reference_fields.get("description") or "").strip()
        details["vision_extracted_text"] = str(reference_fields.get("extracted_text") or "").strip()
        details["vision_ocr_blocks"] = reference_fields.get("ocr_blocks")
        details["vision_visual_regions"] = reference_fields.get("visual_regions")
        details["vision_region_count"] = len(reference_fields.get("visual_regions", [])) if isinstance(reference_fields.get("visual_regions"), list) else 0
        details["vision_no_match"] = bool(reference_fields.get("no_match"))
        details["vision_match_outcome"] = str(reference_fields.get("match_outcome") or "").strip()
        details["vision_error_reason"] = str(reference_fields.get("error_reason") or "").strip()
        details["vision_error_category"] = str(reference_fields.get("error_category") or "").strip()
        details["vision_confidence"] = reference_fields.get("confidence")
        details["vision_warnings"] = reference_fields.get("warnings")
        details["vision_requested_query"] = str(reference_fields.get("requested_query") or "").strip()
        details["vision_source_url"] = str(reference_fields.get("source_url") or "").strip()
        details["vision_source_origin"] = str(reference_fields.get("source_origin") or "").strip()
        details["vision_instruction_trust"] = str(reference_fields.get("instruction_trust") or "").strip()
        details["vision_factual_credibility"] = str(reference_fields.get("factual_credibility") or "").strip()
        details["vision_captcha_suspected"] = bool(reference_fields.get("captcha_suspected"))
        details["vision_captcha_reason"] = str(reference_fields.get("captcha_reason") or "").strip()
        grounding_diagnostics = reference_fields.get("grounding_diagnostics")
        if isinstance(grounding_diagnostics, dict):
            details["vision_grounding_diagnostics"] = dict(grounding_diagnostics)
    return details


def _render_vision_evaluation(evaluation: GoalEvaluation, task: NaturalLanguageTask) -> str:
    if evaluation.status != GoalEvaluationStatus.COMPLETED:
        reason = str(evaluation.details.get("vision_error_reason") or "").strip()
        if reason:
            return f"Vision task could not be completed.\nReason: {reason}"
        return "Vision task could not be completed."
    if task.requested_operation == "vision_describe_image":
        description = str(evaluation.details.get("vision_description") or "").strip()
        return description or "The image was described successfully."
    if task.requested_operation == "vision_extract_text":
        text = str(evaluation.details.get("vision_extracted_text") or "").strip()
        return text or "No visible text was extracted from the image."
    if task.requested_operation == "vision_find_visual_element":
        return "\n".join(_render_visual_find_lines(evaluation, task))
    return "Vision task completed."


def _render_visual_find_lines(evaluation: GoalEvaluation, task: NaturalLanguageTask) -> list[str]:
    details = evaluation.details
    query = str(details.get("vision_requested_query") or "").strip() or _requested_visual_query(task)
    match_outcome = _effective_browser_visual_match_outcome(details) if task.requested_operation == "browser_visual_find_element" else str(details.get("vision_match_outcome") or "").strip()
    if task.requested_operation != "browser_visual_find_element" and not match_outcome:
        if details.get("vision_no_match"):
            match_outcome = "not_found"
        elif details.get("vision_region_count"):
            match_outcome = "found"
    warning_lines = _safe_warning_lines(details.get("vision_warnings"))
    if match_outcome == "not_found":
        return [f"No verified visual match was detected for: {query}."]
    if match_outcome == "uncertain":
        lines = [f"A possible visual match could not be verified for: {query}."]
        lines.extend(warning_lines[:2])
        return lines
    region = _first_visual_region(details)
    if region is None:
        return ["Visual search completed, but no grounded match details were available."]
    label = str(region.get("label") or "match").strip()
    lines = [f"Found visual match: {label}."]
    confidence = region.get("confidence")
    if isinstance(confidence, (int, float)):
        lines.append(f"Confidence: {float(confidence):.2f}")
    visible_text = str(region.get("visible_text") or "").strip()
    if visible_text and visible_text != label:
        lines.append(f"Visible text: {visible_text}")
    attributes = region.get("attributes")
    if isinstance(attributes, dict):
        role = str(attributes.get("role") or "").strip()
        if role:
            lines.append(f"Role: {role}")
        relative_location = str(attributes.get("relative_location") or "").strip()
        if relative_location:
            lines.append(f"Relative location: {relative_location}")
    verification = str(region.get("verification") or "").strip()
    if verification:
        lines.append(f"Verification: {verification}")
    bounding_box = region.get("bounding_box")
    if isinstance(bounding_box, dict):
        x = bounding_box.get("x")
        y = bounding_box.get("y")
        width = bounding_box.get("width")
        height = bounding_box.get("height")
        if all(isinstance(value, (int, float)) for value in (x, y, width, height)):
            lines.append(
                "Bounding box: "
                f"x={float(x):.3f}, y={float(y):.3f}, width={float(width):.3f}, height={float(height):.3f}"
            )
    return lines


def _effective_browser_visual_match_outcome(details: dict[str, Any]) -> str:
    match_outcome = str(details.get("vision_match_outcome") or "").strip()
    if not match_outcome:
        if details.get("vision_no_match"):
            return "not_found"
        if details.get("vision_region_count"):
            match_outcome = "found"
    if match_outcome == "found":
        region = _first_visual_region(details)
        verification = str(region.get("verification") or "").strip() if isinstance(region, dict) else ""
        diagnostics = details.get("vision_grounding_diagnostics") if isinstance(details.get("vision_grounding_diagnostics"), dict) else {}
        if verification not in {"dom_verified", "crop_verified", "pixel_verified"}:
            return "uncertain"
        if verification == "dom_verified" and str(diagnostics.get("dom_grounding_result") or "").strip() != "matched":
            return "uncertain"
        if verification == "crop_verified" and str(diagnostics.get("crop_observer_outcome") or "").strip() != "matched":
            return "uncertain"
        if verification == "pixel_verified" and str(diagnostics.get("full_frame_observer_outcome") or "").strip() != "matched":
            return "uncertain"
    return match_outcome


def _first_visual_region(details: dict[str, Any]) -> dict[str, Any] | None:
    regions = details.get("vision_visual_regions")
    if not isinstance(regions, list):
        return None
    for item in regions:
        if isinstance(item, dict):
            return item
    return None


def _safe_warning_lines(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value[:4] if isinstance(item, str) and str(item).strip()]


def _requested_visual_query(task: NaturalLanguageTask) -> str:
    requested_query = str(getattr(task, "requested_visual_query", "") or "").strip()
    if requested_query:
        return requested_query
    for item in task.requested_contents:
        value = str(item or "").strip()
        if value:
            return value
    return "the requested element"


def _browser_text_summary(text: str) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return "The page did not contain visible text."
    if len(cleaned) <= 220:
        return cleaned
    sentence_end = cleaned.find(". ", 120)
    if sentence_end != -1 and sentence_end <= 220:
        return cleaned[: sentence_end + 1]
    return cleaned[:217].rstrip() + "..."


def _git_failure_reason(value: str) -> str:
    lowered = value.lower()
    if "not a git repository" in lowered:
        return "Not a Git repository."
    if "permission denied" in lowered:
        return "Git command reported permission denied warnings."
    return value[:160]


def _merge_failed_step_details(details: dict[str, Any], task_record: AgentTaskRecord) -> None:
    if task_record.failed_step_index > 0:
        details.setdefault("browser_failed_step_index", task_record.failed_step_index)
    if task_record.failed_tool_name:
        details.setdefault("browser_failed_tool", task_record.failed_tool_name)
    if task_record.failed_error_category:
        details.setdefault("browser_error_category", task_record.failed_error_category)
    if task_record.failure_reason:
        details.setdefault("browser_failed_step_reason", task_record.failure_reason[:160])


def _browser_safe_failure_reason(details: dict[str, Any]) -> str:
    reason = str(details.get("browser_error_reason") or "").strip()
    if reason:
        return reason[:160]
    step_reason = str(details.get("browser_failed_step_reason") or "").strip()
    if step_reason.lower().startswith("step ") and " failed: " in step_reason.lower():
        parts = step_reason.split(": ", 1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()[:160]
    return step_reason[:160]

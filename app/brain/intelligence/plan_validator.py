from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

from app.brain.browser.validation import validate_browser_tool_arguments
from app.brain.filesystem.errors import FilesystemPathError
from app.brain.filesystem.path_policy import resolve_path
from app.brain.intelligence.models import DynamicPlan, NaturalLanguageTask, PlanValidationResult, ToolCatalogEntry
from app.brain.intelligence.vision_query import canonicalize_vision_find_query_text, looks_like_json_container_text, normalize_vision_query
from app.brain.planner.result_references import allows_result_reference, is_result_reference, validate_reference
from app.brain.planner.plan_models import AgentPlan, AgentStep
from app.brain.terminal.errors import TerminalError
from app.brain.terminal.policy import decision_from_arguments as terminal_decision_from_arguments
from app.brain.tools.registry import ToolRegistry
from app.brain.tools.validators import validate_arguments

_WRITE_TOOLS = {
    "filesystem.create_directory",
    "filesystem.create_text_file",
    "filesystem.write_text_file",
    "filesystem.append_text_file",
    "filesystem.rename_path",
    "filesystem.copy_path",
    "filesystem.move_path",
    "filesystem.delete_path",
    "notes.create",
    "memory.remember",
    "tasks.create",
}
_FALLBACK_ARTIFACTS = {"project_notes.txt"}
_BROWSER_SESSION_PRODUCERS = {"browser.start_session", "browser.get_active_session"}
_BROWSER_CAPTURE_PRODUCER = "browser.capture_view"
_BROWSER_CAPTURE_ANALYSIS_TOOLS = {
    "vision.describe_browser_capture",
    "vision.extract_text_from_browser_capture",
    "vision.find_visual_element_in_browser_capture",
}


def validate_dynamic_plan(
    dynamic_plan: DynamicPlan,
    *,
    task: NaturalLanguageTask,
    registry: ToolRegistry,
    tool_catalog: list[ToolCatalogEntry],
    max_steps: int,
) -> PlanValidationResult:
    if not isinstance(dynamic_plan, DynamicPlan):
        return PlanValidationResult(False, "plan is invalid")
    if len(dynamic_plan.steps) > max_steps:
        return PlanValidationResult(False, "plan exceeds the maximum number of steps")

    tool_catalog_names = {entry.name: entry for entry in tool_catalog}
    seen_ids: set[int] = set()
    seen_tools: dict[int, str] = {}
    agent_steps: list[AgentStep] = []

    for index, step in enumerate(dynamic_plan.steps, 1):
        if step.tool not in tool_catalog_names:
            scoped_reason = _task_specific_scoped_tool_reason(step.tool, task, registry)
            if scoped_reason is not None:
                return PlanValidationResult(False, scoped_reason, semantic_incomplete=True)
            return PlanValidationResult(False, f"unknown tool: {step.tool}")
        if not tool_catalog_names[step.tool].enabled:
            return PlanValidationResult(False, f"disabled tool: {step.tool}")
        try:
            definition = registry.get(step.tool)
        except KeyError:
            return PlanValidationResult(False, f"unknown tool: {step.tool}")
        if any(dependency not in seen_ids for dependency in step.depends_on):
            return PlanValidationResult(False, "plan dependency must reference an earlier step")
        try:
            validated_arguments = _validate_tool_arguments(
                step.tool,
                definition.argument_schema,
                step.arguments,
                seen_ids,
                seen_tools,
                index,
                step.depends_on,
            )
        except ValueError as error:
            reason = _task_specific_argument_reason(str(error), step.tool, task)
            return PlanValidationResult(False, reason, policy_rejected=_is_policy_reason(reason))
        seen_ids.add(index)
        seen_tools[index] = step.tool
        agent_steps.append(
            AgentStep(
                step_id=index,
                tool_name=step.tool,
                arguments=validated_arguments,
                depends_on=list(step.depends_on),
                risk_level=definition.risk_level,
                user_visible_description=step.description or definition.description,
            )
        )

    semantic_reason = _validate_semantic_coverage(dynamic_plan, task)
    normalized_plan = AgentPlan(steps=agent_steps, original_request=dynamic_plan.original_request)
    if semantic_reason is not None:
        return PlanValidationResult(False, semantic_reason, normalized_plan=normalized_plan, semantic_incomplete=True)
    return PlanValidationResult(True, "", normalized_plan)


def _validate_tool_arguments(
    tool_name: str,
    schema: dict[str, Any],
    arguments: dict[str, Any],
    seen_ids: set[int],
    seen_tools: dict[int, str],
    current_step_id: int,
    depends_on: list[int],
) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ValueError("invalid arguments")
    normalized = dict(arguments)
    dependency_steps = set(depends_on)
    for key, value in normalized.items():
        if is_result_reference(value):
            if not allows_result_reference(tool_name, key):
                raise ValueError(_literal_only_reference_reason(tool_name, key, value, current_step_id))
            try:
                from_step, field = validate_reference(
                    value,
                    current_step_id=current_step_id,
                    available_steps=seen_ids,
                    dependency_steps=dependency_steps,
                )
            except ValueError as error:
                raise ValueError(_reference_error_reason(str(error))) from error
            if tool_name.startswith("browser.") and key == "session_id":
                if seen_tools.get(from_step) not in _BROWSER_SESSION_PRODUCERS:
                    raise ValueError("browser session reference must come from a browser session step")
                if field not in {"session_id", "display_value"}:
                    raise ValueError("browser session reference field is invalid")
            if tool_name.startswith("browser.") and key == "page_version":
                if not str(seen_tools.get(from_step) or "").startswith("browser."):
                    raise ValueError("browser page_version reference must come from a browser step")
                if field != "page_version":
                    raise ValueError("browser page_version reference field is invalid")
            if tool_name == "browser.capture_view" and key == "tab_id":
                if not str(seen_tools.get(from_step) or "").startswith("browser."):
                    raise ValueError("browser tab reference must come from a browser step")
                if field not in {"tab_id", "active_tab_id"}:
                    raise ValueError("browser tab reference field is invalid")
            if tool_name in _BROWSER_CAPTURE_ANALYSIS_TOOLS and key == "capture_id":
                if seen_tools.get(from_step) != _BROWSER_CAPTURE_PRODUCER:
                    raise ValueError("browser capture reference must come from browser.capture_view")
                if field != "capture_id":
                    raise ValueError("browser capture reference field is invalid")
        elif isinstance(value, dict):
            raise ValueError("invalid result reference")
    if tool_name == "browser.capture_view":
        session_value = normalized.get("session_id")
        if not is_result_reference(session_value):
            raise ValueError("browser capture_view must use the active browser session reference")
    if tool_name in _BROWSER_CAPTURE_ANALYSIS_TOOLS:
        capture_value = normalized.get("capture_id")
        if not is_result_reference(capture_value):
            raise ValueError(f"{tool_name} must use a browser capture result reference")
    placeholder_arguments = dict(normalized)
    for name, value in list(placeholder_arguments.items()):
        spec = schema.get(name)
        if spec is None:
            continue
        if isinstance(value, dict):
            placeholder_arguments[name] = _placeholder_for_schema(spec)
    validate_arguments(schema, placeholder_arguments)
    _validate_restrictions(tool_name, placeholder_arguments)
    return normalized


def _placeholder_for_schema(spec: dict[str, Any]) -> Any:
    field_type = spec.get("type")
    if field_type in {"text", "query", "path"}:
        return "reference"
    if field_type == "app":
        return "notepad"
    if field_type == "folder":
        return "downloads"
    if field_type == "integer":
        return 1
    if field_type == "bool":
        return False
    if field_type == "string_list":
        return ["reference"]
    if field_type == "number":
        return 1
    return "reference"


def _validate_restrictions(tool_name: str, arguments: dict[str, Any]) -> None:
    if tool_name.startswith("filesystem."):
        for field_name in ("path", "source_path", "destination_path"):
            value = arguments.get(field_name)
            if isinstance(value, str):
                try:
                    resolve_path(value, prefer_directory=False)
                except FilesystemPathError as error:
                    raise ValueError(str(error)) from error
    if tool_name == "terminal.execute":
        try:
            decision = terminal_decision_from_arguments(arguments)
        except TerminalError as error:
            raise ValueError(str(error)) from error
        if not decision.allowed:
            raise ValueError(decision.rejection_reason or decision.reason)
    if tool_name.startswith("browser."):
        try:
            validate_browser_tool_arguments(tool_name, arguments)
        except ValueError as error:
            raise ValueError(str(error)) from error


def _task_specific_scoped_tool_reason(tool_name: str, task: NaturalLanguageTask, registry: ToolRegistry) -> str | None:
    try:
        registry.get(tool_name)
    except KeyError:
        return None
    if task.requested_operation == "browser_visual_describe":
        return "browser visual description not covered"
    if task.requested_operation == "browser_visual_extract_text":
        return "browser visual text extraction not covered"
    if task.requested_operation == "browser_visual_find_element":
        return "browser visual-element search not covered"
    if task.requested_operation == "vision_extract_text":
        return "vision text extraction not covered"
    if task.requested_operation == "vision_find_visual_element":
        return "vision visual-element search not covered"
    if task.requested_operation == "vision_describe_image":
        return "vision image description not covered"
    return None


def _task_specific_argument_reason(reason: str, tool_name: str, task: NaturalLanguageTask) -> str:
    if tool_name == "browser.capture_view" and task.requested_operation.startswith("browser_visual_"):
        if reason == "browser capture_view must use the active browser session reference":
            return "browser visual capture must use the active browser session reference"
    if tool_name == "vision.describe_browser_capture" and task.requested_operation == "browser_visual_describe":
        if reason == "vision.describe_browser_capture must use a browser capture result reference":
            return "browser visual capture analysis must reference the captured viewport"
    if tool_name == "vision.extract_text_from_browser_capture" and task.requested_operation == "browser_visual_extract_text":
        if reason == "vision.extract_text_from_browser_capture must use a browser capture result reference":
            return "browser visual capture analysis must reference the captured viewport"
    if tool_name == "vision.find_visual_element_in_browser_capture" and task.requested_operation == "browser_visual_find_element":
        if reason == "missing argument: query":
            return "requested visual target not covered"
        if reason == "vision.find_visual_element_in_browser_capture must use a browser capture result reference":
            return "browser visual capture analysis must reference the captured viewport"
    if tool_name == "vision.find_visual_element" and task.requested_operation == "vision_find_visual_element":
        if reason == "missing argument: query":
            return "requested visual target not covered"
        if reason == "missing argument: path":
            return "requested image path not covered"
    if tool_name == "vision.extract_text" and task.requested_operation == "vision_extract_text" and reason == "missing argument: path":
        return "requested image path not covered"
    if tool_name == "vision.describe_image" and task.requested_operation == "vision_describe_image" and reason == "missing argument: path":
        return "requested image path not covered"
    return reason


def _literal_only_reference_reason(tool_name: str, argument_name: str, value: dict[str, Any], current_step_id: int) -> str:
    if tool_name in _BROWSER_CAPTURE_ANALYSIS_TOOLS:
        from_step = value.get("from_step")
        if from_step == current_step_id:
            return f"invalid self result reference in {tool_name}.{argument_name}"
        if tool_name == "vision.find_visual_element_in_browser_capture" and argument_name == "query":
            return "vision.find_visual_element_in_browser_capture query must be a literal string"
    if tool_name.startswith("vision."):
        from_step = value.get("from_step")
        if from_step == current_step_id:
            return f"invalid self result reference in {tool_name}.{argument_name}"
        if tool_name == "vision.find_visual_element" and argument_name == "query":
            return "vision.find_visual_element query must be a literal string"
        if tool_name == "vision.find_visual_element" and argument_name == "path":
            return "vision.find_visual_element path must be a literal string"
        if tool_name == "vision.extract_text" and argument_name == "path":
            return "vision.extract_text path must be a literal string"
        if tool_name == "vision.describe_image" and argument_name == "path":
            return "vision.describe_image path must be a literal string"
    return "invalid result reference"


def _validate_semantic_coverage(dynamic_plan: DynamicPlan, task: NaturalLanguageTask) -> str | None:
    paths = _step_paths(dynamic_plan)
    lowered_paths = {_normalize_semantic_local_path(path) for path in paths if _normalize_semantic_local_path(path)}
    writes = [step for step in dynamic_plan.steps if step.tool in _WRITE_TOOLS]
    terminal_steps = [step for step in dynamic_plan.steps if step.tool == "terminal.execute"]
    browser_steps = [step for step in dynamic_plan.steps if step.tool.startswith("browser.")]
    capture_vision_steps = [step for step in dynamic_plan.steps if step.tool in _BROWSER_CAPTURE_ANALYSIS_TOOLS]
    explicit_browser_urls = _explicit_browser_urls(task)

    if any(fallback in lowered_paths for fallback in _FALLBACK_ARTIFACTS):
        if not any(artifact.lower().endswith("project_notes.txt") for artifact in task.requested_artifacts):
            return "the generated plan did not cover all requested requirements"

    if task.read_only_task and writes:
        return "the generated plan did not cover all requested requirements"

    if task.requested_operation == "git_status":
        if not _has_git_read_only_step(terminal_steps):
            return "the generated plan did not cover all requested requirements"
        if writes:
            return "the generated plan did not cover all requested requirements"

    if task.requested_operation.startswith("browser_visual_"):
        browser_reason = _validate_browser_visual_plan(
            dynamic_plan,
            task,
            browser_steps,
            capture_vision_steps,
            terminal_steps,
            explicit_browser_urls=explicit_browser_urls,
        )
        if browser_reason is not None:
            return browser_reason
    elif task.requested_operation.startswith("browser_"):
        browser_reason = _validate_browser_plan(dynamic_plan, task, browser_steps, terminal_steps, explicit_browser_urls=explicit_browser_urls)
        if browser_reason is not None:
            return browser_reason
    if task.requested_operation.startswith("vision_"):
        vision_reason = _validate_vision_plan(dynamic_plan, task, terminal_steps, browser_steps)
        if vision_reason is not None:
            return vision_reason
    else:
        for artifact in _coverage_artifacts(task, explicit_browser_urls):
            normalized_artifact = _normalize_semantic_local_path(artifact)
            if normalized_artifact not in lowered_paths and not any(path.endswith(normalized_artifact) for path in lowered_paths):
                return "requested artifact not covered"

    if task.requires_code_write:
        if not _has_file_write_step(dynamic_plan):
            return "python script plan must include a file write step"
        if _python_code_is_wrapped_as_string_literal(dynamic_plan):
            return "python script plan must write executable code, not a quoted code string"
        if task.requested_output_texts and any(artifact.lower().endswith(".py") for artifact in task.requested_artifacts):
            if not _python_output_is_represented(dynamic_plan, task.requested_output_texts):
                return "python script plan must write code that includes the requested stdout text"

    if task.requested_contents and not _content_is_represented(dynamic_plan, task.requested_contents):
        return "the generated plan did not cover all requested requirements"

    if task.requires_execution:
        if task.requested_operation.startswith("browser_"):
            if not any(step.tool == "browser.open_url" for step in dynamic_plan.steps):
                return "browser plan must open the requested URL"
        elif not _has_execution_step(terminal_steps, task.requested_artifacts):
            return "the generated plan did not cover all requested requirements"

    if task.requires_stdout_match:
        if not task.requested_output_texts:
            return "the generated plan did not cover all requested requirements"
        if not _stdout_expectation_is_represented(dynamic_plan, task.requested_output_texts):
            return "the generated plan did not cover all requested requirements"

    if task.requires_tests:
        if not _has_test_coverage(dynamic_plan, terminal_steps):
            return "the generated plan did not cover all requested requirements"

    return None


def _step_paths(dynamic_plan: DynamicPlan) -> list[str]:
    values: list[str] = []
    for step in dynamic_plan.steps:
        for field_name in ("path", "source_path", "destination_path"):
            value = step.arguments.get(field_name)
            if isinstance(value, str):
                values.append(value)
    return values


def _has_file_write_step(dynamic_plan: DynamicPlan) -> bool:
    return any(step.tool in {"filesystem.write_text_file", "filesystem.append_text_file"} for step in dynamic_plan.steps)


def _content_is_represented(dynamic_plan: DynamicPlan, requested_contents: list[str]) -> bool:
    write_texts: list[str] = []
    for step in dynamic_plan.steps:
        value = step.arguments.get("text")
        if isinstance(value, str):
            write_texts.append(value.lower())
    for requested in requested_contents:
        lowered = requested.lower()
        if any(lowered in text for text in write_texts):
            return True
    return False


def _has_execution_step(terminal_steps: list[Any], requested_artifacts: list[str]) -> bool:
    if not terminal_steps:
        return False
    if not requested_artifacts:
        return True
    lowered_artifacts = [artifact.lower() for artifact in requested_artifacts]
    for step in terminal_steps:
        arguments = step.arguments.get("arguments")
        if not isinstance(arguments, list):
            continue
        normalized = [str(item).lower() for item in arguments]
        if any(artifact in normalized or any(item.endswith(artifact) for item in normalized) for artifact in lowered_artifacts):
            return True
    return False


def _stdout_expectation_is_represented(dynamic_plan: DynamicPlan, requested_outputs: list[str]) -> bool:
    all_text = " ".join(dynamic_plan.success_criteria + [step.expected_result or step.description for step in dynamic_plan.steps]).lower()
    for requested in requested_outputs:
        if requested.lower() in all_text:
            return True
    return False


def _python_output_is_represented(dynamic_plan: DynamicPlan, requested_outputs: list[str]) -> bool:
    write_texts = [
        str(step.arguments.get("text") or "").lower()
        for step in dynamic_plan.steps
        if step.tool in {"filesystem.write_text_file", "filesystem.append_text_file"}
    ]
    for requested in requested_outputs:
        lowered = requested.lower()
        if any(lowered in text for text in write_texts):
            return True
    return False


def _python_code_is_wrapped_as_string_literal(dynamic_plan: DynamicPlan) -> bool:
    for step in dynamic_plan.steps:
        if step.tool not in {"filesystem.write_text_file", "filesystem.append_text_file"}:
            continue
        text = step.arguments.get("text")
        if not isinstance(text, str):
            continue
        stripped = text.strip()
        if len(stripped) < 2:
            continue
        if stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
            inner = stripped[1:-1].strip().lower()
            if inner.startswith(("print(", "def ", "class ", "import ", "from ")):
                return True
    return False


def _has_git_read_only_step(terminal_steps: list[Any]) -> bool:
    for step in terminal_steps:
        executable = str(step.arguments.get("executable", "")).lower()
        arguments = step.arguments.get("arguments", [])
        if executable != "git" or not isinstance(arguments, list) or not arguments:
            continue
        normalized = [str(item).lower() for item in arguments]
        if normalized[:1] == ["status"] and "--porcelain" in normalized[1:]:
            return True
    return False


def _validate_browser_plan(
    dynamic_plan: DynamicPlan,
    task: NaturalLanguageTask,
    browser_steps: list[Any],
    terminal_steps: list[Any],
    *,
    explicit_browser_urls: list[str],
) -> str | None:
    if not browser_steps:
        return "browser plan must use browser tools"
    if terminal_steps:
        return "browser plan must not use terminal workarounds"
    producer_steps = [step for step in browser_steps if step.tool in _BROWSER_SESSION_PRODUCERS]
    close_steps = [step for step in browser_steps if step.tool == "browser.close_session"]
    if len(producer_steps) != 1:
        return "browser plan must acquire or reuse exactly one browser session"
    if browser_steps[0].tool not in _BROWSER_SESSION_PRODUCERS:
        return "browser plan must acquire or reuse a browser session first"
    producer_step = producer_steps[0]
    producer_tool = producer_step.tool
    producer_step_id = dynamic_plan.steps.index(producer_step) + 1
    require_close = producer_tool == "browser.start_session"
    if require_close:
        if len(close_steps) != 1:
            return "browser plan must close the browser session"
        if browser_steps[-1].tool != "browser.close_session":
            return "browser plan must close the browser session"
    elif len(close_steps) > 1:
        return "browser plan must not close the browser session more than once"
    elif close_steps and producer_tool != "browser.start_session":
        return "browser plan must not close a reused browser session"
    close_step_id = dynamic_plan.steps.index(close_steps[0]) + 1 if close_steps else None
    if not _browser_session_references_are_valid(dynamic_plan, producer_step_id, close_step_id):
        return "browser plan must use the started session instead of inventing session IDs"
    if producer_tool == "browser.start_session" and not any(step.tool in {"browser.open_url", "browser.open_new_tab"} for step in browser_steps):
        return "browser plan must open the requested URL"
    if explicit_browser_urls and not _browser_requested_url_is_covered(browser_steps, explicit_browser_urls):
        return "requested browser URL not covered"
    if not any(step.tool == "browser.open_url" for step in browser_steps):
        if task.requested_operation not in {
            "browser_click",
            "browser_scroll",
            "browser_scroll_to_element",
            "browser_switch_tab",
            "browser_list_tabs",
            "browser_close_tab",
            "browser_reload",
            "browser_clickable_inspection",
            "browser_new_tab",
            "browser_form_fill",
            "browser_form_clear",
            "browser_form_submit",
            "browser_form_fill_submit",
            "browser_visual_describe",
            "browser_visual_extract_text",
            "browser_visual_find_element",
        }:
            return "browser plan must open the requested URL"
    if task.requested_operation == "browser_title" and not any(step.tool == "browser.get_page_info" for step in browser_steps):
        return "browser title plan must capture page info"
    if task.requested_operation == "browser_summary" and not any(step.tool == "browser.extract_visible_text" for step in browser_steps):
        return "browser summary plan must extract visible text"
    if task.requested_operation == "browser_screenshot" and not any(step.tool == "browser.take_screenshot" for step in browser_steps):
        return "browser screenshot plan must save a screenshot"
    if task.requested_operation == "browser_scroll" and not any(step.tool == "browser.scroll_page" for step in browser_steps):
        return "browser scroll plan must scroll the page"
    if task.requested_operation == "browser_scroll_to_element" and not any(step.tool == "browser.scroll_to_element" for step in browser_steps):
        return "browser scroll-to-element plan must target a visible element"
    if task.requested_operation == "browser_click" and not any(step.tool == "browser.click_element" for step in browser_steps):
        return "browser click plan must click a safe visible element"
    if task.requested_operation == "browser_reload" and not any(step.tool == "browser.reload_page" for step in browser_steps):
        return "browser reload plan must reload the current page"
    if task.requested_operation == "browser_new_tab" and not any(step.tool == "browser.open_new_tab" for step in browser_steps):
        return "browser new-tab plan must open a new tab"
    if task.requested_operation == "browser_switch_tab" and not any(step.tool == "browser.switch_tab" for step in browser_steps):
        return "browser switch-tab plan must switch tabs"
    if task.requested_operation == "browser_list_tabs" and not any(step.tool == "browser.list_tabs" for step in browser_steps):
        return "browser list-tabs plan must enumerate open tabs"
    if task.requested_operation == "browser_close_tab" and not any(step.tool == "browser.close_tab" for step in browser_steps):
        return "browser close-tab plan must close a tab"
    if task.requested_operation == "browser_clickable_inspection" and not any(step.tool in {"browser.inspect_clickable_elements", "browser.inspect_elements"} for step in browser_steps):
        return "browser clickable-inspection plan must inspect clickable elements"
    if task.requested_operation == "browser_form_inspection" and not any(step.tool == "browser.inspect_form_controls" for step in browser_steps):
        return "browser form-inspection plan must inspect visible form controls"
    if task.requested_operation == "browser_form_fill":
        if not any(step.tool == "browser.input_text" for step in browser_steps):
            return "browser form-fill plan must enter text into a visible form control"
        if any(step.tool == "browser.submit_form" for step in browser_steps):
            return "browser form-fill plan must not submit a form unless the user explicitly asked for it"
        if _browser_plan_mentions_submission(dynamic_plan):
            return "browser form-fill plan must not claim form submission"
    if task.requested_operation == "browser_form_clear":
        if not any(step.tool == "browser.clear_input" for step in browser_steps):
            return "browser clear-input plan must clear a visible form control"
        if any(step.tool == "browser.submit_form" for step in browser_steps):
            return "browser clear-input plan must not submit a form unless the user explicitly asked for it"
    if _task_constraint_truthy(task, "must_not_submit"):
        if any(step.tool == "browser.submit_form" for step in browser_steps):
            return "browser form-fill plan must not submit a form unless the user explicitly asked for it"
        if _browser_plan_mentions_submission(dynamic_plan):
            return "browser form-fill plan must not claim form submission"
    if task.requested_operation == "browser_form_submit":
        if not any(step.tool == "browser.submit_form" for step in browser_steps):
            return "browser form-submit plan must submit a visible safe form"
        if not any(step.tool in {"browser.get_page_info", "browser.extract_visible_text"} for step in browser_steps):
            return "browser form-submit plan must capture grounded post-submit browser evidence"
    if task.requested_operation == "browser_form_fill_submit":
        input_step_ids = [index for index, step in enumerate(dynamic_plan.steps, 1) if step.tool == "browser.input_text"]
        submit_step_ids = [index for index, step in enumerate(dynamic_plan.steps, 1) if step.tool == "browser.submit_form"]
        if not input_step_ids:
            return "browser form-fill-submit plan must enter text before submitting the form"
        if not submit_step_ids:
            return "browser form-fill-submit plan must submit the requested form"
        if min(submit_step_ids) <= min(input_step_ids):
            return "browser form-fill-submit plan must enter text before submitting the form"
        if not any(step.tool in {"browser.get_page_info", "browser.extract_visible_text"} for step in browser_steps):
            return "browser form-fill-submit plan must capture grounded post-submit browser evidence"
    return None


def _validate_browser_visual_plan(
    dynamic_plan: DynamicPlan,
    task: NaturalLanguageTask,
    browser_steps: list[Any],
    capture_vision_steps: list[Any],
    terminal_steps: list[Any],
    *,
    explicit_browser_urls: list[str],
) -> str | None:
    browser_reason = _validate_browser_plan(
        dynamic_plan,
        task,
        browser_steps,
        terminal_steps,
        explicit_browser_urls=explicit_browser_urls,
    )
    if browser_reason is not None:
        return browser_reason
    expected_tool = {
        "browser_visual_describe": "vision.describe_browser_capture",
        "browser_visual_extract_text": "vision.extract_text_from_browser_capture",
        "browser_visual_find_element": "vision.find_visual_element_in_browser_capture",
    }.get(task.requested_operation, "")
    if not expected_tool:
        return "browser visual analysis not covered"
    allowed_tools = {
        "browser.start_session",
        "browser.get_active_session",
        "browser.open_url",
        "browser.capture_view",
        "browser.close_session",
        expected_tool,
    }
    if any(step.tool not in allowed_tools for step in dynamic_plan.steps):
        return "browser visual plan must not add unrelated tools"
    capture_steps = [step for step in browser_steps if step.tool == "browser.capture_view"]
    if len(capture_steps) != 1:
        return "browser visual plan must capture the current viewport exactly once"
    if len(capture_vision_steps) != 1 or capture_vision_steps[0].tool != expected_tool:
        if task.requested_operation == "browser_visual_describe":
            return "browser visual description not covered"
        if task.requested_operation == "browser_visual_extract_text":
            return "browser visual text extraction not covered"
        return "browser visual-element search not covered"
    capture_step = capture_steps[0]
    analysis_step = capture_vision_steps[0]
    capture_step_id = dynamic_plan.steps.index(capture_step) + 1
    analysis_step_id = dynamic_plan.steps.index(analysis_step) + 1
    if capture_step_id >= analysis_step_id:
        return "browser visual capture must occur before browser visual analysis"
    close_step_id = next(
        (index for index, step in enumerate(dynamic_plan.steps, 1) if step.tool == "browser.close_session"),
        None,
    )
    if close_step_id is not None and analysis_step_id >= close_step_id:
        return "browser visual analysis must occur before session cleanup"
    session_value = capture_step.arguments.get("session_id")
    if not is_result_reference(session_value):
        return "browser visual capture must use the active browser session reference"
    capture_value = analysis_step.arguments.get("capture_id")
    if not is_result_reference(capture_value):
        return "browser visual capture analysis must reference the captured viewport"
    if capture_value.get("from_step") != capture_step_id or capture_value.get("field") != "capture_id":
        return "browser visual capture analysis must reference the captured viewport"
    if capture_step_id not in analysis_step.depends_on:
        return "browser visual analysis must depend on the capture step"
    if task.requested_operation == "browser_visual_find_element":
        query_value = analysis_step.arguments.get("query")
        if not isinstance(query_value, str) or not query_value.strip():
            return "requested visual target not covered"
        expected_query = _task_constraint_value(task, "browser_visual_query")
        canonical_query = canonicalize_vision_find_query_text(query_value, expected_query=expected_query)
        if canonical_query is None:
            if looks_like_json_container_text(query_value):
                return "requested visual target not covered"
            canonical_query = query_value.strip()
        if expected_query and normalize_vision_query(expected_query) not in normalize_vision_query(canonical_query):
            return "requested visual target not covered"
    return None


def _validate_vision_plan(
    dynamic_plan: DynamicPlan,
    task: NaturalLanguageTask,
    terminal_steps: list[Any],
    browser_steps: list[Any],
) -> str | None:
    if terminal_steps or browser_steps:
        return "vision plan must use only vision tools"
    vision_steps = [step for step in dynamic_plan.steps if step.tool.startswith("vision.")]
    expected_tool = {
        "vision_describe_image": "vision.describe_image",
        "vision_extract_text": "vision.extract_text",
        "vision_find_visual_element": "vision.find_visual_element",
    }.get(task.requested_operation, "")
    matching_steps = [step for step in vision_steps if step.tool == expected_tool] if expected_tool else []
    if not matching_steps:
        if task.requested_operation == "vision_extract_text":
            return "vision text extraction not covered"
        if task.requested_operation == "vision_find_visual_element":
            return "vision visual-element search not covered"
        if task.requested_operation == "vision_describe_image":
            return "vision image description not covered"
        return "vision plan must contain exactly one vision analysis step"
    if len(vision_steps) != 1 or len(dynamic_plan.steps) != 1:
        return "vision plan must contain exactly one vision analysis step"
    step = matching_steps[0]
    requested_paths = [
        _normalize_semantic_local_path(artifact)
        for artifact in task.requested_artifacts
        if artifact.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    requested_paths = [path for path in requested_paths if path]
    step_path = _normalize_semantic_local_path(str(step.arguments.get("path") or ""))
    if not step_path:
        return "requested image path not covered"
    if requested_paths and step_path not in requested_paths:
        return "requested image path not covered"
    if task.requested_operation == "vision_find_visual_element":
        query_value = str(step.arguments.get("query") or "").strip()
        if not query_value:
            return "requested visual target not covered"
        expected_query = _task_constraint_value(task, "vision_query")
        canonical_query = canonicalize_vision_find_query_text(query_value, expected_query=expected_query)
        if canonical_query is None:
            if looks_like_json_container_text(query_value):
                return "requested visual target not covered"
            canonical_query = query_value
        if expected_query and normalize_vision_query(expected_query) not in normalize_vision_query(canonical_query):
            return "requested visual target not covered"
    return None


def _coverage_artifacts(task: NaturalLanguageTask, explicit_browser_urls: list[str]) -> list[str]:
    explicit_suffixes = {_url_artifact_suffix(url) for url in explicit_browser_urls}
    explicit_url_text = [url.lower() for url in explicit_browser_urls] + [url.lower() for url in _raw_browser_url_tokens(task)]
    filtered: list[str] = []
    for artifact in task.requested_artifacts:
        lowered = artifact.lower()
        if task.requested_operation.startswith("browser_") and (
            lowered in explicit_suffixes or any(lowered in url for url in explicit_url_text)
        ):
            continue
        filtered.append(artifact)
    return filtered


def _normalize_semantic_local_path(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    normalized = raw.replace("\\", "/")
    parts: list[str] = []
    for segment in normalized.split("/"):
        token = segment.strip()
        if not token or token == ".":
            continue
        parts.append(token)
    return "/".join(parts).lower()


def _explicit_browser_urls(task: NaturalLanguageTask) -> list[str]:
    if not task.requested_operation.startswith("browser_"):
        return []
    urls: list[str] = []
    for token in _raw_browser_url_tokens(task):
        normalized = _normalize_browser_url(token)
        if normalized and normalized not in urls:
            urls.append(normalized)
    return urls


def _raw_browser_url_tokens(task: NaturalLanguageTask) -> list[str]:
    values: list[str] = []
    for token in task.raw_input.split():
        candidate = token.strip().rstrip(".,)")
        if not candidate.lower().startswith(("http://", "https://")):
            continue
        if candidate not in values:
            values.append(candidate)
    return values


def _browser_requested_url_is_covered(browser_steps: list[Any], explicit_browser_urls: list[str]) -> bool:
    opened_urls: set[str] = set()
    for step in browser_steps:
        if step.tool not in {"browser.open_url", "browser.open_new_tab"}:
            continue
        url = step.arguments.get("url")
        if not isinstance(url, str):
            continue
        normalized = _normalize_browser_url(url)
        if normalized:
            opened_urls.add(normalized)
    return all(url in opened_urls for url in explicit_browser_urls)


def _url_artifact_suffix(url: str) -> str:
    split = urlsplit(url)
    path = split.path or "/"
    if path == "/":
        return split.netloc.lower()
    return f"{split.netloc.lower()}{path}".rstrip("/").lower()


def _normalize_browser_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    value = url.strip()
    if not value:
        return ""
    split = urlsplit(value)
    if split.scheme.lower() not in {"http", "https"} or not split.netloc:
        return ""
    host = (split.hostname or "").lower()
    if not host:
        return ""
    port = split.port
    default_port = 80 if split.scheme.lower() == "http" else 443
    if port and port != default_port:
        netloc = f"{host}:{port}"
    else:
        netloc = host
    path = split.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((split.scheme.lower(), netloc, path, split.query, ""))


def _task_constraint_truthy(task: NaturalLanguageTask, name: str) -> bool:
    for constraint in task.constraints:
        if constraint.name == name and str(constraint.value).strip().lower() == "true":
            return True
    return False


def _browser_plan_mentions_submission(dynamic_plan: DynamicPlan) -> bool:
    values = list(dynamic_plan.success_criteria)
    for step in dynamic_plan.steps:
        if step.description:
            values.append(step.description)
        if step.expected_result:
            values.append(step.expected_result)
    normalized = " ".join(value.lower() for value in values if isinstance(value, str))
    return any(token in normalized for token in ("form submitted", "submit the form", "submitted the requested form"))


def _browser_session_references_are_valid(dynamic_plan: DynamicPlan, producer_step_id: int, close_step_id: int | None) -> bool:
    session_user_step_ids = [
        index
        for index, step in enumerate(dynamic_plan.steps, 1)
        if step.tool.startswith("browser.") and step.tool not in _BROWSER_SESSION_PRODUCERS | {"browser.close_session"}
    ]
    if not session_user_step_ids:
        return False
    final_session_user_step_id = session_user_step_ids[-1]
    for index, step in enumerate(dynamic_plan.steps, 1):
        if not step.tool.startswith("browser.") or step.tool in _BROWSER_SESSION_PRODUCERS:
            continue
        session_value = step.arguments.get("session_id")
        if not is_result_reference(session_value):
            return False
        if session_value.get("from_step") != producer_step_id:
            return False
        if session_value.get("field") not in {"display_value", "session_id"}:
            return False
        required_dependencies = {producer_step_id}
        if step.tool == "browser.close_session":
            required_dependencies.add(final_session_user_step_id)
        if not required_dependencies.issubset(set(step.depends_on)):
            return False
        if step.tool == "browser.close_session" and close_step_id is not None and index != close_step_id:
            return False
    return True


def _reference_error_reason(reason: str) -> str:
    if reason == "forward references are not allowed":
        return "invalid result reference"
    if reason == "reference dependency is missing":
        return "step result reference must also appear in depends_on"
    if reason in {"reference is invalid", "referenced step result is missing"}:
        return "invalid result reference"
    return "invalid result reference"


def _task_constraint_value(task: NaturalLanguageTask, name: str) -> str:
    for constraint in task.constraints:
        if constraint.name == name:
            return constraint.value
    return ""


def _has_test_coverage(dynamic_plan: DynamicPlan, terminal_steps: list[Any]) -> bool:
    wrote_test_artifact = False
    wrote_implementation_artifact = False
    for step in dynamic_plan.steps:
        path = step.arguments.get("path")
        if isinstance(path, str) and path.lower().endswith(".py"):
            if "test" in path.lower():
                wrote_test_artifact = True
            else:
                wrote_implementation_artifact = True
    ran_tests = False
    for step in terminal_steps:
        arguments = step.arguments.get("arguments", [])
        if not isinstance(arguments, list):
            continue
        normalized = [str(item).lower() for item in arguments]
        if any(token in normalized for token in ("pytest", "unittest")) or normalized[:2] == ["-m", "unittest"]:
            ran_tests = True
            break
    return wrote_implementation_artifact and (wrote_test_artifact or ran_tests)


def _is_policy_reason(reason: str) -> bool:
    return reason in {
        "Shell syntax is not allowed.",
        "That executable is not allowed.",
        "Destructive Git operations are not allowed.",
        "Only read-only Git operations are allowed.",
        "Only git remote -v is allowed.",
        "Unsupported terminal operation.",
        "Unknown executable.",
        "Inline Python execution is not allowed.",
        "Package installation is disabled.",
        "Package specification is not allowed.",
        "Unsupported browser wait condition.",
        "Browser timeout is invalid.",
        "Browser text limit is invalid.",
        "Browser element limit is invalid.",
        "Unsupported element type.",
        "Unsupported screenshot image format.",
        "Unsupported browser form control type.",
        "Browser form targeting requires label_hint, placeholder_hint, name_hint, or a positive ordinal.",
        "Browser form targeting requires semantic hints or a positive ordinal.",
        "Sensitive form fields are not supported.",
        "Sensitive or authenticated form submission is not supported.",
        "Browser input text is invalid.",
        "Browser page version is invalid.",
        "Browser tab ID is invalid.",
        "That path is not allowed.",
        "Working directory is invalid.",
    }

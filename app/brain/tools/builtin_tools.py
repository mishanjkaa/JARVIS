from __future__ import annotations

from app.brain.browser.controller import get_browser_controller
from app.brain.computer.app_launcher import open_browser, open_calculator, open_notepad
from app.brain.computer.folder_actions import open_known_folder
from app.brain.computer.system_info import get_computer_name, get_disk_space, get_system_info
from app.brain.configuration.runtime_config import get_effective_runtime_config
from app.brain.filesystem.controller import get_filesystem_controller
from app.brain.filesystem.errors import FilesystemError
from app.brain.internet.web_actions import search_web
from app.brain.location.controller import get_location_controller
from app.brain.location.errors import LocationError
from app.brain.memory.store import (
    DEFAULT_CATEGORY,
    SOURCE_AI_PROPOSED,
    VALID_CATEGORIES,
    forget_memory,
    recall_memory,
    save_memory,
)
from app.brain.planner.notes import add_note, list_notes
from app.brain.planner.tasks import add_task, list_tasks
from app.brain.skills.calculator import calculate_expression
from app.brain.skills.self_check import run_self_check
from app.brain.skills.system_info import get_local_date, get_local_time
from app.brain.terminal.controller import get_terminal_controller
from app.brain.terminal.git_status import extract_git_untracked_evidence
from app.brain.terminal.models import TerminalExecutionResult, TerminalExecutionStatus
from app.brain.tools.models import ToolDefinition, ToolResult
from app.brain.tools.validators import validate_arguments
from app.brain.vision.controller import get_vision_controller
from app.brain.vision.errors import VisionCaptureUnsupportedError, VisionDisabledError, VisionImageError, VisionPolicyError


def _tool_result(success: bool, status_category: str, message: str, display_value: str = "", reference_fields: dict | None = None) -> ToolResult:
    return ToolResult(success=success, status_category=status_category, message=message, display_value=display_value, reference_fields=reference_fields or {})


def _handler_calculate(args: dict) -> ToolResult:
    schema = {"expression": {"type": "text", "max_length": 120}}
    validated = validate_arguments(schema, args)
    result = calculate_expression(validated["expression"])
    display_value = result[len("Result: "):].strip() if result.startswith("Result: ") else result
    return _tool_result(True, "success", result, display_value, {"display_value": display_value})


def _handler_search(args: dict) -> ToolResult:
    schema = {"query": {"type": "query"}}
    validated = validate_arguments(schema, args)
    result = search_web(validated["query"])
    return _tool_result(True, "success", result, result)


def _handler_remember(args: dict) -> ToolResult:
    schema = {
        "key": {"type": "text", "max_length": 80},
        "value": {"type": "text", "max_length": 200},
        "category": {"type": "text", "max_length": 20, "required": False},
    }
    validated = validate_arguments(schema, args)
    category = validated.get("category") or DEFAULT_CATEGORY
    category = category.strip().lower()
    if category not in VALID_CATEGORIES:
        return _tool_result(False, "failed", "Invalid memory category.")
    config = get_effective_runtime_config()
    result = save_memory(
        validated["key"],
        validated["value"],
        category=category,
        source=SOURCE_AI_PROPOSED,
        max_entries=int(config.get("memory_max_entries", 500)),
        learned_capture_enabled=bool(config.get("memory_learned_capture_enabled", True)),
    )
    success = not result.startswith(("Please provide", "Memory limit reached", "Learned-pattern memory capture is disabled"))
    return _tool_result(success, "success" if success else "failed", result, result)


def _handler_recall(args: dict) -> ToolResult:
    schema = {"key": {"type": "text", "max_length": 80}}
    validated = validate_arguments(schema, args)
    result = recall_memory(validated["key"])
    return _tool_result(True, "success", result, result)


def _handler_forget(args: dict) -> ToolResult:
    schema = {"key": {"type": "text", "max_length": 80}}
    validated = validate_arguments(schema, args)
    result = forget_memory(validated["key"])
    success = not result.startswith("Please provide")
    return _tool_result(success, "success" if success else "failed", result, result)


def _handler_notes_create(args: dict) -> ToolResult:
    schema = {"text": {"type": "text", "max_length": 200}}
    validated = validate_arguments(schema, args)
    note_id = add_note(validated["text"])
    return _tool_result(True, "success", f"Added note {note_id}.", str(note_id), {"display_value": str(note_id)})


def _handler_notes_list(args: dict) -> ToolResult:
    return _tool_result(True, "success", list_notes(), list_notes())


def _handler_tasks_create(args: dict) -> ToolResult:
    schema = {"text": {"type": "text", "max_length": 200}}
    validated = validate_arguments(schema, args)
    task_id = add_task(validated["text"])
    return _tool_result(True, "success", f"Added task {task_id}.", str(task_id), {"display_value": str(task_id)})


def _handler_tasks_list(args: dict) -> ToolResult:
    return _tool_result(True, "success", list_tasks(), list_tasks())


def _handler_open_application(args: dict) -> ToolResult:
    schema = {"application": {"type": "app"}}
    validated = validate_arguments(schema, args)
    if validated["application"] == "notepad":
        result = open_notepad()
    elif validated["application"] == "calculator":
        result = open_calculator()
    else:
        result = open_browser()
    return _tool_result(True, "success", result, result)


def _handler_open_folder(args: dict) -> ToolResult:
    schema = {"folder": {"type": "folder"}}
    validated = validate_arguments(schema, args)
    result = open_known_folder(validated["folder"])
    return _tool_result(True, "success", result, result)


def _handler_get_time(args: dict) -> ToolResult:
    return _tool_result(True, "success", get_local_time(), get_local_time())


def _handler_get_date(args: dict) -> ToolResult:
    return _tool_result(True, "success", get_local_date(), get_local_date())


def _handler_system_info(args: dict) -> ToolResult:
    return _tool_result(True, "success", get_system_info(), get_system_info())


def _handler_self_check(args: dict) -> ToolResult:
    return _tool_result(True, "success", run_self_check(), run_self_check())


def _handler_request_lock(args: dict) -> ToolResult:
    return _tool_result(False, "confirmation_needed", "Please confirm lock.")


def _handler_request_restart(args: dict) -> ToolResult:
    return _tool_result(False, "confirmation_needed", "Please confirm restart.")


def _handler_request_shutdown(args: dict) -> ToolResult:
    return _tool_result(False, "confirmation_needed", "Please confirm shutdown.")


def _handler_terminal_execute(args: dict) -> ToolResult:
    schema = {
        "executable": {"type": "text", "max_length": 120},
        "arguments": {"type": "string_list", "required": False, "max_items": 20, "item_max_length": 200},
        "working_directory": {"type": "path", "required": False},
        "timeout_seconds": {"type": "integer", "required": False},
        "operation_type": {"type": "text", "max_length": 40},
        "raw_command": {"type": "text", "max_length": 200, "required": False},
    }
    validated = validate_arguments(schema, args)
    validated.setdefault("arguments", [])
    result = get_terminal_controller().execute_from_arguments(validated)
    success = result.status == TerminalExecutionStatus.COMPLETED
    compact_message = _terminal_tool_message(result)
    reference_fields = _terminal_reference_fields(validated, result)
    return _tool_result(
        success,
        "success" if success else "failed",
        compact_message,
        "",
        reference_fields,
    )


def _terminal_tool_message(result: TerminalExecutionResult, *, stream_chars: int = 160, total_chars: int = 700) -> str:
    status_line = {
        TerminalExecutionStatus.COMPLETED: "Command completed successfully.",
        TerminalExecutionStatus.FAILED: "Command failed.",
        TerminalExecutionStatus.TIMED_OUT: "Command timed out.",
        TerminalExecutionStatus.CANCELLED: "Command cancelled.",
        TerminalExecutionStatus.REJECTED: "Command rejected.",
        TerminalExecutionStatus.RUNNING: "Command is running.",
        TerminalExecutionStatus.PENDING: "Command pending.",
    }[result.status]
    lines = [status_line]
    if result.exit_code is not None:
        lines.append(f"Exit code: {result.exit_code}")
    lines.append(f"Duration: {result.duration_seconds:.3f}s")
    if result.timed_out:
        lines.append("Timed out: yes")
    if result.cancelled:
        lines.append("Cancelled: yes")
    if result.stdout:
        lines.append("")
        lines.append("stdout:")
        lines.append(_compact_terminal_stream(result.stdout, stream_chars))
    if result.stderr:
        lines.append("")
        lines.append("stderr:")
        lines.append(_compact_terminal_stream(result.stderr, stream_chars))
    message = "\n".join(lines)
    if len(message) <= total_chars:
        return message
    marker = "\n[output truncated]"
    return message[: max(0, total_chars - len(marker))].rstrip() + marker


def _compact_terminal_stream(value: str, max_chars: int) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    marker = "\n[output truncated]"
    return text[: max(0, max_chars - len(marker))].rstrip() + marker


def _terminal_reference_fields(arguments: dict[str, object], result: TerminalExecutionResult) -> dict[str, object]:
    operation_type = str(arguments.get("operation_type") or "").strip().lower()
    raw_arguments = arguments.get("arguments", [])
    if operation_type == "git_read_only" and isinstance(raw_arguments, list) and all(isinstance(item, str) for item in raw_arguments):
        evidence = extract_git_untracked_evidence(raw_arguments, result.stdout, result.stderr, result.exit_code)
        if evidence is not None:
            return evidence.to_reference_fields()
    return {}


def _browser_tool_result(evidence, message: str, *, display_value: str = "") -> ToolResult:
    return _tool_result(
        bool(evidence.success),
        "success" if evidence.success else "failed",
        message,
        display_value,
        evidence.to_reference_fields(),
    )


def _vision_tool_result(evidence, message: str, *, display_value: str = "") -> ToolResult:
    return _tool_result(
        bool(evidence.success),
        "success" if evidence.success else "failed",
        message,
        display_value,
        evidence.to_reference_fields(),
    )


def _handler_browser_start_session(args: dict) -> ToolResult:
    schema = {"headless": {"type": "bool", "required": False}}
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().start_session(headless=validated.get("headless"))
    message = f"Started browser session {evidence.session_id}." if evidence.success else evidence.error_reason or "Browser session could not be started."
    return _browser_tool_result(evidence, message, display_value=evidence.session_id)


def _handler_browser_close_session(args: dict) -> ToolResult:
    schema = {"session_id": {"type": "text", "max_length": 80}}
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().close_session(validated["session_id"])
    message = f"Closed browser session {evidence.session_id}." if evidence.success else evidence.error_reason or "Browser session could not be closed."
    return _browser_tool_result(evidence, message)


def _handler_browser_get_active_session(args: dict) -> ToolResult:
    evidence = get_browser_controller().get_active_session()
    message = f"Using browser session {evidence.session_id}." if evidence.success else evidence.error_reason or "No active browser session."
    return _browser_tool_result(evidence, message, display_value=evidence.session_id)


def _handler_browser_open_url(args: dict) -> ToolResult:
    schema = {
        "session_id": {"type": "text", "max_length": 80},
        "url": {"type": "text", "max_length": 400},
        "wait_until": {"type": "text", "max_length": 40},
        "timeout_seconds": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().open_url(
        session_id=validated["session_id"],
        url=validated["url"],
        wait_until=validated["wait_until"],
        timeout_seconds=validated.get("timeout_seconds", 30),
    )
    message = f"Opened {evidence.final_url or evidence.requested_url}." if evidence.success else evidence.error_reason or "Browser navigation failed."
    return _browser_tool_result(evidence, message)


def _handler_browser_get_page_info(args: dict) -> ToolResult:
    schema = {"session_id": {"type": "text", "max_length": 80}}
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().get_page_info(session_id=validated["session_id"])
    message = f"Page title: {evidence.title}" if evidence.success else evidence.error_reason or "Could not read page information."
    return _browser_tool_result(evidence, message)


def _handler_browser_extract_visible_text(args: dict) -> ToolResult:
    schema = {
        "session_id": {"type": "text", "max_length": 80},
        "max_characters": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().extract_visible_text(
        session_id=validated["session_id"],
        max_characters=validated.get("max_characters", 4000),
    )
    message = evidence.text if evidence.success else evidence.error_reason or "Could not extract visible text."
    return _browser_tool_result(evidence, message or "No visible text found.")


def _handler_browser_inspect_elements(args: dict) -> ToolResult:
    schema = {
        "session_id": {"type": "text", "max_length": 80},
        "element_types": {"type": "string_list", "max_items": 6, "item_max_length": 20},
        "max_elements": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().inspect_elements(
        session_id=validated["session_id"],
        element_types=validated["element_types"],
        max_elements=validated.get("max_elements", 20),
    )
    message = f"Inspected {evidence.element_count} page elements." if evidence.success else evidence.error_reason or "Could not inspect page elements."
    return _browser_tool_result(evidence, message)


def _handler_browser_inspect_clickable_elements(args: dict) -> ToolResult:
    schema = {
        "session_id": {"type": "text", "max_length": 80},
        "max_elements": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().inspect_clickable_elements(
        session_id=validated["session_id"],
        max_elements=validated.get("max_elements", 20),
    )
    message = f"Inspected {evidence.element_count} clickable elements." if evidence.success else evidence.error_reason or "Could not inspect clickable elements."
    return _browser_tool_result(evidence, message)


def _handler_browser_inspect_form_controls(args: dict) -> ToolResult:
    schema = {
        "session_id": {"type": "text", "max_length": 80},
        "max_controls": {"type": "integer", "required": False},
        "tab_id": {"type": "text", "max_length": 80, "required": False},
        "page_version": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().inspect_form_controls(
        session_id=validated["session_id"],
        max_controls=validated.get("max_controls", 20),
        tab_id=validated.get("tab_id", ""),
        page_version=validated.get("page_version", 0),
    )
    message = f"Inspected {evidence.control_count} visible form controls." if evidence.success else evidence.error_reason or "Could not inspect form controls."
    return _browser_tool_result(evidence, message)


def _handler_browser_input_text(args: dict) -> ToolResult:
    schema = {
        "session_id": {"type": "text", "max_length": 80},
        "control_type": {"type": "text", "max_length": 20},
        "text": {"type": "text", "max_length": 2000},
        "label_hint": {"type": "text", "max_length": 200, "required": False},
        "placeholder_hint": {"type": "text", "max_length": 200, "required": False},
        "name_hint": {"type": "text", "max_length": 120, "required": False},
        "ordinal": {"type": "integer", "required": False},
        "tab_id": {"type": "text", "max_length": 80, "required": False},
        "page_version": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().input_text(
        session_id=validated["session_id"],
        control_type=validated["control_type"],
        text=validated["text"],
        label_hint=validated.get("label_hint", ""),
        placeholder_hint=validated.get("placeholder_hint", ""),
        name_hint=validated.get("name_hint", ""),
        ordinal=validated.get("ordinal", 0),
        tab_id=validated.get("tab_id", ""),
        page_version=validated.get("page_version", 0),
    )
    target = evidence.target_description or validated.get("label_hint") or validated.get("placeholder_hint") or validated.get("name_hint") or validated["control_type"]
    message = (
        f"Entered text into {target} (length {evidence.text_length})."
        if evidence.success
        else evidence.error_reason or "Could not enter text into the requested form control."
    )
    return _browser_tool_result(evidence, message)


def _handler_browser_clear_input(args: dict) -> ToolResult:
    schema = {
        "session_id": {"type": "text", "max_length": 80},
        "control_type": {"type": "text", "max_length": 20},
        "label_hint": {"type": "text", "max_length": 200, "required": False},
        "placeholder_hint": {"type": "text", "max_length": 200, "required": False},
        "name_hint": {"type": "text", "max_length": 120, "required": False},
        "ordinal": {"type": "integer", "required": False},
        "tab_id": {"type": "text", "max_length": 80, "required": False},
        "page_version": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().clear_input(
        session_id=validated["session_id"],
        control_type=validated["control_type"],
        label_hint=validated.get("label_hint", ""),
        placeholder_hint=validated.get("placeholder_hint", ""),
        name_hint=validated.get("name_hint", ""),
        ordinal=validated.get("ordinal", 0),
        tab_id=validated.get("tab_id", ""),
        page_version=validated.get("page_version", 0),
    )
    target = evidence.target_description or validated.get("label_hint") or validated.get("placeholder_hint") or validated.get("name_hint") or validated["control_type"]
    message = f"Cleared {target}." if evidence.success else evidence.error_reason or "Could not clear the requested form control."
    return _browser_tool_result(evidence, message)


def _handler_browser_submit_form(args: dict) -> ToolResult:
    schema = {
        "session_id": {"type": "text", "max_length": 80},
        "label_hint": {"type": "text", "max_length": 200, "required": False},
        "placeholder_hint": {"type": "text", "max_length": 200, "required": False},
        "name_hint": {"type": "text", "max_length": 120, "required": False},
        "form_text_hint": {"type": "text", "max_length": 200, "required": False},
        "submit_text_hint": {"type": "text", "max_length": 200, "required": False},
        "ordinal": {"type": "integer", "required": False},
        "wait_until": {"type": "text", "max_length": 40, "required": False},
        "timeout_seconds": {"type": "integer", "required": False},
        "allowed_destination_origin": {"type": "text", "max_length": 200, "required": False},
        "page_context": {"type": "text", "max_length": 200, "required": False},
        "tab_id": {"type": "text", "max_length": 80, "required": False},
        "page_version": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().submit_form(
        session_id=validated["session_id"],
        label_hint=validated.get("label_hint", ""),
        placeholder_hint=validated.get("placeholder_hint", ""),
        name_hint=validated.get("name_hint", ""),
        form_text_hint=validated.get("form_text_hint", ""),
        submit_text_hint=validated.get("submit_text_hint", ""),
        ordinal=validated.get("ordinal", 0),
        wait_until=validated.get("wait_until", "domcontentloaded"),
        timeout_seconds=validated.get("timeout_seconds", 30),
        allowed_destination_origin=validated.get("allowed_destination_origin", ""),
        page_context=validated.get("page_context", ""),
        tab_id=validated.get("tab_id", ""),
        page_version=validated.get("page_version", 0),
    )
    message = "Submitted the requested form." if evidence.success else evidence.error_reason or "Could not submit the requested form."
    return _browser_tool_result(evidence, message)


def _handler_browser_take_screenshot(args: dict) -> ToolResult:
    schema = {
        "session_id": {"type": "text", "max_length": 80},
        "path": {"type": "path"},
        "full_page": {"type": "bool", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().take_screenshot(
        session_id=validated["session_id"],
        path=validated["path"],
        full_page=validated.get("full_page", True),
    )
    message = f"Saved screenshot to {evidence.screenshot_path}." if evidence.success else evidence.error_reason or "Could not save screenshot."
    return _browser_tool_result(evidence, message, display_value=evidence.screenshot_path)


def _handler_browser_capture_view(args: dict) -> ToolResult:
    schema = {
        "session_id": {"type": "text", "max_length": 80},
        "tab_id": {"type": "text", "max_length": 80, "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().capture_view(
        session_id=validated["session_id"],
        tab_id=validated.get("tab_id", ""),
    )
    message = (
        f"Captured browser view {evidence.capture_id}."
        if evidence.success
        else evidence.error_reason or "Could not capture the current browser view."
    )
    return _browser_tool_result(evidence, message, display_value=evidence.capture_id)


def _handler_browser_go_back(args: dict) -> ToolResult:
    schema = {"session_id": {"type": "text", "max_length": 80}}
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().go_back(session_id=validated["session_id"])
    message = f"Returned to {evidence.final_url}." if evidence.success else evidence.error_reason or "Could not go back."
    return _browser_tool_result(evidence, message)


def _handler_browser_go_forward(args: dict) -> ToolResult:
    schema = {"session_id": {"type": "text", "max_length": 80}}
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().go_forward(session_id=validated["session_id"])
    message = f"Moved forward to {evidence.final_url}." if evidence.success else evidence.error_reason or "Could not go forward."
    return _browser_tool_result(evidence, message)


def _handler_browser_wait_for_page(args: dict) -> ToolResult:
    schema = {
        "session_id": {"type": "text", "max_length": 80},
        "wait_until": {"type": "text", "max_length": 40},
        "timeout_seconds": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().wait_for_page(
        session_id=validated["session_id"],
        wait_until=validated["wait_until"],
        timeout_seconds=validated.get("timeout_seconds", 30),
    )
    message = f"Page load state reached: {evidence.load_state}." if evidence.success else evidence.error_reason or "Page wait failed."
    return _browser_tool_result(evidence, message)


def _handler_browser_scroll_page(args: dict) -> ToolResult:
    schema = {
        "session_id": {"type": "text", "max_length": 80},
        "direction": {"type": "text", "max_length": 20, "required": False},
        "amount": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().scroll_page(
        session_id=validated["session_id"],
        direction=validated.get("direction", "down"),
        amount=validated.get("amount", 600),
    )
    message = f"Scrolled {validated.get('direction', 'down')}." if evidence.success else evidence.error_reason or "Could not scroll the page."
    return _browser_tool_result(evidence, message)


def _handler_browser_scroll_to_element(args: dict) -> ToolResult:
    schema = {
        "session_id": {"type": "text", "max_length": 80},
        "target_type": {"type": "text", "max_length": 20},
        "text_hint": {"type": "text", "max_length": 200, "required": False},
        "href_hint": {"type": "text", "max_length": 300, "required": False},
        "ordinal": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().scroll_to_element(
        session_id=validated["session_id"],
        target_type=validated["target_type"],
        text_hint=validated.get("text_hint", ""),
        href_hint=validated.get("href_hint", ""),
        ordinal=validated.get("ordinal", 0),
    )
    target = evidence.target_description or validated.get("text_hint") or validated.get("target_type", "element")
    message = f"Scrolled to {target}." if evidence.success else evidence.error_reason or "Could not scroll to the requested element."
    return _browser_tool_result(evidence, message)


def _handler_browser_click_element(args: dict) -> ToolResult:
    schema = {
        "session_id": {"type": "text", "max_length": 80},
        "target_type": {"type": "text", "max_length": 20},
        "text_hint": {"type": "text", "max_length": 200, "required": False},
        "href_hint": {"type": "text", "max_length": 300, "required": False},
        "ordinal": {"type": "integer", "required": False},
        "wait_until": {"type": "text", "max_length": 40, "required": False},
        "timeout_seconds": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().click_element(
        session_id=validated["session_id"],
        target_type=validated["target_type"],
        text_hint=validated.get("text_hint", ""),
        href_hint=validated.get("href_hint", ""),
        ordinal=validated.get("ordinal", 0),
        wait_until=validated.get("wait_until", "domcontentloaded"),
        timeout_seconds=validated.get("timeout_seconds", 30),
    )
    target = evidence.target_description or validated.get("text_hint") or validated.get("target_type", "element")
    if evidence.success and evidence.navigated:
        message = f"Clicked {evidence.target_type or validated['target_type']} and navigated to {evidence.final_url or evidence.url}."
    elif evidence.success:
        message = f"Clicked {target}."
    else:
        message = evidence.error_reason or "Could not click the requested element."
    return _browser_tool_result(evidence, message)


def _handler_browser_reload_page(args: dict) -> ToolResult:
    schema = {
        "session_id": {"type": "text", "max_length": 80},
        "wait_until": {"type": "text", "max_length": 40, "required": False},
        "timeout_seconds": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().reload_page(
        session_id=validated["session_id"],
        wait_until=validated.get("wait_until", "domcontentloaded"),
        timeout_seconds=validated.get("timeout_seconds", 30),
    )
    message = f"Reloaded {evidence.final_url or evidence.url}." if evidence.success else evidence.error_reason or "Could not reload the page."
    return _browser_tool_result(evidence, message)


def _handler_browser_open_new_tab(args: dict) -> ToolResult:
    schema = {
        "session_id": {"type": "text", "max_length": 80},
        "url": {"type": "text", "max_length": 400},
        "wait_until": {"type": "text", "max_length": 40},
        "timeout_seconds": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().open_new_tab(
        session_id=validated["session_id"],
        url=validated["url"],
        wait_until=validated["wait_until"],
        timeout_seconds=validated.get("timeout_seconds", 30),
    )
    message = f"Opened a new tab: {evidence.final_url or evidence.requested_url}." if evidence.success else evidence.error_reason or "Could not open a new browser tab."
    return _browser_tool_result(evidence, message, display_value=evidence.tab_id)


def _handler_browser_switch_tab(args: dict) -> ToolResult:
    schema = {
        "session_id": {"type": "text", "max_length": 80},
        "target": {"type": "text", "max_length": 20, "required": False},
        "tab_id": {"type": "text", "max_length": 80, "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().switch_tab(
        session_id=validated["session_id"],
        target=validated.get("target", "current"),
        tab_id=validated.get("tab_id", ""),
    )
    message = f"Switched to tab {evidence.tab_id}." if evidence.success else evidence.error_reason or "Could not switch browser tabs."
    return _browser_tool_result(evidence, message, display_value=evidence.tab_id)


def _handler_browser_list_tabs(args: dict) -> ToolResult:
    schema = {"session_id": {"type": "text", "max_length": 80}}
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().list_tabs(session_id=validated["session_id"])
    message = f"Found {evidence.tab_count} open browser tab{'s' if evidence.tab_count != 1 else ''}." if evidence.success else evidence.error_reason or "Could not list browser tabs."
    return _browser_tool_result(evidence, message)


def _handler_browser_close_tab(args: dict) -> ToolResult:
    schema = {
        "session_id": {"type": "text", "max_length": 80},
        "target": {"type": "text", "max_length": 20, "required": False},
        "tab_id": {"type": "text", "max_length": 80, "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_browser_controller().close_tab(
        session_id=validated["session_id"],
        target=validated.get("target", "current"),
        tab_id=validated.get("tab_id", ""),
    )
    message = f"Closed browser tab {evidence.tab_id}." if evidence.success else evidence.error_reason or "Could not close the browser tab."
    return _browser_tool_result(evidence, message)


def _handler_vision_describe_image(args: dict) -> ToolResult:
    schema = {
        "path": {"type": "path"},
        "detail_level": {"type": "text", "max_length": 20, "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_vision_controller().describe_image(
        path=validated["path"],
        detail_level=validated.get("detail_level", "normal"),
    )
    message = evidence.description if evidence.success else evidence.error_reason or "Image description failed."
    return _vision_tool_result(evidence, message or "No image description was returned.")


def _handler_vision_extract_text(args: dict) -> ToolResult:
    schema = {
        "path": {"type": "path"},
        "language_hint": {"type": "text", "max_length": 20, "required": False},
        "max_characters": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_vision_controller().extract_text(
        path=validated["path"],
        language_hint=validated.get("language_hint", ""),
        max_characters=validated.get("max_characters"),
    )
    message = evidence.extracted_text if evidence.success else evidence.error_reason or "Image text extraction failed."
    return _vision_tool_result(evidence, message or "No visible text was extracted from the image.")


def _handler_vision_find_visual_element(args: dict) -> ToolResult:
    schema = {
        "path": {"type": "path"},
        "query": {"type": "text", "max_length": 200},
        "max_results": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_vision_controller().find_visual_element(
        path=validated["path"],
        query=validated["query"],
        max_results=validated.get("max_results", 3),
    )
    if evidence.success:
        match_outcome = str(evidence.match_outcome or "").strip()
        if match_outcome == "found":
            message = f"Found {len(evidence.visual_regions)} visual match{'es' if len(evidence.visual_regions) != 1 else ''} for {validated['query']}."
        elif match_outcome == "uncertain":
            message = f"A possible visual match could not be verified for {validated['query']}."
        else:
            message = f"No verified visual match was detected for {validated['query']}."
    else:
        message = evidence.error_reason or "Visual element search failed."
    result = _vision_tool_result(evidence, message)
    result.reference_fields["requested_query"] = validated["query"]
    return result


def _handler_vision_describe_browser_capture(args: dict) -> ToolResult:
    schema = {
        "capture_id": {"type": "text", "max_length": 80},
        "detail_level": {"type": "text", "max_length": 20, "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_vision_controller().describe_browser_capture(
        capture_id=validated["capture_id"],
        detail_level=validated.get("detail_level", "normal"),
    )
    message = evidence.description if evidence.success else evidence.error_reason or "Browser visual description failed."
    return _vision_tool_result(evidence, message or "No browser visual description was returned.")


def _handler_vision_extract_text_from_browser_capture(args: dict) -> ToolResult:
    schema = {
        "capture_id": {"type": "text", "max_length": 80},
        "language_hint": {"type": "text", "max_length": 20, "required": False},
        "max_characters": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_vision_controller().extract_text_from_browser_capture(
        capture_id=validated["capture_id"],
        language_hint=validated.get("language_hint", ""),
        max_characters=validated.get("max_characters"),
    )
    message = evidence.extracted_text if evidence.success else evidence.error_reason or "Browser visual OCR failed."
    return _vision_tool_result(evidence, message or "No visible text was extracted from the browser capture.")


def _handler_vision_find_visual_element_in_browser_capture(args: dict) -> ToolResult:
    schema = {
        "capture_id": {"type": "text", "max_length": 80},
        "query": {"type": "text", "max_length": 200},
        "max_results": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_vision_controller().find_visual_element_in_browser_capture(
        capture_id=validated["capture_id"],
        query=validated["query"],
        max_results=validated.get("max_results", 3),
    )
    if evidence.success:
        match_outcome = str(evidence.match_outcome or "").strip()
        if match_outcome == "found":
            message = f"Found {len(evidence.visual_regions)} visual match{'es' if len(evidence.visual_regions) != 1 else ''} in the browser capture."
        elif match_outcome == "uncertain":
            message = f"A possible visual match could not be verified for {validated['query']}."
        else:
            message = f"No verified visual match was detected for {validated['query']}."
    else:
        message = evidence.error_reason or "Browser visual element search failed."
    result = _vision_tool_result(evidence, message)
    result.reference_fields["requested_query"] = validated["query"]
    return result


def _desktop_capture_error_result(error: Exception) -> ToolResult:
    return _tool_result(False, "failed", str(error))


def _handler_desktop_list_windows(args: dict) -> ToolResult:
    result = get_vision_controller().list_windows()
    if not result.success:
        return _tool_result(False, "failed", result.error_reason or "Could not list open windows.")
    if not result.windows:
        return _tool_result(True, "success", "No open windows were found.")
    message = "\n".join(f"{window['window_id']}: {window['title']}" for window in result.windows)
    return _tool_result(True, "success", message, message)


def _handler_desktop_capture_screen(args: dict) -> ToolResult:
    try:
        capture = get_vision_controller().capture_desktop_screen()
    except (VisionDisabledError, VisionPolicyError, VisionCaptureUnsupportedError, VisionImageError) as error:
        return _desktop_capture_error_result(error)
    message = f"Captured a screenshot of your entire desktop as {capture.capture_id}."
    return _tool_result(True, "success", message, capture.capture_id)


def _handler_desktop_capture_window(args: dict) -> ToolResult:
    schema = {
        "window_id": {"type": "integer"},
        "window_title": {"type": "text", "max_length": 200},
    }
    validated = validate_arguments(schema, args)
    try:
        capture = get_vision_controller().capture_desktop_window(
            window_id=validated["window_id"],
            window_title=validated["window_title"],
        )
    except (VisionDisabledError, VisionPolicyError, VisionCaptureUnsupportedError, VisionImageError) as error:
        return _desktop_capture_error_result(error)
    message = f"Captured a screenshot of the window titled '{capture.window_title}' as {capture.capture_id}."
    return _tool_result(True, "success", message, capture.capture_id)


def _handler_vision_describe_desktop_capture(args: dict) -> ToolResult:
    schema = {
        "capture_id": {"type": "text", "max_length": 80},
        "detail_level": {"type": "text", "max_length": 20, "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_vision_controller().describe_desktop_capture(
        capture_id=validated["capture_id"],
        detail_level=validated.get("detail_level", "normal"),
    )
    message = evidence.description if evidence.success else evidence.error_reason or "Desktop visual description failed."
    return _vision_tool_result(evidence, message or "No desktop visual description was returned.")


def _handler_vision_extract_text_from_desktop_capture(args: dict) -> ToolResult:
    schema = {
        "capture_id": {"type": "text", "max_length": 80},
        "language_hint": {"type": "text", "max_length": 20, "required": False},
        "max_characters": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_vision_controller().extract_text_from_desktop_capture(
        capture_id=validated["capture_id"],
        language_hint=validated.get("language_hint", ""),
        max_characters=validated.get("max_characters"),
    )
    message = evidence.extracted_text if evidence.success else evidence.error_reason or "Desktop visual OCR failed."
    return _vision_tool_result(evidence, message or "No visible text was extracted from the desktop capture.")


def _handler_vision_find_visual_element_in_desktop_capture(args: dict) -> ToolResult:
    schema = {
        "capture_id": {"type": "text", "max_length": 80},
        "query": {"type": "text", "max_length": 200},
        "max_results": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    evidence = get_vision_controller().find_visual_element_in_desktop_capture(
        capture_id=validated["capture_id"],
        query=validated["query"],
        max_results=validated.get("max_results", 3),
    )
    if evidence.success:
        match_outcome = str(evidence.match_outcome or "").strip()
        if match_outcome == "found":
            message = f"Found {len(evidence.visual_regions)} visual match{'es' if len(evidence.visual_regions) != 1 else ''} in the desktop capture."
        elif match_outcome == "uncertain":
            message = f"A possible visual match could not be verified for {validated['query']}."
        else:
            message = f"No verified visual match was detected for {validated['query']}."
    else:
        message = evidence.error_reason or "Desktop visual element search failed."
    result = _vision_tool_result(evidence, message)
    result.reference_fields["requested_query"] = validated["query"]
    return result


def _filesystem_result(method, *args, **kwargs) -> ToolResult:
    controller = get_filesystem_controller()
    try:
        result = method(*args, **kwargs)
    except FilesystemError as error:
        return _tool_result(False, "failed", str(error))
    reference_fields = {key: value for key, value in result.reference_fields.items() if key != "display_value"}
    return _tool_result(True, "success", result.message, result.display_value, reference_fields)


def _handler_filesystem_list_directory(args: dict) -> ToolResult:
    schema = {
        "path": {"type": "path", "required": False},
        "recursive": {"type": "bool", "required": False},
        "max_depth": {"type": "integer", "required": False},
    }
    validated = validate_arguments(schema, args)
    return _filesystem_result(
        get_filesystem_controller().list_directory,
        validated.get("path"),
        recursive=validated.get("recursive", False),
        max_depth=validated.get("max_depth", 1),
    )


def _handler_filesystem_create_directory(args: dict) -> ToolResult:
    schema = {"path": {"type": "path"}}
    validated = validate_arguments(schema, args)
    return _filesystem_result(get_filesystem_controller().create_directory, validated["path"])


def _handler_filesystem_read_text_file(args: dict) -> ToolResult:
    schema = {"path": {"type": "path", "required": False}}
    validated = validate_arguments(schema, args)
    return _filesystem_result(get_filesystem_controller().read_text_file, validated.get("path"))


def _handler_filesystem_create_text_file(args: dict) -> ToolResult:
    schema = {"path": {"type": "path"}}
    validated = validate_arguments(schema, args)
    return _filesystem_result(get_filesystem_controller().create_text_file, validated["path"])


def _handler_filesystem_write_text_file(args: dict) -> ToolResult:
    schema = {"text": {"type": "text", "max_length": 2000}, "path": {"type": "path", "required": False}}
    validated = validate_arguments(schema, args)
    return _filesystem_result(get_filesystem_controller().write_text_file, validated["text"], validated.get("path"))


def _handler_filesystem_append_text_file(args: dict) -> ToolResult:
    schema = {"text": {"type": "text", "max_length": 2000}, "path": {"type": "path", "required": False}}
    validated = validate_arguments(schema, args)
    return _filesystem_result(get_filesystem_controller().append_text_file, validated["text"], validated.get("path"))


def _handler_filesystem_rename_path(args: dict) -> ToolResult:
    schema = {"source_path": {"type": "path", "required": False}, "destination_path": {"type": "path"}}
    validated = validate_arguments(schema, args)
    return _filesystem_result(get_filesystem_controller().rename_path, validated.get("source_path"), validated["destination_path"])


def _handler_filesystem_copy_path(args: dict) -> ToolResult:
    schema = {"source_path": {"type": "path", "required": False}, "destination_path": {"type": "path", "required": False}}
    validated = validate_arguments(schema, args)
    return _filesystem_result(get_filesystem_controller().copy_path, validated.get("source_path"), validated.get("destination_path"))


def _handler_filesystem_move_path(args: dict) -> ToolResult:
    schema = {"source_path": {"type": "path", "required": False}, "destination_path": {"type": "path"}}
    validated = validate_arguments(schema, args)
    return _filesystem_result(get_filesystem_controller().move_path, validated.get("source_path"), validated["destination_path"])


def _handler_filesystem_delete_path(args: dict) -> ToolResult:
    schema = {"path": {"type": "path", "required": False}}
    validated = validate_arguments(schema, args)
    return _filesystem_result(get_filesystem_controller().delete_path, validated.get("path"))


def _handler_filesystem_exists(args: dict) -> ToolResult:
    schema = {"path": {"type": "path", "required": False}}
    validated = validate_arguments(schema, args)
    return _filesystem_result(get_filesystem_controller().exists, validated.get("path"))


def _handler_filesystem_metadata(args: dict) -> ToolResult:
    schema = {"path": {"type": "path", "required": False}}
    validated = validate_arguments(schema, args)
    return _filesystem_result(get_filesystem_controller().metadata, validated.get("path"))


def _location_error_result(error: Exception) -> ToolResult:
    return _tool_result(False, "failed", str(error))


def _handler_location_where_am_i(args: dict) -> ToolResult:
    try:
        message = get_location_controller().where_am_i()
    except LocationError as error:
        return _location_error_result(error)
    return _tool_result(True, "success", message, message)


def _handler_location_distance_to(args: dict) -> ToolResult:
    schema = {"place": {"type": "text", "max_length": 80}}
    validated = validate_arguments(schema, args)
    try:
        message = get_location_controller().distance_to(validated["place"])
    except LocationError as error:
        return _location_error_result(error)
    return _tool_result(True, "success", message, message)


def _handler_location_save_place(args: dict) -> ToolResult:
    schema = {
        "name": {"type": "text", "max_length": 80},
        "latitude": {"type": "number"},
        "longitude": {"type": "number"},
    }
    validated = validate_arguments(schema, args)
    try:
        message = get_location_controller().save_place(validated["name"], float(validated["latitude"]), float(validated["longitude"]))
    except LocationError as error:
        return _location_error_result(error)
    return _tool_result(True, "success", message, message)


def _handler_navigation_start(args: dict) -> ToolResult:
    schema = {"destination": {"type": "text", "max_length": 200}}
    validated = validate_arguments(schema, args)
    try:
        message = get_location_controller().start_navigation(validated["destination"])
    except LocationError as error:
        return _location_error_result(error)
    return _tool_result(True, "success", message, message)


def _handler_navigation_get_next_instruction(args: dict) -> ToolResult:
    schema = {"session_id": {"type": "text", "max_length": 80}}
    validated = validate_arguments(schema, args)
    try:
        message = get_location_controller().get_next_instruction(validated["session_id"])
    except LocationError as error:
        return _location_error_result(error)
    return _tool_result(True, "success", message, message)


def _handler_navigation_stop(args: dict) -> ToolResult:
    schema = {"session_id": {"type": "text", "max_length": 80}}
    validated = validate_arguments(schema, args)
    try:
        message = get_location_controller().stop_navigation(validated["session_id"])
    except LocationError as error:
        return _location_error_result(error)
    return _tool_result(True, "success", message, message)


def build_builtin_tools() -> list[ToolDefinition]:
    return [
        ToolDefinition("calculator.calculate", "Calculate an expression.", {"expression": {"type": "text", "max_length": 120}}, "read_only", False, _handler_calculate, lambda result: result.display_value or result.message),
        ToolDefinition("internet.search", "Search the web.", {"query": {"type": "query"}}, "external_navigation", False, _handler_search, lambda result: result.display_value or result.message),
        ToolDefinition("memory.remember", "Remember a value.", {"key": {"type": "text", "max_length": 80}, "value": {"type": "text", "max_length": 200}, "category": {"type": "text", "max_length": 20, "required": False}}, "persistent_write", False, _handler_remember, lambda result: result.display_value or result.message),
        ToolDefinition("memory.recall", "Recall a remembered value.", {"key": {"type": "text", "max_length": 80}}, "read_only", False, _handler_recall, lambda result: result.display_value or result.message),
        ToolDefinition("memory.forget", "Forget a remembered value.", {"key": {"type": "text", "max_length": 80}}, "persistent_write", False, _handler_forget, lambda result: result.display_value or result.message),
        ToolDefinition("notes.create", "Create a note.", {"text": {"type": "text", "max_length": 200}}, "persistent_write", False, _handler_notes_create, lambda result: result.display_value or result.message),
        ToolDefinition("notes.list", "List notes.", {}, "read_only", False, _handler_notes_list, lambda result: result.display_value or result.message),
        ToolDefinition("tasks.create", "Create a task.", {"text": {"type": "text", "max_length": 200}}, "persistent_write", False, _handler_tasks_create, lambda result: result.display_value or result.message),
        ToolDefinition("tasks.list", "List tasks.", {}, "read_only", False, _handler_tasks_list, lambda result: result.display_value or result.message),
        ToolDefinition("computer.open_application", "Open an allowlisted application.", {"application": {"type": "app"}}, "local_safe", False, _handler_open_application, lambda result: result.display_value or result.message),
        ToolDefinition("computer.open_known_folder", "Open an allowlisted folder.", {"folder": {"type": "folder"}}, "local_safe", False, _handler_open_folder, lambda result: result.display_value or result.message),
        ToolDefinition("filesystem.list_directory", "List a trusted directory.", {"path": {"type": "path", "required": False}, "recursive": {"type": "bool", "required": False}, "max_depth": {"type": "integer", "required": False}}, "read_only", False, _handler_filesystem_list_directory, lambda result: result.display_value or result.message),
        ToolDefinition("filesystem.create_directory", "Create a trusted directory.", {"path": {"type": "path"}}, "persistent_write", False, _handler_filesystem_create_directory, lambda result: result.display_value or result.message),
        ToolDefinition("filesystem.read_text_file", "Read a UTF-8 text file.", {"path": {"type": "path", "required": False}}, "read_only", False, _handler_filesystem_read_text_file, lambda result: result.display_value or result.message),
        ToolDefinition("filesystem.create_text_file", "Create a UTF-8 text file.", {"path": {"type": "path"}}, "persistent_write", False, _handler_filesystem_create_text_file, lambda result: result.display_value or result.message),
        ToolDefinition("filesystem.write_text_file", "Write text to a UTF-8 file.", {"text": {"type": "text", "max_length": 2000}, "path": {"type": "path", "required": False}}, "persistent_write", False, _handler_filesystem_write_text_file, lambda result: result.display_value or result.message),
        ToolDefinition("filesystem.append_text_file", "Append text to a UTF-8 file.", {"text": {"type": "text", "max_length": 2000}, "path": {"type": "path", "required": False}}, "persistent_write", False, _handler_filesystem_append_text_file, lambda result: result.display_value or result.message),
        ToolDefinition("filesystem.rename_path", "Rename a trusted file or directory.", {"source_path": {"type": "path", "required": False}, "destination_path": {"type": "path"}}, "persistent_write", False, _handler_filesystem_rename_path, lambda result: result.display_value or result.message),
        ToolDefinition("filesystem.copy_path", "Copy a trusted file or directory.", {"source_path": {"type": "path", "required": False}, "destination_path": {"type": "path", "required": False}}, "persistent_write", False, _handler_filesystem_copy_path, lambda result: result.display_value or result.message),
        ToolDefinition("filesystem.move_path", "Move a trusted file or directory.", {"source_path": {"type": "path", "required": False}, "destination_path": {"type": "path"}}, "persistent_write", False, _handler_filesystem_move_path, lambda result: result.display_value or result.message),
        ToolDefinition("filesystem.delete_path", "Soft delete a trusted file or directory.", {"path": {"type": "path", "required": False}}, "persistent_write", False, _handler_filesystem_delete_path, lambda result: result.display_value or result.message),
        ToolDefinition("filesystem.exists", "Check whether a trusted path exists.", {"path": {"type": "path", "required": False}}, "read_only", False, _handler_filesystem_exists, lambda result: result.display_value or result.message),
        ToolDefinition("filesystem.metadata", "Read trusted path metadata.", {"path": {"type": "path", "required": False}}, "read_only", False, _handler_filesystem_metadata, lambda result: result.display_value or result.message),
        ToolDefinition("system.get_time", "Get the current time.", {}, "read_only", False, _handler_get_time, lambda result: result.display_value or result.message),
        ToolDefinition("system.get_date", "Get the current date.", {}, "read_only", False, _handler_get_date, lambda result: result.display_value or result.message),
        ToolDefinition("system.system_info", "Get system information.", {}, "read_only", False, _handler_system_info, lambda result: result.display_value or result.message),
        ToolDefinition("terminal.execute", "Execute a validated terminal command.", {"executable": {"type": "text", "max_length": 120}, "arguments": {"type": "string_list", "required": False, "max_items": 20, "item_max_length": 200}, "working_directory": {"type": "path", "required": False}, "timeout_seconds": {"type": "integer", "required": False}, "operation_type": {"type": "text", "max_length": 40}, "raw_command": {"type": "text", "max_length": 200, "required": False}}, "local_safe", False, _handler_terminal_execute, lambda result: result.display_value or result.message),
        ToolDefinition("browser.start_session", "Start an isolated browser session.", {"headless": {"type": "bool", "required": False}}, "local_safe", False, _handler_browser_start_session, lambda result: result.display_value or result.message),
        ToolDefinition("browser.get_active_session", "Use the active isolated browser session.", {}, "read_only", False, _handler_browser_get_active_session, lambda result: result.display_value or result.message),
        ToolDefinition("browser.close_session", "Close an isolated browser session.", {"session_id": {"type": "text", "max_length": 80}}, "local_safe", False, _handler_browser_close_session, lambda result: result.display_value or result.message),
        ToolDefinition("browser.open_url", "Open a policy-approved URL in a browser session.", {"session_id": {"type": "text", "max_length": 80}, "url": {"type": "text", "max_length": 400}, "wait_until": {"type": "text", "max_length": 40}, "timeout_seconds": {"type": "integer", "required": False}}, "local_safe", False, _handler_browser_open_url, lambda result: result.display_value or result.message),
        ToolDefinition("browser.get_page_info", "Read the current page URL and title.", {"session_id": {"type": "text", "max_length": 80}}, "read_only", False, _handler_browser_get_page_info, lambda result: result.display_value or result.message),
        ToolDefinition("browser.extract_visible_text", "Extract visible page text.", {"session_id": {"type": "text", "max_length": 80}, "max_characters": {"type": "integer", "required": False}}, "read_only", False, _handler_browser_extract_visible_text, lambda result: result.display_value or result.message),
        ToolDefinition("browser.inspect_elements", "Inspect visible page elements without interacting.", {"session_id": {"type": "text", "max_length": 80}, "element_types": {"type": "string_list", "max_items": 6, "item_max_length": 20}, "max_elements": {"type": "integer", "required": False}}, "read_only", False, _handler_browser_inspect_elements, lambda result: result.display_value or result.message),
        ToolDefinition("browser.inspect_clickable_elements", "Inspect visible clickable elements without interacting.", {"session_id": {"type": "text", "max_length": 80}, "max_elements": {"type": "integer", "required": False}}, "read_only", False, _handler_browser_inspect_clickable_elements, lambda result: result.display_value or result.message),
        ToolDefinition("browser.inspect_form_controls", "Inspect visible text inputs and textareas without interacting.", {"session_id": {"type": "text", "max_length": 80}, "max_controls": {"type": "integer", "required": False}, "tab_id": {"type": "text", "max_length": 80, "required": False}, "page_version": {"type": "integer", "required": False}}, "read_only", False, _handler_browser_inspect_form_controls, lambda result: result.display_value or result.message),
        ToolDefinition("browser.input_text", "Enter text into a safe visible text input or textarea.", {"session_id": {"type": "text", "max_length": 80}, "control_type": {"type": "text", "max_length": 20}, "text": {"type": "text", "max_length": 2000}, "label_hint": {"type": "text", "max_length": 200, "required": False}, "placeholder_hint": {"type": "text", "max_length": 200, "required": False}, "name_hint": {"type": "text", "max_length": 120, "required": False}, "ordinal": {"type": "integer", "required": False}, "tab_id": {"type": "text", "max_length": 80, "required": False}, "page_version": {"type": "integer", "required": False}}, "local_safe", False, _handler_browser_input_text, lambda result: result.display_value or result.message),
        ToolDefinition("browser.clear_input", "Clear a safe visible text input or textarea.", {"session_id": {"type": "text", "max_length": 80}, "control_type": {"type": "text", "max_length": 20}, "label_hint": {"type": "text", "max_length": 200, "required": False}, "placeholder_hint": {"type": "text", "max_length": 200, "required": False}, "name_hint": {"type": "text", "max_length": 120, "required": False}, "ordinal": {"type": "integer", "required": False}, "tab_id": {"type": "text", "max_length": 80, "required": False}, "page_version": {"type": "integer", "required": False}}, "local_safe", False, _handler_browser_clear_input, lambda result: result.display_value or result.message),
        ToolDefinition("browser.submit_form", "Submit a safe visible form after explicit approval or allowed automatic execution.", {"session_id": {"type": "text", "max_length": 80}, "label_hint": {"type": "text", "max_length": 200, "required": False}, "placeholder_hint": {"type": "text", "max_length": 200, "required": False}, "name_hint": {"type": "text", "max_length": 120, "required": False}, "form_text_hint": {"type": "text", "max_length": 200, "required": False}, "submit_text_hint": {"type": "text", "max_length": 200, "required": False}, "ordinal": {"type": "integer", "required": False}, "wait_until": {"type": "text", "max_length": 40, "required": False}, "timeout_seconds": {"type": "integer", "required": False}, "allowed_destination_origin": {"type": "text", "max_length": 200, "required": False}, "page_context": {"type": "text", "max_length": 200, "required": False}, "tab_id": {"type": "text", "max_length": 80, "required": False}, "page_version": {"type": "integer", "required": False}}, "local_safe", False, _handler_browser_submit_form, lambda result: result.display_value or result.message),
        ToolDefinition("browser.take_screenshot", "Save a screenshot inside trusted roots.", {"session_id": {"type": "text", "max_length": 80}, "path": {"type": "path"}, "full_page": {"type": "bool", "required": False}}, "local_safe", False, _handler_browser_take_screenshot, lambda result: result.display_value or result.message),
        ToolDefinition("browser.capture_view", "Capture the current browser viewport into an opaque temporary vision capture.", {"session_id": {"type": "text", "max_length": 80}, "tab_id": {"type": "text", "max_length": 80, "required": False}}, "read_only", False, _handler_browser_capture_view, lambda result: result.display_value or result.message),
        ToolDefinition("browser.go_back", "Navigate back in browser history.", {"session_id": {"type": "text", "max_length": 80}}, "local_safe", False, _handler_browser_go_back, lambda result: result.display_value or result.message),
        ToolDefinition("browser.go_forward", "Navigate forward in browser history.", {"session_id": {"type": "text", "max_length": 80}}, "local_safe", False, _handler_browser_go_forward, lambda result: result.display_value or result.message),
        ToolDefinition("browser.wait_for_page", "Wait for a page load state.", {"session_id": {"type": "text", "max_length": 80}, "wait_until": {"type": "text", "max_length": 40}, "timeout_seconds": {"type": "integer", "required": False}}, "local_safe", False, _handler_browser_wait_for_page, lambda result: result.display_value or result.message),
        ToolDefinition("browser.scroll_page", "Scroll vertically within the current page.", {"session_id": {"type": "text", "max_length": 80}, "direction": {"type": "text", "max_length": 20, "required": False}, "amount": {"type": "integer", "required": False}}, "local_safe", False, _handler_browser_scroll_page, lambda result: result.display_value or result.message),
        ToolDefinition("browser.scroll_to_element", "Scroll to a visible page element.", {"session_id": {"type": "text", "max_length": 80}, "target_type": {"type": "text", "max_length": 20}, "text_hint": {"type": "text", "max_length": 200, "required": False}, "href_hint": {"type": "text", "max_length": 300, "required": False}, "ordinal": {"type": "integer", "required": False}}, "local_safe", False, _handler_browser_scroll_to_element, lambda result: result.display_value or result.message),
        ToolDefinition("browser.click_element", "Click a safe visible page element.", {"session_id": {"type": "text", "max_length": 80}, "target_type": {"type": "text", "max_length": 20}, "text_hint": {"type": "text", "max_length": 200, "required": False}, "href_hint": {"type": "text", "max_length": 300, "required": False}, "ordinal": {"type": "integer", "required": False}, "wait_until": {"type": "text", "max_length": 40, "required": False}, "timeout_seconds": {"type": "integer", "required": False}}, "local_safe", False, _handler_browser_click_element, lambda result: result.display_value or result.message),
        ToolDefinition("browser.reload_page", "Reload the current browser page.", {"session_id": {"type": "text", "max_length": 80}, "wait_until": {"type": "text", "max_length": 40, "required": False}, "timeout_seconds": {"type": "integer", "required": False}}, "local_safe", False, _handler_browser_reload_page, lambda result: result.display_value or result.message),
        ToolDefinition("browser.open_new_tab", "Open a policy-approved URL in a new browser tab.", {"session_id": {"type": "text", "max_length": 80}, "url": {"type": "text", "max_length": 400}, "wait_until": {"type": "text", "max_length": 40}, "timeout_seconds": {"type": "integer", "required": False}}, "local_safe", False, _handler_browser_open_new_tab, lambda result: result.display_value or result.message),
        ToolDefinition("browser.switch_tab", "Switch the active browser tab.", {"session_id": {"type": "text", "max_length": 80}, "target": {"type": "text", "max_length": 20, "required": False}, "tab_id": {"type": "text", "max_length": 80, "required": False}}, "local_safe", False, _handler_browser_switch_tab, lambda result: result.display_value or result.message),
        ToolDefinition("browser.list_tabs", "List open browser tabs.", {"session_id": {"type": "text", "max_length": 80}}, "read_only", False, _handler_browser_list_tabs, lambda result: result.display_value or result.message),
        ToolDefinition("browser.close_tab", "Close a browser tab.", {"session_id": {"type": "text", "max_length": 80}, "target": {"type": "text", "max_length": 20, "required": False}, "tab_id": {"type": "text", "max_length": 80, "required": False}}, "local_safe", False, _handler_browser_close_tab, lambda result: result.display_value or result.message),
        ToolDefinition("vision.describe_image", "Describe a trusted local image file.", {"path": {"type": "path"}, "detail_level": {"type": "text", "max_length": 20, "required": False}}, "read_only", False, _handler_vision_describe_image, lambda result: result.display_value or result.message),
        ToolDefinition("vision.extract_text", "Extract visible text from a trusted local image file.", {"path": {"type": "path"}, "language_hint": {"type": "text", "max_length": 20, "required": False}, "max_characters": {"type": "integer", "required": False}}, "read_only", False, _handler_vision_extract_text, lambda result: result.display_value or result.message),
        ToolDefinition("vision.find_visual_element", "Find a visually described element inside a trusted local image file.", {"path": {"type": "path"}, "query": {"type": "text", "max_length": 200}, "max_results": {"type": "integer", "required": False}}, "read_only", False, _handler_vision_find_visual_element, lambda result: result.display_value or result.message),
        ToolDefinition("vision.describe_browser_capture", "Describe a temporary browser viewport capture.", {"capture_id": {"type": "text", "max_length": 80}, "detail_level": {"type": "text", "max_length": 20, "required": False}}, "read_only", False, _handler_vision_describe_browser_capture, lambda result: result.display_value or result.message),
        ToolDefinition("vision.extract_text_from_browser_capture", "Extract visible text from a temporary browser viewport capture.", {"capture_id": {"type": "text", "max_length": 80}, "language_hint": {"type": "text", "max_length": 20, "required": False}, "max_characters": {"type": "integer", "required": False}}, "read_only", False, _handler_vision_extract_text_from_browser_capture, lambda result: result.display_value or result.message),
        ToolDefinition("vision.find_visual_element_in_browser_capture", "Find a visually described element inside a temporary browser viewport capture.", {"capture_id": {"type": "text", "max_length": 80}, "query": {"type": "text", "max_length": 200}, "max_results": {"type": "integer", "required": False}}, "read_only", False, _handler_vision_find_visual_element_in_browser_capture, lambda result: result.display_value or result.message),
        ToolDefinition("desktop.list_windows", "List currently open, visible desktop windows by title.", {}, "read_only", False, _handler_desktop_list_windows, lambda result: result.display_value or result.message),
        ToolDefinition("desktop.capture_screen", "Capture a one-shot screenshot of the entire desktop into an opaque temporary vision capture.", {}, "persistent_write", False, _handler_desktop_capture_screen, lambda result: result.display_value or result.message),
        ToolDefinition("desktop.capture_window", "Capture a one-shot screenshot of a specific open window into an opaque temporary vision capture.", {"window_id": {"type": "integer"}, "window_title": {"type": "text", "max_length": 200}}, "persistent_write", False, _handler_desktop_capture_window, lambda result: result.display_value or result.message),
        ToolDefinition("vision.describe_desktop_capture", "Describe a temporary desktop capture.", {"capture_id": {"type": "text", "max_length": 80}, "detail_level": {"type": "text", "max_length": 20, "required": False}}, "read_only", False, _handler_vision_describe_desktop_capture, lambda result: result.display_value or result.message),
        ToolDefinition("vision.extract_text_from_desktop_capture", "Extract visible text from a temporary desktop capture.", {"capture_id": {"type": "text", "max_length": 80}, "language_hint": {"type": "text", "max_length": 20, "required": False}, "max_characters": {"type": "integer", "required": False}}, "read_only", False, _handler_vision_extract_text_from_desktop_capture, lambda result: result.display_value or result.message),
        ToolDefinition("vision.find_visual_element_in_desktop_capture", "Find a visually described element inside a temporary desktop capture.", {"capture_id": {"type": "text", "max_length": 80}, "query": {"type": "text", "max_length": 200}, "max_results": {"type": "integer", "required": False}}, "read_only", False, _handler_vision_find_visual_element_in_desktop_capture, lambda result: result.display_value or result.message),
        ToolDefinition("location.where_am_i", "Reverse-geocode the freshest phone location into a human-readable answer.", {}, "external_navigation", False, _handler_location_where_am_i, lambda result: result.display_value or result.message),
        ToolDefinition("location.distance_to", "Compute the distance from the freshest phone location to an owner-saved place.", {"place": {"type": "text", "max_length": 80}}, "read_only", False, _handler_location_distance_to, lambda result: result.display_value or result.message),
        ToolDefinition("location.save_place", "Save a named place's coordinates for future location.distance_to/navigation.start lookups.", {"name": {"type": "text", "max_length": 80}, "latitude": {"type": "number"}, "longitude": {"type": "number"}}, "persistent_write", False, _handler_location_save_place, lambda result: result.display_value or result.message),
        ToolDefinition("navigation.start", "Start a turn-by-turn navigation session to a destination (saved place, raw coordinates, or place name).", {"destination": {"type": "text", "max_length": 200}}, "external_navigation", False, _handler_navigation_start, lambda result: result.display_value or result.message),
        ToolDefinition("navigation.get_next_instruction", "Recompute the next turn-by-turn instruction from the freshest phone location.", {"session_id": {"type": "text", "max_length": 80}}, "external_navigation", False, _handler_navigation_get_next_instruction, lambda result: result.display_value or result.message),
        ToolDefinition("navigation.stop", "End an active turn-by-turn navigation session.", {"session_id": {"type": "text", "max_length": 80}}, "local_safe", False, _handler_navigation_stop, lambda result: result.display_value or result.message),
        ToolDefinition("power.request_lock", "Request lock confirmation.", {}, "sensitive", True, _handler_request_lock, lambda result: result.display_value or result.message),
        ToolDefinition("power.request_restart", "Request restart confirmation.", {}, "sensitive", True, _handler_request_restart, lambda result: result.display_value or result.message),
        ToolDefinition("power.request_shutdown", "Request shutdown confirmation.", {}, "sensitive", True, _handler_request_shutdown, lambda result: result.display_value or result.message),
    ]

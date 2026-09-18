from __future__ import annotations

from typing import Any

REFERENCE_KEYS = frozenset({"from_step", "field"})
_ALLOWED_REFERENCE_ARGUMENTS = {
    ("notes.create", "text"),
    ("tasks.create", "text"),
    ("memory.remember", "value"),
    ("filesystem.write_text_file", "text"),
    ("filesystem.append_text_file", "text"),
    ("browser.close_session", "session_id"),
    ("browser.open_url", "session_id"),
    ("browser.get_page_info", "session_id"),
    ("browser.extract_visible_text", "session_id"),
    ("browser.inspect_elements", "session_id"),
    ("browser.inspect_clickable_elements", "session_id"),
    ("browser.inspect_form_controls", "session_id"),
    ("browser.inspect_form_controls", "page_version"),
    ("browser.input_text", "session_id"),
    ("browser.input_text", "page_version"),
    ("browser.clear_input", "session_id"),
    ("browser.clear_input", "page_version"),
    ("browser.submit_form", "session_id"),
    ("browser.submit_form", "page_version"),
    ("browser.take_screenshot", "session_id"),
    ("browser.go_back", "session_id"),
    ("browser.go_forward", "session_id"),
    ("browser.wait_for_page", "session_id"),
    ("browser.scroll_page", "session_id"),
    ("browser.scroll_to_element", "session_id"),
    ("browser.click_element", "session_id"),
    ("browser.reload_page", "session_id"),
    ("browser.open_new_tab", "session_id"),
    ("browser.switch_tab", "session_id"),
    ("browser.list_tabs", "session_id"),
    ("browser.close_tab", "session_id"),
    ("browser.capture_view", "session_id"),
    ("browser.capture_view", "tab_id"),
    ("vision.describe_browser_capture", "capture_id"),
    ("vision.extract_text_from_browser_capture", "capture_id"),
    ("vision.find_visual_element_in_browser_capture", "capture_id"),
}


def is_result_reference(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == REFERENCE_KEYS


def allows_result_reference(tool_name: str, argument_name: str) -> bool:
    return (tool_name, argument_name) in _ALLOWED_REFERENCE_ARGUMENTS


def validate_reference(
    reference: dict[str, Any],
    *,
    current_step_id: int,
    available_steps: set[int],
    dependency_steps: set[int] | None = None,
) -> tuple[int, str]:
    if not is_result_reference(reference):
        raise ValueError("reference is invalid")
    from_step = reference.get("from_step")
    field = reference.get("field")
    if not isinstance(from_step, int) or from_step <= 0:
        raise ValueError("reference is invalid")
    if not isinstance(field, str) or not field.strip():
        raise ValueError("reference is invalid")
    if from_step >= current_step_id:
        raise ValueError("forward references are not allowed")
    if from_step not in available_steps:
        raise ValueError("referenced step result is missing")
    if dependency_steps is not None and from_step not in dependency_steps:
        raise ValueError("reference dependency is missing")
    return from_step, field.strip()


def resolve_reference(reference: dict[str, Any], results_by_step: dict[int, dict[str, Any]], current_step_id: int) -> Any:
    if not isinstance(reference, dict):
        raise TypeError("reference must be a dictionary")
    from_step, field = validate_reference(
        reference,
        current_step_id=current_step_id,
        available_steps=set(results_by_step),
    )
    result = results_by_step[from_step]
    if field in result:
        return result[field]
    reference_fields = result.get("reference_fields")
    if isinstance(reference_fields, dict) and field in reference_fields:
        return reference_fields[field]
    raise ValueError("referenced field is missing")

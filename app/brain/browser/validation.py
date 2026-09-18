from __future__ import annotations

from pathlib import Path
from typing import Any

from app.brain.browser.models import (
    ALLOWED_CLICK_TARGET_TYPES,
    ALLOWED_ELEMENT_TYPES,
    ALLOWED_FORM_CONTROL_TYPES,
    ALLOWED_SCREENSHOT_EXTENSIONS,
    ALLOWED_SCROLL_DIRECTIONS,
    ALLOWED_SCROLL_TARGET_TYPES,
    ALLOWED_TAB_TARGETS,
    ALLOWED_WAIT_UNTIL,
)
from app.brain.browser.url_policy import BrowserUrlPolicy
from app.brain.configuration.runtime_config import get_effective_runtime_config
from app.brain.filesystem.errors import FilesystemPathError
from app.brain.filesystem.path_policy import resolve_path
from config.config_loader import load_config

_HIGH_IMPACT_CLICK_MARKERS = (
    "buy",
    "purchase",
    "pay",
    "submit order",
    "place order",
    "delete",
    "remove account",
    "close account",
    "publish",
    "post",
    "send",
    "transfer",
    "confirm",
    "password",
    "privacy",
    "security",
)

_SENSITIVE_FORM_MARKERS = (
    "password",
    "passcode",
    "token",
    "otp",
    "mfa",
    "verification",
    "credit card",
    "card number",
    "cvv",
    "sign in",
    "login",
)


def validate_browser_tool_arguments(tool_name: str, arguments: dict[str, Any]) -> None:
    config = get_effective_runtime_config()
    allowed_test_hosts = config.get("browser_allowed_test_hosts", [])
    resolver = config.get("browser_test_resolver")
    policy = BrowserUrlPolicy(
        allow_http=bool(config.get("browser_allow_http", False)),
        allowed_test_hosts={str(item).lower() for item in allowed_test_hosts} if isinstance(allowed_test_hosts, (list, tuple, set)) else set(),
        resolver=resolver if callable(resolver) else BrowserUrlPolicy().resolver,
    )
    if tool_name in {"browser.open_url", "browser.open_new_tab"}:
        wait_until = str(arguments.get("wait_until") or "").strip().lower()
        if wait_until not in ALLOWED_WAIT_UNTIL:
            raise ValueError("Unsupported browser wait condition.")
        timeout = arguments.get("timeout_seconds", 30)
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0 or timeout > 300:
            raise ValueError("Browser timeout is invalid.")
        url = arguments.get("url")
        if isinstance(url, str):
            policy.validate_url(url)
        return
    if tool_name == "browser.wait_for_page":
        wait_until = str(arguments.get("wait_until") or "").strip().lower()
        if wait_until not in ALLOWED_WAIT_UNTIL:
            raise ValueError("Unsupported browser wait condition.")
        timeout = arguments.get("timeout_seconds", 30)
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0 or timeout > 300:
            raise ValueError("Browser timeout is invalid.")
        return
    if tool_name == "browser.extract_visible_text":
        max_characters = arguments.get("max_characters", 4000)
        if not isinstance(max_characters, int) or isinstance(max_characters, bool) or max_characters <= 0:
            raise ValueError("Browser text limit is invalid.")
        return
    if tool_name == "browser.inspect_elements":
        raw_types = arguments.get("element_types")
        if not isinstance(raw_types, list) or not raw_types:
            raise ValueError("Unsupported element type.")
        normalized = [str(item).strip().lower() for item in raw_types if str(item).strip()]
        if not normalized or any(item not in ALLOWED_ELEMENT_TYPES for item in normalized):
            raise ValueError("Unsupported element type.")
        max_elements = arguments.get("max_elements", 20)
        if not isinstance(max_elements, int) or isinstance(max_elements, bool) or max_elements <= 0:
            raise ValueError("Browser element limit is invalid.")
        return
    if tool_name == "browser.take_screenshot":
        path = arguments.get("path")
        if not isinstance(path, str):
            raise ValueError("Screenshot path is invalid.")
        try:
            resolved = resolve_path(path, prefer_directory=False, allow_missing=True)
        except FilesystemPathError as error:
            raise ValueError(str(error)) from error
        if resolved.absolute_path.suffix.lower() not in ALLOWED_SCREENSHOT_EXTENSIONS:
            raise ValueError("Unsupported screenshot image format.")
        return
    if tool_name == "browser.capture_view":
        _validate_page_binding(arguments)
        return
    if tool_name == "browser.inspect_clickable_elements":
        max_elements = arguments.get("max_elements", 20)
        if not isinstance(max_elements, int) or isinstance(max_elements, bool) or max_elements <= 0:
            raise ValueError("Browser element limit is invalid.")
        return
    if tool_name == "browser.inspect_form_controls":
        max_controls = arguments.get("max_controls", 20)
        if not isinstance(max_controls, int) or isinstance(max_controls, bool) or max_controls <= 0:
            raise ValueError("Browser element limit is invalid.")
        _validate_page_binding(arguments)
        return
    if tool_name == "browser.scroll_page":
        direction = str(arguments.get("direction") or "down").strip().lower()
        if direction not in ALLOWED_SCROLL_DIRECTIONS:
            raise ValueError("Unsupported browser scroll direction.")
        amount = arguments.get("amount", 600)
        if not isinstance(amount, int) or isinstance(amount, bool) or amount <= 0:
            raise ValueError("Browser scroll amount is invalid.")
        return
    if tool_name in {"browser.scroll_to_element", "browser.click_element"}:
        target_type = str(arguments.get("target_type") or "").strip().lower()
        allowed_types = ALLOWED_SCROLL_TARGET_TYPES if tool_name == "browser.scroll_to_element" else ALLOWED_CLICK_TARGET_TYPES
        if target_type not in allowed_types:
            raise ValueError("Unsupported browser target type.")
        text_hint = str(arguments.get("text_hint") or "").strip()
        href_hint = str(arguments.get("href_hint") or "").strip()
        ordinal = arguments.get("ordinal", 0)
        if not text_hint and not href_hint and (not isinstance(ordinal, int) or ordinal <= 0):
            raise ValueError("Browser element targeting requires text_hint, href_hint, or a positive ordinal.")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
            raise ValueError("Browser element ordinal is invalid.")
        if tool_name == "browser.click_element":
            wait_until = str(arguments.get("wait_until") or "").strip().lower()
            if wait_until not in ALLOWED_WAIT_UNTIL:
                raise ValueError("Unsupported browser wait condition.")
            timeout = arguments.get("timeout_seconds", 30)
            if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0 or timeout > 300:
                raise ValueError("Browser timeout is invalid.")
            if _is_high_impact_click(arguments):
                raise ValueError("High-risk browser interaction is not supported in RFC-006B.")
        return
    if tool_name in {"browser.input_text", "browser.clear_input"}:
        control_type = str(arguments.get("control_type") or "").strip().lower()
        if control_type not in ALLOWED_FORM_CONTROL_TYPES:
            raise ValueError("Unsupported browser form control type.")
        label_hint = str(arguments.get("label_hint") or "").strip()
        placeholder_hint = str(arguments.get("placeholder_hint") or "").strip()
        name_hint = str(arguments.get("name_hint") or "").strip()
        ordinal = arguments.get("ordinal", 0)
        if not label_hint and not placeholder_hint and not name_hint and (not isinstance(ordinal, int) or ordinal <= 0):
            raise ValueError("Browser form targeting requires label_hint, placeholder_hint, name_hint, or a positive ordinal.")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
            raise ValueError("Browser element ordinal is invalid.")
        if _contains_sensitive_form_target(arguments):
            raise ValueError("Sensitive form fields are not supported.")
        _validate_page_binding(arguments)
        if tool_name == "browser.input_text":
            text = arguments.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError("Browser input text is invalid.")
        return
    if tool_name == "browser.submit_form":
        label_hint = str(arguments.get("label_hint") or "").strip()
        placeholder_hint = str(arguments.get("placeholder_hint") or "").strip()
        name_hint = str(arguments.get("name_hint") or "").strip()
        form_text_hint = str(arguments.get("form_text_hint") or "").strip()
        submit_text_hint = str(arguments.get("submit_text_hint") or "").strip()
        ordinal = arguments.get("ordinal", 0)
        if not any((label_hint, placeholder_hint, name_hint, form_text_hint, submit_text_hint)) and (
            not isinstance(ordinal, int) or ordinal <= 0
        ):
            raise ValueError("Browser form targeting requires semantic hints or a positive ordinal.")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
            raise ValueError("Browser element ordinal is invalid.")
        wait_until = str(arguments.get("wait_until") or "").strip().lower()
        if wait_until not in ALLOWED_WAIT_UNTIL:
            raise ValueError("Unsupported browser wait condition.")
        timeout = arguments.get("timeout_seconds", 30)
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0 or timeout > 300:
            raise ValueError("Browser timeout is invalid.")
        if _contains_sensitive_form_target(arguments):
            raise ValueError("Sensitive or authenticated form submission is not supported.")
        _validate_page_binding(arguments)
        return
    if tool_name == "browser.reload_page":
        wait_until = str(arguments.get("wait_until") or "").strip().lower()
        if wait_until not in ALLOWED_WAIT_UNTIL:
            raise ValueError("Unsupported browser wait condition.")
        timeout = arguments.get("timeout_seconds", 30)
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0 or timeout > 300:
            raise ValueError("Browser timeout is invalid.")
        return
    if tool_name in {"browser.switch_tab", "browser.close_tab"}:
        target = str(arguments.get("target") or "current").strip().lower()
        if target not in ALLOWED_TAB_TARGETS:
            raise ValueError("Unsupported browser tab target.")
        tab_id = arguments.get("tab_id")
        if tab_id is not None and not isinstance(tab_id, str):
            raise ValueError("Browser tab ID is invalid.")
        return
    if tool_name in {"browser.list_tabs", "browser.get_active_session"}:
        return
    if tool_name.startswith("browser.") and tool_name not in {
        "browser.start_session",
        "browser.close_session",
        "browser.get_page_info",
        "browser.go_back",
        "browser.go_forward",
    }:
        raise ValueError("Unsupported browser operation.")


def _is_high_impact_click(arguments: dict[str, Any]) -> bool:
    text = " ".join(
        str(arguments.get(field) or "")
        for field in ("target_type", "text_hint", "href_hint")
    ).lower()
    return any(marker in text for marker in _HIGH_IMPACT_CLICK_MARKERS)


def _contains_sensitive_form_target(arguments: dict[str, Any]) -> bool:
    text = " ".join(
        str(arguments.get(field) or "")
        for field in ("label_hint", "placeholder_hint", "name_hint", "form_text_hint", "submit_text_hint", "page_context")
    ).lower()
    return any(marker in text for marker in _SENSITIVE_FORM_MARKERS)


def _validate_page_binding(arguments: dict[str, Any]) -> None:
    tab_id = arguments.get("tab_id")
    if tab_id is not None and not isinstance(tab_id, str):
        raise ValueError("Browser tab ID is invalid.")
    page_version = arguments.get("page_version")
    if page_version is not None and (not isinstance(page_version, int) or isinstance(page_version, bool) or page_version < 0):
        raise ValueError("Browser page version is invalid.")

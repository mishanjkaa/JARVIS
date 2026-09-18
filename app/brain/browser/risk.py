from __future__ import annotations

from typing import Any

from app.brain.risk.models import RiskLevel

LOW_RISK_BROWSER_TOOLS = {
    "browser.start_session": "launches an isolated browser session",
    "browser.get_active_session": "uses the current isolated browser session",
    "browser.close_session": "closes an isolated browser session",
    "browser.open_url": "opens a public page",
    "browser.get_page_info": "reads the current page title and URL",
    "browser.extract_visible_text": "extracts visible page text",
    "browser.inspect_elements": "inspects structured page elements",
    "browser.inspect_clickable_elements": "inspects structured clickable elements",
    "browser.inspect_form_controls": "inspects structured visible form controls",
    "browser.take_screenshot": "captures a page screenshot inside trusted roots",
    "browser.go_back": "navigates back in browser history",
    "browser.go_forward": "navigates forward in browser history",
    "browser.wait_for_page": "waits for a page load state",
    "browser.scroll_page": "scrolls vertically within the current page",
    "browser.scroll_to_element": "scrolls to a visible element",
    "browser.reload_page": "reloads the current page",
    "browser.open_new_tab": "opens a public page in a new browser tab",
    "browser.switch_tab": "switches the active browser tab",
    "browser.list_tabs": "lists open browser tabs",
    "browser.close_tab": "closes the current browser tab",
}

FUTURE_BROWSER_RISK = {
    "browser.input_text": (RiskLevel.LOW, "text entry without submission"),
    "browser.clear_input": (RiskLevel.LOW, "clears text in a visible form control"),
    "browser.submit_form": (RiskLevel.MEDIUM, "form submission can change external state"),
}

_HIGH_IMPACT_MARKERS = (
    "payment",
    "purchase",
    "card",
    "password",
    "security",
    "privacy",
    "transfer",
    "delete account",
    "publish",
    "post",
    "message",
    "comment",
    "bid",
    "order",
    "checkout",
    "register",
    "sign in",
    "log in",
)


def classify_browser_step(tool_name: str, arguments: dict[str, Any]) -> tuple[RiskLevel, str] | None:
    if tool_name in LOW_RISK_BROWSER_TOOLS:
        return RiskLevel.LOW, LOW_RISK_BROWSER_TOOLS[tool_name]
    if tool_name == "browser.click_element":
        if _is_semantically_high_impact(arguments):
            return RiskLevel.HIGH, "browser action has a high external consequence"
        target_type = str(arguments.get("target_type") or "").strip().lower()
        if target_type == "link":
            return RiskLevel.LOW, "clicks an ordinary navigation link"
        if target_type == "button":
            return RiskLevel.MEDIUM, "clicking a button can have uncertain external effects"
        return RiskLevel.MEDIUM, "browser interaction has uncertain external effects"
    if tool_name in FUTURE_BROWSER_RISK:
        level, reason = FUTURE_BROWSER_RISK[tool_name]
        if _is_semantically_high_impact(arguments):
            return RiskLevel.HIGH, "browser action has a high external consequence"
        return level, reason
    if tool_name.startswith("browser.") and _is_semantically_high_impact(arguments):
        return RiskLevel.HIGH, "browser action has a high external consequence"
    return None


def _is_semantically_high_impact(arguments: dict[str, Any]) -> bool:
    text = " ".join(f"{key}={value}" for key, value in sorted(arguments.items()) if isinstance(value, (str, int, float, bool)))
    lowered = text.lower()
    return any(marker in lowered for marker in _HIGH_IMPACT_MARKERS)
